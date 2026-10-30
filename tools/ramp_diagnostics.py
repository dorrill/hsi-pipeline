"""Cross-track ramp diagnosis: per-column gain and offset of every band (``viz`` extra).

Diagnosis only; the pipeline applies no ramp correction. Runs the pipeline up to
registration (calibration, offset repair, destriping, co-registration to band
27), takes the mean of all bands as a scene proxy, and fits every band against
it column by column (:func:`hsi_pipeline.radiometry.column_linear_fit`). A trend
across the columns in the intercept is an additive error, one in the slope a
multiplicative error.

The proxy contains any ramp shared by all bands, so only ramps *relative to the
band mean* are seen by the fit. The scene also changes across the track (ocean
and cumulus on the left, city on the right), and each column's line follows the
surfaces it contains, so the fit alone cannot separate instrument from scene.

Second check, on a single surface: the median of every band over open water
(dark at 556 and 729 nm, clear of the cloud mask), column by column. Water is nearly
uniform, so a trend here is instrumental or atmospheric (view angle), and a ramp
shared by all bands is not cancelled. Columns without water cannot be checked.

Writes to the output directory:
    ramp_profiles.png   intercept and relative-slope profiles of all bands
    water_profiles.png  band level over open water per column, all bands
    ramp_profiles.npz   the raw per-column profiles
    ramp_stats.json     per band: left-to-right change of slope, intercept and water level

Usage:
    python tools/ramp_diagnostics.py --input-dir inputs --output-dir outputs/diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hsi_pipeline.calibration import check_offsets  # noqa: E402
from hsi_pipeline.cloud import BRIGHTNESS_SIGMA, cloud_mask, water_mode_and_sigma  # noqa: E402
from hsi_pipeline.io import load_cube, load_metadata  # noqa: E402
from hsi_pipeline.radiometry import (  # noqa: E402
    apply_radiometric_correction,
    column_linear_fit,
    destripe_columns,
    validity_masks,
)
from hsi_pipeline.registration import register_cube  # noqa: E402

#: Reference band for registration, as in the pipeline (the chosen G band).
REFERENCE_BAND = "27"
#: Columns averaged at each edge for the left-to-right change (1/8 of the swath).
EDGE_COLUMNS = 256
#: Running median (columns) applied to the plotted profiles only, to suppress
#: column-to-column scatter (residual stripes, per-column fit noise).
PLOT_SMOOTH_COLUMNS = 31
#: Cloud-mask dilation (px) before taking water: keeps thin cloud edges and the
#: cloud shadows next to them out of the water statistics.
CLOUD_CLEARANCE_PX = 10
#: Minimum water pixels in a column for its water level to be reported.
MIN_WATER_ROWS = 100
#: Right-edge check: change of the per-column median from the inner to the outer
#: column range, per block of rows. The same change in blocks with different land
#: cover points to the instrument rather than the scene.
EDGE_INNER_COLUMNS = slice(-256, -192)
EDGE_OUTER_COLUMNS = slice(-64, -8)
EDGE_ROW_BLOCK = 256
#: Erosion (px) of the water mask: drops isolated dark land pixels and mixed
#: shoreline pixels.
WATER_EROSION_PX = 3
#: Box smoothing (px) before the water tests, as in the cloud mask.
WATER_SMOOTH_PX = 3


def right_edge_change(band: np.ndarray, usable: np.ndarray) -> list[float]:
    """Outer minus inner right-edge column median, per ``EDGE_ROW_BLOCK`` rows."""
    out = []
    for r0 in range(0, band.shape[0], EDGE_ROW_BLOCK):
        block = np.where(usable[r0 : r0 + EDGE_ROW_BLOCK], band[r0 : r0 + EDGE_ROW_BLOCK], np.nan)
        inner = block[:, EDGE_INNER_COLUMNS]
        outer = block[:, EDGE_OUTER_COLUMNS]
        if np.isfinite(inner).any() and np.isfinite(outer).any():
            out.append(round(float(np.nanmedian(outer) - np.nanmedian(inner)), 3))
        else:
            out.append(None)
    return out


def water_mask(green: np.ndarray, red_edge: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Open water: dark at 556 nm *and* at 729 nm, eroded.

    Green alone does not separate water from the city here (both sit below the
    cloud-brightness threshold); land is bright at 729 nm, water is not. Each
    test is ``BRIGHTNESS_SIGMA`` noise sigmas above that band's water mode.
    """
    dark = usable.copy()
    for band in (green, red_edge):
        smooth = ndimage.uniform_filter(np.where(usable, band, 0.0), WATER_SMOOTH_PX)
        mode, sigma = water_mode_and_sigma(smooth[usable])
        dark &= smooth < mode + BRIGHTNESS_SIGMA * sigma
    return ndimage.binary_erosion(dark, iterations=WATER_EROSION_PX)


def edge_change(profile: np.ndarray) -> float:
    """Median of the last ``EDGE_COLUMNS`` finite values minus median of the first."""
    finite = profile[np.isfinite(profile)]
    return float(np.median(finite[-EDGE_COLUMNS:]) - np.median(finite[:EDGE_COLUMNS]))


def main() -> None:
    """Run the diagnosis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("inputs"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bands = load_metadata(args.input_dir / "band_metadata.json", args.input_dir)
    dn = load_cube(bands)
    valid, saturated = validity_masks(dn)
    bands, _ = check_offsets(dn, bands, valid)
    cube = apply_radiometric_correction(dn, bands, valid)
    del dn
    destripe_columns(cube, saturated, args.workers)
    ref = [b.band_id for b in bands].index(REFERENCE_BAND)
    register_cube(cube, valid, ref, args.workers)

    usable = valid.all(axis=0)
    proxy = np.where(usable, cube.mean(axis=0), np.nan)
    n_bands, _, width = cube.shape
    green = ref
    red_edge = int(np.argmax([b.wavelength_nm for b in bands]))
    clouds, _ = cloud_mask(cube[green], cube[red_edge], usable)
    water = water_mask(cube[green], cube[red_edge], usable) & ~ndimage.binary_dilation(
        clouds.astype(bool), iterations=CLOUD_CLEARANCE_PX
    )
    water_cols = water.sum(axis=0) >= MIN_WATER_ROWS
    water_levels = np.full((n_bands, width), np.nan)
    intercepts, slopes = np.empty((n_bands, width)), np.empty((n_bands, width))
    rows = []
    for b in bands:
        a, s = column_linear_fit(cube[b.index], proxy, usable)
        intercepts[b.index], slopes[b.index] = a, s
        level = float(np.nanmedian(cube[b.index][usable]))
        gain = float(np.nanmedian(s))
        on_water = np.where(water[:, water_cols], cube[b.index][:, water_cols], np.nan)
        water_levels[b.index, water_cols] = np.nanmedian(on_water, axis=0)
        water_level = float(np.nanmedian(water_levels[b.index]))
        rows.append(
            {
                "band_id": b.band_id,
                "wavelength_nm": round(b.wavelength_nm, 1),
                "median_level": round(level, 3),
                "median_slope": round(gain, 4),
                "slope_change_pct": round(100 * edge_change(s) / gain, 2),
                "intercept_change": round(edge_change(a), 3),
                "intercept_change_pct_of_level": round(100 * edge_change(a) / level, 2),
                "water_level": round(water_level, 3),
                "water_change_pct": round(
                    100 * edge_change(water_levels[b.index]) / water_level, 2
                ),
                "right_edge_change_per_row_block": right_edge_change(cube[b.index], usable),
            }
        )
    np.savez_compressed(
        args.output_dir / "ramp_profiles.npz",
        intercept=intercepts,
        slope=slopes,
        water_level=water_levels,
        band_id=[b.band_id for b in bands],
        wavelength_nm=[b.wavelength_nm for b in bands],
    )
    water_idx = np.flatnonzero(water_cols)
    stats = {
        "method": "per-column bisquare fit of band vs mean of all registered bands; "
        "median over open water per column",
        "edge_columns": EDGE_COLUMNS,
        "water_columns": [int(water_idx.min()), int(water_idx.max())],
        "water_columns_count": int(water_cols.sum()),
        "right_edge": {
            "inner_columns": [width - 256, width - 192],
            "outer_columns": [width - 64, width - 8],
            "row_block": EDGE_ROW_BLOCK,
            "units": "corrected units (registered, destriped)",
        },
        "bands": rows,
    }
    (args.output_dir / "ramp_stats.json").write_text(json.dumps(stats, indent=2))

    wl = np.array([b.wavelength_nm for b in bands])
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(wl.min(), wl.max())
    cols = np.arange(width)
    fig, (ax_a, ax_s) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for b in bands:
        colour = cmap(norm(b.wavelength_nm))
        level = rows[b.index]["median_level"]
        a = ndimage.median_filter(intercepts[b.index], PLOT_SMOOTH_COLUMNS, mode="nearest")
        s = ndimage.median_filter(slopes[b.index], PLOT_SMOOTH_COLUMNS, mode="nearest")
        ax_a.plot(cols, 100 * (a - np.nanmedian(a)) / level, color=colour, lw=0.8)
        ax_s.plot(cols, 100 * (s / np.nanmedian(s) - 1), color=colour, lw=0.8)
    ax_a.set_ylabel("intercept − its median\n(% of band median level)")
    ax_a.set_title("Additive part: intercept of band vs scene proxy, per column")
    ax_s.set_ylabel("slope / its median − 1 (%)")
    ax_s.set_title("Multiplicative part: slope of band vs scene proxy, per column")
    ax_s.set_xlabel(f"column (running median over {PLOT_SMOOTH_COLUMNS} columns)")
    for ax in (ax_a, ax_s):
        ax.axhline(0, color="0.5", lw=0.5)
        ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=(ax_a, ax_s), label="nm")
    fig.savefig(args.output_dir / "ramp_profiles.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for b in bands:
        level = water_levels[b.index]
        smooth = ndimage.median_filter(level, PLOT_SMOOTH_COLUMNS, mode="nearest")
        smooth[~water_cols] = np.nan
        rel = 100 * (smooth / np.nanmedian(level) - 1)
        ax.plot(cols, rel, color=cmap(norm(b.wavelength_nm)), lw=0.8)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, width)
    ax.set_ylabel("level / its median − 1 (%)")
    ax.set_xlabel(f"column (running median over {PLOT_SMOOTH_COLUMNS} columns; gaps: no water)")
    ax.set_title("Band level over open water, per column")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="nm")
    fig.savefig(args.output_dir / "water_profiles.png", dpi=90)
    print(f"Wrote ramp and water profiles and ramp_stats.json to {args.output_dir}")


if __name__ == "__main__":
    main()
