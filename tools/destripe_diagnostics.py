"""Before/after evaluation of column destriping on the real capture (``viz`` extra).

Writes ``destripe_stats.json`` and ``destripe_crop.png`` to the output directory:

* Stripe amplitude per band, split-half: offsets estimated on the even rows are
  removed from the odd rows, and the stripe amplitude is re-measured on the odd
  rows only, so the "after" value is not the estimator grading itself. The
  noise floor is what the same measurement gives for white noise alone.
  The previous estimator (column-median profile minus its running median) is
  evaluated the same way for comparison.
* Per-band SNR with and without destriping, and the RGB choice in each case.
* A before/after crop of open water for the three chosen bands.

Usage:
    python tools/destripe_diagnostics.py --input-dir inputs --output-dir outputs/diagnostics
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
from hsi_pipeline.io import load_cube, load_metadata  # noqa: E402
from hsi_pipeline.radiometry import (  # noqa: E402
    STRIPE_WINDOW_COLUMNS,
    apply_radiometric_correction,
    column_offsets,
    destripe_columns,
    validity_masks,
)
from hsi_pipeline.rgb import band_snr, select_rgb  # noqa: E402

#: Open water near the frame centre: flat, so stripes dominate what is visible.
WATER_CROP = np.s_[1400:1656, 900:1156]
#: Median of n Gaussian samples has std ~ sqrt(pi / 2) sigma / sqrt(n).
_MEDIAN_EFFICIENCY = np.sqrt(np.pi / 2)


def previous_offsets(band: np.ndarray) -> np.ndarray:
    """Estimator used before: column-median profile minus its running median."""
    profile = np.nanmedian(band, axis=0)
    profile = np.where(np.isfinite(profile), profile, np.nanmedian(profile))
    return profile - ndimage.median_filter(profile, STRIPE_WINDOW_COLUMNS, mode="nearest")


def split_half(band: np.ndarray, saturated: np.ndarray, sigma: float) -> dict[str, float]:
    """Stripe amplitude on the odd rows before and after removing even-row estimates."""
    even, odd, sat_odd = band[0::2], band[1::2], saturated[1::2]
    n_rows = np.isfinite(odd).sum(axis=0).mean()
    return {
        "before": float(column_offsets(odd, sat_odd).std()),
        "after": float(column_offsets(odd - column_offsets(even, saturated[0::2]), sat_odd).std()),
        "after_previous_method": float(column_offsets(odd - previous_offsets(even), sat_odd).std()),
        "noise_floor": float(_MEDIAN_EFFICIENCY * sigma / np.sqrt(n_rows)),
    }


def stretch(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear display stretch to [0, 1]."""
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main() -> None:
    """Run the diagnostics."""
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
    raw = apply_radiometric_correction(dn, bands, valid)
    del dn
    destriped = raw.copy()
    amplitude = destripe_columns(destriped, saturated, args.workers)

    snr_raw = band_snr(raw, bands, valid, saturated)
    snr_ds = band_snr(destriped, bands, valid, saturated)
    chosen_raw, chosen_ds = select_rgb(snr_raw), select_rgb(snr_ds)

    per_band = []
    for b, amp, s0, s1 in zip(bands, amplitude, snr_raw, snr_ds, strict=True):
        per_band.append(
            {
                "band_id": b.band_id,
                "wavelength_nm": round(b.wavelength_nm, 1),
                "removed_amplitude": round(float(amp), 4),
                "split_half": {
                    k: round(v, 4)
                    for k, v in split_half(raw[b.index], saturated[b.index], s1.noise).items()
                },
                "snr_without": round(s0.snr, 2),
                "snr_with": round(s1.snr, 2),
                "noise_without": round(s0.noise, 4),
                "noise_with": round(s1.noise, 4),
            }
        )
    stats = {
        "units": "corrected units (gain * DN + offset)",
        "chosen_without_destripe": {c: snr_raw[k].band_id for c, k in chosen_raw.items()},
        "chosen_with_destripe": {c: snr_ds[k].band_id for c, k in chosen_ds.items()},
        "rgb": {c: per_band[k] for c, k in chosen_ds.items()},
        "bands": per_band,
    }
    (args.output_dir / "destripe_stats.json").write_text(json.dumps(stats, indent=2))
    for c, row in stats["rgb"].items():
        h = row["split_half"]
        print(
            f"{c} band {row['band_id']}: stripe std {h['before']:.3f} -> {h['after']:.3f} "
            f"(previous method {h['after_previous_method']:.3f}, floor {h['noise_floor']:.3f}); "
            f"SNR {row['snr_without']} -> {row['snr_with']}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
    for j, (c, k) in enumerate(chosen_ds.items()):
        before, after = raw[k][WATER_CROP], destriped[k][WATER_CROP]
        lo, hi = np.nanpercentile(before, [1, 99])
        for i, (img, label) in enumerate([(before, "calibrated"), (after, "destriped")]):
            axes[i, j].imshow(stretch(img, lo, hi), cmap="gray", interpolation="nearest")
            axes[i, j].set_title(f"{c}, band {bands[k].band_id}: {label}")
            axes[i, j].set_axis_off()
    fig.suptitle(
        "Open water, rows 1400-1655, columns 900-1155 (1-99 % stretch of the calibrated crop)"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "destripe_crop.png", dpi=80)
    print(f"Wrote {args.output_dir / 'destripe_stats.json'} and destripe_crop.png")


if __name__ == "__main__":
    main()
