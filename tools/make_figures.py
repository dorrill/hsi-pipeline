"""Summary figures for the pipeline outputs (requires the ``viz`` extra).

Run after ``hsi-pipeline``; reads its JSON outputs and recomputes only what is
not stored (three unregistered bands and the R/B displacement fields).

Usage:
    python tools/make_figures.py --input-dir inputs --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hsi_pipeline.calibration import check_offsets  # noqa: E402
from hsi_pipeline.io import load_cube, load_metadata, read_band  # noqa: E402
from hsi_pipeline.radiometry import (  # noqa: E402
    SATURATION_DN,
    apply_radiometric_correction,
    destripe_columns,
    validity_masks,
)
from hsi_pipeline.registration import register_band  # noqa: E402
from hsi_pipeline.rgb import RGB_WINDOWS_NM, rgb_composite  # noqa: E402

COLOURS = {"R": "tab:red", "G": "tab:green", "B": "tab:blue"}
#: Crops shown at full resolution: (title, rows, cols).
CROPS = [
    ("Port and downtown", slice(300, 812), slice(1300, 1812)),
    ("North Island airfield", slice(250, 762), slice(450, 962)),
    ("Clouds over the ocean", slice(1050, 1562), slice(0, 512)),
]


def snr_figure(sel: dict, summary: list[dict], out: Path) -> None:
    """SNR, noise and contrast vs wavelength with colour windows and chosen bands."""
    rows = sel["bands"]
    wl = np.array([r["wavelength_nm"] for r in rows])
    snr = np.array([r["snr"] for r in rows])
    ids = [r["band_id"] for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1])
    for c, (lo, hi) in RGB_WINDOWS_NM.items():
        for a in (ax, ax2):
            a.axvspan(lo, hi, color=COLOURS[c], alpha=0.08)
        ax.text((lo + hi) / 2, snr.max() * 1.07, f"{c} window", ha="center", color=COLOURS[c])
    ax.plot(wl, snr, "o-", color="0.4", ms=4, lw=1)
    for c, bid in sel["chosen"].items():
        k = ids.index(bid)
        ax.plot(wl[k], snr[k], "o", ms=12, mfc="none", mec=COLOURS[c], mew=2.5)
        ax.annotate(
            f"{c}: band {bid}\n{wl[k]:.1f} nm, SNR {snr[k]:.1f}",
            (wl[k], snr[k]), textcoords="offset points", xytext=(10, 12),
            color=COLOURS[c], fontsize=9, weight="bold",
        )  # fmt: skip
    ax.set_ylabel("SNR = contrast / noise")
    ax.set_ylim(0, snr.max() * 1.15)
    ax.set_title("Per-band SNR (destriped, before registration): highest SNR per colour window")
    ax.grid(alpha=0.3)

    # In DN, noise is ~1 DN in every band, so contrast in DN is what drives SNR.
    gain = {r["band_id"]: r["gain"] for r in summary}
    g = np.array([gain[i] for i in ids])
    ax2.plot(
        wl, np.array([r["contrast"] for r in rows]) / g, "s-", ms=3, lw=1, label="scene contrast"
    )
    ax2.plot(wl, np.array([r["noise"] for r in rows]) / g, "^-", ms=3, lw=1, label="noise")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("DN")
    ax2.set_title(
        "In DN: noise ~1 DN everywhere; the low-gain bands (40-44, 75-80) span ~2x more DN"
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "snr_vs_wavelength.png", dpi=110)
    plt.close(fig)


def rgb_figures(rgb: np.ndarray, sel: dict, out: Path) -> None:
    """The composite, its three channels, and full-resolution crops."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
    axes[0].imshow(rgb[::2, ::2])
    axes[0].set_title("rgb_image.tif (2x downsampled)")
    for ax, (c, ch) in zip(axes[1:], [("R", 0), ("G", 1), ("B", 2)], strict=True):
        ax.imshow(rgb[::2, ::2, ch], cmap="gray", vmin=1, vmax=255)
        ax.set_title(f"{c} channel: band {sel['chosen'][c]}", color=COLOURS[c])
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "rgb_overview.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CROPS), figsize=(7 * len(CROPS), 7.4))
    for ax, (title, r, c) in zip(axes, CROPS, strict=True):
        ax.imshow(rgb[r, c], interpolation="nearest")
        ax.set_title(f"{title} (rows {r.start}-{r.stop}, cols {c.start}-{c.stop})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("RGB composite, full-resolution crops")
    fig.tight_layout()
    fig.savefig(out / "rgb_crops.png", dpi=90)
    plt.close(fig)


def registration_figures(input_dir: Path, sel: dict, reg_stats: dict, out: Path) -> None:
    """Unregistered vs registered composite, and the R/B displacement fields."""
    bands = load_metadata(input_dir / "band_metadata.json", input_dir)
    ids = [b.band_id for b in bands]
    order = [ids.index(sel["chosen"][c]) for c in ("R", "G", "B")]
    # Offsets as the pipeline applies them (the repair needs every band's offset).
    full = load_cube(bands)
    full_valid, _ = validity_masks(full)
    repaired, _ = check_offsets(full, bands, full_valid)
    dn_all, valid = full[order], full_valid[order]
    del full, full_valid
    three = [replace(repaired[k], index=i) for i, k in enumerate(order)]
    cube = apply_radiometric_correction(dn_all, three, valid)
    destripe_columns(cube, dn_all == SATURATION_DN)

    single = {"R": {0: 1.0}, "G": {1: 1.0}, "B": {2: 1.0}}
    before = rgb_composite(cube, single, valid.all(axis=0))
    fields = {}
    reg_cube, reg_valid = cube.copy(), valid.copy()
    for i, c in ((0, "R"), (2, "B")):
        reg_cube[i], reg_valid[i], fields[c], _ = register_band(cube[1], cube[i], valid[i])
    after = rgb_composite(reg_cube, single, reg_valid.all(axis=0))

    fig, axes = plt.subplots(2, len(CROPS), figsize=(7 * len(CROPS), 14))
    for j, (title, r, c) in enumerate(CROPS):
        for i, (img, label) in enumerate([(before, "unregistered"), (after, "registered")]):
            axes[i, j].imshow(img[r, c], interpolation="nearest")
            axes[i, j].set_title(f"{title}: {label}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(
        "Same bands, same stretch: before vs after co-registration to G "
        "(B was 4-45 px and R 3-22 px off, varying over the frame)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "registration_before_after.png", dpi=80)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for i, c in enumerate(("B", "R")):
        for j, comp in enumerate(("dy", "dx")):
            im = axes[i, j].imshow(fields[c][j], cmap="viridis")
            axes[i, j].set_title(f"{c} (band {sel['chosen'][c]}) vs G: {comp} (px)")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(im, ax=axes[i, j], fraction=0.046)
    fig.suptitle(
        "Displacement fields: dy varies across the swath and along the rows (time); "
        "the bottom rows are stretched for bands far from the reference"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "registration_fields.png", dpi=80)
    plt.close(fig)


def shift_vs_band_figure(reg_stats: dict, out: Path) -> None:
    """Coarse shift vs band number: linear in detector position, segment jump at 44/45."""
    rows = reg_stats["bands"]
    n = np.array([int(r["band_id"]) for r in rows])
    s = np.array([r["coarse_shift_dy_dx_px"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, label in [(axes[0], "band", "band number"), (axes[1], "wl", "wavelength (nm)")]:
        x = n if key == "band" else np.array([r["wavelength_nm"] for r in rows])
        ax.plot(x, s[:, 0], "o", ms=4, label="dy")
        ax.plot(x, s[:, 1], "s", ms=4, label="dx")
        ax.set_xlabel(label)
        ax.set_ylabel(f"full-frame shift vs band {reg_stats['reference_band']} (px)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].axvline(44.5, color="0.5", ls="--")
    axes[0].text(44.8, s[:, 0].max() * 0.8, "segment gap\n(44 | 45)", fontsize=8, color="0.3")
    axes[0].set_title("Linear in band number (detector row position)")
    axes[1].set_title("Not monotonic in wavelength (bands 22/24/23/25 interleave)")
    fig.tight_layout()
    fig.savefig(out / "shift_vs_band.png", dpi=110)
    plt.close(fig)


def offset_figure(summary: list[dict], out: Path) -> None:
    """Scene-mean corrected spectrum with the metadata offsets vs the repaired ones."""
    wl = np.array([r["wavelength_nm"] for r in summary])
    applied = np.array([r["corrected_mean"] for r in summary])
    meta = applied + np.array([r["offset_metadata"] - r["offset"] for r in summary])
    fixed = np.array([r["offset_repaired"] for r in summary])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(wl, meta, "o-", color="tab:orange", ms=4, lw=1, label="metadata offsets")
    ax.plot(wl, applied, "o-", color="tab:blue", ms=4, lw=1, label="after offset check")
    for w, a, m, bid in zip(
        wl[fixed], applied[fixed], meta[fixed],
        [r["band_id"] for r, f in zip(summary, fixed, strict=True) if f], strict=True,
    ):  # fmt: skip
        ax.annotate("", (w, a), (w, m), arrowprops={"arrowstyle": "->", "color": "0.4"})
        ax.text(w + 1, m, bid, fontsize=8)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("scene-mean corrected value")
    ax.set_title(
        "Offsets of bands 42-44 and 75-76 (+41 to +68) put ~2.5x steps into the spectrum; "
        "repaired to the median offset"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "offset_check_spectrum.png", dpi=110)
    plt.close(fig)


def offset_by_band_figures(summary: list[dict], out: Path) -> None:
    """Scene-mean corrected value vs band number, before and after the offset repair.

    Writes two images with the same axes, so they can be compared side by side
    or flipped between: ``offsets_by_band_before.png`` (metadata offsets) and
    ``offsets_by_band_after.png`` (repaired offsets). Repaired bands are marked.
    """
    rows = sorted(summary, key=lambda r: int(r["band_id"]))
    number = np.array([int(r["band_id"]) for r in rows])
    after = np.array([r["corrected_mean"] for r in rows])
    before = after + np.array([r["offset_metadata"] - r["offset"] for r in rows])
    fixed = np.array([r["offset_repaired"] for r in rows])
    lo, hi = min(before.min(), after.min()), max(before.max(), after.max())
    margin = 0.05 * (hi - lo)
    for name, values, label in (
        ("before", before, "metadata offsets"),
        ("after", after, "repaired offsets"),
    ):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(number, values, "o-", color="tab:blue", ms=4, lw=1, label="all bands")
        ax.plot(
            number[fixed], values[fixed], "o", ms=9, mfc="none", mec="tab:red", mew=2,
            label="offset replaced by the check",
        )  # fmt: skip
        for k, (n, v) in enumerate(zip(number[fixed], values[fixed], strict=True)):
            dy = 8 if k % 2 == 0 else -14  # alternate above/below: neighbours overlap
            ax.annotate(str(n), (n, v), textcoords="offset points", xytext=(-4, dy), fontsize=8)
        ax.set_xlabel("band number (metadata key; not monotonic in wavelength)")
        ax.set_ylabel("scene-mean gain·DN + offset")
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(f"Corrected scene mean by band, {label}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"offsets_by_band_{name}.png", dpi=110)
        plt.close(fig)


def dynamic_range_figure(input_dir: Path, out: Path) -> None:
    """Raw DN range of each band vs band number: extremes, 1st-99th percentiles, median.

    The extremes are set by single pixels; the percentile range is the part of
    the 8-bit scale the scene actually fills. Writes ``dynamic_range_by_band.png``.
    """
    bands = load_metadata(input_dir / "band_metadata.json", input_dir)
    bands = sorted(bands, key=lambda b: int(b.band_id))
    number = np.array([int(b.band_id) for b in bands])
    rows = []
    for b in bands:
        counts = np.bincount(read_band(b.path).ravel(), minlength=256)
        counts[0] = 0  # DN 0 is no-data (none in the challenge capture)
        cdf = np.cumsum(counts) / counts.sum()
        present = np.flatnonzero(counts)
        p1, p50, p99 = np.searchsorted(cdf, [0.01, 0.5, 0.99])
        rows.append((present[0], p1, p50, p99, present[-1], counts[SATURATION_DN]))
    lo, p1, med, p99, hi, saturated = np.array(rows).T

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1])
    ax.vlines(number, lo, hi, color="0.7", lw=1.5, label="min to max")
    ax.vlines(number, p1, p99, color="tab:blue", lw=5, label="1st to 99th percentile")
    ax.plot(number, med, "o", color="white", mec="tab:blue", ms=4, label="median")
    ax.axhline(SATURATION_DN, color="tab:red", ls="--", lw=1, label=f"saturation ({SATURATION_DN})")
    hit = saturated > 0
    if hit.any():
        ax.text(
            0.99, 0.97,
            "saturated pixels: " + ", ".join(
                f"band {n}: {k}" for n, k in zip(number[hit], saturated[hit], strict=True)
            ),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="tab:red",
        )  # fmt: skip
    ax.set_ylabel("raw DN")
    ax.set_ylim(90, 270)
    ax.set_title("Raw DN range per band: the scene sits on a ~100 DN pedestal")
    ax.legend(loc="upper left", bbox_to_anchor=(0.17, 0.92), fontsize=9)
    ax.grid(alpha=0.3)

    ax2.bar(number, hi - lo + 1, color="0.8", width=0.8, label="min to max")
    ax2.bar(number, p99 - p1 + 1, color="tab:blue", width=0.5, label="1st to 99th percentile")
    ax2.set_ylabel("DN levels used (of 256)")
    ax2.set_xlabel("band number (metadata key; not monotonic in wavelength)")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.17, 1.0), fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "dynamic_range_by_band.png", dpi=110)
    plt.close(fig)


def cloud_figure(rgb: np.ndarray, mask: np.ndarray, stats: dict, out: Path) -> None:
    """Cloud mask outline on the RGB composite, full frame and crops."""
    crops = [
        ("Cumulus field", slice(900, 1400), slice(0, 500)),
        ("Airfield and Point Loma", slice(250, 762), slice(450, 962)),
        ("Clouds near the coast", slice(0, 500), slice(0, 500)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.6))
    axes[0].imshow(rgb[::2, ::2])
    axes[0].contour(mask[::2, ::2], [0.5], colors="yellow", linewidths=0.7)
    axes[0].set_title(
        f"cloud_mask.tif outline: {100 * stats['cloud_fraction']:.1f} % cloud, "
        f"{stats['n_objects']} objects"
    )
    for ax, (title, r, c) in zip(axes[1:], crops, strict=True):
        ax.imshow(rgb[r, c], interpolation="nearest")
        ax.contour(mask[r, c], [0.5], colors="yellow", linewidths=1)
        ax.set_title(f"{title} (rows {r.start}-{r.stop}, cols {c.start}-{c.stop})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "cloud_mask_overlay.png", dpi=80)
    plt.close(fig)


def ship_figure(rgb: np.ndarray, mask: np.ndarray, report: dict, out: Path) -> None:
    """Ship boxes on the RGB composite (solid: confidence >= 0.5), and a crop of each."""
    ships = report["ships"]
    fig = plt.figure(figsize=(24, 10.5), layout="constrained")
    full, grid = fig.subfigures(1, 2, width_ratios=[1, 1.4])
    ax = full.subplots()
    ax.imshow(rgb[::2, ::2])
    ax.contour(mask[::2, ::2], [0.5], colors="yellow", linewidths=0.5)
    pad = 12  # px around each box, so small vessels stay visible at this scale
    for s in ships:
        style = "-" if s["confidence"] >= 0.5 else ":"
        ax.add_patch(
            plt.Rectangle(
                ((s["col_min"] - pad) / 2, (s["row_min"] - pad) / 2),
                (s["col_max"] - s["col_min"] + 2 * pad) / 2,
                (s["row_max"] - s["row_min"] + 2 * pad) / 2,
                fill=False, ec="red", lw=1.5, ls=style,
            )
        )  # fmt: skip
    n_conf = sum(s["confidence"] >= 0.5 for s in ships)
    ax.set_title(
        f"{len(ships)} detections, {n_conf} with confidence >= 0.5 (solid); "
        f"searched {100 * report['search_fraction']:.0f} % of the frame; cloud in yellow"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cols = 7
    axes = grid.subplots(-(-len(ships) // cols), cols, squeeze=False).ravel()
    half = 32
    for a, s in zip(axes, ships, strict=False):
        cy, cx = (s["row_min"] + s["row_max"]) // 2, (s["col_min"] + s["col_max"]) // 2
        y0 = int(np.clip(cy - half, 0, rgb.shape[0] - 2 * half))
        x0 = int(np.clip(cx - half, 0, rgb.shape[1] - 2 * half))
        crop = rgb[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(np.float32)
        lo, hi = np.percentile(crop, [1, 99.5])  # water is dark in the global stretch
        a.imshow(np.clip((crop - lo) / max(hi - lo, 1), 0, 1), interpolation="nearest")
        a.add_patch(
            plt.Rectangle(
                (s["col_min"] - x0 - 1.5, s["row_min"] - y0 - 1.5),
                s["col_max"] - s["col_min"] + 2, s["row_max"] - s["row_min"] + 2,
                fill=False, ec="red", lw=0.8, ls="-" if s["confidence"] >= 0.5 else ":",
            )
        )  # fmt: skip
        a.set_title(f"#{s['id']}  {s['confidence']:.2f}  ({cy}, {cx})", fontsize=9)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    for a in axes[len(ships) :]:
        a.set_visible(False)
    fig.savefig(out / "ships_overlay.png", dpi=80)
    plt.close(fig)


def main() -> None:
    """Write the summary figures to ``<output-dir>/figures``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("inputs"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    out = args.output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    sel = json.loads((args.output_dir / "band_selection.json").read_text())
    reg = json.loads((args.output_dir / "registration_stats.json").read_text())
    summary = json.loads((args.output_dir / "band_summary.json").read_text())
    rgb = tifffile.imread(args.output_dir / "rgb_image.tif")
    mask = tifffile.imread(args.output_dir / "cloud_mask.tif")
    cloud_stats = json.loads((args.output_dir / "cloud_stats.json").read_text())
    ships = json.loads((args.output_dir / "ships.json").read_text())

    snr_figure(sel, summary, out)
    rgb_figures(rgb, sel, out)
    cloud_figure(rgb, mask, cloud_stats, out)
    ship_figure(rgb, mask, ships, out)
    shift_vs_band_figure(reg, out)
    offset_figure(summary, out)
    offset_by_band_figures(summary, out)
    dynamic_range_figure(args.input_dir, out)
    registration_figures(args.input_dir, sel, reg, out)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
