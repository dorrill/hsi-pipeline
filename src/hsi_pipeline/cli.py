"""Command-line entry point: ``hsi-pipeline --input-dir inputs --output-dir outputs``."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from .calibration import check_offsets
from .cloud import cloud_mask
from .io import BandInfo, load_cube, load_metadata, write_tiff
from .radiometry import (
    apply_radiometric_correction,
    band_summary,
    destripe_columns,
    validity_masks,
)
from .registration import (
    BandRegistration,
    ResidualStats,
    independent_residual,
    register_cube,
    residual_misregistration,
)
from .rgb import (
    RGB_WINDOWS_NM,
    band_snr,
    channel_weights,
    combined_noise,
    rgb_composite,
    select_rgb,
)
from .ships import CONFIDENT, detect_ships, draw_boxes

log = logging.getLogger("hsi_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments (input/output paths, no-data DN values, gain/offset convention).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder with band files and band_metadata.json",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata path (default: <input-dir>/band_metadata.json)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--nodata",
        type=int,
        nargs="*",
        default=[0],
        help="DN values treated as no-data (default: 0)",
    )
    p.add_argument(
        "--convention",
        choices=["linear", "subtract_offset"],
        default="linear",
        help="Gain/offset formula (default: L = gain*DN + offset)",
    )
    p.add_argument(
        "--keep-metadata-offsets",
        action="store_true",
        help="Report but do not replace offsets that fail the consistency check",
    )
    p.add_argument(
        "--destripe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove per-column offsets after radiometric correction (default: on); "
        "--no-destripe keeps the supplied calibration only",
    )
    p.add_argument(
        "--average-neighbours",
        type=int,
        default=0,
        metavar="K",
        help="Average each RGB band with up to K wavelength neighbours on each side inside "
        "its colour window, inverse-variance weighted (default: 0, single bands)",
    )
    p.add_argument(
        "--reference-band",
        default=None,
        help="Band id (metadata key) to register to (default: middle of the cube)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Processes for band registration (default: min(4, CPUs); ~0.5 GB each)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def registration_report(
    bands: list[BandInfo], reference_index: int, registrations: list[BandRegistration]
) -> dict:
    """Assemble the JSON-serialisable registration statistics.

    Args:
        bands: Band metadata in cube order.
        reference_index: Index of the reference band.
        registrations: Per-band results from ``register_cube``.

    Returns:
        Dict with the reference band and one entry per band.
    """

    def rounded(a):
        return [round(float(v), 3) for v in a]

    rows = []
    for b, r in zip(bands, registrations, strict=True):
        rows.append(
            {
                "band_id": b.band_id,
                "wavelength_nm": b.wavelength_nm,
                "coarse_shift_dy_dx_px": rounded(r.coarse_shift_px),
                # False: too few textured tiles, the band only got the coarse shift
                "field_from_tiles": bool(r.passes) and r.passes[0].n_inliers > 0,
                "field_min_dy_dx_px": rounded(r.field_min_px),
                "field_max_dy_dx_px": rounded(r.field_max_px),
                "spline_smoothing": r.smoothing,
                "passes": [
                    {
                        "tiles_measured": f.n_tiles,
                        "tiles_above_peak_threshold": f.n_good,
                        "tiles_inliers": f.n_inliers,
                        "fit_rms_px": round(f.fit_rms_px, 3),
                        "median_measured_shift_px": round(f.median_update_px, 3),
                    }
                    for f in r.passes
                ],
                "residual_after_registration": _rounded(r.residual),
                "field_blocked_cv_error": _rounded(r.cv),
            }
        )
    return {
        "reference_band": bands[reference_index].band_id,
        "reference_wavelength_nm": bands[reference_index].wavelength_nm,
        "method": "tile phase correlation, robust thin-plate-spline displacement field, "
        "iterative refinement; residual re-measured on non-overlapping tiles after warping; "
        "field error by spatially blocked cross-validation",
        "bands": rows,
    }


def _rounded(stats: ResidualStats) -> dict:
    """JSON-ready residual statistics, floats rounded to 0.001 px."""
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in vars(stats).items()}


def rgb_registration_report(
    cube: np.ndarray,
    bands: list[BandInfo],
    chosen: dict[str, int],
    registrations: list[BandRegistration],
    noise: list[float],
    cloud: np.ndarray,
) -> dict:
    """Registration error between the chosen R, G and B bands, three ways.

    Args:
        cube: ``(n_bands, H, W)`` float32 registered bands.
        bands: Band metadata in cube order.
        chosen: Cube index per colour.
        registrations: Per-band results from ``register_cube``.
        noise: Noise standard deviation per band, in cube order.
        cloud: ``(H, W)`` bool cloud mask.

    Returns:
        The chosen bands; per pair the independent check (clear and cloudy
        windows) and the self-consistency residual; per band the field's CV
        error against the reference.
    """
    pairs = {"R-G": ("R", "G"), "B-G": ("B", "G"), "R-B": ("R", "B")}
    independent, self_consistency = {}, {}
    for name, (a, b) in pairs.items():
        ref, mov = cube[chosen[a]], cube[chosen[b]]
        check = independent_residual(ref, mov, noise[chosen[a]], cloud)
        independent[name] = {region: _rounded(s) for region, s in check.items()}
        self_consistency[name] = _rounded(residual_misregistration(ref, mov))
    return {
        "bands": {c: bands[k].band_id for c, k in chosen.items()},
        "independent_check": independent,
        "self_consistency_residual": self_consistency,
        "field_cv_error_vs_reference": {
            c: _rounded(registrations[k].cv) for c, k in chosen.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline end to end and write outputs to ``--output-dir``.

    Args:
        argv: Argument list without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or args.input_dir / "band_metadata.json"

    # 1. Metadata + raw cube
    bands = load_metadata(metadata_path, args.input_dir)
    log.info(
        "Loaded metadata for %d bands (%.1f–%.1f nm)",
        len(bands),
        bands[0].wavelength_nm,
        bands[-1].wavelength_nm,
    )
    cube_dn = load_cube(bands)
    log.info("Loaded cube %s %s", cube_dn.shape, cube_dn.dtype)

    # 2. Invalid pixels + radiometric correction (before any band selection)
    valid, saturated = validity_masks(cube_dn, tuple(args.nodata))
    bands, offset_checks = check_offsets(
        cube_dn, bands, valid, args.convention, repair=not args.keep_metadata_offsets
    )
    corrected = apply_radiometric_correction(cube_dn, bands, valid, args.convention)
    stripe_amplitude = (
        destripe_columns(corrected, saturated, args.workers) if args.destripe else None
    )
    summary = band_summary(cube_dn, corrected, bands, valid, saturated, offset_checks)
    if stripe_amplitude is not None:
        for row, amp in zip(summary, stripe_amplitude, strict=True):
            row["stripe_amplitude"] = round(float(amp), 4)
    (args.output_dir / "band_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", args.output_dir / "band_summary.json")

    # 3. SNR per band and RGB choice, before registration: resampling smooths
    #    noise and would inflate the SNR of every band but the reference.
    snr = band_snr(corrected, bands, valid, saturated)
    chosen = select_rgb(snr)
    weights = channel_weights(snr, chosen, args.average_neighbours)
    selection = {
        "snr_definition": "noise-free scene std / noise std (Immerkaer Laplacian, "
        f"flattest 50% of pixels), on {'destriped' if args.destripe else 'calibrated'}, "
        "unregistered bands",
        "windows_nm": RGB_WINDOWS_NM,
        "chosen": {c: snr[k].band_id for c, k in chosen.items()},
        "channels": {
            c: {
                "weights": {snr[i].band_id: round(v, 4) for i, v in w.items()},
                # independent-noise estimate; resampling and shared stripes change it
                "noise_reduction_vs_chosen": round(
                    snr[chosen[c]].noise / combined_noise(snr, w), 3
                ),
            }
            for c, w in weights.items()
        },
        "bands": [{k: round(v, 4) if isinstance(v, float) else v for k, v in vars(s).items()}
                  for s in snr],
    }  # fmt: skip
    (args.output_dir / "band_selection.json").write_text(json.dumps(selection, indent=2))
    log.info(
        "RGB bands: %s",
        ", ".join(f"{c}={snr[k].band_id} ({snr[k].wavelength_nm:.1f} nm, SNR {snr[k].snr:.1f})"
                  for c, k in chosen.items()),
    )  # fmt: skip
    if args.average_neighbours:
        log.info(
            "RGB channels: %s",
            "; ".join(f"{c} = " + " + ".join(f"{v:.2f}*b{snr[i].band_id}" for i, v in w.items())
                      for c, w in weights.items()),
        )  # fmt: skip

    # 4. Co-registration of every band, in place. The reference is the green
    #    band, so one RGB channel is never resampled.
    ids = [b.band_id for b in bands]
    ref_index = chosen["G"] if args.reference_band is None else ids.index(args.reference_band)
    log.info(
        "Registering %d bands to band %s (%.1f nm)",
        len(bands),
        ids[ref_index],
        bands[ref_index].wavelength_nm,
    )
    registrations = register_cube(corrected, valid, ref_index, args.workers)

    # 5. One no-data footprint for both outputs: pixels valid in every band the
    #    RGB or the cloud mask uses. The cloud mask also needs the longest band
    #    (the closest the capture gets to NIR), which registration shifts by
    #    ~25 px, so without this the mask would read "clear" where it is unknown.
    red_edge = int(np.argmax([b.wavelength_nm for b in bands]))
    green = chosen["G"]
    used = sorted({i for w in weights.values() for i in w} | {green, red_edge})
    footprint = valid[used].all(axis=0)

    mask, cloud = cloud_mask(corrected[green], corrected[red_edge], footprint)
    write_tiff(args.output_dir / "cloud_mask.tif", mask)
    cloud_report = {
        "bands": {"green": bands[green].band_id, "red_edge": bands[red_edge].band_id},
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in vars(cloud).items()},
        "no_data_pixels": int((~footprint).sum()),
    }
    (args.output_dir / "cloud_stats.json").write_text(json.dumps(cloud_report, indent=2))
    log.info(
        "Wrote %s: %.1f %% cloud, %d objects (ratio threshold %.2f)",
        args.output_dir / "cloud_mask.tif",
        100 * cloud.cloud_fraction,
        cloud.n_objects,
        cloud.ratio_threshold,
    )

    # 6. Registration statistics; the RGB pairs are checked separately on clear
    #    and cloudy windows, so they come after the cloud mask.
    stats = registration_report(bands, ref_index, registrations)
    noise = [s.noise for s in snr]
    stats["rgb"] = rgb_registration_report(
        corrected, bands, chosen, registrations, noise, mask.astype(bool)
    )
    (args.output_dir / "registration_stats.json").write_text(json.dumps(stats, indent=2))
    for pair, r in stats["rgb"]["independent_check"].items():
        log.info(
            "RGB %s misregistration (independent check): clear median %.3f / p95 %.3f px "
            "(%d windows), cloud median %.3f px",
            pair, r["clear"]["median_px"], r["clear"]["p95_px"], r["clear"]["n_tiles"],
            r["cloud"]["median_px"],
        )  # fmt: skip

    # 7. RGB composite (per-channel percentile stretch; see docs/instrument_notes.md,
    #    "Colour balance", for why a white balance on the clouds was not kept)
    rgb = rgb_composite(corrected, weights, footprint)
    write_tiff(args.output_dir / "rgb_image.tif", rgb, photometric="rgb")
    log.info(
        "Wrote %s (%.1f %% of pixels valid)",
        args.output_dir / "rgb_image.tif",
        100 * footprint.mean(),
    )

    # 8. Ships in the cloud-free open water, on the same grid as the RGB image;
    #    written after the RGB and cloud mask so that it cannot prevent them
    ships, search = detect_ships(corrected[red_edge], mask.astype(bool), footprint)
    ship_report = {
        "band": bands[red_edge].band_id,
        "bbox_convention": "pixel rows/columns of rgb_image.tif, inclusive",
        "search_fraction": round(search.search_fraction, 4),
        "noise": round(search.noise, 4),
        "ships": [{"id": i, **_rounded(s)} for i, s in enumerate(ships, 1)],
    }
    (args.output_dir / "ships.json").write_text(json.dumps(ship_report, indent=2))
    write_tiff(args.output_dir / "ships_overlay.tif", draw_boxes(rgb, ships), photometric="rgb")
    log.info(
        "Wrote %s and ships_overlay.tif: %d detections, %d with confidence >= %.1f "
        "(searched %.1f %% of the frame)",
        args.output_dir / "ships.json",
        len(ships),
        sum(s.confidence >= CONFIDENT for s in ships),
        CONFIDENT,
        100 * search.search_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
