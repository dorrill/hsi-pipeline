"""Command-line entry point: ``hsi-pipeline --input-dir inputs --output-dir outputs``."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .calibration import check_offsets
from .io import BandInfo, load_cube, load_metadata
from .radiometry import (
    apply_radiometric_correction,
    band_summary,
    destripe_columns,
    validity_masks,
)
from .registration import BandRegistration, ResidualStats, register_cube

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
        "--no-destripe",
        action="store_true",
        help="Skip the column destriping applied after radiometric correction",
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

    def stats(r: ResidualStats) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in vars(r).items()}

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
                "residual_after_registration": stats(r.residual),
                "field_blocked_cv_error": stats(r.cv),
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
    stripe_amplitude = None if args.no_destripe else destripe_columns(corrected)
    summary = band_summary(cube_dn, corrected, bands, valid, saturated, offset_checks)
    if stripe_amplitude is not None:
        for row, amp in zip(summary, stripe_amplitude, strict=True):
            row["stripe_amplitude"] = round(float(amp), 4)
    (args.output_dir / "band_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", args.output_dir / "band_summary.json")

    # 3. Co-registration (every band to one reference, in place)
    ids = [b.band_id for b in bands]
    ref_index = len(bands) // 2 if args.reference_band is None else ids.index(args.reference_band)
    log.info(
        "Registering %d bands to band %s (%.1f nm)",
        len(bands),
        ids[ref_index],
        bands[ref_index].wavelength_nm,
    )
    registrations = register_cube(corrected, valid, ref_index, args.workers)
    stats = registration_report(bands, ref_index, registrations)
    (args.output_dir / "registration_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "Wrote %s (worst band residual: median %.3f px, p95 %.3f px)",
        args.output_dir / "registration_stats.json",
        max(r.residual.median_px for r in registrations),
        max(r.residual.p95_px for r in registrations),
    )

    # 4. SNR-based RGB band choice  -> rgb_image.tif             (TODO)
    # 5. Cloud mask                 -> cloud_mask.tif            (TODO)
    # 6. Optional ship detection    -> ships.png / ships.json    (TODO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
