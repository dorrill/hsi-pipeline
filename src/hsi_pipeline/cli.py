"""Command-line entry point: ``hsi-pipeline --input-dir inputs --output-dir outputs``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .io import load_cube, load_metadata
from .radiometry import apply_radiometric_correction, band_summary, validity_masks

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
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


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
    corrected = apply_radiometric_correction(cube_dn, bands, valid, args.convention)
    summary = band_summary(cube_dn, corrected, bands, valid, saturated)
    (args.output_dir / "band_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", args.output_dir / "band_summary.json")

    # 3. Co-registration            -> registration_stats.json   (TODO)
    # 4. SNR-based RGB band choice  -> rgb_image.tif             (TODO)
    # 5. Cloud mask                 -> cloud_mask.tif            (TODO)
    # 6. Optional ship detection    -> ships.png / ships.json    (TODO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
