"""Loading of band metadata and raw band images, and writing of TIFF outputs.

Input layout (the challenge format):

* ``band_metadata.json``: ``{"<n>": {"wavelength": nm, "gain": g, "offset": o}, ...}``
* one headerless, row-major uint8 file ``b<n>.raw`` per metadata key ``n``.

Anything else (a missing or extra band file, a missing field, a file of the
wrong size) raises an error rather than being guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

#: Expected image geometry from the challenge specification.
HEIGHT = 2048
WIDTH = 2048

#: Required fields of each metadata entry.
_FIELDS = ("wavelength", "gain", "offset")


@dataclass(frozen=True)
class BandInfo:
    """Metadata for a single spectral band.

    Attributes:
        index: Position of the band in the loaded cube (0-based, sorted by wavelength).
        band_id: Metadata key, e.g. ``"24"`` for ``b24.raw``.
        wavelength_nm: Band centre wavelength in nanometres.
        gain: Multiplicative radiometric coefficient.
        offset: Additive radiometric coefficient.
        path: Path to the band's image file.
    """

    index: int
    band_id: str
    wavelength_nm: float
    gain: float
    offset: float
    path: Path


def load_metadata(metadata_path: Path, input_dir: Path | None = None) -> list[BandInfo]:
    """Parse ``band_metadata.json`` and pair each entry with its ``b<n>.raw`` file.

    Args:
        metadata_path: Path to ``band_metadata.json``.
        input_dir: Folder with the band files (default: the metadata's folder).

    Returns:
        Bands sorted by ascending wavelength, with ``index`` assigned in that
        order. Band number is not monotonic in wavelength in the challenge
        data, so the cube order differs from the key order.

    Raises:
        ValueError: If the JSON is not a mapping of band id to entry, or the
            band files and metadata keys do not match one to one.
        KeyError: If an entry lacks a required field.
        FileNotFoundError: If a band file is missing.
    """
    input_dir = input_dir or metadata_path.parent
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise ValueError(f"{metadata_path.name}: expected {{band_id: {{{', '.join(_FIELDS)}}}}}")

    bands = []
    for band_id, entry in raw.items():
        missing = [f for f in _FIELDS if f not in entry]
        if missing:
            raise KeyError(f"Band {band_id}: missing {missing}; has {sorted(entry)}")
        path = input_dir / f"b{band_id}.raw"
        if not path.is_file():
            raise FileNotFoundError(f"Band {band_id}: {path} not found")
        bands.append((band_id, entry, path))

    # A band file without metadata would be silently ignored; most likely the
    # wrong folder or a truncated metadata file, so refuse.
    expected = {p for _, _, p in bands}
    extra = sorted(p.name for p in input_dir.glob("b*.raw") if p not in expected)
    if extra:
        raise ValueError(f"Band files without metadata entries: {extra[:5]}")

    bands.sort(key=lambda b: float(b[1]["wavelength"]))
    return [
        BandInfo(
            index=i,
            band_id=band_id,
            wavelength_nm=float(entry["wavelength"]),
            gain=float(entry["gain"]),
            offset=float(entry["offset"]),
            path=path,
        )
        for i, (band_id, entry, path) in enumerate(bands)
    ]


def read_band(path: Path, shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    """Read one headerless band file.

    Args:
        path: Path to a ``.raw`` file.
        shape: Expected ``(H, W)``.

    Returns:
        ``(H, W)`` uint8 DN.

    Raises:
        ValueError: If the file is not exactly ``H*W`` bytes. The size is
            reported rather than guessed at, because a wrong header skip would
            shift the image.
    """
    expected = shape[0] * shape[1]
    size = path.stat().st_size
    if size != expected:
        raise ValueError(
            f"{path.name}: {size} bytes, expected {expected} for a headerless "
            f"{shape[0]}x{shape[1]} uint8 image (difference {size - expected} bytes)."
        )
    return np.fromfile(path, dtype=np.uint8).reshape(shape)


def load_cube(bands: list[BandInfo], shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    """Stack all bands into a cube in ``bands`` order.

    Args:
        bands: Band metadata from :func:`load_metadata`.
        shape: Expected ``(H, W)`` of every band.

    Returns:
        ``(n_bands, H, W)`` uint8 DN (~200 MB for 48 bands at 2048 x 2048).
    """
    cube = np.empty((len(bands), *shape), dtype=np.uint8)
    for b in bands:
        cube[b.index] = read_band(b.path, shape)
    return cube


def write_tiff(path: Path, image: np.ndarray, photometric: str = "minisblack") -> None:
    """Write an image as a plain TIFF (uncompressed, no georeferencing).

    Args:
        path: Output path.
        image: ``(H, W)`` or ``(H, W, 3)`` array.
        photometric: ``"rgb"`` for an ``(H, W, 3)`` colour image, else
            ``"minisblack"``.
    """
    tifffile.imwrite(path, image, photometric=photometric)
