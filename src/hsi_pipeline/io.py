"""Loading of band metadata and raw band images.

The exact on-disk layout of the challenge inputs is not known until the data
is inspected, so this module is deliberately strict: it accepts a few common
metadata layouts and image containers, and raises a descriptive error for
anything else rather than guessing silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Expected image geometry from the challenge specification.
HEIGHT = 2048
WIDTH = 2048
N_BANDS = 48

#: File extensions treated as headerless 8-bit raster dumps.
RAW_EXTENSIONS = {".raw", ".bin", ".dat", ".img", ""}
TIFF_EXTENSIONS = {".tif", ".tiff"}

# Accepted spellings for each metadata field (first match wins).
_FIELD_ALIASES = {
    "wavelength": (
        "wavelength",
        "wavelength_nm",
        "center_wavelength",
        "central_wavelength",
        "lambda",
    ),
    "gain": ("gain", "gain_coefficient", "scale", "multiplier"),
    "offset": ("offset", "offset_coefficient", "bias", "add_offset"),
    "filename": ("filename", "file", "file_name", "path", "image"),
    "band_id": ("band", "band_id", "band_index", "index", "id", "name"),
}


@dataclass(frozen=True)
class BandInfo:
    """Metadata for a single spectral band.

    Attributes:
        index: Position of the band in the loaded cube (0-based, sorted by wavelength).
        band_id: Identifier as given in the metadata (or the file stem).
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


def natural_key(s: str) -> list:
    """Sort key that orders 'band_2' before 'band_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _pick(entry: dict, field: str, required: bool = True):
    """Return the value of ``field`` from ``entry`` using the accepted aliases."""
    lowered = {k.lower(): v for k, v in entry.items()}
    for alias in _FIELD_ALIASES[field]:
        if alias in lowered:
            return lowered[alias]
    if required:
        raise KeyError(
            f"Metadata entry is missing '{field}' (looked for {_FIELD_ALIASES[field]}); "
            f"available keys: {sorted(entry)}"
        )
    return None


def _normalise_entries(raw: object) -> list[dict]:
    """Flatten the supported metadata layouts into a list of per-band dicts.

    Supported layouts:
        * ``[ {...}, {...} ]``                      — list of band entries
        * ``{"bands": [ {...}, ... ], ...}``        — list under a top-level key
        * ``{"band_01": {...}, "band_02": {...}}``  — mapping of band id -> entry
        * ``{"wavelength": [...], "gain": [...], "offset": [...]}`` — parallel arrays
    """
    if isinstance(raw, list):
        return [dict(e) for e in raw]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported metadata root type: {type(raw).__name__}")

    for key in ("bands", "band_metadata", "metadata", "data"):
        if isinstance(raw.get(key), list):
            return [dict(e) for e in raw[key]]

    # Parallel arrays: every value is a list of equal length.
    if raw and all(isinstance(v, list) for v in raw.values()):
        lengths = {len(v) for v in raw.values()}
        if len(lengths) != 1:
            raise ValueError(f"Parallel metadata arrays have unequal lengths: {lengths}")
        n = lengths.pop()
        return [{k: v[i] for k, v in raw.items()} for i in range(n)]

    # Mapping of band id -> entry.
    if raw and all(isinstance(v, dict) for v in raw.values()):
        return [{"band_id": k, **v} for k, v in raw.items()]

    raise ValueError(f"Unrecognised metadata layout; top-level keys: {sorted(raw)}")


def find_band_files(input_dir: Path) -> list[Path]:
    """List candidate band image files in ``input_dir`` in natural-sort order."""
    allowed = RAW_EXTENSIONS | TIFF_EXTENSIONS
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in allowed
        and not p.name.startswith(".")
        and p.suffix.lower() != ".json"
    ]
    return sorted(files, key=lambda p: natural_key(p.name))


def _match_by_band_number(entries: list[dict], files: list[Path]) -> list[Path] | None:
    """Pair entries to files by the integer in the band id and the file stem.

    Returns None (so the caller falls back) unless every entry has a numeric
    band id matching exactly one file, e.g. ``"24"`` -> ``b24.raw``.
    """
    by_num: dict[int, list[Path]] = {}
    for f in files:
        digits = re.findall(r"\d+", f.stem)
        if len(digits) == 1:
            by_num.setdefault(int(digits[0]), []).append(f)
    paths = []
    for e in entries:
        band_id = _pick(e, "band_id", required=False)
        try:
            matches = by_num.get(int(str(band_id)), [])
        except (TypeError, ValueError):
            return None
        if len(matches) != 1:
            return None
        paths.append(matches[0])
    return paths


def load_metadata(metadata_path: Path, input_dir: Path | None = None) -> list[BandInfo]:
    """Parse ``band_metadata.json`` and pair each entry with its image file.

    Pairing priority: (1) filenames given in the metadata; (2) the band number
    in the metadata key matched to the number in the file stem (``"24"`` ->
    ``b24.raw``, the layout of the challenge data); (3) natural-sort order,
    which requires equal counts.

    Returns:
        Bands sorted by ascending wavelength, with ``index`` assigned in that order.
    """
    input_dir = input_dir or metadata_path.parent
    entries = _normalise_entries(json.loads(metadata_path.read_text()))

    files = find_band_files(input_dir)
    named = [_pick(e, "filename", required=False) for e in entries]
    if all(named):
        paths = [input_dir / str(n) for n in named]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Metadata references missing files: {missing[:5]}")
    elif (by_number := _match_by_band_number(entries, files)) is not None:
        # e.g. metadata key "24" <-> file "b24.raw"
        paths = by_number
    else:
        if len(files) != len(entries):
            raise ValueError(
                f"{len(entries)} metadata entries but {len(files)} image files in {input_dir}; "
                "cannot pair them by order."
            )
        paths = files

    bands = []
    # Every pairing branch yields one path per entry; strict guards against a silent drop.
    for i, (entry, path) in enumerate(zip(entries, paths, strict=True)):
        band_id = _pick(entry, "band_id", required=False)
        bands.append(
            BandInfo(
                index=i,
                band_id=str(band_id) if band_id is not None else path.stem,
                wavelength_nm=float(_pick(entry, "wavelength")),
                gain=float(_pick(entry, "gain")),
                offset=float(_pick(entry, "offset")),
                path=path,
            )
        )

    # Order the cube spectrally; re-index so index == position in the cube.
    bands.sort(key=lambda b: b.wavelength_nm)
    return [
        BandInfo(i, b.band_id, b.wavelength_nm, b.gain, b.offset, b.path)
        for i, b in enumerate(bands)
    ]


def read_band(path: Path, shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    """Read one band as a ``uint8`` array of ``shape``.

    Headerless files must be exactly ``H*W`` bytes; any other size is reported
    rather than guessed at, because a wrong header skip would shift the image.
    """
    suffix = path.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        import tifffile

        arr = tifffile.imread(path)
    elif suffix in RAW_EXTENSIONS:
        expected = shape[0] * shape[1]
        size = path.stat().st_size
        if size != expected:
            raise ValueError(
                f"{path.name}: {size} bytes, expected {expected} for a headerless "
                f"{shape[0]}x{shape[1]} uint8 image (difference {size - expected} bytes)."
            )
        arr = np.fromfile(path, dtype=np.uint8).reshape(shape)
    else:
        raise ValueError(f"Unsupported image format: {path.name}")

    arr = np.squeeze(arr)
    if arr.shape != shape:
        raise ValueError(f"{path.name}: shape {arr.shape}, expected {shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{path.name}: dtype {arr.dtype}, expected uint8")
    return arr


def write_tiff(path: Path, image: np.ndarray, photometric: str = "minisblack") -> None:
    """Write an image as a plain TIFF (uncompressed, no georeferencing).

    Args:
        path: Output path.
        image: ``(H, W)`` or ``(H, W, 3)`` array.
        photometric: ``"rgb"`` for an ``(H, W, 3)`` colour image, else
            ``"minisblack"``.
    """
    import tifffile

    tifffile.imwrite(path, image, photometric=photometric)


def load_cube(bands: list[BandInfo], shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    """Stack all bands into a ``(n_bands, H, W)`` ``uint8`` cube (~200 MB for 48 bands)."""
    cube = np.empty((len(bands), *shape), dtype=np.uint8)
    for b in bands:
        cube[b.index] = read_band(b.path, shape)
    return cube
