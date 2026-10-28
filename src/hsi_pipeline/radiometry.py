"""Radiometric correction and invalid-pixel handling.

Correction model (assumed, to be confirmed against the metadata):

    L = gain * DN + offset

where DN is the raw 8-bit digital number and L the corrected value. This is
the usual linear calibration convention (e.g. Landsat/Sentinel L1 rescaling).
The alternative ``L = gain * (DN - offset)`` is available via
``convention="subtract_offset"`` in case the metadata uses a dark-level offset.

Invalid pixels:
    * DN values listed in ``nodata_values`` (default: 0) are treated as fill /
      no-data and become NaN in the corrected cube.
    * DN == 255 is flagged as *saturated*. Saturated pixels are kept (they
      carry real, if clipped, signal — bright cloud tops often saturate) but
      are reported so later stages can exclude them from SNR and registration
      statistics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .io import BandInfo

SATURATION_DN = 255

Convention = Literal["linear", "subtract_offset"]


def validity_masks(
    cube: np.ndarray, nodata_values: tuple[int, ...] = (0,)
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-band boolean masks ``(valid, saturated)`` for a uint8 cube.

    ``valid`` is False where the DN is a no-data value. ``saturated`` is True
    where DN == 255.
    """
    valid = ~np.isin(cube, nodata_values)
    saturated = cube == SATURATION_DN
    return valid, saturated


def apply_radiometric_correction(
    cube: np.ndarray,
    bands: list[BandInfo],
    valid: np.ndarray | None = None,
    convention: Convention = "linear",
) -> np.ndarray:
    """Convert a uint8 DN cube to corrected float32 values, band by band.

    Args:
        cube: ``(n_bands, H, W)`` uint8 array, ordered as ``bands``.
        bands: Band metadata providing gain and offset for each band.
        valid: Optional boolean mask of the same shape; invalid pixels become NaN.
        convention: Which gain/offset formula to apply (see module docstring).

    Returns:
        ``(n_bands, H, W)`` float32 array (~800 MB for the full capture).
    """
    if cube.shape[0] != len(bands):
        raise ValueError(f"Cube has {cube.shape[0]} bands but metadata has {len(bands)}")

    out = np.empty(cube.shape, dtype=np.float32)
    for b in bands:
        dn = cube[b.index].astype(np.float32)
        if convention == "linear":
            out[b.index] = b.gain * dn + b.offset
        elif convention == "subtract_offset":
            out[b.index] = b.gain * (dn - b.offset)
        else:
            raise ValueError(f"Unknown convention: {convention!r}")

    if valid is not None:
        out[~valid] = np.nan
    return out


def band_summary(
    cube_dn: np.ndarray,
    corrected: np.ndarray,
    bands: list[BandInfo],
    valid: np.ndarray,
    saturated: np.ndarray,
) -> list[dict]:
    """Per-band diagnostics used to sanity-check loading and correction."""
    rows = []
    for b in bands:
        i = b.index
        vals = corrected[i][valid[i]]
        rows.append(
            {
                "index": i,
                "band_id": b.band_id,
                "wavelength_nm": b.wavelength_nm,
                "gain": b.gain,
                "offset": b.offset,
                "dn_min": int(cube_dn[i].min()),
                "dn_max": int(cube_dn[i].max()),
                "valid_fraction": float(valid[i].mean()),
                "saturated_fraction": float(saturated[i].mean()),
                "corrected_mean": float(vals.mean()) if vals.size else None,
                "corrected_std": float(vals.std()) if vals.size else None,
            }
        )
    return rows
