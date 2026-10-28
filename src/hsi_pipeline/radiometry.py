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

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import ndimage

from .io import BandInfo

if TYPE_CHECKING:  # calibration imports this module, so import for typing only
    from .calibration import OffsetCheck

SATURATION_DN = 255

#: Running-median window (columns) separating column stripes from scene content.
#: The stripes are single columns; 15 is wide enough for the running median to
#: ignore them and narrow enough to follow the column-to-column scene profile.
STRIPE_WINDOW_COLUMNS = 15

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


def correct_dn(
    dn: np.ndarray | float,
    gain: np.ndarray | float,
    offset: np.ndarray | float,
    convention: Convention = "linear",
) -> np.ndarray | float:
    """Apply the gain/offset formula elementwise (shapes broadcast).

    Args:
        dn: Raw digital number(s).
        gain: Multiplicative coefficient(s).
        offset: Additive coefficient(s), in corrected units for ``"linear"`` and
            in DN for ``"subtract_offset"``.
        convention: Which formula to apply (see module docstring).

    Returns:
        Corrected value(s), same broadcast shape as the inputs.

    Raises:
        ValueError: If ``convention`` is unknown.
    """
    if convention == "linear":
        return gain * dn + offset
    if convention == "subtract_offset":
        return gain * (dn - offset)
    raise ValueError(f"Unknown convention: {convention!r}")


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
        out[b.index] = correct_dn(dn, b.gain, b.offset, convention)

    if valid is not None:
        out[~valid] = np.nan
    return out


def destripe_columns(cube: np.ndarray) -> np.ndarray:
    """Remove column fixed-pattern noise from each band, in place.

    The capture has single-column stripes of ~2 DN (twice the random noise),
    shared by all bands (r ~ 0.91), i.e. a detector-column offset. For each band
    the stripe pattern is the column-median profile minus its running median over
    ``STRIPE_WINDOW_COLUMNS`` columns; the pattern is subtracted from every row.
    Medians over all rows make the estimate insensitive to scene content, and
    scene structure wider than the window is left untouched (so is any banding
    wider than the window).

    Args:
        cube: ``(n_bands, H, W)`` float32 corrected values, NaN at no-data.
            Modified in place.

    Returns:
        ``(n_bands,)`` float64 standard deviation of the removed pattern per
        band, in corrected units.
    """
    amplitude = np.empty(cube.shape[0])
    for i in range(cube.shape[0]):
        profile = np.nanmedian(cube[i], axis=0)
        profile = np.where(np.isfinite(profile), profile, np.nanmedian(profile))
        pattern = profile - ndimage.median_filter(profile, STRIPE_WINDOW_COLUMNS, mode="nearest")
        cube[i] -= pattern[None, :].astype(cube.dtype)
        amplitude[i] = pattern.std()
    return amplitude


def band_summary(
    cube_dn: np.ndarray,
    corrected: np.ndarray,
    bands: list[BandInfo],
    valid: np.ndarray,
    saturated: np.ndarray,
    offset_checks: list[OffsetCheck] | None = None,
) -> list[dict]:
    """Per-band diagnostics used to sanity-check loading and correction.

    Args:
        cube_dn: ``(n_bands, H, W)`` uint8 raw DN.
        corrected: ``(n_bands, H, W)`` float32 corrected values, NaN at no-data.
        bands: Band metadata as used for the correction (``offset`` is the applied one).
        valid: ``(n_bands, H, W)`` bool, False for no-data pixels.
        saturated: ``(n_bands, H, W)`` bool, True where DN == 255.
        offset_checks: Optional per-band ``calibration.OffsetCheck`` results to merge in.

    Returns:
        One JSON-serialisable dict per band, in cube order.
    """
    checks = offset_checks or [None] * len(bands)
    rows = []
    for b, check in zip(bands, checks, strict=True):
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
        if check is not None:
            rows[-1] |= {
                "offset_metadata": check.offset_metadata,
                "offset_z": round(check.offset_z, 2),
                "offset_repaired": check.repaired,
                "continuity_residual": round(check.continuity_residual, 4),
            }
    return rows
