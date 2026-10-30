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

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import ndimage

from .io import BandInfo

if TYPE_CHECKING:  # calibration imports this module, so import for typing only
    from .calibration import OffsetCheck

SATURATION_DN = 255

#: Running-median window (columns) separating column stripes from scene content
#: along each row. The stripes are single columns; 15 is wide enough for the
#: running median to ignore them and narrow enough to follow the scene.
STRIPE_WINDOW_COLUMNS = 15

#: Estimation passes. On the real capture (split-half test, band 27) the
#: residual stripe std is 0.16 after one pass, 0.094 after two, 0.083 after three.
STRIPE_PASSES = 2

#: Tukey bisquare tuning constant (95 % efficiency for Gaussian residuals) and
#: fixed number of reweighting iterations for :func:`column_linear_fit`.
BISQUARE_C = 4.685
ROBUST_ITERATIONS = 10
#: MAD to standard deviation for Gaussian residuals.
_MAD_TO_SIGMA = 1.4826

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


def column_offsets(band: np.ndarray, saturated: np.ndarray | None = None) -> np.ndarray:
    """Per-column additive offset (stripe pattern) of one band.

    Each row is smoothed with a running median over ``STRIPE_WINDOW_COLUMNS``
    columns, which follows the scene but ignores single-column stripes; the
    stripe of a column is the median of its row residuals. Removing the scene
    row by row first means a column's estimate is not biased by what that column
    happens to see (a coastline, a cloud), and the median down ~2000 rows
    rejects the residuals left at edges. The stripes themselves pull the
    running median, so the estimate is refined on the destriped band
    (``STRIPE_PASSES``).

    No-data pixels are filled with the nearest valid value before smoothing and
    excluded from the column median, as are saturated pixels (clipping removes
    the column's offset). Two limits: only the part of the pattern that varies
    faster than the window is identifiable (its local mean is indistinguishable
    from scene), and a scene edge that stays in the same column for most rows
    biases the columns next to it.

    Args:
        band: ``(H, W)`` float32 corrected values, NaN at no-data.
        saturated: Optional ``(H, W)`` bool, True where DN == 255.

    Returns:
        ``(W,)`` float64 offsets in corrected units, with zero mean over the
        columns that have usable pixels (so the band mean is unchanged) and 0
        in columns that have none.
    """
    valid = np.isfinite(band)
    filled = band
    if not valid.all():
        # Nearest valid pixel, so windows next to no-data behave like the frame edge.
        nearest = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = band[tuple(nearest)]
    usable = valid if saturated is None else valid & ~saturated
    cols = usable.any(axis=0)
    offsets = np.zeros(band.shape[1])
    for _ in range(STRIPE_PASSES):
        current = filled - offsets[None, :].astype(filled.dtype)
        smooth = ndimage.median_filter(current, size=(1, STRIPE_WINDOW_COLUMNS), mode="mirror")
        residual = np.where(usable, current - smooth, np.nan)
        offsets[cols] += np.nanmedian(residual[:, cols], axis=0)
    offsets[cols] -= offsets[cols].mean()
    return offsets


def destripe_columns(
    cube: np.ndarray, saturated: np.ndarray | None = None, workers: int = 1
) -> np.ndarray:
    """Remove column fixed-pattern noise from each band, in place.

    The capture has single-column stripes of ~2 DN (twice the random noise),
    shared by all bands (r ~ 0.91), i.e. a detector-column offset. Each band's
    :func:`column_offsets` are subtracted from every row. Scene structure and
    banding wider than the window are left untouched. This is an empirical
    correction beyond the supplied gain/offset calibration.

    Args:
        cube: ``(n_bands, H, W)`` float32 corrected values, NaN at no-data.
            Modified in place.
        saturated: Optional ``(n_bands, H, W)`` bool, True where DN == 255.
        workers: Threads; the median filter releases the GIL, and the result
            does not depend on the number of threads.

    Returns:
        ``(n_bands,)`` float64 standard deviation of the removed offsets per
        band, in corrected units.
    """

    def one(i: int) -> float:
        offsets = column_offsets(cube[i], None if saturated is None else saturated[i])
        cube[i] -= offsets[None, :].astype(cube.dtype)
        return float(offsets.std())

    with ThreadPoolExecutor(max(workers, 1)) as pool:
        return np.array(list(pool.map(one, range(cube.shape[0]))))


def column_linear_fit(
    y: np.ndarray, x: np.ndarray, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Robust straight-line fit ``y = intercept + slope * x`` in every column.

    Diagnostic for cross-track (column-dependent) gain and offset errors: with
    ``x`` a scene proxy shared by all bands, a column trend in ``intercept`` is
    an additive error and one in ``slope`` a multiplicative one. Iteratively
    reweighted least squares with Tukey bisquare weights, vectorised over the
    columns; the scale is the MAD of the current residuals per column, so
    clouds, saturation and surfaces with an unusual spectrum get zero weight.

    Args:
        y: ``(H, W)`` float values of one band.
        x: ``(H, W)`` float scene proxy on the same grid.
        usable: ``(H, W)`` bool, pixels to fit (finite in both).

    Returns:
        ``(W,)`` float64 intercepts and slopes; NaN in columns with fewer than
        two usable pixels.
    """
    cols = usable.sum(axis=0) >= 2
    ok = usable[:, cols]
    x = np.where(ok, x[:, cols], 0.0).astype(np.float64)
    y = np.where(ok, y[:, cols], 0.0).astype(np.float64)
    w = ok.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):  # all weights 0: NaN result
        for _ in range(ROBUST_ITERATIONS):
            sw = w.sum(axis=0)
            mx, my = (w * x).sum(axis=0) / sw, (w * y).sum(axis=0) / sw
            dx = x - mx
            b = (w * dx * (y - my)).sum(axis=0) / (w * dx**2).sum(axis=0)
            a = my - b * mx
            r = y - a - b * x
            scale = _MAD_TO_SIGMA * np.nanmedian(np.where(ok, np.abs(r), np.nan), axis=0)
            u = r / (BISQUARE_C * scale)
            w = np.where(ok & (np.abs(u) < 1), (1 - u**2) ** 2, 0.0)
    intercept, slope = np.full(usable.shape[1], np.nan), np.full(usable.shape[1], np.nan)
    intercept[cols], slope[cols] = a, b
    return intercept, slope


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
