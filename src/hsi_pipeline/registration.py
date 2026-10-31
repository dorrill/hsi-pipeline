"""Band-to-band co-registration: local phase correlation and a smooth displacement field.

What the data shows (see README, "Co-registration"):
    The capture behaves like a push-broom sensor whose bands are separate strips
    of detector rows, so each band sees a ground point at a different time.
    Relative to a reference band, displacements reach ~50 px, grow linearly
    with band *number* (detector position, not wavelength) and jump by
    ~(19, 11) px between bands 44 and 45 (a detector-segment gap). They vary
    smoothly over the frame (linearly across the swath, non-monotonically
    along the rows, i.e. in time), and in the last ~450 rows the bands of the
    other segment are stretched along-track by up to ~10 %. A translation or
    affine model leaves errors of 0.3-1.3 px and a global cubic fails near the
    bottom edge, so the field is modelled non-parametrically.

Model:
    For band ``k``, ``band_k(p) = ref(p - s_k(p))`` with ``s_k = (dy, dx)`` a
    smooth dense field. The registered band is ``band_k(p + s_k(p))``, sampled
    with a cubic spline.

Estimation, per band:
    1. Coarse full-frame shift, so that each tile pair below overlaps.
    2. Local shifts on a grid of ``TILE_PX`` tiles by phase correlation:
       mean-filled no-data, a Hann window against FFT edge effects, the
       normalised cross-power spectrum (insensitive to the band's gain, offset
       and contrast, which differ strongly across 470-729 nm), a Gaussian
       low-pass that down-weights noise-dominated frequencies, and subpixel
       refinement with the upsampled DFT of Guizar-Sicairos et al., Opt. Lett.
       33, 156 (2008).
    3. Tiles whose correlation peak is below ``MIN_PEAK`` are dropped: open
       water and other textureless areas give noise peaks, not displacements.
    4. A smoothing thin-plate spline through the remaining tiles, iteratively
       rejecting tiles more than ``OUTLIER_MADS`` robust standard deviations
       from it (moving cloud edges, mismatches). It follows local structure and
       bridges the water gaps with minimum bending.
    5. Iteration: the band is warped with the field and tiles are re-measured
       on the warped band; the remaining shift is composed into the field.
       Tiles that decorrelated because of local stretch correlate again once
       most of the stretch is removed.

Evaluation:
    * ``residual``: shifts re-measured on non-overlapping tiles between the
      registered band and the reference, with the estimator that drove the fit.
      A field flexible enough to follow the tiles absorbs their measurement
      errors, so this is a self-consistency check and optimistic.
    * ``cv``: spatially blocked cross-validation of the field against the total
      tile shifts (field at the tile centre plus what is still measured there).
      It is not flattered by a spline that passes through its own data, and it
      covers predictions into gaps such as open water, so it is pessimistic
      where tiles are dense.
    * ``independent_residual``: Lucas-Kanade, a gradient method the fit did not
      use, on smaller windows offset from the fit's tiles, split into clear and
      cloudy windows. This is the accuracy estimate for the output image.
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
from scipy import fft, interpolate, ndimage

log = logging.getLogger(__name__)

#: Subpixel refinement factor: shifts are resolved to 1/UPSAMPLE px, well below
#: the accuracy achievable from 8-bit data.
UPSAMPLE = 100

#: Gaussian low-pass applied to the normalised cross-power spectrum, in
#: cycles/px (equivalent to ~2 px spatial smoothing). On the real data it raised
#: the median tile peak from ~0.1 to ~0.8 and lowered the field-fit residual.
LOWPASS_SIGMA_CPP = 0.08

#: Tile size and step for local shift estimation. 128 px tiles hold enough
#: texture on land and cloud while resolving the along-track variation of the
#: field (period of several hundred rows); 64 px steps give overlapping tiles.
TILE_PX = 128
TILE_STEP_PX = 64

#: Minimum normalised correlation peak for a tile to be used. Land and cloud
#: tiles on the real data peak at 0.6-0.9; open-water tiles at < 0.3.
MIN_PEAK = 0.5

#: Candidate smoothings of the thin-plate spline through the tile shifts
#: (coordinates normalised by the image size). The right value depends on how
#: noisy the tiles are, so it is chosen per band and pass by blocked
#: cross-validation. On the real data held-out error falls from 1e-2 to 1e-5
#: (sharp structure near the bottom edge) and is flat below; on noisy synthetic
#: tiles the large values win, where the spline tends to an affine fit.
SMOOTHING_CANDIDATES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


#: Spacing of the grid on which the spline is evaluated before cubic
#: interpolation to every pixel.
FIELD_GRID_PX = 32

#: Tiles further than this many robust standard deviations from the fit are
#: rejected in each iteration.
OUTLIER_MADS = 4.0
_FIT_ITERATIONS = 5
#: Floor on the rejection radius, so a near-perfect fit does not reject everything.
_MIN_REJECT_PX = 0.05
#: Fewer usable tiles than this and the field is not updated.
MIN_GOOD_TILES = 10

#: Field refinement stops when the median remaining tile shift is below this
#: (px) or after MAX_PASSES passes.
CONVERGED_PX = 0.05
MAX_PASSES = 4

#: Block size for spatially blocked cross-validation of the field. Larger than
#: the tile size, so a held-out tile's overlapping neighbours are mostly held
#: out with it and cannot leak its content into the training set.
CV_BLOCK_PX = 384

#: Independent check (:func:`independent_residual`): Lucas-Kanade on windows of
#: this size, whose centres are offset from the fit's tile centres.
CHECK_WINDOW_PX = 64
#: A window is used only if its standard deviation is at least this many times
#: the reference band's noise, i.e. it holds scene structure. Open water holds
#: mostly noise and column banding, and registration shifts the banding by a few
#: px between bands, so the check would measure the banding instead.
CHECK_MIN_CONTRAST = 4.0
#: Windows more than this fraction cloud are reported separately: clouds move
#: and sit above the ground the field was fitted to.
CHECK_CLOUD_FRACTION = 0.2
#: Margin of the moving window (px), the largest shift the check can measure.
_CHECK_PAD_PX = 4
_CHECK_ITERATIONS = 10

MAD_TO_SIGMA = 1.4826


# --------------------------------------------------------------------------- #
# Shift estimation
# --------------------------------------------------------------------------- #


@functools.cache
def _hann(shape: tuple[int, int]) -> np.ndarray:
    """2-D Hann window, cached because every tile has the same shape."""
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _prepare(image: np.ndarray) -> np.ndarray:
    """Mean-fill NaNs, remove the mean and apply a 2-D Hann window.

    Args:
        image: ``(H, W)`` float image, NaN at no-data.

    Returns:
        ``(H, W)`` float64 windowed, zero-mean image.
    """
    img = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(img)
    mean = img[finite].mean() if finite.any() else 0.0
    img = np.where(finite, img - mean, 0.0)
    return img * _hann(img.shape)


def _upsampled_dft(
    data: np.ndarray, region_size: int, upsample: int, offsets: np.ndarray
) -> np.ndarray:
    """Inverse DFT of ``data`` on an upsampled grid in a small region.

    Evaluates ``sum_k data[k] exp(+2 pi i k x / N)`` at
    ``x = (arange(region_size) - offsets) / upsample`` along each axis, by two
    matrix products instead of a zero-padded FFT.

    Args:
        data: ``(H, W)`` complex spectrum.
        region_size: Number of output samples per axis.
        upsample: Upsampling factor.
        offsets: ``(2,)`` position of output sample 0, in upsampled pixels.

    Returns:
        ``(region_size, region_size)`` complex array.
    """
    for n, off in zip(data.shape[::-1], offsets[::-1], strict=True):
        kernel = np.exp(
            2j * np.pi * (np.arange(region_size) - off)[:, None] * fft.fftfreq(n, upsample)
        )
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample: int = UPSAMPLE,
    lowpass_cpp: float | None = LOWPASS_SIGMA_CPP,
) -> tuple[np.ndarray, float]:
    """Estimate the translation of ``moving`` relative to ``reference``.

    Args:
        reference: ``(H, W)`` float image, NaN at no-data.
        moving: ``(H, W)`` float image of the same shape, NaN at no-data.
        upsample: Subpixel refinement factor (precision ``1 / upsample`` px).
        lowpass_cpp: Sigma of the Gaussian spectral weight in cycles/px, or
            None for plain phase correlation.

    Returns:
        ``shift_px``: ``(2,)`` float64 ``(dy, dx)`` such that
        ``moving(y, x) ~ reference(y - dy, x - dx)``; and the correlation peak
        normalised to [0, 1] (1 for identical images up to translation).

    Raises:
        ValueError: If the shapes differ.
    """
    if reference.shape != moving.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {moving.shape}")
    shape = np.array(reference.shape)

    cross = fft.fft2(_prepare(moving)) * np.conj(fft.fft2(_prepare(reference)))
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, np.finfo(float).tiny)
    if lowpass_cpp:
        fy = fft.fftfreq(shape[0])[:, None]
        fx = fft.fftfreq(shape[1])[None, :]
        cross *= np.exp(-(fy**2 + fx**2) / (2 * lowpass_cpp**2))
    weight = np.abs(cross).sum()

    corr = np.abs(fft.ifft2(cross))
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape), dtype=np.float64)
    wrap = peak > shape // 2  # indices past the midpoint are negative shifts
    peak[wrap] -= shape[wrap]

    # Refine on a 1.5 px x 1.5 px grid centred on the integer peak.
    region = int(np.ceil(upsample * 1.5))
    centre = region // 2
    local = np.abs(_upsampled_dft(cross, region, upsample, centre - peak * upsample))
    sub = np.array(np.unravel_index(np.argmax(local), local.shape), dtype=np.float64)
    shift_px = peak + (sub - centre) / upsample
    return shift_px, float(local.max() / weight)


@dataclass(frozen=True)
class TileShifts:
    """Local shift measurements on a tile grid.

    Attributes:
        centre_yx_px: ``(n, 2)`` float64 tile centres in reference coordinates.
        shift_px: ``(n, 2)`` float64 measured ``(dy, dx)`` per tile.
        peak: ``(n,)`` float64 normalised correlation peak per tile.
    """

    centre_yx_px: np.ndarray
    shift_px: np.ndarray
    peak: np.ndarray


def tile_shifts(
    reference: np.ndarray,
    moving: np.ndarray,
    offset_px: np.ndarray | None = None,
    tile_px: int = TILE_PX,
    step_px: int = TILE_STEP_PX,
) -> TileShifts:
    """Measure local shifts of ``moving`` relative to ``reference`` on a tile grid.

    Args:
        reference: ``(H, W)`` float image, NaN at no-data.
        moving: ``(H, W)`` float image, NaN at no-data.
        offset_px: ``(2,)`` approximate global shift; each moving tile is cut at
            the reference tile position plus this offset (rounded) so that the
            pair overlaps. Tiles whose moving window leaves the frame are skipped.
        tile_px: Tile side length.
        step_px: Grid step between tile origins.

    Returns:
        Total shifts (offset included) per tile.
    """
    h, w = reference.shape
    off = np.zeros(2, int) if offset_px is None else np.round(offset_px).astype(int)
    centres, shifts, peaks = [], [], []
    for ty in range(0, h - tile_px + 1, step_px):
        for tx in range(0, w - tile_px + 1, step_px):
            my, mx = ty + off[0], tx + off[1]
            if my < 0 or mx < 0 or my + tile_px > h or mx + tile_px > w:
                continue
            ref_tile = reference[ty : ty + tile_px, tx : tx + tile_px]
            mov_tile = moving[my : my + tile_px, mx : mx + tile_px]
            if not (np.isfinite(ref_tile).any() and np.isfinite(mov_tile).any()):
                continue
            s, p = estimate_shift(ref_tile, mov_tile)
            centres.append((ty + tile_px / 2, tx + tile_px / 2))
            shifts.append(s + off)
            peaks.append(p)
    return TileShifts(
        np.array(centres, float).reshape(-1, 2),
        np.array(shifts, float).reshape(-1, 2),
        np.array(peaks, float),
    )


# --------------------------------------------------------------------------- #
# Displacement field
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FieldFit:
    """Statistics of one field update (one pass).

    Attributes:
        n_tiles: Tiles measured.
        n_good: Tiles with peak >= ``MIN_PEAK``.
        n_inliers: Good tiles kept by the robust fit.
        fit_rms_px: RMS distance of the inlier tiles from the fitted field.
        median_update_px: Median magnitude of the measured tile shifts in this
            pass (for passes after the first: the misregistration still left).
        smoothing: Spline smoothing selected by cross-validation.
    """

    n_tiles: int
    n_good: int
    n_inliers: int
    fit_rms_px: float
    median_update_px: float
    smoothing: float


def _smooth_field(
    points: np.ndarray, values: np.ndarray, shape: tuple[int, int], smoothing: float
) -> np.ndarray:
    """Thin-plate-spline smoothing of scattered shifts, evaluated densely.

    The spline is evaluated on a grid with ``FIELD_GRID_PX`` spacing and
    interpolated to every pixel with a cubic spline, which is exact to well
    below 0.01 px for a field this smooth and ~1000x cheaper than evaluating
    the RBF everywhere.

    Args:
        points: ``(n, 2)`` tile centres ``(y, x)`` in px.
        values: ``(n, 2)`` shifts ``(dy, dx)`` in px.
        shape: ``(H, W)`` output shape.
        smoothing: Spline smoothing.

    Returns:
        ``(2, H, W)`` float64 dense field.
    """
    spline = _spline(points, values, shape, smoothing)
    gy = np.arange(0, shape[0] + FIELD_GRID_PX, FIELD_GRID_PX, dtype=float)
    gx = np.arange(0, shape[1] + FIELD_GRID_PX, FIELD_GRID_PX, dtype=float)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    coarse = spline(np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(gy), len(gx), 2)
    fy = (np.arange(shape[0], dtype=float) / FIELD_GRID_PX)[:, None]
    fx = (np.arange(shape[1], dtype=float) / FIELD_GRID_PX)[None, :]
    fy, fx = np.broadcast_arrays(fy, fx)
    return np.stack(
        [ndimage.map_coordinates(coarse[..., c], [fy, fx], order=3, mode="nearest") for c in (0, 1)]
    )


def _spline(points: np.ndarray, values: np.ndarray, shape: tuple[int, int], smoothing: float):
    """Smoothing thin-plate spline in coordinates normalised by the image size."""
    scale = float(max(shape))
    rbf = interpolate.RBFInterpolator(
        points / scale, values, kernel="thin_plate_spline", smoothing=smoothing, degree=1
    )
    return lambda q: rbf(np.asarray(q, float) / scale)


def _robust_inliers(
    points: np.ndarray, values: np.ndarray, shape: tuple[int, int], smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reject points far from a smoothing spline through the rest.

    Args:
        points: ``(n, 2)`` tile centres in px.
        values: ``(n, 2)`` measured shifts in px.
        shape: ``(H, W)`` of the image.
        smoothing: Spline smoothing.

    Returns:
        ``(n,)`` bool inlier mask and ``(n,)`` distance of each point from the
        final spline, in px.
    """
    keep = np.ones(len(points), bool)
    for _ in range(_FIT_ITERATIONS):
        spline = _spline(points[keep], values[keep], shape, smoothing)
        resid = np.linalg.norm(values - spline(points), axis=1)
        radius = max(OUTLIER_MADS * MAD_TO_SIGMA * np.median(resid[keep]), _MIN_REJECT_PX)
        new_keep = resid <= radius
        if np.array_equal(new_keep, keep) or new_keep.sum() < MIN_GOOD_TILES:
            break
        keep = new_keep
    return keep, resid


def _blocked_errors(
    points: np.ndarray,
    values: np.ndarray,
    inliers: np.ndarray,
    shape: tuple[int, int],
    smoothing: float,
) -> np.ndarray:
    """Held-out prediction errors with spatially blocked folds.

    For each ``CV_BLOCK_PX`` block, a spline is fitted to the inlier points
    outside it and evaluated at every point inside it.

    Args:
        points: ``(n, 2)`` tile centres in px.
        values: ``(n, 2)`` measured shifts in px.
        inliers: ``(n,)`` bool, points usable for training.
        shape: ``(H, W)`` of the image.
        smoothing: Spline smoothing.

    Returns:
        ``(m, 2)`` measured minus predicted shift for the held-out points of
        every block that had enough training points, and ``(m,)`` the block
        index of each.
    """
    block = (points // CV_BLOCK_PX).astype(int)
    block_id = block[:, 0] * (shape[1] // CV_BLOCK_PX + 1) + block[:, 1]
    errors, blocks = [], []
    for b in np.unique(block_id):
        test = block_id == b
        train = ~test & inliers
        if train.sum() < MIN_GOOD_TILES:
            continue
        spline = _spline(points[train], values[train], shape, smoothing)
        errors.append(values[test] - spline(points[test]))
        blocks.append(np.full(test.sum(), b))
    if not errors:
        return np.empty((0, 2)), np.empty(0, int)
    return np.concatenate(errors), np.concatenate(blocks)


def select_smoothing(points: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> float:
    """Choose the spline smoothing by blocked cross-validation.

    All candidates are trained and scored on the same tiles: the inliers of the
    most flexible candidate. (Scoring each candidate on its own inliers would
    favour smooth fields, which simply reject the tiles they cannot fit, e.g.
    in a locally stretched region.)

    The returned candidate is the most flexible one whose mean held-out error
    is within one standard error of the best (the standard error over CV
    blocks, since tiles within a block are correlated). Differences smaller
    than that are noise, and the risks are asymmetric: over-smoothing removes
    real structure such as local stretch, while a slightly noisy field costs
    little.

    Args:
        points: ``(n, 2)`` tile centres in px.
        values: ``(n, 2)`` measured shifts in px.
        shape: ``(H, W)`` of the image.

    Returns:
        The selected smoothing.
    """
    keep, _ = _robust_inliers(points, values, shape, min(SMOOTHING_CANDIDATES))
    pts, vals = points[keep], values[keep]
    scores, block_means = [], []
    for smoothing in SMOOTHING_CANDIDATES:
        err, block = _blocked_errors(pts, vals, np.ones(len(pts), bool), shape, smoothing)
        mag = np.linalg.norm(err, axis=1)
        scores.append(mag.mean() if len(mag) else np.inf)
        block_means.append([mag[block == b].mean() for b in np.unique(block)])
    best = int(np.argmin(scores))
    n_blocks = len(block_means[best])
    se = np.std(block_means[best], ddof=1) / np.sqrt(n_blocks) if n_blocks > 1 else 0.0
    ok = np.array(scores) <= scores[best] + se
    return float(np.array(SMOOTHING_CANDIDATES)[ok].min())


def fit_field(
    tiles: TileShifts, shape: tuple[int, int], smoothing: float | None = None
) -> tuple[np.ndarray | None, FieldFit]:
    """Robust smooth displacement field from tile shifts.

    Tiles below ``MIN_PEAK`` are dropped; the rest are fitted with a smoothing
    thin-plate spline, iteratively rejecting tiles more than ``OUTLIER_MADS``
    robust standard deviations from the fit (moving cloud edges, mismatches).

    Args:
        tiles: Local shift measurements.
        shape: ``(H, W)`` of the image.
        smoothing: Spline smoothing; None selects it with :func:`select_smoothing`.

    Returns:
        ``(2, H, W)`` float64 dense field, or None if fewer than
        ``MIN_GOOD_TILES`` tiles are usable; and the fit statistics.
    """
    good = tiles.peak >= MIN_PEAK
    n_good = int(good.sum())
    median_update = float(np.median(np.linalg.norm(tiles.shift_px[good], axis=1))) if n_good else 0
    nan = float("nan")
    if n_good < MIN_GOOD_TILES:
        return None, FieldFit(len(good), n_good, 0, nan, median_update, nan)

    pts, vals = tiles.centre_yx_px[good], tiles.shift_px[good]
    if smoothing is None:
        smoothing = select_smoothing(pts, vals, shape)
    keep, resid = _robust_inliers(pts, vals, shape, smoothing)
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    field = _smooth_field(pts[keep], vals[keep], shape, smoothing)
    fit = FieldFit(len(good), n_good, int(keep.sum()), rms, median_update, smoothing)
    return field, fit


def compose_fields(field: np.ndarray, update: np.ndarray) -> np.ndarray:
    """Field equivalent to warping with ``field`` and then with ``update``.

    Warping with ``field`` gives ``w(q) = band(q + field(q))``; warping that
    with ``update`` gives ``band(p + update(p) + field(p + update(p)))``.

    Args:
        field: ``(2, H, W)`` current field.
        update: ``(2, H, W)`` residual field measured on the warped band.

    Returns:
        ``(2, H, W)`` float64 composed field.
    """
    h, w = field.shape[1:]
    coords = update.copy()
    coords[0] += np.arange(h, dtype=float)[:, None]
    coords[1] += np.arange(w, dtype=float)[None, :]
    resampled = [ndimage.map_coordinates(field[c], coords, order=1, mode="nearest") for c in (0, 1)]
    return update + np.stack(resampled)


# --------------------------------------------------------------------------- #
# Warping
# --------------------------------------------------------------------------- #


def warp_band(
    band: np.ndarray, field: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a band onto the reference grid: ``out(p) = band(p + field(p))``.

    Cubic-spline interpolation preserves sharpness better than bilinear but
    still correlates noise between neighbouring pixels, so noise statistics
    (SNR) should be measured before warping.

    Args:
        band: ``(H, W)`` float image, NaN at no-data.
        field: ``(2, H, W)`` displacement ``(dy, dx)`` of this band relative to
            the reference, in px.
        valid: ``(H, W)`` bool validity of ``band``; defaults to ``isfinite(band)``.

    Returns:
        ``(H, W)`` float32 registered band (NaN where invalid) and its
        ``(H, W)`` bool validity: False where the sample point falls outside
        the frame or within the spline support of an invalid input pixel.
    """
    h, w = band.shape
    valid = np.isfinite(band) if valid is None else valid
    coords = field.copy()
    coords[0] += np.arange(h, dtype=float)[:, None]
    coords[1] += np.arange(w, dtype=float)[None, :]

    fill = float(np.mean(band[valid])) if valid.any() else 0.0
    data = np.where(valid, band, fill)
    out = ndimage.map_coordinates(data, coords, order=3, mode="nearest").astype(np.float32)

    inside = (coords[0] >= 0) & (coords[0] <= h - 1) & (coords[1] >= 0) & (coords[1] <= w - 1)
    # Any invalid pixel within the 4x4 cubic support taints the sample.
    tainted = ndimage.binary_dilation(~valid, iterations=2)
    hit = ndimage.map_coordinates(tainted.astype(np.float32), coords, order=0, mode="nearest")
    out_valid = inside & (hit < 0.5)
    out[~out_valid] = np.nan
    return out, out_valid


# --------------------------------------------------------------------------- #
# Whole-cube registration and evaluation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidualStats:
    """Misregistration remaining between two registered bands, measured on tiles.

    Attributes:
        n_tiles: Non-overlapping tiles with peak >= ``MIN_PEAK``.
        mean_dy_px: Mean residual dy (systematic part).
        mean_dx_px: Mean residual dx (systematic part).
        median_px: Median residual magnitude.
        rms_px: RMS residual magnitude.
        p95_px: 95th percentile of the residual magnitude.
        max_px: Largest residual magnitude.
    """

    n_tiles: int
    mean_dy_px: float
    mean_dx_px: float
    median_px: float
    rms_px: float
    p95_px: float
    max_px: float


def _stats(shifts: np.ndarray) -> ResidualStats:
    """Summary statistics of ``(n, 2)`` residual shifts in px."""
    if not len(shifts):
        nan = float("nan")
        return ResidualStats(0, nan, nan, nan, nan, nan, nan)
    mag = np.linalg.norm(shifts, axis=1)
    return ResidualStats(
        n_tiles=len(shifts),
        mean_dy_px=float(shifts[:, 0].mean()),
        mean_dx_px=float(shifts[:, 1].mean()),
        median_px=float(np.median(mag)),
        rms_px=float(np.sqrt(np.mean(mag**2))),
        p95_px=float(np.percentile(mag, 95)),
        max_px=float(mag.max()),
    )


def blocked_cv_error(tiles: TileShifts, shape: tuple[int, int], smoothing: float) -> ResidualStats:
    """Field prediction error at tiles held out in spatial blocks.

    Tiles are grouped into ``CV_BLOCK_PX`` blocks. For each block, the spline
    is fitted to the inlier tiles outside it and evaluated at all good tiles
    inside it (outliers included, so the statistics are not flattered); the
    statistics are of measured minus predicted shift. This estimates how well
    the field predicts places it was not fitted to, including gaps such as open
    water, and is not flattered by a spline that passes through its own data.

    Args:
        tiles: Shift measurements (e.g. on a registered band vs the reference).
        shape: ``(H, W)`` of the image.
        smoothing: Spline smoothing.

    Returns:
        Statistics of the held-out prediction errors.
    """
    good = tiles.peak >= MIN_PEAK
    pts, vals = tiles.centre_yx_px[good], tiles.shift_px[good]
    if len(pts) < MIN_GOOD_TILES:
        return _stats(np.empty((0, 2)))
    keep, _ = _robust_inliers(pts, vals, shape, smoothing)
    return _stats(_blocked_errors(pts, vals, keep, shape, smoothing)[0])


def residual_misregistration(
    a: np.ndarray, b: np.ndarray, tiles: TileShifts | None = None
) -> ResidualStats:
    """Measure the remaining shift of ``b`` relative to ``a`` on non-overlapping tiles.

    Args:
        a: ``(H, W)`` float registered band, NaN at no-data.
        b: ``(H, W)`` float registered band, NaN at no-data.
        tiles: Optional existing measurement of ``b`` vs ``a`` on the default
            grid; its non-overlapping subset is used instead of measuring again.

    Returns:
        Residual statistics over tiles that pass the quality threshold.
    """
    if tiles is None:
        tiles = tile_shifts(a, b, step_px=TILE_PX)
    else:  # tile origins on the TILE_PX lattice do not overlap
        origin = tiles.centre_yx_px - TILE_PX / 2
        tiles = TileShifts(
            *(arr[np.all(origin % TILE_PX == 0, axis=1)] for arr in vars(tiles).values())
        )
    return _stats(tiles.shift_px[tiles.peak >= MIN_PEAK])


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit standard deviation (removes band gain and offset)."""
    return (x - x.mean()) / x.std()


def lucas_kanade_shift(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Translation of ``moving`` relative to ``reference`` by iterative Lucas-Kanade.

    Solves ``grad(ref) . d = ref - moving(p + d)`` in least squares, using the
    reference gradient for every iteration, with both windows standardised so
    the bands' different gain and offset cancel.

    Args:
        reference: ``(W, W)`` float window.
        moving: ``(W + 2P, W + 2P)`` float window around the same position, with
            a margin of ``P`` px on each side so it can be sampled at ``p + d``.

    Returns:
        ``(2,)`` float64 ``(dy, dx)`` such that ``moving(p + d) ~ reference(p)``;
        NaN if the shift leaves the margin (no reliable estimate).
    """
    pad = (moving.shape[0] - reference.shape[0]) // 2
    ref = _standardise(reference)
    gy, gx = np.gradient(ref)
    hessian = np.array([[np.sum(gy * gy), np.sum(gy * gx)], [np.sum(gy * gx), np.sum(gx * gx)]])
    coeffs = ndimage.spline_filter(moving, order=3)
    yy, xx = np.mgrid[pad : pad + ref.shape[0], pad : pad + ref.shape[1]].astype(np.float64)
    d = np.zeros(2)
    for _ in range(_CHECK_ITERATIONS):
        sampled = ndimage.map_coordinates(coeffs, [yy + d[0], xx + d[1]], order=3, prefilter=False)
        error = ref - _standardise(sampled)
        step = np.linalg.solve(hessian, [np.sum(gy * error), np.sum(gx * error)])
        d += step
        if np.abs(d).max() > pad - 1:
            return np.full(2, np.nan)
        if np.abs(step).max() < 1e-4:
            break
    return d


def independent_residual(
    reference: np.ndarray, moving: np.ndarray, reference_noise: float, cloud: np.ndarray
) -> dict[str, ResidualStats]:
    """Remaining misregistration measured with a method the fit did not use.

    The field and :func:`residual_misregistration` both use phase correlation
    on 128 px tiles, so neither can see that estimator's own errors. This check
    uses Lucas-Kanade (:func:`lucas_kanade_shift`) on ``CHECK_WINDOW_PX``
    windows offset from the fit's tiles, keeping windows with scene structure
    (``CHECK_MIN_CONTRAST``). On the challenge data its noise floor (the green
    band against itself, each copy with independent noise at the band's own
    level added) is 0.013 px median and 0.034 px p95.

    Args:
        reference: ``(H, W)`` float registered band, NaN at no-data.
        moving: ``(H, W)`` float registered band, NaN at no-data.
        reference_noise: Noise standard deviation of ``reference``.
        cloud: ``(H, W)`` bool cloud mask.

    Returns:
        Statistics for ``"clear"`` and ``"cloud"`` windows
        (``CHECK_CLOUD_FRACTION``).
    """
    w, pad = CHECK_WINDOW_PX, _CHECK_PAD_PX
    h, width = reference.shape
    shifts: dict[str, list[np.ndarray]] = {"clear": [], "cloud": []}
    for r0 in range(pad, h - w - pad + 1, w):
        for c0 in range(pad, width - w - pad + 1, w):
            ref = reference[r0 : r0 + w, c0 : c0 + w].astype(np.float64)
            mov = moving[r0 - pad : r0 + w + pad, c0 - pad : c0 + w + pad].astype(np.float64)
            if not (np.isfinite(ref).all() and np.isfinite(mov).all()):
                continue
            if ref.std() < CHECK_MIN_CONTRAST * reference_noise:
                continue
            d = lucas_kanade_shift(ref, mov)
            if np.isfinite(d).all():
                cloudy = cloud[r0 : r0 + w, c0 : c0 + w].mean() > CHECK_CLOUD_FRACTION
                shifts["cloud" if cloudy else "clear"].append(d)
    return {region: _stats(np.reshape(v, (-1, 2))) for region, v in shifts.items()}


@dataclass(frozen=True)
class BandRegistration:
    """Registration result of one band.

    Attributes:
        coarse_shift_px: ``(2,)`` full-frame shift used to seed the first pass.
        passes: Statistics of each field update.
        field_min_px: ``(2,)`` minimum of dy and dx over the frame.
        field_max_px: ``(2,)`` maximum of dy and dx over the frame.
        smoothing: Spline smoothing of the final field (chosen by CV).
        residual: Remaining misregistration vs the reference after warping,
            re-measured on non-overlapping tiles.
        cv: Blocked cross-validation error of the final field against the
            total tile shifts (see :func:`register_band`).
    """

    coarse_shift_px: np.ndarray
    passes: tuple[FieldFit, ...]
    field_min_px: np.ndarray
    field_max_px: np.ndarray
    smoothing: float
    residual: ResidualStats
    cv: ResidualStats


def register_band(
    reference: np.ndarray, band: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BandRegistration]:
    """Register one band to the reference.

    1. Iterative refinement with the most flexible smoothing: pass 1 measures
       tiles against the coarse full-frame shift; every later pass re-measures
       tiles on the band warped with the current field and composes the
       remainder into it, until the median remainder is below ``CONVERGED_PX``
       or after ``MAX_PASSES``. Flexibility here is what lets tiles in locally
       stretched regions correlate again.
    2. Final smoothing: the *total* shift of each tile (field at its centre plus
       what is still measured there) is the honest measurement to judge a field
       by. Residuals alone are not: a tile's measurement error is a fixed
       function of its image content, so an overfitted field absorbs it and
       re-measuring after warping reproduces it. The smoothing is selected by
       blocked cross-validation on the totals; if it is not the flexible one,
       the field is refitted from the totals and the band warped again.

    Args:
        reference: ``(H, W)`` float reference band, NaN at no-data.
        band: ``(H, W)`` float band to register, NaN at no-data.
        valid: ``(H, W)`` bool validity of ``band``.

    Returns:
        ``(H, W)`` float32 registered band, its ``(H, W)`` bool validity, the
        ``(2, H, W)`` final field, and the registration statistics.
    """
    shape = reference.shape
    flexible = min(SMOOTHING_CANDIDATES)
    coarse, _ = estimate_shift(reference, band)
    field = np.broadcast_to(coarse[:, None, None], (2, *shape)).astype(np.float64)
    warped, warped_valid = band, valid
    passes = []
    for k in range(MAX_PASSES):
        offset = coarse if k == 0 else None
        tiles = tile_shifts(reference, band if k == 0 else warped, offset)
        update, fit = fit_field(tiles, shape, flexible)
        passes.append(fit)
        if update is None:
            log.warning("Pass %d: only %d usable tiles; field not updated", k + 1, fit.n_good)
            break
        field = update if k == 0 else compose_fields(field, update)
        warped, warped_valid = warp_band(band, field, valid)
        if k > 0 and fit.median_update_px < CONVERGED_PX:
            break

    final = tile_shifts(reference, warped)
    good = final.peak >= MIN_PEAK
    pts = final.centre_yx_px[good]
    at_centres = np.stack(
        [ndimage.map_coordinates(field[c], pts.T, order=1, mode="nearest") for c in (0, 1)], 1
    )
    totals = TileShifts(pts, at_centres + final.shift_px[good], final.peak[good])
    smoothing = flexible
    if len(pts) >= MIN_GOOD_TILES and passes[0].n_inliers:
        smoothing = select_smoothing(totals.centre_yx_px, totals.shift_px, shape)
        if smoothing != flexible:
            field, _ = fit_field(totals, shape, smoothing)
            warped, warped_valid = warp_band(band, field, valid)
            final = tile_shifts(reference, warped)

    residual = residual_misregistration(reference, warped, final)
    cv = blocked_cv_error(totals, shape, smoothing)
    sub = field[:, ::FIELD_GRID_PX, ::FIELD_GRID_PX].reshape(2, -1)
    result = BandRegistration(
        coarse, tuple(passes), sub.min(1), sub.max(1), smoothing, residual, cv
    )
    return warped, warped_valid, field, result


_worker_reference: np.ndarray | None = None


def _init_worker(reference: np.ndarray) -> None:
    """Process-pool initializer: keep the reference band in the worker."""
    global _worker_reference
    _worker_reference = reference


def _register_in_worker(
    band: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, BandRegistration]:
    """Register one band against the worker's reference (the dense field is dropped)."""
    warped, warped_valid, _, reg = register_band(_worker_reference, band, valid)
    return warped, warped_valid, reg


def register_cube(
    cube: np.ndarray, valid: np.ndarray, reference_index: int, workers: int = 1
) -> list[BandRegistration]:
    """Register every band of ``cube`` to ``cube[reference_index]``, in place.

    Bands are independent, so with ``workers > 1`` they are registered in
    parallel processes. At most ``2 * workers`` bands are in flight, which
    bounds the extra memory to a few bands rather than a copy of the cube.
    Results do not depend on scheduling.

    Args:
        cube: ``(n_bands, H, W)`` float32 corrected values, NaN at no-data.
            Overwritten with the registered bands.
        valid: ``(n_bands, H, W)`` bool validity; overwritten with the
            validity after warping (False where data was shifted in from
            outside the frame).
        reference_index: Index of the reference band, which is left unchanged.
        workers: Number of worker processes; 1 runs in this process.

    Returns:
        One :class:`BandRegistration` per band (identity for the reference).
    """
    ref = cube[reference_index].copy()
    zero, none = np.zeros(2), ResidualStats(0, *[0.0] * 6)
    results: dict[int, BandRegistration] = {
        reference_index: BandRegistration(zero, (), zero, zero, 0.0, none, none)
    }
    todo = [k for k in range(cube.shape[0]) if k != reference_index]

    def store(k: int, out: tuple[np.ndarray, np.ndarray, BandRegistration]) -> None:
        cube[k], valid[k], results[k] = out
        reg = results[k]
        log.info(
            "  band %d/%d: %d passes, residual median %.3f px, CV median %.3f / p95 %.3f px",
            len(results), cube.shape[0], len(reg.passes),
            reg.residual.median_px, reg.cv.median_px, reg.cv.p95_px,
        )  # fmt: skip

    if workers <= 1:
        _init_worker(ref)
        for k in todo:
            store(k, _register_in_worker(cube[k], valid[k]))
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(ref,)) as pool:
            pending: dict = {}
            while todo or pending:
                while todo and len(pending) < 2 * workers:
                    k = todo.pop(0)
                    pending[pool.submit(_register_in_worker, cube[k], valid[k])] = k
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    store(pending.pop(fut), fut.result())
    return [results[k] for k in range(cube.shape[0])]
