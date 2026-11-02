"""Explainable cloud mask from visible bands: brightness, colour of the added light, shape.

The capture has no NIR or SWIR, so the usual cloud tests (high NIR
reflectance, SWIR cirrus bands) are not available. The mask uses the green
band (556 nm) and the longest band (729 nm) and three facts:

* Water is dark at 729 nm, and clouds lie over water in this scene. Every
  pixel is compared with the water *in its own column*: the capture's level
  varies across the swath (a smooth ramp and column banding of a few units),
  which is several times the noise, and comparing within a column removes it.
* Clouds are bright: a cloud adds light to the water below it. The brightness
  test is the green radiance added over the water, ``dg``, in units of the
  local noise.
* Clouds are spectrally flat in reflectance, so the light they add has the
  solar spectrum, while every land surface here (vegetation's red edge, soil,
  roofs, sand) adds relatively more 729 nm light. Radiances add, so a pixel
  that is a fraction ``f`` cloud over water adds ``f * (cloud - water)`` in
  both bands, and the ratio ``r = dn / dg`` of the added light is the same for
  thin and thick cloud. Among bright pixels the histogram of ``r`` is bimodal
  (cloud ~0.7, land ~1.0 on the challenge data); the threshold is the valley
  between the modes, so it is set by the data rather than by hand.

Clean-up is morphological, using the fact that clouds are compact blobs: an
opening, removal of objects smaller than ``MIN_CLOUD_PX``, hole filling, and a
``BUFFER_PX`` dilation that covers mixed edge pixels and the small cloud
parallax between bands (clouds sit above the ground the bands were registered
to).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

#: Brightness test: green radiance added over the water, in units of the water's
#: noise. Noise alone exceeds 5 sigma with probability ~3e-7.
BRIGHTNESS_SIGMA = 5.0

#: Spatial averaging before the per-pixel tests (px). Single-pixel SNR is ~6 in
#: these bands; a 3x3 mean triples it at the cost of a 1 px edge blur.
SMOOTH_PX = 3

#: Water pixels: 729 nm radiance less than this many standard deviations of the
#: dark cluster above its mode. Cloud and land are bright at 729 nm.
WATER_SIGMA = 3.0

#: The water reference is a median per column in row blocks of this height,
#: interpolated between blocks. Columns are where the level varies (ramp,
#: banding); blocks follow slower along-track changes (ocean vs bay water, and
#: the warped bands' banding, which shifts by a few px over the frame).
REFERENCE_BLOCK_PX = 256

#: A column in a block needs this fraction of water pixels for its own median;
#: otherwise it is interpolated from the neighbouring columns.
MIN_WATER_FRACTION = 0.25

#: Pixels at least this many sigma bright set the ratio histogram: the ratio of
#: faint pixels is too noisy to show the two modes.
STRONG_SIGMA = 20.0

#: Smoothing of the ratio histogram (bins) before locating its modes, and the
#: minimum height of a mode relative to the highest one.
_HIST_SMOOTH_BINS = 2.0
_MIN_MODE_FRACTION = 0.05

#: Morphological opening radius (px). The disk is 7 px across; after the 3x3
#: mean widens features by 2 px, it removes features up to ~4 px wide in the
#: original image (roof edges, piers). Clouds that narrow would be removed by
#: MIN_CLOUD_PX anyway.
OPENING_RADIUS_PX = 3

#: Smallest cloud object kept (px), ~ 10 x 10 px.
MIN_CLOUD_PX = 100

#: Dilation buffer around clouds (px), for mixed edge pixels and parallax.
BUFFER_PX = 2


@dataclass(frozen=True)
class CloudStats:
    """Parameters found on the data and summary of the mask.

    Attributes:
        water_fraction: Fraction of valid pixels used as water reference.
        noise_sigma: Noise of the smoothed green band over water.
        brightness_threshold: Added green radiance above which a pixel is bright.
        cloud_ratio_mode: Mode of the added-light ratio ``dn / dg`` for cloud.
        land_ratio_mode: Mode of the ratio for land.
        ratio_threshold: Histogram valley between the two modes.
        cloud_fraction: Fraction of valid pixels flagged as cloud.
        n_objects: Number of connected cloud objects.
    """

    water_fraction: float
    noise_sigma: float
    brightness_threshold: float
    cloud_ratio_mode: float
    land_ratio_mode: float
    ratio_threshold: float
    cloud_fraction: float
    n_objects: int


def _smooth(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Box mean that does not drag invalid (NaN) pixels into valid ones."""
    filled = np.where(valid, band, 0.0).astype(np.float64)
    num = ndimage.uniform_filter(filled, SMOOTH_PX)
    den = ndimage.uniform_filter(valid.astype(np.float64), SMOOTH_PX)
    return num / np.maximum(den, 1e-9)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split maximising the between-class variance.

    Args:
        values: 1-D sample.
        bins: Histogram bins.

    Returns:
        The threshold, in the units of ``values``.
    """
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[1:][np.argmax(between)])


def valley_threshold(values: np.ndarray, bins: int = 200) -> tuple[float, float, float]:
    """Minimum of the smoothed histogram between its two outermost modes.

    Falls back to :func:`otsu_threshold` (modes NaN) if the histogram has a
    single mode, e.g. a scene without land or without cloud.

    Args:
        values: 1-D sample.
        bins: Histogram bins.

    Returns:
        ``(threshold, low_mode, high_mode)`` in the units of ``values``.
    """
    lo_v, hi_v = np.percentile(values, [0.5, 99.5])
    hist, edges = np.histogram(values, bins=bins, range=(lo_v, hi_v))
    centres = (edges[:-1] + edges[1:]) / 2
    h = ndimage.gaussian_filter1d(hist.astype(np.float64), _HIST_SMOOTH_BINS)
    is_mode = (h[1:-1] >= h[:-2]) & (h[1:-1] >= h[2:]) & (h[1:-1] > _MIN_MODE_FRACTION * h.max())
    modes = np.flatnonzero(is_mode) + 1
    if len(modes) < 2:
        log.warning("Ratio histogram has a single mode; falling back to Otsu's threshold")
        return otsu_threshold(values), np.nan, np.nan
    a, b = modes[0], modes[-1]
    return float(centres[a + np.argmin(h[a : b + 1])]), float(centres[a]), float(centres[b])


def water_mode_and_sigma(values: np.ndarray) -> tuple[float, float]:
    """Mode of the dominant dark cluster and its spread from the dark side.

    The dark side of the mode is free of cloud and land, so the half-normal
    spread below the mode measures the water cluster's own noise.

    Args:
        values: 1-D sample of smoothed radiance (or of radiance above the water).

    Returns:
        ``(mode, sigma)`` in the units of ``values``.
    """
    hist, edges = np.histogram(values, bins=400)
    k = int(np.argmax(hist))
    mode = (edges[k] + edges[k + 1]) / 2
    below = values[values <= mode]
    return float(mode), float(np.sqrt(np.mean((below - mode) ** 2)))


def water_mask(red_edge: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Water pixels: the dark cluster at the longest band.

    Args:
        red_edge: ``(H, W)`` float radiance of the longest band (729 nm here).
        valid: ``(H, W)`` bool valid pixels.

    Returns:
        ``(H, W)`` bool, True where the smoothed radiance is less than
        ``WATER_SIGMA`` standard deviations above the dark cluster's mode.
    """
    n = _smooth(red_edge, valid)
    inner = ndimage.binary_erosion(valid, iterations=SMOOTH_PX)
    mode, sigma = water_mode_and_sigma(n[inner])
    return inner & (n < mode + WATER_SIGMA * sigma)


def _fill_gaps(profile: np.ndarray) -> np.ndarray:
    """Linear interpolation over the NaNs of a 1-D profile (unchanged if all NaN)."""
    ok = np.isfinite(profile)
    if ok.all() or not ok.any():
        return profile
    x = np.arange(len(profile))
    return np.interp(x, x[ok], profile[ok])


def water_reference(image: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Level of the water in each column, following slow changes along the rows.

    The median over water pixels per column and ``REFERENCE_BLOCK_PX`` row
    block, interpolated over columns without enough water (land) and blocks
    without any, then linearly between block centres along the rows.

    Args:
        image: ``(H, W)`` float smoothed radiance.
        water: ``(H, W)`` bool water pixels.

    Returns:
        ``(H, W)`` float64 water level at every pixel.

    Raises:
        ValueError: If no column has enough water.
    """
    h, w = image.shape
    block_h = min(REFERENCE_BLOCK_PX, h)
    n_blocks = h // block_h
    blocks = np.where(water, image, np.nan)[: n_blocks * block_h].reshape(n_blocks, block_h, w)
    with warnings.catch_warnings():  # columns without water: all-NaN medians
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(blocks, axis=1)
    level[np.isfinite(blocks).mean(axis=1) < MIN_WATER_FRACTION] = np.nan
    if np.isnan(level).all():
        raise ValueError("No water found to use as reference")
    level = np.apply_along_axis(_fill_gaps, 1, level)  # across columns, per block
    level = np.apply_along_axis(_fill_gaps, 0, level)  # across blocks, per column

    # Row y lies between the centres of blocks k0 and k1, at fraction t.
    pos = np.clip((np.arange(h) + 0.5) / block_h - 0.5, 0, n_blocks - 1)
    k0 = np.floor(pos).astype(int)
    k1 = np.minimum(k0 + 1, n_blocks - 1)
    t = (pos - k0)[:, None]
    return (1 - t) * level[k0] + t * level[k1]


def cloud_mask(
    green: np.ndarray, red_edge: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, CloudStats]:
    """Binary cloud mask on the registered grid (see module docstring).

    Args:
        green: ``(H, W)`` float registered green-band radiance (556 nm here).
        red_edge: ``(H, W)`` float registered radiance of the longest band
            (729 nm here), the closest the capture gets to NIR.
        valid: ``(H, W)`` bool, valid in both bands.

    Returns:
        ``(H, W)`` uint8 mask (1 = cloud, 0 = clear or no-data) and the
        parameters found on the data.

    Raises:
        ValueError: If no water is found to compare with.
    """
    g = _smooth(green, valid)
    n = _smooth(red_edge, valid)
    inner = ndimage.binary_erosion(valid, iterations=SMOOTH_PX)

    # 1. Water reference per column, and the light each pixel adds over it.
    water = water_mask(red_edge, valid)
    dg = g - water_reference(g, water)
    dn = n - water_reference(n, water)

    # 2. Brightness, in units of the noise measured on the water's dark side.
    _, sigma = water_mode_and_sigma(dg[water])
    bright = inner & (dg > BRIGHTNESS_SIGMA * sigma)

    # 3. Colour of the added light: low ratio = flat spectrum = cloud.
    ratio = np.divide(dn, dg, out=np.full_like(dg, np.inf), where=bright)
    strong = bright & (dg > STRONG_SIGMA * sigma)
    ratio_thr, cloud_mode, land_mode = valley_threshold(ratio[strong])
    raw = bright & (ratio < ratio_thr)

    # 4. Shape: clouds are compact blobs.
    mask = ndimage.binary_opening(raw, structure=_disk(OPENING_RADIUS_PX))
    labels, _ = ndimage.label(mask)
    keep = np.bincount(labels.ravel()) >= MIN_CLOUD_PX
    keep[0] = False
    mask = ndimage.binary_fill_holes(keep[labels])
    mask = ndimage.binary_dilation(mask, structure=_disk(BUFFER_PX)) & valid

    stats = CloudStats(
        water_fraction=float(water[valid].mean()),
        noise_sigma=sigma,
        brightness_threshold=BRIGHTNESS_SIGMA * sigma,
        cloud_ratio_mode=cloud_mode,
        land_ratio_mode=land_mode,
        ratio_threshold=ratio_thr,
        cloud_fraction=float(mask[valid].mean()),
        n_objects=int(ndimage.label(mask)[1]),
    )
    return mask.astype(np.uint8), stats


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element."""
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2
