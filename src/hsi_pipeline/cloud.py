"""Explainable cloud mask from visible bands: brightness, spectral shape, and shape.

The capture has no NIR or SWIR, so the usual cloud tests (high NIR
reflectance, SWIR cirrus bands) are not available. Two visible-range facts are:

* Clouds are bright: much brighter than the water they lie over.
* Clouds are spectrally flat in reflectance, so their radiance follows the
  solar spectrum. Every land surface in the scene (vegetation's red edge,
  soils, roofs, sand) gets relatively brighter towards the longest band
  (729 nm), so for the same green radiance it has *more* 729 nm radiance.

Because radiances add, every mixture of water and cloud (thin cloud, cloud
edges) lies on one straight line in the plane L(729) vs L(556): the *cloud
line*. Land lies above it. The mask uses two features:

1. ``brightness``: green radiance above the water mode, in units of the water
   cluster's noise. Test: ``> BRIGHTNESS_SIGMA``.
2. ``red_edge_excess``: height of the pixel above the cloud line, in 729 nm
   radiance. Among bright pixels its histogram is bimodal (cloud ~ +1, land
   ~ +4 on the challenge data); the threshold is the minimum of the smoothed
   histogram between the two modes, so it is set by the data rather than by
   hand. (Otsu's method was tried first and put the split at +3.0, inside the
   land mode, because it is biased towards the larger class.)

Clean-up is morphological, using the fact that clouds are compact blobs while
the remaining false alarms (beach strip, roof edges) are thin lines: an
opening, removal of objects smaller than ``MIN_CLOUD_PX``, hole filling, and a
``BUFFER_PX`` dilation that covers mixed edge pixels and the small cloud
parallax between bands (clouds sit above the ground the bands were registered
to).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: Brightness test, in standard deviations of the water cluster above its mode.
#: The noise of water alone exceeds 5 sigma with probability ~3e-7; on the
#: challenge data 5 sigma is ~5.4 radiance units (~9 DN at 556 nm): about the
#: water's own signal above the ~112 DN pedestal, though only ~8 % of its full level.
BRIGHTNESS_SIGMA = 5.0

#: Spatial averaging before the per-pixel tests (px). Single-pixel SNR is ~6 in
#: these bands; a 3x3 mean triples it at the cost of a 1 px edge blur.
SMOOTH_PX = 3

#: The cloud line is fitted to the lower envelope (this percentile of 729 nm
#: radiance) of brightness bins of this width (radiance units), using bins with
#: at least _MIN_BIN_PX pixels.
ENVELOPE_PERCENTILE = 5.0
_BIN_WIDTH = 2.0
_MIN_BIN_PX = 200

#: Smoothing of the excess histogram (bins) before locating its modes, and the
#: minimum height of a mode relative to the highest one.
_HIST_SMOOTH_BINS = 2.0
_MIN_MODE_FRACTION = 0.05

#: Morphological opening radius (px). The disk is 7 px across; after the 3x3
#: mean widens features by 2 px, it removes features up to ~4 px wide in the
#: original image (beach strip, roof edges, piers). Clouds that narrow would be
#: removed by MIN_CLOUD_PX anyway.
OPENING_RADIUS_PX = 3

#: Smallest cloud object kept (px), ~ 10 x 10 px.
MIN_CLOUD_PX = 100

#: Dilation buffer around clouds (px), for mixed edge pixels and parallax.
BUFFER_PX = 2


@dataclass(frozen=True)
class CloudStats:
    """Parameters found on the data and summary of the mask.

    Attributes:
        water_mode: Green radiance of the water mode.
        water_sigma: Noise of the water cluster in green radiance.
        brightness_threshold: Green radiance threshold used.
        cloud_line_slope: Slope of the cloud line, L(729) per L(556).
        cloud_line_intercept: Intercept of the cloud line.
        excess_threshold: Threshold on the red-edge excess (histogram valley).
        cloud_fraction: Fraction of valid pixels flagged as cloud.
        n_objects: Number of connected cloud objects.
    """

    water_mode: float
    water_sigma: float
    brightness_threshold: float
    cloud_line_slope: float
    cloud_line_intercept: float
    excess_threshold: float
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


def valley_threshold(values: np.ndarray, bins: int = 200) -> float:
    """Minimum of the smoothed histogram between its two outermost modes.

    Falls back to :func:`otsu_threshold` if the histogram has a single mode.

    Args:
        values: 1-D sample.
        bins: Histogram bins.

    Returns:
        The threshold, in the units of ``values``.
    """
    lo_v, hi_v = np.percentile(values, [0.5, 99.5])
    hist, edges = np.histogram(values, bins=bins, range=(lo_v, hi_v))
    centres = (edges[:-1] + edges[1:]) / 2
    h = ndimage.gaussian_filter1d(hist.astype(np.float64), _HIST_SMOOTH_BINS)
    is_mode = (h[1:-1] >= h[:-2]) & (h[1:-1] >= h[2:]) & (h[1:-1] > _MIN_MODE_FRACTION * h.max())
    modes = np.flatnonzero(is_mode) + 1
    if len(modes) < 2:
        return otsu_threshold(values)
    a, b = modes[0], modes[-1]
    return float(centres[a + np.argmin(h[a : b + 1])])


def water_mode_and_sigma(green: np.ndarray) -> tuple[float, float]:
    """Mode of the dominant dark cluster and its noise from the dark side.

    The dark side of the mode is free of cloud and land, so the half-normal
    spread below the mode measures the water cluster's own noise.

    Args:
        green: 1-D sample of smoothed green radiance.

    Returns:
        ``(mode, sigma)`` in radiance units.
    """
    hist, edges = np.histogram(green, bins=400)
    k = int(np.argmax(hist))
    mode = (edges[k] + edges[k + 1]) / 2
    below = green[green <= mode]
    return float(mode), float(np.sqrt(np.mean((below - mode) ** 2)))


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
        ValueError: If there are too few bright pixels to fit the cloud line.
    """
    g = _smooth(green, valid)
    n = _smooth(red_edge, valid)
    inner = ndimage.binary_erosion(valid, iterations=SMOOTH_PX)

    mode, sigma = water_mode_and_sigma(g[inner])
    bright_thr = mode + BRIGHTNESS_SIGMA * sigma
    bright = inner & (g > bright_thr)

    # Cloud line: lower envelope of L(729) vs L(556) above the water mode.
    xs, ys = [], []
    top = np.percentile(g[bright], 99.9) if bright.any() else bright_thr
    for lo in np.arange(bright_thr, top, _BIN_WIDTH):
        sel = bright & (g >= lo) & (g < lo + _BIN_WIDTH)
        if sel.sum() >= _MIN_BIN_PX:
            xs.append(lo + _BIN_WIDTH / 2)
            ys.append(np.percentile(n[sel], ENVELOPE_PERCENTILE))
    if len(xs) < 2:
        raise ValueError("Too few bright pixels to fit the cloud line")
    slope, intercept = np.polyfit(xs, ys, 1)
    excess = n - (slope * g + intercept)
    excess_thr = valley_threshold(excess[bright])

    raw = bright & (excess < excess_thr)
    disk = _disk(OPENING_RADIUS_PX)
    mask = ndimage.binary_opening(raw, structure=disk)
    labels, n_obj = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_CLOUD_PX
    keep[0] = False
    mask = keep[labels]
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, structure=_disk(BUFFER_PX)) & valid
    n_objects = int(ndimage.label(mask)[1])

    stats = CloudStats(
        water_mode=mode,
        water_sigma=sigma,
        brightness_threshold=float(bright_thr),
        cloud_line_slope=float(slope),
        cloud_line_intercept=float(intercept),
        excess_threshold=excess_thr,
        cloud_fraction=float(mask[valid].mean()),
        n_objects=n_objects,
    )
    return mask.astype(np.uint8), stats


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element."""
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2
