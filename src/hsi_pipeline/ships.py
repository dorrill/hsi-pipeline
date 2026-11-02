"""Ship detection in cloud-free open water: compact objects brighter than the water at 729 nm.

Water absorbs strongly beyond 700 nm, so at the longest band (729 nm) it is the
darkest surface in the scene, and a vessel of any colour, dark hull or white
deck, is brighter than the water around it. The detector looks for compact
bright anomalies on the water at that band:

* Search area: the open water found by the cloud module's water test, with
  enclosed objects (the vessels themselves) filled in, minus a
  ``SHORE_BUFFER_PX`` margin at the shore and everything within
  ``CLOUD_BUFFER_PX`` of the cloud mask. The cloud buffer is wide because the
  cloud fields contain small, faint puffs below the cloud mask's minimum size,
  which look like small bright objects; in the challenge capture they all lie
  within ~55 px of masked cloud.
* Anomaly: the 3x3 mean of the 729 nm radiance minus its local median
  (``BACKGROUND_PX``), in units of the anomaly's own noise measured over the
  search area.
* Objects: hysteresis thresholding. A pixel above ``DETECT_SIGMA`` seeds a
  detection; its connected region above ``EXTENT_SIGMA`` gives the bounding box.
* Confidence: a logistic function of the peak contrast. It is 0.5 at
  ``CONFIDENCE_MID_SIGMA``, the contrast of the brightest haze patches that the
  cloud buffer does not remove. It ranks detections; it is not a calibrated
  probability.

Vessels moored against piers or the shore are not holes in the water, so they
fall outside the search area: at this resolution they cannot be separated from
the piers. Surf, wakes, buoys and haze patches far from the cloud mask can still
be detected, mostly with low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .cloud import water_mask

#: Distance from the cloud mask (px) inside which nothing is searched. Faint
#: cloud puffs below the cloud mask's 100 px minimum lie within ~55 px of masked
#: cloud in the challenge capture; the nearest vessel found is ~70 px away.
CLOUD_BUFFER_PX = 64

#: Margin at the shore (px): mixed land/water pixels, pier ends and breaking waves.
SHORE_BUFFER_PX = 5

#: Opening radius (px) that cuts thin dark links (shadows, canals) between the
#: sea and dark patches on land before the sea's holes are filled.
SEA_OPENING_PX = 2

#: Smallest connected water body treated as sea (px), ~ 140 x 140 px; smaller
#: dark patches are shadows, ponds and dark roofs.
MIN_SEA_PX = 20_000

#: Local median window (px), applied along rows and then along columns (12x
#: faster than a square window). It must be more than twice as long as a
#: vessel so the median is set by the water; the largest vessel here is ~25 px.
BACKGROUND_PX = 31

#: Peak contrast that seeds a detection, in noise sigma of the 3x3 mean. Below
#: ~8 sigma the candidates in the challenge capture are noise and haze.
DETECT_SIGMA = 8.0

#: Contrast that bounds a detection's extent (half the seed threshold).
EXTENT_SIGMA = 4.0

#: Confidence is 0.5 at this contrast; the brightest haze patches outside the
#: cloud buffer reach ~11.6 sigma, most vessels 13-43 sigma.
CONFIDENCE_MID_SIGMA = 12.0

#: Width of the logistic (sigma): confidence 0.12 at the detection threshold,
#: 0.88 at 16 sigma.
CONFIDENCE_SCALE_SIGMA = 2.0

#: Confidence at or above which a detection counts as confident (red box in
#: the overlay); below it the box is orange.
CONFIDENT = 0.5
CONFIDENT_COLOUR = (255, 0, 0)
WEAK_COLOUR = (255, 160, 0)

#: Gap between a detection and its box, and the box's line width (px): the box
#: frames the vessel without covering it.
BOX_MARGIN_PX = 3
BOX_WIDTH_PX = 2


@dataclass(frozen=True)
class Ship:
    """One detection, in pixel coordinates of the registered grid.

    Attributes:
        row_min: First row of the bounding box.
        row_max: Last row of the bounding box (inclusive).
        col_min: First column of the bounding box.
        col_max: Last column of the bounding box (inclusive).
        area_px: Pixels in the detection's extent.
        peak_sigma: Peak contrast over the water, in noise sigma.
        confidence: Logistic score in (0, 1) of ``peak_sigma``.
    """

    row_min: int
    row_max: int
    col_min: int
    col_max: int
    area_px: int
    peak_sigma: float
    confidence: float


@dataclass(frozen=True)
class ShipSearch:
    """The search area and the noise the contrast is measured in.

    Attributes:
        search_fraction: Fraction of valid pixels searched (cloud-free open water).
        noise: Robust standard deviation of the anomaly over the search area,
            in radiance units.
    """

    search_fraction: float
    noise: float


def confidence(peak_sigma: np.ndarray | float) -> np.ndarray | float:
    """Logistic confidence of a detection from its peak contrast.

    Args:
        peak_sigma: Peak contrast in noise sigma.

    Returns:
        Confidence in (0, 1), 0.5 at ``CONFIDENCE_MID_SIGMA``.
    """
    return 1 / (1 + np.exp(-(peak_sigma - CONFIDENCE_MID_SIGMA) / CONFIDENCE_SCALE_SIGMA))


def search_area(red_edge: np.ndarray, cloud: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cloud-free open water, including the vessels on it.

    Args:
        red_edge: ``(H, W)`` float registered radiance of the longest band.
        cloud: ``(H, W)`` bool cloud mask.
        valid: ``(H, W)`` bool valid pixels.

    Returns:
        ``(H, W)`` bool search area.
    """
    water = ndimage.binary_opening(water_mask(red_edge, valid), iterations=SEA_OPENING_PX)
    labels, _ = ndimage.label(water)
    big = np.bincount(labels.ravel()) >= MIN_SEA_PX
    big[0] = False
    sea = ndimage.binary_fill_holes(big[labels])  # vessels are holes in the water
    sea = ndimage.binary_erosion(sea, iterations=SHORE_BUFFER_PX)
    return sea & (ndimage.distance_transform_edt(~cloud) > CLOUD_BUFFER_PX)


def detect_ships(
    red_edge: np.ndarray, cloud: np.ndarray, valid: np.ndarray
) -> tuple[list[Ship], ShipSearch]:
    """Detect vessels in the cloud-free open water (see module docstring).

    Args:
        red_edge: ``(H, W)`` float registered radiance of the longest band
            (729 nm here).
        cloud: ``(H, W)`` bool cloud mask on the same grid.
        valid: ``(H, W)`` bool valid pixels.

    Returns:
        Detections sorted by decreasing confidence, and the search summary.
    """
    area = search_area(red_edge, cloud, valid)
    band = np.where(valid, red_edge, np.median(red_edge[area])).astype(np.float64)
    background = ndimage.median_filter(band, size=(1, BACKGROUND_PX))
    background = ndimage.median_filter(background, size=(BACKGROUND_PX, 1))
    anomaly = ndimage.uniform_filter(band - background, 3)
    clear = anomaly[area]
    noise = 1.4826 * float(np.median(np.abs(clear - np.median(clear))))
    z = np.where(area, anomaly / noise, 0.0)

    # Hysteresis: keep the extent regions that contain a seed.
    labels, n = ndimage.label(z > EXTENT_SIGMA)
    index = np.arange(1, n + 1)
    peaks = ndimage.maximum(z, labels, index)
    areas = ndimage.sum_labels(np.ones_like(z), labels, index)
    ships = [
        Ship(
            row_min=sl[0].start,
            row_max=sl[0].stop - 1,
            col_min=sl[1].start,
            col_max=sl[1].stop - 1,
            area_px=int(a),
            peak_sigma=float(p),
            confidence=float(confidence(p)),
        )
        for sl, p, a in zip(ndimage.find_objects(labels), peaks, areas, strict=True)
        if p >= DETECT_SIGMA
    ]
    ships.sort(key=lambda s: -s.peak_sigma)
    return ships, ShipSearch(search_fraction=float(area[valid].mean()), noise=noise)


def draw_boxes(rgb: np.ndarray, ships: list[Ship]) -> np.ndarray:
    """Copy of the RGB image with a box around each detection.

    Args:
        rgb: ``(H, W, 3)`` uint8 RGB composite on the detections' grid.
        ships: Detections from :func:`detect_ships`.

    Returns:
        ``(H, W, 3)`` uint8 image; boxes are ``CONFIDENT_COLOUR`` at or above
        ``CONFIDENT`` and ``WEAK_COLOUR`` below, clipped at the frame edges.
    """
    out = rgb.copy()
    h, w = out.shape[:2]
    pad = BOX_MARGIN_PX + BOX_WIDTH_PX
    for s in sorted(ships, key=lambda s: s.confidence):  # confident boxes drawn last, on top
        r0, r1 = s.row_min - pad, s.row_max + pad
        c0, c1 = s.col_min - pad, s.col_max + pad
        rows = slice(max(r0, 0), min(r1, h - 1) + 1)
        cols = slice(max(c0, 0), min(c1, w - 1) + 1)
        rr = np.arange(rows.start, rows.stop)[:, None]
        cc = np.arange(cols.start, cols.stop)[None, :]
        frame = (
            (rr < r0 + BOX_WIDTH_PX)
            | (rr > r1 - BOX_WIDTH_PX)
            | (cc < c0 + BOX_WIDTH_PX)
            | (cc > c1 - BOX_WIDTH_PX)
        )
        out[rows, cols][frame] = CONFIDENT_COLOUR if s.confidence >= CONFIDENT else WEAK_COLOUR
    return out
