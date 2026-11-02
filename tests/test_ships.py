"""Unit tests for ship detection on a synthetic 729 nm scene."""

import numpy as np
import pytest

from hsi_pipeline.ships import (
    BOX_MARGIN_PX,
    BOX_WIDTH_PX,
    CLOUD_BUFFER_PX,
    CONFIDENCE_MID_SIGMA,
    CONFIDENT_COLOUR,
    WEAK_COLOUR,
    Ship,
    confidence,
    detect_ships,
    draw_boxes,
)

WATER, LAND, VESSEL = 40.0, 52.0, 15.0  # 729 nm radiance; vessel contrast over water


def _scene(seed=0, size=512):
    """Banded water with land on the right, a cloud, one vessel and three decoys.

    The decoys are as bright as the vessel: one near the cloud, one on land and
    one moored against the shore.

    Returns the radiance, the cloud mask and the vessel's mask.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.where(xx >= 420, LAND, WATER) + rng.normal(0, 0.7, size)[xx]  # column banding
    cloud = np.hypot(yy - 400, xx - 120) < 20
    image[cloud] += 25.0
    vessel = (np.abs(yy - 150) <= 6) & (np.abs(xx - 200) <= 2)
    near_cloud = (np.abs(yy - 400) <= 2) & (np.abs(xx - (140 + CLOUD_BUFFER_PX // 2)) <= 2)
    on_land = (np.abs(yy - 100) <= 2) & (np.abs(xx - 470) <= 2)
    moored = (np.abs(yy - 300) <= 6) & (xx >= 416) & (xx < 420)
    image[vessel | near_cloud | on_land | moored] += VESSEL
    return image + rng.normal(0, 1.0, image.shape), cloud, vessel


def test_detects_the_vessel_and_ignores_decoys():
    """Only the vessel in open water is found, with a tight box and high confidence."""
    image, cloud, vessel = _scene()
    ships, search = detect_ships(image, cloud, np.ones(image.shape, bool))
    assert len(ships) == 1
    s = ships[0]
    rows, cols = np.nonzero(vessel)
    assert abs(s.row_min - rows.min()) <= 1 and abs(s.row_max - rows.max()) <= 1
    assert abs(s.col_min - cols.min()) <= 1 and abs(s.col_max - cols.max()) <= 1
    assert s.confidence > 0.99
    assert 0.5 < search.search_fraction < 0.8


def test_no_detections_on_empty_water():
    """Noise and column banding alone produce no detections."""
    image, cloud, vessel = _scene()
    image[vessel] -= VESSEL
    ships, _ = detect_ships(image, cloud, np.ones(image.shape, bool))
    assert ships == []


def test_confidence_is_monotonic_and_centred():
    """Confidence rises with contrast and is 0.5 at the midpoint."""
    z = np.array([8.0, 12.0, 20.0, 40.0])
    c = confidence(z)
    assert np.all(np.diff(c) > 0)
    assert confidence(CONFIDENCE_MID_SIGMA) == pytest.approx(0.5)


def test_draw_boxes_frames_without_covering_and_clips_at_edges():
    """Boxes surround the detection, leave it untouched, and clip at the frame edge."""
    rgb = np.zeros((40, 40, 3), np.uint8)
    inside = Ship(10, 14, 10, 12, area_px=15, peak_sigma=20.0, confidence=0.98)
    corner = Ship(0, 2, 37, 39, area_px=9, peak_sigma=9.0, confidence=0.18)
    out = draw_boxes(rgb, [inside, corner])
    assert not rgb.any()  # input unchanged
    pad = BOX_MARGIN_PX + BOX_WIDTH_PX
    assert (out[10 - pad, 10:13] == CONFIDENT_COLOUR).all()  # top edge
    assert (out[14 + pad, 10:13] == CONFIDENT_COLOUR).all()  # bottom edge
    assert not out[
        10 - BOX_MARGIN_PX : 15 + BOX_MARGIN_PX, 10 - BOX_MARGIN_PX : 13 + BOX_MARGIN_PX
    ].any()
    assert (out[2 + pad, 37:40] == WEAK_COLOUR).all()  # clipped box keeps its bottom edge
    assert (out[0:3, 37 - pad] == WEAK_COLOUR).all()  # and its left edge
