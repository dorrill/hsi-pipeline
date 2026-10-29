"""Unit tests for the cloud mask on a synthetic scene built from the same physics."""

import numpy as np
import pytest
from scipy import ndimage

from hsi_pipeline.cloud import BUFFER_PX, cloud_mask, valley_threshold, water_mode_and_sigma

WATER = np.array([70.0, 40.0])  # (L556, L729)
CLOUD = np.array([95.0, 64.0])  # thick cloud: on the water-cloud mixing line
LAND = np.array([78.0, 50.0])  # above the line: relatively more 729 nm


def _scene(seed=0, size=512):
    """Water with land patches, soft clouds, a thin bright line and a tiny blob.

    Returns green, red-edge, the true cloud fraction, and masks of the decoys.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    land = (xx > 400) | ((yy - 80) ** 2 + (xx - 120) ** 2 < 50**2)
    f = np.zeros((size, size))
    for cy, cx, r in [(250, 150, 45), (330, 260, 35), (180, 300, 30), (420, 120, 40)]:
        f = np.maximum(f, np.clip(1.3 - np.hypot(yy - cy, xx - cx) / r, 0, 1))
    line = (np.abs(yy - 470) <= 1) & (xx > 200) & (xx < 380)  # 3 px wide
    blob = (np.abs(yy - 60) <= 2) & (np.abs(xx - 320) <= 2)  # 25 px
    base = np.where(land[..., None], LAND, WATER)
    pixel = (1 - f[..., None]) * base + f[..., None] * CLOUD
    pixel[line | blob] = CLOUD  # spectrally cloud-like but the wrong shape
    noise = rng.normal(0, 1.0, (size, size, 2))
    return pixel[..., 0] + noise[..., 0], pixel[..., 1] + noise[..., 1], f, land, line | blob


def test_cloud_mask_finds_clouds_and_rejects_decoys():
    """Definite cloud is found; land, thin and tiny features are not; extras are buffer only.

    The brightness test is sensitive to cloud fractions well below 0.3 (5 sigma
    of the smoothed water noise), so thin cloud edges are flagged too, plus the
    ``BUFFER_PX`` dilation.
    """
    g, n, f, land, decoys = _scene()
    mask, stats = cloud_mask(g, n, np.ones(g.shape, bool))
    m = mask.astype(bool)
    assert m[f > 0.3].all()
    assert not (m & land & (f == 0)).any()
    assert not (m & decoys).any()
    near_cloud = ndimage.binary_dilation(f > 0, iterations=BUFFER_PX + 1)
    assert not (m & ~near_cloud).any()
    assert stats.cloud_line_slope == pytest.approx(
        (CLOUD[1] - WATER[1]) / (CLOUD[0] - WATER[0]), abs=0.1
    )


def test_cloud_mask_is_zero_on_nodata():
    """No-data pixels are never flagged."""
    g, n, _, _, _ = _scene()
    valid = np.ones(g.shape, bool)
    valid[200:300, 100:200] = False  # covers part of a cloud
    mask, _ = cloud_mask(g, n, valid)
    assert not mask[~valid].any()


def test_valley_threshold_splits_unequal_modes():
    """The valley lies between the modes even when one class is 10x larger (Otsu does not)."""
    rng = np.random.default_rng(1)
    small, large = rng.normal(1.0, 0.5, 10_000), rng.normal(4.0, 0.8, 100_000)
    thr = valley_threshold(np.concatenate([small, large]))
    assert 1.8 < thr < 2.6


def test_water_mode_and_sigma_ignore_bright_tail():
    """The dark-side spread measures the water noise, not the bright classes."""
    rng = np.random.default_rng(2)
    sample = np.concatenate([rng.normal(70, 1.2, 200_000), rng.uniform(75, 110, 50_000)])
    mode, sigma = water_mode_and_sigma(ndimage.uniform_filter1d(sample, 1))
    assert mode == pytest.approx(70, abs=0.3)
    assert sigma == pytest.approx(1.2, rel=0.1)
