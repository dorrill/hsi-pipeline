"""Unit tests for co-registration, against synthetic data with known displacements."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from hsi_pipeline.io import load_cube, load_metadata
from hsi_pipeline.radiometry import apply_radiometric_correction, validity_masks
from hsi_pipeline.registration import (
    MIN_PEAK,
    estimate_shift,
    independent_residual,
    lucas_kanade_shift,
    register_band,
)

REPO = Path(__file__).resolve().parents[1]


def _texture(shape, seed=0, sigma=2.0):
    """Band-limited random texture, unit standard deviation."""
    f = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma)
    return f / f.std()


def _fourier_shift(image, shift):
    """Exact (periodic) subpixel translation: out(p) = image(p - shift)."""
    return np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(image), shift)).real


@pytest.mark.parametrize("shift", [(0.0, 0.0), (1.3, -2.7), (-0.45, 0.8), (7.25, -12.6)])
def test_estimate_shift_subpixel_with_noise(shift):
    """Known subpixel translations are recovered to < 0.05 px at SNR ~ 20."""
    rng = np.random.default_rng(1)
    ref = _texture((256, 256))
    mov = _fourier_shift(ref, shift) + rng.normal(0, 0.05, ref.shape)
    est, peak = estimate_shift(ref + rng.normal(0, 0.05, ref.shape), mov)
    assert est == pytest.approx(shift, abs=0.05)
    assert peak > MIN_PEAK


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.6), (-1.2, 0.45)])
def test_lucas_kanade_recovers_subpixel_shift_across_gain_and_offset(shift):
    """The independent check recovers a known shift although the bands differ in gain/offset."""
    rng = np.random.default_rng(4)
    ref = _texture((128, 128))
    mov = 3.0 * _fourier_shift(ref, shift) + 10.0 + rng.normal(0, 0.05, ref.shape)
    pad = 4
    est = lucas_kanade_shift(ref[32:96, 32:96], mov[32 - pad : 96 + pad, 32 - pad : 96 + pad])
    assert est == pytest.approx(shift, abs=0.02)


def test_independent_residual_splits_clear_and_cloud():
    """Windows are reported by region; textureless windows are skipped."""
    ref = _texture((256, 256))
    mov = _fourier_shift(ref, (0.25, 0.0))
    ref[:, 128:] = 0.0  # flat: the third column of windows is skipped
    cloud = np.zeros(ref.shape, bool)
    cloud[:128] = True  # windows start at rows 4, 68, 132: the first two are cloudy
    out = independent_residual(ref, mov, reference_noise=0.05, cloud=cloud)
    assert (out["cloud"].n_tiles, out["clear"].n_tiles) == (4, 2)
    for stats in out.values():
        assert stats.median_px == pytest.approx(0.25, abs=0.03)


def test_estimate_shift_ignores_gain_and_offset():
    """Phase correlation is insensitive to a linear radiometric difference between bands."""
    ref = _texture((256, 256))
    mov = 3.0 * _fourier_shift(ref, (2.4, -1.1)) + 50.0
    est, _ = estimate_shift(ref, mov)
    assert est == pytest.approx((2.4, -1.1), abs=0.02)


def test_textureless_tile_has_low_peak():
    """Independent noise (e.g. open water) does not pass the quality threshold."""
    rng = np.random.default_rng(2)
    _, peak = estimate_shift(rng.normal(size=(128, 128)), rng.normal(size=(128, 128)))
    assert peak < MIN_PEAK


def test_register_band_recovers_non_affine_field():
    """A smooth field with along-track curvature (not affine) is recovered to < 0.1 px."""
    h = w = 512

    def s(y, x):
        return 4.0 + 1.5 * np.sin(np.pi * y / h) + 0.002 * x, -2.0 + 0.8 * np.cos(2 * np.pi * y / h)

    scene = _texture((h, w), seed=3)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sy, sx = s(yy, xx)
    band = ndimage.map_coordinates(scene, [yy - sy, xx - sx], order=3, mode="reflect")
    # Registration outputs band(p + F(p)) = scene(p + F - s(p + F)), so the exact
    # field solves F(p) = s(p + F(p)); it differs from s(p) by ~grad(s) * F.
    true_dy, true_dx = sy, sx
    for _ in range(10):
        true_dy, true_dx = s(yy + true_dy, xx + true_dx)
    rng = np.random.default_rng(4)
    ref = scene + rng.normal(0, 0.05, scene.shape)
    band = band + rng.normal(0, 0.05, scene.shape)

    registered, valid, field, reg = register_band(ref, band, np.ones_like(band, bool))

    inner = (slice(64, -64), slice(64, -64))  # the hull of the tile centres
    err = np.hypot(field[0][inner] - true_dy[inner], field[1][inner] - true_dx[inner])
    assert np.median(err) < 0.05
    assert np.percentile(err, 99) < 0.15  # the tail is at the hull corners (extrapolation)
    # Re-measured residual includes the estimator's own noise: an upper bound.
    assert reg.residual.median_px < 0.1
    assert np.isnan(registered[~valid]).all() and valid.mean() > 0.95


def test_synthetic_capture_matches_truth(tmp_path):
    """End to end on tools/make_synthetic.py output, against truth.json.

    The generator shifts with bilinear interpolation, adds 1-5 DN noise and a
    no-data strip that does not move with the scene; its land texture is faint,
    so tiles are much noisier than on the real data. Checks that:

    * bands 40-120 nm from the reference are registered to < 0.15 px (median);
    * the reported cross-validation error does not understate the true error;
    * a band whose land/sea contrast is inverted relative to the reference
      (960 nm vs 400 nm) is reported as not registered from tiles.
    """
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "make_synthetic.py"), "--out", str(tmp_path),
         "--size", "1024", "--n-bands", "16", "--seed", "0"],
        check=True, capture_output=True,
    )  # fmt: skip
    truth = json.loads((tmp_path / "truth.json").read_text())
    bands = load_metadata(tmp_path / "band_metadata.json")
    cube = load_cube(bands, (1024, 1024))
    valid, _ = validity_masks(cube)
    corrected = apply_radiometric_correction(cube, bands, valid)
    ref = truth["reference_band"] - 1

    for k in (1, 2, 3):
        _, _, field, reg = register_band(corrected[ref], corrected[k], valid[k])
        true_shift = np.array(truth["shifts_dy_dx_px"][k])
        err = np.median(np.hypot(*(field[:, 64:-64, 64:-64] - true_shift[:, None, None])))
        assert err < 0.15, f"band {k + 1}: median error {err:.3f} px"
        assert reg.cv.median_px >= err, f"band {k + 1}: reported error understates truth"

    _, _, _, reg = register_band(corrected[ref], corrected[14], valid[14])
    assert reg.passes[0].n_inliers == 0  # flagged: coarse shift only
    assert reg.residual.median_px > 1.0  # and the residual says so
