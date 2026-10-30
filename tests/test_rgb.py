"""Unit tests for SNR estimation, RGB band selection and the RGB composite."""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from hsi_pipeline.io import BandInfo
from hsi_pipeline.rgb import (
    BandSNR,
    band_snr,
    channel_weights,
    combined_noise,
    noise_sigma,
    rgb_composite,
    select_rgb,
)


def _scene(shape=(512, 512), seed=0):
    """Textured scene around 120 DN with a 15 DN standard deviation."""
    f = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), 4)
    return 120 + 15 * f / f.std()


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 4.0])
def test_noise_sigma_recovers_known_noise_on_integer_data(sigma):
    """Noise is recovered within 5 % on DN-quantised data, texture notwithstanding.

    Quantisation adds 1/sqrt(12) DN of noise, which is real noise in the data.
    """
    rng = np.random.default_rng(1)
    img = np.rint(_scene() + rng.normal(0, sigma, (512, 512)))
    expected = np.hypot(sigma, 1 / np.sqrt(12))
    assert noise_sigma(img, np.ones(img.shape, bool)) == pytest.approx(expected, rel=0.05)


def test_band_snr_ignores_gain_and_offset():
    """SNR is a property of the data, not of the radiometric coefficients."""
    rng = np.random.default_rng(2)
    dn = np.rint(_scene() + rng.normal(0, 1.0, (512, 512)))
    cube = np.stack([dn, 0.3 * dn + 60.0]).astype(np.float32)
    bands = [BandInfo(i, str(i), 500.0 + i, 1.0, 0.0, Path("x")) for i in range(2)]
    ones = np.ones(cube.shape, bool)
    snr = band_snr(cube, bands, ones, ~ones)
    assert snr[0].snr == pytest.approx(snr[1].snr, rel=1e-3)
    assert snr[0].snr == pytest.approx(15 / np.hypot(1.0, 1 / np.sqrt(12)), rel=0.1)


def test_select_rgb_takes_highest_snr_inside_each_window():
    """Bands outside the windows are ignored even if their SNR is higher."""
    table = [(480, 5.0), (490, 7.0), (550, 6.0), (600, 99.0), (640, 4.0), (680, 8.0)]
    snr = [BandSNR(str(i), w, 1.0, s, s) for i, (w, s) in enumerate(table)]
    assert select_rgb(snr) == {"R": 5, "G": 2, "B": 1}


def test_select_rgb_reports_an_empty_window():
    """A capture without a blue band raises a clear error."""
    snr = [BandSNR("0", 550.0, 1.0, 5.0, 5.0), BandSNR("1", 650.0, 1.0, 5.0, 5.0)]
    with pytest.raises(ValueError, match="B window"):
        select_rgb(snr)


def test_rgb_composite_reserves_zero_for_nodata():
    """Pixels invalid in any channel are 0 in all; valid pixels lie in 1..255."""
    cube = np.stack([_scene(seed=s) for s in range(3)]).astype(np.float32)
    valid = np.ones(cube.shape, bool)
    valid[1, :10, :] = False
    rgb, joint = rgb_composite(cube, valid, {"R": {0: 1.0}, "G": {1: 1.0}, "B": {2: 1.0}})
    assert rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8
    assert (rgb[:10] == 0).all() and not joint[:10].any()
    assert rgb[10:].min() >= 1


def test_channel_weights_are_inverse_variance_inside_the_window():
    """Neighbours outside the window are skipped; a noisier band gets less weight."""
    table = [(560, 1.0), (600, 1.0), (625, 1.0), (628, 2.0), (632, 1.0)]  # (nm, noise)
    snr = [BandSNR(str(i), w, n, 5.0, 5.0 / n) for i, (w, n) in enumerate(table)]
    chosen = {"R": 2, "G": 0, "B": 0}
    assert channel_weights(snr, chosen, 0)["R"] == {2: 1.0}
    w = channel_weights(snr, chosen, 1)["R"]  # band 1 (600 nm) is outside 620-700
    assert set(w) == {2, 3}
    assert w[2] == pytest.approx(0.8) and w[3] == pytest.approx(0.2)  # 1 : 1/4
    assert combined_noise(snr, w) < snr[2].noise


def test_weighted_composite_matches_single_band_when_unweighted():
    """A one-band channel is the band itself; averaging two equal bands changes nothing."""
    cube = np.stack([_scene(seed=s) for s in range(3)]).astype(np.float32)
    valid = np.ones(cube.shape, bool)
    single, _ = rgb_composite(cube, valid, {"R": {0: 1.0}, "G": {1: 1.0}, "B": {2: 1.0}})
    pair = np.concatenate([cube, cube[1:2]])  # band 3 duplicates band 1
    valid4 = np.ones(pair.shape, bool)
    avg, _ = rgb_composite(pair, valid4, {"R": {0: 1.0}, "G": {1: 0.5, 3: 0.5}, "B": {2: 1.0}})
    assert np.array_equal(single, avg)
