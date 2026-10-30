"""Unit tests for metadata parsing, raw loading and radiometric correction."""

import json

import numpy as np
import pytest

from hsi_pipeline.io import load_cube, load_metadata, read_band
from hsi_pipeline.radiometry import (
    apply_radiometric_correction,
    column_linear_fit,
    column_offsets,
    destripe_columns,
    validity_masks,
)

SHAPE = (8, 8)


def _write_challenge_layout(tmp_path, meta):
    """Write ``band_metadata.json`` and a constant ``b<n>.raw`` (DN = 10 * n) per key."""
    for key in meta:
        np.full(SHAPE, 10 * int(key), np.uint8).tofile(tmp_path / f"b{key}.raw")
    (tmp_path / "band_metadata.json").write_text(json.dumps(meta))


def test_pairs_by_band_number_and_sorts_by_wavelength(tmp_path):
    """Keys '1','2','20' pair with b1/b2/b20.raw; band 20 has the shortest wavelength."""
    _write_challenge_layout(
        tmp_path,
        {
            "1": {"wavelength": 600, "gain": 1, "offset": 0},
            "2": {"wavelength": 650, "gain": 1, "offset": 0},
            "20": {"wavelength": 480, "gain": 1, "offset": 0},
        },
    )
    bands = load_metadata(tmp_path / "band_metadata.json")
    assert [(b.index, b.band_id, b.path.name) for b in bands] == [
        (0, "20", "b20.raw"),
        (1, "1", "b1.raw"),
        (2, "2", "b2.raw"),
    ]


def test_missing_field_is_reported(tmp_path):
    """An entry without an offset names the band and the missing field."""
    _write_challenge_layout(tmp_path, {"1": {"wavelength": 600, "gain": 1}})
    with pytest.raises(KeyError, match="Band 1: missing \\['offset'\\]"):
        load_metadata(tmp_path / "band_metadata.json")


def test_missing_and_extra_band_files_are_reported(tmp_path):
    """A metadata key without a file, or a file without a key, is an error."""
    meta = {"1": {"wavelength": 600, "gain": 1, "offset": 0}}
    _write_challenge_layout(tmp_path, meta)
    (tmp_path / "b1.raw").unlink()
    with pytest.raises(FileNotFoundError, match="b1.raw"):
        load_metadata(tmp_path / "band_metadata.json")

    _write_challenge_layout(tmp_path, meta)
    np.zeros(SHAPE, np.uint8).tofile(tmp_path / "b7.raw")
    with pytest.raises(ValueError, match="b7.raw"):
        load_metadata(tmp_path / "band_metadata.json")


def test_raw_size_mismatch_is_reported(tmp_path):
    """A file whose size is not H*W bytes is rejected, naming the byte difference."""
    p = tmp_path / "bad.raw"
    np.zeros(SHAPE[0] * SHAPE[1] + 16, np.uint8).tofile(p)
    with pytest.raises(ValueError, match="difference 16 bytes"):
        read_band(p, SHAPE)


def test_correction_and_nodata(tmp_path):
    """Both gain/offset conventions are applied; DN 0 -> NaN and DN 255 -> saturated."""
    _write_challenge_layout(
        tmp_path,
        {
            "1": {"wavelength": 500, "gain": 0.5, "offset": 1.0},
            "2": {"wavelength": 600, "gain": 2.0, "offset": -1.0},
        },
    )
    bands = load_metadata(tmp_path / "band_metadata.json")
    cube = load_cube(bands, SHAPE)
    cube[0, 0, 0] = 0  # no-data
    cube[1, 0, 1] = 255  # saturated

    valid, sat = validity_masks(cube)
    out = apply_radiometric_correction(cube, bands, valid)
    assert np.isnan(out[0, 0, 0])
    assert out[0, 1, 1] == pytest.approx(0.5 * 10 + 1.0)
    assert out[1, 1, 1] == pytest.approx(2.0 * 20 - 1.0)
    assert sat[1, 0, 1] and sat.sum() == 1

    alt = apply_radiometric_correction(cube, bands, valid, convention="subtract_offset")
    assert alt[0, 1, 1] == pytest.approx(0.5 * (10 - 1.0))


def _striped_scene(rng, size=256):
    """Scene with a ramp, a slanted coastline and a saturated cloud, plus known stripes.

    Returns the striped band (NaN no-data in the first 3 columns and 10 rows),
    the stripe-free band, the injected zero-mean offsets and the saturation mask.
    """
    rows, cols = np.mgrid[:size, :size]
    clean = 20.0 + 0.05 * cols + 15.0 * (cols > size // 4 + rows // 2)  # ramp + coastline
    saturated = (rows - 60) ** 2 + (cols - 80) ** 2 < 30**2
    clean[saturated] = 80.0  # clipped cloud top: no column offset survives clipping
    offsets = rng.normal(0.0, 2.0, size)
    offsets -= offsets.mean()
    band = clean + rng.normal(0.0, 1.0, clean.shape) + np.where(saturated, 0.0, offsets)
    band[:, :3] = np.nan
    band[100:110] = np.nan
    return band.astype(np.float32), clean, offsets, saturated


def _column_error(band, clean, saturated):
    """Mean of ``band - clean`` down each valid column, excluding saturated pixels."""
    diff = np.where(saturated, np.nan, band - clean)[:, 3:]
    return np.nanmean(diff, axis=0)


def test_column_offsets_follow_injected_stripes():
    """Estimated offsets track injected ones despite a coastline, a cloud and no-data."""
    band, _, offsets, saturated = _striped_scene(np.random.default_rng(0))
    est = column_offsets(band, saturated)
    # The slow part of the pattern is indistinguishable from scene, so r < 1.
    assert np.corrcoef(est[3:], offsets[3:])[0, 1] > 0.9
    assert np.all(est[:3] == 0)  # no-data columns
    assert est[3:].mean() == pytest.approx(0.0, abs=1e-9)


def test_destripe_removes_column_to_column_stripes_and_keeps_the_level():
    """Column-to-column jumps drop ~10x; the band level and the no-data are unchanged."""
    band, clean, _, saturated = _striped_scene(np.random.default_rng(1))
    cube = band[None].copy()
    amplitude = destripe_columns(cube, saturated[None], workers=2)
    before = np.diff(_column_error(band, clean, saturated)).std()
    after = np.diff(_column_error(cube[0], clean, saturated)).std()
    assert before > 2.5  # sqrt(2) * 2 DN
    assert after < 0.35  # noise alone: sqrt(2 / 230) ~ 0.09
    assert amplitude[0] > 1.5
    assert abs(np.nanmean(cube[0] - clean)) < 0.1
    assert np.isnan(cube[0, :, :3]).all() and np.isnan(cube[0, 100:110]).all()


def test_column_linear_fit_recovers_ramps_despite_outliers():
    """Known per-column gain and offset ramps are recovered with 20 % gross outliers."""
    rng = np.random.default_rng(2)
    h, w = 400, 64
    x = rng.uniform(10.0, 100.0, (h, w))
    gain = np.linspace(0.9, 1.1, w)
    offset = np.linspace(-3.0, 3.0, w)
    y = offset + gain * x + rng.normal(0.0, 0.5, (h, w))
    outliers = rng.random((h, w)) < 0.2
    y[outliers] += rng.uniform(20.0, 60.0, outliers.sum())  # clouds: far too bright
    usable = np.ones((h, w), bool)
    usable[:, 0] = False
    intercept, slope = column_linear_fit(y, x, usable)
    assert np.isnan(intercept[0]) and np.isnan(slope[0])
    np.testing.assert_allclose(slope[1:], gain[1:], atol=0.005)
    np.testing.assert_allclose(intercept[1:], offset[1:], atol=0.3)
