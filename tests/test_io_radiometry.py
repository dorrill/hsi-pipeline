"""Unit tests for metadata parsing, raw loading and radiometric correction."""

import json

import numpy as np
import pytest

from hsi_pipeline.io import load_cube, load_metadata, read_band
from hsi_pipeline.radiometry import apply_radiometric_correction, validity_masks

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
