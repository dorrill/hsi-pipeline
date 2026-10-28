"""Unit tests for metadata parsing, raw loading and radiometric correction."""

import json

import numpy as np
import pytest

from hsi_pipeline.io import load_cube, load_metadata, natural_key, read_band
from hsi_pipeline.radiometry import apply_radiometric_correction, validity_masks

SHAPE = (8, 8)


def _write_bands(tmp_path, n=3):
    for i in range(n):
        np.full(SHAPE, 10 * (i + 1), np.uint8).tofile(tmp_path / f"band_{i + 1}.raw")


def test_natural_sort():
    """band_10 sorts after band_2, unlike plain lexicographic order."""
    names = ["band_10.raw", "band_2.raw", "band_1.raw"]
    assert sorted(names, key=natural_key) == ["band_1.raw", "band_2.raw", "band_10.raw"]


@pytest.mark.parametrize("layout", ["list", "bands_key", "mapping", "parallel"])
def test_metadata_layouts(tmp_path, layout):
    """All supported JSON layouts parse to the same spectrally sorted bands."""
    _write_bands(tmp_path)
    wl, g, o = [700.0, 500.0, 600.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]
    entries = [{"wavelength": w, "gain": a, "offset": b} for w, a, b in zip(wl, g, o, strict=True)]
    raw = {
        "list": entries,
        "bands_key": {"bands": entries},
        "mapping": {f"b{i}": e for i, e in enumerate(entries)},
        "parallel": {"wavelength": wl, "gain": g, "offset": o},
    }[layout]
    (tmp_path / "band_metadata.json").write_text(json.dumps(raw))

    bands = load_metadata(tmp_path / "band_metadata.json")
    assert [b.wavelength_nm for b in bands] == [500.0, 600.0, 700.0]  # sorted spectrally
    assert [b.index for b in bands] == [0, 1, 2]
    assert bands[0].path.name == "band_2.raw"  # paired by file order


def test_challenge_layout_matches_by_band_number(tmp_path):
    """Keys '1','2','20' with files b1/b2/b20.raw; band 20 has the shortest wavelength."""
    for n in (1, 2, 20):
        np.full(SHAPE, n, np.uint8).tofile(tmp_path / f"b{n}.raw")
    meta = {
        "1": {"wavelength": 600, "gain": 1, "offset": 0},
        "2": {"wavelength": 650, "gain": 1, "offset": 0},
        "20": {"wavelength": 480, "gain": 1, "offset": 0},
    }
    (tmp_path / "band_metadata.json").write_text(json.dumps(meta))
    bands = load_metadata(tmp_path / "band_metadata.json")
    assert [(b.band_id, b.path.name) for b in bands] == [
        ("20", "b20.raw"),
        ("1", "b1.raw"),
        ("2", "b2.raw"),
    ]


def test_raw_size_mismatch_is_reported(tmp_path):
    """A file whose size is not H*W bytes is rejected, naming the byte difference."""
    p = tmp_path / "bad.raw"
    np.zeros(SHAPE[0] * SHAPE[1] + 16, np.uint8).tofile(p)
    with pytest.raises(ValueError, match="difference 16 bytes"):
        read_band(p, SHAPE)


def test_correction_and_nodata(tmp_path):
    """Both gain/offset conventions are applied; DN 0 -> NaN and DN 255 -> saturated."""
    _write_bands(tmp_path, 2)
    meta = [
        {"filename": "band_1.raw", "wavelength": 500, "gain": 0.5, "offset": 1.0},
        {"filename": "band_2.raw", "wavelength": 600, "gain": 2.0, "offset": -1.0},
    ]
    (tmp_path / "band_metadata.json").write_text(json.dumps(meta))
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
