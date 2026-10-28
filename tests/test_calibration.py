"""Unit tests for the metadata offset consistency check."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hsi_pipeline.calibration import band_median_dn, check_offsets
from hsi_pipeline.io import BandInfo
from hsi_pipeline.radiometry import validity_masks

SHAPE = (32, 32)
WAVELENGTH_NM = np.arange(500.0, 610.0, 10.0)  # 11 bands
TRUE_OFFSET = np.array([-2.0, -1.5, -3.0, -2.5, -1.0, -2.0, -3.5, -1.5, -2.5, -2.0, -3.0])
GAIN = np.linspace(0.5, 1.0, len(WAVELENGTH_NM))
BAD_BAND = 5


def _capture(true_offset):
    """Cube whose true corrected spectrum is smooth, generated with ``true_offset``.

    Returns the uint8 cube and band metadata carrying the true offsets.
    """
    rng = np.random.default_rng(0)
    texture = rng.uniform(0.9, 1.1, SHAPE)
    level = 60.0 + 0.2 * (WAVELENGTH_NM - 500.0)  # smooth corrected spectrum
    radiance = level[:, None, None] * texture
    dn = (radiance - true_offset[:, None, None]) / GAIN[:, None, None]
    cube = np.clip(np.rint(dn), 1, 254).astype(np.uint8)
    bands = [
        BandInfo(i, str(i), float(w), float(g), float(o), Path(f"b{i}.raw"))
        for i, (w, g, o) in enumerate(zip(WAVELENGTH_NM, GAIN, true_offset, strict=True))
    ]
    return cube, bands


def _with_offset(bands, k, offset):
    """Copy of ``bands`` with band ``k``'s metadata offset set to ``offset``."""
    return [replace(b, offset=offset) if b.index == k else b for b in bands]


def test_band_median_dn_matches_numpy():
    """The histogram median equals np.median over valid pixels."""
    rng = np.random.default_rng(1)
    cube = rng.integers(0, 256, (3, *SHAPE), dtype=np.uint8)
    valid, _ = validity_masks(cube)
    expected = [np.median(cube[i][valid[i]]) for i in range(3)]
    # np.median averages the two middle values for even counts; ours returns the lower one.
    assert band_median_dn(cube, valid) == pytest.approx(expected, abs=0.5)


def test_wrong_metadata_offset_is_repaired():
    """A +60 metadata offset on data generated with ~-2 is replaced; other bands untouched."""
    cube, bands = _capture(TRUE_OFFSET)
    bands = _with_offset(bands, BAD_BAND, 60.0)
    valid, _ = validity_masks(cube)

    fixed, checks = check_offsets(cube, bands, valid)

    assert [c.repaired for c in checks] == [k == BAD_BAND for k in range(len(bands))]
    assert fixed[BAD_BAND].offset == pytest.approx(np.median(np.delete(TRUE_OFFSET, BAD_BAND)))
    assert checks[BAD_BAND].offset_metadata == 60.0
    assert abs(checks[BAD_BAND].continuity_residual) < 0.05
    assert [b.offset for k, b in enumerate(fixed) if k != BAD_BAND] == list(
        np.delete(TRUE_OFFSET, BAD_BAND)
    )


def test_outlier_offset_supported_by_data_is_kept():
    """A large offset that the DNs were really generated with passes the continuity test."""
    true_offset = TRUE_OFFSET.copy()
    true_offset[BAD_BAND] = 40.0
    cube, bands = _capture(true_offset)
    valid, _ = validity_masks(cube)

    fixed, checks = check_offsets(cube, bands, valid)

    assert checks[BAD_BAND].offset_z > 5  # flagged by the metadata test...
    assert not any(c.repaired for c in checks)  # ...but cleared by the data
    assert fixed == bands


def test_repair_disabled_reports_only():
    """With repair=False the bands are returned unchanged."""
    cube, bands = _capture(TRUE_OFFSET)
    bands = _with_offset(bands, BAD_BAND, 60.0)
    valid, _ = validity_masks(cube)

    fixed, checks = check_offsets(cube, bands, valid, repair=False)

    assert fixed == bands
    assert not any(c.repaired for c in checks)
    assert checks[BAD_BAND].continuity_residual > 0.5
