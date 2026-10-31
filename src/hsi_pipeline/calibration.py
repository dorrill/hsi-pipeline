"""Sanity check of the metadata offsets against the image data, with a logged repair.

Some bands of the challenge capture carry offsets of +41 to +68 while every
other band sits between about -7 and +3. Applied as given, they put a step of
roughly 2.5x into the corrected spectrum over 2-3 nm, which no physical scene
produces, while the raw DNs of those bands are continuous with (and correlate
at r > 0.96 with) their spectral neighbours. The offsets, not the pixels, are
wrong.

The check has two independent stages so a repair never rests on one heuristic:

1. **Metadata outlier test.** An offset is suspect if its robust z-score
   (distance from the median offset in units of 1.4826 * MAD) exceeds
   ``OFFSET_OUTLIER_Z``. This uses the metadata only.
2. **Spectral-continuity test on the data.** For each suspect band, the
   scene-median corrected level is compared with the value linearly
   interpolated in wavelength from the non-suspect bands, once with the
   metadata offset and once with the median of the non-suspect offsets. The
   offset is replaced only if the replacement fits the neighbours better. A
   large offset that the data supports is therefore kept. This is a relative
   comparison, so it needs no tolerance and holds under either gain/offset
   convention.

Every band also gets a leave-one-out continuity residual as a diagnostic. Bands
above tolerance that were not repaired (e.g. a suspicious gain) are logged as
warnings but left unchanged: a gain cannot be re-estimated from neighbours
without assuming the scene spectrum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .io import BandInfo
from .radiometry import Convention, correct_dn

log = logging.getLogger(__name__)

#: Converts a median absolute deviation to a Gaussian-equivalent standard deviation.
MAD_TO_SIGMA = 1.4826

#: Robust z-score above which a metadata offset is suspect. In the challenge data
#: normal offsets lie within |z| <= 1.7 and the suspect ones at z = 13.7-22.3, so
#: 5 leaves a wide margin on both sides.
OFFSET_OUTLIER_Z = 5.0

#: Relative deviation from the neighbour interpolation above which a band is
#: logged as spectrally inconsistent (diagnostic only, nothing is changed). In the
#: challenge data, bands with sound metadata reach at most 0.25 (band 2, whose
#: prediction is pulled by band 1, and the 628/632 nm gain-group boundary);
#: band 1 sits at 0.72. The corrected level is mostly gain x the ~112 DN pedestal
#: the offsets leave in, so where the gain jumps (band 1 has twice band 2's) the
#: residual tracks the gain ratio, not the spectrum: with 100 DN removed band 1
#: drops to 0.23. The offset repairs do not change for any pedestal 0-112 DN.
CONTINUITY_TOLERANCE = 0.5

N_DN_LEVELS = 256


@dataclass(frozen=True)
class OffsetCheck:
    """Outcome of the offset check for one band.

    Attributes:
        band_id: Identifier as given in the metadata.
        wavelength_nm: Band centre wavelength in nanometres.
        offset_metadata: Offset as read from the metadata.
        offset_applied: Offset used for the correction (differs only if repaired).
        offset_z: Robust z-score of the metadata offset among all bands.
        continuity_residual: Relative deviation of the band's corrected median
            level from the leave-one-out neighbour interpolation, using
            ``offset_applied``.
        repaired: True if the metadata offset was replaced.
    """

    band_id: str
    wavelength_nm: float
    offset_metadata: float
    offset_applied: float
    offset_z: float
    continuity_residual: float
    repaired: bool


def band_median_dn(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Exact median DN of each band over its valid pixels.

    Uses a 256-bin histogram per band, so it needs no float copy of the cube.

    Args:
        cube: ``(n_bands, H, W)`` uint8 raw DN.
        valid: ``(n_bands, H, W)`` bool, False for no-data pixels.

    Returns:
        ``(n_bands,)`` float64 median DN; NaN for a band with no valid pixels.
    """
    medians = np.full(cube.shape[0], np.nan)
    for i in range(cube.shape[0]):
        counts = np.bincount(cube[i][valid[i]], minlength=N_DN_LEVELS)
        total = counts.sum()
        if total:
            medians[i] = np.searchsorted(np.cumsum(counts), total / 2)
    return medians


def continuity_residuals(
    level: np.ndarray, wavelength_nm: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Relative deviation of each band from a neighbour interpolation.

    Each band is predicted by linear interpolation in wavelength between the
    nearest ``reference`` bands below and above it, excluding the band itself.
    At the ends of the spectrum the nearest reference band is used as is.

    Args:
        level: ``(n_bands,)`` scene-level corrected value per band.
        wavelength_nm: ``(n_bands,)`` ascending band centres in nanometres.
        reference: ``(n_bands,)`` bool, bands trusted as interpolation nodes.

    Returns:
        ``(n_bands,)`` float64 ``(level - predicted) / predicted``.
    """
    out = np.empty(len(level))
    for k in range(len(level)):
        nodes = reference.copy()
        nodes[k] = False
        predicted = np.interp(wavelength_nm[k], wavelength_nm[nodes], level[nodes])
        out[k] = (level[k] - predicted) / predicted
    return out


def check_offsets(
    cube: np.ndarray,
    bands: list[BandInfo],
    valid: np.ndarray,
    convention: Convention = "linear",
    repair: bool = True,
) -> tuple[list[BandInfo], list[OffsetCheck]]:
    """Detect metadata offsets inconsistent with the data and optionally replace them.

    See the module docstring for the method.

    Args:
        cube: ``(n_bands, H, W)`` uint8 raw DN, ordered as ``bands``.
        bands: Band metadata sorted by wavelength.
        valid: ``(n_bands, H, W)`` bool, False for no-data pixels.
        convention: Gain/offset formula used for the correction.
        repair: If False, only report; the returned bands equal ``bands``.

    Returns:
        The bands with repaired offsets where applicable, and one
        :class:`OffsetCheck` per band.
    """
    wavelength_nm = np.array([b.wavelength_nm for b in bands])
    gain = np.array([b.gain for b in bands])
    offset_meta = np.array([b.offset for b in bands])
    dn_median = band_median_dn(cube, valid)

    median_offset = np.median(offset_meta)
    mad = np.median(np.abs(offset_meta - median_offset))
    offset_z = (offset_meta - median_offset) / (MAD_TO_SIGMA * mad) if mad else 0 * offset_meta
    suspect = np.abs(offset_z) > OFFSET_OUTLIER_Z
    replacement = float(np.median(offset_meta[~suspect]))

    # Compare each suspect band with the trusted bands only, so a run of adjacent
    # bad bands (42-44) cannot vouch for each other.
    level_meta = correct_dn(dn_median, gain, offset_meta, convention)
    level_repl = correct_dn(dn_median, gain, np.full_like(offset_meta, replacement), convention)
    res_meta = continuity_residuals(level_meta, wavelength_nm, ~suspect)
    res_repl = continuity_residuals(level_repl, wavelength_nm, ~suspect)
    do_repair = repair & suspect & (np.abs(res_repl) < np.abs(res_meta))

    offset_applied = np.where(do_repair, replacement, offset_meta)
    level_applied = correct_dn(dn_median, gain, offset_applied, convention)
    residual = continuity_residuals(level_applied, wavelength_nm, np.ones(len(bands), bool))

    checks = []
    for b, z, off, res, suspect_flag, fixed in zip(
        bands, offset_z, offset_applied, residual, suspect, do_repair, strict=True
    ):
        if fixed:
            log.warning(
                "Band %s (%.1f nm): metadata offset %+.1f is an outlier (z=%.0f) and fits "
                "the spectral neighbours worse than %+.2f; replaced",
                b.band_id,
                b.wavelength_nm,
                b.offset,
                z,
                off,
            )
        elif suspect_flag:
            log.warning(
                "Band %s (%.1f nm): metadata offset %+.1f is an outlier (z=%.0f) but %s",
                b.band_id,
                b.wavelength_nm,
                b.offset,
                z,
                "repair is disabled" if not repair else "the data supports it; kept",
            )
        elif abs(res) > CONTINUITY_TOLERANCE:
            log.warning(
                "Band %s (%.1f nm): level deviates %+.0f%% from its spectral neighbours "
                "(gain %.3f, offset %+.1f); left unchanged",
                b.band_id,
                b.wavelength_nm,
                100 * res,
                b.gain,
                b.offset,
            )
        checks.append(
            OffsetCheck(
                band_id=b.band_id,
                wavelength_nm=b.wavelength_nm,
                offset_metadata=b.offset,
                offset_applied=float(off),
                offset_z=float(z),
                continuity_residual=float(res),
                repaired=bool(fixed),
            )
        )
    repaired_bands = [
        replace(b, offset=float(off)) for b, off in zip(bands, offset_applied, strict=True)
    ]
    return repaired_bands, checks
