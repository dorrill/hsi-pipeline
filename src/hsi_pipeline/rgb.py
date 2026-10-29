"""SNR-based choice of the R, G and B bands, and the RGB composite.

SNR definition:
    ``SNR = contrast / noise`` per band, on the corrected, destriped band
    *before* registration (resampling correlates and smooths noise, which
    would inflate the SNR of every band except the reference).

    * noise: Immerkaer, "Fast noise variance estimation", CVIU 64 (1996). The
      3x3 operator ``[[1,-2,1],[-2,4,-2],[1,-2,1]]`` cancels any locally linear
      signal; for white Gaussian noise the mean absolute response is
      ``6 sigma sqrt(2/pi)``. Texture and edges would inflate it, so only the
      flattest ``FLAT_FRACTION`` of pixels (lowest Sobel gradient) are used.
      A median-based (MAD) version is not usable here: on integer DN its
      result is quantised to a few values (e.g. 0.99 or 1.24 DN for every band).
    * contrast: standard deviation of the scene with the noise removed in
      quadrature, ``sqrt(var - sigma^2)``.

    Both scale with the gain and ignore the offset, so the ratio does not
    depend on the radiometric calibration, which is doubtful for some bands
    (band 1's gain, a ~110 DN floor in every band). A mean-based SNR would be
    dominated by that floor.

Selection:
    The band with the highest SNR inside each colour window (``RGB_WINDOWS_NM``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .io import BandInfo

#: Conventional spectral colour boundaries (nm). The capture covers 470-729 nm
#: with gaps at ~491-538, 559-602 and 685-712 nm, so the candidates are
#: 470-491 nm (B), 538-559 nm (G) and 625-686 nm (R).
RGB_WINDOWS_NM = {"R": (620.0, 700.0), "G": (495.0, 570.0), "B": (450.0, 495.0)}

#: Immerkaer's noise operator; for white Gaussian noise of std s its mean
#: absolute response is 6 s sqrt(2 / pi).
_LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_MEAN_ABS_TO_SIGMA = np.sqrt(np.pi / 2) / 6

#: Fraction of pixels, lowest local gradient first, used for the noise estimate.
#: Half the frame keeps water, flat land and cloud interiors and drops edges and
#: texture, which would otherwise be counted as noise.
FLAT_FRACTION = 0.5

#: Display stretch of the RGB composite: per-channel percentiles mapped to 1 and
#: 255 (0 is reserved for no-data). Per-channel stretching acts as a simple white
#: balance; 1/99 keeps cloud tops from compressing the land and water.
STRETCH_PERCENTILES = (1.0, 99.0)


@dataclass(frozen=True)
class BandSNR:
    """Signal and noise of one band.

    Attributes:
        band_id: Identifier as given in the metadata.
        wavelength_nm: Band centre wavelength in nanometres.
        noise: Noise standard deviation, in corrected units.
        contrast: Noise-free scene standard deviation, in corrected units.
        snr: ``contrast / noise``.
    """

    band_id: str
    wavelength_nm: float
    noise: float
    contrast: float
    snr: float


def noise_sigma(image: np.ndarray, usable: np.ndarray) -> float:
    """Robust white-noise standard deviation of one image.

    Args:
        image: ``(H, W)`` float image.
        usable: ``(H, W)`` bool, pixels that may be used (valid, unsaturated).

    Returns:
        Noise standard deviation in the image's units.
    """
    filled = np.where(usable, image, 0.0)
    response = np.abs(ndimage.convolve(filled, _LAPLACIAN, mode="nearest"))
    gradient = np.hypot(ndimage.sobel(filled, 0), ndimage.sobel(filled, 1))
    # Only responses whose whole 3x3 support is usable, then the flattest of those.
    inner = ndimage.binary_erosion(usable, np.ones((3, 3), bool))
    flat = gradient[inner] <= np.quantile(gradient[inner], FLAT_FRACTION)
    return float(_MEAN_ABS_TO_SIGMA * response[inner][flat].mean())


def band_snr(
    cube: np.ndarray, bands: list[BandInfo], valid: np.ndarray, saturated: np.ndarray
) -> list[BandSNR]:
    """SNR of every band (see module docstring).

    Args:
        cube: ``(n_bands, H, W)`` float32 corrected, destriped, *unregistered*.
        bands: Band metadata in cube order.
        valid: ``(n_bands, H, W)`` bool validity.
        saturated: ``(n_bands, H, W)`` bool, DN == 255.

    Returns:
        One :class:`BandSNR` per band.
    """
    out = []
    for b in bands:
        usable = valid[b.index] & ~saturated[b.index]
        img = cube[b.index].astype(np.float64)
        sigma = noise_sigma(img, usable)
        contrast = float(np.sqrt(max(img[usable].var() - sigma**2, 0.0)))
        out.append(BandSNR(b.band_id, b.wavelength_nm, sigma, contrast, contrast / sigma))
    return out


def select_rgb(snr: list[BandSNR]) -> dict[str, int]:
    """Pick the highest-SNR band in each colour window.

    Args:
        snr: Per-band SNR in cube order.

    Returns:
        Cube index of the chosen band for ``"R"``, ``"G"`` and ``"B"``.

    Raises:
        ValueError: If a colour window contains no band.
    """
    chosen = {}
    for colour, (lo, hi) in RGB_WINDOWS_NM.items():
        candidates = [i for i, s in enumerate(snr) if lo <= s.wavelength_nm <= hi]
        if not candidates:
            raise ValueError(f"No band in the {colour} window {lo}-{hi} nm")
        chosen[colour] = max(candidates, key=lambda i: snr[i].snr)
    return chosen


def rgb_composite(
    cube: np.ndarray, valid: np.ndarray, chosen: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """8-bit display composite of the chosen (registered) bands.

    Args:
        cube: ``(n_bands, H, W)`` float32 registered corrected values.
        valid: ``(n_bands, H, W)`` bool validity after registration.
        chosen: Cube index per colour, from :func:`select_rgb`.

    Returns:
        ``(H, W, 3)`` uint8 RGB, 0 in all channels where any channel is
        invalid; and the ``(H, W)`` bool joint validity.
    """
    order = [chosen["R"], chosen["G"], chosen["B"]]
    joint = valid[order].all(axis=0)
    rgb = np.zeros((*joint.shape, 3), np.uint8)
    for c, k in enumerate(order):
        lo, hi = np.percentile(cube[k][joint], STRETCH_PERCENTILES)
        scaled = (cube[k] - lo) / (hi - lo) * 254 + 1
        rgb[..., c] = np.where(joint, np.clip(np.rint(scaled), 1, 255), 0).astype(np.uint8)
    return rgb, joint
