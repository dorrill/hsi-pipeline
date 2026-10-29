"""Generate a synthetic 48-band capture for exercising the pipeline.

The scene has land, water, bright clouds, a few small "ships" on water, a
no-data border strip, and a known sub-pixel shift per band (written to
``truth.json``) so registration error can be checked against ground truth.

Usage:
    python tools/make_synthetic.py --out synthetic_inputs [--size 2048] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def smooth_field(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Gaussian-filtered white noise rescaled to [0, 1]."""
    f = ndimage.gaussian_filter(rng.standard_normal((size, size)), sigma)
    return (f - f.min()) / (f.max() - f.min())


def main() -> None:
    """Write synthetic bands, ``band_metadata.json`` and ``truth.json`` to ``--out``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("synthetic_inputs"))
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--n-bands", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, wl = args.size, np.linspace(400.0, 1000.0, args.n_bands)
    args.out.mkdir(parents=True, exist_ok=True)

    # Scene layout ---------------------------------------------------------
    water = smooth_field(rng, n, n / 16) < 0.45  # coastline
    cloud = np.clip((smooth_field(rng, n, n / 40) - 0.62) * 6, 0, 1)  # soft cloud opacity
    texture = smooth_field(rng, n, 3)  # land texture

    ships = np.zeros((n, n), bool)
    wy, wx = np.nonzero(water & (cloud == 0))
    for k in rng.choice(len(wy), size=min(8, len(wy)), replace=False):
        y, x = wy[k], wx[k]
        ships[y : y + max(2, n // 400), x : x + max(6, n // 120)] = True

    # Simple reflectance spectra (0–1) ---------------------------------------
    veg = 0.05 + 0.04 * np.exp(-(((wl - 550) / 30) ** 2)) + 0.40 / (1 + np.exp(-(wl - 715) / 12))
    sea = 0.08 * np.exp(-(wl - 400) / 150) + 0.005
    cld = np.full_like(wl, 0.85)
    ship = np.full_like(wl, 0.35)
    solar = np.exp(-(((wl - 500) / 350) ** 2))  # crude irradiance shape

    gains = rng.uniform(0.8, 1.2, args.n_bands) * 1.0 / 255
    offsets = rng.uniform(-0.02, 0.02, args.n_bands)
    noise_sigma_dn = 1.0 + 4.0 * rng.random(args.n_bands)  # band-dependent noise
    shifts = rng.uniform(-3, 3, (args.n_bands, 2))  # (dy, dx) per band
    shifts[0] = 0.0  # band 0 is the reference

    meta, border = {}, max(4, n // 64)
    for i in range(args.n_bands):
        land = veg[i] * (0.7 + 0.6 * texture)
        refl = np.where(water, sea[i], land)
        refl = np.where(ships, ship[i], refl)
        refl = (1 - cloud) * refl + cloud * cld[i]
        radiance = refl * solar[i] * 1.1  # corrected units

        radiance = ndimage.shift(radiance, shifts[i], order=1, mode="nearest")
        dn = (radiance - offsets[i]) / gains[i] + rng.normal(0, noise_sigma_dn[i], (n, n))
        dn = np.clip(np.rint(dn), 1, 255).astype(np.uint8)  # 0 reserved for no-data
        dn[:, :border] = 0  # no-data strip

        # Same layout as the challenge data: key "n" <-> file b<n>.raw
        dn.tofile(args.out / f"b{i + 1}.raw")
        meta[str(i + 1)] = {
            "wavelength": round(float(wl[i]), 2),
            "gain": float(gains[i]),
            "offset": float(offsets[i]),
        }

    (args.out / "band_metadata.json").write_text(json.dumps(meta, indent=2))
    (args.out / "truth.json").write_text(
        json.dumps(
            {
                "reference_band": 1,
                "shifts_dy_dx_px": shifts.round(4).tolist(),
                "noise_sigma_dn": noise_sigma_dn.round(3).tolist(),
                "nodata_border_columns": border,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.n_bands} bands ({n}x{n}) to {args.out}")


if __name__ == "__main__":
    main()
