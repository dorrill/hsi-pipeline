# Hyperspectral Image Processing Pipeline

Processes a 48-band, 2048 × 2048, 8-bit hyperspectral capture into a
radiometrically corrected, co-registered RGB image and a binary cloud mask,
and reports co-registration accuracy between the selected RGB bands.

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for tests, ".[viz]" for plots
hsi-pipeline --input-dir inputs --output-dir outputs
```

Run the tests with `pytest`. To try the pipeline without the real data:

```bash
python tools/make_synthetic.py --out synthetic_inputs
hsi-pipeline --input-dir synthetic_inputs --output-dir outputs_synthetic
```

The run is deterministic: no random state is used in the pipeline, and the
synthetic generator is seeded.

## Outputs

| File | Description |
|---|---|
| `rgb_image.tif` | TODO |
| `cloud_mask.tif` | uint8, same shape as the RGB image, 1 = cloud, 0 = not cloud. TODO |
| `registration_stats.json` | TODO |
| `band_summary.json` | Per-band DN range, valid/saturated fractions, corrected mean/std |

## Method

### 1. Loading and metadata
- Band files are paired with `band_metadata.json` entries by filename when the
  metadata names them, otherwise by natural-sort file order (the counts must match).
- The cube is ordered by ascending wavelength.
- Headerless files must be exactly 2048 × 2048 bytes. Any other size raises an
  error rather than guessing a header offset.
- TODO: describe the actual input format once inspected.

### 2. Radiometric correction
- Corrected value `L = gain · DN + offset`, computed per band in float32
  **before** any band selection or combination.
- `--convention subtract_offset` gives `L = gain · (DN − offset)` instead. The data
  cannot distinguish the two: for every band with plausible metadata the offset is
  only a few units, so the formulas differ by less than the band-to-band variation.

#### Offset consistency check (`calibration.py`)
Five bands carry metadata offsets far outside the rest (all others lie in −7…+3):

| Band | λ (nm) | Metadata offset | Mean with it | Neighbour mean |
|---|---|---|---|---|
| 42, 43, 44 | 608–628 | +64.5, +65.5, +68.4 | 105–112 | ~40 (band 41) |
| 75, 76 | 712, 715 | +57.5, +41.2 | 98, 82 | ~39 (band 77) |

Applied as given, they put a ~2.5× step into the spectrum over 2–3 nm, which no
real scene produces. Their raw DNs, however, are continuous with the neighbouring
bands and correlate with them at r = 0.96–0.99, so the pixels are fine and the
offsets are wrong. Neither convention fixes them (the alternative gives ~20, too low).

The pipeline therefore checks every offset in two independent stages and repairs
it only when both agree:
1. **Metadata outlier:** robust z-score of the offset (median/MAD over all bands)
   above 5. Normal bands reach |z| ≤ 1.7; the five above are at z = 14–22.
2. **Data consistency:** the band's scene-median corrected level is compared with
   a linear interpolation, in wavelength, of the non-suspect bands, once with the
   metadata offset and once with the median of the normal offsets (−2.97). The
   offset is replaced only if the replacement fits better, so a large offset that
   the data supports is kept.

Each repair is logged as a warning and recorded in `band_summary.json`
(`offset_metadata`, `offset_z`, `offset_repaired`; `offset` is the value applied).
`--keep-metadata-offsets` reports without repairing.

Every band also gets a leave-one-out `continuity_residual`. Bands above 0.5 are
logged but **not** changed: band 1 (470 nm, gain 1.368, about twice its
neighbours) comes out 72 % above band 2, which suggests a wrong gain, but a gain
cannot be re-estimated from neighbours without assuming the scene spectrum.

### 3. Invalid pixels
- DN = 0 is treated as no-data (configurable with `--nodata`) and becomes NaN
  after correction.
- DN = 255 is flagged as saturated. It stays in the image but is excluded from
  SNR and registration statistics.
- TODO: describe how NaNs are handled in the RGB and cloud mask outputs.

### 4. Co-registration
TODO

### 5. RGB band selection (SNR)
TODO

### 6. Cloud mask
TODO

### 7. Ship detection (optional)
TODO

## Assumptions
- The gain/offset convention is linear (see above).
- DN = 0 is fill, not real signal.
- The scene-median spectrum is locally smooth in wavelength, so a band that breaks
  continuity while its raw DNs do not has wrong metadata, not a real feature.
- TODO

## Limitations
- Repaired offsets are a consistent replacement (the median of the normal
  offsets), not a recalibration; absolute levels in those bands carry an
  uncertainty of a few units.
- Band 1's gain looks wrong but is used as given.
- TODO

## Project layout
```
src/hsi_pipeline/   io.py (loading), radiometry.py (correction, validity),
                    calibration.py (offset consistency check), cli.py (entry point)
tools/              make_synthetic.py (test data with known per-band shifts)
tests/              unit tests
```
