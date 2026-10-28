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
synthetic generator is seeded. On the challenge data a run takes ~2 min and
~1.6 GB with the default 4 registration workers (`--workers N`; ~0.5 GB each).

## Outputs

| File | Description |
|---|---|
| `rgb_image.tif` | TODO |
| `cloud_mask.tif` | uint8, same shape as the RGB image, 1 = cloud, 0 = not cloud. TODO |
| `registration_stats.json` | Per band: coarse shift, field range, tiles used per pass, spline smoothing, residual misregistration after warping, and blocked cross-validation error of the field (see §4) |
| `band_summary.json` | Per-band DN range, valid/saturated fractions, corrected mean/std, offset check, removed stripe amplitude |

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

#### Column destriping
Every band has single-column stripes of ~2 DN, twice the random noise (~1 DN
from row-to-row differences), and the pattern is the same in all bands
(r ≈ 0.91 between 470 and 659 nm): a detector-column offset. For each band the
stripe pattern is the column-median profile minus its 15-column running median,
subtracted from every row. This lowers the stripe power ~20× above
0.05 cycles/px; banding wider than the window remains. The removed amplitude is
recorded as `stripe_amplitude`; `--no-destripe` skips the step.

### 3. Invalid pixels
- DN = 0 is treated as no-data (configurable with `--nodata`) and becomes NaN
  after correction.
- DN = 255 is flagged as saturated (≤ 0.03 % of pixels, cloud tops). It stays in
  the image and is excluded from SNR statistics.
- Registration resamples each band; output pixels whose source lies outside the
  frame, or within the cubic-spline support of a no-data pixel, become no-data.
  For bands 1–44 this removes up to ~100 rows at the bottom and ~25 columns on
  the right of the frame (the field reaches dy ≈ 98 px, dx ≈ 26 px).
- TODO: describe how NaNs are handled in the RGB and cloud mask outputs.

### 4. Co-registration (`registration.py`)

#### What the data shows
Tile-by-tile phase correlation against a reference band shows a large, structured
misregistration typical of a push-broom sensor whose bands are separate strips of
detector rows (each band images a ground point at a different time):

- Shifts reach ~50 px and grow linearly with the band **number**, not the
  wavelength: bands 22 and 24 (545, 544 nm) follow their numbers. Between bands
  44 and 45 there is an extra jump of ~(19, 11) px, consistent with a gap
  between two detector segments (the gain/level break in §2 is at the same place).
- The shift is not a translation. dy varies linearly across the swath (~11 px
  over 2048 columns for band 1) and non-monotonically along the rows (time),
  as expected from attitude/velocity variation during the capture.
- In the last ~450 rows, bands of the other segment are stretched along-track by
  up to ~10 % relative to the reference (band 6: dy goes from 41 to 78 px between
  rows 1550 and 1930), while bands of the same segment are not.

A single translation, or an affine model, leaves 0.3–1.3 px; a global cubic
polynomial fails near the bottom edge. The field is therefore modelled
non-parametrically.

#### Method
For each band, relative to the reference (default: the middle of the cube, band
47 at 637.5 nm; `--reference-band`):

1. **Tile shifts by phase correlation** on 128 px tiles every 64 px. No-data is
   mean-filled and a Hann window suppresses FFT edge effects. The normalised
   cross-power spectrum keeps only phase, so the estimate does not depend on the
   band's gain, offset or contrast. A Gaussian spectral weight (σ = 0.08
   cycles/px) down-weights noise-dominated frequencies; on the real data it
   raised the median correlation peak from ~0.1 to ~0.8. The peak is refined to
   0.01 px with the upsampled DFT of Guizar-Sicairos et al. (2008).
2. **Tile quality.** Tiles with a normalised peak below 0.5 are dropped. Open
   water gives noise peaks (< 0.3); land and cloud give 0.6–0.9. The column
   stripes turned out not to matter here: excluding the stripe frequencies, or
   destriping first, left the number of failed water tiles unchanged.
3. **Field.** A smoothing thin-plate spline through the tiles, with iterative
   rejection of tiles > 4 robust σ from it (moving cloud edges, mismatches). It
   follows local structure and bridges the water gaps with minimum bending.
4. **Iteration.** The band is warped (cubic spline) and tiles are re-measured on
   the warped band; the remaining shift is composed into the field. Tiles that
   decorrelated because of local stretch correlate again once most of it is
   removed. 2–3 passes are needed.
5. **Smoothing by cross-validation.** The total shift of each tile (field at its
   centre plus what is still measured there) is fitted with each of six
   smoothings, scored by spatially blocked cross-validation (384 px blocks, so a
   held-out tile's overlapping neighbours are held out too), on a common tile
   set. The most flexible candidate within one standard error of the best is
   used. The result follows the sensor geometry: smooth fields for bands in the
   reference's detector segment, flexible ones for the other segment.

#### Evaluation
`registration_stats.json` gives two error measures per band. They answer
different questions:

- `residual_after_registration`: shifts re-measured between the registered band
  and the reference on non-overlapping tiles, i.e. what is left **where there is
  texture to measure**. It cannot see a tile's own measurement error once a
  flexible field has absorbed it, so it is optimistic for flexible fields.
- `field_blocked_cv_error`: the error of predicting the total tile shift in a
  held-out 384 px block from the rest. It includes per-tile measurement error and
  the difficulty of predicting **into gaps** (open water), so it is pessimistic
  where tiles are dense.

| Bands | λ (nm) | Smoothing | Residual median / p95 / max (px) | CV median / p95 (px) |
|---|---|---|---|---|
| 1–8 | 470–491 | 1e-5 | 0.010–0.014 / ≤ 0.07 / ≤ 2.6 | 0.21–0.29 / 1.4–2.8 |
| 20–28 | 538–559 | 1e-5 | 0.014–0.020 / ≤ 0.12 / ≤ 2.3 | 0.16–0.20 / 1.3–2.1 |
| 40–44 | 602–628 | 1e-5 | 0.010–0.014 / ≤ 0.10 / ≤ 0.73 | 0.14–0.15 / 0.86–1.09 |
| 45–64 | 632–686 | 1e-2 | 0.041–0.058 / ≤ 0.29 / ≤ 0.86 | 0.08–0.10 / 0.31–0.45 |
| 75–80 | 712–729 | 1e-3 | 0.030–0.036 / ≤ 0.10 / ≤ 0.65 | 0.12–0.14 / 0.41–0.46 |

Bands of the other detector segment (1–44) are registered to a few hundredths of
a pixel where there is texture; their larger CV tails come from the stretched
bottom rows, where a held-out block cannot be predicted from its neighbours.

**Validation against ground truth** (`tests/test_registration.py`): on
`tools/make_synthetic.py` data (known per-band shifts, 1–5 DN noise, faint land
texture, a no-data strip that does not move), bands 40–120 nm from the reference
are recovered to 0.08–0.12 px median, and the reported CV error is above the true
error for every band. A smooth non-affine field on a textured scene is recovered
to 0.03 px median.

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
- Column stripes are an additive per-column offset, constant along the rows.
- The band-to-reference displacement is smooth over the frame, and the scene is
  static between band acquisitions: the field is fitted to the ground, so moving
  or high clouds keep a parallax/motion residual (their tiles are rejected as
  outliers rather than registered).
- TODO

## Limitations
- Repaired offsets are a consistent replacement (the median of the normal
  offsets), not a recalibration; absolute levels in those bands carry an
  uncertainty of a few units.
- Band 1's gain looks wrong but is used as given.
- Destriping removes single-column stripes only; banding wider than 15 columns
  remains.
- Every band is registered directly to one reference. Where a band's contrast is
  inverted relative to the reference (e.g. land/sea across the vegetation red
  edge), few tiles correlate: on the synthetic 400–1000 nm data, bands > 500 nm
  from the reference fall back to the coarse shift. This is reported
  (`field_from_tiles: false`, large residual). The challenge data (470–729 nm)
  is not affected.
- Registration error is measured with the same estimator that drives the fit;
  the ground-truth check is on synthetic data only.
- TODO

## Project layout
```
src/hsi_pipeline/   io.py (loading), radiometry.py (correction, validity, destriping),
                    calibration.py (offset consistency check),
                    registration.py (co-registration and its evaluation), cli.py (entry point)
tools/              make_synthetic.py (test data with known per-band shifts)
tests/              unit tests
```
