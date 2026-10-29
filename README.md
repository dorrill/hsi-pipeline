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
| `rgb_image.tif` | uint8 (H, W, 3) display composite of the registered R, G, B bands; 0 = no-data in any channel (see §5) |
| `band_selection.json` | Per-band noise, contrast and SNR, the colour windows, and the chosen bands |
| `cloud_mask.tif` | uint8 (H, W) on the same grid as `rgb_image.tif`: 1 = cloud, 0 = clear **or no-data** (no-data = 0 in `rgb_image.tif`) |
| `cloud_stats.json` | Bands used, the thresholds found on the data, cloud fraction and object count |
| `registration_stats.json` | Per band: coarse shift, field range, tiles used per pass, spline smoothing, residual misregistration after warping, blocked cross-validation error of the field; and `rgb`: misregistration measured between each pair of RGB bands (see §4) |
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
  With the green band (27) as reference this removes up to ~75 rows at the top
  for bands 45–80 and up to ~45 rows at the bottom for bands 1–8. The RGB image
  is valid in 97.5 % of the frame.
- In the outputs, no-data is 0 in all three channels of `rgb_image.tif`, and 0
  (not cloud) in `cloud_mask.tif`; the joint no-data area is the same in both.

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
For each band, relative to the reference (default: the chosen green band, so one
RGB channel is never resampled; `--reference-band`):

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
   used. The result follows the sensor geometry: smooth fields for bands close
   to the reference on the detector (small time offset), flexible ones for
   distant bands, whose fields carry the along-track stretch.

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

Reference band 27 (556 nm):

| Bands | λ (nm) | Smoothing | Residual median / p95 / max (px) | CV median / p95 (px) |
|---|---|---|---|---|
| 1–8 | 470–491 | 1e-5 | 0.014–0.022 / ≤ 0.09 / ≤ 0.71 | 0.17–0.26 / 0.69–0.96 |
| 20–28 | 538–559 | 1e-3, 1e-2 | 0.032–0.051 / ≤ 0.17 / ≤ 0.58 | 0.09–0.14 / 0.34–0.48 |
| 40–44 | 602–628 | 1e-5 | 0.010–0.014 / ≤ 0.07 / ≤ 0.31 | 0.15–0.19 / 0.66–0.81 |
| 45–64 | 632–686 | 1e-5 | 0.010–0.020 / ≤ 0.14 / ≤ 1.89 | 0.16–0.20 / 0.88–1.16 |
| 75–80 | 712–729 | 1e-5 | 0.010 / ≤ 0.06 / ≤ 0.26 | 0.19–0.21 / 0.87–1.25 |

**Between the chosen RGB bands** (`rgb.pairs_after_registration`, re-measured
on non-overlapping tiles of the registered bands):

| Pair | Tiles | Median (px) | p95 (px) | Max (px) | Mean (dy, dx) (px) |
|---|---|---|---|---|---|
| R–G (43 vs 27) | 169 | 0.010 | 0.060 | 0.15 | (−0.002, −0.003) |
| B–G (7 vs 27) | 170 | 0.014 | 0.079 | 0.27 | (0.000, −0.001) |
| R–B (43 vs 7) | 168 | 0.041 | 0.202 | 0.41 | (−0.004, +0.006) |

The CV error of the R and B fields (accuracy expected away from textured
tiles, e.g. over water) is 0.17 / 0.18 px median and 0.81 / 0.75 px p95. Before
registration the B band was displaced from G by 4–45 px and the R band by
3–22 px, varying over the frame.

**Validation against ground truth** (`tests/test_registration.py`): on
`tools/make_synthetic.py` data (known per-band shifts, 1–5 DN noise, faint land
texture, a no-data strip that does not move), bands 40–120 nm from the reference
are recovered to 0.08–0.12 px median, and the reported CV error is above the true
error for every band. A smooth non-affine field on a textured scene is recovered
to 0.03 px median.

### 5. RGB band selection (SNR) and composite (`rgb.py`)

**SNR** is measured per band on the corrected, destriped bands *before*
registration (resampling smooths noise and would inflate the SNR of every band
except the reference):

- *Noise*: Immerkær's estimator (CVIU 1996). The 3×3 operator
  `[[1,−2,1],[−2,4,−2],[1,−2,1]]` cancels locally linear signal; for white
  noise its mean absolute response is 6σ√(2/π). It is averaged over the flattest
  50 % of pixels (lowest Sobel gradient), so edges and texture are not counted
  as noise. A median-based version is not usable: on integer DN it returns
  only a few quantised values (0.99 or 1.24 DN for every band). On synthetic
  integer data with known noise the estimate is within 1 %.
- *Signal*: the scene contrast, i.e. the standard deviation of the band with
  the noise removed in quadrature.

`SNR = contrast / noise`. Gain cancels and offset drops out, so the ranking does
not depend on the doubtful calibration (band 1's gain, the ~110 DN floor). A
mean-based SNR would mostly measure that floor.

**Selection**: the highest-SNR band in each conventional colour window, blue
450–495, green 495–570, red 620–700 nm:

| Colour | Band | λ (nm) | SNR | Runner-up |
|---|---|---|---|---|
| R | 43 | 625.5 | 11.8 | 63 (683.5 nm), 9.9 |
| G | 27 | 556.0 | 6.6 | 28 (559 nm), 6.5 |
| B | 7 | 486.0 | 6.4 | 8 (491 nm), 6.4 |

Noise is 0.9–1.2 DN in every band, close to the read-plus-quantisation floor,
so the SNR differences come from how many DN the scene spans. The low-gain
bands (40–44, 75–80, gain ≈ 0.3) span about twice as many DN, so they are the
most sensitive, which puts R on band 43. Its repaired offset does not affect
either the SNR or the display stretch.

**Composite**: the three registered bands, each stretched linearly from its
1st to 99th percentile onto 1–255; 0 marks pixels invalid in any channel.

This balances the scene *average* to grey. Because the land here is redder than
the clouds, the clouds come out slightly cyan. A white balance on the cloud mask
(clouds are spectrally flat) was tried and not kept: it made the clouds neutral
but turned the land magenta and the water purple, and clipped 8 % of the red
channel. Clouds sit above most of the hazy atmosphere while the surface signal
passes through all of it, so a cloud-based white point does not suit the surface.
Natural colour would need atmospheric correction, which the metadata does not
support.

### 6. Cloud mask (`cloud.py`)

There is no NIR or SWIR, so the usual tests (bright NIR, SWIR cirrus band) are
unavailable. Two visible-range facts remain:

- clouds are much brighter than the water they lie over;
- clouds are spectrally flat in reflectance, so their radiance follows the solar
  spectrum, while every land surface here (vegetation's red edge, soil, roofs,
  sand) gets relatively brighter towards the longest band, 729 nm.

Radiances add, so every mixture of water and cloud (thin cloud, cloud edges)
lies on one straight line in the plane L(729) vs L(556), the *cloud line*; land
lies above it. On the registered green (27) and 729 nm (80) bands, after a 3×3
mean (single-pixel SNR is ~6):

1. **Brightness**: green radiance more than 5σ above the water mode, where σ is
   the water cluster's noise measured on its dark side (free of cloud and
   land). Water noise alone passes with probability ~3·10⁻⁷. Here: mode 69.9,
   σ 1.19, threshold 75.8.
2. **Spectral shape**: the *red-edge excess* is the height above the cloud line,
   fitted to the lower envelope of the bright pixels (slope 0.954). Among bright
   pixels its histogram is bimodal (cloud ≈ +1, land ≈ +4). The threshold is the
   minimum of the smoothed histogram between the modes (2.09), so it comes from
   the data. Otsu's method was tried first and put the split at 3.0, inside the
   land mode, because it is biased towards the larger class. 90 % of cloud
   pixels lie below 1.6; 90 % of airfield and beach pixels lie above 2.4.
3. **Shape**: an opening with a 7 px disk removes features up to ~4 px wide
   (beach strip, roof edges, piers); objects under 100 px are dropped; holes are
   filled; a 2 px buffer covers mixed edge pixels and the small cloud parallax
   (clouds sit above the ground the bands were registered to).

Result: 4.2 % of the valid frame, 83 objects (`figures/cloud_mask_overlay.png`
from `tools/make_figures.py`). The tests build a synthetic scene from the same
physics (water, red-edge land, soft clouds mixed linearly with water, and
cloud-coloured lines and blobs) and check that definite cloud is all found, and
that land, thin or tiny decoys, and water beyond the buffer are not.

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
- The per-channel stretch balances the scene average to grey; because the land
  is redder than the clouds, clouds look cyan. A white balance on cloud pixels
  would fix this. A slow across-track response difference between bands
  (reddish top-right corner) and the wider column banding (coloured streaks
  over water) are not corrected.
- Each colour uses one band. Averaging neighbouring bands would raise the SNR
  further, at the cost of spectral purity and an extra registration step.
- Cloud mask: bright, spectrally grey man-made surfaces of cloud-like size
  (a few white roofs, the Shelter Island marina) are false alarms; the faintest
  haze at cloud margins is missed (below the 5σ brightness test). Thin cloud over
  land is harder than over water, because the land below raises the red-edge
  excess; in this scene the clouds are over the ocean. Each band's response
  drifts across the swath (−2 to +3.7 % per 1000 columns on open water, measured
  per band; 729 nm drifts more than 556 nm), which slightly lowers the excess
  towards the left of the frame.
- No cloud-shadow mask.
- TODO

## Project layout
```
src/hsi_pipeline/   io.py (loading), radiometry.py (correction, validity, destriping),
                    calibration.py (offset consistency check),
                    registration.py (co-registration and its evaluation),
                    rgb.py (SNR, band choice, composite), cloud.py (cloud mask),
                    cli.py (entry point)
tools/              make_figures.py (summary figures, needs the viz extra)
                    make_synthetic.py (test data with known per-band shifts)
tests/              unit tests
```
