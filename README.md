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

Summary figures (SNR vs wavelength, registration before/after, offset check,
cloud overlay, …) are written to `outputs/figures/` by
`python tools/make_figures.py --input-dir inputs --output-dir outputs` after a run (needs the `viz` extra).

The run is deterministic: no random state is used in the pipeline, and the
synthetic generator is seeded. A fresh clone installed into a new Python 3.13 venv
reproduces all six outputs byte-for-byte. On the challenge data a run takes ~2.5 min and
~1.6 GB with the default 4 registration workers (`--workers N`; ~0.5 GB each).

## Outputs

| File | Description |
|---|---|
| `rgb_image.tif` | uint8 (H, W, 3) display composite of the registered R, G, B bands; 0 = no-data in any channel (see §5) |
| `band_selection.json` | Per-band noise, contrast and SNR, the colour windows, and the chosen bands |
| `cloud_mask.tif` | uint8 (H, W) on the same grid as `rgb_image.tif`: 1 = cloud, 0 = not cloud; also 0 outside the shared footprint, where `rgb_image.tif` is 0 (§3) |
| `cloud_stats.json` | Bands used, the water reference, noise and thresholds found on the data, cloud and land ratio modes, cloud fraction, object count, no-data pixels |
| `registration_stats.json` | Per band: coarse shift, field range, tiles used per pass, spline smoothing, residual misregistration after warping, blocked cross-validation error of the field; and `rgb`: misregistration between each pair of RGB bands by an independent check (clear and cloudy windows), the self-consistency residual, and the fields' CV error (see §4) |
| `band_summary.json` | Per-band DN range, valid/saturated fractions, corrected mean/std, offset check, removed stripe amplitude |

## Method

### 1. Loading and metadata
The challenge input is a directory of 48 headerless uint8 files `b{n}.raw`
(n = 1…80, not contiguous; each exactly 2048 × 2048 = 4,194,304 bytes, row-major)
and `band_metadata.json` of the form
`{"<n>": {"wavelength": nm, "gain": …, "offset": …}}`.

- Each metadata key `"n"` is paired with `b{n}.raw`. This is the only layout
  accepted (`tools/make_synthetic.py` writes it too): a missing field, a missing
  band file, or a `b*.raw` file with no metadata entry is an error, rather than
  falling back to pairing by file order.
- Band number is not monotonic in wavelength (bands 24 and 25, at 544–547 nm,
  sit among 21–23), so the cube is ordered by ascending wavelength; band ids are
  kept in every output.
- Coverage is 470–729 nm with gaps at ~491–538, 559–602 and 685–712 nm. There
  is no NIR or SWIR, which shapes the cloud test (§6).
- Headerless files must be exactly 2048 × 2048 bytes. Any other size raises an
  error rather than guessing a header offset.

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
logged but **not** changed. Only band 1 (470 nm) is, at 72 % above band 2, and
this is the pedestal, not a wrong gain: the corrected level is mostly gain × the
~112 DN floor the offsets leave in (see Instrument artifacts, "DN floor"), and
band 1's gain is 1.85× band 2's (1.368 vs 0.739). With 100 DN removed from every
band before the comparison, band 1's residual falls to 0.23. Where the gain jumps
(here, and at the 628/632 nm boundary) the residual therefore compares gains as
much as spectra. The repair decision is not affected: the same five offsets are
replaced for any pedestal from 0 to 112 DN.

#### Column destriping (beyond the supplied calibration)
Every band has single-column stripes of ~2 DN, twice the random noise (~1 DN
from row-to-row differences), and the pattern is the same in all bands
(r ≈ 0.91 between 470 and 659 nm): a detector-column offset. The metadata has no
per-column terms, so removing it is an empirical correction on top of the
supplied gain/offset. It is on by default; `--no-destripe` gives the
supplied-calibration-only output.

For each band, every row is smoothed with a 15-column running median, which
follows the scene but not single-column stripes; a column's offset is the median
of its row residuals over valid, unsaturated pixels, and is subtracted from every
row. The estimate is refined with a second pass, because the stripes themselves
pull the running median. Offsets are zero-mean, so band levels are unchanged.
The removed amplitude is recorded as `stripe_amplitude` in `band_summary.json`.

`tools/destripe_diagnostics.py` evaluates this split-half: offsets estimated on
the even rows are removed from the odd rows, and the stripe amplitude is
re-measured there (corrected units; floor = the same measurement on white noise):

| Colour | Band | Before | After | Earlier method¹ | Noise floor |
|---|---|---|---|---|---|
| R | 43 | 0.58 | 0.063 | 0.14 | 0.012 |
| G | 27 | 1.15 | 0.11 | 0.30 | 0.024 |
| B | 7 | 0.92 | 0.10 | 0.22 | 0.022 |

¹ Column-median profile minus its running median, used before the review.

![Destriping, open water](outputs/diagnostics/destripe_crop.png)

What is left is banding a few to ~20 columns wide (visible in the crop). It is
not identifiable this way: a column pattern that varies more slowly than the
window looks the same as scene content that is constant along the track. The
same ambiguity applies to the cross-track ramp (see Instrument artifacts).

Effect downstream: the RGB band choice and the SNR table below are unchanged.
Registration improves slightly (R–B p95 0.20 → 0.18 px). The cloud mask compares
each pixel with the water in its own column (§6), so a column offset cancels
there with or without destriping.

### 3. Invalid pixels
- DN = 0 is treated as no-data (configurable with `--nodata`) and becomes NaN
  after correction. The challenge data contains none (the lowest DN is 103), so
  all no-data in the outputs comes from registration.
- DN = 255 is flagged as saturated (≤ 0.03 % of pixels, cloud tops). It stays in
  the image and is excluded from SNR statistics.
- Registration resamples each band; output pixels whose source lies outside the
  frame, or within the cubic-spline support of a no-data pixel, become no-data.
  With the green band (27) as reference this removes up to ~75 rows at the top
  for bands 45–80 and up to ~45 rows at the bottom for bands 1–8.
- Both outputs share one footprint: the pixels valid in every band either output
  uses (the RGB bands and the 729 nm band of the cloud mask). The 729 nm band is
  shifted most, so it sets the left and top borders. The footprint is 94.9 % of
  the frame. Outside it, `rgb_image.tif` is 0 in all three channels and
  `cloud_mask.tif` is 0, so a 0 in the mask is "not cloud" only where the RGB
  image is non-zero. Without a shared footprint the mask would read "clear" in a
  strip (cols 6–27, rows 8–40) where it cannot be computed.

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
| 1–8 | 470–491 | 1e-5 | 0.012–0.020 / ≤ 0.09 / ≤ 0.46 | 0.16–0.25 / 0.73–0.97 |
| 20–28 | 538–559 | 1e-3, 1e-2 | 0.032–0.051 / ≤ 0.17 / ≤ 0.63 | 0.09–0.14 / 0.35–0.49 |
| 40–44 | 602–628 | 1e-5 | 0.010–0.014 / ≤ 0.07 / ≤ 0.31 | 0.15–0.19 / 0.65–0.92 |
| 45–64 | 632–686 | 1e-5 | 0.010–0.020 / ≤ 0.16 / ≤ 1.68 | 0.15–0.21 / 0.93–1.31 |
| 75–80 | 712–729 | 1e-5 | 0.010 / ≤ 0.07 / ≤ 0.25 | 0.18–0.21 / 0.81–1.15 |

**Between the chosen RGB bands.** Before registration B was displaced from G by
10–37 px and R by 8–23 px, varying over the frame. Afterwards, the accuracy
estimate is `rgb.independent_check`: a method the fit did not use (Lucas–Kanade,
a gradient method, instead of phase correlation), on 64 px windows offset from
the fit's 128 px tiles. Only windows with scene structure are used (standard
deviation ≥ 4× the band noise): open water holds mostly noise and column banding,
which registration shifts by a few px between bands, so it would measure the
banding. Windows more than 20 % cloud are reported separately, because clouds
move between band acquisitions and sit above the ground the field was fitted to.
The check's own noise floor (green against itself with independent noise) is
0.013 px median, 0.034 px p95.

| Pair | Clear windows | Clear median / p95 / max (px) | Cloud windows | Cloud median / p95 (px) |
|---|---|---|---|---|
| R–G (43 vs 27) | 330 | 0.051 / 0.26 / 0.80 | 116 | 0.13 / 0.38 |
| B–G (7 vs 27) | 226 | 0.049 / 0.17 / 0.61 | 70 | 0.11 / 0.25 |
| R–B (43 vs 7) | 330 | 0.064 / 0.36 / 1.17 | 116 | 0.13 / 0.35 |

The other two measures bracket this. The self-consistency residual
(`rgb.self_consistency_residual`, phase correlation again) is 0.014–0.036 px
median, about 4× too optimistic. The CV error of the R and B fields
(`rgb.field_cv_error_vs_reference`) is 0.16 px median, 0.73–0.77 px p95,
pessimistic where tiles are dense. In short: on land, misregistration is ~0.05 px
median with a tail to ~0.3 px; on cloud, ~0.12 px. No colour fringes are visible
at full resolution.

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

Column stripes are invisible to this noise estimator: they are constant down a
column, and the operator's vertical second difference cancels them. Without
destriping they count as scene contrast instead, so destriping lowers SNR by
0–17 % per band (most in the noisier bands) but does not change the RGB choice.

`SNR = contrast / noise`. Gain cancels and offset drops out, so the ranking does
not depend on the doubtful calibration (the repaired offsets, the ~112 DN
pedestal). A mean-based SNR would mostly measure that pedestal.

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

**Higher-SNR bands outside the windows.** Two groups of bands have higher SNR
than band 43 but are deliberately not candidates:

| Bands | λ (nm) | SNR | Why excluded |
|---|---|---|---|
| 40–42 | 602–608 | 13.7–15.0 | Yellow/orange (570–620 nm). Only ~46 nm from the green band, so R and G would be strongly correlated and the composite would lose colour contrast. Red channels of comparable sensors sit at ~630–690 nm (Landsat 8 OLI 636–673, Sentinel-2 B4 650–680). |
| 75–80 | 712–729 | 12.7–13.7 | Beyond the visible (eye sensitivity ~1 % of peak at 720 nm) and on vegetation's red edge: a "red" channel here makes vegetation bright red, i.e. a false-colour composite. Band 80 is used by the cloud mask instead (§6). |

Their SNR is high for an instrument reason, not a scene one: in corrected
units the scene contrast is about the same in every band from 600 to 729 nm
(3.7–4.6), while the ~1 DN noise is only ~0.3 units in these low-gain
(≈ 0.3) bands against ~0.6 in bands 45–64. The 620 nm boundary is a convention;
the CIE red colour-matching function peaks near 600 nm, so a 602–608 nm red is
a defensible alternative for perceived colour, at the cost of colour separation.
`RGB_WINDOWS_NM` in `rgb.py` sets the windows.

**Composite**: the three registered bands, each stretched linearly from its
1st to 99th percentile onto 1–255; 0 marks pixels invalid in any channel.

The chosen bands span only 34–55 DN over that stretch, so the 8-bit input shows
as steps of ~4–7 display levels. In G (band 27, the registration reference and
therefore not resampled) only 63 distinct values occur. The noise (~8 levels in
G) dithers the steps, so they are not visible as contours. `--average-neighbours K`
makes each channel the inverse-variance weighted mean of its band and up to K
neighbours on each side inside the same colour window. With K = 1 the display
noise falls from 3.2/8.1/6.0 to 2.7/3.9/3.7 levels (R/G/B) and G uses all 255
values. The column stripes are unchanged, because they are shared between bands.
The default stays K = 0, the single highest-SNR band per colour that the task asks
for. The weights are recorded in `band_selection.json` under `channels`.

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
unavailable. The mask uses the registered green (27, 556 nm) and longest
(80, 729 nm) bands, after a 3×3 mean (single-pixel SNR is ~6), and three facts:

- water is dark at 729 nm, and the clouds lie over water;
- a cloud adds light to the water below it;
- clouds are spectrally flat in reflectance, so the light they add follows the
  solar spectrum, while every land surface here (vegetation's red edge, soil,
  roofs, sand) adds relatively more light at 729 nm.

1. **Water reference.** Water pixels are those dark at 729 nm (within 3σ of the
   dark cluster's mode; 56 % of the frame). The water level is the median over
   water pixels in each column, per 256-row block, interpolated over land
   columns and between blocks. Comparing within a column matters: across the
   swath the water level varies by ~3 units (the cross-track ramp and column
   banding, see Instrument artifacts), almost ten times the noise.
2. **Brightness.** The light a pixel adds over its column's water,
   `dg = g − g_water`, must exceed 5σ, where σ = 0.35 is the noise of `dg` over
   water, measured on its dark side (free of cloud and land). Noise alone
   passes with probability ~3·10⁻⁷. Threshold: 1.77 units, ~3 DN.
3. **Colour of the added light.** `r = dn / dg`, the 729 nm light added per unit
   of green light added. Radiances add, so a pixel that is a fraction *f* cloud
   adds *f* × (cloud − water) in both bands: `r` is the same for thin and thick
   cloud. Land adds relatively more 729 nm light, so its `r` is higher. Among
   strongly bright pixels (`dg` > 20σ, where `r` is precise) the histogram of
   `r` is bimodal, cloud 0.69 and land 1.06; the threshold is the minimum of the
   smoothed histogram between them (0.86), so it comes from the data. A pixel is
   cloud if it is bright and `r` < 0.86.
4. **Shape.** An opening with a 7 px disk removes features up to ~4 px wide
   (roof edges, piers); objects under 100 px are dropped; holes are filled; a
   2 px buffer covers mixed edge pixels and the small cloud parallax (clouds sit
   above the ground the bands were registered to).

Result: 9.3 % of the footprint, 137 objects, written with the thresholds to
`cloud_stats.json` (`figures/cloud_mask_overlay.png` from
`tools/make_figures.py`). The tests build a synthetic scene from the same
physics (banded water with a cross-track ramp, red-edge land, soft clouds
mixed linearly with water, and cloud-coloured lines and blobs) and check that
cloud down to 20 % cover is found, and that land, thin or tiny decoys and the
banded water are not.

### 7. Ship detection (optional)
Not implemented. A natural extension of the steps above: within the cloud-free,
valid water area (water = below the brightness threshold of §6), flag compact
objects that are bright relative to a local water background (e.g. > k σ of a
ring-shaped neighbourhood), keep those of plausible ship size and elongation, and
score confidence from contrast and shape. Wakes and the column banding over
water would be the main false-alarm sources.

## Instrument artifacts

The supplied calibration is one gain and one offset per band. Two corrections
in the pipeline go **beyond it**, both empirical and both switchable:
**column destriping** (`--no-destripe` turns it off) and the **repair of five
metadata offsets** (`--keep-metadata-offsets`). With both flags the outputs use
the supplied calibration only. Everything else below is documented but not
corrected.

| Artifact | Size | Corrected? |
|---|---|---|
| Single-column stripes | ~2 DN, 2× the noise | Yes, empirically (destriping) |
| Wider column banding | a few to ~20 columns | No |
| Right-edge fall-off | −1 to −9 DN, last ~250 columns | No |
| Smooth level change across the swath | −9 to +3 DN over water | No |
| Wrong metadata offsets (bands 42–44, 75–76) | +41 to +68 | Yes, empirically (replaced) |
| Band-to-band geometry (two detector segments) | up to ~50 px, 10 % stretch | Yes (co-registration) |
| 8-bit quantisation of a narrow DN range | 34–55 DN over the display stretch | Optional (`--average-neighbours`) |
| ~112 DN pedestal in every band | 1st percentile 108–126 DN | No (offset-invariant methods) |
| No-data rows and borders, saturation | ≤ 0.03 % saturated | Masked (§3) |

**Single-column stripes.** *Observed*: ~2 DN per-column offsets in every band,
the same pattern in all bands (r ≈ 0.91), constant down each column: a
detector-column offset. *Diagnosed* from column profiles and verified split-half
(even rows estimate, odd rows test). *Corrected* by subtracting per-column
offsets estimated from row residuals (§2), which lowers the stripe amplitude ~10×
(G 1.15 → 0.11, floor 0.024). The metadata has no per-column terms, so this is an
empirical correction on top of the supplied calibration. *Limits*: the column
pattern is estimated from the scene itself; a scene edge that stays in one column
for most of the frame would bias the columns next to it.

**Wider column banding.** *Observed*: after destriping, soft bands a few to ~20
columns wide remain, most visible over water (`diagnostics/destripe_crop.png`).
*Not corrected*: any per-column method has to separate the column pattern from
the scene with a window across the track, and a pattern that varies more slowly
than the window is indistinguishable from scene content that is constant along
the track. A flat field (or a uniform target) would be needed.

**Right-edge fall-off.** *Observed*: in the last ~250 columns every band drops,
by 8.4–9.0 corrected units at 470 nm, 3.2–3.8 in B (band 7), 4.2 in G (27),
1.5–1.8 in R (43) and 0.6–0.9 at 729 nm. *Diagnosed* with
`tools/ramp_diagnostics.py` (`right_edge_change_per_row_block` in
`diagnostics/ramp_stats.json`): the change from columns 1792–1855 to 1984–2039
is negative in every band in each of the four 256-row blocks of the city (rows
0–1023), whose land cover differs, and agrees between blocks within 0–17 % for
the RGB bands (up to 40 % for the bands with the smallest drop), so it does not
appear to follow the land cover. Rows with bay or salt ponds at the edge do
follow the scene. In raw DN the drop is 5–9 DN in B, G and R (1–5 DN at
729 nm); as a gain loss that would mean 20–50 % vignetting on signal only
~15–30 DN above the floor, so an **additive** effect is more likely. Its
blue-heavy spectrum fits either a dark-level fall-off at the detector edge or
less atmospheric path radiance towards that edge; the data cannot tell them
apart. *Effect*: G's display stretch spans only 34 DN against 55 for R, so the
same few DN darken G by ~37 display levels and R by ~14, which is the magenta
right edge of `rgb_image.tif`. *Not corrected*: an empirical fit would be made on
the city itself, with the cause unknown; dark frames or the viewing geometry
would be needed.

**Smooth level change across the swath.** *Observed*: over open water (dark at
556 and 729 nm, away from clouds; `diagnostics/water_profiles.png`) the level
changes smoothly across the swath. Relative to columns 1000–1500, the outermost
water columns (27–47) are lower in all bands but 548–551 nm (within 0.4 DN),
by up to 9 DN (median 3.7;
B 6.7, G 3.0, R 6.1 DN), and columns 650–750 are 0.1–3.4 DN higher. The plot
shows this as only −7 to +3 % because it divides by the full level, which is
mostly the ~112 DN pedestal (see "DN floor"): water sits just 2–23 DN (median 5)
above it, so the change is comparable to the water signal itself. The bands do
not share one profile: over columns 100–250 they differ by up to 8 DN, with the
blue and 602–628 nm bands lowest, so colour changes as well as brightness.
*Diagnosed* on water because it is a single, nearly uniform surface. The
review's suggested fit of each band against the mean of all bands, column by
column (`diagnostics/ramp_profiles.png`), turned out to follow the scene instead:
its profiles jump where the columns change from ocean and cumulus to land, and
the fitted change depends on the brightness at which it is evaluated.
*Interpretation*: as a gain change it would be tens of per cent of the signal, so,
as at the right edge, an additive change is more likely: a real difference
between open-ocean and bay water, varying path radiance, or a dark level that
varies across the detector. The data cannot separate them. *Not corrected*.
Past column ~1550 the shallow south bay has a different spectrum.

**Metadata offsets.** Bands 42–44 and 75–76 carry offsets of +41 to +68 while
all others lie in −7…+3, which would put a 2.5× step into the spectrum. Their raw
DNs are continuous with the neighbours (r = 0.96–0.99), so the offsets are
replaced by the median of the normal ones (§2). This is the second correction
beyond the supplied calibration. Band 1's corrected level is 72 % above band 2's,
which first looked like a wrong gain; it is explained by the pedestal (§2), and
band 1 is used as given.

**Band-to-band geometry.** Each band is a separate strip of detector rows, in
two segments (break between bands 44 and 45). Bands are displaced from each
other by up to ~50 px, non-uniformly, with an along-track stretch of up to 10 %
in the last ~450 rows for the other segment (§4). Corrected by co-registration
to ~0.05 px median between the RGB bands on land and ~0.12 px on cloud, by an
independent check (§4).

**Quantisation.** The RGB bands span only 34–55 DN over the 1–99 % display
stretch, so the display shows steps of 4–7 levels; G, the unresampled reference,
has 63 distinct values. The noise (~1 DN) dithers the steps, so they do not
show as contours. `--average-neighbours 1` removes the steps and halves the G
display noise (§5); off by default.

**DN floor.** *Observed*: every band sits on a floor that the metadata offsets
(−7…+3) do not remove; the 1st percentile of raw DN is 108–126 (median 111) and
no pixel is below 103. *Diagnosed* as a fixed electronic offset (a pedestal),
not light: a fixed radiance would read ∝ 1/gain, about 5× higher in the
low-gain bands, but a fit over the 48 bands gives 112 + 0.7/gain DN. Consistent
with that, the noise near the floor is ~0.9 DN and grows with the signal above
it (band 40: 0.9 DN at 116 DN, 1.6 DN at 204 DN), as shot noise should, while a
constant adds none. Bias and dark current cannot be told apart without dark
frames. *Effect*: corrected values include about gain × 112, so they are not
absolute radiance, and anything that divides by a level or compares levels
between bands of different gain is biased. The pipeline's steps are
offset-invariant (destriping, SNR, registration with tile means removed, RGB
stretch, cloud mask on light added over the water); the exception is the
continuity residual of the offset check
(§2), whose repair decision is unaffected. *Not corrected*: the pedestal is not
known per band better than ~±4 DN, which is comparable to the water signal.

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
- Noise is white and spatially uncorrelated at the 3×3 scale (Immerkær's
  estimator, §5); the conventional colour windows are an adequate definition of
  "red", "green" and "blue" for a display composite.
- Clouds are spectrally flat in reflectance over 556–729 nm, and the surfaces
  confused with them (land, sand, roofs) are relatively brighter at 729 nm.
  Thresholds are found on each scene, but the method assumes water in most
  columns (columns without water borrow the level of their neighbours) and a
  scene with both cloud and land, so that the ratio histogram has two modes
  (with one mode it falls back to Otsu's threshold and logs a warning).

## Limitations
- Repaired offsets are a consistent replacement (the median of the normal
  offsets), not a recalibration; absolute levels in those bands carry an
  uncertainty of a few units.
- The ~112 DN pedestal is not removed, so corrected values are relative, not
  absolute radiance; band ratios or level comparisons across a gain jump would be
  biased (none of the outputs depend on them).
- Destriping is an empirical correction beyond the supplied calibration and
  removes single-column stripes only; banding a few to ~20 columns wide remains
  (see Instrument artifacts).
- Every band is registered directly to one reference. Where a band's contrast is
  inverted relative to the reference (e.g. land/sea across the vegetation red
  edge), few tiles correlate: on the synthetic 400–1000 nm data, bands > 500 nm
  from the reference fall back to the coarse shift. This is reported
  (`field_from_tiles: false`, large residual). The challenge data (470–729 nm)
  is not affected.
- The ground-truth check of registration is on synthetic data only. On the real
  data the independent check measures only where there is scene structure (land
  and cloud), not over open water, where the field is interpolated.
- The per-channel stretch balances the scene average to grey; because the land
  is redder than the clouds, clouds look cyan. A white balance on cloud pixels
  neutralises them but tints the surface (§5); natural colour needs atmospheric
  correction. The right-edge fall-off (magenta right edge) and the wider column
  banding (coloured streaks over water) are not corrected (see Instrument
  artifacts).
- By default each colour uses one band. `--average-neighbours` averages in-window
  neighbours, which lowers the noise but not the column banding, at a small
  cost in spectral purity.
- Cloud mask: surf along the Silver Strand and a few bright roofs are false
  alarms (whitewater is as bright and as spectrally flat as cloud; about ten
  small objects); the faintest haze at cloud margins is missed (below the 5σ
  brightness test). Cloud over land is not tested: the reference is water, so
  thin cloud over land would add land-coloured light and be rejected. In this
  scene the clouds are over the ocean.
- No cloud-shadow mask.
- No ship detection (§7).
- Only radiance-like corrected values are produced: no conversion to
  reflectance (no solar geometry or atmospheric data) and no georeferencing (the
  outputs are on the reference band's pixel grid).
- Memory: the whole cube is held in float32 (~0.8 GB) plus the registered copy;
  a larger capture would need band-by-band streaming.

## Project layout
```
src/hsi_pipeline/   io.py (loading), radiometry.py (correction, validity, destriping),
                    calibration.py (offset consistency check),
                    registration.py (co-registration and its evaluation),
                    rgb.py (SNR, band choice, composite), cloud.py (cloud mask),
                    cli.py (entry point)
tools/              make_figures.py (summary figures, needs the viz extra)
                    destripe_diagnostics.py (destriping before/after, viz extra)
                    ramp_diagnostics.py (cross-track ramp diagnosis, viz extra)
                    make_synthetic.py (test data with known per-band shifts)
tests/              unit tests
```
