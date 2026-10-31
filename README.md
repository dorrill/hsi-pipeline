# Hyperspectral Image Processing Pipeline

Turns a 48-band, 2048 × 2048, 8-bit hyperspectral capture into a radiometrically
corrected, co-registered RGB image and a binary cloud mask, and reports how well
the RGB bands are co-registered.

## Results on the challenge capture

| | |
|---|---|
| RGB bands (highest SNR per colour window) | R = band 43 (625.5 nm), G = 27 (556 nm), B = 7 (486 nm) |
| Misregistration before | B vs G 10–37 px, R vs G 8–23 px, varying over the frame |
| Misregistration after (independent check) | ~0.05 px median on land (p95 0.17–0.36 px), ~0.12 px on cloud |
| Cloud | 9.3 % of the valid area, 137 objects |
| Valid area | 94.9 % of the frame (no-data only at the borders, from registration) |

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the exact versions the outputs were made with
pip install -e .                  # add ".[dev]" for tests, ".[viz]" for plots
hsi-pipeline --input-dir inputs --output-dir outputs
pytest                            # needs the dev extra
```

`inputs/` holds the `b{n}.raw` files and `band_metadata.json`. A run takes
~2.5 min and ~1.9 GB peak with the default 4 registration workers (`--workers`).
It is deterministic: there is no random state in the pipeline, and a fresh clone
installed with `requirements.txt` reproduces all six outputs byte for byte.

Without the real data, `python tools/make_synthetic.py --out synthetic_inputs`
writes a synthetic capture with known per-band shifts. After a run,
`python tools/make_figures.py --input-dir inputs --output-dir outputs` writes
summary figures to `outputs/figures/` (needs the `viz` extra).

## Outputs

| File | Contents |
|---|---|
| `rgb_image.tif` | uint8 (H, W, 3) composite of the registered R, G, B bands; 0 in all channels = no-data |
| `cloud_mask.tif` | uint8 (H, W) on the same grid: 1 = cloud, 0 = not cloud (also 0 where `rgb_image.tif` is no-data) |
| `registration_stats.json` | Per band: shifts, field, residual and cross-validation error. `rgb`: the accuracy between the RGB bands (below) |
| `band_selection.json` | Per-band noise, contrast and SNR; the colour windows; the chosen bands |
| `cloud_stats.json` | Water reference, noise, thresholds found on the data, cloud fraction, object count |
| `band_summary.json` | Per-band DN range, valid/saturated fractions, offset check, removed stripe amplitude |

## Method

### 1. Loading
Each metadata key `"n"` in `band_metadata.json` is paired with the headerless file
`b{n}.raw` (exactly 2048 × 2048 bytes, row-major uint8). A missing field, a missing
file, an unpaired file or a wrong file size is an error, not a guess. Band
numbers are not in wavelength order (bands 24–25 sit among 21–23), so the cube is
sorted by wavelength; band ids are kept in every output. Coverage is 470–729 nm,
with no NIR or SWIR.

### 2. Radiometric correction
`L = gain · DN + offset`, per band in float32, **before** any band selection or
combination (`--convention subtract_offset` gives `gain · (DN − offset)`; the data
cannot tell the two apart). Two empirical corrections go beyond the supplied
calibration; both can be switched off:

- **Column destriping** (`--no-destripe`). Every band has ~2 DN single-column
  stripes, the same in all bands: a detector-column offset. Each column's offset
  is the median, down the column, of the difference between the row and its
  15-column running median, and is subtracted. Split-half validation lowers the
  stripe amplitude ~10× (details in [docs/instrument_notes.md](docs/instrument_notes.md)).
- **Offset repair** (`--keep-metadata-offsets`). Bands 42–44 and 75–76 have
  offsets of +41 to +68 while all others lie in −7…+3, which puts a 2.5× step into
  the spectrum although their raw DNs are continuous with their neighbours. An
  offset is replaced (by the median normal offset) only if it is a statistical
  outlier *and* the replacement fits the spectral neighbours better. Every later
  step is offset-invariant, so the repair changes none of the outputs; it makes
  the corrected spectrum consistent and flags the metadata problem.

### 3. Invalid pixels
- DN = 0 is no-data (`--nodata`) and becomes NaN. The challenge data has none
  (the lowest DN is 103), so all no-data in the outputs comes from registration.
- DN = 255 is flagged as saturated (≤ 0.03 % of pixels, cloud tops), kept in the
  image and excluded from the SNR statistics.
- Registration resamples each band; pixels whose source lies outside the frame,
  or within the cubic-spline support of a no-data pixel, become no-data.
- Both outputs share one footprint: pixels valid in every band either output uses.
  The cloud mask's 729 nm band is shifted most and sets the left and top borders.
  Outside the footprint (5.1 % of the frame) `rgb_image.tif` is 0 in all channels
  and `cloud_mask.tif` is 0, so a 0 in the mask means "not cloud" wherever the RGB
  image has data.

### 4. Co-registration (`registration.py`)
Each band is a separate strip of detector rows, so bands see the ground at
different times: shifts reach ~50 px, vary across the frame, and in the last ~450
rows one detector segment is stretched along-track by up to 10 %. A translation
or affine model leaves 0.3–1.3 px, so each band gets a smooth, non-parametric
displacement field relative to the green band (which is therefore never
resampled):

1. **Tile shifts** by phase correlation on 128 px tiles every 64 px, refined to
   0.01 px (Guizar-Sicairos et al. 2008). It uses phase only, so the bands'
   different gain and contrast do not matter. Tiles with a weak correlation peak
   (open water) are dropped.
2. **Field**: a smoothing thin-plate spline through the tiles, rejecting
   outliers (moving cloud edges), bridging the water gaps with minimum bending.
3. **Iteration**: warp, re-measure, and add the remainder, 2–3 passes.
4. **Smoothing** chosen per band by spatially blocked cross-validation.

**Accuracy between the RGB bands** (`rgb.independent_check`). The headline numbers
come from a method the fit did not use: Lucas–Kanade, a gradient method, on 64 px
windows offset from the fit's tiles. Only windows with scene structure count
(standard deviation ≥ 4× the band noise); open water holds mostly noise and column
banding, which would be measured instead. Windows over 20 % cloud are reported
separately, because clouds move between band acquisitions. The check's own noise
floor is 0.013 px median.

| Pair | Clear windows | Median / p95 (px) | Cloud windows | Median / p95 (px) |
|---|---|---|---|---|
| R–G | 330 | 0.051 / 0.26 | 116 | 0.13 / 0.38 |
| B–G | 226 | 0.049 / 0.17 | 70 | 0.11 / 0.25 |
| R–B | 330 | 0.064 / 0.36 | 116 | 0.13 / 0.35 |

Two other measures bracket this. Re-measuring with the fit's own phase
correlation (`rgb.self_consistency_residual`) gives 0.014–0.036 px median, ~4×
too optimistic, because a flexible field absorbs the tiles' own errors. The
fields' blocked cross-validation error (`rgb.field_cv_error_vs_reference`,
0.16 px median, 0.73–0.77 px p95) is pessimistic, because it predicts into
held-out blocks and open-water gaps. No colour fringes are visible at full
resolution. On synthetic data with known shifts (`tests/test_registration.py`),
bands 40–120 nm from the reference are recovered to 0.08–0.12 px median.

### 5. RGB band selection (`rgb.py`)
**SNR = scene contrast / noise**, per band, on the corrected, destriped bands
*before* registration (resampling smooths noise and would flatter every band but
the reference). Noise is Immerkær's estimator (a 3×3 operator that cancels locally
linear signal) over the flattest half of the pixels; contrast is the band's
standard deviation with the noise removed in quadrature. Both scale with the gain
and ignore the offset, so the ranking does not depend on the doubtful calibration;
a mean-based SNR would mostly measure the ~112 DN pedestal every band sits on.

The chosen band is the highest-SNR band in each conventional colour window (blue
450–495, green 495–570, red 620–700 nm):

| Colour | Band | λ (nm) | SNR | Runner-up |
|---|---|---|---|---|
| R | 43 | 625.5 | 11.8 | 63 (683.5 nm), 9.9 |
| G | 27 | 556.0 | 6.6 | 28 (559 nm), 6.5 |
| B | 7 | 486.0 | 6.4 | 8 (491 nm), 6.4 |

Noise is ~1 DN in every band, so SNR follows how many DN the scene spans, which
favours the low-gain bands; that puts R on band 43. For G and B the winner leads
its neighbour by under 2 %, within the estimator's uncertainty: bands 27/28 and
7/8 are equivalent choices. Bands 40–42 (602–608 nm, orange) and 75–80
(712–729 nm, beyond the visible) have higher SNR but lie outside the windows.

**Composite**: each registered band is stretched linearly from its 1st to 99th
percentile onto 1–255 (0 = no-data). This balances the scene average to grey, so
the clouds, which are less red than the land, look slightly cyan.
`--average-neighbours K` averages each band with K in-window neighbours to lower
the noise (off by default).

### 6. Cloud mask (`cloud.py`)
Without NIR or SWIR, the mask uses the green (556 nm) and longest (729 nm) bands,
after a 3×3 mean, and three facts: water is dark at 729 nm; a cloud adds light to
the water below it; and that light is spectrally flat, while every land surface
here (vegetation, soil, roofs, sand) adds relatively more 729 nm light.

1. **Water reference.** Water pixels are those dark at 729 nm (56 % of the frame).
   The water level is their median in each column, per 256-row block,
   interpolated over land columns. This matters: across the swath the water level
   varies by ~3 units (a smooth ramp and column banding), almost ten times the
   noise.
2. **Brightness.** The light a pixel adds over its column's water,
   `dg = g − g_water`, must exceed 5σ, with σ = 0.35 the noise of `dg` over water.
   Threshold: 1.77 units (~3 DN).
3. **Colour of the added light.** `r = dn / dg`, the 729 nm light added per unit
   of green light added. A pixel that is a fraction *f* cloud adds
   *f* × (cloud − water) in both bands, so `r` is the same for thin and thick
   cloud. Among strongly bright pixels the histogram of `r` has two modes, cloud
   0.69 and land 1.06; the threshold is the valley between them, 0.86, found on
   the data. Cloud = bright and `r` < 0.86.
4. **Shape.** An opening removes features up to ~4 px wide (roof edges, piers),
   objects under 100 px are dropped, holes are filled, and a 2 px buffer covers
   mixed edge pixels and the clouds' small parallax.

The tests build a synthetic scene from the same physics (banded water with a
cross-track ramp, land, soft clouds mixed linearly with water, cloud-coloured
lines and blobs) and check that cloud down to 20 % cover is found and that land,
small decoys and the banded water are not.

Ship detection (optional) is not implemented.

## Assumptions
- The gain/offset convention is linear, and DN = 0 is fill.
- A band whose corrected level breaks spectral continuity while its raw DNs do
  not has a wrong metadata offset, not a real spectral feature.
- Column stripes are an additive per-column offset, constant along the rows.
- The band-to-band displacement is smooth over the frame and the ground is
  static between band acquisitions (moving clouds keep a residual, reported).
- Noise is white at the 3×3 scale; the conventional colour windows define "red",
  "green" and "blue".
- Clouds are spectrally flat over 556–729 nm and lie over water; the scene has
  water in most columns and both cloud and land (with only one of them, the
  ratio histogram has one mode and the threshold falls back to Otsu's, with a
  warning).

## Limitations
- **Column banding** a few to ~20 columns wide remains after destriping; it
  cannot be separated from the scene without a flat field. Registration shifts it
  by a few px between bands, which turns it into green/magenta streaks over water.
- **Right edge**: the last ~250 columns are darker in every band, most in blue
  and green, so the edge looks magenta. The cause (dark level or path radiance)
  cannot be determined from the data, so it is not corrected.
- **Cloud mask**: surf along the Silver Strand and a few bright roofs are flagged
  (about ten small objects; whitewater is as bright and as flat-spectrum as
  cloud); the faintest haze at cloud edges is missed. Cloud over land is not
  handled: the reference is water. No cloud-shadow mask.
- **Registration accuracy** is checked against ground truth only on synthetic
  data; on the real data the independent check measures land and cloud, not open
  water, where the field is interpolated.
- **Radiometry**: the ~112 DN pedestal is not removed and the repaired offsets
  are a consistent replacement, not a recalibration, so corrected values are
  relative radiance. No reflectance conversion and no georeferencing.
- **Memory**: the whole cube is held in float32 (~0.8 GB) plus the registered
  copy; a larger capture would need band-by-band streaming.

The instrument artifacts behind these (stripes, banding, edge fall-off, swath
ramp, offsets, pedestal, detector geometry) are analysed in
[docs/instrument_notes.md](docs/instrument_notes.md).

## Project layout
```
src/hsi_pipeline/   io.py (loading), radiometry.py (correction, validity, destriping),
                    calibration.py (offset check), registration.py (co-registration
                    and its evaluation), rgb.py (SNR, band choice, composite),
                    cloud.py (cloud mask), cli.py (entry point)
tools/              make_figures.py, destripe_diagnostics.py, ramp_diagnostics.py
                    (figures and diagnostics, viz extra); make_synthetic.py
tests/              unit tests, including synthetic ground truth
docs/               instrument_notes.md
```
