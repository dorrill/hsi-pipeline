# Instrument notes

Detail behind the README: what the capture's artifacts are, how each was
diagnosed, and why each is or is not corrected. Figures and statistics referred
to here are written to `outputs/` by the pipeline and the scripts in `tools/`.

The supplied calibration is one gain and one offset per band. Two corrections
in the pipeline go **beyond it**, both empirical and both switchable: **column
destriping** (`--no-destripe`) and the **repair of five metadata offsets**
(`--keep-metadata-offsets`). With both flags the outputs use the supplied
calibration only.

| Artifact | Size | Corrected? |
|---|---|---|
| Single-column stripes | ~2 DN, 2× the noise | Yes, empirically (destriping) |
| Wider column banding | a few to ~20 columns | No; made coloured by registration |
| Right-edge fall-off | −1 to −9 DN, last ~250 columns | No |
| Smooth level change across the swath | −9 to +3 DN over water | No (the cloud mask works around it) |
| Wrong metadata offsets (bands 42–44, 75–76) | +41 to +68 | Yes, empirically (replaced) |
| Band-to-band geometry (two detector segments) | up to ~50 px, 10 % stretch | Yes (co-registration) |
| 8-bit quantisation of a narrow DN range | 34–55 DN over the display stretch | Optional (`--average-neighbours`) |
| ~112 DN pedestal in every band | 1st percentile 108–126 DN | No (offset-invariant methods) |

## Single-column stripes

*Observed*: ~2 DN per-column offsets in every band, twice the random noise
(~1 DN), with the same pattern in all bands (r ≈ 0.91 between 470 and 659 nm),
constant down each column: a detector-column offset.

*Corrected* (`radiometry.destripe_columns`): every row is smoothed with a
15-column running median, which follows the scene but not single-column
stripes; a column's offset is the median of its row residuals over valid,
unsaturated pixels, and is subtracted from every row. A second pass refines the
estimate, because the stripes pull the running median. Offsets are zero-mean,
so band levels are unchanged.

*Evaluated* split-half (`tools/destripe_diagnostics.py`): offsets estimated on
the even rows are removed from the odd rows, and the stripe amplitude is
re-measured there (corrected units; floor = the same measurement on white noise):

| Colour | Band | Before | After | Noise floor |
|---|---|---|---|---|
| R | 43 | 0.58 | 0.063 | 0.012 |
| G | 27 | 1.15 | 0.11 | 0.024 |
| B | 7 | 0.92 | 0.10 | 0.022 |

*Limits*: the pattern is estimated from the scene; a scene edge that stays in
one column for most of the frame would bias the columns next to it.

## Wider column banding

*Observed*: after destriping, soft bands a few to ~20 columns wide remain, most
visible over water (`outputs/diagnostics/destripe_crop.png`).

*Made coloured by registration*: the banding is largely shared between bands,
so in the unregistered bands it is nearly grey. Registration shifts R and B by
several px across the track relative to G (over open water, R by about −4 px
and B by about +6 px), and the banding moves with them. Column profiles over
open water (rows 1250–1500, cols 1050–1500) show it: the correlation of the R
and B banding with G at zero lag falls from 0.69 and 0.64 before registration
to 0.53 and 0.44 after, and peaks again at the shifts (+4 and −6 px). The
green/magenta streaks over water in `rgb_image.tif` are this.

*Not corrected*: any per-column method has to separate the column pattern from
the scene with a window across the track, and a pattern that varies more slowly
than the window cannot be told from scene content that is constant along the
track. A flat field would be needed. With one, the pattern could be removed in
detector geometry, before registration, which would also stop it turning
coloured.

## Right-edge fall-off

*Observed*: in the last ~250 columns every band drops, by 8.4–9.0 corrected
units at 470 nm, 3.2–3.8 in B (band 7), 4.2 in G (27), 1.5–1.8 in R (43) and
0.6–0.9 at 729 nm.

*Diagnosed* with `tools/ramp_diagnostics.py`: the change from columns 1792–1855
to 1984–2039 is negative in every band in each of the four 256-row blocks of the
city (rows 0–1023), whose land cover differs, and agrees between blocks within
0–17 % for the RGB bands, so it does not follow the land cover. In raw DN the
drop is 5–9 DN in B, G and R; as a gain loss that would mean 20–50 % vignetting
on signal only ~15–30 DN above the floor, so an **additive** effect is more
likely: a dark-level fall-off at the detector edge, or less path radiance
towards that edge. The data cannot tell them apart.

*Effect*: G's display stretch spans only 34 DN against 55 for R, so the same few
DN darken G by ~37 display levels and R by ~14: the magenta right edge of
`rgb_image.tif`. *Not corrected*: an empirical fit would be made on the city
itself, with the cause unknown.

## Smooth level change across the swath

*Observed*: over open water the level changes smoothly across the swath.
Relative to columns 1000–1500, the outermost water columns (27–47) are lower in
nearly all bands by up to 9 DN (median 3.7; B 6.7, G 3.0, R 6.1 DN), and
columns 650–750 are 0.1–3.4 DN higher. Water sits only 2–23 DN above the
pedestal, so the change is comparable to the water signal itself. The bands do
not share one profile (they differ by up to 8 DN over columns 100–250), so colour
changes as well as brightness.

*Interpretation*: as a gain change it would be tens of per cent of the signal,
so an additive change is more likely: a real difference between open-ocean and
bay water, varying path radiance, or a dark level that varies across the
detector. *Not corrected*, but the cloud mask compares each pixel with the water
in its own column, so it does not depend on it. A fit of each band against the
mean of all bands, column by column, was tried as a diagnostic and follows the
scene instead (its profiles jump where columns change from ocean to land).

## Metadata offsets

Five bands carry offsets far outside the rest (all others lie in −7…+3):

| Band | λ (nm) | Metadata offset | Mean with it | Neighbour mean |
|---|---|---|---|---|
| 42, 43, 44 | 608–628 | +64.5, +65.5, +68.4 | 105–112 | ~40 (band 41) |
| 75, 76 | 712, 715 | +57.5, +41.2 | 98, 82 | ~39 (band 77) |

Applied as given, they put a ~2.5× step into the spectrum over 2–3 nm, which no
real scene produces, while the raw DNs are continuous with the neighbouring
bands (r = 0.96–0.99). Neither gain/offset convention fixes them.

`calibration.py` checks every offset in two stages and repairs it only when
both agree: (1) a robust z-score of the offset above 5 (normal bands reach
|z| ≤ 1.7, these five 14–22); (2) the band's scene-median level fits a
wavelength interpolation of the trusted bands better with the median normal
offset (−2.97) than with its own. Each repair is logged and recorded in
`band_summary.json`.

**Effect on the outputs: none.** The SNR, the display stretch, registration and
the cloud mask are all offset-invariant, and the cloud mask uses bands 27 and 80,
which are not repaired. The repair makes the corrected spectrum consistent; it
is reported so that the metadata problem is visible.

Every band also gets a leave-one-out `continuity_residual` (logged, never
acted on). Band 1 is at +72 %, explained by the pedestal: its corrected level is
mostly gain × ~112 DN, and its gain is 1.85× band 2's. With 100 DN removed from
every band its residual falls to 0.23. Band 44's scene contrast is also low
(2.95 against 4.2–4.4 for bands 42 and 45), which suggests the gains of bands
42–44 may be off as well; this is not corrected either, and the display stretch
cancels it.

## Band-to-band geometry

Tile-by-tile phase correlation against the reference shows the misregistration
of a push-broom sensor whose bands are separate strips of detector rows:

- Shifts reach ~50 px and grow linearly with the band **number**, not the
  wavelength. Between bands 44 and 45 there is an extra jump of ~(19, 11) px: a
  gap between two detector segments (the gain break is at the same place).
- The shift is not a translation: dy varies linearly across the swath (~11 px
  over 2048 columns for band 1) and non-monotonically along the rows (time).
- In the last ~450 rows, bands of the other segment are stretched along-track by
  up to ~10 % relative to the reference.

A translation or affine model leaves 0.3–1.3 px, and a global cubic fails near
the bottom edge, hence the non-parametric field. Per band, relative to band 27
(`registration_stats.json`):

| Bands | λ (nm) | Smoothing | Residual median / p95 (px) | CV median / p95 (px) |
|---|---|---|---|---|
| 1–8 | 470–491 | 1e-5 | 0.012–0.020 / ≤ 0.09 | 0.16–0.25 / 0.73–0.97 |
| 20–28 | 538–559 | 1e-3, 1e-2 | 0.032–0.051 / ≤ 0.17 | 0.09–0.14 / 0.35–0.49 |
| 40–44 | 602–628 | 1e-5 | 0.010–0.014 / ≤ 0.07 | 0.15–0.19 / 0.65–0.92 |
| 45–64 | 632–686 | 1e-5 | 0.010–0.020 / ≤ 0.16 | 0.15–0.21 / 0.93–1.31 |
| 75–80 | 712–729 | 1e-5 | 0.010 / ≤ 0.07 | 0.18–0.21 / 0.81–1.15 |

The residual is a self-consistency check (optimistic); the CV error includes
prediction into open-water gaps (pessimistic). For the RGB bands the README
gives an independent check, which falls between them.

## SNR outside the colour windows

Two groups of bands have higher SNR than the chosen red band but are not
candidates:

| Bands | λ (nm) | SNR | Why excluded |
|---|---|---|---|
| 40–42 | 602–608 | 13.7–15.0 | Orange, only ~46 nm from the green band, so R and G would be strongly correlated. Comparable sensors put red at ~630–690 nm (Landsat 8 OLI 636–673, Sentinel-2 B4 650–680). |
| 75–80 | 712–729 | 12.7–13.7 | Beyond the visible and on vegetation's red edge: a "red" channel here gives a false-colour composite. Band 80 is used by the cloud mask instead. |

Their SNR is high for an instrument reason: the scene contrast in corrected
units is similar from 600 to 729 nm, while the ~1 DN noise is only ~0.3 units in
these low-gain (≈ 0.3) bands against ~0.6 in bands 45–64.

## Quantisation and neighbour averaging

The RGB bands span only 34–55 DN over the 1–99 % display stretch, so the display
shows steps of 4–7 levels; G, the unresampled reference, has 63 distinct values.
The noise dithers the steps, so they do not show as contours.
`--average-neighbours K` makes each channel the inverse-variance weighted mean of
its band and up to K neighbours on each side inside the colour window. With
K = 1 the display noise falls from 3.2/8.1/6.0 to 2.7/3.9/3.7 levels (R/G/B). The
default is K = 0, the single highest-SNR band per colour that the task asks for.

## Colour balance

The per-channel stretch balances the scene average to grey. The land is redder
than the clouds, so the clouds come out slightly cyan. A white balance on the
cloud pixels (clouds are spectrally flat) makes them neutral but turns the land
magenta and the water purple, and clips 8 % of the red channel: clouds sit above
most of the hazy atmosphere while the surface signal passes through all of it.
Natural colour would need atmospheric correction.

## DN floor (pedestal)

Every band sits on a floor that the metadata offsets do not remove: the 1st
percentile of raw DN is 108–126 (median 111) and no pixel is below 103. It is a
fixed electronic offset, not light: a fixed radiance would read ∝ 1/gain, about
5× higher in the low-gain bands, but a fit over the 48 bands gives
112 + 0.7/gain DN, and the noise near the floor is ~0.9 DN and grows with the
signal above it, as shot noise should. Corrected values therefore include about
gain × 112, so they are relative, not absolute radiance. Every step of the
pipeline is offset-invariant, so none of the outputs depend on it.
