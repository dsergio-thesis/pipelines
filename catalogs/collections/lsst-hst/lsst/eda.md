1. Spatial Coverage

Use:

coord_ra
coord_dec

Plots:

sky scatter plot
density heatmap

Purpose:

show footprint / sky coverage
identify survey geometry and missing regions
2. Photometric Brightness Distributions

Choose ONLY magnitudes, not raw fluxes.

Use:

u_psfFlux_mag
g_psfFlux_mag
r_psfFlux_mag
i_psfFlux_mag
z_psfFlux_mag
y_psfFlux_mag

Plots:

histogram per band
maybe overlaid KDEs

This is much more interpretable than raw flux.

Why:

demonstrates dynamic range
shows limiting magnitude
identifies outliers / saturation
3. Signal-to-Noise Quality

Use:

*_psfFlux_SNR_log

Probably:

g_psfFlux_SNR_log
r_psfFlux_SNR_log
i_psfFlux_SNR_log

Not all six bands.

Purpose:

data quality assessment
motivates filtering thresholds
4. Color Indices (Most Important)

This is the scientifically meaningful part.

Strong subset:

color_ug
color_gr
color_ri
color_iz

Plots:

histograms
pairplots
color-color diagrams

Especially:

g-r vs r-i
u-g vs g-r

These are canonical astronomical diagnostics.

They separate:

stars
galaxies
quasars
red sequence objects
blue cloud galaxies

This section matters far more than raw flux histograms.

Very Important: Color-Color Diagram

This is probably the single highest-value EDA figure.

Example:

x-axis:

g - r

y-axis:

r - i

You will see clustering structure immediately.

That looks scientifically meaningful in a thesis.

5. Morphology / Object Type

Use:

refExtendedness

Plot:

histogram or bar chart

Purpose:

distinguish point sources vs extended sources
stars vs galaxies

You can also overlay:

g-r color by refExtendedness

Very strong figure scientifically.

6. Flags / Data Quality

Do NOT show all flags.

Instead summarize:

fraction flagged
bad measurements per band

Use:

*_psfFlux_bad_flag

Maybe one table:

Band	Bad Fraction
u	0.04
g	0.02

This is much cleaner.

Columns You Should Probably Exclude

Avoid cluttering appendix with:

Administrative fields
tract
patch

Mention briefly only.

Raw flux + arcsinh + mag together

Too redundant.

Choose:

magnitudes
colors
SNR

not all representations.

Every pairwise color

You generated many:

color_ug
color_ur
color_ui
...

This is too many.

Keep only adjacent-band colors:

u-g
g-r
r-i
i-z
z-y

These are standard in astronomy.

Curvature features

Unless your ML model directly relies on them.

These belong more in:

feature engineering section
ML preprocessing section

not generic EDA.

Ideal Final Appendix Subset

Honestly, this is enough:

Spatial
RA
Dec
Magnitudes
g_mag
r_mag
i_mag
SNR
g/r/i SNR
Colors
u-g
g-r
r-i
i-z
Morphology
refExtendedness
Quality
bad flag summary

That already looks substantial and scientifically motivated.
