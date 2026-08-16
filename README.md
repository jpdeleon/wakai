# `wakai`
`wakai` is the Japanese word for `young.` This repo aims to consolidate data and tools to constrain host star ages.
This also provides a wrapper to run existing tools for convenience.
Given a Gaia DR3 ID, the target is cross-matched with various catalogs using `astroquery.vizier`.
A table of measurements called "ref_table" for stellar parameters including their uncertainties and references is created.
In case of several possibly conflicting independent measurements from various sources, the user should adopt one definitive source.
Whenever possible, any missing stellar parameter is either measured from raw data (e.g. rotation period from TESS lightcurves) or supplied by the user (e.g. LiEW).
Depending on available measurements in ref_table, several tools are run:

### Python modules

The former monolithic `wakai.catalog` implementation is split by responsibility:

* `wakai.star` and `wakai.planet`: target models and archive queries
* `wakai.vizier` and `wakai.catalog_keys`: catalog discovery, download, and identifiers
* `wakai.cluster`: cluster membership, kinematics, and diagnostic plots
* `wakai.iso`: isochrone and stellar-evolution helpers
* `wakai.sed`: stellar-atmosphere and SED calibration helpers
* `wakai.gyro`, `wakai.var`, and `wakai.liew`: domain catalog registries

`wakai.catalog` remains a compatibility facade for existing notebooks.

The GUI's Target page can resolve a target with SIMBAD, query nearby
VizieR tables, populate the default `ref_table` parameters, and search any
additional VizieR column. Catalog candidates are only added after the user
selects a source, and every catalog-backed row includes an **Open ↗** link for
verification in a new browser tab.

Pipeline results are saved under `~/ql/wakai/` by default. This directory is set
by the GUI launcher (`wakai gui --output-dir`, or the `WAKAI_OUTPUT_DIR`
env var) — not by the pipeline scripts. Target state is persisted in SQLite at
`<output-dir>/wakai.sqlite3` (override the file directly with
`WAKAI_DATABASE_PATH`). **Apply Changes** saves target metadata and the full
`ref_table`; successful gyro-interp, BAFFLES, and joint results are saved
automatically and restored by Gaia ID on the next load. SQLite retains the
full floating-point values, while the GUI displays up to ten significant
figures and accepts unrestricted decimal precision in numeric inputs.

Each age-dating method's posterior is defined over some age grid, and the
GUI resamples every method's PDF onto one shared grid (`ageGrid` in
`gui/state.js`) before combining them into the joint posterior. Only
BAFFLES's and EAGLES's grids are an intrinsic property of those packages
(a constant and a function default, respectively); gyro-interp is
grid-agnostic (it evaluates at whatever ages it's asked for), and
isochrones' MCMC fit returns raw age samples with no grid at all, so for
both of those the "grid" below is wakai's own choice -- a query grid for
gyro-interp (`run_gyro.py`), a histogram binning for isochrones
(`run_isochrones.py`) -- not something read off the package itself:

| Method | Grid (native, or wakai's own choice) | Points | Range (Myr) |
|---|---|---|---|
| gyro-interp | linear, ~5.2 Myr steps -- wakai's query grid, not a gyro-interp property | 500 | 0–2600 |
| BAFFLES | log-spaced, ~0.1 Myr steps @10 Myr → ~124 Myr steps @13000 Myr -- native (`baffles_ca.const.AGE`) | 1000 | 1–13000 |
| EAGLES | log-spaced, ~0.12 Myr steps @10 Myr → ~145 Myr steps @12589 Myr -- native (`get_li_age_v2`'s own defaults) | 821 | 1–12589 |
| isochrones | log-spaced, ~0.32 Myr steps @10 Myr → ~124 Myr steps @13000 Myr -- wakai's histogram binning of raw MCMC samples | 300 bins | 1–13000 |
| **shared client grid** | union of two grids (below) | 1500 | 1–13000 |
| — log component | log-spaced, ~0.06 Myr steps @10 Myr → ~82 Myr steps @13000 Myr, matching BAFFLES's own step | 1001 | 1–13000 |
| — linear component | linear, ~5.2 Myr steps, matching the step `run_gyro.py` requests from gyro-interp | 500 | 1–2600 |

The shared grid is the **union** of two component grids, not one scheme
approximating both regimes: a log-spaced grid over the full range (constant
relative/dex resolution, matching BAFFLES/EAGLES's own log-spaced grids and
isochrones' comparably-spaced histogram binning, sized to resolve at least
as finely as BAFFLES's own step, the tightest of the three) merged with a
linear grid over 1–2600 Myr matching the step `run_gyro.py` requests from
gyro-interp. A log grid alone would under-resolve that gyro-interp query
above ~800 Myr, where its ~5.2 Myr step becomes finer than a log grid's
widening dex step; a linear grid alone would need ~2500 points just to
match BAFFLES's step down at 1 Myr. The union meets every method's own
resolution requirement across its own valid range (gyro-interp's
calibration *ceiling* is 2600 Myr, per Bouma+2023; its calibration
*floor*, ~80 Myr, is a reliability caveat on the result -- see
`GYRO_CALIBRATION_FLOOR_MYR` in `gui/summary.js`, and Bouma+2023's note
that below 80 Myr the model doesn't extrapolate, it just returns a uniform
posterior -- not a bound on the grid `run_gyro.py` actually requests, which
runs the full 0-2600 Myr) at the same total cost, 1500 points, as a
log-only grid. Because the log component concentrates points where the
dex-range is largest, roughly 70% of the shared grid's points fall within
1–1000 Myr -- the range that matters most for typical (young to
intermediate-age) transiting-planet hosts.

* module: cluster (or kinematics?)
 - method: match_cluster
   Given a Gaia DR2 or Gaia DR3 ID, the star is matched with Perren+2023 catalog members to check which cluster the target belongs to.
   If there is no direct ID match, a 6-D crossmatching is done to assess which cluster the target most likely belongs to if any.
   inputs: ra, dec, parallax, pmra, pmdec, rv
 - method: find_cluster
   uses comove package in /ut2/jerome/github/research/project/ext_tools/Comove
   inputs: ra, dec, parallax, pmra, pmdec, rv
 - method: run_kalkayotl
   uses kalkayotl in /ut2/jerome/github/research/project/ext_tools/Kalkayotl2
 - method: run_zoomies
   uses zoomies in /ut2/jerome/github/research/project/ext_tools/zoomies
   inputs:
* module: sed (or star_atmosphere?)
 - method: run_sedfit
   uses SedFit in /ut2/jerome/github/research/project/ext_tools/SEDFit
 - method: run_ariadne
   uses astroARIADNE in /ut2/jerome/github/research/project/ext_tools/astroARIADNE
 - method: run_brutus
   uses brutus in in /ut2/jerome/github/research/project/ext_tools/brutus_package
   inputs: broadband magnitudes in from NUV to NIR
   other_source: spherex data
* module: iso (or star_evol? or star_interior?)
 - method: run_madys
   uses madys in /ut2/jerome/github/research/project/ext_tools/madys
 - method: run_isochrones
   uses isochrones2 in /ut2/jerome/github/research/project/ext_tools/isochrones2
   inputs: broadband photometry and optional Teff,logg,[Fe/H]
* module: gyro
 - method: run_gyrointerp
   uses gyro-interp in /ut2/jerome/github/research/project/ext_tools/gyro-interp
   inputs: Prot, Teff
 - method: run_fleck
   uses fleck in
   input: stellar variability/ rotation amplitude
* module: var
 - method: run_eva
   uses eva package in /ut2/jerome/github/research/project/ext_tools/EVA2
   inputs: Gaia magnitudes
* module: liew
 - method: run_eagles
   uses EAGLES in /ut2/jerome/github/research/project/ext_tools/eagles
 - method: run_baffles
   uses BAFFLES in /ut2/jerome/github/research/project/ext_tools/BAFFLES
   inputs: LiEW, Teff, Ca-II R_HK
* others
 - asteroseismology (only for evolved stars) using pysyd
 - X-ray-age relation
 - machine learning

## Methods
According to Age-dating methods [Soderblom+2014](https://ui.adsabs.harvard.edu/abs/2015ASSP...39....3S/abstract),
there are

* Fundamental
radioactive dating of meteorites on Earth for age-dating the Sun
* Semi-fundamental
 - Lithium Depletion Boundary (LDB)
 - Kinematic traceback/ expansion rate
* Model dependent
 - isochrones
 - asteroseismology
 - surface gravity
 - radiative convective gap
* Empirical
 - rotation, amplitude, activity, disks, LiEW
* Statistical
 - metallicity
 - kinematics

See also this [ppt slides](https://beuther.www3.mpia.de/ppvi/talks/soderblom_jeffries.pdf) for summary.

## Cluster data
* 10 Myr
  - Upper Sco
  - TW Hya
  - nu Cha (11)
  - Beta Pic (23)

* 15 - 25 Myr
  - Sco-Cen

* 50 Myr
  - Tuc-Hor
  - IC 2391
  - IC 2602

* 100- 150 Myr
  - Pisces-Eridani (120)
  - Pleiades (125)
  - AB Dor (150)

* 250 Myr
  - M 34 (220 Myr)
  - Melange I
  - Group X

* 300 Myr
  - NGC 3532

* 400 Myr
  - M 48 (450 Myr)

* 600 - 800 Myr
  - Praesepe (670)
  - Hyades (730)

* 1 Gyr
  - NGC 6811

* 2.5 Gyr
  - NGC 6819
  - Ruprecht 147 (2.7 Gyr)

See notebooks/cluster_catalog.ipynb

## Stellar interior evolutionary models
* See [Morton+2015]() using [isochrones v2](https://github.com/timothydmorton/isochrones) code for MIST model
* See [Squicciarini & Bonavita](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract) using [MADYS](https://github.com/vsquicciarini/madys) code

## Stellar atmosphere models
* See [Vines & Jenkins 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.2719V/abstract) using [ARIADNE](https://github.com/jvines/astroARIADNE) code for several models

## Kinematics
* See [Sagear+2024](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract) using [zoomies](https://github.com/ssagear/zoomies) code
* See [Tofflemire+2021](https://ui.adsabs.harvard.edu/abs/2021AJ....161..171T/abstract) using [comove](https://github.com/adamkraus/Comove) code
* [Gagne+2018]() using [BANYAN SIGMA](https://www.exoplanetes.umontreal.ca/banyan/banyansigma.php) code
* [Clusterix](http://clusterix.cab.inta-csic.es/clusterix/)
* [ucc.ar](ucc.ar)

## Gaia excess
* See [Barner & Mann 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...953..127B/abstract) using Gaia excess uncertainty variability [code](https://github.com/madysonb/EVA)

## Gyrochronology
* See [Bouma+2023](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract) using [gyro-interp](https://github.com/lgbouma/gyro-interp) code
* See [Messina+2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.144M/abstract)
* See [Angus+2018]() using [stardate](https://github.com/RuthAngus/stardate) code
* See also [Astraea](https://github.com/lyx12311/Astraea) code
* [Barnes 2007]()
* [Mamajek & Hillenbrand 2008]()
* [Angus+2015]()

## Chromospheric activity indicator
* log(R'_HK)
* S_HK index: ratio of narrow flux to the background continuum flux
* See description in Isaacson & Fischer (2010)

## Chromospheric emission
 - Ca II K: 3933.5 Å
 - Ca II H: 3968.5 Å
 - H epsilon: 3970 Å

## Activity vs color or Li
* See [Stanford-Moore+2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...27S/abstract) using [BAFFLES](https://github.com/adamstanfordmoore/BAFFLES) code
* See [Jeffries+2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523..802J/abstract) using [EAGLES](https://github.com/robdjeff/eagles) code
* See activity-age relations of Mamajek & Hillenbrand (2008)

## Hα absorption
* 6562.8 Å

## Na I doublet absorption
* 5889.95 and 5895.92Å

## Li I doublet absorption
* [Wood+2023](https://ui.adsabs.harvard.edu/abs/2023AJ....166..247W/abstract)
* Li I resonance: 6708 Å: 6707.78 & 6707.91 Å
* See description in [Bouma+2022a](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract)
* Pleiades: [Bouvier+2018](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/613/A63)
* Hyades & Praesepe: [Cummings+2017](https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/153/128)

## Xray and UV emission
* detection by GALEX with (NUV, 1750–2750 Å) and far-UV (FUV, 1350–1750 Å)
* (NUV-J) and (J − K) colors and the empirical relations presented in [Findeisen+2011]()

## Cooling and contraction models
* [Linder+2019](): See Fig. 10 in [Dong+2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...926L...7D/abstract)

## Plots

### color/Teff - absolute magnitude

### Prot vs color
* B-V
* V-K_s
* M 48 (Barnes+2015)
* M 34 (Meibom+2011)

### Prot vs Teff
* See [Bouma+2020]

### Li EW & Teff
* See [Jeffries+2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523..802J/abstract) using [EAGLES](https://github.com/robdjeff/eagles) code

### Summary
* violin plot e.g. Fig. 14 in [David+2018]()
* weighted posterior distribution e.g. [Stanford-Moore+2020]()
