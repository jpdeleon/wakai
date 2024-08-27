# `wakai`
`wakai` is the Japanese word for `young.` This repo contains several youth indicators and other tools to constrain exoplanet ages.

## Cluster
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

## Isochrones
* See [Morton+2015]() using [isochrones v2](https://github.com/timothydmorton/isochrones) code for MIST model
* See [Vines & Jenkins 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.2719V/abstract) using [ARIADNE](https://github.com/jvines/astroARIADNE) code for several models
* See [Squicciarini & Bonavita](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A..15S/abstract) using [MADYS](https://github.com/vsquicciarini/madys) code

## Kinematics
* See [Sagear+2024](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract) using [zoomies](https://github.com/ssagear/zoomies) code
* See [Tofflemire+2021](https://ui.adsabs.harvard.edu/abs/2021AJ....161..171T/abstract) using [comove](https://github.com/adamkraus/Comove) code
* [Gagne+2018]() using [BANYAN SIGMA](https://www.exoplanetes.umontreal.ca/banyan/banyansigma.php) code
  
## Gaia excess
* See [Barner & Mann 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...953..127B/abstract) using Gaia excess uncertainty variability [code](https://github.com/madysonb/EVA)
  
## Gyrochronology
* See [Bouma+2023](https://ui.adsabs.harvard.edu/abs/2020AJ....160..239B/abstract) using [gyro-interp](https://github.com/lgbouma/gyro-interp) code
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
* [Rebull+2016]()
* [Rebull+2017]()
* Melange: [Tofflemire+2021]()
* NGC 3532: [Fritzewski+2021](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/652/A60)
* Group X: [Newton+2022]() & [Messina+2022]()
* Praesepe: [Rampalli+2021]()
* Pleiades: [Curtis+2020]()
* Pis-Eri: [Curtis+2019](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/AJ/158/77)
* NGC 6811: [Curtis+2019](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/879/49)
* Stars with TESS contamination ratios > 0.2, RUWE > 1.3 (likely binary), or C*>0.04 (corrected GBP and GRP flux excess factor) were not analyzed and are not included in figures. C* calculated from phot_bp_rp_excess_factor following Riello et al. (2021, Table 2 and Equation (6)).

### Li EW & Teff
* See [Jeffries+2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523..802J/abstract) using [EAGLES](https://github.com/robdjeff/eagles) code
* Pleiades: [Soderblom+1993]() & [Bouvier+2018]()
* IC 2391/2602: [Randich+2001]()
* young moving groups: [Mentuch+2008]() & [Kraus+2014]() 
* Praesepe: [Cummings+2017]()
* Group X: [Newton+2022]() & [Messina+2022]()

### Summary
* violin plot e.g. Fig. 14 in [David+2018]()
* weighted posterior distribution e.g. [Stanford-Moore+2020]()