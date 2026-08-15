"""VizieR catalog identifiers grouped by measurement type."""

VIZIER_KEYS_STAR_CATALOG = {
    # Kepler Field with Gaia DR3
    "Godoy-Rivera2025": "J/A+A/696/A243",
    # Solar-like oscillators catalogue
    "Hatt2023": "J/A+A/669/A67",
    # Survey of Surveys: II. Stellar parameters for 23 million stars
    "Turchi2025": "J/A+A/700/A195", # see also https://zenodo.org/records/7811871
    # Stellar parameters for 220m stars from Gaia DR3 BP/RP spectra with machine learning
    #
    #The Stellar Abundances and Galactic Evolution Survey (SAGES). I. General Description and the First Data Release (DR1)
    #https://ui.adsabs.harvard.edu/abs/2023ApJS..268....9F/abstract
    #
    #SAGES II. Machine Learning–based Stellar Parameters for 21 Million Stars from the First Data Release
    #https://ui.adsabs.harvard.edu/abs/2025ApJS..277...19G/abstract
    # LAMOST
    "Zhang2023": "local",
    # https://zenodo.org/records/7945154: https://ui.adsabs.harvard.edu/abs/2023ApJS..267....8A/abstract
    "Andrae2023": "local", #Robust Data-driven Metallicities for 175 Million Stars from Gaia XP Spectra
    #SAGES IV. Surface Gravity Estimation and Giant─Dwarf Separation with the DDO51 Filter
    # "Carillo2020": "https://arxiv.org/abs/1911.07825",  # Gaia2+APOGEE14+GALAH+RAVE5+LAMOST+SkyMapper for TESS host stars
    # "Queiroz2020": "https://arxiv.org/abs/1710.09970",  # starhorse: using APOGEE+Gaia2
    # "HardegreeUllman2020": "https://arxiv.org/abs/2001.11511",  # Gaia2+LAMOST for K2 host stars (TICv8)
    # see also: http://kevinkhu.com/table1.txt
    # https://iopscience.iop.org/0067-0049/247/1/28/suppdata/apjsab7230t1_mrt.txt
    # "Anders2019": "https://arxiv.org/abs/1904.11302",  # panstarrs+2MASS+AllWISE+some APOGEE
    # kinematic thin disc, thick disc, and halo membership probabilities:
    # https://zenodo.org/record/3546184#.Xt-UFIFq1Ol
    # Sloan Digital Sky Survey Apache Point Observatory Galaxy Evolution Experiment (APOGEE)
    # "Ahumada2020": "https://arxiv.org/abs/1912.02905",  # SDSS16: using APOGEE2 -southern+eBOSS spectra
    # "Buder2018": "https://arxiv.org/abs/1804.06041",  # GALAH2
    # "Buder2020": "https://arxiv.org/abs/2011.02505",  # GALAH3: Gaia2+SME
    # "Hansen2020": "https://arxiv.org/pdf/2009.08154.pdf",  # Kepler: APOGEE+Gaia+2MASS+Stromgen
    # "Lin2020": "https://arxiv.org/abs/1911.05221",  # GALAH2=isochrone ages and init bulk met
    # distance- and extinction-corrected CMD, extinction maps as a function of distance, and density maps
    # "Guiglion2020": "https://arxiv.org/abs/2004.12666",  # rave w/ CNN
    # "McMillan2018": "https://arxiv.org/abs/1707.04554",  # TGAS+RAVE for radial velocities
    # "Casey2016": "https://arxiv.org/abs/1609.02914",  # RAVE-on, ages
    # "Auge2020": "https://arxiv.org/abs/2003.05459",  # M-giants asteroseismic distances
    # ===with planet
    # "Wittenmyer2020": "https://arxiv.org/abs/2005.10959",  # K2C1-13+GALAH/HERMES
    # "HardegreeUllman2019": "https://arxiv.org/abs/1905.05900",  # keplerMdwarfs
    "Abdurrouf2022": "III/286", #APOGEE-2 DR17 final allStar catalog
    "Buder2018": "J/MNRAS/478/4513", #The GALAH Survey: second data release
}
VIZIER_KEYS_AGE_CATALOG =  {
    # Gaia Galactic open clusters located within 1kpc
    "Alfonso2024": "J/A+A/689/A18",
    # Bayesian ages of local young stellar associations
    "Olivares2025": "J/A+A/699/A122",
    # The GALAH+ Survey DR3
    "Buder2021": "J/MNRAS/506/150", #see table 3 for ages
    # Bayesian Ages For Field LowEr-mass Stars
    "Stanford-Moore2020": "J/ApJ/898/27",
    # Gaia-Kepler stellar properties catalog. II. Planets
    "Berger2020": "J/AJ/160/108",
    # Gyrochronology ages for ~100000 stars & host stars
    "Lu2024": "J/AJ/167/159",
    # ages of 269 OC
    "Bossini2019": "J/A+A/623/A108",
    #"Bouma2024a:" "https://content.cld.iop.org/journals/0004-637X/976/2/234/revision1/apjad855ft1_mrt.txt",
    # "Sanders2018": "https://arxiv.org/abs/1806.02324",  # Isochrone ages for ∼3 million stars
    }
VIZIER_KEYS_VARIABLE_STAR_CATALOG = {
    "Jayasinghe2020":"II/366", #ASAS-SN catalog of variable stars (Jayasinghe+, 2018-2020)
    "Samus2017": "B/gcvs/gcvs_cat", #General Catalogue of Variable Stars
    "Watson2006": "B/vsx/vsx", #AAVSO International Variable Star Index VSX
    "Chen2020": "J/ApJS/249/18", #  The ZTF catalog of periodic variable stars
    "Clement2017": "V/150/variabls", #Updated catalog of variable stars in globular clusters
    "Oelkers2018": "J/AJ/155/39/Variables", #Variability properties of TIC sources with KELT
    "Pojmanski2005": "II/264", #ASAS Variable Stars in Southern hemisphere (Pojmanski+, 2002-2005)
}
VIZIER_KEYS_ACTIVITY_CATALOG = {
    #Gaia DR3 with eRASS1 detection CaII IRT
    'Freund2025': "J/A+A/697/A230",
    # HARPS activity indicators
    "Perdelwitz2024" : "J/A+A/683/A125",
    # SRG/eROSITA all-sky survey catalogs (eRASS1)
    "Merloni2024": "J/A+A/682/A34",
}
# https://docs.google.com/document/d/1s6OgiJlBVwonAYvQ3VONB4ioWOn0SHtJesrnaXJETBA/edit
VIZIER_KEYS_BINARY_STAR_CATALOG = {
    # PAST VIII. Kinematic Characterization and Identification of RV Variables for the LAMOST–Gaia–TESS Stars
    "Wu2025": "local",
    # from astropy.io import ascii
    # fp = "https://content.cld.iop.org/journals/0067-0049/281/2/36/revision1/apjsae0d85t7_mrt.txt"
    # table = ascii.read(fp, format="cds")
    # df = table.to_pandas()
    #Stellar and substellar companions from Gaia EDR3
    "Kervella2022": "J/A+A/657/A7",
    #https://ui.adsabs.harvard.edu/abs/2025ApJ...992..131C/abstract
    #see data/Chance2022/Chance2022_gaia_dr3_binaries.h5
    #Millions of Main-Sequence Binary Stars from Gaia BP/RP Spectra:
    #https://ui.adsabs.harvard.edu/abs/2025arXiv250709622L/abstract
    "Ding2024": "J/AJ/167/192", #TESS catalog of 1322 contact binary candidates
    "IJspeert2024":"J/A+A/691/A242", #TESS OBAF-type eclipsing binaries
    "Prsa2022": "J/ApJS/258/16", #The Eclipsing Binary stars (TESS-EBs) catalog
    "Pesta2023": "J/A+A/672/A176", #Contact binary candidates in the Kepler Eclipsing Binary Catalog (2172 rows)
    "Shi2022": "J/ApJS/259/50", #EA-type eclipsing binaries observed by TESS
    "Hartman2022": "J/ApJ/934/72", #SUPERWIDE wide binary systems with TESS & K2 data (Hartman+, 2022)
    "Zasche2022": "J/A+A/664/A96", # #Multiply eclipsing candidates from TESS satellite
    "Justesen2021": "J/ApJ/912/123", #TESS EBs in the southern hemisphere
    "Birko2019": "J/AJ/158/155", #SB candidates from the RAVE & Gaia DR2 surveys
    "Graczyk2019": "J/ApJ/872/85", #Detached eclipsing binaries with Gaia parallaxes
    "Pawlak2016": "J/AcA/66/421/ecl", #Eclipsing binaries in the Magellanic System
    "Abdul-Masih2016": "J/AJ/151/101", #Kepler Mission. VIII. 285 false positives (Abdul-Masih+, 2016)
    "Collins2018": "J/AJ/156/234", #KELT transit false positive catalog for TESS
    "Schanche2019": "J/MNRAS/488/4905", #SuperWASP transit false positive catalog
    "Oelkers2016": "J/AJ/152/75", # PMS binaries in YMG
    "Southworth2015": "V/152", #The DEBCat detached eclipsing binary catalogue
    # "Qian2019": "",
    # "Liu2024a": "",
    # "El-Badry2021": "",
    # "Jing2024": "",
    # "": J/AJ/158/25"
    #=====no CDS data but worth reading:=====#
    # EB in LAMOST: https://ui.adsabs.harvard.edu/abs/2024ApJ...969..114L/abstract
    # https://ui.adsabs.harvard.edu/abs/2025ApJS..277...15J/abstract
    # https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2269E/abstract
    # See also FPs in https://ui.adsabs.harvard.edu/abs/2025arXiv250209790V/abstract
    # See FPs in TOIs: https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.3749M/abstract
    # table in: https://academic.oup.com/mnras/article/521/3/3749/7081367#supplementary-data
    # See also
    # See also https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.3995K/abstract
    # See also https://ui.adsabs.harvard.edu/abs/2020ApJ...895....2P/abstract
    # classification of variable stars:
    # https://ui.adsabs.harvard.edu/abs/2025ApJS..276...57G/abstract
    #=====Related references=====#
    # binary sequence https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..977H/abstract
    }
VIZIER_KEYS_RV_CATALOG = {
    # HARPS activity indicators
    "Perdelwitz2024" : "J/A+A/683/A125",
    # HARPS radial velocity database
    "Trifonov2020": "J/A+A/636/A74",
    # Radial velocities of 51 exoplanets
    "Osborne2025": "J/A+A/693/A4",
    }
# See more keywords here: http://vizier.cds.unistra.fr/vizier/vizHelp/cats/U.htx
VIZIER_KEYS_LiEW_CATALOG = {
    # Members for 41 open clusters
    "Gutierrez-Albarran2024": "J/A+A/685/A83",
    # Bayesian Ages For Field LowEr-mass Stars
    "Stanford-Moore2020": "J/ApJ/898/27",
    # Members for 20 open clusters
    "Gutierrez2020": "J/A+A/643/A71",
    # The Seven Sisters DANCe. II. Pleiades
    "Barrado2016_Pleiades": "J/A+A/596/A113",
    # WOCS. LXXV. Hyades&Praesepe stellar lithium data
    "Cummings2017_HyadesPraesepe": "J/AJ/153/128",
    # Iz photometry, RV and EW(Li) in IC 4665
    "Manzi2008_IC4665": "J/A+A/479/141",
    # BV photometry and Li abundances in NGC3960
    "Prisinzano2007_NGC3960": "J/A+A/475/539",
}
VIZIER_KEYS_ALi_CATALOG = {
    # Lithium depletion boundary in stellar assoc.
    "Galindo-Guil2022": "J/A+A/664/A70",
    # Membership and lithium of 10-100Myr clusters
    "Franciosini2022": "J/A%2bA/659/A85",
    # Evolution of Li in FGK dwarf stars
    "LlorentedeAndres2021": "J/A+A/654/A137",
    # Lithium abundances from LAMOST MRS DR9 spectra
    "Ding2024": "J/ApJS/271/58",
    # Li abundance and mixing in giant stars
    "Magrini2021": "J/A%2bA/651/A84",
    # Li abundance values for stars in NGC 6819
    "Deliyannis2019": "J/AJ/158/163",
    # Lithium content for 148 Pleiades stars
    "Bouvier2018_Pleiades": "J/A+A/613/A63",
    # Lithium abundances in IC 2602 and IC 2391
    "Randich2001_IC2602": "J/A+A/372/862",
}
VIZIER_KEYS_PROT_CATALOG = {
    "Boyle2026": "local", #https://zenodo.org/records/18342591
    # Gyrochronology ages for ~100000 stars & host stars
    "Lu2024": "J/AJ/167/159",
    # Pleiades, Praesepe, NGC 6811, NGC 752, NGC 6819, and Ruprecht 147
    # See table1: https://arxiv.org/pdf/1905.10588.pdf
    "Curtis2019_Rup147": "J/ApJ/904/140",
    "Curtis2019_PisEri": "J/AJ/158/77",  # 250Gyr
    "Curtis2019_NGC6811": "J/ApJ/879/49",  # 1Gyr
    "Fritzewski2021_NGC3532": "J/A+A/652/A60",    #300 Myr
    # "Feinstein2020_NYMG": "See data/Feinstein2020_NYMG.txt",
    "McQuillan2014_Kepler": "J/ApJS/211/24",
    "Nielsen2013_KeplerMS": "J/A+A/557/L10",
    "Barnes2015_NGC2548": "J/A+A/583/A73",  # , M48/NGC2548
    "Meibom2011_NGC6811": "J/ApJ/733/L9",
    "Douglas2017_Praesepe": "J/ApJ/842/83",  # 680 Myr
    "Rebull2016_Pleiades": "J/AJ/152/114",  # 100 Myr
    "Rebull2017_Praesepe": "J/ApJ/839/92/table1",
    "Rebull2018_USco_rhoOph": "J/AJ/155/196",  # 10 Myr
    "Rebull2020_Taurus": "J/AJ/159/273",
    "Reinhold2020_K2C0C18": "J/A+A/635/A43",
    # "Feinstein+2020"
    # http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=
    # "Douglas2019_Praesepe": "2019ApJ...879..100D",
    # "Fang2020_PleiadesPraesepeHyades": "2020MNRAS.495.2949F",
    # "Gillen2020_BlancoI": "2020MNRAS.492.1008G",
    "Canto2020_TOIs": "J/ApJS/250/20",
    # https://filtergraph.com/tess_rotation_tois
    }
VIZIER_KEYS_CLUSTER_CATALOG = {
    # Gaia Galactic open clusters located within 1kpc
    "Alfonso2024": "J/A+A/689/A18",
    #sco-cen: https://ui.adsabs.harvard.edu/abs/2025arXiv250919693F/abstract
    #beta pic: https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.4760L/abstract
    #https://ui.adsabs.harvard.edu/abs/2025ApJS..280...16W/abstract
    #https://ui.adsabs.harvard.edu/abs/2025arXiv250919693F/abstract
    "Risbud2025": "J/A+A/694/A258", # Tidal tails of nearby open clusters
    # Census of BPMG & other nearby associations with Gaia
    "Luhman2024": "J/AJ/168/159",
    # A Census of the Stellar Populations in the Sco-Cen Complex
    "Luhman2022": "J/AJ/163/24",
    # A Gaia astrometric view of Pleiades, Praesepe, and Blanco 1
    "Alfonso2023": "J/A+A/677/A163",
    # members
    "Ratzenboeck2023a": "J/A%2bA/677/A59",
    # ages
    "Ratzenboeck2023b": "J/A+A/678/A71",
    # Open cluster parameters derived using Gaia XP
    "Nizovkina2025": "J/A+A/703/A100",
    # Using cluster masses, radii, and dynamics to create a cleaned open cluster catalogue
    "Hunt2024": "J/A+A/686/A42", #	Improving the open cluster census. III.
    # UCC
    "Perren2023": "local",
    # 3D Kinematics and age of OCs
    "Tarricq2021": "J/A+A/647/A19",
    # Survey of Surveys. I. Radial velocities
    "Tsantaki2022": "J/A+A/659/A95",
    # Coronae of nearby star clusters
    "Meingast2021" : "J/A+A/645/A84",
    # Stellar populations in IC 2602 and IC 2391
    "Nisak2022": "J/AJ/163/278",
    # Melange4: A 27 Myr Extended Population of Lower Centaurus Crux with a Transiting Two-planet System
    "Wood2023": "J/AJ/165/85",
    # Melange3: A Pleiades-age Association Harboring Two Transiting Planetary Systems from Kepler
    "Barber2023": "J/AJ/164/88",
    # Melange2: Membership, Rotation, and Lithium in the Young Cluster Group-X and a New Young Exoplane
    "Newton2022": "J/AJ/164/115",
    # Melange1: Three Small Planets Orbiting a 120 Myr Old Star in the Pisces-Eridanus Stream
    "Tofflemire2021": "J/AJ/161/171",
    # An all-sky cluster catalogue with Gaia DR3; 7167 total clusters, 2387 new
    "Hunt2023": "J/A+A/673/A114",
    # New Coronae and Stellar Associations Revealed by a HDBSCAN of the Solar Neighborhood
    "Moranta2022": "J/ApJ/939/94",
    # Open cluster non-degenerate stars analysis
    "Gutierrez2022": "J/ApJ/929/29",
    # Open clusters in solar neighborhood
    "Pang2021": "J/ApJ/912/162",
    # 3794 open clusters parameters
    "Hao2022": "J/A+A/660/A4",
    # c.f. Evolution of the local spiral structure of the Milky Way revealed by open clusters
    "Hao2021": "J/A+A/652/A102",
    # 1656 new star clusters found in Gaia EDR3
    "He2022c": "J/ApJS/264/8",
    # 886 Clusters within 1.2 kpc of the Sun
    "He2022b": "J/ApJS/262/7",
    # 541 new open cluster candidates
    "He2022a": "J/ApJS/260/8",
    # GALAH+APOGEE survey. Galactic disc with open clusters
    "Spina2021": "J/MNRAS/503/3279",
    # 628 new open clusters found with OCfinder
    "CastroGinard2022": "J/A+A/661/A118",
    # 570 new open clusters in the Galactic disc
    "CastroGinard2020": "J/A+A/635/A45",
    # Painting a portrait of the Galactic disc with its stellar clusters
    "CantatGaudin2020b": "J/A+A/640/A1/nodup",
    # Clusters and mirages: 1481 clusters and their members
    "CantatGaudin2020": "J/A+A/633/A99",
    # Untangling the Galaxy. II. Structure within 3kpc
    "Kounkel2020": "J/AJ/160/279",
    # Gaia-ESO Survey: members of 32 open clusters
    "Jackson2020" : "J/MNRAS/496/4701",
    # open clusters in the Galactic anticenter
    "CastroGinard2019": "J/A+A/627/A35",
    # Untangling the Galaxy. I. Local Structure and Star Formation History of the Milky Way
    # http://mkounkel.com/mw3d/hr.html
    "Kounkel2019": "J/AJ/158/122",
    # Gaia DR2 open clusters in the Milky Way
    "CantatGaudin2018": "J/A+A/618/A93",
    # HRD of Gaia DR2
    "Babusiaux2018": "J/A+A/616/A10",
    # merged catalogs
    "Bouma2019": "J/ApJS/245/13",
    # 28 GC in APOGEE14 and 11 GC in APOGEE16; no parallax
    "Nataf2019": "J/AJ/158/14",
    # Banyan sigma #https://jgagneastro.com/banyanii/
    "Gagne2018a": "J/ApJ/860/43",  # TGAS
    "Gagne2018b": "J/ApJ/862/138",  # DR2
    # APOGEE14+GALAH2 of open clusters
    "Carrera2019": "J/A+A/623/A80",
    # eta Cha assoc
    "Murphy2013": "J/MNRAS/435/1325",
    # nu Cha assoc
    "Bell2015": "J/MNRAS/454/593",
    # volans-carina: 90 Myr @ 85 pc
    "Gagne2018c": "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2018ApJ...865..136G",
    # Ruprecht 147 DANCe: oldest open cluster @ 300pc
    "Olivares2019": "J/A+A/625/A115",
    # Membership & properties of moving groups with Gaia
    "Ujjwal2020": "J/AJ/159/166",
    # young low-mass stars d<25 pc, spectroscopic obs
    "Shkolnik2009": "J/ApJ/699/649",
    # distances, kinematics, membership
    "Shkolnik2012": "J/ApJ/758/56",
    # Young Nearby Moving Groups from a Sample of Chromospherically Active Stars in RAVE
    "RamirezPreciado2018": "J/ApJ/867/93",
    # Young Nearby Moving Groups from a Sample of Chromospherically Active Stars in RAVE II
    # See also RAVE I: https://ui.adsabs.harvard.edu/abs/2013ApJ...776..127Z/abstract
    "Zerjal2017": "J/ApJ/835/61",
    # 146 nearby, young, low-mass young stars from all-sky search
    "Binks2020": "J/MNRAS/491/215",
    # Young Binaries and Lithium-rich Stars in the Solar Neighborhood
    "Bowler2019": "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2019ApJ...877...60B",
    # GALEX nearby young-star survey
    "Rodriguez2013": "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2013ApJ...774..101R",
    # Young (<100Myr) massive star clusters
    "Portegies2010": "http://simbad.u-strasbg.fr/simbad/sim-ref?querymethod=bib&simbo=on&submit=submit+bibcode&bibcode=2010ARA%26A..48..431P",
    # OC
    "Sampedro2017": "J/MNRAS/470/3937",
    # Gaia-ESO Survey (2011-2018) in 62 clusters with 1 Myr−8 Gyr
    "Randich2022": "J/A+A/666/A121",
    # Gaia-ESO Survey in 7 open star cluster fields
    "Randich2018": "J/A+A/612/A99",
    #
    "Kharchenko2013": "J/A+A/558/A53",
    # OC #"Dias2014"?
    "Dias2016": "B/ocl",
    # Psc Eri
    "Curtis2019": "J/AJ/158/77",
    # praesepe, alpa per
    "Lodieu2019": "J/A+A/628/A66",
    # ACRONYM III: young low-mass stars in the solar neighborhood
    "Schneider2019": "J/AJ/157/234",
    # young harps RV
    "Grandjean2020": "J/A+A/633/A44",
    # Gaia CKS II: Planet radius demographics as a function of stellar mass and age
    "Berger2020": "J/AJ/160/108",
    # Lithium abundances of KOIs from CKS spectra
    "Berger2018": "J/ApJ/855/115",
    # 'BailerJones2018': 'I/347', #distances
    # 'Luo2019': 'V/149', #Lamost
    # "Cody2018": "",
    # "Feinstein2020": "", #NYMG
    "Bianchi2017_GALEX": "II/335",  # see also https://ui.adsabs.harvard.edu/abs/2020ApJS..250...36B/abstract
    # YSO from SED using CNN
    "Chiu2021": "http://scao.astr.nthu.edu.tw/media/scao/files/documents/20191227_SEIP_total_MRT.txt",
    # age of Kepler stars estimated using astraea
    "Lu2021": "https://cfn-live-content-bucket-iop-org.s3.amazonaws.com/journals/1538-3881/161/4/189/1/"
    + "ajabe4d6t1_mrt.txt?AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1622511816&Signature=CVdNivtn2MJv1%2FY%2F4ztoZKBPzaw%3D",
}

ALL_VIZIER_DICTIONARIES = [
    # Preserve the historical lookup priority for duplicate author/year names.
    VIZIER_KEYS_PROT_CATALOG,
    VIZIER_KEYS_STAR_CATALOG,
    VIZIER_KEYS_LiEW_CATALOG,
    VIZIER_KEYS_ALi_CATALOG,
    VIZIER_KEYS_RV_CATALOG,
    VIZIER_KEYS_CLUSTER_CATALOG,
    VIZIER_KEYS_AGE_CATALOG,
    VIZIER_KEYS_VARIABLE_STAR_CATALOG,
    VIZIER_KEYS_ACTIVITY_CATALOG,
    VIZIER_KEYS_BINARY_STAR_CATALOG,
]
