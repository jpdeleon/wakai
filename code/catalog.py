"""
TODO: 
1. replace print with logger.info
"""
import json, os, re
import itertools
import warnings
import requests
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from urllib.request import urlopen
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from pathlib import Path
from pprint import pprint
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.mast import Observations, Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from scipy.interpolate import NearestNDInterpolator
from loguru import logger

home = Path.home()
DATA_PATH = f'{home}/github/research/project/wakai/data/'


VIZIER_KEYS_AGE_CATALOG =  {
    "Berger2020": "J/AJ/160/108",
    "Lu2024": "J/AJ/167/159",
    #"Bouma2024a:" "https://content.cld.iop.org/journals/0004-637X/976/2/234/revision1/apjad855ft1_mrt.txt",
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
# https://docs.google.com/document/d/1s6OgiJlBVwonAYvQ3VONB4ioWOn0SHtJesrnaXJETBA/edit
VIZIER_KEYS_BINARY_STAR_CATALOG = {
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
    "Collins2018": "J/AJ/156/234/table4", #KELT transit false positive catalog for TESS 
    "Schanche2019": "J/MNRAS/488/4905/table2", #SuperWASP transit false positive catalog 
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
    "Perdelwitz2024" : "J/A+A/683/A125", # HARPS radial velocity database
    "Osborne2025": "J/A+A/693/A4", # Radial velocities of 51 exoplanets
    }
# See more keywords here: http://vizier.cds.unistra.fr/vizier/vizHelp/cats/U.htx
VIZIER_KEYS_LiEW_CATALOG = {
    "Buder2023": "J/MNRAS/506/150",
    "Ding2024": "J/ApJS/271/58", #Lithium abundances from LAMOST MRS DR9 spectra
    "Gutierrez-Albarran2024": "J/A+A/685/A83", # Members for 41 open clusters
    "Franciosini2022": "J/A%2bA/659/A85", #Membership and lithium of 10-100Myr clusters
    "Gutierrez2020": "J/A+A/643/A71", #Members for 20 open clusters (Gutierrez Albarran+, 2020)
    "Magrini2021_ALi": "J/A%2bA/651/A84", #Li abundance and mixing in giant stars
    "Deliyannis2019": "J/AJ/158/163", #Li abundance values for stars in NGC 6819
    "Randich2001_IC2602": "J/A+A/372/862", # Lithium abundances in IC 2602 and IC 2391
    "Barrado2016_Pleiades": "J/A+A/596/A113", #
    "Cummings2017_HyadesPraesepe": "J/AJ/153/128",
    "Manzi2008_IC4665": "J/A+A/479/141", #Iz photometry, RV and EW(Li) in IC 4665 
    "Prisinzano2007_NGC3960": "J/A+A/475/539", # BV photometry and Li abundances in NGC3960
    "Stanford-Moore2020": "J/ApJ/898/27",
    "Bouvier2018_Pleiades": "J/A+A/613/A63",
}
VIZIER_KEYS_PROT_CATALOG = {
    # See table1: https://arxiv.org/pdf/1905.10588.pdf
    "Curtis2019_Rup147": "J/ApJ/904/140", #Pleiades, Praesepe, NGC 6811, NGC 752, NGC 6819, and Ruprecht 147 
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
    "Ratzenboeck2023a": "J/A%2bA/677/A59", # members
    "Ratzenboeck2023b": "J/A+A/678/A71", # ages
    # Using cluster masses, radii, and dynamics to create a cleaned open cluster catalogue
    "Hunt2024": "J/A+A/686/A42",
    #
    "Perren2023": "local",
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
    # 3794 open clusters parameters
    "Hao2022": "J/A+A/660/A4",
    # c.f. Evolution of the local spiral structure of the Milky Way revealedby open clusters
    "Hao2021": "J/A+A/652/A102",
    # 1656 new star clusters found in Gaia EDR3
    "He2022c": "J/ApJS/264/8",
    # 886 Clusters within 1.2 kpc of the Sun
    "He2022b": "J/ApJS/262/7",
    # 541 new open cluster candidates
    "He2022a": "J/ApJS/260/8",
    # 628 new open clusters found with OCfinder
    "CastroGinard2022": "J/A+A/661/A118",
    # 570 new open clusters in the Galactic disc
    "CastroGinard2020": "J/A+A/635/A45",
    # 1481 clusters and their members
    "CantatGaudin2020": "J/A+A/633/A99",
    # Untangling the Galaxy. II. Structure within 3kpc (Kounkel+, 2020)
    "Kounkel2020": "J/AJ/160/279",
    # open clusters in the Galactic anticenter
    "CastroGinard2019": "J/A+A/627/A35",
    #
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
    # ages of 269 OC
    "Bossini2019": "J/A+A/623/A108/tablea",
    # APOGEE14+GALAH2 of open clusters
    "Carrera2019": "J/A+A/623/A80",
    # Argus assoc via simbad link
    "Zuckerman2019": "None",
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
    # Local structure & star formation history of the MW
    # http://mkounkel.com/mw3d/hr.html
    "Kounkel2019": "J/AJ/158/122",  # Local structure & star formation history of the MW
    # "Feinstein2020": "", #NYMG
    "Bianchi2017_GALEX": "II/335",  # see also https://ui.adsabs.harvard.edu/abs/2020ApJS..250...36B/abstract
    # YSO from SED using CNN
    "Chiu2021": "http://scao.astr.nthu.edu.tw/media/scao/files/documents/20191227_SEIP_total_MRT.txt",
    # age of Kepler stars estimated using astraea
    "Lu2021": "https://cfn-live-content-bucket-iop-org.s3.amazonaws.com/journals/1538-3881/161/4/189/1/"
    + "ajabe4d6t1_mrt.txt?AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1622511816&Signature=CVdNivtn2MJv1%2FY%2F4ztoZKBPzaw%3D",
}

@dataclass
class Star:
    """
    A class representing a star, extending the Target class with stellar-specific functionality.
    Provides methods to query and analyze data from various astronomical catalogs.
    """
    star_name: str
    source: str = "tic"  # Default source for stellar parameters
    tfop_data: dict = None
    data_json: dict = None
    exofop_url: str = None
    star_names: list = field(default_factory=list)
    gaia_name: str = None
    toiid: int = None
    ticid: int = None
    query_name: str = None
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize the parent Target class with default values."""
        super().__init__(ra_deg=None, dec_deg=None, gaiaid=None, verbose=self.verbose)
        
        # Initialize other attributes that will be populated later
        self.magnitudes = None
        self.rstar = None
        self.mstar = None
        self.rhostar = None
        self.teff = None
        self.logg = None
        self.feh = None

    def __repr__(self):
        """Return string representation of the Target object."""
        coord_str = self.target_coord.to_string("decimal").replace(" ", ", ")
        return (f"{self.__class__.__name__}("
                f"star_name='{getattr(self, 'star_name', 'Unnamed')}', "
                f"gaiaid={self.gaiaid}, "
                f"ra={self.ra}, dec={self.dec}, "
                f"coord=({coord_str}))"
               )

    def target_coord(self) -> SkyCoord:
        """Return SkyCoord object for the target position."""
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

    def query_vizier(self, radius=None, verbose=None, use_cached=True):
        """
        Query Vizier catalogs near the target position.
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec

        if use_cached and self._vizier_tables is not None:
            return self._vizier_tables

        if verbose:
            print(f"Searching Vizier at {self.target_coord().to_string('hmsdms')} with radius={radius}")

        vizier = Vizier(columns=["*", "+_r"])
        try:
            tables = vizier.query_region(self.target_coord(), radius=radius)
        except Exception as e:
            print(f"Vizier query failed: {e}")
            return None

        if tables is None or len(tables) == 0:
            if verbose:
                print("No Vizier tables found.")
            return None

        if verbose:
            print(f"{len(tables)} tables found.")
            from pprint import pprint
            pprint({k: tables[k]._meta.get("description", "") for k in tables.keys()})

        self._vizier_tables = tables
        return tables

    def get_vizier_param(self, param=None, radius=3, use_regex=False):
        """
        Search for a specific parameter across all Vizier tables.
        """
        tables = self.query_vizier(radius=radius, verbose=False)

        if not tables:
            print("No Vizier data to search.")
            return None

        if param is None:
            # Print available columns
            cols = [tab.colnames for tab in tables]
            flat_cols = sorted(set(col for sublist in cols for col in sublist))
            print(f"Available parameters:\n{flat_cols}")
            return None

        results = {}
        for i, tab in enumerate(tables):
            columns = tab.colnames

            if use_regex:
                pattern = re.compile(param.replace("*", ".*"), re.IGNORECASE)
                matched = [col for col in columns if pattern.search(col)]
            else:
                matched = [param] if param in columns else []

            for col in matched:
                value = tab[col][0]
                value = np.nan if isinstance(value, np.ma.core.MaskedConstant) else value
                results.setdefault(tables.keys()[i], {})[col] = value

        if self.verbose:
            match_type = "regex" if use_regex else "exact"
            print(f"{match_type.capitalize()} match: Found {len(results)} matching tables for '{param}'.")

        return results
    
    def query_binary_star_catalogs(self):
        """Query catalogs that contain information about binary stars."""
        self._query_star_catalog(VIZIER_KEYS_BINARY_STAR_CATALOG)                        
        
    def query_variable_star_catalogs(self):
        """Query catalogs that contain information about variable stars."""
        self._query_star_catalog(VIZIER_KEYS_VARIABLE_STAR_CATALOG)
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.get_vizier_table(verbose=False)
        
        if all_tabs is None:
            return
            
        # Check for `var` in catalog title
        idx = [
            n if "var" in t._meta.get("description", "").lower() else False
            for n, t in enumerate(all_tabs)
        ]
        
        for i in idx:
            if i is not False:  # Only process valid indices
                tab = all_tabs[i]
                try:
                    s = tab.to_pandas().squeeze().str.decode("ascii").dropna()
                except Exception:
                    s = tab.to_pandas().squeeze().dropna()
                    
                if len(s) > 0:
                    print(f"\nSee also: {base_url}{tab._meta['name']}\n{s}")
                    self.variable_star = True
        
    def _query_star_catalog(self, catalog_keys):
        """
        Check for stars in specialized catalogs.
        
        Parameters:
            catalog_keys (dict): Dictionary mapping reference names to VizieR keys
        """
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.get_vizier_table(verbose=False)
        
        if all_tabs is None:
            return
            
        for key, tab in zip(all_tabs.keys(), all_tabs.values()):
            for ref, vkey in catalog_keys.items():
                if key == vkey:
                    d = tab.to_pandas().squeeze()
                    print(f"{ref} ({base_url}{key}):\n{d}")
    
    def query_gaia_catalog(self, radius=None, version=2, return_nearest_xmatch=False, verbose=None):
        """
        Query the Gaia DR catalog for stellar sources.
        
        Parameters:
            radius (float, optional): Search radius in arcseconds
            version (int): Gaia catalog version to query (default is 2)
            return_nearest_xmatch (bool): If True, return only the nearest match
            verbose (bool, optional): Whether to print detailed output
            
        Returns:
            pd.DataFrame or pd.Series: Query results
        """
        if self._gaia_sources is not None:
            return self._gaia_sources.copy()
    
        radius_quantity = self.search_radius if radius is None else radius * u.arcsec
        verbose = self.verbose if verbose is None else verbose
    
        if verbose:
            logger.info(f"Querying Gaia DR{version} catalog at {self.target_coord().to_string()} within {radius_quantity:.2f}")
    
        try:
            tab = Catalogs.query_region(self.target_coord(), radius=radius_quantity, catalog="Gaia", version=version).to_pandas()
        except Exception as e:
            raise RuntimeError(f"Gaia query failed: {e}")
    
        if tab.empty:
            raise ValueError(f"No Gaia star found within {radius_quantity}")
    
        tab.rename(columns={"distance": "separation"}, inplace=True)
        tab["separation"] *= u.arcmin.to(u.arcsec)
        tab["source_id"] = tab["source_id"].astype(int)
    
        if not np.allclose(tab["ref_epoch"], 2015.5):
            raise ValueError("Non-standard epoch found (expected 2015.5)")
    
        self._gaia_sources = tab
    
        return self._process_nearest_match(tab, radius_quantity) if return_nearest_xmatch else self._process_multiple_matches(tab, radius_quantity)
    
    
    def _process_multiple_matches(self, tab, radius):
        """Clean and return multiple Gaia matches as a DataFrame."""
        tab["parallax"] = tab["parallax"].where(tab["parallax"] >= 0, np.nan)
    
        if tab["parallax"].isnull().all():
            raise ValueError(f"No valid parallax values found within {radius}")
    
        self._update_target_coord(tab)
        return tab
    
    
    def _process_nearest_match(self, tab, radius):
        """Return the nearest Gaia source as a Series."""
        nearest = tab.iloc[0]
        self._update_target_coord(tab)
        self._assign_gaia_metadata(nearest)
        return nearest
    
    
    def _update_target_coord(self, tab):
        """Update coordinates using nearest Gaia source if gaiaid is not set."""
        if self.gaiaid is None and not tab.empty:
            nearest = tab.iloc[0]
            if self.verbose:
                print(f"Updating target coordinates based on Gaia source (separation: {nearest.separation:.2f}\")")
            # Optional: update self.ra and self.dec if desired    
    
    def _assign_gaia_metadata(self, target):
        """Store Gaia metadata and flag potential binarity or data quality issues."""
        if self.gaiaid is None:
            self.gaiaid = int(target["source_id"])
        self.gaia_params = target
    
        def log_if(condition, message):
            if condition:
                logger.info(message)
    
        log_if(target.get("astrometric_excess_noise_sig", 0) >= 5, 
               f"astrometric_excess_noise_sig={target['astrometric_excess_noise_sig']:.2f} suggests binarity.")
    
        log_if(target.get("astrometric_gof_al", 0) >= 20, 
               f"astrometric_gof_al={target['astrometric_gof_al']:.2f} suggests binarity.")
    
        log_if(target.get("visibility_periods_used", 0) < 6, 
               "visibility_periods_used < 6: no astrometric solution")
    
        ruwe_results = self.get_vizier_param("ruwe")
        if ruwe_results:
            ruwe_val = next(iter(ruwe_results.values()[0].values()), None)
            if ruwe_val and ruwe_val > 1.4:
                logger.info(f"RUWE={ruwe_val:.1f} > 1.4 suggests non-single source")

    def query_tfop_data(self) -> dict:
        """
        Query the ExoFOP-TESS database for information about the star.
        
        Returns:
            dict: JSON data from ExoFOP-TESS
            
        Raises:
            ValueError: If the query fails or no data is found
        """
        if self.tfop_data:
            return self.tfop_data
            
        base_url = "https://exofop.ipac.caltech.edu/tess"
        self.exofop_url = f"{base_url}/target.php?id={self.star_name.replace(' ','')}&json"
        
        try:
            response = requests.get(self.exofop_url)
            response.raise_for_status()
            self.tfop_data = response.json()
            self.data_json = self.tfop_data  # For compatibility with older methods
            return self.tfop_data
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error while fetching data for {self.star_name}.\nError: {e}")
                
    def get_tfop_data(self) -> None:
        """
        Parse the TFOP info to get star names, Gaia name, Gaia ID, and target coordinates.
        """
        if self.tfop_data is None:
            self.tfop_data = self.query_tfop_data()
            
        # Extract star names
        self.star_names = np.array(
            self.tfop_data.get("basic_info", {}).get("star_names", "").split(", ")
        )
        
        if self.star_name is None:
            self.star_name = self.star_names[0] if len(self.star_names) > 0 else "Unknown"
            
        if self.verbose:
            logger.info("Catalog names:")
            for n in self.star_names:
                print(f"\t{n}")
                
        # Extract Gaia information
        gaia_mask = np.array([i[:4].lower() == "gaia" for i in self.star_names])
        if any(gaia_mask):
            self.gaia_name = self.star_names[gaia_mask][0]
            self.gaiaid = int(self.gaia_name.split()[-1])
        
        # Extract coordinates
        coordinates = self.tfop_data.get("coordinates", {})
        ra = coordinates.get("ra")
        dec = coordinates.get("dec")
        
        if ra is not None and dec is not None:
            self.ra_deg = float(ra)
            self.dec_deg = float(dec)
            self.target_coord = SkyCoord(ra=ra, dec=dec, unit="degree")

        # Extract TOI ID
        if self.star_name.lower()[:3] == "toi":
            parts = self.star_name.split("-")
            if len(parts) > 1:
                self.toiid = parts[-1]
            else:
                self.toiid = int(float(self.star_name.replace(" ", "")[3:]))
        else:
            idx = [i[:3].lower() == "toi" for i in self.star_names]
            if sum(idx) > 0:
                self.toiid = int(self.star_names[idx][0].split("-")[-1])
            else:
                self.toiid = None
                
        # Extract TIC ID
        self.ticid = int(self.tfop_data.get("basic_info", {}).get("tic_id", 0))
        if self.ticid:
            self.query_name = f"TIC{self.ticid}"
        else:
            self.query_name = self.star_name.replace("-", " ")

    def get_params_from_tfop(self, name="planet_parameters", idx=None) -> dict:
        """
        Get parameters from TFOP data.
        
        Parameters:
            name (str): Parameter set name ('planet_parameters' or 'stellar_parameters')
            idx (int, optional): Index of the parameter set to return
            
        Returns:
            dict: Selected parameter set
        """
        if self.tfop_data is None:
            self.tfop_data = self.query_tfop_data()
            
        params_dict = self.tfop_data.get(name, [])
        
        if not params_dict:
            return {}
            
        if idx is None:
            key = "pdate" if name == "planet_parameters" else "sdate"
            # Get the latest parameter based on upload date
            dates = [d.get(key, "") for d in params_dict]
            df = pd.DataFrame({"date": dates})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].notna().any():
                idx = df["date"].idxmax()
            else:
                idx = 0
                
        return params_dict[idx] if idx < len(params_dict) else {}
    
    def get_magnitudes(self):
        """Extract magnitude information from TFOP data."""
        if not hasattr(self, "data_json") or not self.data_json:
            self.data_json = self.query_tfop_data()
            
        if 'magnitudes' in self.data_json:
            self.magnitudes = pd.json_normalize(self.data_json['magnitudes'])
            self.magnitudes['value'] = pd.to_numeric(self.magnitudes['value'], errors='coerce')
            self.magnitudes['value_e'] = pd.to_numeric(self.magnitudes['value_e'], errors='coerce')
        else:
            self.magnitudes = pd.DataFrame()

    def get_star_params(self):
        """
        Extract stellar parameters from TFOP data.
        
        Raises:
            ValueError: If parameters cannot be extracted
        """
        if not hasattr(self, "data_json") or not self.data_json:
            self.data_json = self.query_tfop_data()

        # Get coordinates and TIC ID
        self.ra = float(self.data_json.get("coordinates", {}).get("ra", 0))
        self.dec = float(self.data_json.get("coordinates", {}).get("dec", 0))
        self.ticid = self.data_json.get("basic_info", {}).get("tic_id", 0)

        # Find parameters from the desired source
        stellar_params = self.data_json.get("stellar_parameters", [])
        self.tic_params = {}
        
        for i, p in enumerate(stellar_params):
            if p.get("prov") == self.source:
                self.tic_params = p
                break
                
        if not self.tic_params and stellar_params:
            self.tic_params = stellar_params[0]  # Use first set if source not found

        try:
            # Extract stellar radius
            self.rstar = (
                float(self.tic_params.get("srad", np.nan)),
                float(self.tic_params.get("srad_e", np.nan))
            )
            
            # Extract stellar mass
            self.mstar = (
                float(self.tic_params.get("mass", np.nan)),
                float(self.tic_params.get("mass_e", np.nan))
            )
            
            # Calculate stellar density in solar density units
            if not np.isnan(self.mstar[0]) and not np.isnan(self.rstar[0]) and self.rstar[0] > 0:
                density = self.mstar[0] / (self.rstar[0] ** 3)
                # Error propagation for density
                density_err = np.sqrt(
                    (1 / self.rstar[0] ** 3) ** 2 * self.mstar[1] ** 2
                    + (3 * self.mstar[0] / self.rstar[0] ** 4) ** 2 * self.rstar[1] ** 2
                )
                self.rhostar = (density, density_err)
            else:
                self.rhostar = (np.nan, np.nan)
                
            # Extract temperature
            self.teff = (
                float(self.tic_params.get("teff", np.nan)),
                float(self.tic_params.get("teff_e", 500))
            )
            
            # Extract surface gravity
            self.logg = (
                float(self.tic_params.get("logg", np.nan)),
                float(self.tic_params.get("logg_e", 0.1))
            )
            
            # Extract metallicity
            feh_val = self.tic_params.get("feh", 0)
            feh_val = 0 if (feh_val is None or feh_val == "") else float(feh_val)
            
            feh_err = self.tic_params.get("feh_e", 0.1)
            feh_err = 0.1 if (feh_err is None or feh_err == "") else float(feh_err)
            
            self.feh = (feh_val, feh_err)
            
            # Print parameter summary
            if self.verbose:
                if not np.isnan(self.mstar[0]):
                    print(f"Mstar=({self.mstar[0]:.2f},{self.mstar[1]:.2f}) Msun")
                if not np.isnan(self.rstar[0]):
                    print(f"Rstar=({self.rstar[0]:.2f},{self.rstar[1]:.2f}) Rsun")
                if not np.isnan(self.rhostar[0]):
                    print(f"Rhostar=({self.rhostar[0]:.2f},{self.rhostar[1]:.2f}) rhosun")
                if not np.isnan(self.teff[0]):
                    print(f"teff=({self.teff[0]:.0f},{self.teff[1]:.0f}) K")
                if not np.isnan(self.logg[0]):
                    print(f"logg=({self.logg[0]:.2f},{self.logg[1]:.2f}) cgs")
                print(f"feh=({self.feh[0]:.2f},{self.feh[1]:.2f}) dex")
                
        except Exception as e:
            print(f"Error extracting stellar parameters: {e}")
            raise ValueError(f"Check exofop: {self.exofop_url}")
            
    def query_simbad(self):
        """
        Query Simbad to get the object type of the target star.

        Returns:
            SimbadResult or None: Result of the query if target is resolved, otherwise None
        """
        # Add object type field to Simbad query
        Simbad.add_votable_fields("otype")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Make sure star_names is populated
            if not hasattr(self, "star_names") or not self.star_names:
                if self.star_name is None:
                    self.star_name = input("What is the target's name? ")
                self.tfop_data = self.query_tfop_data()
                self.get_tfop_data()
            
            # Try to resolve using each known name
            for name in self.star_names:
                result = Simbad.query_object(name)
                if result is not None:
                    return result
                    
            if self.verbose:
                msg = f"Simbad cannot resolve {self.star_name}"
                msg += f" using any of its names: {self.star_names}"
                logger.warning(msg)
                
            return None
                
    def get_simbad_obj_type(self):
        """
        Retrieve the object type of the target star from Simbad.

        Returns:
            str or None: Description of the object type if found, otherwise None
        """
        # Query Simbad for the target star
        result = self.query_simbad()

        if not result:
            return None
            
        # Extract the object type category
        category = result.to_pandas().squeeze().get("OTYPE", "")

        if not category or len(category) < 1:
            return None
            
        try:
            # Load Simbad object type descriptions
            df = pd.read_csv(simbad_obj_list_file)
            matches = df[df["Id"] == category]

            if len(matches) > 0:
                # Retrieve the description and id
                desc = matches["Description"].iloc[0]
                oid = matches["Id"].iloc[0]

                # Check if the description contains 'binary' and print appropriate message
                if "binary" in desc.lower():
                    logger.info("***" * 15)
                    logger.info(f"Simbad classifies {self.star_name} as {oid}={desc}!")
                    logger.info("***" * 15)
                else:
                    logger.info(f"Simbad classifies {self.star_name} as {oid}={desc}!")

                return desc
            else:
                return category  # Return raw category if not found in description file
        except Exception as e:
            logger.warning(f"Error loading Simbad object types: {e}")
            return category

    def get_spectral_type(
        self,
        columns="Teff Bp-Rp J-H H-Ks W1-W2 W1-W3".split(),
        nsamples=int(1e4),
        return_samples=False,
        plot=False,
        ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.
        c.f. self.get_vizier_table_param("SpT")
    
        Parameters
        ----------
        nsamples : int
            number of Monte Carlo samples (default=1e4)
    
        Returns
        -------
        interpolated spectral type
    
        Notes:
        It may be good to check which color index yields most accurate result
    
        Check sptype from self.query_simbad()
        """
        df = get_Mamajek_table()
        gaia_sources = self.get_gaia_sources()
        self.gaia_params = gaia_sources.iloc[0].squeeze()
    
        # effective temperature
        col = "teff"
        teff = self.gaia_params[f"{col}_val"]
        siglo = (self.gaia_params[f"{col}_val"] - self.gaia_params[f"{col}_percentile_lower"])
        sighi = (self.gaia_params[f"{col}_percentile_upper"] - self.gaia_params[f"{col}_val"])
        uteff = np.sqrt(sighi**2 + siglo**2)
        s_teff = (teff + np.random.randn(nsamples) * uteff)  # Monte Carlo samples
        print(f"Gaia Teff={teff},{uteff} K")
        bands=['Gaia','V','J','H','K',
               'WISE 3.4 micron','WISE 4.6 micron','WISE 12 micron']
        mags = {}
        for band in bands:
            val=self.magnitudes.query("band==@band").value.squeeze()
            err=self.magnitudes.query("band==@band").value_e.squeeze()
            mags[band]=(val,err)
            print(f"{band}=({val},{err})")
        colors = []
        if 'G-V' in columns:
            gv_color =  mags['Gaia'][0] - mags['V'][0]
            ugv_color = mags['Gaia'][1] + mags['V'][1]
            print(f"G-V={gv_color:.2f}+/-{ugv_color:.2f}")
            s_gv_color = (
                 gv_color + np.random.randn(nsamples) * ugv_color
            )  # Monte Carlo samples
            colors.append(s_gv_color)
        if 'Bp-Rp' in columns:
            # Bp-Rp color index
            sources = self.get_gaia_sources(rad_arcsec=30)
            bprp_color =  sources.iloc[0]['bp_rp']
            ubprp_color = 0.01
            print(f"Gaia Bp-Rp={bprp_color:.2f}+/-{ubprp_color:.2f}")
            s_bprp_color = (
                bprp_color + np.random.randn(nsamples) * ubprp_color
            ) 
            colors.append(s_bprp_color)
        if 'J-H' in columns:
            # J-H color index
            jh_color =  mags['J'][0] - mags['H'][0]
            ujh_color = mags['J'][1] + mags['H'][1]
            print(f"J-H={jh_color:.2f}+/-{ujh_color:.2f}")
            s_jh_color = (
                jh_color + np.random.randn(nsamples) * ujh_color
            )  # Monte Carlo samples
            colors.append(s_jh_color)
        if 'H-Ks' in columns:        
            # H-Ks color index
            hk_color =  mags['H'][0] - mags['K'][0]
            uhk_color = mags['H'][1] + mags['K'][1]
            print(f"H-Ks={hk_color:.2f}+/-{uhk_color:.2f}")
            s_hk_color = (
                hk_color + np.random.randn(nsamples) * uhk_color
            )
            colors.append(s_hk_color)
        if 'W1-W2' in columns:
            # W1-W2 color index
            w1w2_color =  mags['WISE 3.4 micron'][0] - mags['WISE 4.6 micron'][0]
            uw1w2_color = mags['WISE 3.4 micron'][1] + mags['WISE 4.6 micron'][1]
            print(f"W1-W2={w1w2_color:.2f}+/-{uw1w2_color:.2f}")
            s_w1w2_color = (
                w1w2_color + np.random.randn(nsamples) * uw1w2_color
            ) 
            colors.append(s_w1w2_color)
        if 'W1-W3' in columns:
            # W1-W3 color index
            w1w3_color =  mags['WISE 3.4 micron'][0] - mags['WISE 12 micron'][0]
            uw1w3_color = mags['WISE 3.4 micron'][1] + mags['WISE 12 micron'][1]
            print(f"W1-W3={w1w3_color:.2f}+/-{uw1w3_color:.2f}")
            s_w1w3_color = (
                w1w3_color + np.random.randn(nsamples) * uw1w3_color
            ) 
            colors.append(s_w1w3_color)
        
        # drop incomplete columns
        cols = columns.copy()
        cols.append("#SpT")
        df = df[cols].dropna()
        # interpolate spec type using given colors 
        interp = NearestNDInterpolator(
            df[columns].values, 
            df['#SpT'].cat.codes.values,
            rescale=False
        )
        samples_code = interp(s_teff, *colors)
        code_spec_mapping = dict(enumerate(df["#SpT"].cat.categories))
        samples_spec_types = pd.Series(samples_code).map(code_spec_mapping)
        # get mode of distribution
        spt = samples_spec_types.mode().values[0]
        # specify dtype
        spec_types = pd.Categorical(samples_spec_types, 
                                    categories=df["#SpT"].cat.categories, 
                                    ordered=True)
        
        if plot:
            d = spec_types.value_counts()
            idx = d>100
            ax = d[idx].plot(kind="barh")
            ax.set_xlabel("Counts")
            print(d[idx].sort_values(ascending=False))
        if return_samples:
            return spt, spec_types
        else:
            return spt
        
    def params_to_dict(self):
        """Convert star parameters to a dictionary."""
        return {
            "rstar": self.rstar,
            "mstar": self.mstar,  # Fixed from self.rstar in original
            "rhostar": self.rhostar,
            "teff": self.teff,
            "logg": self.logg,
            "feh": self.feh,
        }
    

@dataclass
class Planet(Star):
    """
    A class representing an exoplanet, extending the Star class with planet-specific functionality.
    """
    # star_name: str
    # alias: str  # Planet designation number (e.g., "01" for first planet)
    star_params: Dict[str, Tuple[float, float]] = None
    # Default arguments must follow non-default arguments
    source: str = "toi"  # Default source for planetary parameters (overrides Star's default source)
    planet_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize and fetch planet parameters."""
        # super().__init__(star_name=self.star_name, verbose=self.verbose)
        
        if self.star_params is None:
            self.star_params = self.get_star_params()
        self.get_planet_params()
    
    def get_planet_params(self):
        """
        Extract planet parameters from ExoFOP data.
        
        Fetches and processes relevant planetary parameters from the ExoFOP database,
        including transit timing, period, duration, and other orbital characteristics.
        
        Raises:
            ValueError: If parameters cannot be found or processed
        """
        # Ensure data_json is defined by querying ExoFOP if necessary
        if not hasattr(self, "data_json") or self.data_json is None:
            self.data_json = self.query_tfop_data()
        
        # Check if planet_parameters exists in data_json
        if "planet_parameters" not in self.data_json or not self.data_json["planet_parameters"]:
            raise ValueError(f"No planet parameters found for {self.name}")
        
        # Get available sources for planet parameters
        sources = set([p.get("prov") for p in self.data_json["planet_parameters"] if p.get("prov")])
        if not sources:
            raise ValueError(f"No valid sources found in planet parameters for {self.star_name}")
            
        errmsg = f"{self.source} must be in {sources}"
        assert self.source in sources, errmsg
        
        # Parse alias to get index
        try:
            idx = int(self.alias.replace('.', '')) - 1  # Convert to 0-indexed
        except ValueError:
            idx = 0
        
        # Fallback index finding logic
        found = False
        for i, p in enumerate(self.data_json["planet_parameters"]):
            if p.get("prov") == self.source:
                idx = i
                found = True
                break
        
        # Make sure idx is in bounds
        if idx < 0 or idx >= len(self.data_json["planet_parameters"]):
            idx = 0
            
        planet_params = self.data_json["planet_parameters"][idx]
        
        try:
            self.t0 = tuple(
                map(
                    float,
                    (
                        planet_params.get("epoch", np.nan),
                        planet_params.get("epoch_e", 0.1),
                    ),
                )
            )
            self.period = tuple(
                map(
                    float,
                    (
                        planet_params.get("per", np.nan),
                        planet_params.get("per_e", 0.1),
                    ),
                )
            )
            self.tdur = (
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("tdur", 0),
                                planet_params.get("dur_e", 0),
                            ),
                        )
                    )
                )
                / 24
            )
            self.rprs = np.sqrt(
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("dep_p", 0),
                                planet_params.get("dep_p_e", 0),
                            ),
                        )
                    )
                )
                / 1e6
            )
            self.imp = tuple(
                map(
                    float,
                    (
                        float(0 if planet_params.get("imp", 0) == "" else planet_params.get("imp", 0)),
                        float(0.1 if planet_params.get("imp_e", 0.1) == "" else planet_params.get("imp_e", 0.1)),
                    ),
                )
            )
            print(f"t0={self.t0} BJD\nP={self.period} d\nRp/Rs={self.rprs}")
            
            rhostar = self.star_params["rhostar"]
            self.a_Rs = (
                (rhostar[0] / 0.01342 * self.period[0] ** 2) ** (1 / 3),
                1
                / 3
                * (1 / 0.01342 * self.period[0] ** 2) ** (1 / 3)
                * rhostar[0] ** (-2 / 3)
                * rhostar[1],
            )
            
            # Update planet_params dictionary with calculated values
            self.planet_params = self.params_to_dict()
            
        except Exception as e:
            print(f"Error processing planet parameters: {e}")
            raise ValueError(f"Check exofop: {self.exofop_url}")
    
    def params_to_dict(self):
        """
        Convert planet parameters to a dictionary.
        
        Returns:
            Dict[str, Tuple[float, float]]: Dictionary of planet parameters with uncertainties
        """
        return {
            "t0": self.t0,
            "period": self.period,
            "tdur": self.tdur,
            "imp": self.imp,
            "rprs": self.rprs,
            "a_Rs": self.a_Rs,
        }
    
    def __repr__(self):
        """Return string representation of the Target object."""
        coord_str = self.target_coord.to_string("decimal").replace(" ", ", ")
        return (f"{self.__class__.__name__}("
                f"star_name='{getattr(self, 'star_name', 'Unnamed')}' "
                f"{self.alias}, "
                f"gaiaid={self.gaiaid}, "
                f"ra={self.ra}, dec={self.dec}, "
                f"coord=({coord_str}))"
               )
    
class CatalogDownloader:
    """download tables from vizier
    Attributes
    ----------
    tables : astroquery.utils.TableList
        collection of astropy.table.Table downloaded from vizier
    """

    def __init__(
        self, catalog_name, catalog_type="cluster", data_loc=DATA_PATH, verbose=True, clobber=False
        ):
        self.catalog_name = catalog_name
        self.catalog_type = catalog_type
        if catalog_type.lower() in ["prot", "rotation"]:
            self.catalog_dict = VIZIER_KEYS_PROT_CATALOG
        elif catalog_type.lower() in ["liew", "lithium"]:
            self.catalog_dict = VIZIER_KEYS_LiEW_CATALOG
        elif catalog_type.lower() in ["rv"]:
            self.catalog_dict = VIZIER_KEYS_RV_CATALOG
        elif catalog_type.lower()=="cluster":
            self.catalog_dict = VIZIER_KEYS_CLUSTER_CATALOG
        self.verbose = verbose
        self.clobber = clobber
        if not Path(data_loc).exists():
            Path(data_loc).mkdir()
        self.data_loc = Path(data_loc, self.catalog_name)
        self.tables = None
        self.vizier_url = self.get_vizier_url()

    def get_tables_from_vizier(self, row_limit=50, save=False, clobber=None):
        """
        row_limit-1 to download all rows
        TODO: Load data when save=True
        """
        clobber = self.clobber if clobber is None else clobber
        if row_limit == -1:
            msg = "Downloading all tables in "
        else:
            msg = f"Downloading the first {row_limit} rows of each table "
        try:
            msg += f"{self.catalog_dict[self.catalog_name]} from vizier."
            if self.verbose:
                logger.info(msg)
        except:
            errmsg = f"'{self.catalog_name}' not in {list(self.catalog_dict.keys())}.\n"
            errmsg+=f"\nUsing catalog_type={self.catalog_type}."
            raise ValueError(errmsg)

        # set row limit
        Vizier.ROW_LIMIT = row_limit

        if self.catalog_dict[self.catalog_name]=="local":
            raise ValueError("Read the csv data locally.")
        else:
            tables = Vizier.get_catalogs(self.catalog_dict[self.catalog_name])
            errmsg = "No data returned from Vizier."
            assert tables is not None, errmsg
            self.tables = tables
    
            if self.verbose:
                pprint({k: tables[k]._meta["description"] for k in tables.keys()})

            if save:
                self.save_tables(clobber=clobber)
        return tables

    def save_tables(self, clobber=None):
        errmsg = "No tables to save"
        assert self.tables is not None, errmsg
        clobber = self.clobber if clobber is None else clobber

        if not self.data_loc.exists():
            self.data_loc.mkdir()

        for n, table in enumerate(self.tables):
            fp = Path(self.data_loc, f"{self.catalog_name}_tab{n}.txt")
            if not fp.exists() or clobber:
                table.write(fp, format="ascii")
                if self.verbose:
                    logger.info(f"Saved: {fp}")
            else:
                logger.info("Set clobber=True to overwrite.")

    def get_vizier_url(self, catalog_name=None):
        if catalog_name is None:
            catalog_name = self.catalog_name
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        vizier_key = self.catalog_dict[catalog_name]
        url = base_url + vizier_key
        if self.verbose:
            logger.info(f"Data url: {url}")
        return url

    def __repr__(self):
        """Override to print a readable string representation of class"""
        included_args = ["catalog_name", "cluster_name"]
        args = []
        for key in self.__dict__.keys():
            val = self.__dict__.get(key)
            if key in included_args:
                if val is not None:
                    args.append(f"{key}={val}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"
    
def load_plot_params():
    """ Load in plt.rcParams and set (based on paper defaults).
    """
    params = Table.read('../rcParams.txt', format='csv')
    for i, name in enumerate(params['name']):
        try:
            pl.rcParams[name] = float(params['value'][i])
        except:
            pl.rcParams[name] = params['value'][i]
    return params
    
def get_tois(
    clobber=False,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False
):
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        TOI table as dataframe
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    fp = Path(outdir, "TOIs.csv")
    if not Path(outdir).exists():
        Path(outdir).makedir()

    if not fp.exists() or clobber:
        df = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = f"Downloading {dl_link}\n"

        #add coordinates in deg
        coords = SkyCoord(df[['RA','Dec']].values, unit=('hourangle','degree'))
        df['ra_deg'] = coords.ra.deg
        df['dec_deg'] = coords.dec.deg
        #add previously querried Gaia DR3 ids
        tois = pd.read_csv(f'{outdir}/TOIs_with_Gaiaid.csv')
        df = pd.merge(df, tois, on='TOI', how='outer')
        df.to_csv(fp, index=False)
        logger.info("Saved: ", fp)
    else:
        df = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    assert len(df) > 1000, f"{fp} likely has been overwritten!"

    # remove False Positives
    if remove_FP:
        df = df[df["TFOPWG Disposition"] != "FP"]
        msg += "TOIs with TFPWG disposition==FP are removed.\n"
    if remove_known_planets:
        planet_keys = [
            "HD",
            "GJ",
            "LHS",
            "XO",
            "Pi Men", 
            "WASP",
            "SWASP",
            "HAT",
            "HATS",
            "KELT",
            "TrES",
            "QATAR",
            "CoRoT",
            "K2",  # , "EPIC"
            "Kepler",  # "KOI"
        ]
        keys = []
        for key in planet_keys:
            idx = ~np.array(
                df["Comments"].str.contains(key).tolist(), dtype=bool
            )
            df = df[idx]
            if idx.sum() > 0:
                keys.append(key)
        msg += f"{keys} planets are removed.\n"
    msg += f"Saved: {fp}\n"
    if verbose:
        logger.info(msg)
    return df.sort_values("TOI").reset_index(drop=True)

def get_ctois(clobber=True, outdir=DATA_PATH, verbose=False, remove_FP=True):
    """Download Community TOI list from exofop/TESS.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        CTOI table as dataframe

    See interface: https://exofop.ipac.caltech.edu/tess/view_ctoi.php
    See also: https://exofop.ipac.caltech.edu/tess/ctoi_help.php
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
    fp = Path(outdir, "CTOIs.csv")
    if not Path(outdir).exists():
        Path(outdir).makedir()

    if not fp.exists() or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = "Downloading {}\n".format(dl_link)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = "Loaded: {}\n".format(fp)
    d.to_csv(fp, index=False)

    # remove False Positives
    if remove_FP:
        d = d[d["User Disposition"] != "FP"]
        msg += "CTOIs with user disposition==FP are removed.\n"
    msg += "Saved: {}\n".format(fp)
    if verbose:
        logger.info(msg)
    return d.sort_values("CTOI")

def get_nexsci_data(table_name="ps", method="Transit", outdir=DATA_PATH, clobber=False):
    """
    ps: self-consistent set of parameters
    pscomppars: a more complete, though not necessarily self-consistent set of parameters
    See also 
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    except Exception as e:
        logger.info(e)
    assert table_name in ["ps", "pscomppars"]
    url = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
    logger.info("Column definitions: ", url)
    fp = Path(outdir, f'nexsci_{table_name}.csv')        
    if not fp.exists() or clobber:
        #pstable combines data from the Confirmed Planets and Extended Planet Parameters tables
        logger.info(f"Downloading NExSci {table_name} table...")
        tab = NasaExoplanetArchive.query_criteria(table=table_name, 
                                                  where=f"discoverymethod like '{method}'")
        df = tab.to_pandas()
        df.to_csv(fp, index=False)
        logger.info("Saved: ", fp)
    else:
        df = pd.read_csv(fp)
        logger.info("Loaded: ", fp)
    return df

def get_absolute_gmag(gmag, distance, a_g):
    """
    gmag : float
        apparent G band magnitude
    distance : float
        distance in pc
    a_g : float
        extinction in the G-band

    Returns
    -------
    gmag : float
        Gaia magnitude
    """
    assert (gmag is not None) & (str(gmag) != "nan"), "gma is nan"
    assert (distance is not None) & (str(distance) != "nan"), "distance is nan"
    assert (a_g is not None) & (str(a_g) != "nan"), "a_g is nan"
    Gmag = gmag - 5.0 * np.log10(distance) + 5.0 - a_g
    return Gmag

def plot_cmd(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    log_age=None,
    feh=0.0,
    eep_limits=(202, 454),
    target_color="r",
    xaxis="bp_rp0",
    yaxis="abs_gmag",
    color="radius_val",
    figsize=(8, 8),
    estimate_color=False,
    cmap="viridis",
    add_text=True,
    ax=None,
):
    """Plot color-magnitude diagram using absolute G magnitude and dereddened Bp-Rp from Gaia photometry

    Parameters
    ----------
    df : pd.DataFrame
        cluster member properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    estimate_color : bool
        estimate absolute/dereddened color from estimated excess
    log_age : float
        isochrone age (default=None)
    feh : float
        isochrone metallicity
    eep_limits : tuple
        maximum eep (default=(202,454): (ZAMS,TAMS))

    Returns
    -------
    ax : axis
    """
    assert len(df) > 0, "df is empty"
    errmsg = f"color={color} not in {df.columns}"
    assert color in df.columns, errmsg

    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if df.columns.isin([xaxis, yaxis]).sum() == 2:
        x = df[xaxis]
        y = df[yaxis]
        ax.set_xlabel(xaxis, fontsize=16)
        ax.set_ylabel(yaxis, fontsize=16)
    else:
        #compute Gmag and BP-RP
        if "distance" not in df.columns:
            df["parallax"] = df["parallax"].astype(float)
            idx = ~np.isnan(df["parallax"]) & (df["parallax"] > 0)
            df = df[idx]
            if sum(~idx) > 0:
                logger.info(f"{sum(~idx)} removed NaN or negative parallaxes")
    
            df["distance"] = Distance(parallax=df["parallax"].values * u.mas).pc
            
        # compute absolute Gmag
        df["abs_gmag"] = get_absolute_gmag(
            df["phot_g_mean_mag"], df["distance"], df["a_g_val"]
        )
        # compute intrinsic color index
        if estimate_color:
            df[xaxis] = get_absolute_color_index(
                df["a_g_val"], df["phot_bp_mean_mag"], df["phot_rp_mean_mag"]
            )
        else:
            df[xaxis] = df["bp_rp"] - df["e_bp_min_rp_val"]
        ax.set_xlabel(r"$G_{BP} - G_{RP}$ [mag]", fontsize=16)
        ax.set_ylabel(r"$G$ [mag]", fontsize=16)

    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            x, y = df.loc[idx, xaxis], df.loc[idx, yaxis]
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            # compute absolute Gmag
            df_target["abs_gmag"] = get_absolute_gmag(
                df_target["phot_g_mean_mag"],
                df_target["distance"],
                df_target["a_g_val"],
            )
            # compute intrinsic color index
            if estimate_color:
                df_target[xaxis] = get_absolute_color_index(
                    df_target["a_g_val"],
                    df_target["phot_bp_mean_mag"],
                    df_target["phot_rp_mean_mag"],
                )
            else:
                df_target[xaxis] = (
                    df_target["bp_rp"] - df_target["e_bp_min_rp_val"]
                )
            x, y = df_target[xaxis], df_target[yaxis]
        if target_label is not None:
            ax.legend(loc="best")
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms="25",
            label=target_label,
            # cmap=cmap,
            zorder=10,
        )
    if log_age is not None:
        # plot isochrones
        try:
            from isochrones import get_ichrone
    
            iso_grid = get_ichrone("mist")
            assert len(eep_limits) == 2, "eep_limits=(min,max)"
            iso_df = iso_grid.isochrone(log_age, feh)
            idx = (iso_df.eep > eep_limits[0]) & (iso_df.eep < eep_limits[1])
            G = iso_df.G_mag[idx]
            #FIXME: check if xaxis is bp-rp
            BP_RP = iso_df.BP_mag[idx] - iso_df.RP_mag[idx]
            label = f"log(t)={log_age:.2f}\nfeh={feh:.2f}"
            ax.plot(BP_RP, G, c="k", label=label)
            ax.legend(title="MIST isochrones")
        except Exception as e:
            print("Could not plot isochrones. You may need to install the 'isochrones' package:")
            print("    pip install isochrones")
            print(f"Exception: {e}")
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    if color == "radius_val":
        rstar = np.log10(df[color].astype(float))
        c = ax.scatter(df[xaxis], df[yaxis], marker=".", c=rstar, cmap=cmap)
        ax.figure.colorbar(c, ax=ax, label=r"$\log$(R/R$_{\odot}$)")
    else:
        c = ax.scatter(df[xaxis], df[yaxis], c=df[color], marker=".", cmap=cmap)
        ax.figure.colorbar(c, ax=ax, label=color)

    ax.set_xlim(df[xaxis].min(), df[xaxis].max())
    ax.invert_yaxis()
    if add_text:
        text = len(df[[xaxis, yaxis]].dropna())
        ax.text(0.8, 0.8, f"n={text}", fontsize=14, transform=ax.transAxes)
    return ax

def plot_hrd(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    target_color="r",
    log_age=None,
    feh=0.0,
    eep_limits=(202, 454),
    figsize=(8, 8),
    yaxis="lum_val",
    xaxis="teff_val",
    color="radius_val",
    cmap="viridis",
    annotate_Sun=False,
    add_text=True,
    ax=None,
):
    """Plot HR diagram using luminosity and Teff
    and optionally MIST isochrones if log_age is given

    Parameters
    ----------
    df : pd.DataFrame
        cluster memeber properties
    match_id : bool
        checks if target gaiaid in df
    df_target : pd.Series
        info of target
    log_age : float
        isochrone age (default=None)
    feh : float
        isochrone metallicity
    eep_limits : tuple
        maximum eep (default=(202,454): (ZAMS,TAMS))
    xaxis, yaxis : str
        parameter to plot

    Returns
    -------
    ax : axis
    """
    assert len(df) > 0, "df is empty"
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            x, y = df.loc[idx, xaxis], df.loc[idx, yaxis]
        else:
            assert df_target is not None, "provide df_target"
            df_target["distance"] = Distance(
                parallax=df_target["parallax"] * u.mas
            ).pc
            x, y = df_target[xaxis], df_target[yaxis]
        if target_label is not None:
            ax.legend(loc="best")
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms="25",
            label=target_label,
        )
    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    # luminosity can be computed from abs mag; note Mag_sun = 4.85
    # df["abs_gmag"] = get_absolute_gmag(
    #     df["phot_g_mean_mag"], df["distance"], df["a_g_val"])
    # df["lum_val"] = 10**(0.4*(4.85-df["abs_gmag"])
    if color == "radius_val":
        rstar = np.log10(df[color].astype(float))
        c = ax.scatter(df[xaxis], df[yaxis], marker=".", c=rstar, cmap=cmap)
        ax.figure.colorbar(c, ax=ax, label=r"$\log$(R/R$_{\odot}$)")
    else:
        ax.scatter(df[xaxis], df[yaxis], c=df[color], marker=".")

    if annotate_Sun:
        assert (yaxis == "lum_val") & (xaxis == "teff_val")
        ax.plot(5700, 1, marker=r"$\odot$", c="r", ms="15", label="Sun")
    if log_age is not None:
        # plot isochrones
        try:
            from isochrones.mist import MISTIsochroneGrid

            iso_grid = MISTIsochroneGrid()
            # from isochrones import get_ichrone
            # iso_grid = get_ichrone('mist').model_grid
        except Exception:
            errmsg = "pip install isochrones"
            raise ModuleNotFoundError(errmsg)
        assert len(eep_limits) == 2, "eep_limits=(min,max)"
        # check log_age
        ages = iso_grid.df.index.get_level_values(0)
        nearest_log_age = min(ages, key=lambda x: abs(x - log_age))
        errmsg = f"log_age={log_age} not in:\n{[round(x,2) for x in ages.unique().tolist()]}"
        # assert ages.isin([log_age]).any(), errmsg
        assert abs(nearest_log_age - log_age) < 0.1, errmsg
        # check feh
        fehs = iso_grid.df.index.get_level_values(1)
        nearest_feh = min(fehs, key=lambda x: abs(x - feh))
        errmsg = f"feh={feh} not in:\n{[round(x,2) for x in fehs.unique().tolist()]}"
        # assert fehs.isin([feh]).any(), errmsg
        assert abs(nearest_feh - feh) < 0.1, errmsg
        # get isochrone
        iso_df = iso_grid.df.loc[nearest_log_age, nearest_feh]
        iso_df["L"] = iso_df["logL"].apply(lambda x: 10**x)
        iso_df["Teff"] = iso_df["logTeff"].apply(lambda x: 10**x)
        label = f"log(t)={log_age:.2f}\nfeh={feh:.2f}"
        # limit eep
        idx = (iso_df.eep > eep_limits[0]) & (iso_df.eep < eep_limits[1])
        iso_df[idx].plot(x="Teff", y="L", c="k", ax=ax, label=label)
        ax.set_xlim(df[xaxis].min() - 100, df[xaxis].max() + 100)
        ax.legend(title="MIST isochrones")

    ax.set_ylabel(r"$L/L_{\odot}$", fontsize=16)
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_{\rm{eff}}$/K", fontsize=16)
    if add_text:
        text = len(df[[xaxis, yaxis]].dropna())
        ax.text(
            0.8, 0.8, f"nstars={text}", fontsize=14, transform=ax.transAxes
        )
    return ax

def plot_rdp_pmrv(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    target_color="r",
    color="teff_val",
    marker="o",
    figsize=(10, 10),
    cmap="viridis",
):
    """
    Plot ICRS position and proper motions in 2D scatter plots,
    and parallax and radial velocity in kernel density

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, rv columns
    target_gaiaid : int
        target gaia DR2 id

    Returns
    -------
    fig : matplotlib.pyplot.Figure
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    n = 1
    x, y = "ra", "dec"
    # _ = df.plot.scatter(x=x, y=y, c=color, marker=marker, ax=ax[n], cmap=cmap)
    errmsg = f"color={color} not in {df.columns}"
    assert color in df.columns, errmsg
    c = df[color] if color is not None else None
    cbar = ax[n].scatter(df[x], df[y], c=c, marker=marker, cmap=cmap)
    if color is not None:
        fig.colorbar(cbar, ax=ax[n], label=color)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
                label=target_label,
            )
    ax[n].set_xlabel("R.A. [deg]")
    ax[n].set_ylabel("Dec. [deg]")
    text = len(df[["ra", "dec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    if target_label is not None:
        ax[n].legend(loc="best")
    n = 0
    par = "parallax"
    df[par].plot.kde(ax=ax[n])
    ax[n].plot(df[par], np.full_like(df[par], -0.01), "|k", markeredgewidth=1)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].axvline(
                df.loc[idx, par].values[0],
                0,
                1,
                c="k",
                ls="--",
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].axvline(
                df_target[par], 0, 1, c="k", ls="--", label=target_label
            )

        if target_label is not None:
            ax[n].legend(loc="best")
    ax[n].set_xlabel("Parallax [mas]")
    text = len(df[par].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 3
    x, y = "pmra", "pmdec"
    # _ = df.plot.scatter(x=x, y=y, c=c, marker=marker, ax=ax[n], cmap=cmap)
    c = df[color] if color is not None else None
    cbar = ax[n].scatter(df[x], df[y], c=c, marker=marker, cmap=cmap)
    if (color is not None) & (n == 3):
        # show last colorbar only
        fig.colorbar(cbar, ax=ax[n], label=color)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms="25",
            )
    ax[n].set_xlabel("PM R.A. [deg]")
    ax[n].set_ylabel("PM Dec. [deg]")
    text = len(df[["pmra", "pmdec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 2
    par = "radial_velocity"
    errmsg = f"{par} is not available in {df.columns}"
    assert df.columns.isin([par]), errmsg
    try:
        df[par].plot.kde(ax=ax[n])
        ax[n].plot(
            df[par], np.full_like(df[par], -0.01), "|k", markeredgewidth=1
        )
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].axvline(
                    df.loc[idx, par].values[0],
                    0,
                    1,
                    c="k",
                    ls="--",
                    label=target_label,
                )
            else:
                ax[n].axvline(
                    df_target[par], 0, 1, c="k", ls="--", label=target_label
                )
        ax[n].set_xlabel("RV [km/s]")
        text = len(df[par].dropna())
        ax[n].text(
            0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        )
    except Exception as e:
        ax[n].clear()
        logger.error("Error: ", e)
        npar = len(df[par].dropna())
        if npar < 10:
            errmsg = f"Cluster members have only {npar} {par} measurements."
            logger.error("Error: ", errmsg)
            # raise ValueError(errmsg)
    return fig

def get_transformed_coord(df, frame="galactocentric", verbose=True):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        catalog with complete kinematics parameters
    frame : str
        frame conversion

    Returns
    -------
    df : pandas.DataFrame
        catalog with transformed coordinates appended in columns

    Note
    ----
    Assumes galactic center distance distance of 8.1 kpc based on the GRAVITY
    collaboration, and a solar height of z_sun=0 pc.
    See also:
    http://learn.astropy.org/rst-tutorials/gaia-galactic-orbits.html?highlight=filtertutorials
    """
    assert len(df) > 0, "df is empty"
    if any(df["parallax"] < 0):
        # retain non-negative parallaxes including nan
        df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
        if verbose:
            logger.warning("Some parallaxes are negative!")
            print("These are removed for the meantime.")
            print("For proper treatment, see:")
            print("https://arxiv.org/pdf/1804.09366.pdf\n")
    errmsg = f"radial_velocity is not in {df.columns}"
    assert any(df.columns.isin(["radial_velocity"])), errmsg
    df2 = df.copy()
    icrs = SkyCoord(
        ra=df2["ra"].values * u.deg,
        dec=df2["dec"].values * u.deg,
        distance=Distance(parallax=df2["parallax"].values * u.mas),
        radial_velocity=df2["radial_velocity"].values * u.km / u.s,
        pm_ra_cosdec=df2["pmra"].values * u.mas / u.yr,
        pm_dec=df2["pmdec"].values * u.mas / u.yr,
        frame="fk5",
        equinox="J2000.0",
    )
    # transform to galactocentric frame
    if frame == "galactocentric":
        # xyz = icrs.transform_to(
        #     Galactocentric(z_sun=0 * u.pc, galcen_distance=8.1 * u.kpc)
        # )
        xyz = icrs.galactocentric
        df2["X"] = xyz.x.copy()
        df2["Y"] = xyz.y.copy()
        df2["Z"] = xyz.z.copy()
        df2["U"] = xyz.v_x.copy()
        df2["V"] = xyz.v_y.copy()
        df2["W"] = xyz.v_z.copy()

    elif frame == "galactic":
        # transform to galactic frame
        # gal = icrs.transform_to("galactic")
        gal = icrs.galactic
        df2["gal_l"] = gal.l.deg.copy()
        df2["gal_b"] = gal.b.deg.copy()
        df2["gal_pm_b"] = gal.pm_b.copy()
        df2["gal_pm_l_cosb"] = gal.pm_l_cosb.copy()
    else:
        raise ValueError(f"frame={frame} is unavailable")
    return df2

def plot_xyz_uvw(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_color="r",
    color="teff_val",
    marker="o",
    verbose=True,
    figsize=(12, 8),
    cmap="viridis",
):
    """
    Plot 3D position in galactocentric (xyz) frame
    and proper motion with radial velocity in galactic cartesian velocities
    (UVW) frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec, parallax, pmra, pmdec, rv columns
    target_gaiaid : int
        target gaia DR2 id
    df_target : pandas.Series
        target's gaia parameters

    Returns
    -------
    fig : matplotlib.pyplot.Figure

    Note: U is positive towards the direction of the Galactic center (GC);
    V is positive for a star with the same rotational direction as the Sun going around the galaxy,
    with 0 at the same rotation as sources at the Sun’s distance,
    and W positive towards the north Galactic pole

    U,V,W can be converted to Local Standard of Rest (LSR) by subtracting V = 238 km/s,
    the adopted rotation velocity at the position of the Sun from Marchetti et al. (2018).

    See also https://arxiv.org/pdf/1707.00697.pdf which estimates Sun's
    (U,V,W) = (9.03, 255.26, 7.001)

    See also https://arxiv.org/pdf/1804.10607.pdf for modeling Gaia DR2 in 6D
    """
    assert len(df) > 0, "df is empty"
    fig, axs = pl.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax = axs.flatten()

    errmsg = f"color={color} not in {df.columns}"
    assert color in df.columns, errmsg
    if not np.all(df.columns.isin("X Y Z U V W".split())):
        df = get_transformed_coord(df, frame="galactocentric", verbose=verbose)
    if df_target is not None:
        df_target = get_transformed_coord(
            pd.DataFrame(df_target).T, frame="galactocentric"
        )
    n = 0
    for (i, j) in itertools.combinations(["X", "Y", "Z"], r=2):
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
            else:
                assert df_target is not None, "provide df_target"
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
        # _ = df.plot.scatter(x=i, y=j, c=color, marker=marker, ax=ax[n])
        c = df[color] if color is not None else None
        cbar = ax[n].scatter(df[i], df[j], c=c, marker=marker, cmap=cmap)
        # if color is not None:
        #     fig.colorbar(cbar, ax=ax[n], label=color)
        ax[n].set_xlabel(i + " [pc]")
        ax[n].set_ylabel(j + " [pc]")
        #import pdb; pdb.set_trace()
        #text = df[[i, j]].dropna().shape[0]
        #ax[n].text(
        #    0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        #)
        n += 1

    n = 3
    for (i, j) in itertools.combinations(["U", "V", "W"], r=2):
        if target_gaiaid is not None:
            idx = df.source_id.astype(int).isin([target_gaiaid])
            if match_id:
                errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
                assert sum(idx) > 0, errmsg
                ax[n].plot(
                    df.loc[idx, i],
                    df.loc[idx, j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
            else:
                ax[n].plot(
                    df_target[i],
                    df_target[j],
                    marker=r"$\star$",
                    c=target_color,
                    ms="25",
                )
        # _ = df.plot.scatter(x=i, y=j, c=color, marker=marker, ax=ax[n], cmap=cmap)
        c = df[color] if color is not None else None
        cbar = ax[n].scatter(df[i], df[j], c=c, marker=marker, cmap=cmap)
        if (color is not None) and (n == 5):
            # show last colorbar only only
            fig.colorbar(cbar, ax=ax[n], label=color)
        ax[n].set_xlabel(i + " [km/s]")
        ax[n].set_ylabel(j + " [km/s]")
        #text = df[[i, j]].dropna().shape[0]
        #ax[n].text(
        #    0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes
        #)
        n += 1

    return fig


def plot_xyz_3d(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_color="r",
    color="teff_val",
    marker="o",
    xlim=None,
    ylim=None,
    zlim=None,
    figsize=(8, 5),
    cmap="viridis",
):
    """plot 3-d position in galactocentric frame

    Parameters
    ----------
    df : pandas.DataFrame
        contains ra, dec & parallax columns
    target_gaiaid : int
        target gaia DR2 id
    xlim,ylim,zlim : tuple
        lower and upper bounds

    Returns
    -------
    fig : matplotlib.pyplot.Figure
    """
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 120)

    errmsg = f"color={color} not in {df.columns}"
    assert color in df.columns, errmsg
    if "distance" not in df.columns:
        df["distance"] = Distance(parallax=df.parallax.values * u.mas).pc

    coords = SkyCoord(
        ra=df.ra.values * u.deg,
        dec=df.dec.values * u.deg,
        distance=df.distance.values * u.pc,
    )
    xyz = coords.galactocentric
    df["x"] = xyz.x.copy()
    df["y"] = xyz.y.copy()
    df["z"] = xyz.z.copy()

    idx1 = np.zeros_like(df.x, dtype=bool)
    if xlim:
        assert isinstance(xlim, tuple)
        idx1 = (df.x > xlim[0]) & (df.x < xlim[1])
    idx2 = np.zeros_like(df.y, dtype=bool)
    if ylim:
        assert isinstance(ylim, tuple)
        idx2 = (df.y > ylim[0]) & (df.y < ylim[1])
    idx3 = np.zeros_like(df.z, dtype=bool)
    if zlim:
        assert isinstance(zlim, tuple)
        idx3 = (df.z > zlim[0]) & (df.z < zlim[1])
    idx = idx1 | idx2 | idx3
    c = df.loc[idx, color]
    cbar = ax.scatter(
        xs=df[idx].x,
        ys=df[idx].y,
        zs=df[idx].z,
        c=c,
        marker=marker,
        cmap=cmap,
        alpha=0.5,
    )
    if color is not None:
        fig.colorbar(cbar, ax=ax, label=color)
    if target_gaiaid is not None:
        idx = df.source_id == target_gaiaid
        ax.scatter(
            xs=df[idx].x,
            ys=df[idx].y,
            zs=df[idx].z,
            marker=r"$\star$",
            c=target_color,
            s=300,
            zorder=10,
        )
    pl.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")
    return fig


def get_mamajek_table(clobber=False, verbose=True, data_loc=DATA_PATH):
    """
    """
    fp = Path(data_loc, "mamajek_table.csv")
    if not fp.exists() or clobber:
        url = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
        table = Table.read(url,
                           format='ascii',
                           comment='',
                           header_start=22,
                           data_start=23,
                           data_end=141,
                           delimiter=' ',
                           fill_values=('...', np.nan),
                           fast_reader=False,
                           guess=False
                          )
        tab = table.to_pandas()
        df.to_csv(fp, index=False)
        logger.info(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            logger.info(f"Loaded: {fp}")
    return df


def interpolate_mamajek_table(
        df,
        input_col="BP-RP",
        output_col="Teff",
        nsamples=int(1e4),
        return_samples=False,
        plot=False,
        clobber=False,
        verbose=True
    ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.
        c.f. self.get_vizier_table_param("SpT")

        Parameters
        ----------
        columns : list
            column names of input parameters
        nsamples : int
            number of Monte Carlo samples (default=1e4)
        clobber : bool (default=False)
            re-download Mamajek table

        Returns
        -------
        spt: str
            interpolated spectral type

        Notes:
        It may be good to check which color index yields most accurate result

        Check sptype from self.query_simbad()
        """
        df = get_mamajek_table(clobber=clobber, verbose=verbose)
        
        # B-V color index
        bprp_color = df["BPRP"]
        ubprp_color = bprp_color.std()
        s_bprp_color = (
            bprp_color + np.random.randn(nsamples) * ubprp_color
        )  # Monte Carlo samples

        # Interpolate
        interp = NearestNDInterpolator(
            df[input_col].values, df[output_col].values, rescale=False
        )
        samples = interp(s_bprp_color)
        # encode category
        spt_cats = pd.Series(samples, dtype="category")  # .cat.codes
        spt = spt_cats.mode().values[0]
        if plot:
            nbins = np.unique(samples)
            pl.hist(samples, bins=nbins)
        if return_samples:
            return spt, samples
        else:
            return spt


def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))


def query_tfop_data(target_name: str) -> dict:
    base_url = "https://exofop.ipac.caltech.edu/tess"
    url = f"{base_url}/target.php?id={target_name.replace(' ','')}&json"
    response = urlopen(url)
    assert response.code == 200, "Failed to get data from ExoFOP-TESS"
    try:
        data_json = json.loads(response.read())
        return data_json
    except Exception as e:
        logger.error(e)
        raise ValueError(f"No TIC data found for {target_name}")


def get_params_from_tfop(tfop_data, name="planet_parameters", idx=None):
    params_dict = tfop_data.get(name)
    if idx is None:
        key = "pdate" if name == "planet_parameters" else "sdate"
        # get the latest parameter based on upload date
        dates = []
        for d in params_dict:
            t = d.get(key)
            dates.append(t)
        df = pd.DataFrame({"date": dates})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        idx = df["date"].idxmax()
    return params_dict[idx]


def get_tic_id(target_name: str) -> int:
    return int(query_tfop_data(target_name)["basic_info"]["tic_id"])


def get_nexsci_data(table_name="ps", clobber=False):
    """
    ps: self-consistent set of parameters
    pscomppars: a more complete, though not necessarily self-consistent set of parameters
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
    logger.info("Column definitions: ", url)
    fp = Path("../data/",f"nexsci_{table_name}.csv")
    if not fp.exists() or clobber:
        logger.info(f"Downloading NExSci {table_name} table...")
        nexsci_tab = NasaExoplanetArchive.query_criteria(table=table_name, where="discoverymethod like 'Transit'")
        df_nexsci = nexsci_tab.to_pandas()
        df_nexsci.to_csv(fp, index=False)
        logger.info("Saved: ", fp)
    else:
        df_nexsci = pd.read_csv(fp)
        logger.info("Loaded: ", fp)
    return df_nexsci