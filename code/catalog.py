import os
from re import M
import itertools
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

from pathlib import Path
from pprint import pprint

import astropy.units as u
from astropy.coordinates import SkyCoord, Distance, Galactocentric
from astropy.table import Table
from astroquery.vizier import Vizier

from util_funcs import flatten_list

DATA_PATH = '../data/'

def get_tois(
    clobber=False,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False,
    add_FPP=False,
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
    fp = os.path.join(outdir, "TOIs.csv")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(fp) or clobber:
        d = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = f"Downloading {dl_link}\n"
        if add_FPP:
            fp2 = os.path.join(outdir, "Giacalone2020/tab4.txt")
            classified = ascii.read(fp2).to_pandas()
            fp3 = os.path.join(outdir, "Giacalone2020/tab5.txt")
            unclassified = ascii.read(fp3).to_pandas()
            fpp = pd.concat(
                [
                    classified[["TOI", "FPP-2m", "FPP-30m"]],
                    unclassified[["TOI", "FPP"]],
                ],
                sort=True,
            )
            d = pd.merge(d, fpp, how="outer").drop_duplicates()
        d.to_csv(fp, index=False)
    else:
        d = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    assert len(d) > 1000, f"{fp} likely has been overwritten!"

    # remove False Positives
    if remove_FP:
        d = d[d["TFOPWG Disposition"] != "FP"]
        msg += "TOIs with TFPWG disposition==FP are removed.\n"
    if remove_known_planets:
        planet_keys = [
            "HD",
            "GJ",
            "LHS",
            "XO",
            "Pi Men" "WASP",
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
                d["Comments"].str.contains(key).tolist(), dtype=bool
            )
            d = d[idx]
            if idx.sum() > 0:
                keys.append(key)
        msg += f"{keys} planets are removed.\n"
    msg += f"Saved: {fp}\n"
    if verbose:
        print(msg)
    return d.sort_values("TOI")


# See more keywords here: http://vizier.cds.unistra.fr/vizier/vizHelp/cats/U.htx
VIZIER_KEYS_LiEW_CATALOG = {
    "Randich2001_IC2602": "J/A+A/372/862", # 	Lithium abundances in IC 2602 and IC 2391
    "Barrado2016_Pleiades": "J/A+A/596/A113", #
    "Cummings2017_HyadesPraesepe": "J/AJ/153/128",
    "Manzi2008_IC4665": "J/A+A/479/141", #Iz photometry, RV and EW(Li) in IC 4665 (Manzi+, 2008) 
    "Prisinzano2007_NGC3960": "J/A+A/475/539", # BV photometry and Li abundances in NGC3960
    "Stanford-Moore2020": "J/ApJ/898/27",
    "Bouvier2018_Pleiades": "J/A+A/613/A63",
    "Franciosini2022": "J/A%2bA/659/A85", #Membership and lithium of 10-100Myr clusters
    "Gutierrez2020": "J/A+A/643/A71", #Members for 20 open clusters (Gutierrez Albarran+, 2020)
    "Magrini2021_ALi": "J/A%2bA/651/A84", #Li abundance and mixing in giant stars
    "Deliyannis2019": "J/AJ/158/163", #Li abundance values for stars in NGC 6819
}
VIZIER_KEYS_PROT_CATALOG = {
    # See table1: https://arxiv.org/pdf/1905.10588.pdf
    "Curtis2019_Rup147": "J/ApJ/904/140", #Pleiades, Praesepe, NGC 6811, NGC 752, NGC 6819, and Ruprecht 147 
    "Curtis2019_PisEri": "J/AJ/158/77",  # 250Gyr
    "Curtis2019_NGC6811": "J/ApJ/879/49",  # 1Gyr
    "Fritzewski2021_NGC3532": "J/A+A/652/A60",    #300 Myr
    "Feinstein2020_NYMG": "See data/Feinstein2020_NYMG.txt",
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
    "Douglas2019_Praesepe": "2019ApJ...879..100D",
    "Fang2020_PleiadesPraesepeHyades": "2020MNRAS.495.2949F",
    "Gillen2020_BlancoI": "2020MNRAS.492.1008G",
    "Canto2020_TOIs": "J/ApJS/250/20",
    # https://filtergraph.com/tess_rotation_tois
    }
VIZIER_KEYS_CLUSTER_CATALOG = {
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
    # 	Gaia-ESO Survey in 7 open star cluster fields
    "Randich2018": "J/A+A/612/A99",
    "Karchenko2013": "J/A+A/558/A53",
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

class Target:
    def __init__(self, ra_deg, dec_deg, gaiaDR2id=None, verbose=True):
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.ra = ra_deg
        self.dec = dec_deg
        self.target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        self.verbose = verbose
        self.vizier_tables = None
        self.search_radius = 1 * u.arcmin

    def query_vizier(self, radius=3, verbose=None):
        """
        Useful to get relevant catalogs from literature
        See:
        https://astroquery.readthedocs.io/en/latest/vizier/vizier.html
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec
        if verbose:
            print(
                f"Searching Vizier: ({self.target_coord.to_string()}) with radius={radius}."
            )
        # standard column sorted in increasing distance
        v = Vizier(
            columns=["*", "+_r"],
            # column_filters={"Vmag":">10"},
            # keywords=['stars:white_dwarf']
        )
        if self.vizier_tables is None:
            tables = v.query_region(self.target_coord, radius=radius)
            if tables is None:
                print("No result from Vizier.")
            else:
                if verbose:
                    print(f"{len(tables)} tables found.")
                    pprint(
                        {
                            k: tables[k]._meta["description"]
                            for k in tables.keys()
                        }
                    )
                self.vizier_tables = tables
        else:
            tables = self.vizier_tables.filled(fill_value)
        return tables

    def query_vizier_param(self, param=None, radius=3):
        """looks for value of param in each vizier table"""
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is not None:
            idx = [param in tab.columns for tab in tabs]
            vals = {}
            for i in np.argwhere(idx).flatten():
                k = tabs.keys()[int(i)]
                v = tabs[int(i)][param][0] #nearest match
                if isinstance(v, np.ma.core.MaskedConstant):
                    v = np.nan
                vals[k] = v
            if self.verbose:
                print(f"Found {sum(idx)} references in Vizier with `{param}`.")
            return vals
        else:
            #print all available keys
            cols = [tab.to_pandas().columns.tolist() for tab in tabs]
            print(f"Choose parameter:\n{list(np.unique(flatten_list(cols)))}")
    
    def __repr__(self):
        """Override to print a readable string representation of class"""
        # params = signature(self.__init__).parameters
        # val = repr(getattr(self, key))

        included_args = [
            # ===target attributes===
            # "name",
            # "toiid",
            # "ctoiid",
            # "ticid",
            # "epicid",
            "gaiaDR2id",
            "ra_deg",
            "dec_deg",
            "target_coord",
            "search_radius",
        ]
        args = []
        for key in self.__dict__.keys():
            val = self.__dict__.get(key)
            if key in included_args:
                if key == "target_coord":
                    # format coord
                    coord = self.target_coord.to_string("decimal")
                    args.append(f"{key}=({coord.replace(' ',',')})")
                elif val is not None:
                    args.append(f"{key}={val}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"

class CatalogDownloader:
    """download tables from vizier
    Attributes
    ----------
    tables : astroquery.utils.TableList
        collection of astropy.table.Table downloaded from vizier
    """

    def __init__(
        self, catalog_name, catalog_type="prot", data_loc=DATA_PATH, verbose=True, clobber=False
        ):
        self.catalog_name = catalog_name
        self.catalog_type = catalog_type
        if catalog_type.lower()=="prot":
            self.catalog_dict = VIZIER_KEYS_PROT_CATALOG
        elif catalog_type.lower()=="liew":
            self.catalog_dict = VIZIER_KEYS_LiEW_CATALOG
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
        """row_limit-1 to download all rows"""
        clobber = self.clobber if clobber is None else clobber
        if row_limit == -1:
            msg = "Downloading all tables in "
        else:
            msg = f"Downloading the first {row_limit} rows of each table "
        try:
            msg += f"{self.catalog_dict[self.catalog_name]} from vizier."
            if self.verbose:
                print(msg)
        except:
            errmsg = f"'{self.catalog_name}' not in {list(self.catalog_dict.keys())}.\n"
            errmsg+=f"\nUsing catalog_type={self.catalog_type}."
            raise ValueError(errmsg)

        # set row limit
        Vizier.ROW_LIMIT = row_limit

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
                    print(f"Saved: {fp}")
            else:
                print("Set clobber=True to overwrite.")

    def get_vizier_url(self, catalog_name=None):
        if catalog_name is None:
            catalog_name = self.catalog_name
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        vizier_key = self.catalog_dict[catalog_name]
        url = base_url + vizier_key
        if self.verbose:
            print("Data url:", url)
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


def get_absolute_gmag(gmag, distance, a_g):
    """
    gmag : float
        apparent G band magnitude
    distance : float
        distance in pc
    a_g : float
        extinction in the G-band
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

    if "distance" not in df.columns.any():
        df["parallax"] = df["parallax"].astype(float)
        idx = ~np.isnan(df["parallax"]) & (df["parallax"] > 0)
        df = df[idx]
        if sum(~idx) > 0:
            print(f"{sum(~idx)} removed NaN or negative parallaxes")

        df["distance"] = Distance(parallax=df["parallax"].values * u.mas).pc

    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if df.columns.isin([xaxis, yaxis]).sum() == 2:
        x = df[xaxis]
        y = df[yaxis]
        ax.set_xlabel(xaxis, fontsize=16)
        ax.set_ylabel(yaxis, fontsize=16)
    else:
        # compute absolute Gmag
        df["abs_gmag"] = get_absolute_gmag(
            df["phot_g_mean_mag"], df["distance"], df["a_g_val"]
        )
        # compute intrinsic color index
        if estimate_color:
            df["bp_rp0"] = get_absolute_color_index(
                df["a_g_val"], df["phot_bp_mean_mag"], df["phot_rp_mean_mag"]
            )
        else:
            df["bp_rp0"] = df["bp_rp"] - df["e_bp_min_rp_val"]
        ax.set_xlabel(r"$G_{BP} - G_{RP}$ [mag]", fontsize=16)
        ax.set_ylabel(r"$G$ [mag]", fontsize=16)

    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster catalog does not contain the target gaia id [{target_gaiaid}]"
            assert sum(idx) > 0, errmsg
            x, y = df.loc[idx, "bp_rp0"], df.loc[idx, "abs_gmag"]
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
                df_target["bp_rp0"] = get_absolute_color_index(
                    df_target["a_g_val"],
                    df_target["phot_bp_mean_mag"],
                    df_target["phot_rp_mean_mag"],
                )
            else:
                df_target["bp_rp0"] = (
                    df_target["bp_rp"] - df_target["e_bp_min_rp_val"]
                )
            x, y = df_target["bp_rp0"], df_target["abs_gmag"]
        if target_label is not None:
            ax.legend(loc="best")
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms="25",
            label=target_label,
            zorder=10,
        )
    if log_age is not None:
        # plot isochrones
        try:
            from isochrones import get_ichrone

            iso_grid = get_ichrone("mist")
        except Exception:
            errmsg = "pip install isochrones"
        assert len(eep_limits) == 2, "eep_limits=(min,max)"
        iso_df = iso_grid.isochrone(log_age, feh)
        idx = (iso_df.eep > eep_limits[0]) & (iso_df.eep < eep_limits[1])
        G = iso_df.G_mag[idx]
        BP_RP = iso_df.BP_mag[idx] - iso_df.RP_mag[idx]
        label = f"log(t)={log_age:.2f}\nfeh={feh:.2f}"
        ax.plot(BP_RP, G, c="k", label=label)
        ax.legend(title="MIST isochrones")

    # df.plot.scatter(ax=ax, x="bp_rp", y="abs_gmag", marker=".")
    if color == "radius_val":
        rstar = np.log10(df[color].astype(float))
        c = ax.scatter(df[xaxis], df[yaxis], marker=".", c=rstar, cmap=cmap)
        ax.figure.colorbar(c, ax=ax, label=r"$\log$(R/R$_{\odot}$)")
    else:
        c = ax.scatter(df[xaxis], df[yaxis], c=df[color], marker=".")
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
    assert df.columns.isin([par]).any(), errmsg
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
        print("Error: ", e)
        npar = len(df[par].dropna())
        if npar < 10:
            errmsg = f"Cluster members have only {npar} {par} measurements."
            print("Error: ", errmsg)
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
    if np.any(df["parallax"] < 0):
        # retain non-negative parallaxes including nan
        df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
        if verbose:
            print("Some parallaxes are negative!")
            print("These are removed for the meantime.")
            print("For proper treatment, see:")
            print("https://arxiv.org/pdf/1804.09366.pdf\n")
    errmsg = f"radial_velocity is not in {df.columns}"
    assert df.columns.isin(["radial_velocity"]).any(), errmsg
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
    fp = join(data_loc, "mamajek_table.csv")
    if not exists(fp) or clobber:
        url = "http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
        # cols="SpT Teff logT BCv Mv logL B-V Bt-Vt G-V U-B V-Rc V-Ic V-Ks J-H H-Ks Ks-W1 W1-W2 W1-W3 W1-W4 Msun logAge b-y M_J M_Ks Mbol i-z z-Y R_Rsun".split(' ')
        df = pd.read_csv(
            url,
            skiprows=21,
            skipfooter=524,
            delim_whitespace=True,
            engine="python",
        )
        # tab = ascii.read(url, guess=None, data_start=0, data_end=124)
        # df = tab.to_pandas()
        # replace ... with NaN
        df = df.replace(["...", "....", "....."], np.nan)
        # replace header
        # df.columns = cols
        # drop last duplicate column
        df = df.drop(df.columns[-1], axis=1)
        # df['#SpT_num'] = range(df.shape[0])
        # df['#SpT'] = df['#SpT'].astype('category')

        # remove the : type in M_J column
        df["M_J"] = df["M_J"].apply(lambda x: str(x).split(":")[0])
        # convert columns to float
        for col in df.columns:
            if col == "#SpT":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(float)
            # if col=='SpT':
            #     df[col] = df[col].astype('categorical')
            # else:
            #     df[col] = df[col].astype(float)
        df.to_csv(fp, index=False)
        print(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f"Loaded: {fp}")
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
        c.f. self.query_vizier_param("SpT")

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
