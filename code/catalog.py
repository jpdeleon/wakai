import os
from re import M
import numpy as np
import pandas as pd

from pathlib import Path
from pprint import pprint

import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
from astroquery.vizier import Vizier

from utils import flatten_list

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


VIZIER_KEYS_LiEW_CATALOG = {
    "Bouvier2018_Pleiades": "J/A+A/613/A63",
    "Cummings2017_HyadesPraesepe": "J/AJ/153/128"
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
            tables = self.vizier_tables
        return tables

    def query_vizier_param(self, param=None, radius=3):
        """looks for value of param in each vizier table"""
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is not None:
            idx = [param in i.columns for i in tabs]
            vals = {
                tabs.keys()[int(i)]: tabs[int(i)][param][0]
                for i in np.argwhere(idx).flatten()
            }
            if self.verbose:
                print(f"Found {sum(idx)} references in Vizier with `{param}`.")
            return vals
        else:
            cols = [i.to_pandas().columns.tolist() for i in tabs]
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
        self.verbose = verbose
        self.clobber = clobber
        if not Path(data_loc).exists():
            Path(data_loc).mkdir()
        self.data_loc = Path(data_loc, self.catalog_name)
        self.tables = None

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