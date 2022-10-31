import os
import numpy as np
import pandas as pd

from pathlib import Path
from pprint import pprint

from astropy.table import Table
from astroquery.vizier import Vizier

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


VIZIER_KEYS_PROT_CATALOG = {
    # See table1: https://arxiv.org/pdf/1905.10588.pdf
    "Feinstein2020_NYMG": "See data/Feinstein2020_NYMG.txt",
    "McQuillan2014_Kepler": "J/ApJS/211/24",
    "Nielsen2013_KeplerMS": "J/A+A/557/L10",
    "Barnes2015_NGC2548": "J/A+A/583/A73",  # , M48/NGC2548
    "Meibom2011_NGC6811": "J/ApJ/733/L9",
    "Curtis2019_PisEri": "J/AJ/158/77",  # 250Gyr
    "Curtis2019_NGC6811": "J/ApJ/879/49",  # 1Gyr
    "Curtis2019_Rup147": "J/ApJ/904/140", #Pleiades, Praesepe, NGC 6811, NGC 752, NGC 6819, and Ruprecht 147 
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

PROT_CATALOG_LIST = [key for key in VIZIER_KEYS_PROT_CATALOG.keys()]

class CatalogDownloader:
    """download tables from vizier
    Attributes
    ----------
    tables : astroquery.utils.TableList
        collection of astropy.table.Table downloaded from vizier
    """

    def __init__(
        self, catalog_name, data_loc=DATA_PATH, verbose=True, clobber=False
    ):
        self.catalog_name = catalog_name
        self.catalog_dict = VIZIER_KEYS_PROT_CATALOG
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
        msg += f"{self.catalog_dict[self.catalog_name]} from vizier."
        if self.verbose:
            print(msg)
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
        return base_url + vizier_key

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


class ProtCatalog(CatalogDownloader):
    """read and parse downloaded catalog data from vizier"""

    def __init__(
        self,
        catalog_name,
        # df_mem_path=None,
        verbose=True,
        clobber=False,
        data_loc=DATA_PATH,
    ):
        super().__init__(
            catalog_name=catalog_name,
            data_loc=data_loc,
            verbose=verbose,
            clobber=clobber,
        )
        """Initialize the catalog
        # Parameters
        # ----------
        # df_mem_path : str
        #     path to the cluster members file
        Attributes
        ----------
        data_loc : str
            data directory
        all_members: pd.DataFrame
            list of all members in catalog
        all_clusters : pd.DataFrame
            list of all clusters in catalog
        Note:
        setting `all_members` as a method (as opposed to attribute)
        seems not
        """
        self.catalog_list = PROT_CATALOG_LIST
        # self.df_mem_path = df_mem_path
        self.all_clusters = None  # self.query_catalog(return_members=False)
        self.all_members = None  # self.query_catalog(return_members=True)

        # files = glob(join(self.data_loc, "*.txt"))
        if self.data_loc.exists():  # & len(files)<2:
            if self.clobber:
                _ = self.get_tables_from_vizier(
                    row_limit=-1, save=True, clobber=self.clobber
                )
        else:
            _ = self.get_tables_from_vizier(
                row_limit=-1, save=True, clobber=self.clobber
            )
        if self.verbose:
            print("Data url:", self.get_vizier_url())

        # if self.df_mem_path is not None:
        #     self.all_members = pd.read_csv(self.df_mem_path)

    def query_catalog(
        self, name=None, return_members=False, verbose=None, **kwargs
    ):
        """Query catalogs
        Parameters
        ----------
        name : str
            catalog name; see `self.catalog_list`
        return_members : bool
            return parameters for all members instead of the default
        Returns
        -------
        df : pandas.DataFrame
            dataframe parameters of the cluster or its individual members
        Note:
        1. See self.vizier_url() for details
        2. Use the following:
        if np.any(df["parallax"] < 0):
            df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
            if verbose:
                print("Some parallaxes are negative!")
                print("These are removed for the meantime.")
                print("For proper treatment, see:")
                print("https://arxiv.org/pdf/1804.09366.pdf\n")
        FIXME: self.all_clusters and self.all_members are repeated each if else block
        """
        self.catalog_name = name if name is not None else self.catalog_name
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print(f"Using {self.catalog_name} catalog.")
        if self.catalog_name == "Rebull2016":
            if return_members:
                df_mem = self.get_members_Hao2022()
                self.all_members = df_mem
                return df_mem
            else:
                df = self.get_clusters_Hao2022()
                self.all_clusters = df
                return df