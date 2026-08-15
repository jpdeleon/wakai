"""Exoplanet targets and archive access."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from loguru import logger

from .config import DATA_PATH
from .star import Star


@dataclass
class Planet(Star):
    """
    A class representing an exoplanet, extending the Star class with planet-specific functionality.
    """
    # star_name: str
    # alias: str  # Planet designation number (e.g., "01" for first planet)
    star_params: Dict[str, Tuple[float, float]] = None
    alias: str = "01"
    # Default arguments must follow non-default arguments
    source: str = "toi"  # Default source for planetary parameters (overrides Star's default source)
    planet_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    verbose: bool = True

    def __post_init__(self):
        """Initialize and fetch planet parameters."""
        super().__post_init__()

        if self.star_params is None:
            self.get_star_params()
            self.star_params = Star.params_to_dict(self)
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
            raise ValueError(f"No planet parameters found for {self.star_name}")

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
        for i, p in enumerate(self.data_json["planet_parameters"]):
            if p.get("prov") == self.source:
                idx = i
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
        coord_str = "unknown"
        if self.ra_deg is not None and self.dec_deg is not None:
            coord_str = self.target_coord().to_string("decimal").replace(" ", ", ")
        return (f"{self.__class__.__name__}("
                f"star_name='{getattr(self, 'star_name', 'Unnamed')}' "
                f"{self.alias}, "
                f"gaiaid={self.gaiaid}, "
                f"ra={self.ra_deg}, dec={self.dec_deg}, "
                f"coord=({coord_str}))"
               )


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
        Path(outdir).mkdir(parents=True)

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
        logger.info(f"Saved: {fp}")
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
        Path(outdir).mkdir(parents=True)

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
    msg += f"Saved: {fp}\n"
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
        logger.info(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        logger.info(f"Loaded: {fp}")
    return df


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
