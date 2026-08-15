"""VizieR catalog discovery, download, and crossmatching."""

from pathlib import Path
from pprint import pprint

import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from loguru import logger

from .catalog_keys import ALL_VIZIER_DICTIONARIES
from .config import DATA_PATH


class CatalogDownloader:
    """
    Downloads tables from Vizier.

    Behavior:
      - If vizier_key is provided → use it.
      - Else if catalog_name is provided → search all known dictionaries.
      - Else → fail.
    """

    def __init__(
        self,
        catalog_name: str | None = None,
        catalog_type: str | None = None,
        vizier_key: str | None = None,
        data_loc=DATA_PATH,
        verbose: bool = True,
        clobber: bool = False
    ):
        self.catalog_name = catalog_name
        self.catalog_type = catalog_type.lower() if catalog_type else None
        self.verbose = verbose
        self.clobber = clobber

        if vizier_key is None and catalog_name is None:
            raise ValueError("Either 'catalog_name' or 'vizier_key' must be provided.")

        self.vizier_key = self._resolve_vizier_key(vizier_key)

        Path(data_loc).mkdir(parents=True, exist_ok=True)
        self.data_loc = Path(data_loc, catalog_name if catalog_name else "anonymous_catalog")
        self.tables = None

        if self.verbose:
            logger.info(f"Using Vizier key: {self.vizier_key}")

    # ---------------------- Key Resolution ----------------------

    def _resolve_vizier_key(self, user_key):
        """Priority:
        1) user-specified key
        2) search all dictionaries using catalog_name
        """
        if user_key is not None:
            return user_key

        # catalog_name guaranteed non-None here
        name = self.catalog_name

        for d in ALL_VIZIER_DICTIONARIES:
            if name in d:
                return d[name]

        all_keys = sorted(
            {k for d in ALL_VIZIER_DICTIONARIES for k in d.keys()}
        )
        raise ValueError(
            f"'{name}' not found in any known Vizier dictionary.\n"
            f"Available names include:\n{all_keys}"
        )

    # ---------------------- Vizier Access ----------------------

    def get_tables_from_vizier(self, add_cols=None, row_limit=50, save=False, clobber=None):
        clobber = self.clobber if clobber is None else clobber

        if self.vizier_key == "local":
            raise ValueError("Local CSV requested; Vizier fetch disabled.")

        if self.verbose:
            msg = "Downloading all tables" if row_limit == -1 else f"Downloading first {row_limit} rows"
            logger.info(f"{msg} from {self.vizier_key}")

        default_columns = ["*", "+_r"]
        if add_cols is not None:
            if isinstance(add_cols, str):
                default_columns.append(add_cols)
            else:
                default_columns.extend(add_cols)

        viz = Vizier(columns=list(default_columns), row_limit=row_limit)
        tables = viz.get_catalogs(self.vizier_key)

        if tables is None:
            raise RuntimeError("No data returned from Vizier.")

        self.tables = tables

        if self.verbose:
            pprint({k: tables[k]._meta["description"] for k in tables.keys()})

        if save:
            self.save_tables(clobber=clobber)

        return tables

    def save_tables(self, clobber=None):
        if self.tables is None:
            raise RuntimeError("No tables available to save.")

        clobber = self.clobber if clobber is None else clobber
        self.data_loc.mkdir(parents=True, exist_ok=True)

        for n, table in enumerate(self.tables):
            fp = Path(self.data_loc, f"{self.catalog_name or 'anonymous'}_tab{n}.txt")
            if not fp.exists() or clobber:
                table.write(fp, format="ascii")
                if self.verbose:
                    logger.info(f"Saved: {fp}")
            else:
                logger.info("File exists. Enable clobber to overwrite.")

    def get_vizier_url(self):
        base = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        url = base + self.vizier_key
        if self.verbose:
            logger.info(f"Data URL: {url}")
        return url

    def __repr__(self):
        return (
            f"{type(self).__name__}(catalog_name={self.catalog_name}, "
            f"vizier_key={self.vizier_key})"
        )


def crossmatch_catalogs(c1, c2, max_sep_arcsec=None):
    """
    Crossmatch two SkyCoord lists.

    Parameters
    ----------
    c1 : SkyCoord
        First coordinate list.
    c2 : SkyCoord
        Second coordinate list.
    max_sep_arcsec : float or None
        Maximum separation allowed for accepting a match.
        If None, all nearest neighbors are returned.

    Returns
    -------
    idx : ndarray
        Indices in c2 corresponding to each entry in c1.
    sep : Quantity
        Angular separations for each match.
    mask : ndarray (boolean)
        True where matched within max_sep_arcsec.
    """
    idx, sep2d, _ = c1.match_to_catalog_sky(c2)

    if max_sep_arcsec is not None:
        limit = max_sep_arcsec * u.arcsec
        mask = sep2d < limit
    else:
        mask = np.ones(len(c1), dtype=bool)

    return idx, sep2d, mask
