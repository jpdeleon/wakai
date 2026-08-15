"""Stellar targets and catalog queries."""

import re
import warnings
from dataclasses import dataclass, field
from urllib.parse import quote

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from loguru import logger
from scipy.interpolate import NearestNDInterpolator

from .catalog_keys import (
    VIZIER_KEYS_BINARY_STAR_CATALOG,
    VIZIER_KEYS_VARIABLE_STAR_CATALOG,
)
from .sed import get_mamajek_table


@dataclass
class Star:
    """
    A stellar target with helpers for querying astronomical catalogs.
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
    ra_deg: float = None
    dec_deg: float = None
    gaiaid: int = None
    verbose: bool = True

    def __post_init__(self):
        """Initialize query caches and derived stellar parameters."""
        self.search_radius = 3 * u.arcsec
        self._vizier_tables = None
        self._gaia_sources = None
        self.gaia_params = None
        self.variable_star = False
        self.ra = self.ra_deg
        self.dec = self.dec_deg
        self.magnitudes = None
        self.rstar = None
        self.mstar = None
        self.rhostar = None
        self.teff = None
        self.logg = None
        self.feh = None

    def __repr__(self):
        """Return a concise target description without making a query."""
        coord_str = "unknown"
        if self.ra_deg is not None and self.dec_deg is not None:
            coord_str = self.target_coord().to_string("decimal").replace(" ", ", ")
        return (f"{self.__class__.__name__}("
                f"star_name='{getattr(self, 'star_name', 'Unnamed')}', "
                f"gaiaid={self.gaiaid}, "
                f"ra={self.ra_deg}, dec={self.dec_deg}, "
                f"coord=({coord_str}))"
               )

    def target_coord(self) -> SkyCoord:
        """Return SkyCoord object for the target position."""
        if self.ra_deg is None or self.dec_deg is None:
            raise ValueError("Target coordinates are not set; call get_tfop_data() first.")
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

    def query_vizier(self, add_cols=None, radius=None, verbose=None):
        """
        Query Vizier catalogs near the target position.
        add_cols : additional columns
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec

        if self._vizier_tables is not None:
            return self._vizier_tables

        if verbose:
            logger.info(f"Searching Vizier at {self.target_coord().to_string('hmsdms')} with radius={radius}")

        default_columns = ["**", "+_r"]
        if add_cols is not None:
            logger.info(f"Added columns: {add_cols}")
            if isinstance(add_cols, str):
                default_columns.append(add_cols)
            else:
                default_columns.extend(add_cols)
        viz = Vizier(columns=list(default_columns))
        try:
            tables = viz.query_region(self.target_coord(), radius=radius)
        except Exception as e:
            logger.warning(f"Vizier query failed: {e}")
            return None

        if tables is None or len(tables) == 0:
            if verbose:
                logger.info("No Vizier tables found.")
            return None

        if verbose:
            logger.info(f"{len(tables)} tables found.")
            from pprint import pformat
            logger.info(pformat({k: tables[k]._meta.get("description", "") for k in tables.keys()}))

        self._vizier_tables = tables
        return tables

    @staticmethod
    def _vizier_scalar(column):
        """Return the nearest-row scalar without Astropy/NumPy wrappers."""
        value = column[0]
        if np.ma.is_masked(value):
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _vizier_uncertainty_column(table, column_name):
        """Find a conventional symmetric uncertainty column for a value column."""
        normalized = {name.lower(): name for name in table.colnames}
        candidates = (
            f"e_{column_name}",
            f"e{column_name}",
            f"{column_name}_err",
            f"{column_name}_error",
            f"err_{column_name}",
        )
        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]
        return None

    def query_vizier_param(
        self,
        param=None,
        radius=3,
        use_regex=False,
        detailed=False,
    ):
        """Query the nearest row of every VizieR table for a parameter.

        This is adapted from ``kenmon.Target.query_vizier_param``. With no
        parameter it returns all available column names. Exact, wildcard, and
        regular-expression matching are supported. ``detailed=True`` adds the
        catalog description, uncertainty, units, separation, and source URL
        needed by the GUI's source-selection workflow.
        """
        tables = self.query_vizier(radius=radius, verbose=False)
        if not tables:
            return [] if detailed or param is None else {}

        if param is None:
            return sorted({column for table in tables for column in table.colnames})

        pattern = None
        if use_regex:
            pattern = re.compile(param.replace("*", ".*"), re.IGNORECASE)

        matches = []
        values = {}
        for table_key, table in zip(tables.keys(), tables.values()):
            if len(table) == 0:
                continue
            if pattern is None:
                columns = [param] if param in table.colnames else []
            else:
                columns = [column for column in table.colnames if pattern.search(column)]

            description = table.meta.get("description", table._meta.get("description", ""))
            source_url = (
                "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source="
                f"{quote(str(table_key), safe='/+')}"
                f"&-c={quote(f'{self.ra_deg} {self.dec_deg}')}&-c.rs={radius}"
            )
            separation = self._vizier_scalar(table["_r"]) if "_r" in table.colnames else None

            for column_name in columns:
                value = self._vizier_scalar(table[column_name])
                uncertainty_column = self._vizier_uncertainty_column(table, column_name)
                uncertainty = (
                    self._vizier_scalar(table[uncertainty_column])
                    if uncertainty_column is not None
                    else None
                )
                key = f"{table_key}:{column_name}"
                values[key] = value
                matches.append({
                    "key": key,
                    "catalog": str(table_key),
                    "column": column_name,
                    "value": value,
                    "uncertainty": uncertainty,
                    "uncertainty_column": uncertainty_column,
                    "unit": str(getattr(table[column_name], "unit", "") or ""),
                    "separation_arcsec": separation,
                    "reference": description or str(table_key),
                    "source_url": source_url,
                })

        if self.verbose:
            logger.info(f"Found {len(matches)} VizieR matches for `{param}`.")
        return matches if detailed else values

    def get_vizier_param(self, add_cols=None, param=None, radius=3, use_regex=False):
        """
        Search for a specific parameter across all Vizier tables.
        add_cols : additional columns
        """
        if add_cols is not None:
            self.query_vizier(radius=radius, add_cols=add_cols, verbose=False)
        if param is None:
            return self.query_vizier_param(param=None, radius=radius)
        records = self.query_vizier_param(
            param=param,
            radius=radius,
            use_regex=use_regex,
            detailed=True,
        )
        results = {}
        for record in records:
            results.setdefault(record["catalog"], {})[record["column"]] = record["value"]
        return results

    def query_binary_star_catalogs(self):
        """Query catalogs that contain information about binary stars."""
        self._query_star_catalog(VIZIER_KEYS_BINARY_STAR_CATALOG)

    def query_variable_star_catalogs(self):
        """Query catalogs that contain information about variable stars."""
        self._query_star_catalog(VIZIER_KEYS_VARIABLE_STAR_CATALOG)
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.query_vizier(verbose=False)

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
                    logger.info(f"See also: {base_url}{tab._meta['name']}\n{s}")
                    self.variable_star = True

    def _query_star_catalog(self, catalog_keys):
        """
        Check for stars in specialized catalogs.

        Parameters:
            catalog_keys (dict): Dictionary mapping reference names to VizieR keys
        """
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.query_vizier(verbose=False)

        if all_tabs is None:
            return

        for key, tab in zip(all_tabs.keys(), all_tabs.values()):
            for ref, vkey in catalog_keys.items():
                if key == vkey:
                    d = tab.to_pandas().squeeze()
                    logger.info(f"{ref} ({base_url}{key}):\n{d}")

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
            raise RuntimeError(f"Gaia query failed: {e}") from e

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
                logger.info(f"Updating target coordinates based on Gaia source (separation: {nearest.separation:.2f}\")")
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

        ruwe_results = self.get_vizier_param(param="ruwe")
        if ruwe_results:
            first_match = next(iter(ruwe_results.values()))
            ruwe_val = next(iter(first_match.values()), None)
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
            raise ValueError(f"Error while fetching data for {self.star_name}.\nError: {e}") from e

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
                logger.info(f"\t{n}")

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
            self.ra = self.ra_deg
            self.dec = self.dec_deg

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
                    logger.info(f"Mstar=({self.mstar[0]:.2f},{self.mstar[1]:.2f}) Msun")
                if not np.isnan(self.rstar[0]):
                    logger.info(f"Rstar=({self.rstar[0]:.2f},{self.rstar[1]:.2f}) Rsun")
                if not np.isnan(self.rhostar[0]):
                    logger.info(f"Rhostar=({self.rhostar[0]:.2f},{self.rhostar[1]:.2f}) rhosun")
                if not np.isnan(self.teff[0]):
                    logger.info(f"teff=({self.teff[0]:.0f},{self.teff[1]:.0f}) K")
                if not np.isnan(self.logg[0]):
                    logger.info(f"logg=({self.logg[0]:.2f},{self.logg[1]:.2f}) cgs")
                logger.info(f"feh=({self.feh[0]:.2f},{self.feh[1]:.2f}) dex")

        except Exception as e:
            logger.error(f"Error extracting stellar parameters: {e}")
            raise ValueError(f"Check exofop: {self.exofop_url}") from e

    def query_simbad(self):
        """
        Query Simbad to get the object type of the target star.

        Returns:
            SimbadResult or None: Result of the query if target is resolved, otherwise None
        """
        simbad = Simbad()
        simbad.add_votable_fields("otype", "ids")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            names = list(self.star_names) if len(self.star_names) else []
            if self.star_name and self.star_name not in names:
                names.insert(0, self.star_name)

            for name in names:
                result = simbad.query_object(name)
                if result is not None:
                    return result

            if self.verbose:
                msg = f"Simbad cannot resolve {self.star_name}"
                msg += f" using any of its names: {names}"
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
        df = get_mamajek_table()
        gaia_sources = self.query_gaia_catalog()
        self.gaia_params = gaia_sources.iloc[0].squeeze()

        # effective temperature
        col = "teff"
        teff = self.gaia_params[f"{col}_val"]
        siglo = (self.gaia_params[f"{col}_val"] - self.gaia_params[f"{col}_percentile_lower"])
        sighi = (self.gaia_params[f"{col}_percentile_upper"] - self.gaia_params[f"{col}_val"])
        uteff = np.sqrt(sighi**2 + siglo**2)
        s_teff = (teff + np.random.randn(nsamples) * uteff)  # Monte Carlo samples
        logger.info(f"Gaia Teff={teff},{uteff} K")
        bands=['Gaia','V','J','H','K',
               'WISE 3.4 micron','WISE 4.6 micron','WISE 12 micron']
        mags = {}
        for band in bands:
            val=self.magnitudes.query("band==@band").value.squeeze()
            err=self.magnitudes.query("band==@band").value_e.squeeze()
            mags[band]=(val,err)
            logger.info(f"{band}=({val},{err})")
        colors = []
        if 'G-V' in columns:
            gv_color =  mags['Gaia'][0] - mags['V'][0]
            ugv_color = mags['Gaia'][1] + mags['V'][1]
            logger.info(f"G-V={gv_color:.2f}+/-{ugv_color:.2f}")
            s_gv_color = (
                 gv_color + np.random.randn(nsamples) * ugv_color
            )  # Monte Carlo samples
            colors.append(s_gv_color)
        if 'Bp-Rp' in columns:
            # Bp-Rp color index
            sources = self.query_gaia_catalog(radius=30)
            bprp_color =  sources.iloc[0]['bp_rp']
            ubprp_color = 0.01
            logger.info(f"Gaia Bp-Rp={bprp_color:.2f}+/-{ubprp_color:.2f}")
            s_bprp_color = (
                bprp_color + np.random.randn(nsamples) * ubprp_color
            )
            colors.append(s_bprp_color)
        if 'J-H' in columns:
            # J-H color index
            jh_color =  mags['J'][0] - mags['H'][0]
            ujh_color = mags['J'][1] + mags['H'][1]
            logger.info(f"J-H={jh_color:.2f}+/-{ujh_color:.2f}")
            s_jh_color = (
                jh_color + np.random.randn(nsamples) * ujh_color
            )  # Monte Carlo samples
            colors.append(s_jh_color)
        if 'H-Ks' in columns:
            # H-Ks color index
            hk_color =  mags['H'][0] - mags['K'][0]
            uhk_color = mags['H'][1] + mags['K'][1]
            logger.info(f"H-Ks={hk_color:.2f}+/-{uhk_color:.2f}")
            s_hk_color = (
                hk_color + np.random.randn(nsamples) * uhk_color
            )
            colors.append(s_hk_color)
        if 'W1-W2' in columns:
            # W1-W2 color index
            w1w2_color =  mags['WISE 3.4 micron'][0] - mags['WISE 4.6 micron'][0]
            uw1w2_color = mags['WISE 3.4 micron'][1] + mags['WISE 4.6 micron'][1]
            logger.info(f"W1-W2={w1w2_color:.2f}+/-{uw1w2_color:.2f}")
            s_w1w2_color = (
                w1w2_color + np.random.randn(nsamples) * uw1w2_color
            )
            colors.append(s_w1w2_color)
        if 'W1-W3' in columns:
            # W1-W3 color index
            w1w3_color =  mags['WISE 3.4 micron'][0] - mags['WISE 12 micron'][0]
            uw1w3_color = mags['WISE 3.4 micron'][1] + mags['WISE 12 micron'][1]
            logger.info(f"W1-W3={w1w3_color:.2f}+/-{uw1w3_color:.2f}")
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
            logger.info(d[idx].sort_values(ascending=False))
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


class Target(Star):
    """Backward-compatible constructor for the pre-refactor ``Target`` class.

    Historical notebooks build targets two ways: by name, e.g.
    ``Target("TOI-179")``, which resolves coordinates via ExoFOP; or by bare
    coordinates, e.g. ``Target(ra_deg, dec_deg)`` or ``Target(*coord)``, with
    no name available at all (row-wise VizieR lookups over a coordinate
    list). ``Star`` is a dataclass keyed on a mandatory ``star_name`` first
    field, so this subclass restores both calling conventions: a single
    positional string is treated as a name, two positional/keyword values
    are treated as ``ra_deg``/``dec_deg`` with a coordinate-derived
    placeholder name.
    """

    def __init__(self, ra_deg=None, dec_deg=None, star_name=None, verbose=True, **kwargs):
        if isinstance(ra_deg, str) and dec_deg is None:
            star_name, ra_deg = ra_deg, None
        if star_name is None:
            star_name = (
                f"J{ra_deg:.5f}{dec_deg:+.5f}"
                if ra_deg is not None and dec_deg is not None
                else "Unnamed"
            )
        super().__init__(star_name=star_name, ra_deg=ra_deg, dec_deg=dec_deg, verbose=verbose, **kwargs)
        if self.ra_deg is None and self.dec_deg is None and self.star_name != "Unnamed":
            self.get_tfop_data()
