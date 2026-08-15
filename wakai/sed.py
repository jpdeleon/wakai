"""Stellar-atmosphere and SED calibration helpers."""

from pathlib import Path

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from astropy.table import Table
from loguru import logger
from scipy.interpolate import NearestNDInterpolator

from .catalog_keys import VIZIER_KEYS_STAR_CATALOG
from .config import DATA_PATH


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
        tab.to_csv(fp, index=False)
        df = tab
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


def get_bv_from_teff(teff, teff_err=None, feh=None, data_loc=DATA_PATH, clobber=False, verbose=False):
    """
    Interpolate mean (B-V)o color for a dwarf star at the given Teff using
    the Pecaut & Mamajek (2013+) EEM dwarf table (see get_mamajek_table).

    This does plain 1-D linear interpolation over the table's real Teff and
    B-V columns (no polynomial fit). teff_err is propagated to a B-V
    uncertainty via the local slope d(B-V)/d(Teff) bracketing the requested
    Teff. The table itself carries no per-point intrinsic/cosmic scatter
    (from metallicity, reddening, or multiplicity), so the returned bv_err
    is a measurement-error-only lower bound on the true color uncertainty.

    feh is not used to correct the interpolated value -- the table has no
    metallicity axis, so no such correction is calibrated here. It is only
    used to flag a caveat when the target is notably non-solar.

    Parameters
    ----------
    teff : float
        Effective temperature in K.
    teff_err : float, optional
        1-sigma uncertainty on teff in K.
    feh : float, optional
        [Fe/H] in dex.

    Returns
    -------
    dict with keys: bv, bv_err (float or None), out_of_range (bool),
    caveat (str or None).
    """
    df = get_mamajek_table(clobber=clobber, verbose=verbose, data_loc=data_loc)

    teff_col = pd.to_numeric(df["Teff"], errors="coerce")
    bv_col = pd.to_numeric(df["B-V"], errors="coerce")
    mask = teff_col.notna() & bv_col.notna()
    teff_grid = teff_col[mask].to_numpy(dtype=float)
    bv_grid = bv_col[mask].to_numpy(dtype=float)

    order = np.argsort(teff_grid)
    teff_grid = teff_grid[order]
    bv_grid = bv_grid[order]

    bv = float(np.interp(teff, teff_grid, bv_grid))
    out_of_range = bool(teff < teff_grid[0] or teff > teff_grid[-1])

    bv_err = None
    if teff_err is not None:
        idx = np.searchsorted(teff_grid, teff)
        idx = min(max(idx, 1), len(teff_grid) - 1)
        slope = (bv_grid[idx] - bv_grid[idx - 1]) / (teff_grid[idx] - teff_grid[idx - 1])
        bv_err = float(abs(slope) * teff_err)

    caveat = None
    if feh is not None and abs(feh) > 0.25:
        caveat = (
            f"Fe/H={feh:+.2f} deviates from solar; the Pecaut & Mamajek "
            "table has no metallicity axis, so (B-V)o may be systematically "
            "offset for this star."
        )
    if out_of_range:
        rng_caveat = (
            f"Teff={teff:.0f} K is outside the tabulated range "
            f"[{teff_grid[0]:.0f}, {teff_grid[-1]:.0f}] K; (B-V)o is edge-extrapolated."
        )
        caveat = rng_caveat if caveat is None else f"{caveat} {rng_caveat}"

    return {"bv": bv, "bv_err": bv_err, "out_of_range": out_of_range, "caveat": caveat}
