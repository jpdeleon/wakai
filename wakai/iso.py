"""Isochrone helpers and stellar-evolution diagrams."""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from astropy.coordinates import Distance
from loguru import logger

from .catalog_keys import VIZIER_KEYS_AGE_CATALOG
from .config import DATA_PATH


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
    df = df.copy()  # derived columns below must not leak into the caller's frame
    if df_target is not None:
        df_target = df_target.copy()

    def dereddened_bp_rp(frame):
        """Intrinsic BP-RP, either from the catalog E(BP-RP) or estimated from A_G.

        ``estimate_color`` used to call an undefined ``get_absolute_color_index``.
        Without a catalog reddening, E(BP-RP) is approximated as A_G / 2
        (A_G ~ 2 E(BP-RP); Andrae et al. 2018, A&A 616, A8).
        """
        if estimate_color:
            bp_rp = frame["phot_bp_mean_mag"] - frame["phot_rp_mean_mag"]
            return bp_rp - frame["a_g_val"] / 2.0
        return frame["bp_rp"] - frame["e_bp_min_rp_val"]

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
        df[xaxis] = dereddened_bp_rp(df)
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
            df_target[xaxis] = dereddened_bp_rp(df_target)
            x, y = df_target[xaxis], df_target[yaxis]
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms=25,
            label=target_label,
            # cmap=cmap,
            zorder=10,
        )
        if target_label is not None:
            ax.legend(loc="best")
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
        ax.plot(
            x,
            y,
            marker=r"$\star$",
            c=target_color,
            ms=25,
            label=target_label,
        )
        if target_label is not None:
            ax.legend(loc="best")
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
        iso_df = iso_grid.df.loc[nearest_log_age, nearest_feh].copy()
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


def get_mist_eep_table():
    """
    For eep phases, see
    http://waps.cfa.harvard.edu/MIST/README_tables.pdf
    """
    fp = Path(DATA_PATH, "mist_eep_table.csv")
    return pd.read_csv(fp, comment="#")
