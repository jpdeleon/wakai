"""Cluster membership, kinematics, and diagnostic plots."""

from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from astropy.coordinates import Distance, Galactic, Galactocentric, SkyCoord
from loguru import logger
from scipy import stats

from .catalog_keys import VIZIER_KEYS_CLUSTER_CATALOG
from .config import DATA_PATH


def get_cluster_3d_size(group):
    # 1. Create the SkyCoord object for the group
    # Note: allow_negative=True is used to handle noisy parallax data
    coords = SkyCoord(
        ra=group['ra'].values * u.degree,
        dec=group['dec'].values * u.degree,
        distance=Distance(parallax=group['parallax'].values * u.mas, allow_negative=True),
        frame='icrs'
    )

    # 2. Convert to Cartesian (x, y, z) in parsecs
    cart = coords.cartesian
    points = np.stack([cart.x.value, cart.y.value, cart.z.value], axis=1)

    # 3. Calculate all-to-all pairwise distances
    # For large groups, consider using a faster method like scipy.spatial.distance.pdist
    from scipy.spatial.distance import pdist
    if len(points) > 1:
        distances = pdist(points)
        return np.max(distances) #* u.pc
    return 0.0


def plot_rdp_pmrv(
    df,
    target_gaiaid=None,
    match_id=True,
    df_target=None,
    target_label=None,
    target_color="r",
    target_ms=25,
    color="teff_val",
    marker="o",
    figsize=(10, 10),
    cmap="viridis",
    show_colorbar=True,
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
    assert color is None or color in df.columns, errmsg
    c = df[color] if color is not None else None
    cbar = ax[n].scatter(df[x], df[y], c=c, marker=marker, cmap=cmap)
    if color is not None and show_colorbar:
        fig.colorbar(cbar, ax=ax[n], label=color)
    if target_gaiaid is not None:
        idx = df.source_id.astype(int).isin([target_gaiaid])
        if match_id:
            errmsg = f"Given cluster does not contain the target gaia id [{target_gaiaid}]"
            assert idx.any(), errmsg
            ax[n].plot(
                df.loc[idx, x],
                df.loc[idx, y],
                marker=r"$\star$",
                c=target_color,
                ms=target_ms,
                label=target_label,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms=target_ms,
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
            assert idx.any(), errmsg
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
    if (color is not None) & (n == 3) & show_colorbar:
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
                ms=target_ms,
            )
        else:
            assert df_target is not None, "provide df_target"
            ax[n].plot(
                df_target[x],
                df_target[y],
                marker=r"$\star$",
                c=target_color,
                ms=target_ms,
            )
    ax[n].set_xlabel("PM R.A. [mas/yr]")
    ax[n].set_ylabel("PM Dec. [mas/yr]")
    text = len(df[["pmra", "pmdec"]].dropna())
    ax[n].text(0.8, 0.9, f"n={text}", fontsize=14, transform=ax[n].transAxes)
    n = 2
    par = "radial_velocity"
    errmsg = f"{par} is not available in {df.columns}"
    assert par in df.columns, errmsg
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
        # loguru drops positional args with no "{}" placeholder, so the old
        # logger.error("Error: ", e) logged just "Error: ".
        logger.error(f"Error: {e}")
        npar = len(df[par].dropna())
        if npar < 10:
            errmsg = f"Cluster members have only {npar} {par} measurements."
            logger.error(f"Error: {errmsg}")
            # raise ValueError(errmsg)
    return fig


def get_transformed_coord(df, frame="galactic", use_LSR=False, verbose=True):
    """
    df : pandas.DataFrame
        Input catalog containing full 6D astrometry and kinematics.
        Required columns:
            - 'ra' : Right Ascension [deg]
            - 'dec' : Declination [deg]
            - 'parallax' : Parallax [mas]
            - 'pmra' : Proper motion in RA*cos(dec) [mas/yr]
            - 'pmdec' : Proper motion in Dec [mas/yr]
            - 'radial_velocity' : Heliocentric radial velocity [km/s]
    frame : str, optional
        Target frame for transformation:
            - "galactic" (default) : Heliocentric Galactic Cartesian coordinates (Sun-centered)
            - "galactocentric" : Galactocentric Cartesian coordinates
    use_LSR : bool, optional (default=False)
        If True and frame="galactic", compute velocities relative to the Local Standard of Rest (LSR)
        using the solar motion (Schönrich et al. 2010):
            (U, V, W)_sun = (11.1, 12.24, 7.25) km/s
        U positive toward Galactic center, V in direction of rotation, W toward North Galactic Pole.

    Returns
    -------
    df : pandas.DataFrame
        Copy of the input catalog with additional columns for transformed
        coordinates and velocities.

    References
    ----------
    - https://arxiv.org/pdf/1707.00697.pdf (Solar motion estimates)
    - https://arxiv.org/pdf/1804.10607.pdf (Modeling Gaia DR2 in 6D)
    - https://ui.adsabs.harvard.edu/abs/1987AJ.....93..864J/abstract
    - https://learn.astropy.org/rst-tutorials/gaia-galactic-orbits.html

    See also
    --------
    - https://github.com/vsquicciarini/madys/blob/main/madys.py#L5936
    - https://gala.adrian.pw/en/latest/tutorials/mock-stream-heliocentric.html1
    """
    assert len(df) > 0, "Input DataFrame is empty."

    # Handle negative parallaxes
    if any(df["parallax"] < 0):
        df = df[(df["parallax"] >= 0) | (df["parallax"].isnull())]
        if verbose:
            print("Warning: Some negative parallaxes were removed.")
            print("For proper treatment, see Luri et al. (2018): https://arxiv.org/pdf/1804.09366.pdf\n")

    errmsg = "Column 'radial_velocity' not found in DataFrame."
    assert "radial_velocity" in df.columns, errmsg

    df2 = df.copy()

    # Build SkyCoord object in ICRS frame (heliocentric)
    icrs = SkyCoord(
        ra=df2["ra"].values * u.deg,
        dec=df2["dec"].values * u.deg,
        distance=Distance(parallax=df2["parallax"].values * u.mas),
        pm_ra_cosdec=df2["pmra"].values * u.mas / u.yr,
        pm_dec=df2["pmdec"].values * u.mas / u.yr,
        radial_velocity=df2["radial_velocity"].values * u.km / u.s,
        frame="icrs",
        obstime="J2016.0",
    )

    # --- GALACTOCENTRIC TRANSFORMATION ---
    if frame.lower() == "galactocentric":
        galcen_frame = Galactocentric(
            galcen_distance=8.122*u.kpc,  # Sun-Galactic center distance
            z_sun=20.8*u.pc,              # Sun's height above the plane
        )
        galcen = icrs.transform_to(galcen_frame)
        df2["X"] = galcen.x.to(u.pc).value
        df2["Y"] = galcen.y.to(u.pc).value
        df2["Z"] = galcen.z.to(u.pc).value
        df2["U"] = galcen.velocity.d_x.to(u.km / u.s).value
        df2["V"] = galcen.velocity.d_y.to(u.km / u.s).value
        df2["W"] = galcen.velocity.d_z.to(u.km / u.s).value

    # --- GALACTIC TRANSFORMATION ---
    elif frame.lower() == "galactic":
        gal_frame = Galactic()
        gal = icrs.transform_to(gal_frame)
        # gal = icrs.transform_to('galactic')
        df2["gal_l"] = gal.l.deg
        df2["gal_b"] = gal.b.deg
        df2["gal_pm_l_cosb"] = gal.pm_l_cosb.to(u.mas / u.yr).value
        df2["gal_pm_b"] = gal.pm_b.to(u.mas / u.yr).value
        df2["gal_vr"] = gal.radial_velocity.to(u.km/u.s).value

        # --- Positions (heliocentric Galactic coordinates) ---
        df2["X"] = gal.cartesian.x.to(u.pc).value   # toward Galactic center
        df2["Y"] = gal.cartesian.y.to(u.pc).value   # direction of rotation
        df2["Z"] = gal.cartesian.z.to(u.pc).value   # toward NGP

        # --- Velocities (heliocentric Galactic velocities) ---
        df2["U"] = gal.velocity.d_x.to(u.km/u.s).value  # toward Galactic center
        df2["V"] = gal.velocity.d_y.to(u.km/u.s).value  # direction of rotation
        df2["W"] = gal.velocity.d_z.to(u.km/u.s).value  # toward NGP

        # Optional LSR correction
        if use_LSR:
            if verbose:
                print("Setting U,V,W in Local Standard of Rest (LSR).")
            U_sun, V_sun, W_sun = 11.1, 12.24, 7.25  # km/s
            df2["U"] += U_sun
            df2["V"] += V_sun
            df2["W"] += W_sun
    else:
        raise ValueError(f"Unsupported frame: {frame!r}. Choose 'galactocentric' or 'galactic'.")

    if verbose:
        print(f"Coordinate transformation: '{frame}'")
        print("Sun(X,Y,Z)=(–8.2, 0, +0.025) kpc\nSun(U,V,W)=(11.1,12.24,7.25) km/s")

    return df2


def plot_xyz_uvw(df: pd.DataFrame, target_gaiaid=None, match_id: bool=True,
                 df_target=None, target_color: str="r", target_ms: int=14,
                 color: str="teff_val", marker: str="o", verbose: bool=True,
                 figsize: tuple=(12, 8), cmap="viridis", show_colorbar: bool=True,
                 **kwargs):

    assert not df.empty, "df is empty"
    if color not in df.columns:
        raise ValueError(f"color='{color}' not in {df.columns}")

    # 1. Coordinate Transformation (Vectorized)

    required = ["X", "Y", "Z", "U", "V", "W"]
    if all(col in df.columns for col in required):
        # Copy so the in-place kpc rescaling below never touches the caller's
        # frame (repeat calls used to divide X/Y/Z by 1e3 again each time).
        df = df.copy()
    else:
        df = get_transformed_coord(df, **kwargs)

    pos_cols = ["X", "Y", "Z"]
    if np.any([df[i].abs().median()<1000 for i in pos_cols]):
        units = ["pc"] * 3 + ["km/s"] * 3
        scale = 1
    else:
        units = ["kpc"] * 3 + ["km/s"] * 3
        scale = 1e3

    # convert to kpc if necessary
    df[pos_cols] = df[pos_cols]/scale

    if df_target is not None:
        # Ensure df_target is transformed once
        target_df = pd.DataFrame(df_target).T if isinstance(df_target, pd.Series) else df_target
        target_df = get_transformed_coord(target_df, **kwargs)
        # convert to kpc if necessary
        target_df[pos_cols] = target_df[pos_cols]/scale
    else:
        target_df = pd.DataFrame()

    # 2. Setup Plotting
    fig, axs = pl.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axs = axs.flatten()

    # Pre-identify target stars in the main dataframe
    target_mask = pd.Series(False, index=df.index)
    if target_gaiaid is not None:
        tids = np.atleast_1d(target_gaiaid)
        target_mask = df['source_id'].astype(int).isin(tids)

    # 3. Iterative Plotting
    # Pairs: (X,Y), (Y,Z), (Z,X) then (U,V), (V,W), (W,U)
    coords = [('X','Y'), ('Y','Z'), ('Z','X'), ('U','V'), ('V','W'), ('W','U')]

    for i, (col_x, col_y) in enumerate(coords):
        ax = axs[i]
        unit = units[i]

        # Background scatter
        sc = ax.scatter(df[col_x], df[col_y], c=df[color], marker=marker, cmap=cmap, zorder=1)

        # Highlight target star(s) from main DF
        if target_mask.any():
            ax.scatter(df.loc[target_mask, col_x], df.loc[target_mask, col_y],
                       marker="*", s=target_ms**2, c=target_color, edgecolors='k', zorder=3)

        # Plot explicit target_df if provided
        if not target_df.empty:
            ax.scatter(target_df[col_x], target_df[col_y],
                       marker="*", s=target_ms**2, c=target_color, edgecolors='w', zorder=4)

        ax.set_xlabel(f"{col_x} [{unit}]")
        ax.set_ylabel(f"{col_y} [{unit}]")

    # Add Colorbar to the last plot
    if show_colorbar:
        fig.colorbar(sc, ax=axs[5], label=color)

    if verbose:
        print(f"Plotted {len(df)} stars. Target(s): {target_gaiaid}")

    return fig, axs


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
    df = df.copy()  # don't add distance/x/y/z columns to the caller's frame
    if "distance" not in df.columns:
        df["distance"] = Distance(parallax=df.parallax.values * u.mas).pc

    coords = SkyCoord(
        ra=df.ra.values * u.deg,
        dec=df.dec.values * u.deg,
        distance=df.distance.values * u.pc,
    )
    xyz = coords.galactocentric
    df["x"] = xyz.x.to_value(u.pc)
    df["y"] = xyz.y.to_value(u.pc)
    df["z"] = xyz.z.to_value(u.pc)

    # Each limit narrows the selection; an unset limit keeps every star. The
    # old masks defaulted to False and were OR-ed, so with no limits nothing
    # was plotted, and with several limits they widened instead of narrowing.
    idx = np.ones(len(df), dtype=bool)
    for col, lim in (("x", xlim), ("y", ylim), ("z", zlim)):
        if lim:
            assert isinstance(lim, tuple)
            idx &= ((df[col] > lim[0]) & (df[col] < lim[1])).to_numpy()
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


# ---------------------------------------------------------------------------
# Cluster catalog access + membership assessment (the "wakai" cluster methods)
# ---------------------------------------------------------------------------

#: Default cluster catalog. Perren et al. 2023 (the Unified Cluster Catalogue,
#: https://ucc.ar) ships as a small cluster-level table plus a large member list.
DEFAULT_CLUSTER_CATALOG = "Perren2023"

_CLUSTER_CATALOG_FILES = {
    "Perren2023": {
        "clusters": "Perren2023/UCC_cat.csv",
        "members": "Perren2023/UCC_members.parquet",
    }
}

# Map the raw catalog columns onto wakai's canonical astrometry names so the
# plotting helpers (plot_xyz_uvw / plot_rdp_pmrv) work without further munging.
_MEMBER_COLUMN_MAP = {
    "name": "Name",
    "Source": "source_id",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Plx": "parallax",
    "e_Plx": "parallax_error",
    "pmRA": "pmra",
    "e_pmRA": "pmra_error",
    "pmDE": "pmdec",
    "e_pmDE": "pmdec_error",
    "RV": "radial_velocity",
    "e_RV": "radial_velocity_error",
    "Gmag": "phot_g_mean_mag",
    "BP-RP": "bp_rp",
    "probs": "Prob",
}

_CLUSTER_COLUMN_MAP = {
    "name": "Name",
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Plx": "parallax",
    "pmRA": "pmra",
    "pmDE": "pmdec",
    "Rv": "radial_velocity",
}

#: Astrometric features scored against the member distribution. Radial velocity
#: is added on top of these when both the target and enough members have one.
MEMBERSHIP_FEATURES = ("parallax", "pmra", "pmdec")

#: Curated cluster-level properties surfaced in the GUI summary table.
#: Maps the raw Perren column -> (human label, unit).
CLUSTER_SUMMARY_FIELDS = {
    "Name(s)": ("Name(s)", ""),
    "N_membs": ("Members (P>0.5)", ""),
    "Dist_[kpc]": ("Distance", "kpc"),
    "Age_[Myr]": ("Age", "Myr"),
    "FeH_[dex]": ("[Fe/H]", "dex"),
    "Av_[mag]": ("Extinction A_V", "mag"),
    "Mass_[Msun]": ("Mass", "Msun"),
    "r_50": ("Radius (half-member)", "arcmin"),
    "r_core": ("Core radius", "pc"),
    "parallax": ("Parallax", "mas"),
    "pmra": ("pmRA", "mas/yr"),
    "pmdec": ("pmDE", "mas/yr"),
    "radial_velocity": ("Radial velocity", "km/s"),
    "C3": ("Quality class (C3)", ""),
    "UTI": ("UCC trust index", ""),
}

# Minimum number of members with a finite measurement before a feature is
# trusted enough to enter the covariance used for the membership score.
_MIN_MEMBERS_FOR_FEATURE = 10


def _catalog_file(catalog, kind, data_path=None):
    """Resolve the on-disk path for a cluster catalog file."""
    try:
        relative = _CLUSTER_CATALOG_FILES[catalog][kind]
    except KeyError as exc:
        known = ", ".join(sorted(_CLUSTER_CATALOG_FILES))
        raise ValueError(
            f"Unknown cluster catalog {catalog!r} (kind={kind!r}). Known catalogs: {known}"
        ) from exc
    base = Path(data_path) if data_path is not None else Path(DATA_PATH)
    return base / relative


def load_cluster_catalog(catalog=DEFAULT_CLUSTER_CATALOG, data_path=None):
    """Load the cluster-level catalog (one row per cluster) with canonical names."""
    path = _catalog_file(catalog, "clusters", data_path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster catalog not found: {path}")
    df = pd.read_csv(path)
    return df.rename(columns=_CLUSTER_COLUMN_MAP)


def load_cluster_members(
    catalog=DEFAULT_CLUSTER_CATALOG,
    data_path=None,
    columns=None,
    cluster_name=None,
    gaia_id=None,
):
    """Load member stars with canonical names.

    Parameters
    ----------
    columns : list[str], optional
        Canonical column names to read (mapped back to the raw catalog columns).
    cluster_name : str, optional
        If given, use parquet predicate pushdown to read only that cluster's rows.
    gaia_id : int, optional
        If given, read only the row(s) matching this Gaia source id.
    """
    path = _catalog_file(catalog, "members", data_path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster member list not found: {path}")

    inverse = {v: k for k, v in _MEMBER_COLUMN_MAP.items()}
    read_columns = None
    if columns is not None:
        read_columns = [inverse.get(col, col) for col in columns]

    filters = []
    if cluster_name is not None:
        filters.append(("name", "==", str(cluster_name)))
    if gaia_id is not None:
        filters.append(("Source", "==", int(gaia_id)))

    df = pd.read_parquet(
        path,
        columns=read_columns,
        filters=filters or None,
    )
    return df.rename(columns=_MEMBER_COLUMN_MAP)


def find_cluster_by_gaiaid(
    gaia_id, members=None, catalog=DEFAULT_CLUSTER_CATALOG, data_path=None
):
    """Return catalog rows whose Gaia source id matches ``gaia_id``.

    An empty DataFrame means the target is not a listed member of any cluster.
    """
    gaia_id = int(gaia_id)
    if members is None:
        members = load_cluster_members(
            catalog=catalog,
            data_path=data_path,
            columns=["source_id", "Name", "Prob"],
            gaia_id=gaia_id,
        )
        return members.reset_index(drop=True)
    hits = members[members["source_id"].astype("int64") == gaia_id]
    return hits.reset_index(drop=True)


def _feature_vector(row, features):
    """Extract finite feature values from a mapping/Series, else None."""
    values = []
    for key in features:
        value = row.get(key) if hasattr(row, "get") else row[key]
        value = float(value) if value is not None else np.nan
        values.append(value)
    array = np.asarray(values, dtype=float)
    return array if np.all(np.isfinite(array)) else None


def compute_membership_probability(
    target,
    members,
    features=MEMBERSHIP_FEATURES,
    use_rv="auto",
    min_members=_MIN_MEMBERS_FOR_FEATURE,
):
    """Score a target against a cluster's member distribution.

    A multivariate-normal model is fit to the cluster members in astrometric
    feature space (parallax, proper motion, and optionally radial velocity).
    The squared Mahalanobis distance of the target follows a chi-squared
    distribution with ``dof`` degrees of freedom, so the survival function
    ``P = chi2.sf(d^2, dof)`` gives the probability that a genuine member would
    be at least this discrepant. Values near 1 are fully consistent with
    membership; values near 0 are inconsistent.

    Returns
    -------
    dict
        ``probability``, ``mahalanobis``, ``dof``, ``features`` used, and
        ``n_members`` contributing to the covariance. ``probability`` is None
        when no feature has enough members or the target lacks the inputs.
    """
    feature_list = list(features)
    if use_rv == "auto":
        include_rv = "radial_velocity" not in feature_list
    else:
        include_rv = bool(use_rv)
    if include_rv:
        feature_list = [*feature_list, "radial_velocity"]

    # Keep only features the target actually has and that enough members share.
    usable = []
    for key in feature_list:
        target_value = target.get(key) if hasattr(target, "get") else target[key]
        if target_value is None or not np.isfinite(float(target_value)):
            continue
        if key not in members.columns:
            continue
        finite = members[key].dropna()
        if len(finite) >= min_members:
            usable.append(key)

    empty = {
        "probability": None,
        "mahalanobis": None,
        "dof": 0,
        "features": [],
        "n_members": 0,
    }
    if len(usable) < 2:
        return empty

    sample = members[usable].dropna()
    if len(sample) < min_members:
        return empty

    matrix = sample.to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    # Regularize against singular covariance (e.g. a near-degenerate feature).
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * 1e-12
    try:
        inverse_cov = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse_cov = np.linalg.pinv(covariance)

    target_vector = _feature_vector(target, usable)
    if target_vector is None:
        return empty

    delta = target_vector - mean
    d2 = float(delta @ inverse_cov @ delta)
    dof = len(usable)
    probability = float(stats.chi2.sf(d2, dof))
    return {
        "probability": probability,
        "mahalanobis": float(np.sqrt(d2)),
        "dof": dof,
        "features": usable,
        "n_members": int(len(sample)),
    }


def _skycoord_from_frame(ra, dec, parallax):
    """Build an ICRS SkyCoord with distance from parallax (negatives allowed)."""
    return SkyCoord(
        ra=np.atleast_1d(ra) * u.deg,
        dec=np.atleast_1d(dec) * u.deg,
        distance=Distance(parallax=np.atleast_1d(parallax) * u.mas, allow_negative=True),
        frame="icrs",
    )


def _separation_from_core(target, cluster_row):
    """3D separation (pc) between the target and a cluster centroid."""
    required = ("ra", "dec", "parallax")
    for key in required:
        target_value = target.get(key) if hasattr(target, "get") else target[key]
        core_value = cluster_row.get(key)
        if target_value is None or core_value is None:
            return None
        if not (np.isfinite(float(target_value)) and np.isfinite(float(core_value))):
            return None
    target_coord = _skycoord_from_frame(
        target["ra"], target["dec"], target["parallax"]
    )
    core_coord = _skycoord_from_frame(
        cluster_row["ra"], cluster_row["dec"], cluster_row["parallax"]
    )
    return float(target_coord.separation_3d(core_coord).to(u.pc).value[0])


def crossmatch_target_to_clusters(
    target,
    catalog_df=None,
    catalog=DEFAULT_CLUSTER_CATALOG,
    data_path=None,
    min_members=_MIN_MEMBERS_FOR_FEATURE,
):
    """Find the nearest cluster core (3D) and score membership for it.

    Used when the target is not a listed member of any cluster. Returns the
    nearest cluster's name, the loaded member table, the 3D separation from the
    cluster core, the cluster's spatial extent, and the membership score.
    """
    if catalog_df is None:
        catalog_df = load_cluster_catalog(catalog=catalog, data_path=data_path)

    # parallax > 0, not just non-null: match_to_catalog_3d (below) builds a
    # Cartesian KD-tree over every row up front and raises on any NaN entry,
    # unlike match_to_catalog_sky (2D, ignores distance for matching itself).
    # A negative parallax always converts to a NaN Distance regardless of
    # _skycoord_from_frame's allow_negative=True (that flag only suppresses
    # the error, not the NaN -- see astropy.coordinates.Distance), so those
    # rows have to be dropped here rather than left for the KD-tree to choke on.
    valid = catalog_df.dropna(subset=["ra", "dec", "parallax"])
    valid = valid[valid["parallax"] > 0].reset_index(drop=True)
    if valid.empty:
        raise ValueError("Cluster catalog has no usable ra/dec/parallax rows")

    target_coord = _skycoord_from_frame(target["ra"], target["dec"], target["parallax"])
    cluster_coords = _skycoord_from_frame(
        valid["ra"].values, valid["dec"].values, valid["parallax"].values
    )
    # match_to_catalog_3d (not _sky) so the nearest cluster is chosen by true
    # 3D separation, matching this function's own docstring/contract -- two
    # clusters that are close in sky position but far apart in distance
    # would otherwise let the angularly-closer one win over the truly
    # nearest one. Same (idx, sep2d, sep3d) return shape as match_to_catalog_sky.
    #
    # Requires the *target's own* parallax to convert to a finite distance
    # too (match_to_catalog_3d raises ValueError on a NaN query point, unlike
    # match_to_catalog_sky which only uses ra/dec for matching). A target
    # with a non-positive parallax -- plausible for a faint/distant real
    # target, not just bad data -- has no usable distance at all, so "nearest
    # in 3D" is undefined for it anyway; fall back to the old 2D-angular
    # matching for that one case rather than raising.
    target_parallax = target.get("parallax") if hasattr(target, "get") else target["parallax"]
    if target_parallax is not None and np.isfinite(target_parallax) and target_parallax > 0:
        idx, sep2d, sep3d = target_coord.match_to_catalog_3d(cluster_coords)
    else:
        idx, sep2d, sep3d = target_coord.match_to_catalog_sky(cluster_coords)
    index = int(np.atleast_1d(idx)[0])
    nearest = valid.iloc[index]
    cluster_name = nearest["Name"]

    members = load_cluster_members(
        catalog=catalog, data_path=data_path, cluster_name=cluster_name
    )
    extent_pc = get_cluster_3d_size(members) if len(members) > 1 else 0.0
    sep3d_pc = _separation_from_core(target, nearest)
    membership = compute_membership_probability(
        target, members, min_members=min_members
    )
    return {
        "cluster_name": cluster_name,
        "cluster_row": nearest,
        "members": members,
        "sep2d_arcsec": float(np.atleast_1d(sep2d.to(u.arcsec).value)[0]),
        "sep3d_pc": sep3d_pc,
        "extent_pc": float(extent_pc),
        "membership": membership,
    }


def get_cluster_summary(
    cluster_name, catalog_df=None, catalog=DEFAULT_CLUSTER_CATALOG, data_path=None
):
    """Return a curated list of labeled properties for one cluster."""
    if catalog_df is None:
        catalog_df = load_cluster_catalog(catalog=catalog, data_path=data_path)
    match = catalog_df[catalog_df["Name"] == cluster_name]
    if match.empty:
        return []
    row = match.iloc[0]
    summary = []
    for column, (label, unit) in CLUSTER_SUMMARY_FIELDS.items():
        if column not in row.index:
            continue
        value = row[column]
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        summary.append({"label": label, "value": value, "unit": unit})
    return summary


@dataclass
class ClusterMatch:
    """Outcome of assessing whether a target belongs to a known cluster."""

    matched: bool
    method: str  # "gaia_id" or "crossmatch"
    cluster_name: str | None
    membership_probability: float | None
    catalog_probability: float | None
    mahalanobis: float | None
    dof: int | None
    sep3d_pc: float | None
    extent_pc: float | None
    within_extent: bool | None
    n_members: int
    summary: list = field(default_factory=list)
    members: pd.DataFrame = field(default_factory=pd.DataFrame)


def assess_cluster_membership(
    target,
    catalog=DEFAULT_CLUSTER_CATALOG,
    data_path=None,
    min_members=_MIN_MEMBERS_FOR_FEATURE,
):
    """Decide whether ``target`` belongs to a cluster in ``catalog``.

    Strategy
    --------
    1. If the target has a Gaia id and it is listed as a member, report that
       cluster directly (``method="gaia_id"``) with its catalog membership
       probability.
    2. Otherwise crossmatch the target against every cluster core and score the
       nearest one (``method="crossmatch"``) with a Mahalanobis-distance
       membership probability.

    Parameters
    ----------
    target : Mapping
        Requires ``ra``, ``dec``, ``parallax`` for the crossmatch path and
        ``pmra``/``pmdec`` (and optionally ``radial_velocity``) for scoring.
        ``gaiaid`` enables the fast membership-by-id path.

    Returns
    -------
    ClusterMatch
    """
    catalog_df = load_cluster_catalog(catalog=catalog, data_path=data_path)
    gaia_id = target.get("gaiaid") if hasattr(target, "get") else None
    gaia_id = gaia_id or (target.get("source_id") if hasattr(target, "get") else None)

    if gaia_id:
        hits = find_cluster_by_gaiaid(
            gaia_id, catalog=catalog, data_path=data_path
        )
        if not hits.empty:
            hit = hits.sort_values("Prob", ascending=False).iloc[0]
            cluster_name = hit["Name"]
            members = load_cluster_members(
                catalog=catalog, data_path=data_path, cluster_name=cluster_name
            )
            extent_pc = get_cluster_3d_size(members) if len(members) > 1 else 0.0
            sep3d_pc = _separation_from_core(
                target, catalog_df[catalog_df["Name"] == cluster_name].iloc[0]
            ) if (catalog_df["Name"] == cluster_name).any() else None
            return ClusterMatch(
                matched=True,
                method="gaia_id",
                cluster_name=cluster_name,
                membership_probability=float(hit["Prob"]),
                catalog_probability=float(hit["Prob"]),
                mahalanobis=None,
                dof=None,
                sep3d_pc=sep3d_pc,
                extent_pc=float(extent_pc),
                within_extent=(sep3d_pc is not None and sep3d_pc <= extent_pc),
                n_members=int(len(members)),
                summary=get_cluster_summary(cluster_name, catalog_df=catalog_df),
                members=members,
            )

    # Fall back to spatial + kinematic crossmatch.
    result = crossmatch_target_to_clusters(
        target,
        catalog_df=catalog_df,
        catalog=catalog,
        data_path=data_path,
        min_members=min_members,
    )
    membership = result["membership"]
    sep3d_pc = result["sep3d_pc"]
    extent_pc = result["extent_pc"]
    within_extent = (
        sep3d_pc is not None and extent_pc is not None and sep3d_pc <= extent_pc
    )
    return ClusterMatch(
        matched=within_extent and (membership["probability"] or 0) > 0,
        method="crossmatch",
        cluster_name=result["cluster_name"],
        membership_probability=membership["probability"],
        catalog_probability=None,
        mahalanobis=membership["mahalanobis"],
        dof=membership["dof"],
        sep3d_pc=sep3d_pc,
        extent_pc=extent_pc,
        within_extent=within_extent,
        n_members=int(len(result["members"])),
        summary=get_cluster_summary(result["cluster_name"], catalog_df=catalog_df),
        members=result["members"],
    )
