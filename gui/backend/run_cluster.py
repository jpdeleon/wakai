#!/usr/bin/env python
"""Assess cluster membership for a target and render diagnostic plots.

Called via subprocess with a JSON payload on argv[1]. Runs in the ``wakai``
conda environment (needs pandas/pyarrow/astropy/scipy + the wakai package).

The Perren+2023 member list triggers noisy stdout (numexpr threadpool banner,
astropy parallax warnings). Only the final JSON result is allowed on the real
stdout, so everything else is redirected to stderr for the whole run.
"""
import json
import os
import sys

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# numexpr prints a banner to stdout when the host has more cores than its
# thread cap; pin the cap before pandas imports so nothing pollutes stdout.
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _plot_utils import fig_to_data_uri  # noqa: E402
from wakai.cluster import (  # noqa: E402
    ClusterMatch,
    assess_cluster_membership,
    crossmatch_target_to_clusters,
    get_cluster_summary,
    load_cluster_catalog,
    plot_rdp_pmrv,
    plot_xyz_uvw,
)
from wakai.plotting import load_plot_params  # noqa: E402

# Columns the plotting helpers need from the (out-of-catalog) target.
_TARGET_PLOT_KEYS = ("ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity")


def _assess_forced_cluster(target, catalog, cluster_name):
    """Score membership against one specific cluster, bypassing auto-selection.

    Reuses crossmatch_target_to_clusters' own Mahalanobis/separation/extent
    scoring by restricting its candidate catalog to a single named row --
    the function still "finds the nearest cluster" among its candidates,
    but there's only one, so it always resolves to `cluster_name`. Returns
    None if `cluster_name` isn't in `catalog`.
    """
    catalog_df = load_cluster_catalog(catalog=catalog)
    subset = catalog_df[catalog_df["Name"] == cluster_name]
    if subset.empty:
        return None

    result = crossmatch_target_to_clusters(target, catalog_df=subset, catalog=catalog)
    membership = result["membership"]
    sep3d_pc = result["sep3d_pc"]
    extent_pc = result["extent_pc"]
    within_extent = sep3d_pc is not None and extent_pc is not None and sep3d_pc <= extent_pc
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


def _target_series(target):
    """Build a float Series for highlighting a non-member target on the plots."""
    values = {}
    for key in _TARGET_PLOT_KEYS:
        raw = target.get(key)
        try:
            values[key] = float(raw) if raw is not None else np.nan
        except (TypeError, ValueError):
            values[key] = np.nan
    return pd.Series(values, dtype=float)


def _render_plots(match, target):
    """Return {xyz_uvw, rdp_pmrv} data URIs, or an empty dict if unplottable."""
    members = match.members
    if members is None or len(members) < 2:
        return {}

    gaia_id = target.get("gaiaid") or target.get("source_id")
    by_id = match.method == "gaia_id" and gaia_id is not None

    shared = dict(color="Prob", cmap="viridis", target_color="#ad3d3d", verbose=False)
    if by_id:
        common = dict(target_gaiaid=int(gaia_id), match_id=True, **shared)
    else:
        # Target is not in the member list: highlight it via explicit coords.
        common = dict(
            target_gaiaid=int(gaia_id) if gaia_id else 0,
            match_id=False,
            df_target=_target_series(target),
            **shared,
        )

    plots = {}
    try:
        fig_xyz, _ = plot_xyz_uvw(members.copy(), target_ms=18, marker=".", **common)
        plots["xyz_uvw"] = fig_to_data_uri(fig_xyz)
    except Exception as exc:  # noqa: BLE001 - surface as a caveat, not a crash
        plots["xyz_uvw_error"] = str(exc)

    # plot_rdp_pmrv does not accept verbose; drop it for this call.
    rdp_common = {k: v for k, v in common.items() if k != "verbose"}
    try:
        fig_rdp = plot_rdp_pmrv(members.copy(), target_ms=22, marker=".", **rdp_common)
        plots["rdp_pmrv"] = fig_to_data_uri(fig_rdp)
    except Exception as exc:  # noqa: BLE001
        plots["rdp_pmrv_error"] = str(exc)
    return plots


def _clean(value):
    """Coerce numpy scalars / NaNs into JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main():
    data = json.loads(sys.argv[1])
    target = data.get("target", data)

    catalog = data.get("catalog", "Perren2023")
    forced_cluster_name = data.get("cluster_name")

    if forced_cluster_name:
        match = _assess_forced_cluster(target, catalog, forced_cluster_name)
        if match is None:
            print(json.dumps({
                "status": "error",
                "message": f'Cluster "{forced_cluster_name}" not found in {catalog}',
            }), file=REAL_STDOUT)
            return
    else:
        match = assess_cluster_membership(target, catalog=catalog)

    summary = [
        {"label": item["label"], "value": _clean(item["value"]), "unit": item["unit"]}
        for item in match.summary
    ]

    # rcParams affect figure style; load them only for the plotting step.
    load_plot_params()
    plots = _render_plots(match, target)

    output = {
        "status": "ok",
        "catalog": catalog,
        "forced_cluster_name": forced_cluster_name,
        "matched": bool(match.matched),
        "method": match.method,
        "cluster_name": match.cluster_name,
        "membership_probability": _clean(match.membership_probability),
        "catalog_probability": _clean(match.catalog_probability),
        "mahalanobis": _clean(match.mahalanobis),
        "dof": _clean(match.dof),
        "sep3d_pc": _clean(match.sep3d_pc),
        "extent_pc": _clean(match.extent_pc),
        "within_extent": None if match.within_extent is None else bool(match.within_extent),
        "n_members": _clean(match.n_members),
        "summary": summary,
        "plots": plots,
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
