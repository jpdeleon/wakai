#!/usr/bin/env python
"""Run the EVA (Excess Variability-based Age) posterior for a group of
comoving stars. Called via subprocess with JSON input on argv[1], in EVA2's
own .venv (needs the `eva2` package plus its galpy/astroquery deps).

EVA (Barber & Mann 2023, ApJ 953, 76) uses each star's Gaia photometric
excess noise -- how much its G/BP/RP flux scatters beyond Gaia's own
per-magnitude uncertainty model (EDR3_Photometric_Uncertainties, Riello+
2021) -- as a rotation/activity variability proxy, then calibrates a group's
90th-percentile excess (varX90) against age using known young moving
groups. Unlike gyro/EAGLES/BAFFLES/isochrones (single-star methods fed from
one target's ref_table row), EVA needs a *group* of comoving stars -- the
tighter the varX90 percentile, the more members -- so this wrapper takes a
list of Gaia DR3 source ids (the GUI's Cluster-page member list, or a
manually pasted comoving-group list) rather than one target's row.

This bypasses eva2's own `query()`/`gaiaQuery()` (a per-star
`astroquery.gaia.Gaia.cone_search` loop meant for reading a FriendFinder CSV
one row at a time) in favor of a single batched Gaia TAP ADQL query for the
whole group, then builds the DataFrame layout `calculate()`/`age()` expect
by hand (see EVA2/eva2/analysisFunctions.py's own gaiaQuery() for the
reference column layout this mirrors).
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _plot_utils import fig_to_data_uri  # noqa: E402

from eva2 import analysisFunctions as af  # noqa: E402

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# A comoving group's varX90 percentile is only as good as its member count;
# below this, calcVar90's np.nanpercentile/bootstrap over so few points is
# not a meaningful 90th-percentile estimate (Barber & Mann 2023 calibrate
# against groups of dozens to hundreds of members).
MIN_GROUP_MEMBERS = 5
# Bounds the ADQL IN() list size. The Monte Carlo age draw itself
# (var90Age's 50000-iteration loop per band) costs the same regardless of
# group size -- timed locally at ~15s for a 281-star group -- so this cap is
# about the TAP query, not runtime.
MAX_GROUP_MEMBERS = 300


class GaiaQueryError(Exception):
    pass


def _gaia_tap_query(adql):
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql,
    })
    url = f"{GAIA_TAP_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise GaiaQueryError(f"Gaia TAP query failed: {exc}") from exc
    columns = [c["name"] for c in payload.get("metadata", [])]
    return pd.DataFrame(payload.get("data", []), columns=columns)


def _fetch_group_photometry(gaia_ids):
    ids_sql = ",".join(str(g) for g in gaia_ids)
    adql = f"""
        SELECT source_id, ra, dec, parallax, parallax_error, ruwe,
               phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_g_n_obs,
               phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_n_obs,
               phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_n_obs,
               bp_rp
        FROM gaiadr3.gaia_source
        WHERE source_id IN ({ids_sql})
    """
    return _gaia_tap_query(adql)


def _build_group_frame(raw):
    """Reshape the Gaia TAP result into the column layout calculate()/age()
    read (gnums/gfluxs/.../bp-rp/dist/plx/...). Rows missing a BP or RP
    observation, or without a positive parallax (needed for `dist`, the
    distance-calibration term in var90Age), are dropped -- the same
    availability check gaiaQuery() does per star before appending it."""
    usable = raw[
        (raw["phot_bp_n_obs"].fillna(0) != 0)
        & (raw["phot_rp_n_obs"].fillna(0) != 0)
        & (raw["parallax"] > 0)
    ].copy()
    if usable.empty:
        return usable

    def col(name, dtype=float):
        return usable[name].astype(dtype).values

    return pd.DataFrame({
        "source_id": col("source_id", "int64"),
        "ra": col("ra"), "dec": col("dec"),
        "dist": 1000.0 / col("parallax"),
        "plx": col("parallax"), "plx_err": col("parallax_error"),
        "ruwe": col("ruwe"),
        "gmags": col("phot_g_mean_mag"), "gfluxs": col("phot_g_mean_flux"),
        "gfluxerrs": col("phot_g_mean_flux_error"), "gnums": col("phot_g_n_obs", int),
        "bpmags": col("phot_bp_mean_mag"), "bpfluxs": col("phot_bp_mean_flux"),
        "bpfluxerrs": col("phot_bp_mean_flux_error"), "bpnums": col("phot_bp_n_obs", int),
        "rpmags": col("phot_rp_mean_mag"), "rpfluxs": col("phot_rp_mean_flux"),
        "rpfluxerrs": col("phot_rp_mean_flux_error"), "rpnums": col("phot_rp_n_obs", int),
        "bp-rp": col("bp_rp"),
        "bpsnr": col("phot_bp_mean_flux") / col("phot_bp_mean_flux_error"),
        "rpsnr": col("phot_rp_mean_flux") / col("phot_rp_mean_flux_error"),
        "gsnr": col("phot_g_mean_flux") / col("phot_g_mean_flux_error"),
    }).reset_index(drop=True)


def _error_result(message):
    return {"status": "error", "message": message}


def _band_result(age, err):
    # var90Age's per-band return is (age_myr, [err_lo_myr, err_hi_myr]) --
    # both already in Myr (the calibration's own units; unlike EAGLES's
    # log10(age/yr) grid, no Jacobian conversion is needed here).
    return {"age_myr": float(age), "err_lo_myr": float(err[0]), "err_hi_myr": float(err[1])}


def _histogram_plot(group, band):
    """eva2's own plotHist(): a histogram of varX for the filtered group with
    the varX90 threshold marked. Its plotXYZ() sibling is skipped -- it
    raises inside the installed galpy/astropy versions (a bug in EVA2 itself,
    not something to patch around here), while plotHist() runs cleanly."""
    before = set(plt.get_fignums())
    af.plotHist(group, band=band)
    new_figs = set(plt.get_fignums()) - before
    if not new_figs:
        return None
    return fig_to_data_uri(plt.figure(sorted(new_figs)[0]))


def main():
    data = json.loads(sys.argv[1])

    seen = set()
    gaia_ids = []
    for raw_id in data.get("gaia_ids") or []:
        try:
            gid = int(str(raw_id).strip())
        except (TypeError, ValueError):
            continue
        if gid not in seen:
            seen.add(gid)
            gaia_ids.append(gid)

    if len(gaia_ids) < MIN_GROUP_MEMBERS:
        print(json.dumps(_error_result(
            f"EVA needs at least {MIN_GROUP_MEMBERS} comoving group members (got "
            f"{len(gaia_ids)}) for a meaningful varX90 percentile -- see Barber & Mann 2023."
        )), file=REAL_STDOUT)
        return

    truncated = len(gaia_ids) > MAX_GROUP_MEMBERS
    if truncated:
        gaia_ids = gaia_ids[:MAX_GROUP_MEMBERS]

    use_distance = bool(data.get("distance", True))
    plot_band = data.get("plot_band", "G")
    if plot_band not in ("G", "BP", "RP"):
        plot_band = "G"

    try:
        raw = _fetch_group_photometry(gaia_ids)
    except GaiaQueryError as exc:
        print(json.dumps(_error_result(str(exc))), file=REAL_STDOUT)
        return

    group = _build_group_frame(raw)
    n_queried = int(len(raw))
    n_usable = int(len(group))
    if n_usable < MIN_GROUP_MEMBERS:
        print(json.dumps(_error_result(
            f"Only {n_usable} of {len(gaia_ids)} requested stars have usable Gaia BP/RP "
            "photometry and a positive parallax -- too few for a reliable varX90 percentile."
        )), file=REAL_STDOUT)
        return

    n_pass_filters = int(af.filters(group).sum())
    if n_pass_filters < MIN_GROUP_MEMBERS:
        print(json.dumps(_error_result(
            f"Only {n_pass_filters} of {n_usable} stars pass EVA's own quality cuts "
            "(BP/RP/G S/N, parallax S/N>20, white-dwarf and color cuts -- Barber & Mann "
            "2023 Sec 3) -- too few for a reliable varX90 percentile."
        )), file=REAL_STDOUT)
        return

    try:
        af.calculate(group)  # adds varG/varBP/varRP columns in place
        (
            overall_age, overall_err,
            g_age, g_err, bp_age, bp_err, rp_age, rp_err,
        ) = af.age(group, band="all", distance=use_distance)
        var90_g, var90_g_err, var90_bp, var90_bp_err, var90_rp, var90_rp_err = af.calcVar90(group)
    except Exception as exc:  # noqa: BLE001 - surface as a caveat, not a bare 500
        print(json.dumps(_error_result(f"EVA age calculation failed: {exc}")), file=REAL_STDOUT)
        return

    plots = {}
    try:
        plot_uri = _histogram_plot(group, plot_band)
        if plot_uri:
            plots["hist"] = plot_uri
    except Exception as exc:  # noqa: BLE001
        plots["hist_error"] = str(exc)

    output = {
        "status": "ok",
        "n_input": len(gaia_ids),
        "n_truncated": truncated,
        "n_queried": n_queried,
        "n_usable": n_usable,
        "n_pass_filters": n_pass_filters,
        "median_distance_pc": float(np.nanmedian(group["dist"])),
        "distance_calibration": use_distance,
        "plot_band": plot_band,
        "overall": _band_result(overall_age, overall_err),
        "bands": {
            "G": _band_result(g_age, g_err),
            "BP": _band_result(bp_age, bp_err),
            "RP": _band_result(rp_age, rp_err),
        },
        "var90": {
            "G": float(var90_g), "G_err": float(var90_g_err),
            "BP": float(var90_bp), "BP_err": float(var90_bp_err),
            "RP": float(var90_rp), "RP_err": float(var90_rp_err),
        },
        "plots": plots,
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
