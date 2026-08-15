#!/usr/bin/env python
"""Color-magnitude diagram (CMD) for a target via MADYS -- the target's own
photometry against theoretical isochrone tracks, plus (when a matched
cluster name is supplied) that cluster's member stars overlaid, to visually
assess cluster youth/membership. Mirrors notebooks/ext_code_madys.ipynb.

Called via subprocess with JSON input on argv[1]. Runs in the `madys` conda
env (needs madys/pandas/pyarrow/matplotlib).

`models` (list, e.g. ["parsec2", "mist"]) draws one CMD per model, all from
a single MADYS photometry fetch -- the expensive part (~30-45s, see below)
is per-*star-list*, not per-model, so SampleObject is only built once and
reused for every model's SampleObject.CMD() call (confirmed locally: CMD()
itself takes ~1-2s once photometry is already fetched). A bad model name
only drops that one model's plot (reported in `errors`), not the whole run.

MADYS re-queries photometry over the network for every star in the batch --
no reuse of this app's own Gaia/2MASS cache -- at a fixed ~30-45s overhead
plus a small per-star cost (batching many stars in one SampleObject call is
far cheaper than one call per star; timed locally at ~45s for 25 members vs
~35s for 1). Member counts are capped (MEMBER_CAP) for a bounded worst-case
runtime, keeping the highest-probability members when a cluster has more
than the cap.

plot_ages/plot_masses (Myr / M_sun) are user-editable on the page -- see
_plot_ages()/_plot_masses() for how None (never customized) vs. an explicit
empty selection are told apart, and why an empty selection means different
things for each (mass tracks can be turned off; ages can't -- a madys
limitation, not a UI choice). The target's star marker is labeled with its
real name (`name` in the input, falling back to "Target" if absent) rather
than a generic "Target" string.
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

WAKAI_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMBERS_PARQUET = os.path.join(WAKAI_REPO_DIR, "data", "Perren2023", "UCC_members.parquet")

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import madys  # noqa: E402

from _plot_utils import fig_to_data_uri  # noqa: E402

MEMBER_CAP = 100
DEFAULT_MODEL = "parsec2"
DEFAULT_PLOT_AGES = [1, 3, 5, 10, 20, 30, 100, 200, 500, 1000]  # Myr, madys.CMD's own default
DEFAULT_PLOT_MASSES = [0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.3, 2]  # M_sun, madys.CMD's own default


def _plot_masses(values):
    """Which mass tracks to draw, matching madys.CMD's own "array or False"
    contract. `values` is whatever the frontend's checkbox selection sent: a
    non-empty list draws exactly those tracks, an empty list means the user
    unchecked everything (draw none -- confirmed safe: IsochroneGrid.
    plot_isochrones only checks plot_masses for whether to draw mass-track
    lines, nothing upstream depends on it being real values), and None
    (never customized) falls back to the default set.
    """
    if values is None:
        return DEFAULT_PLOT_MASSES
    return list(values) if values else False


def _plot_ages(values):
    """Which age isochrones to draw. Unlike _plot_masses, plot_ages=False is
    NOT safe here -- confirmed locally: IsochroneGrid.plot_isochrones always
    does `kwargs['age_range'] = plot_ages` before constructing the
    IsochroneGrid (needed to interpolate mass tracks too, not just to decide
    whether to draw isochrone lines), and passing it False either hangs or
    raises deep inside madys depending on what else is requested. So an
    empty selection here still falls back to the default set rather than
    disabling isochrones -- unlike mass tracks, "no ages" isn't a real,
    supported option in this madys version.
    """
    if values:
        return list(values)
    return DEFAULT_PLOT_AGES


def _render_cmd(sample, model, plot_ages, plot_masses, target_phot, target_name, cluster_name, member_ids):
    """One model's CMD plot as a base64 PNG data URI, or raises on failure
    (bad model name, grid range mismatch, etc.) -- caller decides whether
    that fails the whole batch or just drops this one model.
    """
    fig, ax = sample.CMD(
        "Gbp-Grp", "G", model,
        close=False, label_points=False, alpha=0.5, s=10, figsize=(8, 8),
        plot_ages=plot_ages, plot_masses=plot_masses,
    )

    target_color = target_phot["Gbp"] - target_phot["Grp"]
    target_mag = target_phot["G"]
    ax.plot(
        target_color, target_mag,
        marker="*", mec="black", mfc="red", ms=22, mew=1.2,
        linestyle="none", label=target_name, zorder=10,
    )
    ax.legend(loc="best")
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$G$")
    ax.set_title(f"{cluster_name} | {model.upper()} isochrones" if member_ids else f"{model.upper()} isochrones")

    return fig_to_data_uri(fig)


def _load_cluster_member_ids(cluster_name):
    """Up to MEMBER_CAP Gaia DR3 source ids for `cluster_name`, highest
    membership probability first. Reads the raw parquet columns directly
    (name/Source/probs -- same as wakai.cluster.load_cluster_members's
    _MEMBER_COLUMN_MAP) rather than importing the wakai package itself,
    which isn't guaranteed to be importable from the madys conda env.
    """
    if not cluster_name or not os.path.exists(MEMBERS_PARQUET):
        return []
    df = pd.read_parquet(
        MEMBERS_PARQUET,
        columns=["name", "Source", "probs"],
        filters=[("name", "==", str(cluster_name))],
    )
    if df.empty:
        return []
    df = df.sort_values("probs", ascending=False).head(MEMBER_CAP)
    return df["Source"].astype(str).tolist()


def main():
    data = json.loads(sys.argv[1])
    gaia_id = str(data.get("gaiaid") or data.get("gaia_id") or "").strip()
    if not gaia_id:
        print(json.dumps({"status": "error", "message": "Target needs a Gaia DR3 ID"}), file=REAL_STDOUT)
        return

    models = [str(m) for m in data.get("models") or []] or [str(data.get("model") or DEFAULT_MODEL)]
    # De-dupe while preserving order -- a repeated checkbox value would
    # otherwise render (and cost render time for) the same plot twice.
    models = list(dict.fromkeys(models))
    cluster_name = data.get("cluster_name")
    member_ids = _load_cluster_member_ids(cluster_name)
    target_name = str(data.get("name") or "").strip() or "Target"
    plot_ages = _plot_ages(data.get("plot_ages"))
    plot_masses = _plot_masses(data.get("plot_masses"))

    target_query_id = f"Gaia DR3 {gaia_id}"
    # Target goes last -- order doesn't otherwise matter, and putting it
    # last keeps the "insert the host, which usually isn't itself a listed
    # member" logic simple, mirroring the notebook's star_list2.append(...).
    star_list = [f"Gaia DR3 {i}" for i in member_ids] + [target_query_id]

    try:
        sample = madys.SampleObject(star_list, id_type="DR3", ext_map="leike", verbose=False)
    except Exception as exc:  # noqa: BLE001 -- network/catalog failures, report cleanly
        print(json.dumps({"status": "error", "message": f"MADYS photometry fetch failed: {exc}"}), file=REAL_STDOUT)
        return

    resolved_ids = list(sample.ID["ID"])
    try:
        target_index = resolved_ids.index(target_query_id)
    except ValueError:
        print(json.dumps({"status": "error", "message": "MADYS could not resolve this target's photometry"}), file=REAL_STDOUT)
        return

    filters = list(sample.filters)
    target_phot = dict(zip(filters, sample.abs_phot[target_index]))
    if "Gbp" not in target_phot or "Grp" not in target_phot or "G" not in target_phot:
        print(json.dumps({"status": "error", "message": "MADYS returned no Gaia photometry for this target"}), file=REAL_STDOUT)
        return

    plots = {}
    errors = {}
    for model in models:
        try:
            plots[model] = _render_cmd(
                sample, model, plot_ages, plot_masses,
                target_phot, target_name, cluster_name, member_ids,
            )
        except Exception as exc:  # noqa: BLE001 -- bad model name, grid range mismatch, etc.
            errors[model] = str(exc)

    if not plots:
        print(json.dumps({
            "status": "error",
            "message": f"MADYS CMD plot failed for every selected model: {errors}",
        }), file=REAL_STDOUT)
        return

    output = {
        "status": "ok",
        "models": list(plots.keys()),
        "cluster_name": cluster_name,
        "n_members": len(member_ids),
        "member_cap": MEMBER_CAP,
        "plot_ages": plot_ages if plot_ages else [],
        "plot_masses": plot_masses if plot_masses else [],
        "plots": plots,
        "errors": errors,
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
