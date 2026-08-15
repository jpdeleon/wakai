#!/usr/bin/env python3
"""Flask backend for wakai GUI — serves frontend + proxies to gyro-interp and BAFFLES."""
import json
import os
import subprocess
import sys
import threading
import urllib.request
import urllib.parse
from collections import OrderedDict

from flask import Flask, jsonify, request, send_from_directory

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_DIR = os.path.dirname(GUI_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from wakai.reference import (  # noqa: E402
    DEFAULT_PARAMETER_SPECS,
    empty_reference_row,
    query_vizier_reference,
    simbad_target_payload,
)
from wakai.database import TargetDatabase  # noqa: E402
from wakai.star import Star  # noqa: E402

app = Flask(__name__, static_folder=None)

# Where pipeline results are persisted. Defaults to ~/ql/wakai so results live
# outside the repo; the GUI launcher (wakai gui --output-dir / WAKAI_OUTPUT_DIR)
# is the only place this is set — the pipeline wrapper scripts stay path-agnostic.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "ql", "wakai")
OUTPUT_DIR = os.environ.get("WAKAI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
app.config["OUTPUT_DIR"] = OUTPUT_DIR
app.config["DATABASE_PATH"] = os.environ.get(
    "WAKAI_DATABASE_PATH", os.path.join(OUTPUT_DIR, "wakai.sqlite3")
)

# Serve static frontend files at root
@app.route("/")
def index():
    return send_from_directory(GUI_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    # SPA fallback: real static assets (app.js, styles.css, favicons, ...)
    # are served as-is; anything else is a client-side route (/gyro-interp,
    # /target=TOI-5973, ...) that app.js's router resolves from index.html.
    # send_from_directory still guards both branches against path traversal
    # via its own safe_join, regardless of this pre-check.
    if os.path.isfile(os.path.join(GUI_DIR, path)):
        return send_from_directory(GUI_DIR, path)
    return send_from_directory(GUI_DIR, "index.html")

# Sibling of the wakai repo root (see CLAUDE.md: external tools live under
# ../ext_tools/), computed relative to this file rather than hardcoded so the
# GUI works from any checkout of this repo (this machine has more than one).
EXT_TOOLS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ext_tools")
GYRO_ENV_PYTHON = f"{EXT_TOOLS_DIR}/gyro-interp/.venv/bin/python"
BAFFLES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/BAFFLES/.venv/bin/python"
ISOCHRONES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/isochrones2/.venv/bin/python"
WAKAI_ENV_PYTHON = "/ut3/jerome/miniconda3/envs/wakai/bin/python"
# madys is a conda env (not a repo-local .venv under ext_tools), same as
# WAKAI_ENV_PYTHON above.
MADYS_ENV_PYTHON = "/ut3/jerome/miniconda3/envs/madys/bin/python"

GYRO_WRAPPER = os.path.join(BACKEND_DIR, "run_gyro.py")
GYRO_SLOW_SEQUENCE_WRAPPER = os.path.join(BACKEND_DIR, "run_gyro_slow_sequence.py")
BAFFLES_WRAPPER = os.path.join(BACKEND_DIR, "run_baffles.py")
BV_LOOKUP_WRAPPER = os.path.join(BACKEND_DIR, "run_bv_lookup.py")
CLUSTER_WRAPPER = os.path.join(BACKEND_DIR, "run_cluster.py")
CLUSTER_NAMES_WRAPPER = os.path.join(BACKEND_DIR, "run_cluster_names.py")
ISOCHRONES_WRAPPER = os.path.join(BACKEND_DIR, "run_isochrones.py")
MADYS_CMD_WRAPPER = os.path.join(BACKEND_DIR, "run_madys_cmd.py")

# gyrointerp's Bouma+2023 rotation-sequence grids are only calibrated for
# 3800-6200 K (gyrointerp.gyro_posterior.gyro_age_posterior docstring); outside
# that range the posterior is all-NaN, which breaks JSON parsing downstream.
GYRO_TEFF_MIN, GYRO_TEFF_MAX = 3800, 6200

TIMEOUT_SECONDS = 180
GYRO_TIMEOUT_SECONDS = 600
# Polynomial evaluation against gyrointerp's cached cluster fits, not an MCMC
# run -- cheap, so a much shorter timeout than GYRO_TIMEOUT_SECONDS is fine.
GYRO_SLOW_SEQUENCE_TIMEOUT_SECONDS = 60
VIZIER_CACHE_MAX_TARGETS = 4
_VIZIER_TARGET_CACHE = OrderedDict()
# Guards _VIZIER_TARGET_CACHE's pop/insert/evict sequence, which isn't safe
# under concurrent requests now that the dev server runs threaded=True.
_VIZIER_CACHE_LOCK = threading.Lock()

def _run_wrapper(python_bin, wrapper_script, input_data, timeout=TIMEOUT_SECONDS):
    input_json = json.dumps(input_data)
    try:
        result = subprocess.run(
            [python_bin, wrapper_script, input_json],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return {"status": "error", "message": stderr or "Unknown error"}
        output = json.loads(result.stdout)
        output["_stderr"] = result.stderr.strip() if result.stderr else ""
        return output
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Process timed out after {timeout}s"}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON parse error: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "wakai backend"})


def _target_database():
    return TargetDatabase(app.config["DATABASE_PATH"])


@app.route("/api/targets/save", methods=["POST"])
def save_target():
    """Persist one target's metadata, ref_table, and model results."""
    data = request.get_json(force=True)
    target = data.get("target")
    ref_table = data.get("ref_table")
    results = data.get("results")
    if not isinstance(target, dict):
        return jsonify({"status": "error", "message": "target must be an object"}), 400
    if not isinstance(ref_table, list):
        return jsonify({"status": "error", "message": "ref_table must be a list"}), 400
    if not isinstance(results, dict):
        return jsonify({"status": "error", "message": "results must be an object"}), 400
    try:
        key = _target_database().save(target, ref_table, results)
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database save failed: {exc}"}), 500
    return jsonify({"status": "ok", "key": key})


@app.route("/api/targets/load", methods=["GET"])
def load_target():
    """Load a target snapshot by Gaia ID or normalized name."""
    gaia_id = str(request.args.get("gaia_id") or "").strip() or None
    name = str(request.args.get("name") or "").strip() or None
    if not gaia_id and not name:
        return jsonify({"status": "error", "message": "Provide gaia_id or name"}), 400
    try:
        record = _target_database().load(gaia_id=gaia_id, name=name)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database load failed: {exc}"}), 500
    if record is None:
        return jsonify({"status": "error", "message": "No saved results for target"}), 404
    return jsonify({"status": "ok", "record": record})


@app.route("/api/targets", methods=["GET"])
def list_targets():
    """List saved targets, most recently updated first."""
    try:
        targets = _target_database().list_targets()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database query failed: {exc}"}), 500
    return jsonify({"status": "ok", "targets": targets})


@app.route("/api/run/gyro", methods=["POST"])
def run_gyro():
    data = request.get_json(force=True)
    needed = ["teff", "prot"]
    for k in needed:
        if k not in data:
            return jsonify({"status": "error", "message": f"Missing required field: {k}"}), 400

    teff = data["teff"]
    if not (GYRO_TEFF_MIN <= teff <= GYRO_TEFF_MAX):
        return jsonify({
            "status": "error",
            "message": (
                f"Teff={teff} K is outside the range gyrochronology is calibrated for "
                f"({GYRO_TEFF_MIN}-{GYRO_TEFF_MAX} K, per Bouma+2023 rotation-sequence grids). "
                "No gyro age can be derived for this star."
            ),
        }), 422

    result = _run_wrapper(GYRO_ENV_PYTHON, GYRO_WRAPPER, data, timeout=GYRO_TIMEOUT_SECONDS)
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


@app.route("/api/gyro/slow-sequence", methods=["GET"])
def gyro_slow_sequence():
    """Bouma+2023 empirical slow-sequence Prot(Teff) isochrones for the
    gyro-interp page's "overplot empirical models" overlay. Target-independent
    reference-cluster curves, not a per-star posterior -- the frontend fetches
    this once and caches it rather than re-requesting per toggle."""
    result = _run_wrapper(
        GYRO_ENV_PYTHON, GYRO_SLOW_SEQUENCE_WRAPPER, {}, timeout=GYRO_SLOW_SEQUENCE_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


@app.route("/api/run/baffles", methods=["POST"])
def run_baffles():
    data = request.get_json(force=True)
    if "teff" not in data:
        return jsonify({"status": "error", "message": "Missing required field: teff"}), 400
    # rhk (activity) and liew (lithium) are each individually optional -- a
    # star can be missing one measurement, or have a B-V outside one
    # indicator's calibrated range (see run_baffles.py) -- but at least one
    # is needed to compute anything.
    if data.get("rhk") is None and data.get("liew") is None:
        return jsonify({
            "status": "error",
            "message": "Provide at least one of log R'HK (activity) or Li EW (lithium)",
        }), 400

    bv_lookup = _run_wrapper(WAKAI_ENV_PYTHON, BV_LOOKUP_WRAPPER, {
        "teff": data["teff"],
        "teff_err": data.get("teff_err"),
        "feh": data.get("feh"),
    })
    if bv_lookup.get("status") == "error":
        return jsonify({
            "status": "error",
            "message": f"(B-V)o lookup from Teff failed: {bv_lookup.get('message')}",
        }), 500

    baffles_input = dict(data)
    baffles_input["bv"] = bv_lookup["bv"]
    baffles_input["bv_err"] = bv_lookup["bv_err"]

    result = _run_wrapper(BAFFLES_ENV_PYTHON, BAFFLES_WRAPPER, baffles_input)
    if result.get("status") == "error":
        return jsonify(result), 500

    result["bv_lookup"] = {
        "bv": bv_lookup["bv"],
        "bv_err": bv_lookup["bv_err"],
        "out_of_range": bv_lookup.get("out_of_range", False),
        "caveat": bv_lookup.get("caveat"),
        "source": "Pecaut & Mamajek EEM dwarf table",
    }
    return jsonify(result)


# Scanning the Perren+2023 member list (~1M rows) can take a few seconds even
# with predicate pushdown, so allow more headroom than the default.
CLUSTER_TIMEOUT_SECONDS = 300
# Just the cluster-level table (~18k rows), not the member list -- a few
# seconds in practice, but generous headroom since it's disk I/O bound.
CLUSTER_NAMES_TIMEOUT_SECONDS = 60


@app.route("/api/run/cluster", methods=["POST"])
def run_cluster():
    """Assess whether a target belongs to a known open cluster.

    Checks the target's Gaia DR3 id against the cluster catalog (Perren+2023 by
    default); if absent, crossmatches against every cluster core and scores
    membership. Returns cluster properties plus XYZ-UVW and RA/Dec-PM diagnostic
    plots (base64 PNGs).
    """
    data = request.get_json(force=True)
    target = data.get("target")
    if not isinstance(target, dict):
        return jsonify({"status": "error", "message": "target must be an object"}), 400

    has_id = bool(str(target.get("gaiaid") or target.get("source_id") or "").strip())
    has_coords = all(
        target.get(k) is not None for k in ("ra", "dec", "parallax")
    )
    if not has_id and not has_coords:
        return jsonify({
            "status": "error",
            "message": "Provide a Gaia ID, or ra/dec/parallax for a crossmatch",
        }), 400

    result = _run_wrapper(
        WAKAI_ENV_PYTHON, CLUSTER_WRAPPER, data, timeout=CLUSTER_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


@app.route("/api/cluster/names", methods=["GET"])
def cluster_names():
    """List every cluster name in a catalog, for the Cluster page's "cluster
    to plot" search box. Target-independent -- the frontend fetches this
    once and caches it, same pattern as GET /api/gyro/slow-sequence."""
    catalog = request.args.get("catalog", "Perren2023")
    result = _run_wrapper(
        WAKAI_ENV_PYTHON, CLUSTER_NAMES_WRAPPER, {"catalog": catalog}, timeout=CLUSTER_NAMES_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


# MADYS re-queries photometry over the network for every star in the batch
# (no reuse of this app's own Gaia/2MASS cache); timed locally at ~35-45s
# for 1-25 stars and the member cap (run_madys_cmd.py's MEMBER_CAP = 100)
# keeps the worst case well under this. Requesting multiple models in one
# run adds only ~6-8s each on top of that single fetch (SampleObject.CMD()
# itself is cheap; only the photometry fetch is slow), so this budget covers
# a full multi-model, full-member-cap run too.
MADYS_CMD_TIMEOUT_SECONDS = 300


@app.route("/api/run/madys-cmd", methods=["POST"])
def run_madys_cmd():
    """Color-magnitude diagram(s) for a target via MADYS -- the target
    plotted against theoretical isochrone tracks, plus (if cluster_name is
    given, typically the Cluster page's own matched cluster) that cluster's
    member stars overlaid for a visual youth/membership check. Independent
    of the Cluster page's own "Check cluster membership" run -- cluster_name
    is optional, and the plot still renders target-only without it.
    plot_ages/plot_masses (Myr/M_sun) are optional too -- which isochrone/
    mass tracks to draw, user-editable on the page; omitted or null falls
    back to run_madys_cmd.py's own defaults (matching madys.CMD's own
    defaults). `name` labels the target's star marker on the plot (falls
    back to "Target" if not given). `models` (list, e.g. ["parsec2",
    "mist"]) renders one CMD per model from a single shared photometry
    fetch -- must be non-empty if given at all; omitted falls back to
    run_madys_cmd.py's single default model.
    """
    data = request.get_json(force=True)
    gaia_id = str(data.get("gaiaid") or data.get("gaia_id") or "").strip()
    if not gaia_id:
        return jsonify({"status": "error", "message": "Target needs a Gaia DR3 ID"}), 400

    models = data.get("models")
    if models is not None and not models:
        return jsonify({"status": "error", "message": "Select at least one isochrone model"}), 400

    payload = {
        "gaiaid": gaia_id,
        "name": data.get("name") or None,
        "cluster_name": data.get("cluster_name") or None,
        "models": models,
        "plot_ages": data.get("plot_ages"),
        "plot_masses": data.get("plot_masses"),
    }
    result = _run_wrapper(
        MADYS_ENV_PYTHON, MADYS_CMD_WRAPPER, payload, timeout=MADYS_CMD_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


# Timed locally against real Gaia+2MASS photometry: the full MCMC run
# (isochrones' own fit_mcmc_old defaults -- 1700 steps x 50 walkers) takes
# ~30s. Generous headroom over that.
ISOCHRONES_TIMEOUT_SECONDS = 180


@app.route("/api/run/isochrones", methods=["POST"])
def run_isochrones():
    """Fit age/mass/radius/Teff/logg/feh from photometry + parallax via
    isochrones2's SingleStarModel (MIST grids). Photometry (Gaia BP/G/RP,
    2MASS J/H/Ks), parallax, Teff and Fe/H all come straight from the
    frontend -- they're sourced from the target's ref_table (populated once
    at target-resolve time by /api/query/gaia, not re-fetched here) and the
    frontend includes only the bands/values whose checkbox the user has
    ticked on the Isochrones page. logg is not a ref_table measurement -- it's
    a fixed dwarf-star prior (default 4.5+/-0.1, matching
    notebooks/ext_code_isochrones.ipynb) that's editable on the page and
    included the same optional way. age_prior_bounds/av_prior_bounds mirror
    the notebook's mod.set_prior(age=FlatPrior(...), AV=FlatPrior(...)) --
    also user-editable, defaulting in run_isochrones.py to the notebook's
    own (5, 10) / (0, 0.1) if omitted. eep_prior_bounds restricts
    SingleStarModel's fourth fit parameter (Equivalent Evolutionary Point,
    not touched by the notebook) -- also optional, defaulting to the MIST
    grid's own full range if omitted.
    """
    data = request.get_json(force=True)
    photometry = data.get("photometry") or {}
    parallax = data.get("parallax")
    parallax_err = data.get("parallax_err")
    if not photometry or parallax is None or parallax_err is None:
        return jsonify({
            "status": "error",
            "message": "Select at least one photometric band plus parallax to run isochrones",
        }), 400

    payload = {
        "photometry": photometry,
        "parallax": parallax,
        "parallax_err": parallax_err,
    }
    if data.get("teff") is not None and data.get("teff_err") is not None:
        payload["teff"] = data["teff"]
        payload["teff_err"] = data["teff_err"]
    if data.get("feh") is not None and data.get("feh_err") is not None:
        payload["feh"] = data["feh"]
        payload["feh_err"] = data["feh_err"]
    if data.get("logg") is not None and data.get("logg_err") is not None:
        payload["logg"] = data["logg"]
        payload["logg_err"] = data["logg_err"]

    for bounds_key in ("age_prior_bounds", "av_prior_bounds", "eep_prior_bounds"):
        bounds = data.get(bounds_key)
        if bounds is None:
            continue
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            return jsonify({"status": "error", "message": f"{bounds_key} must be [min, max]"}), 400
        if lo >= hi:
            return jsonify({"status": "error", "message": f"{bounds_key}: min must be less than max"}), 400
        payload[bounds_key] = [lo, hi]

    result = _run_wrapper(
        ISOCHRONES_ENV_PYTHON, ISOCHRONES_WRAPPER, payload, timeout=ISOCHRONES_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"


def _gaia_tap_query(adql: str) -> list[dict] | None:
    """Execute ADQL query against Gaia TAP service, return rows as dicts."""
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "json",
        "QUERY": adql,
    })
    url = f"{GAIA_TAP_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("data")
    except Exception:
        return None


# 2.5/ln(10), the standard Gaia photometric error propagation constant:
# sigma_mag = MAG_ERR_CONST / (flux / flux_error).
MAG_ERR_CONST = 1.0857362047581294


def _fetch_gaia_2mass_photometry(gaia_id):
    """Fetch parallax plus Gaia BP/G/RP and 2MASS J/H/Ks magnitudes
    (+errors) for the isochrones2 page's SingleStarModel fit, via Gaia's
    own official 2MASS crossmatch table (gaiadr3.tmass_psc_xsc_best_neighbour)
    -- one query, no separate VizieR round trip, and parallax comes from the
    same source row as the photometry rather than a second query. Returns
    {"photometry": {}, "parallax": None, "parallax_err": None} if the source
    isn't found; omits any individual band that's missing (no 2MASS match,
    or that band wasn't observed) rather than failing the whole fetch --
    SingleStarModel only needs whichever bands are actually available.
    """
    adql = f"""
        SELECT g.parallax, g.parallax_error,
               g.phot_bp_mean_mag, g.phot_bp_mean_flux_over_error,
               g.phot_g_mean_mag, g.phot_g_mean_flux_over_error,
               g.phot_rp_mean_mag, g.phot_rp_mean_flux_over_error,
               tm.j_m, tm.j_msigcom, tm.h_m, tm.h_msigcom, tm.ks_m, tm.ks_msigcom
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch ON g.source_id = xmatch.source_id
        JOIN gaiadr1.tmass_original_valid AS tm ON xmatch.original_ext_source_id = tm.designation
        WHERE g.source_id = {int(gaia_id)}
    """
    rows = _gaia_tap_query(adql)
    if not rows:
        return {"photometry": {}, "parallax": None, "parallax_err": None}
    r = rows[0]

    def from_snr(mag, snr):
        if mag is None or snr is None or snr <= 0:
            return None
        return [float(mag), MAG_ERR_CONST / float(snr)]

    def from_err(mag, err):
        return [float(mag), float(err)] if mag is not None and err is not None else None

    photometry = {
        "BP": from_snr(r[2], r[3]),
        "G": from_snr(r[4], r[5]),
        "RP": from_snr(r[6], r[7]),
        "J": from_err(r[8], r[9]),
        "H": from_err(r[10], r[11]),
        "K": from_err(r[12], r[13]),
    }
    return {
        "photometry": {band: value for band, value in photometry.items() if value is not None},
        "parallax": float(r[0]) if r[0] is not None else None,
        "parallax_err": float(r[1]) if r[1] is not None else None,
    }


def _default_parameters():
    return [empty_reference_row(spec) for spec in DEFAULT_PARAMETER_SPECS]


def _set_parameter_row(parameters, param_id, value, uncertainty, unit, reference, catalog, column, source_url):
    """Find `param_id`'s row (by DEFAULT_PARAMETER_SPECS id) in `parameters`
    and fill it in, if a real value was found -- looked up by id rather than
    a hardcoded list position so callers don't need to know row order."""
    row = next((r for r in parameters if r["id"] == param_id), None)
    if row is None or value is None:
        return row
    row.update({
        "value": value,
        "uncertainty": uncertainty,
        "reference": reference,
        "catalog": catalog,
        "column": column,
        "unit": unit,
        "source_url": source_url,
    })
    return row


def _cached_catalog_star(name, ra, dec, radius, gaia_id=None):
    """Reuse downloaded VizieR tables for follow-up parameter selections."""
    key = (round(float(ra), 7), round(float(dec), 7), round(float(radius), 3))
    with _VIZIER_CACHE_LOCK:
        star = _VIZIER_TARGET_CACHE.pop(key, None)
        if star is None:
            star = Star(
                name or str(gaia_id or "target"),
                ra_deg=float(ra),
                dec_deg=float(dec),
                gaiaid=gaia_id,
                verbose=False,
            )
        else:
            star.star_name = name or star.star_name
            star.gaiaid = gaia_id or star.gaiaid
        _VIZIER_TARGET_CACHE[key] = star
        while len(_VIZIER_TARGET_CACHE) > VIZIER_CACHE_MAX_TARGETS:
            _VIZIER_TARGET_CACHE.popitem(last=False)
    return star


@app.route("/api/query/simbad", methods=["POST"])
def query_simbad():
    """Resolve a target name and return its SIMBAD identifiers and coordinates."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Provide a target name"}), 400

    try:
        payload = simbad_target_payload(Star(name, verbose=False))
    except Exception as exc:
        return jsonify({"status": "error", "message": f"SIMBAD query failed: {exc}"}), 502
    if payload is None:
        return jsonify({"status": "error", "message": f"SIMBAD could not resolve {name}"}), 404
    return jsonify({"status": "ok", "target": payload})


@app.route("/api/query/vizier", methods=["POST"])
def query_vizier():
    """Query VizieR near a target and build selectable reference rows."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    ra = data.get("ra")
    dec = data.get("dec")
    parameter = str(data.get("parameter") or "").strip() or None
    try:
        radius = float(data.get("radius", 3))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Radius must be numeric"}), 400
    if radius <= 0 or radius > 60:
        return jsonify({"status": "error", "message": "Radius must be between 0 and 60 arcsec"}), 400

    if ra is None or dec is None:
        if not name:
            return jsonify({"status": "error", "message": "Provide target coordinates or a name"}), 400
        try:
            resolved = simbad_target_payload(Star(name, verbose=False))
        except Exception as exc:
            return jsonify({"status": "error", "message": f"SIMBAD query failed: {exc}"}), 502
        if resolved is None:
            return jsonify({"status": "error", "message": f"SIMBAD could not resolve {name}"}), 404
        ra, dec = resolved["ra"], resolved["dec"]

    try:
        star = _cached_catalog_star(name, ra, dec, radius, data.get("gaia_id"))
        result = query_vizier_reference(star, parameter=parameter, radius=radius)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"VizieR query failed: {exc}"}), 502

    return jsonify({
        "status": "ok",
        "target": {"name": star.star_name, "ra": star.ra_deg, "dec": star.dec_deg},
        **result,
    })


@app.route("/api/query/gaia", methods=["POST"])
def query_gaia():
    """Query Gaia DR3 by source_id or target name."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    gaia_id = str(data.get("gaia_id") or "").strip()
    ra = data.get("ra")
    dec = data.get("dec")

    if gaia_id:
        if not gaia_id.isdigit():
            return jsonify({"status": "error", "message": "Gaia ID must be numeric"}), 400
        where = f"source_id = {gaia_id}"
    elif ra is not None and dec is not None:
        try:
            ra, dec = float(ra), float(dec)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "RA and Dec must be numeric"}), 400
        where = (
            "1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {3 / 3600}))"
        )
    elif name:
        # Try to resolve via Gaia DR3 or fallback to Simbad-like name search
        # Use a broad cone search or name match
        try:
            numeric_id = int(name)
            where = f"source_id = {numeric_id}"
        except ValueError:
            where = f"UPPER(designation) LIKE UPPER('%{name.replace(chr(39), '')}%')"
    else:
        return jsonify({"status": "error", "message": "Provide 'name' or 'gaia_id'"}), 400

    adql = f"""
        SELECT TOP 1 source_id, ra, dec, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error,
               radial_velocity, radial_velocity_error,
               teff_gspphot, teff_gspphot_lower, teff_gspphot_upper
        FROM gaiadr3.gaia_source
        WHERE {where}
    """
    rows = _gaia_tap_query(adql)
    if not rows:
        return jsonify({"status": "error", "message": "No match found in Gaia DR3"}), 404

    r = rows[0]
    source_id = str(r[0])
    ra = r[1] if r[1] is not None else 0.0
    dec = r[2] if r[2] is not None else 0.0
    parallax = r[3] if r[3] is not None else None
    parallax_err = r[4] if r[4] is not None else None
    pmra = r[5] if r[5] is not None else None
    pmra_err = r[6] if r[6] is not None else None
    pmdec = r[7] if r[7] is not None else None
    pmdec_err = r[8] if r[8] is not None else None
    rv = r[9] if r[9] is not None else None
    rv_err = r[10] if r[10] is not None else None
    teff = r[11] if r[11] is not None else None
    teff_lo = r[12] if r[12] is not None else None
    teff_hi = r[13] if r[13] is not None else None

    teff_val = float(teff) if teff is not None else None
    teff_err = (
        (float(teff_hi) - float(teff_lo)) / 2
        if teff_hi is not None and teff_lo is not None
        else None
    )

    gaia_source_url = f"https://gea.esac.esa.int/archive/?source_id={source_id}"
    parameters = _default_parameters()
    _set_parameter_row(
        parameters, "teff", teff_val, teff_err, "K",
        "Gaia DR3 GSP-Phot", "Gaia DR3", "teff_gspphot", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "parallax", float(parallax) if parallax is not None else None,
        float(parallax_err) if parallax_err is not None else None, "mas",
        "Gaia DR3", "Gaia DR3", "parallax", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "pmra", float(pmra) if pmra is not None else None,
        float(pmra_err) if pmra_err is not None else None, "mas/yr",
        "Gaia DR3", "Gaia DR3", "pmra", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "pmdec", float(pmdec) if pmdec is not None else None,
        float(pmdec_err) if pmdec_err is not None else None, "mas/yr",
        "Gaia DR3", "Gaia DR3", "pmdec", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "rv", float(rv) if rv is not None else None,
        float(rv_err) if rv_err is not None else None, "km/s",
        "Gaia DR3", "Gaia DR3", "radial_velocity", gaia_source_url,
    )

    # Photometry (Gaia BP/G/RP + 2MASS J/H/Ks via Gaia's own crossmatch
    # table) for the Isochrones page -- fetched here, once, at target-resolve
    # time, rather than re-fetched on every isochrones run.
    photometry = _fetch_gaia_2mass_photometry(source_id)["photometry"]
    for band, param_id, unit, reference, catalog in (
        ("BP", "bpmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("G", "gmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("RP", "rpmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("J", "jmag", "mag", "2MASS", "2MASS"),
        ("H", "hmag", "mag", "2MASS", "2MASS"),
        ("K", "kmag", "mag", "2MASS", "2MASS"),
    ):
        value = photometry.get(band)
        if value is not None:
            _set_parameter_row(
                parameters, param_id, value[0], value[1], unit,
                reference, catalog, band, gaia_source_url,
            )

    return jsonify({
        "status": "ok",
        "target": {
            "name": name or source_id,
            "gaiaid": source_id,
            "ra": float(ra),
            "dec": float(dec),
            "parallax": float(parallax) if parallax is not None else None,
            "parallax_error": float(parallax_err) if parallax_err is not None else None,
            "pmra": float(pmra) if pmra is not None else None,
            "pmdec": float(pmdec) if pmdec is not None else None,
            "rv": float(rv) if rv is not None else None,
            "parameters": parameters,
        },
    })


if __name__ == "__main__":
    host = os.environ.get("WAKAI_HOST", "127.0.0.1")
    port = int(os.environ.get("WAKAI_PORT", 7000))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting wakai backend on {host}:{port}")
    print(f"  results dir:         {OUTPUT_DIR}")
    print(f"  gyro-interp wrapper: {GYRO_WRAPPER}")
    print(f"  BAFFLES wrapper:     {BAFFLES_WRAPPER}")
    app.run(host=host, port=port, debug=False, threaded=True)
