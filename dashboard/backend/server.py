#!/usr/bin/env python3
"""Flask backend for wakai dashboard — serves frontend + proxies to gyro-interp and BAFFLES."""
import json
import os
import subprocess
import sys
import urllib.request
import urllib.parse
from collections import OrderedDict

from flask import Flask, jsonify, request, send_from_directory

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)
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
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(DASHBOARD_DIR, path)

EXT_TOOLS_DIR = "/raid_ut2/home/jerome/github/research/project/ext_tools"
GYRO_ENV_PYTHON = f"{EXT_TOOLS_DIR}/gyro-interp/.venv/bin/python"
BAFFLES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/BAFFLES/.venv/bin/python"
WAKAI_ENV_PYTHON = "/ut3/jerome/miniconda3/envs/wakai/bin/python"

GYRO_WRAPPER = os.path.join(BACKEND_DIR, "run_gyro.py")
BAFFLES_WRAPPER = os.path.join(BACKEND_DIR, "run_baffles.py")
BV_LOOKUP_WRAPPER = os.path.join(BACKEND_DIR, "run_bv_lookup.py")
CLUSTER_WRAPPER = os.path.join(BACKEND_DIR, "run_cluster.py")

# gyrointerp's Bouma+2023 rotation-sequence grids are only calibrated for
# 3800-6200 K (gyrointerp.gyro_posterior.gyro_age_posterior docstring); outside
# that range the posterior is all-NaN, which breaks JSON parsing downstream.
GYRO_TEFF_MIN, GYRO_TEFF_MAX = 3800, 6200

TIMEOUT_SECONDS = 180
GYRO_TIMEOUT_SECONDS = 600
VIZIER_CACHE_MAX_TARGETS = 4
_VIZIER_TARGET_CACHE = OrderedDict()

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


@app.route("/api/run/baffles", methods=["POST"])
def run_baffles():
    data = request.get_json(force=True)
    needed = ["teff", "rhk"]
    for k in needed:
        if k not in data:
            return jsonify({"status": "error", "message": f"Missing required field: {k}"}), 400

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


def _default_parameters():
    return [empty_reference_row(spec) for spec in DEFAULT_PARAMETER_SPECS]


def _cached_catalog_star(name, ra, dec, radius, gaia_id=None):
    """Reuse downloaded VizieR tables for follow-up parameter selections."""
    key = (round(float(ra), 7), round(float(dec), 7), round(float(radius), 3))
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
               pmra, pmdec, radial_velocity,
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
    pmdec = r[6] if r[6] is not None else None
    rv = r[7] if r[7] is not None else None
    teff = r[8] if r[8] is not None else None
    teff_lo = r[9] if r[9] is not None else None
    teff_hi = r[10] if r[10] is not None else None

    teff_val = float(teff) if teff is not None else None
    teff_err = (
        (float(teff_hi) - float(teff_lo)) / 2
        if teff_hi is not None and teff_lo is not None
        else None
    )

    parameters = _default_parameters()
    teff_row = parameters[0]
    teff_row.update({
        "value": teff_val if teff is not None else None,
        "uncertainty": teff_err if teff is not None else None,
        "reference": "Gaia DR3 GSP-Phot" if teff is not None else "No catalog measurement selected",
        "catalog": "Gaia DR3" if teff is not None else "",
        "column": "teff_gspphot" if teff is not None else "",
        "unit": "K" if teff is not None else "",
        "source_url": f"https://gea.esac.esa.int/archive/?source_id={source_id}",
    })

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
    app.run(host=host, port=port, debug=False)
