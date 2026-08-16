"""Routes that shell out to the run_*.py wrapper scripts (gyro-interp,
BAFFLES, cluster membership, MADYS CMD, isochrones2) -- each isolated in its
own conda/venv via subprocess.run(), per CLAUDE.md's "wrapper to run scripts
in different repos under ../ext_tools/" description of this whole package.
"""
import json
import os
import subprocess

from flask import Blueprint, jsonify, request

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BACKEND_DIR))

# Sibling of the wakai repo root (see CLAUDE.md: external tools live under
# ../ext_tools/), computed relative to this file rather than hardcoded so the
# GUI works from any checkout of this repo (this machine has more than one).
EXT_TOOLS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ext_tools")
GYRO_ENV_PYTHON = f"{EXT_TOOLS_DIR}/gyro-interp/.venv/bin/python"
BAFFLES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/BAFFLES/.venv/bin/python"
ISOCHRONES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/isochrones2/.venv/bin/python"
EAGLES_ENV_PYTHON = f"{EXT_TOOLS_DIR}/eagles/.venv/bin/python"
WAKAI_ENV_PYTHON = "/ut3/jerome/miniconda3/envs/wakai/bin/python"
# madys is a conda env (not a repo-local .venv under ext_tools), same as
# WAKAI_ENV_PYTHON above.
MADYS_ENV_PYTHON = "/ut3/jerome/miniconda3/envs/madys/bin/python"

GYRO_WRAPPER = os.path.join(BACKEND_DIR, "run_gyro.py")
GYRO_SLOW_SEQUENCE_WRAPPER = os.path.join(BACKEND_DIR, "run_gyro_slow_sequence.py")
BAFFLES_WRAPPER = os.path.join(BACKEND_DIR, "run_baffles.py")
BV_LOOKUP_WRAPPER = os.path.join(BACKEND_DIR, "run_bv_lookup.py")
EAGLES_WRAPPER = os.path.join(BACKEND_DIR, "run_eagles.py")
EAGLES_MODELS_WRAPPER = os.path.join(BACKEND_DIR, "run_eagles_models.py")
CLUSTER_WRAPPER = os.path.join(BACKEND_DIR, "run_cluster.py")
CLUSTER_NAMES_WRAPPER = os.path.join(BACKEND_DIR, "run_cluster_names.py")
ISOCHRONES_WRAPPER = os.path.join(BACKEND_DIR, "run_isochrones.py")
MADYS_CMD_WRAPPER = os.path.join(BACKEND_DIR, "run_madys_cmd.py")
MADYS_AGE_WRAPPER = os.path.join(BACKEND_DIR, "run_madys_age.py")

# gyrointerp's Bouma+2023 rotation-sequence grids are only calibrated for
# 3800-6200 K (gyrointerp.gyro_posterior.gyro_age_posterior docstring); outside
# that range the posterior is all-NaN, which breaks JSON parsing downstream.
GYRO_TEFF_MIN, GYRO_TEFF_MAX = 3800, 6200

# EAGLES's EW-age relation is calibrated for 3000 < Teff < 6500 K (Jeffries+
# 2023 / Weaver+2024 -- see run_eagles.py); get_li_age_v2 raises RuntimeError
# outside that range.
EAGLES_TEFF_MIN, EAGLES_TEFF_MAX = 3000, 6500
# The ML model's own grid spans Teff 3000-6590 K (see model.py), so the
# library's 6500 K ceiling is the binding constraint -- keep that one.

TIMEOUT_SECONDS = 180
GYRO_TIMEOUT_SECONDS = 600
# Polynomial evaluation against gyrointerp's cached cluster fits, not an MCMC
# run -- cheap, so a much shorter timeout than GYRO_TIMEOUT_SECONDS is fine.
GYRO_SLOW_SEQUENCE_TIMEOUT_SECONDS = 60

# The ANN model scans EAGLES's ~295k-row EW grid on every age/Teff-marginal
# step (nTeff=21 x nAge=820 when a Teff error is given), timed locally at
# ~50s for the ML model with eTeff -- the analytic model is a few seconds.
EAGLES_TIMEOUT_SECONDS = 300
# Model evaluation against the analytic relation / ANN grid for the empirical
# overlay (both models in one subprocess call), not the posterior scan above
# -- cheap, but the ML model reads the ~295k-row grid once, so keep headroom.
EAGLES_MODELS_TIMEOUT_SECONDS = 60

# Scanning the Perren+2023 member list (~1M rows) can take a few seconds even
# with predicate pushdown, so allow more headroom than the default.
CLUSTER_TIMEOUT_SECONDS = 300
# Just the cluster-level table (~18k rows), not the member list -- a few
# seconds in practice, but generous headroom since it's disk I/O bound.
CLUSTER_NAMES_TIMEOUT_SECONDS = 60

# MADYS re-queries photometry over the network for every star in the batch
# (no reuse of this app's own Gaia/2MASS cache); timed locally at ~35-45s
# for 1-25 stars and the member cap (run_madys_cmd.py's MEMBER_CAP = 100)
# keeps the worst case well under this. Requesting multiple models in one
# run adds only ~6-8s each on top of that single fetch (SampleObject.CMD()
# itself is cheap; only the photometry fetch is slow), so this budget covers
# a full multi-model, full-member-cap run too.
MADYS_CMD_TIMEOUT_SECONDS = 300

# Same ~30-45s MADYS photometry-fetch overhead as MADYS_CMD_TIMEOUT_SECONDS
# above, plus get_params() itself -- timed locally at ~3s for n_steps=[400,
# 500], save_maps=True against real photometry (run_madys_age.py's own
# N_STEPS/N_TRY). Generous headroom over that combined ~40-50s in practice.
MADYS_AGE_TIMEOUT_SECONDS = 180

# Timed locally against real Gaia+2MASS photometry: the full MCMC run
# (isochrones' own fit_mcmc_old defaults -- 1700 steps x 50 walkers) takes
# ~30s. Generous headroom over that.
ISOCHRONES_TIMEOUT_SECONDS = 180

run_bp = Blueprint("routes_run", __name__)


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


@run_bp.route("/api/run/gyro", methods=["POST"])
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


@run_bp.route("/api/gyro/slow-sequence", methods=["GET"])
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


@run_bp.route("/api/run/baffles", methods=["POST"])
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


@run_bp.route("/api/run/eagles", methods=["POST"])
def run_eagles():
    """Lithium-EW age posterior via EAGLES v2.0 (Jeffries et al. 2023;
    Weaver et al. 2024). Inputs are the star's Teff and Li EW (plus their
    uncertainties) straight from ref_table -- EAGLES models Li I 6708A EW as
    a function of Teff and age, so unlike BAFFLES no (B-V)o lookup is needed.
    `model` selects the v2 EW-age relation: "ml" (the ANN model, Weaver+2024,
    default) or "analytic" (the v2 port of the Jeffries+2023 analytical
    model). Output: the posterior over log age (native EAGLES grid plus a Myr
    resampling), EAGLES's summary stats (median age, most-probable age with
    asymmetric 68% errors, and 95% upper/lower limits when the posterior
    peaks at the grid edge), and the best-fit model isochrone for the
    EW-vs-Teff diagnostic plot.
    """
    data = request.get_json(force=True)
    teff = data.get("teff")
    liew = data.get("liew")
    if teff is None:
        return jsonify({"status": "error", "message": "Missing required field: teff"}), 400
    if liew is None:
        return jsonify({"status": "error", "message": "Missing required field: liew"}), 400

    if not (EAGLES_TEFF_MIN <= teff <= EAGLES_TEFF_MAX):
        return jsonify({
            "status": "error",
            "message": (
                f"Teff={teff} K is outside the range EAGLES is calibrated for "
                f"({EAGLES_TEFF_MIN}-{EAGLES_TEFF_MAX} K, per Jeffries+2023 / Weaver+2024). "
                "No lithium-EW age can be derived for this star."
            ),
        }), 422

    # Mirrors get_li_age_v2's own sanity check (fit.py) so the wrapper's
    # RuntimeError -- which would otherwise surface as a bare 500 -- becomes a
    # clear 4xx message instead.
    liew_err = data.get("liew_err") or 0.0
    if liew - liew_err > 800 or liew + liew_err < -200:
        return jsonify({
            "status": "error",
            "message": f"Li EW={liew} ± {liew_err} mÅ is outside a sensible range (-200..800 mÅ)",
        }), 422

    model = data.get("model", "ml")
    if model not in ("ml", "analytic"):
        return jsonify({
            "status": "error",
            "message": "model must be 'ml' (Weaver+2024 ANN) or 'analytic' (Jeffries+2023)",
        }), 400

    result = _run_wrapper(
        EAGLES_ENV_PYTHON, EAGLES_WRAPPER, {
            "teff": teff,
            "teff_err": data.get("teff_err"),
            "liew": liew,
            "liew_err": liew_err,
            "model": model,
        },
        timeout=EAGLES_TIMEOUT_SECONDS,
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


@run_bp.route("/api/eagles/models", methods=["GET"])
def eagles_models():
    """Weaver+2024 (ML) and Jeffries+2023 (analytic) empirical Li EW(Teff)
    isochrones at fixed ages, for the EAGLES page's "overplot empirical
    models" overlay. Target-independent empirical tracks, not a per-star
    posterior -- the frontend fetches this once and caches it rather than
    re-requesting per toggle, same pattern as GET /api/gyro/slow-sequence.
    Both models come back in one call; which one is drawn is a pure
    client-side choice following the page's model selector.
    """
    result = _run_wrapper(
        EAGLES_ENV_PYTHON, EAGLES_MODELS_WRAPPER, {}, timeout=EAGLES_MODELS_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)


@run_bp.route("/api/run/cluster", methods=["POST"])
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


@run_bp.route("/api/cluster/names", methods=["GET"])
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


@run_bp.route("/api/run/madys-cmd", methods=["POST"])
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


@run_bp.route("/api/run/isochrones", methods=["POST"])
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


@run_bp.route("/api/run/madys-age", methods=["POST"])
def run_madys_age():
    """Fit age/mass/radius/Teff/logg from broadband photometry via MADYS's
    SampleObject.get_params() -- PARSEC v2 by default. A second, independent
    age-fitting method alongside /api/run/isochrones (isochrones2/MIST):
    isochrones2 has no PARSEC support to add a model option to (only a MIST
    ModelGridInterpolator exists in ext_tools/isochrones2), so this calls
    MADYS directly instead. Unlike /api/run/isochrones, this re-queries its
    own photometry over the network by Gaia ID (same as /api/run/madys-cmd)
    rather than reusing this app's ref_table photometry -- MADYS's
    get_params() has no way to accept externally-supplied photometry.
    """
    data = request.get_json(force=True)
    gaia_id = str(data.get("gaiaid") or data.get("gaia_id") or "").strip()
    if not gaia_id:
        return jsonify({"status": "error", "message": "Target needs a Gaia DR3 ID"}), 400

    payload = {"gaiaid": gaia_id, "model": data.get("model") or "parsec2"}
    result = _run_wrapper(
        MADYS_ENV_PYTHON, MADYS_AGE_WRAPPER, payload, timeout=MADYS_AGE_TIMEOUT_SECONDS
    )
    if result.get("status") == "error":
        return jsonify(result), 500
    return jsonify(result)
