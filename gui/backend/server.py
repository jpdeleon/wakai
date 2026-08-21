#!/usr/bin/env python3
"""Flask backend for wakai GUI -- serves the SPA and registers the /api/*
route blueprints.

Route handlers live in sibling modules, grouped by concern:
  - routes_query.py   Target resolution: SIMBAD, VizieR, Gaia DR3 (uses gaia_client.py)
  - routes_run.py      Subprocess-isolated science: gyro-interp, BAFFLES, cluster,
                       MADYS CMD, isochrones2 (each in its own conda/venv)
  - routes_targets.py  SQLite persistence of target snapshots (wakai/database.py)
This file itself only owns app setup, static file serving, and error handling.
"""
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_DIR = os.path.dirname(GUI_DIR)
# Needed for the bare sibling-module imports below (routes_query, routes_run,
# routes_targets, gaia_client) to resolve both when this file is run directly
# (python server.py -- sys.path[0] is already BACKEND_DIR then) and when it's
# imported as the gui.backend.server package submodule (e.g. by pytest, where
# sys.path[0] is not BACKEND_DIR), matching the sys.path pattern already used
# by every run_*.py wrapper script for the same reason.
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from routes_query import query_bp  # noqa: E402
from routes_run import BAFFLES_WRAPPER, GYRO_WRAPPER, run_bp  # noqa: E402
from routes_targets import targets_bp  # noqa: E402

app = Flask(__name__, static_folder=None)
app.register_blueprint(query_bp)
app.register_blueprint(run_bp)
app.register_blueprint(targets_bp)

# Where pipeline results are persisted. Defaults to ~/ql/wakai so results live
# outside the repo; the GUI launcher (wakai gui --output-dir / WAKAI_OUTPUT_DIR)
# is the only place this is set — the pipeline wrapper scripts stay path-agnostic.
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "ql", "wakai")
OUTPUT_DIR = os.environ.get("WAKAI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
app.config["OUTPUT_DIR"] = OUTPUT_DIR
app.config["DATABASE_PATH"] = os.environ.get(
    "WAKAI_DATABASE_PATH", os.path.join(OUTPUT_DIR, "wakai.sqlite3")
)


@app.errorhandler(Exception)
def _handle_unexpected_error(error):
    """Every /api/* route's success path returns JSON; make sure its failure
    path does too. Without this, an uncaught exception (e.g. a route doing
    arithmetic/comparison on unvalidated request JSON, like GYRO_TEFF_MIN <=
    teff <= GYRO_TEFF_MAX with a non-numeric teff) falls through to Flask's
    default HTML error page instead of the {"status": "error", ...} shape
    every caller already expects."""
    if isinstance(error, HTTPException):
        return jsonify({"status": "error", "message": error.description}), error.code
    app.logger.exception("Unhandled error in %s %s", request.method, request.path)
    return jsonify({"status": "error", "message": str(error) or error.__class__.__name__}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "wakai backend"})


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


if __name__ == "__main__":
    host = os.environ.get("WAKAI_HOST", "127.0.0.1")
    port = int(os.environ.get("WAKAI_PORT", 7000))
    # Werkzeug's stat reloader (independent of Flask's debug=True, which also
    # turns on the interactive in-browser debugger/pin -- not something to
    # expose just for auto-restart) re-execs this process whenever an
    # imported .py file's mtime changes, so editing routes_run.py/
    # routes_query.py/routes_targets.py/gaia_client.py picks up new routes
    # without a manual restart. It only watches modules actually imported
    # into this process -- static frontend files (gui/*.js/html/css) are
    # already served fresh from disk per-request regardless (see
    # static_files() above), and the gui/backend/run_*.py wrapper scripts
    # are invoked fresh via subprocess.run() per call, so neither needed
    # reload in the first place; this only changes the route-registration
    # gap those didn't have. Set via `wakai gui --reload`, not a bare env
    # var, so it's an explicit per-launch dev choice, not sticky in a shell.
    reload_enabled = os.environ.get("WAKAI_RELOAD") == "1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting wakai backend on {host}:{port}")
    print(f"  results dir:         {OUTPUT_DIR}")
    print(f"  gyro-interp wrapper: {GYRO_WRAPPER}")
    print(f"  BAFFLES wrapper:     {BAFFLES_WRAPPER}")
    if reload_enabled:
        print("  auto-reload:         on (backend .py changes restart the server)")
    app.run(host=host, port=port, debug=False, use_reloader=reload_enabled, threaded=True)
