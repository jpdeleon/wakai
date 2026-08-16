#!/usr/bin/env python
"""Compute EAGLES empirical Li EW(Teff) isochrones at fixed ages, for the
EAGLES page's "overplot empirical models" overlay. Called via subprocess with
JSON input on argv[1].

Target-independent (generic EW-age tracks, not a per-star posterior), so this
is cheap -- evaluation of the analytic Jeffries+2023 relation or
interpolation over the Weaver+2024 ANN grid, not the posterior scan in
run_eagles.py -- and safe to fetch once and reuse across targets. Both models
are computed in one call so the frontend can switch between them without
re-fetching; the caller (GET /api/eagles/models) is a target-independent GET,
same pattern as GET /api/gyro/slow-sequence.
"""
import json
import sys
import os

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# Computed relative to this file (sibling of the wakai repo root) rather than
# hardcoded so this works from any checkout of the repo.
WAKAI_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EAGLES_PATH = os.path.join(os.path.dirname(WAKAI_REPO_DIR), "ext_tools", "eagles")
EAGLES_SRC_PATH = os.path.join(EAGLES_PATH, "src")
sys.path.insert(0, EAGLES_SRC_PATH)

import numpy as np

from eagles.model import EWLi_Model

# Matches EAGLES's calibration range (3000-6500 K, per routes_run.py's
# EAGLES_TEFF_MIN/MAX -- the grid's own Teff ceiling is 6590 K) and the
# EW-vs-Teff plot's inverted [6500, 3000] axis.
TEFF_MIN, TEFF_MAX, N_TEFF = 3000, 6500, 100
# The same ages as EAGLES's own shipped isochrone files (iso_10.txt ...
# iso_1000.txt). All fetched in one call and cached client-side, with age
# selection filtered in the browser rather than re-fetched per toggle.
DEFAULT_AGES_MYR = [10, 30, 50, 100, 500, 1000]


def _iso_curve(model, lteff, age_myr):
    """Predicted Li EW across the Teff grid at one age (log10(Age/yr))."""
    log_age = np.log10(age_myr) + 6.0
    return [
        None if np.isnan(x) else round(float(x), 3)
        for x in np.asarray(model.get_EWm(lteff, log_age)).ravel()
    ]


def main():
    data = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
    ages_myr = data.get("ages_myr") or DEFAULT_AGES_MYR

    lteff = np.log10(np.linspace(TEFF_MIN, TEFF_MAX, N_TEFF))
    models = {}
    for name, use_ml in (("ml", True), ("analytic", False)):
        model = EWLi_Model(use_ml)
        models[name] = {
            "ew": {str(age): _iso_curve(model, lteff, age) for age in ages_myr}
        }

    output = {
        "teff": [round(float(x), 1) for x in (10.0 ** lteff).tolist()],
        "ages_myr": ages_myr,
        "models": models,
        "status": "ok",
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
