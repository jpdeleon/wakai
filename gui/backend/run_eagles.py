#!/usr/bin/env python
"""Run the EAGLES lithium-EW age posterior. Called via subprocess with JSON
input on argv[1].

EAGLES ("Estimating AGes from Lithium Equivalent widthS", Jeffries et al.
2023, MNRAS 523, 802) predicts the Li I 6708A equivalent width of a star as
a function of Teff and age, then turns an observed (Teff, Li EW, eTeff,
eLiEW) into a posterior over log age. Version 2.0 (Weaver et al. 2024, MNRAS
534, 2014) offers an analytical EW-age relation (the v2 port of Jeffries+23)
or a machine-learning (ANN) model -- `model` selects between them. Only the
single-star path is used here (this GUI is target-based); the EAGLES
cluster mode is left to the CLI.
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
# The package lives under src/ (setuptools src layout, see its pyproject.toml)
# and its .venv's editable-install metadata points at a sibling checkout, so
# pin the actual src dir regardless of which checkout imported this wrapper.
EAGLES_SRC_PATH = os.path.join(EAGLES_PATH, "src")
sys.path.insert(0, EAGLES_SRC_PATH)

import numpy as np

from eagles.fit import get_li_age_v2
from eagles.model import EWLi_Model


def _myr(log_age):
    """log10(Age/yr) -> Age in Myr (10**(lAge - 6))."""
    return float(10.0 ** (log_age - 6.0))


def main():
    data = json.loads(sys.argv[1])

    teff = data["teff"]
    teff_err = data.get("teff_err")
    liew = data["liew"]
    liew_err = data.get("liew_err", 15.0)
    model_name = data.get("model", "ml")

    # The EAGLES input file format distinguishes "Teff" from "Teff eTeff" by
    # whether an error column is present; eTeff=None reproduces the former.
    eTeff = np.array([teff_err]) if teff_err else None

    use_ml = model_name == "ml"
    model = EWLi_Model(use_ml)

    lAges, llike, lprob, p, chisq = get_li_age_v2(
        np.array([liew]), np.array([liew_err]), np.array([teff]),
        model.get_EWm, model.get_eEWm,
        eTeff=eTeff, z=0.0,
    )

    # p = [lApk, siglo, sighi, limup, limlo, lmed] (log10(Age/yr)); -1 marks
    # a component that doesn't apply (e.g. no 68% errors when the posterior
    # only yields a 95% limit because the peak sits at the grid edge).
    peak_log = float(p[0]) if p[0] != -1 else None
    sig_lo = float(p[1]) if p[1] != -1 else None
    sig_hi = float(p[2]) if p[2] != -1 else None
    limit_up = float(p[3]) if p[3] != -1 else None
    limit_lo = float(p[4]) if p[4] != -1 else None
    median_log = float(p[5]) if p[5] != -1 else None

    # Best-fit model isochrone (EW vs Teff at the derived age) for the EW-vs-
    # Teff diagnostic plot -- evaluated at the peak age when one exists,
    # otherwise the median. Teff grid 3000-6500 K matches EAGLES's own
    # calibration range (and the ML grid's 3000-6590 K).
    iso_age_log = peak_log if peak_log is not None else median_log
    lteff_iso = np.arange(3.4772, 3.8131, 0.005)
    iso = None
    if iso_age_log is not None:
        iso = {
            "teff": [round(float(x), 1) for x in (10.0 ** lteff_iso).tolist()],
            "ew": [round(float(x), 3) for x in model.get_EWm(lteff_iso, iso_age_log).tolist()],
            "eew": [round(float(x), 3) for x in model.get_eEWm(lteff_iso, iso_age_log).tolist()],
        }

    output = {
        # Native EAGLES posterior: PDF over log10(Age/yr).
        "log_age_grid": [round(float(x), 6) for x in lAges.tolist()],
        "log_pdf": [round(float(x), 8) for x in np.exp(lprob).tolist()],
        # Same posterior on a Myr age axis, for the shared ageGrid resampling.
        "age_grid": [round(float(x), 6) for x in (10.0 ** (lAges - 6.0)).tolist()],
        "pdf": [round(float(x), 8) for x in np.exp(lprob).tolist()],
        "median_myr": _myr(median_log) if median_log is not None else None,
        "median_log_age": median_log,
        "peak_myr": _myr(peak_log) if peak_log is not None else None,
        "peak_log_age": peak_log,
        # Asymmetric 68% errors about the peak, as EAGLES reports them.
        "peak_lo_myr": _myr(peak_log - sig_lo) if (peak_log is not None and sig_lo is not None) else None,
        "peak_hi_myr": _myr(peak_log + sig_hi) if (peak_log is not None and sig_hi is not None) else None,
        # 95% upper/lower age limits (only when the peak is at the grid edge).
        "limit_up_myr": _myr(limit_up) if limit_up is not None else None,
        "limit_lo_myr": _myr(limit_lo) if limit_lo is not None else None,
        "model": model_name,
        "iso": iso,
        "status": "ok",
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
