#!/usr/bin/env python
"""Run BAFFLES age posterior. Called via subprocess with JSON input on argv[1]."""
import json
import sys
import os

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# Computed relative to this file (sibling of the wakai repo root) rather than
# hardcoded so this works from any checkout of the repo.
WAKAI_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BAFFLES_PATH = os.path.join(os.path.dirname(WAKAI_REPO_DIR), "ext_tools", "BAFFLES")
sys.path.insert(0, BAFFLES_PATH)
os.chdir(BAFFLES_PATH)

import numpy as np

# NumPy compatibility: np.trapezoid was added in NumPy 2.0; BAFFLES env uses older NumPy
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from baffles import age_estimator
import probability as prob


def _posterior_or_none(estimator, bv, metallicity, bv_uncertainty, measure_err):
    """Run get_posterior, treating an out-of-calibration-range AssertionError
    as "this indicator isn't usable for this star" rather than a crash.

    BAFFLES asserts bv/metallicity against its own calibrated ranges inside
    get_posterior (age_estimator.const.BV_RANGE/METAL_RANGE*, see baffles.py)
    -- e.g. calcium/activity is only calibrated for B-V in [0.45, 0.9], a
    much narrower range than lithium's [0.35, 1.9], so a cool star like a
    mid-K dwarf can have a valid lithium age but no activity age. Catching
    per-indicator here lets whichever indicator IS in range still return a
    result instead of the whole request failing.
    """
    try:
        return estimator.get_posterior(
            bv=bv, metallicity=metallicity,
            bv_uncertainty=bv_uncertainty, measure_err=measure_err,
            showPlot=False,
        ), None
    except AssertionError as e:
        return None, str(e)


def main():
    data = json.loads(sys.argv[1])

    bv = data["bv"]
    rhk = data.get("rhk")
    liew = data.get("liew")
    bv_err = data.get("bv_err", 0.02)
    liew_err = data.get("liew_err")
    rhk_err = data.get("rhk_err")

    # calcium/activity age_estimator is still constructed even when rhk is
    # absent, purely for its .const.AGE grid -- identical to lithium's grid
    # (same np.logspace(0, log10(13000), 1000) formula in both ca_constants
    # and li_constants), so it's a safe, always-available source for it.
    baffles_ca = age_estimator("calcium")
    post_ca, ca_skip_reason = (None, None) if rhk is None else _posterior_or_none(
        baffles_ca, bv, rhk, bv_err, rhk_err
    )

    post_li = None
    li_skip_reason = None
    if liew is not None:
        baffles_li = age_estimator("lithium")
        post_li, li_skip_reason = _posterior_or_none(baffles_li, bv, liew, bv_err, liew_err)

    if post_ca is None and post_li is None:
        # Neither indicator is usable for this star -- surface why instead
        # of a bare "ok" result with nothing in it.
        reasons = [r for r in (ca_skip_reason, li_skip_reason) if r]
        print(json.dumps({
            "status": "error",
            "message": "; ".join(reasons) or "BAFFLES could not compute an age for this star.",
        }), file=REAL_STDOUT)
        return

    ca_pdf = [round(float(x), 8) for x in post_ca.array.tolist()] if post_ca is not None else None
    ca_stats_arr = [round(float(x), 3) for x in post_ca.stats] if post_ca is not None else None
    li_pdf = [round(float(x), 8) for x in post_li.array.tolist()] if post_li is not None else None
    li_stats_arr = [round(float(x), 3) for x in post_li.stats] if post_li is not None else None
    age_grid = [round(float(x), 3) for x in baffles_ca.const.AGE.tolist()]

    combined_pdf = None
    combined_stats_arr = None
    if post_ca is not None and post_li is not None:
        y = post_ca.array * post_li.array
        prob.normalize(baffles_ca.const.AGE, y)
        combined_stats = prob.stats(baffles_ca.const.AGE, y)
        combined_pdf = [round(float(x), 8) for x in y.tolist()]
        combined_stats_arr = [round(float(x), 3) for x in combined_stats]

    output = {
        "age_grid": age_grid,
        "ca_pdf": ca_pdf,
        "li_pdf": li_pdf,
        "combined_pdf": combined_pdf,
        "ca_stats": ca_stats_arr,
        "li_stats": li_stats_arr,
        "combined_stats": combined_stats_arr,
        "ca_skip_reason": ca_skip_reason,
        "li_skip_reason": li_skip_reason,
        "status": "ok"
    }
    print(json.dumps(output), file=REAL_STDOUT)

if __name__ == "__main__":
    main()
