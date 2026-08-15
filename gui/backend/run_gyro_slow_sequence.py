#!/usr/bin/env python
"""Compute gyrointerp's empirical slow-sequence Prot(Teff) isochrones for a
set of reference ages, for the gyro-interp page's "overplot empirical
models" overlay. Called via subprocess with JSON input on argv[1].

Target-independent (generic reference-cluster curves, not a per-star
posterior), so this is cheap -- polynomial evaluation against gyrointerp's
cached cluster fits, not the MCMC in run_gyro.py -- and safe to fetch once
and reuse across targets.
"""
import json
import sys
import os

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# Computed relative to this file (sibling of the wakai repo root) rather than
# hardcoded so this works from any checkout of the repo.
WAKAI_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GYRO_PATH = os.path.join(os.path.dirname(WAKAI_REPO_DIR), "ext_tools", "gyro-interp")
sys.path.insert(0, GYRO_PATH)

import numpy as np
from gyrointerp.models import slow_sequence

# Matches the Teff range gyrointerp is calibrated for (server.py's
# GYRO_TEFF_MIN/MAX) and the grid density used to explore this overlay.
TEFF_MIN, TEFF_MAX, N_TEFF = 3800, 6200, 100
# All ages the gyro-interp page's overlay lets the user pick between; fetched
# in one call and cached client-side, with age selection filtered in the
# browser rather than re-fetched per selection change.
DEFAULT_AGES_MYR = [80, 120, 300, 500, 1000]


def main():
    data = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
    ages_myr = data.get("ages_myr") or DEFAULT_AGES_MYR

    teff = np.linspace(TEFF_MIN, TEFF_MAX, N_TEFF)
    prot_by_age = {}
    for age in ages_myr:
        prot = slow_sequence(teff, age, poly_order=7, interp_method="pchip_m67", verbose=False)
        prot_by_age[str(age)] = [
            None if np.isnan(x) else round(float(x), 3) for x in np.asarray(prot).ravel()
        ]

    output = {
        "teff": [round(float(x), 2) for x in teff],
        "ages_myr": ages_myr,
        "prot": prot_by_age,
        "status": "ok",
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
