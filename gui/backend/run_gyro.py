#!/usr/bin/env python
"""Run gyro-interp age posterior. Called via subprocess with JSON input on argv[1]."""
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
from gyrointerp import gyro_age_posterior, get_summary_statistics

def main():
    data = json.loads(sys.argv[1])

    Prot = data["prot"]
    Teff = data["teff"]
    Prot_err = data.get("prot_err", 0.01 * Prot)
    Teff_err = data.get("teff_err", 100)

    age_grid = np.linspace(0, 2600, 500)
    age_posterior = gyro_age_posterior(
        Prot, Teff,
        Prot_err=Prot_err,
        Teff_err=Teff_err,
        age_grid=age_grid,
        verbose=False
    )

    result = get_summary_statistics(age_grid, age_posterior)

    # gyrointerp's "+1sigma"/"-1sigma" (and 2/3-sigma) are OFFSETS from the
    # median (both returned as positive magnitudes), not absolute age bounds
    # -- see gyrointerp.helpers._find_sigma_interval, which returns
    # (p_sigma - median, median - m_sigma). Convert to absolute ages here so
    # downstream consumers get an actual [lower, upper] confidence interval.
    median = float(result["median"])
    output = {
        "age_grid": [round(x, 3) for x in age_grid.tolist()],
        "pdf": [0.0 if np.isnan(x) else round(x, 8) for x in age_posterior.tolist()],
        "median": round(median, 2),
        "peak": int(result["peak"]),
        "lower_1sig": round(median - float(result["-1sigma"]), 2),
        "upper_1sig": round(median + float(result["+1sigma"]), 2),
        "lower_2sig": round(median - float(result["-2sigma"]), 2) if result.get("-2sigma") is not None else None,
        "upper_2sig": round(median + float(result["+2sigma"]), 2) if result.get("+2sigma") is not None else None,
        "status": "ok"
    }
    print(json.dumps(output), file=REAL_STDOUT)

if __name__ == "__main__":
    main()
