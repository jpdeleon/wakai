#!/usr/bin/env python
"""Fit a target's age (plus mass/radius/Teff/logg) from broadband photometry
via MADYS's SampleObject.get_params() -- PARSEC v2 by default, or any other
model MADYS ships a local grid for (see AVAILABLE_MODELS). Called via
subprocess with JSON input on argv[1]. Runs in the `madys` conda env, same
as run_madys_cmd.py.

Independent of run_isochrones.py's isochrones2/MIST MCMC fit: isochrones2
has no PARSEC support at all (only a MIST ModelGridInterpolator exists in
ext_tools/isochrones2), so this is a second, separate age-fitting method
rather than a model option added to that one. MADYS re-queries its own
photometry over the network (same ~30-45s overhead as run_madys_cmd.py),
rather than reusing this app's own Gaia/2MASS ref_table cache.

get_params()'s mass_range must be passed explicitly: leaving it unset makes
_get_agemass() auto-detect a mass_range via an internal IsochroneGrid(...)
call that hardcodes its own n_steps=[100,75] and also forwards **kwargs
(including our own n_steps) into that same call, which raises "got multiple
values for keyword argument 'n_steps'" in the installed madys build
(/ut3/jerome/miniconda3/envs/madys). Passing mass_range ourselves skips that
code path entirely (confirmed against real photometry -- see this branch's
PR description for the smoke test).

The 1-1000 Msun default mass_range is intersected against the model grid's
own native range internally, so it's a broad ceiling, not a real constraint,
covering PMS through early-type dwarfs.
"""
import json
import os
import sys

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# numexpr prints a banner to stdout when the host has more cores than its
# thread cap; pin the cap before pandas/madys imports so nothing pollutes
# stdout -- same fix as run_madys_cmd.py, confirmed necessary here too (the
# banner reproduces without this).
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import numpy as np  # noqa: E402
import madys  # noqa: E402

DEFAULT_MODEL = "parsec2"
# Local grids actually present under ext_tools/madys (bhac15, dartmouth,
# etc. are valid madys model names but have no downloaded grid here, and
# would just fail at fit time) -- see this wrapper's PR description.
AVAILABLE_MODELS = ("parsec2", "mist")

# Matches the shared client-side age grid's range (gui/state.js ageGrid).
AGE_RANGE_MYR = [1, 13000]
# [n_masses, n_ages] -- timed locally against real photometry at ~3s for
# get_params() itself (the ~30-45s cost is the photometry fetch, shared
# with run_madys_cmd.py). n_ages=500 log-spaced points over 1-13000 Myr
# resolve to a comparable density as BAFFLES/EAGLES's own native grids.
N_STEPS = [400, 500]
MASS_RANGE_MSUN = [0.05, 3.0]
N_TRY = 1000

# get_params()'s fit_status codes (madys.FitParams docstring).
FIT_STATUS_MESSAGES = {
    1: "All magnitudes have an error beyond the maximum allowed threshold.",
    2: "All magnitudes are more than 0.2 mag from their best theoretical match "
       "-- the star may be outside this model's age/mass range.",
    3: "No acceptable fit (chi2 < 1000) was found on the grid.",
    4: "The best-fit solution's third-closest filter is more than 3 sigma "
       "from its theoretical match.",
    5: "Undefined fitting error.",
}


def _reconstruct_age_axis(age_range, n_ages):
    """The age grid IsochroneGrid.__init__ builds internally for a plain
    [min, max] age_range: log-spaced with a small inward fudge factor at
    each endpoint (1.0001x / 0.9999x) so both ends stay strictly inside the
    grid's own native age coverage for interpolation. Reconstructed here
    (not returned by FitParams) because it's the age axis weight_maps is
    defined over -- see ext_tools/madys's IsochroneGrid.__init__, the
    `isinstance(self.age_range, list)` branch.
    """
    lo, hi = age_range
    return np.exp(
        np.log(1.0001 * lo)
        + (np.log(0.9999 * hi) - np.log(1.0001 * lo)) / (n_ages - 1) * np.arange(n_ages)
    )


def _weight_map_to_age_pdf(weight_map, age_axis):
    """Marginalize the (n_masses, n_ages) chi2-weight grid over mass into a
    1D age distribution, then convert from probability MASS per grid point
    (what weight_map sums to 1 over) into a density (mass / local bin
    width). The density conversion matters because age_axis is log-spaced
    with local spacing varying by ~4 orders of magnitude across its range;
    interpolatePDF() (gui/posterior.js) linearly interpolates whatever
    values it's given as if they were already a density, so passing raw
    per-point mass straight through would make old-age bins (wide, in Myr)
    look artificially denser than young-age bins (narrow) of equal mass.
    """
    mass_marginal = np.nansum(weight_map, axis=0)
    widths = np.empty_like(age_axis)
    widths[1:-1] = (age_axis[2:] - age_axis[:-2]) / 2.0
    widths[0] = age_axis[1] - age_axis[0]
    widths[-1] = age_axis[-1] - age_axis[-2]
    return mass_marginal / widths


def _stat_dict(fit, key):
    """{median, lower, upper} for a FitParams point-estimate + range triple
    (e.g. fit.mass/mass_min/mass_max), matching run_isochrones.py's output
    shape so the frontend's existing fmtStat() helper works unchanged. None
    if this model didn't return the quantity (phys_param=False models).
    """
    value = getattr(fit, key, None)
    lo = getattr(fit, f"{key}_min", None)
    hi = getattr(fit, f"{key}_max", None)
    if value is None or lo is None or hi is None:
        return None
    return {"median": float(value[0]), "lower": float(lo[0]), "upper": float(hi[0])}


def main():
    data = json.loads(sys.argv[1])
    gaia_id = str(data.get("gaiaid") or data.get("gaia_id") or "").strip()
    if not gaia_id:
        print(json.dumps({"status": "error", "message": "Target needs a Gaia DR3 ID"}), file=REAL_STDOUT)
        return

    model_version = str(data.get("model") or DEFAULT_MODEL)
    if model_version not in AVAILABLE_MODELS:
        print(json.dumps({
            "status": "error",
            "message": f"model must be one of {AVAILABLE_MODELS} (no local grid for {model_version!r})",
        }), file=REAL_STDOUT)
        return

    target_query_id = f"Gaia DR3 {gaia_id}"
    try:
        sample = madys.SampleObject([target_query_id], id_type="DR3", ext_map="leike", verbose=False)
    except Exception as exc:  # noqa: BLE001 -- network/catalog failures, report cleanly
        print(json.dumps({"status": "error", "message": f"MADYS photometry fetch failed: {exc}"}), file=REAL_STDOUT)
        return

    try:
        fit = sample.get_params(
            model_version,
            age_range=AGE_RANGE_MYR,
            n_steps=N_STEPS,
            mass_range=MASS_RANGE_MSUN,
            n_try=N_TRY,
            save_maps=True,
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001 -- bad grid range, no acceptable fit, etc.
        print(json.dumps({"status": "error", "message": f"MADYS fit failed: {exc}"}), file=REAL_STDOUT)
        return

    fit_status = int(fit.fit_status[0])
    if fit_status != 0:
        message = FIT_STATUS_MESSAGES.get(fit_status, f"MADYS fit_status={fit_status}")
        print(json.dumps({"status": "error", "message": f"MADYS could not fit this target: {message}"}), file=REAL_STDOUT)
        return

    age_axis = _reconstruct_age_axis(AGE_RANGE_MYR, N_STEPS[1])
    age_pdf = _weight_map_to_age_pdf(fit.weight_maps[0], age_axis)

    median = float(fit.age[0])
    lower = float(fit.age_min[0])
    upper = float(fit.age_max[0])

    output = {
        "status": "ok",
        "age_grid": [round(float(x), 6) for x in age_axis.tolist()],
        "pdf": [round(float(x), 10) for x in age_pdf.tolist()],
        "median": round(median, 3),
        "lower_1sig": round(median - lower, 3),
        "upper_1sig": round(upper - median, 3),
        "mass": _stat_dict(fit, "mass"),
        "radius": _stat_dict(fit, "radius"),
        "teff": _stat_dict(fit, "Teff"),
        "logg": _stat_dict(fit, "logg"),
        "model": model_version,
        "fit_status": fit_status,
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
