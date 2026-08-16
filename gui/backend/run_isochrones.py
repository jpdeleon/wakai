#!/usr/bin/env python
"""Fit a target's age (plus mass/radius/Teff/logg/feh) from broadband
photometry + parallax via isochrones2's SingleStarModel (MIST grids).
Called via subprocess with JSON input on argv[1]. Runs in the
ext_tools/isochrones2 .venv (needs isochrones/emcee/pandas/numpy).

Teff/feh/logg are optional constraints -- the frontend lets the user toggle
whether to include each one (see gui/index.html's isochrones page); this
script just adds them to SingleStarModel's observation dict if present in
the input, exactly like the photometric bands. logg has no ref_table
measurement behind it (it's a fixed dwarf-star prior, default 4.5+/-0.1,
editable on the page) but is otherwise handled identically.

age_prior_bounds/av_prior_bounds/eep_prior_bounds are also user-editable on
the page, mirroring notebooks/ext_code_isochrones.ipynb's
mod.set_prior(age=FlatPrior(...), AV=FlatPrior(...)) plus SingleStarModel's
own (untouched-by-the-notebook, but still a real fit parameter) EEP prior.
"""
import json
import os
import sys

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

WAKAI_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISOCHRONES_PATH = os.path.join(os.path.dirname(WAKAI_REPO_DIR), "ext_tools", "isochrones2")
sys.path.insert(0, ISOCHRONES_PATH)

import numpy as np
import emcee

# The vendored isochrones2 package's SingleStarModel._make_samples() (and
# StarModel.fit_mcmc_old, which we bypass -- see below) are written against
# the emcee v2 API: sampler.flatchain / sampler.lnprobability / .chain /
# .shape, all removed in emcee>=3 (installed here: 3.1.6) in favor of
# get_chain()/get_log_prob(). Rather than patching the vendored package or
# its .venv, shim only the two properties SingleStarModel actually reads.
emcee.EnsembleSampler.flatchain = property(lambda self: self.get_chain(flat=True))
emcee.EnsembleSampler.lnprobability = property(lambda self: self.get_log_prob(flat=True))

from isochrones import get_ichrone, SingleStarModel  # noqa: E402
from isochrones.priors import FlatPrior  # noqa: E402

# Matches the range and log-spacing of the shared client-side age grid
# (gui/state.js ageGrid, 1500 points) and BAFFLES's own AGE grid (1000
# points). Log-spaced because age posteriors are naturally log-normal-ish
# and linear bins would be far too coarse at young ages and needlessly fine
# at old ages -- see README.md's age-grid table for the full comparison
# across all four methods.
AGE_GRID_MIN_MYR, AGE_GRID_MAX_MYR, N_AGE_BINS = 1, 13000, 300

# isochrones' own defaults for StarModel.fit_mcmc_old (nwalkers=50,
# ninitial=500, nburn=200, niter=1000); timed locally against real Gaia+
# 2MASS photometry at ~1.3e-4 s/likelihood-eval, the full run (1700 steps x
# 50 walkers = 85000 evals) takes ~12s -- fast enough to just use these
# as-is rather than inventing reduced settings.
NWALKERS, NINITIAL, NBURN, NITER = 50, 500, 200, 1000

# Default age/AV prior bounds, matching notebooks/ext_code_isochrones.ipynb's
# mod.set_prior(age=FlatPrior(bounds=(5,10)), AV=FlatPrior(bounds=(0,0.1))).
# Both are editable on the Isochrones page; these are the fallback if the
# frontend doesn't send bounds.
DEFAULT_AGE_PRIOR_BOUNDS = (5, 10)
DEFAULT_AV_PRIOR_BOUNDS = (0, 0.1)

# EEP (Equivalent Evolutionary Point) is SingleStarModel's fourth fit
# parameter -- it's not touched by the notebook's mod.set_prior(...) call,
# so it keeps SingleStarModel's own default EEP_prior spanning the full MIST
# grid, get_ichrone('mist').eep_bounds == (0, 1710) as installed here
# (verified locally; pre-MS through post-AGB). Shown here purely so it's
# visible/editable on the page -- if eep_prior_bounds isn't sent at all,
# SingleStarModel falls back to the grid's own bounds itself (see below),
# so this constant is just the UI's starting value, not a hardcoded ceiling.


def _percentiles(samples):
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])
    return {"median": float(p50), "lower": float(p16), "upper": float(p84)}


def _run_mcmc(mod):
    """Drive emcee directly (mod.emcee_p0/mod.lnpost), mirroring
    StarModel.fit_mcmc_old's logic (initial burn -> reseed around the best
    walker -> burn -> production) but with the modern emcee 3 API, since
    fit_mcmc_old itself can't run against emcee>=3 (see the shim comment
    above -- reseeding needs .get_chain()/.get_log_prob(), not .chain/
    .lnprobability/.shape, which fit_mcmc_old still uses internally for
    that specific step).
    """
    npars = mod.n_params
    p0 = mod.emcee_p0(NWALKERS)

    initial_sampler = emcee.EnsembleSampler(NWALKERS, npars, mod.lnpost)
    initial_sampler.run_mcmc(p0, NINITIAL)
    log_prob = initial_sampler.get_log_prob()
    i, j = np.unravel_index(log_prob.argmax(), log_prob.shape)
    p0_best = initial_sampler.get_chain()[i, j, :]
    rng = np.random.default_rng()
    p0 = p0_best * (1 + rng.normal(size=p0.shape) * 0.001)

    sampler = emcee.EnsembleSampler(NWALKERS, npars, mod.lnpost)
    state = sampler.run_mcmc(p0, NBURN)
    sampler.reset()
    sampler.run_mcmc(state, NITER)

    mod._sampler = sampler
    return sampler


def main():
    data = json.loads(sys.argv[1])

    photometry = data.get("photometry") or {}
    if not photometry:
        print(json.dumps({"status": "error", "message": "No photometry available for this target"}), file=REAL_STDOUT)
        return
    parallax = data.get("parallax")
    parallax_err = data.get("parallax_err")
    if parallax is None or parallax_err is None:
        print(json.dumps({"status": "error", "message": "Missing parallax for this target"}), file=REAL_STDOUT)
        return

    params = {band: tuple(value) for band, value in photometry.items()}
    params["parallax"] = (float(parallax), float(parallax_err))

    used_inputs = {"bands": sorted(photometry.keys()), "teff": False, "feh": False, "logg": False}
    if data.get("teff") is not None and data.get("teff_err") is not None:
        params["Teff"] = (float(data["teff"]), float(data["teff_err"]))
        used_inputs["teff"] = True
    if data.get("feh") is not None and data.get("feh_err") is not None:
        params["feh"] = (float(data["feh"]), float(data["feh_err"]))
        used_inputs["feh"] = True
    if data.get("logg") is not None and data.get("logg_err") is not None:
        params["logg"] = (float(data["logg"]), float(data["logg_err"]))
        used_inputs["logg"] = True

    # EEP bounds have to go in at construction time, not via set_prior after
    # the fact: SingleStarModel builds its EEP_prior as a Jacobian-weighted
    # transform of the mass prior (Chabrier IMF), and only the constructor's
    # eep_bounds kwarg preserves that shape while restricting the range.
    # set_prior(eep=FlatPrior(...)) would silently replace it with a flat
    # (non-IMF-weighted) prior instead -- wrong physics, not just a narrower
    # range. Omitting eep_prior_bounds entirely (rather than passing the
    # (0, 1710) UI default) lets SingleStarModel fall back to its own grid
    # bounds, so a stale/mismatched UI default can't accidentally restrict
    # a different MIST grid.
    eep_bounds_input = data.get("eep_prior_bounds")
    eep_bounds = tuple(float(x) for x in eep_bounds_input) if eep_bounds_input else None

    mist = get_ichrone("mist", bands=sorted(photometry.keys()))
    mod = SingleStarModel(mist, name="target", use_emcee=True, eep_bounds=eep_bounds, **params)

    # Matches notebooks/ext_code_isochrones.ipynb: replace the default
    # FlatLogPrior on age with a FlatPrior over log10(age/yr), and restrict
    # extinction AV to a FlatPrior too. Bounds are user-editable on the
    # Isochrones page; fall back to the notebook's own values if omitted.
    age_bounds = tuple(float(x) for x in (data.get("age_prior_bounds") or DEFAULT_AGE_PRIOR_BOUNDS))
    av_bounds = tuple(float(x) for x in (data.get("av_prior_bounds") or DEFAULT_AV_PRIOR_BOUNDS))
    mod.set_prior(age=FlatPrior(bounds=age_bounds), AV=FlatPrior(bounds=av_bounds))

    _run_mcmc(mod)

    derived = mod.derived_samples
    age_myr = (10 ** derived["age"]) / 1e6

    edges = np.logspace(np.log10(AGE_GRID_MIN_MYR), np.log10(AGE_GRID_MAX_MYR), N_AGE_BINS + 1)
    counts, _ = np.histogram(age_myr, bins=edges, density=True)
    age_grid = [round(float(x), 3) for x in ((edges[:-1] + edges[1:]) / 2).tolist()]
    pdf = [round(float(x), 10) for x in counts.tolist()]

    age_stats = _percentiles(age_myr)
    output = {
        "status": "ok",
        "age_grid": age_grid,
        "pdf": pdf,
        "median": round(age_stats["median"], 3),
        "lower_1sig": round(age_stats["median"] - age_stats["lower"], 3),
        "upper_1sig": round(age_stats["upper"] - age_stats["median"], 3),
        "mass": _percentiles(derived["mass"]),
        "radius": _percentiles(derived["radius"]),
        "teff_derived": _percentiles(derived["Teff"]),
        "logg": _percentiles(derived["logg"]),
        "feh_derived": _percentiles(derived["feh"]),
        "used_inputs": used_inputs,
        "prior_bounds": {
            "age": list(age_bounds),
            "AV": list(av_bounds),
            # mod.eep_bounds is the resolved value regardless of whether
            # eep_prior_bounds was sent (SingleStarModel sets this attribute
            # to the grid's own bounds when eep_bounds=None was passed).
            "eep": list(mod.eep_bounds),
        },
        "n_samples": int(len(derived)),
    }
    print(json.dumps(output), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
