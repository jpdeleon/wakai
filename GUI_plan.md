# `wakai` Stellar Age-Dating GUI — Architecture Reference

This document describes the **actual, current architecture** of the `wakai` GUI (`gui/`, formerly `dashboard/`). It supersedes the original pre-implementation design plan: the build evolved substantially from that first draft, most notably by dropping every client-side "mock" age calculation in favor of always calling the real `gyro-interp`/`BAFFLES` packages. Treat this as living documentation — update it when the GUI's structure changes, not just when someone asks.

---

## 1. Design System & Aesthetics

The GUI is a **light, editorial "workspace" theme** — not the dark glassmorphic look originally planned. Serif display type and a paper-white canvas were chosen to read as a scientific notebook rather than a dashboard.

### Design Tokens (`gui/styles.css` `:root`)
- **Typography**: `--font-display: "Iowan Old Style", "Palatino Linotype", Georgia, serif` for headings; `--font-sans: "Avenir Next", Avenir, "Segoe UI", Helvetica, Arial, sans-serif` for body/UI text; `--font-mono` for tabular/numeric values.
- **Surfaces**: `--canvas: #f4f7f9` (page background), `--surface: #ffffff` (cards), `--surface-subtle: #f8fafb`.
- **Text**: `--ink: #16212b` (primary), `--ink-soft: #526171` (secondary), `--ink-faint: #8693a0` (muted).
- **Lines**: `--line: #dce3e8`, `--line-strong: #c6d0d8`.
- **Accents** (one per method, used consistently for that method's buttons/plots/badges across all pages):
  - General/UI accent: teal `#176b87` (`--accent`, `--accent-soft: #e5f2f6`)
  - Gyrochronology: violet `#5b55a5` (`--gyro`, `--gyro-soft: #eeedf8`)
  - BAFFLES (Lithium/Activity): rust `#a85331` (`--baffles`, `--baffles-soft: #f8eee9`)
  - Joint/Summary: green `#20735e` (`--joint`, `--joint-soft: #e8f3ef`)
  - Cluster: blue `#2f6f8f` (`--cluster`, `--cluster-soft: #e6f0f5`)
  - Isochrones: amber `#a6740a` (`--isochrone`, `--isochrone-soft: #f6efe0`)
  - Danger/error: `#ad3d3d` (`--danger`, `--danger-soft: #faecec`)
- **Shape**: `--radius: 8px`; flat `--shadow` (two soft low-opacity layers, no heavy elevation).
- **Charts**: [Plotly.js](https://plotly.com/javascript/) 2.24.1, vendored locally at `gui/vendor/plotly-2.24.1.min.js` (`gui/index.html`) so charts still render on an offline/firewalled host -- not a CDN dependency, and not a custom canvas/D3 implementation.

---

## 2. Navigation & Page Structure (SPA Router)

A fixed-top nav (`gui/index.html`) switches between six pages by toggling `.page`/`.page.active` sections — no page reloads, but each page and the active target *are* reflected in the URL via the History API (`gui/app.js`, "URL Routing" section).

```
+---------------------------------------------------------------------------------------------------------+
|  [w wakai]  ::  [Target]  ::  [gyro-interp]  ::  [BAFFLES]  ::  [Cluster]  ::  [Isochrones]  ::  [Summary]  |
+---------------------------------------------------------------------------------------------------------+
```

### URL Scheme

```
/<page-slug>[/target=<name>]
```

| Page | Slug |
|------|------|
| Target | `/target` |
| gyro-interp | `/gyro-interp` |
| BAFFLES | `/baffles` |
| Cluster | `/cluster` |
| Isochrones | `/isochrones` |
| Summary | `/summary` |

- `/` behaves as `/target`.
- Appending `/target=<name>` to any page URL (e.g. `/gyro-interp/target=TOI-5973`) keeps that page's URL addressable to a specific target; it's added automatically once a target is loaded (`pushState`/`replaceState`, no reload) and read back on load.
- The `target=` segment is also recognized standalone: visiting `/target=TOI-5973` directly (typed, bookmarked, or shared) loads the Target page and resolves `TOI-5973` exactly as a manual search would — SIMBAD/Gaia resolution first, then a lookup against previously saved results (`GET /api/targets/load`) so prior `ref_table` edits and gyro/BAFFLES/Cluster results are restored rather than re-computed. On the Target page itself this collapses to the bare `/target=TOI-5973` form (no redundant `/target/target=...`); other pages keep their slug prefix, e.g. `/gyro-interp/target=TOI-5973`.
- Browser back/forward (`popstate`) re-applies the route the same way.
- Backend support: `gui/backend/server.py`'s catch-all route serves any path that isn't a real static file as `index.html` (SPA fallback) so these URLs work on a hard refresh, not just in-app navigation. `send_from_directory`'s own path-traversal guard (`safe_join`) still applies to both branches.

### Workflow Strip (persistent status indicator)

A `.workflow-strip` bar sits above `<main>`, outside every `.page` section, so it's visible on all six pages at once (page-switching only toggles `display` on the sections beneath it, per `.page { display: none }` / `.page.active`). It shows the active target's name plus three stages -- **Rotation** (gyro-interp), **Lithium + activity** (BAFFLES), **Joint age** -- each lit green/red once `state.results.{gyro,baffles,joint}.status` is `'Calculated'`/`'Error'`/`'No overlap'` (`updateWorkflowStrip()`, `gui/app.js`).

The Joint age stage is kept live by `recomputeJointStatus()`: a pure (no DOM/Plotly) function that rebuilds `state.results.joint` from whichever of gyro/BAFFLES/isochrone currently have a `'Calculated'` status (excluding a flagged-broad isochrone posterior, same rule as the Summary page's joint), requiring at least two. `updateWorkflowStrip()` calls it on every invocation -- so the strip reflects the true joint status on every page, not only after visiting Summary. Previously `state.results.joint` was only (re)computed inside `updateSummaryPage()`, so the strip could show a stale "Idle" even after two+ methods had already succeeded elsewhere; `updateSummaryPage()` now calls the same `recomputeJointStatus()` instead of duplicating the PDF-combination logic, so there's a single source of truth for the joint result everywhere it's shown.

### Page 1: Target (`#page-target`)
- **Select a star**: search box that resolves a name via **SIMBAD** (`POST /api/query/simbad`) — not a mocked/preloaded list — then queries nearby **VizieR** tables (`POST /api/query/vizier`) to populate candidate reference values, and can resolve directly by Gaia DR3 source_id (`POST /api/query/gaia`).
- **Reference table** (`ref_table`): editable rows for every input any other page can use — $T_{\rm eff}$, $P_{\rm rot}$, Li EW, $\log R'_{HK}$, $[\text{Fe/H}]$ (VizieR-sourced, opt-in per candidate), plus Parallax, pmRA, pmDec, RV, and Gaia BP/G/RP + 2MASS J/H/Ks photometry (auto-populated once from Gaia DR3 + 2MASS by `POST /api/query/gaia` at target-resolve time — the astrometry feeds the Cluster page, the photometry+parallax feed the Isochrones page). Each catalog-backed row carries an **Open ↗** link to the source in a new tab; VizieR rows are only added after the user explicitly selects a candidate (no silent auto-fill) — the Gaia/2MASS rows are the one exception, filled automatically since they're an unambiguous single-catalog lookup by Gaia ID, same as Teff already was.
- **Analysis status**: tracks which of gyro-interp / BAFFLES / Cluster have been run for the current target, since those results feed the Summary page.
- Target snapshots (target metadata + ref_table + results) persist to SQLite via `POST/GET /api/targets/save`, `/load`, `/` (list) — see §5.

### Page 2: Gyrochronology (`#page-gyro`)
- **Rotation parameters**: read-only display of $T_{\rm eff}$ and $P_{\rm rot}$, synced from the Target page's `ref_table` (no independent sliders — the ref_table is the single source of truth for inputs).
- **Run gyro-interp** button calls `POST /api/run/gyro`, which shells out to the real `gyrointerp` package (see §5). Rejected client-side/server-side outside its calibrated range.
- **Rotation age** card: median age and 68% confidence interval.
- **Plots**: rotation measurement placement (`#plot-gyro-grid`) and the age posterior PDF (`#plot-gyro-posterior`), both Plotly.

### Page 3: Lithium & Activity (`#page-baffles`)
- **Activity & lithium** panel: read-only $T_{\rm eff}$, $\log R'_{HK}$, Li EW, [Fe/H], synced from the `ref_table`.
- **Run BAFFLES** button calls `POST /api/run/baffles`, which first derives $(B-V)_0$ from $T_{\rm eff}$ (Pecaut & Mamajek dwarf table, see `run_bv_lookup.py`) and then runs the real `BAFFLES` package.
- **BAFFLES age** card: $(B-V)_0$ used, lithium age peak, activity age peak, combined BAFFLES age.
- **Plots**: individual lithium/activity posteriors (`#plot-baffles-indiv`) and the combined BAFFLES posterior (`#plot-baffles-posterior`).

### Page 4: Cluster Membership (`#page-cluster`)
*(Not part of the original plan — added after the initial design.)*
- **Cluster membership** panel: takes a Gaia DR3 id, or falls back to ra/dec/parallax for a crossmatch, and calls `POST /api/run/cluster`.
- Checks the target's Gaia DR3 id against the Perren+2023 cluster catalog by default; if absent, crossmatches against every cluster core and scores membership.
- **Cluster summary**, **XYZ – UVW**, and **RA/Dec, proper motion, parallax & RV** cards/plots (server-rendered base64 PNGs from the wakai plotting stack, not Plotly, since these reuse the existing `wakai.catalog` diagnostic plots).
- **CMD (MADYS)** card, below the membership card, with its own **Run CMD plot** button (`POST /api/run/madys-cmd`) -- independent of "Check cluster membership" above (either can run first). Plots the target's own photometry against theoretical isochrone tracks via [MADYS](https://madys.readthedocs.io/en/latest/) (`madys.SampleObject`/`SampleObject.CMD`, mirroring `notebooks/ext_code_madys.ipynb`); if the membership check already matched a cluster, that cluster's member stars (from `data/Perren2023/UCC_members.parquet`, highest membership probability first, capped at 100 -- `run_madys_cmd.py`'s `MEMBER_CAP`, keeping the worst-case runtime bounded) are overlaid too, for a visual youth/membership check, with the target highlighted as a star marker labeled with its real name (not a generic "Target").
- **Isochrone model(s)**: a model chip row (PARSEC2 checked by default, MIST, BHAC15) -- checking more than one renders a separate CMD per model in a single run, reusing one shared MADYS photometry fetch (the slow, network-bound part) rather than re-fetching per model. `run_madys_cmd.py`'s `_render_cmd()` builds each model's plot independently; a bad/unsupported model name only drops that one model's plot (reported in the response's `errors`), not the whole run. At least one model must stay checked (validated both client-side and as a 400 server-side).
- MADYS re-queries photometry over the network for every star in the batch -- no reuse of this app's own Gaia/2MASS cache -- at a fixed ~30-45s overhead regardless of star count (timed locally: ~35s for 1-3 stars, ~45s for 25, batched in one `SampleObject` call rather than per-star). Additional models only add ~6-8s each on top of that single fetch (`SampleObject.CMD()` itself is cheap), so timeout is a generous 300s that comfortably covers a full multi-model, full-member-cap run.
- **Which ages/masses to plot**: two chip rows (same checkbox-chip pattern as gyro-interp's model-age overlay), matching `madys.CMD`'s own default sets -- ages 1/3/5/10/20/30/100/200/500/1000 Myr, masses 0.1/0.3/0.5/0.7/0.85/1.0/1.3/2 M$_\odot$, all checked by default. Unchecking mass tracks entirely is safe (`plot_masses=False`, confirmed against the installed madys version); unchecking *all* ages is not -- `IsochroneGrid.plot_isochrones` always uses `plot_ages` to build the isochrone grid itself (not just to decide what to draw, needed even for mass tracks alone), and passing it `False` hangs or raises deep inside madys. So `run_madys_cmd.py`'s `_plot_ages()` silently falls back to the default age set on an empty selection rather than honoring "no ages" as a real option -- unlike masses, that's a madys limitation, not a UI choice.

### Page 5: Isochrones (`#page-isochrones`)
*(Not part of the original plan — added after the initial design.)*
- Fits age, mass, radius, Teff, logg, Fe/H against MIST isochrones via `isochrones2`'s `SingleStarModel` (`POST /api/run/isochrones`). Every input is read from the Target page's `ref_table` — Parallax, Gaia BP/G/RP, 2MASS J/H/Ks, Teff, Fe/H — populated once at target-resolve time by `POST /api/query/gaia`, not re-fetched per run.
- A checkbox per relevant `ref_table` row (Parallax, BP, G, RP, J, H, K, Teff, Fe/H) lets the user pick exactly which inputs go into the fit; each is disabled if the row has no value. Parallax and the photometric bands auto-check once available (the fit needs them); Teff/Fe-H stay opt-in, unchecked until the user ticks them, since they're optional spectroscopic constraints. A tenth control, **log g**, is a fixed dwarf-star prior (default $4.5\pm0.1$, editable inline) rather than a `ref_table` measurement — matching `notebooks/ext_code_isochrones.ipynb`'s generic `logg: (4.5, 0.1)` constraint, which has no catalog source of its own.
- **Priors** panel: editable age (log10(age/yr)) and A$_V$ (mag) `FlatPrior` bounds, defaulting to the notebook's own `mod.set_prior(age=FlatPrior(bounds=(5,10)), AV=FlatPrior(bounds=(0,0.1)))` (replacing `SingleStarModel`'s default `FlatLogPrior` on age); plus EEP (Equivalent Evolutionary Point) bounds, default `(0, 1710)` -- the full MIST grid, i.e. no restriction. `SingleStarModel`'s fit parameters are actually `[eep, age, feh, distance, AV]` (`mod.n_params == 5`); EEP isn't touched by the notebook, but it's a real free parameter with its own default `EEP_prior` (an IMF-weighted transform of the mass prior, not a plain flat distribution), so it's exposed here too for transparency/restriction. EEP bounds are applied via `SingleStarModel(..., eep_bounds=(lo, hi))` at construction time (not `mod.set_prior(eep=FlatPrior(...))`, which would incorrectly replace the IMF-weighted prior with an unweighted one) -- omitting them lets the model fall back to the grid's own bounds, so a stale UI default can't silently restrict a different MIST grid. All three bounds are sent as `age_prior_bounds`/`av_prior_bounds`/`eep_prior_bounds` on `POST /api/run/isochrones`, validated server-side (min < max); the bounds actually used are echoed back in the response (`prior_bounds.eep` reads `mod.eep_bounds`, the model's resolved value either way) and shown under the fit's result. A collapsible **EEP phase reference** table sits under the EEP inputs (MIST's primary EEP -> evolutionary phase mapping: PMS=1, ZAMS=202, IAMS=353, TAMS=454, RGBTip=605, ZACHeB=631, TACHeB=707, then TPAGB=808/post-AGB=1409/WDCS=1710 for low-mass stars or C-burn=808 for high-mass stars) so users can pick sensible bounds (e.g. `(1, 454)` to restrict to pre-MS through the main sequence) without needing to know MIST's EEP scale from memory.
- **Reliability caveat**: photometry-only isochrone fits are often very weakly constrained for a main-sequence dwarf (a young and old low-mass star can look nearly identical in color/luminosity — the technique is powerful mainly for pre-main-sequence or evolved stars, or with strong spectroscopic constraints). If the 68% CI spans more than `ISOCHRONE_BROAD_DEX` (0.3 dex) in log10(age), the result is flagged and excluded from the Summary page's joint estimate (shown on its own page/row regardless).
- The vendored `isochrones2` package's own `fit_mcmc`/`fit_multinest` don't work in this environment (no `pymultinest`; the emcee-based path is written against the removed emcee v2 API — `sampler.chain`/`.lnprobability`/`.shape`, installed here: emcee 3.1.6). `run_isochrones.py` drives `emcee.EnsembleSampler` directly via the package's public `lnpost`/`emcee_p0` API (mirroring `fit_mcmc_old`'s burn-in/reseed/production logic with the modern API) and shims only the two properties `SingleStarModel._make_samples()` actually reads (`flatchain`, `lnprobability`) — not a patch to the vendored package or its `.venv`.

### Page 6: Joint Age Summary (`#page-summary`)
- **Age posterior comparison** (`#plot-summary-comparison`): overlaid gyro-interp / BAFFLES / isochrones / joint posteriors, Plotly.
- **Evolutionary stage**: visual placement from pre-main-sequence to main sequence.
- **Comparison** and **Target properties** tables summarizing every method's result side by side.
- Joint posterior = normalized product of *whichever* of {gyro-interp, BAFFLES, isochrones} are usable, requiring at least two (not synthesized from a single method) — computed client-side (`combineAllIndependent`, chains `posterior.js`'s pairwise `combineIndependent`) from results already returned by the backend, no dedicated API call. A flagged-broad isochrone posterior is excluded from the product but still plotted, labeled as excluded.

---

## 3. Data Integrity Policy: No Simulated Ages

The original plan (see git history) specified client-side JavaScript reimplementations of Barnes (2007) gyrochronology, Mamajek & Hillenbrand (2008) activity-age, and an ad hoc lithium-depletion heuristic, intended as instant-feedback mock models and as a fallback when the real backend was unavailable.

**That fallback was deliberately removed** (`gui/app.js`, "2. Statistics helper for real posteriors" section). Rationale: a silently-substituted approximate age is worse than no age — a user could mistake a client-side heuristic for a real `gyro-interp`/`BAFFLES` result. The GUI now:
- Only ever displays ages computed by the actual `gyro-interp` and `BAFFLES` packages, run server-side.
- Shows an explicit error state if a backend call fails or hasn't been run yet — never a simulated number.
- Still computes summary statistics (median, 16th/84th percentile) client-side, but only from a PDF that came back from the backend.

If client-side preview math is wanted again in the future, it must be visually and textually distinguished from real results — not a silent fallback.

---

## 4. Directory Layout

```
gui/
├── index.html              # SPA shell: 5 <section class="page"> blocks, Plotly.js + posterior.js + app.js
├── styles.css               # Design tokens (§1) + component styles
├── app.js                   # State, routing, table editing, API calls, Plotly rendering (~2700 lines)
├── posterior.js             # PDF/statistics helpers shared with app.js; unit-tested via Node (tests/test_gui_posterior.py)
├── favicon.ico / favicon-*.png / apple-touch-icon.png
└── backend/
    ├── server.py            # Flask app: serves the SPA AND all /api/* routes from one process (§5)
    ├── requirements.txt     # flask>=3.0, numpy>=1.24 — no flask-cors (not needed, see §5)
    ├── run_gyro.py           # Wrapper: real gyro-interp, run in the gyro-interp .venv
    ├── run_baffles.py        # Wrapper: real BAFFLES, run in the BAFFLES .venv
    ├── run_bv_lookup.py      # Wrapper: Teff -> (B-V)o via Pecaut & Mamajek, run in the wakai env
    └── run_cluster.py        # Wrapper: cluster membership crossmatch, run in the wakai env
```

---

## 5. Backend Architecture

### Single-Origin Flask App (not the original two-port design)

The original plan proposed a separate frontend host talking cross-origin to a Flask API on port 7000, requiring `flask-cors`. The actual implementation serves **both the static SPA and every `/api/*` route from one Flask process** (`gui/backend/server.py`): the `/` and `/<path:path>` routes serve `index.html`/`app.js`/`styles.css`/etc. directly, and `app.js` calls a relative `API_BASE = '/api'`. Same origin, no CORS needed — `flask-cors` was removed from `requirements.txt` accordingly.

Paths to `ext_tools/` and the repo root are computed relative to `server.py`'s own location (not hardcoded), so the GUI works from any checkout of this repo.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/targets/save` | Persist a target's metadata, `ref_table`, and model results to SQLite |
| `GET` | `/api/targets/load` | Load a saved target snapshot by `gaia_id` or `name` |
| `GET` | `/api/targets` | List saved targets, most recently updated first |
| `POST` | `/api/run/gyro` | Run real gyro-interp (`teff`, `prot`, `+_err`); rejects `teff` outside 3800–6200 K (Bouma+2023 calibration range) with HTTP 422 |
| `GET` | `/api/gyro/slow-sequence` | Bouma+2023 empirical slow-sequence Prot(Teff) isochrones (all default ages, target-independent), for the gyro-interp page's model overlay |
| `POST` | `/api/run/baffles` | Derive $(B-V)_0$ then run real BAFFLES; `teff` required, `rhk`/`liew` each optional but at least one required (each may independently fall outside BAFFLES's calibrated range and be skipped, see `run_baffles.py`) |
| `POST` | `/api/run/cluster` | Crossmatch against a cluster catalog (Perren+2023 by default) by Gaia id, ra/dec/parallax, or an explicit `cluster_name` override (restricts scoring/plots to one named cluster instead of auto-matching) |
| `GET` | `/api/cluster/names` | List every cluster name in a catalog (~18k, Perren+2023), for the Cluster page's "cluster to plot" search box |
| `POST` | `/api/run/isochrones` | Fit age/mass/radius/Teff/logg/Fe-H via isochrones2's `SingleStarModel`; `photometry`+`parallax` required (whichever bands the frontend's checkboxes selected, sourced from `ref_table`), `teff`/`feh`/`logg` each optional, `age_prior_bounds`/`av_prior_bounds`/`eep_prior_bounds` each optional (default to the notebook's `(5,10)`/`(0,0.1)`, and the MIST grid's own EEP range, respectively) |
| `POST` | `/api/run/madys-cmd` | Color-magnitude diagram(s) via MADYS; `gaiaid` required, `models` must be non-empty if given (default `["parsec2"]`, one CMD per model from a single shared photometry fetch), `cluster_name`/`name` (labels the target's star marker)/`plot_ages`/`plot_masses` all optional (default to no cluster overlay and madys.CMD's own age/mass sets) |
| `POST` | `/api/query/simbad` | Resolve a target name via SIMBAD |
| `POST` | `/api/query/vizier` | Query VizieR near a target, build selectable reference-table candidates |
| `POST` | `/api/query/gaia` | Query Gaia DR3 by `source_id`, name, or ra/dec cone search; also populates `ref_table`'s Parallax/pmRA/pmDec/RV/Teff (Gaia DR3) and BP/G/RP/J/H/K (Gaia + 2MASS via Gaia's own crossmatch table) rows in one round trip |

### Execution Model

- **Flask server**: launched by `wakai gui` (`wakai/cli.py`), runs in the `wakai` env, default `127.0.0.1:5200` (`--host`/`--port` to override).
- **gyro-interp**: subprocess into `ext_tools/gyro-interp/.venv/bin/python run_gyro.py '<json>'`. Timeout 600s (real MCMC run, single core).
- **BAFFLES**: subprocess into `ext_tools/BAFFLES/.venv/bin/python run_baffles.py '<json>'`, preceded by a `run_bv_lookup.py` call in the `wakai` env. Timeout 180s.
- **Cluster crossmatch**: subprocess into the `wakai` env, `run_cluster.py`. Timeout 300s.
- **Isochrones**: subprocess into `ext_tools/isochrones2/.venv/bin/python run_isochrones.py '<json>'`. Timeout 180s (real MCMC run, ~30s in practice — isochrones' own `fit_mcmc_old` defaults: 1700 steps × 50 walkers).
- **MADYS CMD**: subprocess into the `madys` conda env (`/ut3/jerome/miniconda3/envs/madys/bin/python`, not a repo-local `.venv` under `ext_tools/` -- same pattern as `WAKAI_ENV_PYTHON`), `run_madys_cmd.py`. Timeout 300s (network-bound photometry fetch, not compute-bound).
- All wrapper scripts take JSON on `argv[1]` and print JSON to stdout; `_run_wrapper()` in `server.py` handles timeouts and non-zero exits uniformly.
- A small `OrderedDict`-based LRU cache (`_VIZIER_TARGET_CACHE`, max 4 targets) avoids re-downloading VizieR tables for follow-up parameter selections on the same target within a session.

### Persistence

- Target metadata, `ref_table`, and per-method results persist to SQLite via `wakai.database.TargetDatabase`.
- Pipeline/database location: `WAKAI_OUTPUT_DIR` (default `~/ql/wakai/`), set only by the `wakai gui` launcher — never by the pipeline wrapper scripts, which stay output-dir-agnostic. Override the DB file directly with `WAKAI_DATABASE_PATH`.
- `model_results` is a generic `{target_id, method, payload_json}` store (one row per key of the `results` object POSTed to `/api/targets/save`) -- the backend has no schema-level notion of what a "method" is. `madysCmd` (`plots`, keyed by model, + `errors` + which cluster produced the overlay) is persisted this way like any other method, even though it's a visualization rather than an age estimate. `gui/app.js` uses the same generic store to also persist per-page UI controls that aren't model results, under a `results.pageSettings` key: gyro-interp's axis limits + model-overlay checkboxes (`collectPageSettings()`/`applyPageSettings()`), the Cluster page's catalog/cluster-name override, the MADYS CMD page's model/age/mass chip selections, and the Isochrones page's per-input checkboxes + log g + age/A$_V$/EEP prior bounds. These are collected fresh from the DOM at every `saveTargetState()` call (not kept continuously in sync via `state`) and applied back to the DOM in `restoreSavedTarget()`, right after `state.results` is populated -- so a browser refresh restores exactly what the user had set on every page, not just the ref_table and computed results. Controls with no other natural save checkpoint (gyro's post-run display tweaks, isochrones' priors before a first run) get an explicit `saveTargetState({silent: true})` on `change`; controls the user would set right before clicking a page's Run button (most isochrones checkboxes, the cluster override) piggyback on that run's own silent save.

### Starting the GUI

```bash
wakai gui                       # http://127.0.0.1:5200, opens a browser tab
wakai gui --port 8080 --no-browser
wakai gui --output-dir /path/to/results
```

(Internally: `wakai/cli.py`'s `gui` command shells out to `gui/backend/server.py` with `sys.executable`, so it runs in whatever env `wakai` itself is installed in — no separate `wakai` conda env subprocess needed for the server itself, only for the tool wrappers above.)
