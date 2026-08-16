/**
 * App state: the single `state` object every page module reads/mutates,
 * plus target-independent reference data (ageGrid, the gyro slow-sequence
 * cache) and the result-shape helpers used when loading/saving a target.
 */

// 1. App State
export const state = {
  activeTarget: null,
  activePage: 'target',
  refTable: [],
  catalogCandidates: [],
  availableVizierParameters: [],
  simbadTarget: null,
  savedTargets: [],
  // defaultResults() is a hoisted function declaration, safe to call here
  // even though it's defined later in the file -- single source of truth
  // for the results shape instead of duplicating it.
  results: defaultResults(),
};

// Shared display grid. Model PDFs are zero-padded outside their native
// support (see interpolatePDF in posterior.js). Built as the union of two
// component grids, each matched to a different method's tightest
// resolution requirement, rather than one scheme approximating both. Only
// BAFFLES's grid is an intrinsic package property here; gyro-interp is
// grid-agnostic (it evaluates at whatever ages it's asked for), so its
// "native step" below is really the grid run_gyro.py requests from it,
// not something read off gyro-interp itself:
//  - a log-spaced grid over the full 1-13000 Myr range, at BAFFLES's own
//    native step (0.0041 dex, its 1000-point AGE grid -- an intrinsic
//    package property, and the tightest of BAFFLES/EAGLES, both natively
//    log-spaced; isochrones' own grid is likewise not a package property
//    -- run_isochrones.py bins raw MCMC age samples into a log-spaced
//    histogram it defines itself -- but at a comparable resolution, so it
//    doesn't change which requirement is tightest);
//  - a linear grid over 1-2600 Myr, at the step run_gyro.py requests from
//    gyro-interp (2600/499 Myr -- 2600 is gyro-interp's calibration
//    ceiling per Bouma+2023; its calibration *floor*,
//    GYRO_CALIBRATION_FLOOR_MYR in summary.js, is a reliability caveat on
//    the result -- Bouma+2023: below 80 Myr the model doesn't extrapolate,
//    it just returns a uniform posterior -- not a bound on the grid this
//    method actually computes over -- run_gyro.py's own request grid runs
//    the full 0-2600 Myr).
// A log grid alone would under-resolve the gyro-interp query above ~800
// Myr, where its ~5.2 Myr step becomes finer than a log grid's widening
// dex step; a linear grid alone would need ~2500 points just to match
// BAFFLES's step down at 1 Myr. The union meets every method's own
// resolution requirement across its own valid range at the same total
// cost (1500 points) as the log-only grid this replaced -- see README.md's
// age-grid table.
const AGE_GRID_MIN_MYR = 1;
const AGE_GRID_MAX_MYR = 13000;
const GYRO_GRID_CEILING_MYR = 2600;
const AGE_GRID_LOG_STEP_DEX = 0.0041139; // BAFFLES's own native step
const AGE_GRID_GYRO_STEP_MYR = 5.2104; // the step run_gyro.py requests from gyro-interp

function logGrid(min, max, stepDex) {
  const n = Math.round(Math.log10(max / min) / stepDex) + 1;
  return Array.from({ length: n }, (_, i) => min * (max / min) ** (i / (n - 1)));
}

function linearGrid(min, max, step) {
  const n = Math.round((max - min) / step) + 1;
  return Array.from({ length: n }, (_, i) => min + (i * (max - min)) / (n - 1));
}

export const ageGrid = Array.from(new Set([
  ...logGrid(AGE_GRID_MIN_MYR, AGE_GRID_MAX_MYR, AGE_GRID_LOG_STEP_DEX),
  ...linearGrid(AGE_GRID_MIN_MYR, GYRO_GRID_CEILING_MYR, AGE_GRID_GYRO_STEP_MYR),
])).sort((a, b) => a - b);

// Real gyrointerp slow-sequence isochrones (Bouma+2023), fetched once from
// GET /api/gyro/slow-sequence and reused for every target -- this reference
// data is target-independent, unlike everything in `state`, so it lives
// outside state/target save-load rather than being refetched per toggle.
// Overlay colors come from the shared colormap system below (viridis by
// default). All ages are fetched and cached in one call; which ones are
// drawn is a pure client-side filter (checked .chk-gyro-age boxes), no
// re-fetch per age toggle.
//
// A plain `let` export can't be reassigned by importers (ES module bindings
// to a `let`/`var` are live but read-only from outside the declaring
// module), so target.js and gyro.js -- both of which populate this cache --
// go through setGyroSlowSequenceCache() instead of `gyroSlowSequenceCache = ...`.
export let gyroSlowSequenceCache = null;
export function setGyroSlowSequenceCache(value) {
  gyroSlowSequenceCache = value;
}

// Real EAGLES empirical Li EW(Teff) isochrones (Jeffries+2023 analytic /
// Weaver+2024 ML), fetched once from GET /api/eagles/models and reused for
// every target -- the same target-independent reference-data pattern as
// gyroSlowSequenceCache above. Both models come back in one call; which one
// is drawn follows the page's model selector. All ages are fetched and
// cached in one call, with age selection a pure client-side filter (checked
// .chk-eagles-age boxes), no re-fetch per toggle.
export let eaglesModelsCache = null;
export function setEaglesModelsCache(value) {
  eaglesModelsCache = value;
}
export const EAGLES_MODEL_AGES = [10, 30, 50, 100, 500, 1000];

// Colormaps for the empirical-model overlays on the gyro-interp and EAGLES
// pages. Each entry is 16 stops sampled across the full matplotlib colormap
// (generated with matplotlib.colormaps; the arrays below are those hex
// values); cmapColor() interpolates between stops so every age gets a color
// that still reads as one sequence. The per-page dropdowns
// (#gyro-cmap-select / #eagles-cmap-select) choose which map the overlay
// uses; viridis matches the exploratory notebook both overlays are based on.
export const EMPIRICAL_COLORMAPS = {
  viridis: ['#440154', '#481a6c', '#472f7d', '#414487', '#39568c', '#31688e', '#2a788e', '#23888e', '#1f988b', '#22a884', '#35b779', '#54c568', '#7ad151', '#a5db36', '#d2e21b', '#fde725'],
  plasma: ['#0d0887', '#330597', '#5002a2', '#6a00a8', '#8405a7', '#9c179e', '#b12a90', '#c33d80', '#d35171', '#e16462', '#ed7953', '#f68f44', '#fca636', '#fec029', '#f9dc24', '#f0f921'],
  inferno: ['#000004', '#0c0826', '#240c4f', '#420a68', '#5d126e', '#781c6d', '#932667', '#ae305c', '#c73e4c', '#dd513a', '#ed6925', '#f8850f', '#fca50a', '#fac62d', '#f2e661', '#fcffa4'],
  magma: ['#000004', '#0b0924', '#20114b', '#3b0f70', '#57157e', '#721f81', '#8c2981', '#a8327d', '#c43c75', '#de4968', '#f1605d', '#fa7f5e', '#fe9f6d', '#febf84', '#fddea0', '#fcfdbf'],
  turbo: ['#30123b', '#4143a7', '#4771e9', '#3e9bfe', '#22c5e2', '#1ae4b6', '#46f884', '#88ff4e', '#b9f635', '#e1dd37', '#faba39', '#fd8d27', '#f05b12', '#d63506', '#af1801', '#7a0403'],
  cool: ['#00ffff', '#11eeff', '#22ddff', '#33ccff', '#44bbff', '#55aaff', '#6699ff', '#7788ff', '#8877ff', '#9966ff', '#aa55ff', '#bb44ff', '#cc33ff', '#dd22ff', '#ee11ff', '#ff00ff'],
  autumn: ['#ff0000', '#ff1100', '#ff2200', '#ff3300', '#ff4400', '#ff5500', '#ff6600', '#ff7700', '#ff8800', '#ff9900', '#ffaa00', '#ffbb00', '#ffcc00', '#ffdd00', '#ffee00', '#ffff00'],
  winter: ['#0000ff', '#0011f6', '#0022ee', '#0033e6', '#0044dd', '#0055d4', '#0066cc', '#0077c3', '#0088bb', '#0099b2', '#00aaaa', '#00bba2', '#00cc99', '#00dd90', '#00ee88', '#00ff80'],
};
export const DEFAULT_EMPIRICAL_COLORMAP = 'viridis';

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function rgbToHex(r, g, b) {
  const channel = v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0');
  return `#${channel(r)}${channel(g)}${channel(b)}`;
}

export function cmapColor(cmapName, t) {
  const stops = EMPIRICAL_COLORMAPS[cmapName] || EMPIRICAL_COLORMAPS[DEFAULT_EMPIRICAL_COLORMAP];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x);
  if (i >= stops.length - 1) return stops[stops.length - 1];
  const f = x - i;
  const from = hexToRgb(stops[i]);
  const to = hexToRgb(stops[i + 1]);
  return rgbToHex(
    from[0] + (to[0] - from[0]) * f,
    from[1] + (to[1] - from[1]) * f,
    from[2] + (to[2] - from[2]) * f,
  );
}

// Color for every age of an overlay, sampled across the whole colormap by
// each age's rank within the full fetched age set -- so an age keeps the
// same color regardless of which ages are currently checked.
export function ageColors(cmapName, ages) {
  const sorted = [...ages].sort((a, b) => a - b);
  return Object.fromEntries(sorted.map((age, i) => [
    age,
    cmapColor(cmapName, sorted.length > 1 ? i / (sorted.length - 1) : 0.5),
  ]));
}

export function emptyDefaultParameters() {
  return [
    ['Teff', 'teff'],
    ['Prot', 'prot'],
    ['Li EW', 'liew'],
    ["log R'HK", 'rhk'],
    ['Fe/H', 'feh'],
    // Astrometry/photometry: inputs to Cluster (parallax/pmRA/pmDec/RV) and
    // Isochrones (parallax + all six mags) -- see wakai/reference.py's
    // DEFAULT_PARAMETER_SPECS, which this mirrors.
    ['Parallax', 'parallax'],
    ['pmRA', 'pmra'],
    ['pmDec', 'pmdec'],
    ['RV', 'rv'],
    ['BP mag', 'bpmag'],
    ['G mag', 'gmag'],
    ['RP mag', 'rpmag'],
    ['J mag', 'jmag'],
    ['H mag', 'hmag'],
    ['K mag', 'kmag'],
  ].map(([parameter, id]) => ({
    parameter,
    id,
    value: null,
    uncertainty: null,
    reference: 'No catalog measurement selected',
    catalog: '',
    column: '',
    unit: '',
    source_url: '',
  }));
}

export function defaultResults() {
  return {
    gyro: { agePDF: null, median: null, ci: null, status: 'Idle' },
    baffles: { agePDF: null, median: null, ci: null, status: 'Idle', liPDF: null, actPDF: null },
    // EAGLES lithium-EW age (Jeffries+2023 / Weaver+2024): the posterior's
    // shared-grid resampling (agePDF) feeds the joint like the other methods;
    // the remaining fields are EAGLES's own summary stats, kept for the page's
    // result card and status dot (peak-based asymmetric 68% errors and 95%
    // limits -- see run_eagles.py). `model` is the v2 relation that produced
    // the result ("ml" ANN or "analytic").
    eagles: {
      agePDF: null, median: null, ci: null, status: 'Idle',
      peak: null, peakLo: null, peakHi: null,
      limitUp: null, limitLo: null, model: null,
      logPdf: null, logAgeGrid: null, iso: null,
    },
    // broad: true when the 68% CI spans more than ISOCHRONE_BROAD_DEX in
    // log10(age) -- photometry-only isochrone fits are often this weakly
    // constrained for main-sequence dwarfs (see run_isochrones.py's
    // docstring); flagged results are shown but excluded from the joint.
    isochrone: { agePDF: null, median: null, ci: null, status: 'Idle', broad: false },
    joint: { agePDF: null, median: null, ci: null, status: 'Idle' },
    cluster: null,
    // MADYS color-magnitude diagram -- a visualization, not an age
    // estimate, so (unlike gyro/baffles/isochrone) it never feeds the
    // joint posterior or workflow strip. Independent of `cluster` above:
    // run separately, though it reuses cluster's matched cluster_name (if
    // any) to overlay member stars -- see run_madys_cmd.py.
    madysCmd: { status: 'Idle', plots: {}, errors: {}, clusterName: null, nMembers: 0 },
    // Per-page UI controls that aren't model results or ref_table rows, but
    // that a user still deliberately sets and expects to survive a refresh
    // (axis limits, which optional inputs are checked, prior bounds, ...).
    // Piggybacks on `results` (rather than a separate save/load field)
    // since wakai/database.py's model_results table is already a generic
    // {method: json_payload} store -- no backend/schema change needed to
    // round-trip an extra key through it. Defaults here match gui/index
    // .html's own hardcoded input defaults, so a target with no saved
    // settings yet looks identical to today.
    pageSettings: {
      gyro: {
        showModels: false,
        modelAges: [80, 120, 300, 500],
        cmap: 'viridis',
        xmin: 3500, xmax: 6500, ymin: 0, ymax: 40,
        logY: false, showGrid: true,
      },
      eagles: {
        // The ANN model (Weaver+2024) is EAGLES v2's recommended relation.
        model: 'ml',
        // Empirical Li EW(Teff) isochrone overlay on the diagnostic plot
        // (same pattern as the gyro page's slow-sequence overlay).
        showModels: false,
        modelAges: [10, 30, 50, 100, 500],
        cmap: 'viridis',
      },
      cluster: {
        catalog: 'Perren2023',
        clusterName: '',
      },
      isochrones: {
        // null = never customized -- let syncIsochroneInputParams()'s
        // normal auto-check-if-available logic run, same as before this
        // feature existed. An array (even empty) means "this is exactly
        // what the user had checked," restored verbatim.
        checkedIds: null,
        logg: { checked: true, value: 4.5, err: 0.1 },
        agePrior: [5, 10],
        avPrior: [0, 0.1],
        eepPrior: [0, 1710],
      },
      madysCmd: {
        // null = never customized -- leave the chips at their HTML default
        // (models: just PARSEC2; ages/masses: madys.CMD's own default
        // sets, all checked) rather than forcing them. An array (even
        // empty) is restored verbatim.
        models: null,
        plotAges: null,
        plotMasses: null,
      },
    },
  };
}

export function positiveUncertainty(row, fallback) {
  const value = Number(row?.uncertainty);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function sanitizeStoredResults(results) {
  const clean = { ...defaultResults(), ...(results || {}) };
  ['gyro', 'baffles', 'eagles'].forEach(method => {
    const result = clean[method];
    if (result?.status === 'Calculated' && result.agePDF?.length !== ageGrid.length) {
      clean[method] = defaultResults()[method];
    }
  });
  if (clean.joint?.agePDF?.length !== ageGrid.length) clean.joint = defaultResults().joint;
  return clean;
}

/**
 * Linear interpolation: resample PDF from one age grid onto another.
 * Both grids must be sorted ascending.
 */
export function interpolatePDF(srcGrid, srcPdf, targetGrid) {
  return WakaiPosterior.interpolatePDF(srcGrid, srcPdf, targetGrid);
}
