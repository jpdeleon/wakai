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

// Shared display grid. Model PDFs are zero-padded outside their native support.
export const ageGrid = [1, ...Array.from({ length: 2600 }, (_, i) => (i + 1) * 5)];

// Real gyrointerp slow-sequence isochrones (Bouma+2023), fetched once from
// GET /api/gyro/slow-sequence and reused for every target -- this reference
// data is target-independent, unlike everything in `state`, so it lives
// outside state/target save-load rather than being refetched per toggle.
// Colors are matplotlib viridis at np.linspace(100/1000, 1, 5), matching the
// exploratory notebook this overlay is based on. All ages are fetched and
// cached in one call; which ones are drawn is a pure client-side filter
// (checked .chk-gyro-age boxes), no re-fetch per age toggle.
//
// A plain `let` export can't be reassigned by importers (ES module bindings
// to a `let`/`var` are live but read-only from outside the declaring
// module), so target.js and gyro.js -- both of which populate this cache --
// go through setGyroSlowSequenceCache() instead of `gyroSlowSequenceCache = ...`.
export let gyroSlowSequenceCache = null;
export function setGyroSlowSequenceCache(value) {
  gyroSlowSequenceCache = value;
}
export const GYRO_SLOW_SEQUENCE_COLORS = { 80: '#482475', 120: '#31668e', 300: '#1e9c89', 500: '#6ccd5a', 1000: '#fde725' };

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
        xmin: 3500, xmax: 6500, ymin: 0, ymax: 40,
        logY: false, showGrid: true,
      },
      eagles: {
        // The ANN model (Weaver+2024) is EAGLES v2's recommended relation.
        model: 'ml',
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
