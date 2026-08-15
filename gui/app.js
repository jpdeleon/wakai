/**
 * WAKAI DASHBOARD APPLICATION LOGIC
 * Manages state, routing, ref_table editing, physical models, and Plotly visualizations.
 */

// 1. App State
const state = {
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
const ageGrid = [1, ...Array.from({ length: 2600 }, (_, i) => (i + 1) * 5)];

// Real gyrointerp slow-sequence isochrones (Bouma+2023), fetched once from
// GET /api/gyro/slow-sequence and reused for every target -- this reference
// data is target-independent, unlike everything in `state`, so it lives
// outside state/target save-load rather than being refetched per toggle.
// Colors are matplotlib viridis at np.linspace(100/1000, 1, 5), matching the
// exploratory notebook this overlay is based on. All ages are fetched and
// cached in one call; which ones are drawn is a pure client-side filter
// (checked .chk-gyro-age boxes), no re-fetch per age toggle.
let gyroSlowSequenceCache = null;
const GYRO_SLOW_SEQUENCE_COLORS = { 80: '#482475', 120: '#31668e', 300: '#1e9c89', 500: '#6ccd5a', 1000: '#fde725' };

// Backend API configuration
const API_BASE = '/api';

async function apiCall(endpoint, data) {
  const response = await fetch(`${API_BASE}/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  const body = await response.json();
  if (!response.ok || body.status === 'error') {
    throw new Error(body.message || `HTTP ${response.status}`);
  }
  return body;
}

async function apiGet(endpoint, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`${API_BASE}/${endpoint}${query ? `?${query}` : ''}`);
  const body = await response.json();
  if (!response.ok || body.status === 'error') {
    const error = new Error(body.message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function safeSourceUrl(value) {
  if (!value) return '';
  try {
    const url = new URL(value, window.location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch (_) {
    return '';
  }
}

function emptyDefaultParameters() {
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

function defaultResults() {
  return {
    gyro: { agePDF: null, median: null, ci: null, status: 'Idle' },
    baffles: { agePDF: null, median: null, ci: null, status: 'Idle', liPDF: null, actPDF: null },
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

function positiveUncertainty(row, fallback) {
  const value = Number(row?.uncertainty);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function sanitizeStoredResults(results) {
  const clean = { ...defaultResults(), ...(results || {}) };
  ['gyro', 'baffles'].forEach(method => {
    const result = clean[method];
    if (result?.status === 'Calculated' && result.agePDF?.length !== ageGrid.length) {
      clean[method] = defaultResults()[method];
    }
  });
  if (clean.joint?.agePDF?.length !== ageGrid.length) clean.joint = defaultResults().joint;
  return clean;
}

function formatSignificant(value, significantFigures = 10) {
  const number = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(number)) return '';
  if (number === 0) return '0';
  const formatted = number.toPrecision(significantFigures);
  if (formatted.includes('e')) {
    const [mantissa, exponent] = formatted.split('e');
    return `${mantissa.replace(/\.?0+$/, '')}e${Number(exponent)}`;
  }
  return formatted.includes('.') ? formatted.replace(/\.?0+$/, '') : formatted;
}

function formatMetadata(value, significantFigures = 10) {
  return formatSignificant(value, significantFigures) || '-';
}

/**
 * Linear interpolation: resample PDF from one age grid onto another.
 * Both grids must be sorted ascending.
 */
function interpolatePDF(srcGrid, srcPdf, targetGrid) {
  return WakaiPosterior.interpolatePDF(srcGrid, srcPdf, targetGrid);
}

// ================= 2. Statistics helper for real posteriors =================
//
// NOTE: this file previously also contained client-side reimplementations of
// gyrochronology (Barnes 2007), chromospheric-activity age (Mamajek 2008),
// and an uncited ad hoc lithium-depletion heuristic, used as a silent
// fallback whenever the real gyro-interp/BAFFLES backend calls failed or
// before the user had run them. That fallback has been removed: this GUI
// only ever displays ages that came from the actual gyro-interp/BAFFLES
// packages via the backend. If those calls fail, the UI now shows a clear
// error instead of a simulated number.

/**
 * Compute statistics (Median, 16th, 84th percentile) from normalized PDF
 */
function getStatsFromPDF(pdf) {
  return WakaiPosterior.stats(ageGrid, pdf) || { median: 0, lower: 0, upper: 0 };
}

// Recomputes state.results.joint from whatever gyro/BAFFLES/isochrone
// results currently exist -- pure state update, no DOM/Plotly work, so it's
// cheap enough to call on every updateWorkflowStrip() run. This is what
// keeps the top nav's "Joint age" stage live on every page: previously
// state.results.joint was only (re)computed inside updateSummaryPage(),
// which only runs when the user actually visits the Summary page, so the
// strip could show "Idle" even after two+ methods had already succeeded.
// updateSummaryPage() calls this too instead of duplicating the same
// usable-methods/combineAllIndependent logic itself.
function recomputeJointStatus() {
  const isGyroRun = state.results.gyro.status === 'Calculated';
  const isBafflesRun = state.results.baffles.status === 'Calculated';
  const isIsochroneRun = state.results.isochrone.status === 'Calculated';
  // Excludes a weakly-constrained (broad) isochrone posterior, same as
  // updateSummaryPage's joint product -- see ISOCHRONE_BROAD_DEX.
  const isIsochroneUsable = isIsochroneRun && !state.results.isochrone.broad;

  const pdfs = [];
  if (isGyroRun) pdfs.push(state.results.gyro.agePDF);
  if (isBafflesRun) pdfs.push(state.results.baffles.agePDF);
  if (isIsochroneUsable) pdfs.push(state.results.isochrone.agePDF);

  if (pdfs.length < 2) {
    state.results.joint = { ...defaultResults().joint, status: 'Idle' };
    return;
  }

  const joint = combineAllIndependent(ageGrid, pdfs);
  if (joint.status !== 'ok') {
    state.results.joint = { ...defaultResults().joint, status: 'No overlap' };
    return;
  }

  state.results.joint = {
    agePDF: joint.pdf,
    median: joint.stats.median,
    lower: joint.stats.lower,
    upper: joint.stats.upper,
    ci: `[${formatSignificant(joint.stats.lower, 8)}, ${formatSignificant(joint.stats.upper, 8)}]`,
    status: 'Calculated',
  };
}

function updateWorkflowStrip() {
  recomputeJointStatus();
  const targetName = state.activeTarget?.name || 'No target selected';
  document.getElementById('workspace-target-name').innerText = targetName;
  const mappings = [
    ['workspace-gyro-stage', state.results.gyro?.status],
    ['workspace-baffles-stage', state.results.baffles?.status],
    ['workspace-joint-stage', state.results.joint?.status],
  ];
  mappings.forEach(([id, status]) => {
    const element = document.getElementById(id);
    element.classList.toggle('complete', status === 'Calculated');
    element.classList.toggle('failed', status === 'Error' || status === 'No overlap');
  });
}

function invalidateJointResult() {
  // updateWorkflowStrip() recomputes state.results.joint itself (via
  // recomputeJointStatus()) from whatever methods are still valid -- e.g.
  // invalidating isochrone shouldn't blank the joint if gyro+BAFFLES alone
  // can still produce one, so this doesn't force-reset it first.
  updateWorkflowStrip();
}

function invalidateMethod(method) {
  const hadResult = ['Calculated', 'Error'].includes(state.results[method]?.status);
  state.results[method] = defaultResults()[method];
  invalidateJointResult();
  const prefix = method === 'gyro' ? 'gyro' : 'baffles';
  document.getElementById(`${prefix}-summary-card`).style.display = 'none';
  document.getElementById(`${prefix}-results-panel`).style.display = 'none';
  document.getElementById(`status-${prefix}-dot`).className = 'dot dot-inactive';
  document.getElementById(`status-${prefix}-text`).innerText = hadResult ? 'Inputs changed' : 'Idle';
}

// ================= 4. Core GUI Actions & Routing =================

async function initApp() {
  setupNavigation();
  setupTargetSelector();
  setupTableActions();
  setupCatalogActions();
  setupModelRunControls();
  setupGyroModelOverlay();
  setupClusterPageControls();
  setupIsochronesPageControls();
  setupSummaryActions();
  // Awaited (unlike everything else above) because applyRoute needs
  // state.savedTargets populated to try the local-cache path first -- see
  // applyRoute.
  await loadSavedTargetIndex();
  updateWorkflowStrip();
  await applyRoute(window.location.pathname);
}

async function loadSavedTargetIndex() {
  try {
    const body = await apiGet('targets');
    state.savedTargets = (body.targets || []).map(target => ({
      name: target.name,
      gaiaid: target.gaia_id,
      parameters: emptyDefaultParameters(),
      isSaved: true,
    }));
  } catch (error) {
    showToast(`Could not list saved targets: ${error.message}`);
  }
}

// ================= URL Routing =================
// Path shape: /<page-slug>[/target=<name>], e.g. /gyro-interp/target=TOI-5973.
// The target= segment can also appear alone (/target=TOI-5973 -> Target page
// with that target loaded) since it's matched independently of page slug.
const PAGE_ROUTES = { target: 'target', gyro: 'gyro-interp', baffles: 'baffles', cluster: 'cluster', isochrones: 'isochrones', summary: 'summary' };
const ROUTE_PAGES = Object.fromEntries(Object.entries(PAGE_ROUTES).map(([pageId, slug]) => [slug, pageId]));

function buildPath(pageId, targetName) {
  const parts = [];
  // On the Target page itself, a target= segment already implies the page
  // (parsePath defaults to it), so skip the redundant /target/target=... .
  if (pageId !== 'target' || !targetName) {
    parts.push(PAGE_ROUTES[pageId] || PAGE_ROUTES.target);
  }
  if (targetName) parts.push(`target=${encodeURIComponent(targetName)}`);
  return `/${parts.join('/')}`;
}

function parsePath(pathname) {
  let pageId = 'target';
  let targetSlug = null;
  for (const segment of pathname.split('/').filter(Boolean)) {
    if (segment.startsWith('target=')) {
      targetSlug = decodeURIComponent(segment.slice('target='.length));
    } else if (ROUTE_PAGES[segment]) {
      pageId = ROUTE_PAGES[segment];
    }
  }
  return { pageId, targetSlug };
}

function updateUrl(pageId, targetName, { replace = false } = {}) {
  const path = buildPath(pageId, targetName);
  if (path === window.location.pathname) return;
  window.history[replace ? 'replaceState' : 'pushState']({ pageId, targetName }, '', path);
}

function showPage(pageId) {
  document.querySelectorAll('.nav-item').forEach(l => l.classList.toggle('active', l.getAttribute('data-page') === pageId));
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${pageId}`));
  state.activePage = pageId;
  onPageEnter(pageId);
}

function targetSlugMatchesActive(targetSlug) {
  if (!state.activeTarget) return false;
  const slug = targetSlug.toLowerCase();
  return String(state.activeTarget.name || '').toLowerCase() === slug
    || String(state.activeTarget.gaiaid || '').toLowerCase() === slug;
}

// Local-cache fast path, but ONLY by Gaia ID -- a Gaia DR3 source_id is
// canonical and unambiguous, so trusting a local match for one is safe
// without asking SIMBAD first. A bare *name* is not: the same star can have
// several valid names (e.g. "TOI-837" and "CPD-63 1435" both resolve to
// Gaia DR3 5251470948229949568 per SIMBAD), and a name-only local match
// can't tell whether a differently-named row already exists for the same
// real star, or whether a stale/wrong-Gaia-ID row happens to share the
// typed name. (This app once shipped hardcoded demo data under a fake
// 12-digit "Gaia ID" for "TOI-837"; a name-only fast path here would have
// kept serving that stale row forever instead of ever re-resolving to the
// real star.) Requires state.savedTargets to be populated (see initApp).
function findSavedTargetByGaiaId(gaiaId) {
  return state.savedTargets.find(t => String(t.gaiaid || '') === gaiaId);
}

// Shared by initial page load and popstate (back/forward): resolves the page
// and, if the URL names a target that isn't already loaded, fetches it. A
// numeric Gaia-ID slug can use the local cache directly (findSavedTargetByGaiaId);
// a name slug always goes through the full SIMBAD/Gaia resolution
// (triggerMockSearch) so that name aliases resolve to the same canonical
// Gaia ID -- and therefore the same saved record -- no matter which alias
// was typed, instead of silently diverging into separate saved targets.
async function applyRoute(pathname) {
  const { pageId, targetSlug } = parsePath(pathname);
  showPage(pageId);
  if (targetSlug && !targetSlugMatchesActive(targetSlug)) {
    // triggerMockSearch's own "Resolving..." state lives on the Target
    // page's search button, which is invisible if the URL landed on a
    // different page (e.g. /gyro-interp/target=...) -- without this, the
    // round trip (real network calls, can take a couple seconds) looks
    // like a blank, unresponsive page until it resolves.
    showToast(`Loading ${targetSlug}...`);
    const known = /^\d+$/.test(targetSlug) ? findSavedTargetByGaiaId(targetSlug) : null;
    if (known) {
      await loadTarget(known);
    } else {
      await triggerMockSearch(targetSlug);
    }
    onPageEnter(pageId);
  }
}

function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const pageId = link.getAttribute('data-page');
      showPage(pageId);
      updateUrl(pageId, state.activeTarget?.name);
    });
  });
  window.addEventListener('popstate', () => {
    applyRoute(window.location.pathname);
  });
}

function syncGyroInputParams() {
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const protRow = state.refTable.find(r => r.parameter === 'Prot');
  document.getElementById('gyro-in-teff').innerText = teffRow && teffRow.value != null
    ? `${formatSignificant(teffRow.value, 8)} ± ${formatSignificant(teffRow.uncertainty, 6)} K`
    : '—';
  document.getElementById('gyro-in-prot').innerText = protRow && protRow.value != null
    ? `${formatSignificant(protRow.value, 8)} ± ${formatSignificant(protRow.uncertainty, 6)} d`
    : '—';
}

function syncBafflesInputParams() {
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const rhkRow = state.refTable.find(r => r.parameter === "log R'HK");
  const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
  const fehRow = state.refTable.find(r => r.parameter === 'Fe/H');
  document.getElementById('baffles-in-teff').innerText = teffRow && teffRow.value != null
    ? `${formatSignificant(teffRow.value, 8)} ± ${formatSignificant(teffRow.uncertainty, 6)} K`
    : '—';
  document.getElementById('baffles-in-rhk').innerText = rhkRow && rhkRow.value != null
    ? `${formatSignificant(rhkRow.value, 8)} ± ${formatSignificant(rhkRow.uncertainty, 6)}`
    : '—';
  document.getElementById('baffles-in-liew').innerText = liewRow && liewRow.value != null
    ? `${formatSignificant(liewRow.value, 8)} ± ${formatSignificant(liewRow.uncertainty, 6)} mÅ`
    : '—';
  document.getElementById('baffles-in-feh').innerText = fehRow && fehRow.value != null
    ? `${formatSignificant(fehRow.value, 8)} ± ${formatSignificant(fehRow.uncertainty, 6)}`
    : '—';
}

function onPageEnter(pageId) {
  if (pageId === 'gyro') {
    syncGyroInputParams();
    drawGyroPlots();
  } else if (pageId === 'baffles') {
    syncBafflesInputParams();
    drawBafflesPlots();
  } else if (pageId === 'cluster') {
    syncClusterInputs();
    if (state.results.cluster) renderClusterResult(state.results.cluster);
    if (Object.keys(state.results.madysCmd?.plots || {}).length) {
      renderMadysCmdResult(state.results.madysCmd);
      document.getElementById('madys-cmd-results-panel').style.display = 'block';
    }
  } else if (pageId === 'isochrones') {
    syncIsochroneInputParams();
    drawIsochronePlots();
  } else if (pageId === 'summary') {
    updateSummaryPage();
  }
}

function showToast(text) {
  const toast = document.getElementById('toast');
  toast.innerText = text;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 2500);
}

// ================= 5. Target Logic =================

function setupTargetSelector() {
  const searchInput = document.getElementById('target-search-input');
  const suggestionsBox = document.getElementById('target-suggestions');
  const queryBtn = document.getElementById('btn-query-target');

  searchInput.addEventListener('input', () => {
    const val = searchInput.value.toLowerCase();
    suggestionsBox.innerHTML = '';

    if (!val) {
      suggestionsBox.style.display = 'none';
      return;
    }

    const targetsByIdentity = new Map();
    state.savedTargets.forEach(target => {
      targetsByIdentity.set(targetIdentity(target), target);
    });
    const filtered = [...targetsByIdentity.values()]
      .filter(t => t.name.toLowerCase().includes(val));

    // Add custom selection option
    filtered.push({ name: `Custom Search: "${searchInput.value}"`, isCustom: true, rawName: searchInput.value });

    filtered.forEach(item => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';

      if (item.isCustom) {
        div.innerHTML = `<span>Query Gaia DR3: <strong>${escapeHtml(item.rawName)}</strong></span><span class="type">TAP Service</span>`;
        div.addEventListener('click', () => {
          triggerMockSearch(item.rawName);
          suggestionsBox.style.display = 'none';
        });
      } else {
        div.innerHTML = `<span>${escapeHtml(item.name)}</span><span class="type">${item.isSaved ? 'Saved Results' : 'Catalog Match'}</span>`;
        div.addEventListener('click', async () => {
          await loadTarget(item);
          suggestionsBox.style.display = 'none';
          searchInput.value = item.name;
        });
      }
      suggestionsBox.appendChild(div);
    });

    suggestionsBox.style.display = 'block';
  });

  // Hide suggestions on outside click
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.target-search-container')) {
      suggestionsBox.style.display = 'none';
    }
  });

  queryBtn.addEventListener('click', () => {
    const query = searchInput.value.trim();
    if (query) {
      triggerMockSearch(query);
    } else {
      showToast("Please enter a target name or Gaia ID first");
    }
  });
}

async function triggerMockSearch(name) {
  const queryBtn = document.getElementById('btn-query-target');
  queryBtn.disabled = true;
  queryBtn.innerText = "Resolving target...";

  try {
    let resolved = null;
    if (!/^\d+$/.test(name)) {
      const simbad = await apiCall('query/simbad', { name });
      resolved = simbad.target;
      state.simbadTarget = resolved;
    }

    const gaiaRequest = {
      name,
      gaia_id: /^\d+$/.test(name) ? name : (resolved?.gaiaid || ''),
      ra: resolved?.ra,
      dec: resolved?.dec,
    };

    try {
      const gaia = await apiCall('query/gaia', gaiaRequest);
      const restored = await loadTarget(gaia.target);
      showToast(`Loaded target "${name}" from Gaia DR3`);
      if (!restored) await queryVizierParameters({ populateDefaults: true });
    } catch (gaiaError) {
      if (!resolved) throw gaiaError;
      const restored = await loadTarget({
        name: resolved.name || name,
        gaiaid: resolved.gaiaid,
        ra: resolved.ra,
        dec: resolved.dec,
        parallax: null,
        rv: null,
        parameters: emptyDefaultParameters(),
      });
      showToast(`Loaded "${name}" from SIMBAD; no Gaia DR3 row was found`);
      if (!restored) await queryVizierParameters({ populateDefaults: true });
    }
  } catch (err) {
    showToast(`Query failed: ${err.message}`);
  } finally {
    queryBtn.disabled = false;
    queryBtn.innerText = "Query Catalog Services";
  }
}

async function loadTarget(target) {
  state.activeTarget = target;
  updateUrl(state.activePage, target.name, { replace: true });
  state.refTable = JSON.parse(JSON.stringify(target.parameters || emptyDefaultParameters()));
  state.catalogCandidates = [];
  document.getElementById('ref-table-card').hidden = false;

  // Reset Results
  state.results = defaultResults();
  document.getElementById('gyro-summary-card').style.display = 'none';
  document.getElementById('baffles-summary-card').style.display = 'none';

  // Update Metadata Card
  renderTargetMetadata();

  // Update Status
  document.getElementById('status-target-name').innerText = target.name;
  document.getElementById('status-gyro-dot').className = "dot dot-inactive";
  document.getElementById('status-gyro-text').innerText = "Idle";
  document.getElementById('status-baffles-dot').className = "dot dot-inactive";
  document.getElementById('status-baffles-text').innerText = "Idle";
  updateWorkflowStrip();

  renderRefTable();
  return restoreSavedTarget(target);
}

function targetIdentity(target) {
  const gaiaId = String(target?.gaiaid || '').trim();
  return gaiaId ? `gaia:${gaiaId}` : `name:${String(target?.name || '').trim().toLowerCase()}`;
}

// Reads the current DOM state of every persisted-but-not-a-result control
// (gyro axis/model-overlay settings, Cluster catalog/name override,
// Isochrones checkboxes + logg + prior bounds) into a plain object matching
// defaultResults().pageSettings's shape. Called at save time rather than
// kept continuously in sync via change listeners -- the DOM is the single
// source of truth, this just snapshots it.
function collectPageSettings() {
  const num = (id, fallback) => {
    const value = Number(document.getElementById(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  };

  return {
    gyro: {
      showModels: document.getElementById('chk-gyro-models')?.checked ?? false,
      modelAges: Array.from(document.querySelectorAll('.chk-gyro-age:checked')).map(el => Number(el.value)),
      xmin: num('gyro-grid-xmin', 3500),
      xmax: num('gyro-grid-xmax', 6500),
      ymin: num('gyro-grid-ymin', 0),
      ymax: num('gyro-grid-ymax', 40),
      logY: document.getElementById('chk-gyro-grid-logy')?.checked ?? false,
      showGrid: document.getElementById('chk-gyro-grid-showgrid')?.checked ?? true,
    },
    cluster: {
      catalog: document.getElementById('cluster-catalog-select')?.value || 'Perren2023',
      clusterName: document.getElementById('cluster-name-input')?.value.trim() || '',
    },
    isochrones: {
      checkedIds: ISOCHRONE_INPUT_SPECS
        .filter(spec => document.getElementById(`chk-isochrone-${spec.id}`)?.checked)
        .map(spec => spec.id),
      logg: {
        checked: document.getElementById('chk-isochrone-logg')?.checked ?? true,
        value: num('isochrone-logg-value', 4.5),
        err: num('isochrone-logg-err', 0.1),
      },
      agePrior: [num('isochrone-age-prior-min', 5), num('isochrone-age-prior-max', 10)],
      avPrior: [num('isochrone-av-prior-min', 0), num('isochrone-av-prior-max', 0.1)],
      eepPrior: [num('isochrone-eep-prior-min', 0), num('isochrone-eep-prior-max', 1710)],
    },
    madysCmd: {
      models: Array.from(document.querySelectorAll('.chk-madys-model:checked')).map(el => el.value),
      plotAges: Array.from(document.querySelectorAll('.chk-madys-age:checked')).map(el => Number(el.value)),
      plotMasses: Array.from(document.querySelectorAll('.chk-madys-mass:checked')).map(el => Number(el.value)),
    },
  };
}

// Reverse of collectPageSettings() -- writes a saved settings object back
// into the DOM controls on target restore. Async because turning the gyro
// model overlay back on may need to fetch gyroSlowSequenceCache (same cache
// setupGyroModelOverlay() populates lazily on first checkbox toggle).
async function applyPageSettings(settings) {
  const s = settings || {};
  const setNum = (id, value) => {
    const el = document.getElementById(id);
    if (el && Number.isFinite(value)) el.value = value;
  };

  const gyro = s.gyro || {};
  const modelsCheckbox = document.getElementById('chk-gyro-models');
  const ageRow = document.getElementById('gyro-model-ages');
  if (modelsCheckbox) modelsCheckbox.checked = Boolean(gyro.showModels);
  if (ageRow) ageRow.hidden = !modelsCheckbox?.checked;
  if (Array.isArray(gyro.modelAges)) {
    document.querySelectorAll('.chk-gyro-age').forEach(el => {
      el.checked = gyro.modelAges.includes(Number(el.value));
    });
  }
  setNum('gyro-grid-xmin', gyro.xmin);
  setNum('gyro-grid-xmax', gyro.xmax);
  setNum('gyro-grid-ymin', gyro.ymin);
  setNum('gyro-grid-ymax', gyro.ymax);
  const logYEl = document.getElementById('chk-gyro-grid-logy');
  if (logYEl) logYEl.checked = Boolean(gyro.logY);
  const showGridEl = document.getElementById('chk-gyro-grid-showgrid');
  if (showGridEl) showGridEl.checked = gyro.showGrid !== false;
  // Don't call fitGyroGridAxisToModels() here -- it would overwrite the
  // axis values just restored above with the models' own auto-fit range.
  if (modelsCheckbox?.checked && !gyroSlowSequenceCache) {
    try {
      gyroSlowSequenceCache = await apiGet('gyro/slow-sequence');
    } catch (error) {
      // Overlay just won't render until the user retoggles the checkbox.
    }
  }

  const cluster = s.cluster || {};
  const catalogSelect = document.getElementById('cluster-catalog-select');
  if (catalogSelect && cluster.catalog) catalogSelect.value = cluster.catalog;
  const clusterNameInput = document.getElementById('cluster-name-input');
  if (clusterNameInput) {
    clusterNameInput.value = cluster.clusterName || '';
    updateClusterNameHint();
  }

  const iso = s.isochrones || {};
  if (Array.isArray(iso.checkedIds)) {
    ISOCHRONE_INPUT_SPECS.forEach(spec => {
      const chk = document.getElementById(`chk-isochrone-${spec.id}`);
      if (!chk) return;
      chk.checked = iso.checkedIds.includes(spec.id);
      // Marks it as already decided so syncIsochroneInputParams() (which
      // runs whenever the Isochrones page is entered) won't auto-check it
      // again over the user's restored choice -- see its own comment.
      chk.dataset.userTouched = '1';
    });
  }
  if (iso.logg) {
    const loggChk = document.getElementById('chk-isochrone-logg');
    if (loggChk && typeof iso.logg.checked === 'boolean') loggChk.checked = iso.logg.checked;
    setNum('isochrone-logg-value', iso.logg.value);
    setNum('isochrone-logg-err', iso.logg.err);
  }
  if (Array.isArray(iso.agePrior)) {
    setNum('isochrone-age-prior-min', iso.agePrior[0]);
    setNum('isochrone-age-prior-max', iso.agePrior[1]);
  }
  if (Array.isArray(iso.avPrior)) {
    setNum('isochrone-av-prior-min', iso.avPrior[0]);
    setNum('isochrone-av-prior-max', iso.avPrior[1]);
  }
  if (Array.isArray(iso.eepPrior)) {
    setNum('isochrone-eep-prior-min', iso.eepPrior[0]);
    setNum('isochrone-eep-prior-max', iso.eepPrior[1]);
  }

  const madysCmdSettings = s.madysCmd || {};
  if (Array.isArray(madysCmdSettings.models)) {
    document.querySelectorAll('.chk-madys-model').forEach(el => {
      el.checked = madysCmdSettings.models.includes(el.value);
    });
  }
  if (Array.isArray(madysCmdSettings.plotAges)) {
    document.querySelectorAll('.chk-madys-age').forEach(el => {
      el.checked = madysCmdSettings.plotAges.includes(Number(el.value));
    });
  }
  if (Array.isArray(madysCmdSettings.plotMasses)) {
    document.querySelectorAll('.chk-madys-mass').forEach(el => {
      el.checked = madysCmdSettings.plotMasses.includes(Number(el.value));
    });
  }
  saveClusterPageSettingsToLocalStorage();
}

async function restoreSavedTarget(target) {
  const identity = targetIdentity(target);
  const params = { name: target.name };
  if (target.gaiaid) params.gaia_id = target.gaiaid;
  try {
    const body = await apiGet('targets/load', params);
    if (targetIdentity(state.activeTarget) !== identity) return false;
    const record = body.record;
    state.activeTarget = { ...state.activeTarget, ...record.target };
    state.refTable = record.ref_table?.length
      ? JSON.parse(JSON.stringify(record.ref_table))
      : state.refTable;
    state.results = sanitizeStoredResults(record.results);
    renderTargetMetadata();
    renderRefTable();
    renderStoredResultStatus();
    await applyPageSettings(state.results.pageSettings);
    // The workflow strip's stages were set from the pre-restore (empty)
    // state.results before this async load resolved -- refresh them now
    // that the saved gyro/BAFFLES/isochrone/joint results are in, so a
    // restored target's status is correct on every page, not just after
    // some later action happens to call updateWorkflowStrip() again.
    updateWorkflowStrip();
    showToast(`Restored saved results for ${state.activeTarget.name}`);
    return true;
  } catch (error) {
    if (error.status !== 404) showToast(`Could not restore saved target: ${error.message}`);
    return false;
  }
}

function renderStoredResultStatus() {
  const gyro = state.results.gyro || {};
  const baffles = state.results.baffles || {};
  const gyroCalculated = gyro.status === 'Calculated';
  const bafflesCalculated = baffles.status === 'Calculated';
  document.getElementById('status-gyro-dot').className = gyroCalculated ? 'dot dot-active' : 'dot dot-inactive';
  document.getElementById('status-gyro-text').innerText = gyroCalculated
    ? `${formatSignificant(gyro.median, 8)} Myr`
    : (gyro.status || 'Idle');
  document.getElementById('status-baffles-dot').className = bafflesCalculated ? 'dot dot-active' : 'dot dot-inactive';
  document.getElementById('status-baffles-text').innerText = bafflesCalculated
    ? `${formatSignificant(baffles.median, 8)} Myr`
    : (baffles.status || 'Idle');
  if (gyroCalculated) {
    document.getElementById('gyro-res-age').innerText = `${formatSignificant(gyro.median, 8)} Myr`;
    document.getElementById('gyro-res-ci').innerText = gyro.lower != null && gyro.upper != null
      ? `${formatSignificant(gyro.lower, 8)} - ${formatSignificant(gyro.upper, 8)} Myr`
      : (gyro.ci || '-');
    document.getElementById('gyro-summary-card').style.display = 'block';
    document.getElementById('gyro-results-panel').style.display = 'block';
  }
  if (bafflesCalculated) {
    document.getElementById('baffles-res-li-age').innerText = baffles.liStats
      ? `${formatSignificant(baffles.liStats.median, 8)} Myr`
      : 'N/A';
    document.getElementById('baffles-res-act-age').innerText = baffles.activityStats
      ? `${formatSignificant(baffles.activityStats.median, 8)} Myr`
      : '-';
    document.getElementById('baffles-res-comb-age').innerText = `${formatSignificant(baffles.median, 8)} Myr`;
    document.getElementById('baffles-summary-card').style.display = 'block';
    document.getElementById('baffles-results-panel').style.display = 'block';
    const caveatEl = document.getElementById('baffles-caveat-hint');
    if (caveatEl) {
      const caveats = [baffles.caSkipReason, baffles.liSkipReason].filter(Boolean);
      caveatEl.textContent = caveats.join(' ');
      caveatEl.hidden = caveats.length === 0;
    }
  }
  if (state.results.isochrone?.status === 'Calculated') {
    renderIsochroneResult(state.results.isochrone);
    document.getElementById('isochrone-results-panel').style.display = 'block';
  }
  if (state.results.cluster) renderClusterResult(state.results.cluster);
  updateWorkflowStrip();
}

function renderTargetMetadata() {
  const target = state.activeTarget || {};
  document.getElementById('meta-gaiaid').innerText = target.gaiaid || '-';
  document.getElementById('meta-ra').innerText = formatMetadata(target.ra, 10);
  document.getElementById('meta-dec').innerText = formatMetadata(target.dec, 10);
  document.getElementById('meta-parallax').innerText = formatMetadata(target.parallax, 8);
  document.getElementById('meta-rv').innerText = formatMetadata(target.rv, 8);
}

function upsertReferenceRow(row, replaceMissingOnly = false) {
  const index = state.refTable.findIndex(existing => existing.id === row.id);
  if (index === -1) {
    state.refTable.push(JSON.parse(JSON.stringify(row)));
    return;
  }
  const previous = state.refTable[index];
  const existingHasValue = state.refTable[index].value !== null && state.refTable[index].value !== '';
  const incomingHasValue = row.value !== null && row.value !== '';
  if (!replaceMissingOnly || (!existingHasValue && incomingHasValue)) {
    state.refTable[index] = JSON.parse(JSON.stringify(row));
    const changed = Number(previous.value) !== Number(row.value)
      || Number(previous.uncertainty) !== Number(row.uncertainty);
    if (changed && ['Teff', 'Prot'].includes(row.parameter)) invalidateMethod('gyro');
    if (changed && ['Teff', 'Li EW', "log R'HK", 'Fe/H'].includes(row.parameter)) invalidateMethod('baffles');
    if (changed && ISOCHRONE_INPUT_SPECS.some(spec => spec.id === row.id)) invalidateIsochrone();
  }
}

async function queryVizierParameters({ parameter = null, populateDefaults = false } = {}) {
  if (!state.activeTarget) {
    showToast('Load a target before querying VizieR');
    return;
  }
  const ra = Number(state.activeTarget.ra);
  const dec = Number(state.activeTarget.dec);
  if (!Number.isFinite(ra) || !Number.isFinite(dec)) {
    showToast('The active target has no coordinates; resolve it with SIMBAD first');
    return;
  }

  const queryButton = document.getElementById('btn-query-vizier');
  const status = document.getElementById('catalog-query-status');
  const panel = document.getElementById('catalog-query-panel');
  queryButton.disabled = true;
  queryButton.innerText = 'Querying VizieR...';
  panel.hidden = false;
  status.innerText = parameter ? `Searching for ${parameter}...` : 'Searching nearby catalogs...';

  try {
    const body = await apiCall('query/vizier', {
      name: state.activeTarget.name,
      gaia_id: state.activeTarget.gaiaid,
      ra,
      dec,
      radius: 3,
      parameter,
    });
    state.catalogCandidates = body.candidates || [];
    state.availableVizierParameters = body.available_parameters || [];

    if (populateDefaults || !parameter) {
      (body.ref_table || []).forEach(row => upsertReferenceRow(row, true));
      renderRefTable();
    }
    renderCatalogCandidates();
    renderVizierParameterList();
    status.innerText = `${body.tables?.length || 0} tables; ${state.catalogCandidates.length} matching measurements`;
    showToast('VizieR query complete');
  } catch (error) {
    status.innerText = error.message;
    showToast(`VizieR query failed: ${error.message}`);
  } finally {
    queryButton.disabled = false;
    queryButton.innerText = 'Query VizieR Parameters';
  }
}

async function querySimbadTarget() {
  const searchName = document.getElementById('target-search-input').value.trim();
  const name = searchName || state.activeTarget?.name;
  if (!name) {
    showToast('Enter or load a target before querying SIMBAD');
    return;
  }

  const button = document.getElementById('btn-query-simbad');
  const panel = document.getElementById('catalog-query-panel');
  const resultBox = document.getElementById('simbad-result');
  button.disabled = true;
  button.innerText = 'Querying SIMBAD...';
  panel.hidden = false;

  try {
    const body = await apiCall('query/simbad', { name });
    const target = body.target;
    state.simbadTarget = target;
    if (state.activeTarget) {
      state.activeTarget.name = target.name || state.activeTarget.name;
      state.activeTarget.ra = target.ra;
      state.activeTarget.dec = target.dec;
      if (target.gaiaid) state.activeTarget.gaiaid = target.gaiaid;
      renderTargetMetadata();
      document.getElementById('status-target-name').innerText = state.activeTarget.name;
    }
    const sourceUrl = safeSourceUrl(target.source_url);
    resultBox.innerHTML = `
      <strong>${escapeHtml(target.name)}</strong>
      ${target.object_type ? ` · ${escapeHtml(target.object_type)}` : ''}
      ${target.gaiaid ? ` · Gaia ${escapeHtml(target.gaiaid)}` : ''}
      ${sourceUrl ? ` · <a class="catalog-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">Verify in SIMBAD ↗</a>` : ''}
    `;
    resultBox.hidden = false;
    showToast(`SIMBAD resolved ${name}`);
  } catch (error) {
    resultBox.innerText = error.message;
    resultBox.hidden = false;
    showToast(`SIMBAD query failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.innerText = 'Query SIMBAD';
  }
}

function renderVizierParameterList() {
  const datalist = document.getElementById('vizier-parameter-list');
  datalist.innerHTML = state.availableVizierParameters
    .map(parameter => `<option value="${escapeHtml(parameter)}"></option>`)
    .join('');
}

function renderCatalogCandidates() {
  const tbody = document.getElementById('catalog-candidate-body');
  if (!state.catalogCandidates.length) {
    tbody.innerHTML = '<tr><td colspan="6">No matching measurements found.</td></tr>';
    return;
  }

  tbody.innerHTML = state.catalogCandidates.map((candidate, index) => {
    const group = escapeHtml(candidate.id || `candidate-${index}`);
    const sourceUrl = safeSourceUrl(candidate.source_url);
    const value = formatSignificant(candidate.value, 10) || '—';
    const uncertainty = formatSignificant(candidate.uncertainty, 8) || '—';
    const unit = candidate.unit ? ` ${escapeHtml(candidate.unit)}` : '';
    return `
      <tr>
        <td><input type="radio" name="catalog-candidate-${group}" data-index="${index}" aria-label="Select ${escapeHtml(candidate.parameter)} from ${escapeHtml(candidate.catalog)}"></td>
        <td><strong>${escapeHtml(candidate.parameter)}</strong></td>
        <td class="table-input-mono">${escapeHtml(value)}${unit}</td>
        <td class="table-input-mono">${escapeHtml(uncertainty)}</td>
        <td>${escapeHtml(candidate.column)}</td>
        <td>${sourceUrl
          ? `<a class="catalog-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(candidate.reference || candidate.catalog)} ↗</a>`
          : escapeHtml(candidate.reference || candidate.catalog)}</td>
      </tr>
    `;
  }).join('');
}

function renderRefTable() {
  const tbody = document.getElementById('ref-table-body');
  tbody.innerHTML = '';

  state.refTable.forEach((row, index) => {
    const tr = document.createElement('tr');
    const sourceUrl = safeSourceUrl(row.source_url);
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.parameter)}</strong></td>
      <td><input type="number" class="table-input table-input-mono" value="${escapeHtml(formatSignificant(row.value, 10))}" step="any" data-field="value" data-index="${index}"></td>
      <td><input type="number" class="table-input table-input-mono" value="${escapeHtml(formatSignificant(row.uncertainty, 8))}" min="0" step="any" data-field="uncertainty" data-index="${index}"></td>
      <td><input type="text" class="table-input" value="${escapeHtml(row.reference)}" data-field="reference" data-index="${index}"></td>
      <td><input type="text" class="table-input" value="${escapeHtml(row.catalog)}" data-field="catalog" data-index="${index}"></td>
      <td>${sourceUrl ? `<a class="catalog-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">Open ↗</a>` : '<span class="source-missing">—</span>'}</td>
      <td style="text-align: center;">
        <button class="btn btn-danger btn-circle btn-sm btn-delete-row" data-index="${index}">×</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Setup input change event delegation
  tbody.querySelectorAll('.table-input').forEach(input => {
    input.addEventListener('change', (e) => {
      const idx = parseInt(e.target.dataset.index);
      const field = e.target.dataset.field;
      let val = e.target.value;

      if (field === 'value' || field === 'uncertainty') {
        val = val === '' ? null : parseFloat(val);
      }
      state.refTable[idx][field] = val;
      const parameter = state.refTable[idx].parameter;
      const paramId = state.refTable[idx].id;
      if (['Teff', 'Prot'].includes(parameter)) invalidateMethod('gyro');
      if (['Teff', 'Li EW', "log R'HK", 'Fe/H'].includes(parameter)) invalidateMethod('baffles');
      if (ISOCHRONE_INPUT_SPECS.some(spec => spec.id === paramId)) invalidateIsochrone();
    });
  });

  // Setup row delete click handlers
    tbody.querySelectorAll('.btn-delete-row').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(btn.dataset.index);
      state.refTable.splice(idx, 1);
      invalidateMethod('gyro');
      invalidateMethod('baffles');
      invalidateIsochrone();
      renderRefTable();
      showToast("Row removed from table");
    });
  });
}

function setupCatalogActions() {
  document.getElementById('btn-query-vizier').addEventListener('click', () => {
    queryVizierParameters({ populateDefaults: true });
  });
  document.getElementById('btn-query-simbad').addEventListener('click', querySimbadTarget);
  document.getElementById('btn-close-catalog-panel').addEventListener('click', () => {
    document.getElementById('catalog-query-panel').hidden = true;
  });
  document.getElementById('btn-find-vizier-param').addEventListener('click', () => {
    const parameter = document.getElementById('vizier-parameter-input').value.trim();
    if (!parameter) {
      showToast('Choose a VizieR parameter first');
      return;
    }
    queryVizierParameters({ parameter, populateDefaults: false });
  });
  document.getElementById('btn-add-catalog-selection').addEventListener('click', () => {
    const selected = document.querySelectorAll('#catalog-candidate-body input[type="radio"]:checked');
    if (!selected.length) {
      showToast('Select at least one catalog measurement');
      return;
    }
    selected.forEach(input => {
      const candidate = state.catalogCandidates[Number(input.dataset.index)];
      if (candidate) upsertReferenceRow(candidate);
    });
    renderRefTable();
    showToast(`Added ${selected.length} catalog measurement${selected.length === 1 ? '' : 's'} to ref_table`);
  });
}

async function saveTargetState({ silent = false } = {}) {
  if (!state.activeTarget) {
    if (!silent) showToast('Select a target before saving');
    return false;
  }
  try {
    state.results.pageSettings = collectPageSettings();
    await apiCall('targets/save', {
      target: state.activeTarget,
      ref_table: state.refTable,
      results: state.results,
    });
    state.activeTarget.parameters = JSON.parse(JSON.stringify(state.refTable));
    const identity = targetIdentity(state.activeTarget);
    state.savedTargets = state.savedTargets.filter(target => targetIdentity(target) !== identity);
    state.savedTargets.unshift({
      name: state.activeTarget.name,
      gaiaid: state.activeTarget.gaiaid,
      parameters: emptyDefaultParameters(),
      isSaved: true,
    });
    if (!silent) showToast(`Saved ref_table and results for ${state.activeTarget.name}`);
    return true;
  } catch (error) {
    showToast(`Database save failed: ${error.message}`);
    return false;
  }
}

function setupTableActions() {
  document.getElementById('btn-add-row').addEventListener('click', () => {
    // Show prompt to select parameter
    const param = prompt("Enter parameter name (e.g., Teff, Prot, Li EW, log R'HK, Fe/H, etc.):");
    if (!param) return;

    // Add default row
    state.refTable.push({
      parameter: param,
      value: 0,
      uncertainty: 0,
      reference: "User manual",
      catalog: "User",
      id: param.toLowerCase().replace(/[^a-z]/g, ''),
      source_url: '',
    });

    invalidateMethod('gyro');
    invalidateMethod('baffles');
    invalidateIsochrone();
    renderRefTable();
    showToast(`Added row for "${param}"`);
  });

  document.getElementById('btn-save-table').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.innerText = 'Saving...';
    await saveTargetState();
    button.disabled = false;
    button.innerText = 'Apply Changes';
  });
}

function setupModelRunControls() {
  document.getElementById('btn-run-gyro').addEventListener('click', runGyro);
  document.getElementById('btn-run-baffles').addEventListener('click', runBaffles);
}

// ================= 6. Gyro-interp Page Logic =================

async function runGyro() {
    const runBtn = document.getElementById('btn-run-gyro');
    const teffRow = state.refTable.find(r => r.parameter === 'Teff');
    const protRow = state.refTable.find(r => r.parameter === 'Prot');
    const teffVal = parseFloat(teffRow?.value);
    const protVal = parseFloat(protRow?.value);
    if (!Number.isFinite(teffVal) || !Number.isFinite(protVal) || protVal <= 0) {
      showToast('Add valid Teff and positive Prot measurements to the reference table');
      return;
    }

    const teffErr = positiveUncertainty(teffRow, 100);
    const protErr = positiveUncertainty(protRow, Math.max(0.01 * protVal, 0.01));

    // Show loading spinner
    document.getElementById('gyro-results-panel').style.display = 'none';
    document.getElementById('gyro-loader').style.display = 'flex';
    runBtn.disabled = true;

    try {
      const result = await apiCall('run/gyro', {
        teff: teffVal, prot: protVal,
        teff_err: teffErr, prot_err: protErr
      });

      // Resample API PDF onto common JS ageGrid
      const pdf = interpolatePDF(result.age_grid, result.pdf, ageGrid);

      // Save results
      state.results.gyro.agePDF = pdf;
      state.results.gyro.median = result.median;
      state.results.gyro.lower = result.lower_1sig;
      state.results.gyro.upper = result.upper_1sig;
      state.results.gyro.ci = `[${formatSignificant(result.lower_1sig, 8)}, ${formatSignificant(result.upper_1sig, 8)}]`;
      state.results.gyro.status = 'Calculated';
      invalidateJointResult();

      // Update UI elements
      document.getElementById('gyro-res-age').innerText = `${formatSignificant(result.median, 8)} Myr`;
      document.getElementById('gyro-res-ci').innerText = `${formatSignificant(result.lower_1sig, 8)} - ${formatSignificant(result.upper_1sig, 8)} Myr`;
      document.getElementById('gyro-summary-card').style.display = 'block';

      // Update Pipeline Status Dot
      document.getElementById('status-gyro-dot').className = "dot dot-active";
      document.getElementById('status-gyro-text').innerText = `${formatSignificant(result.median, 8)} Myr`;
      updateWorkflowStrip();

      document.getElementById('gyro-loader').style.display = 'none';
      document.getElementById('gyro-results-panel').style.display = 'block';
      drawGyroPlots();

      await saveTargetState({ silent: true });
      showToast("gyro-interp run completed successfully");
    } catch (err) {
      // No simulated fallback: an age here would not have come from
      // gyrointerp's Bouma+2023 rotation-sequence grids, so we show the
      // failure instead of a fabricated number.
      state.results.gyro.agePDF = null;
      state.results.gyro.median = null;
      state.results.gyro.ci = null;
      state.results.gyro.status = 'Error';
      invalidateJointResult();

      document.getElementById('gyro-res-age').innerText = 'Error';
      document.getElementById('gyro-res-ci').innerText = err.message;
      document.getElementById('gyro-summary-card').style.display = 'block';
      document.getElementById('status-gyro-dot').className = "dot dot-danger";
      document.getElementById('status-gyro-text').innerText = "Error";
      updateWorkflowStrip();

      document.getElementById('gyro-loader').style.display = 'none';
      document.getElementById('gyro-results-panel').style.display = 'none';

      showToast(`gyro-interp error: ${err.message}`);
    } finally {
      runBtn.disabled = false;
    }
}

function setupGyroModelOverlay() {
  const checkbox = document.getElementById('chk-gyro-models');
  if (!checkbox) return;
  const ageRow = document.getElementById('gyro-model-ages');

  checkbox.addEventListener('change', async () => {
    if (ageRow) ageRow.hidden = !checkbox.checked;
    if (checkbox.checked && !gyroSlowSequenceCache) {
      checkbox.disabled = true;
      try {
        gyroSlowSequenceCache = await apiGet('gyro/slow-sequence');
      } catch (error) {
        showToast(`Could not load empirical models: ${error.message}`);
        checkbox.checked = false;
        if (ageRow) ageRow.hidden = true;
      } finally {
        checkbox.disabled = false;
      }
    }
    if (checkbox.checked && gyroSlowSequenceCache?.status === 'ok') {
      fitGyroGridAxisToModels();
    }
    drawGyroPlots();
    saveTargetState({ silent: true });
  });

  // These are display-only settings (no model re-run needed to see them
  // take effect), so unlike ref_table edits or a run's own success handler,
  // nothing else would ever persist them -- save on every change, not just
  // when some other action happens to call saveTargetState() anyway.
  document.querySelectorAll('.chk-gyro-age').forEach(ageCheckbox => {
    ageCheckbox.addEventListener('change', () => {
      drawGyroPlots();
      saveTargetState({ silent: true });
    });
  });

  ['gyro-grid-xmin', 'gyro-grid-xmax', 'gyro-grid-ymin', 'gyro-grid-ymax'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      drawGyroPlots();
      saveTargetState({ silent: true });
    });
  });
  document.getElementById('chk-gyro-grid-logy')?.addEventListener('change', () => {
    drawGyroPlots();
    saveTargetState({ silent: true });
  });
  document.getElementById('chk-gyro-grid-showgrid')?.addEventListener('change', () => {
    drawGyroPlots();
    saveTargetState({ silent: true });
  });
}

// Sets the axis-settings inputs to the empirical slow-sequence data's own
// Teff/Prot range (across all fetched ages, not just the currently-checked
// ones, so later toggling which ages are shown doesn't require re-fitting).
// Runs once when the overlay is switched on; the user's own edits after
// that aren't overwritten again until they toggle it off and back on.
function fitGyroGridAxisToModels() {
  const { teff, prot } = gyroSlowSequenceCache;
  const teffMin = Math.min(...teff);
  const teffMax = Math.max(...teff);
  const allProt = Object.values(prot).flat().filter(v => v != null);
  const protMin = Math.min(...allProt);
  const protMax = Math.max(...allProt);

  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.value = Math.round(value * 100) / 100;
  };
  setValue('gyro-grid-xmin', teffMin);
  setValue('gyro-grid-xmax', teffMax);
  setValue('gyro-grid-ymin', protMin);
  setValue('gyro-grid-ymax', protMax);
}

function gyroGridAxisSettings() {
  const num = (id, fallback) => {
    const value = Number(document.getElementById(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    xmin: num('gyro-grid-xmin', 3500),
    xmax: num('gyro-grid-xmax', 6500),
    ymin: num('gyro-grid-ymin', 0),
    ymax: num('gyro-grid-ymax', 40),
    logY: document.getElementById('chk-gyro-grid-logy')?.checked ?? false,
    showGrid: document.getElementById('chk-gyro-grid-showgrid')?.checked ?? true,
  };
}

function drawGyroPlots() {
  if (!state.results.gyro.agePDF || state.results.gyro.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }

  const pdf = state.results.gyro.agePDF;
  const targetTeff = Number(state.refTable.find(r => r.parameter === 'Teff')?.value);
  const targetProt = Number(state.refTable.find(r => r.parameter === 'Prot')?.value);
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const protRow = state.refTable.find(r => r.parameter === 'Prot');
  const teffErr = teffRow ? teffRow.uncertainty : 100;
  const protErr = protRow ? protRow.uncertainty : 0.2;

  // PLOT 1: target's (Teff, Prot) input, optionally with the empirical
  // slow-sequence isochrones overlaid. Earlier versions overlaid Pleiades/
  // Hyades/Sun sequences computed from a simplified Barnes (2007) power law
  // reimplemented in JS -- not what gyro-interp actually uses, so plotting
  // it next to a real posterior would have been misleading. The overlay
  // below is different: it's gyrointerp.models.slow_sequence itself (Bouma+
  // 2023 reference-cluster fits), fetched from the real package via
  // GET /api/gyro/slow-sequence, so it's safe to show next to the posterior.
  const dataGrid = [];
  const overlayCheckbox = document.getElementById('chk-gyro-models');
  if (overlayCheckbox?.checked && gyroSlowSequenceCache?.status === 'ok') {
    const { teff, ages_myr, prot } = gyroSlowSequenceCache;
    const checkedAges = new Set(
      Array.from(document.querySelectorAll('.chk-gyro-age:checked')).map(el => Number(el.value))
    );
    ages_myr.filter(age => checkedAges.has(age)).forEach(age => {
      dataGrid.push({
        x: teff,
        y: prot[String(age)],
        mode: 'lines',
        type: 'scatter',
        name: `${age} Myr`,
        line: { color: GYRO_SLOW_SEQUENCE_COLORS[age] || '#9ca3af', width: 2 },
        hoverinfo: 'skip',
      });
    });
  }
  dataGrid.push({
    x: [targetTeff],
    y: [targetProt],
    error_x: { type: 'data', array: [teffErr], visible: true, color: '#4a5568' },
    error_y: { type: 'data', array: [protErr], visible: true, color: '#4a5568' },
    mode: 'markers',
    name: state.activeTarget ? state.activeTarget.name : 'Target',
    marker: { color: '#7c3aed', size: 12, line: { color: '#ffffff', width: 2 }, symbol: 'star' }
  });

  const axis = gyroGridAxisSettings();
  // Plotly log-axis ranges are specified in log10 units, not data units; a
  // ymin of 0 (the linear default) isn't valid on a log axis, so floor it.
  const logYMin = axis.ymin > 0 ? axis.ymin : 0.1;
  const layoutGrid = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: {
      title: 'Teff (K)',
      showgrid: axis.showGrid,
      gridcolor: 'rgba(0,0,0,0.06)',
      color: '#4a5568',
      range: [axis.xmax, axis.xmin], // inverted: hotter stars on the left
    },
    yaxis: {
      title: 'Prot (days)',
      showgrid: axis.showGrid,
      gridcolor: 'rgba(0,0,0,0.06)',
      color: '#4a5568',
      type: axis.logY ? 'log' : 'linear',
      range: axis.logY ? [Math.log10(logYMin), Math.log10(axis.ymax)] : [axis.ymin, axis.ymax],
    },
    legend: { x: 0, y: 1, font: { color: '#4a5568' }, bgcolor: 'rgba(255,255,255,0.8)' }
  };

  Plotly.newPlot('plot-gyro-grid', dataGrid, layoutGrid, { responsive: true, displayModeBar: false, staticPlot: true });

  // PLOT 2: Posterior PDF
  const dataPDF = [{
    x: ageGrid,
    y: pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Posterior',
    line: { color: '#7c3aed', width: 2.5 },
    fillcolor: 'rgba(124, 58, 237, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.newPlot('plot-gyro-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}

// ================= 7. BAFFLES Page Logic =================

async function runBaffles() {
    const runBtn = document.getElementById('btn-run-baffles');
    const teffRow = state.refTable.find(r => r.parameter === 'Teff');
    const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
    const rhkRow = state.refTable.find(r => r.parameter === 'log R\'HK');
    const teffVal = parseFloat(teffRow?.value);
    const liewVal = liewRow?.value == null || liewRow.value === '' ? null : Number(liewRow.value);
    const rhkVal = rhkRow?.value == null || rhkRow.value === '' ? null : Number(rhkRow.value);
    // rhk and liew are each individually optional (a star may have only one
    // measurement, or a B-V outside one indicator's range -- see
    // run_baffles.py's per-indicator skip), but at least one is required.
    const hasValidLiew = liewVal !== null && Number.isFinite(liewVal);
    const hasValidRhk = rhkVal !== null && Number.isFinite(rhkVal);
    if (!Number.isFinite(teffVal) || (!hasValidRhk && !hasValidLiew)) {
      showToast("Add valid Teff and at least one of log R'HK or Li EW to the reference table");
      return;
    }

    const teffErr = positiveUncertainty(teffRow, 100);
    const liewErr = positiveUncertainty(liewRow, 15);
    const rhkErr = positiveUncertainty(rhkRow, 0.05);
    const fehRow = state.refTable.find(r => r.parameter === 'Fe/H');

    // BAFFLES needs (B-V)o, not Teff. The backend derives it server-side from
    // the real Pecaut & Mamajek dwarf table (wakai.catalog.get_bv_from_teff)
    // rather than the client guessing it from an uncited polynomial.
    document.getElementById('baffles-results-panel').style.display = 'none';
    document.getElementById('baffles-loader').style.display = 'flex';
    runBtn.disabled = true;

    try {
      const result = await apiCall('run/baffles', {
        teff: teffVal, teff_err: teffErr,
        rhk: hasValidRhk ? rhkVal : undefined, liew: hasValidLiew ? liewVal : undefined,
        liew_err: liewErr, rhk_err: rhkErr,
        feh: fehRow ? fehRow.value : undefined
      });

      // Resample PDFs onto common JS ageGrid. ca_pdf/ca_stats can be null if
      // this star's B-V falls outside BAFFLES's calibrated range for
      // chromospheric activity (narrower than lithium's range -- see
      // run_baffles.py's _posterior_or_none); the backend only returns
      // status "ok" here if at least one of ca/li succeeded, so combStats
      // below can't end up with nothing to fall back on.
      const liPdf = result.li_pdf ? interpolatePDF(result.age_grid, result.li_pdf, ageGrid) : null;
      const actPdf = result.ca_pdf ? interpolatePDF(result.age_grid, result.ca_pdf, ageGrid) : null;
      const combinedPdf = result.combined_pdf
        ? interpolatePDF(result.age_grid, result.combined_pdf, ageGrid)
        : (actPdf || liPdf);

      // BAFFLES stats: [p2, p16, p50, p84, p97]
      const liStats = result.li_stats ? {
        median: result.li_stats[2], lower: result.li_stats[1], upper: result.li_stats[3]
      } : null;
      const actStats = result.ca_stats ? {
        median: result.ca_stats[2], lower: result.ca_stats[1], upper: result.ca_stats[3]
      } : null;
      const combStats = result.combined_stats ? {
        median: result.combined_stats[2], lower: result.combined_stats[1], upper: result.combined_stats[3]
      } : (actStats || liStats);

      // Save to state (use resampled PDFs for consistency)
      state.results.baffles.liPDF = liPdf;
      state.results.baffles.actPDF = actPdf;
      state.results.baffles.agePDF = combinedPdf;
      state.results.baffles.median = combStats.median;
      state.results.baffles.lower = combStats.lower;
      state.results.baffles.upper = combStats.upper;
      state.results.baffles.liStats = liStats;
      state.results.baffles.activityStats = actStats;
      state.results.baffles.bvLookup = result.bv_lookup || null;
      state.results.baffles.caSkipReason = result.ca_skip_reason || null;
      state.results.baffles.liSkipReason = result.li_skip_reason || null;
      state.results.baffles.ci = `[${formatSignificant(combStats.lower, 8)}, ${formatSignificant(combStats.upper, 8)}]`;
      state.results.baffles.status = 'Calculated';
      invalidateJointResult();

      // Update Summary Cards
      document.getElementById('baffles-res-li-age').innerText = liStats ? `${formatSignificant(liStats.median, 8)} Myr` : 'N/A';
      document.getElementById('baffles-res-act-age').innerText = actStats ? `${formatSignificant(actStats.median, 8)} Myr` : 'N/A';
      document.getElementById('baffles-res-comb-age').innerText = `${formatSignificant(combStats.median, 8)} Myr`;
      const bvEl = document.getElementById('baffles-res-bv');
      if (bvEl && result.bv_lookup) {
        const bv = result.bv_lookup;
        bvEl.innerText = bv.bv_err != null
          ? `${formatSignificant(bv.bv, 8)} ± ${formatSignificant(bv.bv_err, 6)} (from Teff, Pecaut & Mamajek table)`
          : `${formatSignificant(bv.bv, 8)} (from Teff, Pecaut & Mamajek table)`;
        if (bv.caveat) showToast(bv.caveat);
      }
      const caveatEl = document.getElementById('baffles-caveat-hint');
      if (caveatEl) {
        const caveats = [result.ca_skip_reason, result.li_skip_reason].filter(Boolean);
        caveatEl.textContent = caveats.join(' ');
        caveatEl.hidden = caveats.length === 0;
      }
      document.getElementById('baffles-summary-card').style.display = 'block';

      // Update Pipeline Status Dot
      document.getElementById('status-baffles-dot').className = "dot dot-active";
      document.getElementById('status-baffles-text').innerText = `${formatSignificant(combStats.median, 8)} Myr`;
      updateWorkflowStrip();

      document.getElementById('baffles-loader').style.display = 'none';
      document.getElementById('baffles-results-panel').style.display = 'block';
      drawBafflesPlots();

      await saveTargetState({ silent: true });
      showToast("BAFFLES run completed successfully");
    } catch (err) {
      // No simulated fallback: an age here would not have come from the
      // real BAFFLES calcium/lithium calibrations, so we show the failure
      // instead of a fabricated number.
      state.results.baffles.liPDF = null;
      state.results.baffles.actPDF = null;
      state.results.baffles.agePDF = null;
      state.results.baffles.median = null;
      state.results.baffles.ci = null;
      state.results.baffles.status = 'Error';
      invalidateJointResult();

      document.getElementById('baffles-res-li-age').innerText = 'Error';
      document.getElementById('baffles-res-act-age').innerText = 'Error';
      document.getElementById('baffles-res-comb-age').innerText = err.message;
      document.getElementById('baffles-summary-card').style.display = 'block';
      document.getElementById('status-baffles-dot').className = "dot dot-danger";
      document.getElementById('status-baffles-text').innerText = "Error";
      updateWorkflowStrip();

      document.getElementById('baffles-loader').style.display = 'none';
      document.getElementById('baffles-results-panel').style.display = 'none';

      showToast(`BAFFLES error: ${err.message}`);
    } finally {
      runBtn.disabled = false;
    }
}

function drawBafflesPlots() {
  if (!state.results.baffles.agePDF || state.results.baffles.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }

  const liPdf = state.results.baffles.liPDF;
  const actPdf = state.results.baffles.actPDF;
  const combPdf = state.results.baffles.agePDF;

  // PLOT 1: Individual Age Posteriors
  const dataIndiv = [
    {
      x: ageGrid,
      y: liPdf,
      type: 'scatter',
      mode: 'lines',
      name: 'Lithium Depletion (Li EW)',
      line: { color: '#ea580c', width: 2 }
    },
    {
      x: ageGrid,
      y: actPdf,
      type: 'scatter',
      mode: 'lines',
      name: 'Chromospheric Activity (log R\'HK)',
      line: { color: '#d946ef', width: 2 }
    }
  ].filter(trace => Array.isArray(trace.y));

  const layoutIndiv = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 46, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    // Horizontal legend above the plot (not x:0,y:1 inside it) -- a PDF's
    // peak can land anywhere along the age axis depending on the star, so
    // no in-plot corner is reliably clear of the curves themselves. Extra
    // top margin vs. the analogous summary-page legend: this chart's two
    // labels ("Lithium Depletion (Li EW)", "Chromospheric Activity
    // (log R'HK)") are longer and this container is shorter (360px vs
    // 440px), so it needs more headroom to clear the axes area cleanly.
    legend: { x: 0, y: 1.22, orientation: 'h', font: { color: '#4a5568' }, bgcolor: 'transparent' }
  };

  Plotly.newPlot('plot-baffles-indiv', dataIndiv, layoutIndiv, { responsive: true, displayModeBar: false, staticPlot: true });

  // PLOT 2: Combined BAFFLES PDF
  const dataPDF = [{
    x: ageGrid,
    y: combPdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Combined BAFFLES',
    line: { color: '#ea580c', width: 2.5 },
    fillcolor: 'rgba(234, 88, 12, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.newPlot('plot-baffles-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}

// ================= 7B. Cluster Membership Logic =================

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function syncClusterInputs() {
  const target = state.activeTarget;
  const runBtn = document.getElementById('btn-run-cluster');
  const hint = document.getElementById('cluster-input-hint');

  const setField = (id, value, digits = 8) =>
    (document.getElementById(id).innerText = value == null ? '-' : formatMetadata(value, digits));

  document.getElementById('cluster-in-gaiaid').innerText = target?.gaiaid || '-';
  setField('cluster-in-parallax', target?.parallax);
  setField('cluster-in-pmra', target?.pmra);
  setField('cluster-in-pmdec', target?.pmdec);
  setField('cluster-in-rv', target?.rv ?? target?.radial_velocity);

  const hasId = Boolean(String(target?.gaiaid || '').trim());
  const hasCoords = [target?.ra, target?.dec, target?.parallax].every(v => Number.isFinite(Number(v)));
  const ready = Boolean(target) && (hasId || hasCoords);
  runBtn.disabled = !ready;

  // Clear any stale result rendered for a previous target.
  if (!state.results.cluster) {
    document.getElementById('cluster-results-panel').style.display = 'none';
    document.getElementById('cluster-loader').style.display = 'none';
    document.getElementById('cluster-summary-card').style.display = 'none';
  }
  if (!Object.keys(state.results.madysCmd?.plots || {}).length) {
    document.getElementById('madys-cmd-results-panel').style.display = 'none';
    document.getElementById('madys-cmd-loader').style.display = 'none';
  }

  if (!target) {
    hint.innerText = 'Load a target on the Target page first.';
  } else if (!ready) {
    hint.innerText = 'Target needs a Gaia ID, or RA/Dec/parallax for a crossmatch.';
  } else if (!Number.isFinite(Number(target?.pmra)) || !Number.isFinite(Number(target?.pmdec))) {
    hint.innerText = 'No proper motion on this target — membership will be scored on parallax only.';
  } else {
    hint.innerText = 'Ready. Gaia DR3 membership is checked first, then a kinematic crossmatch.';
  }

  // MADYS strictly needs a Gaia DR3 ID (it queries "Gaia DR3 <id>" itself,
  // no ra/dec/parallax crossmatch fallback like the membership check above).
  const madysRunBtn = document.getElementById('btn-run-madys-cmd');
  const madysHint = document.getElementById('madys-cmd-input-hint');
  if (madysRunBtn) madysRunBtn.disabled = !hasId;
  if (madysHint) {
    madysHint.innerText = !target
      ? 'Load a target on the Target page first.'
      : !hasId
        ? 'Target needs a Gaia DR3 ID.'
        : 'Ready.';
  }
}

function saveClusterPageSettingsToLocalStorage() {
  const settings = {
    catalog: document.getElementById('cluster-catalog-select')?.value || 'Perren2023',
    clusterName: document.getElementById('cluster-name-input')?.value.trim() || '',
    models: Array.from(document.querySelectorAll('.chk-madys-model:checked')).map(el => el.value),
    plotAges: Array.from(document.querySelectorAll('.chk-madys-age:checked')).map(el => Number(el.value)),
    plotMasses: Array.from(document.querySelectorAll('.chk-madys-mass:checked')).map(el => Number(el.value)),
  };
  localStorage.setItem('wakai:cluster:settings', JSON.stringify(settings));
}

function loadClusterPageSettingsFromLocalStorage() {
  try {
    const raw = localStorage.getItem('wakai:cluster:settings');
    if (!raw) return;
    const settings = JSON.parse(raw);
    if (!settings) return;

    if (settings.catalog) {
      const catalogSelect = document.getElementById('cluster-catalog-select');
      if (catalogSelect) catalogSelect.value = settings.catalog;
    }
    if (settings.clusterName !== undefined) {
      const clusterNameInput = document.getElementById('cluster-name-input');
      if (clusterNameInput) {
        clusterNameInput.value = settings.clusterName;
        updateClusterNameHint();
      }
    }
    if (Array.isArray(settings.models)) {
      document.querySelectorAll('.chk-madys-model').forEach(el => {
        el.checked = settings.models.includes(el.value);
      });
    }
    if (Array.isArray(settings.plotAges)) {
      document.querySelectorAll('.chk-madys-age').forEach(el => {
        el.checked = settings.plotAges.includes(Number(el.value));
      });
    }
    if (Array.isArray(settings.plotMasses)) {
      document.querySelectorAll('.chk-madys-mass').forEach(el => {
        el.checked = settings.plotMasses.includes(Number(el.value));
      });
    }
  } catch (e) {
    console.error('Error loading cluster settings from localStorage:', e);
  }
}

function setupClusterPageControls() {
  const runBtn = document.getElementById('btn-run-cluster');
  runBtn.addEventListener('click', runCluster);
  setupClusterNameSelector();
  document.getElementById('cluster-catalog-select')?.addEventListener('change', () => {
    saveTargetState({ silent: true });
    saveClusterPageSettingsToLocalStorage();
  });

  document.getElementById('btn-run-madys-cmd')?.addEventListener('click', runMadysCmd);
  document.querySelectorAll('.chk-madys-model, .chk-madys-age, .chk-madys-mass').forEach(chk => {
    chk.addEventListener('change', () => {
      saveTargetState({ silent: true });
      saveClusterPageSettingsToLocalStorage();
    });
  });
  // Load initially saved values from localStorage
  loadClusterPageSettingsFromLocalStorage();
}

// Cache of every cluster name in the selected catalog (~18k strings),
// fetched once from GET /api/cluster/names and reused -- target-independent
// reference data, like gyroSlowSequenceCache, so it lives outside `state`.
let clusterNameCache = null;

async function ensureClusterNameCache() {
  if (!clusterNameCache) {
    clusterNameCache = await apiGet('cluster/names');
  }
  return clusterNameCache;
}

// Lets the user override which cluster the Cluster page's diagnostic plots
// (and membership scoring) are computed against, instead of always using
// the auto-matched/nearest cluster -- see run_cluster.py's cluster_name
// override, which restricts crossmatch_target_to_clusters to this one
// candidate. Free text, not a strict selection: whatever's typed is sent
// as-is if the "Check cluster membership" button is clicked with something
// in the box, and the backend returns a clear "not found" error if it
// doesn't match a real cluster name.
function updateClusterNameHint() {
  const input = document.getElementById('cluster-name-input');
  const hint = document.getElementById('cluster-name-hint');
  if (input && hint) {
    const value = input.value.trim();
    hint.hidden = !value;
    hint.textContent = value ? `Will plot against "${value}" instead of the auto-matched cluster.` : '';
  }
}

function setupClusterNameSelector() {
  const input = document.getElementById('cluster-name-input');
  const suggestionsBox = document.getElementById('cluster-name-suggestions');
  if (!input || !suggestionsBox) return;

  // 'change' (fires on blur/commit, not per keystroke) is the persistence
  // point -- 'input' below just drives the live autosuggest dropdown.
  input.addEventListener('change', () => {
    saveTargetState({ silent: true });
    saveClusterPageSettingsToLocalStorage();
  });

  input.addEventListener('input', async () => {
    const query = input.value.trim().toLowerCase();
    suggestionsBox.innerHTML = '';
    updateClusterNameHint();
    if (!query) {
      suggestionsBox.style.display = 'none';
      return;
    }

    let cache;
    try {
      cache = await ensureClusterNameCache();
    } catch (error) {
      showToast(`Could not load cluster list: ${error.message}`);
      return;
    }
    const matches = (cache.names || []).filter(name => name.toLowerCase().includes(query)).slice(0, 25);
    if (!matches.length) {
      suggestionsBox.style.display = 'none';
      return;
    }
    matches.forEach(name => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.innerHTML = `<span>${escapeHtml(name)}</span>`;
      div.addEventListener('click', () => {
        input.value = name;
        suggestionsBox.style.display = 'none';
        updateClusterNameHint();
        // Setting .value programmatically doesn't fire 'change', so persist
        // explicitly here too.
        saveTargetState({ silent: true });
      });
      suggestionsBox.appendChild(div);
    });
    suggestionsBox.style.display = 'block';
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('#cluster-name-input') && !e.target.closest('#cluster-name-suggestions')) {
      suggestionsBox.style.display = 'none';
    }
  });
}

async function runCluster() {
  const target = state.activeTarget;
  if (!target) {
    showToast('Load a target on the Target page first');
    return;
  }
  const gaiaid = String(target.gaiaid || '').trim();
  const hasCoords = [target.ra, target.dec, target.parallax].every(v => Number.isFinite(Number(v)));
  if (!gaiaid && !hasCoords) {
    showToast('Target needs a Gaia ID or RA/Dec/parallax');
    return;
  }

  const catalog = document.getElementById('cluster-catalog-select').value;
  const clusterNameOverride = document.getElementById('cluster-name-input')?.value.trim() || null;
  const payload = {
    catalog,
    target: {
      gaiaid: gaiaid || undefined,
      ra: numberOrNull(target.ra),
      dec: numberOrNull(target.dec),
      parallax: numberOrNull(target.parallax),
      pmra: numberOrNull(target.pmra),
      pmdec: numberOrNull(target.pmdec),
      radial_velocity: numberOrNull(target.rv ?? target.radial_velocity),
    },
  };
  if (clusterNameOverride) payload.cluster_name = clusterNameOverride;

  const runBtn = document.getElementById('btn-run-cluster');
  document.getElementById('cluster-results-panel').style.display = 'none';
  document.getElementById('cluster-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/cluster', payload);
    state.results.cluster = result;
    renderClusterResult(result);
    await saveTargetState({ silent: true });
    showToast(
      result.matched
        ? `Target is consistent with ${result.cluster_name}`
        : `No cluster membership found (nearest: ${result.cluster_name || 'n/a'})`
    );
  } catch (err) {
    document.getElementById('cluster-loader').style.display = 'none';
    showToast(`Cluster check failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

function renderClusterPlot(containerId, dataUri, errorMessage) {
  const container = document.getElementById(containerId);
  if (dataUri) {
    const img = document.createElement('img');
    img.src = dataUri;
    img.alt = 'Cluster diagnostic plot';
    container.replaceChildren(img);
  } else {
    container.innerHTML = `<p style="padding: 1rem; font-size: 0.85rem; color: var(--ink-faint);">Plot unavailable${
      errorMessage ? `: ${escapeHtml(errorMessage)}` : ''
    }</p>`;
  }
}

function renderClusterResult(result) {
  document.getElementById('cluster-loader').style.display = 'none';
  document.getElementById('cluster-results-panel').style.display = 'block';

  const prob = result.membership_probability;
  const probText = prob == null ? '—' : `${(prob * 100).toFixed(1)}%`;

  // Result summary card
  document.getElementById('cluster-res-name').innerText = result.cluster_name || '-';
  document.getElementById('cluster-res-prob').innerText = probText;
  document.getElementById('cluster-res-prob-bar').style.width = `${prob == null ? 0 : Math.max(0, Math.min(1, prob)) * 100}%`;
  const methodLabel = result.method === 'gaia_id' ? 'Gaia DR3 ID in catalog' : 'Kinematic crossmatch';
  document.getElementById('cluster-res-method').innerText = methodLabel;
  document.getElementById('cluster-summary-card').style.display = 'block';

  // Verdict banner
  const verdict = document.getElementById('cluster-verdict');
  verdict.classList.toggle('is-member', Boolean(result.matched));
  verdict.classList.toggle('no-member', !result.matched);
  document.getElementById('cluster-verdict-icon').innerText = result.matched ? '✓' : '✕';
  document.getElementById('cluster-verdict-title').innerText = result.matched
    ? `Consistent with ${result.cluster_name}`
    : 'No cluster membership';
  const subParts = [];
  if (result.method === 'gaia_id') subParts.push('Listed member in ' + result.catalog);
  else if (result.forced_cluster_name) subParts.push('User-selected: ' + (result.cluster_name || 'n/a'));
  else subParts.push('Nearest core: ' + (result.cluster_name || 'n/a'));
  if (result.sep3d_pc != null && result.extent_pc != null) {
    subParts.push(`3D sep ${result.sep3d_pc.toFixed(1)} pc vs extent ${result.extent_pc.toFixed(1)} pc`);
  }
  if (result.mahalanobis != null) subParts.push(`Mahalanobis d=${result.mahalanobis.toFixed(2)} (dof ${result.dof})`);
  document.getElementById('cluster-verdict-sub').innerText = subParts.join(' · ');

  // Summary table
  const body = document.getElementById('cluster-summary-body');
  body.innerHTML = '';
  (result.summary || []).forEach(item => {
    const tr = document.createElement('tr');
    const value = item.value == null || item.value === ''
      ? '-'
      : `${typeof item.value === 'number' ? formatSignificant(item.value, 8) : escapeHtml(item.value)}${item.unit ? ' ' + escapeHtml(item.unit) : ''}`;
    tr.innerHTML = `<td>${escapeHtml(item.label)}</td><td>${value}</td>`;
    body.appendChild(tr);
  });

  // Diagnostic plots
  const plots = result.plots || {};
  renderClusterPlot('plot-cluster-xyz', plots.xyz_uvw, plots.xyz_uvw_error);
  renderClusterPlot('plot-cluster-rdp', plots.rdp_pmrv, plots.rdp_pmrv_error);
}

// Color-magnitude diagram via MADYS -- a separate run from "Check cluster
// membership" above (either can be run first), but reuses that run's
// matched cluster_name (if any and if matched) to overlay member stars.
// See run_madys_cmd.py's docstring for why member counts are capped and why
// MADYS's own photometry fetch (not this app's Gaia/2MASS cache) is slow.
async function runMadysCmd() {
  const target = state.activeTarget;
  if (!target) {
    showToast('Load a target on the Target page first');
    return;
  }
  const gaiaid = String(target.gaiaid || '').trim();
  if (!gaiaid) {
    showToast('Target needs a Gaia DR3 ID');
    return;
  }

  const models = Array.from(document.querySelectorAll('.chk-madys-model:checked')).map(el => el.value);
  if (!models.length) {
    showToast('Check at least one isochrone model to plot');
    return;
  }
  const clusterResult = state.results.cluster;
  const clusterName = clusterResult?.matched ? (clusterResult.cluster_name || null) : null;
  // Empty arrays are meaningful here (not "unset") -- an unchecked chip row
  // really does mean "draw none of these", forwarded as-is to
  // run_madys_cmd.py's _plot_ages()/_plot_masses(), which fall back to
  // MADYS's own defaults only when the value is missing/null, not when it's
  // an explicit empty list.
  const plotAges = Array.from(document.querySelectorAll('.chk-madys-age:checked')).map(el => Number(el.value));
  const plotMasses = Array.from(document.querySelectorAll('.chk-madys-mass:checked')).map(el => Number(el.value));

  const runBtn = document.getElementById('btn-run-madys-cmd');
  document.getElementById('madys-cmd-results-panel').style.display = 'none';
  document.getElementById('madys-cmd-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/madys-cmd', {
      gaiaid, models, cluster_name: clusterName,
      name: target.name || null,
      plot_ages: plotAges, plot_masses: plotMasses,
    });

    state.results.madysCmd = {
      status: 'Calculated',
      plots: result.plots || {},
      errors: result.errors || {},
      clusterName: result.cluster_name || null,
      nMembers: result.n_members || 0,
    };

    renderMadysCmdResult(state.results.madysCmd);
    document.getElementById('madys-cmd-loader').style.display = 'none';
    document.getElementById('madys-cmd-results-panel').style.display = 'block';

    await saveTargetState({ silent: true });
    const failedCount = Object.keys(state.results.madysCmd.errors).length;
    showToast(failedCount
      ? `MADYS CMD plot completed (${failedCount} model${failedCount === 1 ? '' : 's'} failed -- see below)`
      : 'MADYS CMD plot completed');
  } catch (err) {
    document.getElementById('madys-cmd-loader').style.display = 'none';
    showToast(`MADYS CMD plot failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

function renderMadysCmdResult(result) {
  const plots = result?.plots || {};
  const modelNames = Object.keys(plots);
  if (!modelNames.length) return;

  const container = document.getElementById('plot-madys-cmd');
  container.innerHTML = '';
  modelNames.forEach(model => {
    const block = document.createElement('div');
    block.style.marginBottom = '1rem';
    const heading = document.createElement('div');
    heading.className = 'card-eyebrow';
    heading.style.marginBottom = '0.35rem';
    heading.innerText = model.toUpperCase();
    const plotDiv = document.createElement('div');
    plotDiv.className = 'cluster-plot';
    const img = document.createElement('img');
    img.src = plots[model];
    img.alt = `MADYS color-magnitude diagram (${model})`;
    plotDiv.appendChild(img);
    block.appendChild(heading);
    block.appendChild(plotDiv);
    container.appendChild(block);
  });

  const caveat = document.getElementById('madys-cmd-caveat');
  if (!caveat) return;
  const errorEntries = Object.entries(result.errors || {});
  const clusterText = result.nMembers
    ? `overlaid with ${result.nMembers} member${result.nMembers === 1 ? '' : 's'} of ${result.clusterName} (highest membership probability first)`
    : 'target only -- no cluster matched yet';
  const errorText = errorEntries.length
    ? ` Failed: ${errorEntries.map(([m, msg]) => `${m.toUpperCase()} (${msg})`).join('; ')}.`
    : '';
  caveat.innerText = `${modelNames.map(m => m.toUpperCase()).join(', ')} isochrones, ${clusterText}.${errorText}`;
}

// ================= 7C. Isochrones Page Logic =================

// 68% CI width in log10(age) beyond which the fit is treated as too weakly
// constrained to trust -- see run_isochrones.py's docstring and
// defaultResults()'s isochrone.broad comment for why this happens (a
// young and old main-sequence dwarf can be nearly degenerate along a MIST
// isochrone track in color/luminosity space, with no independent age
// handle beyond parallax + photometry). Chosen from real test runs against
// TOI-6715 (a real, verified young star): a photometry-only fit gave a
// ~0.55 dex 68% CI (correctly flagged); a well-constrained fit for an
// evolved/subgiant star would be well under this.
const ISOCHRONE_BROAD_DEX = 0.3;

// Every ref_table row the Isochrones page's SingleStarModel fit can use --
// parallax + all six bands (required to run at all) plus the two optional
// spectroscopic constraints. Drives both the checkbox list (built once by
// setupIsochronesPageControls) and the payload built in runIsochrones, so
// the two stay in sync by construction. `band` maps a row to the photometry
// dict key isochrones2 expects; rows without `band` (parallax/teff/feh) are
// top-level SingleStarModel kwargs instead. logg has no ref_table row (it's
// a fixed, editable dwarf-star prior, not a measurement) so it isn't listed
// here -- see the dedicated #chk-isochrone-logg controls in index.html.
const ISOCHRONE_INPUT_SPECS = [
  { id: 'parallax', label: 'Parallax', unit: 'mas', digits: 8, errDigits: 4, fallbackErr: 0.05, required: true },
  { id: 'bpmag', label: 'BP mag', band: 'BP', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'gmag', label: 'G mag', band: 'G', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'rpmag', label: 'RP mag', band: 'RP', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'jmag', label: 'J mag', band: 'J', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'hmag', label: 'H mag', band: 'H', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'kmag', label: 'K mag', band: 'K', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'teff', label: 'Teff', unit: 'K', digits: 8, errDigits: 6, fallbackErr: 100 },
  { id: 'feh', label: 'Fe/H', digits: 8, errDigits: 6, fallbackErr: 0.1 },
];

function isochroneInputRowHtml(spec) {
  const reqTag = spec.required
    ? ' <span style="color: var(--ink-faint); font-size: 0.75rem;">(required)</span>'
    : '';
  return `
    <div class="form-group" style="display: flex; align-items: center; gap: 0.5rem;">
      <input type="checkbox" id="chk-isochrone-${spec.id}" class="chk-isochrone-input" style="accent-color: var(--isochrone);">
      <label for="chk-isochrone-${spec.id}" style="margin: 0; flex: 1;">${spec.label}${reqTag}</label>
      <span class="meta-value" id="isochrone-in-${spec.id}" style="font-size: 0.85rem;">-</span>
    </div>`;
}

function syncIsochroneInputParams() {
  const target = state.activeTarget;
  const runBtn = document.getElementById('btn-run-isochrones');
  const hint = document.getElementById('isochrone-input-hint');

  let parallaxAvailable = false;
  let anyBandAvailable = false;

  ISOCHRONE_INPUT_SPECS.forEach(spec => {
    const row = state.refTable.find(r => r.id === spec.id);
    const available = Boolean(row && Number.isFinite(Number(row.value)));

    const valueSpan = document.getElementById(`isochrone-in-${spec.id}`);
    if (valueSpan) {
      valueSpan.innerText = available
        ? `${formatSignificant(row.value, spec.digits)} ± ${formatSignificant(row.uncertainty, spec.errDigits)}${spec.unit ? ` ${spec.unit}` : ''}`
        : '—';
    }

    const chk = document.getElementById(`chk-isochrone-${spec.id}`);
    if (chk) {
      chk.disabled = !available;
      if (!available) {
        chk.checked = false;
      } else if ((spec.required || spec.band) && !chk.dataset.userTouched) {
        // Parallax + photometry are auto-checked once available, since the
        // fit can't run without them; Teff/Fe-H stay opt-in (unchecked)
        // until the user explicitly ticks them, matching the page's
        // original teff/feh-only behavior. Either way, once the user
        // manually toggles a box, this sync stops overriding their choice.
        chk.checked = true;
      }
    }

    if (spec.id === 'parallax') parallaxAvailable = available;
    if (spec.band && available) anyBandAvailable = true;
  });

  const hasId = Boolean(String(target?.gaiaid || '').trim());
  const ready = hasId && parallaxAvailable && anyBandAvailable;
  if (runBtn) runBtn.disabled = !ready;
  if (hint) {
    hint.innerText = !target
      ? 'Load a target on the Target page first.'
      : !parallaxAvailable
        ? 'Missing parallax -- fetch it on the Target page first.'
        : !anyBandAvailable
          ? 'No photometry available -- fetch it on the Target page first.'
          : 'Ready. Uncheck any input above to exclude it from the fit.';
  }
}

function setupIsochronesPageControls() {
  const list = document.getElementById('isochrone-input-list');
  if (list) {
    list.innerHTML = ISOCHRONE_INPUT_SPECS.map(isochroneInputRowHtml).join('');
    list.querySelectorAll('.chk-isochrone-input').forEach(chk => {
      chk.addEventListener('change', () => {
        chk.dataset.userTouched = '1';
        saveTargetState({ silent: true });
      });
    });
  }
  const runBtn = document.getElementById('btn-run-isochrones');
  if (runBtn) runBtn.addEventListener('click', runIsochrones);

  // logg + prior bounds have no other natural save checkpoint before the
  // user clicks Run (unlike the checkboxes above, which mirror ref_table
  // availability that a run would already persist) -- save on every change.
  [
    'chk-isochrone-logg', 'isochrone-logg-value', 'isochrone-logg-err',
    'isochrone-age-prior-min', 'isochrone-age-prior-max',
    'isochrone-av-prior-min', 'isochrone-av-prior-max',
    'isochrone-eep-prior-min', 'isochrone-eep-prior-max',
  ].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      saveTargetState({ silent: true });
    });
  });
}

function invalidateIsochrone() {
  state.results.isochrone = defaultResults().isochrone;
  document.getElementById('isochrone-summary-card').style.display = 'none';
  document.getElementById('isochrone-results-panel').style.display = 'none';
  invalidateJointResult();
}

async function runIsochrones() {
  const target = state.activeTarget;
  if (!target) {
    showToast('Load a target on the Target page first');
    return;
  }

  const photometry = {};
  let parallax = null;
  let parallaxErr = null;
  const payload = {};

  ISOCHRONE_INPUT_SPECS.forEach(spec => {
    const chk = document.getElementById(`chk-isochrone-${spec.id}`);
    if (!chk || !chk.checked) return;
    const row = state.refTable.find(r => r.id === spec.id);
    if (!row || !Number.isFinite(Number(row.value))) return;

    const value = Number(row.value);
    const err = positiveUncertainty(row, spec.fallbackErr);
    if (spec.id === 'parallax') {
      parallax = value;
      parallaxErr = err;
    } else if (spec.band) {
      photometry[spec.band] = [value, err];
    } else {
      payload[spec.id] = value;
      payload[`${spec.id}_err`] = err;
    }
  });

  if (parallax === null || Object.keys(photometry).length === 0) {
    showToast('Check parallax plus at least one photometric band to run isochrones');
    return;
  }
  payload.photometry = photometry;
  payload.parallax = parallax;
  payload.parallax_err = parallaxErr;

  const chkLogg = document.getElementById('chk-isochrone-logg');
  if (chkLogg?.checked) {
    const loggValue = Number(document.getElementById('isochrone-logg-value')?.value);
    const loggErr = Number(document.getElementById('isochrone-logg-err')?.value);
    if (Number.isFinite(loggValue) && Number.isFinite(loggErr) && loggErr > 0) {
      payload.logg = loggValue;
      payload.logg_err = loggErr;
    }
  }

  // Age/AV/EEP prior bounds -- age+AV match notebooks/ext_code_isochrones
  // .ipynb's mod.set_prior(age=FlatPrior(...), AV=FlatPrior(...)); EEP isn't
  // touched by the notebook but is still one of SingleStarModel's fit
  // parameters, so it's exposed the same way (default: full MIST grid).
  const ageMin = Number(document.getElementById('isochrone-age-prior-min')?.value);
  const ageMax = Number(document.getElementById('isochrone-age-prior-max')?.value);
  const avMin = Number(document.getElementById('isochrone-av-prior-min')?.value);
  const avMax = Number(document.getElementById('isochrone-av-prior-max')?.value);
  const eepMin = Number(document.getElementById('isochrone-eep-prior-min')?.value);
  const eepMax = Number(document.getElementById('isochrone-eep-prior-max')?.value);
  if (![ageMin, ageMax, avMin, avMax, eepMin, eepMax].every(Number.isFinite)
    || ageMin >= ageMax || avMin >= avMax || eepMin >= eepMax) {
    showToast('Check the age/AV/EEP prior bounds -- min must be less than max');
    return;
  }
  payload.age_prior_bounds = [ageMin, ageMax];
  payload.av_prior_bounds = [avMin, avMax];
  payload.eep_prior_bounds = [eepMin, eepMax];

  const runBtn = document.getElementById('btn-run-isochrones');
  document.getElementById('isochrone-results-panel').style.display = 'none';
  document.getElementById('isochrone-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/isochrones', payload);

    const pdf = interpolatePDF(result.age_grid, result.pdf, ageGrid);
    const median = result.median;
    const lower = median - result.lower_1sig;
    const upper = median + result.upper_1sig;
    const broad = (Math.log10(upper) - Math.log10(lower)) > ISOCHRONE_BROAD_DEX;

    state.results.isochrone = {
      agePDF: pdf,
      median,
      lower,
      upper,
      ci: `[${formatSignificant(lower, 8)}, ${formatSignificant(upper, 8)}]`,
      status: 'Calculated',
      broad,
      mass: result.mass,
      radius: result.radius,
      teffDerived: result.teff_derived,
      logg: result.logg,
      fehDerived: result.feh_derived,
      usedInputs: result.used_inputs,
      priorBounds: result.prior_bounds,
    };
    invalidateJointResult();

    renderIsochroneResult(state.results.isochrone);

    document.getElementById('isochrone-loader').style.display = 'none';
    document.getElementById('isochrone-results-panel').style.display = 'block';
    drawIsochronePlots();

    await saveTargetState({ silent: true });
    showToast(
      broad
        ? 'Isochrone fit complete, but weakly constrained (excluded from joint estimate)'
        : 'Isochrone fit completed successfully'
    );
  } catch (err) {
    document.getElementById('isochrone-loader').style.display = 'none';
    showToast(`Isochrone fit failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

function renderIsochroneResult(iso) {
  document.getElementById('isochrone-res-age').innerText = iso.median != null
    ? `${formatSignificant(iso.median, 8)} Myr`
    : '-';
  const fmtStat = (stat, digits = 6) => stat
    ? `${formatSignificant(stat.median, digits)} [${formatSignificant(stat.lower, digits)}, ${formatSignificant(stat.upper, digits)}]`
    : '-';
  document.getElementById('isochrone-res-mass').innerText = fmtStat(iso.mass);
  document.getElementById('isochrone-res-radius').innerText = fmtStat(iso.radius);
  document.getElementById('isochrone-res-teff').innerText = fmtStat(iso.teffDerived, 6);
  document.getElementById('isochrone-res-logg').innerText = fmtStat(iso.logg, 4);
  document.getElementById('isochrone-res-feh').innerText = fmtStat(iso.fehDerived, 4);

  const used = iso.usedInputs;
  const priors = iso.priorBounds;
  const priorsText = priors
    ? ` Priors: log10(age) in [${priors.age?.[0]}, ${priors.age?.[1]}], AV in [${priors.AV?.[0]}, ${priors.AV?.[1]}], EEP in [${priors.eep?.[0]}, ${priors.eep?.[1]}].`
    : '';
  document.getElementById('isochrone-inputs-used').innerText = used
    ? `Fit used: ${used.bands.join(', ')} photometry, parallax${used.teff ? ', Teff' : ''}${used.feh ? ', Fe/H' : ''}${used.logg ? ', log g' : ''}.${priorsText}`
    : '';
  document.getElementById('isochrone-summary-card').style.display = 'block';

  const caveatCard = document.getElementById('isochrone-caveat-card');
  if (caveatCard) caveatCard.style.display = iso.broad ? 'block' : 'none';
}

function drawIsochronePlots() {
  if (!state.results.isochrone.agePDF || state.results.isochrone.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }
  const pdf = state.results.isochrone.agePDF;

  const dataPDF = [{
    x: ageGrid,
    y: pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Isochrone age',
    line: { color: '#a6740a', width: 2.5 },
    fillcolor: 'rgba(166, 116, 10, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.newPlot('plot-isochrone-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}

// ================= 8. Summary Page Logic =================

// Reduces 2 or 3 independent PDFs (already on the shared `grid`) into one
// product posterior by chaining WakaiPosterior.combineIndependent pairwise
// -- its own scaling-before-integrating (to avoid underflow) still applies
// at each step. Short-circuits on the first non-overlapping pair, since a
// null intermediate .pdf can't feed the next combine.
function combineAllIndependent(grid, pdfs) {
  let acc = { status: 'ok', pdf: pdfs[0], stats: null };
  for (let i = 1; i < pdfs.length; i += 1) {
    acc = WakaiPosterior.combineIndependent(grid, acc.pdf, pdfs[i]);
    if (acc.status !== 'ok') return acc;
  }
  return acc;
}

function updateSummaryPage() {
  const isGyroRun = state.results.gyro.status === 'Calculated';
  const isBafflesRun = state.results.baffles.status === 'Calculated';
  const isIsochroneRun = state.results.isochrone.status === 'Calculated';
  // A very broad (weakly constrained) isochrone posterior is shown on its
  // own page/row but excluded here so it can't dominate/distort the joint
  // product -- see ISOCHRONE_BROAD_DEX.
  const isIsochroneUsable = isIsochroneRun && !state.results.isochrone.broad;

  const gyroPdf = state.results.gyro.agePDF;
  const bafflesPdf = state.results.baffles.agePDF;
  const isochronePdf = state.results.isochrone.agePDF;

  const statusBadge = (calculated, label) => calculated
    ? `<span class="status-indicator"><div class="dot dot-active"></div>${label}</span>`
    : '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Not yet run</span>';

  document.getElementById('td-gyro-status').innerHTML = statusBadge(isGyroRun, 'Active');
  document.getElementById('td-baffles-status').innerHTML = statusBadge(isBafflesRun, 'Active');
  document.getElementById('td-isochrone-status').innerHTML = !isIsochroneRun
    ? statusBadge(false, '')
    : isIsochroneUsable
      ? statusBadge(true, 'Active')
      : '<span class="status-indicator" style="color: var(--danger);"><div class="dot dot-danger"></div>Excluded (weak)</span>';

  const setRow = (prefix, calculated, pdf) => {
    if (calculated) {
      const stats = getStatsFromPDF(pdf);
      document.getElementById(`td-${prefix}-age`).innerText = `${formatSignificant(stats.median, 8)} Myr`;
      document.getElementById(`td-${prefix}-ci`).innerText = `${formatSignificant(stats.lower, 8)}-${formatSignificant(stats.upper, 8)} Myr`;
    } else {
      document.getElementById(`td-${prefix}-age`).innerText = '-';
      document.getElementById(`td-${prefix}-ci`).innerText = '-';
    }
  };
  setRow('gyro', isGyroRun, gyroPdf);
  setRow('baffles', isBafflesRun, bafflesPdf);
  setRow('isochrone', isIsochroneRun, isochronePdf);

  updatePropertiesSummary();
  renderReportedAges();

  // Joint posterior: product of whichever of {gyro, BAFFLES, isochrone}
  // are usable, requiring at least two -- a "joint" of a single method
  // isn't a joint constraint, so don't synthesize one from a partial result.
  // The actual PDF product lives in recomputeJointStatus() (shared with
  // updateWorkflowStrip(), so the top nav's "Joint age" stage stays live on
  // every page, not just this one) -- this only builds the plot's trace
  // list and then renders whatever state.results.joint ends up holding.
  const usable = [];
  if (isGyroRun) usable.push({ label: 'gyro-interp PDF', pdf: gyroPdf, color: '124, 58, 237' });
  if (isBafflesRun) usable.push({ label: 'BAFFLES Combined PDF', pdf: bafflesPdf, color: '234, 88, 12' });
  if (isIsochroneUsable) usable.push({ label: 'Isochrone PDF', pdf: isochronePdf, color: '166, 116, 10' });

  const comparisonTraces = usable.map(m => ({ ...m }));
  if (isIsochroneRun && !isIsochroneUsable) {
    comparisonTraces.push({ label: 'Isochrone PDF (excluded, weakly constrained)', pdf: isochronePdf, color: '166, 116, 10' });
  }

  recomputeJointStatus();

  if (state.results.joint.status === 'Idle') {
    document.getElementById('summary-joint-age').innerText = '-';
    document.getElementById('summary-joint-ci').innerText = 'Run at least two of gyro-interp, BAFFLES, or Isochrones to compute a joint age';
    document.getElementById('td-joint-age').innerText = '-';
    document.getElementById('td-joint-ci').innerText = '-';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Awaiting 2+ methods</span>';

    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    return;
  }

  // Gyrochronology validity guard: Bouma+2023 rotation-sequence grids are
  // only calibrated down to ~80 Myr (youngest reference cluster). If a
  // significant fraction of the gyro posterior mass falls below this floor,
  // the gyro age is an extrapolation and the joint result is unreliable.
  const GYRO_CALIBRATION_FLOOR_MYR = 80;
  const gyroMassBelowFloor = isGyroRun
    ? WakaiPosterior.probabilityBelow(ageGrid, gyroPdf, GYRO_CALIBRATION_FLOOR_MYR)
    : 0;
  document.getElementById('gyro-caveat-card').style.display = gyroMassBelowFloor > 0.5 ? 'block' : 'none';

  if (state.results.joint.status === 'No overlap') {
    document.getElementById('summary-joint-age').innerText = '—';
    document.getElementById('summary-joint-ci').innerText = 'The method posteriors do not overlap';
    document.getElementById('td-joint-age').innerText = '—';
    document.getElementById('td-joint-ci').innerText = '—';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator status-error"><div class="dot dot-danger"></div>Methods disagree</span>';
    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    saveTargetState({ silent: true });
    showToast('No joint age: method posteriors do not overlap');
    return;
  }

  const jointPdf = state.results.joint.agePDF;
  const jointStats = state.results.joint;
  updateWorkflowStrip();

  // Update Page Labels
  document.getElementById('summary-joint-age').innerText = formatSignificant(jointStats.median, 8);
  document.getElementById('summary-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)} - ${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-age').innerText = `${formatSignificant(jointStats.median, 8)} Myr`;
  document.getElementById('td-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)}-${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-status').innerHTML =
    `<span class="status-indicator"><div class="dot dot-active"></div>Synthesized (${usable.length}-way)</span>`;

  // Update Evolutionary Stage Timeline
  updateEvolutionTimeline(jointStats.median);

  // Draw Comparison Plot -- every run method (including an excluded, broad
  // isochrone posterior, for transparency) plus the joint product.
  const data = comparisonTraces.map(t => ({
    x: ageGrid,
    y: t.pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: t.label,
    line: { color: `rgba(${t.color}, 0.7)`, width: 2 },
    fillcolor: `rgba(${t.color}, 0.08)`,
  }));
  data.push({
    x: ageGrid,
    y: jointPdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Joint Posterior PDF',
    line: { color: 'rgba(5, 150, 105, 1)', width: 3 },
    fillcolor: 'rgba(5, 150, 105, 0.18)'
  });

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: 'Stellar age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Normalized Probability', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    legend: { x: 0, y: 1.15, orientation: 'h', font: { color: '#4a5568' }, bgcolor: 'transparent' }
  };

  Plotly.newPlot('plot-summary-comparison', data, layout, { responsive: true, displayModeBar: false, staticPlot: true });
  saveTargetState({ silent: true });
}

function updateEvolutionTimeline(age) {
  // Reset timeline node states
  document.querySelectorAll('.timeline-node').forEach(node => node.classList.remove('active'));

  if (age < 50) {
    document.getElementById('tl-pms').classList.add('active');
  } else if (age >= 50 && age < 200) {
    document.getElementById('tl-zams').classList.add('active');
  } else if (age >= 200 && age < 800) {
    document.getElementById('tl-youngms').classList.add('active');
  } else {
    document.getElementById('tl-oldms').classList.add('active');
  }
}

function updatePropertiesSummary() {
  const container = document.getElementById('summary-target-properties');
  container.innerHTML = '';

  if (!state.activeTarget) return;

  const titleDiv = document.createElement('div');
  titleDiv.innerHTML = `Target: <strong>${state.activeTarget.name}</strong> (Gaia DR3 ${state.activeTarget.gaiaid})`;
  titleDiv.style.marginBottom = '0.5rem';
  container.appendChild(titleDiv);

  state.refTable.forEach(row => {
    const item = document.createElement('div');
    item.style.display = 'flex';
    item.style.justifyContent = 'space-between';
    item.style.padding = '0.25rem 0';
    item.style.borderBottom = '1px solid rgba(0,0,0,0.06)';
    item.innerHTML = `
      <span style="color: var(--text-secondary);">${escapeHtml(row.parameter)}:</span>
      <span style="font-family: var(--font-mono);">${escapeHtml(formatSignificant(row.value, 10) || '—')} ± ${escapeHtml(formatSignificant(row.uncertainty, 8) || '—')} <span style="font-size: 0.75rem; color: var(--text-muted);">(${escapeHtml(row.reference)})</span></span>
    `;
    container.appendChild(item);
  });
}

// Literature ages the user has recorded directly in the reference table
// (e.g. via "+ Add Parameter Row", parameter "Age" with a citation in
// Reference) -- distinct from the computed gyro/BAFFLES/joint rows above,
// so shown as extra rows in the same Comparison table rather than mixed in.
function renderReportedAges() {
  const tbody = document.getElementById('reported-ages-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const reported = state.refTable.filter(row =>
    /age/i.test(row.parameter || '') && row.value != null && Number.isFinite(Number(row.value))
  );
  if (!reported.length) return;

  const header = document.createElement('tr');
  header.innerHTML = `<td colspan="4" style="border-top: 1px solid var(--glass-border); padding-top: 0.75rem; font-size: 0.75rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.04em;">Reported (from reference table)</td>`;
  tbody.appendChild(header);

  reported.forEach(row => {
    const range = row.uncertainty != null && Number.isFinite(Number(row.uncertainty))
      ? `± ${formatSignificant(row.uncertainty, 6)}`
      : '-';
    const hasReference = row.reference && row.reference !== 'No catalog measurement selected';
    const sourceUrl = safeSourceUrl(row.source_url);
    const referenceHtml = sourceUrl
      ? `<a class="catalog-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(hasReference ? row.reference : 'Source')} ↗</a>`
      : escapeHtml(hasReference ? row.reference : 'No reference given');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.parameter)}</strong><br><span style="font-size: 0.75rem; color: var(--ink-faint);">${referenceHtml}</span></td>
      <td>${escapeHtml(formatSignificant(row.value, 8))}</td>
      <td>${range}</td>
      <td><span class="status-indicator"><div class="dot dot-inactive"></div>Reported</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function setupSummaryActions() {
  document.getElementById('btn-export-report').addEventListener('click', () => {
    // Refresh state.results.joint right before reading it, same as
    // updateSummaryPage()/updateWorkflowStrip() -- recomputeJointStatus() is
    // the single source of truth for joint status (see its own comment);
    // reimplementing the gyro+baffles-only check here previously produced a
    // self-contradictory export (e.g. joint.status:'Idle' while joint.age
    // held a real value from a gyro+isochrone-only joint).
    recomputeJointStatus();
    const jointResult = state.results.joint;
    const isochroneResult = state.results.isochrone;
    const clusterResult = state.results.cluster;
    const madysResult = state.results.madysCmd;

    const reportData = {
      target: state.activeTarget ? state.activeTarget.name : "Custom",
      gaiaid: state.activeTarget ? state.activeTarget.gaiaid : null,
      refTable: state.refTable,
      runs: {
        gyro: { age: state.results.gyro.median, ci: state.results.gyro.ci, status: state.results.gyro.status },
        baffles: { age: state.results.baffles.median, ci: state.results.baffles.ci, status: state.results.baffles.status },
        isochrone: {
          age: isochroneResult.median, ci: isochroneResult.ci, status: isochroneResult.status,
          broad: isochroneResult.broad,
        },
        joint: { age: jointResult.median, ci: jointResult.ci, status: jointResult.status },
        // Base64 plot images are intentionally omitted here -- this is a
        // numeric/text report, not a plot archive.
        cluster: clusterResult ? {
          matched: clusterResult.matched,
          clusterName: clusterResult.cluster_name,
          membershipProbability: clusterResult.membership_probability,
          nMembers: clusterResult.n_members,
        } : null,
        madysCmd: {
          status: madysResult.status,
          clusterName: madysResult.clusterName,
          nMembers: madysResult.nMembers,
        },
      },
      exportTime: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wakai_report_${reportData.target.replace(/\s+/g, '_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast("Report exported successfully!");
  });

  document.getElementById('btn-print-report').addEventListener('click', () => {
    window.print();
  });
}

// Initialize on load
window.addEventListener('DOMContentLoaded', initApp);
