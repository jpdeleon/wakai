/**
 * WAKAI DASHBOARD APPLICATION LOGIC
 * Manages state, routing, ref_table editing, physical models, and Plotly visualizations.
 */

// 1. App State
const state = {
  activeTarget: null,
  activePage: 'target',
  clusterResult: null,
  refTable: [],
  catalogCandidates: [],
  availableVizierParameters: [],
  simbadTarget: null,
  savedTargets: [],
  preloadedTargets: [
    {
      name: "TOI-837",
      gaiaid: "593149204901",
      ra: 156.4521,
      dec: -64.3012,
      parallax: 6.942,
      rv: 12.4,
      parameters: [
        { parameter: "Teff", value: 6040, uncertainty: 80, reference: "Bouma+2020", catalog: "TIC", id: "teff" },
        { parameter: "Prot", value: 3.02, uncertainty: 0.05, reference: "Bouma+2020", catalog: "TESS", id: "prot" },
        { parameter: "Li EW", value: 215, uncertainty: 15, reference: "Jeffries+2023", catalog: "VLT/UVES", id: "liew" },
        { parameter: "log R'HK", value: -4.05, uncertainty: 0.05, reference: "Mamajek+2008", catalog: "Keck/HIRES", id: "rhk" },
        { parameter: "Fe/H", value: 0.08, uncertainty: 0.05, reference: "Ahumada+2020", catalog: "VizieR", id: "feh" }
      ]
    },
    {
      name: "TOI-1201",
      gaiaid: "193740920472",
      ra: 45.1982,
      dec: 18.2394,
      parallax: 24.312,
      rv: -4.5,
      parameters: [
        { parameter: "Teff", value: 4100, uncertainty: 100, reference: "Kossakowski+2021", catalog: "TIC", id: "teff" },
        { parameter: "Prot", value: 12.2, uncertainty: 0.2, reference: "Kossakowski+2021", catalog: "TESS", id: "prot" },
        { parameter: "Li EW", value: 145, uncertainty: 20, reference: "Vines+2022", catalog: "HARPS", id: "liew" },
        { parameter: "log R'HK", value: -4.25, uncertainty: 0.08, reference: "Perdelwitz+2024", catalog: "ESO", id: "rhk" },
        { parameter: "Fe/H", value: -0.10, uncertainty: 0.08, reference: "Kossakowski+2021", catalog: "TIC", id: "feh" }
      ]
    },
    {
      name: "HD 189733",
      gaiaid: "293847920194",
      ra: 300.2312,
      dec: 22.7123,
      parallax: 50.56,
      rv: -2.2,
      parameters: [
        { parameter: "Teff", value: 5050, uncertainty: 50, reference: "Bouchy+2005", catalog: "TIC", id: "teff" },
        { parameter: "Prot", value: 11.9, uncertainty: 0.1, reference: "Henry+2008", catalog: "Ground", id: "prot" },
        { parameter: "Li EW", value: 5, uncertainty: 2, reference: "Mishenina+2012", catalog: "McDonald", id: "liew" },
        { parameter: "log R'HK", value: -4.50, uncertainty: 0.04, reference: "Knutson+2010", catalog: "Keck", id: "rhk" },
        { parameter: "Fe/H", value: -0.03, uncertainty: 0.03, reference: "Bouchy+2005", catalog: "TIC", id: "feh" }
      ]
    }
  ],
  results: {
    gyro: { agePDF: null, median: null, ci: null, status: 'Idle' },
    baffles: { agePDF: null, median: null, ci: null, status: 'Idle', liPDF: null, actPDF: null },
    joint: { agePDF: null, median: null, ci: null, status: 'Idle' }
  }
};

// Shared display grid. Model PDFs are zero-padded outside their native support.
const ageGrid = [1, ...Array.from({ length: 2600 }, (_, i) => (i + 1) * 5)];

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
    joint: { agePDF: null, median: null, ci: null, status: 'Idle' },
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

function updateWorkflowStrip() {
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
  state.results.joint = defaultResults().joint;
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

function initApp() {
  setupNavigation();
  setupTargetSelector();
  setupTableActions();
  setupCatalogActions();
  setupModelRunControls();
  setupClusterPageControls();
  setupSummaryActions();
  loadSavedTargetIndex();
  updateWorkflowStrip();
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

function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const pageId = link.getAttribute('data-page');

      // Update UI active states
      document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.getElementById(`page-${pageId}`).classList.add('active');

      state.activePage = pageId;
      onPageEnter(pageId);
    });
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
    if (state.clusterResult) renderClusterResult(state.clusterResult);
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

// ================= 5. Target Hub Logic =================

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
    [...state.preloadedTargets, ...state.savedTargets].forEach(target => {
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
  state.refTable = JSON.parse(JSON.stringify(target.parameters || emptyDefaultParameters()));
  state.catalogCandidates = [];
  state.clusterResult = null;
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
  }
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
      if (['Teff', 'Prot'].includes(parameter)) invalidateMethod('gyro');
      if (['Teff', 'Li EW', "log R'HK", 'Fe/H'].includes(parameter)) invalidateMethod('baffles');
    });
  });

  // Setup row delete click handlers
    tbody.querySelectorAll('.btn-delete-row').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(btn.dataset.index);
      state.refTable.splice(idx, 1);
      invalidateMethod('gyro');
      invalidateMethod('baffles');
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

  // PLOT 1: target's (Teff, Prot) input.
  // Earlier versions overlaid Pleiades/Hyades/Sun reference sequences here,
  // computed from a simplified Barnes (2007) power law reimplemented in JS.
  // That relation isn't what gyro-interp actually uses (Bouma+2023
  // rotation-sequence grids, which include the "stalled" spin-down regime
  // the power law can't capture), so plotting it next to a real gyro-interp
  // posterior would be misleading. Only the real input point is shown here.
  const dataGrid = [
    {
      x: [targetTeff],
      y: [targetProt],
      error_x: { type: 'data', array: [teffErr], visible: true, color: '#4a5568' },
      error_y: { type: 'data', array: [protErr], visible: true, color: '#4a5568' },
      mode: 'markers',
      name: state.activeTarget ? state.activeTarget.name : 'Target',
      marker: { color: '#7c3aed', size: 12, line: { color: '#ffffff', width: 2 }, symbol: 'star' }
    }
  ];

  const layoutGrid = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: 'Teff (K)', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568', range: [6500, 3500] },
    yaxis: { title: 'Prot (days)', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568', range: [0, 40] },
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
    const rhkVal = parseFloat(rhkRow?.value);
    if (!Number.isFinite(teffVal) || !Number.isFinite(rhkVal) || (liewVal !== null && !Number.isFinite(liewVal))) {
      showToast('Add valid Teff, log R\'HK, and optional Li EW measurements to the reference table');
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
        rhk: rhkVal, liew: liewVal,
        liew_err: liewErr, rhk_err: rhkErr,
        feh: fehRow ? fehRow.value : undefined
      });

      // Resample PDFs onto common JS ageGrid
      const liPdf = result.li_pdf ? interpolatePDF(result.age_grid, result.li_pdf, ageGrid) : null;
      const actPdf = interpolatePDF(result.age_grid, result.ca_pdf, ageGrid);
      const combinedPdf = result.combined_pdf
        ? interpolatePDF(result.age_grid, result.combined_pdf, ageGrid)
        : actPdf;

      // BAFFLES stats: [p2, p16, p50, p84, p97]
      const liStats = result.li_stats ? {
        median: result.li_stats[2], lower: result.li_stats[1], upper: result.li_stats[3]
      } : null;
      const actStats = {
        median: result.ca_stats[2], lower: result.ca_stats[1], upper: result.ca_stats[3]
      };
      const combStats = result.combined_stats ? {
        median: result.combined_stats[2], lower: result.combined_stats[1], upper: result.combined_stats[3]
      } : actStats;

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
      state.results.baffles.ci = `[${formatSignificant(combStats.lower, 8)}, ${formatSignificant(combStats.upper, 8)}]`;
      state.results.baffles.status = 'Calculated';
      invalidateJointResult();

      // Update Summary Cards
      document.getElementById('baffles-res-li-age').innerText = liStats ? `${formatSignificant(liStats.median, 8)} Myr` : 'N/A';
      document.getElementById('baffles-res-act-age').innerText = `${formatSignificant(actStats.median, 8)} Myr`;
      document.getElementById('baffles-res-comb-age').innerText = `${formatSignificant(combStats.median, 8)} Myr`;
      const bvEl = document.getElementById('baffles-res-bv');
      if (bvEl && result.bv_lookup) {
        const bv = result.bv_lookup;
        bvEl.innerText = bv.bv_err != null
          ? `${formatSignificant(bv.bv, 8)} ± ${formatSignificant(bv.bv_err, 6)} (from Teff, Pecaut & Mamajek table)`
          : `${formatSignificant(bv.bv, 8)} (from Teff, Pecaut & Mamajek table)`;
        if (bv.caveat) showToast(bv.caveat);
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
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    legend: { x: 0, y: 1, font: { color: '#4a5568' }, bgcolor: 'rgba(255,255,255,0.8)' }
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
  if (!state.clusterResult) {
    document.getElementById('cluster-results-panel').style.display = 'none';
    document.getElementById('cluster-loader').style.display = 'none';
    document.getElementById('cluster-summary-card').style.display = 'none';
  }

  if (!target) {
    hint.innerText = 'Load a target on the Target Hub first.';
  } else if (!ready) {
    hint.innerText = 'Target needs a Gaia ID, or RA/Dec/parallax for a crossmatch.';
  } else if (!Number.isFinite(Number(target?.pmra)) || !Number.isFinite(Number(target?.pmdec))) {
    hint.innerText = 'No proper motion on this target — membership will be scored on parallax only.';
  } else {
    hint.innerText = 'Ready. Gaia DR3 membership is checked first, then a kinematic crossmatch.';
  }
}

function setupClusterPageControls() {
  const runBtn = document.getElementById('btn-run-cluster');
  runBtn.addEventListener('click', runCluster);
}

async function runCluster() {
  const target = state.activeTarget;
  if (!target) {
    showToast('Load a target on the Target Hub first');
    return;
  }
  const gaiaid = String(target.gaiaid || '').trim();
  const hasCoords = [target.ra, target.dec, target.parallax].every(v => Number.isFinite(Number(v)));
  if (!gaiaid && !hasCoords) {
    showToast('Target needs a Gaia ID or RA/Dec/parallax');
    return;
  }

  const catalog = document.getElementById('cluster-catalog-select').value;
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

  const runBtn = document.getElementById('btn-run-cluster');
  document.getElementById('cluster-results-panel').style.display = 'none';
  document.getElementById('cluster-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/cluster', payload);
    state.clusterResult = result;
    renderClusterResult(result);
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

// ================= 8. Summary Page Logic =================

function updateSummaryPage() {
  const isGyroRun = state.results.gyro.status === 'Calculated';
  const isBafflesRun = state.results.baffles.status === 'Calculated';

  const gyroPdf = state.results.gyro.agePDF;
  const bafflesPdf = state.results.baffles.agePDF;

  document.getElementById('td-gyro-status').innerHTML = isGyroRun
    ? '<span class="status-indicator"><div class="dot dot-active"></div>Active</span>'
    : '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Not yet run</span>';
  document.getElementById('td-baffles-status').innerHTML = isBafflesRun
    ? '<span class="status-indicator"><div class="dot dot-active"></div>Active</span>'
    : '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Not yet run</span>';

  if (isGyroRun) {
    const gyroStats = getStatsFromPDF(gyroPdf);
    document.getElementById('td-gyro-age').innerText = `${formatSignificant(gyroStats.median, 8)} Myr`;
    document.getElementById('td-gyro-ci').innerText = `${formatSignificant(gyroStats.lower, 8)}-${formatSignificant(gyroStats.upper, 8)} Myr`;
  } else {
    document.getElementById('td-gyro-age').innerText = '-';
    document.getElementById('td-gyro-ci').innerText = '-';
  }

  if (isBafflesRun) {
    const bafflesStats = getStatsFromPDF(bafflesPdf);
    document.getElementById('td-baffles-age').innerText = `${formatSignificant(bafflesStats.median, 8)} Myr`;
    document.getElementById('td-baffles-ci').innerText = `${formatSignificant(bafflesStats.lower, 8)}-${formatSignificant(bafflesStats.upper, 8)} Myr`;
  } else {
    document.getElementById('td-baffles-age').innerText = '-';
    document.getElementById('td-baffles-ci').innerText = '-';
  }

  updatePropertiesSummary();

  if (!isGyroRun || !isBafflesRun) {
    // Joint posterior requires both real posteriors; do not synthesize one
    // from a partial or estimated result.
    state.results.joint.agePDF = null;
    state.results.joint.median = null;
    state.results.joint.ci = null;
    state.results.joint.status = 'Idle';

    document.getElementById('summary-joint-age').innerText = '-';
    document.getElementById('summary-joint-ci').innerText = 'Run gyro-interp and BAFFLES to compute a joint age';
    document.getElementById('td-joint-age').innerText = '-';
    document.getElementById('td-joint-ci').innerText = '-';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Awaiting both methods</span>';

    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    return;
  }

  // Gyrochronology validity guard: Bouma+2023 rotation-sequence grids are
  // only calibrated down to ~80 Myr (youngest reference cluster). If a
  // significant fraction of the gyro posterior mass falls below this floor,
  // the gyro age is an extrapolation and the joint result is unreliable.
  const GYRO_CALIBRATION_FLOOR_MYR = 80;
  const gyroMassBelowFloor = WakaiPosterior.probabilityBelow(ageGrid, gyroPdf, GYRO_CALIBRATION_FLOOR_MYR);
  const caveatCard = document.getElementById('gyro-caveat-card');
  if (gyroMassBelowFloor > 0.5) {
    caveatCard.style.display = 'block';
  } else {
    caveatCard.style.display = 'none';
  }

  // Multiply the normalized method PDFs on their shared grid. Scaling inside
  if (joint.status !== 'ok') {
    state.results.joint = { ...defaultResults().joint, status: 'No overlap' };
    document.getElementById('summary-joint-age').innerText = '—';
    document.getElementById('summary-joint-ci').innerText = 'The method posteriors do not overlap';
    document.getElementById('td-joint-age').innerText = '—';
    document.getElementById('td-joint-ci').innerText = '—';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator status-error"><div class="dot dot-danger"></div>Methods disagree</span>';
    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    saveTargetState({ silent: true });
    showToast('No joint age: gyro-interp and BAFFLES posteriors do not overlap');
    return;
  }
  const jointPdf = joint.pdf;

  const gyroStats = getStatsFromPDF(gyroPdf);
  const bafflesStats = getStatsFromPDF(bafflesPdf);
  const jointStats = joint.stats;

  // Update State Results
  state.results.joint.agePDF = jointPdf;
  state.results.joint.median = jointStats.median;
  state.results.joint.lower = jointStats.lower;
  state.results.joint.upper = jointStats.upper;
  state.results.joint.ci = `[${formatSignificant(jointStats.lower, 8)}, ${formatSignificant(jointStats.upper, 8)}]`;
  state.results.joint.status = 'Calculated';
  updateWorkflowStrip();

  // Update Page Labels
  document.getElementById('summary-joint-age').innerText = formatSignificant(jointStats.median, 8);
  document.getElementById('summary-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)} - ${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-age').innerText = `${formatSignificant(jointStats.median, 8)} Myr`;
  document.getElementById('td-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)}-${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-status').innerHTML = '<span class="status-indicator"><div class="dot dot-active"></div>Synthesized</span>';

  // Update Evolutionary Stage Timeline
  updateEvolutionTimeline(jointStats.median);

  // Draw Comparison Plot
  const data = [
    {
      x: ageGrid,
      y: gyroPdf,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      name: 'gyro-interp PDF',
      line: { color: 'rgba(124, 58, 237, 0.7)', width: 2 },
      fillcolor: 'rgba(124, 58, 237, 0.08)'
    },
    {
      x: ageGrid,
      y: bafflesPdf,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      name: 'BAFFLES Combined PDF',
      line: { color: 'rgba(234, 88, 12, 0.7)', width: 2 },
      fillcolor: 'rgba(234, 88, 12, 0.08)'
    },
    {
      x: ageGrid,
      y: jointPdf,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      name: 'Joint Posterior PDF',
      line: { color: 'rgba(5, 150, 105, 1)', width: 3 },
      fillcolor: 'rgba(5, 150, 105, 0.18)'
    }
  ];

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

function setupSummaryActions() {
  document.getElementById('btn-export-report').addEventListener('click', () => {
    const reportData = {
      target: state.activeTarget ? state.activeTarget.name : "Custom",
      gaiaid: state.activeTarget ? state.activeTarget.gaiaid : null,
      refTable: state.refTable,
      runs: {
        gyro: { age: state.results.gyro.median, ci: state.results.gyro.ci, status: state.results.gyro.status },
        baffles: { age: state.results.baffles.median, ci: state.results.baffles.ci, status: state.results.baffles.status },
        joint: {
          age: state.results.joint.median, ci: state.results.joint.ci,
          status: (state.results.gyro.status === 'Calculated' && state.results.baffles.status === 'Calculated')
            ? 'Calculated' : 'Idle'
        }
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
