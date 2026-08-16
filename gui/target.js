/**
 * Target page: search/resolve, reference table editing, VizieR/SIMBAD
 * catalog lookups, per-page settings persistence, and target save/load.
 */
import {
  state, emptyDefaultParameters, defaultResults, sanitizeStoredResults,
  gyroSlowSequenceCache, setGyroSlowSequenceCache,
  eaglesModelsCache, setEaglesModelsCache,
} from './state.js';
import { escapeHtml, safeSourceUrl, formatSignificant, showToast } from './dom-utils.js';
import { apiCall, apiGet } from './api.js';
import { updateWorkflowStrip, invalidateMethod } from './joint-status.js';
import { updateUrl } from './routing.js';
import { ISOCHRONE_INPUT_SPECS, invalidateIsochrone, renderIsochroneResult } from './isochrones.js';
import { renderClusterResult, updateClusterNameHint, saveClusterPageSettingsToLocalStorage } from './cluster.js';
import { runGyro } from './gyro.js';
import { runBaffles } from './baffles.js';
import { runEagles, EAGLES_MODEL_LABELS } from './eagles.js';

// JSON of the last payload actually POSTed to /api/targets/save by
// saveTargetState() below -- lets it skip the round trip when called again
// with nothing changed (it's wired to many low-stakes UI toggles, each of
// which would otherwise re-send the full results object, including any
// large base64 plot images already saved).
let _lastSavedPayloadJson = null;

export function setupTargetSelector() {
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

export async function triggerMockSearch(name) {
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
      if (Array.isArray(resolved?.identifiers) && resolved.identifiers.length) {
        state.activeTarget.identifiers = resolved.identifiers;
        renderTargetMetadata();
      }
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
        identifiers: resolved.identifiers,
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

export async function loadTarget(target) {
  state.activeTarget = target;
  updateUrl(state.activePage, target.name, { replace: true });
  state.refTable = JSON.parse(JSON.stringify(target.parameters || emptyDefaultParameters()));
  state.catalogCandidates = [];
  document.getElementById('ref-table-card').hidden = false;

  // Reset Results
  state.results = defaultResults();
  document.getElementById('gyro-summary-card').style.display = 'none';
  document.getElementById('baffles-summary-card').style.display = 'none';
  document.getElementById('eagles-summary-card').style.display = 'none';

  // Update Metadata Card
  renderTargetMetadata();

  // Update Status
  document.getElementById('status-target-name').innerText = target.name;
  document.getElementById('status-gyro-dot').className = "dot dot-inactive";
  document.getElementById('status-gyro-text').innerText = "Idle";
  document.getElementById('status-baffles-dot').className = "dot dot-inactive";
  document.getElementById('status-baffles-text').innerText = "Idle";
  document.getElementById('status-eagles-dot').className = "dot dot-inactive";
  document.getElementById('status-eagles-text').innerText = "Idle";
  updateWorkflowStrip();

  renderRefTable();
  return restoreSavedTarget(target);
}

export function targetIdentity(target) {
  const gaiaId = String(target?.gaiaid || '').trim();
  return gaiaId ? `gaia:${gaiaId}` : `name:${String(target?.name || '').trim().toLowerCase()}`;
}

// Reads the current DOM state of every persisted-but-not-a-result control
// (gyro axis/model-overlay settings, Cluster catalog/name override,
// Isochrones checkboxes + logg + prior bounds) into a plain object matching
// defaultResults().pageSettings's shape. Called at save time rather than
// kept continuously in sync via change listeners -- the DOM is the single
// source of truth, this just snapshots it.
export function collectPageSettings() {
  const num = (id, fallback) => {
    const value = Number(document.getElementById(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  };

  return {
    gyro: {
      showModels: document.getElementById('chk-gyro-models')?.checked ?? false,
      modelAges: Array.from(document.querySelectorAll('.chk-gyro-age:checked')).map(el => Number(el.value)),
      cmap: document.getElementById('gyro-cmap-select')?.value || 'viridis',
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
    eagles: {
      model: document.getElementById('eagles-model-select')?.value || 'ml',
      showModels: document.getElementById('chk-eagles-models')?.checked ?? false,
      modelAges: Array.from(document.querySelectorAll('.chk-eagles-age:checked')).map(el => Number(el.value)),
      cmap: document.getElementById('eagles-cmap-select')?.value || 'viridis',
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
export async function applyPageSettings(settings) {
  const s = settings || {};
  const setNum = (id, value) => {
    const el = document.getElementById(id);
    if (el && Number.isFinite(value)) el.value = value;
  };

  const gyro = s.gyro || {};
  const modelsCheckbox = document.getElementById('chk-gyro-models');
  const ageRow = document.getElementById('gyro-model-ages');
  const gyroCmapRow = document.getElementById('gyro-cmap-row');
  if (modelsCheckbox) modelsCheckbox.checked = Boolean(gyro.showModels);
  if (ageRow) ageRow.hidden = !modelsCheckbox?.checked;
  if (gyroCmapRow) gyroCmapRow.hidden = !modelsCheckbox?.checked;
  if (Array.isArray(gyro.modelAges)) {
    document.querySelectorAll('.chk-gyro-age').forEach(el => {
      el.checked = gyro.modelAges.includes(Number(el.value));
    });
  }
  const gyroCmapSelect = document.getElementById('gyro-cmap-select');
  if (gyroCmapSelect && gyro.cmap) gyroCmapSelect.value = gyro.cmap;
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
      setGyroSlowSequenceCache(await apiGet('gyro/slow-sequence'));
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

  const eagles = s.eagles || {};
  const eaglesModelSelect = document.getElementById('eagles-model-select');
  if (eaglesModelSelect && eagles.model) eaglesModelSelect.value = eagles.model;
  // Same lazy-fetch pattern as the gyro model overlay (see the gyro block
  // above): the overlay's tracks are target-independent reference data.
  const eaglesOverlay = document.getElementById('chk-eagles-models');
  const eaglesAgeRow = document.getElementById('eagles-model-ages');
  const eaglesCmapRow = document.getElementById('eagles-cmap-row');
  if (eaglesOverlay) eaglesOverlay.checked = Boolean(eagles.showModels);
  if (eaglesAgeRow) eaglesAgeRow.hidden = !eaglesOverlay?.checked;
  if (eaglesCmapRow) eaglesCmapRow.hidden = !eaglesOverlay?.checked;
  const eaglesCmapSelect = document.getElementById('eagles-cmap-select');
  if (eaglesCmapSelect && eagles.cmap) eaglesCmapSelect.value = eagles.cmap;
  if (Array.isArray(eagles.modelAges)) {
    document.querySelectorAll('.chk-eagles-age').forEach(el => {
      el.checked = eagles.modelAges.includes(Number(el.value));
    });
  }
  if (eaglesOverlay?.checked && !eaglesModelsCache) {
    try {
      setEaglesModelsCache(await apiGet('eagles/models'));
    } catch (error) {
      // Overlay just won't render until the user retoggles the checkbox.
    }
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

export async function restoreSavedTarget(target) {
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

export function renderStoredResultStatus() {
  const gyro = state.results.gyro || {};
  const baffles = state.results.baffles || {};
  const eagles = state.results.eagles || {};
  const gyroCalculated = gyro.status === 'Calculated';
  const bafflesCalculated = baffles.status === 'Calculated';
  const eaglesCalculated = eagles.status === 'Calculated';
  document.getElementById('status-gyro-dot').className = gyroCalculated ? 'dot dot-active' : 'dot dot-inactive';
  document.getElementById('status-gyro-text').innerText = gyroCalculated
    ? `${formatSignificant(gyro.median, 8)} Myr`
    : (gyro.status || 'Idle');
  document.getElementById('status-baffles-dot').className = bafflesCalculated ? 'dot dot-active' : 'dot dot-inactive';
  document.getElementById('status-baffles-text').innerText = bafflesCalculated
    ? `${formatSignificant(baffles.median, 8)} Myr`
    : (baffles.status || 'Idle');
  document.getElementById('status-eagles-dot').className = eaglesCalculated ? 'dot dot-active' : 'dot dot-inactive';
  document.getElementById('status-eagles-text').innerText = eaglesCalculated
    ? `${formatSignificant(eagles.median, 8)} Myr`
    : (eagles.status || 'Idle');
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
  if (eaglesCalculated) {
    document.getElementById('eagles-res-median').innerText = `${formatSignificant(eagles.median, 8)} Myr`;
    const peakEl = document.getElementById('eagles-res-peak');
    if (eagles.peak != null && eagles.peakLo != null && eagles.peakHi != null) {
      peakEl.innerText = `${formatSignificant(eagles.peak, 8)} +${formatSignificant(eagles.peakHi - eagles.peak, 6)}/-${formatSignificant(eagles.peak - eagles.peakLo, 6)} Myr`;
    } else if (eagles.limitUp != null) {
      peakEl.innerText = `< ${formatSignificant(eagles.limitUp, 8)} Myr (95% upper limit)`;
    } else if (eagles.limitLo != null) {
      peakEl.innerText = `> ${formatSignificant(eagles.limitLo, 8)} Myr (95% lower limit)`;
    } else {
      peakEl.innerText = '—';
    }
    const modelEl = document.getElementById('eagles-res-model');
    if (modelEl) modelEl.innerText = EAGLES_MODEL_LABELS[eagles.model] || eagles.model;
    document.getElementById('eagles-summary-card').style.display = 'block';
    document.getElementById('eagles-results-panel').style.display = 'block';
  }
  if (state.results.isochrone?.status === 'Calculated') {
    renderIsochroneResult(state.results.isochrone);
    document.getElementById('isochrone-results-panel').style.display = 'block';
  }
  if (state.results.cluster) renderClusterResult(state.results.cluster);
  updateWorkflowStrip();
}

export function renderTargetMetadata() {
  const target = state.activeTarget || {};
  const aliases = Array.isArray(target.identifiers) && target.identifiers.length
    ? target.identifiers
    : (state.simbadTarget?.identifiers
        && String(state.simbadTarget.gaiaid) === String(target.gaiaid || '')
        && String(state.simbadTarget.gaiaid) !== 'undefined' ? state.simbadTarget.identifiers : []);
  document.getElementById('meta-aliases').innerHTML = aliases.length
    ? aliases.map(escapeHtml).join('<br>')
    : '-';
}

export function upsertReferenceRow(row, replaceMissingOnly = false) {
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
    if (changed && ['Teff', 'Li EW', "log R'HK"].includes(row.parameter)) invalidateMethod('baffles');
    if (changed && ['Teff', 'Li EW'].includes(row.parameter)) invalidateMethod('eagles');
    if (changed && ISOCHRONE_INPUT_SPECS.some(spec => spec.id === row.id)) invalidateIsochrone();
  }
}

export async function queryVizierParameters({ parameter = null, populateDefaults = false } = {}) {
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

export async function querySimbadTarget() {
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
      if (Array.isArray(target.identifiers)) state.activeTarget.identifiers = target.identifiers;
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

export function renderVizierParameterList() {
  const datalist = document.getElementById('vizier-parameter-list');
  datalist.innerHTML = state.availableVizierParameters
    .map(parameter => `<option value="${escapeHtml(parameter)}"></option>`)
    .join('');
}

export function renderCatalogCandidates() {
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

export function renderRefTable() {
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
      if (['Teff', 'Li EW', "log R'HK"].includes(parameter)) invalidateMethod('baffles');
      if (['Teff', 'Li EW'].includes(parameter)) invalidateMethod('eagles');
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
      invalidateMethod('eagles');
      invalidateIsochrone();
      renderRefTable();
      showToast("Row removed from table");
    });
  });
}

export function setupCatalogActions() {
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

export async function saveTargetState({ silent = false } = {}) {
  if (!state.activeTarget) {
    if (!silent) showToast('Select a target before saving');
    return false;
  }
  state.results.pageSettings = collectPageSettings();
  const payload = {
    target: state.activeTarget,
    ref_table: state.refTable,
    results: state.results,
  };
  const payloadJson = JSON.stringify(payload);
  if (payloadJson === _lastSavedPayloadJson) {
    // Nothing changed since the last successful save -- most calls here are
    // silent, from low-stakes UI toggles (axis limits, checkboxes), so this
    // is the common case, not an edge case.
    if (!silent) showToast(`Saved ref_table and results for ${state.activeTarget.name}`);
    return true;
  }
  try {
    await apiCall('targets/save', payload);
    _lastSavedPayloadJson = payloadJson;
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

export function setupTableActions() {
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
    invalidateMethod('eagles');
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

export function setupModelRunControls() {
  document.getElementById('btn-run-gyro').addEventListener('click', runGyro);
  document.getElementById('btn-run-baffles').addEventListener('click', runBaffles);
  document.getElementById('btn-run-eagles').addEventListener('click', runEagles);
}
