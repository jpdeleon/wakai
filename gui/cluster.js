/**
 * Cluster membership + MADYS CMD page.
 */
import { state, defaultResults } from './state.js';
import { escapeHtml, formatMetadata, formatSignificant, showToast, showErrorLog, hideErrorLog } from './dom-utils.js';
import { apiCall, apiGet } from './api.js';
import { saveTargetState } from './target.js';

export function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function syncClusterInputs() {
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

  // Clear any stale result (and any stale error log) rendered for a
  // previous target.
  if (!state.results.cluster) {
    document.getElementById('cluster-results-panel').style.display = 'none';
    document.getElementById('cluster-loader').style.display = 'none';
    document.getElementById('cluster-summary-card').style.display = 'none';
    hideErrorLog('cluster-error-log');
  }
  if (!Object.keys(state.results.madysCmd?.plots || {}).length) {
    document.getElementById('madys-cmd-results-panel').style.display = 'none';
    document.getElementById('madys-cmd-loader').style.display = 'none';
    hideErrorLog('madys-cmd-error-log');
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

export function saveClusterPageSettingsToLocalStorage() {
  const settings = {
    catalog: document.getElementById('cluster-catalog-select')?.value || 'Perren2023',
    clusterName: document.getElementById('cluster-name-input')?.value.trim() || '',
    models: Array.from(document.querySelectorAll('.chk-madys-model:checked')).map(el => el.value),
    plotAges: Array.from(document.querySelectorAll('.chk-madys-age:checked')).map(el => Number(el.value)),
    plotMasses: Array.from(document.querySelectorAll('.chk-madys-mass:checked')).map(el => Number(el.value)),
  };
  localStorage.setItem('wakai:cluster:settings', JSON.stringify(settings));
}

export function loadClusterPageSettingsFromLocalStorage() {
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

// state.results.cluster's "no result" sentinel is null (see defaultResults()
// in state.js), not a status-flagged object like the joint-feeding methods
// -- syncClusterInputs() already treats a falsy value as "nothing to show".
export function deleteClusterResults() {
  state.results.cluster = null;
  document.getElementById('cluster-summary-card').style.display = 'none';
  document.getElementById('cluster-results-panel').style.display = 'none';
  hideErrorLog('cluster-error-log');
  saveTargetState({ silent: true });
  showToast('Cluster membership results deleted');
}

export function deleteMadysCmdResults() {
  state.results.madysCmd = defaultResults().madysCmd;
  document.getElementById('madys-cmd-results-panel').style.display = 'none';
  hideErrorLog('madys-cmd-error-log');
  saveTargetState({ silent: true });
  showToast('MADYS CMD plot deleted');
}

export function setupClusterPageControls() {
  const runBtn = document.getElementById('btn-run-cluster');
  runBtn.addEventListener('click', runCluster);
  setupClusterNameSelector();
  document.getElementById('cluster-catalog-select')?.addEventListener('change', () => {
    saveTargetState({ silent: true });
    saveClusterPageSettingsToLocalStorage();
  });
  document.getElementById('btn-delete-cluster')?.addEventListener('click', deleteClusterResults);

  document.getElementById('btn-run-madys-cmd')?.addEventListener('click', runMadysCmd);
  document.getElementById('btn-delete-madys-cmd')?.addEventListener('click', deleteMadysCmdResults);
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

export async function ensureClusterNameCache() {
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
export function updateClusterNameHint() {
  const input = document.getElementById('cluster-name-input');
  const hint = document.getElementById('cluster-name-hint');
  if (input && hint) {
    const value = input.value.trim();
    hint.hidden = !value;
    hint.textContent = value ? `Will plot against "${value}" instead of the auto-matched cluster.` : '';
  }
}

export function setupClusterNameSelector() {
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

export async function runCluster() {
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
  hideErrorLog('cluster-error-log');
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
    showErrorLog('cluster-error-log', err.message);
  } finally {
    runBtn.disabled = false;
  }
}

export function renderClusterPlot(containerId, dataUri, errorMessage) {
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

export function renderClusterResult(result) {
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
export async function runMadysCmd() {
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
  hideErrorLog('madys-cmd-error-log');
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
    showErrorLog('madys-cmd-error-log', err.message);
  } finally {
    runBtn.disabled = false;
  }
}

export function renderMadysCmdResult(result) {
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
