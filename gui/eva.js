/**
 * EVA (Excess Variability-based Age, Barber & Mann 2023) page. Unlike every
 * other method page (gyro/BAFFLES/EAGLES/isochrones/MADYS age), EVA is not a
 * single-star fit off the active target's reference table -- it needs a
 * *group* of comoving stars (Gaia DR3 ids), typically the target's matched
 * cluster members from the Cluster page, or a manually pasted comoving-group
 * list. It never feeds the joint posterior or workflow strip, same as the
 * Cluster page's own membership check -- see run_eva.py and state.js's
 * defaultResults() for why.
 */
import { state, defaultResults } from './state.js';
import { escapeHtml, formatSignificant, showToast, showErrorLog, hideErrorLog } from './dom-utils.js';
import { apiCall } from './api.js';
import { saveTargetState } from './target.js';

const EVA_MIN_MEMBERS = 5;

// Accepts one Gaia DR3 id per line, comma-separated, or the "Gaia DR3
// <id>" format eva2's own gaiaQuery()/testFile.csv use -- whatever the user
// pastes in, most likely straight out of a FriendFinder/cluster-member CSV.
function parseGaiaIdList(text) {
  const ids = [];
  const seen = new Set();
  (text || '').split(/[\n,]/).forEach(token => {
    const match = token.match(/(\d{5,})/);
    if (!match) return;
    if (!seen.has(match[1])) {
      seen.add(match[1]);
      ids.push(match[1]);
    }
  });
  return ids;
}

export function syncEvaInputs() {
  // The textarea's value is restored once from pageSettings.eva by
  // applyPageSettings() (target.js), same as cluster-name-input -- this just
  // refreshes the derived hint/button state, which can go stale if the
  // Cluster page's match changed since this page was last shown.
  updateEvaGroupHint();

  const useClusterBtn = document.getElementById('btn-eva-use-cluster');
  if (useClusterBtn) {
    const { disabled, title } = evaluateClusterMembersAvailability(state.results.cluster);
    useClusterBtn.disabled = disabled;
    useClusterBtn.title = title;
  }

  if (state.results.eva?.status === 'Calculated' && state.results.eva.result) {
    renderEvaResult(state.results.eva.result);
  }
}

function updateEvaGroupHint() {
  const textarea = document.getElementById('eva-group-ids');
  const hint = document.getElementById('eva-group-hint');
  if (!textarea || !hint) return;
  const n = parseGaiaIdList(textarea.value).length;
  hint.innerText = n
    ? `${n} Gaia DR3 id${n === 1 ? '' : 's'} parsed${n < EVA_MIN_MEMBERS ? ` -- need at least ${EVA_MIN_MEMBERS}` : ''}.`
    : `Paste Gaia DR3 ids (one per line, comma-separated, or "Gaia DR3 <id>" rows) -- at least ${EVA_MIN_MEMBERS}.`;
}

// Distinguishes "never checked" from "checked, but the saved result predates
// member_gaia_ids being added to run_cluster.py's output" -- a cluster
// result saved by an older backend has `matched: true` and a real
// cluster_name but no member list at all, which otherwise looks identical
// (silently disabled button) to never having run the check, with no way for
// the user to tell why. Re-running "Check cluster membership" always fixes
// it, since that always calls the current run_cluster.py.
function evaluateClusterMembersAvailability(cluster) {
  if (!cluster?.matched) {
    return { disabled: true, title: 'Run "Check cluster membership" on the Cluster page first' };
  }
  const ids = cluster.member_gaia_ids;
  if (!Array.isArray(ids) || ids.length < EVA_MIN_MEMBERS) {
    return {
      disabled: true,
      title: Array.isArray(ids)
        ? `Only ${ids.length} member(s) available for ${cluster.cluster_name} -- too few for EVA`
        : `This cluster result predates member-list support -- re-run "Check cluster membership" on the Cluster page for ${cluster.cluster_name}`,
    };
  }
  return { disabled: false, title: `Fill in ${ids.length} member(s) of ${cluster.cluster_name}` };
}

function useMatchedClusterMembers() {
  const cluster = state.results.cluster;
  const { disabled, title } = evaluateClusterMembersAvailability(cluster);
  if (disabled) {
    showToast(title);
    return;
  }
  const ids = cluster.member_gaia_ids;
  const textarea = document.getElementById('eva-group-ids');
  textarea.value = ids.join('\n');
  updateEvaGroupHint();
  saveTargetState({ silent: true });
  showToast(`Filled in ${ids.length} member(s) of ${cluster.cluster_name}`);
}

export async function runEva() {
  const textarea = document.getElementById('eva-group-ids');
  const gaiaIds = parseGaiaIdList(textarea?.value);
  if (gaiaIds.length < EVA_MIN_MEMBERS) {
    showToast(`Provide at least ${EVA_MIN_MEMBERS} Gaia DR3 ids for the comoving group`);
    return;
  }

  const useDistance = document.getElementById('chk-eva-distance')?.checked ?? true;
  const plotBand = document.getElementById('eva-plot-band-select')?.value || 'G';

  const runBtn = document.getElementById('btn-run-eva');
  document.getElementById('eva-results-panel').style.display = 'none';
  document.getElementById('eva-loader').style.display = 'flex';
  hideErrorLog('eva-error-log');
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/eva', {
      gaia_ids: gaiaIds,
      distance: useDistance,
      plot_band: plotBand,
    });

    state.results.eva = { status: 'Calculated', result };
    renderEvaResult(result);

    document.getElementById('eva-loader').style.display = 'none';
    document.getElementById('eva-results-panel').style.display = 'block';

    await saveTargetState({ silent: true });
    const truncatedNote = result.n_truncated ? ' (group truncated to the first 300 ids)' : '';
    showToast(`EVA run completed successfully${truncatedNote}`);
  } catch (err) {
    state.results.eva = { status: 'Error', result: null };
    document.getElementById('eva-loader').style.display = 'none';
    document.getElementById('eva-results-panel').style.display = 'none';
    showErrorLog('eva-error-log', err.message);
  } finally {
    runBtn.disabled = false;
  }
}

function formatBandAge(band) {
  if (!band) return '-';
  return `${formatSignificant(band.age_myr, 8)} +${formatSignificant(band.err_hi_myr, 6)}/-${formatSignificant(band.err_lo_myr, 6)} Myr`;
}

export function renderEvaResult(result) {
  document.getElementById('eva-res-overall').innerText = formatBandAge(result.overall);
  document.getElementById('eva-res-distance-note').innerText = result.distance_calibration
    ? `Distance-calibrated (median group distance ${formatSignificant(result.median_distance_pc, 6)} pc)`
    : 'No distance calibration';
  document.getElementById('eva-res-counts').innerText =
    `${result.n_pass_filters} of ${result.n_usable} queried stars passed EVA's quality cuts `
    + `(${result.n_input} ids requested${result.n_truncated ? ', truncated to 300' : ''}, ${result.n_queried} matched in Gaia DR3).`;
  document.getElementById('eva-summary-card').style.display = 'block';

  const bandBody = document.getElementById('eva-band-body');
  bandBody.innerHTML = '';
  const var90 = result.var90 || {};
  ['G', 'BP', 'RP'].forEach(band => {
    const row = result.bands?.[band];
    const var90Value = var90[band];
    const var90Err = var90[`${band}_err`];
    const var90Text = var90Value != null
      ? `${formatSignificant(var90Value, 6)} ± ${formatSignificant(var90Err, 4)}`
      : '-';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${band}</td><td>${formatBandAge(row)}</td><td>${var90Text}</td>`;
    bandBody.appendChild(tr);
  });

  const plots = result.plots || {};
  const plotContainer = document.getElementById('plot-eva-hist');
  if (plots.hist) {
    const img = document.createElement('img');
    img.src = plots.hist;
    img.alt = `EVA var${result.plot_band} histogram`;
    plotContainer.replaceChildren(img);
  } else {
    plotContainer.innerHTML = `<p style="padding: 1rem; font-size: 0.85rem; color: var(--ink-faint);">Plot unavailable${
      plots.hist_error ? `: ${escapeHtml(plots.hist_error)}` : ''
    }</p>`;
  }
}

export function deleteEvaResults() {
  state.results.eva = defaultResults().eva;
  document.getElementById('eva-summary-card').style.display = 'none';
  document.getElementById('eva-results-panel').style.display = 'none';
  hideErrorLog('eva-error-log');
  saveTargetState({ silent: true });
  showToast('EVA results deleted');
}

export function setupEvaPageControls() {
  document.getElementById('btn-run-eva')?.addEventListener('click', runEva);
  document.getElementById('btn-eva-use-cluster')?.addEventListener('click', useMatchedClusterMembers);
  document.getElementById('btn-delete-eva')?.addEventListener('click', deleteEvaResults);

  const textarea = document.getElementById('eva-group-ids');
  textarea?.addEventListener('input', updateEvaGroupHint);
  textarea?.addEventListener('change', () => saveTargetState({ silent: true }));

  document.getElementById('chk-eva-distance')?.addEventListener('change', () => {
    saveTargetState({ silent: true });
  });
  document.getElementById('eva-plot-band-select')?.addEventListener('change', () => {
    saveTargetState({ silent: true });
  });
}
