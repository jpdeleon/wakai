/**
 * Joint-posterior computation and the workflow strip / per-method status
 * that depends on it. Kept separate from state.js so page modules that only
 * need e.g. getStatsFromPDF don't have to pull in the whole state shape
 * story, and separate from summary.js since updateWorkflowStrip() (and
 * therefore recomputeJointStatus()) needs to run from every page, not just
 * the Summary page.
 */
import { state, ageGrid, defaultResults } from './state.js';
import { formatSignificant } from './dom-utils.js';

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
export function getStatsFromPDF(pdf) {
  return WakaiPosterior.stats(ageGrid, pdf) || { median: 0, lower: 0, upper: 0 };
}

// Reduces 2 or 3 independent PDFs (already on the shared `grid`) into one
// product posterior by chaining WakaiPosterior.combineIndependent pairwise
// -- its own scaling-before-integrating (to avoid underflow) still applies
// at each step. Short-circuits on the first non-overlapping pair, since a
// null intermediate .pdf can't feed the next combine.
export function combineAllIndependent(grid, pdfs) {
  let acc = { status: 'ok', pdf: pdfs[0], stats: null };
  for (let i = 1; i < pdfs.length; i += 1) {
    acc = WakaiPosterior.combineIndependent(grid, acc.pdf, pdfs[i]);
    if (acc.status !== 'ok') return acc;
  }
  return acc;
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
export function recomputeJointStatus() {
  const isGyroRun = state.results.gyro.status === 'Calculated';
  const isBafflesRun = state.results.baffles.status === 'Calculated';
  const isEaglesRun = state.results.eagles.status === 'Calculated';
  const isIsochroneRun = state.results.isochrone.status === 'Calculated';
  // Excludes a weakly-constrained (broad) isochrone posterior, same as
  // updateSummaryPage's joint product -- see ISOCHRONE_BROAD_DEX.
  const isIsochroneUsable = isIsochroneRun && !state.results.isochrone.broad;

  // BAFFLES and EAGLES are BOTH lithium-EW age estimates, so multiplying
  // both posteriors would double-count the lithium constraint (same
  // observable, same open-cluster calibration anchors). The joint therefore
  // takes at most one lithium-based posterior: EAGLES's, or -- when BAFFLES
  // ran with a chromospheric-activity (Ca II R'HK) posterior available --
  // BAFFLES contributes only its activity component, which is genuinely
  // independent of lithium. When BAFFLES ran without a usable activity
  // posterior (actPDF null), it contributes nothing here rather than falling
  // back to its combined/lithium PDF and double-counting with EAGLES.
  const pdfs = [];
  if (isGyroRun) pdfs.push(state.results.gyro.agePDF);
  if (isBafflesRun && state.results.baffles.actPDF) pdfs.push(state.results.baffles.actPDF);
  if (isEaglesRun) pdfs.push(state.results.eagles.agePDF);
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

export function updateWorkflowStrip() {
  recomputeJointStatus();
  const targetName = state.activeTarget?.name || 'No target selected';
  document.getElementById('workspace-target-name').innerText = targetName;
  const mappings = [
    ['workspace-gyro-stage', state.results.gyro?.status],
    ['workspace-baffles-stage', state.results.baffles?.status],
    ['workspace-eagles-stage', state.results.eagles?.status],
    ['workspace-isochrones-stage', state.results.isochrone?.status],
    ['workspace-joint-stage', state.results.joint?.status],
  ];
  mappings.forEach(([id, status]) => {
    const element = document.getElementById(id);
    element.classList.toggle('complete', status === 'Calculated');
    element.classList.toggle('failed', status === 'Error' || status === 'No overlap');
  });
}

export function invalidateJointResult() {
  // updateWorkflowStrip() recomputes state.results.joint itself (via
  // recomputeJointStatus()) from whatever methods are still valid -- e.g.
  // invalidating isochrone shouldn't blank the joint if gyro+BAFFLES alone
  // can still produce one, so this doesn't force-reset it first.
  updateWorkflowStrip();
}

export function invalidateMethod(method) {
  const hadResult = ['Calculated', 'Error'].includes(state.results[method]?.status);
  state.results[method] = defaultResults()[method];
  invalidateJointResult();
  const prefix = method === 'gyro' ? 'gyro' : method === 'eagles' ? 'eagles' : 'baffles';
  document.getElementById(`${prefix}-summary-card`).style.display = 'none';
  document.getElementById(`${prefix}-results-panel`).style.display = 'none';
  document.getElementById(`status-${prefix}-dot`).className = 'dot dot-inactive';
  document.getElementById(`status-${prefix}-text`).innerText = hadResult ? 'Inputs changed' : 'Idle';
}
