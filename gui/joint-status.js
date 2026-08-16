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

// The five PDFs that can feed the joint product, each user-toggleable via a
// checkbox on the Summary page's Comparison table (#chk-joint-<key>).
// `pdf(state)` is the candidate PDF; `eligible(state)` is whether it exists
// at all (a checkbox can only be checked when eligible); `defaultChecked`
// is what the checkbox is set to the first time it becomes eligible (before
// the user has touched it), preserving this app's previous hardcoded
// behavior as the out-of-the-box default:
//  - BAFFLES contributes only its chromospheric-activity (Ca II R'HK)
//    component by default, not its combined (lithium x activity) PDF --
//    BAFFLES and EAGLES are both lithium-EW age estimates, so including
//    both by default would double-count that one observable. The user can
//    still check both deliberately; this only governs the untouched default.
//  - isochrone (MIST/isochrones2) and madysAge (PARSEC2/MIST via MADYS) fit
//    the same photometry+parallax observable through a different
//    stellar-model grid, so only one is checked by default (isochrone,
//    when it's not too weakly constrained -- ISOCHRONE_BROAD_DEX).
const JOINT_METHODS = {
  gyro: {
    checkboxId: 'chk-joint-gyro',
    pdf: s => s.results.gyro.agePDF,
    eligible: s => s.results.gyro.status === 'Calculated',
    defaultChecked: () => true,
  },
  bafflesActivity: {
    checkboxId: 'chk-joint-baffles',
    pdf: s => s.results.baffles.actPDF,
    eligible: s => s.results.baffles.status === 'Calculated' && Boolean(s.results.baffles.actPDF),
    defaultChecked: () => true,
  },
  eagles: {
    checkboxId: 'chk-joint-eagles',
    pdf: s => s.results.eagles.agePDF,
    eligible: s => s.results.eagles.status === 'Calculated',
    defaultChecked: () => true,
  },
  isochrone: {
    checkboxId: 'chk-joint-isochrone',
    pdf: s => s.results.isochrone.agePDF,
    eligible: s => s.results.isochrone.status === 'Calculated',
    defaultChecked: s => !s.results.isochrone.broad,
  },
  madysAge: {
    checkboxId: 'chk-joint-madys-age',
    pdf: s => s.results.madysAge.agePDF,
    eligible: s => s.results.madysAge.status === 'Calculated',
    defaultChecked: s => s.results.isochrone.status !== 'Calculated' || s.results.isochrone.broad,
  },
};

// What recomputeJointStatus() last decided for each method -- eligible
// (a real PDF exists) and included (eligible AND its checkbox is checked).
// summary.js reads this (via getJointInclusion()) to build the comparison
// plot/status badges from the same decision recomputeJointStatus() made,
// instead of re-deriving eligibility/inclusion itself and risking drift.
let lastInclusion = Object.fromEntries(Object.keys(JOINT_METHODS).map(key => [key, { eligible: false, included: false }]));

export function getJointInclusion() {
  return lastInclusion;
}

// Syncs each method's checkbox against current eligibility (disables it,
// and visually unchecks it, when the method has no PDF at all -- e.g. never
// run, or just invalidated ahead of a re-run) and applies defaultChecked
// the first time a method becomes eligible -- afterwards the checkbox's
// own .checked is authoritative, whatever the user last set it to
// (dataset.userTouched isn't cleared just because the method is
// momentarily ineligible, so a deliberate exclusion survives a re-run --
// same pattern as isochrones.js's syncIsochroneInputParams()). Checkboxes
// live only in the Summary page's markup, but that markup exists in the
// DOM on every page load (this is a single-document SPA -- see routing.js),
// so this can run unconditionally from recomputeJointStatus(), which
// itself runs on every page via updateWorkflowStrip().
function syncJointCheckboxes() {
  const included = {};
  Object.entries(JOINT_METHODS).forEach(([key, method]) => {
    const isEligible = method.eligible(state);
    const chk = document.getElementById(method.checkboxId);
    if (!chk) {
      included[key] = { eligible: isEligible, included: isEligible && method.defaultChecked(state) };
      return;
    }
    chk.disabled = !isEligible;
    if (!isEligible) {
      chk.checked = false;
    } else if (!chk.dataset.userTouched) {
      chk.checked = method.defaultChecked(state);
    }
    included[key] = { eligible: isEligible, included: isEligible && chk.checked };
  });
  return included;
}

// Recomputes state.results.joint from whichever of the five JOINT_METHODS
// are both eligible and checked -- pure state update (plus the checkbox
// sync above), no Plotly work, so it's cheap enough to call on every
// updateWorkflowStrip() run. This is what keeps the top nav's "Joint age"
// stage live on every page: previously state.results.joint was only
// (re)computed inside updateSummaryPage(), which only runs when the user
// actually visits the Summary page, so the strip could show "Idle" even
// after two+ methods had already succeeded. updateSummaryPage() calls this
// too instead of duplicating the same usable-methods/combineAllIndependent
// logic itself.
export function recomputeJointStatus() {
  lastInclusion = syncJointCheckboxes();

  const pdfs = Object.entries(JOINT_METHODS)
    .filter(([key]) => lastInclusion[key].included)
    .map(([key, method]) => method.pdf(state));

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
    ['workspace-madys-age-stage', state.results.madysAge?.status],
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
