/**
 * SPA routing (History API, path shape /<page-slug>[/target=<name>]) and the
 * per-page enter hooks that keep each page's inputs in sync with state.
 */
import { state } from './state.js';
import { formatSignificant, showToast, hideToast } from './dom-utils.js';
import { loadTarget, triggerMockSearch } from './target.js';
import { drawGyroPlots } from './gyro.js';
import { drawBafflesPlots } from './baffles.js';
import { syncEaglesInputParams, drawEaglesPlots } from './eagles.js';
import { syncClusterInputs, renderClusterResult, renderMadysCmdResult } from './cluster.js';
import { syncIsochroneInputParams, drawIsochronePlots } from './isochrones.js';
import { renderMadysAgeResult, drawMadysAgePlot } from './madys-age.js';
import { updateSummaryPage } from './summary.js';

// Path shape: /<page-slug>[/target=<name>], e.g. /gyro-interp/target=TOI-5973.
// The target= segment can also appear alone (/target=TOI-5973 -> Target page
// with that target loaded) since it's matched independently of page slug.
export const PAGE_ROUTES = { target: 'target', gyro: 'gyro-interp', baffles: 'baffles', eagles: 'eagles', cluster: 'cluster', isochrones: 'isochrones', summary: 'summary' };
export const ROUTE_PAGES = Object.fromEntries(Object.entries(PAGE_ROUTES).map(([pageId, slug]) => [slug, pageId]));

export function buildPath(pageId, targetName) {
  const parts = [];
  // On the Target page itself, a target= segment already implies the page
  // (parsePath defaults to it), so skip the redundant /target/target=... .
  if (pageId !== 'target' || !targetName) {
    parts.push(PAGE_ROUTES[pageId] || PAGE_ROUTES.target);
  }
  if (targetName) parts.push(`target=${encodeURIComponent(targetName)}`);
  return `/${parts.join('/')}`;
}

export function parsePath(pathname) {
  let pageId = 'target';
  let targetSlug = null;
  for (const segment of pathname.split('/').filter(Boolean)) {
    if (segment.startsWith('target=')) {
      // A malformed percent-escape (truncated/garbled share link, typed by
      // hand) throws URIError; degrade to "no target in the URL" rather
      // than letting that propagate out of applyRoute()/initApp() and
      // leave the SPA stuck on a blank page.
      try {
        targetSlug = decodeURIComponent(segment.slice('target='.length));
      } catch (error) {
        targetSlug = null;
      }
    } else if (ROUTE_PAGES[segment]) {
      pageId = ROUTE_PAGES[segment];
    }
  }
  return { pageId, targetSlug };
}

export function updateUrl(pageId, targetName, { replace = false } = {}) {
  const path = buildPath(pageId, targetName);
  if (path === window.location.pathname) return;
  window.history[replace ? 'replaceState' : 'pushState']({ pageId, targetName }, '', path);
}

export function showPage(pageId) {
  document.querySelectorAll('.nav-item').forEach(l => l.classList.toggle('active', l.getAttribute('data-page') === pageId));
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${pageId}`));
  state.activePage = pageId;
  onPageEnter(pageId);
}

export function targetSlugMatchesActive(targetSlug) {
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
export function findSavedTargetByGaiaId(gaiaId) {
  return state.savedTargets.find(t => String(t.gaiaid || '') === gaiaId);
}

// Shared by initial page load and popstate (back/forward): resolves the page
// and, if the URL names a target that isn't already loaded, fetches it. A
// numeric Gaia-ID slug can use the local cache directly (findSavedTargetByGaiaId);
// a name slug always goes through the full SIMBAD/Gaia resolution
// (triggerMockSearch) so that name aliases resolve to the same canonical
// Gaia ID -- and therefore the same saved record -- no matter which alias
// was typed, instead of silently diverging into separate saved targets.
export async function applyRoute(pathname) {
  const { pageId, targetSlug } = parsePath(pathname);
  showPage(pageId);
  if (targetSlug && !targetSlugMatchesActive(targetSlug)) {
    // triggerMockSearch's own "Resolving..." state lives on the Target
    // page's search button, which is invisible if the URL landed on a
    // different page (e.g. /gyro-interp/target=...) -- without this, the
    // round trip (real network calls, can take a couple seconds) looks
    // like a blank, unresponsive page until it resolves. Persist this
    // message for the whole resolution instead of letting the toast's own
    // 2.5s timer hide it mid-load.
    showToast(`Loading ${targetSlug}...`, { persist: true });
    const known = /^\d+$/.test(targetSlug) ? findSavedTargetByGaiaId(targetSlug) : null;
    try {
      if (known) {
        await loadTarget(known);
      } else {
        await triggerMockSearch(targetSlug);
      }
    } finally {
      // The successful paths above replace the loading message with their
      // own toast (e.g. "Restored saved results...", "Loaded target ...
      // from Gaia DR3"); only hide it ourselves if it is still the stale
      // loading message (the silent 404 restore path replaces it with
      // nothing), so we never yank a fresh success toast away.
      if (document.getElementById('toast')?.innerText === `Loading ${targetSlug}...`) {
        hideToast();
      }
    }
    onPageEnter(pageId);
  }
}

export function setupNavigation() {
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

export function syncGyroInputParams() {
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const protRow = state.refTable.find(r => r.parameter === 'Prot');
  document.getElementById('gyro-in-teff').innerText = teffRow && teffRow.value != null
    ? `${formatSignificant(teffRow.value, 8)} ± ${formatSignificant(teffRow.uncertainty, 6)} K`
    : '—';
  document.getElementById('gyro-in-prot').innerText = protRow && protRow.value != null
    ? `${formatSignificant(protRow.value, 8)} ± ${formatSignificant(protRow.uncertainty, 6)} d`
    : '—';
}

export function syncBafflesInputParams() {
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const rhkRow = state.refTable.find(r => r.parameter === "log R'HK");
  const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
  document.getElementById('baffles-in-teff').innerText = teffRow && teffRow.value != null
    ? `${formatSignificant(teffRow.value, 8)} ± ${formatSignificant(teffRow.uncertainty, 6)} K`
    : '—';
  const bv = state.results.baffles.bvLookup;
  document.getElementById('baffles-in-bv').innerText = bv && bv.bv != null
    ? (bv.bv_err != null
      ? `${formatSignificant(bv.bv, 8)} ± ${formatSignificant(bv.bv_err, 6)}`
      : `${formatSignificant(bv.bv, 8)}`)
    : '—';
  document.getElementById('baffles-in-rhk').innerText = rhkRow && rhkRow.value != null
    ? `${formatSignificant(rhkRow.value, 8)} ± ${formatSignificant(rhkRow.uncertainty, 6)}`
    : '—';
  document.getElementById('baffles-in-liew').innerText = liewRow && liewRow.value != null
    ? `${formatSignificant(liewRow.value, 8)} ± ${formatSignificant(liewRow.uncertainty, 6)} mÅ`
    : '—';
}

export function onPageEnter(pageId) {
  if (pageId === 'gyro') {
    syncGyroInputParams();
    drawGyroPlots();
  } else if (pageId === 'baffles') {
    syncBafflesInputParams();
    drawBafflesPlots();
  } else if (pageId === 'eagles') {
    syncEaglesInputParams();
    drawEaglesPlots();
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
    if (state.results.madysAge.status === 'Calculated') {
      renderMadysAgeResult(state.results.madysAge);
      document.getElementById('madys-age-results-panel').style.display = 'block';
    }
    drawMadysAgePlot();
  } else if (pageId === 'summary') {
    updateSummaryPage();
  }
}
