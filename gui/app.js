/**
 * WAKAI DASHBOARD APPLICATION ENTRY POINT
 * Boots the SPA: registers every page's setup() handlers, loads the saved
 * target index, and applies the initial URL route. All application logic
 * lives in the ES modules imported below (see index.html's type="module"
 * script tag); this file only wires them together.
 */
import { state, emptyDefaultParameters } from './state.js';
import { apiGet } from './api.js';
import { showToast } from './dom-utils.js';
import { setupNavigation, applyRoute } from './routing.js';
import { updateWorkflowStrip } from './joint-status.js';
import {
  setupTargetSelector,
  setupTableActions,
  setupCatalogActions,
  setupModelRunControls,
} from './target.js';
import { setupGyroModelOverlay } from './gyro.js';
import { setupEaglesPageControls, setupEaglesModelOverlay } from './eagles.js';
import { setupClusterPageControls } from './cluster.js';
import { setupIsochronesPageControls } from './isochrones.js';
import { setupMadysAgePageControls } from './madys-age.js';
import { setupSummaryActions } from './summary.js';

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

async function initApp() {
  setupNavigation();
  setupTargetSelector();
  setupTableActions();
  setupCatalogActions();
  setupModelRunControls();
  setupGyroModelOverlay();
  setupEaglesPageControls();
  setupEaglesModelOverlay();
  setupClusterPageControls();
  setupIsochronesPageControls();
  setupMadysAgePageControls();
  setupSummaryActions();
  // Awaited (unlike everything else above) because applyRoute needs
  // state.savedTargets populated to try the local-cache path first -- see
  // applyRoute.
  await loadSavedTargetIndex();
  updateWorkflowStrip();
  await applyRoute(window.location.pathname);
}

// Initialize on load
window.addEventListener('DOMContentLoaded', initApp);
