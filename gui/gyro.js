/**
 * Gyrochronology (gyro-interp) page: run button, empirical model overlay,
 * axis-fit controls, and the two Plotly plots.
 */
import {
  state, ageGrid, GYRO_SLOW_SEQUENCE_COLORS, gyroSlowSequenceCache, setGyroSlowSequenceCache,
  interpolatePDF, positiveUncertainty,
} from './state.js';
import { formatSignificant, showToast } from './dom-utils.js';
import { apiCall, apiGet } from './api.js';
import { invalidateJointResult, updateWorkflowStrip } from './joint-status.js';
import { saveTargetState } from './target.js';

export async function runGyro() {
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

export function setupGyroModelOverlay() {
  const checkbox = document.getElementById('chk-gyro-models');
  if (!checkbox) return;
  const ageRow = document.getElementById('gyro-model-ages');

  checkbox.addEventListener('change', async () => {
    if (ageRow) ageRow.hidden = !checkbox.checked;
    if (checkbox.checked && !gyroSlowSequenceCache) {
      checkbox.disabled = true;
      try {
        setGyroSlowSequenceCache(await apiGet('gyro/slow-sequence'));
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
export function fitGyroGridAxisToModels() {
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

export function gyroGridAxisSettings() {
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

export function drawGyroPlots() {
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

  Plotly.react('plot-gyro-grid', dataGrid, layoutGrid, { responsive: true, displayModeBar: false, staticPlot: true });

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

  Plotly.react('plot-gyro-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}
