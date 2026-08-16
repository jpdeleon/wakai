/**
 * EAGLES (lithium-EW age) page: model selector, run button, and the two
 * Plotly plots (best-fit EW-vs-Teff isochrone with the star, and the age
 * posterior). EAGLES models Li I 6708A EW as a function of Teff and age
 * (Jeffries+2023 / Weaver+2024, see run_eagles.py), so -- unlike BAFFLES --
 * it needs no (B-V)o derivation; inputs are Teff and Li EW straight from the
 * reference table.
 */
import { state, ageGrid, interpolatePDF, positiveUncertainty } from './state.js';
import { formatSignificant, showToast } from './dom-utils.js';
import { apiCall } from './api.js';
import { invalidateJointResult, updateWorkflowStrip } from './joint-status.js';
import { saveTargetState } from './target.js';

export const EAGLES_MODEL_LABELS = {
  ml: 'Machine learning (Weaver+2024)',
  analytic: 'Analytical (Jeffries+2023)',
};

export async function runEagles() {
  const runBtn = document.getElementById('btn-run-eagles');
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
  const teffVal = parseFloat(teffRow?.value);
  const liewVal = liewRow?.value == null || liewRow.value === '' ? null : Number(liewRow.value);
  if (!Number.isFinite(teffVal) || liewVal === null || !Number.isFinite(liewVal)) {
    showToast('Add valid Teff and Li EW measurements to the reference table');
    return;
  }

  const teffErr = positiveUncertainty(teffRow, 100);
  const liewErr = positiveUncertainty(liewRow, 15);
  const model = document.getElementById('eagles-model-select')?.value || 'ml';

  document.getElementById('eagles-results-panel').style.display = 'none';
  document.getElementById('eagles-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/eagles', {
      teff: teffVal, teff_err: teffErr,
      liew: liewVal, liew_err: liewErr,
      model,
    });

    // Resample the EAGLES posterior (native log-age grid, already expressed
    // on the Myr axis by the backend) onto the shared JS ageGrid, same as the
    // other methods.
    const pdf = interpolatePDF(result.age_grid, result.pdf, ageGrid);

    state.results.eagles.agePDF = pdf;
    state.results.eagles.median = result.median_myr;
    state.results.eagles.peak = result.peak_myr;
    state.results.eagles.peakLo = result.peak_lo_myr;
    state.results.eagles.peakHi = result.peak_hi_myr;
    state.results.eagles.limitUp = result.limit_up_myr;
    state.results.eagles.limitLo = result.limit_lo_myr;
    state.results.eagles.model = result.model;
    state.results.eagles.logAgeGrid = result.log_age_grid;
    state.results.eagles.logPdf = result.log_pdf;
    state.results.eagles.iso = result.iso || null;
    state.results.eagles.ci = result.peak_lo_myr != null && result.peak_hi_myr != null
      ? `[${formatSignificant(result.peak_lo_myr, 8)}, ${formatSignificant(result.peak_hi_myr, 8)}]`
      : (result.limit_up_myr != null
        ? `< ${formatSignificant(result.limit_up_myr, 8)}`
        : (result.limit_lo_myr != null
          ? `> ${formatSignificant(result.limit_lo_myr, 8)}`
          : null));
    state.results.eagles.status = 'Calculated';
    invalidateJointResult();

    // Update Summary Card
    document.getElementById('eagles-res-median').innerText = `${formatSignificant(result.median_myr, 8)} Myr`;
    const peakEl = document.getElementById('eagles-res-peak');
    if (result.peak_myr != null && result.peak_lo_myr != null && result.peak_hi_myr != null) {
      const hi = result.peak_hi_myr - result.peak_myr;
      const lo = result.peak_myr - result.peak_lo_myr;
      peakEl.innerText = `${formatSignificant(result.peak_myr, 8)} +${formatSignificant(hi, 6)}/-${formatSignificant(lo, 6)} Myr`;
    } else if (result.limit_up_myr != null) {
      peakEl.innerText = `< ${formatSignificant(result.limit_up_myr, 8)} Myr (95% upper limit)`;
    } else if (result.limit_lo_myr != null) {
      peakEl.innerText = `> ${formatSignificant(result.limit_lo_myr, 8)} Myr (95% lower limit)`;
    } else {
      peakEl.innerText = '—';
    }
    document.getElementById('eagles-res-model').innerText = EAGLES_MODEL_LABELS[result.model] || result.model;
    document.getElementById('eagles-summary-card').style.display = 'block';

    // Update Pipeline Status Dot
    document.getElementById('status-eagles-dot').className = "dot dot-active";
    document.getElementById('status-eagles-text').innerText = `${formatSignificant(result.median_myr, 8)} Myr`;
    updateWorkflowStrip();

    document.getElementById('eagles-loader').style.display = 'none';
    document.getElementById('eagles-results-panel').style.display = 'block';
    drawEaglesPlots();

    await saveTargetState({ silent: true });
    showToast("EAGLES run completed successfully");
  } catch (err) {
    // No simulated fallback: an age here would not have come from the real
    // EAGLES EW-age model, so we show the failure instead of a fabricated
    // number.
    state.results.eagles.agePDF = null;
    state.results.eagles.median = null;
    state.results.eagles.ci = null;
    state.results.eagles.peak = null;
    state.results.eagles.status = 'Error';
    invalidateJointResult();

    document.getElementById('eagles-res-median').innerText = 'Error';
    document.getElementById('eagles-res-peak').innerText = err.message;
    document.getElementById('eagles-summary-card').style.display = 'block';
    document.getElementById('status-eagles-dot').className = "dot dot-danger";
    document.getElementById('status-eagles-text').innerText = "Error";
    updateWorkflowStrip();

    document.getElementById('eagles-loader').style.display = 'none';
    document.getElementById('eagles-results-panel').style.display = 'none';

    showToast(`EAGLES error: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

export function syncEaglesInputParams() {
  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
  document.getElementById('eagles-in-teff').innerText = teffRow && teffRow.value != null
    ? `${formatSignificant(teffRow.value, 8)} ± ${formatSignificant(teffRow.uncertainty, 6)} K`
    : '—';
  document.getElementById('eagles-in-liew').innerText = liewRow && liewRow.value != null
    ? `${formatSignificant(liewRow.value, 8)} ± ${formatSignificant(liewRow.uncertainty, 6)} mÅ`
    : '—';
}

export function drawEaglesPlots() {
  if (!state.results.eagles.agePDF || state.results.eagles.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }

  // PLOT 1: best-fit EW-vs-Teff isochrone at the derived age, with the
  // target's (Teff, Li EW) overlaid -- the same diagnostic the EAGLES
  // package's own *_prob.pdf figures show.
  const dataIso = [];
  const iso = state.results.eagles.iso;
  if (iso) {
    const eew = iso.eew.map((v, i) => iso.ew[i] + v);
    const eem = iso.eew.map((v, i) => iso.ew[i] - v);
    dataIso.push(
      {
        x: iso.teff, y: eem, mode: 'lines', type: 'scatter',
        name: 'model ± dispersion', line: { width: 0 },
        fillcolor: 'rgba(14, 116, 144, 0.15)', fill: 'tonexty',
        hoverinfo: 'skip', showlegend: false,
      },
      {
        x: iso.teff, y: eew, mode: 'lines', type: 'scatter',
        line: { width: 0 }, fill: 'tonexty',
        fillcolor: 'rgba(14, 116, 144, 0.15)',
        hoverinfo: 'skip', showlegend: false,
      },
      {
        x: iso.teff, y: iso.ew, mode: 'lines', type: 'scatter',
        name: 'EAGLES model',
        line: { color: '#0e7490', width: 2.5 },
      },
    );
  }

  const teffRow = state.refTable.find(r => r.parameter === 'Teff');
  const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
  const targetTeff = Number(teffRow?.value);
  const targetLiew = Number(liewRow?.value);
  if (Number.isFinite(targetTeff) && Number.isFinite(targetLiew)) {
    dataIso.push({
      x: [targetTeff],
      y: [targetLiew],
      error_x: { type: 'data', array: [positiveUncertainty(teffRow, 100)], visible: true, color: '#4a5568' },
      error_y: { type: 'data', array: [positiveUncertainty(liewRow, 15)], visible: true, color: '#4a5568' },
      mode: 'markers',
      name: state.activeTarget ? state.activeTarget.name : 'Target',
      marker: { color: '#0e7490', size: 12, line: { color: '#ffffff', width: 2 }, symbol: 'star' }
    });
  }

  const layoutIso = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 46, b: 40 },
    xaxis: {
      title: 'Teff (K)',
      range: [6500, 3000], // inverted, matching EAGLES's own figures
      gridcolor: 'rgba(0,0,0,0.06)',
      color: '#4a5568',
    },
    yaxis: {
      title: 'Li EW (mÅ)',
      gridcolor: 'rgba(0,0,0,0.06)',
      color: '#4a5568',
    },
    legend: { x: 0, y: 1.22, orientation: 'h', font: { color: '#4a5568' }, bgcolor: 'transparent' }
  };

  Plotly.react('plot-eagles-iso', dataIso, layoutIso, { responsive: true, displayModeBar: false, staticPlot: true });

  // PLOT 2: Posterior PDF (EAGLES reports the posterior over log age; the
  // backend's age_grid already expresses it on a Myr axis).
  const dataPDF = [{
    x: ageGrid,
    y: state.results.eagles.agePDF,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'EAGLES posterior',
    line: { color: '#0e7490', width: 2.5 },
    fillcolor: 'rgba(14, 116, 144, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.react('plot-eagles-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}

export function setupEaglesPageControls() {
  const select = document.getElementById('eagles-model-select');
  if (!select) return;
  select.addEventListener('change', () => {
    saveTargetState({ silent: true });
  });
}
