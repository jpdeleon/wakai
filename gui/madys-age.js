/**
 * MADYS/PARSEC2 age fit -- a second, independent age-fitting method (card on
 * the Isochrones page) alongside isochrones2/MIST: isochrones2 has no PARSEC
 * support to add a model option to (only a MIST ModelGridInterpolator exists
 * in ext_tools/isochrones2), so this calls MADYS directly instead. Unlike
 * the isochrones2 fit, this re-queries its own photometry over the network
 * by Gaia ID rather than reusing this app's ref_table photometry -- see
 * run_madys_age.py.
 */
import { state, ageGrid, interpolatePDF, defaultResults } from './state.js';
import { formatSignificant, showToast } from './dom-utils.js';
import { apiCall } from './api.js';
import { invalidateJointResult } from './joint-status.js';
import { saveTargetState } from './target.js';

export function setupMadysAgePageControls() {
  const runBtn = document.getElementById('btn-run-madys-age');
  if (runBtn) runBtn.addEventListener('click', runMadysAge);

  document.getElementById('madys-age-model-select')?.addEventListener('change', () => {
    saveTargetState({ silent: true });
  });
}

export function invalidateMadysAge() {
  state.results.madysAge = defaultResults().madysAge;
  document.getElementById('madys-age-summary-card').style.display = 'none';
  document.getElementById('madys-age-results-panel').style.display = 'none';
  invalidateJointResult();
}

export async function runMadysAge() {
  const target = state.activeTarget;
  const gaiaId = String(target?.gaiaid || '').trim();
  if (!gaiaId) {
    showToast('Load a target with a Gaia DR3 ID on the Target page first');
    return;
  }

  const model = document.getElementById('madys-age-model-select')?.value || 'parsec2';
  const runBtn = document.getElementById('btn-run-madys-age');
  document.getElementById('madys-age-results-panel').style.display = 'none';
  document.getElementById('madys-age-loader').style.display = 'flex';
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/madys-age', { gaiaid: gaiaId, model });

    const pdf = interpolatePDF(result.age_grid, result.pdf, ageGrid);
    const median = result.median;
    const lower = median - result.lower_1sig;
    const upper = median + result.upper_1sig;

    state.results.madysAge = {
      agePDF: pdf,
      median,
      lower,
      upper,
      ci: `[${formatSignificant(lower, 8)}, ${formatSignificant(upper, 8)}]`,
      status: 'Calculated',
      model: result.model,
      mass: result.mass,
      radius: result.radius,
      teff: result.teff,
      logg: result.logg,
    };
    invalidateJointResult();

    renderMadysAgeResult(state.results.madysAge);

    document.getElementById('madys-age-loader').style.display = 'none';
    document.getElementById('madys-age-results-panel').style.display = 'block';
    drawMadysAgePlot();

    await saveTargetState({ silent: true });
    showToast('MADYS age fit completed successfully');
  } catch (err) {
    document.getElementById('madys-age-loader').style.display = 'none';
    showToast(`MADYS age fit failed: ${err.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

export function renderMadysAgeResult(result) {
  document.getElementById('madys-age-res-age').innerText = result.median != null
    ? `${formatSignificant(result.median, 8)} Myr`
    : '-';
  const fmtStat = (stat, digits = 6) => stat
    ? `${formatSignificant(stat.median, digits)} [${formatSignificant(stat.lower, digits)}, ${formatSignificant(stat.upper, digits)}]`
    : '-';
  document.getElementById('madys-age-res-mass').innerText = fmtStat(result.mass);
  document.getElementById('madys-age-res-radius').innerText = fmtStat(result.radius);
  document.getElementById('madys-age-res-teff').innerText = fmtStat(result.teff, 6);
  document.getElementById('madys-age-res-logg').innerText = fmtStat(result.logg, 4);
  document.getElementById('madys-age-model-used').innerText = result.model
    ? `Fit used the ${result.model.toUpperCase()} grid via MADYS.`
    : '';
  document.getElementById('madys-age-summary-card').style.display = 'block';
}

export function drawMadysAgePlot() {
  if (!state.results.madysAge.agePDF || state.results.madysAge.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }
  const pdf = state.results.madysAge.agePDF;

  const dataPDF = [{
    x: ageGrid,
    y: pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'MADYS age',
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

  Plotly.react('plot-madys-age-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}
