/**
 * Isochrones (isochrones2/MIST) page.
 */
import { state, ageGrid, interpolatePDF, defaultResults, positiveUncertainty, isDefaultUncertainty } from './state.js';
import { formatSignificant, showToast, describeDefaultUncertainties, showErrorLog, hideErrorLog } from './dom-utils.js';
import { apiCall } from './api.js';
import { invalidateJointResult } from './joint-status.js';
import { saveTargetState } from './target.js';

// 68% CI width in log10(age) beyond which the fit is treated as too weakly
// constrained to trust -- see run_isochrones.py's docstring and
// defaultResults()'s isochrone.broad comment for why this happens (a
// young and old main-sequence dwarf can be nearly degenerate along a MIST
// isochrone track in color/luminosity space, with no independent age
// handle beyond parallax + photometry). Chosen from real test runs against
// TOI-6715 (a real, verified young star): a photometry-only fit gave a
// ~0.55 dex 68% CI (correctly flagged); a well-constrained fit for an
// evolved/subgiant star would be well under this.
export const ISOCHRONE_BROAD_DEX = 0.3;

// Every ref_table row the Isochrones page's SingleStarModel fit can use --
// parallax + all six bands (required to run at all) plus the two optional
// spectroscopic constraints. Drives both the checkbox list (built once by
// setupIsochronesPageControls) and the payload built in runIsochrones, so
// the two stay in sync by construction. `band` maps a row to the photometry
// dict key isochrones2 expects; rows without `band` (parallax/teff/feh) are
// top-level SingleStarModel kwargs instead. logg has no ref_table row (it's
// a fixed, editable dwarf-star prior, not a measurement) so it isn't listed
// here -- see the dedicated #chk-isochrone-logg controls in index.html.
export const ISOCHRONE_INPUT_SPECS = [
  { id: 'parallax', label: 'Parallax', unit: 'mas', digits: 8, errDigits: 4, fallbackErr: 0.05, required: true },
  { id: 'bpmag', label: 'BP mag', band: 'BP', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'gmag', label: 'G mag', band: 'G', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'rpmag', label: 'RP mag', band: 'RP', digits: 6, errDigits: 4, fallbackErr: 0.02 },
  { id: 'jmag', label: 'J mag', band: 'J', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'hmag', label: 'H mag', band: 'H', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'kmag', label: 'K mag', band: 'K', digits: 6, errDigits: 4, fallbackErr: 0.03 },
  { id: 'teff', label: 'Teff', unit: 'K', digits: 8, errDigits: 6, fallbackErr: 100 },
  { id: 'feh', label: 'Fe/H', digits: 8, errDigits: 6, fallbackErr: 0.1 },
];

export function isochroneInputRowHtml(spec) {
  const reqTag = spec.required
    ? ' <span style="color: var(--ink-faint); font-size: 0.75rem;">(required)</span>'
    : '';
  return `
    <div class="form-group" style="display: flex; align-items: center; gap: 0.5rem;">
      <input type="checkbox" id="chk-isochrone-${spec.id}" class="chk-isochrone-input" style="accent-color: var(--isochrone);">
      <label for="chk-isochrone-${spec.id}" style="margin: 0; flex: 1;">${spec.label}${reqTag}</label>
      <span class="meta-value" id="isochrone-in-${spec.id}" style="font-size: 0.85rem;">-</span>
    </div>`;
}

export function syncIsochroneInputParams() {
  const target = state.activeTarget;
  const runBtn = document.getElementById('btn-run-isochrones');
  const hint = document.getElementById('isochrone-input-hint');

  let parallaxAvailable = false;
  let anyBandAvailable = false;

  ISOCHRONE_INPUT_SPECS.forEach(spec => {
    const row = state.refTable.find(r => r.id === spec.id);
    const available = Boolean(row && Number.isFinite(Number(row.value)));

    const valueSpan = document.getElementById(`isochrone-in-${spec.id}`);
    if (valueSpan) {
      valueSpan.innerText = available
        ? `${formatSignificant(row.value, spec.digits)} ± ${formatSignificant(row.uncertainty, spec.errDigits)}${spec.unit ? ` ${spec.unit}` : ''}`
        : '—';
    }

    const chk = document.getElementById(`chk-isochrone-${spec.id}`);
    if (chk) {
      chk.disabled = !available;
      if (!available) {
        chk.checked = false;
      } else if ((spec.required || spec.band) && !chk.dataset.userTouched) {
        // Parallax + photometry are auto-checked once available, since the
        // fit can't run without them; Teff/Fe-H stay opt-in (unchecked)
        // until the user explicitly ticks them, matching the page's
        // original teff/feh-only behavior. Either way, once the user
        // manually toggles a box, this sync stops overriding their choice.
        chk.checked = true;
      }
    }

    if (spec.id === 'parallax') parallaxAvailable = available;
    if (spec.band && available) anyBandAvailable = true;
  });

  const hasId = Boolean(String(target?.gaiaid || '').trim());
  const ready = hasId && parallaxAvailable && anyBandAvailable;
  if (runBtn) runBtn.disabled = !ready;
  if (hint) {
    hint.innerText = !target
      ? 'Load a target on the Target page first.'
      : !parallaxAvailable
        ? 'Missing parallax -- fetch it on the Target page first.'
        : !anyBandAvailable
          ? 'No photometry available -- fetch it on the Target page first.'
          : 'Ready. Uncheck any input above to exclude it from the fit.';
  }
}

export function setupIsochronesPageControls() {
  const list = document.getElementById('isochrone-input-list');
  if (list) {
    list.innerHTML = ISOCHRONE_INPUT_SPECS.map(isochroneInputRowHtml).join('');
    list.querySelectorAll('.chk-isochrone-input').forEach(chk => {
      chk.addEventListener('change', () => {
        chk.dataset.userTouched = '1';
        saveTargetState({ silent: true });
      });
    });
  }
  const runBtn = document.getElementById('btn-run-isochrones');
  if (runBtn) runBtn.addEventListener('click', runIsochrones);
  document.getElementById('btn-delete-isochrone')?.addEventListener('click', () => {
    invalidateIsochrone();
    saveTargetState({ silent: true });
    showToast('Isochrone fit results deleted');
  });

  // logg + prior bounds have no other natural save checkpoint before the
  // user clicks Run (unlike the checkboxes above, which mirror ref_table
  // availability that a run would already persist) -- save on every change.
  [
    'chk-isochrone-logg', 'isochrone-logg-value', 'isochrone-logg-err',
    'isochrone-age-prior-min', 'isochrone-age-prior-max',
    'isochrone-av-prior-min', 'isochrone-av-prior-max',
    'isochrone-eep-prior-min', 'isochrone-eep-prior-max',
  ].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      saveTargetState({ silent: true });
    });
  });
}

export function invalidateIsochrone() {
  state.results.isochrone = defaultResults().isochrone;
  document.getElementById('isochrone-summary-card').style.display = 'none';
  document.getElementById('isochrone-results-panel').style.display = 'none';
  hideErrorLog('isochrone-error-log');
  invalidateJointResult();
}

export async function runIsochrones() {
  const target = state.activeTarget;
  if (!target) {
    showToast('Load a target on the Target page first');
    return;
  }

  const photometry = {};
  let parallax = null;
  let parallaxErr = null;
  const payload = {};
  const defaultedInputs = [];

  ISOCHRONE_INPUT_SPECS.forEach(spec => {
    const chk = document.getElementById(`chk-isochrone-${spec.id}`);
    if (!chk || !chk.checked) return;
    const row = state.refTable.find(r => r.id === spec.id);
    if (!row || !Number.isFinite(Number(row.value))) return;

    const value = Number(row.value);
    const err = positiveUncertainty(row, spec.fallbackErr);
    if (isDefaultUncertainty(row)) {
      const unit = spec.unit ? ` ${spec.unit}` : (spec.band ? ' mag' : '');
      defaultedInputs.push({ label: spec.label, value: err, unit });
    }
    if (spec.id === 'parallax') {
      parallax = value;
      parallaxErr = err;
    } else if (spec.band) {
      photometry[spec.band] = [value, err];
    } else {
      payload[spec.id] = value;
      payload[`${spec.id}_err`] = err;
    }
  });
  const defaultUncertaintyNote = describeDefaultUncertainties(defaultedInputs);

  if (parallax === null || Object.keys(photometry).length === 0) {
    showToast('Check parallax plus at least one photometric band to run isochrones');
    return;
  }
  payload.photometry = photometry;
  payload.parallax = parallax;
  payload.parallax_err = parallaxErr;

  const chkLogg = document.getElementById('chk-isochrone-logg');
  if (chkLogg?.checked) {
    const loggValue = Number(document.getElementById('isochrone-logg-value')?.value);
    const loggErr = Number(document.getElementById('isochrone-logg-err')?.value);
    if (Number.isFinite(loggValue) && Number.isFinite(loggErr) && loggErr > 0) {
      payload.logg = loggValue;
      payload.logg_err = loggErr;
    }
  }

  // Age/AV/EEP prior bounds -- age+AV match notebooks/ext_code_isochrones
  // .ipynb's mod.set_prior(age=FlatPrior(...), AV=FlatPrior(...)); EEP isn't
  // touched by the notebook but is still one of SingleStarModel's fit
  // parameters, so it's exposed the same way (default: full MIST grid).
  const ageMin = Number(document.getElementById('isochrone-age-prior-min')?.value);
  const ageMax = Number(document.getElementById('isochrone-age-prior-max')?.value);
  const avMin = Number(document.getElementById('isochrone-av-prior-min')?.value);
  const avMax = Number(document.getElementById('isochrone-av-prior-max')?.value);
  const eepMin = Number(document.getElementById('isochrone-eep-prior-min')?.value);
  const eepMax = Number(document.getElementById('isochrone-eep-prior-max')?.value);
  if (![ageMin, ageMax, avMin, avMax, eepMin, eepMax].every(Number.isFinite)
    || ageMin >= ageMax || avMin >= avMax || eepMin >= eepMax) {
    showToast('Check the age/AV/EEP prior bounds -- min must be less than max');
    return;
  }
  payload.age_prior_bounds = [ageMin, ageMax];
  payload.av_prior_bounds = [avMin, avMax];
  payload.eep_prior_bounds = [eepMin, eepMax];

  const runBtn = document.getElementById('btn-run-isochrones');
  document.getElementById('isochrone-results-panel').style.display = 'none';
  document.getElementById('isochrone-loader').style.display = 'flex';
  hideErrorLog('isochrone-error-log');
  runBtn.disabled = true;

  try {
    const result = await apiCall('run/isochrones', payload);

    const pdf = interpolatePDF(result.age_grid, result.pdf, ageGrid);
    const median = result.median;
    const lower = median - result.lower_1sig;
    const upper = median + result.upper_1sig;
    const broad = (Math.log10(upper) - Math.log10(lower)) > ISOCHRONE_BROAD_DEX;

    state.results.isochrone = {
      agePDF: pdf,
      median,
      lower,
      upper,
      ci: `[${formatSignificant(lower, 8)}, ${formatSignificant(upper, 8)}]`,
      status: 'Calculated',
      broad,
      mass: result.mass,
      radius: result.radius,
      teffDerived: result.teff_derived,
      logg: result.logg,
      fehDerived: result.feh_derived,
      usedInputs: result.used_inputs,
      priorBounds: result.prior_bounds,
      convergence: result.convergence,
    };
    invalidateJointResult();

    renderIsochroneResult(state.results.isochrone);

    document.getElementById('isochrone-loader').style.display = 'none';
    document.getElementById('isochrone-results-panel').style.display = 'block';
    drawIsochronePlots();

    await saveTargetState({ silent: true });
    const notConverged = result.convergence && !result.convergence.converged;
    const caveats = [
      broad ? 'weakly constrained (excluded from joint estimate)' : null,
      notConverged ? 'MCMC may not have converged (see caveat below)' : null,
    ].filter(Boolean);
    const message = caveats.length ? `Isochrone fit complete, but ${caveats.join('; ')}` : 'Isochrone fit completed successfully';
    showToast(
      defaultUncertaintyNote ? `${message} ${defaultUncertaintyNote}` : message,
      { persist: caveats.length > 0 || Boolean(defaultUncertaintyNote) }
    );
  } catch (err) {
    document.getElementById('isochrone-loader').style.display = 'none';
    showErrorLog('isochrone-error-log', err.message);
  } finally {
    runBtn.disabled = false;
  }
}

export function renderIsochroneResult(iso) {
  document.getElementById('isochrone-res-age').innerText = iso.median != null
    ? `${formatSignificant(iso.median, 8)} Myr`
    : '-';
  const fmtStat = (stat, digits = 6) => stat
    ? `${formatSignificant(stat.median, digits)} [${formatSignificant(stat.lower, digits)}, ${formatSignificant(stat.upper, digits)}]`
    : '-';
  document.getElementById('isochrone-res-mass').innerText = fmtStat(iso.mass);
  document.getElementById('isochrone-res-radius').innerText = fmtStat(iso.radius);
  document.getElementById('isochrone-res-teff').innerText = fmtStat(iso.teffDerived, 6);
  document.getElementById('isochrone-res-logg').innerText = fmtStat(iso.logg, 4);
  document.getElementById('isochrone-res-feh').innerText = fmtStat(iso.fehDerived, 4);

  const used = iso.usedInputs;
  const priors = iso.priorBounds;
  const priorsText = priors
    ? ` Priors: log10(age) in [${priors.age?.[0]}, ${priors.age?.[1]}], AV in [${priors.AV?.[0]}, ${priors.AV?.[1]}], EEP in [${priors.eep?.[0]}, ${priors.eep?.[1]}].`
    : '';
  document.getElementById('isochrone-inputs-used').innerText = used
    ? `Fit used: ${used.bands.join(', ')} photometry, parallax${used.teff ? ', Teff' : ''}${used.feh ? ', Fe/H' : ''}${used.logg ? ', log g' : ''}.${priorsText}`
    : '';
  document.getElementById('isochrone-summary-card').style.display = 'block';

  const caveatCard = document.getElementById('isochrone-caveat-card');
  if (caveatCard) caveatCard.style.display = iso.broad ? 'block' : 'none';

  const convergenceCard = document.getElementById('isochrone-convergence-card');
  const convergenceText = document.getElementById('isochrone-convergence-text');
  const convergence = iso.convergence;
  if (convergenceCard) {
    convergenceCard.style.display = convergence && !convergence.converged ? 'block' : 'none';
    if (convergenceText && convergence) {
      convergenceText.innerText = `${convergence.warning} (acceptance fraction ${convergence.acceptance_fraction}, `
        + `autocorrelation time ${convergence.autocorr_time ?? 'undetermined'} steps over ${convergence.n_iter} `
        + 'production steps). Treat this age posterior\'s shape with caution.';
    }
  }
}

export function drawIsochronePlots() {
  if (!state.results.isochrone.agePDF || state.results.isochrone.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }
  const pdf = state.results.isochrone.agePDF;

  const dataPDF = [{
    x: ageGrid,
    y: pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Isochrone age',
    line: { color: '#a6740a', width: 2.5 },
    fillcolor: 'rgba(166, 116, 10, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.react('plot-isochrone-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}
