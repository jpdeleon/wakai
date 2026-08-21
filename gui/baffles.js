/**
 * BAFFLES (lithium + chromospheric activity) page: run button and the two
 * Plotly plots.
 */
import { state, ageGrid, interpolatePDF, positiveUncertainty, isDefaultUncertainty } from './state.js';
import { formatSignificant, showToast, describeDefaultUncertainties, showErrorLog, hideErrorLog } from './dom-utils.js';
import { apiCall } from './api.js';
import { invalidateJointResult, updateWorkflowStrip } from './joint-status.js';
import { saveTargetState } from './target.js';

export async function runBaffles() {
    const runBtn = document.getElementById('btn-run-baffles');
    const teffRow = state.refTable.find(r => r.parameter === 'Teff');
    const liewRow = state.refTable.find(r => r.parameter === 'Li EW');
    const rhkRow = state.refTable.find(r => r.parameter === 'log R\'HK');
    const teffVal = parseFloat(teffRow?.value);
    const liewVal = liewRow?.value == null || liewRow.value === '' ? null : Number(liewRow.value);
    const rhkVal = rhkRow?.value == null || rhkRow.value === '' ? null : Number(rhkRow.value);
    // rhk and liew are each individually optional (a star may have only one
    // measurement, or a B-V outside one indicator's range -- see
    // run_baffles.py's per-indicator skip), but at least one is required.
    const hasValidLiew = liewVal !== null && Number.isFinite(liewVal);
    const hasValidRhk = rhkVal !== null && Number.isFinite(rhkVal);
    if (!Number.isFinite(teffVal) || (!hasValidRhk && !hasValidLiew)) {
      showToast("Add valid Teff and at least one of log R'HK or Li EW to the reference table");
      return;
    }

    const teffErr = positiveUncertainty(teffRow, 100);
    const liewErr = positiveUncertainty(liewRow, 15);
    const rhkErr = positiveUncertainty(rhkRow, 0.05);
    const defaultUncertaintyNote = describeDefaultUncertainties([
      ...(isDefaultUncertainty(teffRow) ? [{ label: 'Teff', value: teffErr, unit: ' K' }] : []),
      ...(hasValidLiew && isDefaultUncertainty(liewRow) ? [{ label: 'Li EW', value: liewErr, unit: ' mÅ' }] : []),
      ...(hasValidRhk && isDefaultUncertainty(rhkRow) ? [{ label: "log R'HK", value: rhkErr, unit: '' }] : []),
    ]);

    // BAFFLES needs (B-V)o, not Teff. The backend derives it server-side from
    // the real Pecaut & Mamajek dwarf table (wakai.catalog.get_bv_from_teff)
    // rather than the client guessing it from an uncited polynomial.
    document.getElementById('baffles-results-panel').style.display = 'none';
    document.getElementById('baffles-loader').style.display = 'flex';
    hideErrorLog('baffles-error-log');
    runBtn.disabled = true;

    try {
      const result = await apiCall('run/baffles', {
        teff: teffVal, teff_err: teffErr,
        rhk: hasValidRhk ? rhkVal : undefined, liew: hasValidLiew ? liewVal : undefined,
        liew_err: liewErr, rhk_err: rhkErr
      });

      // Resample PDFs onto common JS ageGrid. ca_pdf/ca_stats can be null if
      // this star's B-V falls outside BAFFLES's calibrated range for
      // chromospheric activity (narrower than lithium's range -- see
      // run_baffles.py's _posterior_or_none); the backend only returns
      // status "ok" here if at least one of ca/li succeeded, so combStats
      // below can't end up with nothing to fall back on.
      const liPdf = result.li_pdf ? interpolatePDF(result.age_grid, result.li_pdf, ageGrid) : null;
      const actPdf = result.ca_pdf ? interpolatePDF(result.age_grid, result.ca_pdf, ageGrid) : null;
      const combinedPdf = result.combined_pdf
        ? interpolatePDF(result.age_grid, result.combined_pdf, ageGrid)
        : (actPdf || liPdf);

      // BAFFLES stats: [p2, p16, p50, p84, p97]
      const liStats = result.li_stats ? {
        median: result.li_stats[2], lower: result.li_stats[1], upper: result.li_stats[3]
      } : null;
      const actStats = result.ca_stats ? {
        median: result.ca_stats[2], lower: result.ca_stats[1], upper: result.ca_stats[3]
      } : null;
      const combStats = result.combined_stats ? {
        median: result.combined_stats[2], lower: result.combined_stats[1], upper: result.combined_stats[3]
      } : (actStats || liStats);

      // Save to state (use resampled PDFs for consistency)
      state.results.baffles.liPDF = liPdf;
      state.results.baffles.actPDF = actPdf;
      state.results.baffles.agePDF = combinedPdf;
      state.results.baffles.median = combStats.median;
      state.results.baffles.lower = combStats.lower;
      state.results.baffles.upper = combStats.upper;
      state.results.baffles.liStats = liStats;
      state.results.baffles.activityStats = actStats;
      state.results.baffles.bvLookup = result.bv_lookup || null;
      state.results.baffles.caSkipReason = result.ca_skip_reason || null;
      state.results.baffles.liSkipReason = result.li_skip_reason || null;
      state.results.baffles.ci = `[${formatSignificant(combStats.lower, 8)}, ${formatSignificant(combStats.upper, 8)}]`;
      state.results.baffles.status = 'Calculated';
      invalidateJointResult();

      // Update Summary Cards
      document.getElementById('baffles-res-li-age').innerText = liStats ? `${formatSignificant(liStats.median, 8)} Myr` : 'N/A';
      document.getElementById('baffles-res-act-age').innerText = actStats ? `${formatSignificant(actStats.median, 8)} Myr` : 'N/A';
      document.getElementById('baffles-res-comb-age').innerText = `${formatSignificant(combStats.median, 8)} Myr`;
      const bvEl = document.getElementById('baffles-res-bv');
      if (bvEl && result.bv_lookup) {
        const bv = result.bv_lookup;
        bvEl.innerText = bv.bv_err != null
          ? `${formatSignificant(bv.bv, 8)} ± ${formatSignificant(bv.bv_err, 6)} (from Teff, Pecaut & Mamajek table)`
          : `${formatSignificant(bv.bv, 8)} (from Teff, Pecaut & Mamajek table)`;
        if (bv.caveat) showToast(bv.caveat);
      }
      const bvInEl = document.getElementById('baffles-in-bv');
      if (bvInEl && result.bv_lookup && result.bv_lookup.bv != null) {
        const bv = result.bv_lookup;
        bvInEl.innerText = bv.bv_err != null
          ? `${formatSignificant(bv.bv, 8)} ± ${formatSignificant(bv.bv_err, 6)}`
          : `${formatSignificant(bv.bv, 8)}`;
      }
      const caveatEl = document.getElementById('baffles-caveat-hint');
      if (caveatEl) {
        const caveats = [result.ca_skip_reason, result.li_skip_reason].filter(Boolean);
        caveatEl.textContent = caveats.join(' ');
        caveatEl.hidden = caveats.length === 0;
      }
      document.getElementById('baffles-summary-card').style.display = 'block';

      // Update Pipeline Status Dot
      document.getElementById('status-baffles-dot').className = "dot dot-active";
      document.getElementById('status-baffles-text').innerText = `${formatSignificant(combStats.median, 8)} Myr`;
      updateWorkflowStrip();

      document.getElementById('baffles-loader').style.display = 'none';
      document.getElementById('baffles-results-panel').style.display = 'block';
      drawBafflesPlots();

      await saveTargetState({ silent: true });
      showToast(
        defaultUncertaintyNote ? `BAFFLES run completed successfully. ${defaultUncertaintyNote}` : "BAFFLES run completed successfully",
        { persist: Boolean(defaultUncertaintyNote) }
      );
    } catch (err) {
      // No simulated fallback: an age here would not have come from the
      // real BAFFLES calcium/lithium calibrations, so we show the failure
      // instead of a fabricated number.
      state.results.baffles.liPDF = null;
      state.results.baffles.actPDF = null;
      state.results.baffles.agePDF = null;
      state.results.baffles.median = null;
      state.results.baffles.ci = null;
      state.results.baffles.status = 'Error';
      invalidateJointResult();

      document.getElementById('baffles-res-li-age').innerText = 'Error';
      document.getElementById('baffles-res-act-age').innerText = 'Error';
      document.getElementById('baffles-res-comb-age').innerText = err.message;
      document.getElementById('baffles-summary-card').style.display = 'block';
      document.getElementById('status-baffles-dot').className = "dot dot-danger";
      document.getElementById('status-baffles-text').innerText = "Error";
      updateWorkflowStrip();

      document.getElementById('baffles-loader').style.display = 'none';
      document.getElementById('baffles-results-panel').style.display = 'none';

      showErrorLog('baffles-error-log', err.message);
    } finally {
      runBtn.disabled = false;
    }
}

export function drawBafflesPlots() {
  if (!state.results.baffles.agePDF || state.results.baffles.status !== 'Calculated') {
    // Nothing real to show yet -- do not fabricate a placeholder posterior.
    return;
  }

  const liPdf = state.results.baffles.liPDF;
  const actPdf = state.results.baffles.actPDF;
  const combPdf = state.results.baffles.agePDF;

  // PLOT 1: Individual Age Posteriors
  const dataIndiv = [
    {
      x: ageGrid,
      y: liPdf,
      type: 'scatter',
      mode: 'lines',
      name: 'Lithium Depletion (Li EW)',
      line: { color: '#ea580c', width: 2 }
    },
    {
      x: ageGrid,
      y: actPdf,
      type: 'scatter',
      mode: 'lines',
      name: 'Chromospheric Activity (log R\'HK)',
      line: { color: '#d946ef', width: 2 }
    }
  ].filter(trace => Array.isArray(trace.y));

  const layoutIndiv = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 46, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    // Horizontal legend above the plot (not x:0,y:1 inside it) -- a PDF's
    // peak can land anywhere along the age axis depending on the star, so
    // no in-plot corner is reliably clear of the curves themselves. Extra
    // top margin vs. the analogous summary-page legend: this chart's two
    // labels ("Lithium Depletion (Li EW)", "Chromospheric Activity
    // (log R'HK)") are longer and this container is shorter (360px vs
    // 440px), so it needs more headroom to clear the axes area cleanly.
    legend: { x: 0, y: 1.22, orientation: 'h', font: { color: '#4a5568' }, bgcolor: 'transparent' }
  };

  Plotly.react('plot-baffles-indiv', dataIndiv, layoutIndiv, { responsive: true, displayModeBar: false, staticPlot: true });

  // PLOT 2: Combined BAFFLES PDF
  const dataPDF = [{
    x: ageGrid,
    y: combPdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Combined BAFFLES',
    line: { color: '#ea580c', width: 2.5 },
    fillcolor: 'rgba(234, 88, 12, 0.15)'
  }];

  const layoutPDF = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Probability Density', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    showlegend: false
  };

  Plotly.react('plot-baffles-posterior', dataPDF, layoutPDF, { responsive: true, displayModeBar: false, staticPlot: true });
}
