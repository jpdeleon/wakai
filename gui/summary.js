/**
 * Summary page: joint posterior comparison, evolutionary timeline, target
 * properties, reported literature ages, and export/print actions.
 */
import { state, ageGrid } from './state.js';
import { escapeHtml, safeSourceUrl, formatSignificant, referenceLabel, showToast } from './dom-utils.js';
import { getStatsFromPDF, recomputeJointStatus, updateWorkflowStrip } from './joint-status.js';
import { saveTargetState } from './target.js';

export function updateSummaryPage() {
  const isGyroRun = state.results.gyro.status === 'Calculated';
  const isBafflesRun = state.results.baffles.status === 'Calculated';
  const isEaglesRun = state.results.eagles.status === 'Calculated';
  const isIsochroneRun = state.results.isochrone.status === 'Calculated';
  // A very broad (weakly constrained) isochrone posterior is shown on its
  // own page/row but excluded here so it can't dominate/distort the joint
  // product -- see ISOCHRONE_BROAD_DEX.
  const isIsochroneUsable = isIsochroneRun && !state.results.isochrone.broad;

  const gyroPdf = state.results.gyro.agePDF;
  const bafflesPdf = state.results.baffles.agePDF;
  const eaglesPdf = state.results.eagles.agePDF;
  const isochronePdf = state.results.isochrone.agePDF;

  const statusBadge = (calculated, label) => calculated
    ? `<span class="status-indicator"><div class="dot dot-active"></div>${label}</span>`
    : '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Not yet run</span>';

  document.getElementById('td-gyro-status').innerHTML = statusBadge(isGyroRun, 'Active');
  document.getElementById('td-baffles-status').innerHTML = statusBadge(isBafflesRun, 'Active');
  document.getElementById('td-eagles-status').innerHTML = statusBadge(isEaglesRun, 'Active');
  document.getElementById('td-isochrone-status').innerHTML = !isIsochroneRun
    ? statusBadge(false, '')
    : isIsochroneUsable
      ? statusBadge(true, 'Active')
      : '<span class="status-indicator" style="color: var(--danger);"><div class="dot dot-danger"></div>Excluded (weak)</span>';

  const setRow = (prefix, calculated, pdf) => {
    if (calculated) {
      const stats = getStatsFromPDF(pdf);
      document.getElementById(`td-${prefix}-age`).innerText = `${formatSignificant(stats.median, 8)} Myr`;
      document.getElementById(`td-${prefix}-ci`).innerText = `${formatSignificant(stats.lower, 8)}-${formatSignificant(stats.upper, 8)} Myr`;
    } else {
      document.getElementById(`td-${prefix}-age`).innerText = '-';
      document.getElementById(`td-${prefix}-ci`).innerText = '-';
    }
  };
  setRow('gyro', isGyroRun, gyroPdf);
  setRow('baffles', isBafflesRun, bafflesPdf);
  setRow('eagles', isEaglesRun, eaglesPdf);
  setRow('isochrone', isIsochroneRun, isochronePdf);

  updatePropertiesSummary();
  renderReportedAges();

  // Joint posterior: product of whichever of {gyro, BAFFLES-activity,
  // EAGLES, isochrone} are usable, requiring at least two -- a "joint" of a
  // single method isn't a joint constraint, so don't synthesize one from a
  // partial result. BAFFLES and EAGLES are both lithium-EW ages, so the
  // joint takes at most one lithium posterior (EAGLES's); BAFFLES
  // contributes only its chromospheric-activity component, which is
  // independent of lithium -- see recomputeJointStatus().
  // The actual PDF product lives in recomputeJointStatus() (shared with
  // updateWorkflowStrip(), so the top nav's "Joint age" stage stays live on
  // every page, not just this one) -- this only builds the plot's trace
  // list and then renders whatever state.results.joint ends up holding.
  const usable = [];
  if (isGyroRun) usable.push({ label: 'gyro-interp PDF', pdf: gyroPdf, color: '124, 58, 237' });
  if (isBafflesRun && state.results.baffles.actPDF) {
    usable.push({ label: 'BAFFLES activity PDF', pdf: state.results.baffles.actPDF, color: '234, 88, 12' });
  }
  if (isEaglesRun) usable.push({ label: 'EAGLES PDF', pdf: eaglesPdf, color: '14, 116, 144' });
  if (isIsochroneUsable) usable.push({ label: 'Isochrone PDF', pdf: isochronePdf, color: '166, 116, 10' });

  const comparisonTraces = usable.map(m => ({ ...m }));
  if (isIsochroneRun && !isIsochroneUsable) {
    comparisonTraces.push({ label: 'Isochrone PDF (excluded, weakly constrained)', pdf: isochronePdf, color: '166, 116, 10' });
  }
  if (isBafflesRun && state.results.baffles.actPDF) {
    // Shown for transparency but not fed to the joint: BAFFLES's combined
    // (lithium x activity) posterior would double-count the lithium
    // constraint already carried by EAGLES.
    comparisonTraces.push({ label: 'BAFFLES combined PDF (excluded: lithium)', pdf: bafflesPdf, color: '234, 88, 12' });
  }

  recomputeJointStatus();

  if (state.results.joint.status === 'Idle') {
    document.getElementById('summary-joint-age').innerText = '-';
    document.getElementById('summary-joint-ci').innerText = 'Run at least two of gyro-interp, BAFFLES activity, EAGLES, or Isochrones to compute a joint age';
    document.getElementById('td-joint-age').innerText = '-';
    document.getElementById('td-joint-ci').innerText = '-';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator" style="color: var(--text-muted);"><div class="dot dot-inactive"></div>Awaiting 2+ methods</span>';

    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    return;
  }

  // Gyrochronology validity guard: Bouma+2023 rotation-sequence grids are
  // only calibrated down to ~80 Myr (youngest reference cluster). If a
  // significant fraction of the gyro posterior mass falls below this floor,
  // the gyro age is an extrapolation and the joint result is unreliable.
  const GYRO_CALIBRATION_FLOOR_MYR = 80;
  const gyroMassBelowFloor = isGyroRun
    ? WakaiPosterior.probabilityBelow(ageGrid, gyroPdf, GYRO_CALIBRATION_FLOOR_MYR)
    : 0;
  document.getElementById('gyro-caveat-card').style.display = gyroMassBelowFloor > 0.5 ? 'block' : 'none';

  if (state.results.joint.status === 'No overlap') {
    document.getElementById('summary-joint-age').innerText = '—';
    document.getElementById('summary-joint-ci').innerText = 'The method posteriors do not overlap';
    document.getElementById('td-joint-age').innerText = '—';
    document.getElementById('td-joint-ci').innerText = '—';
    document.getElementById('td-joint-status').innerHTML =
      '<span class="status-indicator status-error"><div class="dot dot-danger"></div>Methods disagree</span>';
    Plotly.purge('plot-summary-comparison');
    updateWorkflowStrip();
    saveTargetState({ silent: true });
    showToast('No joint age: method posteriors do not overlap');
    return;
  }

  const jointPdf = state.results.joint.agePDF;
  const jointStats = state.results.joint;
  updateWorkflowStrip();

  // Update Page Labels
  document.getElementById('summary-joint-age').innerText = formatSignificant(jointStats.median, 8);
  document.getElementById('summary-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)} - ${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-age').innerText = `${formatSignificant(jointStats.median, 8)} Myr`;
  document.getElementById('td-joint-ci').innerText = `${formatSignificant(jointStats.lower, 8)}-${formatSignificant(jointStats.upper, 8)} Myr`;
  document.getElementById('td-joint-status').innerHTML =
    `<span class="status-indicator"><div class="dot dot-active"></div>Synthesized (${usable.length}-way)</span>`;

  // Update Evolutionary Stage Timeline
  updateEvolutionTimeline(jointStats.median);

  // Draw Comparison Plot -- every run method (including an excluded, broad
  // isochrone posterior, for transparency) plus the joint product.
  const data = comparisonTraces.map(t => ({
    x: ageGrid,
    y: t.pdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: t.label,
    line: { color: `rgba(${t.color}, 0.7)`, width: 2 },
    fillcolor: `rgba(${t.color}, 0.08)`,
  }));
  data.push({
    x: ageGrid,
    y: jointPdf,
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    name: 'Joint Posterior PDF',
    line: { color: 'rgba(5, 150, 105, 1)', width: 3 },
    fillcolor: 'rgba(5, 150, 105, 0.18)'
  });

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(0,0,0,0.015)',
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: 'Stellar age (Myr)', gridcolor: 'rgba(15,23,42,0.08)', color: '#526171', type: 'log', range: [0, 4.12] },
    yaxis: { title: 'Normalized Probability', gridcolor: 'rgba(0,0,0,0.06)', color: '#4a5568' },
    legend: { x: 0, y: 1.15, orientation: 'h', font: { color: '#4a5568' }, bgcolor: 'transparent' }
  };

  Plotly.react('plot-summary-comparison', data, layout, { responsive: true, displayModeBar: false, staticPlot: true });
  saveTargetState({ silent: true });
}

export function updateEvolutionTimeline(age) {
  // Reset timeline node states
  document.querySelectorAll('.timeline-node').forEach(node => node.classList.remove('active'));

  if (age < 50) {
    document.getElementById('tl-pms').classList.add('active');
  } else if (age >= 50 && age < 200) {
    document.getElementById('tl-zams').classList.add('active');
  } else if (age >= 200 && age < 800) {
    document.getElementById('tl-youngms').classList.add('active');
  } else {
    document.getElementById('tl-oldms').classList.add('active');
  }
}

export function updatePropertiesSummary() {
  const container = document.getElementById('summary-target-properties');
  container.innerHTML = '';

  if (!state.activeTarget) return;

  const titleDiv = document.createElement('div');
  titleDiv.innerHTML = `Target: <strong>${escapeHtml(state.activeTarget.name)}</strong> (Gaia DR3 ${escapeHtml(state.activeTarget.gaiaid)})`;
  titleDiv.style.marginBottom = '0.5rem';
  container.appendChild(titleDiv);

  state.refTable.forEach(row => {
    const item = document.createElement('div');
    item.style.display = 'flex';
    item.style.justifyContent = 'space-between';
    item.style.padding = '0.25rem 0';
    item.style.borderBottom = '1px solid rgba(0,0,0,0.06)';
    item.innerHTML = `
      <span style="color: var(--text-secondary);">${escapeHtml(row.parameter)}:</span>
      <span style="font-family: var(--font-mono);">${escapeHtml(formatSignificant(row.value, 10) || '—')} ± ${escapeHtml(formatSignificant(row.uncertainty, 8) || '—')} <span style="font-size: 0.75rem; color: var(--text-muted);">(${escapeHtml(referenceLabel(row))})</span></span>
    `;
    container.appendChild(item);
  });
}

// Literature ages the user has recorded directly in the reference table
// (e.g. via "+ Add Parameter Row", parameter "Age" with a citation in
// Reference) -- distinct from the computed gyro/BAFFLES/joint rows above,
// so shown as extra rows in the same Comparison table rather than mixed in.
export function renderReportedAges() {
  const tbody = document.getElementById('reported-ages-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const reported = state.refTable.filter(row =>
    /age/i.test(row.parameter || '') && row.value != null && Number.isFinite(Number(row.value))
  );
  if (!reported.length) return;

  const header = document.createElement('tr');
  header.innerHTML = `<td colspan="4" style="border-top: 1px solid var(--glass-border); padding-top: 0.75rem; font-size: 0.75rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.04em;">Reported (from reference table)</td>`;
  tbody.appendChild(header);

  reported.forEach(row => {
    const range = row.uncertainty != null && Number.isFinite(Number(row.uncertainty))
      ? `± ${formatSignificant(row.uncertainty, 6)}`
      : '-';
    const label = referenceLabel(row);
    const hasReference = label && label !== 'No catalog measurement selected';
    const sourceUrl = safeSourceUrl(row.source_url);
    const referenceHtml = sourceUrl
      ? `<a class="catalog-source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(hasReference ? label : 'Source')} ↗</a>`
      : escapeHtml(hasReference ? label : 'No reference given');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.parameter)}</strong><br><span style="font-size: 0.75rem; color: var(--ink-faint);">${referenceHtml}</span></td>
      <td>${escapeHtml(formatSignificant(row.value, 8))}</td>
      <td>${range}</td>
      <td><span class="status-indicator"><div class="dot dot-inactive"></div>Reported</span></td>
    `;
    tbody.appendChild(tr);
  });
}

export function setupSummaryActions() {
  document.getElementById('btn-export-report').addEventListener('click', () => {
    // Refresh state.results.joint right before reading it, same as
    // updateSummaryPage()/updateWorkflowStrip() -- recomputeJointStatus() is
    // the single source of truth for joint status (see its own comment);
    // reimplementing the gyro+baffles-only check here previously produced a
    // self-contradictory export (e.g. joint.status:'Idle' while joint.age
    // held a real value from a gyro+isochrone-only joint).
    recomputeJointStatus();
    const jointResult = state.results.joint;
    const isochroneResult = state.results.isochrone;
    const clusterResult = state.results.cluster;
    const madysResult = state.results.madysCmd;

    const reportData = {
      target: state.activeTarget ? state.activeTarget.name : "Custom",
      gaiaid: state.activeTarget ? state.activeTarget.gaiaid : null,
      refTable: state.refTable,
      runs: {
        gyro: { age: state.results.gyro.median, ci: state.results.gyro.ci, status: state.results.gyro.status },
        baffles: { age: state.results.baffles.median, ci: state.results.baffles.ci, status: state.results.baffles.status },
        eagles: {
          age: state.results.eagles.median, ci: state.results.eagles.ci, status: state.results.eagles.status,
          model: state.results.eagles.model,
        },
        isochrone: {
          age: isochroneResult.median, ci: isochroneResult.ci, status: isochroneResult.status,
          broad: isochroneResult.broad,
        },
        joint: { age: jointResult.median, ci: jointResult.ci, status: jointResult.status },
        // Base64 plot images are intentionally omitted here -- this is a
        // numeric/text report, not a plot archive.
        cluster: clusterResult ? {
          matched: clusterResult.matched,
          clusterName: clusterResult.cluster_name,
          membershipProbability: clusterResult.membership_probability,
          nMembers: clusterResult.n_members,
        } : null,
        madysCmd: {
          status: madysResult.status,
          clusterName: madysResult.clusterName,
          nMembers: madysResult.nMembers,
        },
      },
      exportTime: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wakai_report_${reportData.target.replace(/\s+/g, '_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast("Report exported successfully!");
  });

  document.getElementById('btn-print-report').addEventListener('click', () => {
    window.print();
  });
}
