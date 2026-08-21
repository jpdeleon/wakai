/**
 * Pure DOM/formatting helpers with no dependency on app state.
 */

export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export function safeSourceUrl(value) {
  if (!value) return '';
  try {
    const url = new URL(value, window.location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch (_) {
    return '';
  }
}

export function formatSignificant(value, significantFigures = 10) {
  const number = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(number)) return '';
  if (number === 0) return '0.00';

  // Round to the requested significant figures first -- a value's real
  // precision, not a fixed decimal count -- then format for display.
  // Visualization defaults to 2 decimal places, except when that would
  // erase a genuinely nonzero value (e.g. a 0.0035 B-V uncertainty
  // rounding to "0.00"): fall back to enough decimals to show it instead
  // of silently reporting zero.
  const precise = Number(number.toPrecision(significantFigures));
  if (Math.round(precise * 100) === 0) {
    const magnitude = Math.floor(Math.log10(Math.abs(precise)));
    return precise.toFixed(-magnitude + 1);
  }
  return precise.toFixed(2);
}

export function formatMetadata(value, significantFigures = 10) {
  return formatSignificant(value, significantFigures) || '-';
}

// A ref_table row starts with the 'No catalog measurement selected'
// placeholder reference (see emptyDefaultParameters in state.js) and keeps
// it if the user types a value directly into the row instead of picking a
// VizieR/SIMBAD candidate. Once the row actually holds a value, that literal
// placeholder reads as an error message rather than a description, so show
// 'user input' instead wherever the reference is displayed.
export function referenceLabel(row) {
  const hasValue = row.value != null && Number.isFinite(Number(row.value));
  if (hasValue && row.reference === 'No catalog measurement selected') return 'user input';
  return row.reference;
}

// Builds a short caveat note listing which inputs to a run had no measured
// uncertainty in the reference table, so a generic fallback (e.g. a flat
// +-100 K Teff error) was silently substituted for the fit. `entries` is
// the subset of a page's inputs that isDefaultUncertainty() (state.js)
// already flagged as defaulted, as {label, value, unit}; value is the
// fallback actually sent to the backend, so the note names the exact
// number used instead of a generic warning.
export function describeDefaultUncertainties(entries) {
  if (!entries.length) return '';
  const list = entries
    .map(({ label, value, unit }) => `${label} (±${formatSignificant(value, 3)}${unit || ''} default)`)
    .join(', ');
  return `No measured uncertainty for: ${list} -- used a generic default. Enter a real value in the reference table for a more reliable result.`;
}

let toastTimeout = null;

export function showToast(text, { persist = false } = {}) {
  const toast = document.getElementById('toast');
  toast.innerText = text;
  toast.style.display = 'block';
  if (toastTimeout) {
    clearTimeout(toastTimeout);
    toastTimeout = null;
  }
  if (!persist) {
    toastTimeout = setTimeout(() => {
      toast.style.display = 'none';
      toastTimeout = null;
    }, 2500);
  }
}

export function hideToast() {
  const toast = document.getElementById('toast');
  if (toastTimeout) {
    clearTimeout(toastTimeout);
    toastTimeout = null;
  }
  toast.style.display = 'none';
}

// Persistent, embedded run-failure display: unlike showToast() (auto-hides
// after 2.5s, easy to miss on a long backend/network error message), an
// error-log block stays on the page -- inside the card it belongs to --
// until the user dismisses it or the next run replaces/clears it. Each page
// with a Run button gets its own `<div class="error-log" id="...">` (see
// styles.css's .error-log rule) holding a `.error-log-message` node for the
// text; showErrorLog()/hideErrorLog() just target it by that container id.
// textContent (not innerHTML) throughout -- the message can be raw
// subprocess stderr, not markup-safe.
export function showErrorLog(containerId, message) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const messageEl = container.querySelector('.error-log-message');
  if (messageEl) messageEl.textContent = message;
  container.hidden = false;
}

export function hideErrorLog(containerId) {
  const container = document.getElementById(containerId);
  if (container) container.hidden = true;
}

// Wires every page's error-log dismiss ("x") button in one pass -- called
// once from app.js's initApp(), rather than each page module wiring its own
// (the dismiss behavior is identical everywhere: hide the ancestor
// .error-log). closest() finds that ancestor regardless of which page's
// error-log the button belongs to.
export function setupErrorLogDismissButtons() {
  document.querySelectorAll('.error-log-close').forEach(button => {
    button.addEventListener('click', () => {
      const container = button.closest('.error-log');
      if (container) container.hidden = true;
    });
  });
}
