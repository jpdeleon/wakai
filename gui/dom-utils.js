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
