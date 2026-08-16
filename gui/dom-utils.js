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

export function formatSignificant(value) {
  const number = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(number)) return '';
  return number.toFixed(2);
}

export function formatMetadata(value, significantFigures = 10) {
  return formatSignificant(value, significantFigures) || '-';
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
