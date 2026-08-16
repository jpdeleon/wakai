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
  if (number === 0) return '0';
  const formatted = number.toPrecision(significantFigures);
  if (formatted.includes('e')) {
    const [mantissa, exponent] = formatted.split('e');
    return `${mantissa.replace(/\.?0+$/, '')}e${Number(exponent)}`;
  }
  return formatted.includes('.') ? formatted.replace(/\.?0+$/, '') : formatted;
}

export function formatMetadata(value, significantFigures = 10) {
  return formatSignificant(value, significantFigures) || '-';
}

export function showToast(text) {
  const toast = document.getElementById('toast');
  toast.innerText = text;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 2500);
}
