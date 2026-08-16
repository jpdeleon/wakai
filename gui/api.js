/**
 * Backend API configuration and fetch wrappers.
 */

export const API_BASE = '/api';

async function readJson(response, endpoint) {
  try {
    return await response.json();
  } catch {
    // e.g. a stale backend whose SPA catch-all serves index.html for
    // routes registered after it was started -- surface that instead of a
    // confusing "Unexpected token '<'" JSON.parse error.
    const text = await response.text().catch(() => '');
    const hint = text.startsWith('<!doctype') || text.startsWith('<html')
      ? ' (server returned an HTML page instead of JSON -- restart the backend so it registers this route)'
      : '';
    throw new Error(`HTTP ${response.status} from /api/${endpoint}${hint}`);
  }
}

export async function apiCall(endpoint, data) {
  const response = await fetch(`${API_BASE}/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  const body = await readJson(response, endpoint);
  if (!response.ok || body.status === 'error') {
    throw new Error(body.message || `HTTP ${response.status}`);
  }
  return body;
}

export async function apiGet(endpoint, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`${API_BASE}/${endpoint}${query ? `?${query}` : ''}`);
  const body = await readJson(response, endpoint);
  if (!response.ok || body.status === 'error') {
    const error = new Error(body.message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}
