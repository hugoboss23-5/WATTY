const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || 'Request failed');
  }
  return res.json();
}

// Chat
export const sendMessage = (message, sessionId, provider, model) =>
  request('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, sessionId, provider, model }),
  });

// Memories
export const getMemories = (search) =>
  request(`/memories${search ? `?search=${encodeURIComponent(search)}` : ''}`);
export const deleteMemory = (id) =>
  request(`/memories/${id}`, { method: 'DELETE' });

// Directives
export const getDirectives = (filters = {}) => {
  const params = new URLSearchParams(filters).toString();
  return request(`/directives${params ? `?${params}` : ''}`);
};
export const updateDirective = (id, data) =>
  request(`/directives/${id}`, { method: 'PATCH', body: JSON.stringify(data) });

// Skills
export const getSkills = (status) =>
  request(`/skills${status ? `?status=${status}` : ''}`);
export const updateSkill = (id, data) =>
  request(`/skills/${id}`, { method: 'PATCH', body: JSON.stringify(data) });

// State
export const getState = () => request('/state');

// Sessions
export const getSessions = () => request('/sessions');
export const getSession = (id) => request(`/sessions/${id}`);

// Pending (Approval Queue)
export const getPending = () => request('/pending');
export const resolvePending = (id, action) =>
  request(`/pending/${id}`, { method: 'PATCH', body: JSON.stringify({ action }) });

// Settings
export const getSettings = () => request('/settings');
export const updateSettings = (data) =>
  request('/settings', { method: 'POST', body: JSON.stringify(data) });

// Changelog
export const getChangelog = () => request('/settings/changelog');
