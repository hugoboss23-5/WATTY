import React, { useState, useEffect } from 'react';
import { getSettings, updateSettings, getChangelog } from '../utils/api.js';

const styles = {
  container: { padding: '20px', overflowY: 'auto', height: '100%' },
  title: { fontSize: '22px', fontWeight: 'bold', marginBottom: '20px', color: '#e0e0e0' },
  section: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '16px',
    marginBottom: '16px',
  },
  sectionTitle: { fontSize: '16px', fontWeight: 'bold', marginBottom: '12px', color: '#888' },
  field: { marginBottom: '12px' },
  label: { display: 'block', fontSize: '12px', color: '#888', marginBottom: '4px' },
  input: {
    width: '100%',
    padding: '8px 12px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #2a2a3e',
    borderRadius: '6px',
    color: '#e0e0e0',
    fontSize: '13px',
  },
  saveBtn: {
    padding: '8px 20px',
    backgroundColor: '#2563eb',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '13px',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginTop: '8px',
  },
  status: (has) => ({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: has ? '#22c55e' : '#ef4444',
    marginRight: '6px',
  }),
  changelogItem: {
    padding: '8px 0',
    borderBottom: '1px solid #1e1e2e',
    fontSize: '12px',
    color: '#ccc',
  },
  changelogMeta: { fontSize: '11px', color: '#666' },
  success: { color: '#22c55e', fontSize: '13px', marginTop: '8px' },
};

export default function Settings() {
  const [settings, setSettings] = useState(null);
  const [anthropicKey, setAnthropicKey] = useState('');
  const [openaiKey, setOpenaiKey] = useState('');
  const [saved, setSaved] = useState(false);
  const [changelog, setChangelog] = useState([]);

  useEffect(() => {
    getSettings().then(setSettings).catch(console.error);
    getChangelog().then(setChangelog).catch(console.error);
  }, []);

  const handleSave = async () => {
    try {
      const data = {};
      if (anthropicKey) data.anthropicKey = anthropicKey;
      if (openaiKey) data.openaiKey = openaiKey;
      const result = await updateSettings(data);
      setSettings(result);
      setAnthropicKey('');
      setOpenaiKey('');
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to save settings:', err);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.title}>Settings</div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>API Keys</div>
        <div style={{ marginBottom: '12px', fontSize: '13px' }}>
          <span style={styles.status(settings?.hasAnthropicKey)} />
          Anthropic (Claude): {settings?.hasAnthropicKey ? 'Configured' : 'Not set'}
        </div>
        <div style={{ marginBottom: '16px', fontSize: '13px' }}>
          <span style={styles.status(settings?.hasOpenaiKey)} />
          OpenAI (GPT): {settings?.hasOpenaiKey ? 'Configured' : 'Not set'}
        </div>

        <div style={styles.field}>
          <label style={styles.label}>Anthropic API Key</label>
          <input
            style={styles.input}
            type="password"
            placeholder="sk-ant-..."
            value={anthropicKey}
            onChange={(e) => setAnthropicKey(e.target.value)}
          />
        </div>
        <div style={styles.field}>
          <label style={styles.label}>OpenAI API Key</label>
          <input
            style={styles.input}
            type="password"
            placeholder="sk-..."
            value={openaiKey}
            onChange={(e) => setOpenaiKey(e.target.value)}
          />
        </div>
        <button style={styles.saveBtn} onClick={handleSave}>Save Keys</button>
        {saved && <div style={styles.success}>Saved successfully</div>}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Current Configuration</div>
        <div style={{ fontSize: '13px', color: '#ccc' }}>
          <div>Default Provider: {settings?.defaultProvider || 'anthropic'}</div>
          <div>Default Model: {settings?.defaultModel || 'claude-sonnet-4-20250514'}</div>
        </div>
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Changelog (Audit Trail)</div>
        {changelog.length === 0 ? (
          <div style={{ color: '#555', fontStyle: 'italic', fontSize: '13px' }}>
            No changes recorded yet
          </div>
        ) : (
          changelog.slice(0, 50).map((entry) => (
            <div key={entry.id} style={styles.changelogItem}>
              <div>
                <strong>{entry.action}</strong> on {entry.table_name} #{entry.record_id}
                {entry.reason && <span> — {entry.reason}</span>}
              </div>
              <div style={styles.changelogMeta}>
                By {entry.changed_by} at {new Date(entry.changed_at).toLocaleString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
