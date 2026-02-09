import React, { useState, useEffect } from 'react';
import MemoryPanel from '../components/MemoryPanel.jsx';
import DirectiveCard from '../components/DirectiveCard.jsx';
import SkillCard from '../components/SkillCard.jsx';
import { getDirectives, getSkills, getState } from '../utils/api.js';

const styles = {
  container: { padding: '20px', overflowY: 'auto', height: '100%' },
  title: { fontSize: '22px', fontWeight: 'bold', marginBottom: '20px', color: '#e0e0e0' },
  section: { marginBottom: '24px' },
  sectionTitle: { fontSize: '16px', fontWeight: 'bold', marginBottom: '12px', color: '#888' },
  statePanel: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '16px',
    marginBottom: '16px',
  },
  stateItem: {
    padding: '6px 0',
    borderBottom: '1px solid #1e1e2e',
    fontSize: '13px',
    color: '#ccc',
    display: 'flex',
    justifyContent: 'space-between',
  },
  stateKey: { color: '#3b82f6', fontWeight: 'bold' },
  expiry: { fontSize: '11px', color: '#666' },
  empty: { color: '#555', fontStyle: 'italic', fontSize: '13px' },
  tabs: { display: 'flex', gap: '8px', marginBottom: '16px' },
  tab: (active) => ({
    padding: '6px 14px',
    borderRadius: '6px',
    border: active ? '1px solid #2563eb' : '1px solid #2a2a3e',
    backgroundColor: active ? '#1a2a4e' : 'transparent',
    color: active ? '#3b82f6' : '#888',
    fontSize: '12px',
    cursor: 'pointer',
  }),
};

export default function Dashboard() {
  const [directives, setDirectives] = useState([]);
  const [skills, setSkills] = useState([]);
  const [stateItems, setStateItems] = useState([]);
  const [directiveFilter, setDirectiveFilter] = useState('');

  const loadAll = async () => {
    try {
      const [d, s, st] = await Promise.all([
        getDirectives(directiveFilter ? { type: directiveFilter } : {}),
        getSkills(),
        getState(),
      ]);
      setDirectives(d);
      setSkills(s);
      setStateItems(st);
    } catch (err) {
      console.error('Dashboard load error:', err);
    }
  };

  useEffect(() => { loadAll(); }, [directiveFilter]);

  const directiveTypes = ['', 'SAFETY', 'EPISTEMICS', 'STYLE', 'WORKFLOW', 'REASONING'];

  return (
    <div style={styles.container}>
      <div style={styles.title}>Cognitive Dashboard</div>

      <div style={styles.section}>
        <MemoryPanel />
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Directives</div>
        <div style={styles.tabs}>
          {directiveTypes.map((t) => (
            <button
              key={t || 'all'}
              style={styles.tab(directiveFilter === t)}
              onClick={() => setDirectiveFilter(t)}
            >
              {t || 'All'}
            </button>
          ))}
        </div>
        {directives.length === 0 ? (
          <div style={styles.empty}>No directives</div>
        ) : (
          directives.map((d) => (
            <DirectiveCard key={d.id} directive={d} onUpdate={loadAll} />
          ))
        )}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Skills</div>
        {skills.length === 0 ? (
          <div style={styles.empty}>No skills</div>
        ) : (
          skills.map((s) => (
            <SkillCard key={s.id} skill={s} onUpdate={loadAll} />
          ))
        )}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Active State</div>
        <div style={styles.statePanel}>
          {stateItems.length === 0 ? (
            <div style={styles.empty}>No active state</div>
          ) : (
            stateItems.map((s) => (
              <div key={s.id} style={styles.stateItem}>
                <div>
                  <span style={styles.stateKey}>{s.key}</span>: {s.value}
                </div>
                <span style={styles.expiry}>
                  {s.expires_at
                    ? `Expires: ${new Date(s.expires_at).toLocaleString()}`
                    : 'No expiry'}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
