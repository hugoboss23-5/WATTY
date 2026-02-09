import React from 'react';

const typeColors = {
  MEMORY: '#22c55e',
  STATE: '#3b82f6',
  DIRECTIVE: '#f59e0b',
  SKILL: '#a855f7',
  EVOLVE: '#ef4444',
  CRYSTAL: '#06b6d4',
};

const styles = {
  container: {
    margin: '8px 16px',
    padding: '10px 14px',
    borderRadius: '8px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #2a2a3e',
    fontSize: '13px',
  },
  badge: (type) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '4px',
    backgroundColor: typeColors[type] || '#666',
    color: '#fff',
    fontSize: '11px',
    fontWeight: 'bold',
    marginRight: '8px',
  }),
  status: (autoApproved) => ({
    color: autoApproved ? '#22c55e' : '#f59e0b',
    fontSize: '11px',
    marginLeft: '8px',
  }),
};

export default function SelfModNotification({ mod }) {
  return (
    <div style={styles.container}>
      <span style={styles.badge(mod.type)}>{mod.type}</span>
      <span style={{ color: '#ccc' }}>
        {mod.type === 'MEMORY' && 'New memory stored'}
        {mod.type === 'STATE' && 'State updated'}
        {mod.type === 'DIRECTIVE' && 'New directive proposed'}
        {mod.type === 'SKILL' && 'New skill proposed'}
        {mod.type === 'EVOLVE' && 'Evolution requested'}
        {mod.type === 'CRYSTAL' && 'Session crystal written'}
      </span>
      <span style={styles.status(mod.autoApproved)}>
        {mod.autoApproved ? 'auto-approved' : 'needs approval'}
      </span>
    </div>
  );
}
