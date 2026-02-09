import React from 'react';
import { updateDirective } from '../utils/api.js';

const typeColors = {
  SAFETY: '#ef4444',
  EPISTEMICS: '#a855f7',
  STYLE: '#3b82f6',
  WORKFLOW: '#f59e0b',
  REASONING: '#22c55e',
};

const statusColors = {
  active: '#22c55e',
  shadow: '#f59e0b',
  candidate: '#888',
};

const styles = {
  card: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '12px 16px',
    marginBottom: '8px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  typeBadge: (type) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '4px',
    backgroundColor: typeColors[type] || '#666',
    color: '#fff',
    fontSize: '11px',
    fontWeight: 'bold',
  }),
  statusBadge: (status) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '4px',
    border: `1px solid ${statusColors[status] || '#666'}`,
    color: statusColors[status] || '#666',
    fontSize: '11px',
  }),
  content: { fontSize: '13px', color: '#ccc', lineHeight: '1.5' },
  meta: { fontSize: '11px', color: '#666', marginTop: '8px' },
  actions: { display: 'flex', gap: '6px', marginTop: '8px' },
  btn: (color) => ({
    padding: '4px 10px',
    borderRadius: '4px',
    border: `1px solid ${color}`,
    backgroundColor: 'transparent',
    color,
    fontSize: '11px',
    cursor: 'pointer',
  }),
};

export default function DirectiveCard({ directive, onUpdate }) {
  const promote = async (newStatus) => {
    try {
      await updateDirective(directive.id, { status: newStatus });
      if (onUpdate) onUpdate();
    } catch (err) {
      console.error('Failed to update directive:', err);
    }
  };

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <span style={styles.typeBadge(directive.type)}>{directive.type}</span>
          <span style={styles.statusBadge(directive.status)}>{directive.status}</span>
          <span style={{ fontSize: '11px', color: '#666' }}>P{directive.precedence}</span>
        </div>
        <span style={{ fontSize: '11px', color: '#555' }}>{directive.scope}</span>
      </div>
      <div style={styles.content}>{directive.content}</div>
      {directive.evidence && (
        <div style={styles.meta}>Evidence: {directive.evidence}</div>
      )}
      <div style={styles.actions}>
        {directive.status === 'candidate' && (
          <button style={styles.btn('#f59e0b')} onClick={() => promote('shadow')}>
            Promote to Shadow
          </button>
        )}
        {directive.status === 'shadow' && (
          <button style={styles.btn('#22c55e')} onClick={() => promote('active')}>
            Promote to Active
          </button>
        )}
        {directive.status === 'active' && (
          <button style={styles.btn('#ef4444')} onClick={() => promote('candidate')}>
            Demote
          </button>
        )}
      </div>
    </div>
  );
}
