import React from 'react';
import { updateSkill } from '../utils/api.js';

const statusColors = {
  active: '#22c55e',
  pending: '#f59e0b',
  disabled: '#ef4444',
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
    marginBottom: '6px',
  },
  name: { fontWeight: 'bold', fontSize: '14px', color: '#a855f7' },
  statusBadge: (status) => ({
    padding: '2px 8px',
    borderRadius: '4px',
    border: `1px solid ${statusColors[status] || '#666'}`,
    color: statusColors[status] || '#666',
    fontSize: '11px',
  }),
  trigger: { fontSize: '12px', color: '#888', marginBottom: '4px' },
  content: { fontSize: '13px', color: '#ccc', lineHeight: '1.5' },
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

export default function SkillCard({ skill, onUpdate }) {
  const setStatus = async (status) => {
    try {
      await updateSkill(skill.id, { status });
      if (onUpdate) onUpdate();
    } catch (err) {
      console.error('Failed to update skill:', err);
    }
  };

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.name}>{skill.name}</span>
        <span style={styles.statusBadge(skill.status)}>{skill.status}</span>
      </div>
      {skill.trigger_pattern && (
        <div style={styles.trigger}>Trigger: "{skill.trigger_pattern}"</div>
      )}
      <div style={styles.content}>{skill.content}</div>
      <div style={styles.actions}>
        {skill.status !== 'active' && (
          <button style={styles.btn('#22c55e')} onClick={() => setStatus('active')}>
            Activate
          </button>
        )}
        {skill.status !== 'disabled' && (
          <button style={styles.btn('#ef4444')} onClick={() => setStatus('disabled')}>
            Disable
          </button>
        )}
      </div>
    </div>
  );
}
