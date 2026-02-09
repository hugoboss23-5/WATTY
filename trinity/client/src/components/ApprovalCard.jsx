import React from 'react';
import { resolvePending } from '../utils/api.js';

const styles = {
  card: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #3a2a1e',
    padding: '14px 16px',
    marginBottom: '10px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  typeBadge: {
    padding: '2px 8px',
    borderRadius: '4px',
    backgroundColor: '#f59e0b',
    color: '#000',
    fontSize: '11px',
    fontWeight: 'bold',
  },
  date: { fontSize: '11px', color: '#666' },
  content: {
    fontSize: '13px',
    color: '#ccc',
    lineHeight: '1.5',
    padding: '8px',
    backgroundColor: '#1a1a2e',
    borderRadius: '6px',
    marginBottom: '8px',
    whiteSpace: 'pre-wrap',
  },
  evidence: { fontSize: '12px', color: '#888', marginBottom: '8px', fontStyle: 'italic' },
  actions: { display: 'flex', gap: '8px' },
  approveBtn: {
    padding: '6px 16px',
    borderRadius: '6px',
    border: '1px solid #22c55e',
    backgroundColor: 'transparent',
    color: '#22c55e',
    fontSize: '12px',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
  rejectBtn: {
    padding: '6px 16px',
    borderRadius: '6px',
    border: '1px solid #ef4444',
    backgroundColor: 'transparent',
    color: '#ef4444',
    fontSize: '12px',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
};

export default function ApprovalCard({ item, onResolve }) {
  const handleAction = async (action) => {
    try {
      await resolvePending(item.id, action);
      if (onResolve) onResolve();
    } catch (err) {
      console.error(`Failed to ${action}:`, err);
    }
  };

  let displayContent;
  try {
    const parsed = JSON.parse(item.proposed_content);
    displayContent = JSON.stringify(parsed, null, 2);
  } catch {
    displayContent = item.proposed_content;
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.typeBadge}>{item.change_type.toUpperCase()}</span>
        <span style={styles.date}>{new Date(item.created_at).toLocaleString()}</span>
      </div>
      <div style={styles.content}>{displayContent}</div>
      {item.evidence && <div style={styles.evidence}>{item.evidence}</div>}
      {item.status === 'pending' && (
        <div style={styles.actions}>
          <button style={styles.approveBtn} onClick={() => handleAction('approve')}>
            Approve
          </button>
          <button style={styles.rejectBtn} onClick={() => handleAction('reject')}>
            Reject
          </button>
        </div>
      )}
    </div>
  );
}
