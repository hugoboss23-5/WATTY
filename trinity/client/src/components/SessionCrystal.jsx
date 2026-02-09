import React from 'react';

const styles = {
  card: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '14px 16px',
    marginBottom: '10px',
    cursor: 'pointer',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px',
  },
  id: { fontSize: '14px', fontWeight: 'bold', color: '#06b6d4' },
  model: { fontSize: '12px', color: '#888' },
  date: { fontSize: '11px', color: '#666' },
  crystal: {
    fontSize: '13px',
    color: '#ccc',
    lineHeight: '1.5',
    padding: '8px',
    backgroundColor: '#1a1a2e',
    borderRadius: '6px',
    whiteSpace: 'pre-wrap',
  },
  noCrystal: { color: '#555', fontSize: '12px', fontStyle: 'italic' },
};

export default function SessionCrystal({ session, onClick }) {
  return (
    <div style={styles.card} onClick={onClick}>
      <div style={styles.header}>
        <div>
          <span style={styles.id}>Session #{session.id}</span>
          {session.model && <span style={styles.model}> ({session.model})</span>}
        </div>
        <span style={styles.date}>{new Date(session.started_at).toLocaleString()}</span>
      </div>
      {session.context && (
        <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>
          Context: {session.context}
        </div>
      )}
      {session.crystal ? (
        <div style={styles.crystal}>{session.crystal}</div>
      ) : (
        <div style={styles.noCrystal}>No crystal written</div>
      )}
    </div>
  );
}
