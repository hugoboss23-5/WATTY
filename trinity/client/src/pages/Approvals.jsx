import React, { useState, useEffect } from 'react';
import ApprovalCard from '../components/ApprovalCard.jsx';
import { getPending } from '../utils/api.js';

const styles = {
  container: { padding: '20px', overflowY: 'auto', height: '100%' },
  title: { fontSize: '22px', fontWeight: 'bold', marginBottom: '6px', color: '#e0e0e0' },
  subtitle: { fontSize: '13px', color: '#888', marginBottom: '20px' },
  empty: { color: '#555', fontStyle: 'italic', fontSize: '13px', padding: '40px 0', textAlign: 'center' },
  count: {
    display: 'inline-block',
    padding: '2px 10px',
    borderRadius: '12px',
    backgroundColor: '#3a2a1e',
    color: '#f59e0b',
    fontSize: '12px',
    fontWeight: 'bold',
    marginLeft: '10px',
  },
};

export default function Approvals() {
  const [pending, setPending] = useState([]);

  const load = async () => {
    try {
      const data = await getPending();
      setPending(data);
    } catch (err) {
      console.error('Failed to load pending:', err);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <div style={styles.container}>
      <div style={styles.title}>
        Approval Queue
        {pending.length > 0 && <span style={styles.count}>{pending.length}</span>}
      </div>
      <div style={styles.subtitle}>Sunlight Governance — Review AI-proposed changes</div>
      {pending.length === 0 ? (
        <div style={styles.empty}>No pending changes. The AI hasn't proposed anything yet.</div>
      ) : (
        pending.map((item) => (
          <ApprovalCard key={item.id} item={item} onResolve={load} />
        ))
      )}
    </div>
  );
}
