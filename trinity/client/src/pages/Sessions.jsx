import React, { useState, useEffect } from 'react';
import SessionCrystal from '../components/SessionCrystal.jsx';
import MessageBubble from '../components/MessageBubble.jsx';
import { getSessions, getSession } from '../utils/api.js';

const styles = {
  container: { padding: '20px', overflowY: 'auto', height: '100%' },
  title: { fontSize: '22px', fontWeight: 'bold', marginBottom: '20px', color: '#e0e0e0' },
  empty: { color: '#555', fontStyle: 'italic', fontSize: '13px' },
  back: {
    padding: '6px 14px',
    borderRadius: '6px',
    border: '1px solid #2a2a3e',
    backgroundColor: 'transparent',
    color: '#888',
    fontSize: '12px',
    cursor: 'pointer',
    marginBottom: '16px',
  },
  detail: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '16px',
  },
  detailHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '16px',
    fontSize: '14px',
    color: '#888',
  },
};

export default function Sessions() {
  const [sessions, setSessions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    getSessions().then(setSessions).catch(console.error);
  }, []);

  const openSession = async (id) => {
    try {
      const data = await getSession(id);
      setDetail(data);
      setSelected(id);
    } catch (err) {
      console.error('Failed to load session:', err);
    }
  };

  if (selected && detail) {
    return (
      <div style={styles.container}>
        <button style={styles.back} onClick={() => { setSelected(null); setDetail(null); }}>
          Back to Sessions
        </button>
        <div style={styles.title}>Session #{detail.id}</div>
        <div style={styles.detail}>
          <div style={styles.detailHeader}>
            <span>Model: {detail.model || 'N/A'}</span>
            <span>{new Date(detail.started_at).toLocaleString()}</span>
          </div>
          {detail.crystal && (
            <div style={{
              padding: '12px',
              backgroundColor: '#1a2a3e',
              borderRadius: '6px',
              marginBottom: '16px',
              fontSize: '13px',
              color: '#06b6d4',
              lineHeight: '1.5',
            }}>
              Crystal: {detail.crystal}
            </div>
          )}
          <div>
            {Array.isArray(detail.messages) && detail.messages.length > 0 ? (
              detail.messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} />
              ))
            ) : (
              <div style={styles.empty}>No messages in this session</div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.title}>Session History</div>
      {sessions.length === 0 ? (
        <div style={styles.empty}>No sessions yet. Start chatting to create one.</div>
      ) : (
        sessions.map((s) => (
          <SessionCrystal key={s.id} session={s} onClick={() => openSession(s.id)} />
        ))
      )}
    </div>
  );
}
