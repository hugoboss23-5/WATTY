import React, { useState, useEffect } from 'react';
import { getMemories, deleteMemory } from '../utils/api.js';

const styles = {
  panel: {
    backgroundColor: '#12121e',
    borderRadius: '8px',
    border: '1px solid #2a2a3e',
    padding: '16px',
    marginBottom: '16px',
  },
  title: { fontSize: '16px', fontWeight: 'bold', marginBottom: '12px', color: '#22c55e' },
  search: {
    width: '100%',
    padding: '8px 12px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #2a2a3e',
    borderRadius: '6px',
    color: '#e0e0e0',
    marginBottom: '12px',
    fontSize: '13px',
  },
  item: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: '8px 0',
    borderBottom: '1px solid #1e1e2e',
    fontSize: '13px',
    color: '#ccc',
  },
  badge: {
    fontSize: '10px',
    padding: '2px 6px',
    borderRadius: '3px',
    backgroundColor: '#1e3a1e',
    color: '#22c55e',
    marginRight: '8px',
  },
  deleteBtn: {
    background: 'none',
    border: '1px solid #4a2020',
    color: '#ef4444',
    padding: '4px 8px',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    flexShrink: 0,
  },
  empty: { color: '#666', fontStyle: 'italic', fontSize: '13px' },
};

export default function MemoryPanel() {
  const [memories, setMemories] = useState([]);
  const [search, setSearch] = useState('');

  const load = async () => {
    try {
      const data = await getMemories(search);
      setMemories(data);
    } catch (err) {
      console.error('Failed to load memories:', err);
    }
  };

  useEffect(() => { load(); }, [search]);

  const handleDelete = async (id) => {
    try {
      await deleteMemory(id);
      load();
    } catch (err) {
      console.error('Failed to delete memory:', err);
    }
  };

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Memories</div>
      <input
        style={styles.search}
        placeholder="Search memories..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      {memories.length === 0 ? (
        <div style={styles.empty}>No memories yet</div>
      ) : (
        memories.map((m) => (
          <div key={m.id} style={styles.item}>
            <div>
              <span style={styles.badge}>{m.confidence}</span>
              {m.content}
            </div>
            <button style={styles.deleteBtn} onClick={() => handleDelete(m.id)}>
              archive
            </button>
          </div>
        ))
      )}
    </div>
  );
}
