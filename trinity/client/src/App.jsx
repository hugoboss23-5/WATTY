import React, { useState } from 'react';
import Chat from './pages/Chat.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Sessions from './pages/Sessions.jsx';
import Approvals from './pages/Approvals.jsx';
import Settings from './pages/Settings.jsx';

const NAV_ITEMS = [
  { id: 'chat', label: 'Chat' },
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'sessions', label: 'Sessions' },
  { id: 'approvals', label: 'Approvals' },
  { id: 'settings', label: 'Settings' },
];

const styles = {
  layout: {
    display: 'flex',
    height: '100vh',
    backgroundColor: '#0a0a0f',
  },
  sidebar: {
    width: '200px',
    borderRight: '1px solid #1e1e2e',
    display: 'flex',
    flexDirection: 'column',
    padding: '16px 0',
    flexShrink: 0,
  },
  logo: {
    padding: '8px 20px 24px',
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#2563eb',
    letterSpacing: '2px',
  },
  version: {
    fontSize: '11px',
    color: '#555',
    fontWeight: 'normal',
    letterSpacing: '0',
  },
  navItem: (active) => ({
    padding: '10px 20px',
    fontSize: '14px',
    color: active ? '#2563eb' : '#888',
    backgroundColor: active ? '#0d1a2e' : 'transparent',
    borderLeft: active ? '3px solid #2563eb' : '3px solid transparent',
    cursor: 'pointer',
    transition: 'all 0.15s',
  }),
  content: {
    flex: 1,
    overflow: 'hidden',
  },
};

export default function App() {
  const [page, setPage] = useState('chat');

  const renderPage = () => {
    switch (page) {
      case 'chat': return <Chat />;
      case 'dashboard': return <Dashboard />;
      case 'sessions': return <Sessions />;
      case 'approvals': return <Approvals />;
      case 'settings': return <Settings />;
      default: return <Chat />;
    }
  };

  return (
    <div style={styles.layout}>
      <nav style={styles.sidebar}>
        <div style={styles.logo}>
          TRINITY <span style={styles.version}>v0.1</span>
        </div>
        {NAV_ITEMS.map((item) => (
          <div
            key={item.id}
            style={styles.navItem(page === item.id)}
            onClick={() => setPage(item.id)}
          >
            {item.label}
          </div>
        ))}
      </nav>
      <main style={styles.content}>
        {renderPage()}
      </main>
    </div>
  );
}
