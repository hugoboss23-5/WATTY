import React from 'react';

const styles = {
  container: (isUser) => ({
    display: 'flex',
    justifyContent: isUser ? 'flex-end' : 'flex-start',
    marginBottom: '12px',
    padding: '0 16px',
  }),
  bubble: (isUser) => ({
    maxWidth: '75%',
    padding: '12px 16px',
    borderRadius: '12px',
    backgroundColor: isUser ? '#2563eb' : '#1e1e2e',
    color: '#e0e0e0',
    fontSize: '14px',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    border: isUser ? 'none' : '1px solid #2a2a3e',
  }),
  label: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '4px',
    padding: '0 4px',
  },
};

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  return (
    <div style={styles.container(isUser)}>
      <div>
        <div style={{ ...styles.label, textAlign: isUser ? 'right' : 'left' }}>
          {isUser ? 'You' : 'AI'}
        </div>
        <div style={styles.bubble(isUser)}>{message.content}</div>
      </div>
    </div>
  );
}
