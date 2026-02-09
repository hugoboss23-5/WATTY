import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from '../components/MessageBubble.jsx';
import SelfModNotification from '../components/SelfModNotification.jsx';
import { sendMessage, getSettings } from '../utils/api.js';

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
  },
  header: {
    padding: '12px 20px',
    borderBottom: '1px solid #1e1e2e',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: { fontSize: '18px', fontWeight: 'bold', color: '#e0e0e0' },
  controls: { display: 'flex', gap: '10px', alignItems: 'center' },
  select: {
    padding: '6px 10px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #2a2a3e',
    borderRadius: '6px',
    color: '#e0e0e0',
    fontSize: '13px',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '16px 0',
  },
  inputArea: {
    padding: '12px 16px',
    borderTop: '1px solid #1e1e2e',
    display: 'flex',
    gap: '10px',
    alignItems: 'flex-end',
  },
  textarea: {
    flex: 1,
    padding: '10px 14px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #2a2a3e',
    borderRadius: '8px',
    color: '#e0e0e0',
    fontSize: '14px',
    resize: 'none',
    fontFamily: 'inherit',
    minHeight: '44px',
    maxHeight: '160px',
  },
  sendBtn: {
    padding: '10px 20px',
    backgroundColor: '#2563eb',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
  sendBtnDisabled: {
    padding: '10px 20px',
    backgroundColor: '#1e3a5f',
    border: 'none',
    borderRadius: '8px',
    color: '#666',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'not-allowed',
  },
  noKey: {
    padding: '20px',
    textAlign: 'center',
    color: '#f59e0b',
    fontSize: '14px',
  },
};

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [selfMods, setSelfMods] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [provider, setProvider] = useState('anthropic');
  const [model, setModel] = useState('');
  const [settings, setSettings] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    getSettings().then(setSettings).catch(console.error);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, selfMods]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendMessage(text, sessionId, provider, model || undefined);
      setSessionId(res.sessionId);
      setMessages((prev) => [...prev, { role: 'assistant', content: res.response }]);
      if (res.selfMods?.length) {
        setSelfMods((prev) => [...prev, ...res.selfMods]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const newSession = () => {
    setMessages([]);
    setSelfMods([]);
    setSessionId(null);
  };

  const hasKey = settings && (
    (provider === 'anthropic' && settings.hasAnthropicKey) ||
    (provider === 'openai' && settings.hasOpenaiKey)
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>Trinity Chat</div>
        <div style={styles.controls}>
          <select
            style={styles.select}
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
          >
            <option value="anthropic">Claude</option>
            <option value="openai">GPT</option>
          </select>
          <input
            style={{ ...styles.select, width: '180px' }}
            placeholder="Model (default)"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
          <button
            style={{ ...styles.select, cursor: 'pointer', border: '1px solid #3a3a4e' }}
            onClick={newSession}
          >
            New Session
          </button>
        </div>
      </div>

      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#555' }}>
            <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
              Trinity v0.1
            </div>
            <div>The Self-Evolving AI Interface</div>
            {!hasKey && settings && (
              <div style={styles.noKey}>
                No API key configured for {provider}. Go to Settings to add one.
              </div>
            )}
          </div>
        )}
        {messages.map((msg, i) => (
          <React.Fragment key={i}>
            <MessageBubble message={msg} />
            {msg.role === 'assistant' &&
              selfMods
                .filter((_, j) => j === i - Math.floor(i / 2))
                .map((mod, j) => <SelfModNotification key={`mod-${j}`} mod={mod} />)}
          </React.Fragment>
        ))}
        {loading && (
          <div style={{ padding: '12px 32px', color: '#888', fontSize: '13px' }}>
            Thinking...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputArea}>
        <textarea
          style={styles.textarea}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Talk to your AI..."
          disabled={loading}
        />
        <button
          style={loading ? styles.sendBtnDisabled : styles.sendBtn}
          onClick={handleSend}
          disabled={loading}
        >
          Send
        </button>
      </div>
    </div>
  );
}
