import { Router } from 'express';
import { getDb } from '../db.js';
import { callAI } from '../ai.js';
import { parseSelfModCommands, stripCommands } from '../parser.js';
import { applyCommand } from '../governance.js';
import { buildSystemPrompt } from '../prompt-builder.js';

const router = Router();

// POST /api/chat — Send message, get AI response, parse self-mods
router.post('/', async (req, res) => {
  const { message, sessionId, provider = 'anthropic', model } = req.body;
  const db = getDb();

  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  // Get or create session
  let currentSessionId = sessionId;
  if (!currentSessionId) {
    const result = db.prepare(
      "INSERT INTO sessions (model, messages) VALUES (?, ?)"
    ).run(model || provider, JSON.stringify([]));
    currentSessionId = result.lastInsertRowid;
  }

  // Load existing messages for this session
  const session = db.prepare('SELECT * FROM sessions WHERE id = ?').get(currentSessionId);
  let messages = [];
  try {
    messages = session?.messages ? JSON.parse(session.messages) : [];
  } catch {
    messages = [];
  }

  // Add user message
  messages.push({ role: 'user', content: message });

  // Build system prompt from cognitive layers
  const systemPrompt = buildSystemPrompt();

  // Get API key from settings-like approach (env vars or stored)
  const apiKey = provider === 'anthropic'
    ? process.env.ANTHROPIC_API_KEY
    : process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return res.status(400).json({
      error: `No API key configured for ${provider}. Set ${provider === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY'} environment variable or configure in Settings.`,
    });
  }

  try {
    // Call AI
    const aiResponse = await callAI(systemPrompt, messages, provider, apiKey, model);

    // Parse self-mod commands from AI response
    const commands = parseSelfModCommands(aiResponse);
    const commandResults = commands.map(cmd => applyCommand(cmd, currentSessionId));

    // Strip commands from displayed response
    const cleanResponse = stripCommands(aiResponse);

    // Add assistant message
    messages.push({ role: 'assistant', content: aiResponse });

    // Update session
    db.prepare('UPDATE sessions SET messages = ?, model = ? WHERE id = ?')
      .run(JSON.stringify(messages), model || provider, currentSessionId);

    res.json({
      response: cleanResponse,
      rawResponse: aiResponse,
      sessionId: currentSessionId,
      selfMods: commandResults.filter(r => r.applied),
      pendingApprovals: commandResults.filter(r => r.needsHuman),
    });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

export default router;
