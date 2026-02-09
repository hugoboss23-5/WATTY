import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// POST /api/settings — Update API keys, preferences
router.post('/', (req, res) => {
  const { anthropicKey, openaiKey, defaultModel, defaultProvider } = req.body;

  // Store settings in environment (in-memory for this session)
  // In production, these would be stored securely
  if (anthropicKey) process.env.ANTHROPIC_API_KEY = anthropicKey;
  if (openaiKey) process.env.OPENAI_API_KEY = openaiKey;
  if (defaultModel) process.env.DEFAULT_MODEL = defaultModel;
  if (defaultProvider) process.env.DEFAULT_PROVIDER = defaultProvider;

  res.json({
    success: true,
    hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
    hasOpenaiKey: !!process.env.OPENAI_API_KEY,
    defaultModel: process.env.DEFAULT_MODEL || 'claude-sonnet-4-20250514',
    defaultProvider: process.env.DEFAULT_PROVIDER || 'anthropic',
  });
});

// GET /api/settings — Get current settings (no keys exposed)
router.get('/', (req, res) => {
  res.json({
    hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
    hasOpenaiKey: !!process.env.OPENAI_API_KEY,
    defaultModel: process.env.DEFAULT_MODEL || 'claude-sonnet-4-20250514',
    defaultProvider: process.env.DEFAULT_PROVIDER || 'anthropic',
  });
});

// GET /api/changelog — Full audit trail
router.get('/changelog', (req, res) => {
  const db = getDb();
  const rows = db.prepare('SELECT * FROM changelog ORDER BY changed_at DESC LIMIT 100').all();
  res.json(rows);
});

export default router;
