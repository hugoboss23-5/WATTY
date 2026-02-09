import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/sessions — List past sessions with crystals
router.get('/', (req, res) => {
  const db = getDb();
  const rows = db.prepare(
    'SELECT id, started_at, ended_at, model, context, crystal FROM sessions ORDER BY id DESC'
  ).all();
  res.json(rows);
});

// GET /api/sessions/:id — Full session with messages
router.get('/:id', (req, res) => {
  const db = getDb();
  const { id } = req.params;
  const session = db.prepare('SELECT * FROM sessions WHERE id = ?').get(id);
  if (!session) return res.status(404).json({ error: 'Session not found' });

  try {
    session.messages = JSON.parse(session.messages || '[]');
  } catch {
    session.messages = [];
  }
  res.json(session);
});

export default router;
