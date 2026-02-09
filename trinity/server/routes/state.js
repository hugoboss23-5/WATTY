import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/state — List active state (auto-filters expired)
router.get('/', (req, res) => {
  const db = getDb();
  const now = new Date().toISOString();
  const rows = db.prepare(
    'SELECT * FROM state WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC'
  ).all(now);
  res.json(rows);
});

export default router;
