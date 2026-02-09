import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/skills — List skills
router.get('/', (req, res) => {
  const db = getDb();
  const { status } = req.query;
  let query = 'SELECT * FROM skills';
  const params = [];

  if (status) {
    query += ' WHERE status = ?';
    params.push(status);
  }

  query += ' ORDER BY name ASC';
  const rows = db.prepare(query).all(...params);
  res.json(rows);
});

// PATCH /api/skills/:id — Approve / disable
router.patch('/:id', (req, res) => {
  const db = getDb();
  const { id } = req.params;
  const { status } = req.body;

  const existing = db.prepare('SELECT * FROM skills WHERE id = ?').get(id);
  if (!existing) return res.status(404).json({ error: 'Skill not found' });

  if (!['active', 'disabled', 'pending'].includes(status)) {
    return res.status(400).json({ error: 'Invalid status' });
  }

  db.prepare("UPDATE skills SET status = ?, approved_by = 'human' WHERE id = ?").run(status, id);

  db.prepare(
    "INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by) VALUES ('skills', ?, 'update', ?, ?, ?, 'human')"
  ).run(id, existing.status, status, `Human set skill to ${status}`);

  const updated = db.prepare('SELECT * FROM skills WHERE id = ?').get(id);
  res.json(updated);
});

export default router;
