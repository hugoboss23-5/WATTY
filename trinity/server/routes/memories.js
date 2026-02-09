import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/memories — List memories
router.get('/', (req, res) => {
  const db = getDb();
  const { status = 'active', search } = req.query;

  let query = 'SELECT * FROM memories WHERE status = ?';
  const params = [status];

  if (search) {
    query += ' AND content LIKE ?';
    params.push(`%${search}%`);
  }

  query += ' ORDER BY created_at DESC';
  const rows = db.prepare(query).all(...params);
  res.json(rows);
});

// DELETE /api/memories/:id — Archive a memory
router.delete('/:id', (req, res) => {
  const db = getDb();
  const { id } = req.params;

  const existing = db.prepare('SELECT * FROM memories WHERE id = ?').get(id);
  if (!existing) return res.status(404).json({ error: 'Memory not found' });

  db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(id);

  db.prepare(
    "INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by) VALUES ('memories', ?, 'update', ?, 'archived', 'Human archived memory', 'human')"
  ).run(id, existing.status);

  res.json({ success: true });
});

export default router;
