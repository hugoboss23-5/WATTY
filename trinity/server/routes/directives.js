import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/directives — List directives (filterable by type, scope, status)
router.get('/', (req, res) => {
  const db = getDb();
  const { type, scope, status } = req.query;

  let query = 'SELECT * FROM directives WHERE 1=1';
  const params = [];

  if (type) { query += ' AND type = ?'; params.push(type); }
  if (scope) { query += ' AND scope = ?'; params.push(scope); }
  if (status) { query += ' AND status = ?'; params.push(status); }

  query += ' ORDER BY precedence DESC, created_at DESC';
  const rows = db.prepare(query).all(...params);
  res.json(rows);
});

// PATCH /api/directives/:id — Promote / demote / edit
router.patch('/:id', (req, res) => {
  const db = getDb();
  const { id } = req.params;
  const { status, content, precedence } = req.body;

  const existing = db.prepare('SELECT * FROM directives WHERE id = ?').get(id);
  if (!existing) return res.status(404).json({ error: 'Directive not found' });

  const updates = [];
  const params = [];

  if (status) {
    // Validate promotion ladder: candidate → shadow → active
    const validTransitions = {
      candidate: ['shadow', 'active'],
      shadow: ['active', 'candidate'],
      active: ['candidate'],
    };
    if (validTransitions[existing.status] && !validTransitions[existing.status].includes(status)) {
      return res.status(400).json({ error: `Cannot transition from ${existing.status} to ${status}` });
    }
    updates.push('status = ?');
    params.push(status);
    if (status === 'active' || status === 'shadow') {
      updates.push('promoted_at = CURRENT_TIMESTAMP');
      updates.push("approved_by = 'human'");
    }
  }
  if (content) { updates.push('content = ?'); params.push(content); }
  if (precedence !== undefined) { updates.push('precedence = ?'); params.push(precedence); }

  if (updates.length === 0) {
    return res.status(400).json({ error: 'No updates provided' });
  }

  params.push(id);
  db.prepare(`UPDATE directives SET ${updates.join(', ')} WHERE id = ?`).run(...params);

  const action = status ? 'promote' : 'update';
  db.prepare(
    "INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by) VALUES ('directives', ?, ?, ?, ?, ?, 'human')"
  ).run(id, action, JSON.stringify(existing), JSON.stringify(req.body), `Human ${action}d directive`);

  const updated = db.prepare('SELECT * FROM directives WHERE id = ?').get(id);
  res.json(updated);
});

export default router;
