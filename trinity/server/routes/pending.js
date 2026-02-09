import { Router } from 'express';
import { getDb } from '../db.js';

const router = Router();

// GET /api/pending — Approval queue
router.get('/', (req, res) => {
  const db = getDb();
  const { status = 'pending' } = req.query;
  const rows = db.prepare(
    'SELECT * FROM pending_changes WHERE status = ? ORDER BY created_at DESC'
  ).all(status);
  res.json(rows);
});

// PATCH /api/pending/:id — Approve / reject
router.patch('/:id', (req, res) => {
  const db = getDb();
  const { id } = req.params;
  const { action } = req.body; // 'approve' or 'reject'

  const pending = db.prepare('SELECT * FROM pending_changes WHERE id = ?').get(id);
  if (!pending) return res.status(404).json({ error: 'Pending change not found' });

  if (action === 'approve') {
    // Apply the change based on type
    const content = JSON.parse(pending.proposed_content);

    if (pending.change_type === 'directive' && content.id) {
      // Promote the directive from candidate to shadow
      db.prepare(
        "UPDATE directives SET status = 'shadow', promoted_at = CURRENT_TIMESTAMP, approved_by = 'human' WHERE id = ?"
      ).run(content.id);
      db.prepare(
        "INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by) VALUES ('directives', ?, 'promote', 'candidate', 'shadow', 'Human approved pending change', 'human')"
      ).run(content.id);
    } else if (pending.change_type === 'skill') {
      // Create the skill
      const result = db.prepare(
        "INSERT INTO skills (name, trigger_pattern, content, status, approved_by) VALUES (?, ?, ?, 'active', 'human')"
      ).run(content.name, content.trigger, content.content);
      db.prepare(
        "INSERT INTO changelog (table_name, record_id, action, new_value, reason, changed_by) VALUES ('skills', ?, 'create', ?, 'Human approved pending skill', 'human')"
      ).run(result.lastInsertRowid, JSON.stringify(content));
    }

    db.prepare(
      "UPDATE pending_changes SET status = 'approved', resolved_at = CURRENT_TIMESTAMP, resolved_by = 'human' WHERE id = ?"
    ).run(id);
  } else if (action === 'reject') {
    db.prepare(
      "UPDATE pending_changes SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP, resolved_by = 'human' WHERE id = ?"
    ).run(id);

    // If it was a directive, also reject the directive record
    if (pending.change_type === 'directive') {
      const content = JSON.parse(pending.proposed_content);
      if (content.id) {
        db.prepare("DELETE FROM directives WHERE id = ? AND status = 'candidate'").run(content.id);
      }
    }
  } else {
    return res.status(400).json({ error: "Action must be 'approve' or 'reject'" });
  }

  db.prepare(
    "INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by) VALUES ('pending_changes', ?, ?, 'pending', ?, ?, 'human')"
  ).run(id, action, action === 'approve' ? 'approved' : 'rejected', `Human ${action}d change`);

  const updated = db.prepare('SELECT * FROM pending_changes WHERE id = ?').get(id);
  res.json(updated);
});

export default router;
