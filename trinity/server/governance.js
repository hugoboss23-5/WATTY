// Sunlight Governance Engine v0.1
// Applies rules for self-modification commands

import { getDb } from './db.js';

// Governance rules: which layers auto-approve vs need human approval
const GOVERNANCE_RULES = {
  MEMORY: { autoApprove: true, needsHuman: false },
  STATE: { autoApprove: true, needsHuman: false },
  DIRECTIVE: { autoApprove: false, needsHuman: true },
  SKILL: { autoApprove: false, needsHuman: true },
  IDENTITY: { autoApprove: false, needsHuman: true },
  EVOLVE: { autoApprove: false, needsHuman: true },
  CRYSTAL: { autoApprove: true, needsHuman: false },
};

function logChange(tableName, recordId, action, oldValue, newValue, reason, changedBy) {
  const db = getDb();
  db.prepare(`
    INSERT INTO changelog (table_name, record_id, action, old_value, new_value, reason, changed_by)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(tableName, recordId, action, oldValue, newValue, reason, changedBy);
}

export function applyCommand(command, sessionId) {
  const db = getDb();
  const rule = GOVERNANCE_RULES[command.type];
  const results = [];

  if (!rule) {
    return { applied: false, error: `Unknown command type: ${command.type}` };
  }

  switch (command.type) {
    case 'MEMORY': {
      const result = db.prepare(
        'INSERT INTO memories (content, source_session, confidence) VALUES (?, ?, ?)'
      ).run(command.data.content, sessionId, 'stated');
      logChange('memories', result.lastInsertRowid, 'create', null, command.data.content, 'AI self-mod: memory', 'ai');
      return { applied: true, autoApproved: true, id: result.lastInsertRowid, type: 'MEMORY' };
    }

    case 'STATE': {
      const expiresAt = command.data.ttlHours
        ? new Date(Date.now() + command.data.ttlHours * 60 * 60 * 1000).toISOString()
        : null;
      const result = db.prepare(
        'INSERT INTO state (key, value, expires_at, source_session) VALUES (?, ?, ?, ?)'
      ).run(command.data.key, command.data.value, expiresAt, sessionId);
      logChange('state', result.lastInsertRowid, 'create', null, JSON.stringify(command.data), 'AI self-mod: state', 'ai');
      return { applied: true, autoApproved: true, id: result.lastInsertRowid, type: 'STATE' };
    }

    case 'DIRECTIVE': {
      // Enters as 'candidate' — needs human to promote
      const result = db.prepare(
        "INSERT INTO directives (type, scope, content, status, source_session, evidence, approved_by) VALUES (?, ?, ?, 'candidate', ?, ?, NULL)"
      ).run(command.data.directiveType, command.data.scope, command.data.content, sessionId, command.data.evidence);

      // Also add to pending_changes queue
      db.prepare(
        "INSERT INTO pending_changes (change_type, target_table, proposed_content, evidence) VALUES ('directive', 'directives', ?, ?)"
      ).run(JSON.stringify({ id: result.lastInsertRowid, ...command.data }), command.data.evidence);

      logChange('directives', result.lastInsertRowid, 'create', null, command.data.content, 'AI proposed directive (candidate)', 'ai');
      return { applied: true, autoApproved: false, needsHuman: true, id: result.lastInsertRowid, type: 'DIRECTIVE' };
    }

    case 'SKILL': {
      // Enters pending_changes queue — human must approve
      const result = db.prepare(
        "INSERT INTO pending_changes (change_type, target_table, proposed_content, evidence) VALUES ('skill', 'skills', ?, ?)"
      ).run(JSON.stringify(command.data), `AI proposed skill: ${command.data.name}`);
      logChange('pending_changes', result.lastInsertRowid, 'create', null, JSON.stringify(command.data), 'AI proposed skill', 'ai');
      return { applied: true, autoApproved: false, needsHuman: true, id: result.lastInsertRowid, type: 'SKILL' };
    }

    case 'EVOLVE': {
      const result = db.prepare(
        "INSERT INTO pending_changes (change_type, target_table, proposed_content, evidence) VALUES ('evolve', 'system', ?, ?)"
      ).run(command.data.description, command.data.evidence);
      logChange('pending_changes', result.lastInsertRowid, 'create', null, command.data.description, 'AI proposed evolution', 'ai');
      return { applied: true, autoApproved: false, needsHuman: true, id: result.lastInsertRowid, type: 'EVOLVE' };
    }

    case 'CRYSTAL': {
      // Store crystal in current session
      if (sessionId) {
        db.prepare('UPDATE sessions SET crystal = ? WHERE id = ?').run(command.data.summary, sessionId);
        logChange('sessions', sessionId, 'update', null, command.data.summary, 'Session crystal written', 'ai');
      }
      return { applied: true, autoApproved: true, type: 'CRYSTAL' };
    }

    default:
      return { applied: false, error: `Unhandled command type: ${command.type}` };
  }
}

export function resolveConflictingDirectives(directives) {
  // Sort by precedence (higher wins), then by created_at (newer wins)
  return [...directives].sort((a, b) => {
    if (b.precedence !== a.precedence) return b.precedence - a.precedence;
    return new Date(b.created_at) - new Date(a.created_at);
  });
}
