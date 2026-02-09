// Prompt Builder — Assembles system prompt from cognitive layers

import { getDb } from './db.js';
import { resolveConflictingDirectives } from './governance.js';

function getIdentityBlock() {
  const db = getDb();
  const rows = db.prepare('SELECT key, value FROM identity').all();
  if (rows.length === 0) return '';

  const lines = rows.map(r => `${r.key}: ${r.value}`);
  return `## IDENTITY\n${lines.join('\n')}`;
}

function getDirectivesBlock(type, heading) {
  const db = getDb();
  const rows = db.prepare(
    "SELECT * FROM directives WHERE type = ? AND status = 'active' ORDER BY precedence DESC"
  ).all(type);

  if (rows.length === 0) return '';

  const resolved = resolveConflictingDirectives(rows);
  const lines = resolved.map(r => `- [P${r.precedence}] ${r.content}`);
  return `## ${heading}\n${lines.join('\n')}`;
}

function getActiveSkills() {
  const db = getDb();
  const rows = db.prepare("SELECT * FROM skills WHERE status = 'active'").all();
  if (rows.length === 0) return '';

  const lines = rows.map(r => `- **${r.name}** (trigger: "${r.trigger_pattern}"): ${r.content}`);
  return `## ACTIVE SKILLS\n${lines.join('\n')}`;
}

function getRelevantMemories(limit = 20) {
  const db = getDb();
  const rows = db.prepare(
    "SELECT * FROM memories WHERE status = 'active' ORDER BY created_at DESC LIMIT ?"
  ).all(limit);

  if (rows.length === 0) return '';

  const lines = rows.map(r => `- [${r.confidence}] ${r.content}`);
  return `## RELEVANT MEMORIES\n${lines.join('\n')}`;
}

function getActiveState() {
  const db = getDb();
  const now = new Date().toISOString();
  const rows = db.prepare(
    'SELECT * FROM state WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC'
  ).all(now);

  if (rows.length === 0) return '';

  const lines = rows.map(r => {
    const expiry = r.expires_at ? ` (expires: ${r.expires_at})` : ' (no expiry)';
    return `- ${r.key}: ${r.value}${expiry}`;
  });
  return `## ACTIVE STATE\n${lines.join('\n')}`;
}

function getLastCrystal() {
  const db = getDb();
  const row = db.prepare(
    'SELECT crystal FROM sessions WHERE crystal IS NOT NULL ORDER BY id DESC LIMIT 1'
  ).get();

  if (!row) return '';
  return `## LAST SESSION CRYSTAL\n${row.crystal}`;
}

function getSelfModProtocol() {
  return `## SELF-MODIFICATION PROTOCOL
You can write the following commands in your responses to modify your own cognitive architecture:

[MEMORY: content]
  → Auto-approved. Stores a fact about the world.

[DIRECTIVE: type | scope | content | evidence]
  → Enters as 'candidate'. Human approval required to promote.
  → Types: SAFETY, EPISTEMICS, STYLE, WORKFLOW, REASONING
  → Scopes: global, or project-name

[SKILL: name | trigger | content]
  → Enters pending queue. Human must approve.

[STATE: key | value | ttl_hours]
  → Auto-approved. TTL enforced (default 24hr).

[EVOLVE: description | evidence]
  → Structural change request. Human reviews.

[CRYSTAL: summary]
  → End-of-session summary. Include: threads active, frameworks used, predictions, open questions, what worked.`;
}

export function buildSystemPrompt() {
  const blocks = [
    getIdentityBlock(),
    getDirectivesBlock('REASONING', 'METHODOLOGY'),
    getDirectivesBlock('EPISTEMICS', 'EPISTEMICS'),
    getActiveSkills(),
    getDirectivesBlock('STYLE', 'STYLE RULES'),
    getDirectivesBlock('WORKFLOW', 'WORKFLOW RULES'),
    getDirectivesBlock('SAFETY', 'SAFETY RULES'),
    getRelevantMemories(),
    getActiveState(),
    getLastCrystal(),
    getSelfModProtocol(),
  ].filter(Boolean);

  return blocks.join('\n\n');
}
