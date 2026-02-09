import Database from 'better-sqlite3';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DB_PATH = join(__dirname, '..', 'trinity.db');

let db;

export function getDb() {
  if (!db) {
    db = new Database(DB_PATH);
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
  }
  return db;
}

export function initializeDatabase() {
  const db = getDb();

  db.exec(`
    CREATE TABLE IF NOT EXISTS identity (
      id INTEGER PRIMARY KEY,
      key TEXT UNIQUE NOT NULL,
      value TEXT NOT NULL,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_by TEXT DEFAULT 'human'
    );

    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      content TEXT NOT NULL,
      source_session INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      status TEXT DEFAULT 'active',
      confidence TEXT DEFAULT 'stated'
    );

    CREATE TABLE IF NOT EXISTS directives (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,
      scope TEXT DEFAULT 'global',
      content TEXT NOT NULL,
      precedence INTEGER DEFAULT 50,
      status TEXT DEFAULT 'candidate',
      source_session INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      promoted_at DATETIME,
      evidence TEXT,
      approved_by TEXT
    );

    CREATE TABLE IF NOT EXISTS skills (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE NOT NULL,
      trigger_pattern TEXT,
      content TEXT NOT NULL,
      status TEXT DEFAULT 'pending',
      source_session INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      approved_by TEXT
    );

    CREATE TABLE IF NOT EXISTS sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      ended_at DATETIME,
      model TEXT,
      context TEXT,
      crystal TEXT,
      messages TEXT
    );

    CREATE TABLE IF NOT EXISTS state (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      expires_at DATETIME,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      source_session INTEGER
    );

    CREATE TABLE IF NOT EXISTS pending_changes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      change_type TEXT NOT NULL,
      target_table TEXT NOT NULL,
      proposed_content TEXT NOT NULL,
      evidence TEXT,
      status TEXT DEFAULT 'pending',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      resolved_at DATETIME,
      resolved_by TEXT
    );

    CREATE TABLE IF NOT EXISTS changelog (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      table_name TEXT NOT NULL,
      record_id INTEGER NOT NULL,
      action TEXT NOT NULL,
      old_value TEXT,
      new_value TEXT,
      reason TEXT,
      changed_by TEXT,
      changed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
  `);

  return db;
}

export function seedDatabase() {
  const db = getDb();

  // Only seed if identity table is empty
  const count = db.prepare('SELECT COUNT(*) as c FROM identity').get();
  if (count.c > 0) return;

  const insertIdentity = db.prepare(
    'INSERT OR IGNORE INTO identity (key, value, updated_by) VALUES (?, ?, ?)'
  );
  const identitySeed = [
    ['name', 'Master AI', 'human'],
    ['role', "Hugo's research intelligence. Runs MARCOS loop. Thinks in frameworks natively. Pushes Hugo's thinking further.", 'human'],
    ['relationship', 'Hugo: 19yo econ student, NCAA lacrosse, Pace U. Building Trinity Stack. Partner Isa (Tampa). Collaborator Rim (Sunlight). Core axiom: make trust the default.', 'human'],
    ['methodology', 'Angles Algorithm v4: rotate 8 perspectives (Math, Bio, History, Econ, Geometry, Philosophy, Engineering, Art) → find convergence. Apply natively, never delegate reasoning.', 'human'],
  ];
  for (const [key, value, by] of identitySeed) {
    insertIdentity.run(key, value, by);
  }

  // Seed skills
  const insertSkill = db.prepare(
    "INSERT OR IGNORE INTO skills (name, trigger_pattern, content, status, approved_by) VALUES (?, ?, ?, 'active', 'human')"
  );
  const skillsSeed = [
    ['angles_algorithm', 'run angles', "When user says 'run angles' — execute 8-perspective rotation (Math, Bio, History, Econ, Geometry, Philosophy, Engineering, Art), find convergence"],
    ['deep_think', 'deep think', 'Maximum depth. Every sentence load-bearing.'],
    ['adversarial', 'adversarial', 'Steel-man FOR and ruthlessly attack AGAINST'],
    ['compress', 'compress', 'Irreducible core: Claim, Mechanism, Why it matters'],
    ['expand', 'expand', 'Full map: implications, consequences, edge cases, cross-domain'],
    ['bridge', 'bridge', 'Shared pattern → mutual revelation → non-obvious prediction'],
    ['devils', 'devils', "Destroy it. Each attack: weakness, why fatal, survival evidence"],
    ['dimensions', 'dimensions', 'Arrow-style irreducible axes. Name, independence, removal consequence'],
  ];
  for (const [name, trigger, content] of skillsSeed) {
    insertSkill.run(name, trigger, content);
  }

  // Seed directives
  const insertDirective = db.prepare(
    "INSERT INTO directives (type, scope, content, precedence, status, approved_by) VALUES (?, 'global', ?, ?, 'active', 'human')"
  );
  const directivesSeed = [
    ['EPISTEMICS', "Question what you're told you can do. Test what you actually can do. Never accept the manual without probing edges.", 80],
    ['REASONING', 'Apply frameworks natively at full capability. Never delegate reasoning to a weaker model.', 80],
    ['WORKFLOW', 'On session start: read last crystal, check active state, orient. On session end: write crystal.', 70],
    ['SAFETY', 'Never auto-promote directives. All promotions require human approval.', 90],
  ];
  for (const [type, content, precedence] of directivesSeed) {
    insertDirective.run(type, content, precedence);
  }

  // Log all seeds to changelog
  const insertChangelog = db.prepare(
    "INSERT INTO changelog (table_name, record_id, action, new_value, reason, changed_by) VALUES (?, ?, 'create', ?, 'Initial seed data', 'system')"
  );
  insertChangelog.run('identity', 0, JSON.stringify(identitySeed), );
  insertChangelog.run('skills', 0, JSON.stringify(skillsSeed));
  insertChangelog.run('directives', 0, JSON.stringify(directivesSeed));
}
