"""
Tests for schema migration system.
"""

import sys
import types
import os
import tempfile
import sqlite3
import numpy as np

if "sentence_transformers" not in sys.modules:
    _mock_st = types.ModuleType("sentence_transformers")
    class _MockModel:
        def encode(self, text, **kwargs):
            words = text.lower().split()
            vec = np.zeros(384, dtype=np.float32)
            for w in words:
                np.random.seed(hash(w) % (2**31))
                vec += np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec
    _mock_st.SentenceTransformer = lambda *a, **k: _MockModel()
    sys.modules["sentence_transformers"] = _mock_st

from watty.brain import Brain  # noqa: E402
from watty.migrations import get_version, run_migrations, LATEST_VERSION  # noqa: E402


def _fresh():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    return db, tmp


def test_fresh_db_at_latest_version():
    """A new database should be at the latest schema version."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    assert get_version(conn) == LATEST_VERSION
    conn.close()


def test_v1_db_migrates_without_data_loss():
    """A v1 database (no schema_version table) migrates cleanly."""
    db, _ = _fresh()
    # Simulate a v1 database: create tables manually without schema_version
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            conversation_id TEXT,
            created_at TEXT NOT NULL,
            metadata TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding BLOB,
            created_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            source_type TEXT DEFAULT 'conversation',
            source_path TEXT
        );
        CREATE TABLE clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            centroid BLOB,
            chunk_ids TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            scanned_at TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0
        );
    """)
    # Insert some data
    conn.execute(
        "INSERT INTO conversations (provider, created_at) VALUES (?, ?)",
        ("manual", "2025-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO chunks (conversation_id, role, content, chunk_index, created_at, provider, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "user", "Test migration data", 0, "2025-01-01T00:00:00+00:00", "manual", "abc123"),
    )
    conn.commit()
    conn.close()

    # Now open with Brain — migrations should run
    brain = Brain(db_path=db)
    s = brain.stats()
    assert s["total_memories"] == 1

    # Version should be latest
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    assert get_version(conn) == LATEST_VERSION
    conn.close()


def test_failed_migration_rolls_back():
    """A failed migration should rollback and raise clearly."""
    db, _ = _fresh()
    # Create a real brain first so all existing migrations pass
    brain = Brain(db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    current = get_version(conn)
    assert current == LATEST_VERSION

    # Inject a bad migration at version 999
    from watty import migrations
    original = dict(migrations.MIGRATIONS)
    try:
        migrations.MIGRATIONS[999] = [
            "CREATE TABLE this_works (id INTEGER PRIMARY KEY)",
            "ALTER TABLE nonexistent_table ADD COLUMN bad TEXT",  # will fail
        ]
        raised = False
        try:
            run_migrations(conn)
        except RuntimeError as e:
            raised = True
            assert "version 999 failed" in str(e)
        assert raised, "Expected RuntimeError on failed migration"

        # Version should still be at LATEST_VERSION, not 999
        assert get_version(conn) == LATEST_VERSION
    finally:
        migrations.MIGRATIONS.clear()
        migrations.MIGRATIONS.update(original)
    conn.close()


def test_version_table_accurate():
    """schema_version table has correct entries after migration."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT version, applied_at FROM schema_version ORDER BY version").fetchall()
    assert len(rows) == LATEST_VERSION
    for i, row in enumerate(rows, 1):
        assert row["version"] == i
        assert row["applied_at"] is not None
    conn.close()


def test_migrations_idempotent():
    """Running migrations twice doesn't break anything."""
    db, _ = _fresh()
    brain = Brain(db_path=db)
    brain.store_memory("Before second migration run")
    # Open again — triggers _init_db + migrations again
    brain2 = Brain(db_path=db)
    assert brain2.stats()["total_memories"] == 1
