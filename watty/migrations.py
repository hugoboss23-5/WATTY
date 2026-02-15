"""
Watty Schema Migrations
========================
Ordered list of versioned migrations. Each runs in a transaction.
If one fails, it rolls back and Watty refuses to start — no silent corruption.

To add a migration:
    1. Add a new entry to MIGRATIONS with the next version number
    2. Each entry is a list of SQL statements
    3. All statements in a version run in one transaction
"""

from collections import OrderedDict
from datetime import datetime, timezone


MIGRATIONS = OrderedDict()

# Version 1: core indexes on existing tables.
# Existing databases already have these (created by old _init_db),
# but IF NOT EXISTS makes this idempotent.
MIGRATIONS[1] = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type)",
    "CREATE INDEX IF NOT EXISTS idx_scan_log_hash ON scan_log(file_hash)",
]

# Version 2: provider index for fast provider-filtered recalls.
MIGRATIONS[2] = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_provider ON chunks(provider)",
]

LATEST_VERSION = max(MIGRATIONS.keys())


def get_version(conn) -> int:
    """Get current schema version. Returns 0 if no version table."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except Exception:
        return 0


def run_migrations(conn):
    """Run all pending migrations sequentially. Each in its own transaction."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    conn.commit()

    current = get_version(conn)

    for version, statements in MIGRATIONS.items():
        if version <= current:
            continue
        try:
            for stmt in statements:
                conn.execute(stmt)
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, now),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(
                f"Schema migration to version {version} failed: {e}\n"
                f"Database may need manual repair. Back up ~/.watty/brain.db before retrying."
            ) from e
