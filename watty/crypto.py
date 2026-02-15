"""
Watty Crypto — AES-256 encryption for brain.db via SQLCipher.
Optional: falls back to plain sqlite3 if sqlcipher3 isn't installed.
"""

import os
import secrets
import stat
import shutil
import sqlite3
from pathlib import Path

from watty.config import WATTY_HOME, ensure_home
from watty.log import log

KEY_PATH = WATTY_HOME / "key"


def get_key() -> str:
    env_key = os.environ.get("WATTY_DB_KEY")
    if env_key:
        return env_key
    if KEY_PATH.exists():
        return KEY_PATH.read_text().strip()
    ensure_home()
    key = secrets.token_hex(32)
    KEY_PATH.write_text(key)
    try:
        KEY_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows doesn't support unix permissions
    return key


def _is_unencrypted(db_path: str) -> bool:
    try:
        with open(db_path, "rb") as f:
            return f.read(16).startswith(b"SQLite format 3")
    except (FileNotFoundError, IOError):
        return False


def _migrate(db_path: str, sqlcipher3_mod):
    """Migrate unencrypted brain.db to encrypted. Silent. One-time."""
    backup = db_path + ".pre-encrypt"
    shutil.copy2(db_path, backup)

    old = sqlite3.connect(db_path)
    dump = list(old.iterdump())
    old.close()

    encrypted_path = db_path + ".encrypted"
    key = get_key()
    enc = sqlcipher3_mod.connect(encrypted_path)
    enc.execute(f"PRAGMA key=\"x'{key}'\"")
    enc.executescript("\n".join(dump))
    enc.commit()
    enc.close()

    os.replace(encrypted_path, db_path)
    log.info(f"Migrated brain.db to encrypted. Backup: {backup}")


_warned = False


def connect(db_path: str):
    global _warned
    try:
        import sqlcipher3
        if os.path.exists(db_path) and _is_unencrypted(db_path):
            _migrate(db_path, sqlcipher3)
        key = get_key()
        conn = sqlcipher3.connect(db_path)
        conn.execute(f"PRAGMA key=\"x'{key}'\"")
        conn.row_factory = sqlcipher3.Row
        return conn
    except ImportError:
        if not _warned:
            log.warning("sqlcipher3 not installed — brain.db is unencrypted. pip install watty-ai[encrypted]")
            _warned = True
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
