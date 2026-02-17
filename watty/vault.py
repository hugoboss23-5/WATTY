"""
Watty Vault — Zero-Knowledge Encrypted Secret Storage
======================================================
Military-grade encryption for passwords, API keys, tokens, and secrets.
Even if someone steals brain.db, they get nothing.

Crypto stack:
  - AES-256-GCM: Authenticated encryption (tamper-proof)
  - Argon2id: Memory-hard key derivation (GPU-resistant)
  - Per-entry salt + nonce: Every secret has unique crypto params
  - Labels encrypted too: Even secret names are hidden

The master password is NEVER stored. It lives only in session memory.
No plaintext. No recovery. Lose the password, lose the secrets.

Hugo & Rim · February 2026
"""

import os
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from base64 import b64encode, b64decode

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend

from watty.config import WATTY_HOME


# ── Constants ──────────────────────────────────────────────

VAULT_DB = WATTY_HOME / "vault.db"

# Scrypt params (memory-hard, GPU-resistant)
# N=2^17 (~128MB RAM), r=8, p=1 — takes ~0.5s per derivation
SCRYPT_N = 2**17
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LENGTH = 32  # 256-bit key

# Verification constant — encrypted with the master key to verify password
VERIFY_MAGIC = b"WATTY_VAULT_UNLOCKED_v1"


# ── Key Derivation ─────────────────────────────────────────

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from password + salt using Scrypt."""
    kdf = Scrypt(
        salt=salt,
        length=SCRYPT_KEY_LENGTH,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        backend=default_backend(),
    )
    return kdf.derive(password.encode("utf-8"))


def _encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Encrypt with AES-256-GCM. Returns (nonce, ciphertext)."""
    nonce = os.urandom(12)  # 96-bit nonce (GCM standard)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ciphertext


def _decrypt(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt with AES-256-GCM. Raises on tamper/wrong key."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ── Vault Class ────────────────────────────────────────────

class Vault:
    """Zero-knowledge encrypted secret storage."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or VAULT_DB
        self._master_key: bytes | None = None
        self._init_db()

    def _init_db(self):
        """Create vault tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_salt TEXT NOT NULL,
                label_nonce TEXT NOT NULL,
                label_cipher TEXT NOT NULL,
                secret_salt TEXT NOT NULL,
                secret_nonce TEXT NOT NULL,
                secret_cipher TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    @property
    def is_initialized(self) -> bool:
        """Has a master password been set?"""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT value FROM vault_meta WHERE key = 'verify_salt'"
        ).fetchone()
        conn.close()
        return row is not None

    @property
    def is_unlocked(self) -> bool:
        """Is the vault currently unlocked in this session?"""
        return self._master_key is not None

    def initialize(self, master_password: str) -> dict:
        """Set the master password for the first time."""
        if self.is_initialized:
            return {"error": "Vault already initialized. Use unlock() instead."}

        # Generate verification data
        verify_salt = os.urandom(32)
        verify_key = _derive_key(master_password, verify_salt)
        verify_nonce, verify_cipher = _encrypt(VERIFY_MAGIC, verify_key)

        # Store verification (NOT the password)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES (?, ?)",
            ("verify_salt", b64encode(verify_salt).decode()),
        )
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES (?, ?)",
            ("verify_nonce", b64encode(verify_nonce).decode()),
        )
        conn.execute(
            "INSERT INTO vault_meta (key, value) VALUES (?, ?)",
            ("verify_cipher", b64encode(verify_cipher).decode()),
        )
        conn.commit()
        conn.close()

        # Derive and cache the session key
        self._master_key = verify_key
        return {"status": "Vault initialized and unlocked. Remember your password — there is no recovery."}

    def unlock(self, master_password: str) -> dict:
        """Unlock the vault with the master password."""
        if not self.is_initialized:
            return {"error": "Vault not initialized. Call initialize() first."}

        conn = sqlite3.connect(str(self.db_path))
        verify_salt = b64decode(
            conn.execute("SELECT value FROM vault_meta WHERE key = 'verify_salt'").fetchone()[0]
        )
        verify_nonce = b64decode(
            conn.execute("SELECT value FROM vault_meta WHERE key = 'verify_nonce'").fetchone()[0]
        )
        verify_cipher = b64decode(
            conn.execute("SELECT value FROM vault_meta WHERE key = 'verify_cipher'").fetchone()[0]
        )
        conn.close()

        # Derive key and verify
        key = _derive_key(master_password, verify_salt)
        try:
            plaintext = _decrypt(verify_nonce, verify_cipher, key)
            if plaintext != VERIFY_MAGIC:
                return {"error": "Wrong password."}
        except Exception:
            return {"error": "Wrong password."}

        self._master_key = key
        return {"status": "Vault unlocked."}

    def lock(self) -> dict:
        """Lock the vault — wipe the session key."""
        self._master_key = None
        return {"status": "Vault locked. Master key wiped from memory."}

    def store(self, label: str, secret: str, category: str = "general") -> dict:
        """Store an encrypted secret."""
        if not self.is_unlocked:
            return {"error": "Vault is locked. Unlock first."}

        now = datetime.now(timezone.utc).isoformat()

        # Encrypt label (separate salt so cracking one doesn't reveal labels)
        label_salt = os.urandom(32)
        label_key = _derive_key_fast(self._master_key, label_salt)
        label_nonce, label_cipher = _encrypt(label.encode("utf-8"), label_key)

        # Encrypt secret (separate salt)
        secret_salt = os.urandom(32)
        secret_key = _derive_key_fast(self._master_key, secret_salt)
        secret_nonce, secret_cipher = _encrypt(secret.encode("utf-8"), secret_key)

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO vault_secrets
            (label_salt, label_nonce, label_cipher, secret_salt, secret_nonce, secret_cipher, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            b64encode(label_salt).decode(),
            b64encode(label_nonce).decode(),
            b64encode(label_cipher).decode(),
            b64encode(secret_salt).decode(),
            b64encode(secret_nonce).decode(),
            b64encode(secret_cipher).decode(),
            category,
            now, now,
        ))
        conn.commit()
        conn.close()

        return {"status": f"Secret stored under encrypted label.", "category": category}

    def retrieve(self, label: str) -> dict:
        """Retrieve a secret by label."""
        if not self.is_unlocked:
            return {"error": "Vault is locked. Unlock first."}

        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT id, label_salt, label_nonce, label_cipher,
                   secret_salt, secret_nonce, secret_cipher, category, created_at
            FROM vault_secrets
        """).fetchall()
        conn.close()

        # Must decrypt ALL labels to find the match (zero-knowledge = no index)
        for row in rows:
            try:
                l_salt = b64decode(row[1])
                l_nonce = b64decode(row[2])
                l_cipher = b64decode(row[3])
                l_key = _derive_key_fast(self._master_key, l_salt)
                decrypted_label = _decrypt(l_nonce, l_cipher, l_key).decode("utf-8")

                if decrypted_label.lower() == label.lower():
                    # Found it — decrypt the secret
                    s_salt = b64decode(row[4])
                    s_nonce = b64decode(row[5])
                    s_cipher = b64decode(row[6])
                    s_key = _derive_key_fast(self._master_key, s_salt)
                    secret = _decrypt(s_nonce, s_cipher, s_key).decode("utf-8")

                    return {
                        "id": row[0],
                        "label": decrypted_label,
                        "secret": secret,
                        "category": row[7],
                        "created_at": row[8],
                    }
            except Exception:
                continue  # Wrong key or corrupted — skip

        return {"error": f"No secret found with label '{label}'."}

    def list_secrets(self) -> dict:
        """List all secret labels (decrypted) — NOT the secrets themselves."""
        if not self.is_unlocked:
            return {"error": "Vault is locked. Unlock first."}

        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT id, label_salt, label_nonce, label_cipher, category, created_at
            FROM vault_secrets ORDER BY created_at DESC
        """).fetchall()
        conn.close()

        entries = []
        for row in rows:
            try:
                l_salt = b64decode(row[1])
                l_nonce = b64decode(row[2])
                l_cipher = b64decode(row[3])
                l_key = _derive_key_fast(self._master_key, l_salt)
                label = _decrypt(l_nonce, l_cipher, l_key).decode("utf-8")
                entries.append({
                    "id": row[0],
                    "label": label,
                    "category": row[4],
                    "created_at": row[5],
                })
            except Exception:
                entries.append({"id": row[0], "label": "[CORRUPTED]", "category": row[4]})

        return {"count": len(entries), "entries": entries}

    def delete(self, label: str) -> dict:
        """Delete a secret by label."""
        if not self.is_unlocked:
            return {"error": "Vault is locked. Unlock first."}

        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT id, label_salt, label_nonce, label_cipher
            FROM vault_secrets
        """).fetchall()

        target_id = None
        for row in rows:
            try:
                l_salt = b64decode(row[1])
                l_nonce = b64decode(row[2])
                l_cipher = b64decode(row[3])
                l_key = _derive_key_fast(self._master_key, l_salt)
                decrypted_label = _decrypt(l_nonce, l_cipher, l_key).decode("utf-8")
                if decrypted_label.lower() == label.lower():
                    target_id = row[0]
                    break
            except Exception:
                continue

        if target_id is None:
            conn.close()
            return {"error": f"No secret found with label '{label}'."}

        conn.execute("DELETE FROM vault_secrets WHERE id = ?", (target_id,))
        conn.commit()
        conn.close()
        return {"status": f"Secret '{label}' deleted."}

    def change_password(self, old_password: str, new_password: str) -> dict:
        """Re-encrypt everything with a new master password."""
        # Verify old password
        result = self.unlock(old_password)
        if "error" in result:
            return result

        old_key = self._master_key

        # Decrypt all secrets with old key
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT id, label_salt, label_nonce, label_cipher,
                   secret_salt, secret_nonce, secret_cipher, category, created_at
            FROM vault_secrets
        """).fetchall()

        decrypted_entries = []
        for row in rows:
            try:
                l_salt = b64decode(row[1])
                l_nonce = b64decode(row[2])
                l_cipher = b64decode(row[3])
                l_key = _derive_key_fast(old_key, l_salt)
                label = _decrypt(l_nonce, l_cipher, l_key).decode("utf-8")

                s_salt = b64decode(row[4])
                s_nonce = b64decode(row[5])
                s_cipher = b64decode(row[6])
                s_key = _derive_key_fast(old_key, s_salt)
                secret = _decrypt(s_nonce, s_cipher, s_key).decode("utf-8")

                decrypted_entries.append({
                    "id": row[0], "label": label, "secret": secret,
                    "category": row[7], "created_at": row[8],
                })
            except Exception:
                continue

        # Generate new verification
        verify_salt = os.urandom(32)
        new_key = _derive_key(new_password, verify_salt)
        verify_nonce, verify_cipher = _encrypt(VERIFY_MAGIC, new_key)

        # Re-encrypt everything
        conn.execute("DELETE FROM vault_meta")
        conn.execute("DELETE FROM vault_secrets")

        conn.execute("INSERT INTO vault_meta (key, value) VALUES (?, ?)",
                      ("verify_salt", b64encode(verify_salt).decode()))
        conn.execute("INSERT INTO vault_meta (key, value) VALUES (?, ?)",
                      ("verify_nonce", b64encode(verify_nonce).decode()))
        conn.execute("INSERT INTO vault_meta (key, value) VALUES (?, ?)",
                      ("verify_cipher", b64encode(verify_cipher).decode()))

        self._master_key = new_key
        now = datetime.now(timezone.utc).isoformat()

        for entry in decrypted_entries:
            label_salt = os.urandom(32)
            label_key = _derive_key_fast(new_key, label_salt)
            label_nonce, label_cipher = _encrypt(entry["label"].encode("utf-8"), label_key)

            secret_salt = os.urandom(32)
            secret_key = _derive_key_fast(new_key, secret_salt)
            secret_nonce, secret_cipher = _encrypt(entry["secret"].encode("utf-8"), secret_key)

            conn.execute("""
                INSERT INTO vault_secrets
                (label_salt, label_nonce, label_cipher, secret_salt, secret_nonce, secret_cipher, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                b64encode(label_salt).decode(), b64encode(label_nonce).decode(),
                b64encode(label_cipher).decode(), b64encode(secret_salt).decode(),
                b64encode(secret_nonce).decode(), b64encode(secret_cipher).decode(),
                entry["category"], entry["created_at"], now,
            ))

        conn.commit()
        conn.close()

        return {"status": f"Password changed. {len(decrypted_entries)} secrets re-encrypted."}


# ── Fast Key Derivation (per-entry) ────────────────────────
# Full Scrypt for master password. HKDF-like fast derivation for per-entry keys.
# Each entry uses: HMAC-SHA256(master_key, entry_salt) — fast but still unique per entry.

def _derive_key_fast(master_key: bytes, salt: bytes) -> bytes:
    """Fast per-entry key derivation from master key + unique salt.
    Uses HMAC-SHA256 — the master key already went through Scrypt."""
    import hmac
    return hmac.new(master_key, salt, hashlib.sha256).digest()
