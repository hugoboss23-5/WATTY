"""
Watty Snapshots — Brain Backup & Rollback
==========================================
Safety net for destructive operations. Creates timestamped
copies of brain.db before dream cycles or schema changes.

Usage:
    from watty.snapshots import create_snapshot, rollback, list_snapshots

Hugo & Rim · February 2026
"""

import shutil
import json
from datetime import datetime, timezone
from pathlib import Path

from watty.config import WATTY_HOME, DB_PATH

SNAPSHOT_DIR = WATTY_HOME / "snapshots"
SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "manifest.json"
MAX_SNAPSHOTS = 10  # Keep last N snapshots, auto-prune older ones


def _ensure_dir():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> list[dict]:
    if not SNAPSHOT_MANIFEST.exists():
        return []
    try:
        return json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(entries: list[dict]):
    _ensure_dir()
    SNAPSHOT_MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def create_snapshot(reason: str = "manual") -> dict:
    """
    Create a timestamped backup of brain.db.
    Returns snapshot metadata.
    """
    _ensure_dir()
    db_path = Path(DB_PATH)

    if not db_path.exists():
        return {"error": "No brain.db to snapshot"}

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    filename = f"brain-{timestamp}.db"
    dest = SNAPSHOT_DIR / filename

    # Copy the database
    shutil.copy2(str(db_path), str(dest))

    # Get file size
    size_mb = dest.stat().st_size / (1024 * 1024)

    entry = {
        "filename": filename,
        "path": str(dest),
        "timestamp": now.isoformat(),
        "reason": reason,
        "size_mb": round(size_mb, 2),
    }

    # Update manifest
    manifest = _load_manifest()
    manifest.append(entry)

    # Auto-prune old snapshots
    if len(manifest) > MAX_SNAPSHOTS:
        to_remove = manifest[:-MAX_SNAPSHOTS]
        manifest = manifest[-MAX_SNAPSHOTS:]
        for old in to_remove:
            old_path = Path(old["path"])
            if old_path.exists():
                old_path.unlink()

    _save_manifest(manifest)
    return entry


def list_snapshots() -> list[dict]:
    """List all available snapshots."""
    manifest = _load_manifest()
    # Verify files still exist
    valid = []
    for entry in manifest:
        if Path(entry["path"]).exists():
            valid.append(entry)
    if len(valid) != len(manifest):
        _save_manifest(valid)
    return valid


def rollback(snapshot_filename: str = None) -> dict:
    """
    Restore brain.db from a snapshot.
    If no filename given, uses the most recent snapshot.
    """
    manifest = _load_manifest()
    if not manifest:
        return {"error": "No snapshots available"}

    if snapshot_filename:
        entry = next((e for e in manifest if e["filename"] == snapshot_filename), None)
        if not entry:
            return {"error": f"Snapshot not found: {snapshot_filename}"}
    else:
        entry = manifest[-1]

    source = Path(entry["path"])
    if not source.exists():
        return {"error": f"Snapshot file missing: {source}"}

    db_path = Path(DB_PATH)

    # Safety: backup current state before rollback
    if db_path.exists():
        pre_rollback = SNAPSHOT_DIR / f"brain-pre-rollback-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
        shutil.copy2(str(db_path), str(pre_rollback))

    # Restore
    shutil.copy2(str(source), str(db_path))

    return {
        "restored": True,
        "from_snapshot": entry["filename"],
        "snapshot_date": entry["timestamp"],
        "reason": entry["reason"],
        "pre_rollback_saved": True,
    }


def get_latest_snapshot() -> dict | None:
    """Get the most recent snapshot entry."""
    manifest = _load_manifest()
    return manifest[-1] if manifest else None
