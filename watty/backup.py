"""
Watty Backup & Restore
watty-backup: archive brain.db + key + manifest into a .tar.gz
watty-restore: extract and validate a backup archive
"""

import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from watty.config import WATTY_HOME, DB_PATH, SERVER_VERSION
from watty.crypto import KEY_PATH
from watty.log import log


def backup(output: str = None, db_path: str = None) -> str:
    import watty.config as cfg
    db = str(db_path or cfg.DB_PATH)
    if not os.path.exists(db):
        log.error("No brain.db found. Nothing to back up.")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    if not output:
        backup_dir = cfg.WATTY_HOME / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        output = str(backup_dir / f"watty-{ts}.tar.gz")

    from watty.brain import Brain
    brain = Brain(db_path=db)
    stats = brain.stats()
    manifest = {
        "version": SERVER_VERSION,
        "created": ts,
        "memories": stats["total_memories"],
        "conversations": stats["total_conversations"],
        "providers": stats["providers"],
        "encrypted": KEY_PATH.exists(),
    }

    with tarfile.open(output, "w:gz") as tar:
        tar.add(db, arcname="brain.db")
        if KEY_PATH.exists():
            tar.add(str(KEY_PATH), arcname="key")
        # Write manifest
        manifest_json = json.dumps(manifest, indent=2).encode()
        import io
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_json)
        tar.addfile(info, io.BytesIO(manifest_json))

    log.info(f"Backed up to {output} ({manifest['memories']} memories)")
    return output


def restore(archive: str, force: bool = False, db_path: str = None) -> dict:
    import watty.config as cfg
    target_db = str(db_path or cfg.DB_PATH)
    archive = str(Path(archive).expanduser().resolve())
    if not os.path.exists(archive):
        log.error(f"Backup not found: {archive}")
        sys.exit(1)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        if "manifest.json" not in names or "brain.db" not in names:
            log.error("Invalid backup: missing manifest.json or brain.db")
            sys.exit(1)

        manifest = json.loads(tar.extractfile("manifest.json").read())

        if os.path.exists(target_db) and not force:
            log.error(f"brain.db already exists ({target_db}). Use --force to overwrite.")
            sys.exit(1)

        if manifest.get("version") != SERVER_VERSION:
            log.warning(f"Backup version {manifest.get('version')} != current {SERVER_VERSION}")

        Path(target_db).parent.mkdir(parents=True, exist_ok=True)

        tmp = tempfile.mkdtemp()
        tar.extractall(tmp, filter="data")

        shutil.move(os.path.join(tmp, "brain.db"), target_db)
        if "key" in names:
            shutil.move(os.path.join(tmp, "key"), str(KEY_PATH))
            try:
                KEY_PATH.chmod(0o600)
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)

    log.info(f"Restored: {manifest['memories']} memories, {len(manifest.get('providers', []))} providers")
    return manifest


def backup_main():
    import argparse
    from watty.log import setup
    setup()
    p = argparse.ArgumentParser(description="Back up Watty brain")
    p.add_argument("-o", "--output", help="Output path (default: ~/.watty/backups/)")
    args = p.parse_args()
    backup(output=args.output)


def restore_main():
    import argparse
    from watty.log import setup
    setup()
    p = argparse.ArgumentParser(description="Restore Watty brain from backup")
    p.add_argument("archive", help="Path to .tar.gz backup")
    p.add_argument("--force", action="store_true", help="Overwrite existing brain.db")
    args = p.parse_args()
    restore(args.archive, force=args.force)
