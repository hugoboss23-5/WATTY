"""
Watty Doctor — health check for the brain.
Checks: database, embeddings, schema, memory count, disk usage, last backup.
"""

import json
import os
import sys

from watty.config import DB_PATH, WATTY_HOME, SERVER_VERSION, EMBEDDING_BACKEND


def check_all(db_path: str = None) -> dict:
    """Run all health checks. Returns dict with status and details."""
    db = str(db_path or DB_PATH)
    checks = {}
    healthy = True

    # 1. Database exists and is readable
    if os.path.exists(db):
        try:
            from watty.crypto import connect as crypto_connect
            conn = crypto_connect(db)
            conn.execute("SELECT 1").fetchone()
            size_mb = os.path.getsize(db) / (1024 * 1024)
            checks["database"] = {"status": "ok", "path": db, "size_mb": round(size_mb, 2)}
            conn.close()
        except Exception as e:
            checks["database"] = {"status": "error", "path": db, "error": str(e)}
            healthy = False
    else:
        checks["database"] = {"status": "missing", "path": db}
        healthy = False

    # 2. Embedding backend
    try:
        from watty.embeddings_loader import embed_text
        vec = embed_text("test")
        checks["embeddings"] = {"status": "ok", "backend": EMBEDDING_BACKEND, "dimension": len(vec)}
    except ImportError as e:
        checks["embeddings"] = {"status": "error", "backend": EMBEDDING_BACKEND, "error": str(e)}
        healthy = False
    except Exception as e:
        checks["embeddings"] = {"status": "error", "backend": EMBEDDING_BACKEND, "error": str(e)}
        healthy = False

    # 3. Schema version
    if checks.get("database", {}).get("status") == "ok":
        try:
            from watty.migrations import get_version, LATEST_VERSION
            from watty.crypto import connect as crypto_connect
            conn = crypto_connect(db)
            version = get_version(conn)
            conn.close()
            current = version == LATEST_VERSION
            checks["schema"] = {"status": "ok" if current else "outdated", "version": version, "latest": LATEST_VERSION}
            if not current:
                healthy = False
        except Exception as e:
            checks["schema"] = {"status": "error", "error": str(e)}
            healthy = False

    # 4. Memory stats
    if checks.get("database", {}).get("status") == "ok":
        try:
            from watty.brain import Brain
            brain = Brain(db_path=db)
            stats = brain.stats()
            checks["memories"] = {
                "status": "ok",
                "total": stats["total_memories"],
                "conversations": stats["total_conversations"],
                "files_scanned": stats["total_files_scanned"],
                "providers": stats["providers"],
                "pending_embeddings": stats["pending_embeddings"],
            }
        except Exception as e:
            checks["memories"] = {"status": "error", "error": str(e)}

    # 5. Last backup
    backup_dir = WATTY_HOME / "backups"
    if backup_dir.exists():
        backups = sorted(backup_dir.glob("watty-*.tar.gz"), reverse=True)
        if backups:
            latest = backups[0]
            checks["backup"] = {
                "status": "ok",
                "latest": str(latest.name),
                "count": len(backups),
            }
        else:
            checks["backup"] = {"status": "none", "message": "No backups found"}
    else:
        checks["backup"] = {"status": "none", "message": "No backups directory"}

    return {
        "healthy": healthy,
        "version": SERVER_VERSION,
        "checks": checks,
    }


def main():
    """CLI entry point for watty doctor."""
    import argparse
    from watty.log import setup
    setup()

    p = argparse.ArgumentParser(description="Watty health check")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--db-path", help="Custom database path")
    args = p.parse_args()

    result = check_all(db_path=args.db_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "HEALTHY" if result["healthy"] else "UNHEALTHY"
        print(f"Watty v{result['version']} — {status}")
        print()
        for name, check in result["checks"].items():
            icon = "+" if check["status"] == "ok" else "-" if check["status"] == "error" else "?"
            details = {k: v for k, v in check.items() if k != "status"}
            detail_str = " ".join(f"{k}={v}" for k, v in details.items())
            print(f"  [{icon}] {name}: {check['status']} {detail_str}")

    sys.exit(0 if result["healthy"] else 1)


if __name__ == "__main__":
    main()
