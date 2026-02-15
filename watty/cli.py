"""
Watty Unified CLI
==================
One entry point. Subcommands for everything.

Usage:
    watty serve              # stdio MCP server (default)
    watty serve --http       # HTTP/SSE transport
    watty import chatgpt ~/export.zip
    watty import claude ~/export.json
    watty import json ~/messages.json
    watty doctor             # health check
    watty stats              # quick brain stats
    watty backup             # backup brain
    watty restore backup.tar.gz
    watty --version
"""

import argparse
import sys

from watty.config import SERVER_VERSION


def cmd_serve(args):
    from watty.log import setup
    setup()
    if args.http:
        from watty.server_http import run
        run()
    else:
        from watty.server import run
        run()


def cmd_import(args):
    from watty.log import setup
    setup()
    fmt = args.format
    path = args.path

    if fmt == "chatgpt":
        from watty.importers.chatgpt import import_chatgpt
        result = import_chatgpt(path)
    elif fmt == "claude":
        from watty.importers.claude import import_claude
        result = import_claude(path)
    elif fmt == "json":
        from watty.importers.generic import import_json
        result = import_json(path, provider=args.provider)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Imported {result['conversations']} conversations ({result['chunks']} chunks)")


def cmd_doctor(args):
    from watty.doctor import main as doctor_main
    # Rewrite sys.argv for doctor's argparse
    new_argv = ["watty-doctor"]
    if args.json:
        new_argv.append("--json")
    if args.db_path:
        new_argv.extend(["--db-path", args.db_path])
    sys.argv = new_argv
    doctor_main()


def cmd_stats(args):
    from watty.brain import Brain
    brain = Brain(db_path=args.db_path)
    s = brain.stats()
    pending = s.get("pending_embeddings", 0)
    pending_text = f"\n  Pending embeddings: {pending}" if pending else ""
    print(
        f"Watty Brain Status:\n"
        f"  Total memories: {s['total_memories']}\n"
        f"  Conversations: {s['total_conversations']}\n"
        f"  Files scanned: {s['total_files_scanned']}\n"
        f"  Providers: {', '.join(s['providers']) if s['providers'] else 'None yet'}\n"
        f"  Database: {s['db_path']}{pending_text}"
    )


def cmd_backup(args):
    from watty.log import setup
    setup()
    from watty.backup import backup
    backup(output=args.output, db_path=args.db_path)


def cmd_restore(args):
    from watty.log import setup
    setup()
    from watty.backup import restore
    restore(args.archive, force=args.force, db_path=args.db_path)


def cmd_re_embed(args):
    from watty.log import setup
    setup()
    from watty.brain import Brain
    brain = Brain(db_path=args.db_path)
    count = brain.re_embed()
    print(f"Re-embedded {count} chunks")


def main():
    parser = argparse.ArgumentParser(
        prog="watty",
        description="One memory. Every AI.",
    )
    parser.add_argument("--version", action="version", version=f"watty {SERVER_VERSION}")
    parser.add_argument("--db-path", help="Custom database path", default=None)
    parser.add_argument("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)", default=None)

    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start MCP server (default: stdio)")
    p_serve.add_argument("--http", action="store_true", help="Use HTTP/SSE transport")

    # import
    p_import = sub.add_parser("import", help="Import conversation history")
    p_import.add_argument("format", choices=["chatgpt", "claude", "json"], help="Import format")
    p_import.add_argument("path", help="Path to export file")
    p_import.add_argument("--provider", default="import", help="Provider name for json imports")

    # doctor
    p_doctor = sub.add_parser("doctor", help="Health check")
    p_doctor.add_argument("--json", action="store_true", help="Output as JSON")

    # stats
    sub.add_parser("stats", help="Quick brain stats")

    # backup
    p_backup = sub.add_parser("backup", help="Backup brain")
    p_backup.add_argument("-o", "--output", help="Output path")

    # restore
    p_restore = sub.add_parser("restore", help="Restore brain from backup")
    p_restore.add_argument("archive", help="Path to .tar.gz backup")
    p_restore.add_argument("--force", action="store_true", help="Overwrite existing brain.db")

    # re-embed
    sub.add_parser("re-embed", help="Re-embed chunks with missing vectors")

    args = parser.parse_args()

    # Apply global flags
    if args.log_level:
        import os
        os.environ["WATTY_LOG_LEVEL"] = args.log_level

    if args.command is None:
        # Default to serve
        args.http = False
        cmd_serve(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "backup":
        cmd_backup(args)
    elif args.command == "restore":
        cmd_restore(args)
    elif args.command == "re-embed":
        cmd_re_embed(args)


if __name__ == "__main__":
    main()
