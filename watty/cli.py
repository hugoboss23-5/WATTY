"""
Watty CLI
=========
Command-line interface for Watty brain operations and daemon management.

Usage:
    watty daemon start     Start the autonomous daemon
    watty daemon stop      Stop the daemon
    watty daemon status    Check daemon status
    watty daemon log       View recent daemon activity
    watty daemon config    View/edit daemon configuration

    watty scan <path>      Scan a directory into memory
    watty recall <query>   Search memory
    watty stats            Brain health check
    watty dream            Run consolidation cycle
    watty cluster          Organize knowledge graph
    watty serve            Start MCP server (default)

    watty queue <type> <action> [params_json]
                           Queue a task for the daemon
"""

import sys
import os
import json
import subprocess
from pathlib import Path


def main():
    args = sys.argv[1:]
    if not args:
        cmd_serve()
        return

    cmd = args[0].lower()
    rest = args[1:]

    commands = {
        "chat": cmd_chat,
        "serve": cmd_serve,
        "serve-remote": cmd_serve_remote,
        "daemon": cmd_daemon,
        "scan": cmd_scan,
        "recall": cmd_recall,
        "stats": cmd_stats,
        "dream": cmd_dream,
        "cluster": cmd_cluster,
        "queue": cmd_queue,
        "setup": cmd_setup,
        "web": cmd_web,
        "explore": cmd_explore,
        "snapshot": cmd_snapshot,
        "rollback": cmd_rollback,
        "version": cmd_version,
        "help": cmd_help,
        "--help": cmd_help,
        "-h": cmd_help,
    }

    handler = commands.get(cmd)
    if handler:
        handler(rest)
    else:
        print(f"Unknown command: {cmd}")
        cmd_help([])


def cmd_chat(args=None):
    """Talk to Watty directly in the terminal."""
    from watty.chat import run
    run(args)


def cmd_serve(args=None):
    """Start the MCP server (default behavior)."""
    from watty.server import run
    run()


def cmd_serve_remote(args=None):
    """Start the remote HTTP MCP server for phone/web connectivity."""
    host = "0.0.0.0"
    port = 8765

    # Parse --host and --port flags
    if args:
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                port = int(args[i + 1])
            elif arg == "--host" and i + 1 < len(args):
                host = args[i + 1]
            elif arg.isdigit():
                port = int(arg)

    from watty.server_remote import run_remote
    run_remote(host=host, port=port)


def cmd_daemon(args=None):
    """Daemon management: start, stop, status, log, config."""
    if not args:
        print("Usage: watty daemon <start|stop|status|log|config|insights|queue>")
        return

    sub = args[0].lower()

    if sub == "start":
        _daemon_start(args[1:])
    elif sub == "stop":
        _daemon_stop()
    elif sub == "status":
        _daemon_status()
    elif sub in ("log", "logs", "activity"):
        _daemon_log(args[1:])
    elif sub == "config":
        _daemon_config(args[1:])
    elif sub == "insights":
        _daemon_insights(args[1:])
    else:
        print(f"Unknown daemon command: {sub}")


def _daemon_start(args):
    """Start the daemon as a background process."""
    from watty.daemon import daemon_status, DAEMON_DIR

    status = daemon_status()
    if status.get("running"):
        print(f"Daemon already running (PID {status['pid']})")
        return

    # Start as background process
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DAEMON_DIR / "daemon.log"

    if sys.platform == "win32":
        # Windows: use START /B
        cmd = f'start /B python -m watty.daemon > "{log_file}" 2>&1'
        subprocess.Popen(cmd, shell=True, creationflags=subprocess.DETACHED_PROCESS)
    else:
        # Unix: nohup
        cmd = f'nohup python -m watty.daemon > "{log_file}" 2>&1 &'
        subprocess.Popen(cmd, shell=True)

    import time
    time.sleep(2)
    status = daemon_status()
    if status.get("running"):
        print(f"Daemon started (PID {status['pid']})")
        print(f"Log: {log_file}")
    else:
        print("Daemon may still be starting. Check: watty daemon status")
        print(f"Log: {log_file}")


def _daemon_stop():
    from watty.daemon import daemon_stop
    result = daemon_stop()
    if result.get("success"):
        print(f"Daemon stopped (PID {result['pid']})")
    else:
        print(f"Stop failed: {result.get('error', '?')}")


def _daemon_status():
    from watty.daemon import daemon_status
    status = daemon_status()
    running = "RUNNING" if status.get("running") else "STOPPED"
    print(f"Daemon: {running}")
    if status.get("pid"):
        print(f"  PID: {status['pid']}")
    if status.get("started_at"):
        print(f"  Started: {status['started_at']}")
    if status.get("last_heartbeat"):
        print(f"  Last heartbeat: {status['last_heartbeat']}")


def _daemon_log(args):
    from watty.daemon import daemon_activity
    n = int(args[0]) if args else 20
    entries = daemon_activity(n)
    if not entries:
        print("No daemon activity yet.")
        return
    for e in entries:
        t = e.get("time_local", "?")
        action = e.get("action", "?")
        detail = e.get("detail", "")
        result = e.get("result", "")
        line = f"  {t} | {action}"
        if detail:
            line += f" | {detail[:60]}"
        if result and result != "ok":
            line += f" -> {result[:40]}"
        print(line)


def _daemon_config(args):
    from watty.daemon import daemon_config, daemon_update_config
    if not args:
        config = daemon_config()
        print(json.dumps(config, indent=2))
    else:
        # args[0] should be a JSON string of updates
        try:
            updates = json.loads(args[0])
            config = daemon_update_config(updates)
            print("Config updated:")
            print(json.dumps(config, indent=2))
        except json.JSONDecodeError:
            print(f"Invalid JSON: {args[0]}")


def _daemon_insights(args):
    from watty.daemon import daemon_insights
    n = int(args[0]) if args else 10
    insights = daemon_insights(n)
    if not insights:
        print("No insights yet. Daemon surfaces them every 2 hours.")
        return
    for i, ins in enumerate(insights):
        print(f"\n[{i+1}] {ins.get('timestamp', '?')}")
        print(f"    {ins.get('content', '?')[:200]}")


def cmd_scan(args=None):
    """Scan a directory into Watty's memory."""
    if not args:
        print("Usage: watty scan <path>")
        return
    from watty.brain import Brain
    brain = Brain()
    path = args[0]
    recursive = "--no-recurse" not in args
    print(f"Scanning {path}...")
    result = brain.scan_directory(path, recursive=recursive)
    print(f"Done: {result.get('files_scanned', 0)} files, {result.get('chunks_stored', 0)} chunks stored")


def cmd_recall(args=None):
    """Search Watty's memory."""
    if not args:
        print("Usage: watty recall <query>")
        return
    from watty.brain import Brain
    brain = Brain()
    query = " ".join(args)
    results = brain.recall(query)
    if not results:
        print("No matching memories.")
        return
    for r in results:
        score = r.get("score", 0)
        content = r.get("content", "")[:150]
        provider = r.get("provider", "?")
        print(f"  [{score:.2f}] ({provider}) {content}")


def cmd_stats(args=None):
    """Show brain health stats."""
    from watty.brain import Brain
    brain = Brain()
    s = brain.stats()
    print(f"Memories: {s.get('total_memories', 0)}")
    print(f"Conversations: {s.get('total_conversations', 0)}")
    print(f"Files scanned: {s.get('total_files_scanned', 0)}")
    print(f"Providers: {', '.join(s.get('providers', []))}")
    print(f"Database: {s.get('db_path', '?')}")


def cmd_dream(args=None):
    """Run a dream/consolidation cycle."""
    from watty.brain import Brain
    brain = Brain()
    print("Dreaming...")
    result = brain.dream()
    print(f"Promoted: {result.get('promoted', 0)}")
    print(f"Decayed: {result.get('decayed', 0)}")
    print(f"Associations strengthened: {result.get('associations_strengthened', 0)}")
    print(f"Associations pruned: {result.get('associations_pruned', 0)}")
    if result.get('compressed', 0):
        print(f"Compressed: {result['compressed']} ({result.get('chars_saved', 0):,} chars saved)")
    if result.get('duplicates_pruned', 0):
        print(f"Duplicates pruned: {result['duplicates_pruned']}")


def cmd_cluster(args=None):
    """Organize knowledge into clusters."""
    from watty.brain import Brain
    brain = Brain()
    print("Clustering...")
    clusters = brain.cluster()
    print(f"Found {len(clusters)} clusters")
    for c in clusters[:10]:
        label = c.get("label", "?")[:50]
        size = c.get("size", 0)
        print(f"  [{size}] {label}")


def cmd_queue(args=None):
    """Queue a task for the daemon."""
    if len(args) < 2:
        print("Usage: watty queue <type> <action> [params_json]")
        print("Types: brain, shell, gpu")
        print("Examples:")
        print('  watty queue brain dream')
        print('  watty queue brain scan \'{"path": "~/Documents"}\'')
        print('  watty queue gpu stop')
        print('  watty queue shell "nvidia-smi"')
        return
    from watty.daemon import daemon_queue_task
    task_type = args[0]
    action = args[1]
    params = json.loads(args[2]) if len(args) > 2 else {}
    task_id = daemon_queue_task(task_type, action, params)
    print(f"Task queued: {task_id}")


def cmd_setup(args=None):
    """First-run setup wizard."""
    from watty.config import ensure_home, WATTY_HOME
    from watty.brain import Brain
    from watty.daemon import DAEMON_DIR, _save_config, DEFAULT_CONFIG

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║         WATTY SETUP WIZARD           ║")
    print("  ║  Your brain's external hard drive.   ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # Step 1: Create home
    ensure_home()
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [1/5] Created {WATTY_HOME}")

    # Step 2: Scan default directories
    brain = Brain()
    scan_dirs = [
        Path.home() / "Documents",
        Path.home() / "Desktop",
        Path.home() / "Downloads",
    ]

    print("  [2/5] Scanning your files...")
    total_files = 0
    total_chunks = 0
    for d in scan_dirs:
        if d.exists():
            print(f"        {d.name}/...", end=" ", flush=True)
            try:
                result = brain.scan_directory(str(d), recursive=True)
                f = result.get("files_scanned", 0)
                c = result.get("chunks_stored", 0)
                total_files += f
                total_chunks += c
                print(f"{f} files, {c} chunks")
            except Exception as e:
                print(f"error: {e}")

    print(f"        Total: {total_files} files -> {total_chunks} memories")

    # Step 3: Cluster
    print("  [3/5] Clustering knowledge...", end=" ", flush=True)
    clusters = brain.cluster()
    print(f"{len(clusters)} clusters found")

    # Step 4: Dream
    print("  [4/5] First dream cycle...", end=" ", flush=True)
    result = brain.dream()
    print("done")

    # Step 5: Auto-configure MCP for Claude Desktop
    print("  [5/5] Configuring Claude Desktop...", end=" ", flush=True)
    mcp_configured = _configure_mcp()
    if mcp_configured:
        print("done")
    else:
        print("skipped (config not found)")

    # Save daemon config
    _save_config(DEFAULT_CONFIG)

    print()
    print("  Watty is ready.")
    print(f"  {total_chunks} memories | {len(clusters)} clusters")
    print()
    print("  Next steps:")
    print("    watty daemon start     Start the autonomous daemon")
    print("    watty serve-remote     Start remote server for phone connectivity")
    print("    watty recall <query>   Search your memory")
    print("    watty stats            Check brain health")
    print()


def _configure_mcp():
    """Auto-configure Watty as an MCP server for Claude Desktop."""
    import shutil

    # Find Claude Desktop config path
    if sys.platform == "win32":
        config_dir = Path(os.environ.get("APPDATA", "")) / "Claude"
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        config_dir = Path.home() / ".config" / "claude"

    config_file = config_dir / "claude_desktop_config.json"

    # Find watty executable
    watty_path = shutil.which("watty")
    if not watty_path:
        watty_path = "watty"  # fallback, assume it's on PATH after install

    # Read existing config or start fresh
    config = {}
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Add/update watty MCP server entry
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["watty"] = {
        "command": watty_path,
        "args": ["serve"],
    }

    # Write back
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def cmd_explore(args=None):
    """Launch the Brain Explorer visualization."""
    port = int(args[0]) if args else 8765
    print(f"Starting Brain Explorer on http://localhost:{port}/explorer")
    print(f"Dashboard: http://localhost:{port}/")
    from watty.server_remote import run_remote
    run_remote(host="127.0.0.1", port=port)


def cmd_web(args=None):
    """Launch the Watty web dashboard (via remote server)."""
    port = int(args[0]) if args else 7777
    print(f"Starting Watty web dashboard on http://localhost:{port}")
    try:
        from watty.server_remote import run_remote
        run_remote(host="127.0.0.1", port=port)
    except ImportError as e:
        print(f"Web dashboard not available: {e}")
        print("Try: watty serve-remote")


def cmd_snapshot(args=None):
    """Create a brain.db backup snapshot."""
    from watty.snapshots import create_snapshot, list_snapshots
    if args and args[0] == "list":
        snapshots = list_snapshots()
        if not snapshots:
            print("No snapshots yet.")
            return
        for s in snapshots:
            print(f"  {s['filename']}  ({s['size_mb']} MB)  {s['reason']}  {s['timestamp'][:16]}")
        return
    reason = " ".join(args) if args else "manual"
    result = create_snapshot(reason)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Snapshot created: {result['filename']} ({result['size_mb']} MB)")


def cmd_rollback(args=None):
    """Restore brain.db from a snapshot."""
    from watty.snapshots import rollback, list_snapshots
    if not args:
        snapshots = list_snapshots()
        if not snapshots:
            print("No snapshots available.")
            return
        latest = snapshots[-1]
        print(f"Rolling back to: {latest['filename']} ({latest['timestamp'][:16]})")
        result = rollback()
    else:
        result = rollback(args[0])

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Restored from: {result['from_snapshot']}")
        print("Pre-rollback backup saved.")


def cmd_version(args=None):
    from watty.config import SERVER_VERSION
    print(f"Watty v{SERVER_VERSION}")


def cmd_help(args=None):
    print("""
Watty — Your brain's external hard drive.

Quick start:
  watty setup              One-click setup (scan + cluster + configure)

Server:
  watty serve              Start MCP server (default if no command given)
  watty serve-remote       Start remote HTTP server for phone connectivity
                           (default: 0.0.0.0:8765, --port/--host to customize)
  watty explore [port]     Launch Brain Explorer visualization (default: 8765)
  watty web [port]         Launch web dashboard (default: 7777)

Daemon:
  watty daemon start       Start autonomous daemon
  watty daemon stop        Stop daemon
  watty daemon status      Check if daemon is running
  watty daemon log [n]     View recent activity
  watty daemon insights    View surfaced insights
  watty daemon config      View/edit daemon config

Memory:
  watty scan <path>        Scan directory into memory
  watty recall <query>     Search memory
  watty stats              Brain health check
  watty dream              Run consolidation cycle
  watty cluster            Organize knowledge graph

Safety:
  watty snapshot [reason]  Create brain.db backup
  watty snapshot list      List all snapshots
  watty rollback [file]    Restore from snapshot (latest if no file)

Advanced:
  watty queue <type> <action> [json]
                           Queue task for daemon
  watty version            Show version
  watty help               This message
""")


if __name__ == "__main__":
    main()
