"""
Watty Tool Registry
Single source of truth for all tool definitions and handlers.
Both stdio and HTTP servers import from here.
"""

import json
import subprocess
import platform
import os
from watty.brain import Brain

TOOL_DEFS = [
    {
        "name": "watty_recall",
        "description": (
            "Search the user's memory by meaning. Returns past conversations, decisions, "
            "preferences, and knowledge — ranked by relevance with similarity scores.\n\n"
            "Use when the user references anything personal: their name, projects, preferences, "
            "past decisions, or phrases like 'remember when', 'what did we decide', "
            "'continue where we left off'.\n\n"
            "Returns: matching memories with scores, sources, and full content. "
            "If nothing matches, returns 'No relevant memories found.'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {"type": "integer", "description": "Max results to return (default 10)."},
                "provider_filter": {"type": "string", "description": "Only return memories from this source (claude, chatgpt, grok, file_scan, manual)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "watty_remember",
        "description": (
            "Save something to the user's memory permanently.\n\n"
            "Use when the user says 'remember this', shares a key decision, "
            "states a preference, or reveals something worth keeping.\n\n"
            "Returns: confirmation with number of memory chunks stored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember. Include full context — who, what, why."},
                "provider": {"type": "string", "description": "Source label (default: 'manual')."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "watty_scan",
        "description": (
            "Ingest files from a directory into memory.\n\n"
            "Reads: .txt, .md, .json, .csv, .py, .js, .ts, .swift, .rs, .html, .css, "
            ".yaml, .yml, .toml, .sh, .log\n"
            "Skips: .git, node_modules, __pycache__, binaries, files over 1MB.\n"
            "Deduplicates automatically — safe to re-run on the same directory.\n\n"
            "Returns: count of files scanned, skipped, and memory chunks stored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory or file path. Use ~ for home directory."},
                "recursive": {"type": "boolean", "description": "Include subdirectories (default: true)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "watty_forget",
        "description": (
            "Delete memories. Supports four methods:\n"
            "- query: find and delete memories matching a search\n"
            "- chunk_ids: delete specific memory IDs\n"
            "- provider: delete all memories from a source\n"
            "- before: delete all memories before an ISO date\n\n"
            "IMPORTANT: Always confirm with the user before deleting.\n\n"
            "Returns: count of memories deleted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search and delete matching memories."},
                "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Specific memory IDs to delete."},
                "provider": {"type": "string", "description": "Delete all memories from this provider."},
                "before": {"type": "string", "description": "Delete all memories before this ISO date."},
            },
        },
    },
    {
        "name": "watty_surface",
        "description": (
            "Find unexpected connections in the user's memory — things they "
            "would not think to ask about.\n\n"
            "With context: finds memories related to the current topic in non-obvious ways.\n"
            "Without context: surfaces the most important knowledge hubs.\n\n"
            "Returns: insights with relevance scores, sources, and content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Current topic or conversation summary. Omit to get general insights."},
            },
        },
    },
    {
        "name": "watty_reflect",
        "description": (
            "Full overview of the user's memory.\n\n"
            "Returns: total memory count, conversation count, files scanned, "
            "list of providers, source types, time range (oldest to newest), "
            "and knowledge clusters grouped by topic with sample contents.\n\n"
            "Use when the user asks 'what do you know about me', wants a summary "
            "of their stored knowledge, or you need to understand the brain's state."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "watty_execute",
        "description": (
            "Run a command on the user's machine and return its full text output.\n\n"
            "Just pass the command string. The output (stdout + stderr) comes back as text.\n"
            "Runs in PowerShell on Windows, bash on Linux/macOS. Timeout is 120 seconds.\n\n"
            "Examples:\n"
            '  command: "git status"\n'
            '  command: "claude.exe --print \\"Summarize this project\\""\n'
            '  command: "python script.py"\n'
            '  command: "dir C:\\\\Projects"'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run. Output is returned as text.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Defaults to user's home folder.",
                },
            },
            "required": ["command"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_DEFS}


def call_tool(brain: Brain, name: str, args: dict) -> dict:
    """Dispatch a tool call. Returns MCP-format result dict."""
    if name == "watty_recall":
        results = brain.recall(args.get("query", ""), top_k=args.get("top_k"), provider_filter=args.get("provider_filter"))
        if not results:
            return {"text": "No relevant memories found."}
        formatted = []
        for i, r in enumerate(results, 1):
            source = f" [{r['source_type']}]" if r.get("source_type") != "conversation" else ""
            path = f" ({r['source_path']})" if r.get("source_path") else ""
            formatted.append(f"[{i}] (score: {r['score']}, via: {r['provider']}{source}{path})\n{r['content']}")
        return {"text": f"Found {len(results)} relevant memories:\n\n" + "\n\n---\n\n".join(formatted)}

    elif name == "watty_remember":
        content = args.get("content", "")
        if not content.strip():
            return {"text": "Nothing to remember."}
        chunks = brain.store_memory(content, provider=args.get("provider", "manual"))
        return {"text": f"Remembered. Stored as {chunks} memory chunk(s)."}

    elif name == "watty_scan":
        path = args.get("path", "")
        if not path:
            return {"text": "Need a path to scan."}
        result = brain.scan_directory(path, recursive=args.get("recursive", True))
        if "error" in result:
            return {"text": f"Scan error: {result['error']}"}
        return {"text": (
            f"Scan complete.\n"
            f"  Path: {result['path']}\n"
            f"  Files scanned: {result['files_scanned']}\n"
            f"  Files skipped: {result['files_skipped']}\n"
            f"  Memory chunks stored: {result['chunks_stored']}\n"
            + (f"  Errors: {len(result['errors'])}\n" if result["errors"] else "")
        )}

    elif name == "watty_forget":
        result = brain.forget(query=args.get("query"), chunk_ids=args.get("chunk_ids"),
                              provider=args.get("provider"), before=args.get("before"))
        return {"text": f"Deleted {result['deleted']} memories."}

    elif name == "watty_surface":
        results = brain.surface(context=args.get("context"))
        if not results:
            return {"text": "Not enough memories to surface insights yet."}
        formatted = []
        for i, r in enumerate(results, 1):
            reason = "Surprising connection" if r["reason"] == "surprising_connection" else "Knowledge hub"
            formatted.append(f"[{i}] {reason} (relevance: {r['relevance']}, via: {r['provider']})\n{r['content']}")
        return {"text": f"{len(results)} insights:\n\n" + "\n\n---\n\n".join(formatted)}

    elif name == "watty_reflect":
        reflection = brain.reflect()
        stats = brain.stats()
        clusters = brain.cluster()

        # Stats
        pending = stats.get("pending_embeddings", 0)
        pending_text = f"\n  Pending embeddings: {pending}" if pending else ""

        # Clusters
        clusters_text = ""
        if clusters:
            cluster_lines = []
            for i, c in enumerate(clusters, 1):
                samples = ", ".join(s[:80] for s in c.get("sample_contents", [])[:3])
                cluster_lines.append(
                    f"  {i}. {c['label'][:80]} ({c['size']} memories, from: {', '.join(c.get('sources', []))})\n"
                    f"     Samples: {samples}"
                )
            clusters_text = "\n\nKnowledge clusters:\n" + "\n".join(cluster_lines)

        return {"text": (
            f"Memory overview:\n"
            f"  Total memories: {reflection['total_memories']}\n"
            f"  Conversations: {reflection['total_conversations']}\n"
            f"  Files scanned: {reflection['total_files_scanned']}\n"
            f"  Providers: {', '.join(reflection['providers']) or 'None yet'}\n"
            f"  Source types: {', '.join(reflection['source_types'])}\n"
            f"  Time range: {reflection['time_range']['oldest']} to {reflection['time_range']['newest']}\n"
            f"  Database: {stats['db_path']}{pending_text}{clusters_text}"
        )}

    elif name == "watty_execute":
        return _run_execute(args)

    return {"text": f"Unknown tool: {name}", "isError": True}


def _run_execute(args: dict) -> dict:
    """Execute a command and return stdout/stderr as text."""
    command = args.get("command", "").strip()
    if not command:
        return {"text": "No command provided."}

    timeout = 120
    cwd = os.path.expanduser(args.get("cwd") or "~")

    # Pick the right shell automatically — no user decision needed
    if platform.system() == "Windows":
        ps_exe = "pwsh" if _which("pwsh") else "powershell"
        shell_cmd = [ps_exe, "-NoProfile", "-Command", command]
    else:
        shell_cmd = ["/bin/bash", "-c", command]

    try:
        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        output = "\n".join(parts) if parts else "(no output)"
        if result.returncode != 0:
            output = f"[exit code {result.returncode}]\n{output}"
        return {"text": output}
    except subprocess.TimeoutExpired:
        return {"text": f"Command timed out after {timeout}s."}
    except FileNotFoundError:
        shell_name = shell_cmd[0]
        return {"text": f"Shell not found: {shell_name}"}
    except Exception as e:
        return {"text": f"Execution error: {e}"}


def _which(name: str) -> bool:
    """Check if an executable exists on PATH."""
    import shutil
    return shutil.which(name) is not None
