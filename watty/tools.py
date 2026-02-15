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
            "The user's personal memory across ALL their AI platforms. Contains past conversations, "
            "decisions, preferences, projects, and knowledge from Claude, ChatGPT, Grok, Gemini — "
            "plus scanned documents and notes you CANNOT access any other way.\n\n"
            "Call this FIRST whenever the user mentions: their name, their work, preferences, "
            "past projects, 'we discussed', 'you know', 'remember when', or asks ANY personal question.\n\n"
            "Example triggers: 'What tech stack did I choose?', 'Help me with my project', "
            "'What do I think about...', 'Continue where we left off', 'What did we decide?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for. Be specific."},
                "provider_filter": {"type": "string", "description": "Optional: filter by provider (claude, chatgpt, grok, file_scan, manual)"},
                "top_k": {"type": "integer", "description": "Number of results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "watty_remember",
        "description": (
            "Store something important in Watty's memory. Use when the user says "
            "'remember this', shares a key decision, preference, or insight."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember. Include full context."},
                "provider": {"type": "string", "description": "Source (default: 'manual')"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "watty_scan",
        "description": (
            "Watty finds his own food. Point him at a directory and he eats everything "
            "worth eating — documents, code, notes, configs. Unsupervised. No hand-feeding.\n\n"
            "Supports: .txt, .md, .json, .csv, .py, .js, .ts, .swift, .rs, .html, .css, "
            ".yaml, .yml, .toml, .sh, .log\n"
            "Skips: .git, node_modules, __pycache__, binaries, files >1MB\n"
            "Deduplicates automatically. Re-scanning same files is safe."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory or file path to scan. Use ~ for home directory."},
                "recursive": {"type": "boolean", "description": "Scan subdirectories (default: true)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "watty_cluster",
        "description": (
            "Watty organizes his own mind. Groups related memories into clusters "
            "without being told how. Returns the knowledge graph — what topics exist, "
            "how big each cluster is, sample contents. Use to understand the shape "
            "of the user's knowledge."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "watty_forget",
        "description": (
            "Delete memories. Your soul, your rules. Can forget by:\n"
            "- Search query (finds and deletes matching memories)\n"
            "- Specific chunk IDs\n"
            "- All memories from a provider\n"
            "- All memories before a date\n\n"
            "ALWAYS confirm with the user before deleting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search and delete matching memories"},
                "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Specific memory IDs to delete"},
                "provider": {"type": "string", "description": "Delete all memories from this provider"},
                "before": {"type": "string", "description": "Delete all memories before this ISO date"},
            },
        },
    },
    {
        "name": "watty_surface",
        "description": (
            "Watty proactively tells you something you didn't ask for. "
            "Finds surprising connections and relevant insights from memory.\n\n"
            "With context: finds memories that are related but not obvious — "
            "the connections you wouldn't have made yourself.\n"
            "Without context: surfaces the most important knowledge hubs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Optional: current topic or conversation context."},
            },
        },
    },
    {
        "name": "watty_reflect",
        "description": (
            "Deep synthesis. Watty looks at everything he knows and maps the mind. "
            "Returns: total memories, providers, source types, time range, "
            "knowledge clusters, and top topics."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "watty_context",
        "description": (
            "Lightning-fast pre-check: does Watty know anything about this topic? "
            "Returns relevance scores and short previews — not full memories.\n\n"
            "Use BEFORE watty_recall when unsure if the user's question relates to stored knowledge. "
            "Costs almost nothing. If top_score > 0.5, call watty_recall next for full context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or question to check against memory."},
                "top_k": {"type": "integer", "description": "Number of matches to preview (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "watty_stats",
        "description": "Quick brain health check. Memory count, providers, database location.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "watty_shell",
        "description": (
            "Execute a shell command on the user's machine and return the output.\n\n"
            "Use this when the user asks you to run commands, check system state, "
            "manage files, run scripts, install packages, or interact with their OS.\n\n"
            "The command runs in the user's home directory by default. "
            "Use the 'cwd' parameter to run from a specific directory.\n\n"
            "Shell selection:\n"
            "- 'auto' (default): PowerShell on Windows, bash on Linux/macOS\n"
            "- 'powershell': Force PowerShell (works on any OS with pwsh/powershell installed)\n"
            "- 'bash': Force bash\n"
            "- 'cmd': Force cmd.exe (Windows only)\n\n"
            "Examples: 'dir', 'git status', 'npm install', 'python script.py', 'Get-Process'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "shell": {
                    "type": "string",
                    "enum": ["auto", "powershell", "bash", "cmd"],
                    "description": "Shell to use (default: auto). 'powershell' for PowerShell, 'bash' for bash, 'cmd' for cmd.exe.",
                },
                "cwd": {"type": "string", "description": "Working directory (default: user home)."},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120, max: 300)."},
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

    elif name == "watty_cluster":
        clusters = brain.cluster()
        if not clusters:
            return {"text": "Not enough memories to cluster yet. Keep feeding Watty."}
        formatted = []
        for i, c in enumerate(clusters, 1):
            samples = "\n    ".join(s[:150] for s in c.get("sample_contents", []))
            formatted.append(
                f"Cluster {i}: ({c['size']} memories, sources: {', '.join(c.get('sources', []))})\n"
                f"  Topic: {c['label'][:100]}\n  Samples:\n    {samples}"
            )
        return {"text": f"Knowledge Graph: {len(clusters)} clusters\n\n" + "\n\n".join(formatted)}

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
        return {"text": f"Watty surfaces {len(results)} insights:\n\n" + "\n\n---\n\n".join(formatted)}

    elif name == "watty_reflect":
        reflection = brain.reflect()
        clusters_text = ""
        if reflection.get("top_clusters"):
            clusters_text = "\n  Top knowledge areas:\n" + "\n".join(
                f"    - {c['label'][:60]}... ({c['size']} memories)" for c in reflection["top_clusters"]
            )
        return {"text": (
            f"Watty Mind Map:\n"
            f"  Total memories: {reflection['total_memories']}\n"
            f"  Conversations: {reflection['total_conversations']}\n"
            f"  Files scanned: {reflection['total_files_scanned']}\n"
            f"  Providers: {', '.join(reflection['providers'])}\n"
            f"  Source types: {', '.join(reflection['source_types'])}\n"
            f"  Time range: {reflection['time_range']['oldest']} → {reflection['time_range']['newest']}\n"
            f"  Knowledge clusters: {reflection['knowledge_clusters']}{clusters_text}"
        )}

    elif name == "watty_context":
        ctx = brain.context(args.get("query", ""), top_k=args.get("top_k", 5))
        if not ctx["has_memories"]:
            return {"text": f"No relevant memories found ({ctx['total']} total in brain)."}
        previews = "\n".join(
            f"  [{m['score']}] {m['provider']}/{m['source_type']}: {m['preview']}"
            for m in ctx["matches"]
        )
        return {"text": (
            f"Watty has relevant memories (top score: {ctx['top_score']}, {ctx['total']} total):\n{previews}\n\n"
            f"Call watty_recall for full content."
        )}

    elif name == "watty_stats":
        stats = brain.stats()
        pending = stats.get("pending_embeddings", 0)
        pending_text = f"\n  Pending embeddings: {pending}" if pending else ""
        return {"text": (
            f"Watty Brain Status:\n"
            f"  Total memories: {stats['total_memories']}\n"
            f"  Conversations: {stats['total_conversations']}\n"
            f"  Files scanned: {stats['total_files_scanned']}\n"
            f"  Providers: {', '.join(stats['providers']) if stats['providers'] else 'None yet'}\n"
            f"  Database: {stats['db_path']}{pending_text}"
        )}

    elif name == "watty_shell":
        return _run_shell(args)

    return {"text": f"Unknown tool: {name}", "isError": True}


def _run_shell(args: dict) -> dict:
    """Execute a shell command and return stdout/stderr."""
    command = args.get("command", "").strip()
    if not command:
        return {"text": "No command provided."}

    timeout = min(args.get("timeout", 120), 300)
    cwd = args.get("cwd") or os.path.expanduser("~")
    cwd = os.path.expanduser(cwd)
    shell = args.get("shell", "auto").lower()

    is_windows = platform.system() == "Windows"

    if shell == "auto":
        shell = "powershell" if is_windows else "bash"

    if shell == "powershell":
        # Try pwsh (PowerShell 7+) first, fall back to powershell (Windows PowerShell 5.x)
        ps_exe = "pwsh" if _which("pwsh") else "powershell"
        shell_cmd = [ps_exe, "-NoProfile", "-Command", command]
        shell_label = ps_exe
    elif shell == "bash":
        shell_cmd = ["/bin/bash", "-c", command]
        shell_label = "/bin/bash"
    elif shell == "cmd":
        if not is_windows:
            return {"text": "cmd is only available on Windows."}
        shell_cmd = ["cmd.exe", "/C", command]
        shell_label = "cmd.exe"
    else:
        return {"text": f"Unknown shell: {shell}. Use auto, powershell, bash, or cmd."}

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
        return {"text": f"Shell not found: {shell_label}"}
    except Exception as e:
        return {"text": f"Execution error: {e}"}


def _which(name: str) -> bool:
    """Check if an executable exists on PATH."""
    import shutil
    return shutil.which(name) is not None
