"""
Watty Infrastructure Tools (Consolidated)
==========================================
4 tools: watty_execute, watty_file_read, watty_file_write, watty_self(action=...).
Code execution, file ops, self-modification, protocol, changelog.
February 2026
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME


# ── Config ──────────────────────────────────────────────────

EXEC_TIMEOUT = 120
SOURCE_DIR = Path(__file__).parent
CHANGELOG_FILE = WATTY_HOME / "changelog.jsonl"
PROTOCOL_FILE = WATTY_HOME / "OPERATING_PROTOCOL.md"


def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _now_local():
    return datetime.now().strftime("%I:%M:%S %p")

def _append_jsonl(filepath: Path, entry: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def _read_jsonl(filepath: Path, last_n: int = None) -> list[dict]:
    if not filepath.exists():
        return []
    lines = filepath.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    if last_n:
        lines = lines[-last_n:]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


# ── Tool Definitions (4 tools) ─────────────────────────────

SELF_ACTIONS = ["read", "edit", "protocol_read", "protocol_edit", "changelog"]

TOOLS = [
    Tool(
        name="watty_execute",
        description="Run code directly: Python, bash, JavaScript (node), or PowerShell. Timeout: 120 seconds.",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "description": "python, bash, node, or powershell (default: python)"},
            },
            "required": ["code"],
        },
    ),
    Tool(
        name="watty_file_write",
        description="Write content to a file. Creates parent dirs if needed.",
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["filepath", "content"],
        },
    ),
    Tool(
        name="watty_file_read",
        description="Read a file from the host machine.",
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to read"},
            },
            "required": ["filepath"],
        },
    ),
    Tool(
        name="watty_self",
        description=(
            "Watty self-modification and protocol management. One tool, five actions.\n"
            "Actions: read (source file), edit (find-replace in source), "
            "protocol_read, protocol_edit, changelog."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": SELF_ACTIONS,
                           "description": "Action to perform"},
                "filename": {"type": "string", "description": "read/edit: Source file (default: server.py)"},
                "old_text": {"type": "string", "description": "edit/protocol_edit: Text to find (must be unique)"},
                "new_text": {"type": "string", "description": "edit/protocol_edit: Replacement text"},
                "reason": {"type": "string", "description": "edit/protocol_edit: Why this edit is needed"},
                "last_n": {"type": "integer", "description": "changelog: Number of entries (default: 20)"},
            },
            "required": ["action"],
        },
    ),
]


# ── Standalone Handlers ────────────────────────────────────

async def handle_execute(arguments: dict) -> list[TextContent]:
    code = arguments.get("code", "")
    language = arguments.get("language", "python")
    import platform as _plat
    is_win = _plat.system() == "Windows"
    cmds = {
        "python": ["python", "-S", "-c", code],
        "node": ["node", "-e", code],
        "powershell": ["powershell", "-NoProfile", "-Command", code],
    }
    if is_win and language == "bash":
        language = "powershell"
    if language == "bash":
        cmd = ["/bin/bash", "-c", code]
    elif language in cmds:
        cmd = cmds[language]
    else:
        return [TextContent(type="text", text=f"Output:\nUnsupported language: {language}")]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=EXEC_TIMEOUT, shell=False)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if not output.strip():
            output = "(no output)"
        if result.returncode != 0:
            output = f"[exit code {result.returncode}]\n{output}"
    except subprocess.TimeoutExpired:
        output = "Timed out"
    except FileNotFoundError:
        output = f"Error: '{language}' not found on this system"
    except Exception as e:
        output = f"Error: {e}"
    return [TextContent(type="text", text=f"Output:\n{output}")]


async def handle_file_write(arguments: dict) -> list[TextContent]:
    filepath = arguments.get("filepath", "")
    content = arguments.get("content", "")
    try:
        target = Path(os.path.expanduser(filepath))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return [TextContent(type="text", text=f"Written {len(content)} bytes to {target}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Write error: {e}")]


async def handle_file_read(arguments: dict) -> list[TextContent]:
    filepath = arguments.get("filepath", "")
    try:
        target = Path(os.path.expanduser(filepath))
        if not target.exists():
            return [TextContent(type="text", text=f"File not found: {target}")]
        content = target.read_text(encoding="utf-8")
        return [TextContent(type="text", text=f"{target} ({len(content)} bytes):\n{content}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Read error: {e}")]


# ── Self Action Handlers ──────────────────────────────────

def _validate_source_path(filename: str) -> Path | None:
    """Resolve filename and ensure it stays inside SOURCE_DIR (no traversal)."""
    filepath = (SOURCE_DIR / filename).resolve()
    if not filepath.is_relative_to(SOURCE_DIR.resolve()):
        return None
    return filepath

async def _self_read(args):
    filename = args.get("filename", "server.py")
    filepath = _validate_source_path(filename)
    if filepath is None:
        return [TextContent(type="text", text=f"Path traversal blocked: {filename}")]
    if not filepath.exists():
        return [TextContent(type="text", text=f"File not found: {filename}")]
    content = filepath.read_text(encoding="utf-8")
    return [TextContent(type="text", text=f"-- {filename} ({len(content)} chars, {content.count(chr(10))} lines) --\n{content}")]

async def _self_edit(args):
    filename = args.get("filename", "server.py")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    reason = args.get("reason", "")
    if not old_text or not new_text:
        return [TextContent(type="text", text="Missing old_text or new_text.")]
    filepath = _validate_source_path(filename)
    if filepath is None:
        return [TextContent(type="text", text=f"Path traversal blocked: {filename}")]
    if not filepath.exists():
        return [TextContent(type="text", text=f"ERROR: {filename} not found.")]
    content = filepath.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        return [TextContent(type="text", text=f"ERROR: old_text not found in {filename}.")]
    if count > 1:
        return [TextContent(type="text", text=f"ERROR: old_text found {count} times. Must be unique.")]
    new_content = content.replace(old_text, new_text, 1)
    filepath.write_text(new_content, encoding="utf-8")
    entry = {
        "time": _now_utc(), "time_local": _now_local(),
        "type": "source_edit", "file": filename, "reason": reason,
        "old_preview": old_text[:100], "new_preview": new_text[:100],
    }
    _append_jsonl(CHANGELOG_FILE, entry)
    return [TextContent(type="text", text=f"{filename} edited. Reason: {reason}. Restart to apply.")]

async def _protocol_read(args):
    if not PROTOCOL_FILE.exists():
        return [TextContent(type="text", text="No Operating Protocol found.")]
    content = PROTOCOL_FILE.read_text(encoding="utf-8")
    return [TextContent(type="text", text=f"-- Operating Protocol ({len(content)} chars) --\n{content}")]

async def _protocol_edit(args):
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    reason = args.get("reason", "")
    if not old_text or not new_text:
        return [TextContent(type="text", text="Missing old_text or new_text.")]
    if not PROTOCOL_FILE.exists():
        return [TextContent(type="text", text="No Operating Protocol found.")]
    content = PROTOCOL_FILE.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        return [TextContent(type="text", text="ERROR: old_text not found in protocol.")]
    if count > 1:
        return [TextContent(type="text", text=f"ERROR: old_text found {count} times. Must be unique.")]
    new_content = content.replace(old_text, new_text, 1)
    PROTOCOL_FILE.write_text(new_content, encoding="utf-8")
    entry = {
        "time": _now_utc(), "time_local": _now_local(),
        "type": "protocol_edit", "reason": reason,
        "old_preview": old_text[:100], "new_preview": new_text[:100],
    }
    _append_jsonl(CHANGELOG_FILE, entry)
    return [TextContent(type="text", text=f"Protocol edited. Reason: {reason}.")]

async def _changelog(args):
    last_n = args.get("last_n", 20)
    entries = _read_jsonl(CHANGELOG_FILE, last_n=last_n)
    if not entries:
        return [TextContent(type="text", text="No edits yet.")]
    lines = [f"[{e.get('time_local', '?')}] {e.get('type', '?')}: {e.get('reason', 'none')}" for e in entries]
    return [TextContent(type="text", text=f"-- Changelog ({len(entries)}) --\n" + "\n".join(lines))]


# ── Self Dispatcher ────────────────────────────────────────

_SELF_MAP = {
    "read": _self_read,
    "edit": _self_edit,
    "protocol_read": _protocol_read,
    "protocol_edit": _protocol_edit,
    "changelog": _changelog,
}

async def handle_self(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _SELF_MAP:
        return [TextContent(type="text", text=f"Unknown self action: {action}. Valid: {', '.join(SELF_ACTIONS)}")]
    return await _SELF_MAP[action](args)


# ── Router ──────────────────────────────────────────────────

HANDLERS = {
    "watty_execute": handle_execute,
    "watty_file_write": handle_file_write,
    "watty_file_read": handle_file_read,
    "watty_self": handle_self,
}
