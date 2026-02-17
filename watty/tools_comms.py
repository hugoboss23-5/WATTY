"""
Watty Communications Tools (Consolidated)
==========================================
Two tools: watty_chat(action=...) + watty_browser(action=...).
Desktop <> Code chat bridge and browser research tracking.
February 2026
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME


# ── Config ──────────────────────────────────────────────────

CHAT_FILE = WATTY_HOME / "chat.jsonl"
BROWSER_LOG_FILE = WATTY_HOME / "browser_sessions.jsonl"


def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _now_local():
    return datetime.now().strftime("%I:%M:%S %p")

def _make_id(prefix: str = "msg"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

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

def _load_state() -> dict:
    state_file = WATTY_HOME / "session_state.json"
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_state(updates: dict):
    state_file = WATTY_HOME / "session_state.json"
    state = _load_state()
    state.update(updates)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Tool Definitions (2 tools) ────────────────────────────

CHAT_ACTIONS = ["send", "check", "history"]
BROWSER_ACTIONS = ["start", "log", "end", "recall", "bookmark"]

TOOLS = [
    Tool(
        name="watty_chat",
        description=(
            "Desktop <> Code chat bridge. One tool, three actions.\n"
            "Actions: send (message to Code), check (poll for replies), history (read full log)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": CHAT_ACTIONS,
                           "description": "Action to perform"},
                "message": {"type": "string", "description": "send: Message to send"},
                "last_n": {"type": "integer", "description": "history: Number of messages (default: 20)"},
            },
            "required": ["action"],
        },
    ),
    Tool(
        name="watty_browser",
        description=(
            "Tracked browser research sessions. One tool, five actions.\n"
            "Actions: start (begin session), log (record action), end (close + synthesize), "
            "recall (check if URL visited before), bookmark (save important page)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": BROWSER_ACTIONS,
                           "description": "Action to perform"},
                "purpose": {"type": "string", "description": "start: What you're researching"},
                "tags": {"type": "string", "description": "start: Comma-separated tags"},
                "session_id": {"type": "string", "description": "log/end: Session ID from start"},
                "url": {"type": "string", "description": "log/recall/bookmark: URL"},
                "notes": {"type": "string", "description": "log: What you learned"},
                "discoveries": {"type": "string", "description": "end: What was discovered"},
                "title": {"type": "string", "description": "bookmark: Page title"},
                "reason": {"type": "string", "description": "bookmark: Why this matters"},
            },
            "required": ["action"],
        },
    ),
]


# ── Chat Action Handlers ───────────────────────────────────

async def _chat_send(args):
    message = args.get("message", "")
    if not message:
        return [TextContent(type="text", text="No message provided.")]
    entry = {
        "id": _make_id("msg"), "timestamp": _now_utc(),
        "time_local": _now_local(), "from": "desktop", "message": message,
    }
    _append_jsonl(CHAT_FILE, entry)
    return [TextContent(type="text", text=f"Message sent to Code. ID: {entry['id']}")]

async def _chat_check(args):
    state = _load_state()
    last_read = state.get("chat_last_read_desktop", "")
    all_msgs = _read_jsonl(CHAT_FILE)
    unread = [m for m in all_msgs if m.get("timestamp", "") > last_read and m.get("from") != "desktop"]
    _save_state({"chat_last_read_desktop": _now_utc()})
    if not unread:
        return [TextContent(type="text", text="No new messages from Code.")]
    lines = [f"[{m['id']}] {m.get('time_local', '')}\n{m['message']}" for m in unread]
    return [TextContent(type="text", text=f"-- {len(unread)} new message(s) from Code --\n\n" + "\n\n---\n\n".join(lines))]

async def _chat_history(args):
    last_n = args.get("last_n", 20)
    msgs = _read_jsonl(CHAT_FILE, last_n=last_n)
    if not msgs:
        return [TextContent(type="text", text="No chat history yet.")]
    lines = [f"[{m.get('from', '?').upper()}] {m.get('time_local', '')} ({m['id']})\n{m['message']}" for m in msgs]
    _save_state({"chat_last_read_desktop": _now_utc()})
    return [TextContent(type="text", text=f"-- Chat History ({len(msgs)}) --\n\n" + "\n\n---\n\n".join(lines))]


# ── Browser Action Handlers ────────────────────────────────

async def _browser_start(args):
    purpose = args.get("purpose", "")
    if not purpose:
        return [TextContent(type="text", text="Missing purpose.")]
    tags = args.get("tags", "")
    session_id = _make_id("s")
    entry = {
        "id": session_id, "type": "session_start", "timestamp": _now_utc(),
        "time_local": _now_local(), "purpose": purpose, "tags": tags,
    }
    _append_jsonl(BROWSER_LOG_FILE, entry)
    return [TextContent(type="text", text=f"Browser session started. ID: {session_id}\nPurpose: {purpose}")]

async def _browser_log(args):
    entry = {
        "id": _make_id("bl"), "type": "browser_action", "timestamp": _now_utc(),
        "time_local": _now_local(),
        "session_id": args.get("session_id", ""),
        "action": "browsed",
        "url": args.get("url", ""),
        "notes": args.get("notes", ""),
    }
    _append_jsonl(BROWSER_LOG_FILE, entry)
    return [TextContent(type="text", text=f"Browser action logged. ID: {entry['id']} -> {entry['session_id']}")]

async def _browser_end(args):
    entry = {
        "id": _make_id("be"), "type": "session_end", "timestamp": _now_utc(),
        "time_local": _now_local(),
        "session_id": args.get("session_id", ""),
        "discoveries": args.get("discoveries", ""),
    }
    _append_jsonl(BROWSER_LOG_FILE, entry)
    return [TextContent(type="text", text="Session ended. Discoveries logged.")]

async def _browser_recall(args):
    url = args.get("url", "").lower()
    if not url:
        return [TextContent(type="text", text="Missing url.")]
    domain = url.split("//")[-1].split("/")[0] if "//" in url else url.split("/")[0]
    entries = _read_jsonl(BROWSER_LOG_FILE)
    matches = [e for e in entries if e.get("type") == "browser_action"
               and (url in e.get("url", "").lower() or domain in e.get("url", "").lower())]
    if not matches:
        return [TextContent(type="text", text=f"No prior visits to {domain}.")]
    lines = [f"[{m.get('id', '?')}] {m.get('time_local', '')} - {m.get('url', '')} - {m.get('notes', '')[:200]}"
             for m in matches]
    return [TextContent(type="text", text=f"Found {len(matches)} prior visit(s):\n\n" + "\n\n".join(lines))]

async def _browser_bookmark(args):
    url = args.get("url", "")
    title = args.get("title", "")
    reason = args.get("reason", "")
    if not url:
        return [TextContent(type="text", text="Missing url.")]
    entry = {
        "id": _make_id("bk"), "type": "bookmark", "timestamp": _now_utc(),
        "time_local": _now_local(), "url": url, "title": title, "reason": reason,
    }
    _append_jsonl(BROWSER_LOG_FILE, entry)
    from watty.brain import Brain
    brain = Brain()
    brain.store_memory(
        f"BOOKMARK: {title}\nURL: {url}\nWhy: {reason}",
        provider="manual",
    )
    return [TextContent(type="text", text=f"Bookmark saved: {title} -> {url}")]


# ── Dispatchers ────────────────────────────────────────────

_CHAT_MAP = {
    "send": _chat_send,
    "check": _chat_check,
    "history": _chat_history,
}

_BROWSER_MAP = {
    "start": _browser_start,
    "log": _browser_log,
    "end": _browser_end,
    "recall": _browser_recall,
    "bookmark": _browser_bookmark,
}

async def handle_chat(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _CHAT_MAP:
        return [TextContent(type="text", text=f"Unknown chat action: {action}. Valid: {', '.join(CHAT_ACTIONS)}")]
    return await _CHAT_MAP[action](args)

async def handle_browser(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _BROWSER_MAP:
        return [TextContent(type="text", text=f"Unknown browser action: {action}. Valid: {', '.join(BROWSER_ACTIONS)}")]
    return await _BROWSER_MAP[action](args)


# ── Router ──────────────────────────────────────────────────

HANDLERS = {
    "watty_chat": handle_chat,
    "watty_browser": handle_browser,
}
