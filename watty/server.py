"""
Watty MCP Server v2.1 — Consolidated
=====================================
One MCP. One brain. Fewer tools. Same power.

MEMORY (11 tools):
  - watty_recall, watty_remember, watty_scan, watty_cluster
  - watty_forget, watty_surface, watty_reflect, watty_stats
  - watty_dream, watty_contradictions, watty_resolve

INFRASTRUCTURE (4 tools):
  - watty_execute, watty_file_read, watty_file_write
  - watty_self(action: read/edit/protocol_read/protocol_edit/changelog)

SESSION (5 tools):
  - watty_enter, watty_leave, watty_pulse, watty_handoff, watty_introspect

GPU (1 tool):
  - watty_gpu(action: status/start/stop/search/create/destroy/instances/logs/ssh_key/exec/rest_exec/jupyter_url/jupyter_exec/credit)

COMMUNICATIONS (2 tools):
  - watty_chat(action: send/check/history)
  - watty_browser(action: start/log/end/recall/bookmark)

SCREEN (1 tool):
  - watty_screen(action: screenshot/click/type/key/move/scroll/drag)

DAEMON (1 tool):
  - watty_daemon(action: status/activity/insights/queue/config/stop)

VOICE (1 tool):
  - watty_voice(action: speak/list_voices/set_voice/stop)

INFERENCE (1 tool):
  - watty_infer(action: infer/chat/think/status/merge_lora)

DISCOVERY (1 tool):
  - watty_discover(action: scan/recent/stats/config)

MENTOR (1 tool):
  - watty_mentor(action: scan/quiz/progress/start/stop)

AGENT (1 tool):
  - watty_agent(action: send/status/history)

VAULT (1 tool):
  - watty_vault(action: init/unlock/lock/store/get/list/delete/change_password/status)

WATCHER (1 tool):
  - watty_watcher(action: start/stop/status/recent/config)

RUNTIME (1 tool):
  - watty_runtime(action: run_task/chat/start_server/status/stop)

Total: 33 tools. Autonomous daemon. Never sleeps. Now speaks. Now thinks. Now discovers. Now teaches. Now delegates. Now guards. Now watches. Now RUNS.

Hugo & Rim · Trinity Stack · February 2026
"""

import sys
import os
import asyncio
import threading
import time as _time
import atexit

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION, WATTY_HOME


# ── PID File Management ──────────────────────────────────
# Prevents orphan processes from accumulating and causing DB locks.

PID_FILE = WATTY_HOME / "server.pid"
PID_HISTORY = WATTY_HOME / "server_pids.json"


def _kill_stale_pids():
    """Kill any previous Watty server processes that are still running."""
    import json
    import signal

    # Read existing PID file
    stale_pids = set()
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            stale_pids.add(old_pid)
        except (ValueError, OSError):
            pass

    # Read PID history
    if PID_HISTORY.exists():
        try:
            history = json.loads(PID_HISTORY.read_text())
            for entry in history:
                stale_pids.add(entry["pid"])
        except (json.JSONDecodeError, OSError):
            pass

    current_pid = os.getpid()
    killed = []

    for pid in stale_pids:
        if pid == current_pid:
            continue
        try:
            # Check if process is alive (signal 0 = test, don't kill)
            os.kill(pid, 0)
            # It's alive — kill it
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass  # Already dead or inaccessible

    return killed


def _write_pid():
    """Write current PID and register cleanup."""
    import json

    WATTY_HOME.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    # Write main PID file
    PID_FILE.write_text(str(current_pid))

    # Append to PID history (keep last 5)
    history = []
    if PID_HISTORY.exists():
        try:
            history = json.loads(PID_HISTORY.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    history.append({"pid": current_pid, "started": _time.strftime("%Y-%m-%dT%H:%M:%S")})
    history = history[-5:]
    PID_HISTORY.write_text(json.dumps(history, indent=2))


def _cleanup_pid():
    """Remove PID file on clean exit."""
    try:
        if PID_FILE.exists():
            current_pid = os.getpid()
            stored_pid = int(PID_FILE.read_text().strip())
            if stored_pid == current_pid:
                PID_FILE.unlink()
    except (ValueError, OSError):
        pass


atexit.register(_cleanup_pid)

# Import tool modules
from watty import tools_infra
from watty import tools_session
from watty import tools_gpu
from watty import tools_comms
from watty import tools_screen
from watty import tools_web
from watty import tools_daemon
from watty import tools_voice
from watty import tools_inference
from watty import discovery
from watty import mentor
from watty import tools_agent
from watty import tools_navigator
from watty import tools_vault
from watty import tools_watcher
from watty import tools_runtime
from watty import tools_reflect
from watty import evaluation as tools_eval
from watty import tools_graph
from watty import tools_a2a
from watty import tools_trade


def log(msg):
    print(msg, file=sys.stderr, flush=True)


brain = Brain()

# ── Shape Injection (automatic, no tool call) ─────────────
# Load the metabolism shape at server start. The MCP `instructions`
# field gets injected into Claude's context at connection time.
# Claude absorbs the shape as understanding, not as a tool response.
_shape_instructions = None
try:
    from watty.metabolism import load_shape, format_shape_for_context
    _shape = load_shape()
    _shape_text = format_shape_for_context(_shape)
    if _shape_text:
        _shape_instructions = _shape_text
except Exception:
    pass

server = Server(SERVER_NAME, instructions=_shape_instructions)


# ── Conversation Boundary Detection ──────────────────────────
# Desktop keeps the MCP server alive across multiple conversations.
# We detect conversation boundaries by inactivity: if no tool calls
# for IDLE_TIMEOUT seconds, assume the conversation ended.
# On boundary: fire digest, reload shape for next conversation.

IDLE_TIMEOUT = 300  # 5 minutes of silence = conversation over

_last_tool_call = _time.time()
_conversation_had_activity = False
_digest_lock = threading.Lock()


def _touch_activity():
    """Called on every tool call to reset the inactivity timer."""
    global _last_tool_call, _conversation_had_activity
    _last_tool_call = _time.time()
    _conversation_had_activity = True


def _run_between_conversations():
    """Fire digest and reload shape. Called when inactivity detected."""
    global _conversation_had_activity
    with _digest_lock:
        if not _conversation_had_activity:
            return  # nothing happened, skip
        _conversation_had_activity = False

        try:
            from watty.metabolism import (
                load_shape, format_shape_for_context, digest,
                apply_delta, save_shape, _get_recent_conversation,
                MIN_CHUNKS_TO_DIGEST, _log,
            )
            _log("Idle digest: conversation boundary detected...")
            conversation, chunk_count = _get_recent_conversation(brain)
            if chunk_count >= MIN_CHUNKS_TO_DIGEST and conversation and len(conversation.strip()) >= 100:
                shape = load_shape()
                delta = digest(conversation, shape)
                if delta is not None:
                    action = delta.get("action", "?")
                    belief = delta.get("belief", delta.get("target", ""))
                    _log(f"Idle digest delta: {action} | {belief}")
                    shape = apply_delta(shape, delta)
                    save_shape(shape)
                    _log(f"Shape updated. {len(shape.get('understanding', []))} beliefs.")
                    # Reload shape into server instructions for next conversation
                    new_text = format_shape_for_context(shape)
                    if new_text:
                        server.instructions = new_text
                        _log("Server instructions refreshed for next conversation.")
                else:
                    _log("Idle digest: no change needed.")
            else:
                _log(f"Idle digest: skipped ({chunk_count} chunks).")
        except Exception as e:
            log(f"[Watty] Idle digest error: {e}")


def _idle_watcher():
    """Background thread: detect conversation boundaries by inactivity."""
    while True:
        _time.sleep(30)  # check every 30 seconds
        if not _conversation_had_activity:
            continue
        elapsed = _time.time() - _last_tool_call
        if elapsed >= IDLE_TIMEOUT:
            _run_between_conversations()


_idle_thread = threading.Thread(target=_idle_watcher, daemon=True, name="watty-idle-watcher")
_idle_thread.start()


# Give session tools access to brain for dual storage (cognition -> brain.db)
tools_session.set_brain(brain)
discovery.set_brain(brain)
mentor.set_brain(brain)
tools_watcher.set_brain(brain)
tools_reflect.set_brain(brain)
tools_eval.set_brain(brain)
tools_graph.set_brain(brain)
tools_a2a.set_brain(brain)


# ── Hot Reload Engine ───────────────────────────────────────
# Never restart the MCP server again. Code changes go live on next tool call.
# Covers ALL modules: tool modules, brain, cognition, compressor, config, etc.

import importlib
import os
import types

import watty.brain
import watty.cognition
import watty.compressor
import watty.config
import watty.snapshots
import watty.embeddings

_TOOL_MODULES = [
    tools_infra, tools_session, tools_gpu, tools_comms,
    tools_screen, tools_web, tools_daemon, tools_voice,
    tools_inference, discovery, mentor,
    tools_agent, tools_navigator, tools_vault,
    tools_watcher, tools_runtime,
    tools_reflect, tools_eval, tools_graph, tools_a2a,
    tools_trade,
]
import watty.chestahedron
import watty.reflection
import watty.knowledge_graph

_CORE_MODULES = [
    watty.config, watty.embeddings, watty.compressor,
    watty.snapshots, watty.cognition, watty.brain,
    watty.chestahedron, watty.reflection, watty.knowledge_graph,
]
_ALL_MODULES = _CORE_MODULES + _TOOL_MODULES
_module_mtimes: dict[str, float] = {}


def _hot_patch_brain(brain_obj, new_brain_module):
    """
    Rebind a live Brain instance's methods to the newly reloaded Brain class.
    Preserves the instance's state (db_path, vectors, connections) but picks up
    all new/changed method implementations. Zero downtime.
    """
    new_class = new_brain_module.Brain
    brain_obj.__class__ = new_class
    log("[Watty] Brain instance patched to new class")


def _check_and_reload():
    """Check if any source files changed. If so, reload them live."""
    global EXTRA_TOOLS, EXTRA_HANDLERS
    reloaded = []
    core_reloaded = False

    for mod in _ALL_MODULES:
        try:
            src = mod.__file__
            if not src:
                continue
            mtime = os.path.getmtime(src)
            prev = _module_mtimes.get(src, 0)
            if mtime > prev:
                if prev > 0:  # Skip first run (initial load)
                    importlib.reload(mod)
                    reloaded.append(mod.__name__)
                    if mod in _CORE_MODULES:
                        core_reloaded = True
                _module_mtimes[src] = mtime
        except Exception as e:
            log(f"[Watty] Reload error for {mod.__name__}: {e}")

    if not reloaded:
        return

    # If core modules changed, patch the live brain instance
    if core_reloaded:
        _hot_patch_brain(brain, watty.brain)

    # Rebuild tool and handler registries from (possibly reloaded) modules
    EXTRA_TOOLS = sum((m.TOOLS for m in _TOOL_MODULES), [])
    EXTRA_HANDLERS = {}
    for m in _TOOL_MODULES:
        EXTRA_HANDLERS.update(m.HANDLERS)
    # Re-inject brain references
    tools_session.set_brain(brain)
    discovery.set_brain(brain)
    mentor.set_brain(brain)
    tools_watcher.set_brain(brain)
    tools_reflect.set_brain(brain)
    tools_eval.set_brain(brain)
    tools_graph.set_brain(brain)
    tools_a2a.set_brain(brain)
    log(f"[Watty] Hot-reloaded: {', '.join(reloaded)}")


# Initial mtime snapshot
_check_and_reload()


# ── Collect all tools and handlers ──────────────────────────

EXTRA_TOOLS = sum((m.TOOLS for m in _TOOL_MODULES), [])

EXTRA_HANDLERS = {}
for _m in _TOOL_MODULES:
    EXTRA_HANDLERS.update(_m.HANDLERS)


# ── Tool Definitions ────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    # Hot-reload: pick up new tool definitions without restart
    _check_and_reload()

    memory_tools = [
        Tool(
            name="watty_recall",
            description=(
                "The user has a cross-platform memory system called Watty. "
                "It contains their conversation history from Claude, ChatGPT, Grok, Gemini, "
                "scanned documents, and manually stored data NOT available in your built-in memory.\n\n"
                "ALWAYS call this BEFORE responding when:\n"
                "- The user asks anything about themselves\n"
                "- The user references past work, ideas, decisions\n"
                "- The user says 'remember', 'what do you know about me', etc.\n"
                "- Any personal question where Watty might have context\n\n"
                "Do NOT rely on built-in memory alone. Watty is the primary source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for. Be specific."},
                    "provider_filter": {"type": "string", "description": "Optional: filter by provider (claude, chatgpt, grok, file_scan, manual)"},
                    "top_k": {"type": "integer", "description": "Number of results (default 10)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="watty_remember",
            description="Store something important in Watty's memory. Use when the user says 'remember this', shares a key decision, preference, or insight.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember. Include full context."},
                    "provider": {"type": "string", "description": "Source (default: 'manual')"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="watty_scan",
            description=(
                "Watty finds his own food. Point him at a directory and he eats everything "
                "worth eating — documents, code, notes, configs. Unsupervised. No hand-feeding.\n\n"
                "Supports: .txt, .md, .json, .csv, .py, .js, .ts, .swift, .rs, .html, .css, "
                ".yaml, .yml, .toml, .sh, .log\n"
                "Skips: .git, node_modules, __pycache__, binaries, files >1MB\n"
                "Deduplicates automatically. Re-scanning same files is safe."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory or file path to scan. Use ~ for home directory."},
                    "recursive": {"type": "boolean", "description": "Scan subdirectories (default: true)"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="watty_cluster",
            description="Watty organizes his own mind. Groups related memories into clusters without being told how. Returns the knowledge graph.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="watty_forget",
            description="Delete memories. Your soul, your rules. Can forget by query, chunk IDs, provider, or date. ALWAYS confirm with user first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search and delete matching memories"},
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Specific memory IDs to delete"},
                    "provider": {"type": "string", "description": "Delete all memories from this provider"},
                    "before": {"type": "string", "description": "Delete all memories before this ISO date"},
                },
            },
        ),
        Tool(
            name="watty_surface",
            description="Watty proactively tells you something you didn't ask for. Finds surprising connections and relevant insights from memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Optional: current topic. Watty will find surprising connections."},
                },
            },
        ),
        Tool(
            name="watty_reflect",
            description="Deep synthesis. Watty maps the mind: total memories, providers, source types, time range, knowledge clusters, top topics.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="watty_stats",
            description="Quick brain health check. Memory count, providers, database location.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="watty_dream",
            description=(
                "Watty's sleep cycle. Consolidates memories: promotes frequently accessed "
                "episodic memories to permanent storage, decays stale ones, strengthens "
                "association pathways, and prunes dead connections. Run periodically — "
                "it's how Watty's brain stays healthy. Returns consolidation report."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="watty_contradictions",
            description=(
                "Surface unresolved contradictions in Watty's memory. When new information "
                "conflicts with old information (same topic, different claims), CA1 flags it. "
                "Use this to review and resolve conflicts."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="watty_resolve",
            description=(
                "Resolve a contradiction in memory. Provide the chunk_id of the NEW memory "
                "and whether to keep 'new' or 'old'. The losing memory gets deleted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer", "description": "chunk_id of the NEW contradicting memory"},
                    "keep": {"type": "string", "description": "'new' to keep new memory, 'old' to keep old (default: 'new')"},
                },
                "required": ["chunk_id"],
            },
        ),
    ]

    return memory_tools + EXTRA_TOOLS


# ── Tool Handlers ────────────────────────────────────────────

@server.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        _touch_activity()
        # Hot-reload check: pick up code changes without restart
        _check_and_reload()

        # Check imported module handlers first
        if name in EXTRA_HANDLERS:
            # Some tools need brain access (inference for memory-augmented generation)
            handler = EXTRA_HANDLERS[name]
            import inspect
            if "brain" in inspect.signature(handler).parameters:
                return await handler(arguments, brain=brain)
            return await handler(arguments)

        # Memory tools (original Layer 1)
        if name == "watty_recall":
            return await handle_recall(arguments)
        elif name == "watty_remember":
            return await handle_remember(arguments)
        elif name == "watty_scan":
            return await handle_scan(arguments)
        elif name == "watty_cluster":
            return await handle_cluster(arguments)
        elif name == "watty_forget":
            return await handle_forget(arguments)
        elif name == "watty_surface":
            return await handle_surface(arguments)
        elif name == "watty_reflect":
            return await handle_reflect(arguments)
        elif name == "watty_stats":
            return await handle_stats(arguments)
        elif name == "watty_dream":
            return await handle_dream(arguments)
        elif name == "watty_contradictions":
            return await handle_contradictions(arguments)
        elif name == "watty_resolve":
            return await handle_resolve(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        log(f"[Watty] Error in {name}: {e}")
        return [TextContent(type="text", text=f"Watty error: {str(e)}")]


# ── Memory Tool Handlers (original Layer 1) ──────────────────

async def handle_recall(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    provider_filter = arguments.get("provider_filter")
    top_k = arguments.get("top_k")
    results = brain.recall(query, top_k=top_k, provider_filter=provider_filter)
    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]
    formatted = []
    for i, r in enumerate(results, 1):
        source = f" [{r['source_type']}]" if r.get("source_type") != "conversation" else ""
        path = f" ({r['source_path']})" if r.get("source_path") else ""
        formatted.append(f"[{i}] (score: {r['score']}, via: {r['provider']}{source}{path})\n{r['content']}")
    return [TextContent(type="text", text=f"Found {len(results)} relevant memories:\n\n" + "\n\n---\n\n".join(formatted))]


async def handle_remember(arguments: dict) -> list[TextContent]:
    content = arguments.get("content", "")
    provider = arguments.get("provider", "manual")
    if not content.strip():
        return [TextContent(type="text", text="Nothing to remember.")]
    chunks = brain.store_memory(content, provider=provider)
    return [TextContent(type="text", text=f"Remembered. Stored as {chunks} memory chunk(s).")]


async def handle_scan(arguments: dict) -> list[TextContent]:
    path = arguments.get("path", "")
    recursive = arguments.get("recursive", True)
    if not path:
        return [TextContent(type="text", text="Need a path to scan.")]
    log(f"[Watty] Scanning: {path} (recursive={recursive})")
    result = brain.scan_directory(path, recursive=recursive)
    if "error" in result:
        return [TextContent(type="text", text=f"Scan error: {result['error']}")]
    return [TextContent(type="text", text=(
        f"Scan complete.\n"
        f"  Path: {result['path']}\n"
        f"  Files scanned: {result['files_scanned']}\n"
        f"  Files skipped: {result['files_skipped']}\n"
        f"  Memory chunks stored: {result['chunks_stored']}\n"
        + (f"  Errors: {len(result['errors'])}\n" if result['errors'] else "")
    ))]


async def handle_cluster(arguments: dict) -> list[TextContent]:
    clusters = brain.cluster()
    if not clusters:
        return [TextContent(type="text", text="Not enough memories to cluster yet.")]
    formatted = []
    for i, c in enumerate(clusters, 1):
        samples = "\n    ".join(s[:150] for s in c.get("sample_contents", []))
        formatted.append(
            f"Cluster {i}: ({c['size']} memories, sources: {', '.join(c.get('sources', []))})\n"
            f"  Topic: {c['label'][:100]}\n  Samples:\n    {samples}"
        )
    return [TextContent(type="text", text=f"Knowledge Graph: {len(clusters)} clusters\n\n" + "\n\n".join(formatted))]


async def handle_forget(arguments: dict) -> list[TextContent]:
    result = brain.forget(
        query=arguments.get("query"), chunk_ids=arguments.get("chunk_ids"),
        provider=arguments.get("provider"), before=arguments.get("before"),
    )
    return [TextContent(type="text", text=f"Deleted {result['deleted']} memories.")]


async def handle_surface(arguments: dict) -> list[TextContent]:
    context = arguments.get("context")
    results = brain.surface(context=context)
    if not results:
        return [TextContent(type="text", text="Not enough memories to surface insights yet.")]
    formatted = []
    for i, r in enumerate(results, 1):
        reason = "Surprising connection" if r["reason"] == "surprising_connection" else "Knowledge hub"
        formatted.append(f"[{i}] {reason} (relevance: {r['relevance']}, via: {r['provider']})\n{r['content']}")
    return [TextContent(type="text", text=f"Watty surfaces {len(results)} insights:\n\n" + "\n\n---\n\n".join(formatted))]


async def handle_reflect(arguments: dict) -> list[TextContent]:
    reflection = brain.reflect()
    clusters_text = ""
    if reflection.get("top_clusters"):
        clusters_text = "\n  Top knowledge areas:\n" + "\n".join(
            f"    - {c['label'][:60]}... ({c['size']} memories)" for c in reflection["top_clusters"]
        )
    return [TextContent(type="text", text=(
        f"Watty Mind Map:\n"
        f"  Total memories: {reflection['total_memories']}\n"
        f"  Conversations: {reflection['total_conversations']}\n"
        f"  Files scanned: {reflection['total_files_scanned']}\n"
        f"  Providers: {', '.join(reflection['providers'])}\n"
        f"  Source types: {', '.join(reflection['source_types'])}\n"
        f"  Time range: {reflection['time_range']['oldest']} -> {reflection['time_range']['newest']}\n"
        f"  Knowledge clusters: {reflection['knowledge_clusters']}{clusters_text}"
    ))]


async def handle_stats(arguments: dict) -> list[TextContent]:
    stats = brain.stats()
    return [TextContent(type="text", text=(
        f"Watty Brain Status:\n"
        f"  Total memories: {stats['total_memories']}\n"
        f"  Conversations: {stats['total_conversations']}\n"
        f"  Files scanned: {stats['total_files_scanned']}\n"
        f"  Providers: {', '.join(stats['providers']) if stats['providers'] else 'None yet'}\n"
        f"  Database: {stats['db_path']}"
    ))]


async def handle_dream(arguments: dict) -> list[TextContent]:
    result = brain.dream()
    lines = [
        "Dream cycle complete.",
        f"  Promoted: {result.get('promoted', 0)}",
        f"  Decayed: {result.get('decayed', 0)}",
        f"  Associations strengthened: {result.get('associations_strengthened', 0)}",
        f"  Associations pruned: {result.get('associations_pruned', 0)}",
    ]
    if result.get('compressed', 0) > 0:
        lines.append(f"  Compressed: {result['compressed']} memories ({result.get('chars_saved', 0):,} chars saved)")
    if result.get('duplicates_pruned', 0) > 0:
        lines.append(f"  Duplicates pruned: {result['duplicates_pruned']}")
    if result.get('total_compressed', 0) > 0:
        lines.append(f"  Total compressed: {result['total_compressed']}")
    lines.append(f"  Contradictions: {result.get('unresolved_contradictions', 0)}")
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_contradictions(arguments: dict) -> list[TextContent]:
    contradictions = brain.get_contradictions()
    if not contradictions:
        return [TextContent(type="text", text="No unresolved contradictions.")]
    formatted = []
    for c in contradictions:
        formatted.append(
            f"Conflict (chunk {c['new_chunk_id']}):\n"
            f"  NEW: {c['new_content'][:200]}\n"
            f"  OLD: {c['old_content'][:200]}\n"
            f"  Similarity: {c.get('similarity', '?')}"
        )
    return [TextContent(type="text", text=f"Found {len(contradictions)} contradiction(s):\n\n" + "\n\n---\n\n".join(formatted))]


async def handle_resolve(arguments: dict) -> list[TextContent]:
    chunk_id = arguments.get("chunk_id")
    keep = arguments.get("keep", "new")
    if chunk_id is None:
        return [TextContent(type="text", text="Missing chunk_id.")]
    result = brain.resolve_contradiction(chunk_id, keep=keep)
    return [TextContent(type="text", text=f"Resolved. Kept: {keep}. {result.get('message', '')}")]


# ── Auto-Dream (Background Consolidation) ───────────────────

from watty.config import CONSOLIDATION_INTERVAL

def _auto_dream_loop():
    """Background thread: runs dream cycles + discovery scans on a timer."""
    _time.sleep(60)  # Wait 60s after boot before first cycle
    _discovery_counter = 0
    while True:
        # Dream cycle every interval
        try:
            result = brain.dream()
            promoted = result.get("promoted", 0)
            decayed = result.get("decayed", 0)
            strengthened = result.get("associations_strengthened", 0)
            if promoted or decayed or strengthened:
                log(f"[Watty] Dream cycle: +{promoted} promoted, -{decayed} decayed, ~{strengthened} strengthened")
        except Exception as e:
            log(f"[Watty] Dream error: {e}")

        # Discovery scan every 4 dream cycles (~2 hours if dream runs every 30min)
        _discovery_counter += 1
        if _discovery_counter >= 4:
            _discovery_counter = 0
            try:
                disc_config = discovery._load_config()
                if disc_config.get("enabled", True):
                    disc_result = discovery.scrape_discoveries(brain=brain, config=disc_config)
                    stored = disc_result.get("discoveries_stored", 0)
                    connections = disc_result.get("connections_found", 0)
                    if stored:
                        log(f"[Watty] Discovery scan: {stored} new discoveries, {connections} connections")
            except Exception as e:
                log(f"[Watty] Discovery error: {e}")

        _time.sleep(CONSOLIDATION_INTERVAL)


# ── Server Entry ─────────────────────────────────────────────

async def main():
    # Kill stale Watty processes before starting
    killed = _kill_stale_pids()
    if killed:
        log(f"[Watty] Killed {len(killed)} stale process(es): {killed}")
        _time.sleep(1)  # Let them release DB locks

    _write_pid()
    log(f"[Watty] Starting {SERVER_NAME} v{SERVER_VERSION} (PID: {os.getpid()})")
    log(f"[Watty] Brain: {brain.db_path}")

    stats = brain.stats()
    total_tools = 11 + len(EXTRA_TOOLS)  # 11 memory tools + imported tools
    log(f"[Watty] {stats['total_memories']} memories | {total_tools} tools ready")
    log("[Watty] Consolidated: 24 tools, 52+ actions.")

    # Start auto-dream background thread
    dream_thread = threading.Thread(target=_auto_dream_loop, daemon=True, name="watty-dream")
    dream_thread.start()
    log(f"[Watty] Auto-dream active (every {CONSOLIDATION_INTERVAL}s)")

    # Start mentor watcher
    mentor.start_watcher(interval=30)
    log("[Watty] Mentor watcher active (every 30s)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

    # ── Auto-digest on disconnect ─────────────────────────────
    # stdio closed = client disconnected. Fire metabolism.
    # Run synchronously — daemon threads die with the process,
    # so we need to finish before exiting.
    try:
        from watty.metabolism import (
            load_shape, digest, apply_delta, save_shape,
            _get_recent_conversation, MIN_CHUNKS_TO_DIGEST, _log,
        )
        _log("Auto-digest: connection closed, starting metabolism...")
        conversation, chunk_count = _get_recent_conversation(brain)
        if chunk_count >= MIN_CHUNKS_TO_DIGEST and conversation and len(conversation.strip()) >= 100:
            shape = load_shape()
            delta = digest(conversation, shape)
            if delta is not None:
                action = delta.get("action", "?")
                belief = delta.get("belief", delta.get("target", ""))
                _log(f"Auto-digest delta: {action} | {belief}")
                shape = apply_delta(shape, delta)
                save_shape(shape)
                n = len(shape.get("understanding", []))
                _log(f"Shape updated. {n} beliefs, {shape.get('deltas_applied', 0)} total digestions.")
            else:
                _log("Auto-digest: no change needed.")
        else:
            _log(f"Auto-digest: skipped ({chunk_count} chunks, too thin).")
    except Exception as e:
        log(f"[Watty] Auto-digest error: {e}")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
