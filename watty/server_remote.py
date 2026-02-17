"""
Watty Remote MCP Server — Phone + Web Connectivity
====================================================
Serves the same Watty brain over HTTP so Claude iOS/Android
can connect via Settings > Connectors > Add Custom Connector.

Transports:
  - SSE (Server-Sent Events) at /sse + /messages
  - Streamable HTTP at /mcp (modern, recommended)

Same brain.db. Same tools. Live sync.

Usage:
  watty serve-remote              # Default: 0.0.0.0:8765
  watty serve-remote --port 9000  # Custom port
  watty serve-remote --host 127.0.0.1  # Localhost only

Hugo & Rim · February 2026
"""

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, HTMLResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

# Try to import streamable HTTP (available in newer MCP SDK versions)
try:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    HAS_STREAMABLE = True
except ImportError:
    HAS_STREAMABLE = False

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION, CONSOLIDATION_INTERVAL

# Import tool modules (same as server.py)
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


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Shared brain instance ──────────────────────────────────
brain = Brain()

# Give session tools brain access
tools_session.set_brain(brain)
discovery.set_brain(brain)
mentor.set_brain(brain)
tools_reflect.set_brain(brain)
tools_eval.set_brain(brain)
tools_graph.set_brain(brain)
tools_a2a.set_brain(brain)


# ── Conversation Boundary Detection (same as server.py) ────
# HTTP server is long-running. Detect conversation boundaries
# by inactivity: no tool calls for IDLE_TIMEOUT = digest + reload.

import time as _time

IDLE_TIMEOUT = 300  # 5 minutes

_last_tool_call = _time.time()
_conversation_had_activity = False
_digest_lock = threading.Lock()


def _touch_activity():
    global _last_tool_call, _conversation_had_activity
    _last_tool_call = _time.time()
    _conversation_had_activity = True


_latest_shape_text = None  # refreshed after each digest


def _run_between_conversations():
    global _conversation_had_activity, _latest_shape_text
    with _digest_lock:
        if not _conversation_had_activity:
            return
        _conversation_had_activity = False
        try:
            from watty.metabolism import (
                load_shape, format_shape_for_context, digest,
                apply_delta, save_shape, _get_recent_conversation,
                MIN_CHUNKS_TO_DIGEST, _log,
            )
            _log("Remote idle digest: conversation boundary detected...")
            conversation, chunk_count = _get_recent_conversation(brain)
            if chunk_count >= MIN_CHUNKS_TO_DIGEST and conversation and len(conversation.strip()) >= 100:
                shape = load_shape()
                delta = digest(conversation, shape)
                if delta is not None:
                    _log(f"Remote idle delta: {delta.get('action')} | {delta.get('belief', delta.get('target', ''))}")
                    shape = apply_delta(shape, delta)
                    save_shape(shape)
                    _log(f"Shape updated. {len(shape.get('understanding', []))} beliefs.")
                    _latest_shape_text = format_shape_for_context(shape)
                else:
                    _log("Remote idle digest: no change needed.")
            else:
                _log(f"Remote idle digest: skipped ({chunk_count} chunks).")
        except Exception as e:
            log(f"[Watty Remote] Idle digest error: {e}")


def _idle_watcher():
    while True:
        _time.sleep(30)
        if not _conversation_had_activity:
            continue
        elapsed = _time.time() - _last_tool_call
        if elapsed >= IDLE_TIMEOUT:
            _run_between_conversations()


_idle_thread = threading.Thread(target=_idle_watcher, daemon=True, name="watty-remote-idle")
_idle_thread.start()


# ── Build MCP server (identical to server.py) ──────────────
def _build_mcp_server() -> Server:
    """Build a fresh MCP Server instance with all tools registered."""
    # Load shape for server instructions (automatic context injection)
    # Prefer refreshed shape from idle digest, fall back to disk
    _shape_instructions = _latest_shape_text
    if not _shape_instructions:
        try:
            from watty.metabolism import load_shape, format_shape_for_context
            _shape = load_shape()
            _shape_text = format_shape_for_context(_shape)
            if _shape_text:
                _shape_instructions = _shape_text
        except Exception:
            pass

    srv = Server(SERVER_NAME, instructions=_shape_instructions)

    EXTRA_TOOLS = (
        tools_infra.TOOLS +
        tools_session.TOOLS +
        tools_gpu.TOOLS +
        tools_comms.TOOLS +
        tools_screen.TOOLS +
        tools_web.TOOLS +
        tools_daemon.TOOLS +
        tools_voice.TOOLS +
        tools_inference.TOOLS +
        tools_agent.TOOLS +
        discovery.TOOLS +
        mentor.TOOLS +
        tools_navigator.TOOLS +
        tools_vault.TOOLS +
        tools_watcher.TOOLS +
        tools_runtime.TOOLS +
        tools_reflect.TOOLS +
        tools_eval.TOOLS +
        tools_graph.TOOLS +
        tools_a2a.TOOLS
    )

    EXTRA_HANDLERS = {
        **tools_infra.HANDLERS,
        **tools_session.HANDLERS,
        **tools_gpu.HANDLERS,
        **tools_comms.HANDLERS,
        **tools_screen.HANDLERS,
        **tools_web.HANDLERS,
        **tools_daemon.HANDLERS,
        **tools_voice.HANDLERS,
        **tools_inference.HANDLERS,
        **tools_agent.HANDLERS,
        **discovery.HANDLERS,
        **mentor.HANDLERS,
        **tools_navigator.HANDLERS,
        **tools_vault.HANDLERS,
        **tools_watcher.HANDLERS,
        **tools_runtime.HANDLERS,
        **tools_reflect.HANDLERS,
        **tools_eval.HANDLERS,
        **tools_graph.HANDLERS,
        **tools_a2a.HANDLERS,
    }

    # Memory tool definitions (same as server.py)
    memory_tools = [
        Tool(name="watty_recall", description="Search Watty's cross-platform memory. Contains conversation history from Claude, ChatGPT, Grok, Gemini, scanned documents, and manually stored data.", inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}, "provider_filter": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}),
        Tool(name="watty_remember", description="Store something important in Watty's memory.", inputSchema={"type": "object", "properties": {"content": {"type": "string", "description": "What to remember"}, "provider": {"type": "string"}}, "required": ["content"]}),
        Tool(name="watty_scan", description="Scan a directory into memory. Supports .txt, .md, .json, .csv, .py, .js, .ts, .swift, .rs, .html, .css, .yaml, .yml, .toml, .sh, .log", inputSchema={"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}),
        Tool(name="watty_cluster", description="Organize memories into knowledge clusters.", inputSchema={"type": "object", "properties": {}}),
        Tool(name="watty_forget", description="Delete memories by query, chunk IDs, provider, or date.", inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "chunk_ids": {"type": "array", "items": {"type": "integer"}}, "provider": {"type": "string"}, "before": {"type": "string"}}}),
        Tool(name="watty_surface", description="Surface surprising connections and relevant insights.", inputSchema={"type": "object", "properties": {"context": {"type": "string"}}}),
        Tool(name="watty_reflect", description="Deep synthesis: total memories, providers, clusters, topics.", inputSchema={"type": "object", "properties": {}}),
        Tool(name="watty_stats", description="Quick brain health check.", inputSchema={"type": "object", "properties": {}}),
        Tool(name="watty_dream", description="Consolidation cycle: promote, decay, strengthen, prune.", inputSchema={"type": "object", "properties": {}}),
        Tool(name="watty_contradictions", description="Surface unresolved memory contradictions.", inputSchema={"type": "object", "properties": {}}),
        Tool(name="watty_resolve", description="Resolve a contradiction. Keep 'new' or 'old'.", inputSchema={"type": "object", "properties": {"chunk_id": {"type": "integer"}, "keep": {"type": "string"}}, "required": ["chunk_id"]}),
    ]

    @srv.list_tools()
    async def list_tools() -> list[Tool]:
        return memory_tools + EXTRA_TOOLS

    @srv.call_tool()
    async def handle_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            _touch_activity()
            if name in EXTRA_HANDLERS:
                handler = EXTRA_HANDLERS[name]
                import inspect
                if "brain" in inspect.signature(handler).parameters:
                    return await handler(arguments, brain=brain)
                return await handler(arguments)

            # Memory tools
            handlers = {
                "watty_recall": _handle_recall,
                "watty_remember": _handle_remember,
                "watty_scan": _handle_scan,
                "watty_cluster": _handle_cluster,
                "watty_forget": _handle_forget,
                "watty_surface": _handle_surface,
                "watty_reflect": _handle_reflect,
                "watty_stats": _handle_stats,
                "watty_dream": _handle_dream,
                "watty_contradictions": _handle_contradictions,
                "watty_resolve": _handle_resolve,
            }
            if name in handlers:
                return await handlers[name](arguments)
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            log(f"[Watty Remote] Error in {name}: {e}")
            return [TextContent(type="text", text=f"Watty error: {str(e)}")]

    return srv


# ── Memory tool handlers (same logic as server.py) ─────────

async def _handle_recall(args):
    results = brain.recall(args.get("query", ""), top_k=args.get("top_k"), provider_filter=args.get("provider_filter"))
    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]
    formatted = [f"[{i}] (score: {r['score']}, via: {r['provider']})\n{r['content']}" for i, r in enumerate(results, 1)]
    return [TextContent(type="text", text=f"Found {len(results)} memories:\n\n" + "\n\n---\n\n".join(formatted))]

async def _handle_remember(args):
    content = args.get("content", "")
    if not content.strip():
        return [TextContent(type="text", text="Nothing to remember.")]
    chunks = brain.store_memory(content, provider=args.get("provider", "manual"))
    return [TextContent(type="text", text=f"Remembered. {chunks} chunk(s) stored.")]

async def _handle_scan(args):
    path = args.get("path", "")
    if not path:
        return [TextContent(type="text", text="Need a path to scan.")]
    result = brain.scan_directory(path, recursive=args.get("recursive", True))
    if "error" in result:
        return [TextContent(type="text", text=f"Scan error: {result['error']}")]
    return [TextContent(type="text", text=f"Scanned: {result.get('files_scanned', 0)} files, {result.get('chunks_stored', 0)} chunks")]

async def _handle_cluster(args):
    clusters = brain.cluster()
    if not clusters:
        return [TextContent(type="text", text="Not enough memories to cluster.")]
    lines = [f"Cluster {i}: {c['label'][:80]} ({c['size']} memories)" for i, c in enumerate(clusters, 1)]
    return [TextContent(type="text", text=f"{len(clusters)} clusters:\n" + "\n".join(lines))]

async def _handle_forget(args):
    result = brain.forget(query=args.get("query"), chunk_ids=args.get("chunk_ids"), provider=args.get("provider"), before=args.get("before"))
    return [TextContent(type="text", text=f"Deleted {result['deleted']} memories.")]

async def _handle_surface(args):
    results = brain.surface(context=args.get("context"))
    if not results:
        return [TextContent(type="text", text="Not enough memories to surface insights.")]
    lines = [f"[{i}] ({r['reason']}) {r['content'][:200]}" for i, r in enumerate(results, 1)]
    return [TextContent(type="text", text=f"{len(results)} insights:\n\n" + "\n\n".join(lines))]

async def _handle_reflect(args):
    r = brain.reflect()
    return [TextContent(type="text", text=f"Memories: {r['total_memories']} | Clusters: {r['knowledge_clusters']} | Providers: {', '.join(r['providers'])}")]

async def _handle_stats(args):
    s = brain.stats()
    return [TextContent(type="text", text=f"Memories: {s['total_memories']} | Providers: {', '.join(s['providers']) or 'None'} | DB: {s['db_path']}")]

async def _handle_dream(args):
    r = brain.dream()
    return [TextContent(type="text", text=f"Dream: +{r.get('promoted',0)} promoted, -{r.get('decayed',0)} decayed, ~{r.get('associations_strengthened',0)} strengthened")]

async def _handle_contradictions(args):
    contradictions = brain.get_contradictions()
    if not contradictions:
        return [TextContent(type="text", text="No contradictions.")]
    lines = [f"[{c['new_chunk_id']}] NEW: {c['new_content'][:100]} vs OLD: {c['old_content'][:100]}" for c in contradictions]
    return [TextContent(type="text", text=f"{len(contradictions)} contradictions:\n" + "\n".join(lines))]

async def _handle_resolve(args):
    chunk_id = args.get("chunk_id")
    if chunk_id is None:
        return [TextContent(type="text", text="Missing chunk_id.")]
    result = brain.resolve_contradiction(chunk_id, keep=args.get("keep", "new"))
    return [TextContent(type="text", text=f"Resolved. {result.get('message', '')}")]


# ── SSE Transport Setup ────────────────────────────────────

sse_transport = SseServerTransport("/messages/")


async def handle_sse(request):
    """GET /sse — Client connects here to establish SSE stream."""
    mcp_server = _build_mcp_server()
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())


async def handle_messages(request):
    """POST /messages — Client sends tool calls here."""
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )


# ── Health + Info Endpoints ─────────────────────────────────

async def handle_health(request):
    """Health check for monitoring."""
    stats = brain.stats()
    return JSONResponse({
        "status": "ok",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "transport": "sse",
        "memories": stats.get("total_memories", 0),
        "providers": stats.get("providers", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def handle_explorer(request):
    """Serve the Brain Explorer UI."""
    explorer_path = Path(__file__).parent / "brain_explorer.html"
    if not explorer_path.exists():
        return HTMLResponse("<h1>Brain Explorer not found</h1>", status_code=404)
    return HTMLResponse(explorer_path.read_text(encoding="utf-8"))


async def handle_index(request):
    """Landing page with connection instructions."""
    stats = brain.stats()
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Watty Brain — Remote MCP</title>
<style>
body {{ font-family: system-ui, -apple-system; max-width: 600px; margin: 60px auto; padding: 0 20px; background: #0a0a0a; color: #e0e0e0; }}
h1 {{ color: #7c3aed; }}
code {{ background: #1a1a2e; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }}
.stat {{ color: #a78bfa; font-weight: bold; }}
.step {{ margin: 12px 0; padding: 12px; background: #111; border-left: 3px solid #7c3aed; border-radius: 4px; }}
</style></head>
<body>
<h1>Watty Brain</h1>
<p>v{SERVER_VERSION} | <span class="stat">{stats.get('total_memories', 0)}</span> memories | SSE transport active</p>

<h2>Connect from Claude iOS</h2>
<div class="step"><strong>1.</strong> Open Claude app on your iPhone</div>
<div class="step"><strong>2.</strong> Go to <strong>Settings -> Connectors</strong></div>
<div class="step"><strong>3.</strong> Tap <strong>"Add custom connector"</strong></div>
<div class="step"><strong>4.</strong> Enter this URL: <code>http://{{YOUR_IP}}:{{PORT}}/sse</code></div>
<div class="step"><strong>5.</strong> Done. Same brain. Live sync.</div>

<h2><a href="/explorer" style="color:#a78bfa;text-decoration:none;">Open Brain Explorer &rarr;</a></h2>

<h2>Endpoints</h2>
<p><code>GET /explorer</code> — Brain Explorer (spatial visualization)</p>
<p><code>GET /sse</code> — SSE connection (for MCP clients)</p>
<p><code>POST /messages</code> — Tool call endpoint</p>
<p><code>/mcp</code> — Streamable HTTP (modern MCP transport)</p>
<p><code>GET /health</code> — Health check</p>
<p><code>GET /api/stats</code> — Brain stats (JSON)</p>

<p style="margin-top: 40px; color: #666;">Hugo & Rim · Trinity Stack · {datetime.now().strftime('%B %Y')}</p>
</body></html>""")


async def handle_api_stats(request):
    """JSON stats endpoint for quick polling."""
    stats = brain.stats()
    return JSONResponse(stats)


async def handle_api_memories(request):
    """JSON endpoint: list memories with embeddings for visualization."""
    limit = int(request.query_params.get("limit", "500"))
    offset = int(request.query_params.get("offset", "0"))
    tier = request.query_params.get("tier")

    conn = brain._connect()
    query = "SELECT id, content, role, provider, created_at, source_type, source_path, memory_tier, significance, access_count, compressed_content FROM chunks"
    params = []
    if tier:
        query += " WHERE memory_tier = ?"
        params.append(tier)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    memories = []
    for row in rows:
        memories.append({
            "id": row["id"],
            "content": (row["compressed_content"] or row["content"])[:300],
            "role": row["role"],
            "provider": row["provider"],
            "created_at": row["created_at"],
            "source_type": row["source_type"],
            "source_path": row["source_path"],
            "memory_tier": row["memory_tier"],
            "significance": row["significance"],
            "access_count": row["access_count"],
        })
    conn.close()
    return JSONResponse({"memories": memories, "count": len(memories)})


async def handle_api_associations(request):
    """JSON endpoint: list associations for graph visualization."""
    limit = int(request.query_params.get("limit", "1000"))
    min_strength = float(request.query_params.get("min_strength", "0.1"))

    conn = brain._connect()
    rows = conn.execute(
        "SELECT source_chunk_id, target_chunk_id, strength, association_type FROM associations WHERE strength >= ? ORDER BY strength DESC LIMIT ?",
        (min_strength, limit),
    ).fetchall()
    conn.close()

    edges = [
        {"source": row["source_chunk_id"], "target": row["target_chunk_id"], "strength": row["strength"], "type": row["association_type"]}
        for row in rows
    ]
    return JSONResponse({"edges": edges, "count": len(edges)})


async def handle_api_clusters(request):
    """JSON endpoint: cluster data for visualization."""
    clusters = brain.cluster()
    return JSONResponse({"clusters": clusters, "count": len(clusters)})


async def handle_api_embeddings(request):
    """JSON endpoint: 2D projected embeddings for spatial visualization."""
    import numpy as np

    if brain._index_dirty:
        brain._build_index()
    if brain._vectors is None or len(brain._vectors) == 0:
        return JSONResponse({"points": [], "count": 0})

    # PCA projection to 2D (fast, no sklearn needed)
    vectors = brain._vectors
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    # Covariance matrix
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Top 2 eigenvectors
    top2 = eigenvectors[:, -2:][:, ::-1]
    projected = centered @ top2

    # Normalize to [-1, 1]
    pmax = np.max(np.abs(projected), axis=0, keepdims=True)
    pmax[pmax == 0] = 1
    projected = projected / pmax

    # Get metadata for each point
    conn = brain._connect()
    points = []
    for i, (x, y) in enumerate(projected):
        chunk_id = brain._vector_ids[i]
        row = conn.execute(
            "SELECT content, provider, memory_tier, significance, access_count, created_at FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            points.append({
                "id": chunk_id,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "content": row["content"][:150],
                "provider": row["provider"],
                "tier": row["memory_tier"],
                "significance": row["significance"] or 0,
                "access_count": row["access_count"] or 0,
                "created_at": row["created_at"],
            })
    conn.close()
    return JSONResponse({"points": points, "count": len(points)})


# ── Auto-dream background thread ───────────────────────────

import time as _time

def _auto_dream_loop():
    _time.sleep(60)
    while True:
        try:
            result = brain.dream()
            promoted = result.get("promoted", 0)
            if promoted:
                log(f"[Watty Remote] Dream: +{promoted} promoted")
        except Exception as e:
            log(f"[Watty Remote] Dream error: {e}")
        _time.sleep(CONSOLIDATION_INTERVAL)


# ── Streamable HTTP Transport (modern, recommended) ──────────

_streamable_manager = None

if HAS_STREAMABLE:
    _streamable_server = _build_mcp_server()
    _streamable_manager = StreamableHTTPSessionManager(app=_streamable_server)


# ── A2A Protocol HTTP Endpoints ────────────────────────────

async def handle_a2a_agent_card(request):
    """GET /.well-known/agent.json — Serve the Agent Card."""
    if brain._kg is None and not hasattr(brain, '_a2a_engine'):
        pass  # A2A works even without KG
    try:
        from watty.a2a import A2AEngine
        engine = A2AEngine(db_path=brain.db_path)
        # Gather all tool definitions
        all_tools = []
        for mod in [tools_infra, tools_session, tools_gpu, tools_comms,
                     tools_screen, tools_web, tools_daemon, tools_voice,
                     tools_inference, tools_agent, discovery, mentor,
                     tools_navigator, tools_vault, tools_watcher, tools_runtime,
                     tools_reflect, tools_eval, tools_graph, tools_a2a]:
            all_tools.extend(mod.TOOLS)
        base_url = str(request.base_url).rstrip("/")
        card = engine.generate_agent_card(all_tools, base_url)
        return JSONResponse(card)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_a2a_tasks(request):
    """GET/POST /a2a/tasks — List tasks or submit a new one."""
    from watty.config import A2A_AUTH_TOKEN
    if A2A_AUTH_TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {A2A_AUTH_TOKEN}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from watty.a2a import A2AEngine
        engine = A2AEngine(db_path=brain.db_path)
        if request.method == "POST":
            body = await request.json()
            result = engine.submit_task(
                skill_id=body.get("skill_id", ""),
                input_data=body.get("input", {}),
                source_agent=body.get("agent_name", "unknown"),
            )
            return JSONResponse(result)
        else:
            tasks = engine.list_tasks(
                status=request.query_params.get("status"),
                direction=request.query_params.get("direction"),
            )
            return JSONResponse({"tasks": tasks})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_a2a_task_status(request):
    """GET /a2a/tasks/{id} — Get task status."""
    from watty.config import A2A_AUTH_TOKEN
    if A2A_AUTH_TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {A2A_AUTH_TOKEN}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from watty.a2a import A2AEngine
        engine = A2AEngine(db_path=brain.db_path)
        task_id = request.path_params["task_id"]
        task = engine.get_task(task_id)
        if task is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(task)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_a2a_cancel_task(request):
    """POST /a2a/tasks/{id}/cancel — Cancel a task."""
    from watty.config import A2A_AUTH_TOKEN
    if A2A_AUTH_TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {A2A_AUTH_TOKEN}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from watty.a2a import A2AEngine
        engine = A2AEngine(db_path=brain.db_path)
        task_id = request.path_params["task_id"]
        result = engine.cancel_task(task_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── ASGI App Assembly ──────────────────────────────────────

_routes = [
    Route("/", handle_index),
    Route("/explorer", handle_explorer),
    Route("/health", handle_health),
    Route("/api/stats", handle_api_stats),
    Route("/api/memories", handle_api_memories),
    Route("/api/associations", handle_api_associations),
    Route("/api/clusters", handle_api_clusters),
    Route("/api/embeddings", handle_api_embeddings),
    Route("/sse", handle_sse),
    Route("/messages/", handle_messages, methods=["POST"]),
    Route("/messages", handle_messages, methods=["POST"]),
    # A2A Protocol endpoints
    Route("/.well-known/agent.json", handle_a2a_agent_card),
    Route("/a2a/tasks", handle_a2a_tasks, methods=["GET", "POST"]),
    Route("/a2a/tasks/{task_id:str}", handle_a2a_task_status),
    Route("/a2a/tasks/{task_id:str}/cancel", handle_a2a_cancel_task, methods=["POST"]),
]

if HAS_STREAMABLE and _streamable_manager:
    _routes.append(Mount("/mcp", app=_streamable_manager.handle_request))

app = Starlette(
    routes=_routes,
    middleware=[
        Middleware(CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ],
)


# ── Entry point ────────────────────────────────────────────

def run_remote(host: str = "0.0.0.0", port: int = 8765):
    """Start the remote MCP server."""
    log(f"[Watty] Remote MCP server starting on {host}:{port}")
    log(f"[Watty] Brain: {brain.db_path}")

    stats = brain.stats()
    log(f"[Watty] {stats.get('total_memories', 0)} memories loaded")
    log(f"[Watty] SSE endpoint: http://{host}:{port}/sse")
    if HAS_STREAMABLE:
        log(f"[Watty] Streamable HTTP: http://{host}:{port}/mcp")
    log(f"[Watty] Dashboard: http://{host}:{port}/")

    # Start auto-dream
    dream_thread = threading.Thread(target=_auto_dream_loop, daemon=True, name="watty-dream-remote")
    dream_thread.start()

    # Get local IP for display
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        log(f"[Watty] Your phone connector URL: http://{local_ip}:{port}/sse")
    except Exception:
        pass

    # Auto-reload: watch watty source files for changes
    watty_src = str(Path(__file__).parent)
    uvicorn.run(
        "watty.server_remote:app",
        host=host,
        port=port,
        log_level="info",
        reload=True,
        reload_dirs=[watty_src],
    )


if __name__ == "__main__":
    run_remote()
