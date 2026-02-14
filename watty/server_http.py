"""
Watty HTTP/SSE Transport
Same 8 tools, same Brain, but over HTTP instead of stdio.
Enables ChatGPT, Gemini, Grok, and any HTTP-capable MCP client.

Usage:
    watty-http                    # starts on localhost:8766
    WATTY_HTTP_PORT=9000 watty-http
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION

PORT = int(os.environ.get("WATTY_HTTP_PORT", "8766"))
HOST = os.environ.get("WATTY_HTTP_HOST", "localhost")

brain = Brain()


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Tool dispatch (mirrors server.py logic) ──────────────

TOOLS_SCHEMA = [
    {"name": "watty_recall", "description": "Semantic search across all memory", "inputSchema": {
        "type": "object", "properties": {
            "query": {"type": "string"}, "provider_filter": {"type": "string"}, "top_k": {"type": "integer"}
        }, "required": ["query"]}},
    {"name": "watty_remember", "description": "Store something important in memory", "inputSchema": {
        "type": "object", "properties": {
            "content": {"type": "string"}, "provider": {"type": "string"}
        }, "required": ["content"]}},
    {"name": "watty_scan", "description": "Scan a directory and ingest all supported files", "inputSchema": {
        "type": "object", "properties": {
            "path": {"type": "string"}, "recursive": {"type": "boolean"}
        }, "required": ["path"]}},
    {"name": "watty_cluster", "description": "Generate knowledge graph from memory clusters", "inputSchema": {
        "type": "object", "properties": {}}},
    {"name": "watty_forget", "description": "Delete memories by query, IDs, provider, or date", "inputSchema": {
        "type": "object", "properties": {
            "query": {"type": "string"}, "chunk_ids": {"type": "array", "items": {"type": "integer"}},
            "provider": {"type": "string"}, "before": {"type": "string"}
        }}},
    {"name": "watty_surface", "description": "Surface surprising connections from memory", "inputSchema": {
        "type": "object", "properties": {"context": {"type": "string"}}}},
    {"name": "watty_reflect", "description": "Deep synthesis — map the entire mind", "inputSchema": {
        "type": "object", "properties": {}}},
    {"name": "watty_stats", "description": "Brain health check", "inputSchema": {
        "type": "object", "properties": {}}},
]


def call_tool(name: str, args: dict) -> dict:
    if name == "watty_recall":
        results = brain.recall(args.get("query", ""), top_k=args.get("top_k"), provider_filter=args.get("provider_filter"))
        return {"content": [{"type": "text", "text": json.dumps(results, default=str)}]}
    elif name == "watty_remember":
        chunks = brain.store_memory(args.get("content", ""), provider=args.get("provider", "manual"))
        return {"content": [{"type": "text", "text": f"Stored as {chunks} chunk(s)."}]}
    elif name == "watty_scan":
        result = brain.scan_directory(args.get("path", ""), recursive=args.get("recursive", True))
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
    elif name == "watty_cluster":
        clusters = brain.cluster()
        return {"content": [{"type": "text", "text": json.dumps(clusters, default=str)}]}
    elif name == "watty_forget":
        result = brain.forget(query=args.get("query"), chunk_ids=args.get("chunk_ids"),
                              provider=args.get("provider"), before=args.get("before"))
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
    elif name == "watty_surface":
        results = brain.surface(context=args.get("context"))
        return {"content": [{"type": "text", "text": json.dumps(results, default=str)}]}
    elif name == "watty_reflect":
        return {"content": [{"type": "text", "text": json.dumps(brain.reflect(), default=str)}]}
    elif name == "watty_stats":
        return {"content": [{"type": "text", "text": json.dumps(brain.stats(), default=str)}]}
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


# ── HTTP/SSE Server ──────────────────────────────────────

async def handle_sse(request):
    from aiohttp import web
    from aiohttp.web import StreamResponse

    session_id = str(uuid.uuid4())
    resp = StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    })
    await resp.prepare(request)

    # Send endpoint event per MCP SSE spec
    messages_url = f"/messages?session_id={session_id}"
    await resp.write(f"event: endpoint\ndata: {messages_url}\n\n".encode())

    # Keep connection alive
    request.app["sessions"][session_id] = resp
    try:
        while not resp.task.done():
            await asyncio.sleep(15)
            try:
                await resp.write(b": keepalive\n\n")
            except (ConnectionResetError, ConnectionAbortedError):
                break
    finally:
        request.app["sessions"].pop(session_id, None)
    return resp


async def handle_messages(request):
    from aiohttp import web

    session_id = request.query.get("session_id")
    if not session_id or session_id not in request.app["sessions"]:
        return web.json_response({"error": "Invalid session"}, status=400)

    body = await request.json()
    method = body.get("method", "")
    msg_id = body.get("id")
    params = body.get("params", {})
    sse_resp = request.app["sessions"][session_id]

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS_SCHEMA}
    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        try:
            result = call_tool(tool_name, tool_args)
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
    elif method == "notifications/initialized":
        # Client ack — no response needed
        return web.json_response({"ok": True})
    else:
        result = {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

    response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    event_data = json.dumps(response)
    try:
        await sse_resp.write(f"event: message\ndata: {event_data}\n\n".encode())
    except (ConnectionResetError, ConnectionAbortedError):
        pass

    return web.json_response({"ok": True})


async def handle_cors_preflight(request):
    from aiohttp import web
    return web.Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })


def create_app():
    from aiohttp import web
    app = web.Application()
    app["sessions"] = {}
    app.router.add_get("/sse", handle_sse)
    app.router.add_post("/messages", handle_messages)
    app.router.add_options("/messages", handle_cors_preflight)
    return app


async def main():
    from aiohttp import web
    log(f"[Watty] Starting HTTP/SSE server on {HOST}:{PORT}")
    log(f"[Watty] Brain: {brain.db_path}")
    stats = brain.stats()
    log(f"[Watty] {stats['total_memories']} memories | {stats['total_conversations']} conversations")
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    log(f"[Watty] SSE endpoint: http://{HOST}:{PORT}/sse")
    log(f"[Watty] 8 tools ready over HTTP. Connect any MCP client.")
    await asyncio.Event().wait()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
