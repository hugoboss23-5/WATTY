"""
Watty HTTP/SSE Transport
Same tools, same Brain, but over HTTP instead of stdio.
All tool definitions live in tools.py.

Usage:
    watty-http                    # starts on localhost:8766
    WATTY_HTTP_PORT=9000 watty-http
"""

import asyncio
import json
import os
import uuid

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION
from watty.tools import TOOL_DEFS, call_tool
from watty.log import log, setup

PORT = int(os.environ.get("WATTY_HTTP_PORT", "8766"))
HOST = os.environ.get("WATTY_HTTP_HOST", "localhost")

brain = Brain()


async def handle_sse(request):
    from aiohttp.web import StreamResponse

    session_id = str(uuid.uuid4())
    resp = StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    })
    await resp.prepare(request)
    await resp.write(f"event: endpoint\ndata: /messages?session_id={session_id}\n\n".encode())

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
        result = {"tools": TOOL_DEFS}
    elif method == "tools/call":
        try:
            # Run in thread pool so blocking calls (like subprocess.run in watty_shell)
            # don't freeze the event loop and prevent the response from being sent back.
            raw = await asyncio.to_thread(
                call_tool, brain, params.get("name", ""), params.get("arguments", {})
            )
            result = {"content": [{"type": "text", "text": raw["text"]}]}
            if raw.get("isError"):
                result["isError"] = True
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
    elif method == "notifications/initialized":
        return web.json_response({"ok": True})
    else:
        result = {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

    response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    try:
        await sse_resp.write(f"event: message\ndata: {json.dumps(response)}\n\n".encode())
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
    setup()
    log.info(f"Starting HTTP/SSE server on {HOST}:{PORT}")
    log.info(f"Brain: {brain.db_path}")
    stats = brain.stats()
    log.info(f"{stats['total_memories']} memories | {stats['total_conversations']} conversations")
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    log.info(f"SSE endpoint: http://{HOST}:{PORT}/sse")
    log.info(f"{len(TOOL_DEFS)} tools ready over HTTP. Connect any MCP client.")
    await asyncio.Event().wait()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
