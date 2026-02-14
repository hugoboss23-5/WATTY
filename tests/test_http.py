"""
Tests for HTTP/SSE transport.
Verifies the server starts, handles MCP messages, and returns results.
"""

import sys
import types
import json
import tempfile
import os
import asyncio
import numpy as np

# ── Mock sentence_transformers before any watty imports ────

if "sentence_transformers" not in sys.modules:
    _mock_st = types.ModuleType("sentence_transformers")

    class _MockModel:
        def encode(self, text, **kwargs):
            words = text.lower().split()
            vec = np.zeros(384, dtype=np.float32)
            for w in words:
                np.random.seed(hash(w) % (2**31))
                vec += np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec

    _mock_st.SentenceTransformer = lambda *a, **k: _MockModel()
    sys.modules["sentence_transformers"] = _mock_st


from watty.server_http import create_app, brain  # noqa: E402
from watty.tools import TOOL_DEFS, TOOL_NAMES, call_tool  # noqa: E402
from watty.brain import Brain  # noqa: E402


# ── Unit tests for tool dispatch ─────────────────────────

def test_tools_schema_complete():
    assert TOOL_NAMES == {
        "watty_recall", "watty_remember", "watty_scan", "watty_cluster",
        "watty_forget", "watty_surface", "watty_reflect", "watty_context", "watty_stats",
    }


def test_call_tool_stats():
    result = call_tool(brain, "watty_stats", {})
    assert "text" in result
    assert "Total memories" in result["text"]


def test_call_tool_remember_and_recall():
    call_tool(brain, "watty_remember", {"content": "HTTP transport test memory XYZ123"})
    result = call_tool(brain, "watty_recall", {"query": "HTTP transport test memory XYZ123"})
    assert "text" in result


def test_call_tool_unknown():
    result = call_tool(brain, "nonexistent_tool", {})
    assert result.get("isError") is True


def test_stdio_http_same_tools():
    """stdio and HTTP expose identical tool names."""
    http_names = {t["name"] for t in TOOL_DEFS}
    assert http_names == TOOL_NAMES


# ── Integration test with aiohttp test client ───────────

def test_http_sse_and_messages():
    try:
        from aiohttp import web
        from aiohttp.test_utils import AioHTTPTestCase, TestServer, TestClient
    except ImportError:
        return  # aiohttp not installed, skip

    async def _test():
        app = create_app()
        async with TestClient(TestServer(app)) as client:
            # Start SSE connection
            sse_resp = await client.get("/sse")
            assert sse_resp.status == 200
            # Read the endpoint event
            line = await sse_resp.content.readline()
            assert b"event: endpoint" in line
            data_line = await sse_resp.content.readline()
            assert b"/messages?session_id=" in data_line
            session_id = data_line.decode().split("session_id=")[1].strip()

            # Send initialize
            init_resp = await client.post(
                f"/messages?session_id={session_id}",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            assert init_resp.status == 200

            # Send tools/list
            list_resp = await client.post(
                f"/messages?session_id={session_id}",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            assert list_resp.status == 200

            # Send tools/call for stats
            call_resp = await client.post(
                f"/messages?session_id={session_id}",
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "watty_stats", "arguments": {}}},
            )
            assert call_resp.status == 200

            # Read SSE events (they should contain our responses)
            # The responses are sent via SSE, so read them
            await sse_resp.content.readline()  # empty line after endpoint event
            # Read initialize response
            event_line = await asyncio.wait_for(sse_resp.content.readline(), timeout=2)
            assert b"event: message" in event_line

            sse_resp.close()

    asyncio.run(_test())


def test_cors_preflight():
    try:
        from aiohttp.test_utils import TestServer, TestClient
    except ImportError:
        return

    async def _test():
        app = create_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.options("/messages")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    asyncio.run(_test())
