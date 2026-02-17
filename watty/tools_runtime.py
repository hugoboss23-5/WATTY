"""
Watty Runtime Tools — MCP Interface
=====================================
One tool: watty_runtime(action=...)
Actions: start_server, run_task, chat, status, stop

The autonomous agent loop — Watty as a computer.
February 2026
"""

import json
import os
import sys
import threading
import asyncio
import time

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME


# ── Singleton Runtime State ─────────────────────────────────

_runtime = None
_server_thread = None


def _get_runtime(api_key: str = None, model: str = None):
    """Lazy-init the runtime."""
    global _runtime
    if _runtime is None:
        from watty.runtime import WattyRuntime
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            # Try vault
            try:
                from watty.vault import Vault
                v = Vault()
                if v.is_unlocked:
                    result = v.retrieve("anthropic_api_key")
                    if "error" not in result:
                        key = result["secret"]
            except Exception:
                pass
        if not key:
            return None
        _runtime = WattyRuntime(api_key=key, model=model)
    return _runtime


# ── Tool Definition ─────────────────────────────────────────

RUNTIME_ACTIONS = ["run_task", "chat", "start_server", "status", "stop"]

TOOLS = [
    Tool(
        name="watty_runtime",
        description=(
            "Watty's autonomous agent runtime. Gives Watty a persistent think-act loop.\n"
            "Like giving an AI a computer — it can think, execute, and loop at API speed.\n\n"
            "Actions:\n"
            "  run_task — Run an autonomous task (loops until done, up to max_turns)\n"
            "  chat — Single-turn conversation with tool access\n"
            "  start_server — Launch HTTP API on port 7778 for external access\n"
            "  status — Runtime status (running, model, memory count)\n"
            "  stop — Stop the current autonomous task\n\n"
            "Requires: ANTHROPIC_API_KEY env var or 'anthropic_api_key' in vault.\n"
            "Default model: claude-opus-4-6"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": RUNTIME_ACTIONS,
                    "description": "Action to perform",
                },
                "task": {
                    "type": "string",
                    "description": "run_task: The task to execute autonomously",
                },
                "message": {
                    "type": "string",
                    "description": "chat: Message to send",
                },
                "model": {
                    "type": "string",
                    "description": "Model override (default: claude-opus-4-6)",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "run_task: Max autonomous turns (default: 50)",
                },
                "port": {
                    "type": "integer",
                    "description": "start_server: HTTP port (default: 7778)",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ─────────────────────────────────────────────────

async def handle_runtime(arguments: dict) -> list[TextContent]:
    global _runtime, _server_thread

    action = arguments.get("action", "status")

    if action == "status":
        if _runtime is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            has_key = bool(api_key)
            if not has_key:
                try:
                    from watty.vault import Vault
                    v = Vault()
                    if v.is_unlocked:
                        result = v.retrieve("anthropic_api_key")
                        has_key = "error" not in result
                except Exception:
                    pass

            return [TextContent(type="text", text=(
                f"Watty Runtime Status:\n"
                f"  Initialized: False\n"
                f"  API Key Available: {has_key}\n"
                f"  Server Running: {_server_thread is not None and _server_thread.is_alive()}\n"
                f"  Note: Runtime starts on first run_task or chat call."
            ))]

        return [TextContent(type="text", text=(
            f"Watty Runtime Status:\n"
            f"  Initialized: True\n"
            f"  Model: {_runtime.model}\n"
            f"  Running Task: {_runtime._running}\n"
            f"  Messages in Context: {len(_runtime.messages)}\n"
            f"  Brain Memories: {_runtime.brain.stats()['total_memories']}\n"
            f"  Server Running: {_server_thread is not None and _server_thread.is_alive()}"
        ))]

    elif action == "run_task":
        task = arguments.get("task", "")
        if not task:
            return [TextContent(type="text", text="Need a 'task' to run.")]

        model = arguments.get("model")
        max_turns = arguments.get("max_turns", 50)

        rt = _get_runtime(model=model)
        if rt is None:
            return [TextContent(type="text", text=(
                "No API key found. Set ANTHROPIC_API_KEY env var or store in vault:\n"
                "  watty_vault(action='store', label='anthropic_api_key', secret='sk-...', category='api_key')"
            ))]

        # Run in thread so we don't block MCP
        result_container = {"result": None, "error": None}

        def _run():
            try:
                result_container["result"] = rt.run_task(task, max_turns=max_turns)
            except Exception as e:
                result_container["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=300)  # 5 minute timeout

        if result_container["error"]:
            return [TextContent(type="text", text=f"Runtime error: {result_container['error']}")]

        result = result_container["result"] or "(no response)"
        return [TextContent(type="text", text=f"Task complete:\n\n{result}")]

    elif action == "chat":
        message = arguments.get("message", "")
        if not message:
            return [TextContent(type="text", text="Need a 'message' to chat.")]

        model = arguments.get("model")
        rt = _get_runtime(model=model)
        if rt is None:
            return [TextContent(type="text", text="No API key. Set ANTHROPIC_API_KEY or store in vault.")]

        try:
            result = rt.chat(message)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Chat error: {e}")]

    elif action == "start_server":
        port = arguments.get("port", 7778)
        model = arguments.get("model")

        rt = _get_runtime(model=model)
        if rt is None:
            return [TextContent(type="text", text="No API key. Set ANTHROPIC_API_KEY or store in vault.")]

        if _server_thread is not None and _server_thread.is_alive():
            return [TextContent(type="text", text="Server already running.")]

        def _serve():
            from watty.runtime import _run_server
            asyncio.run(_run_server(rt, port=port))

        _server_thread = threading.Thread(target=_serve, daemon=True, name="watty-runtime-server")
        _server_thread.start()
        time.sleep(1)  # Give it a moment to start

        return [TextContent(type="text", text=(
            f"Runtime server started on http://localhost:{port}\n"
            f"  POST /task  — Run autonomous task\n"
            f"  POST /chat  — Single-turn chat\n"
            f"  GET  /status — Runtime status\n"
            f"  POST /stop  — Stop current task"
        ))]

    elif action == "stop":
        if _runtime is None:
            return [TextContent(type="text", text="Runtime not initialized.")]
        _runtime.stop()
        return [TextContent(type="text", text="Runtime stopped.")]

    else:
        return [TextContent(type="text", text=f"Unknown action: {action}. Valid: {', '.join(RUNTIME_ACTIONS)}")]


HANDLERS = {"watty_runtime": handle_runtime}
