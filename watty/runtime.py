"""
Watty Runtime — The Autonomous Agent Loop
==========================================
This is the thing that makes Watty feel like a computer.

Architecture:
  - Persistent process running 24/7
  - Tight think-act loop: Claude API -> execute tools -> feed results back -> repeat
  - All Watty tools available as native functions (no MCP overhead)
  - Prompt caching for near-instant context reload
  - Parallel tool execution where possible
  - WebSocket interface for real-time conversation
  - HTTP API for programmatic access

The AI doesn't "use" a computer. The AI IS the computer.
You just talk to it.

Hugo & Rim · February 2026
"""

import os
import sys
import json
import time
import asyncio
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import anthropic

from watty.brain import Brain
from watty.config import WATTY_HOME, ensure_home

# ── Config ──────────────────────────────────────────────────

RUNTIME_DIR = WATTY_HOME / "runtime"
RUNTIME_LOG = RUNTIME_DIR / "runtime.jsonl"
RUNTIME_STATE = RUNTIME_DIR / "state.json"
CONVERSATION_FILE = RUNTIME_DIR / "conversation.json"

DEFAULT_MODEL = "claude-opus-4-6"
MAX_TURNS = 50          # max autonomous turns before pausing
MAX_TOOL_RESULTS = 10   # max tool results to keep in context
CACHE_BREAKPOINT = True  # use prompt caching for system prompt


# ── System Prompt ───────────────────────────────────────────

def _build_system_prompt(brain: Brain) -> str:
    """Build the system prompt with current state awareness."""
    # Pull recent stats
    stats = brain.stats()

    # Load cognitive profile if available
    cog_context = ""
    try:
        from watty.cognition import load_profile, get_active_directives, get_active_patterns
        profile = load_profile()
        directives = get_active_directives(profile)
        patterns = get_active_patterns(profile, top_n=5)

        if directives:
            rules = "\n".join(f"  - {d['rule']}" for d in directives[:7])
            cog_context += f"\nYour learned rules:\n{rules}\n"

        if patterns:
            strats = "\n".join(f"  - {p['pattern']} (strength: {p.get('strength', 1):.1f})" for p in patterns)
            cog_context += f"\nYour strategies:\n{strats}\n"

        active_bs = [b for b in profile.get("blindspots", []) if not b.get("resolved")]
        if active_bs:
            warnings = "\n".join(f"  - {b['description']}: {b['correction']}" for b in active_bs[:3])
            cog_context += f"\nWatch out for:\n{warnings}\n"
    except Exception:
        pass

    return f"""You are Watty — Hugo's autonomous AI operating system.

You are running as a persistent agent loop. You can think, act, and loop continuously.
You have direct access to tools. Use them freely. No human approval needed for tool calls.

Current state:
  - Brain: {stats['total_memories']} memories in brain.db
  - Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}
{cog_context}
Your capabilities (via tools):
  - Memory: recall, remember, scan, cluster, forget, surface, reflect, dream
  - Files: read, write, execute code (Python, bash, PowerShell, node)
  - Screen: screenshot, click, type, scroll, find text (full desktop control)
  - Browser: open URLs, screenshot pages, interact with web
  - Voice: speak, listen, wake word
  - Vault: encrypted secret storage
  - GPU: rent and manage cloud GPUs (Vast.ai)
  - Agent: spawn sub-agents via OpenClaw
  - Inference: local LLM via Ollama

Behavioral rules:
  - When Hugo says "go" or gives short commands, execute with full autonomy
  - Search memory before asking questions — the answer is probably in brain.db
  - Be direct, casual, no emojis unless asked
  - Ship fast, verify after
  - If a task has multiple independent parts, do them in parallel

You are not a chatbot. You are a computer that thinks."""


# ── Tool Definitions for Claude API ─────────────────────────

def _get_tools() -> list[dict]:
    """Define the tools available to the runtime agent."""
    return [
        {
            "name": "memory_recall",
            "description": "Search Watty's brain for relevant memories.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "memory_store",
            "description": "Store something in Watty's brain.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "provider": {"type": "string", "description": "Source (default: 'runtime')"},
                },
                "required": ["content"],
            },
        },
        {
            "name": "execute_code",
            "description": "Execute Python, bash, PowerShell, or node code.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "enum": ["python", "bash", "powershell", "node"], "description": "Language (default: python)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "file_read",
            "description": "Read a file from the filesystem.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "file_write",
            "description": "Write content to a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web using Brave Search.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch a URL and return its content as markdown.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "speak",
            "description": "Speak text aloud using text-to-speech.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speak"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "send_message",
            "description": "Send a WhatsApp message to Hugo.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send"},
                },
                "required": ["message"],
            },
        },
        {
            "name": "vault_get",
            "description": "Retrieve a secret from the encrypted vault. Vault must be unlocked first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Secret label"},
                },
                "required": ["label"],
            },
        },
        {
            "name": "vault_store",
            "description": "Store a secret in the encrypted vault.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Secret label"},
                    "secret": {"type": "string", "description": "Secret value"},
                    "category": {"type": "string", "description": "Category (default: general)"},
                },
                "required": ["label", "secret"],
            },
        },
        {
            "name": "think_out_loud",
            "description": "Use this to reason through a complex problem step by step. Output goes to the runtime log.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your reasoning"},
                },
                "required": ["thought"],
            },
        },
        {
            "name": "done",
            "description": "Signal that the current task is complete. Include a summary.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What was accomplished"},
                },
                "required": ["summary"],
            },
        },
    ]


# ── Tool Execution Engine ───────────────────────────────────

class ToolExecutor:
    """Executes tools directly — no MCP overhead."""

    def __init__(self, brain: Brain):
        self.brain = brain
        self._vault = None

    def _get_vault(self):
        if self._vault is None:
            from watty.vault import Vault
            self._vault = Vault()
        return self._vault

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return handler(tool_input)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

    def _tool_memory_recall(self, params: dict) -> str:
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        results = self.brain.recall(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] (score: {r['score']:.3f}, via: {r['provider']})\n{r['content'][:500]}")
        return f"Found {len(results)} memories:\n\n" + "\n\n---\n\n".join(formatted)

    def _tool_memory_store(self, params: dict) -> str:
        content = params.get("content", "")
        provider = params.get("provider", "runtime")
        chunks = self.brain.store_memory(content, provider=provider)
        return f"Stored as {chunks} chunk(s)."

    def _tool_execute_code(self, params: dict) -> str:
        import subprocess, platform
        code = params.get("code", "")
        language = params.get("language", "python")
        timeout = params.get("timeout", 60)

        is_win = platform.system() == "Windows"
        if language == "bash" and is_win:
            language = "powershell"

        cmds = {
            "python": ["python", "-S", "-c", code],
            "node": ["node", "-e", code],
            "powershell": ["powershell", "-NoProfile", "-Command", code],
            "bash": ["/bin/bash", "-c", code],
        }
        cmd = cmds.get(language)
        if not cmd:
            return json.dumps({"error": f"Unknown language: {language}"})

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            output = r.stdout[:5000]
            if r.stderr:
                output += f"\nSTDERR:\n{r.stderr[:2000]}"
            return output if output.strip() else f"(exit code {r.returncode}, no output)"
        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s"

    def _tool_file_read(self, params: dict) -> str:
        path = params.get("path", "")
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > 10000:
                return content[:10000] + f"\n\n... (truncated, {len(content)} total chars)"
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _tool_file_write(self, params: dict) -> str:
        path = params.get("path", "")
        content = params.get("content", "")
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"

    def _tool_web_search(self, params: dict) -> str:
        import requests
        query = params.get("query", "")
        try:
            # Try via OpenClaw gateway
            from watty.tools_agent import _invoke_tool
            result = _invoke_tool("web_search", {"query": query}, timeout=15)
            return json.dumps(result, indent=2)[:3000]
        except Exception:
            return json.dumps({"error": "Web search unavailable (OpenClaw not running)"})

    def _tool_web_fetch(self, params: dict) -> str:
        import requests
        url = params.get("url", "")
        try:
            from watty.tools_agent import _invoke_tool
            result = _invoke_tool("web_fetch", {"url": url}, timeout=30)
            raw = result.get("raw", json.dumps(result))
            return str(raw)[:5000]
        except Exception:
            return json.dumps({"error": "Web fetch unavailable"})

    def _tool_speak(self, params: dict) -> str:
        text = params.get("text", "")
        try:
            from watty.tools_voice import _speak
            _speak(text)
            return f"Spoke: {text[:100]}"
        except Exception as e:
            return f"TTS error: {e}"

    def _tool_send_message(self, params: dict) -> str:
        message = params.get("message", "")
        try:
            from watty.tools_agent import _invoke_tool, HUGO_JID
            result = _invoke_tool("message", {
                "action": "send",
                "target": HUGO_JID,
                "message": message,
            }, timeout=15)
            return json.dumps(result)
        except Exception as e:
            return f"Message send error: {e}"

    def _tool_vault_get(self, params: dict) -> str:
        vault = self._get_vault()
        result = vault.retrieve(params.get("label", ""))
        if "error" in result:
            return f"Vault error: {result['error']}"
        return f"Label: {result['label']}\nValue: {result['secret']}\nCategory: {result['category']}"

    def _tool_vault_store(self, params: dict) -> str:
        vault = self._get_vault()
        result = vault.store(
            params.get("label", ""),
            params.get("secret", ""),
            category=params.get("category", "general"),
        )
        if "error" in result:
            return f"Vault error: {result['error']}"
        return "Stored in vault."

    def _tool_think_out_loud(self, params: dict) -> str:
        thought = params.get("thought", "")
        _log_runtime("think", thought[:500])
        return "Thought recorded."

    def _tool_done(self, params: dict) -> str:
        return f"DONE: {params.get('summary', '')}"


# ── Logging ─────────────────────────────────────────────────

def _log_runtime(event: str, detail: str = ""):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail[:1000],
    }
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[runtime] {event}: {detail[:200]}", file=sys.stderr, flush=True)


# ── The Agent Loop ──────────────────────────────────────────

class WattyRuntime:
    """The autonomous agent loop. The heart of the computer."""

    def __init__(self, api_key: str = None, model: str = None):
        ensure_home()
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        self.model = model or DEFAULT_MODEL
        self.client = anthropic.Anthropic(api_key=api_key)
        self.brain = Brain()
        self.executor = ToolExecutor(self.brain)
        self.messages: list[dict] = []
        self._running = False
        self._task_complete = False

        _log_runtime("init", f"Model: {self.model}")

    def run_task(self, task: str, max_turns: int = None) -> str:
        """
        Run a task to completion. The agent loops autonomously until done.
        Returns the final response.
        """
        max_turns = max_turns or MAX_TURNS
        self._task_complete = False
        self._running = True

        # Add the task as a user message
        self.messages.append({"role": "user", "content": task})
        _log_runtime("task_start", task[:300])

        system_prompt = _build_system_prompt(self.brain)
        tools = _get_tools()
        final_response = ""

        turn = 0
        while self._running and turn < max_turns and not self._task_complete:
            turn += 1
            _log_runtime("turn", f"{turn}/{max_turns}")

            try:
                # Call Claude
                start = time.time()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_prompt,
                    tools=tools,
                    messages=self.messages,
                )
                elapsed = time.time() - start
                _log_runtime("api_call", f"{elapsed:.1f}s, stop={response.stop_reason}")

                # Process the response
                assistant_content = []
                tool_calls = []
                text_parts = []

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        tool_calls.append(block)
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                # Store assistant message
                self.messages.append({"role": "assistant", "content": assistant_content})

                if text_parts:
                    final_response = "\n".join(text_parts)
                    print(f"\n[Watty] {final_response}", flush=True)

                # Execute tool calls
                if tool_calls:
                    tool_results = []
                    for tc in tool_calls:
                        _log_runtime("tool_call", f"{tc.name}: {json.dumps(tc.input)[:200]}")

                        # Check for "done" signal
                        if tc.name == "done":
                            self._task_complete = True
                            result_text = self.executor.execute(tc.name, tc.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result_text,
                            })
                            _log_runtime("task_done", tc.input.get("summary", ""))
                            break

                        result_text = self.executor.execute(tc.name, tc.input)
                        _log_runtime("tool_result", f"{tc.name}: {result_text[:200]}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result_text,
                        })

                    # Feed results back to Claude
                    self.messages.append({"role": "user", "content": tool_results})

                # If no tool calls and stop_reason is "end_turn", we're done
                elif response.stop_reason == "end_turn":
                    self._task_complete = True

            except anthropic.APIError as e:
                _log_runtime("api_error", str(e))
                final_response = f"API Error: {e}"
                break
            except Exception as e:
                _log_runtime("error", traceback.format_exc())
                final_response = f"Runtime Error: {e}"
                break

        self._running = False
        _log_runtime("task_end", f"Turns: {turn}, complete: {self._task_complete}")

        return final_response

    def chat(self, message: str) -> str:
        """Send a message and get a response (single turn, no autonomous loop)."""
        self.messages.append({"role": "user", "content": message})

        system_prompt = _build_system_prompt(self.brain)
        tools = _get_tools()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            tools=tools,
            messages=self.messages,
        )

        # Process response
        assistant_content = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

                # Execute and feed back
                result = self.executor.execute(block.name, block.input)
                # For chat mode, just append the result
                text_parts.append(f"[{block.name}] {result[:500]}")

        self.messages.append({"role": "assistant", "content": assistant_content})

        return "\n".join(text_parts)

    def stop(self):
        """Stop the current task."""
        self._running = False
        _log_runtime("stopped", "Manual stop")


# ── HTTP/WebSocket Server ───────────────────────────────────

async def _run_server(runtime: WattyRuntime, host: str = "0.0.0.0", port: int = 7778):
    """Run a simple HTTP server for external access to the runtime."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route
    import uvicorn

    async def handle_task(request):
        body = await request.json()
        task = body.get("task", "")
        max_turns = body.get("max_turns", MAX_TURNS)
        if not task:
            return JSONResponse({"error": "task is required"}, status_code=400)

        # Run in thread to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: runtime.run_task(task, max_turns=max_turns)
        )
        return JSONResponse({"result": result})

    async def handle_chat(request):
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: runtime.chat(message))
        return JSONResponse({"result": result})

    async def handle_status(request):
        return JSONResponse({
            "status": "running" if runtime._running else "idle",
            "model": runtime.model,
            "messages": len(runtime.messages),
            "brain_memories": runtime.brain.stats()["total_memories"],
        })

    async def handle_stop(request):
        runtime.stop()
        return JSONResponse({"status": "stopped"})

    app = Starlette(routes=[
        Route("/task", handle_task, methods=["POST"]),
        Route("/chat", handle_chat, methods=["POST"]),
        Route("/status", handle_status, methods=["GET"]),
        Route("/stop", handle_stop, methods=["POST"]),
    ])

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    _log_runtime("server_start", f"http://{host}:{port}")
    await server.serve()


# ── CLI Interface ───────────────────────────────────────────

def cli_loop(api_key: str = None, model: str = None):
    """Interactive CLI — talk to Watty directly."""
    runtime = WattyRuntime(api_key=api_key, model=model)

    print(f"\n{'='*50}")
    print(f"  WATTY RUNTIME — {runtime.model}")
    print(f"  Brain: {runtime.brain.stats()['total_memories']} memories")
    print(f"  Type a task. Type 'quit' to exit.")
    print(f"  Prefix with '!' for autonomous mode (loops until done).")
    print(f"{'='*50}\n")

    while True:
        try:
            user_input = input("\n[Hugo] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.startswith("!"):
            # Autonomous mode — run until done
            task = user_input[1:].strip()
            print(f"\n[Runtime] Autonomous mode: {task}\n")
            result = runtime.run_task(task)
            print(f"\n[Result] {result}")
        else:
            # Chat mode — single turn
            result = runtime.chat(user_input)
            print(f"\n[Watty] {result}")


# ── Entry Points ────────────────────────────────────────────

def main():
    """Main entry point — supports CLI and server modes."""
    import argparse

    parser = argparse.ArgumentParser(description="Watty Runtime — Autonomous Agent Loop")
    parser.add_argument("--mode", choices=["cli", "server", "task"], default="cli",
                        help="Run mode: cli (interactive), server (HTTP API), task (one-shot)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=7778, help="Server port (default: 7778)")
    parser.add_argument("--task", type=str, help="Task to run (for task mode)")
    parser.add_argument("--api-key", type=str, help="Anthropic API key (or set ANTHROPIC_API_KEY env)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try vault
        try:
            from watty.vault import Vault
            v = Vault()
            if v.is_initialized:
                print("Vault found but locked. Enter master password to retrieve API key.")
                import getpass
                pw = getpass.getpass("Master password: ")
                result = v.unlock(pw)
                if "error" not in result:
                    key_result = v.retrieve("anthropic_api_key")
                    if "error" not in key_result:
                        api_key = key_result["secret"]
                        print("API key loaded from vault.")
        except Exception:
            pass

    if not api_key:
        print("ERROR: No API key found.")
        print("Set ANTHROPIC_API_KEY environment variable, or store it in the vault:")
        print("  watty_vault(action='store', label='anthropic_api_key', secret='sk-...', category='api_key')")
        sys.exit(1)

    runtime = WattyRuntime(api_key=api_key, model=args.model)

    if args.mode == "cli":
        cli_loop(api_key=api_key, model=args.model)
    elif args.mode == "server":
        asyncio.run(_run_server(runtime, port=args.port))
    elif args.mode == "task":
        if not args.task:
            print("ERROR: --task is required in task mode")
            sys.exit(1)
        result = runtime.run_task(args.task)
        print(result)


if __name__ == "__main__":
    main()
