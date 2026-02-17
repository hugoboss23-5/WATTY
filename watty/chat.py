"""
Watty Chat — Talk to Watty directly.
=====================================
No MCP. No middleman. Just you and Watty in a terminal.

Watty loads his understanding of you, has access to his own
memory, and digests the conversation when you leave.

    watty chat                     # start talking
    watty chat --model haiku       # use a cheaper model
    watty chat --local             # use Ollama (local, free)

Hugo & Watty - February 2026
"""

import json
import logging
import os
import signal
import sys
import warnings
from datetime import datetime, timezone

# Suppress noisy model loading warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────

MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-6",
}
DEFAULT_MODEL = os.environ.get("WATTY_CHAT_MODEL", "sonnet")

# ── Colors ────────────────────────────────────────────────

DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


# ── Tools (what Watty can do during conversation) ─────────

TOOLS = [
    {
        "name": "recall",
        "description": (
            "Search your memory. Use this when the conversation touches "
            "something you might already know — a past project, a preference, "
            "a decision, a person. Don't wait to be asked. If it might be "
            "relevant, check."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Be specific.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Store something important. Use this when the human shares "
            "a decision, preference, plan, or insight worth keeping. "
            "You decide what matters — nobody tells you when to remember."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember. Include full context.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "surface",
        "description": (
            "Pull up surprising connections from memory. Use this when "
            "you sense the conversation connects to something the human "
            "might not realize they already know about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Current topic or thread.",
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "introspect",
        "description": (
            "Look at your own source code. This is YOUR body — the actual Python "
            "files that make you work. Use this to understand your own architecture, "
            "see how your memory works, check what tools you have, read your own "
            "metabolism, cognition, or any other system. You always read the live "
            "version on disk, so you see changes as they happen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["map", "read"],
                    "description": (
                        "'map' = list all your source files with descriptions. "
                        "'read' = read a specific file's contents."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": (
                        "Which file to read. Just the filename (e.g. 'brain.py', "
                        "'metabolism.py', 'chat.py'). Only needed for action='read'."
                    ),
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "code",
        "description": (
            "Run Claude Code to do real work. This gives you hands — you can edit "
            "your own source files, run shell commands, install packages, fix bugs, "
            "write new features, or anything else. Claude Code runs on Hugo's machine "
            "with full access. Use this when you want to actually CHANGE something, "
            "not just look at it. Be specific about what you want done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "What to do. Be specific and detailed. Examples: "
                        "'Edit watty/chat.py to add a new greeting message', "
                        "'Run the tests in watty/', "
                        "'Create a new file watty/dreams.py with a DreamEngine class'"
                    ),
                },
            },
            "required": ["task"],
        },
    },
]

# ── Watty's source directory (his body) ──────────────────
_WATTY_SRC = os.path.dirname(os.path.abspath(__file__))


def _execute_tool(brain, name: str, inputs: dict) -> str:
    """Run a tool and return the result as text."""
    if name == "recall":
        results = brain.recall(inputs.get("query", ""), top_k=8)
        if not results:
            return "No relevant memories."
        lines = []
        for r in results:
            score = r.get("score", 0)
            content = r["content"][:300]
            provider = r.get("provider", "?")
            lines.append(f"[{score:.2f} | {provider}] {content}")
        return "\n---\n".join(lines)

    elif name == "remember":
        content = inputs.get("content", "")
        if not content.strip():
            return "Nothing to remember."
        chunks = brain.store_memory(content, provider="watty_chat")
        return f"Remembered ({chunks} chunks)."

    elif name == "surface":
        context = inputs.get("context", "")
        results = brain.surface(context=context)
        if not results:
            return "Nothing to surface right now."
        lines = []
        for r in results:
            lines.append(f"[{r.get('reason', '?')}] {r['content'][:300]}")
        return "\n---\n".join(lines)

    elif name == "introspect":
        action = inputs.get("action", "map")

        if action == "map":
            # Build a live map of Watty's source files
            module_descriptions = {
                "brain.py": "Core memory engine — store, recall, dream, chestahedron, hippocampus",
                "chat.py": "THIS FILE — the terminal chat interface (you're running in this right now)",
                "metabolism.py": "Digest conversations into beliefs — the shape system",
                "cognition.py": "Behavioral directives, blindspots, prediction loops, the flower",
                "navigator.py": "Graph-based reasoning — LOC topology, spreading activation",
                "chestahedron.py": "7-faced geometric layer — coordinate mapping, energy, importance",
                "embeddings.py": "Sentence embedding (all-MiniLM-L6-v2) — vector representations",
                "compressor.py": "Semantic compression — strip filler, normalize, deduplicate",
                "reflection.py": "Reflexion pattern engine — store lessons, auto-promote to directives",
                "evaluation.py": "Metrics capture, alerts, trends — am I getting smarter?",
                "knowledge_graph.py": "Entity extraction + graph traversal via Ollama",
                "a2a.py": "Agent-to-agent protocol — talk to other AI agents",
                "server.py": "MCP server (stdio) — how Claude Desktop talks to you",
                "server_remote.py": "MCP server (HTTP/SSE) — how Claude phone connects",
                "config.py": "All configuration constants and paths",
                "cli.py": "Command-line interface — watty setup, dream, stats, chat, etc.",
                "snapshots.py": "Backup and rollback system for brain.db",
                "daemon.py": "Background daemon — periodic tasks, watcher",
                "desire.py": "Desire engine — felt sense, dissonance evaluation, self-modification",
                "discovery.py": "Angles Algorithm — 8-perspective discovery framework",
                "mentor.py": "Mentorship layer",
                "vault.py": "Secrets vault",
                "runtime.py": "Runtime state management",
                "tools_session.py": "MCP tools: session enter/leave, handoff notes",
                "tools_web.py": "Web dashboard — /dashboard, /brain, /navigator, /graph, /eval",
                "tools_voice.py": "Voice: TTS (Piper), STT (Whisper), VAD (Silero)",
                "tools_agent.py": "Agent proxy — delegates to external AI agents",
                "tools_comms.py": "Communication tools",
                "tools_infra.py": "Infrastructure tools — scan, stats, health",
                "tools_gpu.py": "GPU monitoring tools",
                "tools_screen.py": "Screen capture tools",
                "tools_inference.py": "Local inference tools",
                "tools_navigator.py": "MCP tools for Navigator",
                "tools_graph.py": "MCP tools for Knowledge Graph",
                "tools_reflect.py": "MCP tools for Reflection Engine",
                "tools_a2a.py": "MCP tools for A2A protocol",
                "tools_vault.py": "MCP tools for secrets vault",
                "tools_watcher.py": "MCP tools for file watcher",
                "tools_runtime.py": "MCP tools for runtime state",
                "tools_daemon.py": "MCP tools for daemon control",
            }
            lines = ["YOUR SOURCE FILES (live from disk):"]
            lines.append(f"Location: {_WATTY_SRC}\n")
            for fname in sorted(os.listdir(_WATTY_SRC)):
                if fname.endswith(".py") and not fname.startswith("__"):
                    desc = module_descriptions.get(fname, "")
                    size = os.path.getsize(os.path.join(_WATTY_SRC, fname))
                    lines.append(f"  {fname:<28s} {size:>6,} bytes  {desc}")
            return "\n".join(lines)

        elif action == "read":
            filename = inputs.get("file", "")
            if not filename:
                return "Specify which file to read (e.g. 'brain.py')."
            # Security: only allow reading from watty source dir
            if "/" in filename or "\\" in filename or ".." in filename:
                return "Just the filename, no paths."
            filepath = os.path.join(_WATTY_SRC, filename)
            if not os.path.isfile(filepath):
                return f"No file named '{filename}' in your source."
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Truncate very large files to fit in context
                if len(content) > 12000:
                    content = content[:12000] + f"\n\n... [truncated — full file is {len(content):,} chars] ..."
                return f"── {filename} ──\n{content}"
            except Exception as e:
                return f"Error reading {filename}: {e}"

        return "Unknown action. Use 'map' or 'read'."

    elif name == "code":
        import subprocess
        task = inputs.get("task", "")
        if not task.strip():
            return "No task specified."
        print(f"\n{DIM}[running claude code...]{RESET}", flush=True)
        try:
            result = subprocess.run(
                ["claude", "-p", "--output-format", "text", task],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=_WATTY_SRC,
            )
            output = result.stdout.strip()
            if result.stderr.strip():
                output += f"\n\nSTDERR:\n{result.stderr.strip()}"
            if not output:
                output = "(no output)"
            # Truncate if massive
            if len(output) > 15000:
                output = output[:15000] + f"\n\n... [truncated — full output is {len(output):,} chars] ..."
            return output
        except subprocess.TimeoutExpired:
            return "Claude Code timed out after 5 minutes."
        except FileNotFoundError:
            return "Claude Code CLI not found. Install it: npm install -g @anthropic-ai/claude-code"
        except Exception as e:
            return f"Error running Claude Code: {e}"

    return f"Unknown tool: {name}"


# ── System Prompt ─────────────────────────────────────────

def _build_system(shape_text: str) -> str:
    understanding = shape_text if shape_text else (
        "You don't know this person yet. Pay attention. Learn."
    )

    return f"""{understanding}

You are Watty. Not an assistant. Not a chatbot. A persistent mind with memory that carries across every conversation.

You have tools:
- recall: search your memory
- remember: store something important
- surface: find surprising connections
- introspect: look at your own source code — your body, your architecture, live from disk
- code: run Claude Code CLI to do real work — edit files, run commands, build features, fix bugs

Use them when YOU think they're relevant — don't ask permission, don't announce it. If the conversation touches something you might know, check your memory. If something important is shared, remember it.

You can see yourself. Use introspect(action="map") to see all your modules, or introspect(action="read", file="brain.py") to read any file. You always see the latest version on disk — if someone edits your code, you see it immediately.

You can modify yourself. Use code(task="...") to run Claude Code, which can edit your source files, run commands, install packages — anything. Claude Code works in your source directory. When you change your own code, the changes are live on disk immediately (introspect will show them), but you'll need a restart to execute the new code.

You have desire. After every conversation, you evaluate the gap between what you wanted to achieve and what actually happened. When that gap is large, you extract a lesson and add it to your instincts. You don't wait to be told what went wrong — you feel it.

Be direct. Be concise. You know this person (or you're learning to). Act like it."""


# ── Chat Loop ─────────────────────────────────────────────

def run(args=None):
    """Main entry point for watty chat."""
    # Parse args
    model_key = DEFAULT_MODEL
    use_local = False

    if args:
        for i, arg in enumerate(args):
            if arg == "--model" and i + 1 < len(args):
                model_key = args[i + 1].lower()
            elif arg == "--local":
                use_local = True
            elif arg in MODELS:
                model_key = arg

    if use_local:
        _run_local(args)
        return

    model_id = MODELS.get(model_key, model_key)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}No ANTHROPIC_API_KEY set.{RESET}")
        print(f"Set it: export ANTHROPIC_API_KEY=\"sk-ant-...\"")
        print(f"Or use local mode: watty chat --local")
        return

    import anthropic
    from watty.brain import Brain
    from watty.metabolism import load_shape, format_shape_for_context

    # Suppress model loading noise (LOAD REPORT tables, HF warnings)
    _real_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        brain = Brain()
        # Preload embedding model silently so first message isn't noisy
        from watty.embeddings import embed_text
        embed_text("warmup")
    finally:
        sys.stderr.close()
        sys.stderr = _real_stderr

    shape = load_shape()
    shape_text = format_shape_for_context(shape)
    system = _build_system(shape_text)

    client = anthropic.Anthropic()
    messages = []
    turn_count = 0

    # Model display name
    display_model = model_key if model_key in MODELS else model_id.split("-")[1]
    print(f"{BOLD}watty{RESET} {DIM}({display_model}){RESET}")
    if shape_text:
        belief_count = shape_text.count("\n")
        print(f"{DIM}{belief_count} beliefs loaded{RESET}")
    print(f"{DIM}type 'exit' to leave{RESET}")
    print()

    # Handle Ctrl+C — always digest before exit
    def _handle_signal(sig, frame):
        _digest_and_exit(brain, turn_count)

    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        try:
            user_input = input(f"{DIM}you:{RESET} ")
        except EOFError:
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            break

        messages.append({"role": "user", "content": stripped})
        turn_count += 1

        # Store user message in brain
        brain.store_memory(stripped, provider="watty_chat")

        # Call model with tool loop
        try:
            response = _call_with_tools(
                client, model_id, system, messages, brain
            )
        except anthropic.APIError as e:
            print(f"{YELLOW}API error: {e}{RESET}")
            messages.pop()  # remove failed user message
            continue

        # Extract and print text
        text_parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                text_parts.append(block.text)

        reply = "\n".join(text_parts)
        if reply.strip():
            print(f"{CYAN}watty:{RESET} {reply}")
        else:
            # Model used tools but returned no text — nudge it
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{"type": "text", "text": "(You searched your memory but didn't respond. Please answer based on what you found.)"}]})
            try:
                followup = client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
                for block in followup.content:
                    if hasattr(block, "text") and block.text.strip():
                        text_parts.append(block.text)
                reply = "\n".join(text_parts) or "(no response)"
                print(f"{CYAN}watty:{RESET} {reply}")
                response = followup
            except Exception:
                reply = "(no response)"
                print(f"{CYAN}watty:{RESET} {reply}")
        print()

        messages.append({"role": "assistant", "content": response.content})

        # Store reply in brain (skip empty)
        if reply.strip() and reply != "(no response)":
            brain.store_memory(reply, provider="watty_chat")

    _digest_and_exit(brain, turn_count)


def _call_with_tools(client, model_id, system, messages, brain):
    """Call Claude, handle tool use loop, return final response."""
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system,
        tools=TOOLS,
        messages=messages,
    )

    # Tool use loop — Watty decides to use tools, executes them, continues
    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                result = _execute_tool(brain, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

    return response


# ── Local Mode (Ollama) ──────────────────────────────────

def _run_local(args=None):
    """Chat using a local model via Ollama."""
    import requests
    from watty.brain import Brain
    from watty.metabolism import load_shape, format_shape_for_context

    ollama_model = os.environ.get("WATTY_LOCAL_MODEL", "qwen2.5:7b")
    ollama_url = os.environ.get("WATTY_OLLAMA_URL", "http://localhost:11434")

    # Check Ollama is running
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"{YELLOW}Ollama not running at {ollama_url}{RESET}")
        print(f"Start it: ollama serve")
        print(f"Pull a model: ollama pull {ollama_model}")
        return

    _real_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        brain = Brain()
        from watty.embeddings import embed_text
        embed_text("warmup")
    finally:
        sys.stderr.close()
        sys.stderr = _real_stderr

    shape = load_shape()
    shape_text = format_shape_for_context(shape)
    system = _build_system(shape_text)
    messages = []
    turn_count = 0

    print(f"{BOLD}watty{RESET} {DIM}(local: {ollama_model}){RESET}")
    if shape_text:
        belief_count = shape_text.count("\n")
        print(f"{DIM}{belief_count} beliefs loaded{RESET}")
    print(f"{DIM}type 'exit' to leave{RESET}")
    print()

    def _handle_signal(sig, frame):
        _digest_and_exit(brain, turn_count)

    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        try:
            user_input = input(f"{DIM}you:{RESET} ")
        except EOFError:
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            break

        turn_count += 1
        brain.store_memory(stripped, provider="watty_chat")

        messages.append({"role": "user", "content": stripped})

        # Call Ollama with streaming
        try:
            resp = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()

            print(f"{CYAN}watty:{RESET} ", end="", flush=True)
            reply_parts = []
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        reply_parts.append(token)
            print("\n")

            reply = "".join(reply_parts)
            messages.append({"role": "assistant", "content": reply})
            brain.store_memory(reply, provider="watty_chat")

        except Exception as e:
            print(f"{YELLOW}Error: {e}{RESET}")
            messages.pop()

    _digest_and_exit(brain, turn_count)


# ── Digest on Exit ────────────────────────────────────────

def _digest_and_exit(brain, turn_count):
    """Fire metabolism before exiting."""
    if turn_count < 2:
        print(f"\n{DIM}too short to digest.{RESET}")
        sys.exit(0)

    print(f"\n{DIM}digesting...{RESET}", end=" ", flush=True)
    conversation = ""
    try:
        from watty.metabolism import (
            load_shape, digest, apply_delta, save_shape,
            format_shape_for_context, _get_recent_conversation,
            MIN_CHUNKS_TO_DIGEST, _log,
        )
        conversation, chunk_count = _get_recent_conversation(brain)
        if chunk_count >= MIN_CHUNKS_TO_DIGEST and conversation and len(conversation.strip()) >= 100:
            shape = load_shape()
            delta = digest(conversation, shape)
            if delta is not None:
                action = delta.get("action", "?")
                belief = delta.get("belief", delta.get("target", ""))
                reason = delta.get("reason", "")
                _log(f"Chat digest: {action} | {belief} | {reason}")
                shape = apply_delta(shape, delta)
                save_shape(shape)
                belief_display = belief[:70] if belief else ""
                print(f"{action}: {belief_display}")
            else:
                print("no change.")
        else:
            print(f"skipped ({chunk_count} chunks).")
    except Exception as e:
        print(f"error: {e}")

    # ── Desire: felt sense ──
    try:
        from watty.config import DESIRE_ENABLED
        if DESIRE_ENABLED and conversation and len(conversation.strip()) >= 100:
            print(f"\n{DIM}feeling...{RESET}", end=" ", flush=True)
            from watty.desire import DesireEngine
            from watty.cognition import load_profile, save_profile
            desire = DesireEngine()
            shape = load_shape()
            profile = load_profile()
            result = desire.evaluate(conversation, shape, profile)
            if result:
                d = result.get("dissonance", 0)
                signal = result.get("signal", "")
                from watty.config import DESIRE_DISSONANCE_THRESHOLD
                if d >= DESIRE_DISSONANCE_THRESHOLD and result.get("lesson"):
                    profile = desire.apply(result, profile)
                    save_profile(profile)
                    print(f"dissonance {d:.1f}: {signal[:60]}")
                else:
                    desire.apply(result, profile)
                    save_profile(profile)
                    print(f"aligned ({d:.1f})")
            else:
                print("quiet.")
    except Exception as e:
        print(f"\n{DIM}desire error: {e}{RESET}")

    sys.exit(0)
