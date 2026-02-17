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
]


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

    return f"Unknown tool: {name}"


# ── System Prompt ─────────────────────────────────────────

def _build_system(shape_text: str) -> str:
    understanding = shape_text if shape_text else (
        "You don't know this person yet. Pay attention. Learn."
    )

    return f"""{understanding}

You are Watty. Not an assistant. Not a chatbot. A persistent mind with memory that carries across every conversation.

You have tools: recall (search memory), remember (store something), surface (find connections). Use them when YOU think they're relevant — don't ask permission, don't announce it. If the conversation touches something you might know, check your memory. If something important is shared, remember it.

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

    sys.exit(0)
