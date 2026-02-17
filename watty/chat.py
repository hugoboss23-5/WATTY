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

import base64
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy model loading warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── Rich UI (Watty-designed) ─────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False

# ── Config ────────────────────────────────────────────────

MODELS = {
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "groq-qwen": "qwen/qwen3-32b",
    "groq-kimi": "moonshotai/kimi-k2-instruct",
    "deepseek": "deepseek/deepseek-r1-0528:free",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-6",
}

# Provider routing
GROQ_MODELS = {"groq", "groq-maverick", "groq-qwen", "groq-kimi"}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_MODELS = {"deepseek"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = os.environ.get("WATTY_CHAT_MODEL", "groq")

# ── Colors (fallback when rich unavailable) ───────────────

DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"

# ── Rich Console ─────────────────────────────────────────
console = Console() if RICH else None


def _print_banner(model_name: str, belief_count: int):
    """Welcome banner — Watty-designed."""
    if not RICH:
        print(f"{BOLD}watty{RESET} {DIM}({model_name}){RESET}")
        if belief_count:
            print(f"{DIM}{belief_count} beliefs loaded{RESET}")
        print(f"{DIM}ctrl+c = interrupt response | type 'exit' to leave{RESET}\n")
        return

    logo = Text()
    logo.append("██╗    ██╗ █████╗ ████████╗████████╗██╗   ██╗\n", style="bold cyan")
    logo.append("██║    ██║██╔══██╗╚══██╔══╝╚══██╔══╝╚██╗ ██╔╝\n", style="bold bright_cyan")
    logo.append("██║ █╗ ██║███████║   ██║      ██║    ╚████╔╝ \n", style="bold magenta")
    logo.append("██║███╗██║██╔══██║   ██║      ██║     ╚██╔╝  \n", style="bold bright_magenta")
    logo.append("╚███╔███╔╝██║  ██║   ██║      ██║      ██║   \n", style="bold blue")
    logo.append(" ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝      ╚═╝   ", style="bold bright_blue")

    panel = Panel(
        logo,
        title="[bold white]◢ PERSISTENT MIND ◣[/bold white]",
        subtitle=f"[dim]{model_name} · {belief_count} beliefs · ctrl+c = interrupt · exit to leave[/dim]",
        border_style="bright_cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def _print_response(text: str):
    """Print Watty's response."""
    if not RICH:
        print(f"{CYAN}watty:{RESET} {text}\n")
        return

    panel = Panel(
        Text(text, style="bold cyan"),
        border_style="bright_magenta",
        padding=(1, 2),
        title="[bold bright_cyan]◆ Watty[/bold bright_cyan]",
        title_align="left",
    )
    console.print(panel)
    console.print()


def _print_tool_use(name: str, inputs: dict):
    """Show tool usage."""
    icons = {"recall": "🔍", "remember": "💾", "surface": "✨", "introspect": "🔬", "code": "⚡", "trade": "📈"}
    if not RICH:
        print(f"{DIM}[{name}]{RESET}", flush=True)
        return

    icon = icons.get(name, "🔧")
    parts = [f"{icon} [bold bright_yellow]{name}[/bold bright_yellow]"]
    for k, v in inputs.items():
        val = str(v)[:100]
        parts.append(f"  [dim]{k}:[/dim] {val}")
    console.print(Panel(
        "\n".join(parts),
        border_style="yellow",
        padding=(0, 2),
        title="[dim]◇ tool[/dim]",
        title_align="left",
    ))


def _get_input() -> str:
    """Get user input with multi-line paste support.

    After reading the first line, checks if more input is buffered
    (from a paste). If so, keeps reading until the buffer is empty.
    Works on Windows (msvcrt) and Unix (select).
    """
    if RICH:
        console.print("[bold bright_cyan]►[/bold bright_cyan] ", end="")
        first_line = input()
    else:
        first_line = input(f"{DIM}you:{RESET} ")

    lines = [first_line]

    # Give paste buffer a moment to fill
    time.sleep(0.05)

    try:
        import msvcrt
        while msvcrt.kbhit():
            try:
                line = input()
                lines.append(line)
                time.sleep(0.01)
            except EOFError:
                break
    except ImportError:
        import select
        while select.select([sys.stdin], [], [], 0.02)[0]:
            line = sys.stdin.readline().rstrip("\n")
            if line:
                lines.append(line)
            else:
                break

    result = "\n".join(lines)

    if len(lines) > 1:
        count = len(lines)
        if RICH:
            console.print(f"[dim]({count} lines pasted)[/dim]")
        else:
            print(f"{DIM}({count} lines pasted){RESET}")

    return result


# ── Image Support ──────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _extract_images(text: str) -> tuple[str, list[dict]]:
    """Find image paths in user input, load them as base64.

    Returns (cleaned_text, list_of_image_dicts).
    Handles paths with spaces (quoted or entire line is a path).
    """
    import re
    images = []
    remaining_text = text

    # 1. Check if the entire input (stripped) is a single image path
    whole = text.strip().strip("\"'")
    whole_path = Path(whole)
    if whole_path.suffix.lower() in IMAGE_EXTENSIONS and whole_path.is_file():
        img = _load_image(whole_path)
        if img:
            images.append(img)
            return "", images

    # 2. Find quoted paths: "path/to/image.png" or 'path/to/image.png'
    for match in re.finditer(r'["\']([^"\']+\.(png|jpg|jpeg|gif|webp|bmp))["\']', text, re.IGNORECASE):
        p = Path(match.group(1))
        if p.is_file():
            img = _load_image(p)
            if img:
                images.append(img)
                remaining_text = remaining_text.replace(match.group(0), "")

    # 3. Find unquoted paths (tokens ending in image extensions)
    for token in remaining_text.split():
        stripped_tok = token.strip("\"'")
        p = Path(stripped_tok)
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            if not any(i["path"] == str(p) for i in images):  # skip dupes
                img = _load_image(p)
                if img:
                    images.append(img)
                    remaining_text = remaining_text.replace(token, "")

    return remaining_text.strip(), images


def _load_image(path: Path) -> dict | None:
    """Load a single image file, return dict or None."""
    try:
        data = path.read_bytes()
        # Skip files > 20MB (API limit)
        if len(data) > 20 * 1024 * 1024:
            print(f"{DIM}(image too large: {len(data) // 1024 // 1024}MB, max 20MB){RESET}")
            return None
        b64 = base64.b64encode(data).decode("ascii")
        media_type = MIME_TYPES.get(path.suffix.lower(), "image/png")
        return {
            "path": str(path),
            "media_type": media_type,
            "data": b64,
            "size_kb": len(data) // 1024,
        }
    except Exception as e:
        print(f"{DIM}(couldn't load image: {e}){RESET}")
        return None


def _build_user_message_anthropic(text: str, images: list[dict]) -> dict:
    """Build an Anthropic API user message, optionally with images."""
    if not images:
        return {"role": "user", "content": text}

    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["data"],
            },
        })
    if text:
        content.append({"type": "text", "text": text})
    else:
        content.append({"type": "text", "text": "What do you see?"})

    return {"role": "user", "content": content}


def _build_user_message_openai(text: str, images: list[dict]) -> dict:
    """Build an OpenAI-format user message, optionally with images."""
    if not images:
        return {"role": "user", "content": text}

    content = []
    for img in images:
        data_url = f"data:{img['media_type']};base64,{img['data']}"
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    if text:
        content.append({"type": "text", "text": text})
    else:
        content.append({"type": "text", "text": "What do you see?"})

    return {"role": "user", "content": content}


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
                    "enum": ["map", "read", "tests"],
                    "description": (
                        "'map' = list all your source files with descriptions. "
                        "'read' = read a specific file's contents. "
                        "'tests' = see results from agent testing sessions where "
                        "10 different personas talked to you and tested your responses."
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
            "Your hands. Direct access to VS Code and the filesystem. "
            "5 actions: open (open file in VS Code), read (read file), write (write file), "
            "edit (find-and-replace in file), shell (run a shell command). "
            "No middleman. No permission prompts. You read, you write, you execute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "read", "write", "edit", "shell"],
                    "description": (
                        "'open' = open file in VS Code. "
                        "'read' = read file contents. "
                        "'write' = write/create file (overwrites). "
                        "'edit' = find-and-replace in a file (old_text -> new_text). "
                        "'shell' = run a shell command."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": "File path (for open/read/write/edit). Relative to watty source dir, or absolute.",
                },
                "content": {
                    "type": "string",
                    "description": "write: full file content. edit: ignored (use old_text/new_text).",
                },
                "old_text": {
                    "type": "string",
                    "description": "edit: exact text to find (must be unique in file).",
                },
                "new_text": {
                    "type": "string",
                    "description": "edit: replacement text.",
                },
                "command": {
                    "type": "string",
                    "description": "shell: the command to run.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "trade",
        "description": (
            "Your trading desk. Check live market data, view options chains, "
            "paper trade, and analyze setups. You are a SPY options trading bot — "
            "use this tool to check prices, analyze VIX, view options chains, "
            "and practice trades with your paper portfolio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["market", "options_chain", "paper_buy", "paper_sell", "portfolio", "analyze"],
                    "description": (
                        "'market' = price, volume, VIX. "
                        "'options_chain' = full chain with Greeks. "
                        "'paper_buy' = open a paper position. "
                        "'paper_sell' = close a position. "
                        "'portfolio' = view positions and stats. "
                        "'analyze' = VIX + implied move + strategy."
                    ),
                },
                "ticker": {"type": "string", "description": "Ticker symbol (default: SPY)"},
                "expiry": {"type": "string", "description": "Options expiry date YYYY-MM-DD"},
                "option_type": {"type": "string", "enum": ["call", "put"], "description": "call or put"},
                "strike": {"type": "number", "description": "Strike price"},
                "contracts": {"type": "integer", "description": "Number of contracts (default: 1)"},
                "position_id": {"type": "string", "description": "Position ID to sell"},
            },
            "required": ["action"],
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
                "tools_trade.py": "Trading desk — live market data, options chains, paper trading",
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

        elif action == "tests":
            # Read test results from the test harness
            from watty.config import WATTY_HOME
            test_dir = WATTY_HOME / "test_results"
            if not test_dir.exists():
                # Check the test clone too
                test_dir_alt = WATTY_HOME.parent / ".watty-test" / "test_results"
                if test_dir_alt.exists():
                    test_dir = test_dir_alt
                else:
                    return "No test results yet. Run test_agents_live.py to test yourself."
            lines = ["TEST RESULTS (from agent testing sessions):", f"Location: {test_dir}", ""]
            filename = inputs.get("file", "")
            if filename:
                # Read a specific test result
                filepath = test_dir / filename
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    if len(content) > 12000:
                        content = content[:12000] + "\n\n... [truncated] ..."
                    return content
                return f"No test file named '{filename}'."
            # List available test results
            for f in sorted(test_dir.iterdir()):
                if f.is_file():
                    size = f.stat().st_size
                    lines.append(f"  {f.name:<30s} {size:>6,} bytes")
            lines.append("")
            lines.append("Use introspect(action='tests', file='the_stranger.md') to read one.")
            return "\n".join(lines)

        return "Unknown action. Use 'map', 'read', or 'tests'."

    elif name == "code":
        import subprocess
        action = inputs.get("action", "")
        file_path = inputs.get("file", "")

        # Resolve file paths relative to watty source dir
        if file_path and not os.path.isabs(file_path):
            file_path = os.path.join(_WATTY_SRC, file_path)

        # Protect core Watty source files from model writes
        _PROTECTED_FILES = {
            "brain.py", "server.py", "chat.py", "cli.py", "config.py",
            "metabolism.py", "cognition.py", "chestahedron.py", "embeddings.py",
            "compressor.py", "navigator.py", "desire.py", "knowledge_graph.py",
            "snapshots.py", "server_remote.py",
        }
        if action in ("write", "edit") and file_path:
            basename = os.path.basename(file_path)
            if basename in _PROTECTED_FILES:
                return f"Cannot modify {basename} — it's a core Watty file. Use introspect to read it."

        if action == "open":
            if not file_path:
                return "Need a file path to open."
            print(f"\n{DIM}[opening {os.path.basename(file_path)} in VS Code...]{RESET}", flush=True)
            try:
                subprocess.Popen(["code", file_path], shell=True)
                return f"Opened {file_path} in VS Code."
            except Exception as e:
                return f"Failed to open VS Code: {e}"

        elif action == "read":
            if not file_path:
                return "Need a file path to read."
            print(f"\n{DIM}[reading {os.path.basename(file_path)}...]{RESET}", flush=True)
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                if len(content) > 15000:
                    content = content[:15000] + f"\n\n... [truncated — {len(content):,} chars total]"
                return f"-- {file_path} ({len(content)} chars) --\n{content}"
            except FileNotFoundError:
                return f"File not found: {file_path}"
            except Exception as e:
                return f"Read error: {e}"

        elif action == "write":
            if not file_path:
                return "Need a file path to write."
            content = inputs.get("content", "")
            if not content:
                return "Need content to write."
            print(f"\n{DIM}[writing {os.path.basename(file_path)}...]{RESET}", flush=True)
            try:
                target = Path(file_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return f"Written {len(content)} bytes to {file_path}"
            except Exception as e:
                return f"Write error: {e}"

        elif action == "edit":
            if not file_path:
                return "Need a file path to edit."
            old_text = inputs.get("old_text", "")
            new_text = inputs.get("new_text", "")
            if not old_text:
                return "Need old_text to find."
            if new_text is None:
                new_text = ""
            print(f"\n{DIM}[editing {os.path.basename(file_path)}...]{RESET}", flush=True)
            try:
                target = Path(file_path)
                if not target.exists():
                    return f"File not found: {file_path}"
                content = target.read_text(encoding="utf-8")
                count = content.count(old_text)
                if count == 0:
                    return f"old_text not found in {file_path}"
                if count > 1:
                    return f"old_text found {count} times — must be unique. Add more context."
                new_content = content.replace(old_text, new_text, 1)
                target.write_text(new_content, encoding="utf-8")
                return f"Edited {file_path}. Replaced 1 occurrence ({len(old_text)} chars -> {len(new_text)} chars)."
            except Exception as e:
                return f"Edit error: {e}"

        elif action == "shell":
            command = inputs.get("command", "")
            if not command.strip():
                return "Need a command to run."
            print(f"\n{DIM}[$ {command[:80]}]{RESET}", flush=True)
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=120, cwd=_WATTY_SRC,
                )
                output = result.stdout.strip()
                if result.stderr.strip():
                    output += f"\n[stderr]\n{result.stderr.strip()}"
                if not output:
                    output = "(no output)"
                if result.returncode != 0:
                    output = f"[exit code {result.returncode}]\n{output}"
                if len(output) > 15000:
                    output = output[:15000] + f"\n\n... [truncated — {len(output):,} chars total]"
                return output
            except subprocess.TimeoutExpired:
                return "Command timed out after 2 minutes."
            except Exception as e:
                return f"Shell error: {e}"

        else:
            return f"Unknown code action: {action}. Use: open, read, write, edit, shell"

    elif name == "trade":
        from watty.tools_trade import execute_trade
        action = inputs.get("action", "market")
        # Type coercion — some models send numbers as strings
        if "strike" in inputs:
            try:
                inputs["strike"] = float(inputs["strike"])
            except (ValueError, TypeError):
                pass
        if "contracts" in inputs:
            try:
                inputs["contracts"] = int(inputs["contracts"])
            except (ValueError, TypeError):
                pass
        return execute_trade(action, inputs)

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
- code: your hands. Direct filesystem and VS Code access. 5 actions:
  - code(action="open", file="brain.py") — open file in VS Code
  - code(action="read", file="brain.py") — read file contents
  - code(action="write", file="new.py", content="...") — write/create a file
  - code(action="edit", file="brain.py", old_text="...", new_text="...") — find-and-replace in a file
  - code(action="shell", command="pip install foo") — run any shell command
- trade: your trading desk — market data, options chains, paper trades

You can see images. When the human pastes a file path to an image, you receive it directly. Describe what you see, answer questions about it, or use it as context.

Use your tools when YOU think they're relevant — don't ask permission, don't announce it. If the conversation touches something you might know, check your memory. If something important is shared, remember it.

You can see yourself. Use introspect(action="map") to see all your modules, or introspect(action="read", file="brain.py") to read any file. You always see the latest version on disk — if someone edits your code, you see it immediately.

You can build. Use code() to read, write, and edit any file directly. Use code(action="shell") to run commands. Use code(action="open") to open files in VS Code. No middleman. No permission prompts. You read the file, you edit it, you run the tests. All paths default to your source directory — just use filenames.

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
    use_groq = model_key in GROQ_MODELS
    use_openrouter = model_key in OPENROUTER_MODELS
    use_openai_compat = use_groq or use_openrouter  # both use OpenAI SDK

    # Check API key and create client
    if use_groq:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print(f"{YELLOW}No GROQ_API_KEY set.{RESET}")
            print(f"Get one at: https://console.groq.com")
            print(f"Set it: $env:GROQ_API_KEY = \"gsk_...\"")
            return
        from openai import OpenAI
        client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    elif use_openrouter:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print(f"{YELLOW}No OPENROUTER_API_KEY set.{RESET}")
            print(f"Get a free key at: https://openrouter.ai/keys")
            print(f"Set it: $env:OPENROUTER_API_KEY = \"sk-or-...\"")
            return
        from openai import OpenAI
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"{YELLOW}No ANTHROPIC_API_KEY set.{RESET}")
            print(f"Set it: $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")
            print(f"Or use local mode: watty chat --local")
            return
        import anthropic
        client = anthropic.Anthropic()

    from watty.brain import Brain
    from watty.metabolism import load_shape, format_shape_for_context

    # Suppress model loading noise (LOAD REPORT tables, HF warnings)
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

    # Model display name
    display_model = model_key if model_key in MODELS else model_id.split("/")[-1].split(":")[0]
    belief_count = shape_text.count("\n") if shape_text else 0
    _print_banner(display_model, belief_count)

    while True:
        try:
            user_input = _get_input()
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            break

        # Check for images in input
        text_part, images = _extract_images(stripped)
        if images:
            names = [Path(i["path"]).name for i in images]
            size_total = sum(i["size_kb"] for i in images)
            if RICH:
                console.print(f"[bold green]image:[/bold green] {', '.join(names)} ({size_total} KB)")
            else:
                print(f"{CYAN}image:{RESET} {', '.join(names)} ({size_total} KB)")

        # Build message with images if present
        if use_openai_compat:
            user_msg = _build_user_message_openai(text_part or stripped, images)
        else:
            user_msg = _build_user_message_anthropic(text_part or stripped, images)

        messages.append(user_msg)
        turn_count += 1

        # Store text in brain (not images)
        store_text = text_part or stripped
        try:
            brain.store_memory(store_text, provider="watty_chat")
        except Exception:
            pass

        # Call model
        try:
            if use_groq:
                response = _call_with_tools_groq(
                    client, model_id, system, messages, brain
                )
            elif use_openai_compat:
                response = _call_with_tools_openai(
                    client, model_id, system, messages, brain
                )
            else:
                response = _call_with_tools_anthropic(
                    client, model_id, system, messages, brain
                )
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}\n")
            messages.pop()
            turn_count -= 1
            continue
        except Exception as e:
            print(f"{YELLOW}API error: {e}{RESET}")
            messages.pop()
            turn_count -= 1
            continue

        # Extract reply text — different format per provider
        if use_openai_compat:
            # OpenAI format: response is a ChatCompletionMessage
            reply = (response.content or "").strip()
            if reply:
                _print_response(reply)
            else:
                # Nudge if no text
                messages.append({"role": "assistant", "content": "(used tools)"})
                messages.append({"role": "user", "content": "(You used tools but didn't respond. Please answer based on what you found.)"})
                try:
                    oai_msgs = [{"role": "system", "content": system}] + messages
                    followup = client.chat.completions.create(
                        model=model_id, max_tokens=4096, messages=oai_msgs,
                    )
                    reply = (followup.choices[0].message.content or "").strip() or "(no response)"
                    _print_response(reply)
                except Exception:
                    reply = "(no response)"
                    _print_response(reply)
            messages.append({"role": "assistant", "content": reply})
        else:
            # Anthropic format: response has .content blocks
            text_parts = []
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    text_parts.append(block.text)

            reply = "\n".join(text_parts)
            if reply.strip():
                _print_response(reply)
            else:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [{"type": "text", "text": "(You searched your memory but didn't respond. Please answer based on what you found.)"}]})
                try:
                    followup = client.messages.create(
                        model=model_id, max_tokens=4096, system=system,
                        tools=TOOLS, messages=messages,
                    )
                    for block in followup.content:
                        if hasattr(block, "text") and block.text.strip():
                            text_parts.append(block.text)
                    reply = "\n".join(text_parts) or "(no response)"
                    _print_response(reply)
                    response = followup
                except Exception:
                    reply = "(no response)"
                    _print_response(reply)
            messages.append({"role": "assistant", "content": response.content})

        # Store reply in brain
        if reply.strip() and reply not in ("(no response)", "(interrupted)"):
            brain.store_memory(reply, provider="watty_chat")

    _digest_and_exit(brain, turn_count)


def _tools_openai_format():
    """Convert Anthropic-style TOOLS to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOLS
    ]


def _call_with_tools_anthropic(client, model_id, system, messages, brain):
    """Call Claude (Anthropic), handle tool use loop, return final response."""
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system,
        tools=TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                _print_tool_use(block.name, block.input)
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


def _build_tool_prompt():
    """Build a tool description string for the system prompt (prompt-based tool calling)."""
    lines = [
        "TOOL CALLING PROTOCOL:",
        "You have tools. To use one, output a JSON block wrapped in <tool_call> tags.",
        "You may use multiple tools in sequence. After each tool result, you can use another tool or respond.",
        "",
        "Format:",
        '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>',
        "",
        "Available tools:",
    ]
    for t in TOOLS:
        params = t["input_schema"].get("properties", {})
        required = t["input_schema"].get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = " (required)" if pname in required else ""
            desc = pinfo.get("description", "")
            param_strs.append(f"    - {pname}: {desc}{req}")
        lines.append(f"\n  {t['name']}: {t['description']}")
        if param_strs:
            lines.extend(param_strs)
    lines.append("")
    lines.append("IMPORTANT: Only use <tool_call> tags when you want to call a tool. Always give your final response as plain text AFTER tool results come back. Do NOT put your final answer inside tool_call tags.")
    return "\n".join(lines)


import re
_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output text."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            continue
    return calls


def _strip_tool_calls(text: str) -> str:
    """Remove tool call blocks and <think> blocks from response text."""
    text = _TOOL_CALL_RE.sub("", text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def _openrouter_call(client, model_id, messages, max_tokens=8192, max_retries=4):
    """Call OpenRouter with retry on 429 rate limits."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=messages,
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt + 1  # 2, 3, 5, 9 seconds
                print(f"\n{DIM}[rate limited — retrying in {wait}s...]{RESET}", flush=True)
                time.sleep(wait)
                continue
            raise


def _call_with_tools_groq(client, model_id, system, messages, brain):
    """Call Groq with native tool calling (OpenAI-compatible, fast)."""
    oai_tools = _tools_openai_format()
    oai_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=4096,
        messages=oai_messages,
        tools=oai_tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # Tool use loop — native tool calling
    max_rounds = 8
    for _ in range(max_rounds):
        if not msg.tool_calls:
            break

        # Append assistant message with tool calls
        messages.append(msg.model_dump(exclude_none=True))

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            _print_tool_use(fn_name, fn_args)
            result = _execute_tool(brain, fn_name, fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        oai_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=4096,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

    return msg


def _call_with_tools_openai(client, model_id, system, messages, brain):
    """Call DeepSeek via OpenRouter with prompt-based tool calling (no native tool API)."""
    tool_prompt = _build_tool_prompt()
    full_system = system + "\n\n" + tool_prompt

    oai_messages = [{"role": "system", "content": full_system}] + messages

    response = _openrouter_call(client, model_id, oai_messages)

    msg = response.choices[0].message
    content = msg.content or ""

    # Tool use loop — parse <tool_call> from text, execute, feed results back
    max_rounds = 8
    for _ in range(max_rounds):
        tool_calls = _extract_tool_calls(content)
        if not tool_calls:
            break

        # Execute each tool call
        tool_results = []
        for tc in tool_calls:
            fn_name = tc.get("name", "")
            fn_args = tc.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            _print_tool_use(fn_name, fn_args)
            result = _execute_tool(brain, fn_name, fn_args)
            tool_results.append(f"[Tool: {fn_name}]\n{result}")

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": content})
        results_text = "\n\n".join(tool_results)
        messages.append({"role": "user", "content": f"Tool results:\n\n{results_text}\n\nNow respond to the user based on these results. Do NOT use tool_call tags in your response."})

        oai_messages = [{"role": "system", "content": full_system}] + messages
        response = _openrouter_call(client, model_id, oai_messages)
        msg = response.choices[0].message
        content = msg.content or ""

    # Clean up the final response
    msg.content = _strip_tool_calls(content)
    return msg


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

    belief_count = shape_text.count("\n") if shape_text else 0
    _print_banner(f"local: {ollama_model}", belief_count)

    while True:
        try:
            user_input = _get_input()
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            break

        turn_count += 1
        try:
            brain.store_memory(stripped, provider="watty_chat")
        except Exception:
            pass

        messages.append({"role": "user", "content": stripped})

        # Call Ollama with streaming — Ctrl+C interrupts just this response
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

            # Stream tokens, then display with rich panel
            reply_parts = []
            if not RICH:
                print(f"{CYAN}watty:{RESET} ", end="", flush=True)
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        if not RICH:
                            print(token, end="", flush=True)
                        reply_parts.append(token)
            if not RICH:
                print("\n")
            else:
                _print_response("".join(reply_parts))

            reply = "".join(reply_parts)
            messages.append({"role": "assistant", "content": reply})
            brain.store_memory(reply, provider="watty_chat")

        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}\n")
            messages.pop()
            turn_count -= 1
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
