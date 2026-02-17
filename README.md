# Watty

**Every AI you talk to forgets you. Watty fixes that.**

Install Watty once and every Claude conversation — Desktop, CLI, phone — starts with an understanding of who you are. Not a profile. Not a fact sheet. An actual developing understanding that grows with every conversation.

## What Watty Actually Does

When you close a conversation, Watty digests it. It calls a small AI model and asks one question: *"What single thing should change about how I understand this person?"* The answer is one delta — one belief added, strengthened, weakened, revised, or removed. That delta gets applied to a structure called the **shape**.

Next conversation, Claude reads the shape before your first message. It looks like this:

```
What I know about you:
You think in spatial metaphors — structure over statistics, geometry over inference.
I've noticed you reject abstraction for its own sake — you demand working code before philosophy.
I think you communicate directly — action over explanation.
I'm not sure yet, but you might prefer building alone over collaborating on design.
```

No confidence scores. No metadata. Just natural language with conviction levels built into the phrasing. Claude absorbs this as understanding, not as instructions.

Over time, beliefs that keep getting confirmed become settled ("You think spatially."). Beliefs that get contradicted weaken and eventually die. The shape changes because your conversations change it — not because you told it to.

This is **memory as metabolism**, not memory as storage. Raw conversations are consumed. What remains is changed structure.

## Install

### Prerequisites

- Python 3.10+
- An Anthropic API key (for the metabolism digest — costs fractions of a cent per conversation)

### Setup

```bash
git clone https://github.com/hugoboss23-5/WATTY.git
cd WATTY
pip install -e .
watty setup
```

`watty setup` does everything: creates the data directory, scans your Documents/Desktop/Downloads, runs a dream cycle, and auto-configures Claude Desktop.

### Connect to Claude

**Claude Desktop** (auto-configured by setup, verify with):
```bash
# macOS
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
type %APPDATA%\Claude\claude_desktop_config.json
```

Should contain:
```json
{
  "mcpServers": {
    "watty": {
      "command": "watty",
      "args": ["serve"]
    }
  }
}
```

If not there, create it manually and restart Claude Desktop.

**Claude Code (CLI)**:
```bash
claude mcp add watty -- watty serve
```

**Phone (iOS/Android)**:
```bash
watty serve-remote --port 8765
```
Then add `http://YOUR_IP:8765/sse` as a custom connector in Claude mobile settings.

### Set your API key

The metabolism needs an Anthropic API key to digest conversations:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add this to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.) so it persists. Without it, everything works except the digest — conversations won't update the shape.

## How It Works

### Automatic triggers (no tool calls needed)

**Session start**: When Claude connects, Watty loads `~/.watty/shape.json` and injects your understanding into the conversation via the MCP protocol. Claude reads it before your first message. You don't ask for it. It's just there.

**Session end**: When the conversation ends (CLI process exits, Desktop goes idle for 5 minutes, phone disconnects), Watty pulls recent conversation chunks from the database and calls a small Claude model to extract one pattern. Not an event — a pattern. "You think spatially" is a pattern. "You asked about a UI layout" is an event. Events get ignored. Patterns become beliefs.

### The confidence system

New beliefs start at 0.5 (developing). Each confirmation strengthens them with diminishing returns — easy to go from 0.5 to 0.6, very hard to go from 0.9 to 1.0. Contradictions hit harder the more confident a belief is — one contradiction can drop a 0.9 belief below the settled threshold in a single hit.

Beliefs above 0.7 are "settled" — they render as declarative statements and are protected from being evicted when the belief cap (50) is reached. Beliefs below 0.1 are automatically removed.

Building conviction is slow. Shattering it is fast. That's how real understanding works.

### Everything else

Watty is also a full memory system with 33+ MCP tools:

- **Semantic search** across all your conversations and documents
- **File scanning** — point it at a folder, it reads everything worth reading
- **Dream cycles** — consolidation that promotes, decays, strengthens, and prunes memories
- **Knowledge graph** — entity extraction and graph traversal
- **Web dashboard** — visual brain explorer at `localhost:8765`
- **Voice** — local speech with Piper TTS and faster-whisper STT
- **Session handoffs** — notes passed between AI instances
- **Agent-to-agent protocol** — Watty can talk to other AI agents

But the metabolism is the core. Everything else is infrastructure. The shape is the product.

## CLI

```
watty serve            Start MCP server (default, runs on stdio)
watty serve-remote     Start HTTP server for phone/web
watty setup            First-time setup wizard
watty stats            Brain health check
watty recall "query"   Search memory
watty scan ~/path      Scan a directory
watty dream            Run consolidation cycle
watty snapshot         Backup brain.db
watty rollback         Restore from backup
watty version          Current version
```

## Data

Everything lives at `~/.watty/`:

| File | What |
|------|------|
| `brain.db` | All memories, embeddings, associations (SQLite) |
| `shape.json` | Current understanding — the metabolism output |
| `cognition/profile.json` | Behavioral patterns, directives, session history |
| `metabolism.log` | Digest history — what changed and why |
| `snapshots/` | Timestamped backups of brain.db |

Your data never leaves your machine. No cloud. No telemetry. Delete `~/.watty/` to completely reset.

## Troubleshooting

**"watty: command not found"** — pip installed it somewhere not on your PATH. Try `python3 -m watty version`. If that works, use the full path in your Claude config.

**Claude Desktop doesn't show Watty tools** — make sure the config JSON is valid, then fully quit and reopen Desktop (not just close the window).

**First run is slow** — downloading the embedding model (~100MB). One-time only.

**PyTorch is huge** — yes, ~2GB. For CPU-only (no GPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing Watty.

**Metabolism not running** — check that `ANTHROPIC_API_KEY` is set. Check `~/.watty/metabolism.log` for errors.

---

*Built by Hugo & Rim. February 2026.*
