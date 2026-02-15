<p align="center">
  <img src="assets/banner.svg" alt="Watty — One memory. Every AI." width="100%">
</p>

<p align="center">
  <a href="https://github.com/watty-ai/watty/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-compatible-green.svg" alt="MCP Compatible"></a>
  <a href="https://github.com/watty-ai/watty/actions/workflows/tests.yml"><img src="https://github.com/watty-ai/watty/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/watty-ai/watty/stargazers"><img src="https://img.shields.io/github/stars/watty-ai/watty?style=social" alt="GitHub Stars"></a>
</p>

---

Your AI has amnesia. Every conversation starts from zero. Watty fixes that.

Watty is an [MCP](https://modelcontextprotocol.io) server that gives **any AI** persistent memory. Install it once, and your AI remembers everything: conversations, documents, code, decisions. Locally. Privately. Forever.

**No cloud. No API keys. No monthly fees. Your data never leaves your machine.**

## Install in 60 seconds

```bash
git clone https://github.com/watty-ai/watty.git
cd watty
pip install -e .
```

> ⚠️ **Heads up:** `sentence-transformers` pulls in PyTorch (~2GB download). First install takes a few minutes. After that, Watty starts in seconds.

> 🔧 **Claude CLI Users:** See **[CLI-SETUP.md](CLI-SETUP.md)** for detailed connection configuration and `claude-cli-config.json` for a ready-to-use config file.

Add to your Claude Desktop config:

**Mac/Linux:** `~/.config/claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "watty": {
      "command": "watty",
      "args": []
    }
  }
}
```

Restart Claude Desktop. Done. Watty is alive.

## Platform support

Watty speaks MCP over **stdio** — the standard local transport.

| Platform | Status | Notes |
|----------|--------|-------|
| **Claude Desktop** | ✅ Works now | Native stdio MCP support |
| **Cursor** | ✅ Works now | Native stdio MCP support |
| **Windsurf** | ✅ Works now | Native stdio MCP support |
| **Claude Code** | ✅ Works now | Native stdio MCP support |
| **VS Code + Copilot** | ✅ Works now | Via MCP extension |
| **ChatGPT** | 🔜 v1.1 | Requires HTTP transport — [tracking issue](https://github.com/watty-ai/watty/issues) |
| **Gemini** | 🔜 v1.1 | Requires HTTP transport |
| **Grok** | 🔜 v1.1 | Requires HTTP transport |

HTTP transport is the #1 priority for v1.1 — one wrapper and Watty works everywhere.

## What happens next

Talk to your AI normally. Watty works in the background — no commands to memorize, no workflows to learn.

```
You:    "Hey Claude, scan my Documents folder"
Watty:  → Eats 847 files. Indexes everything. Deduplicates automatically.

You:    "What was that thing I wrote about distributed systems last month?"  
Watty:  → Finds it instantly. Semantic search, not keyword matching.

You:    "Remember that I prefer TypeScript over JavaScript for new projects"
Watty:  → Stored. Every future coding conversation just knows this.

You:    "What patterns do you see across all my notes?"
Watty:  → Clusters your knowledge. Surfaces connections you missed.
```

**Your AI doesn't announce that it's using Watty. It just _knows_ things.** The way a colleague remembers your preferences without being asked.

## 8 Tools

| Tool | What it does |
|------|-------------|
| `watty_recall` | Search memory by **meaning**, not keywords |
| `watty_remember` | Store something important |
| `watty_scan` | Point at a folder — Watty eats everything worth eating |
| `watty_cluster` | Watty organizes his own mind into a knowledge graph |
| `watty_forget` | Delete anything. Your soul, your rules |
| `watty_surface` | Watty tells you what you didn't know you needed |
| `watty_reflect` | Map your entire mind — providers, topics, time range |
| `watty_stats` | Quick brain health check |

## How it works

```
┌─────────────────────────────────────────┐
│           YOU (type normally)            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         ANY MCP CLIENT                  │
│   Claude Desktop · Cursor · Windsurf    │
│                                         │
│   Before responding, the AI queries     │
│   Watty for relevant memories           │
└─────────────────┬───────────────────────┘
                  │ stdio
                  ▼
┌─────────────────────────────────────────┐
│          WATTY (MCP Server)             │
│                                         │
│   SQLite + Semantic Vectors             │
│   Everything indexed by meaning         │
│   Runs locally · Zero latency           │
│                                         │
│   Your conversations, documents,        │
│   code, notes — all searchable          │
│   by what they MEAN, not what           │
│   words they contain                    │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         AI RESPONDS                     │
│   With natural awareness of your        │
│   history. No citation. No fanfare.     │
│   It just knows.                        │
└─────────────────────────────────────────┘
```

## What Watty can eat

Point Watty at any folder. He finds his own food:

`.txt` `.md` `.json` `.csv` `.log` `.py` `.js` `.ts` `.swift` `.rs` `.html` `.css` `.yaml` `.yml` `.toml` `.sh` `.bat` `.ps1`

He skips what he should (`.git`, `node_modules`, `__pycache__`, binaries) and deduplicates automatically. Re-scanning is always safe.

## Why Watty

| | Watty | Cloud memory tools | Other local tools |
|---|---|---|---|
| **Install** | `pip install -e .` | API keys + cloud accounts | Docker + databases + config |
| **Works with** | Any stdio MCP client | Usually one platform | Varies |
| **Data storage** | Local SQLite. Period. | Their servers | Local, but complex setup |
| **Search** | Semantic (by meaning) | Usually semantic | Keyword or hybrid |
| **File ingestion** | Point at folder, done | Manual upload | Complex pipelines |
| **Self-organization** | Automatic clustering | You organize it | You organize it |
| **Privacy** | Your machine. Nothing leaves. | Read the terms. | Usually local |
| **Cost** | Free forever | Free tier → paid | Free (usually) |
| **Code** | ~850 lines of logic. Audit in 20 min. | Proprietary | Thousands+ |

## Configuration

All optional. Watty works out of the box with zero configuration.

| Variable | Default | What it does |
|----------|---------|-------------|
| `WATTY_HOME` | `~/.watty/` | Where the brain lives |
| `WATTY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `WATTY_TOP_K` | `10` | Max memories per search |
| `WATTY_RELEVANCE_THRESHOLD` | `0.35` | Min similarity score (0-1) |
| `WATTY_CHUNK_SIZE` | `1500` | Characters per memory chunk |
| `WATTY_CHUNK_OVERLAP` | `200` | Overlap between chunks |

## FAQ

**Does it work with ChatGPT / Gemini / Grok?**  
Not yet — Watty v1 uses stdio transport, which works with Claude Desktop, Cursor, and other local MCP clients. ChatGPT, Gemini, and Grok require HTTP/SSE transport for remote MCP servers. HTTP support is the top priority for v1.1. Follow the repo for updates.

**Why is the first install so large?**  
`sentence-transformers` depends on PyTorch (~2GB). This is a one-time download. After that, Watty starts instantly. We're exploring `onnxruntime` as a lighter alternative for a future release — same model, ~100MB total, no GPU dependency.

**How much disk space does it use after install?**  
The embedding model is ~80MB. After that, your memories are just SQLite rows — thousands of documents fit in megabytes.

**Is my data really private?**  
Watty is a local MCP server. Your data lives in `~/.watty/brain.db` on your machine. No network calls except downloading the embedding model the first time. Read the code — it's ~850 lines of logic. You can audit it in 20 minutes.

**Can I back up my brain?**  
Copy `~/.watty/brain.db`. That's your entire memory. One file.

**Can I delete specific memories?**  
Yes. `watty_forget` can delete by search query, specific IDs, provider, or date. Your soul, your rules.

**What's the embedding model?**  
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — a 22M parameter sentence transformer. Small, fast, runs on any machine. No GPU needed.

**Will it slow down with thousands of memories?**  
Watty does brute-force cosine similarity over numpy arrays. This is fast up to ~50k memories on any modern machine. Beyond that, we'll add approximate nearest neighbor search in a future version.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

17 tests, runs in ~2 seconds, no PyTorch download needed (uses mock embeddings).

## Built by

[Hugo Bulliard](https://github.com/watty-ai) — 19, economics student, building infrastructure for human-AI collaboration.

Watty is Layer 1 of the [Trinity Stack](https://github.com/watty-ai) — persistence, governance, and payments for the AI era.

## License

[MIT](LICENSE) — Use it, fork it, build on it. Free forever.

---

<p align="center">
  <strong>If Watty gave your AI a brain, consider giving this repo a ⭐</strong>
</p>
