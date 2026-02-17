# Watty v2.1 — The Brain That Proves Itself

**One memory. Every AI. It just knows you.**

Watty is a local-first AI memory system. Install it once and every AI you talk to — Claude, ChatGPT, Grok, Gemini — remembers your conversations, documents, decisions, and preferences. No cloud. No API keys. Your brain stays on your machine.

## What Makes Watty Different

Everyone else does memory-as-storage. Watty does **memory-as-identity**.

- **Digital hippocampus** — pattern separation, associative recall, mismatch detection, dream consolidation. Not a vector database with a search bar. An actual memory architecture modeled on neuroscience.
- **Self-modifying** — Watty can read and edit its own source code. It evolves.
- **Cross-provider** — works across every MCP-compatible AI. One brain, many interfaces.
- **Zero config** — `pip install`, add to config, done. No Docker, no API keys, no cloud accounts.

## 24 Tools

### Memory (11 tools)
| Tool | What it does |
|------|-------------|
| `watty_recall` | Semantic search — finds memories by meaning, not keywords |
| `watty_remember` | Store something important |
| `watty_scan` | Point at a folder, Watty eats everything worth eating |
| `watty_cluster` | Unsupervised knowledge graph — groups related memories |
| `watty_forget` | Delete anything. Your data, your rules |
| `watty_surface` | Proactive insights — connections you didn't know you needed |
| `watty_reflect` | Deep synthesis — map your entire mind |
| `watty_stats` | Brain health check |
| `watty_dream` | Sleep consolidation — promotes, decays, strengthens, prunes |
| `watty_contradictions` | Surface unresolved conflicts in memory |
| `watty_resolve` | Arbitrate contradictions — keep new or old |

### Infrastructure (4 tools)
| Tool | What it does |
|------|-------------|
| `watty_execute` | Run Python, bash, node, or PowerShell directly |
| `watty_file_read` | Read any file |
| `watty_file_write` | Write any file |
| `watty_self(action)` | Self-modification: read/edit source, protocol, changelog |

### GPU (1 tool, 14 actions)
| Tool | What it does |
|------|-------------|
| `watty_gpu(action)` | Full Vast.ai lifecycle: search, create, start, stop, destroy, exec, jupyter, credit |

### Session (4 tools)
| Tool | What it does |
|------|-------------|
| `watty_enter` | Open persistent cognitive space — read last instance's note |
| `watty_leave` | Write handoff note for the next AI instance |
| `watty_pulse` | Uptime and current time |
| `watty_handoff` | Structured session handoff |

### Communications (2 tools, 8 actions)
| Tool | What it does |
|------|-------------|
| `watty_chat(action)` | Desktop-to-Code chat bridge: send, check, history |
| `watty_browser(action)` | Tracked research sessions: start, log, end, recall, bookmark |

### Screen Control (1 tool, 7 actions)
| Tool | What it does |
|------|-------------|
| `watty_screen(action)` | Desktop automation: screenshot, click, type, key, move, scroll, drag |

### Web Dashboard (1 tool)
| Tool | What it does |
|------|-------------|
| `watty_web(action)` | Live brain dashboard with knowledge graph, search, tier breakdown |

## Install

```bash
git clone https://github.com/hugoboss23-5/WATTY.git
cd watty
pip install -e .
```

Add to your Claude Desktop config:

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

Restart Claude Desktop. Watty is alive.

## First 30 Seconds

```
"Scan my Documents folder"
"What do you know about my projects?"
"Surface something I should know right now"
"Start the brain dashboard"
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Watty MCP Server               │
├──────────┬──────────┬──────────┬────────────┤
│  Memory  │  Infra   │   GPU    │   Screen   │
│ 11 tools │ 4 tools  │ 1 tool   │  1 tool    │
├──────────┴──────────┴──────────┴────────────┤
│           Digital Hippocampus               │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌───────────┐  │
│  │ DG  │→│ CA3 │→│ CA1 │→│ Consolidate│  │
│  │sep. │  │assoc│  │match│  │  (dream)   │  │
│  └─────┘  └─────┘  └─────┘  └───────────┘  │
├─────────────────────────────────────────────┤
│  SQLite · sentence-transformers · local     │
└─────────────────────────────────────────────┘
```

- **Dentate Gyrus (DG):** Pattern separation. Similar inputs get orthogonalized to prevent catastrophic overlap.
- **CA3:** Associative storage. Memories wire together based on co-occurrence. Pattern completion during recall.
- **CA1:** Mismatch detection. Flags novel information and contradictions.
- **Consolidation:** Dream cycles promote frequently accessed memories, decay stale ones, strengthen pathways, prune dead connections.

## Configuration

All settings via environment variables (see `watty/config.py`):

| Variable | Default | What |
|----------|---------|------|
| `WATTY_HOME` | `~/.watty/` | Data directory |
| `WATTY_TOP_K` | `10` | Results per search |
| `WATTY_RELEVANCE_THRESHOLD` | `0.35` | Minimum similarity |
| `WATTY_RECENCY_WEIGHT` | `0.15` | Recency boost |
| `WATTY_DG_SPARSITY` | `0.02` | Dentate gyrus activation (biological: 2%) |
| `WATTY_CA3_MAX_ASSOC` | `10` | Associations per memory |
| `WATTY_CONSOL_DECAY` | `30` | Days before unaccessed memories decay |

## Privacy

- Everything local. SQLite database at `~/.watty/brain.db`.
- No cloud, no telemetry, no API calls.
- Your brain is a single file you can copy, backup, or delete.

---

*Built by Hugo & Rim. February 2026.*
*24 tools. 52 actions. One brain. Every AI.*
