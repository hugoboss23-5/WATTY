# Claude CLI Connection Setup

This guide shows you how to connect the Claude CLI to your Watty MCP server.

## Your Claude CLI Path

```
C:\Users\bulli\AppData\Roaming\npm\claude
```

## Quick Setup

### Option 1: Use the Configuration File (Recommended)

A ready-to-use configuration file is available in this repository:

**File:** `claude-cli-config.json`

This file contains:
- ✅ Your Claude CLI path
- ✅ Watty MCP server configuration
- ✅ All environment variables
- ✅ Optimal default settings

### Option 2: Manual Configuration

Copy the following configuration to your Claude config location:

**Location:**
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac/Linux:** `~/.config/claude/claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "watty": {
      "command": "watty",
      "args": [],
      "env": {
        "WATTY_HOME": "${HOME}/.watty/",
        "WATTY_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "WATTY_TOP_K": "10",
        "WATTY_RELEVANCE_THRESHOLD": "0.35",
        "WATTY_CHUNK_SIZE": "1500",
        "WATTY_CHUNK_OVERLAP": "200"
      }
    }
  }
}
```

## Verify Connection

After setting up, restart Claude Desktop or your Claude CLI client.

You can verify Watty is connected by:
1. Opening Claude
2. Asking: "What MCP tools do you have access to?"
3. Look for `watty_` tools in the response

## Available Tools

Once connected, you'll have access to:

| Tool | Description |
|------|-------------|
| `watty_recall` | Search memory by meaning |
| `watty_remember` | Store something important |
| `watty_scan` | Index entire folders |
| `watty_cluster` | Organize knowledge graph |
| `watty_forget` | Delete memories |
| `watty_surface` | Surface relevant knowledge |
| `watty_reflect` | Map your entire mind |
| `watty_stats` | Memory statistics |

## Configuration Files in This Repo

- **`claude-cli-config.json`** - Complete connection configuration
- **`CLI-SETUP.md`** - This guide
- **`README.md`** - Full Watty documentation

## Environment Variables

All variables are optional. Defaults work out of the box:

| Variable | Default | Description |
|----------|---------|-------------|
| `WATTY_HOME` | `~/.watty/` | Where the brain lives |
| `WATTY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `WATTY_TOP_K` | `10` | Max memories per search |
| `WATTY_RELEVANCE_THRESHOLD` | `0.35` | Min similarity score (0-1) |
| `WATTY_CHUNK_SIZE` | `1500` | Characters per memory chunk |
| `WATTY_CHUNK_OVERLAP` | `200` | Overlap between chunks |

## Troubleshooting

### Claude CLI not found?
Make sure `watty` is installed and in your PATH:
```bash
pip install -e .
```

### MCP server not connecting?
1. Check that the `watty` command works from terminal
2. Restart Claude Desktop completely
3. Check Claude Desktop logs for errors

### Need help?
- Check [README.md](README.md) for full documentation
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- Report issues on GitHub

---

**Your AI memory is now persistent, local, and private.** 🧠
