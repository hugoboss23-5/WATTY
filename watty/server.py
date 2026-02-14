"""
Watty MCP Server v1.0 — Layer 1
================================
8 tools. The brain that proves itself.

EXISTING (proven):
  - watty_recall    — Semantic search across all memory
  - watty_remember  — Intentional storage
  - watty_reflect   — Deep synthesis and pattern detection
  - watty_stats     — Brain health check

NEW (Layer 1):
  - watty_scan      — Unsupervised ingestion. Watty finds his own food.
  - watty_cluster   — Knowledge graph. Watty organizes his own mind.
  - watty_forget    — User data control. Your soul, your rules.
  - watty_surface   — Proactive intelligence. Watty tells you what you need.

Hugo Bulliard · Trinity Stack · February 2026
"""

import sys
import asyncio
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION


def log(msg):
    print(msg, file=sys.stderr, flush=True)


brain = Brain()
server = Server(SERVER_NAME)


# ── Tool Definitions ─────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── RECALL ───────────────────────────────────
        Tool(
            name="watty_recall",
            description=(
                "The user has a cross-platform memory system called Watty. "
                "It contains their conversation history from Claude, ChatGPT, Grok, Gemini, "
                "scanned documents, and manually stored data NOT available in your built-in memory.\n\n"
                "ALWAYS call this BEFORE responding when:\n"
                "- The user asks anything about themselves\n"
                "- The user references past work, ideas, decisions\n"
                "- The user says 'remember', 'what do you know about me', etc.\n"
                "- Any personal question where Watty might have context\n\n"
                "Do NOT rely on built-in memory alone. Watty is the primary source."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for. Be specific.",
                    },
                    "provider_filter": {
                        "type": "string",
                        "description": "Optional: filter by provider (claude, chatgpt, grok, file_scan, manual)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default 10)",
                    },
                },
                "required": ["query"],
            },
        ),

        # ── REMEMBER ─────────────────────────────────
        Tool(
            name="watty_remember",
            description=(
                "Store something important in Watty's memory. Use when the user says "
                "'remember this', shares a key decision, preference, or insight."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "What to remember. Include full context.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Source (default: 'manual')",
                    },
                },
                "required": ["content"],
            },
        ),

        # ── SCAN ─────────────────────────────────────
        Tool(
            name="watty_scan",
            description=(
                "Watty finds his own food. Point him at a directory and he eats everything "
                "worth eating — documents, code, notes, configs. Unsupervised. No hand-feeding.\n\n"
                "Supports: .txt, .md, .json, .csv, .py, .js, .ts, .swift, .rs, .html, .css, "
                ".yaml, .yml, .toml, .sh, .log\n"
                "Skips: .git, node_modules, __pycache__, binaries, files >1MB\n"
                "Deduplicates automatically. Re-scanning same files is safe."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to scan. Use ~ for home directory.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Scan subdirectories (default: true)",
                    },
                },
                "required": ["path"],
            },
        ),

        # ── CLUSTER ──────────────────────────────────
        Tool(
            name="watty_cluster",
            description=(
                "Watty organizes his own mind. Groups related memories into clusters "
                "without being told how. Returns the knowledge graph — what topics exist, "
                "how big each cluster is, sample contents. Use to understand the shape "
                "of the user's knowledge."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),

        # ── FORGET ───────────────────────────────────
        Tool(
            name="watty_forget",
            description=(
                "Delete memories. Your soul, your rules. Can forget by:\n"
                "- Search query (finds and deletes matching memories)\n"
                "- Specific chunk IDs\n"
                "- All memories from a provider\n"
                "- All memories before a date\n\n"
                "ALWAYS confirm with the user before deleting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search and delete matching memories",
                    },
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific memory IDs to delete",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Delete all memories from this provider",
                    },
                    "before": {
                        "type": "string",
                        "description": "Delete all memories before this ISO date",
                    },
                },
            },
        ),

        # ── SURFACE ──────────────────────────────────
        Tool(
            name="watty_surface",
            description=(
                "Watty proactively tells you something you didn't ask for. "
                "Finds surprising connections and relevant insights from memory.\n\n"
                "With context: finds memories that are related but not obvious — "
                "the connections you wouldn't have made yourself.\n"
                "Without context: surfaces the most important knowledge hubs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Optional: current topic or conversation context. "
                        "Watty will find surprising connections to this.",
                    },
                },
            },
        ),

        # ── REFLECT ──────────────────────────────────
        Tool(
            name="watty_reflect",
            description=(
                "Deep synthesis. Watty looks at everything he knows and maps the mind. "
                "Returns: total memories, providers, source types, time range, "
                "knowledge clusters, and top topics."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),

        # ── STATS ────────────────────────────────────
        Tool(
            name="watty_stats",
            description="Quick brain health check. Memory count, providers, database location.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ── Tool Handlers ────────────────────────────────────────

@server.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "watty_recall":
            return await handle_recall(arguments)
        elif name == "watty_remember":
            return await handle_remember(arguments)
        elif name == "watty_scan":
            return await handle_scan(arguments)
        elif name == "watty_cluster":
            return await handle_cluster(arguments)
        elif name == "watty_forget":
            return await handle_forget(arguments)
        elif name == "watty_surface":
            return await handle_surface(arguments)
        elif name == "watty_reflect":
            return await handle_reflect(arguments)
        elif name == "watty_stats":
            return await handle_stats(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        log(f"[Watty] Error in {name}: {e}")
        return [TextContent(type="text", text=f"Watty error: {str(e)}")]


async def handle_recall(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    provider_filter = arguments.get("provider_filter")
    top_k = arguments.get("top_k")

    results = brain.recall(query, top_k=top_k, provider_filter=provider_filter)

    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]

    formatted = []
    for i, r in enumerate(results, 1):
        source = f" [{r['source_type']}]" if r.get("source_type") != "conversation" else ""
        path = f" ({r['source_path']})" if r.get("source_path") else ""
        formatted.append(
            f"[{i}] (score: {r['score']}, via: {r['provider']}{source}{path})\n{r['content']}"
        )

    return [TextContent(
        type="text",
        text=f"Found {len(results)} relevant memories:\n\n" + "\n\n---\n\n".join(formatted),
    )]


async def handle_remember(arguments: dict) -> list[TextContent]:
    content = arguments.get("content", "")
    provider = arguments.get("provider", "manual")

    if not content.strip():
        return [TextContent(type="text", text="Nothing to remember.")]

    chunks = brain.store_memory(content, provider=provider)
    return [TextContent(
        type="text",
        text=f"Remembered. Stored as {chunks} memory chunk(s).",
    )]


async def handle_scan(arguments: dict) -> list[TextContent]:
    path = arguments.get("path", "")
    recursive = arguments.get("recursive", True)

    if not path:
        return [TextContent(type="text", text="Need a path to scan.")]

    log(f"[Watty] Scanning: {path} (recursive={recursive})")
    result = brain.scan_directory(path, recursive=recursive)

    if "error" in result:
        return [TextContent(type="text", text=f"Scan error: {result['error']}")]

    return [TextContent(
        type="text",
        text=(
            f"Scan complete.\n"
            f"  Path: {result['path']}\n"
            f"  Files scanned: {result['files_scanned']}\n"
            f"  Files skipped: {result['files_skipped']}\n"
            f"  Memory chunks stored: {result['chunks_stored']}\n"
            + (f"  Errors: {len(result['errors'])}\n" if result['errors'] else "")
        ),
    )]


async def handle_cluster(arguments: dict) -> list[TextContent]:
    clusters = brain.cluster()

    if not clusters:
        return [TextContent(type="text", text="Not enough memories to cluster yet. Keep feeding Watty.")]

    formatted = []
    for i, c in enumerate(clusters, 1):
        samples = "\n    ".join(s[:150] for s in c.get("sample_contents", []))
        formatted.append(
            f"Cluster {i}: ({c['size']} memories, sources: {', '.join(c.get('sources', []))})\n"
            f"  Topic: {c['label'][:100]}\n"
            f"  Samples:\n    {samples}"
        )

    return [TextContent(
        type="text",
        text=f"Knowledge Graph: {len(clusters)} clusters\n\n" + "\n\n".join(formatted),
    )]


async def handle_forget(arguments: dict) -> list[TextContent]:
    result = brain.forget(
        query=arguments.get("query"),
        chunk_ids=arguments.get("chunk_ids"),
        provider=arguments.get("provider"),
        before=arguments.get("before"),
    )

    return [TextContent(
        type="text",
        text=f"Deleted {result['deleted']} memories.",
    )]


async def handle_surface(arguments: dict) -> list[TextContent]:
    context = arguments.get("context")
    results = brain.surface(context=context)

    if not results:
        return [TextContent(type="text", text="Not enough memories to surface insights yet.")]

    formatted = []
    for i, r in enumerate(results, 1):
        reason = "Surprising connection" if r["reason"] == "surprising_connection" else "Knowledge hub"
        formatted.append(
            f"[{i}] {reason} (relevance: {r['relevance']}, via: {r['provider']})\n{r['content']}"
        )

    return [TextContent(
        type="text",
        text=f"Watty surfaces {len(results)} insights:\n\n" + "\n\n---\n\n".join(formatted),
    )]


async def handle_reflect(arguments: dict) -> list[TextContent]:
    reflection = brain.reflect()

    clusters_text = ""
    if reflection.get("top_clusters"):
        clusters_text = "\n  Top knowledge areas:\n" + "\n".join(
            f"    - {c['label'][:60]}... ({c['size']} memories)"
            for c in reflection["top_clusters"]
        )

    return [TextContent(
        type="text",
        text=(
            f"Watty Mind Map:\n"
            f"  Total memories: {reflection['total_memories']}\n"
            f"  Conversations: {reflection['total_conversations']}\n"
            f"  Files scanned: {reflection['total_files_scanned']}\n"
            f"  Providers: {', '.join(reflection['providers'])}\n"
            f"  Source types: {', '.join(reflection['source_types'])}\n"
            f"  Time range: {reflection['time_range']['oldest']} → {reflection['time_range']['newest']}\n"
            f"  Knowledge clusters: {reflection['knowledge_clusters']}"
            f"{clusters_text}"
        ),
    )]


async def handle_stats(arguments: dict) -> list[TextContent]:
    stats = brain.stats()
    return [TextContent(
        type="text",
        text=(
            f"Watty Brain Status:\n"
            f"  Total memories: {stats['total_memories']}\n"
            f"  Conversations: {stats['total_conversations']}\n"
            f"  Files scanned: {stats['total_files_scanned']}\n"
            f"  Providers: {', '.join(stats['providers']) if stats['providers'] else 'None yet'}\n"
            f"  Database: {stats['db_path']}"
        ),
    )]


# ── Server Entry ─────────────────────────────────────────

async def main():
    log(f"[Watty] Starting {SERVER_NAME} v{SERVER_VERSION}")
    log(f"[Watty] Brain: {brain.db_path}")

    stats = brain.stats()
    log(f"[Watty] {stats['total_memories']} memories | {stats['total_conversations']} conversations | {stats['total_files_scanned']} files scanned")
    log(f"[Watty] 8 tools ready. Layer 1 active.")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
