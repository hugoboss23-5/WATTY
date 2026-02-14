"""
Watty MCP Server — stdio transport.
All tool definitions and handlers live in tools.py.
"""

import sys
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION
from watty.tools import TOOL_DEFS, call_tool


def log(msg):
    print(msg, file=sys.stderr, flush=True)


brain = Brain()
server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in TOOL_DEFS]


@server.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = call_tool(brain, name, arguments)
        return [TextContent(type="text", text=result["text"])]
    except Exception as e:
        log(f"[Watty] Error in {name}: {e}")
        return [TextContent(type="text", text=f"Watty error: {str(e)}")]


async def main():
    log(f"[Watty] Starting {SERVER_NAME} v{SERVER_VERSION}")
    log(f"[Watty] Brain: {brain.db_path}")
    stats = brain.stats()
    log(f"[Watty] {stats['total_memories']} memories | {stats['total_conversations']} conversations | {stats['total_files_scanned']} files scanned")
    log(f"[Watty] {len(TOOL_DEFS)} tools ready. Layer 1 active.")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
