"""
Watty MCP Server — stdio transport.
All tool definitions and handlers live in tools.py.
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from watty.brain import Brain
from watty.config import SERVER_NAME, SERVER_VERSION
from watty.tools import TOOL_DEFS, call_tool
from watty.log import log, setup


brain = Brain()
server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in TOOL_DEFS]


@server.call_tool()
async def handle_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        # Run in thread pool so blocking calls (like subprocess.run in watty_execute)
        # don't freeze the event loop — which would prevent the MCP response from
        # being sent back before the client times out.
        result = await asyncio.to_thread(call_tool, brain, name, arguments)
        return [TextContent(type="text", text=result["text"])]
    except Exception as e:
        log.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=f"Watty error: {str(e)}")]


async def main():
    setup()
    log.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    log.info(f"Brain: {brain.db_path}")
    stats = brain.stats()
    log.info(f"{stats['total_memories']} memories | {stats['total_conversations']} conversations | {stats['total_files_scanned']} files scanned")
    log.info(f"{len(TOOL_DEFS)} tools ready. Layer 1 active.")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
