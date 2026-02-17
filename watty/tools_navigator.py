"""
tools_navigator.py — Navigator MCP Tool
=========================================
Deep memory navigation using spreading activation through
the association graph, shaped by Chestahedron geometry.

MCP tool: watty_navigate

Hugo & Rim & Claude — February 2026
"""

from mcp.types import Tool, TextContent

_navigator_instance = None


def _get_navigator(brain):
    global _navigator_instance
    if _navigator_instance is None or _navigator_instance.brain is not brain:
        from watty.navigator import Navigator
        _navigator_instance = Navigator(brain)
    return _navigator_instance


TOOLS = [
    Tool(
        name="watty_navigate",
        description=(
            "Deep memory navigation using spreading activation. "
            "Unlike recall (flat similarity), Navigate follows association paths "
            "and reads the geometric shape of activated memories through organs. "
            "Use for complex queries needing context discovery or connection tracing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to navigate through memory for. Be specific.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)",
                },
            },
            "required": ["query"],
        },
    ),
]


async def handle_navigate(arguments: dict, brain=None) -> list[TextContent]:
    query = arguments.get("query", "")
    if not query.strip():
        return [TextContent(type="text", text="Need a query to navigate.")]

    top_k = arguments.get("top_k", 10)
    nav = _get_navigator(brain)
    result = nav.navigate(query, top_k=top_k)

    # Format results
    results = result.get("results", [])
    if not results:
        return [TextContent(type="text", text="Navigation found no relevant memories.")]

    lines = []

    # Results
    for i, r in enumerate(results, 1):
        source = f" [{r['source_type']}]" if r.get("source_type") != "conversation" else ""
        path = f" ({r['source_path']})" if r.get("source_path") else ""
        lines.append(
            f"[{i}] (activation: {r['activation']}, similarity: {r['similarity']}, "
            f"via: {r['provider']}{source}{path})\n{r['content']}"
        )

    # Diagnostics summary
    diag = []
    diag.append(f"Circulations: {result['circulations']}")
    diag.append(f"Final coherence: {result['final_coherence']}")
    if result.get("blood_strategies"):
        diag.append(f"Strategies: {' -> '.join(result['blood_strategies'])}")
    if result.get("organ_readings"):
        last_organs = result["organ_readings"][-1]
        signals = [r.get("signal", "?") for r in last_organs]
        diag.append(f"Final organs: coherence={signals[0]}, depth={signals[1]}, bridge={signals[2]}")

    formatted = (
        f"Navigator found {len(results)} memories ({result['circulations']} circulations):\n\n"
        + "\n\n---\n\n".join(lines)
        + "\n\n── Diagnostics ──\n" + "\n".join(diag)
    )

    return [TextContent(type="text", text=formatted)]


HANDLERS = {
    "watty_navigate": handle_navigate,
}
