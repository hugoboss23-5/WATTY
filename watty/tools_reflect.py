"""
Watty Reflection Engine — MCP Tool Interface
==============================================
One tool: watty_reflect_engine(action=...)
Actions: add_reflection, get_reflections, search_reflections,
         promote_to_directive, reflection_stats

Verbal reinforcement learning. Stores self-critiques, retrieves them
before similar tasks, auto-promotes recurring lessons to directives.

Hugo & Watty · February 2026
"""

from mcp.types import Tool, TextContent


# ── Valid enums (mirrored from reflection.py for schema) ──

VALID_CATEGORIES = [
    "code", "architecture", "communication", "planning",
    "debugging", "research", "integration", "testing",
    "data", "security", "performance", "other",
]

VALID_SEVERITIES = ["critical", "major", "minor"]

REFLECT_ACTIONS = [
    "add_reflection", "get_reflections", "search_reflections",
    "promote_to_directive", "reflection_stats",
]


# ── Brain Reference ──────────────────────────────────────

_brain_ref = None


def set_brain(brain):
    """Called by server.py to inject brain reference."""
    global _brain_ref
    _brain_ref = brain


# ── Tool Definition ──────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_reflect_engine",
        description=(
            "Watty's reflection engine — verbal reinforcement learning.\n"
            "Stores self-critiques from failures, retrieves them before similar future tasks,\n"
            "and auto-promotes recurring lessons to behavioral directives.\n\n"
            "Actions:\n"
            "  add_reflection — Store a self-critique (task + outcome + reflection + lessons)\n"
            "  get_reflections — Retrieve reflections similar to a query (semantic search)\n"
            "  search_reflections — Browse reflections with filters (no semantic query)\n"
            "  promote_to_directive — Manually promote a reflection's lesson to a directive\n"
            "  reflection_stats — Aggregate statistics about all reflections\n\n"
            "Use add_reflection after any task with a suboptimal outcome.\n"
            "Use get_reflections BEFORE starting a task to check for past lessons.\n"
            "Lessons that recur >= 3 times auto-promote to directives."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": REFLECT_ACTIONS,
                    "description": "Action to perform",
                },
                "task_description": {
                    "type": "string",
                    "description": "add_reflection: What was attempted",
                },
                "outcome": {
                    "type": "string",
                    "description": "add_reflection: What happened (success/failure/partial)",
                },
                "reflection": {
                    "type": "string",
                    "description": "add_reflection: Self-critique — what went wrong, why, what to change",
                },
                "lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "add_reflection: Concrete lessons learned (short strings)",
                },
                "query": {
                    "type": "string",
                    "description": "get_reflections: What to search for (typically the current task)",
                },
                "category": {
                    "type": "string",
                    "enum": VALID_CATEGORIES,
                    "description": "Filter or tag by category",
                },
                "severity": {
                    "type": "string",
                    "enum": VALID_SEVERITIES,
                    "description": "Filter or tag by severity",
                },
                "top_k": {
                    "type": "integer",
                    "description": "get_reflections: Max results (default: 5)",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["recent", "most_retrieved", "severity"],
                    "description": "search_reflections: Sort order (default: recent)",
                },
                "limit": {
                    "type": "integer",
                    "description": "search_reflections: Max results (default: 20)",
                },
                "reflection_id": {
                    "type": "integer",
                    "description": "promote_to_directive: ID of the reflection to promote",
                },
                "rule": {
                    "type": "string",
                    "description": "promote_to_directive: Custom directive rule (default: first lesson)",
                },
                "related_chunk_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "add_reflection: Related memory chunk IDs",
                },
                "session_number": {
                    "type": "integer",
                    "description": "add_reflection: Current session number",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ──────────────────────────────────────────────

def _get_engine():
    """Get the ReflectionEngine from the brain, or create a standalone one."""
    if _brain_ref is not None and hasattr(_brain_ref, "_reflection") and _brain_ref._reflection is not None:
        return _brain_ref._reflection
    # Fallback: create standalone engine using brain's db_path
    from watty.reflection import ReflectionEngine
    if _brain_ref is not None:
        return ReflectionEngine(db_path=_brain_ref.db_path)
    return ReflectionEngine()


async def handle_reflect_engine(arguments: dict) -> list[TextContent]:
    """Dispatch reflection engine actions."""
    action = arguments.get("action", "")

    engine = _get_engine()
    if engine is None:
        return [TextContent(type="text", text="Reflection Engine not available.")]

    # ── add_reflection ───────────────────────────────
    if action == "add_reflection":
        task_desc = arguments.get("task_description")
        outcome = arguments.get("outcome")
        reflection = arguments.get("reflection")
        lessons = arguments.get("lessons", [])

        if not task_desc or not reflection:
            return [TextContent(
                type="text",
                text="Need 'task_description' and 'reflection' to add a reflection.",
            )]
        if not outcome:
            outcome = "unspecified"

        try:
            rid = engine.add_reflection(
                task_description=task_desc,
                outcome=outcome,
                reflection_text=reflection,
                lessons=lessons,
                category=arguments.get("category", "other"),
                severity=arguments.get("severity", "minor"),
                related_chunk_ids=arguments.get("related_chunk_ids"),
                session_number=arguments.get("session_number"),
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error storing reflection: {e}")]

        lesson_str = ""
        if lessons:
            lesson_str = "\nLessons stored:\n" + "\n".join(f"  - {l}" for l in lessons)

        return [TextContent(
            type="text",
            text=(
                f"Reflection #{rid} stored.\n"
                f"  Task: {task_desc[:100]}\n"
                f"  Outcome: {outcome}\n"
                f"  Category: {arguments.get('category', 'other')}\n"
                f"  Severity: {arguments.get('severity', 'minor')}"
                f"{lesson_str}"
            ),
        )]

    # ── get_reflections ──────────────────────────────
    elif action == "get_reflections":
        query = arguments.get("query")
        if not query:
            return [TextContent(
                type="text",
                text="Need 'query' to search reflections.",
            )]

        try:
            results = engine.get_reflections(
                query=query,
                top_k=arguments.get("top_k"),
                category=arguments.get("category"),
                severity=arguments.get("severity"),
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching reflections: {e}")]

        if not results:
            return [TextContent(type="text", text="No relevant reflections found.")]

        lines = [f"Found {len(results)} relevant reflection(s):\n"]
        for r in results:
            promoted = " [PROMOTED]" if r.get("promoted_to_directive") else ""
            lines.append(
                f"--- Reflection #{r['id']} (similarity: {r['similarity']}, "
                f"{r['category']}/{r['severity']}{promoted}) ---"
            )
            lines.append(f"  Task: {r['task_description']}")
            lines.append(f"  Outcome: {r['outcome']}")
            lines.append(f"  Reflection: {r['reflection_text']}")
            if r.get("lessons"):
                lines.append("  Lessons:")
                for lesson in r["lessons"]:
                    lines.append(f"    - {lesson}")
            lines.append(f"  Retrieved {r['times_retrieved']}x | Created: {r['created_at'][:10]}")
            if r.get("session_number") is not None:
                lines.append(f"  Session: #{r['session_number']}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── search_reflections ───────────────────────────
    elif action == "search_reflections":
        try:
            results = engine.search_reflections(
                category=arguments.get("category"),
                severity=arguments.get("severity"),
                limit=arguments.get("limit", 20),
                sort_by=arguments.get("sort_by", "recent"),
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error browsing reflections: {e}")]

        if not results:
            return [TextContent(type="text", text="No reflections found.")]

        lines = [f"Reflections ({len(results)} results):\n"]
        for r in results:
            promoted = " [PROMOTED]" if r.get("promoted_to_directive") else ""
            retrieved = f", retrieved {r['times_retrieved']}x" if r["times_retrieved"] else ""
            lines.append(
                f"  [{r['id']}] {r['category']}/{r['severity']}{promoted}{retrieved} "
                f"({r['created_at'][:10]})"
            )
            lines.append(f"       Task: {r['task_description'][:80]}")
            lines.append(f"       Outcome: {r['outcome'][:60]}")
            if r.get("lessons"):
                lessons_preview = "; ".join(r["lessons"][:2])
                if len(r["lessons"]) > 2:
                    lessons_preview += f" (+{len(r['lessons']) - 2} more)"
                lines.append(f"       Lessons: {lessons_preview}")
            lines.append("")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── promote_to_directive ─────────────────────────
    elif action == "promote_to_directive":
        reflection_id = arguments.get("reflection_id")
        if reflection_id is None:
            return [TextContent(
                type="text",
                text="Need 'reflection_id' to promote.",
            )]

        try:
            result = engine.promote_to_directive(
                reflection_id=reflection_id,
                rule=arguments.get("rule"),
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error promoting reflection: {e}")]

        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        return [TextContent(
            type="text",
            text=(
                f"Reflection #{result['reflection_id']} promoted to directive.\n"
                f"  Rule: {result['directive_rule']}\n"
                f"  Source: {result['source']}"
            ),
        )]

    # ── reflection_stats ─────────────────────────────
    elif action == "reflection_stats":
        try:
            stats = engine.reflection_stats()
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting stats: {e}")]

        lines = [f"Reflection Engine Stats:\n"]
        lines.append(f"  Total reflections: {stats['total']}")
        lines.append(f"  Promoted to directives: {stats['promoted_to_directives']}")

        if stats["by_category"]:
            lines.append("\n  By category:")
            for cat, cnt in stats["by_category"].items():
                lines.append(f"    {cat}: {cnt}")

        if stats["by_severity"]:
            lines.append("\n  By severity:")
            for sev, cnt in stats["by_severity"].items():
                lines.append(f"    {sev}: {cnt}")

        if stats["most_retrieved"]:
            lines.append("\n  Most retrieved:")
            for r in stats["most_retrieved"]:
                lines.append(f"    [{r['id']}] {r['task_description'][:60]} ({r['times_retrieved']}x)")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── Unknown action ───────────────────────────────
    else:
        return [TextContent(
            type="text",
            text=f"Unknown reflect_engine action: {action}. Valid: {', '.join(REFLECT_ACTIONS)}",
        )]


HANDLERS = {"watty_reflect_engine": handle_reflect_engine}
