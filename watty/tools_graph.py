"""
tools_graph.py -- Knowledge Graph MCP Tool
=============================================
One tool: watty_graph(action=...)
Actions: search_entities, get_entity, traverse, add_entity, merge_entities, graph_stats, backfill

Structured entity/relationship graph over Watty's flat memory.
Entities are people, projects, concepts, tools. Relationships connect them.
Graph traversal finds what similarity search alone cannot.

Hugo & Claude -- February 2026
"""

import json
from datetime import datetime, timezone

from mcp.types import Tool, TextContent

from watty.embeddings import embed_text


# ── Brain Reference ───────────────────────────────────────────

_brain_ref = None


def set_brain(brain):
    """Called by server.py to give graph tools access to the KnowledgeGraph via brain._kg."""
    global _brain_ref
    _brain_ref = brain


# ── Actions ───────────────────────────────────────────────────

GRAPH_ACTIONS = [
    "search_entities", "get_entity", "traverse",
    "add_entity", "merge_entities",
    "graph_stats", "backfill",
]


# ── Tool Definition ───────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_graph",
        description=(
            "Watty's Knowledge Graph. Structured entities and relationships "
            "extracted from memories via local Ollama.\n\n"
            "Actions:\n"
            "  search_entities -- Find entities by name or semantic similarity\n"
            "  get_entity -- Full detail on one entity (relationships, chunks)\n"
            "  traverse -- BFS graph traversal from an entity (follow relationships)\n"
            "  add_entity -- Manually create an entity\n"
            "  merge_entities -- Merge two entities into one\n"
            "  graph_stats -- Graph overview (counts, distributions, top entities)\n"
            "  backfill -- Extract entities from existing chunks not yet processed\n\n"
            "Use traverse to discover connections that flat recall misses. "
            "Use search_entities to find who/what is in the graph. "
            "Use backfill to populate the graph from existing memories."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": GRAPH_ACTIONS,
                    "description": "Action to perform",
                },
                "query": {
                    "type": "string",
                    "description": "search_entities: Search query",
                },
                "entity_id": {
                    "type": "integer",
                    "description": "get_entity: Entity ID to look up",
                },
                "entity_name": {
                    "type": "string",
                    "description": "traverse/add_entity: Entity name",
                },
                "entity_type": {
                    "type": "string",
                    "description": (
                        "add_entity: Entity type. "
                        "One of: person, project, concept, tool, location, organization, event, other"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "add_entity: Entity description",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "traverse: Max traversal depth (default 2, max 3)",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "traverse: Filter by relationship types. "
                        "Options: works_on, uses, created, part_of, related_to, "
                        "knows, located_in, depends_on, contradicts, evolved_from"
                    ),
                },
                "keep_id": {
                    "type": "integer",
                    "description": "merge_entities: ID of entity to keep",
                },
                "merge_id": {
                    "type": "integer",
                    "description": "merge_entities: ID of entity to merge (absorbed)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "search_entities: Number of results (default 10)",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "backfill: Chunks to process per call (default 50)",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ───────────────────────────────────────────────────

async def handle_graph(arguments: dict) -> list[TextContent]:
    if _brain_ref is None or not hasattr(_brain_ref, '_kg') or _brain_ref._kg is None:
        return [TextContent(
            type="text",
            text="Knowledge Graph not available. Ensure KG is enabled and Ollama is running.",
        )]

    kg = _brain_ref._kg
    action = arguments.get("action", "")

    # ── search_entities ──────────────────────────────────
    if action == "search_entities":
        query = arguments.get("query")
        if not query:
            return [TextContent(type="text", text="Need a 'query' to search entities.")]

        top_k = arguments.get("top_k", 10)
        results = kg.entity_search(query, top_k=top_k)

        if not results:
            return [TextContent(type="text", text=f"No entities found for: {query}")]

        lines = [f"Found {len(results)} entities:\n"]
        for r in results:
            desc = f" -- {r['description']}" if r.get("description") else ""
            lines.append(
                f"  [{r['entity_id']}] {r['name']} ({r['entity_type']}) "
                f"mentions={r['mention_count']} match={r['match_type']} "
                f"sim={r['similarity']}{desc}"
            )
        return [TextContent(type="text", text="\n".join(lines))]

    # ── get_entity ───────────────────────────────────────
    elif action == "get_entity":
        entity_id = arguments.get("entity_id")
        if entity_id is None:
            return [TextContent(type="text", text="Need 'entity_id' to look up.")]

        detail = kg.get_entity(int(entity_id))
        if not detail:
            return [TextContent(type="text", text=f"Entity {entity_id} not found.")]

        lines = [
            f"Entity: {detail['name']} (ID: {detail['entity_id']})",
            f"  Type: {detail['entity_type']}",
            f"  Description: {detail.get('description', 'none')}",
            f"  Mentions: {detail['mention_count']}",
            f"  First seen: {detail['first_seen']}",
            f"  Last seen: {detail['last_seen']}",
        ]

        if detail["outgoing"]:
            lines.append(f"\nOutgoing relationships ({len(detail['outgoing'])}):")
            for r in detail["outgoing"]:
                lines.append(
                    f"  -> {r['target_name']} ({r['relationship_type']}) "
                    f"strength={r['strength']:.2f}"
                )

        if detail["incoming"]:
            lines.append(f"\nIncoming relationships ({len(detail['incoming'])}):")
            for r in detail["incoming"]:
                lines.append(
                    f"  <- {r['source_name']} ({r['relationship_type']}) "
                    f"strength={r['strength']:.2f}"
                )

        if detail["linked_chunks"]:
            lines.append(f"\nLinked chunks ({len(detail['linked_chunks'])}):")
            for ch in detail["linked_chunks"][:5]:
                preview = ch["content"][:120].replace("\n", " ")
                lines.append(f"  [{ch['chunk_id']}] {preview}...")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── traverse ─────────────────────────────────────────
    elif action == "traverse":
        entity_name = arguments.get("entity_name")
        if not entity_name:
            return [TextContent(type="text", text="Need 'entity_name' to traverse from.")]

        max_hops = min(arguments.get("max_hops", 2), 3)  # Cap at 3
        rel_types = arguments.get("relationship_types")

        result = kg.traverse_graph(
            entity_name,
            max_hops=max_hops,
            relationship_types=rel_types,
        )

        if result.get("error"):
            return [TextContent(type="text", text=result["error"])]

        entities = result.get("entities", [])
        rels = result.get("relationships", [])

        lines = [
            f"Graph traversal from '{result['root_entity']}' ({max_hops} hops):",
            f"  {len(entities)} entities, {len(rels)} relationships\n",
        ]

        # Group entities by hop distance
        by_hop: dict[int, list] = {}
        for e in entities:
            hop = e["hop_distance"]
            if hop not in by_hop:
                by_hop[hop] = []
            by_hop[hop].append(e)

        for hop in sorted(by_hop.keys()):
            label = "ROOT" if hop == 0 else f"Hop {hop}"
            lines.append(f"  {label}:")
            for e in by_hop[hop]:
                desc = f" -- {e['description']}" if e.get("description") else ""
                lines.append(
                    f"    [{e['entity_id']}] {e['name']} ({e['entity_type']}) "
                    f"mentions={e['mention_count']}{desc}"
                )

        if rels:
            lines.append(f"\n  Relationships:")
            for r in rels[:30]:  # Cap display at 30
                lines.append(
                    f"    {r['source_id']} --[{r['relationship_type']}]--> "
                    f"{r['target_id']} (strength={r['strength']:.2f})"
                )
            if len(rels) > 30:
                lines.append(f"    ... and {len(rels) - 30} more")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── add_entity ───────────────────────────────────────
    elif action == "add_entity":
        entity_name = arguments.get("entity_name")
        if not entity_name:
            return [TextContent(type="text", text="Need 'entity_name' to create.")]

        entity_type = arguments.get("entity_type", "concept")
        description = arguments.get("description", "")
        name_norm = entity_name.lower().strip()
        now = datetime.now(timezone.utc).isoformat()

        valid_types = {
            "person", "project", "concept", "tool",
            "location", "organization", "event", "other",
        }
        if entity_type not in valid_types:
            entity_type = "other"

        # Check for existing
        import sqlite3
        conn = sqlite3.connect(kg.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT id, name FROM entities WHERE name_normalized = ?",
            (name_norm,)
        ).fetchone()

        if existing:
            conn.close()
            return [TextContent(
                type="text",
                text=f"Entity already exists: [{existing['id']}] {existing['name']}",
            )]

        # Embed and insert
        embed_input = f"{entity_name}: {description}" if description else entity_name
        embedding = embed_text(embed_input)

        cursor = conn.execute(
            "INSERT INTO entities "
            "(name, name_normalized, entity_type, description, embedding, "
            "first_seen, last_seen, mention_count, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, '{}')",
            (entity_name, name_norm, entity_type, description,
             embedding.tobytes(), now, now)
        )
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return [TextContent(
            type="text",
            text=f"Created entity [{new_id}] {entity_name} ({entity_type})"
            + (f": {description}" if description else ""),
        )]

    # ── merge_entities ───────────────────────────────────
    elif action == "merge_entities":
        keep_id = arguments.get("keep_id")
        merge_id = arguments.get("merge_id")
        if keep_id is None or merge_id is None:
            return [TextContent(type="text", text="Need both 'keep_id' and 'merge_id'.")]

        result = kg.merge_entities(int(keep_id), int(merge_id))
        if result.get("error"):
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        return [TextContent(
            type="text",
            text=(
                f"Merged: [{result['absorbed']['id']}] {result['absorbed']['name']} "
                f"-> [{result['kept']['id']}] {result['kept']['name']}"
            ),
        )]

    # ── graph_stats ──────────────────────────────────────
    elif action == "graph_stats":
        stats = kg.graph_stats()

        lines = [
            "Knowledge Graph Stats:",
            f"  Entities: {stats['total_entities']}",
            f"  Relationships: {stats['total_relationships']}",
            f"  Entity-Chunk links: {stats['total_links']}",
        ]

        if stats["entity_type_distribution"]:
            lines.append("\n  Entity types:")
            for etype, count in stats["entity_type_distribution"].items():
                lines.append(f"    {etype}: {count}")

        if stats["relationship_type_distribution"]:
            lines.append("\n  Relationship types:")
            for rtype, count in stats["relationship_type_distribution"].items():
                lines.append(f"    {rtype}: {count}")

        if stats["top_entities"]:
            lines.append("\n  Top entities (by mentions):")
            for e in stats["top_entities"][:15]:
                lines.append(
                    f"    [{e['entity_id']}] {e['name']} ({e['entity_type']}) "
                    f"mentions={e['mention_count']} chunks={e['chunk_count']}"
                )

        return [TextContent(type="text", text="\n".join(lines))]

    # ── backfill ─────────────────────────────────────────
    elif action == "backfill":
        batch_size = arguments.get("batch_size", 50)
        result = kg.batch_extract_existing(batch_size=batch_size)

        return [TextContent(
            type="text",
            text=(
                f"Backfill complete:\n"
                f"  Processed: {result['processed']} chunks\n"
                f"  Remaining: {result['remaining']} chunks\n"
                f"  Entities extracted: {result['entities_extracted']}\n"
                f"  Relationships extracted: {result['relationships_extracted']}"
            ),
        )]

    # ── Unknown action ───────────────────────────────────
    else:
        return [TextContent(
            type="text",
            text=f"Unknown graph action: {action}. Valid: {', '.join(GRAPH_ACTIONS)}",
        )]


# ── Router ────────────────────────────────────────────────────

HANDLERS = {"watty_graph": handle_graph}
