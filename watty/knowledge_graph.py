"""
Watty Knowledge Graph Engine
==============================
Extracts structured entities and relationships from memories via local Ollama,
stores in SQLite, provides graph traversal and graph-augmented recall.

The graph turns flat memory into a connected web. Entities are people, projects,
concepts, tools. Relationships are how they connect. Traversal follows the web
to find things similarity search misses.

Extraction is async and non-blocking. Ollama runs locally. No data leaves the machine.

Hugo & Claude -- February 2026
"""

import json
import sqlite3
import threading
import queue
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

from watty.config import (
    DB_PATH, EMBEDDING_DIMENSION, KG_OLLAMA_URL, KG_OLLAMA_MODEL,
    KG_EXTRACTION_TIMEOUT, KG_MAX_ENTITIES_PER_CHUNK, KG_MERGE_SIMILARITY_THRESHOLD,
    KG_TRAVERSAL_MAX_HOPS, KG_RRF_K,
)
from watty.embeddings import embed_text


# ── Extraction Prompt ────────────────────────────────────────

_EXTRACT_PROMPT = """You are a knowledge graph extraction engine. Extract structured entities and relationships from the text below.

Return a JSON object with exactly two keys:
- "entities": array of objects with keys: "name" (string), "type" (string), "description" (string, 1 sentence)
- "relationships": array of objects with keys: "source" (string, entity name), "target" (string, entity name), "type" (string), "description" (string, 1 sentence)

Entity types (pick one): person, project, concept, tool, location, organization, event, other
Relationship types (pick one): works_on, uses, created, part_of, related_to, knows, located_in, depends_on, contradicts, evolved_from

Rules:
- Extract up to {max_entities} entities maximum.
- Entity names should be normalized (e.g. "Python" not "python language").
- Only extract entities and relationships that are clearly stated or strongly implied.
- If the text has no extractable entities, return empty arrays.

Text:
{text}

JSON:"""


class KnowledgeGraph:
    """
    The graph layer on top of Watty's brain. Entities are nodes.
    Relationships are edges. Chunks are evidence. The graph connects
    what similarity search alone cannot.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._write_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    # ── Connection ────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # ── Extraction ────────────────────────────────────────────

    def _ollama_extract(self, text: str) -> dict:
        """
        Ask local Ollama to extract entities and relationships from text.
        Returns {"entities": [...], "relationships": [...]}.
        On any error, returns empty lists -- never blocks the pipeline.
        """
        if not text or len(text.strip()) < 20:
            return {"entities": [], "relationships": []}

        prompt = _EXTRACT_PROMPT.format(
            max_entities=KG_MAX_ENTITIES_PER_CHUNK,
            text=text[:3000],  # Cap input to avoid Ollama timeouts
        )

        try:
            resp = requests.post(
                f"{KG_OLLAMA_URL}/api/generate",
                json={
                    "model": KG_OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                },
                timeout=KG_EXTRACTION_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

            parsed = json.loads(raw)
            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            # Validate entity structure
            valid_entity_types = {
                "person", "project", "concept", "tool",
                "location", "organization", "event", "other",
            }
            validated_entities = []
            for e in entities[:KG_MAX_ENTITIES_PER_CHUNK]:
                if not isinstance(e, dict) or not e.get("name"):
                    continue
                etype = e.get("type", "other").lower()
                if etype not in valid_entity_types:
                    etype = "other"
                validated_entities.append({
                    "name": str(e["name"]).strip(),
                    "type": etype,
                    "description": str(e.get("description", "")).strip(),
                })

            # Validate relationship structure
            valid_rel_types = {
                "works_on", "uses", "created", "part_of", "related_to",
                "knows", "located_in", "depends_on", "contradicts", "evolved_from",
            }
            validated_rels = []
            for r in relationships:
                if not isinstance(r, dict):
                    continue
                src = str(r.get("source", "")).strip()
                tgt = str(r.get("target", "")).strip()
                if not src or not tgt:
                    continue
                rtype = r.get("type", "related_to").lower()
                if rtype not in valid_rel_types:
                    rtype = "related_to"
                validated_rels.append({
                    "source": src,
                    "target": tgt,
                    "type": rtype,
                    "description": str(r.get("description", "")).strip(),
                })

            return {"entities": validated_entities, "relationships": validated_rels}

        except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError):
            return {"entities": [], "relationships": []}

    def extract_and_store(self, chunk_id: int, content: str) -> dict:
        """
        Extract entities and relationships from a chunk, then store in the graph.
        Upserts entities by normalized name. Links to chunk. Creates relationships.
        """
        extracted = self._ollama_extract(content)
        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        if not entities and not relationships:
            return {"entities_stored": 0, "relationships_stored": 0}

        with self._write_lock:
            conn = self._connect()
            try:
                now = datetime.now(timezone.utc).isoformat()
                name_to_id: dict[str, int] = {}
                entities_stored = 0
                relationships_stored = 0

                # ── Upsert entities ──
                for ent in entities:
                    name = ent["name"]
                    name_norm = name.lower().strip()
                    etype = ent["type"]
                    desc = ent.get("description", "")

                    existing = conn.execute(
                        "SELECT id, mention_count FROM entities WHERE name_normalized = ?",
                        (name_norm,)
                    ).fetchone()

                    if existing:
                        entity_id = existing["id"]
                        conn.execute(
                            "UPDATE entities SET mention_count = mention_count + 1, "
                            "last_seen = ? WHERE id = ?",
                            (now, entity_id)
                        )
                    else:
                        # Embed entity name + description for semantic search
                        embed_text_input = f"{name}: {desc}" if desc else name
                        embedding = embed_text(embed_text_input)

                        cursor = conn.execute(
                            "INSERT INTO entities "
                            "(name, name_normalized, entity_type, description, embedding, "
                            "first_seen, last_seen, mention_count, metadata) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, '{}')",
                            (name, name_norm, etype, desc, embedding.tobytes(), now, now)
                        )
                        entity_id = cursor.lastrowid
                        entities_stored += 1

                    name_to_id[name_norm] = entity_id

                    # Link entity to chunk
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id, extracted_at) "
                        "VALUES (?, ?, ?)",
                        (entity_id, chunk_id, now)
                    )

                # ── Upsert relationships ──
                for rel in relationships:
                    src_norm = rel["source"].lower().strip()
                    tgt_norm = rel["target"].lower().strip()
                    rtype = rel["type"]
                    desc = rel.get("description", "")

                    # Resolve entity IDs (may need to look up if entity wasn't in this batch)
                    src_id = name_to_id.get(src_norm)
                    tgt_id = name_to_id.get(tgt_norm)

                    if src_id is None:
                        row = conn.execute(
                            "SELECT id FROM entities WHERE name_normalized = ?",
                            (src_norm,)
                        ).fetchone()
                        if row:
                            src_id = row["id"]
                    if tgt_id is None:
                        row = conn.execute(
                            "SELECT id FROM entities WHERE name_normalized = ?",
                            (tgt_norm,)
                        ).fetchone()
                        if row:
                            tgt_id = row["id"]

                    if src_id is None or tgt_id is None:
                        continue  # Can't create relationship without both ends
                    if src_id == tgt_id:
                        continue  # Skip self-referencing

                    existing_rel = conn.execute(
                        "SELECT id, strength, evidence_chunk_ids FROM relationships "
                        "WHERE source_entity_id = ? AND target_entity_id = ? AND relationship_type = ?",
                        (src_id, tgt_id, rtype)
                    ).fetchone()

                    if existing_rel:
                        # Strengthen and append evidence
                        new_strength = min(1.0, existing_rel["strength"] + 0.1)
                        try:
                            evidence = json.loads(existing_rel["evidence_chunk_ids"] or "[]")
                        except json.JSONDecodeError:
                            evidence = []
                        if chunk_id not in evidence:
                            evidence.append(chunk_id)
                        conn.execute(
                            "UPDATE relationships SET strength = ?, evidence_chunk_ids = ?, "
                            "last_seen = ? WHERE id = ?",
                            (new_strength, json.dumps(evidence), now, existing_rel["id"])
                        )
                    else:
                        conn.execute(
                            "INSERT INTO relationships "
                            "(source_entity_id, target_entity_id, relationship_type, "
                            "description, strength, evidence_chunk_ids, first_seen, last_seen) "
                            "VALUES (?, ?, ?, ?, 0.5, ?, ?, ?)",
                            (src_id, tgt_id, rtype, desc,
                             json.dumps([chunk_id]), now, now)
                        )
                        relationships_stored += 1

                conn.commit()
                return {
                    "entities_stored": entities_stored,
                    "relationships_stored": relationships_stored,
                    "entities_total": len(entities),
                    "relationships_total": len(relationships),
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ── Async Extraction Queue ────────────────────────────────

    def enqueue_extraction(self, chunk_id: int, content: str):
        """Put a chunk on the extraction queue for background processing."""
        self._queue.put((chunk_id, content))

    def start_worker(self):
        """Start the background extraction worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="watty-kg-worker"
        )
        self._worker_thread.start()

    def stop_worker(self):
        """Stop the background extraction worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

    def _worker_loop(self):
        """Background loop: pull from queue, extract, store. Never crash."""
        while self._running:
            try:
                chunk_id, content = self._queue.get(timeout=2)
            except queue.Empty:
                continue
            try:
                self.extract_and_store(chunk_id, content)
            except Exception:
                pass  # Log in production; never kill the worker

    # ── Search Methods ────────────────────────────────────────

    def entity_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Two-stage entity search: name match (SQL LIKE) + semantic (embedding cosine).
        Merged and deduplicated. Returns entities with match_type and similarity.
        """
        if not query or len(query.strip()) < 2:
            return []

        conn = self._connect()
        results: dict[int, dict] = {}

        # Stage 1: SQL LIKE name match
        pattern = f"%{query.lower()}%"
        name_rows = conn.execute(
            "SELECT id, name, name_normalized, entity_type, description, "
            "mention_count, first_seen, last_seen "
            "FROM entities WHERE name_normalized LIKE ? "
            "ORDER BY mention_count DESC LIMIT ?",
            (pattern, top_k)
        ).fetchall()

        for row in name_rows:
            results[row["id"]] = {
                "entity_id": row["id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "description": row["description"],
                "mention_count": row["mention_count"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "match_type": "name",
                "similarity": 1.0,  # Exact name match gets max
            }

        # Stage 2: Semantic embedding search
        query_vec = embed_text(query)
        all_entities = conn.execute(
            "SELECT id, name, name_normalized, entity_type, description, "
            "mention_count, first_seen, last_seen, embedding "
            "FROM entities WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        semantic_candidates = []
        for row in all_entities:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(emb) != EMBEDDING_DIMENSION:
                continue
            sim = float(np.dot(query_vec, emb))
            if sim > 0.3:
                semantic_candidates.append((row, sim))

        semantic_candidates.sort(key=lambda x: x[1], reverse=True)

        for row, sim in semantic_candidates[:top_k]:
            eid = row["id"]
            if eid in results:
                # Upgrade similarity if semantic is higher
                if sim > results[eid]["similarity"]:
                    results[eid]["similarity"] = round(sim, 4)
                    results[eid]["match_type"] = "both"
            else:
                results[eid] = {
                    "entity_id": eid,
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "description": row["description"],
                    "mention_count": row["mention_count"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "match_type": "semantic",
                    "similarity": round(sim, 4),
                }

        # Sort by similarity descending, then by mention_count
        sorted_results = sorted(
            results.values(),
            key=lambda x: (x["similarity"], x["mention_count"]),
            reverse=True,
        )
        return sorted_results[:top_k]

    def get_entity(self, entity_id: int) -> Optional[dict]:
        """
        Full entity detail: info + outgoing relationships + incoming relationships
        + linked chunks (last 20).
        """
        conn = self._connect()

        entity = conn.execute(
            "SELECT id, name, name_normalized, entity_type, description, "
            "mention_count, first_seen, last_seen, metadata "
            "FROM entities WHERE id = ?",
            (entity_id,)
        ).fetchone()

        if not entity:
            conn.close()
            return None

        result = {
            "entity_id": entity["id"],
            "name": entity["name"],
            "name_normalized": entity["name_normalized"],
            "entity_type": entity["entity_type"],
            "description": entity["description"],
            "mention_count": entity["mention_count"],
            "first_seen": entity["first_seen"],
            "last_seen": entity["last_seen"],
            "metadata": json.loads(entity["metadata"] or "{}"),
        }

        # Outgoing relationships
        outgoing = conn.execute(
            "SELECT r.id, r.target_entity_id, r.relationship_type, r.description, "
            "r.strength, r.evidence_chunk_ids, e.name as target_name, e.entity_type as target_type "
            "FROM relationships r JOIN entities e ON r.target_entity_id = e.id "
            "WHERE r.source_entity_id = ? ORDER BY r.strength DESC",
            (entity_id,)
        ).fetchall()
        result["outgoing"] = [{
            "relationship_id": r["id"],
            "target_id": r["target_entity_id"],
            "target_name": r["target_name"],
            "target_type": r["target_type"],
            "relationship_type": r["relationship_type"],
            "description": r["description"],
            "strength": r["strength"],
            "evidence_chunks": json.loads(r["evidence_chunk_ids"] or "[]"),
        } for r in outgoing]

        # Incoming relationships
        incoming = conn.execute(
            "SELECT r.id, r.source_entity_id, r.relationship_type, r.description, "
            "r.strength, r.evidence_chunk_ids, e.name as source_name, e.entity_type as source_type "
            "FROM relationships r JOIN entities e ON r.source_entity_id = e.id "
            "WHERE r.target_entity_id = ? ORDER BY r.strength DESC",
            (entity_id,)
        ).fetchall()
        result["incoming"] = [{
            "relationship_id": r["id"],
            "source_id": r["source_entity_id"],
            "source_name": r["source_name"],
            "source_type": r["source_type"],
            "relationship_type": r["relationship_type"],
            "description": r["description"],
            "strength": r["strength"],
            "evidence_chunks": json.loads(r["evidence_chunk_ids"] or "[]"),
        } for r in incoming]

        # Linked chunks (last 20)
        chunks = conn.execute(
            "SELECT ec.chunk_id, ec.extracted_at, c.content, c.created_at, c.source_type "
            "FROM entity_chunks ec JOIN chunks c ON ec.chunk_id = c.id "
            "WHERE ec.entity_id = ? ORDER BY ec.extracted_at DESC LIMIT 20",
            (entity_id,)
        ).fetchall()
        result["linked_chunks"] = [{
            "chunk_id": ch["chunk_id"],
            "extracted_at": ch["extracted_at"],
            "content": ch["content"][:300],
            "created_at": ch["created_at"],
            "source_type": ch["source_type"],
        } for ch in chunks]

        conn.close()
        return result

    def traverse_graph(
        self,
        entity_name: str,
        max_hops: Optional[int] = None,
        relationship_types: Optional[list[str]] = None,
    ) -> dict:
        """
        BFS traversal from a named entity through the relationship graph.
        Returns entities with hop distance and all traversed relationships.
        Fuzzy name matching if exact not found.
        """
        if max_hops is None:
            max_hops = KG_TRAVERSAL_MAX_HOPS

        conn = self._connect()

        # Find root entity -- exact match first, then fuzzy
        name_norm = entity_name.lower().strip()
        root = conn.execute(
            "SELECT id, name FROM entities WHERE name_normalized = ?",
            (name_norm,)
        ).fetchone()

        if not root:
            # Fuzzy fallback
            root = conn.execute(
                "SELECT id, name FROM entities WHERE name_normalized LIKE ? "
                "ORDER BY mention_count DESC LIMIT 1",
                (f"%{name_norm}%",)
            ).fetchone()

        if not root:
            conn.close()
            return {
                "entities": [],
                "relationships": [],
                "root_entity": None,
                "hops": 0,
                "error": f"Entity not found: {entity_name}",
            }

        root_id = root["id"]

        # BFS
        visited: dict[int, int] = {root_id: 0}  # entity_id -> hop distance
        frontier = [root_id]
        all_relationships = []
        current_hop = 0

        while frontier and current_hop < max_hops:
            current_hop += 1
            next_frontier = []

            for eid in frontier:
                # Outgoing edges
                query = (
                    "SELECT r.id, r.source_entity_id, r.target_entity_id, "
                    "r.relationship_type, r.description, r.strength "
                    "FROM relationships r WHERE r.source_entity_id = ?"
                )
                params: list = [eid]

                if relationship_types:
                    placeholders = ",".join("?" * len(relationship_types))
                    query += f" AND r.relationship_type IN ({placeholders})"
                    params.extend(relationship_types)

                outgoing = conn.execute(query, params).fetchall()

                for rel in outgoing:
                    target_id = rel["target_entity_id"]
                    all_relationships.append({
                        "relationship_id": rel["id"],
                        "source_id": rel["source_entity_id"],
                        "target_id": target_id,
                        "relationship_type": rel["relationship_type"],
                        "description": rel["description"],
                        "strength": rel["strength"],
                    })
                    if target_id not in visited:
                        visited[target_id] = current_hop
                        next_frontier.append(target_id)

                # Incoming edges
                query = (
                    "SELECT r.id, r.source_entity_id, r.target_entity_id, "
                    "r.relationship_type, r.description, r.strength "
                    "FROM relationships r WHERE r.target_entity_id = ?"
                )
                params = [eid]

                if relationship_types:
                    placeholders = ",".join("?" * len(relationship_types))
                    query += f" AND r.relationship_type IN ({placeholders})"
                    params.extend(relationship_types)

                incoming = conn.execute(query, params).fetchall()

                for rel in incoming:
                    source_id = rel["source_entity_id"]
                    all_relationships.append({
                        "relationship_id": rel["id"],
                        "source_id": source_id,
                        "target_id": rel["target_entity_id"],
                        "relationship_type": rel["relationship_type"],
                        "description": rel["description"],
                        "strength": rel["strength"],
                    })
                    if source_id not in visited:
                        visited[source_id] = current_hop
                        next_frontier.append(source_id)

            frontier = next_frontier

        # Fetch entity details for all visited nodes
        entity_ids = list(visited.keys())
        entities = []
        if entity_ids:
            placeholders = ",".join("?" * len(entity_ids))
            rows = conn.execute(
                f"SELECT id, name, name_normalized, entity_type, description, mention_count "
                f"FROM entities WHERE id IN ({placeholders})",
                entity_ids
            ).fetchall()
            for row in rows:
                entities.append({
                    "entity_id": row["id"],
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "description": row["description"],
                    "mention_count": row["mention_count"],
                    "hop_distance": visited[row["id"]],
                })

        entities.sort(key=lambda x: (x["hop_distance"], -x["mention_count"]))

        # Deduplicate relationships by id
        seen_rel_ids = set()
        deduped_rels = []
        for rel in all_relationships:
            rid = rel["relationship_id"]
            if rid not in seen_rel_ids:
                seen_rel_ids.add(rid)
                deduped_rels.append(rel)

        conn.close()

        return {
            "entities": entities,
            "relationships": deduped_rels,
            "root_entity": root["name"],
            "hops": max_hops,
        }

    def get_entity_neighborhood(self, entity_id: int) -> dict:
        """Convenience: 1-hop traversal from an entity by ID."""
        conn = self._connect()
        entity = conn.execute(
            "SELECT name FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        conn.close()

        if not entity:
            return {"entities": [], "relationships": [], "root_entity": None, "hops": 1}

        return self.traverse_graph(entity["name"], max_hops=1)

    # ── Recall Integration ────────────────────────────────────

    def graph_recall_chunks(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Graph-augmented chunk recall. Finds entities matching the query,
        gathers chunk IDs from those entities and their 1-hop neighbors,
        scores using Reciprocal Rank Fusion.

        Returns (chunk_id, score) pairs sorted by score descending.
        """
        # Find top-5 entities matching query
        top_entities = self.entity_search(query, top_k=5)
        if not top_entities:
            return []

        conn = self._connect()
        chunk_scores: dict[int, float] = defaultdict(float)

        for rank, ent in enumerate(top_entities):
            eid = ent["entity_id"]

            # Direct chunks from this entity
            direct_chunks = conn.execute(
                "SELECT chunk_id FROM entity_chunks WHERE entity_id = ? "
                "ORDER BY extracted_at DESC LIMIT 20",
                (eid,)
            ).fetchall()

            for i, row in enumerate(direct_chunks):
                cid = row["chunk_id"]
                score = 1.0 / (KG_RRF_K + rank + i + 1)
                chunk_scores[cid] += score

            # 1-hop neighbor chunks (half weight)
            neighbors = conn.execute(
                "SELECT DISTINCT target_entity_id as neighbor_id FROM relationships "
                "WHERE source_entity_id = ? "
                "UNION "
                "SELECT DISTINCT source_entity_id as neighbor_id FROM relationships "
                "WHERE target_entity_id = ?",
                (eid, eid)
            ).fetchall()

            for neighbor in neighbors:
                nid = neighbor["neighbor_id"]
                neighbor_chunks = conn.execute(
                    "SELECT chunk_id FROM entity_chunks WHERE entity_id = ? "
                    "ORDER BY extracted_at DESC LIMIT 10",
                    (nid,)
                ).fetchall()

                for i, row in enumerate(neighbor_chunks):
                    cid = row["chunk_id"]
                    score = 0.5 / (KG_RRF_K + rank + i + 1)
                    chunk_scores[cid] += score

        conn.close()

        # Sort by score, return top_k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:top_k]

    # ── Dream Maintenance ─────────────────────────────────────

    def dream_maintenance(self) -> dict:
        """
        Graph hygiene during dream cycle.
        Phase 1: Merge duplicate entities (same name_normalized)
        Phase 2: Strengthen relationships with 3+ evidence chunks
        Phase 3: Prune orphaned entities
        Phase 4: Decay weak relationships
        """
        with self._write_lock:
            conn = self._connect()
            try:
                entities_merged = 0
                entities_pruned = 0
                relationships_pruned = 0
                relationships_strengthened = 0

                # ── Phase 1: Merge duplicate entities ──
                # Find name_normalized values that appear more than once
                dupes = conn.execute(
                    "SELECT name_normalized, COUNT(*) as cnt FROM entities "
                    "GROUP BY name_normalized HAVING cnt > 1"
                ).fetchall()

                for dupe in dupes:
                    name_norm = dupe["name_normalized"]
                    instances = conn.execute(
                        "SELECT id, mention_count FROM entities "
                        "WHERE name_normalized = ? ORDER BY mention_count DESC",
                        (name_norm,)
                    ).fetchall()

                    if len(instances) < 2:
                        continue

                    keep_id = instances[0]["id"]
                    for merge_inst in instances[1:]:
                        merge_id = merge_inst["id"]
                        self._merge_entities_inner(conn, keep_id, merge_id)
                        entities_merged += 1

                # ── Phase 2: Strengthen relationships with 3+ evidence chunks ──
                strong_rels = conn.execute(
                    "SELECT id, strength, evidence_chunk_ids FROM relationships "
                    "WHERE strength < 1.0"
                ).fetchall()

                for rel in strong_rels:
                    try:
                        evidence = json.loads(rel["evidence_chunk_ids"] or "[]")
                    except json.JSONDecodeError:
                        evidence = []
                    if len(evidence) >= 3:
                        new_strength = min(1.0, rel["strength"] + 0.05)
                        conn.execute(
                            "UPDATE relationships SET strength = ? WHERE id = ?",
                            (new_strength, rel["id"])
                        )
                        relationships_strengthened += 1

                # ── Phase 3: Prune orphaned entities ──
                # No chunks, no relationships, mention_count <= 1
                orphans = conn.execute(
                    "SELECT e.id FROM entities e "
                    "WHERE e.mention_count <= 1 "
                    "AND e.id NOT IN (SELECT entity_id FROM entity_chunks) "
                    "AND e.id NOT IN (SELECT source_entity_id FROM relationships) "
                    "AND e.id NOT IN (SELECT target_entity_id FROM relationships)"
                ).fetchall()

                if orphans:
                    orphan_ids = [o["id"] for o in orphans]
                    placeholders = ",".join("?" * len(orphan_ids))
                    conn.execute(
                        f"DELETE FROM entities WHERE id IN ({placeholders})",
                        orphan_ids
                    )
                    entities_pruned = len(orphan_ids)

                # ── Phase 4: Decay weak relationships ──
                conn.execute(
                    "UPDATE relationships SET strength = strength * 0.95 WHERE strength < 0.5"
                )
                pruned = conn.execute(
                    "DELETE FROM relationships WHERE strength < 0.05"
                ).rowcount
                relationships_pruned = pruned if isinstance(pruned, int) else 0

                # Clean self-referencing relationships
                self_refs = conn.execute(
                    "DELETE FROM relationships WHERE source_entity_id = target_entity_id"
                ).rowcount
                relationships_pruned += self_refs if isinstance(self_refs, int) else 0

                conn.commit()

                return {
                    "entities_merged": entities_merged,
                    "entities_pruned": entities_pruned,
                    "relationships_pruned": relationships_pruned,
                    "relationships_strengthened": relationships_strengthened,
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ── Utility Methods ───────────────────────────────────────

    def _merge_entities_inner(self, conn, keep_id: int, merge_id: int):
        """
        Merge merge_id entity into keep_id. Transfers all chunk links
        and relationship references. Absorbs mention_count.
        Internal: caller must hold write_lock and manage transaction.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Absorb mention_count
        merge_row = conn.execute(
            "SELECT mention_count FROM entities WHERE id = ?", (merge_id,)
        ).fetchone()
        if merge_row:
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + ?, last_seen = ? WHERE id = ?",
                (merge_row["mention_count"], now, keep_id)
            )

        # Transfer chunk links
        conn.execute(
            "UPDATE OR IGNORE entity_chunks SET entity_id = ? WHERE entity_id = ?",
            (keep_id, merge_id)
        )
        # Delete any remaining (duplicates that hit UNIQUE constraint)
        conn.execute(
            "DELETE FROM entity_chunks WHERE entity_id = ?", (merge_id,)
        )

        # Remap relationships -- source side
        conn.execute(
            "UPDATE OR IGNORE relationships SET source_entity_id = ? WHERE source_entity_id = ?",
            (keep_id, merge_id)
        )
        # Remap relationships -- target side
        conn.execute(
            "UPDATE OR IGNORE relationships SET target_entity_id = ? WHERE target_entity_id = ?",
            (keep_id, merge_id)
        )
        # Delete any orphaned relationships from merge_id (hit UNIQUE constraint)
        conn.execute(
            "DELETE FROM relationships WHERE source_entity_id = ? OR target_entity_id = ?",
            (merge_id, merge_id)
        )

        # Delete the merged entity
        conn.execute("DELETE FROM entities WHERE id = ?", (merge_id,))

    def merge_entities(self, keep_id: int, merge_id: int) -> dict:
        """
        Manual entity merge. Merges merge_id into keep_id.
        Returns info about the surviving entity.
        """
        with self._write_lock:
            conn = self._connect()
            try:
                # Verify both exist
                keep = conn.execute(
                    "SELECT name FROM entities WHERE id = ?", (keep_id,)
                ).fetchone()
                merge = conn.execute(
                    "SELECT name FROM entities WHERE id = ?", (merge_id,)
                ).fetchone()

                if not keep:
                    return {"error": f"Keep entity {keep_id} not found"}
                if not merge:
                    return {"error": f"Merge entity {merge_id} not found"}

                self._merge_entities_inner(conn, keep_id, merge_id)
                conn.commit()

                return {
                    "merged": True,
                    "kept": {"id": keep_id, "name": keep["name"]},
                    "absorbed": {"id": merge_id, "name": merge["name"]},
                }
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def graph_stats(self) -> dict:
        """
        Graph overview: total entities, relationships, links, type distributions,
        top 20 entities by mention count.
        """
        conn = self._connect()

        total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total_relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        total_links = conn.execute("SELECT COUNT(*) FROM entity_chunks").fetchone()[0]

        # Entity type distribution
        entity_types = conn.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities "
            "GROUP BY entity_type ORDER BY cnt DESC"
        ).fetchall()

        # Relationship type distribution
        rel_types = conn.execute(
            "SELECT relationship_type, COUNT(*) as cnt FROM relationships "
            "GROUP BY relationship_type ORDER BY cnt DESC"
        ).fetchall()

        # Top 20 entities by mentions
        top_entities = conn.execute(
            "SELECT id, name, entity_type, mention_count, "
            "(SELECT COUNT(*) FROM entity_chunks WHERE entity_id = entities.id) as chunk_count "
            "FROM entities ORDER BY mention_count DESC LIMIT 20"
        ).fetchall()

        conn.close()

        return {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "total_links": total_links,
            "entity_type_distribution": {
                r["entity_type"]: r["cnt"] for r in entity_types
            },
            "relationship_type_distribution": {
                r["relationship_type"]: r["cnt"] for r in rel_types
            },
            "top_entities": [{
                "entity_id": r["id"],
                "name": r["name"],
                "entity_type": r["entity_type"],
                "mention_count": r["mention_count"],
                "chunk_count": r["chunk_count"],
            } for r in top_entities],
        }

    def batch_extract_existing(self, batch_size: int = 50) -> dict:
        """
        Backfill: extract entities from existing chunks that aren't in entity_chunks yet.
        Processes batch_size chunks per call. Returns progress.
        """
        conn = self._connect()

        # Find chunks not yet processed
        unprocessed = conn.execute(
            "SELECT c.id, c.content FROM chunks c "
            "WHERE c.id NOT IN (SELECT DISTINCT chunk_id FROM entity_chunks) "
            "AND c.content IS NOT NULL AND LENGTH(c.content) > 30 "
            "ORDER BY c.created_at DESC LIMIT ?",
            (batch_size,)
        ).fetchall()

        total_remaining = conn.execute(
            "SELECT COUNT(*) FROM chunks c "
            "WHERE c.id NOT IN (SELECT DISTINCT chunk_id FROM entity_chunks) "
            "AND c.content IS NOT NULL AND LENGTH(c.content) > 30"
        ).fetchone()[0]

        conn.close()

        processed = 0
        entities_total = 0
        relationships_total = 0

        for row in unprocessed:
            try:
                result = self.extract_and_store(row["id"], row["content"])
                entities_total += result.get("entities_stored", 0)
                relationships_total += result.get("relationships_stored", 0)
                processed += 1
            except Exception:
                continue  # Skip failures, keep going

        return {
            "processed": processed,
            "remaining": total_remaining - processed,
            "entities_extracted": entities_total,
            "relationships_extracted": relationships_total,
        }
