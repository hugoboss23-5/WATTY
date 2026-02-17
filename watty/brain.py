"""
Watty Brain v1.0
================
The core memory engine. Stores everything. Retrieves what matters.
Scans unsupervised. Clusters knowledge. Surfaces insights. Forgets on command.

The user can modify it. It learns and adapts.
"""

import sqlite3
import json
import os
import hashlib
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from watty.config import (
    DB_PATH, TOP_K, RELEVANCE_THRESHOLD, RECENCY_WEIGHT,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION, ensure_home,
    SCAN_EXTENSIONS, SCAN_MAX_FILE_SIZE, SCAN_IGNORE_DIRS,
    CLUSTER_MIN_MEMORIES, CLUSTER_SIMILARITY_THRESHOLD,
    SURFACE_TOP_K, SURFACE_NOVELTY_WEIGHT,
    TIER_EPISODIC, TIER_CONSOLIDATED, DG_SEPARATION_THRESHOLD, DG_PROJECTION_DIM, DG_SPARSITY,
    CA3_MAX_ASSOCIATIONS, CA3_COMPLETION_THRESHOLD, CA3_ASSOCIATION_DECAY,
    CA1_NOVELTY_THRESHOLD, CA1_CONTRADICTION_THRESHOLD,
    CONSOLIDATION_DECAY_DAYS, CONSOLIDATION_PROMOTION_THRESHOLD,
    CHESTAHEDRON_GEO_WEIGHT, CHESTAHEDRON_COHERENCE_CONTRA, CHESTAHEDRON_MIGRATION_BATCH,
    KG_ENABLED, REFLECTION_ENABLED, EVAL_ENABLED,
)
from watty.embeddings import embed_text, cosine_similarity
from watty.chestahedron import (
    Chestahedron as ChestahedronCore, ChestaHippocampus, embedding_to_signal,
    CHESTAHEDRON_DIM,
)


class Brain:
    """
    The Watty Brain. Stores everything. Retrieves what matters.
    Organizes itself. Cleans up on command. Surfaces what you need
    before you know you need it.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        ensure_home()
        self._write_lock = threading.Lock()
        self._init_db()
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: list[int] = []
        self._vector_id_to_idx: dict[int, int] = {}
        self._coordinates: Optional[np.ndarray] = None
        self._index_dirty = True
        self._index_lock = threading.Lock()

        # Chestahedron geometric layer
        self.chestahedron = ChestahedronCore()
        self.chesta_hippocampus = ChestaHippocampus()
        self._load_chestahedron_state()

        # v2.3 engines — feature-flagged, lazy, never crash init
        self._reflection = None
        self._eval = None
        self._kg = None
        try:
            if REFLECTION_ENABLED:
                from watty.reflection import ReflectionEngine
                self._reflection = ReflectionEngine(db_path=self.db_path)
        except Exception:
            pass
        try:
            if EVAL_ENABLED:
                from watty.evaluation import EvalEngine
                self._eval = EvalEngine(db_path=self.db_path)
        except Exception:
            pass
        try:
            if KG_ENABLED:
                from watty.knowledge_graph import KnowledgeGraph
                self._kg = KnowledgeGraph(db_path=self.db_path)
                self._kg.start_worker()
        except Exception:
            pass
        self._a2a = None
        try:
            from watty.config import A2A_ENABLED
            if A2A_ENABLED:
                from watty.a2a import A2AEngine
                self._a2a = A2AEngine(db_path=self.db_path)
        except Exception:
            pass

    @property
    def write_lock(self) -> threading.Lock:
        """Lazy write lock — survives hot-reload where __init__ doesn't re-run."""
        if not hasattr(self, '_write_lock') or self._write_lock is None:
            self._write_lock = threading.Lock()
        return self._write_lock

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                conversation_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_type TEXT DEFAULT 'conversation',
                source_path TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                centroid BLOB,
                chunk_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                weight REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scan_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_provider ON chunks(provider);
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type);
            CREATE INDEX IF NOT EXISTS idx_prompts_hash ON prompts(content_hash);
            CREATE INDEX IF NOT EXISTS idx_scan_log_hash ON scan_log(file_hash);
        """)

        # ── Hippocampus schema (additive, won't break existing data) ──
        # ALTER TABLE is not idempotent — check before adding
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        hippo_cols = {
            "memory_tier": "TEXT DEFAULT 'episodic'",
            "significance": "REAL DEFAULT 0.0",
            "access_count": "INTEGER DEFAULT 0",
            "last_accessed": "TEXT",
            "sparse_hash": "BLOB",
            "compressed_content": "TEXT",
            "compression_ratio": "REAL",
            "coordinate": "BLOB",
            "energy": "REAL",
            "importance": "REAL",
        }
        for col, typedef in hippo_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typedef}")

        conn.executescript("""
            -- CA3: Associative links between memories (recurrent connections)
            CREATE TABLE IF NOT EXISTS associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_chunk_id INTEGER NOT NULL,
                target_chunk_id INTEGER NOT NULL,
                strength REAL DEFAULT 1.0,
                association_type TEXT DEFAULT 'co_occurrence',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_chunk_id) REFERENCES chunks(id),
                FOREIGN KEY (target_chunk_id) REFERENCES chunks(id),
                UNIQUE(source_chunk_id, target_chunk_id)
            );

            -- CA1: Mismatch/novelty log
            CREATE TABLE IF NOT EXISTS novelty_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                novelty_score REAL NOT NULL,
                is_contradiction BOOLEAN DEFAULT 0,
                contradicts_chunk_id INTEGER,
                detected_at TEXT NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            -- Consolidation: Schema memories (abstracted patterns)
            CREATE TABLE IF NOT EXISTS schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                embedding BLOB,
                source_chunk_ids TEXT NOT NULL,
                consolidation_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Consolidation: Replay log (which memories got replayed when)
            CREATE TABLE IF NOT EXISTS replay_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                replayed_at TEXT NOT NULL,
                significance_before REAL,
                significance_after REAL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assoc_source ON associations(source_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_assoc_target ON associations(target_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_tier ON chunks(memory_tier);
            CREATE INDEX IF NOT EXISTS idx_chunks_significance ON chunks(significance);
            CREATE INDEX IF NOT EXISTS idx_novelty_unresolved ON novelty_log(resolved);
            CREATE INDEX IF NOT EXISTS idx_chunks_energy ON chunks(energy);

            -- Chestahedron state persistence (singleton)
            CREATE TABLE IF NOT EXISTS chestahedron_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- ── v2.3: Knowledge Graph ──────────────────────────
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                description TEXT,
                embedding BLOB,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                mention_count INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL DEFAULT 'related_to',
                description TEXT,
                strength REAL DEFAULT 1.0,
                evidence_chunk_ids TEXT DEFAULT '[]',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id),
                UNIQUE(source_entity_id, target_entity_id, relationship_type)
            );

            CREATE TABLE IF NOT EXISTS entity_chunks (
                entity_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                extracted_at TEXT NOT NULL,
                PRIMARY KEY (entity_id, chunk_id),
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name_normalized);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_mentions ON entities(mention_count DESC);
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
            CREATE INDEX IF NOT EXISTS idx_entity_chunks_entity ON entity_chunks(entity_id);
            CREATE INDEX IF NOT EXISTS idx_entity_chunks_chunk ON entity_chunks(chunk_id);

            -- ── v2.3: Reflection Engine ────────────────────────
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reflection_text TEXT NOT NULL,
                lessons TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                created_at TEXT NOT NULL,
                category TEXT DEFAULT 'other',
                severity TEXT DEFAULT 'minor',
                times_retrieved INTEGER DEFAULT 0,
                last_retrieved TEXT,
                related_chunk_ids TEXT DEFAULT '[]',
                promoted_to_directive INTEGER DEFAULT 0,
                session_number INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_reflections_category ON reflections(category);
            CREATE INDEX IF NOT EXISTS idx_reflections_severity ON reflections(severity);
            CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);
            CREATE INDEX IF NOT EXISTS idx_reflections_retrieved ON reflections(times_retrieved DESC);

            -- ── v2.3: Evaluation Framework ─────────────────────
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                context TEXT,
                measured_at TEXT NOT NULL,
                session_number INTEGER,
                category TEXT NOT NULL,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS eval_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT DEFAULT 'warning',
                created_at TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_eval_category ON evaluations(category);
            CREATE INDEX IF NOT EXISTS idx_eval_metric ON evaluations(metric_name);
            CREATE INDEX IF NOT EXISTS idx_eval_measured ON evaluations(measured_at);
            CREATE INDEX IF NOT EXISTS idx_eval_session ON evaluations(session_number);
            CREATE INDEX IF NOT EXISTS idx_alerts_ack ON eval_alerts(acknowledged);

            -- ── v2.3: A2A Protocol ─────────────────────────────
            CREATE TABLE IF NOT EXISTS a2a_tasks (
                id TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                agent_url TEXT,
                agent_name TEXT,
                skill_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                input_json TEXT,
                output_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_a2a_status ON a2a_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_a2a_direction ON a2a_tasks(direction);
            CREATE INDEX IF NOT EXISTS idx_a2a_created ON a2a_tasks(created_at);
        """)

        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # ── Hippocampus Pipeline Hooks ────────────────────────
    # These are insertion points for DG, CA3, CA1.
    # Currently pass-through. Hippocampus build fills them.

    def _get_projection_matrix(self) -> np.ndarray:
        """Lazy-load deterministic random projection matrix for DG sparse coding."""
        if not hasattr(self, '_dg_projection') or self._dg_projection is None:
            rng = np.random.RandomState(42)  # deterministic across restarts
            self._dg_projection = rng.randn(EMBEDDING_DIMENSION, DG_PROJECTION_DIM).astype(np.float32)
            self._dg_projection /= np.linalg.norm(self._dg_projection, axis=0, keepdims=True)
        return self._dg_projection

    def _sparse_encode(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding into high-dim sparse space. Only top DG_SPARSITY% neurons fire."""
        proj = self._get_projection_matrix()
        expanded = embedding @ proj  # (384,) @ (384, 2048) → (2048,)
        # ReLU + top-k sparsification (biological: ~2% granule cell activation)
        expanded = np.maximum(expanded, 0)
        k = max(1, int(DG_PROJECTION_DIM * DG_SPARSITY))
        threshold = np.partition(expanded, -k)[-k]
        expanded[expanded < threshold] = 0
        # Normalize surviving activations
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded /= norm
        return expanded

    def _pattern_separate(self, embedding: np.ndarray, content_hash: str) -> tuple[np.ndarray, bool]:
        """
        Dentate Gyrus: Pattern separation before storage.
        Projects to high-dimensional sparse space. If new memory is too similar
        to an existing one (above DG_SEPARATION_THRESHOLD), orthogonalizes
        the embedding to create distinct memory traces. This prevents catastrophic
        overlap — similar experiences get distinct neural codes.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return embedding, True

        # Check similarity against existing memories
        similarities = np.dot(self._vectors, embedding)
        max_sim = float(np.max(similarities))

        if max_sim >= DG_SEPARATION_THRESHOLD:
            # Too similar — orthogonalize to create distinct trace
            most_similar_idx = int(np.argmax(similarities))
            existing = self._vectors[most_similar_idx]
            # Gram-Schmidt: subtract the projection onto existing memory
            projection = np.dot(embedding, existing) * existing
            orthogonal = embedding - projection
            norm = np.linalg.norm(orthogonal)
            if norm < 1e-8:
                # Perfectly identical direction — suppress storage
                return embedding, False
            orthogonal /= norm
            # Blend: keep 70% original direction + 30% orthogonalized
            # This preserves semantic meaning while creating separation
            separated = 0.7 * embedding + 0.3 * orthogonal
            separated /= np.linalg.norm(separated)
            return separated, True

        return embedding, True

    def _create_associations(self, chunk_id: int, embedding: np.ndarray, conn):
        """
        CA3: Create recurrent associative links to related memories.
        New memory gets connected to its nearest neighbors,
        forming the attractor network for pattern completion.
        Like neurons that fire together wire together.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return

        # Rebuild index if needed (we just added a new memory)
        # Use the current vectors for association (new one not yet indexed)
        similarities = np.dot(self._vectors, embedding)
        now = datetime.now(timezone.utc).isoformat()

        # Find top-N most similar existing memories
        top_indices = np.argsort(similarities)[::-1][:CA3_MAX_ASSOCIATIONS]

        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < CA3_COMPLETION_THRESHOLD:
                break

            target_id = self._vector_ids[idx]
            if target_id == chunk_id:
                continue

            # Upsert bidirectional association with strength = similarity
            for src, tgt in [(chunk_id, target_id), (target_id, chunk_id)]:
                existing = conn.execute(
                    "SELECT id, strength FROM associations WHERE source_chunk_id = ? AND target_chunk_id = ?",
                    (src, tgt)
                ).fetchone()
                if existing:
                    # Strengthen existing association (Hebbian: co-activation reinforces)
                    new_strength = min(1.0, existing["strength"] + sim * 0.1)
                    conn.execute(
                        "UPDATE associations SET strength = ?, updated_at = ? WHERE id = ?",
                        (new_strength, now, existing["id"])
                    )
                else:
                    conn.execute(
                        "INSERT OR IGNORE INTO associations (source_chunk_id, target_chunk_id, strength, association_type, created_at, updated_at) "
                        "VALUES (?, ?, ?, 'co_occurrence', ?, ?)",
                        (src, tgt, sim, now, now)
                    )

    def _check_novelty(self, chunk_id: int, embedding: np.ndarray, content: str, conn) -> dict:
        """
        CA1: Mismatch detection on storage.
        Compare new memory against existing memories.
        Novel info gets flagged. Contradictions get logged.
        Returns {novelty_score, is_contradiction, contradicts_chunk_id}.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return {"novelty_score": 1.0, "is_contradiction": False}

        similarities = np.dot(self._vectors, embedding)
        max_sim = float(np.max(similarities))
        novelty_score = 1.0 - max_sim  # Lower similarity = higher novelty

        result = {"novelty_score": novelty_score, "is_contradiction": False}

        # Check for contradiction using geometric coherence
        if novelty_score > CA1_NOVELTY_THRESHOLD:
            most_similar_idx = int(np.argmax(similarities))
            most_similar_id = self._vector_ids[most_similar_idx]

            # Geometric contradiction: moderate embedding similarity but low geometric coherence
            # means the memories are topically related but geometrically divergent
            is_contradiction = False
            if max_sim > 0.4:
                # Check geometric coherence between new memory and closest existing
                new_signal = embedding_to_signal(embedding)
                new_coord, _ = self.chestahedron.process(new_signal)

                existing_row = conn.execute(
                    "SELECT coordinate FROM chunks WHERE id = ?", (most_similar_id,)
                ).fetchone()
                if existing_row and existing_row["coordinate"]:
                    stored_coord = np.frombuffer(existing_row["coordinate"], dtype=np.float64)
                    if len(stored_coord) == CHESTAHEDRON_DIM:
                        geo_coherence = self.chestahedron.coherence(new_coord, stored_coord)
                        if geo_coherence < CHESTAHEDRON_COHERENCE_CONTRA:
                            is_contradiction = True

            if is_contradiction:
                result["is_contradiction"] = True
                result["contradicts_chunk_id"] = most_similar_id

                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO novelty_log (chunk_id, novelty_score, is_contradiction, contradicts_chunk_id, detected_at) "
                    "VALUES (?, ?, 1, ?, ?)",
                    (chunk_id, novelty_score, most_similar_id, now)
                )

            # Log high novelty even if not contradiction
            if not result["is_contradiction"] and novelty_score > CA1_NOVELTY_THRESHOLD * 2:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO novelty_log (chunk_id, novelty_score, is_contradiction, detected_at) "
                    "VALUES (?, ?, 0, ?)",
                    (chunk_id, novelty_score, now)
                )

        return result

    def _pattern_complete(self, query_vec: np.ndarray, initial_results: list[tuple]) -> list[tuple]:
        """
        CA3: Pattern completion during retrieval.
        Takes partial match results, follows associations to reconstruct
        full context. A partial cue activates an attractor basin, pulling
        in associated memories that weren't in the initial search.
        """
        if not initial_results:
            return initial_results

        conn = self._connect()
        seen_ids = set()
        expanded = list(initial_results)

        # Collect initial chunk IDs
        initial_chunk_ids = []
        for idx, sim in initial_results:
            if idx < len(self._vector_ids):
                chunk_id = self._vector_ids[idx]
                initial_chunk_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        # Follow associations from top results (1 hop)
        for chunk_id in initial_chunk_ids[:5]:  # Only top 5 to avoid explosion
            assocs = conn.execute(
                "SELECT target_chunk_id, strength FROM associations "
                "WHERE source_chunk_id = ? AND strength >= ? "
                "ORDER BY strength DESC LIMIT ?",
                (chunk_id, CA3_COMPLETION_THRESHOLD, CA3_MAX_ASSOCIATIONS)
            ).fetchall()

            for assoc in assocs:
                target_id = assoc["target_chunk_id"]
                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                # Find this chunk's index in our vector store
                if target_id in self._vector_id_to_idx:
                    target_idx = self._vector_id_to_idx[target_id]
                    # Score = association strength * similarity to query
                    assoc_sim = float(np.dot(self._vectors[target_idx], query_vec))
                    # Must still be somewhat relevant to query
                    if assoc_sim >= RELEVANCE_THRESHOLD:
                        # Blend: association strength guides, query relevance gates
                        blended = assoc_sim * 0.7 + float(assoc["strength"]) * 0.3
                        expanded.append((target_idx, blended))

        conn.close()
        return expanded

    def _detect_mismatch(self, query_vec: np.ndarray, retrieved: list[dict]) -> list[dict]:
        """
        CA1: Compare retrieved memories against query during recall.
        Annotate results with novelty flags. If a retrieved memory
        has known contradictions, surface that metadata.
        """
        if not retrieved:
            return retrieved

        conn = self._connect()
        for r in retrieved:
            chunk_id = r.get("chunk_id")
            if not chunk_id:
                continue

            # Check if this memory has contradiction flags
            novelty_entry = conn.execute(
                "SELECT novelty_score, is_contradiction, contradicts_chunk_id "
                "FROM novelty_log WHERE chunk_id = ? AND resolved = 0 "
                "ORDER BY detected_at DESC LIMIT 1",
                (chunk_id,)
            ).fetchone()

            if novelty_entry and novelty_entry["is_contradiction"]:
                r["has_contradiction"] = True
                r["contradicts_chunk_id"] = novelty_entry["contradicts_chunk_id"]
                # Fetch the contradicting content for context
                contra_row = conn.execute(
                    "SELECT content FROM chunks WHERE id = ?",
                    (novelty_entry["contradicts_chunk_id"],)
                ).fetchone()
                if contra_row:
                    r["contradiction_context"] = contra_row["content"][:200]

        conn.close()
        return retrieved

    def _bump_access(self, chunk_ids: list[int], conn):
        """Track access for consolidation. Frequently accessed memories get promoted."""
        now = datetime.now(timezone.utc).isoformat()
        for cid in chunk_ids:
            conn.execute(
                "UPDATE chunks SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, cid)
            )

    # ── Chunking ─────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if end < len(text):
                last_break = max(
                    chunk.rfind('\n\n'),
                    chunk.rfind('. '),
                    chunk.rfind('\n'),
                )
                if last_break > CHUNK_SIZE // 2:
                    end = start + last_break + 1
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - CHUNK_OVERLAP
        return [c for c in chunks if c]

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── Storage ──────────────────────────────────────────

    def store_conversation(
        self,
        messages: list[dict],
        provider: str = "unknown",
        conversation_id: str = None,
        metadata: dict = None,
    ) -> int:
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO conversations (provider, conversation_id, created_at, metadata) VALUES (?, ?, ?, ?)",
            (provider, conversation_id, now, json.dumps(metadata or {})),
        )
        conv_id = cursor.lastrowid
        chunks_stored = 0

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content.strip():
                continue

            text_chunks = self._chunk_text(content)
            for i, chunk in enumerate(text_chunks):
                content_hash = self._hash(chunk)
                existing = conn.execute(
                    "SELECT id FROM chunks WHERE content_hash = ?", (content_hash,)
                ).fetchone()
                if existing:
                    continue

                embedding = embed_text(chunk)

                # ── Hippocampus encode pipeline ──
                # DG: pattern separation
                embedding, should_store = self._pattern_separate(embedding, content_hash)
                if not should_store:
                    continue

                # ── Chestahedron: geometric coordinate ──
                signal = embedding_to_signal(embedding)
                coordinate, energy = self.chestahedron.process(signal)
                chesta_importance, is_deep = self.chesta_hippocampus.evaluate(energy)
                self.chestahedron.learn(coordinate, energy, chesta_importance, is_deep)

                conn.execute(
                    "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type, memory_tier, coordinate, energy, importance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (conv_id, role, chunk, i, embedding.tobytes(), now, provider, content_hash, "conversation", TIER_EPISODIC,
                     coordinate.tobytes(), energy, chesta_importance),
                )
                new_chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                # CA3: create associations
                self._create_associations(new_chunk_id, embedding, conn)

                # CA1: novelty check
                self._check_novelty(new_chunk_id, embedding, chunk, conn)

                # Knowledge Graph: enqueue entity extraction (non-blocking)
                if self._kg is not None:
                    try:
                        self._kg.enqueue_extraction(new_chunk_id, chunk)
                    except Exception:
                        pass

                chunks_stored += 1

        self._save_chestahedron_state(conn)
        conn.commit()
        conn.close()
        self._index_dirty = True
        return chunks_stored

    def store_memory(self, content: str, provider: str = "manual", metadata: dict = None) -> int:
        with self.write_lock:
            return self.store_conversation(
                [{"role": "user", "content": content}],
                provider=provider,
                metadata=metadata,
            )

    # ── Scanning (Unsupervised Ingestion) ────────────────

    def scan_directory(self, path: str, recursive: bool = True) -> dict:
        """
        Watty finds his own food. Point him at a directory,
        he eats everything worth eating. No hand-feeding.
        """
        if not self.write_lock.acquire(timeout=30):
            return {"error": "Could not acquire write lock", "files_scanned": 0, "chunks_stored": 0, "files_skipped": 0, "errors": []}
        try:
            return self._scan_directory_inner(path, recursive)
        finally:
            self.write_lock.release()

    def _scan_directory_inner(self, path: str, recursive: bool) -> dict:
        scan_path = Path(path).expanduser().resolve()
        if not scan_path.exists():
            return {"error": f"Path does not exist: {path}", "files_scanned": 0, "chunks_stored": 0}

        conn = self._connect()
        files_scanned = 0
        chunks_stored = 0
        files_skipped = 0
        errors = []

        if scan_path.is_file():
            files_to_scan = [scan_path]
        else:
            files_to_scan = []
            if recursive:
                for root, dirs, files in os.walk(scan_path):
                    dirs[:] = [d for d in dirs if d not in SCAN_IGNORE_DIRS]
                    for f in files:
                        fp = Path(root) / f
                        if fp.suffix.lower() in SCAN_EXTENSIONS:
                            files_to_scan.append(fp)
            else:
                files_to_scan = [
                    f for f in scan_path.iterdir()
                    if f.is_file() and f.suffix.lower() in SCAN_EXTENSIONS
                ]

        now = datetime.now(timezone.utc).isoformat()

        for filepath in files_to_scan:
            try:
                if filepath.stat().st_size > SCAN_MAX_FILE_SIZE:
                    files_skipped += 1
                    continue

                content = filepath.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    files_skipped += 1
                    continue

                file_hash = self._hash(content)

                # Skip already scanned files
                existing = conn.execute(
                    "SELECT id FROM scan_log WHERE file_hash = ?", (file_hash,)
                ).fetchone()
                if existing:
                    files_skipped += 1
                    continue

                # Create a conversation entry for this file
                cursor = conn.execute(
                    "INSERT INTO conversations (provider, conversation_id, created_at, metadata) VALUES (?, ?, ?, ?)",
                    ("file_scan", str(filepath), now, json.dumps({"source": str(filepath), "type": "file_scan"})),
                )
                conv_id = cursor.lastrowid

                text_chunks = self._chunk_text(content)
                file_chunks = 0

                for i, chunk in enumerate(text_chunks):
                    content_hash = self._hash(chunk)
                    dup = conn.execute("SELECT id FROM chunks WHERE content_hash = ?", (content_hash,)).fetchone()
                    if dup:
                        continue

                    embedding = embed_text(chunk)

                    # ── Hippocampus encode pipeline ──
                    embedding, should_store = self._pattern_separate(embedding, content_hash)
                    if not should_store:
                        continue

                    # ── Chestahedron: geometric coordinate ──
                    signal = embedding_to_signal(embedding)
                    coordinate, energy = self.chestahedron.process(signal)
                    chesta_importance, is_deep = self.chesta_hippocampus.evaluate(energy)
                    self.chestahedron.learn(coordinate, energy, chesta_importance, is_deep)

                    conn.execute(
                        "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type, source_path, memory_tier, coordinate, energy, importance) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (conv_id, "document", chunk, i, embedding.tobytes(), now, "file_scan", content_hash, "file", str(filepath), TIER_EPISODIC,
                         coordinate.tobytes(), energy, chesta_importance),
                    )
                    new_chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    self._create_associations(new_chunk_id, embedding, conn)
                    self._check_novelty(new_chunk_id, embedding, chunk, conn)

                    file_chunks += 1

                conn.execute(
                    "INSERT INTO scan_log (path, file_hash, scanned_at, chunk_count) VALUES (?, ?, ?, ?)",
                    (str(filepath), file_hash, now, file_chunks),
                )

                files_scanned += 1
                chunks_stored += file_chunks

            except Exception as e:
                errors.append(f"{filepath}: {str(e)}")

        self._save_chestahedron_state(conn)
        conn.commit()
        conn.close()
        self._index_dirty = True

        return {
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
            "chunks_stored": chunks_stored,
            "errors": errors[:10],
            "path": str(scan_path),
        }

    # ── Search ───────────────────────────────────────────

    def _build_index(self):
        conn = self._connect()
        rows = conn.execute("SELECT id, embedding, coordinate FROM chunks WHERE embedding IS NOT NULL").fetchall()
        conn.close()

        if not rows:
            with self._index_lock:
                self._vectors = None
                self._coordinates = None
                self._vector_ids = []
                self._vector_id_to_idx = {}
                self._index_dirty = False
            return

        vectors = []
        coords = []
        ids = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(vec) == EMBEDDING_DIMENSION:
                vectors.append(vec)
                ids.append(row["id"])
                # Load coordinate if present
                if row["coordinate"]:
                    coord = np.frombuffer(row["coordinate"], dtype=np.float64)
                    if len(coord) == CHESTAHEDRON_DIM:
                        coords.append(coord)
                    else:
                        coords.append(None)
                else:
                    coords.append(None)

        with self._index_lock:
            self._vectors = np.array(vectors) if vectors else None
            self._coordinates = coords  # list, may contain None for unmigrated memories
            self._vector_ids = ids
            self._vector_id_to_idx = {vid: i for i, vid in enumerate(ids)}
            self._index_dirty = False

    def recall(self, query: str, top_k: int = None, provider_filter: str = None) -> list[dict]:
        if self._index_dirty:
            self._build_index()

        if self._vectors is None or len(self._vectors) == 0:
            return []

        query_vec = embed_text(query)
        similarities = np.dot(self._vectors, query_vec)

        # ── Chestahedron: compute query coordinate for geo boosting ──
        query_signal = embedding_to_signal(query_vec)
        query_coord, _ = self.chestahedron.process(query_signal)

        # ── Collect initial candidates ──
        initial_candidates = []
        for idx in np.argsort(similarities)[::-1][:max(top_k or TOP_K, TOP_K) * 2]:
            sim = float(similarities[idx])
            if sim < RELEVANCE_THRESHOLD:
                continue
            initial_candidates.append((idx, sim))

        # ── CA3: Pattern completion (expand via associations) ──
        initial_candidates = self._pattern_complete(query_vec, initial_candidates)

        conn = self._connect()
        now_ts = time.time()
        results = []
        accessed_ids = []

        for idx, sim in initial_candidates:
            chunk_id = self._vector_ids[idx]

            row = conn.execute(
                "SELECT c.*, conv.provider as conv_provider, conv.metadata as conv_metadata "
                "FROM chunks c JOIN conversations conv ON c.conversation_id = conv.id "
                "WHERE c.id = ?",
                (chunk_id,),
            ).fetchone()

            if not row:
                continue
            if provider_filter and row["provider"] != provider_filter:
                continue

            created_ts = datetime.fromisoformat(row["created_at"]).timestamp()
            age_days = (now_ts - created_ts) / 86400
            recency_boost = RECENCY_WEIGHT * max(0, 1 - age_days / 365)

            # ── Chestahedron: geometric coherence boost ──
            geo_boost = 0.0
            if row["coordinate"]:
                stored_coord = np.frombuffer(row["coordinate"], dtype=np.float64)
                if len(stored_coord) == CHESTAHEDRON_DIM:
                    coherence = self.chestahedron.coherence(query_coord, stored_coord)
                    geo_boost = max(0, coherence) * CHESTAHEDRON_GEO_WEIGHT

            final_score = sim + recency_boost + geo_boost

            # Serve compressed content if available, fall back to original
            content = row["content"]
            is_compressed = False
            if "compressed_content" in row.keys() and row["compressed_content"]:
                content = row["compressed_content"]
                is_compressed = True

            results.append({
                "chunk_id": chunk_id,
                "content": content,
                "score": round(final_score, 4),
                "similarity": round(sim, 4),
                "provider": row["provider"],
                "role": row["role"],
                "created_at": row["created_at"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
                "memory_tier": row["memory_tier"] if "memory_tier" in row.keys() else TIER_EPISODIC,
                "compressed": is_compressed,
            })
            accessed_ids.append(chunk_id)

        # ── CA1: Mismatch detection ──
        results = self._detect_mismatch(query_vec, results)

        # ── Track access for consolidation ──
        self._bump_access(accessed_ids, conn)

        conn.commit()
        conn.close()
        results.sort(key=lambda x: x["score"], reverse=True)
        final = results[: top_k or TOP_K]

        # Evaluation: auto-capture retrieval quality (fire-and-forget)
        if self._eval is not None:
            try:
                self._eval.log_retrieval_quality(query, final, query_vec)
            except Exception:
                pass

        return final

    def keyword_search(self, keyword: str, limit: int = 10) -> list[tuple[int, float]]:
        """
        Fast keyword-based search on chunk content. Returns (chunk_id, score) pairs.
        Used by Navigator for multi-resolution seeding alongside embedding similarity.
        Score is based on keyword match frequency normalized by content length.
        """
        if not keyword or len(keyword) < 2:
            return []

        conn = self._connect()
        # Use LIKE for keyword matching — case-insensitive on SQLite by default for ASCII
        pattern = f"%{keyword}%"
        rows = conn.execute(
            "SELECT id, content, compressed_content FROM chunks "
            "WHERE content LIKE ? OR compressed_content LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, limit * 3),  # Over-fetch then rank
        ).fetchall()
        conn.close()

        if not rows:
            return []

        kw_lower = keyword.lower()
        scored = []
        for row in rows:
            text = (row["compressed_content"] or row["content"] or "").lower()
            if not text:
                continue
            count = text.count(kw_lower)
            if count == 0:
                continue
            # Score: match frequency normalized by length (longer docs get slight penalty)
            score = count / (1.0 + len(text) / 500.0)
            scored.append((row["id"], min(1.0, score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ── Clustering (Unsupervised Organization) ───────────

    def cluster(self) -> list[dict]:
        """
        Watty organizes his own mind. Groups related memories
        without being told how. Returns the knowledge graph.
        """
        if self._index_dirty:
            self._build_index()

        if self._vectors is None or len(self._vectors) < CLUSTER_MIN_MEMORIES:
            return []

        # Simple agglomerative clustering — no sklearn dependency
        n = len(self._vectors)
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue

            cluster_indices = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                sim = cosine_similarity(self._vectors[i], self._vectors[j])
                if sim >= CLUSTER_SIMILARITY_THRESHOLD:
                    cluster_indices.append(j)
                    assigned[j] = True

            if len(cluster_indices) >= 2:
                # Get representative content for labeling
                conn = self._connect()
                chunk_ids = [self._vector_ids[idx] for idx in cluster_indices]
                placeholders = ",".join("?" * len(chunk_ids))
                rows = conn.execute(
                    f"SELECT id, content, source_type FROM chunks WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                conn.close()

                # Use first chunk as label seed (longest content = most representative)
                contents = sorted(rows, key=lambda r: len(r["content"]), reverse=True)
                label_text = contents[0]["content"][:100] if contents else "Unknown cluster"

                centroid = np.mean(self._vectors[cluster_indices], axis=0)

                clusters.append({
                    "label": label_text,
                    "size": len(cluster_indices),
                    "chunk_ids": chunk_ids,
                    "sample_contents": [r["content"][:200] for r in contents[:3]],
                    "sources": list(set(r["source_type"] for r in rows)),
                })

        # Store clusters
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("DELETE FROM clusters")
        for c in clusters:
            centroid_bytes = np.mean(
                self._vectors[[self._vector_id_to_idx[cid] for cid in c["chunk_ids"] if cid in self._vector_id_to_idx]],
                axis=0
            ).tobytes() if c["chunk_ids"] else b""
            conn.execute(
                "INSERT INTO clusters (label, centroid, chunk_ids, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (c["label"], centroid_bytes, json.dumps(c["chunk_ids"]), now, now),
            )
        conn.commit()
        conn.close()

        return clusters

    # ── Forget (User Data Control) ───────────────────────

    def forget(self, query: str = None, chunk_ids: list[int] = None, provider: str = None, before: str = None) -> dict:
        """
        Your soul, your rules. Delete memories by search, ID, provider, or date.
        """
        conn = self._connect()
        deleted = 0

        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)
            deleted = len(chunk_ids)

        elif query:
            results = self.recall(query, top_k=20)
            if results:
                ids_to_delete = [r["chunk_id"] for r in results if r["similarity"] >= 0.5]
                if ids_to_delete:
                    placeholders = ",".join("?" * len(ids_to_delete))
                    conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids_to_delete)
                    # Also clean up associations referencing deleted chunks
                    conn.execute(
                        f"DELETE FROM associations WHERE source_chunk_id IN ({placeholders}) OR target_chunk_id IN ({placeholders})",
                        ids_to_delete + ids_to_delete,
                    )
                    deleted = len(ids_to_delete)

        elif provider:
            cursor = conn.execute("DELETE FROM chunks WHERE provider = ?", (provider,))
            deleted = cursor.rowcount

        elif before:
            cursor = conn.execute("DELETE FROM chunks WHERE created_at < ?", (before,))
            deleted = cursor.rowcount

        conn.commit()
        conn.close()
        self._index_dirty = True

        return {"deleted": deleted, "criteria": {
            "query": query, "chunk_ids": chunk_ids, "provider": provider, "before": before
        }}

    # ── Surface (Proactive Intelligence) ─────────────────

    def surface(self, context: str = None) -> list[dict]:
        """
        Watty tells you something you didn't ask for.
        Finds the most surprising and relevant connections
        in your memory based on current context.
        """
        if self._index_dirty:
            self._build_index()

        if self._vectors is None or len(self._vectors) < CLUSTER_MIN_MEMORIES:
            return []

        conn = self._connect()

        if context:
            # Find memories that are relevant but NOT obvious
            query_vec = embed_text(context)
            similarities = np.dot(self._vectors, query_vec)

            # Sweet spot: similar enough to be relevant (>0.3) but not so similar
            # that it's just echoing what you already know (<0.7)
            candidates = []
            for idx in range(len(similarities)):
                sim = float(similarities[idx])
                if 0.3 < sim < 0.7:
                    novelty_score = sim * (1 - sim) * 4  # peaks at 0.5
                    final = sim * (1 - SURFACE_NOVELTY_WEIGHT) + novelty_score * SURFACE_NOVELTY_WEIGHT
                    candidates.append((idx, final, sim))

            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            # No context — surface most connected memories (cluster centroids)
            clusters = self.cluster()
            candidates = []
            for i, c in enumerate(clusters):
                if c["chunk_ids"]:
                    first_id = c["chunk_ids"][0]
                    if first_id in self._vector_id_to_idx:
                        idx = self._vector_id_to_idx[first_id]
                        candidates.append((idx, c["size"] / 10, 0.5))

        results = []
        for idx, score, sim in candidates[:SURFACE_TOP_K]:
            chunk_id = self._vector_ids[idx]
            row = conn.execute(
                "SELECT content, provider, created_at, source_type, source_path FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                results.append({
                    "content": row["content"][:500],
                    "relevance": round(score, 4),
                    "provider": row["provider"],
                    "created_at": row["created_at"],
                    "source_type": row["source_type"],
                    "source_path": row["source_path"],
                    "reason": "surprising_connection" if context else "knowledge_hub",
                })

        conn.close()
        return results

    # ── Reflect (Deep Synthesis) ─────────────────────────

    def reflect(self) -> dict:
        """
        Watty looks at everything he knows and finds patterns.
        Returns a map of the mind.
        """
        conn = self._connect()

        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        total_scanned = conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0]

        providers = [row[0] for row in conn.execute("SELECT DISTINCT provider FROM chunks").fetchall()]
        source_types = [row[0] for row in conn.execute("SELECT DISTINCT source_type FROM chunks").fetchall()]

        # Time range
        oldest = conn.execute("SELECT MIN(created_at) FROM chunks").fetchone()[0]
        newest = conn.execute("SELECT MAX(created_at) FROM chunks").fetchone()[0]

        # Top topics via clustering
        clusters = self.cluster()

        conn.close()

        return {
            "total_memories": total_chunks,
            "total_conversations": total_convs,
            "total_files_scanned": total_scanned,
            "providers": providers,
            "source_types": source_types,
            "time_range": {"oldest": oldest, "newest": newest},
            "knowledge_clusters": len(clusters),
            "top_clusters": [
                {"label": c["label"][:80], "size": c["size"]}
                for c in sorted(clusters, key=lambda x: x["size"], reverse=True)[:5]
            ],
        }

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> dict:
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        providers = [r[0] for r in conn.execute("SELECT DISTINCT provider FROM chunks").fetchall()]
        scanned = conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0]
        associations = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        conn.close()

        # Database file size
        db_size_mb = 0
        try:
            db_size_mb = round(Path(self.db_path).stat().st_size / (1024 * 1024), 2)
        except OSError:
            pass

        # Chestahedron stats
        conn2 = self._connect()
        with_coords = conn2.execute("SELECT COUNT(*) FROM chunks WHERE coordinate IS NOT NULL").fetchone()[0]
        conn2.close()

        return {
            "total_memories": total,
            "total_conversations": convs,
            "total_files_scanned": scanned,
            "total_associations": associations,
            "providers": providers,
            "db_path": self.db_path,
            "db_size_mb": db_size_mb,
            "chestahedron_processed": self.chestahedron._n_processed,
            "felt_state": self.chestahedron.felt_state.tolist(),
            "memories_with_coordinates": with_coords,
            "plasticity": self.chestahedron.plasticity_report(),
        }

    # ── Consolidation (Dream Cycle) ──────────────────────

    def dream(self) -> dict:
        """
        Sleep consolidation cycle v2. Replays, promotes, decays, strengthens,
        compresses, and deduplicates. Call periodically — it's how Watty dreams.

        Phases:
          1. Promote frequently accessed episodic -> consolidated
          2. Decay old, unaccessed episodic memories
          3. Strengthen high-traffic association pathways
          4. Decay weak associations, prune dead ones
          5. Count unresolved contradictions
          6. Semantic compression (consolidated + old episodic)
          7. Near-duplicate pruning
        """
        if not self.write_lock.acquire(timeout=30):
            return {"error": "Could not acquire write lock (another operation in progress)"}

        try:
            return self._dream_inner()
        finally:
            self.write_lock.release()

    def _dream_inner(self) -> dict:
        from watty.snapshots import create_snapshot
        from watty.compressor import compress

        # Safety snapshot before dream cycle
        try:
            create_snapshot("pre-dream")
        except Exception:
            pass  # Don't block dream if snapshot fails

        conn = self._connect()
        try:
            return self._dream_execute(conn, compress)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _dream_execute(self, conn, compress) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()

        promoted = 0
        decayed = 0
        associations_strengthened = 0
        contradictions_found = 0
        compressed_count = 0
        chars_saved = 0
        duplicates_pruned = 0

        # ── Phase 1: Promote frequently accessed episodic -> consolidated ──
        candidates = conn.execute(
            "SELECT id, access_count, content, embedding FROM chunks "
            "WHERE memory_tier = ? AND access_count >= ?",
            (TIER_EPISODIC, CONSOLIDATION_PROMOTION_THRESHOLD)
        ).fetchall()

        for row in candidates:
            conn.execute(
                "UPDATE chunks SET memory_tier = ?, significance = significance + 0.2 WHERE id = ?",
                (TIER_CONSOLIDATED, row["id"])
            )
            conn.execute(
                "INSERT INTO replay_log (chunk_id, replayed_at, significance_before, significance_after) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], now, 0, 0.2)
            )
            promoted += 1

        # ── Phase 2: Decay old, unaccessed episodic memories ──
        decay_cutoff = datetime.fromtimestamp(
            now_ts - (CONSOLIDATION_DECAY_DAYS * 86400), tz=timezone.utc
        ).isoformat()

        stale = conn.execute(
            "SELECT id, significance FROM chunks "
            "WHERE memory_tier = ? AND access_count = 0 AND created_at < ?",
            (TIER_EPISODIC, decay_cutoff)
        ).fetchall()

        for row in stale:
            new_sig = max(0, (row["significance"] or 0) - 0.1)
            conn.execute(
                "UPDATE chunks SET significance = ? WHERE id = ?",
                (new_sig, row["id"])
            )
            decayed += 1

        # ── Phase 3: Strengthen high-traffic association pathways ──
        hot_assocs = conn.execute(
            "SELECT a.id, a.source_chunk_id, a.target_chunk_id, a.strength "
            "FROM associations a "
            "JOIN chunks c1 ON a.source_chunk_id = c1.id "
            "JOIN chunks c2 ON a.target_chunk_id = c2.id "
            "WHERE c1.access_count > 2 AND c2.access_count > 2 "
            "AND a.strength < 1.0"
        ).fetchall()

        for assoc in hot_assocs:
            new_strength = min(1.0, assoc["strength"] + 0.05)
            conn.execute(
                "UPDATE associations SET strength = ?, updated_at = ? WHERE id = ?",
                (new_strength, now, assoc["id"])
            )
            associations_strengthened += 1

        # ── Phase 4: Decay weak associations ──
        conn.execute(
            "UPDATE associations SET strength = strength * ? WHERE strength < 0.5",
            (CA3_ASSOCIATION_DECAY,)
        )
        pruned_assocs = conn.execute("DELETE FROM associations WHERE strength < 0.05").rowcount

        # ── Phase 5: Count unresolved contradictions ──
        contradictions_found = conn.execute(
            "SELECT COUNT(*) FROM novelty_log WHERE is_contradiction = 1 AND resolved = 0"
        ).fetchone()[0]

        # ── Phase 6: Semantic compression ──
        # Compress consolidated memories and old episodic memories (>7 days)
        # that haven't been compressed yet.
        compress_cutoff = datetime.fromtimestamp(
            now_ts - (7 * 86400), tz=timezone.utc
        ).isoformat()

        # Target: consolidated (always) + old episodic (>7 days)
        to_compress = conn.execute(
            "SELECT id, content, memory_tier, created_at FROM chunks "
            "WHERE compressed_content IS NULL "
            "AND (memory_tier = ? OR (memory_tier = ? AND created_at < ?)) "
            "LIMIT 200",
            (TIER_CONSOLIDATED, TIER_EPISODIC, compress_cutoff)
        ).fetchall()

        for row in to_compress:
            content = row["content"]
            if not content or len(content) < 30:
                continue

            # Aggressive compression for old episodic, normal for consolidated
            aggressive = (row["memory_tier"] == TIER_EPISODIC)
            compressed_text, ratio = compress(content, aggressive=aggressive)

            # Only store if we actually saved space (ratio < 0.95)
            if ratio < 0.95:
                saved = len(content) - len(compressed_text)
                conn.execute(
                    "UPDATE chunks SET compressed_content = ?, compression_ratio = ? WHERE id = ?",
                    (compressed_text, ratio, row["id"])
                )
                compressed_count += 1
                chars_saved += saved
            else:
                # Mark as "checked but not worth compressing" so we don't retry
                conn.execute(
                    "UPDATE chunks SET compression_ratio = 1.0 WHERE id = ?",
                    (row["id"],)
                )

        # ── Phase 7: Near-duplicate pruning ──
        # Find memories with very high similarity (>0.95) and merge them.
        # Keep the one with higher access_count, delete the other.
        # Only check within same provider to avoid cross-source deletion.
        if self._index_dirty:
            conn.commit()  # Commit compression before rebuilding index
            self._build_index()

        if self._vectors is not None and len(self._vectors) > 50:
            # Sample a random subset to check (avoid O(n^2) on full corpus)
            check_count = min(200, len(self._vectors))
            rng = np.random.RandomState(int(now_ts) % (2**31))
            indices = rng.choice(len(self._vectors), check_count, replace=False)

            to_delete = set()
            for i in indices:
                if self._vector_ids[i] in to_delete:
                    continue
                # Check against nearby vectors
                sims = np.dot(self._vectors, self._vectors[i])
                near_dupes = np.where(sims > 0.96)[0]

                for j in near_dupes:
                    if i == j or self._vector_ids[j] in to_delete:
                        continue

                    id_i = self._vector_ids[i]
                    id_j = self._vector_ids[j]

                    # Get access counts to decide which to keep
                    row_i = conn.execute("SELECT access_count, provider, content FROM chunks WHERE id = ?", (id_i,)).fetchone()
                    row_j = conn.execute("SELECT access_count, provider, content FROM chunks WHERE id = ?", (id_j,)).fetchone()

                    if not row_i or not row_j:
                        continue

                    # Only merge within same provider
                    if row_i["provider"] != row_j["provider"]:
                        continue

                    # Keep the one with more access, delete the other
                    if (row_i["access_count"] or 0) >= (row_j["access_count"] or 0):
                        to_delete.add(id_j)
                    else:
                        to_delete.add(id_i)

            # Delete duplicates
            if to_delete:
                delete_ids = list(to_delete)
                placeholders = ",".join("?" * len(delete_ids))
                conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", delete_ids)
                conn.execute(
                    f"DELETE FROM associations WHERE source_chunk_id IN ({placeholders}) OR target_chunk_id IN ({placeholders})",
                    delete_ids + delete_ids,
                )
                duplicates_pruned = len(delete_ids)
                self._index_dirty = True

        # ── Phase 8: Chestahedron coordinate migration ──
        migrated = 0
        unmigrated = conn.execute(
            "SELECT id, embedding FROM chunks "
            "WHERE coordinate IS NULL AND embedding IS NOT NULL "
            "ORDER BY id ASC LIMIT ?",
            (CHESTAHEDRON_MIGRATION_BATCH,)
        ).fetchall()

        for row in unmigrated:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(emb) != EMBEDDING_DIMENSION:
                continue
            signal = embedding_to_signal(emb)
            coordinate, energy = self.chestahedron.process(signal)
            chesta_importance, is_deep = self.chesta_hippocampus.evaluate(energy)
            self.chestahedron.learn(coordinate, energy, chesta_importance, is_deep)
            conn.execute(
                "UPDATE chunks SET coordinate = ?, energy = ?, importance = ? WHERE id = ?",
                (coordinate.tobytes(), energy, chesta_importance, row["id"])
            )
            migrated += 1

        if migrated > 0:
            self._save_chestahedron_state(conn)
            self._index_dirty = True

        # ── Phase 9: Knowledge Graph maintenance ──
        kg_result = {}
        if self._kg is not None:
            try:
                kg_result = self._kg.dream_maintenance()
            except Exception:
                pass

        # ── Stats ──
        total_episodic = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE memory_tier = ?", (TIER_EPISODIC,)
        ).fetchone()[0]
        total_consolidated = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE memory_tier = ?", (TIER_CONSOLIDATED,)
        ).fetchone()[0]
        total_compressed = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE compressed_content IS NOT NULL"
        ).fetchone()[0]
        total_associations = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]

        conn.commit()

        dream_result = {
            "promoted": promoted,
            "decayed": decayed,
            "associations_strengthened": associations_strengthened,
            "associations_pruned": pruned_assocs if isinstance(pruned_assocs, int) else 0,
            "unresolved_contradictions": contradictions_found,
            "compressed": compressed_count,
            "chars_saved": chars_saved,
            "duplicates_pruned": duplicates_pruned,
            "coordinates_migrated": migrated,
            "memory_tiers": {
                "episodic": total_episodic,
                "consolidated": total_consolidated,
            },
            "total_compressed": total_compressed,
            "total_associations": total_associations,
        }

        # Add KG maintenance stats
        if kg_result:
            dream_result["kg_entities_merged"] = kg_result.get("entities_merged", 0)
            dream_result["kg_entities_pruned"] = kg_result.get("entities_pruned", 0)
            dream_result["kg_relationships_pruned"] = kg_result.get("relationships_pruned", 0)
            dream_result["kg_relationships_strengthened"] = kg_result.get("relationships_strengthened", 0)

        # Evaluation: log dream health (fire-and-forget)
        if self._eval is not None:
            try:
                self._eval.log_dream_health(dream_result)
            except Exception:
                pass

        return dream_result

    # ── Chestahedron State Persistence ────────────────────

    def _save_chestahedron_state(self, conn=None):
        """Persist chestahedron + hippocampus state to DB. Non-fatal — geometric state
        is supplementary and can be saved on the next successful write.
        Pass an existing connection to avoid opening a second one during a transaction."""
        try:
            state = {
                'chestahedron': self.chestahedron.save_state(),
                'hippocampus': self.chesta_hippocampus.save_state(),
            }
            now = datetime.now(timezone.utc).isoformat()
            own_conn = conn is None
            if own_conn:
                conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO chestahedron_state (id, state_json, updated_at) VALUES (1, ?, ?)",
                    (json.dumps(state), now)
                )
                if own_conn:
                    conn.commit()
            finally:
                if own_conn:
                    conn.close()
        except Exception:
            pass  # DB locked or other issue — geometric state will persist next time

    def _load_chestahedron_state(self):
        """Restore chestahedron + hippocampus state from DB. Fails silently if no prior state."""
        try:
            conn = self._connect()
            row = conn.execute("SELECT state_json FROM chestahedron_state WHERE id = 1").fetchone()
            conn.close()
            if row:
                state = json.loads(row["state_json"])
                self.chestahedron.load_state(state['chestahedron'])
                self.chesta_hippocampus.load_state(state['hippocampus'])
        except Exception:
            pass  # No prior state or table doesn't exist yet — start fresh

    def snapshot(self, reason: str = "manual") -> dict:
        """Create a backup snapshot of brain.db."""
        from watty.snapshots import create_snapshot
        return create_snapshot(reason)

    def rollback(self, snapshot_filename: str = None) -> dict:
        """Restore brain.db from a snapshot. Reloads index after."""
        from watty.snapshots import rollback
        result = rollback(snapshot_filename)
        if result.get("restored"):
            self._index_dirty = True
        return result

    def list_snapshots(self) -> list[dict]:
        """List available snapshots."""
        from watty.snapshots import list_snapshots
        return list_snapshots()

    def get_contradictions(self) -> list[dict]:
        """Surface unresolved contradictions for the AI to arbitrate."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT n.chunk_id, n.contradicts_chunk_id, n.novelty_score, n.detected_at, "
            "c1.content as new_content, c2.content as old_content "
            "FROM novelty_log n "
            "JOIN chunks c1 ON n.chunk_id = c1.id "
            "JOIN chunks c2 ON n.contradicts_chunk_id = c2.id "
            "WHERE n.is_contradiction = 1 AND n.resolved = 0 "
            "ORDER BY n.detected_at DESC LIMIT 10"
        ).fetchall()
        conn.close()

        return [{
            "new_chunk_id": r["chunk_id"],
            "old_chunk_id": r["contradicts_chunk_id"],
            "new_content": r["new_content"][:300],
            "old_content": r["old_content"][:300],
            "novelty_score": r["novelty_score"],
            "detected_at": r["detected_at"],
        } for r in rows]

    def resolve_contradiction(self, chunk_id: int, keep: str = "new") -> dict:
        """Resolve a contradiction. keep='new' keeps the new memory, 'old' keeps the old."""
        conn = self._connect()
        entry = conn.execute(
            "SELECT * FROM novelty_log WHERE chunk_id = ? AND is_contradiction = 1 AND resolved = 0",
            (chunk_id,)
        ).fetchone()

        if not entry:
            conn.close()
            return {"resolved": False, "reason": "No unresolved contradiction found for this chunk"}

        # Mark resolved
        conn.execute("UPDATE novelty_log SET resolved = 1 WHERE id = ?", (entry["id"],))

        # Optionally delete the losing memory
        if keep == "new":
            conn.execute("DELETE FROM chunks WHERE id = ?", (entry["contradicts_chunk_id"],))
        elif keep == "old":
            conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

        conn.commit()
        conn.close()
        self._index_dirty = True
        return {"resolved": True, "kept": keep}
