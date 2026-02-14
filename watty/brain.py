"""
Watty Brain v1.0
================
The core memory engine. Stores everything. Retrieves what matters.
Scans unsupervised. Clusters knowledge. Surfaces insights. Forgets on command.

The user never touches it. It just works.
"""

import sqlite3
import json
import os
import hashlib
import time
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
)
from watty.embeddings import embed_text, cosine_similarity


class Brain:
    """
    The Watty Brain. Stores everything. Retrieves what matters.
    Organizes itself. Cleans up on command. Surfaces what you need
    before you know you need it.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        ensure_home()
        self._init_db()
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: list[int] = []
        self._index_dirty = True

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
            CREATE INDEX IF NOT EXISTS idx_scan_log_hash ON scan_log(file_hash);
        """)
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
                conn.execute(
                    "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (conv_id, role, chunk, i, embedding.tobytes(), now, provider, content_hash, "conversation"),
                )
                chunks_stored += 1

        conn.commit()
        conn.close()
        self._index_dirty = True
        return chunks_stored

    def store_memory(self, content: str, provider: str = "manual", metadata: dict = None) -> int:
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
                    conn.execute(
                        "INSERT INTO chunks (conversation_id, role, content, chunk_index, embedding, created_at, provider, content_hash, source_type, source_path) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (conv_id, "document", chunk, i, embedding.tobytes(), now, "file_scan", content_hash, "file", str(filepath)),
                    )
                    file_chunks += 1

                conn.execute(
                    "INSERT INTO scan_log (path, file_hash, scanned_at, chunk_count) VALUES (?, ?, ?, ?)",
                    (str(filepath), file_hash, now, file_chunks),
                )

                files_scanned += 1
                chunks_stored += file_chunks

            except Exception as e:
                errors.append(f"{filepath}: {str(e)}")

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
        rows = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        conn.close()

        if not rows:
            self._vectors = None
            self._vector_ids = []
            self._index_dirty = False
            return

        vectors = []
        ids = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(vec) == EMBEDDING_DIMENSION:
                vectors.append(vec)
                ids.append(row["id"])

        self._vectors = np.array(vectors) if vectors else None
        self._vector_ids = ids
        self._index_dirty = False

    def recall(self, query: str, top_k: int = None, provider_filter: str = None) -> list[dict]:
        if self._index_dirty:
            self._build_index()

        if self._vectors is None or len(self._vectors) == 0:
            return []

        query_vec = embed_text(query)
        similarities = np.dot(self._vectors, query_vec)

        conn = self._connect()
        now_ts = time.time()
        results = []

        for idx in np.argsort(similarities)[::-1][:max(top_k or TOP_K, TOP_K) * 2]:
            chunk_id = self._vector_ids[idx]
            sim = float(similarities[idx])

            if sim < RELEVANCE_THRESHOLD:
                continue

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
            final_score = sim + recency_boost

            results.append({
                "content": row["content"],
                "score": round(final_score, 4),
                "similarity": round(sim, 4),
                "provider": row["provider"],
                "role": row["role"],
                "created_at": row["created_at"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
            })

        conn.close()
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: top_k or TOP_K]

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
                self._vectors[[self._vector_ids.index(cid) for cid in c["chunk_ids"] if cid in self._vector_ids]],
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
                # Find matching chunk IDs
                for r in results:
                    if r["similarity"] >= 0.5:
                        content_hash = self._hash(r["content"])
                        conn.execute("DELETE FROM chunks WHERE content_hash = ?", (content_hash,))
                        deleted += 1

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
                    if first_id in self._vector_ids:
                        idx = self._vector_ids.index(first_id)
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
        conn.close()

        return {
            "total_memories": total,
            "total_conversations": convs,
            "total_files_scanned": scanned,
            "providers": providers,
            "db_path": self.db_path,
        }
