"""
Watty Reflection Engine v1.0
==============================
Verbal reinforcement learning. Stores self-critiques from failures,
retrieves them before similar future tasks, and auto-promotes
recurring lessons to behavioral directives.

Not just "what went wrong" — "what will I do differently next time."

The reflection loop:
  1. Task fails or produces suboptimal outcome
  2. Store reflection: what happened, why, what I learned
  3. Before similar future tasks, retrieve relevant reflections
  4. If the same lesson appears >= 3 times, auto-promote to directive
  5. Directive fires pre-action via Never-Twice loop

This is how Watty learns from mistakes without a gradient step.

Hugo & Watty · February 2026
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from watty.config import (
    DB_PATH,
    EMBEDDING_DIMENSION,
    REFLECTION_AUTO_PROMOTE_THRESHOLD,
    REFLECTION_MAX_LENGTH,
    REFLECTION_TOP_K,
    REFLECTION_SIMILARITY_THRESHOLD,
)
from watty.embeddings import embed_text


# ── Valid Categories & Severities ────────────────────────

VALID_CATEGORIES = {
    "code", "architecture", "communication", "planning",
    "debugging", "research", "integration", "testing",
    "data", "security", "performance", "other",
}

VALID_SEVERITIES = {"critical", "major", "minor"}


# ── Helpers ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def _word_overlap(text_a: str, text_b: str) -> float:
    """Word-level Jaccard-style overlap between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


class ReflectionEngine:
    """
    Verbal reinforcement learning for Watty.

    Stores self-critiques, retrieves them semantically before similar tasks,
    and auto-promotes recurring lessons to behavioral directives in the
    cognition profile.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._write_lock = threading.Lock()

    # ── DB Connection ────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and busy timeout (same as brain.py)."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # ── Core Methods ─────────────────────────────────────

    def add_reflection(
        self,
        task_description: str,
        outcome: str,
        reflection_text: str,
        lessons: list[str],
        category: str = "other",
        severity: str = "minor",
        related_chunk_ids: Optional[list[int]] = None,
        session_number: Optional[int] = None,
    ) -> int:
        """
        Store a self-critique from a task outcome.

        Args:
            task_description: What was attempted.
            outcome: What happened (success/failure/partial).
            reflection_text: The self-critique — what went wrong, why, what to change.
            lessons: List of concrete lessons learned (short strings).
            category: Domain category for filtering.
            severity: How impactful the mistake was.
            related_chunk_ids: Optional memory chunk IDs related to this reflection.
            session_number: Current session number, if known.

        Returns:
            The reflection ID.
        """
        # Validate inputs
        if category not in VALID_CATEGORIES:
            category = "other"
        if severity not in VALID_SEVERITIES:
            severity = "minor"

        # Truncate reflection text if needed
        if len(reflection_text) > REFLECTION_MAX_LENGTH * 10:
            reflection_text = reflection_text[:REFLECTION_MAX_LENGTH * 10]

        # Embed for semantic retrieval
        embed_input = f"{task_description} {reflection_text}"
        embedding = embed_text(embed_input)

        # Serialize lists
        lessons_json = json.dumps(lessons, ensure_ascii=False)
        related_json = json.dumps(related_chunk_ids or [], ensure_ascii=False)
        now = _now()

        with self._write_lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO reflections
                        (task_description, outcome, reflection_text, lessons,
                         embedding, created_at, category, severity,
                         related_chunk_ids, session_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_description,
                        outcome,
                        reflection_text,
                        lessons_json,
                        embedding.tobytes(),
                        now,
                        category,
                        severity,
                        related_json,
                        session_number,
                    ),
                )
                conn.commit()
                reflection_id = cursor.lastrowid
            finally:
                conn.close()

        # Check if any lessons should auto-promote to directives
        self._check_auto_promote(lessons)

        return reflection_id

    def get_reflections(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve reflections semantically similar to a query.

        Embeds the query, computes cosine similarity against all stored
        reflection embeddings, filters by threshold, and returns top-k.
        Updates retrieval counters on returned results.

        Args:
            query: What to search for — typically the current task description.
            top_k: Max results (default: REFLECTION_TOP_K).
            category: Filter to this category only.
            severity: Filter to this severity only.

        Returns:
            List of reflection dicts with similarity scores, without embedding blobs.
        """
        if top_k is None:
            top_k = REFLECTION_TOP_K

        query_embedding = embed_text(query)

        conn = self._connect()
        try:
            # Build query with optional filters
            where_clauses = []
            params = []
            if category and category in VALID_CATEGORIES:
                where_clauses.append("category = ?")
                params.append(category)
            if severity and severity in VALID_SEVERITIES:
                where_clauses.append("severity = ?")
                params.append(severity)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            rows = conn.execute(
                f"SELECT * FROM reflections {where_sql}",
                params,
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Score by cosine similarity
        scored = []
        for row in rows:
            if row["embedding"] is None:
                continue
            row_emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if row_emb.shape[0] != EMBEDDING_DIMENSION:
                continue
            sim = _cosine_similarity(query_embedding, row_emb)
            if sim >= REFLECTION_SIMILARITY_THRESHOLD:
                scored.append((sim, row))

        # Sort by similarity descending
        scored.sort(key=lambda x: -x[0])
        scored = scored[:top_k]

        if not scored:
            return []

        # Update retrieval counters
        now = _now()
        returned_ids = [row["id"] for _, row in scored]
        with self._write_lock:
            conn = self._connect()
            try:
                for rid in returned_ids:
                    conn.execute(
                        "UPDATE reflections SET times_retrieved = times_retrieved + 1, "
                        "last_retrieved = ? WHERE id = ?",
                        (now, rid),
                    )
                conn.commit()
            finally:
                conn.close()

        # Format results (exclude embedding blob)
        results = []
        for sim, row in scored:
            results.append({
                "id": row["id"],
                "task_description": row["task_description"],
                "outcome": row["outcome"],
                "reflection_text": row["reflection_text"],
                "lessons": json.loads(row["lessons"]),
                "created_at": row["created_at"],
                "category": row["category"],
                "severity": row["severity"],
                "times_retrieved": row["times_retrieved"] + 1,  # Include this retrieval
                "last_retrieved": now,
                "related_chunk_ids": json.loads(row["related_chunk_ids"] or "[]"),
                "promoted_to_directive": bool(row["promoted_to_directive"]),
                "session_number": row["session_number"],
                "similarity": round(sim, 4),
            })

        return results

    def search_reflections(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 20,
        sort_by: str = "recent",
    ) -> list[dict]:
        """
        Browse reflections without semantic query.

        Args:
            category: Filter to this category.
            severity: Filter to this severity.
            limit: Max results.
            sort_by: 'recent' (created_at DESC), 'most_retrieved' (times_retrieved DESC),
                     'severity' (critical first, then major, then minor).

        Returns:
            List of reflection dicts without embedding blobs.
        """
        where_clauses = []
        params = []
        if category and category in VALID_CATEGORIES:
            where_clauses.append("category = ?")
            params.append(category)
        if severity and severity in VALID_SEVERITIES:
            where_clauses.append("severity = ?")
            params.append(severity)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Determine sort order
        order_map = {
            "recent": "created_at DESC",
            "most_retrieved": "times_retrieved DESC",
            "severity": "CASE severity WHEN 'critical' THEN 0 WHEN 'major' THEN 1 ELSE 2 END ASC, created_at DESC",
        }
        order_sql = order_map.get(sort_by, "created_at DESC")

        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM reflections {where_sql} ORDER BY {order_sql} LIMIT ?",
                params,
            ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "task_description": row["task_description"],
                "outcome": row["outcome"],
                "reflection_text": row["reflection_text"],
                "lessons": json.loads(row["lessons"]),
                "created_at": row["created_at"],
                "category": row["category"],
                "severity": row["severity"],
                "times_retrieved": row["times_retrieved"],
                "last_retrieved": row["last_retrieved"],
                "related_chunk_ids": json.loads(row["related_chunk_ids"] or "[]"),
                "promoted_to_directive": bool(row["promoted_to_directive"]),
                "session_number": row["session_number"],
            })

        return results

    def reflection_stats(self) -> dict:
        """
        Aggregate statistics about the reflection store.

        Returns:
            Dict with total, by_category, by_severity, most_retrieved (top 5),
            and promoted_to_directives count.
        """
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]

            # By category
            by_category = {}
            for row in conn.execute(
                "SELECT category, COUNT(*) as cnt FROM reflections GROUP BY category ORDER BY cnt DESC"
            ).fetchall():
                by_category[row["category"]] = row["cnt"]

            # By severity
            by_severity = {}
            for row in conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM reflections GROUP BY severity ORDER BY cnt DESC"
            ).fetchall():
                by_severity[row["severity"]] = row["cnt"]

            # Most retrieved (top 5)
            most_retrieved = []
            for row in conn.execute(
                "SELECT id, task_description, times_retrieved FROM reflections "
                "WHERE times_retrieved > 0 ORDER BY times_retrieved DESC LIMIT 5"
            ).fetchall():
                most_retrieved.append({
                    "id": row["id"],
                    "task_description": row["task_description"],
                    "times_retrieved": row["times_retrieved"],
                })

            # Promoted count
            promoted = conn.execute(
                "SELECT COUNT(*) FROM reflections WHERE promoted_to_directive = 1"
            ).fetchone()[0]
        finally:
            conn.close()

        return {
            "total": total,
            "by_category": by_category,
            "by_severity": by_severity,
            "most_retrieved": most_retrieved,
            "promoted_to_directives": promoted,
        }

    def promote_to_directive(
        self,
        reflection_id: int,
        rule: Optional[str] = None,
    ) -> dict:
        """
        Promote a reflection's lesson to a behavioral directive in the cognition profile.

        Loads the profile, adds the directive, saves, and marks the reflection as promoted.

        Args:
            reflection_id: ID of the reflection to promote.
            rule: Custom rule text. If None, uses the first lesson from the reflection.

        Returns:
            Dict with status and the directive rule text.
        """
        # Load the reflection
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM reflections WHERE id = ?", (reflection_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return {"error": f"Reflection {reflection_id} not found."}

        lessons = json.loads(row["lessons"])
        if not rule:
            if not lessons:
                return {"error": "Reflection has no lessons and no custom rule provided."}
            rule = lessons[0]

        source = f"Reflection #{reflection_id}: {row['task_description'][:80]}"

        # Load cognition profile, add directive, save
        from watty.cognition import load_profile, add_directive, save_profile

        profile = load_profile()
        profile = add_directive(profile, rule=rule, source=source, confidence=0.6)
        save_profile(profile)

        # Mark as promoted
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE reflections SET promoted_to_directive = 1 WHERE id = ?",
                    (reflection_id,),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "status": "promoted",
            "reflection_id": reflection_id,
            "directive_rule": rule,
            "source": source,
        }

    def get_session_reflections(
        self,
        context: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Convenience wrapper for session enter — retrieve reflections relevant
        to the current session context.

        Args:
            context: Description of what this session is about.
            top_k: Max reflections to return.

        Returns:
            List of relevant reflection dicts with similarity scores.
        """
        return self.get_reflections(query=context, top_k=top_k)

    # ── Auto-Promotion ───────────────────────────────────

    def _check_auto_promote(self, lessons: list[str]):
        """
        For each lesson, check if similar lessons appear across multiple reflections.
        If count >= REFLECTION_AUTO_PROMOTE_THRESHOLD, auto-create a directive
        and mark matching reflections as promoted.

        Similarity is measured by word overlap > 0.6 between lesson strings.
        """
        if not lessons:
            return

        conn = self._connect()
        try:
            all_rows = conn.execute(
                "SELECT id, lessons FROM reflections WHERE promoted_to_directive = 0"
            ).fetchall()
        finally:
            conn.close()

        if not all_rows:
            return

        for lesson in lessons:
            matching_ids = []

            for row in all_rows:
                try:
                    row_lessons = json.loads(row["lessons"])
                except (json.JSONDecodeError, TypeError):
                    continue

                for rl in row_lessons:
                    if _word_overlap(lesson, rl) > 0.6:
                        matching_ids.append(row["id"])
                        break

            if len(matching_ids) >= REFLECTION_AUTO_PROMOTE_THRESHOLD:
                # Auto-promote this recurring lesson
                source = f"Auto-promoted: appeared in {len(matching_ids)} reflections"

                try:
                    from watty.cognition import load_profile, add_directive, save_profile

                    profile = load_profile()
                    profile = add_directive(
                        profile, rule=lesson, source=source, confidence=0.5
                    )
                    save_profile(profile)
                except Exception:
                    # Don't let cognition failures break reflection storage
                    continue

                # Mark matching reflections as promoted
                with self._write_lock:
                    conn = self._connect()
                    try:
                        for rid in matching_ids:
                            conn.execute(
                                "UPDATE reflections SET promoted_to_directive = 1 WHERE id = ?",
                                (rid,),
                            )
                        conn.commit()
                    finally:
                        conn.close()
