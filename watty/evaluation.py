"""
Watty Evaluation Framework v1.0
================================
Measures whether Watty is getting smarter over time.

Auto-captures metrics during recall/dream/sessions, computes trends,
generates alerts on degradation. Every metric capture method is wrapped
to never raise -- evaluation should never break the brain.

Hugo & Watty -- February 2026
"""

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from mcp.types import Tool, TextContent

from watty.config import (
    DB_PATH,
    EMBEDDING_DIMENSION,
    EVAL_ENABLED,
    EVAL_ALERT_WINDOW_DAYS,
    EVAL_TREND_SHORT_DAYS,
    EVAL_TREND_LONG_DAYS,
    EVAL_ALERT_PRECISION_DROP,
    EVAL_ALERT_CONTRADICTION_RISE,
    EVAL_ALERT_SUCCESS_DROP,
)

# Valid metric categories
VALID_CATEGORIES = frozenset({
    "retrieval_quality",
    "task_success",
    "memory_health",
    "context_efficiency",
    "reflection_quality",
    "custom",
})


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalEngine:
    """
    The evaluation engine. Tracks metrics, computes trends, fires alerts.
    Designed to be fast and silent -- never crashes the brain.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        """Open a WAL-mode connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # ── Metric Capture ───────────────────────────────────────

    def log_metric(
        self,
        metric_name: str,
        value: float,
        category: str,
        context: Optional[str] = None,
        session_number: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Log a single evaluation metric.

        Args:
            metric_name: Name of the metric (e.g. 'precision_proxy', 'task_success_rate').
            value: Numeric value of the metric.
            category: One of VALID_CATEGORIES.
            context: Optional free-text context string.
            session_number: Optional session number for correlation.
            metadata: Optional dict of extra data, stored as JSON.

        Returns:
            Row ID of the inserted metric, or -1 on failure.
        """
        try:
            if category not in VALID_CATEGORIES:
                category = "custom"

            metadata_json = json.dumps(metadata) if metadata else None

            with self._lock:
                conn = self._connect()
                try:
                    cur = conn.execute(
                        "INSERT INTO evaluations "
                        "(metric_name, metric_value, context, measured_at, session_number, category, metadata_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (metric_name, float(value), context, _now_utc(), session_number, category, metadata_json),
                    )
                    conn.commit()
                    return cur.lastrowid
                finally:
                    conn.close()
        except Exception:
            return -1

    def log_retrieval_quality(
        self,
        query: str,
        results: list[dict],
        query_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Auto-compute retrieval quality metrics from a recall() result set.

        Computes:
            - precision_proxy: average similarity score of returned results
            - diversity: std dev of pairwise similarities (higher = more spread)
            - result_count: number of results returned
            - avg_score: mean of result scores

        Args:
            query: The original query string.
            results: List of result dicts, each expected to have 'score' and optionally 'embedding'.
            query_embedding: The query's embedding vector (optional, used for precision).

        Returns:
            Dict of computed metrics, or empty dict on failure.
        """
        try:
            if not results:
                metrics = {
                    "precision_proxy": 0.0,
                    "diversity": 0.0,
                    "result_count": 0,
                    "avg_score": 0.0,
                }
                for name, val in metrics.items():
                    self.log_metric(name, val, "retrieval_quality", context=query[:200])
                return metrics

            # Extract scores
            scores = []
            for r in results:
                s = r.get("score", r.get("similarity", 0.0))
                if s is not None:
                    scores.append(float(s))

            avg_score = float(np.mean(scores)) if scores else 0.0
            precision_proxy = avg_score  # average similarity IS the precision proxy

            # Compute diversity: std dev of pairwise similarities between result embeddings
            diversity = 0.0
            embeddings = []
            for r in results:
                emb = r.get("embedding")
                if emb is not None:
                    if isinstance(emb, bytes):
                        emb = np.frombuffer(emb, dtype=np.float32)
                    if isinstance(emb, np.ndarray) and len(emb) == EMBEDDING_DIMENSION:
                        # Normalize for cosine similarity
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            embeddings.append(emb / norm)

            if len(embeddings) >= 2:
                emb_matrix = np.stack(embeddings)
                # Pairwise cosine similarities (embeddings already normalized)
                sim_matrix = emb_matrix @ emb_matrix.T
                # Extract upper triangle (exclude diagonal)
                triu_indices = np.triu_indices(len(embeddings), k=1)
                pairwise_sims = sim_matrix[triu_indices]
                diversity = float(np.std(pairwise_sims)) if len(pairwise_sims) > 0 else 0.0

            metrics = {
                "precision_proxy": precision_proxy,
                "diversity": diversity,
                "result_count": len(results),
                "avg_score": avg_score,
            }

            for name, val in metrics.items():
                self.log_metric(name, val, "retrieval_quality", context=query[:200])

            return metrics

        except Exception:
            return {}

    def log_dream_health(self, dream_result: dict) -> dict:
        """
        Extract and log health metrics from a dream cycle result.

        Args:
            dream_result: The dict returned by brain.py _dream_execute().
                Expected keys: promoted, decayed, associations_strengthened,
                duplicates_pruned, unresolved_contradictions, compressed.

        Returns:
            Dict of logged metrics, or empty dict on failure.
        """
        try:
            if not dream_result or not isinstance(dream_result, dict):
                return {}

            # Map dream result keys to metric names
            metric_map = {
                "promoted": "dream_promoted",
                "decayed": "dream_decayed",
                "associations_strengthened": "dream_associations_strengthened",
                "duplicates_pruned": "dream_duplicates_pruned",
                "unresolved_contradictions": "dream_contradictions_found",
                "compressed": "dream_compressed",
                "associations_pruned": "dream_associations_pruned",
                "coordinates_migrated": "dream_coordinates_migrated",
                "chars_saved": "dream_chars_saved",
            }

            metrics = {}
            for src_key, metric_name in metric_map.items():
                val = dream_result.get(src_key)
                if val is not None:
                    metrics[metric_name] = float(val)
                    self.log_metric(metric_name, float(val), "memory_health")

            # Log tier counts if present
            tiers = dream_result.get("memory_tiers", {})
            if isinstance(tiers, dict):
                for tier_name, count in tiers.items():
                    metric_name = f"tier_{tier_name}"
                    metrics[metric_name] = float(count)
                    self.log_metric(metric_name, float(count), "memory_health")

            # Total associations
            total_assoc = dream_result.get("total_associations")
            if total_assoc is not None:
                metrics["total_associations"] = float(total_assoc)
                self.log_metric("total_associations", float(total_assoc), "memory_health")

            return metrics

        except Exception:
            return {}

    def log_task_outcome(
        self,
        task_description: str,
        outcome: str,
        category: str = "general",
        context: Optional[str] = None,
        session_number: Optional[int] = None,
    ) -> int:
        """
        Log a task outcome as a numeric metric.

        Args:
            task_description: What the task was.
            outcome: "success", "partial", or "failure".
            category: Sub-category for grouping (default "general").
            context: Optional extra context.
            session_number: Optional session number.

        Returns:
            Row ID, or -1 on failure.
        """
        try:
            outcome_map = {"success": 1.0, "partial": 0.5, "failure": 0.0}
            value = outcome_map.get(outcome.lower().strip(), 0.0)

            metadata = {
                "task": task_description[:500],
                "outcome_label": outcome,
                "sub_category": category,
            }

            return self.log_metric(
                "task_outcome",
                value,
                "task_success",
                context=context or task_description[:200],
                session_number=session_number,
                metadata=metadata,
            )

        except Exception:
            return -1

    # ── Trend Computation ────────────────────────────────────

    def get_trends(
        self,
        metric_name: Optional[str] = None,
        category: Optional[str] = None,
        days: int = 30,
    ) -> dict:
        """
        Compute daily average trends for metrics.

        Groups by date, computes daily averages for each metric.

        Args:
            metric_name: Filter to a specific metric. If None, returns all in category.
            category: Filter to a specific category. Used when metric_name is None.
            days: How many days back to look (default 30).

        Returns:
            {metric_name: [{date, avg_value, count}, ...]}
        """
        try:
            conn = self._connect()
            try:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

                conditions = ["measured_at >= ?"]
                params = [cutoff]

                if metric_name:
                    conditions.append("metric_name = ?")
                    params.append(metric_name)
                elif category:
                    conditions.append("category = ?")
                    params.append(category)

                where_clause = " AND ".join(conditions)

                rows = conn.execute(
                    f"SELECT metric_name, DATE(measured_at) as day, "
                    f"AVG(metric_value) as avg_value, COUNT(*) as count "
                    f"FROM evaluations WHERE {where_clause} "
                    f"GROUP BY metric_name, day ORDER BY metric_name, day",
                    params,
                ).fetchall()

                trends = {}
                for row in rows:
                    name = row["metric_name"]
                    if name not in trends:
                        trends[name] = []
                    trends[name].append({
                        "date": row["day"],
                        "avg_value": round(row["avg_value"], 4),
                        "count": row["count"],
                    })

                return trends

            finally:
                conn.close()

        except Exception:
            return {}

    def get_stats(self) -> dict:
        """
        Get overall evaluation statistics.

        Returns:
            {total_metrics, categories: {name: count}, date_range: {earliest, latest}, recent_alerts}
        """
        try:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]

                cats = conn.execute(
                    "SELECT category, COUNT(*) as cnt FROM evaluations GROUP BY category"
                ).fetchall()
                categories = {row["category"]: row["cnt"] for row in cats}

                date_range = conn.execute(
                    "SELECT MIN(measured_at) as earliest, MAX(measured_at) as latest FROM evaluations"
                ).fetchone()

                recent_alerts = conn.execute(
                    "SELECT COUNT(*) FROM eval_alerts WHERE acknowledged = 0"
                ).fetchone()[0]

                return {
                    "total_metrics": total,
                    "categories": categories,
                    "date_range": {
                        "earliest": date_range["earliest"],
                        "latest": date_range["latest"],
                    } if date_range["earliest"] else {"earliest": None, "latest": None},
                    "recent_alerts": recent_alerts,
                }

            finally:
                conn.close()

        except Exception:
            return {
                "total_metrics": 0,
                "categories": {},
                "date_range": {"earliest": None, "latest": None},
                "recent_alerts": 0,
            }

    # ── Alert Engine ─────────────────────────────────────────

    def check_alerts(self) -> list[dict]:
        """
        Compare recent window vs previous window and generate alerts on degradation.

        Checks:
            - retrieval precision_proxy drops > EVAL_ALERT_PRECISION_DROP
            - task success rate drops > EVAL_ALERT_SUCCESS_DROP

        Returns:
            List of newly generated alert dicts.
        """
        try:
            conn = self._connect()
            try:
                now = datetime.now(timezone.utc)
                recent_start = (now - timedelta(days=EVAL_ALERT_WINDOW_DAYS)).isoformat()
                previous_start = (now - timedelta(days=EVAL_ALERT_WINDOW_DAYS * 2)).isoformat()

                new_alerts = []

                # Check precision_proxy degradation
                new_alerts.extend(
                    self._check_metric_drop(
                        conn,
                        metric_name="precision_proxy",
                        alert_type="precision_drop",
                        threshold=EVAL_ALERT_PRECISION_DROP,
                        recent_start=recent_start,
                        previous_start=previous_start,
                        previous_end=recent_start,
                    )
                )

                # Check task success rate degradation
                new_alerts.extend(
                    self._check_metric_drop(
                        conn,
                        metric_name="task_outcome",
                        alert_type="success_drop",
                        threshold=EVAL_ALERT_SUCCESS_DROP,
                        recent_start=recent_start,
                        previous_start=previous_start,
                        previous_end=recent_start,
                    )
                )

                return new_alerts

            finally:
                conn.close()

        except Exception:
            return []

    def _check_metric_drop(
        self,
        conn: sqlite3.Connection,
        metric_name: str,
        alert_type: str,
        threshold: float,
        recent_start: str,
        previous_start: str,
        previous_end: str,
    ) -> list[dict]:
        """Check if a metric has dropped beyond a threshold between two time windows."""
        alerts = []

        recent_avg = conn.execute(
            "SELECT AVG(metric_value) as avg_val, COUNT(*) as cnt "
            "FROM evaluations WHERE metric_name = ? AND measured_at >= ?",
            (metric_name, recent_start),
        ).fetchone()

        previous_avg = conn.execute(
            "SELECT AVG(metric_value) as avg_val, COUNT(*) as cnt "
            "FROM evaluations WHERE metric_name = ? AND measured_at >= ? AND measured_at < ?",
            (metric_name, previous_start, previous_end),
        ).fetchone()

        # Need data in both windows to compare
        if (
            recent_avg["cnt"] < 1
            or previous_avg["cnt"] < 1
            or recent_avg["avg_val"] is None
            or previous_avg["avg_val"] is None
        ):
            return alerts

        recent_val = recent_avg["avg_val"]
        previous_val = previous_avg["avg_val"]

        # Avoid division by zero
        if previous_val == 0:
            return alerts

        drop = (previous_val - recent_val) / abs(previous_val)

        if drop > threshold:
            message = (
                f"{metric_name} dropped {drop:.1%} "
                f"(from {previous_val:.3f} to {recent_val:.3f}) "
                f"over the last {EVAL_ALERT_WINDOW_DAYS} days. "
                f"Threshold: {threshold:.0%}."
            )

            severity = "critical" if drop > threshold * 2 else "warning"

            with self._lock:
                cur = conn.execute(
                    "INSERT INTO eval_alerts (alert_type, metric_name, message, severity, created_at, acknowledged) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (alert_type, metric_name, message, severity, _now_utc()),
                )
                conn.commit()

            alert = {
                "id": cur.lastrowid,
                "alert_type": alert_type,
                "metric_name": metric_name,
                "message": message,
                "severity": severity,
                "created_at": _now_utc(),
                "drop_pct": round(drop, 4),
            }
            alerts.append(alert)

        return alerts

    def get_alerts(self, include_acknowledged: bool = False) -> list[dict]:
        """
        Get evaluation alerts.

        Args:
            include_acknowledged: If True, include already-acknowledged alerts.

        Returns:
            List of alert dicts.
        """
        try:
            conn = self._connect()
            try:
                if include_acknowledged:
                    rows = conn.execute(
                        "SELECT * FROM eval_alerts ORDER BY created_at DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM eval_alerts WHERE acknowledged = 0 ORDER BY created_at DESC"
                    ).fetchall()

                return [dict(row) for row in rows]

            finally:
                conn.close()

        except Exception:
            return []

    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as acknowledged.

        Args:
            alert_id: The ID of the alert to acknowledge.

        Returns:
            True if the alert was found and updated, False otherwise.
        """
        try:
            with self._lock:
                conn = self._connect()
                try:
                    cur = conn.execute(
                        "UPDATE eval_alerts SET acknowledged = 1 WHERE id = ?",
                        (alert_id,),
                    )
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()

        except Exception:
            return False

    # ── Export ────────────────────────────────────────────────

    def export_csv(
        self,
        category: Optional[str] = None,
        days: Optional[int] = None,
    ) -> str:
        """
        Export evaluation metrics as a CSV-formatted string.

        Args:
            category: Filter to a specific category. None for all.
            days: Limit to the last N days. None for all time.

        Returns:
            CSV string with headers, or empty string on failure.
        """
        try:
            conn = self._connect()
            try:
                conditions = []
                params = []

                if category:
                    conditions.append("category = ?")
                    params.append(category)
                if days:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                    conditions.append("measured_at >= ?")
                    params.append(cutoff)

                where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

                rows = conn.execute(
                    f"SELECT id, metric_name, metric_value, category, context, "
                    f"measured_at, session_number, metadata_json "
                    f"FROM evaluations{where_clause} ORDER BY measured_at DESC",
                    params,
                ).fetchall()

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([
                    "id", "metric_name", "metric_value", "category",
                    "context", "measured_at", "session_number", "metadata_json",
                ])

                for row in rows:
                    writer.writerow([
                        row["id"],
                        row["metric_name"],
                        row["metric_value"],
                        row["category"],
                        row["context"],
                        row["measured_at"],
                        row["session_number"],
                        row["metadata_json"],
                    ])

                return output.getvalue()

            finally:
                conn.close()

        except Exception:
            return ""

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Retrieve evaluation metrics with optional filtering and pagination.

        Args:
            metric_name: Filter by metric name. None for all.
            category: Filter by category. None for all.
            limit: Max rows to return (default 100).
            offset: Row offset for pagination (default 0).

        Returns:
            List of metric dicts.
        """
        try:
            conn = self._connect()
            try:
                conditions = []
                params = []

                if metric_name:
                    conditions.append("metric_name = ?")
                    params.append(metric_name)
                if category:
                    conditions.append("category = ?")
                    params.append(category)

                where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

                params.extend([limit, offset])

                rows = conn.execute(
                    f"SELECT * FROM evaluations{where_clause} "
                    f"ORDER BY measured_at DESC LIMIT ? OFFSET ?",
                    params,
                ).fetchall()

                results = []
                for row in rows:
                    d = dict(row)
                    # Parse metadata JSON for convenience
                    if d.get("metadata_json"):
                        try:
                            d["metadata"] = json.loads(d["metadata_json"])
                        except (json.JSONDecodeError, TypeError):
                            d["metadata"] = None
                    else:
                        d["metadata"] = None
                    results.append(d)

                return results

            finally:
                conn.close()

        except Exception:
            return []


# ── MCP Tool Definition ─────────────────────────────────────

_brain_ref = None


def set_brain(brain):
    """Called by server.py to give eval tools access to brain (and brain._eval)."""
    global _brain_ref
    _brain_ref = brain


TOOLS = [
    Tool(
        name="watty_eval",
        description=(
            "Evaluation framework -- measures whether Watty is getting smarter. "
            "Tracks retrieval quality, task success, memory health, and generates "
            "degradation alerts. Actions: log_metric, get_metrics, get_trends, "
            "get_alerts, ack_alert, eval_stats, export."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "log_metric",
                        "get_metrics",
                        "get_trends",
                        "get_alerts",
                        "ack_alert",
                        "eval_stats",
                        "export",
                    ],
                    "description": "Which evaluation action to perform",
                },
                "metric_name": {
                    "type": "string",
                    "description": "Name of the metric (for log_metric, get_metrics, get_trends)",
                },
                "value": {
                    "type": "number",
                    "description": "Numeric value (for log_metric)",
                },
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Metric category",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context string",
                },
                "session_number": {
                    "type": "integer",
                    "description": "Optional session number",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata dict (for log_metric)",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (for get_trends, export)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (for get_metrics, default 100)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (for get_metrics)",
                },
                "alert_id": {
                    "type": "integer",
                    "description": "Alert ID (for ack_alert)",
                },
                "include_acknowledged": {
                    "type": "boolean",
                    "description": "Include acknowledged alerts (for get_alerts)",
                },
            },
            "required": ["action"],
        },
    ),
]


async def handle_eval(arguments: dict) -> list[TextContent]:
    """Route watty_eval actions to the EvalEngine instance on the brain."""
    action = arguments.get("action", "")

    # Get the eval engine from brain
    if _brain_ref is None:
        return [TextContent(type="text", text="Eval engine not initialized -- brain not connected.")]

    engine = getattr(_brain_ref, "_eval", None)
    if engine is None:
        return [TextContent(type="text", text="Eval engine not attached to brain. Set brain._eval = EvalEngine().")]

    try:
        if action == "log_metric":
            metric_name = arguments.get("metric_name")
            value = arguments.get("value")
            category = arguments.get("category", "custom")

            if not metric_name or value is None:
                return [TextContent(type="text", text="log_metric requires metric_name and value.")]

            row_id = engine.log_metric(
                metric_name=metric_name,
                value=value,
                category=category,
                context=arguments.get("context"),
                session_number=arguments.get("session_number"),
                metadata=arguments.get("metadata"),
            )
            return [TextContent(
                type="text",
                text=f"Logged {metric_name}={value} [{category}] (id={row_id})",
            )]

        elif action == "get_metrics":
            metrics = engine.get_metrics(
                metric_name=arguments.get("metric_name"),
                category=arguments.get("category"),
                limit=arguments.get("limit", 100),
                offset=arguments.get("offset", 0),
            )
            if not metrics:
                return [TextContent(type="text", text="No metrics found.")]

            lines = [f"Found {len(metrics)} metric(s):", ""]
            for m in metrics[:50]:  # Cap display at 50
                meta_str = ""
                if m.get("metadata"):
                    meta_str = f" | meta={json.dumps(m['metadata'], default=str)[:100]}"
                lines.append(
                    f"  [{m.get('measured_at', '?')[:16]}] "
                    f"{m.get('metric_name', '?')}={m.get('metric_value', '?'):.4f} "
                    f"({m.get('category', '?')})"
                    f"{meta_str}"
                )

            if len(metrics) > 50:
                lines.append(f"  ... and {len(metrics) - 50} more (use limit/offset to paginate)")

            return [TextContent(type="text", text="\n".join(lines))]

        elif action == "get_trends":
            trends = engine.get_trends(
                metric_name=arguments.get("metric_name"),
                category=arguments.get("category"),
                days=arguments.get("days", EVAL_TREND_SHORT_DAYS),
            )
            if not trends:
                return [TextContent(type="text", text="No trend data found.")]

            lines = [f"Trends ({len(trends)} metric(s)):", ""]
            for name, data_points in trends.items():
                lines.append(f"  {name}:")
                for dp in data_points[-14:]:  # Show last 14 days max
                    bar = "#" * max(1, int(dp["avg_value"] * 20))
                    lines.append(
                        f"    {dp['date']} | avg={dp['avg_value']:.4f} | n={dp['count']} | {bar}"
                    )
                if len(data_points) > 14:
                    lines.append(f"    ... {len(data_points) - 14} earlier days omitted")
                lines.append("")

            return [TextContent(type="text", text="\n".join(lines))]

        elif action == "get_alerts":
            include_ack = arguments.get("include_acknowledged", False)
            alerts = engine.get_alerts(include_acknowledged=include_ack)
            if not alerts:
                return [TextContent(type="text", text="No alerts. Watty is healthy.")]

            lines = [f"{len(alerts)} alert(s):", ""]
            for a in alerts:
                ack_marker = " [ACK]" if a.get("acknowledged") else ""
                lines.append(
                    f"  #{a.get('id', '?')} [{a.get('severity', '?').upper()}]{ack_marker} "
                    f"{a.get('alert_type', '?')}: {a.get('message', '?')}"
                )

            return [TextContent(type="text", text="\n".join(lines))]

        elif action == "ack_alert":
            alert_id = arguments.get("alert_id")
            if alert_id is None:
                return [TextContent(type="text", text="ack_alert requires alert_id.")]

            ok = engine.acknowledge_alert(int(alert_id))
            if ok:
                return [TextContent(type="text", text=f"Alert #{alert_id} acknowledged.")]
            else:
                return [TextContent(type="text", text=f"Alert #{alert_id} not found.")]

        elif action == "eval_stats":
            stats = engine.get_stats()

            lines = [
                "Evaluation Stats:",
                f"  Total metrics: {stats.get('total_metrics', 0)}",
                f"  Active alerts: {stats.get('recent_alerts', 0)}",
            ]

            date_range = stats.get("date_range", {})
            if date_range.get("earliest"):
                lines.append(
                    f"  Date range: {date_range['earliest'][:10]} to {date_range['latest'][:10]}"
                )

            cats = stats.get("categories", {})
            if cats:
                lines.append("  Categories:")
                for cat_name, count in sorted(cats.items(), key=lambda x: -x[1]):
                    lines.append(f"    {cat_name}: {count}")

            # Also run alert check
            new_alerts = engine.check_alerts()
            if new_alerts:
                lines.append("")
                lines.append(f"  NEW ALERTS GENERATED: {len(new_alerts)}")
                for a in new_alerts:
                    lines.append(f"    [{a.get('severity', '?').upper()}] {a.get('message', '')}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif action == "export":
            csv_str = engine.export_csv(
                category=arguments.get("category"),
                days=arguments.get("days"),
            )
            if not csv_str:
                return [TextContent(type="text", text="No data to export.")]

            row_count = csv_str.count("\n") - 1  # Subtract header
            return [TextContent(
                type="text",
                text=f"Exported {row_count} row(s):\n\n{csv_str[:5000]}",
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown action '{action}'. Use: log_metric, get_metrics, get_trends, get_alerts, ack_alert, eval_stats, export.",
            )]

    except Exception as e:
        return [TextContent(type="text", text=f"Eval error: {e}")]


HANDLERS = {"watty_eval": handle_eval}
