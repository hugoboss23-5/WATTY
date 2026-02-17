"""
Watty A2A Protocol Engine
==========================
Agent-to-Agent communication following the A2A protocol spec.
Publishes Agent Cards, accepts incoming tasks, discovers remote agents,
delegates tasks outbound, rate-limits, and cleans up expired work.

DB table: a2a_tasks (already exists in brain.db)
Config:   A2A_ENABLED, A2A_MAX_CONCURRENT, A2A_TASK_TTL_HOURS,
          A2A_AUTH_TOKEN, A2A_RATE_LIMIT_PER_MINUTE

Hugo & Watty · February 2026
"""

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from watty.config import (
    DB_PATH,
    A2A_MAX_CONCURRENT,
    A2A_TASK_TTL_HOURS,
    A2A_AUTH_TOKEN,
    A2A_RATE_LIMIT_PER_MINUTE,
    SERVER_NAME,
    SERVER_VERSION,
)


# ── Rate Limiter ─────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter. Thread-safe."""

    def __init__(self, max_per_minute: int = A2A_RATE_LIMIT_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._windows: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "global") -> bool:
        """Return True if the request is within the rate limit, False otherwise."""
        now = time.time()
        cutoff = now - 60.0

        with self._lock:
            timestamps = self._windows.get(key, [])
            # Prune expired timestamps
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.max_per_minute:
                self._windows[key] = timestamps
                return False

            timestamps.append(now)
            self._windows[key] = timestamps
            return True

    def remaining(self, key: str = "global") -> int:
        """Return how many requests remain in the current window."""
        now = time.time()
        cutoff = now - 60.0

        with self._lock:
            timestamps = self._windows.get(key, [])
            active = [t for t in timestamps if t > cutoff]
            return max(0, self.max_per_minute - len(active))


# ── A2A Engine ───────────────────────────────────────────────

class A2AEngine:
    """
    Agent-to-Agent protocol engine for Watty.

    Handles:
      - Agent Card generation (publish this agent's capabilities)
      - Incoming task submission, execution, and lifecycle
      - Remote agent discovery (/.well-known/agent.json)
      - Outbound task delegation to remote agents
      - Rate limiting and TTL-based cleanup
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._write_lock = threading.Lock()
        self._agent_registry: dict[str, dict] = {}
        self._rate_limiter = RateLimiter()
        self._auth_token = A2A_AUTH_TOKEN or ""

    @property
    def write_lock(self) -> threading.Lock:
        """Lazy write lock — survives hot-reload where __init__ doesn't re-run."""
        if not hasattr(self, "_write_lock") or self._write_lock is None:
            self._write_lock = threading.Lock()
        return self._write_lock

    # ── DB Connection ────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and busy_timeout (same as brain.py)."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # ── Agent Card ───────────────────────────────────────────

    def generate_agent_card(
        self,
        tools_list: list | None = None,
        base_url: str = "http://localhost:8765",
    ) -> dict:
        """
        Auto-generate an A2A Agent Card from the MCP tools list.

        Returns a dict conforming to the A2A Agent Card spec:
          name, description, url, version, skills[], capabilities,
          defaultInputModes, defaultOutputModes.
        """
        skills = []
        if tools_list:
            for tool in tools_list:
                # MCP Tool objects have .name, .description, .inputSchema
                name = getattr(tool, "name", None) or tool.get("name", "unknown")
                desc = getattr(tool, "description", None) or tool.get("description", "")
                schema = getattr(tool, "inputSchema", None) or tool.get("inputSchema", {})

                skills.append({
                    "id": name,
                    "name": name,
                    "description": desc[:200] if desc else "",
                    "inputSchema": schema,
                    "tags": [name.replace("watty_", "")],
                })

        card = {
            "name": SERVER_NAME,
            "description": (
                "Watty — Persistent AI memory agent with hippocampus pipeline, "
                "knowledge graph, navigator, reflection engine, and 30+ MCP tools."
            ),
            "url": base_url,
            "version": SERVER_VERSION,
            "protocol": "a2a",
            "protocolVersion": "0.2",
            "provider": {
                "organization": "Hugo & Watty",
                "url": base_url,
            },
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": skills,
            "authentication": {
                "schemes": ["bearer"] if self._auth_token else [],
            },
        }
        return card

    # ── Incoming Tasks ───────────────────────────────────────

    def submit_task(
        self,
        skill_id: str,
        input_data: dict,
        source_agent: Optional[str] = None,
    ) -> dict:
        """
        Accept an incoming task from an external agent.

        Checks max concurrent limit, generates UUID, inserts into a2a_tasks
        with direction='incoming'. Returns {"task_id": ..., "status": "pending"}
        or {"error": ...}.
        """
        # Rate limit check
        if not self._rate_limiter.allow(source_agent or "anonymous"):
            return {"error": "Rate limit exceeded. Try again later."}

        with self.write_lock:
            conn = self._connect()
            try:
                # Check concurrent task limit
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM a2a_tasks WHERE status IN ('pending', 'running')"
                ).fetchone()
                active_count = row["cnt"] if row else 0

                if active_count >= A2A_MAX_CONCURRENT:
                    return {
                        "error": f"Max concurrent tasks reached ({A2A_MAX_CONCURRENT}). "
                                 "Try again later.",
                    }

                task_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()

                conn.execute(
                    """INSERT INTO a2a_tasks
                       (id, direction, agent_url, agent_name, skill_id,
                        status, input_json, created_at, updated_at)
                       VALUES (?, 'incoming', ?, ?, ?, 'pending', ?, ?, ?)""",
                    (
                        task_id,
                        source_agent,
                        source_agent,
                        skill_id,
                        json.dumps(input_data),
                        now,
                        now,
                    ),
                )
                conn.commit()
                return {"task_id": task_id, "status": "pending"}
            except Exception as e:
                return {"error": f"Failed to submit task: {e}"}
            finally:
                conn.close()

    def execute_task(self, task_id: str, handlers: dict) -> dict:
        """
        Execute an incoming task.

        Loads task from DB, finds the matching handler in 'handlers'
        (a dict of {tool_name: handler_fn}), runs it, stores result or error.
        Returns the updated task dict.
        """
        with self.write_lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM a2a_tasks WHERE id = ?", (task_id,)
                ).fetchone()

                if not row:
                    return {"error": f"Task {task_id} not found"}

                task = dict(row)

                if task["status"] != "pending":
                    return {"error": f"Task {task_id} is {task['status']}, not pending"}

                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE a2a_tasks SET status = 'running', updated_at = ? WHERE id = ?",
                    (now, task_id),
                )
                conn.commit()
            finally:
                conn.close()

        # Execute outside the write lock to avoid blocking other operations
        skill_id = task["skill_id"]
        input_data = json.loads(task["input_json"] or "{}")
        result = None
        error = None

        try:
            handler = handlers.get(skill_id)
            if handler is None:
                error = f"No handler registered for skill: {skill_id}"
            else:
                import asyncio
                import inspect

                if inspect.iscoroutinefunction(handler):
                    # Run async handler
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(handler(input_data))
                    finally:
                        loop.close()
                else:
                    result = handler(input_data)

                # Normalize result to JSON-serializable
                if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
                    # List of TextContent objects from MCP handlers
                    texts = []
                    for item in result:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        else:
                            texts.append(str(item))
                    result = {"output": "\n".join(texts)}
                elif isinstance(result, str):
                    result = {"output": result}
                elif not isinstance(result, dict):
                    result = {"output": str(result)}

        except Exception as e:
            error = f"Execution failed: {e}"

        # Write result back
        with self.write_lock:
            conn = self._connect()
            try:
                now = datetime.now(timezone.utc).isoformat()
                if error:
                    conn.execute(
                        """UPDATE a2a_tasks
                           SET status = 'failed', error = ?, updated_at = ?, completed_at = ?
                           WHERE id = ?""",
                        (error, now, now, task_id),
                    )
                else:
                    conn.execute(
                        """UPDATE a2a_tasks
                           SET status = 'completed', output_json = ?,
                               updated_at = ?, completed_at = ?
                           WHERE id = ?""",
                        (json.dumps(result), now, now, task_id),
                    )
                conn.commit()
                return self.get_task(task_id, conn=conn)
            finally:
                conn.close()

    # ── Task Queries ─────────────────────────────────────────

    def get_task(self, task_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
        """Return a single task dict by ID, or None if not found."""
        close_conn = conn is None
        if conn is None:
            conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM a2a_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return None
            task = dict(row)
            # Parse JSON fields
            if task.get("input_json"):
                try:
                    task["input_data"] = json.loads(task["input_json"])
                except (json.JSONDecodeError, TypeError):
                    task["input_data"] = task["input_json"]
            if task.get("output_json"):
                try:
                    task["output_data"] = json.loads(task["output_json"])
                except (json.JSONDecodeError, TypeError):
                    task["output_data"] = task["output_json"]
            return task
        finally:
            if close_conn:
                conn.close()

    def cancel_task(self, task_id: str) -> dict:
        """Cancel a pending or running task. Returns updated task or error."""
        with self.write_lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT status FROM a2a_tasks WHERE id = ?", (task_id,)
                ).fetchone()

                if not row:
                    return {"error": f"Task {task_id} not found"}

                if row["status"] not in ("pending", "running"):
                    return {"error": f"Cannot cancel task in '{row['status']}' state"}

                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """UPDATE a2a_tasks
                       SET status = 'cancelled', updated_at = ?, completed_at = ?
                       WHERE id = ?""",
                    (now, now, task_id),
                )
                conn.commit()
                return {"task_id": task_id, "status": "cancelled"}
            finally:
                conn.close()

    def list_tasks(
        self,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query tasks with optional status and direction filters."""
        conn = self._connect()
        try:
            query = "SELECT * FROM a2a_tasks WHERE 1=1"
            params: list = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if direction:
                query += " AND direction = ?"
                params.append(direction)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            tasks = []
            for row in rows:
                task = dict(row)
                if task.get("input_json"):
                    try:
                        task["input_data"] = json.loads(task["input_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if task.get("output_json"):
                    try:
                        task["output_data"] = json.loads(task["output_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                tasks.append(task)
            return tasks
        finally:
            conn.close()

    # ── Agent Discovery ──────────────────────────────────────

    def discover_agent(self, agent_url: str) -> dict:
        """
        Discover a remote agent by fetching its Agent Card from
        {agent_url}/.well-known/agent.json.

        Caches the result in self._agent_registry keyed by URL.
        Returns the parsed Agent Card dict or {"error": ...}.
        """
        url = agent_url.rstrip("/")
        card_url = f"{url}/.well-known/agent.json"

        try:
            headers = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

            resp = requests.get(card_url, headers=headers, timeout=15)
            resp.raise_for_status()
            card = resp.json()

            # Cache it
            self._agent_registry[url] = {
                "url": url,
                "card": card,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            }
            return card

        except requests.exceptions.ConnectionError:
            return {"error": f"Cannot connect to {card_url}"}
        except requests.exceptions.Timeout:
            return {"error": f"Timeout fetching {card_url}"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error from {card_url}: {e.response.status_code}"}
        except (json.JSONDecodeError, ValueError):
            return {"error": f"Invalid JSON from {card_url}"}
        except Exception as e:
            return {"error": f"Discovery failed: {e}"}

    def list_agents(self) -> dict:
        """Return all cached discovered agents."""
        return dict(self._agent_registry)

    # ── Outbound Delegation ──────────────────────────────────

    def delegate_task(
        self,
        agent_url: str,
        skill_id: str,
        input_data: dict,
        timeout: int = 30,
    ) -> dict:
        """
        Delegate a task to a remote agent.

        POSTs to {agent_url}/a2a/tasks, then polls for completion.
        Records the task locally with direction='outgoing'.
        Returns the final task result or error.
        """
        url = agent_url.rstrip("/")
        task_endpoint = f"{url}/a2a/tasks"
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Record outgoing task locally
        with self.write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO a2a_tasks
                       (id, direction, agent_url, agent_name, skill_id,
                        status, input_json, created_at, updated_at)
                       VALUES (?, 'outgoing', ?, ?, ?, 'running', ?, ?, ?)""",
                    (task_id, url, url, skill_id, json.dumps(input_data), now, now),
                )
                conn.commit()
            finally:
                conn.close()

        # Send to remote agent
        headers = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        payload = {
            "skill_id": skill_id,
            "input": input_data,
            "source_agent": f"{SERVER_NAME}/{SERVER_VERSION}",
        }

        try:
            resp = requests.post(task_endpoint, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            remote_result = resp.json()
            remote_task_id = remote_result.get("task_id")

            if not remote_task_id:
                self._update_task_error(task_id, "Remote agent did not return a task_id")
                return {"error": "Remote agent did not return a task_id", "task_id": task_id}

            # Poll for completion
            status_url = f"{url}/a2a/tasks/{remote_task_id}"
            deadline = time.time() + timeout
            poll_interval = 1.0

            while time.time() < deadline:
                time.sleep(poll_interval)
                try:
                    status_resp = requests.get(status_url, headers=headers, timeout=10)
                    status_resp.raise_for_status()
                    status_data = status_resp.json()

                    remote_status = status_data.get("status", "unknown")
                    if remote_status in ("completed", "failed", "cancelled"):
                        # Update local record
                        self._finalize_outgoing(task_id, status_data)
                        return {
                            "task_id": task_id,
                            "remote_task_id": remote_task_id,
                            "status": remote_status,
                            "result": status_data,
                        }
                except Exception:
                    pass  # Keep polling on transient errors

                # Exponential backoff (cap at 5s)
                poll_interval = min(poll_interval * 1.5, 5.0)

            # Timeout
            self._update_task_error(task_id, f"Timed out after {timeout}s waiting for remote agent")
            return {
                "task_id": task_id,
                "remote_task_id": remote_task_id,
                "status": "timeout",
                "error": f"Remote agent did not complete within {timeout}s",
            }

        except requests.exceptions.ConnectionError:
            self._update_task_error(task_id, f"Cannot connect to {task_endpoint}")
            return {"error": f"Cannot connect to {task_endpoint}", "task_id": task_id}
        except requests.exceptions.Timeout:
            self._update_task_error(task_id, f"Timeout connecting to {task_endpoint}")
            return {"error": f"Timeout connecting to {task_endpoint}", "task_id": task_id}
        except requests.exceptions.HTTPError as e:
            err = f"HTTP {e.response.status_code} from {task_endpoint}"
            self._update_task_error(task_id, err)
            return {"error": err, "task_id": task_id}
        except Exception as e:
            self._update_task_error(task_id, str(e))
            return {"error": str(e), "task_id": task_id}

    def _update_task_error(self, task_id: str, error: str):
        """Mark a task as failed with an error message."""
        with self.write_lock:
            conn = self._connect()
            try:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """UPDATE a2a_tasks
                       SET status = 'failed', error = ?, updated_at = ?, completed_at = ?
                       WHERE id = ?""",
                    (error, now, now, task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def _finalize_outgoing(self, task_id: str, remote_data: dict):
        """Update an outgoing task with the remote agent's final result."""
        with self.write_lock:
            conn = self._connect()
            try:
                now = datetime.now(timezone.utc).isoformat()
                remote_status = remote_data.get("status", "completed")
                output = remote_data.get("output", remote_data.get("result"))
                error = remote_data.get("error")

                if remote_status == "failed":
                    conn.execute(
                        """UPDATE a2a_tasks
                           SET status = 'failed', error = ?, output_json = ?,
                               updated_at = ?, completed_at = ?
                           WHERE id = ?""",
                        (
                            error or "Remote agent reported failure",
                            json.dumps(output) if output else None,
                            now, now, task_id,
                        ),
                    )
                else:
                    conn.execute(
                        """UPDATE a2a_tasks
                           SET status = ?, output_json = ?,
                               updated_at = ?, completed_at = ?
                           WHERE id = ?""",
                        (
                            remote_status,
                            json.dumps(output) if output else None,
                            now, now, task_id,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

    # ── Cleanup ──────────────────────────────────────────────

    def cleanup_expired(self) -> dict:
        """
        Delete tasks older than A2A_TASK_TTL_HOURS.
        Returns {"deleted": count}.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=A2A_TASK_TTL_HOURS)
        cutoff_iso = cutoff.isoformat()

        with self.write_lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM a2a_tasks WHERE created_at < ?",
                    (cutoff_iso,),
                )
                deleted = cursor.rowcount
                conn.commit()
                return {"deleted": deleted, "ttl_hours": A2A_TASK_TTL_HOURS}
            finally:
                conn.close()

    # ── Auth ─────────────────────────────────────────────────

    def set_auth_token(self, token: str):
        """Set the bearer auth token (in-memory only, not persisted)."""
        self._auth_token = token

    def get_auth_token(self) -> str:
        """Return the current auth token (masked)."""
        if not self._auth_token:
            return "(not set)"
        return self._auth_token[:4] + "****" + self._auth_token[-4:]
