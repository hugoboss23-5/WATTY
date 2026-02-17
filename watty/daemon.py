"""
Watty Daemon — The Heartbeat
=============================
A background process that gives Watty autonomy.
Runs 24/7. Watches. Learns. Consolidates. Acts.

Architecture:
  Scheduler    — Cron-like jobs (dream, cluster, surface, scan)
  FileWatcher  — Auto-ingest new files in watched directories
  TaskQueue    — Accept and execute queued work
  GPUManager   — Auto-stop idle GPUs to save money
  ActivityLog  — Everything the daemon does, logged

All state lives in ~/.watty/daemon/
No extra HTTP server — reads/writes JSON files that MCP + frontend can access.

Hugo Bulliard · February 2026
"""

import asyncio
import json
import os
import signal
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from watty.brain import Brain
from watty.config import (
    WATTY_HOME, SCAN_EXTENSIONS, SCAN_IGNORE_DIRS, SCAN_MAX_FILE_SIZE,
    ensure_home,
)

# ── Paths ───────────────────────────────────────────────────

DAEMON_DIR = WATTY_HOME / "daemon"
PID_FILE = DAEMON_DIR / "pid"
STATE_FILE = DAEMON_DIR / "state.json"
ACTIVITY_LOG = DAEMON_DIR / "activity.jsonl"
INSIGHTS_FILE = DAEMON_DIR / "insights.jsonl"
TASK_QUEUE_FILE = DAEMON_DIR / "task_queue.jsonl"
TASK_RESULTS_FILE = DAEMON_DIR / "task_results.jsonl"
WATCH_CONFIG_FILE = DAEMON_DIR / "watch_dirs.json"
DAEMON_CONFIG_FILE = DAEMON_DIR / "config.json"

DEFAULT_CONFIG = {
    "schedule": {
        "dream_interval_hours": 6,
        "scan_interval_hours": 1,
        "cluster_interval_hours": 12,
        "surface_interval_hours": 2,
        "gpu_check_interval_minutes": 5,
    },
    "watch_dirs": [
        str(Path.home() / "Documents"),
        str(Path.home() / "Desktop"),
        str(Path.home() / "Downloads"),
    ],
    "gpu": {
        "auto_stop_idle_minutes": 30,
        "cost_alert_threshold": 5.0,
    },
    "limits": {
        "max_scan_files_per_cycle": 50,
        "max_insights_kept": 100,
        "max_activity_lines": 5000,
    },
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _now_local():
    return datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")


def _log_print(msg):
    print(f"[watty-daemon] {_now_local()} | {msg}", file=sys.stderr, flush=True)


# ── Activity Logger ─────────────────────────────────────────

class ActivityLog:
    def __init__(self):
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, detail: str = "", result: str = "ok"):
        entry = {
            "timestamp": _now(),
            "time_local": _now_local(),
            "action": action,
            "detail": detail,
            "result": result,
        }
        with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        _log_print(f"{action}: {detail[:80]}" if detail else action)

        # Trim if too long
        self._trim()

    def _trim(self):
        if not ACTIVITY_LOG.exists():
            return
        lines = ACTIVITY_LOG.read_text(encoding="utf-8").strip().split("\n")
        max_lines = _load_config().get("limits", {}).get("max_activity_lines", 5000)
        if len(lines) > max_lines:
            trimmed = lines[-max_lines:]
            ACTIVITY_LOG.write_text("\n".join(trimmed) + "\n", encoding="utf-8")

    def recent(self, n: int = 20) -> list[dict]:
        if not ACTIVITY_LOG.exists():
            return []
        lines = ACTIVITY_LOG.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries


# ── State Manager ───────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_config() -> dict:
    if DAEMON_CONFIG_FILE.exists():
        try:
            return json.loads(DAEMON_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CONFIG


def _save_config(config: dict):
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    DAEMON_CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ── File Watcher ────────────────────────────────────────────

class FileWatcher:
    """Polls watched directories for new/changed files."""

    def __init__(self, brain: Brain, activity: ActivityLog):
        self.brain = brain
        self.activity = activity
        self._seen_hashes: dict[str, str] = {}  # path -> file_hash
        self._load_seen()

    def _seen_file(self):
        return DAEMON_DIR / "seen_files.json"

    def _load_seen(self):
        f = self._seen_file()
        if f.exists():
            try:
                self._seen_hashes = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                self._seen_hashes = {}

    def _save_seen(self):
        self._seen_file().write_text(
            json.dumps(self._seen_hashes), encoding="utf-8"
        )

    def _hash_file(self, path: Path) -> str:
        h = hashlib.md5()
        h.update(str(path.stat().st_size).encode())
        h.update(str(path.stat().st_mtime).encode())
        return h.hexdigest()

    def scan_watched_dirs(self) -> dict:
        """Scan all watched directories for new/changed files."""
        config = _load_config()
        watch_dirs = config.get("watch_dirs", DEFAULT_CONFIG["watch_dirs"])
        max_files = config.get("limits", {}).get("max_scan_files_per_cycle", 50)

        new_files = 0
        total_chunks = 0
        errors = 0

        for dir_path in watch_dirs:
            dp = Path(dir_path)
            if not dp.exists() or not dp.is_dir():
                continue

            files_this_cycle = 0
            try:
                for item in dp.rglob("*"):
                    if files_this_cycle >= max_files:
                        break

                    # Skip ignored dirs
                    if any(ign in item.parts for ign in SCAN_IGNORE_DIRS):
                        continue

                    if not item.is_file():
                        continue

                    # Check extension
                    if item.suffix.lower() not in SCAN_EXTENSIONS:
                        continue

                    # Check size
                    try:
                        if item.stat().st_size > SCAN_MAX_FILE_SIZE:
                            continue
                        if item.stat().st_size == 0:
                            continue
                    except OSError:
                        continue

                    # Check if new or changed
                    file_hash = self._hash_file(item)
                    str_path = str(item)
                    if self._seen_hashes.get(str_path) == file_hash:
                        continue

                    # New or modified file — ingest it
                    try:
                        result = self.brain.scan_directory(str(item), recursive=False)
                        chunks = result.get("chunks_stored", 0)
                        total_chunks += chunks
                        new_files += 1
                        files_this_cycle += 1
                        self._seen_hashes[str_path] = file_hash
                        self.activity.log(
                            "file_ingested",
                            f"{item.name} ({chunks} chunks)",
                        )
                    except Exception as e:
                        errors += 1
                        self.activity.log("file_error", str(item), str(e))

            except PermissionError:
                self.activity.log("dir_permission_error", dir_path)
            except Exception as e:
                self.activity.log("dir_scan_error", dir_path, str(e))

        self._save_seen()
        return {"new_files": new_files, "chunks": total_chunks, "errors": errors}


# ── Task Queue ──────────────────────────────────────────────

class TaskQueue:
    """Simple file-based task queue. Drop tasks in, daemon picks them up."""

    def __init__(self, brain: Brain, activity: ActivityLog):
        self.brain = brain
        self.activity = activity

    def _read_queue(self) -> list[dict]:
        if not TASK_QUEUE_FILE.exists():
            return []
        lines = TASK_QUEUE_FILE.read_text(encoding="utf-8").strip().split("\n")
        tasks = []
        for line in lines:
            if line.strip():
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return tasks

    def _write_queue(self, tasks: list[dict]):
        with open(TASK_QUEUE_FILE, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

    def _write_result(self, result: dict):
        with open(TASK_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def add_task(self, task_type: str, action: str, params: dict = None,
                 priority: int = 5) -> str:
        """Add a task to the queue. Returns task ID."""
        import uuid
        task_id = f"t_{uuid.uuid4().hex[:8]}"
        task = {
            "id": task_id,
            "type": task_type,
            "action": action,
            "params": params or {},
            "priority": priority,
            "status": "pending",
            "created_at": _now(),
        }
        with open(TASK_QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(task) + "\n")
        return task_id

    def process_next(self) -> Optional[dict]:
        """Process the next pending task. Returns result or None."""
        tasks = self._read_queue()
        pending = [t for t in tasks if t.get("status") == "pending"]
        if not pending:
            return None

        # Sort by priority (lower = higher priority)
        pending.sort(key=lambda t: t.get("priority", 5))
        task = pending[0]
        task["status"] = "running"
        task["started_at"] = _now()
        self._write_queue(tasks)

        self.activity.log("task_started", f"{task['type']}:{task['action']} ({task['id']})")

        result = self._execute_task(task)

        # Update task status
        task["status"] = "completed" if result.get("success") else "failed"
        task["completed_at"] = _now()
        task["result"] = result

        # Remove from queue, add to results
        remaining = [t for t in tasks if t["id"] != task["id"]]
        self._write_queue(remaining)
        self._write_result(task)

        self.activity.log(
            "task_completed" if result.get("success") else "task_failed",
            f"{task['type']}:{task['action']} ({task['id']})",
            json.dumps(result)[:200],
        )
        return result

    def _execute_task(self, task: dict) -> dict:
        """Execute a single task."""
        try:
            t_type = task.get("type", "")
            action = task.get("action", "")
            params = task.get("params", {})

            if t_type == "brain":
                return self._exec_brain(action, params)
            elif t_type == "shell":
                return self._exec_shell(action, params)
            elif t_type == "gpu":
                return self._exec_gpu(action, params)
            elif t_type == "inference":
                return self._exec_inference(action, params)
            else:
                return {"success": False, "error": f"Unknown task type: {t_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_brain(self, action: str, params: dict) -> dict:
        if action == "scan":
            path = params.get("path", str(Path.home() / "Documents"))
            result = self.brain.scan_directory(path, recursive=params.get("recursive", True))
            return {"success": True, **result}
        elif action == "dream":
            result = self.brain.dream()
            return {"success": True, **result}
        elif action == "cluster":
            clusters = self.brain.cluster()
            return {"success": True, "clusters": len(clusters)}
        elif action == "recall":
            results = self.brain.recall(params.get("query", ""), top_k=params.get("top_k", 10))
            return {"success": True, "results": len(results)}
        elif action == "surface":
            insights = self.brain.surface(context=params.get("context"))
            return {"success": True, "insights": len(insights)}
        elif action == "forget":
            result = self.brain.forget(**params)
            return {"success": True, **result}
        else:
            return {"success": False, "error": f"Unknown brain action: {action}"}

    def _exec_shell(self, command: str, params: dict) -> dict:
        import subprocess
        timeout = params.get("timeout", 60)
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return {
                "success": r.returncode == 0,
                "stdout": r.stdout[:2000],
                "stderr": r.stderr[:500],
                "exit_code": r.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout after {timeout}s"}

    def _exec_gpu(self, action: str, params: dict) -> dict:
        import subprocess
        cmd_map = {
            "start": "vastai start instance {iid}",
            "stop": "vastai stop instance {iid}",
            "status": "vastai show instance {iid}",
        }
        iid_file = Path.home() / ".basho_gpu" / "instance"
        iid = iid_file.read_text(encoding="utf-8").strip() if iid_file.exists() else ""
        if not iid:
            return {"success": False, "error": "No GPU instance configured"}
        template = cmd_map.get(action)
        if not template:
            return {"success": False, "error": f"Unknown GPU action: {action}"}
        cmd = template.format(iid=iid)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return {"success": r.returncode == 0, "output": r.stdout[:500]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_inference(self, action: str, params: dict) -> dict:
        """Execute inference tasks via Ollama."""
        from watty.tools_inference import handle_watty_infer
        import json as _json
        try:
            result_str = handle_watty_infer(
                {"action": action, **params},
                brain=self.brain,
            )
            result = _json.loads(result_str)
            return {"success": "error" not in result, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ── GPU Manager ─────────────────────────────────────────────

class GPUManager:
    """Auto-stop idle GPUs to save money."""

    def __init__(self, activity: ActivityLog):
        self.activity = activity
        self._last_active = None

    def check(self):
        """Check GPU status and auto-stop if idle too long."""
        import subprocess
        config = _load_config()
        idle_minutes = config.get("gpu", {}).get("auto_stop_idle_minutes", 30)
        cost_alert = config.get("gpu", {}).get("cost_alert_threshold", 5.0)

        iid_file = Path.home() / ".basho_gpu" / "instance"
        if not iid_file.exists():
            return

        iid = iid_file.read_text(encoding="utf-8").strip()
        if not iid:
            return

        try:
            r = subprocess.run(
                ["vastai", "show", "instance", iid, "--raw"],
                capture_output=True, text=True, timeout=20
            )
            if r.returncode != 0:
                return

            data = json.loads(r.stdout.strip())
            status = data.get("actual_status", "")
            if status != "running":
                return

            gpu_util = data.get("gpu_util", 0) or 0
            cost_per_hr = data.get("dph_total", 0) or 0

            # Track utilization
            if gpu_util > 5:
                self._last_active = time.time()
            elif self._last_active is None:
                self._last_active = time.time()

            # Auto-stop if idle
            idle_secs = time.time() - (self._last_active or time.time())
            if idle_secs > idle_minutes * 60:
                self.activity.log(
                    "gpu_auto_stop",
                    f"Idle {idle_secs/60:.0f}min, saving ${cost_per_hr:.3f}/hr",
                )
                subprocess.run(
                    ["vastai", "stop", "instance", iid],
                    capture_output=True, timeout=20
                )
                return

            # Cost alert
            try:
                cr = subprocess.run(
                    ["vastai", "show", "user", "--raw"],
                    capture_output=True, text=True, timeout=15
                )
                if cr.returncode == 0:
                    user_data = json.loads(cr.stdout.strip())
                    credit = float(user_data.get("credit", 0))
                    if credit < cost_alert:
                        self.activity.log(
                            "gpu_low_credit",
                            f"${credit:.2f} remaining at ${cost_per_hr:.3f}/hr",
                        )
            except Exception:
                pass

        except Exception as e:
            self.activity.log("gpu_check_error", str(e))


# ── Insight Engine ──────────────────────────────────────────

class InsightEngine:
    """Proactively surfaces insights and saves them."""

    def __init__(self, brain: Brain, activity: ActivityLog):
        self.brain = brain
        self.activity = activity

    def generate(self):
        """Surface new insights and save them."""
        try:
            insights = self.brain.surface()
            if not insights:
                self.activity.log("surface", "no new insights")
                return

            config = _load_config()
            max_kept = config.get("limits", {}).get("max_insights_kept", 100)

            for insight in insights:
                entry = {
                    "timestamp": _now(),
                    "content": insight.get("content", ""),
                    "score": insight.get("score", 0),
                    "connections": insight.get("connections", []),
                }
                with open(INSIGHTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

            self.activity.log("surface", f"{len(insights)} insights generated")

            # Trim old insights
            if INSIGHTS_FILE.exists():
                lines = INSIGHTS_FILE.read_text(encoding="utf-8").strip().split("\n")
                if len(lines) > max_kept:
                    trimmed = lines[-max_kept:]
                    INSIGHTS_FILE.write_text("\n".join(trimmed) + "\n", encoding="utf-8")

        except Exception as e:
            self.activity.log("surface_error", str(e))


# ── Scheduler ───────────────────────────────────────────────

class Scheduler:
    """Cron-like scheduler using asyncio."""

    def __init__(self):
        self._jobs: list[dict] = []

    def every(self, hours: float, name: str, func):
        self._jobs.append({
            "name": name,
            "interval_seconds": hours * 3600,
            "func": func,
            "last_run": 0,
        })

    def every_minutes(self, minutes: float, name: str, func):
        self._jobs.append({
            "name": name,
            "interval_seconds": minutes * 60,
            "func": func,
            "last_run": 0,
        })

    async def run(self, activity: ActivityLog):
        """Main scheduler loop."""
        _log_print(f"Scheduler started with {len(self._jobs)} jobs")
        while True:
            now = time.time()
            for job in self._jobs:
                elapsed = now - job["last_run"]
                if elapsed >= job["interval_seconds"]:
                    try:
                        activity.log("job_started", job["name"])
                        result = job["func"]()
                        job["last_run"] = time.time()
                        activity.log("job_completed", job["name"],
                                     json.dumps(result)[:200] if result else "ok")
                    except Exception as e:
                        activity.log("job_error", job["name"], str(e))
                        job["last_run"] = time.time()  # Don't retry immediately

            await asyncio.sleep(30)  # Check every 30 seconds


# ── Main Daemon ─────────────────────────────────────────────

class WattyDaemon:
    """The autonomous heart of Watty."""

    def __init__(self):
        ensure_home()
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure config exists
        if not DAEMON_CONFIG_FILE.exists():
            _save_config(DEFAULT_CONFIG)

        self.brain = Brain()
        self.activity = ActivityLog()
        self.file_watcher = FileWatcher(self.brain, self.activity)
        self.task_queue = TaskQueue(self.brain, self.activity)
        self.gpu_manager = GPUManager(self.activity)
        self.insight_engine = InsightEngine(self.brain, self.activity)
        self.scheduler = Scheduler()
        self._running = False

        self._setup_schedule()

    def _setup_schedule(self):
        config = _load_config()
        sched = config.get("schedule", DEFAULT_CONFIG["schedule"])

        # Dream cycle — consolidate memories
        self.scheduler.every(
            sched.get("dream_interval_hours", 6),
            "dream",
            self._job_dream,
        )

        # File scan — watch directories for new content
        self.scheduler.every(
            sched.get("scan_interval_hours", 1),
            "file_scan",
            self._job_scan,
        )

        # Cluster — reorganize knowledge graph
        self.scheduler.every(
            sched.get("cluster_interval_hours", 12),
            "cluster",
            self._job_cluster,
        )

        # Surface insights — proactive intelligence
        self.scheduler.every(
            sched.get("surface_interval_hours", 2),
            "surface",
            self._job_surface,
        )

        # GPU cost management
        self.scheduler.every_minutes(
            sched.get("gpu_check_interval_minutes", 5),
            "gpu_check",
            self._job_gpu_check,
        )

        # Task queue processing — every minute
        self.scheduler.every_minutes(1, "task_queue", self._job_process_tasks)

    # ── Scheduled Jobs ──────────────────────────────────────

    def _job_dream(self):
        result = self.brain.dream()
        return result

    def _job_scan(self):
        result = self.file_watcher.scan_watched_dirs()
        return result

    def _job_cluster(self):
        clusters = self.brain.cluster()
        return {"clusters": len(clusters)}

    def _job_surface(self):
        self.insight_engine.generate()
        return {"surfaced": True}

    def _job_gpu_check(self):
        self.gpu_manager.check()
        return {"checked": True}

    def _job_process_tasks(self):
        result = self.task_queue.process_next()
        if result:
            return result
        return None

    # ── Lifecycle ───────────────────────────────────────────

    def _write_pid(self):
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    def _clear_pid(self):
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _update_state(self, **kwargs):
        state = _load_state()
        state.update(kwargs)
        state["last_heartbeat"] = _now()
        state["pid"] = os.getpid()
        _save_state(state)

    async def _heartbeat_loop(self):
        """Update state file periodically so others know we're alive."""
        while self._running:
            self._update_state(status="running")
            await asyncio.sleep(15)

    async def run(self):
        """Main daemon entry point."""
        self._running = True
        self._write_pid()

        self._update_state(
            status="starting",
            started_at=_now(),
            version="1.0.0",
        )

        self.activity.log("daemon_started", f"PID {os.getpid()}")

        # Run initial operations on first start
        self.activity.log("initial_scan", "Running first file scan...")
        try:
            scan_result = self.file_watcher.scan_watched_dirs()
            self.activity.log("initial_scan_complete",
                              f"{scan_result['new_files']} files, {scan_result['chunks']} chunks")
        except Exception as e:
            self.activity.log("initial_scan_error", str(e))

        self._update_state(status="running")

        # Handle shutdown signals
        def shutdown(sig, frame):
            _log_print(f"Received signal {sig}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, shutdown)
        if hasattr(signal, 'SIGTERM'):
            try:
                signal.signal(signal.SIGTERM, shutdown)
            except (OSError, ValueError):
                pass  # Windows may not support SIGTERM in all contexts

        try:
            await asyncio.gather(
                self.scheduler.run(self.activity),
                self._heartbeat_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._update_state(status="stopped", stopped_at=_now())
            self._clear_pid()
            self.activity.log("daemon_stopped", "Clean shutdown")
            _log_print("Daemon stopped.")


# ── Status / Control (called from CLI and MCP) ─────────────

def daemon_status() -> dict:
    """Get daemon status without starting it."""
    state = _load_state()
    if not state:
        return {"running": False, "status": "not_started"}

    pid = state.get("pid")
    alive = False
    if pid:
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    alive = True
            else:
                os.kill(pid, 0)
                alive = True
        except (OSError, ProcessLookupError, Exception):
            alive = False

    return {
        "running": alive,
        "status": state.get("status", "unknown"),
        "pid": pid,
        "started_at": state.get("started_at"),
        "last_heartbeat": state.get("last_heartbeat"),
        "version": state.get("version"),
    }


def daemon_activity(n: int = 20) -> list[dict]:
    """Get recent daemon activity."""
    log = ActivityLog()
    return log.recent(n)


def daemon_insights(n: int = 10) -> list[dict]:
    """Get recent insights."""
    if not INSIGHTS_FILE.exists():
        return []
    lines = INSIGHTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def daemon_queue_task(task_type: str, action: str, params: dict = None,
                      priority: int = 5) -> str:
    """Queue a task for the daemon to execute."""
    brain = Brain()
    activity = ActivityLog()
    tq = TaskQueue(brain, activity)
    return tq.add_task(task_type, action, params, priority)


def daemon_config() -> dict:
    """Get current daemon configuration."""
    return _load_config()


def daemon_update_config(updates: dict) -> dict:
    """Update daemon configuration."""
    config = _load_config()
    _deep_merge(config, updates)
    _save_config(config)
    return config


def _deep_merge(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def daemon_stop():
    """Stop the running daemon."""
    state = _load_state()
    pid = state.get("pid")
    if not pid:
        return {"success": False, "error": "No daemon PID found"}
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
        else:
            os.kill(pid, signal.SIGTERM)
        _save_state({**state, "status": "stopped"})
        return {"success": True, "pid": pid}
    except ProcessLookupError:
        _save_state({**state, "status": "stopped"})
        return {"success": False, "error": "Daemon not running (stale PID)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Entry Point ─────────────────────────────────────────────

def main():
    """Start the daemon. Called from CLI."""
    daemon = WattyDaemon()
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        _log_print("Interrupted. Shutting down.")


if __name__ == "__main__":
    main()
