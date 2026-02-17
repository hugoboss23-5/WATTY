"""
Watty Screen Watcher — Persistent Background Observer
=====================================================
One tool: watty_watcher(action=...). 5 actions.
Watches Hugo's screen activity and stores meaningful observations
into the brain. Runs as a daemon thread. Captures screenshots via
the existing _Vision._eye, runs OCR, detects window changes, and
stores meaningful observations with provider='watcher'.

Hugo & Watty · February 2026
"""

import json
import hashlib
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME

# ── Paths & Constants ──────────────────────────────────────

WATCHER_DIR = WATTY_HOME / "watcher"
OBSERVATIONS_FILE = WATCHER_DIR / "observations.jsonl"
WATCHER_CONFIG_FILE = WATCHER_DIR / "config.json"

PROVIDER = "watcher"

DEFAULT_CONFIG = {
    "interval_seconds": 45,
    "min_interval_seconds": 30,
    "text_change_threshold": 0.30,
    "max_text_length": 500,
    "ocr_enabled": True,
    "ocr_timeout_seconds": 10,
    "ignore_windows": ["Task Manager"],
    "max_observations_kept": 500,
}


# ── Helpers ────────────────────────────────────────────────

def _log(msg):
    print(f"[Watty Watcher] {msg}", file=sys.stderr, flush=True)


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _now_local():
    return datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")


def _ensure_dir():
    WATCHER_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    if WATCHER_CONFIG_FILE.exists():
        try:
            return json.loads(WATCHER_CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_CONFIG.copy()


def _save_config(config: dict):
    _ensure_dir()
    WATCHER_CONFIG_FILE.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


# ── Brain Reference ───────────────────────────────────────

_brain_ref = None


def set_brain(brain):
    global _brain_ref
    _brain_ref = brain


# ── Observation Storage ───────────────────────────────────

def _append_observation(obs: dict):
    _ensure_dir()
    with open(OBSERVATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")
    _trim_observations()


def _trim_observations():
    if not OBSERVATIONS_FILE.exists():
        return
    config = _load_config()
    max_kept = config.get("max_observations_kept", 500)
    try:
        lines = OBSERVATIONS_FILE.read_text(encoding="utf-8").strip().split("\n")
    except OSError:
        return
    if len(lines) > max_kept:
        trimmed = lines[-max_kept:]
        OBSERVATIONS_FILE.write_text("\n".join(trimmed) + "\n", encoding="utf-8")


def _load_recent(n: int = 20) -> list[dict]:
    if not OBSERVATIONS_FILE.exists():
        return []
    try:
        lines = OBSERVATIONS_FILE.read_text(encoding="utf-8").strip().split("\n")
    except OSError:
        return []
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))


def _store_to_brain(obs: dict):
    if not _brain_ref:
        return
    lines = [
        f"[SCREEN OBSERVATION] {obs['timestamp_local']}",
        f"App: {obs['app_name']} | Window: {obs['window_title'][:80]}",
        f"Change: {obs['change_type']} - {obs['change_summary'][:100]}",
    ]
    if obs.get("text_digest"):
        lines.append(f"Visible: {obs['text_digest'][:300]}")
    content = "\n".join(lines)
    try:
        _brain_ref.store_memory(content, provider=PROVIDER)
    except Exception as e:
        _log(f"Brain store error: {e}")


# ── Screen Observation Pipeline ───────────────────────────

def _get_active_window() -> tuple[str, str]:
    """Returns (window_title, app_name)."""
    # Primary: win32gui (most reliable on Windows)
    try:
        import win32gui
        import win32process
        import psutil
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc = psutil.Process(pid)
            app_name = proc.name().replace(".exe", "")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            app_name = "unknown"
        return title, app_name
    except ImportError:
        pass

    # Fallback: pyautogui
    try:
        import pyautogui
        win = pyautogui.getActiveWindow()
        if win:
            title = win.title or ""
            app_name = title.split(" - ")[-1].strip() if " - " in title else title[:30]
            return title, app_name
    except Exception:
        pass

    return "", "unknown"


def _text_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word sets. 0.0 = different, 1.0 = identical."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union) if union else 1.0


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _run_ocr(img) -> str:
    config = _load_config()
    if not config.get("ocr_enabled", True):
        return ""
    try:
        from PIL import Image as PILImage
        import pytesseract
        # Downscale for speed
        w, h = img.size
        if w > 1280:
            scale = 1280 / w
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        text = pytesseract.image_to_string(
            img,
            timeout=config.get("ocr_timeout_seconds", 10),
        )
        return text.strip()
    except Exception as e:
        _log(f"OCR error: {e}")
        return ""


def _capture_observation() -> dict | None:
    """Capture a single screen observation. Returns dict or None if nothing changed."""
    config = _load_config()

    window_title, app_name = _get_active_window()
    if not window_title:
        return None

    # Check ignore list
    for pattern in config.get("ignore_windows", []):
        if pattern.lower() in window_title.lower():
            return None

    # Capture via existing _eye
    try:
        from watty.tools_screen import _eye
        img, w, h = _eye.see()
        if img is None:
            return None
    except Exception as e:
        _log(f"Vision error: {e}")
        return None

    window_changed = (window_title != _watcher.last_window_title)

    # OCR
    ocr_text = _run_ocr(img)
    text_hash = _hash_text(ocr_text)
    max_text = config.get("max_text_length", 500)

    # Check content change
    content_changed = False
    if not window_changed and _watcher.last_text_hash:
        if text_hash != _watcher.last_text_hash:
            threshold = config.get("text_change_threshold", 0.30)
            if _watcher.last_observation and _watcher.last_observation.get("text_digest"):
                sim = _text_similarity(ocr_text[:max_text], _watcher.last_observation["text_digest"])
                content_changed = sim < (1.0 - threshold)
            else:
                content_changed = True

    if not window_changed and not content_changed:
        return None

    # Build observation
    if window_changed:
        change_type = "window_switch"
        old_app = _watcher.last_observation.get("app_name", "?") if _watcher.last_observation else "none"
        change_summary = f"Switched from {old_app} to {app_name}"
    else:
        change_type = "content_change"
        change_summary = f"Screen content updated in {app_name}"

    # Compress text
    text_digest = ocr_text[:max_text]
    try:
        from watty.compressor import compress
        text_digest, _ = compress(text_digest)
    except Exception:
        pass

    return {
        "timestamp": _now_utc(),
        "timestamp_local": _now_local(),
        "window_title": window_title[:200],
        "app_name": app_name,
        "text_digest": text_digest[:max_text],
        "change_type": change_type,
        "change_summary": change_summary,
        "text_hash": text_hash,
    }


# ── Watcher Daemon Thread ────────────────────────────────

class _WatcherDaemon:

    def __init__(self):
        self.running = False
        self.thread = None
        self.last_observation = None
        self.last_store_time = 0.0
        self.observation_count = 0
        self.last_window_title = ""
        self.last_text_hash = ""
        self.started_at = None

    def start(self) -> str:
        if self.running:
            return "already_running"
        self.running = True
        self.started_at = time.time()
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name="watty-watcher"
        )
        self.thread.start()
        _log("Started")
        return "started"

    def stop(self) -> str:
        if not self.running:
            return "already_stopped"
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        _log("Stopped")
        return "stopped"

    def status(self) -> dict:
        uptime = time.time() - self.started_at if self.started_at and self.running else 0
        return {
            "running": self.running,
            "observation_count": self.observation_count,
            "last_observation_time": (
                self.last_observation.get("timestamp_local", "never")
                if self.last_observation else "never"
            ),
            "last_window": self.last_window_title or "none",
            "uptime_seconds": round(uptime),
        }

    def _loop(self):
        config = _load_config()
        interval = config.get("interval_seconds", 45)
        min_interval = config.get("min_interval_seconds", 30)
        _log(f"Loop started (interval={interval}s, min_store={min_interval}s)")

        while self.running:
            try:
                obs = _capture_observation()
                if obs:
                    now = time.time()
                    if now - self.last_store_time >= min_interval:
                        _append_observation(obs)
                        _store_to_brain(obs)
                        self.last_store_time = now
                        self.observation_count += 1

                    # Always update tracking state
                    self.last_observation = obs
                    self.last_window_title = obs["window_title"]
                    self.last_text_hash = obs["text_hash"]
            except Exception as e:
                _log(f"Observation error: {e}")

            # Reload config each cycle
            config = _load_config()
            interval = config.get("interval_seconds", 45)
            min_interval = config.get("min_interval_seconds", 30)
            time.sleep(interval)


# Module-level singleton (NOT auto-started)
_watcher = _WatcherDaemon()


# ── MCP Tool Definition ──────────────────────────────────

WATCHER_ACTIONS = ["start", "stop", "status", "recent", "config"]

TOOLS = [
    Tool(
        name="watty_watcher",
        description=(
            "Watty's persistent screen observer. Watches Hugo's screen activity "
            "and stores meaningful observations into the brain.\n"
            "Actions:\n"
            "  start  -- Begin watching (starts background observer)\n"
            "  stop   -- Pause watching\n"
            "  status -- Is watching? Last observation, count, uptime\n"
            "  recent -- Show last N observations\n"
            "  config -- View or update settings (interval, sensitivity, OCR toggle)"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": WATCHER_ACTIONS,
                    "description": "Action to perform",
                },
                "n": {
                    "type": "integer",
                    "description": "recent: Number of observations to show (default: 10)",
                },
                "config_updates": {
                    "type": "object",
                    "description": "config: JSON settings to update",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Action Handlers ───────────────────────────────────────

async def _action_start(args: dict) -> list[TextContent]:
    result = _watcher.start()
    if result == "already_running":
        return [TextContent(type="text", text="Watcher is already running.")]
    return [TextContent(type="text", text=(
        "Watcher started. Observing screen activity.\n"
        "Observations stored with provider='watcher'.\n"
        "Use action='recent' to see what I've captured."
    ))]


async def _action_stop(args: dict) -> list[TextContent]:
    result = _watcher.stop()
    if result == "already_stopped":
        return [TextContent(type="text", text="Watcher is not running.")]
    return [TextContent(type="text", text=(
        f"Watcher stopped. {_watcher.observation_count} observations this session."
    ))]


async def _action_status(args: dict) -> list[TextContent]:
    s = _watcher.status()
    running_str = "RUNNING" if s["running"] else "STOPPED"
    lines = [
        f"Watcher: {running_str}",
        f"Observations: {s['observation_count']}",
        f"Last observed: {s['last_observation_time']}",
        f"Last window: {s['last_window'][:80]}",
    ]
    if s["uptime_seconds"] > 0:
        mins = s["uptime_seconds"] // 60
        secs = s["uptime_seconds"] % 60
        lines.append(f"Uptime: {mins}m {secs}s")
    config = _load_config()
    lines.append(f"Interval: {config.get('interval_seconds', 45)}s")
    lines.append(f"OCR: {'on' if config.get('ocr_enabled', True) else 'off'}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _action_recent(args: dict) -> list[TextContent]:
    n = args.get("n", 10)
    observations = _load_recent(n)
    if not observations:
        return [TextContent(type="text", text="No observations yet. Start the watcher first.")]
    lines = [f"-- Last {len(observations)} observations --\n"]
    for i, obs in enumerate(observations, 1):
        lines.append(
            f"[{i}] {obs.get('timestamp_local', '?')} | "
            f"{obs.get('app_name', '?')} | "
            f"{obs.get('change_type', '?')}"
        )
        lines.append(f"    Window: {obs.get('window_title', '?')[:70]}")
        lines.append(f"    Change: {obs.get('change_summary', '?')[:80]}")
        if obs.get("text_digest"):
            preview = obs["text_digest"][:120].replace("\n", " ")
            lines.append(f"    Text: {preview}...")
        lines.append("")
    return [TextContent(type="text", text="\n".join(lines))]


async def _action_config(args: dict) -> list[TextContent]:
    updates = args.get("config_updates")
    if updates:
        config = _load_config()
        config.update(updates)
        _save_config(config)
        return [TextContent(type="text", text=f"Watcher config updated:\n{json.dumps(config, indent=2)}")]
    config = _load_config()
    return [TextContent(type="text", text=f"-- Watcher Config --\n{json.dumps(config, indent=2)}")]


# ── Dispatcher ────────────────────────────────────────────

_ACTION_MAP = {
    "start": _action_start,
    "stop": _action_stop,
    "status": _action_status,
    "recent": _action_recent,
    "config": _action_config,
}


async def handle_watcher(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _ACTION_MAP:
        return [TextContent(
            type="text",
            text=f"Unknown watcher action: {action}. Valid: {', '.join(WATCHER_ACTIONS)}"
        )]
    return await _ACTION_MAP[action](args)


HANDLERS = {
    "watty_watcher": handle_watcher,
}
