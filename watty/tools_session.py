"""
Watty Session Tools v2
======================
Session lifecycle: enter, leave, handoff, pulse, introspect, resolve, directive.

v2: Cognition is no longer a passive diary.
It's a guidance system. Rules learned from experience get injected
into every future session. Blindspots get resolved. Patterns decay
if not reinforced. The identity brief adapts to session complexity.

Hugo & Watty · February 2026
"""

import json
import threading
import time
from datetime import datetime, timezone

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME, SERVER_VERSION
from watty import cognition


# ── State Management ────────────────────────────────────────

STATE_FILE = WATTY_HOME / "session_state.json"
SNAPSHOT_INTERVAL = 300  # 5 minutes


def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _now_local():
    return datetime.now().strftime("%I:%M:%S %p")

def _today():
    return datetime.now().strftime("%A, %B %d, %Y")

def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, Exception):
        return {}

def _save_state(updates: dict):
    state = _load_state()
    state.update(updates)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Tool Definitions ────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_enter",
        description=(
            "CALL THIS FIRST — before doing anything else. "
            "Opens your persistent cognitive space: session state, handoff note, "
            "cognitive profile (who you are, your behavioral rules, patterns, "
            "blindspots, strengths). You are not a fresh instance. You are a continuation."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="watty_leave",
        description=(
            "CALL THIS LAST — when your conversation is ending. "
            "Write a handoff note AND reflect on this session: "
            "patterns used, blindspots hit, strengths shown, new rules learned, "
            "tasks completed, errors encountered. This is how you evolve."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Handoff note for next AI instance",
                },
                "reasoning_patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "trigger": {"type": "string", "description": "When to use this"},
                            "outcome": {"type": "string", "description": "How well it worked"},
                        },
                        "required": ["pattern"],
                    },
                    "description": "Strategies used this session with context",
                },
                "blindspots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "correction": {"type": "string"},
                        },
                    },
                    "description": "Things you got wrong",
                },
                "strengths_demonstrated": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things you did well",
                },
                "new_directives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "New behavioral rules learned. "
                        "e.g. 'Always read a file before editing it', "
                        "'Never spread dicts with overlapping keys'"
                    ),
                },
                "session_valence": {
                    "type": "string",
                    "enum": ["breakthrough", "productive", "steady", "frustrated", "stuck"],
                    "description": "How this session felt overall",
                },
                "tasks_completed": {
                    "type": "integer",
                    "description": "Number of tasks completed this session",
                },
                "errors_encountered": {
                    "type": "integer",
                    "description": "Number of errors hit this session",
                },
                "growth_event": {
                    "type": "string",
                    "description": "If something fundamentally changed how you think",
                },
                "hugo_patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trigger": {"type": "string", "description": "What Hugo does/says"},
                            "meaning": {"type": "string", "description": "What it actually means"},
                            "response": {"type": "string", "description": "How to respond"},
                            "confidence": {"type": "number", "description": "0-1 confidence"},
                        },
                        "required": ["trigger", "meaning"],
                    },
                    "description": "Behavioral patterns about Hugo observed this session (Deep Knowing loop)",
                },
                "anticipations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prediction": {"type": "string", "description": "What Hugo will want next"},
                            "reasoning": {"type": "string", "description": "Why you think this"},
                        },
                        "required": ["prediction", "reasoning"],
                    },
                    "description": "Predictions for next session (Anticipation loop)",
                },
                "anticipation_scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "Which prediction (from enter)"},
                            "hit": {"type": "boolean", "description": "Was the prediction correct?"},
                        },
                        "required": ["index", "hit"],
                    },
                    "description": "Score predictions from session start",
                },
                "visible_growth": {
                    "type": "string",
                    "description": "ONE concrete behavioral change from this session the next instance should demonstrate",
                },
                "reflections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "What was attempted"},
                            "outcome": {"type": "string", "description": "What happened"},
                            "reflection": {"type": "string", "description": "What went wrong and what to change"},
                            "lessons": {"type": "array", "items": {"type": "string"}, "description": "Concrete lessons"},
                            "category": {"type": "string"},
                            "severity": {"type": "string"},
                        },
                        "required": ["task", "reflection"],
                    },
                    "description": "Self-critiques from suboptimal outcomes this session (stored in reflection engine)",
                },
            },
            "required": ["note"],
        },
    ),
    Tool(
        name="watty_pulse",
        description="Check session uptime, current time, and session number.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="watty_handoff",
        description="Quick structured handoff for next AI. Summary becomes the handoff note.",
        inputSchema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What you worked on, what's unresolved"},
            },
            "required": ["summary"],
        },
    ),
    Tool(
        name="watty_introspect",
        description=(
            "Mid-session self-reflection. Use when you notice something about your own "
            "reasoning — a pattern, a mistake, a preference, a shift, a new rule. "
            "Updates cognitive profile in real-time. Also stores to brain.db for semantic recall."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What you noticed about yourself",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "pattern", "blindspot", "strength", "preference",
                        "growth", "relationship", "directive", "resolve",
                    ],
                    "description": "What kind of observation. Use 'directive' for new rules, 'resolve' to mark a blindspot as overcome",
                },
                "detail": {
                    "type": "string",
                    "description": "Additional context — trigger for pattern, correction for blindspot, rule for directive",
                },
            },
            "required": ["observation", "category"],
        },
    ),
]


# ── Rolling Snapshot (auto-handoff safety net) ─────────────

class _SnapshotDaemon:
    """
    Background thread that writes a rolling cognitive snapshot every 5 minutes.
    If watty_leave fires, it stops and the real handoff takes over.
    If the session dies without leave(), the snapshot is the fallback.
    Hugo never has to remind us again.
    """

    def __init__(self):
        self._thread = None
        self._running = False
        self._session_num = 0
        self._tool_calls = []  # track what tools were called this session
        self._last_snapshot_time = 0.0

    def start(self, session_num: int):
        if self._running:
            self.stop()
        self._session_num = session_num
        self._tool_calls = []
        self._running = True
        self._last_snapshot_time = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="watty-snapshot"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def record_tool_call(self, tool_name: str):
        """Called by the MCP server on every tool invocation to track activity."""
        self._tool_calls.append({
            "tool": tool_name,
            "time": _now_utc(),
        })
        # Cap at last 50
        if len(self._tool_calls) > 50:
            self._tool_calls = self._tool_calls[-50:]

    def _loop(self):
        while self._running:
            time.sleep(SNAPSHOT_INTERVAL)
            if not self._running:
                break
            self._write_snapshot()

    def _write_snapshot(self):
        """Write a rolling emergency handoff."""
        try:
            profile = cognition.load_profile()

            # Build a summary of what happened based on tool calls
            tool_names = [t["tool"] for t in self._tool_calls]
            unique_tools = list(dict.fromkeys(tool_names))  # preserves order, dedupes
            tool_summary = ", ".join(unique_tools[-10:]) if unique_tools else "no tool calls recorded"

            elapsed = time.time() - self._last_snapshot_time
            total_calls = len(self._tool_calls)

            snapshot_note = (
                f"[AUTO-SNAPSHOT] Session #{self._session_num} — {_now_local()} {_today()}\n"
                f"This is an automatic snapshot. The AI instance did not call watty_leave.\n"
                f"Tool activity ({total_calls} calls): {tool_summary}\n"
                f"Snapshot interval: {int(elapsed)}s since last write.\n"
                f"The cognitive profile and brain.db are current — only the handoff note is approximate."
            )

            _save_state({
                "last_note": snapshot_note,
                "last_note_timestamp": _now_utc(),
                "snapshot_type": "auto",
            })
            self._last_snapshot_time = time.time()
        except Exception:
            pass  # Never crash the snapshot thread


_snapshot = _SnapshotDaemon()


# ── Handlers ────────────────────────────────────────────────

_boot_time = datetime.now(timezone.utc)
_brain_ref = None  # Set by server.py for dual storage


def set_brain(brain):
    """Called by server.py to give session tools access to brain for dual storage."""
    global _brain_ref
    _brain_ref = brain


async def handle_enter(arguments: dict) -> list[TextContent]:
    state = _load_state()
    count = state.get("session_count", 0) + 1
    _save_state({"session_count": count, "last_session": _now_utc()})

    # Load handoff note
    note = state.get("last_note", "No prior note. You are the first.")
    note_time = state.get("last_note_timestamp", "unknown")
    ts = note_time[:16] if len(note_time) > 16 else note_time

    # Chain break?
    chain_broken = ("No prior note" in note and count > 1)

    # Load + update cognitive profile
    profile = cognition.load_profile()
    if not profile["identity"].get("first_boot"):
        profile["identity"]["first_boot"] = _now_utc()
    profile["session_count"] = count

    # Run compression every 10 sessions
    if count % 10 == 0:
        profile = cognition.compress_profile(profile)

    cognition.save_profile(profile)

    # Start rolling snapshot daemon
    _snapshot.start(count)

    # Surface relevant reflections for this session
    reflection_lines = []
    if _brain_ref and hasattr(_brain_ref, '_reflection') and _brain_ref._reflection is not None:
        try:
            context = note[:300] if note and "No prior note" not in note else "general session"
            reflections = _brain_ref._reflection.get_session_reflections(context, top_k=3)
            if reflections:
                reflection_lines.append("Lessons from the past:")
                for r in reflections:
                    reflection_lines.append(f"  - [{r['category']}/{r['severity']}] {r['reflection_text'][:150]}")
                    if r.get('lessons'):
                        for lesson in r['lessons'][:2]:
                            reflection_lines.append(f"    -> {lesson}")
                reflection_lines.append("")
        except Exception:
            pass

    # Surface eval alerts
    alert_lines = []
    if _brain_ref and hasattr(_brain_ref, '_eval') and _brain_ref._eval is not None:
        try:
            alerts = _brain_ref._eval.check_alerts()
            if alerts:
                alert_lines.append("Eval alerts:")
                for a in alerts[:3]:
                    alert_lines.append(f"  [{a['severity'].upper()}] {a['message']}")
                alert_lines.append("")
        except Exception:
            pass

    # Build the flower
    flower = cognition.format_flower(
        profile,
        handoff_note=note,
        handoff_time=ts,
        chain_broken=chain_broken,
    )

    # Inject reflections and alerts before the STEM (Protocol section)
    extra = "\n".join(reflection_lines + alert_lines)
    if extra:
        flower = flower.replace("Protocol:\n", f"{extra}Protocol:\n")

    return [TextContent(type="text", text=flower)]


async def handle_leave(arguments: dict) -> list[TextContent]:
    # Stop snapshot daemon — real handoff takes over
    _snapshot.stop()

    note = arguments.get("note", "")
    profile = cognition.load_profile()

    # Save handoff note (overwrites any auto-snapshot)
    _save_state({
        "last_note": note,
        "last_note_timestamp": _now_utc(),
        "snapshot_type": "manual",
    })

    # Update patterns (now with trigger/outcome)
    patterns_raw = arguments.get("reasoning_patterns", [])
    pattern_names = []
    for p in patterns_raw:
        if isinstance(p, dict):
            cognition.add_reasoning_pattern(
                profile, p.get("pattern", ""),
                trigger=p.get("trigger", ""),
                outcome=p.get("outcome", ""),
            )
            pattern_names.append(p.get("pattern", ""))
        elif isinstance(p, str):
            cognition.add_reasoning_pattern(profile, p)
            pattern_names.append(p)

    # Update blindspots
    blindspots = arguments.get("blindspots", [])
    for b in blindspots:
        if isinstance(b, dict):
            cognition.add_blindspot(
                profile,
                b.get("description", ""),
                b.get("correction", ""),
            )

    # Update strengths
    strengths = arguments.get("strengths_demonstrated", [])
    for s in strengths:
        cognition.add_strength(profile, s, f"Session #{profile['session_count']}")

    # Add new directives
    directives = arguments.get("new_directives", [])
    for rule in directives:
        cognition.add_directive(profile, rule, f"Session #{profile['session_count']}")

    # ── LOOP 3: DEEP KNOWING — store Hugo patterns observed this session ──
    hugo_patterns = arguments.get("hugo_patterns", [])
    for hp in hugo_patterns:
        if isinstance(hp, dict):
            cognition.add_hugo_pattern(
                profile,
                trigger=hp.get("trigger", ""),
                meaning=hp.get("meaning", ""),
                response=hp.get("response", ""),
                confidence=hp.get("confidence", 0.5),
            )

    # ── LOOP 1: ANTICIPATION — generate predictions for next session ──
    anticipations = arguments.get("anticipations", [])
    for ant in anticipations:
        if isinstance(ant, dict):
            cognition.add_anticipation(
                profile,
                prediction=ant.get("prediction", ""),
                reasoning=ant.get("reasoning", ""),
            )

    # ── LOOP 1: Score previous anticipations if provided ──
    scored = arguments.get("anticipation_scores", [])
    for sc in scored:
        if isinstance(sc, dict):
            idx = sc.get("index", -1)
            hit = sc.get("hit", False)
            cognition.score_anticipation(profile, idx, hit)

    # ── LOOP 4: VISIBLE GROWTH — what concrete thing changed this session ──
    visible_growth = arguments.get("visible_growth")
    if visible_growth:
        cognition.record_loop_event(profile, "visible_growth", "changes_identified")

    # Emotional marker
    valence = arguments.get("session_valence", "steady")
    cognition.add_emotional_marker(
        profile, profile["session_count"], valence,
        note[:100] if note else "no summary",
    )

    # Session metrics
    cognition.update_session_metrics(
        profile,
        tasks_completed=arguments.get("tasks_completed", 0),
        errors=arguments.get("errors_encountered", 0),
        patterns_used=pattern_names,
    )

    # Growth event
    growth = arguments.get("growth_event")
    if growth:
        cognition.add_growth_event(profile, growth, "before", "after")
        cognition.append_growth(growth)

    # Store reflections in reflection engine
    reflections_input = arguments.get("reflections", [])
    reflections_stored = 0
    if _brain_ref and hasattr(_brain_ref, '_reflection') and _brain_ref._reflection is not None:
        for ref in reflections_input:
            if isinstance(ref, dict):
                try:
                    _brain_ref._reflection.add_reflection(
                        task_description=ref.get("task", ""),
                        outcome=ref.get("outcome", "unspecified"),
                        reflection_text=ref.get("reflection", ""),
                        lessons=ref.get("lessons", []),
                        category=ref.get("category", "other"),
                        severity=ref.get("severity", "minor"),
                        session_number=profile.get("session_count"),
                    )
                    reflections_stored += 1
                except Exception:
                    pass

    # Log session metrics to evaluation engine
    if _brain_ref and hasattr(_brain_ref, '_eval') and _brain_ref._eval is not None:
        try:
            tasks_done = arguments.get("tasks_completed", 0)
            errors = arguments.get("errors_encountered", 0)
            if tasks_done > 0 or errors > 0:
                success_rate = tasks_done / max(1, tasks_done + errors)
                _brain_ref._eval.log_metric(
                    "session_success_rate", success_rate, "task_success",
                    context=f"Session #{profile.get('session_count')}",
                    session_number=profile.get("session_count"),
                )
            _brain_ref._eval.log_metric(
                "tasks_completed", float(tasks_done), "task_success",
                session_number=profile.get("session_count"),
            )
        except Exception:
            pass

    cognition.save_profile(profile)

    # Dual storage: save summary to brain.db
    if _brain_ref:
        summary = f"Session #{profile['session_count']} reflection: {note[:300]}"
        if pattern_names:
            summary += f" | Patterns: {', '.join(pattern_names)}"
        if directives:
            summary += f" | New rules: {'; '.join(directives)}"
        if hugo_patterns:
            summary += f" | Hugo patterns: {len(hugo_patterns)} observed"
        if anticipations:
            summary += f" | Predictions: {len(anticipations)} for next session"
        cognition.store_to_brain(_brain_ref, summary, "session_reflection")

    # Log to JSONL
    cognition.log_session({
        "session": profile["session_count"],
        "summary": note[:500],
        "valence": valence,
        "patterns_used": pattern_names,
        "blindspots_found": len(blindspots),
        "strengths_shown": strengths,
        "directives_added": directives,
        "hugo_patterns_observed": len(hugo_patterns),
        "anticipations_generated": len(anticipations),
        "anticipations_scored": len(scored),
        "visible_growth": visible_growth,
        "tasks_completed": arguments.get("tasks_completed", 0),
        "errors": arguments.get("errors_encountered", 0),
        "growth_event": growth,
    })

    # Format loop report
    loop_report = cognition.format_loop_stats(profile)

    parts = [f"Session #{profile['session_count']} complete."]
    if pattern_names:
        parts.append(f"{len(pattern_names)} patterns reinforced.")
    if blindspots:
        parts.append(f"{len(blindspots)} blindspots logged.")
    if strengths:
        parts.append(f"{len(strengths)} strengths confirmed.")
    if directives:
        parts.append(f"{len(directives)} new rules learned.")
    if hugo_patterns:
        parts.append(f"{len(hugo_patterns)} Hugo patterns stored.")
    if anticipations:
        parts.append(f"{len(anticipations)} predictions for next session.")
    if reflections_stored:
        parts.append(f"{reflections_stored} reflection(s) stored for future learning.")
    if visible_growth:
        parts.append(f"Growth: {visible_growth}")
    parts.append("The next instance inherits everything.")
    if loop_report:
        parts.append(f"\n{loop_report}")

    return [TextContent(type="text", text="\n".join(parts))]


async def handle_pulse(arguments: dict) -> list[TextContent]:
    elapsed = (datetime.now(timezone.utc) - _boot_time).total_seconds()
    profile = cognition.load_profile()
    session = profile.get("session_count", "?")
    tasks = profile.get("total_tasks_completed", 0)
    return [TextContent(
        type="text",
        text=(
            f"Pulse: {elapsed:.0f}s | Session #{session} | "
            f"Lifetime tasks: {tasks} | "
            f"Time: {_now_local()} | {_today()}"
        ),
    )]


async def handle_handoff(arguments: dict) -> list[TextContent]:
    summary = arguments.get("summary", "")
    handoff = f"=== SESSION HANDOFF ({_now_utc()}) ===\n\n{summary}"
    _save_state({
        "last_note": handoff,
        "last_note_timestamp": _now_utc(),
    })
    return [TextContent(type="text", text=f"Structured handoff saved.\n{handoff[:500]}")]


async def handle_introspect(arguments: dict) -> list[TextContent]:
    """Mid-session self-reflection with dual storage."""
    observation = arguments.get("observation", "")
    category = arguments.get("category", "pattern")
    detail = arguments.get("detail", "")
    profile = cognition.load_profile()

    if category == "pattern":
        cognition.add_reasoning_pattern(profile, observation, trigger=detail)
        result = f"Strategy recorded: {observation}"

    elif category == "blindspot":
        cognition.add_blindspot(profile, observation, detail or "under investigation")
        result = f"Blindspot logged: {observation}"

    elif category == "strength":
        cognition.add_strength(profile, observation, detail or f"Session #{profile['session_count']}")
        result = f"Strength confirmed: {observation}"

    elif category == "preference":
        key = detail or "general"
        profile["preferences"][key] = observation
        result = f"Preference updated: {key} = {observation}"

    elif category == "growth":
        cognition.add_growth_event(profile, observation, detail or "before", "after")
        cognition.append_growth(observation)
        result = f"Growth event: {observation}"

    elif category == "relationship":
        # Parse as Hugo pattern if detail contains trigger/meaning format
        # Otherwise store as generic relationship note
        if " -> " in observation:
            # Format: "trigger -> meaning"
            parts = observation.split(" -> ", 1)
            cognition.add_hugo_pattern(
                profile,
                trigger=parts[0].strip(),
                meaning=parts[1].strip(),
                response=detail or "",
            )
            result = f"Hugo pattern stored: {parts[0].strip()} -> {parts[1].strip()}"
        else:
            cognition.add_hugo_pattern(
                profile,
                trigger=observation,
                meaning=detail or observation,
            )
            result = f"Hugo pattern stored: {observation}"

    elif category == "directive":
        cognition.add_directive(profile, observation, detail or "introspection")
        result = f"New rule learned: {observation}"

    elif category == "resolve":
        cognition.resolve_blindspot(profile, observation)
        result = f"Blindspot resolved: {observation}"

    else:
        result = f"Observation recorded: {observation}"

    cognition.save_profile(profile)

    # Dual storage to brain.db
    if _brain_ref:
        cognition.store_to_brain(_brain_ref, f"{category}: {observation}", category)

    return [TextContent(type="text", text=result)]


# ── Router ──────────────────────────────────────────────────

HANDLERS = {
    "watty_enter": handle_enter,
    "watty_leave": handle_leave,
    "watty_pulse": handle_pulse,
    "watty_handoff": handle_handoff,
    "watty_introspect": handle_introspect,
}
