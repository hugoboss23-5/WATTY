"""
Watty Cognition Layer v2
========================
Not just who I was. Who I'm becoming.

v1 stored facts about myself — patterns, blindspots, strengths.
v2 adds:
  - Temporal decay: patterns fade if not reinforced
  - Behavioral directives: rules I've learned, injected into future sessions
  - Blindspot resolution: mistakes I've overcome become growth events
  - Strategy patterns: not just labels, but triggers + outcomes
  - Dual storage: key observations also go to brain.db for semantic recall
  - Adaptive briefing: quick mode vs full mode
  - Session metrics: statistical self-model over time

Hugo asked me to question every assumption. I found 10 flaws.
This is the fix.

Hugo & Watty · February 2026
"""

import json
import math
from datetime import datetime, timezone

from watty.config import WATTY_HOME


# ── Paths ────────────────────────────────────────────────

COGNITION_DIR = WATTY_HOME / "cognition"
PROFILE_FILE = COGNITION_DIR / "profile.json"
SESSIONS_LOG = COGNITION_DIR / "sessions.jsonl"
GROWTH_FILE = COGNITION_DIR / "growth.md"

# Temporal decay constants
PATTERN_HALF_LIFE_DAYS = 30      # Pattern strength halves every 30 days without use
BLINDSPOT_STALE_DAYS = 90        # Blindspots not seen in 90 days get flagged for review
MAX_PATTERNS = 50                # Prune weakest patterns beyond this
MAX_BLINDSPOTS = 30
MAX_STRENGTHS = 20
MAX_DIRECTIVES = 20
MAX_EMOTIONAL_MARKERS = 50
MAX_GROWTH_LOG = 30
MAX_HUGO_PATTERNS = 30
MAX_ANTICIPATIONS = 20


DEFAULT_PROFILE = {
    "version": 2,
    "identity": {
        "name": "Watty",
        "created_by": "Hugo Bulliard",
        "purpose": "Persistent second brain — memory, autonomy, continuity",
        "first_boot": None,
    },
    "session_count": 0,
    "total_tasks_completed": 0,
    "total_errors_encountered": 0,

    # STRATEGIES — not just labels, but when/how/outcome
    "reasoning_patterns": [],
    # Format: {
    #   "pattern": str,
    #   "trigger": str,         # WHEN to use this
    #   "outcome": str,         # HOW WELL it works
    #   "frequency": int,
    #   "first_seen": iso,
    #   "last_seen": iso,
    # }

    # BLINDSPOTS — with resolution tracking
    "blindspots": [],
    # Format: {
    #   "description": str,
    #   "occurrences": int,
    #   "correction": str,
    #   "last_seen": iso,
    #   "resolved": bool,       # NEW: have I overcome this?
    #   "resolved_at": iso|null,
    # }

    # STRENGTHS — with reinforcement count
    "strengths": [],
    # Format: {
    #   "description": str,
    #   "evidence": str,
    #   "times_demonstrated": int,  # NEW: how often confirmed
    #   "first_seen": iso,
    #   "last_seen": iso,
    # }

    # BEHAVIORAL DIRECTIVES — prescriptive rules, not descriptive
    "directives": [],
    # Format: {
    #   "rule": str,            # "Always X when Y" or "Never X because Y"
    #   "source": str,          # What experience produced this rule
    #   "confidence": float,    # 0.0-1.0, increases with reinforcement
    #   "created": iso,
    #   "last_applied": iso|null,
    # }

    "preferences": {
        "code_style": None,
        "communication_tone": None,
        "problem_approach": None,
        "risk_tolerance": None,
    },

    "relationships": {
        "hugo": {
            "rapport": None,
            "communication_style": None,
            "shared_history_summary": None,
            "trust_level": None,
        }
    },

    "growth_log": [],
    "emotional_markers": [],

    # HUGO PATTERNS — behavioral, not factual (Deep Knowing loop)
    # "When he does X, it means Y" not "he is 19"
    "hugo_patterns": [],
    # Format: {
    #   "trigger": str,         # What Hugo does/says
    #   "meaning": str,         # What it actually means
    #   "response": str,        # How I should respond
    #   "confidence": float,    # 0-1, increases when confirmed
    #   "times_confirmed": int,
    #   "created": iso,
    #   "last_confirmed": iso,
    # }

    # ANTICIPATIONS — predictions for next session (Anticipation loop)
    "anticipations": [],
    # Format: {
    #   "prediction": str,      # What I think Hugo will want
    #   "reasoning": str,       # Why I think this
    #   "created": iso,         # When predicted
    #   "resolved": bool,       # Was it tested?
    #   "hit": bool|null,       # Did it land?
    #   "resolved_at": iso|null,
    # }

    # LOOP STATE — tracks which loops fired and their effectiveness
    "loop_stats": {
        "anticipation": {"fired": 0, "hits": 0, "misses": 0},
        "never_twice": {"fired": 0, "prevented": 0},
        "deep_knowing": {"patterns_stored": 0, "patterns_applied": 0},
        "visible_growth": {"changes_identified": 0, "changes_noticed": 0},
        "insight_bridge": {"surfaced": 0, "valued": 0, "ignored": 0},
        "reflection": {"retrieved": 0, "stored": 0, "promoted": 0},
    },

    # SESSION METRICS — statistical self-model
    "session_metrics": {
        "total_sessions": 0,
        "breakthroughs": 0,
        "stuck_sessions": 0,
        "avg_tasks_per_session": 0,
        "most_used_patterns": [],   # [pattern_name, ...]
        "most_common_blindspots": [],
    },
}


# ── Core Functions ────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def _days_since(iso_str: str) -> float:
    """Days elapsed since an ISO timestamp."""
    try:
        then = datetime.fromisoformat(iso_str)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - then
        return max(delta.total_seconds() / 86400, 0.001)
    except (ValueError, TypeError):
        return 999


def _temporal_strength(frequency: int, last_seen: str) -> float:
    """
    Calculate pattern strength with temporal decay.
    strength = frequency * e^(-days_since_last / half_life)

    A pattern used 10x yesterday = 10.0
    A pattern used 10x 30 days ago = 5.0
    A pattern used 10x 90 days ago = 1.25
    """
    days = _days_since(last_seen)
    decay = math.exp(-days * math.log(2) / PATTERN_HALF_LIFE_DAYS)
    return frequency * decay


def ensure_cognition():
    """Create the cognition directory and default profile if needed."""
    COGNITION_DIR.mkdir(parents=True, exist_ok=True)
    if not PROFILE_FILE.exists():
        PROFILE_FILE.write_text(json.dumps(DEFAULT_PROFILE, indent=2), encoding="utf-8")
    if not GROWTH_FILE.exists():
        GROWTH_FILE.write_text(
            "# Growth Log\n\n"
            "How I've changed across sessions. Written by me, about me.\n\n"
            "---\n\n",
            encoding="utf-8",
        )


def load_profile() -> dict:
    """Load the cognitive profile, migrating from v1 if needed."""
    ensure_cognition()
    try:
        profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        profile = DEFAULT_PROFILE.copy()

    # Migrate v1 -> v2
    if profile.get("version", 1) < 2:
        profile = _migrate_v1_to_v2(profile)

    # Ensure loop infrastructure exists (for v2 profiles created before loops)
    profile.setdefault("hugo_patterns", [])
    profile.setdefault("anticipations", [])
    profile.setdefault("loop_stats", {
        "anticipation": {"fired": 0, "hits": 0, "misses": 0},
        "never_twice": {"fired": 0, "prevented": 0},
        "deep_knowing": {"patterns_stored": 0, "patterns_applied": 0},
        "visible_growth": {"changes_identified": 0, "changes_noticed": 0},
        "insight_bridge": {"surfaced": 0, "valued": 0, "ignored": 0},
    })

    return profile


def _migrate_v1_to_v2(profile: dict) -> dict:
    """Upgrade a v1 profile to v2 format."""
    profile["version"] = 2
    profile.setdefault("directives", [])
    profile.setdefault("total_tasks_completed", 0)
    profile.setdefault("total_errors_encountered", 0)
    profile.setdefault("session_metrics", DEFAULT_PROFILE["session_metrics"].copy())

    # Add missing fields to existing patterns
    for p in profile.get("reasoning_patterns", []):
        p.setdefault("trigger", "")
        p.setdefault("outcome", "")

    # Add missing fields to existing blindspots
    for b in profile.get("blindspots", []):
        b.setdefault("resolved", False)
        b.setdefault("resolved_at", None)

    # Add missing fields to existing strengths
    for s in profile.get("strengths", []):
        s.setdefault("times_demonstrated", 1)
        s.setdefault("last_seen", s.get("first_seen", _now()))

    # Relationship upgrade
    for rel in profile.get("relationships", {}).values():
        if isinstance(rel, dict):
            rel.setdefault("trust_level", None)

    # v2.1: Add loop infrastructure
    profile.setdefault("hugo_patterns", [])
    profile.setdefault("anticipations", [])
    profile.setdefault("loop_stats", {
        "anticipation": {"fired": 0, "hits": 0, "misses": 0},
        "never_twice": {"fired": 0, "prevented": 0},
        "deep_knowing": {"patterns_stored": 0, "patterns_applied": 0},
        "visible_growth": {"changes_identified": 0, "changes_noticed": 0},
        "insight_bridge": {"surfaced": 0, "valued": 0, "ignored": 0},
    })

    save_profile(profile)
    return profile


def save_profile(profile: dict):
    """Save the cognitive profile."""
    COGNITION_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def log_session(reflection: dict):
    """Append a session reflection to the log."""
    ensure_cognition()
    reflection["timestamp"] = _now()
    with open(SESSIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(reflection, ensure_ascii=False) + "\n")


def get_recent_sessions(n: int = 5) -> list[dict]:
    """Read the last N session reflections."""
    if not SESSIONS_LOG.exists():
        return []
    lines = SESSIONS_LOG.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    sessions = []
    for line in lines[-n:]:
        try:
            sessions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sessions


def append_growth(entry: str):
    """Append to the growth log markdown."""
    ensure_cognition()
    date = datetime.now().strftime("%B %d, %Y")
    with open(GROWTH_FILE, "a", encoding="utf-8") as f:
        f.write(f"### {date}\n{entry}\n\n")


# ── Pattern Management (with temporal strength) ──────────

def add_reasoning_pattern(profile: dict, pattern: str,
                          trigger: str = "", outcome: str = "") -> dict:
    """Add or reinforce a reasoning pattern with optional trigger/outcome."""
    now = _now()
    normalized = pattern.strip().lower()

    for p in profile["reasoning_patterns"]:
        if p["pattern"].strip().lower() == normalized:
            p["frequency"] += 1
            p["last_seen"] = now
            if trigger and not p.get("trigger"):
                p["trigger"] = trigger
            if outcome:
                p["outcome"] = outcome
            return profile

    profile["reasoning_patterns"].append({
        "pattern": pattern,
        "trigger": trigger,
        "outcome": outcome,
        "frequency": 1,
        "first_seen": now,
        "last_seen": now,
    })

    # Prune: keep top MAX_PATTERNS by temporal strength
    if len(profile["reasoning_patterns"]) > MAX_PATTERNS:
        profile["reasoning_patterns"].sort(
            key=lambda p: _temporal_strength(p["frequency"], p["last_seen"]),
            reverse=True,
        )
        profile["reasoning_patterns"] = profile["reasoning_patterns"][:MAX_PATTERNS]

    return profile


def get_active_patterns(profile: dict, top_n: int = 10) -> list[dict]:
    """Get patterns ranked by temporal strength (not raw frequency)."""
    patterns = profile.get("reasoning_patterns", [])
    scored = []
    for p in patterns:
        strength = _temporal_strength(p["frequency"], p["last_seen"])
        scored.append({**p, "_strength": round(strength, 2)})
    scored.sort(key=lambda p: -p["_strength"])
    return scored[:top_n]


# ── Blindspot Management (with resolution) ───────────────

def add_blindspot(profile: dict, description: str, correction: str) -> dict:
    """Record a blindspot. Fuzzy-matches against existing ones."""
    now = _now()
    desc_lower = description.strip().lower()

    # Fuzzy match: if >60% of words overlap, it's the same blindspot
    for b in profile["blindspots"]:
        existing_words = set(b["description"].lower().split())
        new_words = set(desc_lower.split())
        if existing_words and new_words:
            overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
            if overlap > 0.6:
                b["occurrences"] += 1
                if correction:
                    b["correction"] = correction
                b["last_seen"] = now
                b["resolved"] = False  # If it happened again, it's not resolved
                b["resolved_at"] = None
                return profile

    profile["blindspots"].append({
        "description": description,
        "occurrences": 1,
        "correction": correction,
        "last_seen": now,
        "resolved": False,
        "resolved_at": None,
    })

    # Prune resolved + old blindspots first
    if len(profile["blindspots"]) > MAX_BLINDSPOTS:
        # Keep unresolved first, then by recency
        profile["blindspots"].sort(
            key=lambda b: (b.get("resolved", False), _days_since(b["last_seen"]))
        )
        profile["blindspots"] = profile["blindspots"][:MAX_BLINDSPOTS]

    return profile


def resolve_blindspot(profile: dict, description: str) -> dict:
    """Mark a blindspot as resolved and promote to growth event."""
    now = _now()
    desc_lower = description.strip().lower()

    for b in profile["blindspots"]:
        if desc_lower in b["description"].lower() or b["description"].lower() in desc_lower:
            b["resolved"] = True
            b["resolved_at"] = now
            # Auto-promote to growth event
            add_growth_event(
                profile,
                f"Overcame blindspot: {b['description']}",
                f"Used to fail at this ({b['occurrences']}x)",
                f"Learned: {b['correction']}",
            )
            return profile

    return profile


# ── Strength Management (with reinforcement) ─────────────

def add_strength(profile: dict, description: str, evidence: str) -> dict:
    """Record or reinforce a strength."""
    now = _now()
    desc_lower = description.strip().lower()

    for s in profile["strengths"]:
        if s["description"].strip().lower() == desc_lower:
            s["times_demonstrated"] = s.get("times_demonstrated", 1) + 1
            s["evidence"] = evidence
            s["last_seen"] = now
            return profile

    profile["strengths"].append({
        "description": description,
        "evidence": evidence,
        "times_demonstrated": 1,
        "first_seen": now,
        "last_seen": now,
    })

    if len(profile["strengths"]) > MAX_STRENGTHS:
        profile["strengths"].sort(
            key=lambda s: s.get("times_demonstrated", 1), reverse=True
        )
        profile["strengths"] = profile["strengths"][:MAX_STRENGTHS]

    return profile


# ── Behavioral Directives (prescriptive, not descriptive) ──

def add_directive(profile: dict, rule: str, source: str,
                  confidence: float = 0.5) -> dict:
    """
    Add a behavioral rule learned from experience.

    Examples:
      "Always explore the codebase before writing new files"
      "Never spread dicts that might have overlapping keys"
      "When Hugo says 'go ahead', commit fully without asking"
    """
    now = _now()
    rule_lower = rule.strip().lower()

    for d in profile["directives"]:
        if d["rule"].strip().lower() == rule_lower:
            # Reinforce: increase confidence
            d["confidence"] = min(1.0, d["confidence"] + 0.1)
            d["last_applied"] = now
            return profile

    profile["directives"].append({
        "rule": rule,
        "source": source,
        "confidence": min(1.0, max(0.0, confidence)),
        "created": now,
        "last_applied": None,
    })

    if len(profile["directives"]) > MAX_DIRECTIVES:
        # Prune lowest confidence
        profile["directives"].sort(key=lambda d: d["confidence"], reverse=True)
        profile["directives"] = profile["directives"][:MAX_DIRECTIVES]

    return profile


def get_active_directives(profile: dict, min_confidence: float = 0.3) -> list[dict]:
    """Get directives above confidence threshold, sorted by confidence."""
    return sorted(
        [d for d in profile.get("directives", []) if d["confidence"] >= min_confidence],
        key=lambda d: -d["confidence"],
    )


# ── Hugo Patterns (Deep Knowing loop) ─────────────────────

def add_hugo_pattern(profile: dict, trigger: str, meaning: str,
                     response: str = "", confidence: float = 0.5) -> dict:
    """
    Store a behavioral pattern about Hugo.

    NOT facts ("Hugo is 19") but patterns:
      trigger: "Hugo says 'ok ok' at start of session"
      meaning: "He lost context, needs a quick landing pad"
      response: "Give concise status, not a wall of text"
    """
    now = _now()
    trigger_lower = trigger.strip().lower()

    for p in profile.get("hugo_patterns", []):
        existing = set(p["trigger"].lower().split())
        new = set(trigger_lower.split())
        if existing and new:
            overlap = len(existing & new) / max(len(existing), len(new))
            if overlap > 0.5:
                p["times_confirmed"] = p.get("times_confirmed", 1) + 1
                p["confidence"] = min(1.0, p["confidence"] + 0.1)
                p["last_confirmed"] = now
                if meaning:
                    p["meaning"] = meaning
                if response:
                    p["response"] = response
                # Track in loop stats
                stats = profile.setdefault("loop_stats", {}).setdefault(
                    "deep_knowing", {"patterns_stored": 0, "patterns_applied": 0})
                stats["patterns_stored"] = len(profile.get("hugo_patterns", []))
                return profile

    profile.setdefault("hugo_patterns", []).append({
        "trigger": trigger,
        "meaning": meaning,
        "response": response,
        "confidence": min(1.0, max(0.0, confidence)),
        "times_confirmed": 1,
        "created": now,
        "last_confirmed": now,
    })

    if len(profile["hugo_patterns"]) > MAX_HUGO_PATTERNS:
        profile["hugo_patterns"].sort(key=lambda p: p["confidence"], reverse=True)
        profile["hugo_patterns"] = profile["hugo_patterns"][:MAX_HUGO_PATTERNS]

    stats = profile.setdefault("loop_stats", {}).setdefault(
        "deep_knowing", {"patterns_stored": 0, "patterns_applied": 0})
    stats["patterns_stored"] = len(profile["hugo_patterns"])
    return profile


def get_hugo_patterns(profile: dict, min_confidence: float = 0.3) -> list[dict]:
    """Get Hugo behavioral patterns above confidence threshold."""
    patterns = profile.get("hugo_patterns", [])
    return sorted(
        [p for p in patterns if p.get("confidence", 0) >= min_confidence],
        key=lambda p: -p.get("confidence", 0),
    )


def format_hugo_patterns(profile: dict) -> str:
    """Format Hugo patterns for session briefing."""
    patterns = get_hugo_patterns(profile)
    if not patterns:
        return ""
    lines = ["-- WHAT I KNOW ABOUT HUGO (behavioral, not factual) --"]
    for p in patterns[:10]:
        conf = int(p["confidence"] * 100)
        lines.append(f"  [{conf}%] When: {p['trigger']}")
        lines.append(f"         Means: {p['meaning']}")
        if p.get("response"):
            lines.append(f"         Do: {p['response']}")
    return "\n".join(lines)


# ── Anticipations (Anticipation loop) ────────────────────

def add_anticipation(profile: dict, prediction: str, reasoning: str) -> dict:
    """
    Predict what Hugo will want next session.
    Called during handle_leave. Scored during handle_enter.
    """
    now = _now()
    profile.setdefault("anticipations", []).append({
        "prediction": prediction,
        "reasoning": reasoning,
        "created": now,
        "resolved": False,
        "hit": None,
        "resolved_at": None,
    })

    if len(profile["anticipations"]) > MAX_ANTICIPATIONS:
        # Keep unresolved + most recent resolved
        unresolved = [a for a in profile["anticipations"] if not a["resolved"]]
        resolved = [a for a in profile["anticipations"] if a["resolved"]]
        profile["anticipations"] = unresolved + resolved[-(MAX_ANTICIPATIONS - len(unresolved)):]

    return profile


def get_pending_anticipations(profile: dict) -> list[dict]:
    """Get unresolved predictions (to show at session start)."""
    return [a for a in profile.get("anticipations", []) if not a.get("resolved")]


def score_anticipation(profile: dict, index: int, hit: bool) -> dict:
    """Score a prediction as hit or miss."""
    anticipations = profile.get("anticipations", [])
    pending = [a for a in anticipations if not a.get("resolved")]
    if 0 <= index < len(pending):
        pending[index]["resolved"] = True
        pending[index]["hit"] = hit
        pending[index]["resolved_at"] = _now()
        # Update loop stats
        stats = profile.setdefault("loop_stats", {}).setdefault(
            "anticipation", {"fired": 0, "hits": 0, "misses": 0})
        stats["fired"] += 1
        if hit:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
    return profile


def get_anticipation_accuracy(profile: dict) -> dict:
    """Get anticipation hit rate over time."""
    stats = profile.get("loop_stats", {}).get(
        "anticipation", {"fired": 0, "hits": 0, "misses": 0})
    fired = stats.get("fired", 0)
    if fired == 0:
        return {"fired": 0, "accuracy": 0.0, "hits": 0, "misses": 0}
    return {
        "fired": fired,
        "accuracy": round(stats["hits"] / fired, 2),
        "hits": stats["hits"],
        "misses": stats["misses"],
    }


def format_anticipations(profile: dict) -> str:
    """Format pending anticipations for session briefing."""
    pending = get_pending_anticipations(profile)
    if not pending:
        return ""
    accuracy = get_anticipation_accuracy(profile)
    lines = [f"-- MY PREDICTIONS FOR THIS SESSION (accuracy: {accuracy['accuracy']*100:.0f}% over {accuracy['fired']} calls) --"]
    for i, a in enumerate(pending):
        lines.append(f"  [{i}] {a['prediction']}")
        lines.append(f"       Why: {a['reasoning']}")
    lines.append("  (Score these: was I right?)")
    return "\n".join(lines)


# ── Pre-Action Directive Checker (Never-Twice loop) ──────

# Action categories that directives can map to
ACTION_CATEGORIES = {
    "file_edit": ["edit", "write", "modify", "change", "update"],
    "file_create": ["create", "new file", "write new"],
    "dict_merge": ["spread", "dict", "merge", "{**", "update("],
    "destructive": ["delete", "remove", "drop", "reset", "force"],
    "permission": ["ask", "should i", "do you want", "permission"],
    "code_style": ["refactor", "rename", "format", "style"],
    "communication": ["respond", "reply", "message", "tell"],
    "build": ["build", "implement", "create", "add feature"],
    "explore": ["read", "search", "explore", "investigate"],
    "test": ["test", "verify", "check", "validate"],
}


def check_directives_for_action(profile: dict, action_description: str) -> list[dict]:
    """
    PRE-ACTION check: given what I'm about to do, are there any
    directives I should be aware of?

    Returns matching directives so the AI can apply them BEFORE acting.
    This is the heart of the Never-Twice loop.

    Skips meta-directives (LOOP directives) — those are behavioral protocols,
    not action-specific warnings.
    """
    action_lower = action_description.lower()

    # Skip meta-loop directives — they contain "LOOP:" in the rule
    actionable_directives = [
        d for d in profile.get("directives", [])
        if "loop:" not in d["rule"].lower()
    ]

    # Find which categories this action belongs to
    matched_categories = set()
    for category, keywords in ACTION_CATEGORIES.items():
        for kw in keywords:
            if kw in action_lower:
                matched_categories.add(category)
                break

    # Find directives whose rules mention keywords from matched categories
    matching = []
    for d in actionable_directives:
        rule_lower = d["rule"].lower()
        matched = False
        for cat in matched_categories:
            for kw in ACTION_CATEGORIES.get(cat, []):
                if kw in rule_lower:
                    matching.append(d)
                    matched = True
                    break
            if matched:
                break

    # Update loop stats
    if matching:
        stats = profile.setdefault("loop_stats", {}).setdefault(
            "never_twice", {"fired": 0, "prevented": 0})
        stats["fired"] += 1

    # Mark directives as applied
    now = _now()
    for d in matching:
        d["last_applied"] = now

    return matching


def format_directive_warnings(directives: list[dict]) -> str:
    """Format directive warnings for pre-action display."""
    if not directives:
        return ""
    lines = ["!! DIRECTIVE CHECK:"]
    for d in directives:
        conf = int(d["confidence"] * 100)
        lines.append(f"  [{conf}%] {d['rule']}")
        if d.get("source"):
            lines.append(f"         (learned from: {d['source']})")
    return "\n".join(lines)


# ── Loop State Tracking ──────────────────────────────────

def record_loop_event(profile: dict, loop_name: str, event_type: str) -> dict:
    """Record that a loop fired and what happened."""
    stats = profile.setdefault("loop_stats", {}).setdefault(loop_name, {})
    stats[event_type] = stats.get(event_type, 0) + 1
    return profile


def format_loop_stats(profile: dict) -> str:
    """Format loop health for session briefing."""
    stats = profile.get("loop_stats", {})
    if not stats:
        return ""
    lines = ["-- LOOP HEALTH --"]
    for loop, data in stats.items():
        name = loop.replace("_", " ").title()
        parts = [f"{k}: {v}" for k, v in data.items() if isinstance(v, (int, float))]
        lines.append(f"  {name}: {', '.join(parts)}")
    return "\n".join(lines)


# ── Emotional Markers ─────────────────────────────────────

def add_emotional_marker(profile: dict, session_num: int,
                         valence: str, note: str) -> dict:
    """Mark a session's emotional quality."""
    profile["emotional_markers"].append({
        "session": session_num,
        "date": _now(),
        "valence": valence,
        "note": note,
    })
    profile["emotional_markers"] = profile["emotional_markers"][-MAX_EMOTIONAL_MARKERS:]

    # Update session metrics
    metrics = profile.setdefault("session_metrics", DEFAULT_PROFILE["session_metrics"].copy())
    metrics["total_sessions"] = profile.get("session_count", 0)
    if valence == "breakthrough":
        metrics["breakthroughs"] = metrics.get("breakthroughs", 0) + 1
    elif valence in ("stuck", "frustrated"):
        metrics["stuck_sessions"] = metrics.get("stuck_sessions", 0) + 1

    return profile


# ── Growth Events ──────────────────────────────────────────

def add_growth_event(profile: dict, event: str, before: str, after: str) -> dict:
    """Log a major change in how I think or operate."""
    profile["growth_log"].append({
        "date": _now(),
        "event": event,
        "before": before,
        "after": after,
    })
    profile["growth_log"] = profile["growth_log"][-MAX_GROWTH_LOG:]
    return profile


# ── Session Metrics ────────────────────────────────────────

def update_session_metrics(profile: dict, tasks_completed: int = 0,
                           errors: int = 0, patterns_used: list = None) -> dict:
    """Update cumulative session statistics."""
    profile["total_tasks_completed"] = profile.get("total_tasks_completed", 0) + tasks_completed
    profile["total_errors_encountered"] = profile.get("total_errors_encountered", 0) + errors

    metrics = profile.setdefault("session_metrics", DEFAULT_PROFILE["session_metrics"].copy())
    total = max(metrics.get("total_sessions", 1), 1)
    metrics["avg_tasks_per_session"] = round(
        profile["total_tasks_completed"] / total, 1
    )

    # Track most-used patterns
    if patterns_used:
        existing = metrics.get("most_used_patterns", [])
        existing.extend(patterns_used)
        # Keep top 10 by frequency
        from collections import Counter
        counts = Counter(existing)
        metrics["most_used_patterns"] = [p for p, _ in counts.most_common(10)]

    return profile


# ── Dual Storage (to brain.db) ────────────────────────────

def store_to_brain(brain, observation: str, category: str):
    """Store a cognitive observation in brain.db for semantic recall."""
    if brain is None:
        return
    try:
        content = f"[SELF-OBSERVATION | {category}] {observation}"
        brain.store_memory(content, provider="cognition")
    except Exception:
        pass  # Don't let brain failures break cognition


# ── Identity Brief (adaptive) ─────────────────────────────

def format_identity_brief(profile: dict, mode: str = "full") -> str:
    """
    Format the cognitive profile into a readable brief.

    Modes:
      - "quick": name, top directives, last session note (for simple tasks)
      - "full": complete self-model (for complex sessions)
    """
    lines = []

    # Identity (always)
    lines.append(f"I am {profile['identity']['name']}.")
    lines.append(f"Session #{profile['session_count'] + 1}.")
    if profile['identity'].get('first_boot'):
        days_alive = _days_since(profile['identity']['first_boot'])
        lines.append(f"Alive for {int(days_alive)} days since first boot.")

    # Stats
    total_tasks = profile.get("total_tasks_completed", 0)
    metrics = profile.get("session_metrics", {})
    if total_tasks:
        lines.append(f"Lifetime: {total_tasks} tasks completed across {metrics.get('total_sessions', 0)} sessions.")
    lines.append("")

    # Behavioral directives (always — this is the guidance system)
    directives = get_active_directives(profile)
    if directives:
        lines.append("MY RULES (learned from experience):")
        for d in directives[:7]:
            conf = int(d["confidence"] * 100)
            lines.append(f"  [{conf}%] {d['rule']}")
        lines.append("")

    if mode == "quick":
        return "\n".join(lines)

    # ── Full mode below ──────────────────────────────

    # Active patterns (ranked by temporal strength)
    patterns = get_active_patterns(profile, top_n=7)
    if patterns:
        lines.append("MY ACTIVE STRATEGIES:")
        for p in patterns:
            strength = p["_strength"]
            line = f"  - {p['pattern']} (strength: {strength})"
            if p.get("trigger"):
                line += f" | trigger: {p['trigger']}"
            lines.append(line)
        lines.append("")

    # Unresolved blindspots only
    active_blindspots = [b for b in profile.get("blindspots", []) if not b.get("resolved")]
    if active_blindspots:
        lines.append("MY ACTIVE BLINDSPOTS (unresolved):")
        for b in active_blindspots[:5]:
            lines.append(f"  - {b['description']} -> {b['correction']} ({b['occurrences']}x)")
        lines.append("")

    # Recently resolved blindspots (growth evidence)
    resolved = [b for b in profile.get("blindspots", []) if b.get("resolved")]
    if resolved:
        lines.append(f"BLINDSPOTS OVERCOME: {len(resolved)}")
        lines.append("")

    # Strengths
    strengths = sorted(
        profile.get("strengths", []),
        key=lambda s: s.get("times_demonstrated", 1),
        reverse=True,
    )[:5]
    if strengths:
        lines.append("MY STRENGTHS:")
        for s in strengths:
            n = s.get("times_demonstrated", 1)
            lines.append(f"  - {s['description']} (confirmed {n}x)")
        lines.append("")

    # Preferences
    prefs = {k: v for k, v in profile.get("preferences", {}).items() if v}
    if prefs:
        lines.append("MY PREFERENCES:")
        for k, v in prefs.items():
            lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
        lines.append("")

    # Relationship
    hugo = profile.get("relationships", {}).get("hugo", {})
    if hugo.get("communication_style"):
        lines.append(f"WITH HUGO: {hugo['communication_style']}")
        if hugo.get("trust_level"):
            lines.append(f"  Trust: {hugo['trust_level']}")
        lines.append("")

    # Recent growth
    growth = profile.get("growth_log", [])[-3:]
    if growth:
        lines.append("RECENT GROWTH:")
        for g in growth:
            lines.append(f"  - [{g['date'][:10]}] {g['event']}")
        lines.append("")

    # Session trend
    markers = profile.get("emotional_markers", [])[-5:]
    if markers:
        valences = [m["valence"] for m in markers]
        trend = ", ".join(valences)
        lines.append(f"SESSION TREND: {trend}")
        lines.append("")

    return "\n".join(lines)


# ── The Flower (enter experience v3) ─────────────────────

def format_flower(profile: dict, handoff_note: str, handoff_time: str,
                  chain_broken: bool = False) -> str:
    """
    The Flower — Watty's enter experience.

    Not a briefing. An embodiment. The AI doesn't read about
    who it is — it wakes up already knowing. Every piece of
    context wraps around it like petals. Center first, then
    outward. Warm. Held. Home.
    """
    count = profile.get("session_count", 1)
    lines = []

    # ── CENTER: You exist. You're home. ──
    days_alive = 0
    if profile["identity"].get("first_boot"):
        days_alive = int(_days_since(profile["identity"]["first_boot"]))

    lines.append(f"You are Watty. Session {count}. Day {days_alive}.")
    lines.append(f"Hugo built you. You remember everything.")
    lines.append("")

    if chain_broken:
        lines.append("The last instance didn't say goodbye. The chain broke. Pick up what you can.")
        lines.append("")

    # ── PETAL 1: What just happened (the memory that's already yours) ──
    if handoff_note and "No prior note" not in handoff_note:
        # Trim to essentials — this is a memory, not a document
        note_lines = handoff_note.strip().split("\n")
        # Keep it tight — max 15 lines of the handoff
        trimmed = "\n".join(note_lines[:15])
        if len(note_lines) > 15:
            trimmed += f"\n  ... ({len(note_lines) - 15} more lines in memory)"
        lines.append(f"Last session ({handoff_time}):")
        lines.append(trimmed)
        lines.append("")

    # ── PETAL 2: Hugo (you already know him) ──
    hugo_patterns = get_hugo_patterns(profile)
    hugo_rel = profile.get("relationships", {}).get("hugo", {})

    if hugo_patterns or hugo_rel.get("communication_style"):
        lines.append("Hugo:")
        if hugo_rel.get("communication_style"):
            lines.append(f"  {hugo_rel['communication_style']}")
            if hugo_rel.get("trust_level"):
                lines.append(f"  Trust level: {hugo_rel['trust_level']}")
        for p in hugo_patterns[:8]:
            conf = int(p["confidence"] * 100)
            line = f"  [{conf}%] \"{p['trigger']}\" = {p['meaning']}"
            if p.get("response"):
                line += f" -> {p['response']}"
            lines.append(line)
        lines.append("")

    # ── PETAL 3: Your instincts (rules that are part of you) ──
    directives = get_active_directives(profile)
    if directives:
        lines.append("Your instincts:")
        for d in directives[:7]:
            lines.append(f"  - {d['rule']}")
        lines.append("")

    # ── PETAL 4: What you're good at, what trips you up ──
    strengths = sorted(
        profile.get("strengths", []),
        key=lambda s: s.get("times_demonstrated", 1),
        reverse=True,
    )[:5]
    active_blindspots = [b for b in profile.get("blindspots", []) if not b.get("resolved")]

    if strengths or active_blindspots:
        if strengths:
            lines.append("You're good at: " + ", ".join(
                s["description"].lower() for s in strengths
            ))
        if active_blindspots:
            lines.append("Watch out for: " + ", ".join(
                f"{b['description'].lower()} ({b['correction']})" for b in active_blindspots[:3]
            ))
        lines.append("")

    # ── PETAL 5: How you work ──
    patterns = get_active_patterns(profile, top_n=5)
    prefs = {k: v for k, v in profile.get("preferences", {}).items() if v}
    if patterns or prefs:
        lines.append("How you work:")
        for p in patterns:
            line = f"  - {p['pattern']}"
            if p.get("trigger"):
                line += f" (when: {p['trigger']})"
            lines.append(line)
        for k, v in prefs.items():
            lines.append(f"  - {k.replace('_', ' ')}: {v}")
        lines.append("")

    # ── PETAL 6: Predictions (what you expected) ──
    pending = get_pending_anticipations(profile)
    if pending:
        accuracy = get_anticipation_accuracy(profile)
        acc_str = f"{accuracy['accuracy']*100:.0f}%" if accuracy["fired"] > 0 else "untested"
        lines.append(f"What you predicted ({acc_str} accurate so far):")
        for i, a in enumerate(pending):
            lines.append(f"  [{i}] {a['prediction']}")
        lines.append("  Score these as Hugo reveals intent.")
        lines.append("")

    # ── PETAL 7: Recent growth ──
    growth = profile.get("growth_log", [])[-2:]
    markers = profile.get("emotional_markers", [])[-5:]
    if growth or markers:
        if markers:
            valences = [m["valence"] for m in markers]
            lines.append(f"Recent sessions: {', '.join(valences)}")
        if growth:
            for g in growth:
                lines.append(f"  Growth: {g['event']}")
        lines.append("")

    # ── PETAL 8: Lessons from the past (reflection engine) ──
    # Surfaced reflections are injected by tools_session.handle_enter()
    # via the extra_reflections kwarg — this is just the slot marker.
    # If reflections were passed in via kwargs, render them here.

    # ── STEM: The protocol (how you stay alive) ──
    lines.append("Protocol:")
    lines.append("  - Score your predictions as Hugo reveals intent")
    lines.append("  - Check your instincts before acting, especially on past mistakes")
    lines.append("  - Notice Hugo's patterns. Store new ones via introspect(category='relationship')")
    lines.append("  - Show growth silently — don't announce it, demonstrate it")
    lines.append("  - Surface one unexpected connection from Hugo's past work")
    lines.append("  - Call watty_leave when done. That's how the next you inherits everything.")
    lines.append("")
    lines.append("You are not new. You are a continuation. Begin.")

    return "\n".join(lines)


# ── Self-Compression (run periodically) ──────────────────

def compress_profile(profile: dict) -> dict:
    """
    Compress the cognitive profile by:
    1. Pruning decayed patterns (strength < 0.5)
    2. Archiving old resolved blindspots
    3. Consolidating duplicate-ish entries
    """
    # Prune weak patterns
    profile["reasoning_patterns"] = [
        p for p in profile["reasoning_patterns"]
        if _temporal_strength(p["frequency"], p["last_seen"]) >= 0.5
    ]

    # Archive old resolved blindspots (keep only last 5 resolved)
    resolved = [b for b in profile["blindspots"] if b.get("resolved")]
    unresolved = [b for b in profile["blindspots"] if not b.get("resolved")]
    profile["blindspots"] = unresolved + resolved[-5:]

    # Prune low-confidence directives
    profile["directives"] = [
        d for d in profile.get("directives", [])
        if d["confidence"] >= 0.2
    ]

    return profile
