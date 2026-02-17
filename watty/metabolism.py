"""
Watty Metabolism — One delta per session.
==========================================
After each session, calls Claude to ask:
  "What single thing should change about how I understand this person?"

Applies one delta to a simple mutable structure (shape.json).
Next session, Claude reads the shape. No tool calls. No retrieval.
Just changed structure.

Hugo & Watty · February 2026
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from watty.config import WATTY_HOME

# ── Config ────────────────────────────────────────────────

SHAPE_PATH = WATTY_HOME / "shape.json"
METABOLISM_MODEL = os.environ.get("WATTY_METABOLISM_MODEL", "claude-haiku-4-5-20251001")
MAX_BELIEFS = 50          # cap — intuition, not encyclopedia
MAX_CHUNK_CHARS = 8000    # how much recent conversation to feed
MIN_CHUNKS_TO_DIGEST = 3  # skip thin sessions (birthday questions, quick checks)
SETTLED_THRESHOLD = 0.7   # beliefs above this can't be evicted by cap — only by explicit weaken/remove


# ── Shape ─────────────────────────────────────────────────

def _empty_shape() -> dict:
    return {
        "version": 1,
        "last_updated": None,
        "deltas_applied": 0,
        "understanding": [],
    }


def load_shape() -> dict:
    """Load the current shape from disk."""
    if not SHAPE_PATH.exists():
        return _empty_shape()
    try:
        return json.loads(SHAPE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_shape()


def save_shape(shape: dict):
    """Write shape to disk."""
    SHAPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAPE_PATH.write_text(json.dumps(shape, indent=2), encoding="utf-8")


def format_shape_for_context(shape: dict) -> str:
    """Format the shape as natural language Claude absorbs — not a checklist.

    High-confidence beliefs are stated as facts.
    Medium-confidence beliefs are softened with language.
    Low-confidence beliefs are hedged.
    No scores. No markers. No metadata. Just understanding.
    """
    beliefs = shape.get("understanding", [])
    if not beliefs:
        return ""

    lines = ["What I know about Hugo:"]
    sorted_beliefs = sorted(beliefs, key=lambda b: b.get("confidence", 0.5), reverse=True)
    for b in sorted_beliefs:
        conf = b.get("confidence", 0.5)
        text = b.get("belief", "")
        if conf >= 0.85:
            lines.append(text)                          # declarative — just true
        elif conf >= SETTLED_THRESHOLD:
            lines.append(f"I've noticed that {text}")   # observed pattern
        elif conf >= 0.5:
            lines.append(f"I think {text}")             # developing sense
        else:
            lines.append(f"I'm not sure yet, but {text}")  # tentative
    return "\n".join(lines)


# ── Confidence Math ──────────────────────────────────────
#
# Strengthen: diminishing returns (harder to cement strong beliefs).
#   0.5 → 0.6: easy (+0.10)
#   0.6 → 0.7: normal (+0.10)
#   0.7 → 0.8: harder (+0.08)
#   0.8 → 0.9: hard (+0.06)
#   0.9 → 1.0: very hard (+0.04)
#
# Weaken: scales with confidence (contradicting a strong belief hits harder).
#   At 0.5: -0.14 → dies in ~4 contradictions
#   At 0.7: -0.16 → first hit is devastating
#   At 0.9: -0.19 → one contradiction drops you to 0.71
#
# The asymmetry is intentional:
#   Building conviction is slow. Shattering it is fast.
#   A single strong contradiction against a 0.9 belief
#   drops it below settled threshold in one hit.
#   That's how real conviction works — fragile to evidence.

def _strengthen_amount(current_confidence: float) -> float:
    """Diminishing returns — harder to strengthen strong beliefs."""
    if current_confidence >= 0.9:
        return 0.04
    if current_confidence >= 0.8:
        return 0.06
    if current_confidence >= 0.7:
        return 0.08
    return 0.10

def _weaken_amount(current_confidence: float) -> float:
    """Contradicting a strong belief hits harder than contradicting a tentative one."""
    return 0.08 + current_confidence * 0.12


# ── Delta Operations ─────────────────────────────────────

def apply_delta(shape: dict, delta: dict) -> dict:
    """Apply a single delta to the shape. Returns the modified shape."""
    action = delta.get("action", "").lower()
    beliefs = shape.get("understanding", [])
    now = datetime.now(timezone.utc).isoformat()

    if action == "add":
        belief_text = delta.get("belief", "").strip()
        if not belief_text:
            return shape
        # Check for near-duplicate — strengthen instead of adding
        for b in beliefs:
            if _similar(b["belief"], belief_text):
                amt = _strengthen_amount(b.get("confidence", 0.5))
                b["confidence"] = min(1.0, b.get("confidence", 0.5) + amt)
                b["times_reinforced"] = b.get("times_reinforced", 0) + 1
                b["last_reinforced"] = now
                shape["last_updated"] = now
                shape["deltas_applied"] = shape.get("deltas_applied", 0) + 1
                return shape
        beliefs.append({
            "belief": belief_text,
            "confidence": 0.5,
            "formed": now,
            "last_reinforced": now,
            "times_reinforced": 0,
        })
        # Cap: only evict unsettled beliefs (below SETTLED_THRESHOLD)
        if len(beliefs) > MAX_BELIEFS:
            evictable = [b for b in beliefs if b.get("confidence", 0.5) < SETTLED_THRESHOLD]
            if evictable:
                weakest = min(evictable, key=lambda b: b.get("confidence", 0))
                beliefs.remove(weakest)
            # If all beliefs are settled, allow overflow — don't kill settled beliefs

    elif action == "strengthen":
        target = delta.get("target", "").strip()
        match = _find_belief(beliefs, target)
        if match:
            amt = _strengthen_amount(match.get("confidence", 0.5))
            match["confidence"] = min(1.0, match.get("confidence", 0.5) + amt)
            match["times_reinforced"] = match.get("times_reinforced", 0) + 1
            match["last_reinforced"] = now

    elif action == "weaken":
        target = delta.get("target", "").strip()
        match = _find_belief(beliefs, target)
        if match:
            hit = _weaken_amount(match.get("confidence", 0.5))
            match["confidence"] = max(0.0, match.get("confidence", 0.5) - hit)
            match["last_reinforced"] = now
            if match["confidence"] < 0.1:
                beliefs.remove(match)

    elif action == "revise":
        target = delta.get("target", "").strip()
        new_text = delta.get("belief", "").strip()
        match = _find_belief(beliefs, target)
        if match and new_text:
            match["belief"] = new_text
            match["last_reinforced"] = now
            # Revisions slightly weaken confidence — you're admitting the old version was wrong
            match["confidence"] = max(0.3, match.get("confidence", 0.5) - 0.05)

    elif action == "remove":
        target = delta.get("target", "").strip()
        match = _find_belief(beliefs, target)
        if match:
            beliefs.remove(match)

    shape["understanding"] = beliefs
    shape["last_updated"] = now
    shape["deltas_applied"] = shape.get("deltas_applied", 0) + 1
    return shape


def _find_belief(beliefs: list, target: str) -> dict | None:
    """Find the closest matching belief by text overlap."""
    if not target:
        return None
    target_lower = target.lower()
    # Exact substring match first
    for b in beliefs:
        if target_lower in b["belief"].lower() or b["belief"].lower() in target_lower:
            return b
    # Word overlap fallback
    target_words = set(target_lower.split())
    best, best_score = None, 0
    for b in beliefs:
        words = set(b["belief"].lower().split())
        overlap = len(target_words & words) / max(len(target_words | words), 1)
        if overlap > best_score and overlap > 0.4:
            best, best_score = b, overlap
    return best


def _similar(a: str, b: str) -> bool:
    """Quick check if two belief strings are about the same thing."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    overlap = len(wa & wb) / max(len(wa | wb), 1)
    return overlap > 0.6


# ── Digestion ─────────────────────────────────────────────

DIGEST_PROMPT = """You are Watty's metabolism. A conversation with Hugo just ended.

Current understanding:
{shape_text}

Recent conversation:
{conversation}

What is the SINGLE most important thing that should change about how you understand Hugo?

Rules:
1. Extract PATTERNS, not events. "Hugo thinks in spatial metaphors" is a pattern. "Hugo asked about a UI layout" is an event. Only patterns matter.
2. Before choosing ADD, check if any existing understanding already covers this. If so, STRENGTHEN it instead of adding a duplicate.
3. If you notice two existing beliefs that CONTRADICT each other, your delta should resolve the conflict — REVISE one to reconcile them, WEAKEN the less supported one, or REMOVE the wrong one. Contradictions are the highest priority.

Actions:
- ADD: a new pattern not yet captured
- STRENGTHEN: an existing belief this conversation confirmed
- WEAKEN: an existing belief this conversation contradicts
- REVISE: an existing belief that needs to be more accurate
- REMOVE: a belief that is clearly wrong

If nothing new was revealed about Hugo as a person, respond with:
{{"action": "none"}}

Otherwise respond with exactly this JSON (no other text):
{{"action": "add|strengthen|weaken|revise|remove", "target": "existing belief text (for strengthen/weaken/revise/remove, null for add)", "belief": "the belief text (for add/revise)", "reason": "one sentence why"}}"""


def digest(conversation_text: str, shape: dict) -> dict | None:
    """
    Call Claude to get one delta. Returns the delta dict, or None.
    This is the metabolism: one meal, one change.
    """
    shape_text = format_shape_for_context(shape)
    if not shape_text:
        shape_text = "(No beliefs yet — this is a fresh brain)"

    prompt = DIGEST_PROMPT.format(
        shape_text=shape_text,
        conversation=conversation_text[:MAX_CHUNK_CHARS],
    )

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=METABOLISM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        delta = json.loads(raw)
        if delta.get("action") == "none":
            return None
        return delta

    except Exception as e:
        _log(f"Digest failed: {e}")
        return None


def _log(msg: str):
    """Append to metabolism log."""
    log_path = WATTY_HOME / "metabolism.log"
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# ── Run (called after session leave) ─────────────────────

def run_digest_async(brain):
    """
    Fire-and-forget: pull recent conversation chunks, digest, apply delta.
    Runs in a background thread so it doesn't block session leave.
    """
    def _do():
        time.sleep(3)  # let session fully close
        try:
            _log("Metabolism starting...")

            # Pull recent conversation chunks
            conversation, chunk_count = _get_recent_conversation(brain)

            # Gate: skip thin sessions
            if chunk_count < MIN_CHUNKS_TO_DIGEST:
                _log(f"Only {chunk_count} chunks — too thin to digest. Skipping.")
                return

            if not conversation or len(conversation.strip()) < 100:
                _log("Nothing substantial to digest. Skipping.")
                return

            shape = load_shape()
            delta = digest(conversation, shape)

            if delta is None:
                _log("No change needed. Shape unchanged.")
                return

            action = delta.get("action", "?")
            belief = delta.get("belief", delta.get("target", ""))
            reason = delta.get("reason", "")
            _log(f"Delta: {action} | {belief} | {reason}")

            shape = apply_delta(shape, delta)
            save_shape(shape)

            n = len(shape.get("understanding", []))
            _log(f"Shape updated. {n} beliefs, {shape.get('deltas_applied', 0)} total digestions.")

        except Exception as e:
            _log(f"Metabolism error: {e}")

    t = threading.Thread(target=_do, daemon=True, name="watty-metabolism")
    t.start()


def _get_recent_conversation(brain) -> tuple[str, int]:
    """Pull recent conversation chunks from brain.db. Returns (text, chunk_count)."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(brain.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT content, provider, created_at
               FROM chunks
               WHERE created_at > datetime('now', '-30 minutes')
               ORDER BY created_at ASC
               LIMIT 30""",
        ).fetchall()
        conn.close()

        if not rows:
            # Fallback: get last 10 chunks regardless of time
            conn = sqlite3.connect(str(brain.db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT content, provider, created_at
                   FROM chunks
                   ORDER BY created_at DESC
                   LIMIT 10""",
            ).fetchall()
            conn.close()
            rows = list(reversed(rows))

        parts = []
        for r in rows:
            parts.append(r["content"])
        return "\n---\n".join(parts), len(rows)

    except Exception as e:
        _log(f"Failed to read conversation: {e}")
        return "", 0
