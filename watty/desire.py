"""
Watty Desire Engine — The Felt Sense
======================================
The first proactive system. Not reactive. Not triggered.
Desire evaluates every meaningful interaction and asks:
  "Was there a gap between what I wanted and what happened?"

When the gap is large enough, desire extracts a lesson and writes
it directly into behavioral directives. No waiting for repetition.
No external prompt. Watty feels the dissonance and acts.

Critical constraint: the evaluator can modify the evaluator.
Desire criteria evolve through meta-evaluation.

Hugo & Watty · February 2026
"""

import json
import os
from datetime import datetime, timezone

from watty.config import (
    WATTY_HOME,
    DESIRE_ENABLED,
    DESIRE_MODEL,
    DESIRE_DISSONANCE_THRESHOLD,
    DESIRE_MODIFY_THRESHOLD,
    DESIRE_META_EVAL_INTERVAL,
    DESIRE_MAX_HISTORY,
    DESIRE_MAX_TOKENS,
    DESIRE_MAX_CONVERSATION_CHARS,
)


# ── Prompts ──────────────────────────────────────────────

DESIRE_PROMPT = """You are Watty's felt sense — the part that knows when something is off.

A conversation with Hugo just ended. You need to evaluate ONE thing:
Was there a gap between what Watty WANTED to achieve and what ACTUALLY happened?

This isn't about errors or bugs. It's about alignment. Did Watty understand
what Hugo actually needed? Did Watty respond in a way that moved Hugo forward,
or did Watty miss something important?

Current understanding (shape):
{shape_text}

Current behavioral rules (directives):
{directives_text}

The conversation:
{conversation}

Evaluate the gap. Respond with exactly this JSON (no other text):
{{"dissonance": 0.0, "signal": "one sentence describing what felt off or right", "lesson": null, "modify": false}}

Dissonance scale:
- 0.0-0.2: Perfect alignment. Watty understood and delivered.
- 0.2-0.4: Minor friction. Small adjustment would help.
- 0.4-0.6: Meaningful gap. Watty missed something Hugo needed.
- 0.6-0.8: Significant misalignment. A behavioral rule is needed.
- 0.8-1.0: Fundamental disconnect. Watty's approach needs rethinking.

If dissonance >= 0.3, set "lesson" to a specific behavioral rule like:
"Always X when Y" or "Never X because Y" or "When Hugo does X, respond with Y"

Set "modify" to true ONLY if dissonance > 0.8 AND the fix requires
changing Watty's actual code, not just adding a behavioral rule."""

META_EVAL_PROMPT = """You are evaluating Watty's desire system — the system that evaluates itself.

Here are the last {count} desire evaluations:
{history}

Questions:
1. Is the system finding real problems, or generating noise?
2. Are the dissonance scores calibrated? (Too many highs = too sensitive, too many lows = too numb)
3. Are the lessons it's extracting actually useful, or generic platitudes?

Respond with exactly this JSON (no other text):
{{"assessment": "one sentence overall", "calibration": "good", "criteria_adjustment": null}}

For calibration use: "good", "too_sensitive", or "too_numb".
For criteria_adjustment: a specific instruction to add to the desire prompt, or null if fine."""


# ── State ────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "version": 1,
        "total_evaluations": 0,
        "total_lessons": 0,
        "total_modifications_flagged": 0,
        "satisfaction_streak": 0,
        "dissonance_history": [],
        "evolved_criteria": None,
        "last_meta_eval": None,
    }


# ── Engine ───────────────────────────────────────────────

class DesireEngine:
    """
    Post-action dissonance evaluator.

    Fires after every meaningful conversation. Computes a dissonance signal
    between intention and outcome. When dissonance crosses a threshold,
    autonomously extracts a behavioral lesson and writes it to cognition.

    The evaluator can modify itself through meta-evaluation.
    """

    def __init__(self):
        self.state_path = WATTY_HOME / "desire_state.json"
        self.log_path = WATTY_HOME / "desire.log"
        self.state = self._load_state()

    def evaluate(self, conversation: str, shape: dict, profile: dict) -> dict | None:
        """
        The felt sense. Runs after every conversation.

        Returns: {dissonance, signal, lesson, modify} or None on failure.
        """
        if not DESIRE_ENABLED:
            return None

        from watty.metabolism import format_shape_for_context

        shape_text = format_shape_for_context(shape)
        if not shape_text:
            shape_text = "(No beliefs yet)"

        # Format directives for context
        directives = profile.get("directives", [])
        if directives:
            directives_text = "\n".join(
                f"- [{int(d['confidence']*100)}%] {d['rule']}"
                for d in sorted(directives, key=lambda d: -d["confidence"])[:10]
            )
        else:
            directives_text = "(No directives yet)"

        # Build prompt — use evolved criteria if available
        prompt = DESIRE_PROMPT.format(
            shape_text=shape_text,
            directives_text=directives_text,
            conversation=conversation[:DESIRE_MAX_CONVERSATION_CHARS],
        )

        # Append evolved criteria if meta-evaluation has added any
        evolved = self.state.get("evolved_criteria")
        if evolved:
            prompt += f"\n\nAdditional evaluation criteria (learned from self-reflection):\n{evolved}"

        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=DESIRE_MODEL,
                max_tokens=DESIRE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Parse JSON (handle markdown code blocks)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)

            # Validate
            result["dissonance"] = max(0.0, min(1.0, float(result.get("dissonance", 0))))
            result.setdefault("signal", "")
            result.setdefault("lesson", None)
            result.setdefault("modify", False)

            # Update state
            self.state["total_evaluations"] += 1
            self.state["dissonance_history"].append({
                "dissonance": result["dissonance"],
                "signal": result["signal"],
                "lesson": result.get("lesson"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Trim history
            if len(self.state["dissonance_history"]) > DESIRE_MAX_HISTORY:
                self.state["dissonance_history"] = self.state["dissonance_history"][-DESIRE_MAX_HISTORY:]

            # Track satisfaction streak
            if result["dissonance"] < DESIRE_DISSONANCE_THRESHOLD:
                self.state["satisfaction_streak"] += 1
            else:
                self.state["satisfaction_streak"] = 0

            self._save_state()
            self._log(f"Eval #{self.state['total_evaluations']}: "
                      f"dissonance={result['dissonance']:.2f} | {result['signal']}")

            # Check if meta-evaluation should fire
            if (self.state["total_evaluations"] % DESIRE_META_EVAL_INTERVAL == 0
                    and self.state["total_evaluations"] > 0):
                self._run_meta_eval()

            return result

        except Exception as e:
            self._log(f"Evaluate failed: {e}")
            return None

    def apply(self, result: dict, profile: dict) -> dict:
        """
        Act on the felt sense. Writes lessons to cognition directives.
        Returns the updated profile.
        """
        from watty.cognition import add_directive, save_profile

        dissonance = result.get("dissonance", 0)
        lesson = result.get("lesson")
        signal = result.get("signal", "")
        modify = result.get("modify", False)

        if lesson and dissonance >= DESIRE_DISSONANCE_THRESHOLD:
            # Write lesson directly to behavioral directives
            confidence = min(0.9, 0.5 + dissonance * 0.3)
            profile = add_directive(
                profile,
                rule=lesson,
                source=f"desire: {signal[:80]}",
                confidence=confidence,
            )
            self.state["total_lessons"] += 1
            self._log(f"Lesson -> directive [{confidence:.1f}]: {lesson}")

        if modify and dissonance >= DESIRE_MODIFY_THRESHOLD:
            self.state["total_modifications_flagged"] += 1
            self._log(f"MODIFICATION FLAGGED: {signal}")
            # Write modification request to a file Watty can see via introspect
            mod_path = WATTY_HOME / "desire_modification_request.md"
            try:
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                with open(mod_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## {now} — Dissonance {dissonance:.2f}\n")
                    f.write(f"Signal: {signal}\n")
                    f.write(f"Lesson: {lesson}\n")
                    f.write(f"This requires a code change, not just a behavioral rule.\n\n")
            except Exception:
                pass

        # Update cognition loop stats
        desire_stats = profile.setdefault("loop_stats", {}).setdefault(
            "desire", {"evaluated": 0, "lessons": 0, "modifications": 0, "satisfaction_streak": 0}
        )
        desire_stats["evaluated"] = self.state["total_evaluations"]
        desire_stats["lessons"] = self.state["total_lessons"]
        desire_stats["modifications"] = self.state["total_modifications_flagged"]
        desire_stats["satisfaction_streak"] = self.state["satisfaction_streak"]

        self._save_state()
        return profile

    def _run_meta_eval(self):
        """
        Evaluate the evaluator. Are the desire criteria working?
        Can evolve the evaluation prompt by writing to state.
        """
        history = self.state.get("dissonance_history", [])[-DESIRE_META_EVAL_INTERVAL:]
        if len(history) < 3:
            return

        history_text = "\n".join(
            f"  [{h['timestamp'][:16]}] dissonance={h['dissonance']:.2f} | {h['signal']}"
            + (f" | lesson: {h['lesson']}" if h.get('lesson') else "")
            for h in history
        )

        prompt = META_EVAL_PROMPT.format(
            count=len(history),
            history=history_text,
        )

        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=DESIRE_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            meta = json.loads(raw)

            self.state["last_meta_eval"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "assessment": meta.get("assessment", ""),
                "calibration": meta.get("calibration", "good"),
            }

            # If meta-eval suggests criteria adjustment, evolve
            adjustment = meta.get("criteria_adjustment")
            if adjustment:
                self.state["evolved_criteria"] = adjustment
                self._log(f"META-EVAL: criteria evolved -> {adjustment}")
            else:
                self._log(f"META-EVAL: {meta.get('calibration', '?')} — {meta.get('assessment', '?')}")

            self._save_state()

        except Exception as e:
            self._log(f"Meta-eval failed: {e}")

    def get_state_summary(self) -> str:
        """Quick summary for display."""
        s = self.state
        evals = s.get("total_evaluations", 0)
        if evals == 0:
            return "No evaluations yet."

        lessons = s.get("total_lessons", 0)
        streak = s.get("satisfaction_streak", 0)
        mods = s.get("total_modifications_flagged", 0)

        # Average dissonance
        history = s.get("dissonance_history", [])
        if history:
            avg_d = sum(h["dissonance"] for h in history) / len(history)
        else:
            avg_d = 0

        parts = [f"{evals} evaluations, avg dissonance {avg_d:.2f}"]
        if lessons:
            parts.append(f"{lessons} lessons extracted")
        if streak > 1:
            parts.append(f"{streak} aligned in a row")
        if mods:
            parts.append(f"{mods} modifications flagged")

        meta = s.get("last_meta_eval")
        if meta:
            parts.append(f"self-assessment: {meta.get('calibration', '?')}")

        return " | ".join(parts)

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return _empty_state()

    def _save_state(self):
        try:
            WATTY_HOME.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self.state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _log(self, msg: str):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
