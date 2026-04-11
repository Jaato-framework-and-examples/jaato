"""Session reflector for the learning advisor plugin.

At the end of a session (or on-demand via user command), the reflector
analyses the conversation history and distills a structured ``Lesson``.

The reflection uses the session's own model via a lightweight one-shot
``generate()`` call — no tool calling, no history pollution.  The prompt
asks the model to introspect on:

1. What was the session's goal?
2. What approach was taken?
3. What worked (key insights)?
4. What didn't work (anti-patterns)?
5. What context tags describe this work?

The model's response is parsed from a JSON block, with a plain-text
fallback if JSON parsing fails.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import Lesson, QualityScore

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession


# The reflection prompt — asks the model to summarise the session as a
# structured lesson.  The model receives this as a one-shot generation
# (no conversation context) so it must include enough session context
# inline.
_REFLECTION_PROMPT = """\
You are analysing a completed work session to extract lessons learned.

Below is a summary of the session — the first user message (the goal),
the total number of turns, and a selection of key moments.

<session_context>
{session_context}
</session_context>

Produce a JSON object with exactly these keys:

{{
  "goal_summary": "One-sentence summary of what the session tried to accomplish",
  "approach_summary": "Brief description of the strategy/approach taken (2-3 sentences max)",
  "key_insights": ["Insight 1", "Insight 2", ...],
  "anti_patterns": ["Anti-pattern 1", ...],
  "context_tags": ["tag1", "tag2", ...]
}}

Rules:
- goal_summary: Normalised, generic enough to match similar future tasks.
  BAD: "Fix bug in auth.py line 42".  GOOD: "Fix authentication token validation bug".
- approach_summary: Focus on the *strategy*, not line-by-line details.
- key_insights: What worked well.  Be specific enough to be actionable.
  If nothing notable worked, return an empty list.
- anti_patterns: Approaches tried that failed or wasted time.
  If none, return an empty list.
- context_tags: 3-8 lowercase keywords for matching similar future goals.
  Use technical terms, framework names, problem categories.
- Do NOT include secrets, credentials, or user-specific paths.
- Output ONLY the JSON object, no surrounding text.
"""


def _build_session_context(
    history: List[Dict[str, Any]],
    turn_count: int,
    max_context_chars: int = 4000,
) -> str:
    """Build a compact session context string for the reflection prompt.

    Includes:
    - The first user message (the goal).
    - The last assistant message (the outcome).
    - Turn count and basic stats.
    - A middle sample if the session is long.

    Args:
        history: Serialised message history (list of dicts with
            ``role`` and ``content``/``text`` keys).
        turn_count: Number of completed turns.
        max_context_chars: Character budget for the context block.

    Returns:
        A formatted string suitable for embedding in the reflection prompt.
    """
    if not history:
        return f"Empty session ({turn_count} turns)."

    parts = [f"Total turns: {turn_count}"]

    # First user message (the goal)
    for msg in history:
        role = msg.get("role", "")
        text = _extract_text(msg)
        if role == "user" and text:
            goal_text = text[:800]
            parts.append(f"\n## Initial goal\n{goal_text}")
            break

    # Last assistant message (the outcome)
    for msg in reversed(history):
        role = msg.get("role", "")
        text = _extract_text(msg)
        if role == "model" and text:
            outcome_text = text[:800]
            parts.append(f"\n## Final response\n{outcome_text}")
            break

    # If long session, sample a middle message
    if len(history) > 6:
        mid = len(history) // 2
        mid_msg = history[mid]
        mid_text = _extract_text(mid_msg)
        if mid_text:
            parts.append(f"\n## Mid-session sample (turn ~{mid // 2})\n{mid_text[:400]}")

    result = "\n".join(parts)

    # Truncate if over budget
    if len(result) > max_context_chars:
        result = result[:max_context_chars] + "\n[truncated]"

    return result


def _extract_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message dict.

    Handles both ``content`` (string) and ``parts`` (list) formats.
    """
    # Simple string content
    content = msg.get("content")
    if isinstance(content, str):
        return content

    # Text field (some serialisations)
    text = msg.get("text")
    if isinstance(text, str):
        return text

    # Parts list
    parts = msg.get("parts") or msg.get("content")
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                t = part.get("text", "")
                if t:
                    texts.append(t)
        return " ".join(texts)

    return ""


class SessionReflector:
    """Distills a structured ``Lesson`` from a completed session.

    The reflector uses the session's model for a single one-shot call
    (``session.generate()``) that doesn't pollute conversation history.

    Usage::

        reflector = SessionReflector()
        lesson = reflector.reflect(session, quality_score)

    If the model call fails or returns unparseable output, the reflector
    falls back to a heuristic lesson built from the first user message.
    """

    def reflect(
        self,
        session: "JaatoSession",
        quality: QualityScore,
        session_id: Optional[str] = None,
    ) -> Optional[Lesson]:
        """Reflect on a session and produce a Lesson.

        Args:
            session: The completed JaatoSession with history intact.
            quality: Pre-computed quality score for this session.
            session_id: Optional session identifier for provenance.

        Returns:
            A ``Lesson`` if reflection succeeded, or None if the session
            has insufficient data (e.g. zero turns).
        """
        history_messages = session.get_history()
        turn_accounting = session.get_turn_accounting()

        if not history_messages or not turn_accounting:
            return None

        # Serialise history to dicts for context extraction
        serialised = []
        for msg in history_messages:
            serialised.append({
                "role": getattr(msg, "role", "unknown"),
                "content": getattr(msg, "text", "") or str(getattr(msg, "parts", "")),
            })

        context = _build_session_context(
            serialised,
            turn_count=len(turn_accounting),
        )

        prompt = _REFLECTION_PROMPT.format(session_context=context)

        # Try LLM-based reflection
        try:
            raw_response = session.generate(prompt)
            lesson_data = self._parse_response(raw_response)
            if lesson_data:
                return self._build_lesson(lesson_data, quality, session_id)
        except Exception:
            pass  # Fall through to heuristic

        # Heuristic fallback: build a minimal lesson from the first message
        return self._heuristic_lesson(serialised, quality, session_id)

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the model's JSON response.

        Tries to extract a JSON object from the response, handling
        cases where the model wraps it in markdown code fences.

        Returns:
            Parsed dict or None if parsing fails.
        """
        text = response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "goal_summary" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON within the response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "goal_summary" in data:
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def _build_lesson(
        self,
        data: Dict[str, Any],
        quality: QualityScore,
        session_id: Optional[str],
    ) -> Lesson:
        """Build a ``Lesson`` from parsed reflection data."""
        return Lesson(
            id=str(uuid.uuid4()),
            goal_summary=data.get("goal_summary", "Unknown goal"),
            approach_summary=data.get("approach_summary", ""),
            key_insights=data.get("key_insights", []),
            anti_patterns=data.get("anti_patterns", []),
            context_tags=data.get("context_tags", []),
            quality=quality,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _heuristic_lesson(
        self,
        serialised_history: List[Dict[str, Any]],
        quality: QualityScore,
        session_id: Optional[str],
    ) -> Optional[Lesson]:
        """Build a minimal lesson from the first user message.

        Used as a fallback when the LLM reflection call fails.
        Extracts a goal summary from the first user message and
        generates basic context tags via keyword extraction.
        """
        # Find first user message
        goal_text = ""
        for msg in serialised_history:
            if msg.get("role") == "user":
                goal_text = msg.get("content", "")[:200]
                break

        if not goal_text:
            return None

        # Simple keyword extraction for tags
        from .matcher import GoalMatcher
        tags = GoalMatcher.extract_keywords(goal_text)[:8]

        return Lesson(
            id=str(uuid.uuid4()),
            goal_summary=goal_text[:200],
            approach_summary="(Heuristic — LLM reflection unavailable)",
            key_insights=[],
            anti_patterns=[],
            context_tags=tags,
            quality=quality,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
