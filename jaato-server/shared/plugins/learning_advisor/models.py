"""Data models for the learning advisor plugin.

Lessons capture what worked (and what didn't) during sessions with similar
goals.  They are the currency of the advisor's institutional memory —
quality-filtered observations that propagate successful approaches to future
sessions.

Lifecycle:
    1. A session completes → the reflector distills a ``Lesson`` from the
       conversation history + outcome signals.
    2. The matcher compares the new lesson's goal signature against the
       existing pool.
    3. If a matching cluster exists and the new lesson scored higher, the
       old lesson is archived and the new one promoted.  Otherwise the new
       lesson is appended (up to ``MAX_LESSONS_PER_CLUSTER``).
    4. When a future session starts with a similar goal, the top-ranked
       lesson(s) are injected via prompt enrichment as non-prescriptive
       guidance.

Quality scoring is composite: tool failure rate, token efficiency, and
an optional explicit user signal always take precedence.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Lesson status constants.
STATUS_ACTIVE = "active"
"""Lesson is eligible for injection into future sessions."""

STATUS_ARCHIVED = "archived"
"""Lesson was superseded by a higher-quality lesson for the same goal cluster."""

STATUS_DISMISSED = "dismissed"
"""User explicitly marked this lesson as unhelpful."""

VALID_STATUSES = frozenset({STATUS_ACTIVE, STATUS_ARCHIVED, STATUS_DISMISSED})

# Maximum lessons retained per goal cluster before eviction.
MAX_LESSONS_PER_CLUSTER = 3


@dataclass
class QualityScore:
    """Composite quality score for a session outcome.

    Each component is normalised to [0.0, 1.0].  The ``composite`` score is
    a weighted average of the components, overridden entirely when
    ``user_rating`` is set.

    Attributes:
        tool_success_rate: Fraction of tool calls that succeeded
            (no errors / total calls).  1.0 means every tool call worked
            on the first try.
        token_efficiency: Inverse of token usage relative to the goal
            complexity.  Higher is better — the session achieved its goal
            with fewer tokens.
        completion_signal: Whether the session completed its goal without
            a reset or abandonment.  1.0 = completed, 0.5 = partial,
            0.0 = abandoned/reset.
        user_rating: Explicit user override (0.0–1.0).  When set, this
            value *replaces* the composite score entirely.
        composite: Final blended score.  Computed automatically unless
            ``user_rating`` is set, in which case ``composite`` equals
            ``user_rating``.
    """

    tool_success_rate: float = 0.0
    token_efficiency: float = 0.0
    completion_signal: float = 0.0
    user_rating: Optional[float] = None
    composite: float = 0.0

    def compute(self) -> float:
        """Compute and store the composite quality score.

        Weights:
            - tool_success_rate:  40%  (efficiency matters most)
            - token_efficiency:   30%  (cost-aware)
            - completion_signal:  30%  (did it actually finish?)

        If ``user_rating`` is set, it overrides the computed score.

        Returns:
            The composite score in [0.0, 1.0].
        """
        if self.user_rating is not None:
            self.composite = max(0.0, min(1.0, self.user_rating))
        else:
            self.composite = (
                0.4 * self.tool_success_rate
                + 0.3 * self.token_efficiency
                + 0.3 * self.completion_signal
            )
        return self.composite


@dataclass
class Lesson:
    """A distilled lesson learned from a completed session.

    Lessons are the primary data structure stored by the learning advisor.
    Each captures:
    - *What* the session tried to do (``goal_summary``, ``context_tags``).
    - *How* it went about it (``approach_summary``).
    - *What worked* (``key_insights``).
    - *What didn't* (``anti_patterns``).
    - *How well* it turned out (``quality``).

    Provenance fields (``session_id``, ``created_at``) link back to the
    originating session for auditability.

    Attributes:
        id: Unique identifier (UUID4).
        goal_summary: Normalised, concise description of what the session
            tried to accomplish.  Used as the primary input for goal
            similarity matching.
        approach_summary: High-level strategy the agent chose — tool
            sequences, decomposition approach, ordering decisions.
        key_insights: Extracted patterns that contributed to success.
            e.g. "Splitting migration into per-table steps avoided timeout."
        anti_patterns: Approaches that were tried and failed before the
            working approach was found.  Helps future sessions avoid
            known dead-ends.
        context_tags: Normalised keywords for fast pre-filtering before
            more expensive similarity checks.
            e.g. ["refactoring", "database", "migration", "large-codebase"]
        quality: Composite quality score for this lesson's originating
            session.
        status: Lifecycle state — ``active``, ``archived``, or ``dismissed``.
        session_id: ID of the session that produced this lesson.
        created_at: ISO-8601 timestamp when the lesson was created.
        used_count: Number of times this lesson has been injected into a
            future session via prompt enrichment.
        last_used_at: ISO-8601 timestamp of the most recent injection.
        scope: ``project`` for workspace-specific lessons, ``universal``
            for cross-project patterns.
    """

    id: str
    goal_summary: str
    approach_summary: str
    key_insights: List[str]
    anti_patterns: List[str]
    context_tags: List[str]
    quality: QualityScore
    status: str = STATUS_ACTIVE
    session_id: Optional[str] = None
    created_at: str = ""
    used_count: int = 0
    last_used_at: Optional[str] = None
    scope: str = "project"


@dataclass
class LessonMetadata:
    """Lightweight metadata for in-memory indexing and prompt enrichment.

    Mirrors the ``MemoryMetadata`` pattern from the memory plugin — stores
    just enough information for the indexer to match and rank without
    loading full lesson content from disk.

    Attributes:
        id: Unique identifier matching the full ``Lesson``.
        goal_summary: For display in enrichment hints.
        context_tags: For keyword-based matching.
        quality_composite: Pre-computed composite score for ranking.
        status: Current lifecycle state.
        created_at: ISO-8601 timestamp.
        scope: ``project`` or ``universal``.
    """

    id: str
    goal_summary: str
    context_tags: List[str]
    quality_composite: float
    status: str = STATUS_ACTIVE
    created_at: str = ""
    scope: str = "project"
