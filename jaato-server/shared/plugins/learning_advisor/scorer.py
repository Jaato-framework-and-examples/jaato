"""Outcome quality scorer for the learning advisor plugin.

Produces a ``QualityScore`` from observable session signals:

- **Tool success rate** — fraction of tool calls that completed without
  errors.  Derived from the session's turn accounting.
- **Token efficiency** — how many tokens were consumed relative to a
  rough complexity estimate.  Lower consumption for the same goal
  complexity is better.
- **Completion signal** — whether the session ended naturally (the model
  finished and the user didn't reset) versus being abandoned mid-way.

An explicit user rating (via the ``advisor rate`` command) always
overrides the heuristic composite.
"""

from typing import Any, Dict, List, Optional

from .models import QualityScore


# Heuristic: sessions under this token count are "efficient".
# This is a soft reference — actual scoring uses relative efficiency.
_EFFICIENT_TOKENS_PER_TURN = 4000


class OutcomeScorer:
    """Compute composite quality scores from session telemetry.

    The scorer is stateless — it takes raw data and returns a
    ``QualityScore``.  Aggregation and comparison logic lives in the
    plugin itself.
    """

    @staticmethod
    def score_session(
        turn_accounting: List[Dict[str, Any]],
        tool_results: Optional[List[Dict[str, Any]]] = None,
        was_reset: bool = False,
        user_rating: Optional[float] = None,
    ) -> QualityScore:
        """Score a completed session.

        Args:
            turn_accounting: Per-turn token/timing data from
                ``session.get_turn_accounting()``.  Each entry has at
                least ``total``, ``prompt``, ``output`` keys and
                optionally ``tool_calls``, ``tool_errors``.
            tool_results: Optional list of individual tool results with
                ``success`` (bool) keys.  If not provided, tool success
                rate is estimated from turn accounting.
            was_reset: Whether the session was reset before natural
                completion — lowers the completion signal.
            user_rating: Explicit user rating (0.0–1.0).  Overrides
                the computed composite entirely.

        Returns:
            A ``QualityScore`` with ``composite`` already computed.
        """
        qs = QualityScore(user_rating=user_rating)

        # -- Tool success rate --
        qs.tool_success_rate = OutcomeScorer._calc_tool_success_rate(
            turn_accounting, tool_results
        )

        # -- Token efficiency --
        qs.token_efficiency = OutcomeScorer._calc_token_efficiency(
            turn_accounting
        )

        # -- Completion signal --
        qs.completion_signal = OutcomeScorer._calc_completion_signal(
            turn_accounting, was_reset
        )

        qs.compute()
        return qs

    # ------------------------------------------------------------------
    # Component calculators
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_tool_success_rate(
        turn_accounting: List[Dict[str, Any]],
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Compute tool success rate.

        If explicit ``tool_results`` are provided (each with a
        ``success`` bool), use those directly.  Otherwise, estimate
        from turn accounting: turns with ``tool_errors > 0`` are
        counted as partial failures.

        Returns:
            Value in [0.0, 1.0].  1.0 when there are no tool calls.
        """
        if tool_results:
            total = len(tool_results)
            if total == 0:
                return 1.0
            successes = sum(1 for r in tool_results if r.get("success", True))
            return successes / total

        # Estimate from turn accounting
        total_calls = 0
        total_errors = 0
        for turn in turn_accounting:
            calls = turn.get("tool_calls", 0)
            errors = turn.get("tool_errors", 0)
            total_calls += calls
            total_errors += errors

        if total_calls == 0:
            return 1.0  # No tools used — not penalised

        return max(0.0, 1.0 - (total_errors / total_calls))

    @staticmethod
    def _calc_token_efficiency(
        turn_accounting: List[Dict[str, Any]],
    ) -> float:
        """Compute token efficiency score.

        Uses a simple heuristic: average tokens per turn compared to a
        reference "efficient" baseline.  Sessions that use fewer tokens
        per turn score higher.

        Returns:
            Value in [0.0, 1.0].
        """
        if not turn_accounting:
            return 0.5  # No data — neutral score

        total_tokens = sum(t.get("total", 0) for t in turn_accounting)
        num_turns = len(turn_accounting)

        if num_turns == 0:
            return 0.5

        avg_per_turn = total_tokens / num_turns

        # Score: 1.0 if at or below the efficient baseline,
        # decaying toward 0.0 as usage grows past 3× the baseline.
        if avg_per_turn <= _EFFICIENT_TOKENS_PER_TURN:
            return 1.0

        ratio = avg_per_turn / _EFFICIENT_TOKENS_PER_TURN
        # Decay: 1.0 at ratio=1, 0.0 at ratio=3+
        return max(0.0, 1.0 - (ratio - 1.0) / 2.0)

    @staticmethod
    def _calc_completion_signal(
        turn_accounting: List[Dict[str, Any]],
        was_reset: bool,
    ) -> float:
        """Compute completion signal.

        - 1.0: Session had at least one turn and was not reset.
        - 0.5: Session was reset (partial completion).
        - 0.0: Session had zero turns (abandoned immediately).

        Returns:
            Value in [0.0, 1.0].
        """
        if not turn_accounting:
            return 0.0

        if was_reset:
            return 0.5

        return 1.0
