"""Tests for learning advisor outcome quality scorer."""

import pytest

from shared.plugins.learning_advisor.scorer import OutcomeScorer


class TestOutcomeScorer:
    """Tests for OutcomeScorer.score_session()."""

    def test_perfect_session(self):
        """Session with no errors and efficient tokens scores high."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000,
             "tool_calls": 5, "tool_errors": 0},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        assert qs.tool_success_rate == pytest.approx(1.0)
        assert qs.token_efficiency == pytest.approx(1.0)
        assert qs.completion_signal == pytest.approx(1.0)
        assert qs.composite == pytest.approx(1.0)

    def test_empty_session(self):
        """Empty session (no turns) scores low."""
        qs = OutcomeScorer.score_session([])
        assert qs.completion_signal == pytest.approx(0.0)
        assert qs.token_efficiency == pytest.approx(0.5)  # Neutral

    def test_reset_session(self):
        """Reset session gets partial completion signal."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000},
        ]
        qs = OutcomeScorer.score_session(turn_accounting, was_reset=True)
        assert qs.completion_signal == pytest.approx(0.5)

    def test_tool_errors_reduce_score(self):
        """Tool errors reduce the tool success rate."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000,
             "tool_calls": 10, "tool_errors": 3},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        assert qs.tool_success_rate == pytest.approx(0.7)

    def test_all_tools_failed(self):
        """All tool calls failing gives zero success rate."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000,
             "tool_calls": 5, "tool_errors": 5},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        assert qs.tool_success_rate == pytest.approx(0.0)

    def test_no_tool_calls(self):
        """Sessions without tool calls get 1.0 success rate."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        assert qs.tool_success_rate == pytest.approx(1.0)

    def test_explicit_tool_results(self):
        """Explicit tool_results override turn-accounting estimation."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000,
             "tool_calls": 10, "tool_errors": 0},  # Would be 1.0
        ]
        tool_results = [
            {"success": True},
            {"success": False},
            {"success": True},
            {"success": False},
        ]
        qs = OutcomeScorer.score_session(turn_accounting, tool_results=tool_results)
        assert qs.tool_success_rate == pytest.approx(0.5)

    def test_high_token_usage_reduces_efficiency(self):
        """High token usage per turn reduces efficiency score."""
        turn_accounting = [
            {"total": 12000, "prompt": 8000, "output": 4000},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        # 12000/4000 = 3.0 ratio → efficiency = max(0, 1 - (3-1)/2) = 0.0
        assert qs.token_efficiency == pytest.approx(0.0)

    def test_moderate_token_usage(self):
        """Moderate token usage gives partial efficiency score."""
        turn_accounting = [
            {"total": 6000, "prompt": 4000, "output": 2000},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        # 6000/4000 = 1.5 ratio → efficiency = 1 - (1.5-1)/2 = 0.75
        assert qs.token_efficiency == pytest.approx(0.75)

    def test_user_rating_overrides_all(self):
        """User rating overrides the computed composite."""
        turn_accounting = [
            {"total": 3000, "prompt": 2000, "output": 1000,
             "tool_calls": 10, "tool_errors": 10},  # Bad tools
        ]
        qs = OutcomeScorer.score_session(
            turn_accounting, user_rating=0.95
        )
        assert qs.composite == pytest.approx(0.95)

    def test_multi_turn_averaging(self):
        """Token efficiency averages across turns."""
        turn_accounting = [
            {"total": 2000, "prompt": 1500, "output": 500},
            {"total": 6000, "prompt": 4000, "output": 2000},
        ]
        qs = OutcomeScorer.score_session(turn_accounting)
        # Average: (2000+6000)/2 = 4000 per turn → exactly at baseline → 1.0
        assert qs.token_efficiency == pytest.approx(1.0)
