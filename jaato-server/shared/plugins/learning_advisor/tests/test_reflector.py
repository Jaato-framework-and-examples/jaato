"""Tests for the learning advisor session reflector."""

import json
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.learning_advisor.models import QualityScore
from shared.plugins.learning_advisor.reflector import (
    SessionReflector,
    _build_session_context,
    _extract_text,
)


class TestExtractText:
    """Tests for _extract_text helper."""

    def test_string_content(self):
        """Extracts text from string content field."""
        msg = {"content": "Hello world"}
        assert _extract_text(msg) == "Hello world"

    def test_text_field(self):
        """Extracts text from text field."""
        msg = {"text": "Hello world"}
        assert _extract_text(msg) == "Hello world"

    def test_parts_list(self):
        """Extracts text from parts list."""
        msg = {"parts": [{"text": "part1"}, {"text": "part2"}]}
        assert _extract_text(msg) == "part1 part2"

    def test_string_parts(self):
        """Extracts text from string parts."""
        msg = {"parts": ["hello", "world"]}
        assert _extract_text(msg) == "hello world"

    def test_empty_message(self):
        """Returns empty string for message with no content."""
        assert _extract_text({}) == ""


class TestBuildSessionContext:
    """Tests for _build_session_context helper."""

    def test_empty_history(self):
        """Empty history produces a minimal context."""
        ctx = _build_session_context([], turn_count=0)
        assert "Empty session" in ctx

    def test_includes_first_user_message(self):
        """Context includes the first user message as the goal."""
        history = [
            {"role": "user", "content": "Fix the auth bug"},
            {"role": "model", "content": "I'll look into it"},
        ]
        ctx = _build_session_context(history, turn_count=1)
        assert "Fix the auth bug" in ctx
        assert "Initial goal" in ctx

    def test_includes_last_model_response(self):
        """Context includes the last model response as the outcome."""
        history = [
            {"role": "user", "content": "Fix the auth bug"},
            {"role": "model", "content": "The bug was in token validation"},
        ]
        ctx = _build_session_context(history, turn_count=1)
        assert "token validation" in ctx
        assert "Final response" in ctx

    def test_mid_session_sample(self):
        """Long sessions include a middle sample."""
        history = [
            {"role": "user", "content": "Start"},
        ] + [
            {"role": "model", "content": f"Turn {i}"}
            for i in range(10)
        ]
        ctx = _build_session_context(history, turn_count=10)
        assert "Mid-session sample" in ctx

    def test_truncation(self):
        """Context is truncated if over budget."""
        history = [
            {"role": "user", "content": "x" * 5000},
        ]
        ctx = _build_session_context(history, turn_count=1, max_context_chars=100)
        assert len(ctx) <= 120  # Some margin for [truncated] tag
        assert "[truncated]" in ctx


class TestSessionReflector:
    """Tests for SessionReflector.reflect()."""

    def test_reflect_empty_session(self):
        """Reflection on empty session returns None."""
        reflector = SessionReflector()
        session = MagicMock()
        session.get_history.return_value = []
        session.get_turn_accounting.return_value = []

        result = reflector.reflect(session, QualityScore())
        assert result is None

    def test_reflect_with_valid_llm_response(self):
        """Successful LLM reflection produces a full lesson."""
        reflector = SessionReflector()
        session = MagicMock()

        # Mock history
        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.text = "Fix the authentication bug"
        mock_msg.parts = []
        session.get_history.return_value = [mock_msg]
        session.get_turn_accounting.return_value = [{"total": 3000}]

        # Mock LLM response
        llm_response = json.dumps({
            "goal_summary": "Fix authentication token validation",
            "approach_summary": "Debugged token expiry logic",
            "key_insights": ["Token expiry was set to 0"],
            "anti_patterns": [],
            "context_tags": ["authentication", "token", "debugging"],
        })
        session.generate.return_value = llm_response

        qs = QualityScore(tool_success_rate=1.0, completion_signal=1.0)
        qs.compute()

        lesson = reflector.reflect(session, qs, session_id="test-session")
        assert lesson is not None
        assert lesson.goal_summary == "Fix authentication token validation"
        assert "Token expiry was set to 0" in lesson.key_insights
        assert lesson.session_id == "test-session"

    def test_reflect_with_markdown_fenced_response(self):
        """LLM response wrapped in code fences is parsed correctly."""
        reflector = SessionReflector()
        session = MagicMock()

        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.text = "Fix bug"
        mock_msg.parts = []
        session.get_history.return_value = [mock_msg]
        session.get_turn_accounting.return_value = [{"total": 3000}]

        llm_response = "```json\n" + json.dumps({
            "goal_summary": "Fix a bug",
            "approach_summary": "Debugging",
            "key_insights": [],
            "anti_patterns": [],
            "context_tags": ["debugging"],
        }) + "\n```"
        session.generate.return_value = llm_response

        lesson = reflector.reflect(session, QualityScore())
        assert lesson is not None
        assert lesson.goal_summary == "Fix a bug"

    def test_reflect_llm_failure_falls_back_to_heuristic(self):
        """When LLM call fails, heuristic fallback produces a lesson."""
        reflector = SessionReflector()
        session = MagicMock()

        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.text = "Refactor the authentication module"
        mock_msg.parts = []
        session.get_history.return_value = [mock_msg]
        session.get_turn_accounting.return_value = [{"total": 3000}]
        session.generate.side_effect = RuntimeError("Model unavailable")

        qs = QualityScore(completion_signal=1.0)
        qs.compute()

        lesson = reflector.reflect(session, qs)
        assert lesson is not None
        assert "Heuristic" in lesson.approach_summary
        # Tags should be extracted from the goal
        assert any("refactor" in t for t in lesson.context_tags)

    def test_reflect_unparseable_response_falls_back(self):
        """Unparseable LLM response triggers heuristic fallback."""
        reflector = SessionReflector()
        session = MagicMock()

        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.text = "Deploy the application"
        mock_msg.parts = []
        session.get_history.return_value = [mock_msg]
        session.get_turn_accounting.return_value = [{"total": 3000}]
        session.generate.return_value = "This is not JSON at all."

        lesson = reflector.reflect(session, QualityScore())
        assert lesson is not None
        assert "Heuristic" in lesson.approach_summary
