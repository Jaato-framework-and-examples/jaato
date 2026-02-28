"""Tests for SessionHistory - canonical conversation history owned by the session.

Tests cover:
1. SessionHistory class behavior (append, replace, clear, dirty tracking)
2. Integration with JaatoSession (get_history returns from SessionHistory,
   reset_session updates history, _history.replace/clear set history)
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from jaato_sdk.plugins.model_provider.types import Message, Part, Role

from ..session_history import SessionHistory
from ..jaato_session import JaatoSession


# ==================== SessionHistory Unit Tests ====================


class TestSessionHistoryInit:
    """Tests for SessionHistory initialization."""

    def test_empty_on_creation(self):
        """New SessionHistory has no messages and is not dirty."""
        sh = SessionHistory()
        assert len(sh) == 0
        assert sh.messages == []
        assert not sh.dirty
        assert not bool(sh)

    def test_repr(self):
        """__repr__ shows count and dirty flag."""
        sh = SessionHistory()
        assert "messages=0" in repr(sh)
        assert "dirty=False" in repr(sh)


class TestSessionHistoryAppend:
    """Tests for SessionHistory.append()."""

    def test_append_adds_message(self):
        """append() adds a message to the list."""
        sh = SessionHistory()
        msg = Message(role=Role.USER, parts=[Part.from_text("hello")])
        sh.append(msg)

        assert len(sh) == 1
        assert sh.messages[0].role == Role.USER

    def test_append_sets_dirty(self):
        """append() sets the dirty flag."""
        sh = SessionHistory()
        msg = Message(role=Role.USER, parts=[Part.from_text("hello")])
        sh.append(msg)

        assert sh.dirty

    def test_append_multiple(self):
        """Multiple append calls accumulate messages."""
        sh = SessionHistory()
        for i in range(3):
            sh.append(Message(role=Role.USER, parts=[Part.from_text(f"msg{i}")]))

        assert len(sh) == 3
        assert bool(sh)


class TestSessionHistoryReplace:
    """Tests for SessionHistory.replace()."""

    def test_replace_sets_messages(self):
        """replace() replaces all messages."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("old")]))

        new_msgs = [
            Message(role=Role.USER, parts=[Part.from_text("new1")]),
            Message(role=Role.MODEL, parts=[Part.from_text("new2")]),
        ]
        sh.replace(new_msgs)

        assert len(sh) == 2
        assert sh.messages[0].parts[0].text == "new1"
        assert sh.messages[1].parts[0].text == "new2"

    def test_replace_sets_dirty(self):
        """replace() sets the dirty flag."""
        sh = SessionHistory()
        sh.replace([Message(role=Role.USER, parts=[Part.from_text("x")])])

        assert sh.dirty

    def test_replace_makes_shallow_copy(self):
        """replace() copies the list to prevent aliasing."""
        sh = SessionHistory()
        original = [Message(role=Role.USER, parts=[Part.from_text("x")])]
        sh.replace(original)

        # Mutating original should not affect SessionHistory
        original.append(Message(role=Role.MODEL, parts=[Part.from_text("y")]))
        assert len(sh) == 1


class TestSessionHistoryClear:
    """Tests for SessionHistory.clear()."""

    def test_clear_empties_messages(self):
        """clear() removes all messages."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("hello")]))
        sh.clear()

        assert len(sh) == 0
        assert sh.messages == []

    def test_clear_resets_dirty(self):
        """clear() resets the dirty flag."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("hello")]))
        assert sh.dirty

        sh.clear()
        assert not sh.dirty


class TestSessionHistoryMessages:
    """Tests for SessionHistory.messages property."""

    def test_messages_returns_copy(self):
        """messages property returns a shallow copy."""
        sh = SessionHistory()
        msg = Message(role=Role.USER, parts=[Part.from_text("hello")])
        sh.append(msg)

        msgs = sh.messages
        msgs.append(Message(role=Role.MODEL, parts=[Part.from_text("world")]))

        # Mutating returned list should not affect SessionHistory
        assert len(sh) == 1


class TestSessionHistoryPopLast:
    """Tests for SessionHistory.pop_last()."""

    def test_pop_last_returns_and_removes(self):
        """pop_last() returns the last message and removes it."""
        sh = SessionHistory()
        msg1 = Message(role=Role.USER, parts=[Part.from_text("first")])
        msg2 = Message(role=Role.MODEL, parts=[Part.from_text("second")])
        sh.append(msg1)
        sh.append(msg2)

        popped = sh.pop_last()
        assert popped.parts[0].text == "second"
        assert len(sh) == 1

    def test_pop_last_empty(self):
        """pop_last() returns None for empty history."""
        sh = SessionHistory()
        assert sh.pop_last() is None


class TestSessionHistoryMessagesRef:
    """Tests for SessionHistory.messages_ref property."""

    def test_messages_ref_returns_same_list(self):
        """messages_ref returns the internal list (not a copy)."""
        sh = SessionHistory()
        msg = Message(role=Role.USER, parts=[Part.from_text("hello")])
        sh.append(msg)

        ref1 = sh.messages_ref
        ref2 = sh.messages_ref
        assert ref1 is ref2


# ==================== JaatoSession Integration Tests ====================


def _make_configured_session():
    """Create a configured JaatoSession with mock provider."""
    mock_runtime = MagicMock()
    mock_provider = MagicMock()
    mock_runtime.create_provider.return_value = mock_provider
    mock_runtime.get_tool_schemas.return_value = []
    mock_runtime.get_executors.return_value = {}
    mock_runtime.get_system_instructions.return_value = None
    mock_runtime.registry = None
    mock_runtime.permission_plugin = None

    session = JaatoSession(mock_runtime, "test-model")
    session.configure()
    return session, mock_provider


class TestSessionHistoryInSession:
    """Tests for SessionHistory integration with JaatoSession."""

    def test_session_has_session_history(self):
        """JaatoSession creates a SessionHistory on __init__."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "test-model")

        assert isinstance(session._history, SessionHistory)
        assert len(session._history) == 0

    def test_get_history_returns_from_session_history(self):
        """get_history() returns from SessionHistory."""
        session, mock_provider = _make_configured_session()

        # Put data directly in SessionHistory
        session._history.replace([
            Message(role=Role.USER, parts=[Part.from_text("from_session")]),
        ])

        history = session.get_history()
        assert len(history) == 1
        assert history[0].parts[0].text == "from_session"

    def test_get_history_empty_without_provider(self):
        """get_history() returns empty list without provider."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "test-model")

        # No provider configured
        assert session.get_history() == []

    def test_reset_session_updates_session_history(self):
        """reset_session(history) updates SessionHistory directly."""
        session, mock_provider = _make_configured_session()

        new_history = [
            Message(role=Role.USER, parts=[Part.from_text("restored")]),
            Message(role=Role.MODEL, parts=[Part.from_text("response")]),
        ]

        session.reset_session(new_history)

        assert len(session._history) == 2
        assert session._history.messages[0].parts[0].text == "restored"

    def test_reset_session_fresh_clears_history(self):
        """reset_session() without history clears SessionHistory."""
        session, mock_provider = _make_configured_session()

        # Add some history first
        session._history.append(
            Message(role=Role.USER, parts=[Part.from_text("old")])
        )
        assert len(session._history) == 1

        session.reset_session()

        assert len(session._history) == 0
        assert not session._history.dirty

    def test_history_replace_sets_history(self):
        """_history.replace() sets history directly on SessionHistory."""
        session, mock_provider = _make_configured_session()

        new_msgs = [
            Message(role=Role.USER, parts=[Part.from_text("hello")]),
            Message(role=Role.MODEL, parts=[Part.from_text("world")]),
        ]

        session._history.replace(new_msgs)

        assert len(session._history) == 2
        assert session._history.messages[0].parts[0].text == "hello"

    def test_history_clear_removes_all(self):
        """_history.clear() removes all messages from SessionHistory."""
        session, mock_provider = _make_configured_session()

        # Add some existing history
        session._history.append(
            Message(role=Role.USER, parts=[Part.from_text("old")])
        )

        session._history.clear()

        assert len(session._history) == 0

    def test_history_sync_preserves_across_gc_cycle(self):
        """History is preserved across a GC cycle (reset_session with history)."""
        session, mock_provider = _make_configured_session()

        # Simulate initial history
        initial_msgs = [
            Message(role=Role.USER, parts=[Part.from_text("turn1")]),
            Message(role=Role.MODEL, parts=[Part.from_text("response1")]),
            Message(role=Role.USER, parts=[Part.from_text("turn2")]),
            Message(role=Role.MODEL, parts=[Part.from_text("response2")]),
        ]
        session._history.replace(initial_msgs)

        # Simulate GC producing reduced history
        gc_msgs = [
            Message(role=Role.USER, parts=[Part.from_text("turn2")]),
            Message(role=Role.MODEL, parts=[Part.from_text("response2")]),
        ]

        session.reset_session(gc_msgs)

        assert len(session._history) == 2
        assert session._history.messages[0].parts[0].text == "turn2"
