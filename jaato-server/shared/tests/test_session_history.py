"""Tests for SessionHistory - canonical conversation history owned by the session.

Tests cover:
1. SessionHistory class behavior (append, replace, clear, dirty tracking, sync)
2. Integration with JaatoSession (get_history returns from SessionHistory,
   reset_session syncs, _create_provider_session syncs)
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


class TestSessionHistorySyncFromProvider:
    """Tests for SessionHistory.sync_from_provider()."""

    def test_sync_copies_provider_history(self):
        """sync_from_provider() copies provider's history."""
        sh = SessionHistory()
        provider = MagicMock()
        provider.get_history.return_value = [
            Message(role=Role.USER, parts=[Part.from_text("hello")]),
            Message(role=Role.MODEL, parts=[Part.from_text("world")]),
        ]

        sh.sync_from_provider(provider)

        assert len(sh) == 2
        assert sh.messages[0].parts[0].text == "hello"
        assert sh.messages[1].parts[0].text == "world"

    def test_sync_clears_dirty(self):
        """sync_from_provider() clears the dirty flag."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("dirty")]))
        assert sh.dirty

        provider = MagicMock()
        provider.get_history.return_value = [
            Message(role=Role.USER, parts=[Part.from_text("clean")]),
        ]
        sh.sync_from_provider(provider)

        assert not sh.dirty

    def test_sync_with_no_get_history(self):
        """sync_from_provider() is a no-op if provider lacks get_history."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("hello")]))

        provider = object()  # No get_history attribute
        sh.sync_from_provider(provider)

        # Should not have changed
        assert len(sh) == 1
        assert sh.dirty

    def test_sync_replaces_existing_content(self):
        """sync_from_provider() replaces existing messages."""
        sh = SessionHistory()
        sh.append(Message(role=Role.USER, parts=[Part.from_text("old")]))

        provider = MagicMock()
        provider.get_history.return_value = [
            Message(role=Role.USER, parts=[Part.from_text("new")]),
        ]
        sh.sync_from_provider(provider)

        assert len(sh) == 1
        assert sh.messages[0].parts[0].text == "new"

    def test_sync_makes_copy_of_provider_list(self):
        """sync_from_provider() copies the list from provider."""
        sh = SessionHistory()
        provider_list = [
            Message(role=Role.USER, parts=[Part.from_text("hello")]),
        ]
        provider = MagicMock()
        provider.get_history.return_value = provider_list

        sh.sync_from_provider(provider)

        # Mutating the original list should not affect SessionHistory
        provider_list.append(Message(role=Role.MODEL, parts=[Part.from_text("x")]))
        assert len(sh) == 1


# ==================== JaatoSession Integration Tests ====================


def _make_configured_session():
    """Create a configured JaatoSession with mock provider."""
    mock_runtime = MagicMock()
    mock_provider = MagicMock()
    mock_provider.get_history.return_value = []
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
        """get_history() returns from SessionHistory, not from provider."""
        session, mock_provider = _make_configured_session()

        # Put different data in provider vs SessionHistory
        mock_provider.get_history.return_value = [
            Message(role=Role.USER, parts=[Part.from_text("from_provider")]),
        ]
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
        """reset_session(history) updates SessionHistory.

        During Phase 1, _create_provider_session → sync_from_provider
        overwrites the SessionHistory from the provider. So we must
        set the mock provider to return the expected history.
        """
        session, mock_provider = _make_configured_session()

        new_history = [
            Message(role=Role.USER, parts=[Part.from_text("restored")]),
            Message(role=Role.MODEL, parts=[Part.from_text("response")]),
        ]
        # Provider returns same history after create_session (simulating
        # the provider accepting and storing the passed-in history)
        mock_provider.get_history.return_value = new_history

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

    def test_create_provider_session_syncs_history(self):
        """_create_provider_session() syncs history from provider."""
        session, mock_provider = _make_configured_session()

        # Set up provider to return history after create_session
        new_msgs = [
            Message(role=Role.USER, parts=[Part.from_text("hello")]),
            Message(role=Role.MODEL, parts=[Part.from_text("world")]),
        ]
        mock_provider.get_history.return_value = new_msgs

        session._create_provider_session(new_msgs)

        # SessionHistory should be synced
        assert len(session._history) == 2
        assert session._history.messages[0].parts[0].text == "hello"

    def test_sync_history_from_provider_no_provider(self):
        """_sync_history_from_provider() is no-op without provider."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "test-model")

        # Should not raise
        session._sync_history_from_provider()
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
        mock_provider.get_history.return_value = gc_msgs

        session.reset_session(gc_msgs)

        assert len(session._history) == 2
        assert session._history.messages[0].parts[0].text == "turn2"
