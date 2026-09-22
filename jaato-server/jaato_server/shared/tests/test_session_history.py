"""Tests for SessionHistory - canonical conversation history owned by the session.

Tests cover:
1. SessionHistory class behavior (append, replace, clear, dirty tracking)
2. Integration with JaatoSession (get_history returns from SessionHistory,
   reset_session updates history, _history.replace/clear set history)
3. Message provenance (_add_model_response_to_history stamps model/provider)
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from jaato_sdk.plugins.model_provider.types import (
    Message, Part, Role, ProviderResponse, FunctionCall,
)

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
    """Create a configured JaatoSession with mock provider.

    Triggers ``_ensure_provider()`` post-configure so tests that
    interact with the provider don't see ``None`` due to the
    2026-05-13 deferred-provider-INIT change.  Production callers
    that need the provider always go through send_message /
    generate / etc. which call ``_ensure_provider()`` first; the
    helper mirrors that lifecycle so unit tests don't have to."""
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
    # Trigger lazy provider creation so downstream test code sees
    # session._provider == mock_provider (matches production where
    # send_message would have already done this on first use).
    session._ensure_provider()
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


# ==================== Message Provenance Tests ====================


class TestAddModelResponseProvenance:
    """Tests that _add_model_response_to_history stamps model and provider."""

    def test_provenance_stamped_on_model_response(self):
        """_add_model_response_to_history sets model and provider on the message."""
        session, mock_provider = _make_configured_session()
        mock_provider.name = "google_genai"
        session._model_name = "gemini-2.5-flash"

        response = ProviderResponse(
            parts=[Part.from_text("Hello from Gemini")],
        )
        session._add_model_response_to_history(response)

        assert len(session._history) == 1
        msg = session._history.messages[0]
        assert msg.role == Role.MODEL
        assert msg.model == "gemini-2.5-flash"
        assert msg.provider == "google_genai"

    def test_provenance_with_different_provider(self):
        """Provenance reflects the actual provider and model at call time."""
        session, mock_provider = _make_configured_session()
        mock_provider.name = "anthropic"
        session._model_name = "claude-sonnet-4-5"

        response = ProviderResponse(
            parts=[Part.from_text("Hello from Claude")],
        )
        session._add_model_response_to_history(response)

        msg = session._history.messages[0]
        assert msg.model == "claude-sonnet-4-5"
        assert msg.provider == "anthropic"

    def test_provenance_none_when_no_provider(self):
        """Provider is None when session has no provider set."""
        session, _ = _make_configured_session()
        session._provider = None

        response = ProviderResponse(
            parts=[Part.from_text("orphan response")],
        )
        session._add_model_response_to_history(response)

        msg = session._history.messages[0]
        assert msg.model == "test-model"
        assert msg.provider is None

    def test_empty_response_not_appended(self):
        """Responses with no text or function_call parts are not appended."""
        session, mock_provider = _make_configured_session()
        mock_provider.name = "google_genai"

        response = ProviderResponse(parts=[])
        session._add_model_response_to_history(response)

        assert len(session._history) == 0


# ==================== Plug-in Transformer Tests ====================
#
# Seat 1 of the four-seat pseudonymization design — see
# project_backlog_pseudonymization_plugin_surface.md and
# jaato-premium/docs/design/pseudonymization-four-seat.md.  The
# transformers are deliberately generic (any per-Message rewrite is
# allowed); these tests cover the behavioural contract, not specific
# pseudonymization use cases.


def _redact_text(prefix: str):
    """Build a transformer that prepends a marker to every text part.

    Cheap stand-in for "redact PII" so the tests can verify ordering
    and chokepoint coverage without pulling in Presidio.
    """
    def _fn(msg: Message) -> Message:
        new_parts = [
            Part.from_text(prefix + p.text) if p.text else p
            for p in msg.parts
        ]
        return Message(
            role=msg.role,
            parts=new_parts,
            message_id=msg.message_id,
            model=msg.model,
            provider=msg.provider,
        )
    return _fn


class TestSessionHistoryInboundTransformer:
    """The inbound transformer fires on every append() / replace()
    before the message lands in the canonical container."""

    def test_unset_inbound_is_identity(self):
        h = SessionHistory()
        msg = Message.from_text(Role.USER, "hello")
        h.append(msg)
        assert h.messages[0].parts[0].text == "hello"

    def test_inbound_transformer_runs_on_append(self):
        h = SessionHistory()
        h.set_inbound_transformer(_redact_text("[REDACTED]"))
        h.append(Message.from_text(Role.USER, "secret"))
        assert h.messages[0].parts[0].text == "[REDACTED]secret"

    def test_inbound_transformer_runs_on_replace(self):
        h = SessionHistory()
        h.set_inbound_transformer(_redact_text("[R]"))
        h.replace([
            Message.from_text(Role.USER, "a"),
            Message.from_text(Role.MODEL, "b"),
        ])
        assert [m.parts[0].text for m in h.messages] == ["[R]a", "[R]b"]

    def test_inbound_transformer_can_be_cleared(self):
        h = SessionHistory()
        h.set_inbound_transformer(_redact_text("[R]"))
        h.append(Message.from_text(Role.USER, "first"))
        h.set_inbound_transformer(None)
        h.append(Message.from_text(Role.USER, "second"))
        assert h.messages[0].parts[0].text == "[R]first"
        assert h.messages[1].parts[0].text == "second"

    def test_canonical_container_holds_transformed_form(self):
        """Subsequent reads via every accessor see the transformed view
        — pop_last, last, messages_ref all return what landed."""
        h = SessionHistory()
        h.set_inbound_transformer(_redact_text("[R]"))
        h.append(Message.from_text(Role.USER, "x"))
        assert h.last.parts[0].text == "[R]x"
        assert h.messages_ref[0].parts[0].text == "[R]x"
        popped = h.pop_last()
        assert popped.parts[0].text == "[R]x"

    def test_dirty_flag_still_set_after_transform(self):
        h = SessionHistory()
        h.set_inbound_transformer(_redact_text("[R]"))
        assert not h.dirty
        h.append(Message.from_text(Role.USER, "x"))
        assert h.dirty


class TestSessionHistoryRawViewTransformer:
    """The raw-view transformer fires when ``messages_raw`` is accessed,
    giving trusted callers an un-transformed view."""

    def test_unset_raw_view_returns_stored(self):
        h = SessionHistory()
        h.append(Message.from_text(Role.USER, "x"))
        assert h.messages_raw[0].parts[0].text == "x"

    def test_raw_view_transformer_runs_on_each_message(self):
        h = SessionHistory()
        h.set_raw_view_transformer(_redact_text("[U]"))
        h.append(Message.from_text(Role.USER, "stored"))
        assert h.messages_raw[0].parts[0].text == "[U]stored"

    def test_raw_view_does_not_mutate_stored(self):
        """Stored messages stay untouched; the transformer runs on
        each read."""
        h = SessionHistory()
        h.set_raw_view_transformer(_redact_text("[U]"))
        h.append(Message.from_text(Role.USER, "stored"))
        # Read once — get transformed
        assert h.messages_raw[0].parts[0].text == "[U]stored"
        # Stored copy untouched
        assert h.messages[0].parts[0].text == "stored"
        assert h.messages_ref[0].parts[0].text == "stored"

    def test_raw_view_transformer_can_be_cleared(self):
        h = SessionHistory()
        h.set_raw_view_transformer(_redact_text("[U]"))
        h.append(Message.from_text(Role.USER, "x"))
        assert h.messages_raw[0].parts[0].text == "[U]x"
        h.set_raw_view_transformer(None)
        assert h.messages_raw[0].parts[0].text == "x"


class TestSessionHistoryInboundAndRawViewCompose:
    """Premium's typical pseudonymization pattern: inbound redacts on
    write, raw-view un-redacts on trusted read.  Confirms the two
    transformers compose to round-trip the original value."""

    def test_round_trip_via_lookup_table(self):
        # Stand-in for premium's PseudonymTable: a dict lookup.
        table = {}
        counter = [0]

        def inbound(msg: Message) -> Message:
            new_parts = []
            for p in msg.parts:
                if not p.text:
                    new_parts.append(p)
                    continue
                counter[0] += 1
                placeholder = f"<TOKEN_{counter[0]}>"
                table[placeholder] = p.text
                new_parts.append(Part.from_text(placeholder))
            return Message(
                role=msg.role, parts=new_parts,
                message_id=msg.message_id,
                model=msg.model, provider=msg.provider,
            )

        def raw_view(msg: Message) -> Message:
            new_parts = []
            for p in msg.parts:
                if not p.text:
                    new_parts.append(p)
                    continue
                # Replace each placeholder back to its raw value.
                text = p.text
                for placeholder, raw in table.items():
                    text = text.replace(placeholder, raw)
                new_parts.append(Part.from_text(text))
            return Message(
                role=msg.role, parts=new_parts,
                message_id=msg.message_id,
                model=msg.model, provider=msg.provider,
            )

        h = SessionHistory()
        h.set_inbound_transformer(inbound)
        h.set_raw_view_transformer(raw_view)

        h.append(Message.from_text(Role.USER, "email me at alice@example.com"))
        # Stored form is redacted
        assert "<TOKEN_1>" in h.messages[0].parts[0].text
        assert "alice@example.com" not in h.messages[0].parts[0].text
        # Raw view round-trips back to original
        assert h.messages_raw[0].parts[0].text == "email me at alice@example.com"


class TestJaatoSessionTransformerPassthroughs:
    """JaatoSession exposes the same plug-in surface so callers don't
    need to reach through .history."""

    def _session(self):
        mock_runtime = MagicMock()
        mock_runtime.create_provider.return_value = MagicMock()
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.permission_plugin = None
        mock_runtime.registry = None
        mock_runtime.reliability_plugin = None
        return JaatoSession(mock_runtime, "model")

    def test_set_history_inbound_transformer_propagates(self):
        s = self._session()
        s.set_history_inbound_transformer(_redact_text("[R]"))
        s._history.append(Message.from_text(Role.USER, "x"))
        assert s.get_history()[0].parts[0].text == "[R]x"

    def test_get_history_raw_returns_transformed_view(self):
        s = self._session()
        s.set_history_raw_view_transformer(_redact_text("[U]"))
        s._history.append(Message.from_text(Role.USER, "stored"))
        assert s.get_history_raw()[0].parts[0].text == "[U]stored"

    def test_get_history_returns_canonical_form(self):
        """get_history is the redacted (canonical) view; get_history_raw
        is the trusted-caller view.  Confirm they diverge when both
        transformers are set."""
        s = self._session()
        s.set_history_inbound_transformer(_redact_text("[R]"))
        s.set_history_raw_view_transformer(_redact_text("[U]"))
        s._history.append(Message.from_text(Role.USER, "x"))
        # Stored: [R]x.  Raw view runs [U] on top of stored.
        assert s.get_history()[0].parts[0].text == "[R]x"
        assert s.get_history_raw()[0].parts[0].text == "[U][R]x"

    def test_passthroughs_accept_none_to_clear(self):
        s = self._session()
        s.set_history_inbound_transformer(_redact_text("[R]"))
        s.set_history_inbound_transformer(None)
        s._history.append(Message.from_text(Role.USER, "x"))
        assert s.get_history()[0].parts[0].text == "x"
