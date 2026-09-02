"""Tests for session serialization utilities."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from ..serializer import (
    serialize_part,
    deserialize_part,
    serialize_message,
    deserialize_message,
    serialize_history,
    deserialize_history,
    serialize_session_state,
    deserialize_session_state,
)
from ..base import SessionState
from jaato_sdk.plugins.model_provider.types import Message, Part, Role, FunctionCall, ToolResult


class TestPartSerialization:
    """Tests for Part serialization/deserialization."""

    def test_serialize_text_part(self):
        """Test serializing a text part."""
        part = Part.from_text("Hello, world!")
        result = serialize_part(part)

        assert result["type"] == "text"
        assert result["text"] == "Hello, world!"

    def test_deserialize_text_part(self):
        """Test deserializing a text part."""
        data = {"type": "text", "text": "Hello, world!"}
        part = deserialize_part(data)

        assert part.text == "Hello, world!"

    def test_serialize_function_response_part(self):
        """Test serializing a function response part."""
        tool_result = ToolResult(
            call_id="test_id",
            name="my_function",
            result={"result": "success", "value": 42}
        )
        part = Part.from_function_response(tool_result)
        result = serialize_part(part)

        assert result["type"] == "function_response"
        assert result["name"] == "my_function"
        assert result["result"]["result"] == "success"
        assert result["result"]["value"] == 42

    def test_deserialize_function_response_part(self):
        """Test deserializing a function response part."""
        data = {
            "type": "function_response",
            "call_id": "test_id",
            "name": "my_function",
            "response": {"result": "success"}
        }
        part = deserialize_part(data)

        assert part.function_response is not None
        assert part.function_response.name == "my_function"

    def test_round_trip_text_part(self):
        """Test round-trip serialization of text part."""
        original = Part.from_text("Test message")
        data = serialize_part(original)
        restored = deserialize_part(data)

        assert restored.text == original.text


class TestMessageSerialization:
    """Tests for Message serialization/deserialization."""

    def test_serialize_user_message(self):
        """Test serializing user message."""
        message = Message(
            role=Role.USER,
            parts=[Part.from_text("Hello")]
        )
        result = serialize_message(message)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["text"] == "Hello"

    def test_deserialize_user_message(self):
        """Test deserializing user message."""
        data = {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello"}]
        }
        message = deserialize_message(data)

        assert message.role == Role.USER
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello"

    def test_serialize_model_message_with_function_response(self):
        """Test serializing model message with function response."""
        tool_result = ToolResult(
            call_id="test_id",
            name="get_weather",
            result={"temp": 72, "conditions": "sunny"}
        )
        message = Message(
            role=Role.MODEL,
            parts=[Part.from_function_response(tool_result)]
        )
        result = serialize_message(message)

        assert result["role"] == "model"
        assert result["parts"][0]["type"] == "function_response"
        assert result["parts"][0]["name"] == "get_weather"


class TestHistorySerialization:
    """Tests for conversation history serialization."""

    def test_serialize_empty_history(self):
        """Test serializing empty history."""
        result = serialize_history([])
        assert result == []

    def test_deserialize_empty_history(self):
        """Test deserializing empty history."""
        result = deserialize_history([])
        assert result == []

    def test_round_trip_conversation(self):
        """Test round-trip serialization of a conversation."""
        history = [
            Message(
                role=Role.USER,
                parts=[Part.from_text("What's the weather?")]
            ),
            Message(
                role=Role.MODEL,
                parts=[Part.from_text("Let me check...")]
            ),
        ]

        data = serialize_history(history)
        restored = deserialize_history(data)

        assert len(restored) == 2
        assert restored[0].role == Role.USER
        assert restored[0].parts[0].text == "What's the weather?"
        assert restored[1].role == Role.MODEL
        assert restored[1].parts[0].text == "Let me check..."


class TestSessionStateSerialization:
    """Tests for SessionState serialization."""

    def test_serialize_session_state(self):
        """Test serializing a complete session state.

        Post-2.3: connection dict (project/location/model) retired in
        favour of ``profile_name`` — the profile is the authoritative
        recipe source for model + provider + plugin_configs.
        """
        history = [
            Message(
                role=Role.USER,
                parts=[Part.from_text("Hello")]
            )
        ]

        state = SessionState(
            session_id="20251207_143022",
            history=history,
            created_at=datetime(2025, 12, 7, 14, 30, 22),
            updated_at=datetime(2025, 12, 7, 15, 0, 0),
            description="Test session",
            turn_count=1,
            turn_accounting=[{"prompt": 10, "output": 20, "total": 30}],
            profile_name="discovery",
        )

        data = serialize_session_state(state)

        assert data["version"] == "2.8"
        assert data["session_id"] == "20251207_143022"
        assert data["description"] == "Test session"
        assert data["turn_count"] == 1
        assert data["profile_name"] == "discovery"
        assert "connection" not in data, (
            "post-2.3 must NOT write connection dict — retired field"
        )
        assert len(data["history"]) == 1

    def test_deserialize_session_state(self):
        """Test deserializing a session state (post-2.3 shape)."""
        data = {
            "version": "2.3",
            "session_id": "20251207_143022",
            "description": "Test session",
            "created_at": "2025-12-07T14:30:22",
            "updated_at": "2025-12-07T15:00:00",
            "turn_count": 1,
            "turn_accounting": [{"prompt": 10, "output": 20, "total": 30}],
            "profile_name": "discovery",
            "history": [
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}]
                }
            ],
        }

        state = deserialize_session_state(data)

        assert state.session_id == "20251207_143022"
        assert state.description == "Test session"
        assert state.turn_count == 1
        assert state.profile_name == "discovery"
        assert len(state.history) == 1

    def test_config_root_round_trip(self):
        """2.4: ``config_root`` paired with ``profile_name`` so
        disk-restore can hand the right config tier to
        ``discover_profiles`` (needed for the JAATO_PROFILE_SET
        subdir scan + workspace-tier scan)."""
        state = SessionState(
            session_id="cfg_root_test",
            history=[],
            created_at=datetime(2026, 6, 3, 20, 0, 0),
            updated_at=datetime(2026, 6, 3, 20, 0, 0),
            description="config_root test",
            turn_count=0,
            profile_name="discovery",
            workspace_path="/repo/tests/runs/cascade_smoke",
            config_root="/repo/.jaato",
        )

        data = serialize_session_state(state)
        assert data["version"] == "2.8"
        assert data["config_root"] == "/repo/.jaato"
        assert data["profile_name"] == "discovery"

        restored = deserialize_session_state(data)
        assert restored.config_root == "/repo/.jaato"
        assert restored.profile_name == "discovery"

    def test_profile_spec_round_trip(self):
        """2.7: the UNRESOLVED inline-profile spec persists so disk-restore
        reconstructs an inline session's recipe by id alone — the
        ``profile_name`` ("<inline>") isn't re-resolvable from disk."""
        spec = {
            "model": "gemini-nano",
            "provider": "chrome_ai",
            "plugins": [],
            "suppress_base_instructions": True,
            # plugin_configs MUST survive — resume can't reconnect chrome_ai
            # without cdp_url/reuse_page.
            "plugin_configs": {"chrome_ai": {
                "cdp_url": "http://[::1]:9222", "page_url": "https://example.com",
                "reuse_page": True}},
        }
        state = SessionState(
            session_id="inline_test",
            history=[],
            created_at=datetime(2026, 7, 8, 12, 0, 0),
            updated_at=datetime(2026, 7, 8, 12, 0, 0),
            description="inline spec test",
            turn_count=0,
            profile_name="<inline>",
            profile_spec=spec,
        )
        data = serialize_session_state(state)
        assert data["version"] == "2.8"
        assert data["profile_spec"] == spec              # full recipe on the wire

        restored = deserialize_session_state(data)
        assert restored.profile_spec == spec
        # plugin_configs survives intact — the crux for chrome_ai resume.
        assert (restored.profile_spec["plugin_configs"]["chrome_ai"]["cdp_url"]
                == "http://[::1]:9222")

    def test_profile_spec_absent_on_named_and_old_records(self):
        # Named-profile sessions carry no spec; pre-2.7 records (no key)
        # deserialize to None, not a crash.
        state = SessionState(
            session_id="named", history=[],
            created_at=datetime(2026, 7, 8, 12, 0, 0),
            updated_at=datetime(2026, 7, 8, 12, 0, 0),
            profile_name="researcher")
        assert serialize_session_state(state)["profile_spec"] is None
        # A 2.6 record with no profile_spec key at all:
        old = {"version": "2.6", "session_id": "old", "history": [],
               "created_at": "2026-07-08T12:00:00", "updated_at": "2026-07-08T12:00:00",
               "profile_name": "researcher"}
        assert deserialize_session_state(old).profile_spec is None

    def test_sandbox_mode_round_trip(self):
        """2.5: ``sandbox_mode`` persisted so orphan-revive / disk-restore
        re-applies the SAME confinement on runner re-spawn (else the revive
        ran unconfined after any idle detach — a security regression)."""
        state = SessionState(
            session_id="sandbox_test",
            history=[],
            created_at=datetime(2026, 7, 1, 16, 22, 28),
            updated_at=datetime(2026, 7, 1, 16, 22, 28),
            sandbox_mode="apparmor",
        )
        data = serialize_session_state(state)
        assert data["sandbox_mode"] == "apparmor"
        assert deserialize_session_state(data).sandbox_mode == "apparmor"

    def test_pre_2_5_record_deserializes_sandbox_mode_none(self):
        """Pre-2.5 JSONs lack ``sandbox_mode`` — deserialize to None
        (unchanged behavior; the revive apparmor= computation reads None)."""
        data = {
            "version": "2.4",
            "session_id": "old",
            "created_at": "2026-06-30T12:00:00",
            "updated_at": "2026-06-30T12:00:00",
            "history": [],
        }
        assert deserialize_session_state(data).sandbox_mode is None

    def test_deserialize_pre_2_4_no_config_root(self):
        """Pre-2.4 session JSONs lack ``config_root``.  Deserialize
        cleanly with ``config_root=None``; ``_resolve_profile`` then
        falls back to workspace_path-only resolution (works for
        sessions where the profile lives under
        ``<workspace>/.jaato/profiles/`` — not the multi-profile
        cascade layout)."""
        data = {
            "version": "2.3",
            "session_id": "pre_2_4_session",
            "description": "Pre-2.4 session",
            "created_at": "2026-06-03T19:30:00",
            "updated_at": "2026-06-03T19:30:00",
            "turn_count": 1,
            "profile_name": "discovery",
            "workspace_path": "/repo/tests/runs/cascade_smoke",
            "history": [],
        }
        state = deserialize_session_state(data)
        assert state.profile_name == "discovery"
        assert state.config_root is None

    def test_deserialize_pre_2_3_backward_compat(self):
        """Pre-2.3 session JSONs carry a ``connection`` dict with
        project/location/model.  Deserialization MUST tolerate them
        — silently drops the retired fields, leaves ``profile_name``
        as None so the disk-restore path falls through to env-only
        resolution (same constraint as fresh-spawn-without-profile).
        """
        data = {
            "version": "2.2",
            "session_id": "old_session",
            "description": "Pre-2.3 session",
            "created_at": "2025-12-07T14:30:22",
            "updated_at": "2025-12-07T15:00:00",
            "turn_count": 1,
            "connection": {
                "project": "my-gcp-project",
                "location": "us-central1",
                "model": "gemini-2.5-flash",
            },
            "history": [],
        }

        state = deserialize_session_state(data)

        # The connection dict is silently dropped — state has no
        # project/location/model attributes anymore.
        assert state.session_id == "old_session"
        assert state.profile_name is None, (
            "pre-2.3 sessions have no profile_name; disk-restore "
            "falls through to env-only model resolution"
        )
        assert not hasattr(state, 'project'), (
            "project field retired in 2.3 — Google-coupled"
        )
        assert not hasattr(state, 'location'), (
            "location field retired in 2.3 — Google-coupled"
        )
        assert not hasattr(state, 'model'), (
            "model field retired in 2.3 — superseded by profile_name"
        )

    def test_round_trip_session_state(self):
        """Test round-trip serialization of session state."""
        history = [
            Message(
                role=Role.USER,
                parts=[Part.from_text("Test")]
            )
        ]

        original = SessionState(
            session_id="test_session",
            history=history,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Round trip test",
            turn_count=1,
        )

        data = serialize_session_state(original)
        restored = deserialize_session_state(data)

        assert restored.session_id == original.session_id
        assert restored.description == original.description
        assert restored.turn_count == original.turn_count
        assert len(restored.history) == len(original.history)

    def test_deserialize_incompatible_version(self):
        """Test that incompatible versions raise ValueError."""
        data = {
            "version": "3.0",
            "session_id": "test",
            "created_at": "2025-12-07T14:30:22",
            "updated_at": "2025-12-07T15:00:00",
        }

        with pytest.raises(ValueError, match="Unsupported session version"):
            deserialize_session_state(data)


class TestMessageProvenanceSerialization:
    """Tests for message provenance (model/provider) serialization."""

    def test_serialize_message_with_provenance(self):
        """Test that model and provider are included in serialized output."""
        message = Message(
            role=Role.MODEL,
            parts=[Part.from_text("Hello")],
            model="gemini-2.5-flash",
            provider="google_genai",
        )
        result = serialize_message(message)

        assert result["model"] == "gemini-2.5-flash"
        assert result["provider"] == "google_genai"

    def test_serialize_message_without_provenance(self):
        """Test that model/provider keys are omitted when None."""
        message = Message(
            role=Role.USER,
            parts=[Part.from_text("Hello")],
        )
        result = serialize_message(message)

        assert "model" not in result
        assert "provider" not in result

    def test_deserialize_message_with_provenance(self):
        """Test that model and provider are restored from serialized data."""
        data = {
            "role": "model",
            "parts": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
        }
        message = deserialize_message(data)

        assert message.model == "claude-sonnet-4-5"
        assert message.provider == "anthropic"

    def test_deserialize_message_without_provenance_backward_compat(self):
        """Test backward compat: old data without model/provider deserializes to None."""
        data = {
            "role": "model",
            "parts": [{"type": "text", "text": "Hello"}],
        }
        message = deserialize_message(data)

        assert message.model is None
        assert message.provider is None

    def test_round_trip_provenance(self):
        """Test that provenance round-trips through serialize/deserialize."""
        original = Message(
            role=Role.MODEL,
            parts=[Part.from_text("Response")],
            model="gemini-2.5-flash",
            provider="google_genai",
        )
        data = serialize_message(original)
        restored = deserialize_message(data)

        assert restored.model == original.model
        assert restored.provider == original.provider
        assert restored.role == original.role
        assert restored.parts[0].text == original.parts[0].text

    def test_session_state_round_trip_preserves_provenance(self):
        """Test that provenance survives full session state round-trip."""
        history = [
            Message(
                role=Role.USER,
                parts=[Part.from_text("What's the weather?")],
            ),
            Message(
                role=Role.MODEL,
                parts=[Part.from_text("Let me check...")],
                model="gemini-2.5-flash",
                provider="google_genai",
            ),
        ]

        state = SessionState(
            session_id="provenance_test",
            history=history,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Provenance round-trip test",
            turn_count=1,
        )

        data = serialize_session_state(state)
        restored = deserialize_session_state(data)

        assert len(restored.history) == 2
        # User message has no provenance
        assert restored.history[0].model is None
        assert restored.history[0].provider is None
        # Model message preserves provenance
        assert restored.history[1].model == "gemini-2.5-flash"
        assert restored.history[1].provider == "google_genai"

    def test_agent_name_round_trip(self):
        """2.6: ``agent_name`` (persona/--agent) persisted so orphan-revive
        rebinds the persona — else revived multimodal sessions lose their
        enter_tier guidance and confabulate on images."""
        state = SessionState(
            session_id="agent_test",
            history=[],
            created_at=datetime(2026, 7, 1, 16, 22, 28),
            updated_at=datetime(2026, 7, 1, 16, 22, 28),
            agent_name="telegram_chat",
        )
        data = serialize_session_state(state)
        assert data["agent_name"] == "telegram_chat"
        assert deserialize_session_state(data).agent_name == "telegram_chat"

    def test_pre_2_6_record_deserializes_agent_name_none(self):
        """Pre-2.6 JSONs lack ``agent_name`` — deserialize to None (unchanged;
        JaatoServer falls back agent id to 'main')."""
        data = {
            "version": "2.5",
            "session_id": "old",
            "created_at": "2026-06-30T12:00:00",
            "updated_at": "2026-06-30T12:00:00",
            "history": [],
        }
        assert deserialize_session_state(data).agent_name is None
