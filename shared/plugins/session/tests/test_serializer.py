"""Tests for session serialization utilities."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from google.genai import types

from ..serializer import (
    serialize_part,
    deserialize_part,
    serialize_content,
    deserialize_content,
    serialize_history,
    deserialize_history,
    serialize_session_state,
    deserialize_session_state,
)
from ..base import SessionState


class TestPartSerialization:
    """Tests for Part serialization/deserialization."""

    def test_serialize_text_part(self):
        """Test serializing a text part."""
        part = types.Part.from_text(text="Hello, world!")
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
        part = types.Part.from_function_response(
            name="my_function",
            response={"result": "success", "value": 42}
        )
        result = serialize_part(part)

        assert result["type"] == "function_response"
        assert result["name"] == "my_function"
        assert result["response"]["result"] == "success"
        assert result["response"]["value"] == 42

    def test_deserialize_function_response_part(self):
        """Test deserializing a function response part."""
        data = {
            "type": "function_response",
            "name": "my_function",
            "response": {"result": "success"}
        }
        part = deserialize_part(data)

        assert part.function_response is not None
        assert part.function_response.name == "my_function"

    def test_round_trip_text_part(self):
        """Test round-trip serialization of text part."""
        original = types.Part.from_text(text="Test message")
        data = serialize_part(original)
        restored = deserialize_part(data)

        assert restored.text == original.text


class TestContentSerialization:
    """Tests for Content serialization/deserialization."""

    def test_serialize_user_content(self):
        """Test serializing user content."""
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hello")]
        )
        result = serialize_content(content)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["text"] == "Hello"

    def test_deserialize_user_content(self):
        """Test deserializing user content."""
        data = {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello"}]
        }
        content = deserialize_content(data)

        assert content.role == "user"
        assert len(content.parts) == 1
        assert content.parts[0].text == "Hello"

    def test_serialize_model_content_with_function_response(self):
        """Test serializing model content with function response."""
        content = types.Content(
            role="model",
            parts=[
                types.Part.from_function_response(
                    name="get_weather",
                    response={"temp": 72, "conditions": "sunny"}
                )
            ]
        )
        result = serialize_content(content)

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
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="What's the weather?")]
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Let me check...")]
            ),
        ]

        data = serialize_history(history)
        restored = deserialize_history(data)

        assert len(restored) == 2
        assert restored[0].role == "user"
        assert restored[0].parts[0].text == "What's the weather?"
        assert restored[1].role == "model"
        assert restored[1].parts[0].text == "Let me check..."


class TestSessionStateSerialization:
    """Tests for SessionState serialization."""

    def test_serialize_session_state(self):
        """Test serializing a complete session state."""
        history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello")]
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
            project="my-project",
            location="us-central1",
            model="gemini-2.5-flash",
        )

        data = serialize_session_state(state)

        assert data["version"] == "1.0"
        assert data["session_id"] == "20251207_143022"
        assert data["description"] == "Test session"
        assert data["turn_count"] == 1
        assert data["connection"]["project"] == "my-project"
        assert len(data["history"]) == 1

    def test_deserialize_session_state(self):
        """Test deserializing a session state."""
        data = {
            "version": "1.0",
            "session_id": "20251207_143022",
            "description": "Test session",
            "created_at": "2025-12-07T14:30:22",
            "updated_at": "2025-12-07T15:00:00",
            "turn_count": 1,
            "turn_accounting": [{"prompt": 10, "output": 20, "total": 30}],
            "connection": {
                "project": "my-project",
                "location": "us-central1",
                "model": "gemini-2.5-flash",
            },
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
        assert state.project == "my-project"
        assert len(state.history) == 1

    def test_round_trip_session_state(self):
        """Test round-trip serialization of session state."""
        history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Test")]
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
            "version": "2.0",
            "session_id": "test",
            "created_at": "2025-12-07T14:30:22",
            "updated_at": "2025-12-07T15:00:00",
        }

        with pytest.raises(ValueError, match="Unsupported session version"):
            deserialize_session_state(data)
