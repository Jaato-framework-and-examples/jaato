"""Tests for the Antigravity provider.

These tests cover the basic functionality of the Antigravity provider.
Most tests use mocking to avoid requiring actual OAuth tokens.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from shared.plugins.model_provider.antigravity import (
    AntigravityProvider,
    AuthenticationError,
    ModelNotFoundError,
    create_provider,
)
from shared.plugins.model_provider.antigravity.converters import (
    build_generate_request,
    build_generation_config,
    messages_to_api,
    parse_sse_event,
    response_from_api,
    serialize_history,
    deserialize_history,
    tool_schemas_to_api,
)
from shared.plugins.model_provider.antigravity.oauth import (
    Account,
    AccountManager,
    OAuthTokens,
    _generate_pkce_pair,
)
from shared.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)


# ==================== Provider Tests ====================


class TestAntigravityProvider:
    """Tests for AntigravityProvider class."""

    def test_create_provider(self):
        """Test factory function creates provider."""
        provider = create_provider()
        assert isinstance(provider, AntigravityProvider)
        assert provider.name == "antigravity"

    def test_provider_not_connected_initially(self):
        """Test provider is not connected after creation."""
        provider = AntigravityProvider()
        assert not provider.is_connected
        assert provider.model_name is None

    def test_initialize_without_accounts_raises(self):
        """Test initialize raises when no accounts exist."""
        provider = AntigravityProvider()

        with patch(
            "shared.plugins.model_provider.antigravity.provider.load_accounts"
        ) as mock_load:
            mock_load.return_value = AccountManager()

            with pytest.raises(AuthenticationError) as exc_info:
                provider.initialize()

            assert "No Antigravity accounts found" in str(exc_info.value)

    def test_connect_valid_model(self):
        """Test connecting to a valid model."""
        provider = AntigravityProvider()

        # Mock account
        tokens = OAuthTokens(
            access_token="test_token",
            refresh_token="test_refresh",
            expires_at=9999999999,
            email="test@example.com",
        )
        account = Account(email="test@example.com", tokens=tokens)
        manager = AccountManager(accounts=[account])

        with patch(
            "shared.plugins.model_provider.antigravity.provider.load_accounts"
        ) as mock_load:
            mock_load.return_value = manager
            provider.initialize()

        provider.connect("antigravity-gemini-3-flash")
        assert provider.is_connected
        assert provider.model_name == "antigravity-gemini-3-flash"

    def test_connect_invalid_model_raises(self):
        """Test connecting to invalid model raises error."""
        provider = AntigravityProvider()

        # Mock account
        tokens = OAuthTokens(
            access_token="test_token",
            refresh_token="test_refresh",
            expires_at=9999999999,
            email="test@example.com",
        )
        account = Account(email="test@example.com", tokens=tokens)
        manager = AccountManager(accounts=[account])

        with patch(
            "shared.plugins.model_provider.antigravity.provider.load_accounts"
        ) as mock_load:
            mock_load.return_value = manager
            provider.initialize()

        with pytest.raises(ModelNotFoundError):
            provider.connect("nonexistent-model")

    def test_list_models(self):
        """Test listing available models."""
        provider = AntigravityProvider()
        models = provider.list_models()

        assert len(models) > 0
        assert "antigravity-gemini-3-flash" in models
        assert "gemini-2.5-flash" in models

    def test_list_models_with_prefix(self):
        """Test listing models with prefix filter."""
        provider = AntigravityProvider()
        models = provider.list_models(prefix="antigravity-")

        assert all(m.startswith("antigravity-") for m in models)

    def test_supports_streaming(self):
        """Test streaming support."""
        provider = AntigravityProvider()
        assert provider.supports_streaming()

    def test_supports_stop(self):
        """Test stop support."""
        provider = AntigravityProvider()
        assert provider.supports_stop()


# ==================== Converter Tests ====================


class TestConverters:
    """Tests for type converters."""

    def test_messages_to_api(self):
        """Test converting messages to API format."""
        messages = [
            Message.from_text(Role.USER, "Hello"),
            Message(role=Role.MODEL, parts=[Part(text="Hi there!")]),
        ]

        api_messages = messages_to_api(messages)

        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["parts"][0]["text"] == "Hello"
        assert api_messages[1]["role"] == "model"
        assert api_messages[1]["parts"][0]["text"] == "Hi there!"

    def test_tool_schemas_to_api(self):
        """Test converting tool schemas to API format."""
        schemas = [
            ToolSchema(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]

        api_tools = tool_schemas_to_api(schemas)

        assert len(api_tools) == 1
        assert "functionDeclarations" in api_tools[0]
        assert api_tools[0]["functionDeclarations"][0]["name"] == "read_file"

    def test_response_from_api(self):
        """Test converting API response to ProviderResponse."""
        api_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello, world!"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        response = response_from_api(api_response)

        assert response.get_text() == "Hello, world!"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage.prompt_tokens == 10
        assert response.usage.output_tokens == 5

    def test_response_with_function_call(self):
        """Test converting response with function call."""
        api_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "/tmp/test.txt"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
        }

        response = response_from_api(api_response)

        assert len(response.function_calls) == 1
        assert response.function_calls[0].name == "read_file"
        assert response.function_calls[0].args == {"path": "/tmp/test.txt"}
        assert response.finish_reason == FinishReason.TOOL_USE

    def test_parse_sse_event(self):
        """Test parsing SSE events."""
        # Data event
        event = parse_sse_event('data: {"test": "value"}')
        assert event == {"test": "value"}

        # Done event
        event = parse_sse_event("data: [DONE]")
        assert event == {"done": True}

        # Empty line
        event = parse_sse_event("")
        assert event is None

        # Invalid JSON
        event = parse_sse_event("data: invalid json")
        assert event is None

    def test_build_generate_request(self):
        """Test building generate request."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        request = build_generate_request(
            contents=contents,
            system_instruction="You are helpful.",
            generation_config={"maxOutputTokens": 1000},
        )

        assert request["contents"] == contents
        assert request["systemInstruction"]["parts"][0]["text"] == "You are helpful."
        assert request["generationConfig"]["maxOutputTokens"] == 1000

    def test_build_generation_config(self):
        """Test building generation config."""
        config = build_generation_config(
            max_output_tokens=2000,
            temperature=0.7,
            thinking_config={"thinkingLevel": "high"},
        )

        assert config["maxOutputTokens"] == 2000
        assert config["temperature"] == 0.7
        assert config["thinkingConfig"]["thinkingLevel"] == "high"

    def test_serialize_deserialize_history(self):
        """Test history serialization round-trip."""
        original = [
            Message.from_text(Role.USER, "Hello"),
            Message(
                role=Role.MODEL,
                parts=[
                    Part(text="Hi!"),
                    Part(function_call=FunctionCall(
                        id="call1",
                        name="test",
                        args={"x": 1},
                    )),
                ],
            ),
        ]

        serialized = serialize_history(original)
        deserialized = deserialize_history(serialized)

        assert len(deserialized) == 2
        assert deserialized[0].role == Role.USER
        assert deserialized[0].parts[0].text == "Hello"
        assert deserialized[1].parts[0].text == "Hi!"
        assert deserialized[1].parts[1].function_call.name == "test"


# ==================== OAuth Tests ====================


class TestOAuth:
    """Tests for OAuth functionality."""

    def test_generate_pkce_pair(self):
        """Test PKCE pair generation."""
        verifier, challenge = _generate_pkce_pair()

        # Verifier should be base64url encoded
        assert len(verifier) > 20
        assert "+" not in verifier
        assert "/" not in verifier

        # Challenge should be different from verifier
        assert challenge != verifier
        assert len(challenge) > 20

    def test_oauth_tokens_expiration(self):
        """Test OAuth token expiration check."""
        import time

        # Not expired
        tokens = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() + 3600,
        )
        assert not tokens.is_expired

        # Expired
        tokens = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() - 100,
        )
        assert tokens.is_expired

        # About to expire (within 5 min buffer)
        tokens = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() + 60,
        )
        assert tokens.is_expired

    def test_account_rate_limiting(self):
        """Test account rate limit tracking."""
        import time

        tokens = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() + 3600,
        )
        account = Account(email="test@example.com", tokens=tokens)

        assert not account.is_rate_limited()

        account.mark_rate_limited(duration=1.0)
        assert account.is_rate_limited()

        account.clear_rate_limit()
        assert not account.is_rate_limited()

    def test_account_manager_rotation(self):
        """Test account manager rotation."""
        import time

        tokens1 = OAuthTokens(
            access_token="token1",
            refresh_token="refresh1",
            expires_at=time.time() + 3600,
        )
        tokens2 = OAuthTokens(
            access_token="token2",
            refresh_token="refresh2",
            expires_at=time.time() + 3600,
        )

        account1 = Account(email="user1@example.com", tokens=tokens1)
        account2 = Account(email="user2@example.com", tokens=tokens2)

        manager = AccountManager(accounts=[account1, account2])

        # First account
        active = manager.get_active_account()
        assert active.email == "user1@example.com"

        # Rotate on rate limit
        next_account = manager.rotate_on_rate_limit("user1@example.com")
        assert next_account.email == "user2@example.com"
