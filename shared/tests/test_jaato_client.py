"""Tests for JaatoClient - core client for the jaato framework."""

import os
import pytest
from unittest.mock import MagicMock, patch

from ..jaato_client import JaatoClient, get_default_provider, DEFAULT_PROVIDER


class TestGetDefaultProvider:
    """Tests for get_default_provider() function."""

    def test_returns_google_genai_when_no_env(self):
        """Test that google_genai is returned when JAATO_PROVIDER is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove JAATO_PROVIDER if it exists
            os.environ.pop("JAATO_PROVIDER", None)
            assert get_default_provider() == "google_genai"

    def test_returns_env_value_when_set(self):
        """Test that JAATO_PROVIDER env var is returned when set."""
        with patch.dict(os.environ, {"JAATO_PROVIDER": "github_models"}):
            assert get_default_provider() == "github_models"

    def test_returns_custom_provider_from_env(self):
        """Test that any custom provider name is returned from env."""
        with patch.dict(os.environ, {"JAATO_PROVIDER": "anthropic"}):
            assert get_default_provider() == "anthropic"


class TestJaatoClientInitialization:
    """Tests for JaatoClient initialization."""

    def test_init_no_args_uses_env_default(self):
        """Test that init with no args uses get_default_provider()."""
        with patch.dict(os.environ, {"JAATO_PROVIDER": "github_models"}):
            client = JaatoClient()
            assert client.provider_name == "github_models"

    def test_init_no_args_falls_back_to_google(self):
        """Test that init with no args falls back to google_genai."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            client = JaatoClient()
            assert client.provider_name == "google_genai"

    def test_init_explicit_provider_overrides_env(self):
        """Test that explicit provider_name overrides env var."""
        with patch.dict(os.environ, {"JAATO_PROVIDER": "github_models"}):
            client = JaatoClient(provider_name="anthropic")
            assert client.provider_name == "anthropic"

    def test_init_explicit_none_uses_env(self):
        """Test that explicit None uses env var."""
        with patch.dict(os.environ, {"JAATO_PROVIDER": "github_models"}):
            client = JaatoClient(provider_name=None)
            assert client.provider_name == "github_models"

    def test_not_connected_initially(self):
        """Test that client is not connected initially."""
        client = JaatoClient()
        assert not client.is_connected

    def test_model_name_none_initially(self):
        """Test that model_name is None initially."""
        client = JaatoClient()
        assert client.model_name is None


class TestJaatoClientConnect:
    """Tests for JaatoClient.connect()."""

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_requires_model(self, mock_runtime_class):
        """Test that connect requires model parameter."""
        client = JaatoClient()

        with pytest.raises(ValueError, match="model is required"):
            client.connect()

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_sets_model_name(self, mock_runtime_class):
        """Test that connect sets model_name."""
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        client = JaatoClient()
        client.connect(model="gemini-2.5-flash")

        assert client.model_name == "gemini-2.5-flash"

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_creates_runtime_with_provider(self, mock_runtime_class):
        """Test that connect creates runtime with correct provider."""
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        client = JaatoClient(provider_name="github_models")
        client.connect(model="gpt-4o")

        mock_runtime_class.assert_called_once_with(provider_name="github_models")

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_with_project_and_location(self, mock_runtime_class):
        """Test that connect passes project and location to runtime."""
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        client = JaatoClient()
        client.connect(project="my-project", location="us-central1", model="gemini-2.5-flash")

        mock_runtime.connect.assert_called_once_with("my-project", "us-central1")


class TestJaatoClientProviderProperty:
    """Tests for JaatoClient.provider_name property."""

    def test_provider_name_returns_configured_provider(self):
        """Test that provider_name returns the configured provider."""
        client = JaatoClient(provider_name="anthropic")
        assert client.provider_name == "anthropic"

    def test_provider_name_returns_default_provider(self):
        """Test that provider_name returns default when not specified."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            client = JaatoClient()
            assert client.provider_name == "google_genai"


class TestDefaultProviderConstant:
    """Tests for DEFAULT_PROVIDER constant (backwards compatibility)."""

    def test_default_provider_is_google_genai(self):
        """Test that DEFAULT_PROVIDER constant is google_genai."""
        assert DEFAULT_PROVIDER == "google_genai"
