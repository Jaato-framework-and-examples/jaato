"""Tests for JaatoClient - core client for the jaato framework."""

import os
import pytest
from unittest.mock import MagicMock, patch

from ..jaato_client import JaatoClient, get_default_provider, get_default_model


class TestGetDefaultProvider:
    """Tests for get_default_provider() function."""

    def test_returns_none_when_no_env(self):
        """Test that None is returned when JAATO_PROVIDER is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            assert get_default_provider() is None

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

    def test_init_no_args_no_env_provider_is_none(self):
        """Test that init with no args and no env var gives None provider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            client = JaatoClient()
            assert client.provider_name is None

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
        client = JaatoClient(provider_name="google_genai")

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MODEL_NAME", None)
            with pytest.raises(ValueError, match="model is required"):
                client.connect()

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_uses_model_from_env(self, mock_runtime_class):
        """Test that connect falls back to MODEL_NAME env var."""
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        client = JaatoClient(provider_name="google_genai")
        with patch.dict(os.environ, {"MODEL_NAME": "gemini-2.5-pro"}):
            client.connect()

        assert client.model_name == "gemini-2.5-pro"

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_requires_provider(self, mock_runtime_class):
        """Test that connect requires provider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            client = JaatoClient()

            with pytest.raises(ValueError, match="provider is required"):
                client.connect(model="gemini-2.5-flash")

    @patch('shared.jaato_client.JaatoRuntime')
    def test_connect_sets_model_name(self, mock_runtime_class):
        """Test that connect sets model_name."""
        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        client = JaatoClient(provider_name="google_genai")
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

        client = JaatoClient(provider_name="google_genai")
        client.connect(project="my-project", location="us-central1", model="gemini-2.5-flash")

        mock_runtime.connect.assert_called_once_with("my-project", "us-central1")


class TestJaatoClientProviderProperty:
    """Tests for JaatoClient.provider_name property."""

    def test_provider_name_returns_configured_provider(self):
        """Test that provider_name returns the configured provider."""
        client = JaatoClient(provider_name="anthropic")
        assert client.provider_name == "anthropic"

    def test_provider_name_returns_none_when_not_specified(self):
        """Test that provider_name returns None when not specified and no env."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JAATO_PROVIDER", None)
            client = JaatoClient()
            assert client.provider_name is None


class TestGetDefaultModel:
    """Tests for get_default_model() function."""

    def test_returns_none_when_no_env(self):
        """Test that None is returned when MODEL_NAME is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MODEL_NAME", None)
            assert get_default_model() is None

    def test_returns_env_value_when_set(self):
        """Test that MODEL_NAME env var is returned when set."""
        with patch.dict(os.environ, {"MODEL_NAME": "gemini-2.5-flash"}):
            assert get_default_model() == "gemini-2.5-flash"
