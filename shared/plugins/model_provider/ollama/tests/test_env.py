"""Tests for Ollama environment variable resolution."""

import pytest
from unittest.mock import patch

from ..env import (
    DEFAULT_OLLAMA_HOST,
    resolve_host,
    resolve_model,
    resolve_context_length,
)


class TestResolveHost:
    """Tests for resolve_host()."""

    def test_default_host(self):
        """Should return default host when env var not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_host() == DEFAULT_OLLAMA_HOST

    def test_host_from_env(self):
        """Should return host from OLLAMA_HOST env var."""
        with patch.dict('os.environ', {'OLLAMA_HOST': 'http://custom:11434'}):
            assert resolve_host() == "http://custom:11434"


class TestResolveModel:
    """Tests for resolve_model()."""

    def test_no_model(self):
        """Should return None when env var not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_model() is None

    def test_model_from_env(self):
        """Should return model from OLLAMA_MODEL env var."""
        with patch.dict('os.environ', {'OLLAMA_MODEL': 'qwen3:32b'}):
            assert resolve_model() == "qwen3:32b"


class TestResolveContextLength:
    """Tests for resolve_context_length()."""

    def test_no_context_length(self):
        """Should return None when env var not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_context_length() is None

    def test_context_length_from_env(self):
        """Should return context length from OLLAMA_CONTEXT_LENGTH env var."""
        with patch.dict('os.environ', {'OLLAMA_CONTEXT_LENGTH': '65536'}):
            assert resolve_context_length() == 65536

    def test_invalid_context_length(self):
        """Should return None for invalid context length."""
        with patch.dict('os.environ', {'OLLAMA_CONTEXT_LENGTH': 'invalid'}):
            assert resolve_context_length() is None
