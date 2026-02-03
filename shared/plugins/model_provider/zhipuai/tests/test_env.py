"""Tests for Zhipu AI environment resolution."""

import pytest
from unittest.mock import patch

from ..env import (
    DEFAULT_ZHIPUAI_BASE_URL,
    DEFAULT_ZHIPUAI_MODEL,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_model,
)


class TestResolveApiKey:
    """Tests for API key resolution."""

    def test_no_env_var(self):
        """Should return None when ZHIPUAI_API_KEY not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_api_key() is None

    def test_from_env_var(self):
        """Should return API key from ZHIPUAI_API_KEY."""
        with patch.dict('os.environ', {'ZHIPUAI_API_KEY': 'test-key-123'}):
            assert resolve_api_key() == 'test-key-123'


class TestResolveBaseUrl:
    """Tests for base URL resolution."""

    def test_default_url(self):
        """Should return default URL when ZHIPUAI_BASE_URL not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_base_url() == DEFAULT_ZHIPUAI_BASE_URL

    def test_from_env_var(self):
        """Should return URL from ZHIPUAI_BASE_URL."""
        with patch.dict('os.environ', {'ZHIPUAI_BASE_URL': 'https://custom.api.com'}):
            assert resolve_base_url() == 'https://custom.api.com'


class TestResolveModel:
    """Tests for model resolution."""

    def test_no_env_var(self):
        """Should return None when ZHIPUAI_MODEL not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_model() is None

    def test_from_env_var(self):
        """Should return model from ZHIPUAI_MODEL."""
        with patch.dict('os.environ', {'ZHIPUAI_MODEL': 'glm-4.7-flash'}):
            assert resolve_model() == 'glm-4.7-flash'


class TestResolveContextLength:
    """Tests for context length resolution."""

    def test_no_env_var(self):
        """Should return None when ZHIPUAI_CONTEXT_LENGTH not set."""
        with patch.dict('os.environ', {}, clear=True):
            assert resolve_context_length() is None

    def test_from_env_var(self):
        """Should return context length from ZHIPUAI_CONTEXT_LENGTH."""
        with patch.dict('os.environ', {'ZHIPUAI_CONTEXT_LENGTH': '65536'}):
            assert resolve_context_length() == 65536

    def test_invalid_value(self):
        """Should return None for invalid context length."""
        with patch.dict('os.environ', {'ZHIPUAI_CONTEXT_LENGTH': 'invalid'}):
            assert resolve_context_length() is None


class TestDefaults:
    """Tests for default values."""

    def test_default_base_url(self):
        """Default base URL should be Z.AI's Anthropic endpoint."""
        assert DEFAULT_ZHIPUAI_BASE_URL == "https://api.z.ai/api/anthropic/v1"

    def test_default_model(self):
        """Default model should be glm-4.7."""
        assert DEFAULT_ZHIPUAI_MODEL == "glm-4.7"
