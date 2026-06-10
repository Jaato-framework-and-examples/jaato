"""Tier-1 context-window auto-detection (2026-06-10 design).

Generic precedence (``base.resolve_context_window``):
  1. PRIMARY  — provider auto-detect from its own capacity endpoint
     (optional per-provider hook; vLLM reads ``max_model_len`` from
     ``GET /v1/models``).
  2. fallback — per-profile ``context_length`` knob.
  3. fallback — provider env var.

Closes the staleness class behind the codegen context-overflow 400: a
stale per-stage profile ``context_length`` (e.g. 40944) can no longer
silently under-declare a window the server actually serves at 71680.
Live evidence: ``GET /v1/models`` returns ``data[0].max_model_len=71680``
for the running qwen3-14b-vllm server (the provider's old "does not
surface max_model_len" comment was stale for current vLLM).
"""

from unittest.mock import patch

import pytest

from shared.plugins.model_provider.base import (
    ProviderConfig,
    resolve_context_window,
)
from shared.plugins.model_provider.vllm.provider import VLLMProvider
from shared.plugins.model_provider.vllm.tests.test_provider import (
    _route_get,
    _models_response,
)


# ============================================================
# Generic precedence helper
# ============================================================


class TestResolveContextWindow:

    def test_detect_wins_over_knob_and_env(self):
        assert resolve_context_window(
            detect_capacity=lambda: 71680,
            profile_value=40944,
            env_value=8192,
        ) == 71680

    def test_knob_used_when_detect_returns_none(self):
        assert resolve_context_window(
            detect_capacity=lambda: None,
            profile_value=40944,
            env_value=8192,
        ) == 40944

    def test_env_used_when_detect_none_and_no_knob(self):
        assert resolve_context_window(
            detect_capacity=lambda: None,
            profile_value=None,
            env_value=8192,
        ) == 8192

    def test_none_when_all_absent(self):
        assert resolve_context_window(
            detect_capacity=lambda: None,
            profile_value=None,
            env_value=None,
        ) is None

    def test_no_detect_hook_falls_through_to_knob(self):
        # Providers without a capacity endpoint omit the hook entirely —
        # tier-1 is a no-op and resolution is knob -> env, as before.
        assert resolve_context_window(
            profile_value=40944,
            env_value=8192,
        ) == 40944

    def test_zero_detect_is_treated_as_absent(self):
        # A falsy (0) detection must not win — fall through to the knob.
        assert resolve_context_window(
            detect_capacity=lambda: 0,
            profile_value=40944,
        ) == 40944


# ============================================================
# vLLM tier-1 hook: _detect_context_capacity
# ============================================================


class TestDetectContextCapacity:

    def _provider(self):
        p = VLLMProvider()
        p._host = "http://srv:8000"
        return p

    def test_returns_max_model_len_from_catalog(self):
        p = self._provider()
        with patch.object(
            p, "_fetch_catalog",
            return_value=[{"id": "qwen3-14b-vllm", "max_model_len": 71680}],
        ):
            assert p._detect_context_capacity() == 71680

    def test_none_when_catalog_empty(self):
        p = self._provider()
        with patch.object(p, "_fetch_catalog", return_value=[]):
            assert p._detect_context_capacity() is None

    def test_none_when_field_absent(self):
        p = self._provider()
        with patch.object(
            p, "_fetch_catalog", return_value=[{"id": "m", "object": "model"}],
        ):
            assert p._detect_context_capacity() is None

    def test_prefers_id_match_among_multiple(self):
        p = self._provider()
        p._model_name = "wanted"
        catalog = [
            {"id": "other", "max_model_len": 100},
            {"id": "wanted", "max_model_len": 71680},
        ]
        with patch.object(p, "_fetch_catalog", return_value=catalog):
            assert p._detect_context_capacity() == 71680

    def test_falls_back_to_first_entry_when_model_unknown(self):
        p = self._provider()
        p._model_name = None
        with patch.object(
            p, "_fetch_catalog",
            return_value=[{"id": "first", "max_model_len": 65520}],
        ):
            assert p._detect_context_capacity() == 65520


# ============================================================
# Integration: initialize() resolution precedence
# ============================================================


class TestInitializeContextResolution:

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_detected_max_model_len_wins_over_profile_knob(
        self, mock_get, monkeypatch,
    ):
        """The codegen-400 regression: a stale profile context_length must
        NOT under-declare the window when the server reports max_model_len.
        """
        monkeypatch.setenv("VLLM_CONTEXT_LENGTH", "8192")
        mock_get.side_effect = _route_get(
            models_resp=_models_response(
                model_id="qwen3-14b-vllm", max_model_len=71680,
            ),
        )
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={
            "host": "http://srv:8000",
            "context_length": 40944,  # stale per-stage profile value
        }))
        assert provider.get_context_limit() == 71680

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_falls_back_to_profile_knob_when_server_omits_field(
        self, mock_get, monkeypatch,
    ):
        """Older vLLM that doesn't surface max_model_len → the manual
        context_length knob is honoured (tier-2)."""
        monkeypatch.delenv("VLLM_CONTEXT_LENGTH", raising=False)
        mock_get.side_effect = _route_get(
            models_resp=_models_response(model_id="m"),  # no max_model_len
        )
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={
            "host": "http://srv:8000",
            "context_length": 40944,
        }))
        assert provider.get_context_limit() == 40944

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_raises_when_neither_server_nor_manual_override(
        self, mock_get, monkeypatch,
    ):
        """No max_model_len, no knob, no env → fail-fast (no hardcoded
        fallback)."""
        monkeypatch.delenv("VLLM_CONTEXT_LENGTH", raising=False)
        mock_get.side_effect = _route_get(
            models_resp=_models_response(model_id="m"),  # no max_model_len
        )
        provider = VLLMProvider()
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.initialize(ProviderConfig(extra={"host": "http://srv:8000"}))
