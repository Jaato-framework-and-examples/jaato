"""Tests for the TensorRT-LLM provider.

Covers:
- Environment resolution (host default, context override, optional token)
- ``initialize()`` probes ``/health`` and surfaces errors as
  ``TensorRTLLMConnectionError`` / ``TensorRTLLMAuthenticationError``
- ``verify_auth()`` is network-free pre-initialize (CLAUDE.md contract)
- ``connect()`` validates the model against ``/v1/models`` and raises
  ``TensorRTLLMModelNotFoundError`` for a mismatch
- ``get_context_limit()`` priority: override > default
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.tensorrt_llm.env import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
)
from shared.plugins.model_provider.tensorrt_llm.errors import (
    TensorRTLLMAuthenticationError,
    TensorRTLLMConnectionError,
    TensorRTLLMModelNotFoundError,
)
from shared.plugins.model_provider.tensorrt_llm.provider import (
    TensorRTLLMProvider,
    create_provider,
)


# ============================================================
# Fixtures
# ============================================================


def _health_response(status: int = 200):
    """Build a fake GET /health response."""
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}",
            request=MagicMock(),
            response=MagicMock(status_code=status),
        )
    return resp


def _models_response(model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Build a fake GET /v1/models response with one engine."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "trtllm",
            }
        ],
    }
    return resp


def _route_get(*, health_resp=None, models_resp=None):
    """Build a side_effect for httpx.get that routes by URL."""
    def _route(url, *args, **kwargs):
        if "/v1/models" in url:
            return models_resp if models_resp is not None else _models_response()
        if "/health" in url:
            return health_resp if health_resp is not None else _health_response()
        raise AssertionError(f"Unexpected GET: {url}")
    return _route


# ============================================================
# env.py
# ============================================================


class TestEnv:

    def test_host_default_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("TENSORRT_LLM_HOST", raising=False)
        assert resolve_host() == DEFAULT_HOST

    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LLM_HOST", "http://gpu-box:8000")
        assert resolve_host() == "http://gpu-box:8000"

    def test_context_length_parsed(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LLM_CONTEXT_LENGTH", "131072")
        assert resolve_context_length() == 131072

    def test_context_length_invalid_is_none(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LLM_CONTEXT_LENGTH", "not-a-number")
        assert resolve_context_length() is None

    def test_api_token_absent_returns_none(self, monkeypatch):
        monkeypatch.delenv("TENSORRT_LLM_API_TOKEN", raising=False)
        assert resolve_api_token() is None

    def test_api_token_from_env(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LLM_API_TOKEN", "secret-123")
        assert resolve_api_token() == "secret-123"


# ============================================================
# create_provider factory
# ============================================================


class TestFactory:

    def test_create_provider_returns_instance(self):
        provider = create_provider()
        assert isinstance(provider, TensorRTLLMProvider)
        assert provider.name == "tensorrt_llm"
        assert provider.is_connected is False


# ============================================================
# initialize()
# ============================================================


class TestInitialize:

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_reads_host_from_extra(self, mock_get, monkeypatch):
        monkeypatch.delenv("TENSORRT_LLM_HOST", raising=False)
        mock_get.return_value = _health_response()

        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://gpu-box:8000"}))

        assert provider._host == "http://gpu-box:8000"
        # Trailing slash stripped
        assert not provider._host.endswith("/")

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_strips_trailing_slash(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://gpu-box:8000/"}))
        assert provider._host == "http://gpu-box:8000"

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_reads_max_tokens_from_extra(self, mock_get):
        """``extra.max_tokens`` populates ``provider._max_tokens`` so the
        completion call forwards it as the OpenAI ``max_tokens`` field.

        Without this knob trtllm-serve defaults the per-request output
        budget to (``max_seq_len`` - prompt) which exhausts KV-cache
        under sustained generation, surfacing as a mid-stream
        ``RemoteProtocolError: peer closed connection``.  Empirically
        demonstrated by the kb-orchestrator cascade hit on the
        WSL2 trtllm-serve endpoint (2026-06-06).
        """
        mock_get.return_value = _health_response()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"max_tokens": 4096}))
        assert provider._max_tokens == 4096

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_max_tokens_stays_none_when_absent(self, mock_get):
        """When ``extra.max_tokens`` is not set, ``provider._max_tokens``
        stays ``None`` so the completion call does NOT carry a
        ``max_tokens`` field — letting trtllm-serve apply its own
        default.  Mirrors the temperature / top_p contract elsewhere
        in the provider stack.
        """
        mock_get.return_value = _health_response()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        assert provider._max_tokens is None

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_connection_failure_raises(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        provider = TensorRTLLMProvider()
        with pytest.raises(TensorRTLLMConnectionError):
            provider.initialize(ProviderConfig(extra={"host": "http://offline:8000"}))

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_timeout_raises_connection_error(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        provider = TensorRTLLMProvider()
        with pytest.raises(TensorRTLLMConnectionError):
            provider.initialize(ProviderConfig())

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_401_raises_auth_error(self, mock_get):
        mock_response = MagicMock(status_code=401)
        mock_get.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response,
        )
        provider = TensorRTLLMProvider()
        with pytest.raises(TensorRTLLMAuthenticationError):
            provider.initialize(ProviderConfig(extra={"api_token": "bad-token"}))

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_probes_health_endpoint(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://srv:8000"}))
        # The single httpx.get call must hit /health, not /v1/models —
        # /health is the cheap liveness probe.
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://srv:8000/health"

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_initialize_forwards_bearer_to_health_probe(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"api_token": "tok-abc"}))
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok-abc"


# ============================================================
# verify_auth — must NOT touch the network
# ============================================================


class TestVerifyAuth:
    """verify_auth must satisfy the contract in shared/plugins/CLAUDE.md:
    runs on a fresh, uninitialized provider; checks only credential
    availability; never touches the network."""

    def test_returns_true_with_no_credentials_required(self, monkeypatch):
        monkeypatch.delenv("TENSORRT_LLM_API_TOKEN", raising=False)
        provider = TensorRTLLMProvider()
        msgs = []
        ok = provider.verify_auth(on_message=msgs.append)
        assert ok is True
        assert any("no authentication required" in m for m in msgs)

    def test_does_not_make_any_http_request(self, monkeypatch):
        monkeypatch.delenv("TENSORRT_LLM_API_TOKEN", raising=False)
        provider = TensorRTLLMProvider()
        with patch(
            "shared.plugins.model_provider.tensorrt_llm.provider.httpx.get"
        ) as gget, patch(
            "shared.plugins.model_provider.tensorrt_llm.provider.httpx.post"
        ) as gpost:
            provider.verify_auth()
            assert gget.call_count == 0
            assert gpost.call_count == 0

    def test_reports_token_when_supplied_via_config(self):
        provider = TensorRTLLMProvider()
        msgs = []
        ok = provider.verify_auth(
            on_message=msgs.append,
            config=ProviderConfig(extra={"api_token": "secret-token-abc-1234"}),
        )
        assert ok is True
        assert any("bearer token configured" in m for m in msgs)
        # Token is masked, never echoed in full
        assert all("secret-token-abc-1234" not in m for m in msgs)

    def test_config_token_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LLM_API_TOKEN", "env-token-xxxx")
        provider = TensorRTLLMProvider()
        msgs = []
        provider.verify_auth(
            on_message=msgs.append,
            config=ProviderConfig(extra={"api_token": "profile-token-yyyy"}),
        )
        # Mask shows profile token's prefix/suffix, not env's
        assert any("prof…yyyy" in m for m in msgs)


# ============================================================
# connect()
# ============================================================


class TestConnect:

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_connect_records_model_name(self, mock_get):
        mock_get.side_effect = _route_get(
            models_resp=_models_response("meta-llama/Llama-3.1-8B-Instruct"),
        )
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("meta-llama/Llama-3.1-8B-Instruct")
        assert provider.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert provider.is_connected is True

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_connect_unknown_model_raises(self, mock_get):
        mock_get.side_effect = _route_get(
            models_resp=_models_response("different-model"),
        )
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        with pytest.raises(TensorRTLLMModelNotFoundError) as excinfo:
            provider.connect("meta-llama/Llama-3.1-8B-Instruct")
        # Error message should mention the engine the server *is* hosting.
        assert "different-model" in str(excinfo.value)

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_connect_skip_model_test_bypasses_catalog(self, mock_get):
        """skip_model_test=True must not hit /v1/models."""
        mock_get.return_value = _health_response()  # only /health probed
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("any-model", skip_model_test=True)
        assert provider.model_name == "any-model"
        # Only the initialize() health probe was called — no /v1/models GET.
        for call in mock_get.call_args_list:
            assert "/v1/models" not in call[0][0]

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_connect_with_empty_catalog_skips_validation(self, mock_get):
        """An empty catalog (transient blip) is treated as 'skip validation'
        rather than 'model not found' — see _fetch_catalog docstring."""
        empty_models = MagicMock()
        empty_models.status_code = 200
        empty_models.raise_for_status = MagicMock()
        empty_models.json.return_value = {"object": "list", "data": []}
        mock_get.side_effect = _route_get(models_resp=empty_models)
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("meta-llama/Llama-3.1-8B-Instruct")
        assert provider.model_name == "meta-llama/Llama-3.1-8B-Instruct"


# ============================================================
# get_context_limit priority
# ============================================================


class TestContextLimit:
    """trtllm-serve's /v1/models doesn't surface max_seq_len, so the only
    sources are explicit override and the conservative default."""

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_override_wins(self, mock_get):
        mock_get.side_effect = _route_get()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))
        provider.connect("meta-llama/Llama-3.1-8B-Instruct")
        assert provider.get_context_limit() == 131072

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_default_used_when_no_override(self, mock_get):
        mock_get.side_effect = _route_get()
        provider = TensorRTLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("meta-llama/Llama-3.1-8B-Instruct")
        assert provider.get_context_limit() == DEFAULT_CONTEXT_LENGTH

    def test_default_used_before_connect(self):
        provider = TensorRTLLMProvider()
        assert provider.get_context_limit() == DEFAULT_CONTEXT_LENGTH


# ============================================================
# list_models
# ============================================================


class TestListModels:

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_list_models_returns_engine_id(self, mock_get):
        mock_get.return_value = _models_response("meta-llama/Llama-3.1-8B-Instruct")
        provider = TensorRTLLMProvider()
        provider.initialize.__wrapped__ if False else None  # silence linter

        # Skip initialize() since list_models() only needs the host.
        provider._host = "http://srv:8000"
        names = provider.list_models()
        assert names == ["meta-llama/Llama-3.1-8B-Instruct"]

    @patch("shared.plugins.model_provider.tensorrt_llm.provider.httpx.get")
    def test_list_models_prefix_filter(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "meta-llama/Llama-3.1-8B-Instruct"},
                {"id": "Qwen/Qwen2.5-32B"},
            ],
        }
        mock_get.return_value = resp
        provider = TensorRTLLMProvider()
        provider._host = "http://srv:8000"
        names = provider.list_models(prefix="meta-llama/")
        assert names == ["meta-llama/Llama-3.1-8B-Instruct"]


# ============================================================
# Capabilities
# ============================================================


class TestCapabilities:

    def test_supports_streaming(self):
        assert TensorRTLLMProvider().supports_streaming() is True

    def test_supports_stop(self):
        assert TensorRTLLMProvider().supports_stop() is True

    def test_supports_structured_output(self):
        assert TensorRTLLMProvider().supports_structured_output() is True

    def test_does_not_advertise_thinking(self):
        """trtllm-serve's /v1 surface has no reasoning delta channel."""
        assert TensorRTLLMProvider().supports_thinking() is False


# ============================================================
# Error handling — mid-stream vs pre-flight connection failures
# ============================================================


class TestMidStreamErrorDistinction:
    """``_handle_api_error`` must distinguish a mid-stream connection drop
    (engine-error AFTER HTTP 200 — e.g. prompt exceeds engine
    ``max_input_length``, KV-cache OOM) from a pre-flight connection
    failure (host unreachable / firewall / DNS).  The two have completely
    different fix trees — surfacing both as ``TensorRTLLMConnectionError``
    misroutes the user toward network debugging when the real bug is
    engine config.  Empirically demonstrated by the 2026-06-06
    kb-orchestrator cascade incident (prompt 19,977 tokens against
    engine ``max_input_length=16384``).
    """

    def _make_apicon_error(self, message: str):
        """Construct an ``openai.APIConnectionError`` whose ``str()`` carries
        the given text — that's the substring ``_handle_api_error``
        regex-matches on.
        """
        from shared.plugins.model_provider.tensorrt_llm.provider import (
            get_openai_module,
        )
        openai = get_openai_module()
        try:
            return openai.APIConnectionError(message=message, request=None)
        except TypeError:
            return openai.APIConnectionError(message)

    def test_remote_protocol_error_routes_to_mid_stream(self):
        from shared.plugins.model_provider.tensorrt_llm.errors import (
            TensorRTLLMMidStreamError,
        )
        provider = TensorRTLLMProvider()
        provider._host = "http://192.168.50.154:8000"
        err = self._make_apicon_error(
            "RemoteProtocolError: peer closed connection without "
            "sending complete message body (incomplete chunked read)"
        )
        with pytest.raises(TensorRTLLMMidStreamError) as ctx:
            provider._handle_api_error(err)
        msg = str(ctx.value)
        assert "192.168.50.154:8000" in msg
        assert "max_input_length" in msg
        assert "trtllm-build" in msg

    def test_clean_connection_failure_still_routes_to_connection_error(self):
        from shared.plugins.model_provider.tensorrt_llm.errors import (
            TensorRTLLMConnectionError,
            TensorRTLLMMidStreamError,
        )
        provider = TensorRTLLMProvider()
        provider._host = "http://192.168.50.154:8000"
        err = self._make_apicon_error(
            "Connection refused at http://192.168.50.154:8000"
        )
        with pytest.raises(TensorRTLLMConnectionError) as ctx:
            provider._handle_api_error(err)
        assert not isinstance(ctx.value, TensorRTLLMMidStreamError)
        msg = str(ctx.value)
        assert "Cannot connect" in msg
        assert "max_input_length" not in msg
