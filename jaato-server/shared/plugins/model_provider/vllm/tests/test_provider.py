"""Tests for the vLLM provider.

Covers:
- Environment resolution (host default, context override, optional token)
- ``initialize()`` probes ``/health`` and surfaces errors as
  ``VLLMConnectionError`` / ``VLLMAuthenticationError``
- ``verify_auth()`` is network-free pre-initialize (CLAUDE.md contract)
- ``connect()`` validates the model against ``/v1/models`` and raises
  ``VLLMModelNotFoundError`` for a mismatch
- ``get_context_limit()`` priority: override > default
- ``_handle_api_error`` distinguishes mid-stream drops from pre-flight
  connection failures (the ``feedback_no_hardcoded_likely_cause...``
  classifier shape)
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.vllm.env import (
    resolve_api_token,
    resolve_context_length,
    resolve_host,
)
from shared.plugins.model_provider.vllm.errors import (
    VLLMAuthenticationError,
    VLLMConnectionError,
    VLLMModelNotFoundError,
)
from shared.plugins.model_provider.vllm.provider import (
    VLLMProvider,
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


def _models_response(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Build a fake GET /v1/models response matching vLLM's shape.

    vLLM's catalog entries (per vLLM stable docs, verified 2026-06-07
    via context7) are ``{id, object, created, owned_by, root, parent,
    permission}`` — ``max_model_len`` is NOT included.
    """
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
                "owned_by": "vllm",
                "root": model_id,
                "parent": None,
                "permission": [],
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


@pytest.fixture(autouse=True)
def _vllm_env_defaults(monkeypatch):
    """Autouse: provide canonical env values so ``initialize()`` doesn't
    fail-fast on missing host/context_length in tests that aren't
    specifically exercising the no-fallback rule.  Tests that need to
    assert the fail-fast behavior call
    ``monkeypatch.delenv(...)`` explicitly in their body to override
    this fixture.
    """
    monkeypatch.setenv("VLLM_HOST", "http://test.local:8000")
    monkeypatch.setenv("VLLM_CONTEXT_LENGTH", "8192")


# ============================================================
# env.py
# ============================================================


class TestEnv:

    def test_host_returns_none_when_env_missing(self, monkeypatch):
        """No hardcoded localhost fallback — ``resolve_host`` returns
        ``None`` when the env var is unset, and the caller
        (``VLLMProvider.initialize``) fails fast.  See the project's
        "no fallback" rule."""
        monkeypatch.delenv("VLLM_HOST", raising=False)
        assert resolve_host() is None

    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_HOST", "http://gpu-box:8000")
        assert resolve_host() == "http://gpu-box:8000"

    def test_context_length_parsed(self, monkeypatch):
        monkeypatch.setenv("VLLM_CONTEXT_LENGTH", "131072")
        assert resolve_context_length() == 131072

    def test_context_length_invalid_is_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_CONTEXT_LENGTH", "not-a-number")
        assert resolve_context_length() is None

    def test_api_token_absent_returns_none(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_TOKEN", raising=False)
        assert resolve_api_token() is None

    def test_api_token_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_TOKEN", "secret-123")
        assert resolve_api_token() == "secret-123"


# ============================================================
# create_provider factory
# ============================================================


class TestFactory:

    def test_create_provider_returns_instance(self):
        provider = create_provider()
        assert isinstance(provider, VLLMProvider)
        assert provider.name == "vllm"
        assert provider.is_connected is False


# ============================================================
# initialize()
# ============================================================


class TestInitialize:

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_reads_host_from_extra(self, mock_get, monkeypatch):
        monkeypatch.delenv("VLLM_HOST", raising=False)
        mock_get.return_value = _health_response()

        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://gpu-box:8000"}))

        assert provider._host == "http://gpu-box:8000"
        assert not provider._host.endswith("/")

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_strips_trailing_slash(self, mock_get):
        mock_get.return_value = _health_response()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://gpu-box:8000/"}))
        assert provider._host == "http://gpu-box:8000"

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_reads_max_tokens_from_extra(self, mock_get):
        """``extra.max_tokens`` populates ``provider._max_tokens`` so the
        completion call forwards it as the OpenAI ``max_tokens`` field.

        Mirrors the trtllm-serve cap pattern — set this knob when the
        per-request output budget under ``--max-model-len`` would
        otherwise drop the engine mid-stream.
        """
        mock_get.return_value = _health_response()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={"max_tokens": 4096}))
        assert provider._max_tokens == 4096

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_max_tokens_stays_none_when_absent(self, mock_get):
        """When ``extra.max_tokens`` is not set, ``provider._max_tokens``
        stays ``None`` so the completion call does NOT carry a
        ``max_tokens`` field — letting vLLM apply its own default
        (typically bounded by ``--max-model-len`` minus prompt).
        """
        mock_get.return_value = _health_response()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        assert provider._max_tokens is None

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_connection_failure_raises(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        provider = VLLMProvider()
        with pytest.raises(VLLMConnectionError):
            provider.initialize(ProviderConfig(extra={"host": "http://offline:8000"}))

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_timeout_raises_connection_error(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        provider = VLLMProvider()
        with pytest.raises(VLLMConnectionError):
            provider.initialize(ProviderConfig())

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_401_raises_auth_error(self, mock_get):
        """vLLM's native ``--api-key <token>`` flag rejects bad bearers
        with HTTP 401; this should surface as VLLMAuthenticationError.
        """
        mock_response = MagicMock(status_code=401)
        mock_get.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response,
        )
        provider = VLLMProvider()
        with pytest.raises(VLLMAuthenticationError):
            provider.initialize(ProviderConfig(extra={"api_token": "bad-token"}))

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_probes_health_endpoint(self, mock_get):
        mock_get.return_value = _health_response()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={"host": "http://srv:8000"}))
        # The single httpx.get call must hit /health, not /v1/models —
        # /health is the cheap liveness probe.
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://srv:8000/health"

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_initialize_forwards_bearer_to_health_probe(self, mock_get):
        mock_get.return_value = _health_response()
        provider = VLLMProvider()
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
        monkeypatch.delenv("VLLM_API_TOKEN", raising=False)
        provider = VLLMProvider()
        msgs = []
        ok = provider.verify_auth(on_message=msgs.append)
        assert ok is True
        assert any("no authentication required" in m for m in msgs)

    def test_does_not_make_any_http_request(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_TOKEN", raising=False)
        provider = VLLMProvider()
        with patch(
            "shared.plugins.model_provider.vllm.provider.httpx.get"
        ) as gget, patch(
            "shared.plugins.model_provider.vllm.provider.httpx.post"
        ) as gpost:
            provider.verify_auth()
            assert gget.call_count == 0
            assert gpost.call_count == 0

    def test_reports_token_when_supplied_via_config(self):
        provider = VLLMProvider()
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
        monkeypatch.setenv("VLLM_API_TOKEN", "env-token-xxxx")
        provider = VLLMProvider()
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

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_connect_records_model_name(self, mock_get):
        mock_get.side_effect = _route_get(
            models_resp=_models_response("Qwen/Qwen2.5-7B-Instruct"),
        )
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("Qwen/Qwen2.5-7B-Instruct")
        assert provider.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert provider.is_connected is True

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_connect_unknown_model_raises(self, mock_get):
        mock_get.side_effect = _route_get(
            models_resp=_models_response("different-model"),
        )
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        with pytest.raises(VLLMModelNotFoundError) as excinfo:
            provider.connect("Qwen/Qwen2.5-7B-Instruct")
        assert "different-model" in str(excinfo.value)

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_connect_skip_model_test_bypasses_catalog(self, mock_get):
        """skip_model_test=True must not hit /v1/models."""
        mock_get.return_value = _health_response()  # only /health probed
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("any-model", skip_model_test=True)
        assert provider.model_name == "any-model"
        # Only the initialize() health probe was called — no /v1/models GET.
        for call in mock_get.call_args_list:
            assert "/v1/models" not in call[0][0]

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_connect_with_empty_catalog_skips_validation(self, mock_get):
        """An empty catalog (transient blip) is treated as 'skip validation'
        rather than 'model not found' — see _fetch_catalog docstring."""
        empty_models = MagicMock()
        empty_models.status_code = 200
        empty_models.raise_for_status = MagicMock()
        empty_models.json.return_value = {"object": "list", "data": []}
        mock_get.side_effect = _route_get(models_resp=empty_models)
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("Qwen/Qwen2.5-7B-Instruct")
        assert provider.model_name == "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# get_context_limit priority
# ============================================================


class TestContextLimit:
    """vLLM's /v1/models doesn't surface max_model_len, so the only
    sources are explicit override and the conservative default."""

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_override_wins(self, mock_get):
        mock_get.side_effect = _route_get()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))
        provider.connect("Qwen/Qwen2.5-7B-Instruct")
        assert provider.get_context_limit() == 131072

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_env_value_used_when_no_config_override(self, mock_get):
        """When ``ProviderConfig`` has no ``context_length`` override,
        the value flows from ``VLLM_CONTEXT_LENGTH`` (autouse fixture
        sets it to 8192).  No hardcoded fallback inside the provider."""
        mock_get.side_effect = _route_get()
        provider = VLLMProvider()
        provider.initialize(ProviderConfig())
        provider.connect("Qwen/Qwen2.5-7B-Instruct")
        assert provider.get_context_limit() == 8192

    def test_initialize_fails_fast_when_context_length_missing(
        self, monkeypatch,
    ):
        """No hardcoded fallback for context_length — ``initialize``
        raises ``ValueError`` when neither config nor env provides one."""
        monkeypatch.delenv("VLLM_CONTEXT_LENGTH", raising=False)
        provider = VLLMProvider()
        with pytest.raises(ValueError, match="context_length is not configured"):
            provider.initialize(ProviderConfig())

    def test_initialize_fails_fast_when_host_missing(self, monkeypatch):
        """No hardcoded fallback for host — ``initialize`` raises
        ``ValueError`` when neither config nor env provides one."""
        monkeypatch.delenv("VLLM_HOST", raising=False)
        provider = VLLMProvider()
        with pytest.raises(ValueError, match="host is not configured"):
            provider.initialize(ProviderConfig())

    def test_get_context_limit_raises_before_initialize(self):
        """``get_context_limit`` raises ``RuntimeError`` when called on
        an uninitialized provider — no silent fallback to a hardcoded
        default."""
        provider = VLLMProvider()
        with pytest.raises(RuntimeError, match="before initialize"):
            provider.get_context_limit()


# ============================================================
# list_models
# ============================================================


class TestListModels:

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_list_models_returns_engine_id(self, mock_get):
        mock_get.return_value = _models_response("Qwen/Qwen2.5-7B-Instruct")
        provider = VLLMProvider()
        provider._host = "http://srv:8000"
        names = provider.list_models()
        assert names == ["Qwen/Qwen2.5-7B-Instruct"]

    @patch("shared.plugins.model_provider.vllm.provider.httpx.get")
    def test_list_models_prefix_filter(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "Qwen/Qwen2.5-7B-Instruct"},
                {"id": "meta-llama/Llama-3.1-8B-Instruct"},
            ],
        }
        mock_get.return_value = resp
        provider = VLLMProvider()
        provider._host = "http://srv:8000"
        names = provider.list_models(prefix="Qwen/")
        assert names == ["Qwen/Qwen2.5-7B-Instruct"]


# ============================================================
# Capabilities
# ============================================================


class TestCapabilities:

    def test_supports_streaming(self):
        assert VLLMProvider().supports_streaming() is True

    def test_supports_stop(self):
        assert VLLMProvider().supports_stop() is True

    def test_supports_structured_output(self):
        assert VLLMProvider().supports_structured_output() is True

    def test_does_not_advertise_thinking(self):
        """vLLM's /v1 surface does not expose reasoning by default —
        the provider does not extract the optional ``message.reasoning``
        field that ``--reasoning-parser`` produces for reasoning models.
        """
        assert VLLMProvider().supports_thinking() is False


# ============================================================
# Error handling — mid-stream vs pre-flight connection failures
# ============================================================


class TestMidStreamErrorDistinction:
    """``_handle_api_error`` must distinguish a mid-stream connection drop
    (engine-error AFTER HTTP 200 — e.g. prompt exceeds
    ``--max-model-len``, KV-cache OOM) from a pre-flight connection
    failure (host unreachable / firewall / DNS).  Symmetric to the
    trtllm/triton mid-stream classifiers — surfacing both as
    ``VLLMConnectionError`` misroutes the user toward network
    debugging when the real bug is engine config.
    """

    def _make_apicon_error(self, message: str):
        """Construct an ``openai.APIConnectionError`` whose ``str()``
        carries the given text — that's the substring
        ``_handle_api_error`` regex-matches on.
        """
        from shared.plugins.model_provider.vllm.provider import (
            get_openai_module,
        )
        openai = get_openai_module()
        try:
            return openai.APIConnectionError(message=message, request=None)
        except TypeError:
            return openai.APIConnectionError(message)

    def test_remote_protocol_error_routes_to_mid_stream(self):
        from shared.plugins.model_provider.vllm.errors import (
            VLLMMidStreamError,
        )
        provider = VLLMProvider()
        provider._host = "http://192.168.50.154:8000"
        err = self._make_apicon_error(
            "RemoteProtocolError: peer closed connection without "
            "sending complete message body (incomplete chunked read)"
        )
        with pytest.raises(VLLMMidStreamError) as ctx:
            provider._handle_api_error(err)
        msg = str(ctx.value)
        assert "192.168.50.154:8000" in msg
        # Per feedback_no_hardcoded_likely_cause_in_error_messages.md,
        # the message points at the server log instead of guessing at
        # the cause — make sure the log-pointer phrasing is present
        # and NO causal-guess phrasing is.
        assert "vLLM server log" in msg

    def test_mid_stream_message_does_not_guess_at_cause(self):
        """Falsification guard for the no-hardcoded-likely-cause rule
        (``feedback_no_hardcoded_likely_cause_in_error_messages.md``).

        Uses the shared ``assert_no_likely_cause_prose`` helper from
        ``shared/plugins/model_provider/conftest.py`` so the forbidden
        keyword list is enforced identically across all self-hosted
        OpenAI-compatible providers (tensorrt_llm, triton, vllm).
        When a fourth such provider lands, point its mid-stream test
        at the same helper.
        """
        from shared.plugins.model_provider.conftest import (
            assert_no_likely_cause_prose,
        )
        from shared.plugins.model_provider.vllm.errors import (
            VLLMMidStreamError,
        )
        err = VLLMMidStreamError("http://srv:8000", original_error="x")
        assert_no_likely_cause_prose(str(err))

    def test_clean_connection_failure_still_routes_to_connection_error(self):
        from shared.plugins.model_provider.vllm.errors import (
            VLLMConnectionError,
            VLLMMidStreamError,
        )
        provider = VLLMProvider()
        provider._host = "http://192.168.50.154:8000"
        err = self._make_apicon_error(
            "Connection refused at http://192.168.50.154:8000"
        )
        with pytest.raises(VLLMConnectionError) as ctx:
            provider._handle_api_error(err)
        assert not isinstance(ctx.value, VLLMMidStreamError)
        msg = str(ctx.value)
        assert "Cannot connect" in msg
