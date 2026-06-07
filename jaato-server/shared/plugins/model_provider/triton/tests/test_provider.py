"""Tests for the Triton provider.

Covers:
- Environment URL resolution (defaults, TRITON_HOST shorthand, explicit
  overrides, mixed env)
- ``initialize()`` probes ``/v2/health/live`` on the control URL
- Passive mode: no ``load`` → no POST to /v2/repository/models/.../load
- Active mode: ``load`` present → POST with expected body and URL
- Unknown model surfaces TritonModelNotFoundError with the repository
  catalog
- Load failure (non-2xx) surfaces TritonLoadError
- ``list_models()`` uses /v2/repository/index (so unloaded models appear)
- ``verify_auth()`` is network-free pre-initialize
- ``get_context_limit()`` priority: override > default
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.triton.env import (
    resolve_api_token,
    resolve_context_length,
    resolve_urls,
)
from shared.plugins.model_provider.triton.errors import (
    TritonAuthenticationError,
    TritonConnectionError,
    TritonLoadError,
    TritonModelNotFoundError,
)
from shared.plugins.model_provider.triton.provider import (
    TritonProvider,
    create_provider,
)


# ============================================================
# Fixtures
# ============================================================


def _health_response(status: int = 200):
    """Build a fake GET /v2/health/live response."""
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


def _index_response(
    models=(("llama-3.1-8b-instruct", "READY"),),
):
    """Build a fake POST /v2/repository/index response.

    Triton returns a top-level JSON array of ``{name, version, state}``
    objects — distinct shape from OpenAI's /v1/models.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [
        {"name": name, "version": "1", "state": state}
        for name, state in models
    ]
    return resp


def _load_response(status: int = 200, body: str = ""):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    return resp


@pytest.fixture(autouse=True)
def _triton_env_defaults(monkeypatch):
    """Autouse: provide canonical env values so ``initialize()`` doesn't
    fail-fast on missing URLs/context_length in tests that aren't
    specifically exercising the no-fallback rule.  Tests that need to
    assert the fail-fast behavior call ``monkeypatch.delenv(...)``
    explicitly in their body to override this fixture.
    """
    monkeypatch.setenv("TRITON_OPENAI_URL", "http://test.local:9000")
    monkeypatch.setenv("TRITON_CONTROL_URL", "http://test.local:8000")
    monkeypatch.setenv("TRITON_CONTEXT_LENGTH", "8192")


# ============================================================
# env.py — URL resolution
# ============================================================


class TestEnvUrls:

    def test_returns_none_when_no_env(self, monkeypatch):
        """No hardcoded localhost fallback — ``resolve_urls`` returns
        ``(None, None)`` when neither URL env vars nor HOST shorthand
        are set, and the caller (``TritonProvider._resolve_urls_from_config``)
        fails fast.  See the project's "no fallback" rule."""
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL", "TRITON_HOST"):
            monkeypatch.delenv(var, raising=False)
        openai, control = resolve_urls()
        assert openai is None
        assert control is None

    def test_host_shorthand_derives_both_urls_with_default_ports(self, monkeypatch):
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TRITON_HOST", "gpu-box")
        openai, control = resolve_urls()
        assert openai == "http://gpu-box:9000"
        assert control == "http://gpu-box:8000"

    def test_host_shorthand_accepts_scheme(self, monkeypatch):
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TRITON_HOST", "https://gpu-box")
        openai, control = resolve_urls()
        assert openai == "https://gpu-box:9000"
        assert control == "https://gpu-box:8000"

    def test_explicit_urls_win_over_host(self, monkeypatch):
        monkeypatch.setenv("TRITON_HOST", "gpu-box")
        monkeypatch.setenv("TRITON_OPENAI_URL", "http://lb:9999/v1")
        monkeypatch.setenv("TRITON_CONTROL_URL", "http://internal:7000")
        openai, control = resolve_urls()
        assert openai == "http://lb:9999/v1"
        assert control == "http://internal:7000"

    def test_partial_url_env_falls_through_to_host_for_other(self, monkeypatch):
        """Only OPENAI_URL set → CONTROL_URL derived from TRITON_HOST."""
        monkeypatch.setenv("TRITON_HOST", "gpu-box")
        monkeypatch.setenv("TRITON_OPENAI_URL", "http://lb:9999")
        monkeypatch.delenv("TRITON_CONTROL_URL", raising=False)
        openai, control = resolve_urls()
        assert openai == "http://lb:9999"
        assert control == "http://gpu-box:8000"

    def test_context_length_parsed(self, monkeypatch):
        monkeypatch.setenv("TRITON_CONTEXT_LENGTH", "131072")
        assert resolve_context_length() == 131072

    def test_context_length_invalid_is_none(self, monkeypatch):
        monkeypatch.setenv("TRITON_CONTEXT_LENGTH", "not-a-number")
        assert resolve_context_length() is None

    def test_api_token_from_env(self, monkeypatch):
        monkeypatch.setenv("TRITON_API_TOKEN", "secret-123")
        assert resolve_api_token() == "secret-123"


# ============================================================
# create_provider factory
# ============================================================


class TestFactory:

    def test_create_provider_returns_instance(self):
        provider = create_provider()
        assert isinstance(provider, TritonProvider)
        assert provider.name == "triton"
        assert provider.is_connected is False


# ============================================================
# initialize()
# ============================================================


class TestInitialize:

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_reads_explicit_urls_from_extra(self, mock_get, monkeypatch):
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL", "TRITON_HOST"):
            monkeypatch.delenv(var, raising=False)
        mock_get.return_value = _health_response()

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={
            "openai_url": "http://lb:9000",
            "control_url": "http://internal:8000",
        }))

        assert provider._openai_url == "http://lb:9000"
        assert provider._control_url == "http://internal:8000"

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_host_shorthand(self, mock_get, monkeypatch):
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL", "TRITON_HOST"):
            monkeypatch.delenv(var, raising=False)
        mock_get.return_value = _health_response()

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"host": "gpu-box"}))

        assert provider._openai_url == "http://gpu-box:9000"
        assert provider._control_url == "http://gpu-box:8000"

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_probes_control_health_endpoint(self, mock_get, monkeypatch):
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL", "TRITON_HOST"):
            monkeypatch.delenv(var, raising=False)
        mock_get.return_value = _health_response()

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"host": "srv"}))

        # The probe must hit the CONTROL URL's /v2/health/live, not the
        # OpenAI URL — the control URL is what we need for load/unload.
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://srv:8000/v2/health/live"

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_connection_failure_raises_with_surface(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        provider = TritonProvider()
        with pytest.raises(TritonConnectionError) as excinfo:
            provider.initialize(ProviderConfig(extra={"control_url": "http://offline:8000"}))
        # Error must identify which surface (control vs openai) failed.
        assert "control" in str(excinfo.value)

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_timeout_raises_connection_error(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        provider = TritonProvider()
        with pytest.raises(TritonConnectionError):
            provider.initialize(ProviderConfig())

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_401_raises_auth_error(self, mock_get):
        mock_response = MagicMock(status_code=401)
        mock_get.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response,
        )
        provider = TritonProvider()
        with pytest.raises(TritonAuthenticationError):
            provider.initialize(ProviderConfig(extra={"api_token": "bad"}))

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_rejects_non_dict_load(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TritonProvider()
        with pytest.raises(ValueError, match="load config must be a dict"):
            provider.initialize(ProviderConfig(extra={"load": "not a dict"}))

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_stores_load_config(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TritonProvider()
        provider.initialize(
            ProviderConfig(extra={"load": {"parameters": {"config": "{}"}}})
        )
        assert provider._load_config == {"parameters": {"config": "{}"}}

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_passive_mode_load_is_none(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        assert provider._load_config is None

    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_initialize_forwards_bearer_to_health_probe(self, mock_get):
        mock_get.return_value = _health_response()
        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"api_token": "tok-abc"}))
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok-abc"


# ============================================================
# verify_auth — must NOT touch the network
# ============================================================


class TestVerifyAuth:

    def test_returns_true_with_no_credentials_required(self, monkeypatch):
        monkeypatch.delenv("TRITON_API_TOKEN", raising=False)
        provider = TritonProvider()
        msgs = []
        ok = provider.verify_auth(on_message=msgs.append)
        assert ok is True
        assert any("no authentication required" in m for m in msgs)

    def test_does_not_make_any_http_request(self, monkeypatch):
        monkeypatch.delenv("TRITON_API_TOKEN", raising=False)
        provider = TritonProvider()
        with patch(
            "shared.plugins.model_provider.triton.provider.httpx.get"
        ) as gget, patch(
            "shared.plugins.model_provider.triton.provider.httpx.post"
        ) as gpost:
            provider.verify_auth()
            assert gget.call_count == 0
            assert gpost.call_count == 0

    def test_reports_token_when_supplied_via_config(self):
        provider = TritonProvider()
        msgs = []
        ok = provider.verify_auth(
            on_message=msgs.append,
            config=ProviderConfig(extra={"api_token": "secret-token-abc-1234"}),
        )
        assert ok is True
        assert any("bearer token configured" in m for m in msgs)
        assert all("secret-token-abc-1234" not in m for m in msgs)


# ============================================================
# connect() — passive mode
# ============================================================


class TestConnectPassive:

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_without_load_does_not_load_post(self, mock_get, mock_post):
        """Passive mode: index POST happens (for catalog validation),
        but no /load POST."""
        mock_get.return_value = _health_response()
        mock_post.return_value = _index_response()

        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        provider.connect("llama-3.1-8b-instruct")

        # POST should have been called once — for /v2/repository/index,
        # NOT /v2/repository/models/.../load
        assert mock_post.call_count == 1
        url = mock_post.call_args[0][0]
        assert url.endswith("/v2/repository/index")
        assert "/load" not in url
        assert provider.model_name == "llama-3.1-8b-instruct"

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_unknown_model_raises(self, mock_get, mock_post):
        mock_get.return_value = _health_response()
        mock_post.return_value = _index_response(models=(("different-model", "READY"),))

        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        with pytest.raises(TritonModelNotFoundError) as excinfo:
            provider.connect("llama-3.1-8b-instruct")
        # Error message should list what IS available.
        assert "different-model" in str(excinfo.value)

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_unloaded_model_is_valid_target(self, mock_get, mock_post):
        """A model present in the repo but in UNAVAILABLE state is still
        a valid target — that's the active-mode use case."""
        mock_get.return_value = _health_response()
        mock_post.return_value = _index_response(
            models=(("llama-3.1-8b-instruct", "UNAVAILABLE"),)
        )

        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        provider.connect("llama-3.1-8b-instruct")
        assert provider.model_name == "llama-3.1-8b-instruct"

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_skip_model_test_bypasses_index(self, mock_get, mock_post):
        mock_get.return_value = _health_response()

        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        provider.connect("any-model", skip_model_test=True)
        # No /v2/repository/index POST happened.
        mock_post.assert_not_called()
        assert provider.model_name == "any-model"


# ============================================================
# connect() — active mode (load)
# ============================================================


class TestConnectActive:

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_with_load_posts_expected_url_and_body(self, mock_get, mock_post):
        mock_get.return_value = _health_response()
        # Two POSTs: index (validation) then load.
        mock_post.side_effect = [_index_response(), _load_response(status=200)]

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={
            "control_url": "http://srv:8000",
            "load": {"parameters": {"config": "{\"max_batch_size\": 8}"}},
        }))
        provider.connect("llama-3.1-8b-instruct")

        # Second POST should be the /load call.
        assert mock_post.call_count == 2
        load_call = mock_post.call_args_list[1]
        url = load_call[0][0]
        assert url == "http://srv:8000/v2/repository/models/llama-3.1-8b-instruct/load"
        body = load_call[1]["json"]
        assert body == {"parameters": {"config": "{\"max_batch_size\": 8}"}}

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_empty_load_dict_posts_empty_body(self, mock_get, mock_post):
        """An empty dict means 'load with stored config' — common case."""
        mock_get.return_value = _health_response()
        mock_post.side_effect = [_index_response(), _load_response(status=200)]

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"load": {}}))
        provider.connect("llama-3.1-8b-instruct")

        load_call = mock_post.call_args_list[1]
        assert load_call[1]["json"] == {}

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_with_load_failure_raises(self, mock_get, mock_post):
        mock_get.return_value = _health_response()
        mock_post.side_effect = [
            _index_response(),
            _load_response(status=400, body='{"error": "bad parameters"}'),
        ]

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"load": {"parameters": {"bad": "1"}}}))
        with pytest.raises(TritonLoadError) as excinfo:
            provider.connect("llama-3.1-8b-instruct")
        assert excinfo.value.status_code == 400
        assert excinfo.value.load_config == {"parameters": {"bad": "1"}}

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_connect_active_sends_bearer_to_load(self, mock_get, mock_post):
        mock_get.return_value = _health_response()
        mock_post.side_effect = [_index_response(), _load_response(status=200)]

        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={
            "api_token": "tok-abc",
            "load": {},
        }))
        provider.connect("llama-3.1-8b-instruct")

        load_call = mock_post.call_args_list[1]
        assert load_call[1]["headers"]["Authorization"] == "Bearer tok-abc"


# ============================================================
# get_context_limit priority
# ============================================================


class TestContextLimit:
    """Triton's model config doesn't carry a standard "context length"
    field (backend-specific), so the value must come from either the
    profile-level ``context_length`` override or the
    ``TRITON_CONTEXT_LENGTH`` env var.  No hardcoded fallback exists
    per the project's no-fallback rule — ``initialize()`` fails fast
    when neither is set."""

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_override_wins(self, mock_get, mock_post):
        mock_get.return_value = _health_response()
        mock_post.return_value = _index_response()
        provider = TritonProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))
        provider.connect("llama-3.1-8b-instruct")
        assert provider.get_context_limit() == 131072

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    @patch("shared.plugins.model_provider.triton.provider.httpx.get")
    def test_env_value_used_when_no_config_override(self, mock_get, mock_post):
        """When ``ProviderConfig`` has no ``context_length`` override,
        the value flows from ``TRITON_CONTEXT_LENGTH`` (autouse fixture
        sets it to 8192).  No hardcoded fallback inside the provider."""
        mock_get.return_value = _health_response()
        mock_post.return_value = _index_response()
        provider = TritonProvider()
        provider.initialize(ProviderConfig())
        provider.connect("llama-3.1-8b-instruct")
        assert provider.get_context_limit() == 8192

    def test_initialize_fails_fast_when_context_length_missing(
        self, monkeypatch,
    ):
        """No hardcoded fallback for context_length — ``initialize``
        raises ``ValueError`` when neither config nor env provides one."""
        monkeypatch.delenv("TRITON_CONTEXT_LENGTH", raising=False)
        provider = TritonProvider()
        with pytest.raises(ValueError, match="context_length is not configured"):
            provider.initialize(ProviderConfig())

    def test_initialize_fails_fast_when_urls_missing(self, monkeypatch):
        """No hardcoded localhost fallback for URLs — ``initialize``
        raises ``ValueError`` when neither URL env vars nor HOST
        shorthand are set."""
        for var in ("TRITON_OPENAI_URL", "TRITON_CONTROL_URL", "TRITON_HOST"):
            monkeypatch.delenv(var, raising=False)
        provider = TritonProvider()
        with pytest.raises(ValueError, match="OpenAI frontend URL is not configured"):
            provider.initialize(ProviderConfig())

    def test_get_context_limit_raises_before_initialize(self):
        """``get_context_limit`` raises ``RuntimeError`` when called on
        an uninitialized provider — no silent fallback to a hardcoded
        default."""
        provider = TritonProvider()
        with pytest.raises(RuntimeError, match="before initialize"):
            provider.get_context_limit()


# ============================================================
# list_models — uses /v2/repository/index, not /v1/models
# ============================================================


class TestListModels:

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    def test_list_models_uses_repository_index(self, mock_post):
        """list_models() must use the repository index (which includes
        unloaded models), not the OpenAI /v1/models endpoint."""
        mock_post.return_value = _index_response(models=(
            ("llama-3.1-8b-instruct", "READY"),
            ("qwen-32b", "UNAVAILABLE"),
        ))
        provider = TritonProvider()
        provider._control_url = "http://srv:8000"
        names = provider.list_models()
        assert names == ["llama-3.1-8b-instruct", "qwen-32b"]
        # And the call went to /v2/repository/index, NOT /v1/models.
        url = mock_post.call_args[0][0]
        assert url == "http://srv:8000/v2/repository/index"

    @patch("shared.plugins.model_provider.triton.provider.httpx.post")
    def test_list_models_prefix_filter(self, mock_post):
        mock_post.return_value = _index_response(models=(
            ("llama-3.1-8b-instruct", "READY"),
            ("qwen-32b", "READY"),
            ("llama-3.3-70b", "READY"),
        ))
        provider = TritonProvider()
        provider._control_url = "http://srv:8000"
        names = provider.list_models(prefix="llama-")
        assert names == ["llama-3.1-8b-instruct", "llama-3.3-70b"]


# ============================================================
# Capabilities
# ============================================================


class TestCapabilities:

    def test_supports_streaming(self):
        assert TritonProvider().supports_streaming() is True

    def test_supports_stop(self):
        assert TritonProvider().supports_stop() is True

    def test_supports_structured_output(self):
        assert TritonProvider().supports_structured_output() is True

    def test_does_not_advertise_thinking(self):
        assert TritonProvider().supports_thinking() is False


# ============================================================
# Error handling — mid-stream vs pre-flight connection failures
# ============================================================


class TestMidStreamErrorDistinction:
    """Triton's OpenAI frontend exposes the same envelope-commit-before-
    validate failure mode as trtllm-serve (HTTP 200 + chunked headers
    committed before the engine validates → wire-level connection
    drop with no extractable cause).  ``_handle_api_error`` must
    distinguish these from pre-flight failures the same way the
    tensorrt_llm provider does — see the kb-orchestrator cascade
    incident (2026-06-06) which falsely surfaced as a generic
    connection error and burned forensics time.

    The mid-stream message itself must be GENERIC per
    ``feedback_no_hardcoded_likely_cause_in_error_messages.md`` —
    enforced by the shared ``assert_no_likely_cause_prose`` helper in
    ``shared/plugins/model_provider/conftest.py``.
    """

    def _make_apicon_error(self, message: str):
        from shared.plugins.model_provider.triton.provider import (
            get_openai_module,
        )
        openai = get_openai_module()
        try:
            return openai.APIConnectionError(message=message, request=None)
        except TypeError:
            return openai.APIConnectionError(message)

    def test_remote_protocol_error_routes_to_mid_stream(self):
        from shared.plugins.model_provider.triton.errors import (
            TritonMidStreamError,
        )
        provider = TritonProvider()
        provider._openai_url = "http://192.168.50.154:9000"
        err = self._make_apicon_error(
            "RemoteProtocolError: peer closed connection without "
            "sending complete message body"
        )
        with pytest.raises(TritonMidStreamError) as ctx:
            provider._handle_api_error(err)
        msg = str(ctx.value)
        assert "192.168.50.154:9000" in msg
        # Per feedback_no_hardcoded_likely_cause_in_error_messages.md,
        # the message points the operator at Triton's log instead of
        # naming a likely cause.
        assert "Triton's log" in msg

    def test_mid_stream_message_does_not_guess_at_cause(self):
        """Falsification guard for the no-hardcoded-likely-cause rule
        (``feedback_no_hardcoded_likely_cause_in_error_messages.md``).
        Shared with tensorrt_llm + vllm via the model_provider
        conftest.
        """
        from shared.plugins.model_provider.conftest import (
            assert_no_likely_cause_prose,
        )
        from shared.plugins.model_provider.triton.errors import (
            TritonMidStreamError,
        )
        err = TritonMidStreamError("http://srv:9000", original_error="x")
        assert_no_likely_cause_prose(str(err))

    def test_clean_connection_failure_still_routes_to_connection_error(self):
        from shared.plugins.model_provider.triton.errors import (
            TritonConnectionError,
            TritonMidStreamError,
        )
        provider = TritonProvider()
        provider._openai_url = "http://192.168.50.154:9000"
        err = self._make_apicon_error(
            "Connection refused at http://192.168.50.154:9000"
        )
        with pytest.raises(TritonConnectionError) as ctx:
            provider._handle_api_error(err)
        assert not isinstance(ctx.value, TritonMidStreamError)
        msg = str(ctx.value)
        assert "max_input_length" not in msg
