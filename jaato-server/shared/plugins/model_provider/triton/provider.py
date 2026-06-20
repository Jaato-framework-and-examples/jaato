"""Triton provider — OpenAI-compatible chat + KServe v2 model lifecycle.

Triton serves models via two cooperating HTTP surfaces:

- Chat:    ``POST /v1/chat/completions``      on the OpenAI frontend (port 9000)
- List:    ``GET  /v1/models``                on the OpenAI frontend
- Load:    ``POST /v2/repository/models/<name>/load``   on tritonserver (port 8000)
- Unload:  ``POST /v2/repository/models/<name>/unload`` on tritonserver
- Index:   ``POST /v2/repository/index``      on tritonserver
- Health:  ``GET  /v2/health/live``           on tritonserver

This provider drives both endpoints — chat through the OpenAI URL, model
control through the Triton control URL — using the same active/passive
pattern as the LM Studio provider:

- **Passive mode** (``load`` absent): use whatever model is currently
  loaded in Triton.  Cheapest, most idempotent.
- **Active mode** (``load`` is a dict): POST the dict to
  ``/v2/repository/models/<model>/load`` during ``connect()`` so Triton
  loads the model (or reloads it with overridden parameters) before the
  first chat.

Authentication: Triton has no native bearer-token mechanism — the
OpenAI frontend can be gated by ``--openai-restricted-api`` headers,
and production deployments commonly sit behind a reverse proxy.  When
``TRITON_API_TOKEN`` is set (or ``plugin_configs.triton.api_token``),
the bearer is sent as ``Authorization: Bearer <token>`` on both URLs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

# Reuse NIM's OpenAI lazy imports and converters — identical SDK,
# identical wire format on the OpenAI frontend.
from .._openai_compat._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    ModalityCapabilityMixin,
    FunctionCallDetectedCallback,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    ThinkingConfig,
    ToolSchema,
    TokenUsage,
    TurnResult,
)
from .._openai_compat.converters import (
    clear_tool_name_mapping,
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .env import (
    ENV_CONTEXT_LENGTH,
    ENV_HOST,
    ENV_OPENAI_URL,
    _origin_for_host,
    DEFAULT_CONTROL_PORT,
    DEFAULT_OPENAI_PORT,
    resolve_api_token,
    resolve_context_length,
    resolve_model,
    resolve_urls,
)
from .errors import (
    TritonAuthenticationError,
    TritonConnectionError,
    TritonLoadError,
    TritonMidStreamError,
    TritonModelNotFoundError,
)


logger = logging.getLogger(__name__)


def _extract_cache_tokens(usage_obj) -> Optional[int]:
    """Extract cached_tokens from an OpenAI usage object, if present.

    Triton's OpenAI frontend forwards backend-reported caching stats in
    the OpenAI-standard ``usage.prompt_tokens_details.cached_tokens``
    shape when the backend (TRT-LLM, vLLM) emits them.
    """
    if not usage_obj:
        return None
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if not details:
        return None
    cached = getattr(details, "cached_tokens", None)
    if cached is not None and isinstance(cached, int) and cached > 0:
        return cached
    return None


class TritonProvider(ModalityCapabilityMixin):
    """Triton provider: OpenAI-compatible chat + KServe v2 model control.

    Lifecycle:
        1. ``__init__()``              — no state yet
        2. ``initialize(config)``      — resolve URLs / auth, create the
                                         OpenAI client, probe ``/v2/health/live``
        3. ``connect(model)``          — verify the model exists in the
                                         repository; if ``load`` is set,
                                         POST it to
                                         ``/v2/repository/models/<model>/load``
        4. ``complete(messages, ...)`` — stateless chat
        5. ``shutdown()``              — release the OpenAI client

    Load body passthrough: Triton's load endpoint accepts a JSON body
    documented at
    ``docs/protocol/extension_model_repository.md``.  Common shape:

        {
          "parameters": {
            "config": "{\\"max_batch_size\\": 8}",
            "file:1/model.json": "<base64>",
            ...
          }
        }

    An empty body is also valid and means "load with the repository's
    stored config", which is the most common operation.  The provider
    posts the dict verbatim — no whitelist, no translation — so newer
    Triton load parameters work without code changes here.
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional["OpenAI"] = None
        self._model_name: Optional[str] = None

        # URLs + context_length are required at session init time; the
        # empty-string / None sentinels here are placeholders until
        # ``initialize()`` validates them.  No localhost / 8K fallback.
        self._openai_url: str = ""
        self._control_url: str = ""
        self._api_token: Optional[str] = None
        self._auth_info: str = ""

        self._last_usage: TokenUsage = TokenUsage()
        self._context_length_override: Optional[int] = None

        # Load body passthrough.  None means "passive mode": do not
        # invoke /v2/repository/models/<name>/load during connect().
        self._load_config: Optional[Dict[str, Any]] = None

        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Record agent context so trace messages are attributable."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write a trace message to the provider trace log."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "triton:main"
        elif self._agent_name:
            prefix = f"triton:subagent:{self._agent_name}"
        else:
            prefix = f"triton:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "triton"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        Resolves the two URLs from ``config.extra`` / env (see
        :func:`resolve_urls`), optional bearer token, optional context
        override, and the optional ``load`` body, then verifies Triton's
        control surface is reachable via ``GET /v2/health/live``.

        Args:
            config: ``ProviderConfig`` whose ``extra`` dict may contain:

                - ``openai_url`` (str): Chat URL override
                - ``control_url`` (str): Model-control URL override
                - ``host`` (str): Shorthand — when set without explicit
                  URLs, both URLs use this host with default ports
                  (9000 for OpenAI, 8000 for control)
                - ``context_length`` (int): Override context window size
                - ``api_token`` (str): Bearer token (auth-proxy gating)
                - ``load`` (dict): Passthrough body for
                  ``POST /v2/repository/models/<model>/load`` (active mode).

        Raises:
            TritonConnectionError: Control URL not reachable.
        """
        self._trace("[INIT] Starting initialization")

        if config is None:
            config = ProviderConfig()

        self._resolve_urls_from_config(config)

        self._api_token = config.extra.get("api_token") or resolve_api_token()

        context_extra = config.extra.get("context_length")
        if context_extra:
            self._context_length_override = int(context_extra)
        else:
            self._context_length_override = resolve_context_length()
        if not self._context_length_override:
            raise ValueError(
                f"Triton provider: context_length is not configured.  "
                f"Triton's model config does not carry a standard "
                f"'context length' field (backend-specific — TRT-LLM "
                f"has max_seq_len, vLLM has max_model_len, etc.), so the "
                f"framework cannot auto-discover it.  Set "
                f"{ENV_CONTEXT_LENGTH} in the environment, or "
                f"plugin_configs.triton.context_length in the profile.  "
                f"No hardcoded fallback exists per the project's "
                f"no-fallback rule."
            )

        load = config.extra.get("load")
        if load is not None and not isinstance(load, dict):
            raise ValueError(
                f"triton load config must be a dict, got {type(load).__name__}"
            )
        self._load_config = load

        self._auth_info = (
            f"openai={self._openai_url}, control={self._control_url}, bearer"
            if self._api_token
            else f"openai={self._openai_url}, control={self._control_url}"
        )
        self._trace(
            f"[INIT] openai_url={self._openai_url} control_url={self._control_url} "
            f"load_params={'yes' if load else 'no'}"
        )

        self._client = self._create_client()
        self._verify_connectivity()
        self._trace("[INIT] Initialization complete")

    def _resolve_urls_from_config(self, config: ProviderConfig) -> None:
        """Pick OpenAI + control URLs from config.extra, falling back to env.

        Precedence: explicit ``openai_url`` / ``control_url`` keys >
        the ``host`` shorthand (with the canonical Triton ports) > env
        vars.  When NEITHER explicit URLs NOR the host shorthand NOR
        the env vars provide a value, ``ValueError`` is raised — no
        localhost fallback per the project's no-fallback rule.
        """
        env_openai, env_control = resolve_urls()

        explicit_openai = config.extra.get("openai_url")
        explicit_control = config.extra.get("control_url")
        host_shorthand = config.extra.get("host")

        if explicit_openai:
            self._openai_url = explicit_openai.rstrip("/")
        elif host_shorthand:
            self._openai_url = _origin_for_host(host_shorthand, DEFAULT_OPENAI_PORT)
        elif env_openai:
            self._openai_url = env_openai
        else:
            raise ValueError(
                f"Triton provider: OpenAI frontend URL is not configured.  "
                f"Set {ENV_OPENAI_URL} (or {ENV_HOST} for a shorthand "
                f"using the canonical port {DEFAULT_OPENAI_PORT}) in the "
                f"environment, or plugin_configs.triton.openai_url / "
                f"plugin_configs.triton.host in the profile.  No "
                f"hardcoded localhost fallback exists per the project's "
                f"no-fallback rule."
            )

        if explicit_control:
            self._control_url = explicit_control.rstrip("/")
        elif host_shorthand:
            self._control_url = _origin_for_host(host_shorthand, DEFAULT_CONTROL_PORT)
        elif env_control:
            self._control_url = env_control
        else:
            raise ValueError(
                f"Triton provider: control (KServe v2) URL is not "
                f"configured.  Set TRITON_CONTROL_URL (or {ENV_HOST} for "
                f"a shorthand using the canonical port "
                f"{DEFAULT_CONTROL_PORT}) in the environment, or "
                f"plugin_configs.triton.control_url / "
                f"plugin_configs.triton.host in the profile.  No "
                f"hardcoded localhost fallback exists per the project's "
                f"no-fallback rule."
            )

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client pointing at Triton's OpenAI frontend.

        Triton's OpenAI frontend ignores the ``api_key`` field by
        default, but we forward the real bearer token when one is
        configured so deployments behind an auth proxy (or with
        ``--openai-restricted-api`` enabled) also work.
        """
        client_class = get_openai_client_class()
        return client_class(
            base_url=f"{self._openai_url}/v1",
            api_key=self._api_token or "triton",
        )

    def _auth_headers(self) -> Dict[str, str]:
        """Return auth headers for direct ``httpx`` calls to ``/v2``."""
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _verify_connectivity(self) -> None:
        """Confirm tritonserver is reachable via ``GET /v2/health/live``.

        We probe the *control* URL rather than the OpenAI URL because
        the control URL is what we need for load/unload — if it's down,
        active mode will fail at ``connect()``.  Passive-mode users
        whose control URL is misconfigured but whose chat URL works
        still surface the error here, which is the right tradeoff:
        better an upfront failure than a confusing one at first chat.
        """
        try:
            response = httpx.get(
                f"{self._control_url}/v2/health/live",
                headers=self._auth_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise TritonConnectionError(self._control_url, surface="control")
        except httpx.TimeoutException:
            raise TritonConnectionError(
                self._control_url, "Connection timed out", surface="control",
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise TritonAuthenticationError(original_error=str(exc))
            raise TritonConnectionError(
                self._control_url, str(exc), surface="control",
            )
        except httpx.HTTPError as exc:
            raise TritonConnectionError(
                self._control_url, str(exc), surface="control",
            )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Credentials-only check — Triton has no native bearer mechanism.

        Per the provider-plugin contract (``shared/plugins/CLAUDE.md``),
        ``verify_auth`` runs on a *fresh, uninitialized* instance and
        must only check whether credentials are **available** — not
        whether they are valid, and not whether the remote service is
        reachable.  Reachability is the job of ``initialize()``.

        For Triton that means: there is nothing to authenticate in the
        default setup, so we always return ``True``.  When the upstream
        is behind an auth proxy or ``--openai-restricted-api``, the
        bearer is read either from ``plugin_configs['triton']['api_token']``
        (via ``config.extra`` when supplied by the runtime) or from
        ``TRITON_API_TOKEN``.  Its presence is reported for operator
        visibility but is never a hard requirement.
        """
        token: Optional[str] = None
        if config is not None and config.extra:
            token = config.extra.get("api_token")
        if not token:
            token = resolve_api_token()

        if on_message:
            if token:
                masked = (
                    f"{token[:4]}…{token[-4:]}" if len(token) > 8 else "***"
                )
                on_message(f"Triton bearer token configured ({masked})")
            else:
                on_message(
                    "Triton: no authentication required "
                    "(reachability validated at session start)"
                )
        return True

    def shutdown(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Short human-readable description of auth state."""
        return self._auth_info or f"Triton ({self._openai_url})"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model for subsequent ``complete()`` calls.

        Verifies the model is present in the repository (via
        ``POST /v2/repository/index``), then — if ``load`` was supplied
        in the profile — POSTs that dict to
        ``/v2/repository/models/<model>/load`` to load the model with
        the requested parameters.  Without ``load``, the provider is
        passive and assumes Triton was started with the model already
        loaded (or in poll mode and discovered it).

        Args:
            model: Model identifier as listed in Triton's model
                repository (the directory name under ``--model-repository``).
            skip_model_test: Skip the repository-index validation call.
                Load control is still invoked when requested.

        Raises:
            TritonModelNotFoundError: Model is not in the repository.
            TritonLoadError: ``/load`` returned a non-2xx.
        """
        if not skip_model_test:
            catalog = self._fetch_repository_index()
            if catalog and model not in {entry["name"] for entry in catalog}:
                raise TritonModelNotFoundError(
                    model, available=[entry["name"] for entry in catalog],
                )

        self._model_name = model

        if self._load_config is not None:
            self._load_model(model, self._load_config)

        logger.info(
            "Connected to Triton model: %s (context=%d, load_applied=%s)",
            model, self.get_context_limit(), bool(self._load_config),
        )

    def _load_model(self, model: str, load_params: Dict[str, Any]) -> None:
        """POST the load body to ``/v2/repository/models/<model>/load``.

        Triton's load is idempotent at the protocol level: loading an
        already-loaded model with the same parameters is a fast no-op
        on the server side.  We don't pre-query for a "matching loaded
        instance" the way the LM Studio provider does — Triton's load
        is much cheaper than LM Studio's because the model is already
        on disk and Triton tracks load state internally.

        The body is the ``load_params`` dict passed verbatim.  An empty
        dict (``{}``) is valid and means "load with stored config".
        """
        url = f"{self._control_url}/v2/repository/models/{model}/load"
        self._trace(
            f"[LOAD] POST {url} body_keys={sorted(load_params.keys())}"
        )
        try:
            response = httpx.post(
                url,
                json=load_params or {},
                headers=self._auth_headers(),
                # Loading a large model can legitimately take a while
                # (engine deserialization + KV-cache init + GPU transfer).
                timeout=600.0,
            )
        except httpx.HTTPError as exc:
            raise TritonConnectionError(
                self._control_url, f"load failed: {exc}", surface="control",
            ) from exc

        if response.status_code >= 400:
            raise TritonLoadError(
                model=model,
                status_code=response.status_code,
                body=response.text,
                load_config=load_params,
            )
        self._trace(f"[LOAD] status={response.status_code}")

    def _fetch_repository_index(self) -> List[Dict[str, Any]]:
        """Query ``POST /v2/repository/index`` and return the list of models.

        Returns entries shaped ``{"name": "...", "version": "...",
        "state": "READY"|"UNAVAILABLE"|...}``.  Includes both loaded
        and unloaded models, which is what we want for the connect-time
        validation: a model that's *available* but not yet loaded is
        still a legitimate target for active mode.

        Returns an empty list when the server is unreachable so callers
        can degrade gracefully (the connect-time test treats an empty
        catalog as "skip validation" rather than "model not found").
        """
        try:
            response = httpx.post(
                f"{self._control_url}/v2/repository/index",
                json={},
                headers=self._auth_headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            # /v2/repository/index returns a top-level array, not an
            # object with a "data" field — distinct from OpenAI's
            # /v1/models shape.
            if isinstance(payload, list):
                return payload
            return []
        except httpx.HTTPError as exc:
            logger.warning("Failed to list Triton models: %s", exc)
            return []

    @property
    def is_connected(self) -> bool:
        """True when both client and model are set."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Currently selected model name, or ``None`` before ``connect()``."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models in Triton's repository (loaded or not).

        Uses ``POST /v2/repository/index`` rather than the OpenAI
        frontend's ``GET /v1/models`` so users see models they could
        load via active mode, not just the ones currently loaded.
        """
        catalog = self._fetch_repository_index()
        names = [entry["name"] for entry in catalog]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    # ==================== Stateless Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[StreamingCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> TurnResult:
        """Run one stateless chat completion through Triton's OpenAI frontend.

        Streaming and non-streaming paths mirror ``lmstudio``'s
        implementation — Triton's OpenAI frontend speaks OpenAI's wire
        format faithfully, including ``tools`` / ``tool_choice`` for
        backends whose chat template supports function calling.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )

        clear_tool_name_mapping()

        openai_messages: List[Dict[str, Any]] = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(history_to_openai(list(messages)))

        kwargs: Dict[str, Any] = {}
        if tools:
            openai_tools = tool_schemas_to_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            if on_chunk:
                provider_response = self._stream_response(
                    messages=openai_messages,
                    kwargs=kwargs,
                    on_chunk=on_chunk,
                    cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_thinking=on_thinking,
                    trace_prefix="COMPLETE_STREAM",
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    **kwargs,
                )
                provider_response = response_from_openai(response)
                if response and response.usage:
                    cached = _extract_cache_tokens(response.usage)
                    if cached is not None:
                        provider_response.usage.cache_read_tokens = cached

            self._last_usage = provider_response.usage

            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return TurnResult.from_provider_response(provider_response)
        except Exception as exc:
            self._handle_api_error(exc)
            raise

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
        trace_prefix: str = "STREAM",
    ) -> ProviderResponse:
        """Accumulate text, tool calls, and usage from a streaming response."""
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block():
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            nonlocal tool_call_accumulators
            for idx in sorted(tool_call_accumulators.keys()):
                tc = tool_call_accumulators[idx]
                func_name = tc.get("function", {}).get("name")
                if func_name:
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_id = tc.get("id")
                    original_name = get_original_tool_name(func_name)
                    fc = FunctionCall(id=tool_id, name=original_name, args=args)
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                if cancel_token and cancel_token.is_cancelled:
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                            cache_read_tokens=_extract_cache_tokens(chunk.usage),
                        )
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_accumulators[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += tc_delta.function.arguments

                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)

                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cache_read_tokens=_extract_cache_tokens(chunk.usage),
                    )
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(
                f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}"
            )

        except Exception as exc:
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        flush_text_block()
        flush_tool_calls()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=None,
        )

    # ==================== Error Handling ====================

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to Triton-specific error types."""
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise TritonAuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.NotFoundError):
            raise TritonModelNotFoundError(
                model=self._model_name or "unknown"
            ) from error

        if isinstance(error, openai.APIConnectionError):
            # Same mid-stream-vs-pre-flight distinction as the
            # tensorrt_llm provider — see TensorRTLLMMidStreamError
            # docstring for the empirical incident (2026-06-06
            # kb-orchestrator cascade against trtllm-serve, but the
            # engine config bug surfaces identically on Triton's
            # OpenAI frontend when it wraps the same engine).
            error_msg = str(error).lower()
            is_mid_stream = (
                "peer closed connection" in error_msg
                or "incomplete chunked" in error_msg
                or "remoteprotocolerror" in error_msg
            )
            if is_mid_stream:
                raise TritonMidStreamError(
                    self._openai_url, original_error=str(error),
                ) from error
            raise TritonConnectionError(
                self._openai_url, str(error), surface="openai",
            ) from error

        # Other APIStatusError subclasses fall through.

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Heuristic token count (~4 chars/token).

        Triton's OpenAI frontend does not expose a tokenization
        endpoint.  Per-backend tokenizers exist but vary too widely to
        rely on.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Return the context window size for the currently connected model.

        Returns the value validated at ``initialize()`` time (sourced
        from ``plugin_configs.triton.context_length`` or
        ``TRITON_CONTEXT_LENGTH``).  Triton's per-model config is too
        backend-specific for reliable auto-discovery (TRT-LLM exposes
        ``max_seq_len``, vLLM exposes ``max_model_len``, ONNX/PyTorch
        backends have no equivalent concept), so the value must come
        from the operator — ``initialize()`` raises if missing.

        Raises:
            RuntimeError: When called before ``initialize()`` has run.
        """
        if self._context_length_override is None:
            raise RuntimeError(
                "Triton provider: get_context_limit() called before "
                "initialize() set the override.  This is a programmer "
                "error — the provider must be connected first."
            )
        return self._context_length_override

    def get_token_usage(self) -> TokenUsage:
        """Token usage from the most recent completion."""
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """The OpenAI frontend accepts ``response_format={'type': 'json_object'}``."""
        return True

    def supports_streaming(self) -> bool:
        """Streaming works through the OpenAI-compatible endpoint."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses can be interrupted via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """Triton's OpenAI frontend does not surface reasoning deltas."""
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is not exposed through Triton's /v1 API."""
        pass

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify exceptions for the reliability layer's retry policy."""
        if isinstance(exc, TritonConnectionError):
            return {"transient": True, "rate_limit": False, "infra": True}
        if isinstance(exc, TritonLoadError):
            # Load failure is deterministic given the body — don't retry.
            return {"transient": False, "rate_limit": False, "infra": False}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Triton does not emit retry-after hints."""
        return None

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — Triton does not use interactive auth."""
        if on_message:
            on_message(
                "Triton has no native bearer auth. If your deployment uses "
                "--openai-restricted-api or sits behind an auth proxy, set "
                "TRITON_API_TOKEN."
            )


def create_provider() -> TritonProvider:
    """Factory function consumed by the provider discovery machinery."""
    return TritonProvider()
