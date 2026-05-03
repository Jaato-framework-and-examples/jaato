"""OpenRouter model provider implementation.

OpenRouter (https://openrouter.ai) is a unified gateway over hundreds of
models from many vendors (OpenAI, Anthropic, Google, Meta, Mistral,
DeepSeek, ...).  It speaks the OpenAI chat-completions wire format, so
this provider uses the ``openai`` Python SDK as its transport.

Endpoints
---------

- ``POST /api/v1/chat/completions`` — chat (OpenAI-compatible)
- ``GET  /api/v1/models``           — model catalog (no auth required)
- ``GET  /api/v1/key``              — auth introspection

Authentication
--------------

- ``JAATO_OPENROUTER_API_KEY`` (``sk-or-...`` from
  https://openrouter.ai/settings/keys), or run ``openrouter-auth key``.

Optional attribution
--------------------

OpenRouter exposes app-level rankings when integrators send two
optional headers:

- ``HTTP-Referer``        — site URL
- ``X-OpenRouter-Title``  — site / app title

Both are defaulted but can be overridden via env vars.

Environment variables
---------------------

    JAATO_OPENROUTER_API_KEY        API key (sk-or-...)
    JAATO_OPENROUTER_BASE_URL       Endpoint (default: https://openrouter.ai/api/v1)
    JAATO_OPENROUTER_MODEL          Default model name
    JAATO_OPENROUTER_CONTEXT_LENGTH Override context window
    JAATO_OPENROUTER_HTTP_REFERER   Attribution: site URL
    JAATO_OPENROUTER_APP_TITLE      Attribution: app title
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from ._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
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
    ToolSchema,
    TokenUsage,
    ThinkingConfig,
    TurnResult,
)
from .converters import (
    deserialize_history as _deserialize_history,
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    serialize_history as _serialize_history,
    tool_schemas_to_openai,
)
from .env import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_LENGTH,
    HEADER_APP_TITLE,
    HEADER_HTTP_REFERER,
    get_checked_credential_locations,
    resolve_api_key,
    resolve_app_title,
    resolve_base_url,
    resolve_context_length,
    resolve_http_referer,
    resolve_model,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)


# Substrings that signal the upstream model exposes reasoning content
# via OpenRouter's ``reasoning`` field.  OpenRouter normalises this for
# DeepSeek-R1, OpenAI o-series, Anthropic thinking models, ...  The
# match is best-effort and conservative; missing entries just skip
# extended-thinking emission and don't break the chat path.
REASONING_CAPABLE_HINTS = (
    "deepseek-r1",
    "deepseek/deepseek-r1",
    "o1",
    "o3",
    "o4",
    "thinking",
    "reasoner",
)


class OpenRouterProvider:
    """OpenRouter model provider.

    Provides access to OpenRouter's catalog via the OpenAI-compatible
    chat-completions API.  The provider is stateless with respect to
    conversation history — the session owns the canonical message
    list and passes it to ``complete()`` on every call.

    Lifecycle:
        1. ``__init__()`` — create instance (no connections yet)
        2. ``initialize(config)`` — resolve credentials, build client
        3. ``connect(model)`` — set the active model and (best-effort)
           cache its catalog-reported context length
        4. ``complete(messages, ...)`` — stateless completion
        5. ``shutdown()`` — release resources
    """

    def __init__(self) -> None:
        """Initialize the provider (not yet connected)."""
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL
        self._http_referer: str = ""
        self._app_title: str = ""

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length: int = 0
        self._context_length_override: bool = False

        # OpenRouter request-time routing controls.  ``provider`` is the
        # killer feature of OpenRouter — it constrains which upstream
        # provider serves each request (sort by price/throughput/latency,
        # blacklist providers, require non-training upstreams, etc.).
        # Stored once at initialize() and forwarded as a top-level body
        # field on every ``chat.completions.create`` via ``extra_body``.
        self._provider_routing: Optional[Dict[str, Any]] = None

        # Thinking / reasoning knobs — unified gate for both
        # request-side emission of OpenRouter's ``reasoning`` block
        # and response-side extraction of reasoning content.  Flat-key
        # convention shared with Anthropic (``enable_thinking``,
        # ``thinking_budget``) and Antigravity (``thinking_level``) so
        # a profile can set them provider-agnostically — switching from
        # anthropic to openrouter doesn't require rewriting the profile.
        # Translated into OpenRouter's ``reasoning`` wire shape in
        # ``_build_extra_body()`` and used to gate streaming/batch
        # extraction below.
        self._thinking_budget: int = 0
        self._thinking_level: Optional[str] = None

        # Sampling parameters.  ``None`` defers to the upstream's default
        # so the wire stays minimal unless the profile opts in.  Values
        # are read from ``config.extra.{temperature, top_p, top_k}`` —
        # this is how a profile expresses per-model tuning (e.g. Qwen
        # 2.5 wants temperature=0.55 / top_p=1 per the upstream's docs;
        # rather than hardcode that, the dumb-set profile sets the knobs).
        self._temperature: Optional[float] = None
        self._top_p: Optional[float] = None
        self._top_k: Optional[int] = None

        # Cached catalog so connect() can look up per-model context lengths
        # without re-fetching for every model switch.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

        # Thinking/reasoning configuration.  Defaults to False so
        # nothing extra goes on the wire and reasoning content is not
        # surfaced unless the profile / runtime opts in.  Matches the
        # Anthropic provider's default.
        self._enable_thinking: bool = False

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    # ==================== Agent Context / Tracing ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Set agent context for trace identification."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write a trace message to the provider trace log."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "openrouter:main"
        elif self._agent_name:
            prefix = f"openrouter:subagent:{self._agent_name}"
        else:
            prefix = f"openrouter:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "openrouter"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Builds an OpenAI client pointed at OpenRouter and configures the
        attribution headers OpenRouter uses for app rankings.

        ``config.extra`` is namespaced into four layers (server 0.6.23+):

        - **Top-level** (auth / identity):
            - ``api_key``, ``http_referer``, ``app_title``
        - ``api_params`` (OpenAI Chat Completions request body fields):
            - ``temperature``, ``top_p``, ``top_k``, ``max_tokens``
            - ``enable_thinking``, ``thinking_budget``, ``thinking_level``
        - ``routing`` (OpenRouter ``provider`` extension — a dict
          forwarded verbatim via ``extra_body`` on every request):
            - ``order``, ``allow_fallbacks``, ``require_parameters``,
              ``data_collection``, ``ignore``, ``quantizations``,
              ``sort``, ... (see
              https://openrouter.ai/docs/features/provider-routing)
        - ``framework_overrides`` (rare escape hatches):
            - ``context_length``, ``base_url``

        For backward compatibility, all of these are also accepted at
        the legacy flat position (``config.extra['temperature']``, ...)
        with a one-time deprecation warning.  The legacy ``provider:``
        key is also accepted as an alias for ``routing:`` with the
        same warning.  Flat-key support will be removed in a future
        release.

        Args:
            config: Configuration with authentication and provider knobs
                (see namespacing layers above).

        Raises:
            APIKeyNotFoundError: No API key found.
            TypeError: If ``routing`` (or legacy ``provider``) is set
                but isn't a dict.
            ValueError: If ``thinking_level`` is not low/medium/high.
        """
        if config is None:
            config = ProviderConfig()

        # The profile config is namespaced into four layers (see
        # docs/design/provider-config-namespacing or the backlog memory
        # ``project_backlog_provider_config_namespacing``):
        #
        #   plugin_configs.openrouter:
        #     <top-level>           # auth / identity (api_key, http_referer, app_title)
        #     api_params:           # OpenAI Chat Completions request body params
        #       temperature, top_p, top_k, max_tokens
        #       enable_thinking, thinking_budget, thinking_level
        #     routing:              # OpenRouter `provider` extension (sort/ignore/order/...)
        #     framework_overrides:  # rare escape hatches (context_length, base_url)
        #
        # Backward compatibility: every nested key is also read from the
        # legacy flat position with a one-time deprecation warning.  The
        # flat fallback will be removed in a future server release.
        api_params = config.extra.get("api_params") or {}
        routing_dict = config.extra.get("routing")
        framework_overrides = config.extra.get("framework_overrides") or {}

        def _knob(
            key: str, *, layer: Dict[str, Any], default: Any = None,
        ) -> Any:
            """Read a config knob from its nested layer first, falling
            back to the legacy flat ``config.extra[key]`` position with
            a deprecation warning when only the flat form is present."""
            if key in layer:
                return layer[key]
            if key in config.extra:
                logger.warning(
                    "OpenRouter profile uses legacy flat config key %r — "
                    "move under the appropriate nested layer "
                    "(api_params / routing / framework_overrides) per the "
                    "0.6.22+ namespacing.  Flat-key support will be "
                    "removed in a future release.",
                    key,
                )
                return config.extra[key]
            return default

        self._api_key = config.api_key or resolve_api_key()
        # Top-level: framework auth / identity.  These live at the top
        # of plugin_configs.openrouter regardless of namespacing tier.
        self._base_url = (
            _knob("base_url", layer=framework_overrides) or resolve_base_url()
        )
        self._http_referer = (
            config.extra.get("http_referer") or resolve_http_referer()
        )
        self._app_title = (
            config.extra.get("app_title") or resolve_app_title()
        )

        # framework_overrides: rare escape hatches.  Normally
        # context_length is discovered from the OpenRouter catalog at
        # connect() time; only set the override when the catalog is
        # wrong / unavailable / for self-hosted gateways.
        context_length_extra = _knob(
            "context_length", layer=framework_overrides,
        )
        if context_length_extra:
            self._context_length = int(context_length_extra)
            self._context_length_override = True
        else:
            self._context_length = resolve_context_length()
            # Treat the env-var fallback as authoritative only if the
            # user actually set it; otherwise we'll let connect() learn
            # the per-model context length from the catalog.
            self._context_length_override = (
                self._context_length != DEFAULT_CONTEXT_LENGTH
            )

        # routing: OpenRouter's ``provider`` extension field (sort,
        # ignore, order, allow_fallbacks, ...).  Stored as a dict and
        # passed through verbatim via ``extra_body`` on each request.
        # Legacy flat key was ``provider:`` which read confusingly as
        # "provider config" rather than "OpenRouter routing extension".
        if routing_dict is None:
            # Fallback: legacy flat ``provider:`` key.
            routing_dict = config.extra.get("provider")
            if routing_dict is not None:
                logger.warning(
                    "OpenRouter profile uses legacy flat key 'provider:' for "
                    "routing config — rename to 'routing:' under "
                    "plugin_configs.openrouter.  Flat-key support will be "
                    "removed in a future release."
                )
        if routing_dict is not None:
            if not isinstance(routing_dict, dict):
                raise TypeError(
                    "OpenRouter 'routing' config must be a dict, "
                    f"got {type(routing_dict).__name__}"
                )
            # Deep copy so later mutations to the profile dict (incl.
            # nested ``ignore`` / ``order`` lists) don't leak into our
            # request body.
            self._provider_routing = copy.deepcopy(routing_dict)

        # api_params: OpenAI Chat Completions request body fields.
        # ``None`` means "let the upstream apply its own default".
        # Some upstreams reject requests without explicit per-model
        # temperature / top_p values; per-profile configuration of
        # these is the principled way to express model-specific tuning.
        temp_extra = _knob("temperature", layer=api_params)
        if temp_extra is not None:
            self._temperature = float(temp_extra)
        top_p_extra = _knob("top_p", layer=api_params)
        if top_p_extra is not None:
            self._top_p = float(top_p_extra)
        top_k_extra = _knob("top_k", layer=api_params)
        if top_k_extra is not None:
            self._top_k = int(top_k_extra)

        # Thinking knobs — framework abstractions that translate to
        # OpenRouter's ``reasoning`` extension on the wire.  Live under
        # api_params because they're agent-tuning knobs (not gateway
        # routing concerns).
        enable_thinking_extra = _knob(
            "enable_thinking", layer=api_params, default=self._enable_thinking,
        )
        self._enable_thinking = bool(enable_thinking_extra)
        budget_extra = _knob("thinking_budget", layer=api_params)
        if budget_extra is not None:
            self._thinking_budget = int(budget_extra)
        level_extra = _knob("thinking_level", layer=api_params)
        if level_extra is not None:
            level_str = str(level_extra).lower()
            if level_str not in ("low", "medium", "high"):
                raise ValueError(
                    "OpenRouter 'thinking_level' must be one of "
                    f"'low' / 'medium' / 'high', got {level_extra!r}"
                )
            self._thinking_level = level_str

        if not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

        self._client = self._create_client()
        self._trace(
            f"[INIT] client created, base_url={self._base_url}, "
            f"referer={self._http_referer!r}, title={self._app_title!r}, "
            f"provider_routing={'configured' if self._provider_routing else 'none'}"
        )

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client configured for OpenRouter.

        Sets the optional attribution headers via ``default_headers``
        so every request automatically includes them.
        """
        client_class = get_openai_client_class()
        default_headers: Dict[str, str] = {}
        if self._http_referer:
            default_headers[HEADER_HTTP_REFERER] = self._http_referer
        if self._app_title:
            default_headers[HEADER_APP_TITLE] = self._app_title

        return client_class(
            base_url=self._base_url,
            api_key=self._api_key,
            default_headers=default_headers or None,
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key in
        env / stored credentials without ever touching ``self._client``.

        ``config`` is accepted for protocol compatibility; the provider
        also reads ``api_key`` from ``config.extra`` when supplied so
        profile-level credentials can take effect during the pre-init
        check.

        When the stored credential file exists but cannot be loaded
        (corrupt JSON, missing field, permission error), the failure
        reason is surfaced via ``on_message`` instead of being swallowed
        into a generic "no key" error.
        """
        import os
        from .auth import try_load_credentials_with_reason
        from .env import ENV_OPENROUTER_API_KEY

        # Profile-supplied key wins — useful for headless / multi-tenant runs.
        profile_key: Optional[str] = None
        if config is not None and config.extra:
            profile_key = config.extra.get("api_key")
        if profile_key:
            if on_message:
                on_message("Found OpenRouter API key (profile config)")
            return True

        env_key = os.environ.get(ENV_OPENROUTER_API_KEY)
        if env_key:
            if on_message:
                on_message("Found OpenRouter API key (environment variable)")
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found OpenRouter API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"OpenRouter credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'openrouter-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_OPENROUTER_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations()
                )
            return False

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        return False

    def shutdown(self) -> None:
        """Close the OpenAI client and clear connection state."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os
        from .env import ENV_OPENROUTER_API_KEY

        if os.environ.get(ENV_OPENROUTER_API_KEY):
            return f"OpenRouter API key ({ENV_OPENROUTER_API_KEY})"

        try:
            from .auth import get_credential_file_path
            cred_path = get_credential_file_path()
            if cred_path:
                return f"OpenRouter API key ({cred_path})"
        except ImportError:
            pass

        return "OpenRouter API key"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the model to use and learn its context length from the catalog.

        Catalog lookup is best-effort: a network failure or missing
        entry just falls back to ``DEFAULT_CONTEXT_LENGTH`` (or the
        explicit override the user set via env / config).  Real model
        validation happens on the first chat-completion call.

        Args:
            model: Model ID (e.g. ``anthropic/claude-3.5-sonnet``,
                ``openai/gpt-4o``, ``openrouter/auto``).
            skip_model_test: When True, skip the catalog lookup for
                fast bootstrap; context length stays at whatever
                ``initialize()`` resolved.
        """
        self._model_name = model

        if skip_model_test or self._context_length_override:
            return

        catalog_length = self._lookup_context_length(model)
        if catalog_length:
            self._context_length = catalog_length
            self._trace(
                f"[CONNECT] model={model} context_length={catalog_length} (catalog)"
            )

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from OpenRouter's public catalog.

        Calls ``GET /api/v1/models`` (no auth required) and returns
        model IDs sorted alphabetically.  Returns an empty list on
        network failure — callers can surface that as a clear error
        rather than a fake catalog.

        Args:
            prefix: Optional vendor / name prefix filter
                (e.g. ``"anthropic/"`` or ``"openai/gpt-4"``).

        Returns:
            Sorted list of model IDs (e.g. ``["anthropic/claude-3.5-sonnet",
            "openai/gpt-4o", ...]``).
        """
        catalog = self._fetch_catalog()
        ids = [entry["id"] for entry in catalog if entry.get("id")]
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the OpenRouter model catalog.

        Returns the cached list on subsequent calls so per-connect
        lookups don't re-hit the network.  An empty list is returned
        and *not* cached on failure, so the next call will retry.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        try:
            response = httpx.get(url, timeout=15)
            response.raise_for_status()
        except Exception as exc:
            self._trace(f"[CATALOG] fetch failed: {type(exc).__name__}: {exc}")
            return []

        try:
            data = response.json()
        except ValueError as exc:
            self._trace(f"[CATALOG] invalid JSON: {exc}")
            return []

        catalog = data.get("data") if isinstance(data, dict) else None
        if not isinstance(catalog, list):
            self._trace("[CATALOG] response missing 'data' list")
            return []

        self._catalog_cache = catalog
        return catalog

    def _lookup_context_length(self, model: str) -> Optional[int]:
        """Return the catalog-reported context length for ``model``."""
        for entry in self._fetch_catalog():
            if entry.get("id") == model:
                ctx = entry.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
        return None

    def _build_extra_body(self) -> Dict[str, Any]:
        """Build OpenRouter-specific top-level body fields for each request.

        Currently emits ``provider`` (upstream routing) and ``reasoning``
        (request-time thinking control) when configured.  Future
        OpenRouter knobs (``transforms``, ``models`` fallback list,
        server-side ``plugins``) will land here too — this is the
        single funnel that translates session-level config into
        per-request OpenRouter extras.

        Reasoning emission rules:
          - Gated on ``_enable_thinking`` (False ⇒ omit).
          - Prefer ``thinking_level`` (→ ``effort``) over
            ``thinking_budget`` (→ ``max_tokens``) when both are set —
            ``effort`` is more portable across upstreams (OpenAI o-series
            doesn't accept arbitrary token caps).

        Returns:
            Dict to merge into the request body via ``extra_body``.
            Empty when no OpenRouter-specific knobs are configured.
        """
        body: Dict[str, Any] = {}
        if self._provider_routing:
            body["provider"] = self._provider_routing
        if self._enable_thinking:
            reasoning: Dict[str, Any] = {}
            if self._thinking_level:
                reasoning["effort"] = self._thinking_level
            elif self._thinking_budget > 0:
                reasoning["max_tokens"] = self._thinking_budget
            if reasoning:
                body["reasoning"] = reasoning
        return body

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
        """Stateless completion: convert messages, call OpenRouter, return.

        Returns ``TurnResult.from_provider_response(r)`` on success and
        **raises** transient errors so ``with_retry`` can retry.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )

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

        # Sampling parameters from the profile (None ⇒ upstream default).
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["extra_body"] = {"top_k": self._top_k}

        # OpenRouter request-body extras (e.g. ``provider`` routing).  The
        # OpenAI SDK has no typed parameter for these, so we pass them
        # through ``extra_body`` which the SDK merges into the JSON body.
        extra_body = self._build_extra_body()
        if extra_body:
            existing = kwargs.get("extra_body", {})
            existing.update(extra_body)
            kwargs["extra_body"] = existing

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

            # Mirror the streaming-path gate: if reasoning isn't
            # enabled for this session, drop any reasoning content
            # the upstream emitted unsolicited (e.g. DeepSeek-R1
            # always reasons regardless of request kwargs).
            if not self._enable_thinking:
                provider_response.thinking = None

            self._last_usage = provider_response.usage

            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return TurnResult.from_provider_response(provider_response)
        except Exception as e:
            self._handle_api_error(e)
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
        """Streaming loop shared between text and tool-result turns."""
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        accumulated_thinking: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        # Tool calls stream in pieces — accumulate by index.
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block() -> None:
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls() -> None:
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
                    if not tool_id:
                        self._trace(f"ERROR: Missing tool call ID for {func_name}")
                    fc = FunctionCall(
                        id=tool_id,
                        name=original_name,
                        args=args,
                    )
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
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                        self._trace(
                            f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} "
                            f"output={usage.output_tokens}"
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

                    # Reasoning content — OpenRouter normalizes upstream
                    # thinking via either ``reasoning`` or ``reasoning_content``.
                    if self._enable_thinking:
                        for attr in ("reasoning", "reasoning_content"):
                            value = getattr(delta, attr, None)
                            if value and isinstance(value, str):
                                self._trace(f"{trace_prefix}_THINKING len={len(value)}")
                                accumulated_thinking.append(value)
                                if on_thinking:
                                    on_thinking(value)
                                break

                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                self._trace(
                                    f"TOOL_CALL_START idx={idx} "
                                    f"id={tc_delta.id!r} "
                                    f"name={getattr(tc_delta.function, 'name', '')!r}"
                                )
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
                                    acc["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)

                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    )
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(
                f"{trace_prefix}_END chunks={chunk_count} "
                f"finish_reason={finish_reason}"
            )

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        flush_text_block()
        flush_tool_calls()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = "".join(accumulated_thinking) if accumulated_thinking else None

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    # ==================== Error Handling ====================

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to provider-specific exceptions."""
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.RateLimitError):
            retry_after = None
            response = getattr(error, "response", None)
            if response:
                retry_header = getattr(
                    response.headers, "get", lambda *a: None
                )("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        pass
            raise RateLimitError(
                retry_after=retry_after,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.NotFoundError):
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIConnectionError):
            raise InfrastructureError(
                status_code=0,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.InternalServerError):
            status_code = getattr(error, "status_code", 500)
            raise InfrastructureError(
                status_code=status_code,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIStatusError):
            status_code = getattr(error, "status_code", 0)
            error_str = str(error).lower()

            if any(
                x in error_str
                for x in ("context_length", "too large", "max size", "tokens_limit")
            ):
                max_tokens = None
                match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
                if match:
                    max_tokens = int(match.group(1))
                raise ContextLimitError(
                    model=self._model_name or "unknown",
                    max_tokens=max_tokens,
                    original_error=str(error),
                ) from error

            if 500 <= status_code < 600:
                raise InfrastructureError(
                    status_code=status_code,
                    original_error=str(error),
                ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Estimate token count (~4 chars / token).

        OpenRouter doesn't expose a token-counting endpoint, and tokenizer
        choice depends on the upstream model.  The 4-char heuristic is
        good enough for GC budgeting.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Return the context window for the current model."""
        return self._context_length

    def get_token_usage(self) -> TokenUsage:
        """Return token usage from the last response."""
        return self._last_usage

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to a JSON string."""
        return _serialize_history(history)

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize a JSON string back to conversation history."""
        return _deserialize_history(data)

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """OpenRouter passes ``response_format`` through to upstream models."""
        return True

    def supports_streaming(self) -> bool:
        """OpenRouter supports streaming via the OpenAI SDK."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses are cancellable via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """Return True if the current model name suggests reasoning support."""
        return self._is_reasoning_capable()

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Apply a runtime thinking-config update.

        Updates the gate (request emission + response extraction) and
        the budget.  ``thinking_level`` is profile-only — there's no
        equivalent in the framework's ``ThinkingConfig``, so dynamic
        runtime updates can only adjust enabled/budget.  A profile that
        sets ``thinking_level`` keeps it in effect across runtime
        ``set_thinking_config`` calls.
        """
        self._enable_thinking = config.enabled
        if config.budget > 0:
            self._thinking_budget = config.budget

    def _is_reasoning_capable(self) -> bool:
        """Best-effort check for upstream-model reasoning support."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        return any(hint in name_lower for hint in REASONING_CAPABLE_HINTS)

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes."""
        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}
        if isinstance(exc, InfrastructureError):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract a retry-after hint from an exception, if available."""
        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)
        return None


def create_provider() -> OpenRouterProvider:
    """Factory function for plugin discovery."""
    return OpenRouterProvider()
