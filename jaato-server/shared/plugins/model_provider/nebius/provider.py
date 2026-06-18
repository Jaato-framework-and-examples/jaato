"""Nebius Token Factory model provider implementation.

This provider enables access to AI models through Nebius Token Factory's
serverless inference API.  Nebius exposes an OpenAI-compatible chat
completions API, so this provider uses the ``openai`` Python SDK as its
transport layer.

The provider "bootstraps" the initial endpoint at connect time: it points
at the fixed serverless base URL (``https://api.tokenfactory.nebius.com/v1``,
overridable) and resolves the active model's metadata from the
``GET /v1/models`` catalog — Nebius returns RichModel entries carrying
``context_length`` and ``architecture.modality``, so the context window and
input modalities are auto-detected (the catalog is the PRIMARY tier), with
profile knob / env override fallbacks and a fail-loud "not configured"
error if nothing resolves (no hardcoded fallback).

Supported models include Llama, Qwen, DeepSeek-R1, Mistral, and the other
open models in the Nebius Token Factory catalog (``deepseek-ai/DeepSeek-R1``,
``meta-llama/Llama-3.3-70B-Instruct``, ...).

Authentication (API key, Bearer token):
- JAATO_NEBIUS_API_KEY (jaato namespace) or NEBIUS_API_KEY (vendor's own
  documented variable), or stored credentials via nebius-auth.

Environment variables:
    JAATO_NEBIUS_API_KEY / NEBIUS_API_KEY: API key
    JAATO_NEBIUS_BASE_URL: Endpoint (default: https://api.tokenfactory.nebius.com/v1)
    JAATO_NEBIUS_MODEL: Default model name
    JAATO_NEBIUS_CONTEXT_LENGTH: Override the catalog-detected context window
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

logger = logging.getLogger(__name__)

from ._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
    ThinkingConfig,
    TurnResult,
)
from .converters import (
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .env import (
    DEFAULT_BASE_URL,
    ENV_NEBIUS_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_model,
    is_self_hosted,
    get_checked_credential_locations,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

# Models known to expose reasoning/thinking content via `reasoning_content`.
REASONING_CAPABLE_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek-r1",
]

# OpenAI Chat Completions request-body fields forwarded verbatim from
# ``plugin_configs.nebius.api_params`` onto every ``chat.completions.create``.
# Nebius is a plain OpenAI-compatible endpoint (no routing / ``extra_body``
# extension like OpenRouter), so these are direct ``create()`` kwargs.
# Allowlisted (not blind passthrough) so a typo'd / unsupported key surfaces
# as a profile warning rather than an opaque OpenAI 400, and so we never
# forward a key the OpenAI SDK's ``create()`` signature would reject.
_FORWARDED_API_PARAMS = frozenset({
    "temperature",
    "top_p",
    "max_tokens",
    "tool_choice",
    "parallel_tool_calls",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
})


class NebiusProvider(ModalityCapabilityMixin):
    """Nebius Token Factory model provider.

    Provides access to the Nebius Token Factory model catalog via the
    OpenAI-compatible serverless inference API.  Context window and input
    modalities are bootstrapped from the ``GET /v1/models`` catalog at
    connect time.

    Providers are stateless with respect to conversation history. The
    session owns the canonical message list and passes it to
    ``complete()`` on each call.  Providers hold only connection/auth
    state set by ``initialize()`` and ``connect()``.

    Lifecycle:
        1. ``__init__()`` — create instance (no connections yet)
        2. ``initialize(config)`` — resolve credentials, create OpenAI client
        3. ``connect(model)`` — set the active model
        4. ``complete(messages, ...)`` — stateless completion
        5. ``shutdown()`` — release resources

    Usage:
        provider = NebiusProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('meta/llama-3.1-70b-instruct')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length: int = 0
        # Manual override tiers (the catalog auto-detect is PRIMARY).
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        # Cached ``GET /v1/models`` catalog (RichModel entries carry
        # context_length + architecture.modality).  Fetched once, lazily.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        # OpenAI Chat Completions body fields from
        # plugin_configs.nebius.api_params (temperature, tool_choice,
        # max_tokens, ...), filtered to _FORWARDED_API_PARAMS and forwarded
        # on every complete() call.
        self._api_params: Dict[str, Any] = {}

        # Thinking/reasoning configuration
        self._enable_thinking: bool = True

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent.
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "nebius:main"
        elif self._agent_name:
            prefix = f"nebius:subagent:{self._agent_name}"
        else:
            prefix = f"nebius:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "nebius"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Creates an OpenAI client configured for the Nebius endpoint.
        For hosted Nebius (api.tokenfactory.nebius.com), an API key is required.
        For self-hosted Nebius containers, the key is optional.

        Args:
            config: Configuration with authentication details.
                - api_key: Nebius API key
                - extra['base_url']: Override API endpoint
                - extra['context_length']: Override context window size

        Raises:
            APIKeyNotFoundError: No API key found and endpoint is not self-hosted.
        """
        if config is None:
            config = ProviderConfig()

        # Stash the config so post-init helpers (e.g.
        # ``_get_credential_source_description``) can reuse the
        # workspace_path / config_root that the runtime injected.
        self._config = config

        # Pull workspace_path / config_root from config.extra (injected
        # by JaatoRuntime.create_provider) so credential lookup
        # resolves under the session's explicit config_root rather
        # than relying on the JAATO_CONFIG_ROOT env var (which is
        # unreliable for headless reactor-spawned sessions running in
        # fresh threads outside any active ``_in_workspace`` context).
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        # Resolve configuration
        self._api_key = config.api_key or resolve_api_key(
            workspace_path=_ws_path, config_root=_config_root,
        )
        self._base_url = config.extra.get("base_url") or resolve_base_url()

        # Validate API key (required for the hosted service; optional only
        # when fronted by a local proxy — see is_self_hosted).
        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

        # Stash the manual override tiers.  The PRIMARY tier — Nebius's
        # ``GET /v1/models`` catalog (RichModel.context_length /
        # architecture.modality) — is consulted lazily at connect()/
        # modalities() so init stays a cheap, network-light credential
        # check.  No context resolution (or fail-loud) happens here: the
        # window is bootstrapped from the catalog once the active model is
        # known.
        self._context_length_knob = (
            config.extra.get("context_length") or resolve_context_length()
        )
        modalities_override = config.extra.get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "Nebius 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

        # api_params passthrough — OpenAI Chat Completions body fields
        # (temperature, top_p, max_tokens, tool_choice, parallel_tool_calls,
        # ...) forwarded verbatim on every chat.completions.create.  This is
        # the determinism + tool_choice knob: e.g. api_params.temperature: 0.0
        # for reproducibility, api_params.tool_choice: "required" to force a
        # tool call (models that stall under auto with hashed tool ids).
        api_params = config.extra.get("api_params") or {}
        if api_params:
            if not isinstance(api_params, dict):
                raise TypeError(
                    "Nebius 'api_params' config must be a dict of OpenAI Chat "
                    f"Completions fields, got {type(api_params).__name__}"
                )
            self._api_params = {
                k: v for k, v in api_params.items()
                if k in _FORWARDED_API_PARAMS
            }
            dropped = set(api_params) - _FORWARDED_API_PARAMS
            if dropped:
                logger.warning(
                    "Nebius api_params: ignoring unsupported key(s) %s; "
                    "forwarded fields are %s",
                    sorted(dropped), sorted(_FORWARDED_API_PARAMS),
                )

        # Create client
        self._client = self._create_client()
        self._trace(f"[INIT] client created, base_url={self._base_url}")

    def _create_client(self) -> "OpenAI":
        """Create the OpenAI client configured for Nebius.

        Returns:
            Initialized OpenAI client.
        """
        client_class = get_openai_client_class()
        # For self-hosted without key, use a dummy key (OpenAI SDK requires one)
        api_key = self._api_key or "not-needed"
        return client_class(
            base_url=self._base_url,
            api_key=api_key,
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key without
        ever touching ``self._client`` (not yet initialized).

        ``config`` is honored: a profile-supplied ``api_key`` (e.g.
        ``plugin_configs.nebius.api_key: pass://...``, which the daemon
        expands) takes effect during this pre-init gate.  The runtime hands
        verify_auth a ``ProviderConfig`` whose ``extra`` carries the
        provider's profile overrides (``ProviderConfig(extra=
        plugin_configs[provider])``), so the expanded key lands in
        ``config.extra['api_key']`` (and may also be on ``config.api_key``).
        Without reading it here, a valid profile credential is rejected
        before ``initialize()`` — which DOES honor it — ever runs.  Mirrors
        openrouter / zhipuai.

        When the stored credential file exists but cannot be loaded
        (corrupt JSON, permission error, missing ``api_key`` field),
        the failure reason is surfaced via ``on_message`` instead of
        being swallowed into "No credentials found".  Without this a
        broken auth file produces the same message as a missing one,
        hiding the real problem from the user.

        Args:
            allow_interactive: Ignored (Nebius uses API keys only).
            on_message: Optional callback for status messages.
            config: Optional profile ``ProviderConfig``; its ``api_key`` /
                ``extra['api_key']`` is consulted first.

        Returns:
            True if authentication is configured or endpoint is self-hosted.

        Raises:
            APIKeyNotFoundError: If no key found and not self-hosted.
        """
        import os
        from .auth import try_load_credentials_with_reason
        from .env import ENV_NEBIUS_API_KEY

        base_url = resolve_base_url()

        # Profile-supplied key wins — the daemon expands pass:// secrets into
        # the verify-time ProviderConfig (extra carries plugin_configs[nebius]),
        # so without this the pre-init gate rejects a perfectly good
        # profile-level credential that initialize() would accept.
        profile_key: Optional[str] = None
        if config is not None:
            profile_key = config.api_key or (
                config.extra.get("api_key") if config.extra else None
            )
        if profile_key:
            if on_message:
                on_message("Found Nebius API key (profile config)")
            return True

        # Check env var first (highest priority)
        env_key = os.environ.get(ENV_NEBIUS_API_KEY)
        if env_key:
            if on_message:
                on_message("Found Nebius API key (environment variable)")
            return True

        # Check stored credentials, differentiating missing-file from
        # broken-file so users can see parse/read errors instead of a
        # generic "not configured" message.
        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found Nebius API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"Nebius credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'nebius-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_NEBIUS_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations()
                )
            return False

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted Nebius endpoint ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used.

        Differentiates between env var, stored credentials, and self-hosted.

        Returns:
            Human-readable auth description.
        """
        import os
        from .env import ENV_NEBIUS_API_KEY

        if is_self_hosted(self._base_url):
            return f"Self-hosted Nebius ({self._base_url})"

        if os.environ.get(ENV_NEBIUS_API_KEY):
            return f"Nebius API key ({ENV_NEBIUS_API_KEY})"

        try:
            from .auth import get_credential_file_path
            # Threading config_root through the auth resolver — see
            # the connect() docstring for the rationale.
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"Nebius API key ({cred_path})"
        except ImportError:
            pass

        return "Nebius API key"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window.

        Context resolution is catalog auto-detect PRIMARY: the per-model
        ``context_length`` from ``GET /v1/models`` (RichModel) wins, then a
        manual override (profile knob / ``JAATO_NEBIUS_CONTEXT_LENGTH``),
        else a fail-fast error (no hardcoded default).  The catalog lookup
        is a cacheable HTTP GET, orthogonal to ``skip_model_test`` (which
        gates *real-completion* validation, not metadata), so it always
        runs.  Real model validation happens on the first chat call.

        Args:
            model: Model ID (e.g. ``deepseek-ai/DeepSeek-R1``,
                ``meta-llama/Llama-3.3-70B-Instruct``).
            skip_model_test: Accepted for protocol compatibility; this
                provider already defers completion validation to the first
                API call.
        """
        self._model_name = model
        self._context_length = resolve_context_window(
            detect_capacity=lambda: self._lookup_context_length(model),
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                "Nebius provider: context_length could not be resolved.  The "
                f"model {model!r} is absent from GET /v1/models (so no per-model "
                "context_length was found), and no manual override is set.  Set "
                "plugin_configs.nebius.context_length in the profile, or "
                f"{ENV_NEBIUS_CONTEXT_LENGTH} in the environment.  No hardcoded "
                "fallback exists per the project's no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} context_length={self._context_length}"
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
        """List available models from Nebius's ``GET /v1/models`` catalog.

        Returns model IDs sorted alphabetically.  Returns an empty list on
        network failure — callers surface that as a clear error rather than
        a fake catalog.

        Args:
            prefix: Optional model-id prefix filter (e.g. ``"meta-llama/"``).

        Returns:
            Sorted list of model IDs.
        """
        ids = [entry["id"] for entry in self._fetch_catalog() if entry.get("id")]
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    # ==================== Catalog (bootstrap metadata) ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the Nebius ``GET /v1/models`` catalog.

        RichModel entries carry ``context_length`` and
        ``architecture.modality`` (an ``"input->output"`` string, e.g.
        ``"text->text"``).  Returns the cached list on subsequent calls so
        per-connect lookups don't re-hit the network.  An empty list is
        returned and *not* cached on failure, so the next call retries.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = httpx.get(url, headers=headers, timeout=15)
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

    def _lookup_modalities(self, model: str) -> Optional[List[str]]:
        """Return the catalog-reported INPUT modalities for ``model``.

        Parses ``architecture.modality`` — Nebius uses the OpenRouter-style
        ``"input->output"`` form (e.g. ``"text->text"``, or
        ``"text+image->text"`` for a vision model).  The INPUT side (left of
        ``->``) is split on ``+``/``,`` and the known modality tokens
        (text / image / audio / video) are extracted.  Returns ``None`` when
        the model is absent or the field is missing, so resolution falls
        through to the manual knob / text floor.
        """
        for entry in self._fetch_catalog():
            if entry.get("id") != model:
                continue
            arch = entry.get("architecture")
            if not isinstance(arch, dict):
                return None
            modality = arch.get("modality")
            if not isinstance(modality, str) or not modality.strip():
                return None
            input_side = modality.split("->", 1)[0]
            tokens = re.split(r"[+,/\s]+", input_side.strip().lower())
            known = {"text", "image", "audio", "video"}
            mods = [t for t in tokens if t in known]
            return mods or None
        return None

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (default: the active model) accepts.

        Catalog auto-detect PRIMARY (``architecture.modality``) → manual
        ``modalities`` knob → text-only floor.  ``model`` lets callers
        resolve a model other than the connected one (the catalog lookup is
        keyed by id, so no :meth:`connect` is needed).
        """
        model = model or self._model_name
        if not model:
            return {MODALITY_TEXT}
        resolved = resolve_modalities(
            detect=lambda: self._lookup_modalities(model),
            profile_value=self._modalities_knob,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    # ==================== Stateless Completion ====================

    def _apply_api_params(
        self, kwargs: Dict[str, Any], tool_choice: Optional[Any],
    ) -> None:
        """Forward profile ``api_params`` (+ per-call ``tool_choice``) into the
        ``chat.completions.create`` kwargs.

        Profile fields (``plugin_configs.nebius.api_params``, already filtered
        to :data:`_FORWARDED_API_PARAMS` at init) apply to every call; a
        per-call ``tool_choice`` overrides the profile's.  ``tool_choice`` is
        dropped when no tools are present this turn — OpenAI rejects
        ``tool_choice`` without ``tools`` — so a profile-wide
        ``tool_choice: "required"`` doesn't break tool-less calls (e.g. GC
        summarization).
        """
        for key, value in self._api_params.items():
            kwargs[key] = value
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if "tool_choice" in kwargs and "tools" not in kwargs:
            kwargs.pop("tool_choice")

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
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to OpenAI format, call API, return response.

        The caller (session) is responsible for maintaining the message
        list and passing it in full each call.  This method does not hold
        any conversation state.

        Returns ``TurnResult.from_provider_response(r)`` on success and
        **raises** transient errors for ``with_retry``.

        Args:
            messages: Full conversation history in provider-agnostic Message
                format.  Must already include the latest user message or tool
                results — the provider does not append anything.
            system_instruction: System prompt text.
            tools: Available tool schemas.
            response_schema: Optional JSON Schema for structured output.
            cancel_token: Optional cancellation signal.
            on_chunk: If provided, enables streaming mode.
            on_usage_update: Real-time token usage callback (streaming).
            on_function_call: Callback when function call detected mid-stream.
            on_thinking: Callback for extended thinking content.
            tool_choice: Optional per-call ``tool_choice`` override (the
                session's lifecycle-retry path).  When set, overrides the
                profile's ``api_params.tool_choice`` for this call only.

        Returns:
            A ``TurnResult`` classifying the outcome.

        Raises:
            RuntimeError: If provider is not initialized/connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Build OpenAI-format messages from explicit parameters
        openai_messages: List[Dict[str, Any]] = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(history_to_openai(list(messages)))

        # Build kwargs
        kwargs: Dict[str, Any] = {}
        if tools:
            openai_tools = tool_schemas_to_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        # Profile api_params (temperature / tool_choice / max_tokens / ...)
        # apply to every call; a per-call tool_choice overrides the profile.
        # Injected into the shared kwargs here so BOTH the streaming and
        # batch paths below forward them.
        self._apply_api_params(kwargs, tool_choice)

        try:
            if on_chunk:
                # Streaming mode
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
                # Batch mode
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    **kwargs,
                )
                provider_response = response_from_openai(response)

            # Per-call accounting (NOT conversation state)
            self._last_usage = provider_response.usage

            # Parse structured output if schema was requested
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
        """Core streaming loop shared by send_message and send_tool_results.

        Accumulates text chunks, tool call deltas, reasoning content, and
        usage information from the streaming response. Handles cancellation
        via cancel_token.

        Args:
            messages: OpenAI-format message list.
            kwargs: Additional kwargs for chat.completions.create().
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            on_usage_update: Optional usage callback.
            on_thinking: Optional callback for reasoning/thinking chunks.
            trace_prefix: Prefix for trace logging.

        Returns:
            ProviderResponse with accumulated response.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        accumulated_thinking: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        # Track tool call accumulation (streaming sends tool calls in pieces)
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            """Flush accumulated tool calls as Parts."""
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

        response_stream = None
        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    # Final chunk may have only usage
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                        self._trace(f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens}")
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    # Extract reasoning/thinking (e.g. DeepSeek-R1 on Nebius)
                    if self._enable_thinking:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning and isinstance(reasoning, str):
                            self._trace(f"{trace_prefix}_THINKING len={len(reasoning)}")
                            accumulated_thinking.append(reasoning)
                            if on_thinking:
                                on_thinking(reasoning)

                    # Accumulate text
                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    # Accumulate tool calls (they come in pieces)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                self._trace(f"TOOL_CALL_START idx={idx} id={tc_delta.id!r} name={getattr(tc_delta.function, 'name', '')!r}")
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

                    # Extract finish reason
                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)

                # Extract usage from chunk (some providers include it on each chunk)
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    )
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise
        finally:
            # Close the underlying HTTP connection so Nebius stops generating
            # immediately on cancel. The SDK ``Stream`` object only sends
            # TCP-close at garbage-collection time, which on a cancelled
            # turn keeps the upstream billing the entire response.
            if response_stream is not None:
                try:
                    response_stream.close()
                except Exception as close_exc:  # pragma: no cover - best effort
                    self._trace(
                        f"{trace_prefix}_CLOSE_ERROR "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )

            # SHAPE B (cancel-leak fix, 2026-06-09): close the openai
            # client's httpx pool when cancelled.  See vllm/provider.py
            # for the empirical evidence — ``response_stream.close()``
            # alone does NOT propagate TCP-FIN to the upstream.
            if was_cancelled and self._client is not None:
                try:
                    self._client.close()
                except Exception:  # pragma: no cover - best effort
                    pass

        # Flush remaining text and tool calls
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
        """Handle API errors and convert to provider-specific exceptions.

        Maps OpenAI SDK exception types to jaato error types for
        consistent error handling across the framework.

        Args:
            error: The original exception from the OpenAI SDK.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(
                original_error=str(error),
            ) from error

        if isinstance(error, openai.RateLimitError):
            retry_after = None
            # Try to extract retry-after from response headers
            response = getattr(error, "response", None)
            if response:
                retry_header = getattr(response.headers, "get", lambda *a: None)("retry-after")
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

            # Context limit errors
            if any(x in error_str for x in ("context_length", "too large", "max size", "tokens_limit")):
                max_tokens = None
                match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
                if match:
                    max_tokens = int(match.group(1))
                raise ContextLimitError(
                    model=self._model_name or "unknown",
                    max_tokens=max_tokens,
                    original_error=str(error),
                ) from error

            # 5xx infrastructure errors
            if 500 <= status_code < 600:
                raise InfrastructureError(
                    status_code=status_code,
                    original_error=str(error),
                ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Uses a heuristic estimate (~4 chars per token) since Nebius does
        not provide a token counting endpoint.

        Args:
            content: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Resolved at :meth:`connect` — catalog auto-detect PRIMARY (the
        per-model ``context_length`` from ``GET /v1/models``), then a manual
        override (``plugin_configs.nebius.context_length`` /
        ``JAATO_NEBIUS_CONTEXT_LENGTH``).

        Returns:
            Maximum tokens the model can handle.
        """
        return self._context_length

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if structured output (json_object response format) is supported.

        Returns:
            True — Nebius's OpenAI-compatible API supports response_format.
        """
        return True

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True — Nebius supports streaming via the OpenAI-compatible API.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True — streaming responses can be cancelled via cancel_token.
        """
        return True

    def supports_thinking(self) -> bool:
        """Check if reasoning/thinking content is supported.

        Returns True for models known to expose ``reasoning_content``
        (e.g. DeepSeek-R1). Other models return False.

        Returns:
            True if the current model exposes reasoning content.
        """
        return self._is_reasoning_capable()

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set thinking configuration.

        For reasoning-capable models this enables/disables extraction of
        ``reasoning_content`` from responses.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        self._enable_thinking = config.enabled

    def _is_reasoning_capable(self) -> bool:
        """Check if the current model exposes reasoning content."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        for prefix in REASONING_CAPABLE_MODELS:
            if name_lower.startswith(prefix) or name_lower.endswith(prefix):
                return True
        return False

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Args:
            exc: The exception to classify.

        Returns:
            Classification dict or None to use global fallback.
        """
        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}

        if isinstance(exc, InfrastructureError):
            return {"transient": True, "rate_limit": False, "infra": True}

        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.
        """
        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)

        return None


def create_provider() -> NebiusProvider:
    """Factory function for plugin discovery.

    Returns:
        A new NebiusProvider instance.
    """
    return NebiusProvider()
