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

OpenRouter exposes app-level rankings when integrators send the
attribution headers documented at
https://openrouter.ai/docs/app-attribution:

- ``HTTP-Referer``              — site URL (required for app rankings)
- ``X-OpenRouter-Title``        — site / app display name
- ``X-OpenRouter-Categories``   — comma-separated marketplace categories
                                  (e.g. ``cli-agent``)

All three are defaulted (jaato itself attributes as ``cli-agent``);
each can be overridden per-profile or via env vars.

Environment variables
---------------------

    JAATO_OPENROUTER_API_KEY        API key (sk-or-...)
    JAATO_OPENROUTER_BASE_URL       Endpoint (default: https://openrouter.ai/api/v1)
    JAATO_OPENROUTER_MODEL          Default model name
    JAATO_OPENROUTER_CONTEXT_LENGTH Override context window
    JAATO_OPENROUTER_HTTP_REFERER   Attribution: site URL
    JAATO_OPENROUTER_APP_TITLE      Attribution: app title
    JAATO_OPENROUTER_APP_CATEGORIES Attribution: comma-separated categories
    JAATO_OPENROUTER_REQUEST_TIMEOUT     Byte-level request deadline (seconds)
    JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT Streaming payload idle deadline (seconds)
"""

from __future__ import annotations

import copy
import json
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

logger = logging.getLogger(__name__)

from ._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
    resolve_context_window,
    resolve_modalities,
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
    parse_tool_call_arguments,
    StreamInterruptedError,
    require_terminated_stream,
    resolve_tool_use_finish,
)
from .cache import (
    make_cache_control,
    model_supports_explicit_cache,
    resolve_cache_active,
)
from .converters import (
    apply_cache_usage,
    deserialize_history as _deserialize_history,
    get_original_tool_name,
    history_to_openai,
    read_chunk_error,
    read_native_finish_reason,
    read_response_native_finish_reason,
    resolve_choice_finish_reason,
    response_from_openai,
    serialize_history as _serialize_history,
    system_message_with_cache,
    tool_schemas_to_openai,
)
from .._prose_tools import (
    augment_system_with_tools,
    messages_to_prose_chat,
    read_prose_tool_calls_quirk,
    rewrite_prose_tool_calls,
)
from shared.app_identity import AppIdentity, resolve_app_identity

from .env import (
    DEFAULT_BASE_URL,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STREAM_IDLE_TIMEOUT,
    HEADER_APP_CATEGORIES,
    HEADER_APP_TITLE,
    HEADER_HTTP_REFERER,
    MAX_CATEGORIES_PER_REQUEST,
    MAX_CATEGORY_LENGTH,
    get_checked_credential_locations,
    resolve_api_key,
    resolve_app_categories,
    resolve_app_title,
    resolve_base_url,
    resolve_context_length,
    resolve_http_referer,
    resolve_model,
    resolve_request_timeout,
    resolve_stream_idle_timeout,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    OpenRouterError,
    RateLimitError,
    StallTimeoutError,
    UpstreamFinishError,
)
from .stall import StreamStallGuard


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


# Format check for OpenRouter's marketplace category slugs: lowercase
# letters, digits, hyphens.  OpenRouter additionally requires
# recognized values from a curated taxonomy, but unrecognized strings
# are silently dropped server-side — so the regex covers wire shape
# only.  See https://openrouter.ai/docs/app-attribution.
_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _resolve_identity(extra: Dict[str, Any]) -> AppIdentity:
    """Resolve the application identity behind a provider config.

    ``extra["app_identity"]`` is what ``JaatoRuntime._inject_session_extras``
    stamps when an application named itself; it is absent for the framework's
    own identity and for a provider constructed outside a session, in which
    case the environment is consulted directly.  Split out of
    ``initialize()`` so the branch does not add to that method's (already
    baselined) complexity.
    """
    stamped = extra.get("app_identity")
    if stamped:
        return AppIdentity.from_dict(stamped)
    return resolve_app_identity()


def _validate_categories(categories: List[str]) -> List[str]:
    """Validate marketplace categories against OpenRouter's documented rules.

    Returns the input list unchanged on success.  Raises ``TypeError``
    on shape errors and ``ValueError`` on format violations — both
    chosen to fail loud so a typo in a profile doesn't silently send
    a header OpenRouter drops on the floor.

    Validates:
      * Each entry is a non-empty lowercase string matching
        ``[a-z0-9-]+`` (no leading hyphen).
      * Each entry is ≤ :data:`MAX_CATEGORY_LENGTH` characters.
      * The list contains ≤ :data:`MAX_CATEGORIES_PER_REQUEST` entries.
    """
    if not isinstance(categories, list):
        raise TypeError(
            f"OpenRouter app_categories must be a list, "
            f"got {type(categories).__name__}"
        )
    if len(categories) > MAX_CATEGORIES_PER_REQUEST:
        raise ValueError(
            f"OpenRouter accepts at most {MAX_CATEGORIES_PER_REQUEST} "
            f"app_categories per request, got {len(categories)}"
        )
    for entry in categories:
        if not isinstance(entry, str):
            raise TypeError(
                f"OpenRouter app_categories entries must be strings, "
                f"got {type(entry).__name__}: {entry!r}"
            )
        if len(entry) > MAX_CATEGORY_LENGTH:
            raise ValueError(
                f"OpenRouter app_categories entries must be ≤ "
                f"{MAX_CATEGORY_LENGTH} characters; {entry!r} is "
                f"{len(entry)} characters"
            )
        if not _CATEGORY_RE.match(entry):
            raise ValueError(
                f"OpenRouter app_categories entries must be lowercase "
                f"hyphen-separated slugs matching [a-z0-9-]+; got {entry!r}"
            )
    return categories


# OpenRouter sets ``X-Generation-Id`` on every chat / completions
# response (see https://openrouter.ai/docs/api/reference/streaming
# "Additional Information").  Capture is best-effort: both the OpenAI
# SDK's batch response and ``Stream`` wrapper expose the underlying
# httpx response, but the exact attribute path has shifted across
# SDK versions.  This helper tries the documented paths and returns
# ``None`` quietly when none are available — never an exception, since
# the ID is observability-only.
_GENERATION_ID_HEADERS = ("x-generation-id", "X-Generation-Id")


def _extract_generation_id(response_or_stream: Any) -> Optional[str]:
    """Return the value of OpenRouter's ``X-Generation-Id`` header, if any.

    Looks at the object's ``response`` attribute (Stream wrapper),
    then ``_raw_response`` (newer SDK), then the object itself (batch
    response), trying common header accessor shapes (``.headers``
    mapping or dict).  Never raises.
    """
    if response_or_stream is None:
        return None
    for attr in ("response", "_raw_response", None):
        target = (
            getattr(response_or_stream, attr, None) if attr else response_or_stream
        )
        if target is None:
            continue
        headers = getattr(target, "headers", None)
        if headers is None:
            continue
        getter = getattr(headers, "get", None)
        if callable(getter):
            for key in _GENERATION_ID_HEADERS:
                value = getter(key)
                if isinstance(value, str) and value:
                    return value
        elif isinstance(headers, dict):
            for key in _GENERATION_ID_HEADERS:
                value = headers.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _resolve_deadline(value: Any, fallback: float, key: str) -> float:
    """Validate one ``framework_overrides`` deadline knob (#732).

    Args:
        value: The raw profile value, or ``None`` when the profile is
            silent — in which case ``fallback`` (already resolved from
            env / the module default) is returned untouched.
        fallback: Seconds to use when the profile doesn't set the knob.
        key: Knob name, for the error message.

    Returns:
        The deadline in seconds.  ``0`` is legal and means "no deadline"
        — the documented opt-out for a model that legitimately thinks
        for longer than any bound the operator is willing to set.

    Raises:
        ValueError: The value isn't a non-negative number.  Fails loud
            rather than silently reverting to the default, because a
            typo'd deadline is exactly the config whose failure mode is
            an unbounded hang.
    """
    if value is None:
        return fallback
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"OpenRouter framework_overrides.{key} must be a number of "
            f"seconds (0 disables the deadline), got {value!r}"
        ) from None
    if seconds < 0:
        raise ValueError(
            f"OpenRouter framework_overrides.{key} must be >= 0 "
            f"(0 disables the deadline), got {value!r}"
        )
    return seconds


def _retry_after_from_headers(exc: Exception) -> Optional[float]:
    """Read a ``Retry-After`` hint from an SDK error's response HEADERS.

    The header counterpart of :func:`_retry_after_from_body`.  Returns
    ``None`` for anything it cannot parse, so a missing or malformed
    header just falls through to the standard backoff.
    """
    response = getattr(exc, "response", None)
    if not response:
        return None
    raw = getattr(response.headers, "get", lambda *a: None)("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _retry_after_from_body(exc: Exception) -> Optional[float]:
    """Read a ``Retry-After`` hint out of an OpenRouter error BODY.

    ``retry_utils.get_retry_after`` looks at ``exc.retry_after`` and at
    ``exc.response.headers``.  OpenRouter also reports the hint inside the
    JSON payload, at ``error.metadata.headers['Retry-After']`` — so for an
    error carrying it only there, both readers return ``None`` and the
    ladder falls back to a generic guess.

    Measured (jaato #719): a 402 ``in_flight_budget_exhausted`` carrying
    ``Retry-After: 120`` was retried at 1.4s, 1.5s, 3.5s and 6.3s, burning
    every attempt in about thirteen seconds against a server that had said
    to wait two minutes.  The delays themselves are the evidence the hint
    was never found: ``calculate_backoff`` already lets a hint override
    ``max_delay`` (``retry_utils.py:311``), so a hint that had been read
    would have produced a 120s wait.

    Returns ``None`` for anything it cannot parse — the caller then falls
    back to the standard readers, so a malformed body costs nothing.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    meta = err.get("metadata")
    if not isinstance(meta, dict):
        return None
    headers = meta.get("headers")
    if not isinstance(headers, dict):
        return None
    for key in ("Retry-After", "retry-after"):
        raw = headers.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return value if value > 0 else None
    return None


class OpenRouterProvider(ModalityCapabilityMixin):
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

        # Per-session identity stamped by the framework (see initialize).
        # Declared here so ``_build_extra_body`` is safe on the
        # ``initialize(config=None)`` path, which never reaches the
        # ``config.extra`` read.
        self._session_id: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL
        self._http_referer: str = ""
        self._app_title: str = ""

        # The APPLICATION this provider is speaking for — the product built
        # on the SDK, not the framework.  Resolved at initialize() from the
        # identity the runtime stamped onto ``config.extra``, falling back to
        # the environment.  Only used to derive the two attribution values
        # above, but retained whole: a header that wants more than a display
        # name (a ``User-Agent``, which AppIdentity already knows how to
        # render) should not have to re-resolve it.
        self._app_identity: Optional[AppIdentity] = None

        # Marketplace categories for OpenRouter's app rankings — emitted
        # as the ``X-OpenRouter-Categories`` header (comma-separated).
        # Empty list means the header is omitted entirely.  Validated
        # at initialize() against OpenRouter's documented format
        # (lowercase, hyphen-separated, ≤30 chars, ≤5 per request);
        # unrecognized categories are silently dropped server-side.
        # See https://openrouter.ai/docs/app-attribution.
        self._app_categories: List[str] = []

        # Arbitrary additional HTTP headers stamped on every request via
        # the OpenAI client's ``default_headers``.  Profile-only knob;
        # primary use is OpenRouter's provider-specific beta header
        # passthrough (e.g. ``x-anthropic-beta: fine-grained-tool-streaming-2025-05-14``,
        # ``interleaved-thinking-2025-05-14``, ``structured-outputs-2025-11-13``).
        # See https://openrouter.ai/docs/features/provider-routing
        # "Provider-Specific Headers".  Merged into the headers OpenRouter
        # already gets (``HTTP-Referer`` / ``X-OpenRouter-Title`` / ``Authorization``)
        # — user values win on key collisions.
        self._extra_headers: Dict[str, str] = {}

        # Cross-model fallback list — OpenRouter's request-level
        # ``models`` body field (sibling of ``model`` / ``messages``).
        # When set, OpenRouter tries each candidate in order on
        # outage / context-limit / safety failures of the primary.
        # Distinct from ``routing.order`` (which constrains *providers*
        # for a single model) and from any framework-level model
        # fallback (which is profile-scoped, not per-request).
        # Required by routing's ``partition: "none"`` examples
        # (see https://openrouter.ai/docs/features/provider-routing
        # "Advanced Sorting with Partition" / Use Cases 1-3).
        self._models_fallback: List[str] = []

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length: int = 0
        # Optional manual context-window override (framework escape hatch
        # or env var) — the FALLBACK tier; the catalog auto-detect wins.
        self._context_length_knob: Optional[int] = None
        # Optional manual INPUT-modality assertion — the escape hatch for a
        # model the catalog doesn't list yet (or to correct it).  The
        # catalog ``architecture.input_modalities`` auto-detect wins; this
        # is only consulted when the catalog has no entry for the model.
        self._modalities_knob: Optional[List[str]] = None

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
        # ``api_params.max_tokens`` — cap on response size, forwarded as the
        # OpenAI Chat Completions ``max_tokens`` top-level field.  ``None``
        # ⇒ let the upstream apply its own default (typically the model's
        # full output budget, e.g. ~64K for Claude Sonnet 4.5).  Setting
        # this is required to fit requests within OpenRouter's pre-flight
        # credit-vs-max-tokens check on low-balance accounts (the
        # 402 "you requested up to N tokens but can only afford M" error
        # is driven by this knob, not by actual consumption).
        self._max_tokens: Optional[int] = None
        # ``api_params.parallel_tool_calls`` — OpenAI Chat Completions
        # request-body field forwarded verbatim to upstream.  When
        # ``false``, OpenAI / vLLM / compatible servers return at most
        # one tool call per response.  Closes the small-model
        # parallel-batching failure mode (e.g. qwen3-14b via OpenRouter
        # emitting N readFiles + signal_completion in one assistant
        # message).  Default ``None`` ⇒ omit the field, upstream default.
        self._parallel_tool_calls: Optional[bool] = None
        # ``api_params.service_tier`` — OpenAI-style processing-tier
        # selector, forwarded to upstreams that price by tier: ``"flex"``
        # buys ~50%-off best-effort processing (slower responses, may 429
        # under load), ``"priority"`` buys low-latency processing at a
        # premium; ``"auto"`` / ``"default"`` defer to the upstream.
        # OpenRouter passes the field through to tier-supporting upstreams
        # (OpenAI, Gemini, ...) and reports the tier actually used in the
        # response.  Default ``None`` ⇒ omit the field entirely.
        # See https://openrouter.ai/docs/guides/features/service-tiers.
        self._service_tier: Optional[str] = None

        # Strict tool-use mode (server 0.6.118+, 2026-05-16).  When True,
        # ``tool_schema_to_openai`` emits ``"strict": True`` as a sibling
        # of ``parameters`` in each function definition.  OpenRouter
        # forwards this to supported upstreams (Anthropic Sonnet 4.5 /
        # Opus 4.1+, OpenAI GPT-4o+, Gemini, OSS, Fireworks per
        # https://openrouter.ai/docs/guides/features/structured-outputs),
        # which then grammar-constrain tool-argument sampling to the
        # schema.  Unsupported models (e.g. claude-haiku-4.5) are
        # documented as "not in the supported list" — enabling strict
        # for them is at best a no-op, at worst a 400 from upstream.
        # The framework intentionally does NOT auto-rewrite schemas to
        # satisfy OpenAI's strict-mode requirements
        # (``additionalProperties: false`` on every object, exhaustive
        # ``required`` arrays, no ``oneOf``/``anyOf``); kb authors own
        # schema shape.  Wire errors surface mismatches.
        self._strict_tools: bool = False

        # Quirk: prose_tool_calls (opt-in via profile.quirks).  For cheap
        # upstream models that cannot emit native tool calls and answer in
        # prose instead: the tools array is withheld, schemas are
        # prompt-injected (hashed wire ids), tool traffic in history is
        # replayed as text, and fenced tool_call blocks in the response
        # are parsed back into FunctionCall parts.  See
        # shared/plugins/model_provider/_prose_tools.py.
        self._prose_tool_calls: bool = False

        # Request deadlines (#732).  Nothing inside the provider used to
        # bound a single request: a stalled upstream or a silently dead
        # connection left the provider waiting forever, and the timeout
        # was delegated to whoever happened to sit above it (a harness
        # arm-timeout, a budget ceiling).  Two layers now bound it:
        #
        #   _connect_timeout / _request_timeout  BYTE level, enforced by
        #       httpx (connect, and read/write/pool respectively).
        #   _stream_idle_timeout                 PAYLOAD level, enforced
        #       by StreamStallGuard around the streaming chunk loop —
        #       the layer httpx cannot provide, because OpenRouter's
        #       ``: OPENROUTER PROCESSING`` keep-alive comments reset the
        #       byte clock while no chunk is ever yielded.
        #
        # All three are settable per-profile under
        # ``framework_overrides`` (or by env var); ``0`` disables a
        # deadline, restoring the pre-#732 unbounded wait.
        self._connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
        self._request_timeout: float = DEFAULT_REQUEST_TIMEOUT
        self._stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT

        # Cached catalog so connect() can look up per-model context lengths
        # without re-fetching for every model switch.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

        # Thinking/reasoning configuration.  Defaults to False so
        # nothing extra goes on the wire and reasoning content is not
        # surfaced unless the profile / runtime opts in.  Matches the
        # Anthropic provider's default.
        self._enable_thinking: bool = False

        # Prompt-caching configuration.  ``_cache_prompt`` is the raw
        # profile knob (``"auto"`` | True | False | "on"/"off"/...);
        # :func:`resolve_cache_active` interprets it against the active
        # model to decide whether to stamp ``cache_control`` breakpoints
        # on outgoing requests.  ``_cache_ttl`` selects the 5-minute
        # (default) or 1-hour ephemeral cache.
        #
        # Cache usage parsing (``prompt_tokens_details.cached_tokens``,
        # ``cache_creation_input_tokens``, ``cost``) is unconditional —
        # automatic-cache upstreams (OpenAI, DeepSeek, Grok) report
        # savings even without breakpoints, and the daemon's ledger
        # should reflect them either way.
        self._cache_prompt: Any = "auto"
        self._cache_ttl: str = "5m"

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

        # OpenRouter returns ``X-Generation-Id`` on every chat / completions
        # response (see https://openrouter.ai/docs/api/reference/streaming
        # "Additional Information").  Capturing the most recent one lets
        # operators correlate trace logs and ledger entries with
        # OpenRouter's dashboard / support tickets.  Updated by both
        # batch and streaming paths; cleared on shutdown.
        self._last_generation_id: Optional[str] = None

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
            - ``app_categories`` (``List[str]``) — OpenRouter
              marketplace categories for app rankings (emitted as the
              ``X-OpenRouter-Categories`` header).  See
              https://openrouter.ai/docs/app-attribution for the
              taxonomy.  Defaults to :data:`DEFAULT_APP_CATEGORIES`
              (``["cli-agent"]`` for jaato); pass ``[]`` to opt out of
              category attribution entirely.  Validated strictly:
              lowercase hyphen-separated slugs, ≤30 chars each, ≤5
              entries per request.
            - ``extra_headers`` (``Dict[str, str]``) — additional HTTP
              headers stamped on every request via the OpenAI client's
              ``default_headers``.  Used for OpenRouter's
              provider-specific beta-header passthrough:
              ``x-anthropic-beta: fine-grained-tool-streaming-2025-05-14``,
              ``interleaved-thinking-2025-05-14``,
              ``structured-outputs-2025-11-13``, etc.  See
              https://openrouter.ai/docs/features/provider-routing
              "Provider-Specific Headers".
        - ``api_params`` (OpenAI Chat Completions request body fields):
            - ``temperature``, ``top_p``, ``top_k``, ``max_tokens``
            - ``service_tier`` (``"auto"`` / ``"default"`` / ``"flex"`` /
              ``"priority"`` / ``"scale"``) — OpenAI-style processing
              tier, forwarded to tier-supporting upstreams.  ``"flex"``
              trades latency for ~50% off; ``"priority"`` the reverse.
            - ``models`` (``List[str]``) — OpenRouter's request-level
              cross-model fallback list (sibling of ``model`` in the
              request body).  OpenRouter walks each candidate on
              outage / context-limit / safety failures.  Pairs with
              ``routing.sort = {by: ..., partition: "none"}`` to find
              the best provider across all candidate models.
            - ``enable_thinking``, ``thinking_budget``, ``thinking_level``
            - ``cache_prompt`` (``"auto"`` / True / False; default
              ``"auto"`` enables explicit caching only for upstreams
              that need ``cache_control`` breakpoints — Anthropic
              Claude, Gemini 1.5/2.5/3 — and lets other upstreams
              cache automatically)
            - ``cache_ttl`` (``"5m"`` default, ``"1h"`` for the
              extended-TTL variant)
        - ``routing`` (OpenRouter ``provider`` extension — a dict
          forwarded verbatim via ``extra_body`` on every request):
            - Common keys: ``order``, ``allow_fallbacks``,
              ``require_parameters``, ``data_collection``, ``ignore``,
              ``only``, ``quantizations``, ``sort`` (string or
              ``{by, partition}`` object), ``max_price``, ``zdr``,
              ``enforce_distillable_text``, ``preferred_min_throughput``,
              ``preferred_max_latency`` (number or ``{p50, p75, p90, p99}``)
            - Any new OpenRouter routing key works without code changes
              — the dict is opaque pass-through.  See
              https://openrouter.ai/docs/features/provider-routing
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
        #     <top-level>           # auth / identity (api_key, http_referer,
        #                           #   app_title, app_categories, extra_headers)
        #     api_params:           # OpenAI Chat Completions request body params
        #       temperature, top_p, top_k, max_tokens, models (fallback list)
        #       enable_thinking, thinking_budget, thinking_level
        #       cache_prompt, cache_ttl
        #     routing:              # OpenRouter `provider` extension (sort/ignore/
        #                           #   order/only/zdr/preferred_min_throughput/...)
        #     framework_overrides:  # rare escape hatches (context_length, base_url)
        #
        # Backward compatibility: every nested key is also read from the
        # legacy flat position with a one-time deprecation warning.  The
        # flat fallback will be removed in a future server release.
        api_params = config.extra.get("api_params") or {}
        routing_dict = config.extra.get("routing")
        framework_overrides = config.extra.get("framework_overrides") or {}

        # Not a profile knob — the framework stamps this per session in
        # ``JaatoRuntime.create_provider`` so OpenRouter can group a
        # session's requests together (dashboard grouping and
        # session-aware upstream routing).  Absent for callers that
        # construct a provider outside a session, which is why the
        # emission below is conditional rather than assumed.
        self._session_id: Optional[str] = config.extra.get("session_id")

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
        # App attribution, three tiers (highest first): the profile knob,
        # the OpenRouter-specific env var, then the framework-resolved
        # application identity — which is what makes a product built on the
        # SDK report under ITS name rather than jaato's.  The identity is
        # stamped onto ``config.extra`` per session by
        # ``JaatoRuntime._inject_session_extras`` (not a profile knob, same
        # as ``session_id``); absent that — a provider constructed outside a
        # session — it is resolved from the environment here.
        identity = _resolve_identity(config.extra)
        self._app_identity = identity
        self._http_referer = (
            config.extra.get("http_referer") or resolve_http_referer(identity)
        )
        self._app_title = (
            config.extra.get("app_title") or resolve_app_title(identity)
        )

        # Marketplace categories — profile takes precedence over env
        # (and env is parsed from a comma-separated string).  Both
        # paths share the same validation: format violations raise
        # immediately so a typo can't silently invalidate the header.
        # Pass an explicit empty list to opt out of category attribution
        # entirely without touching DEFAULT_APP_CATEGORIES.
        categories_extra = config.extra.get("app_categories")
        if categories_extra is None:
            self._app_categories = _validate_categories(resolve_app_categories())
        else:
            if not isinstance(categories_extra, (list, tuple)):
                raise TypeError(
                    "OpenRouter 'app_categories' config must be a list of "
                    f"strings, got {type(categories_extra).__name__}"
                )
            self._app_categories = _validate_categories(list(categories_extra))

        # Top-level: arbitrary HTTP headers (e.g. ``x-anthropic-beta``).
        # Profile-only; no env-var fallback because headers don't compose
        # well with environment configuration (would mask per-profile
        # intent).  Validated strictly so a typo (``"x-anthropic-beta"``
        # as a string instead of a dict) fails loud rather than being
        # silently dropped on the floor.
        extra_headers_extra = config.extra.get("extra_headers")
        if extra_headers_extra is not None:
            if not isinstance(extra_headers_extra, dict):
                raise TypeError(
                    "OpenRouter 'extra_headers' config must be a dict of "
                    f"str→str, got {type(extra_headers_extra).__name__}"
                )
            normalised: Dict[str, str] = {}
            for k, v in extra_headers_extra.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError(
                        "OpenRouter 'extra_headers' entries must be str→str; "
                        f"got {type(k).__name__} -> {type(v).__name__}"
                    )
                normalised[k] = v
            self._extra_headers = normalised

        # Context window is auto-detected from the OpenRouter catalog
        # (per-model ``context_length``) at connect() — the PRIMARY tier.
        # Stash the optional manual override (framework escape hatch or
        # env var) as the fallback, used only when the catalog has no
        # entry for the model (self-hosted gateways, catalog gaps).
        context_length_extra = _knob(
            "context_length", layer=framework_overrides,
        )
        self._context_length_knob = (
            int(context_length_extra) if context_length_extra
            else resolve_context_length()
        )

        # Optional INPUT-modality assertion (framework escape hatch).  Only
        # consulted when the catalog lacks an entry for the active model.
        modalities_extra = _knob("modalities", layer=framework_overrides)
        if modalities_extra is not None:
            if not isinstance(modalities_extra, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_extra
            ):
                raise TypeError(
                    "OpenRouter 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_extra).__name__}"
                )
            self._modalities_knob = list(modalities_extra)

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

        # Cross-model fallback list (OpenRouter's ``models`` body field).
        # Stored as a clean list of strings so ``_build_extra_body``
        # can forward it verbatim.  Validated strictly — silently
        # accepting a non-list would mask profile typos and result
        # in OpenRouter ignoring the fallback.
        models_extra = _knob("models", layer=api_params)
        if models_extra is not None:
            if not isinstance(models_extra, (list, tuple)):
                raise TypeError(
                    "OpenRouter 'models' (cross-model fallback list) must be "
                    f"a list of strings, got {type(models_extra).__name__}"
                )
            cleaned_models: List[str] = []
            for entry in models_extra:
                if not isinstance(entry, str) or not entry.strip():
                    raise TypeError(
                        "OpenRouter 'models' entries must be non-empty "
                        f"strings, got {entry!r}"
                    )
                cleaned_models.append(entry)
            self._models_fallback = cleaned_models

        temp_extra = _knob("temperature", layer=api_params)
        if temp_extra is not None:
            self._temperature = float(temp_extra)
        top_p_extra = _knob("top_p", layer=api_params)
        if top_p_extra is not None:
            self._top_p = float(top_p_extra)
        top_k_extra = _knob("top_k", layer=api_params)
        if top_k_extra is not None:
            self._top_k = int(top_k_extra)
        strict_tools_extra = _knob("strict_tools", layer=api_params)
        if strict_tools_extra is not None:
            self._strict_tools = bool(strict_tools_extra)
        max_tokens_extra = _knob("max_tokens", layer=api_params)
        if max_tokens_extra is not None:
            self._max_tokens = int(max_tokens_extra)

        parallel_extra = _knob("parallel_tool_calls", layer=api_params)
        if parallel_extra is not None:
            self._parallel_tool_calls = bool(parallel_extra)

        # Processing-tier selector (OpenAI-style ``service_tier``).
        # Validated against the values OpenRouter documents so a profile
        # typo fails loud instead of being silently ignored upstream.
        tier_extra = _knob("service_tier", layer=api_params)
        if tier_extra is not None:
            tier_str = str(tier_extra).lower()
            if tier_str not in ("auto", "default", "flex", "priority", "scale"):
                raise ValueError(
                    "OpenRouter 'service_tier' must be one of 'auto' / "
                    "'default' / 'flex' / 'priority' / 'scale', "
                    f"got {tier_extra!r}"
                )
            self._service_tier = tier_str

        # Model quirks (profile.quirks → config.extra["quirks"]).
        self._prose_tool_calls = read_prose_tool_calls_quirk(
            config.extra, provider=self.name)

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

        # Prompt-caching knobs live alongside the other request-body
        # params.  ``cache_prompt`` defaults to ``"auto"`` (on for
        # explicit-cache upstreams, off otherwise).  ``cache_ttl`` of
        # ``"1h"`` opts in to the extended-TTL variant (double write
        # premium, but no mid-session cache miss on hour-long agents).
        cache_prompt_extra = _knob("cache_prompt", layer=api_params)
        if cache_prompt_extra is not None:
            self._cache_prompt = cache_prompt_extra
        cache_ttl_extra = _knob("cache_ttl", layer=api_params)
        if cache_ttl_extra is not None:
            ttl_str = str(cache_ttl_extra).lower()
            if ttl_str not in ("5m", "1h"):
                raise ValueError(
                    "OpenRouter 'cache_ttl' must be '5m' or '1h', "
                    f"got {cache_ttl_extra!r}"
                )
            self._cache_ttl = ttl_str

        # Request deadlines (#732) — framework escape hatches, not wire
        # fields, so they live under ``framework_overrides`` next to
        # ``context_length`` / ``base_url``.  Profile > env > default.
        self._connect_timeout = _resolve_deadline(
            _knob("connect_timeout", layer=framework_overrides),
            DEFAULT_CONNECT_TIMEOUT, "connect_timeout",
        )
        self._request_timeout = _resolve_deadline(
            _knob("request_timeout", layer=framework_overrides),
            resolve_request_timeout(), "request_timeout",
        )
        self._stream_idle_timeout = _resolve_deadline(
            _knob("stream_idle_timeout", layer=framework_overrides),
            resolve_stream_idle_timeout(), "stream_idle_timeout",
        )

        if not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

        self._client = self._create_client()
        self._trace(
            f"[INIT] client created, base_url={self._base_url}, "
            f"referer={self._http_referer!r}, title={self._app_title!r}, "
            f"categories={self._app_categories!r}, "
            f"provider_routing={'configured' if self._provider_routing else 'none'}, "
            f"connect_timeout={self._connect_timeout:g}s, "
            f"request_timeout={self._request_timeout:g}s, "
            f"stream_idle_timeout={self._stream_idle_timeout:g}s"
        )

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client configured for OpenRouter.

        Sets the optional attribution headers via ``default_headers``
        so every request automatically includes them.  Profile-supplied
        ``extra_headers`` (e.g. ``x-anthropic-beta``) are merged in
        after the framework defaults — profile values win on key
        collisions, so a profile can override the attribution headers
        if needed, but normally just adds new ones (beta opt-ins).

        Two transport-level decisions are made here (#732):

        ``timeout``
            Explicit and configurable rather than the SDK's implicit
            600s default.  ``connect`` is separated out because a TLS
            handshake that hasn't finished in 15s never will, while a
            generation legitimately can run for minutes.  Passed as an
            ``httpx.Timeout`` so read/write/pool track
            ``request_timeout`` independently of connect.  ``0``
            disables the byte-level deadline (``httpx`` reads ``None``
            as "no timeout").

        ``max_retries=0``
            The SDK's default of 2 silently *multiplies* every deadline
            by three and is invisible in the daemon log.  The framework
            already owns retries — ``retry_utils.with_retry``, which
            classifies via :meth:`classify_error`, applies exponential
            backoff and honours the ``Retry-After`` hint OpenRouter puts
            in the response body (#720).  Two nested retry budgets is
            one too many; the framework's is the one that is observable.
        """
        import httpx

        client_class = get_openai_client_class()
        default_headers: Dict[str, str] = {}
        if self._http_referer:
            default_headers[HEADER_HTTP_REFERER] = self._http_referer
        if self._app_title:
            default_headers[HEADER_APP_TITLE] = self._app_title
        if self._app_categories:
            default_headers[HEADER_APP_CATEGORIES] = ",".join(self._app_categories)
        if self._extra_headers:
            default_headers.update(self._extra_headers)

        return client_class(
            base_url=self._base_url,
            api_key=self._api_key,
            default_headers=default_headers or None,
            timeout=httpx.Timeout(
                self._request_timeout or None,
                connect=self._connect_timeout or None,
            ),
            max_retries=0,
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
                    checked_locations=get_checked_credential_locations(config=config)
                )
            return False

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config)
            )

        return False

    def shutdown(self) -> None:
        """Close the OpenAI client and clear connection state."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None
        self._last_generation_id = None

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
        """Set the model and resolve its context window from the catalog.

        Context resolution is auto-detect PRIMARY: the per-model
        ``context_length`` from ``GET /api/v1/models`` wins, falling back
        to an explicit manual override (env / config) and failing fast if
        neither resolves (no hardcoded default).  Real model validation
        happens on the first chat-completion call.

        Args:
            model: Model ID (e.g. ``anthropic/claude-3.5-sonnet``,
                ``openai/gpt-4o``, ``openrouter/auto``).
            skip_model_test: When True, skip the catalog lookup for
                fast bootstrap; context length stays at whatever
                ``initialize()`` resolved.
        """
        self._model_name = model

        # Resolve the context window: catalog auto-detect PRIMARY ->
        # manual override -> fail-fast (no hardcoded default).  The
        # catalog lookup is a public, cacheable HTTP GET
        # (``GET /api/v1/models``) orthogonal to ``skip_model_test`` —
        # that flag is for *real-completion* model validation, not the
        # metadata fetch — so it always runs, even on fast-bootstrap
        # paths.  A stale per-profile context_length can no longer
        # silently under-declare a 200 K-window model.
        self._context_length = resolve_context_window(
            detect_capacity=lambda: self._lookup_context_length(model),
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                "OpenRouter provider: context_length could not be resolved.  "
                "The model is absent from GET /api/v1/models (so no per-model "
                "context_length was found), and no manual override is set.  "
                "Set plugin_configs.openrouter.framework_overrides."
                "context_length in the profile, or "
                "JAATO_OPENROUTER_CONTEXT_LENGTH in the environment.  No "
                "hardcoded fallback exists per the project's no-fallback rule."
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

    def _lookup_modalities(self, model: str) -> Optional[List[str]]:
        """Return the catalog-reported INPUT modalities for ``model``.

        Reads ``architecture.input_modalities`` from
        ``GET /api/v1/models`` — present for all catalog models
        (``["text","image"]`` for vision, ``["text"]`` for text-only).
        Returns ``None`` when the model is absent or the field is missing
        (self-hosted gateways, catalog gaps) so resolution falls through
        to the manual knob.
        """
        for entry in self._fetch_catalog():
            if entry.get("id") == model:
                arch = entry.get("architecture")
                if isinstance(arch, dict):
                    mods = arch.get("input_modalities")
                    if isinstance(mods, list) and mods:
                        return [str(m) for m in mods]
                return None
        return None

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (default: the active model) accepts.

        Catalog auto-detect PRIMARY (``architecture.input_modalities``) →
        manual ``modalities`` knob → text-only floor.  The detect tier
        self-updates with OpenRouter's catalog, so vision models are
        recognised without any per-model configuration.  ``model`` lets
        callers resolve a model other than the connected one (e.g.
        validating a vision-tier model); the catalog lookup is keyed by
        the requested id, so no :meth:`connect` is needed.
        """
        model = model or self._model_name
        if not model:
            return {MODALITY_TEXT}
        resolved = resolve_modalities(
            detect=lambda: self._lookup_modalities(model),
            profile_value=self._modalities_knob,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def _caching_active(self) -> bool:
        """Whether to stamp ``cache_control`` breakpoints on this request.

        Resolved against the active model: ``"auto"`` only flips on for
        explicit-cache upstreams (Anthropic, Gemini 1.5+/2.5+/3+), while
        an explicit True / False profile setting wins regardless.
        """
        return resolve_cache_active(self._cache_prompt, self._model_name)

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
        if self._models_fallback:
            # Cross-model fallback list — sibling of ``model`` in the
            # request body.  OpenRouter walks this list on outage /
            # context-limit / safety failures of the primary ``model``.
            # Forwarded as a fresh list so later mutations of
            # ``self._models_fallback`` don't bleed into in-flight requests.
            body["models"] = list(self._models_fallback)
        if self._enable_thinking:
            reasoning: Dict[str, Any] = {}
            if self._thinking_level:
                reasoning["effort"] = self._thinking_level
            elif self._thinking_budget > 0:
                reasoning["max_tokens"] = self._thinking_budget
            if reasoning:
                body["reasoning"] = reasoning
        # Opt in to OpenRouter's detailed usage block.  Cost and the
        # cache-creation token count are only emitted when this flag is
        # set; standard ``prompt_tokens_details.cached_tokens`` comes
        # back regardless but the opt-in costs us nothing extra.
        if self._session_id:
            # OpenRouter's documented request-body field for grouping the
            # requests of one conversation
            # (https://openrouter.ai/docs) — sent as a sibling of
            # ``model``/``messages`` via ``extra_body``, the same way the
            # official OpenAI-SDK example does it.
            body["session_id"] = self._session_id
        body["usage"] = {"include": True}
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

        cache_active = self._caching_active()
        cache_control = make_cache_control(self._cache_ttl) if cache_active else None

        # Quirk path: models that cannot emit native tool calls get the
        # text-encoded protocol — schemas prompt-injected, tool traffic in
        # history replayed as prose, no tools array on the wire.
        prose_mode = bool(self._prose_tool_calls and tools)

        openai_messages: List[Dict[str, Any]] = []
        if prose_mode:
            system_text = augment_system_with_tools(system_instruction, tools)
            if system_text:
                openai_messages.append(
                    system_message_with_cache(system_text, cache_control)
                )
            openai_messages.extend(messages_to_prose_chat(list(messages)))
        else:
            if system_instruction:
                openai_messages.append(
                    system_message_with_cache(system_instruction, cache_control)
                )
            openai_messages.extend(history_to_openai(list(messages)))

        kwargs: Dict[str, Any] = {}
        if tools and not prose_mode:
            openai_tools = tool_schemas_to_openai(
                tools,
                cache_control=cache_control,
                strict=self._strict_tools,
            )
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
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        if self._service_tier is not None:
            kwargs["service_tier"] = self._service_tier

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
                gen_id = _extract_generation_id(response)
                if gen_id:
                    self._last_generation_id = gen_id
                    self._trace(f"COMPLETE_GENERATION_ID {gen_id}")
                # A non-streamed turn carries the shape too — it is
                # the upstream's verdict, not an artefact of SSE.
                self._reject_error_finish(
                    provider_response.finish_reason,
                    read_response_native_finish_reason(response),
                )

            # Mirror the streaming-path gate: if reasoning isn't
            # enabled for this session, drop any reasoning content
            # the upstream emitted unsolicited (e.g. DeepSeek-R1
            # always reasons regardless of request kwargs).
            if not self._enable_thinking:
                provider_response.thinking = None

            # Prose-mode counterpart of the native tool-call flush: parse
            # fenced tool_call blocks out of the text into FunctionCall
            # parts (no-op on cancelled / call-free responses).
            if prose_mode:
                provider_response = rewrite_prose_tool_calls(
                    provider_response, call_id_prefix=self.name)

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
        # The upstream's own word for why it stopped, before OpenRouter
        # normalised it.  Kept because ``finish_reason`` alone cannot
        # carry it, and for ``error`` it is the ONLY diagnosis there is
        # (#766).  Traced on every outcome, not just failures: a
        # ``UNEXPECTED_TOOL_CALL`` on the turn before a
        # ``MALFORMED_FUNCTION_CALL`` is the adjacency that shortens the
        # next investigation.
        native_finish_reason: Optional[str] = None
        # Whether a choice ever carried a finish reason.  Tracked apart
        # from ``finish_reason`` because the normaliser answers UNKNOWN
        # for a label it does not map, which is still a terminated turn.
        # Absent it entirely, the stream was cut (#687) -- the sibling
        # of the stall guard below, for the case where the connection
        # ends instead of going quiet.
        terminal_seen = False
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        # Tool calls stream in pieces — accumulate by index.
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def record_finish(choice: Any) -> None:
            """Resolve a choice's reported finish reason, keeping the native one.

            Both call sites in the chunk loop go through here so the
            normalised reason and the upstream's raw word never drift
            apart.  The native reason is traced as it arrives (it is
            dropped from the returned :class:`ProviderResponse`, which
            has nowhere to put it) and retained for the ``error``
            outcome, where :meth:`_reject_error_finish` turns it into
            the raised diagnosis.
            """
            nonlocal finish_reason, native_finish_reason, terminal_seen
            terminal_seen = True
            finish_reason = resolve_choice_finish_reason(choice)
            native = read_native_finish_reason(choice)
            if native:
                native_finish_reason = native
                self._trace(
                    f"{trace_prefix}_NATIVE_FINISH_REASON {native!r} "
                    f"normalised={finish_reason}"
                )

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
                    # Unreadable arguments stay unreadable (#750): a parse
                    # failure must not present as a zero-argument call.
                    args, unreadable_args = parse_tool_call_arguments(
                        tc.get("function", {}).get("arguments")
                    )
                    tool_id = tc.get("id")
                    original_name = get_original_tool_name(func_name)
                    if not tool_id:
                        self._trace(f"ERROR: Missing tool call ID for {func_name}")
                    if unreadable_args is not None:
                        self._trace(
                            f"UNREADABLE_TOOL_ARGS name={original_name} "
                            f"chars={len(unreadable_args)}"
                        )
                    fc = FunctionCall(
                        id=tool_id,
                        name=original_name,
                        args=args,
                        unreadable_args=unreadable_args,
                    )
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        response_stream = None
        chunk_count = 0
        # #732: the payload-idle watchdog.  Covers the whole exchange —
        # started before ``create()`` so a POST that never comes back is
        # bounded too, then pinged by every chunk the SDK yields.  The
        # holder exists because the stream to tear down doesn't exist yet
        # when the guard is armed.
        stream_holder: Dict[str, Any] = {}
        guard = StreamStallGuard(
            self._stream_idle_timeout,
            on_stall=partial(self._abort_stalled_stream, stream_holder, trace_prefix),
            name=f"openrouter-stall-{self._agent_id}",
        )
        try:
            self._trace(f"{trace_prefix}_START")
            guard.start()
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            stream_holder["stream"] = response_stream

            # Capture OpenRouter's generation ID for log correlation
            # (header is set on the response that the stream wraps).
            gen_id = _extract_generation_id(response_stream)
            if gen_id:
                self._last_generation_id = gen_id
                self._trace(f"{trace_prefix}_GENERATION_ID {gen_id}")

            for chunk in response_stream:
                guard.ping()
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Mid-stream error per OpenRouter's streaming spec: a
                # top-level ``error`` field arrives in a chunk alongside
                # a sentinel choice with ``finish_reason: "error"``.
                # Raise so retry/error-handling layers see it rather
                # than silently truncating the response.
                chunk_error = read_chunk_error(chunk)
                if chunk_error is not None:
                    msg = chunk_error.get("message") or "Upstream provider error"
                    code = chunk_error.get("code")
                    self._trace(
                        f"{trace_prefix}_MID_STREAM_ERROR code={code!r} msg={msg!r}"
                    )
                    raise InfrastructureError(
                        status_code=0,
                        original_error=(
                            f"{code}: {msg}" if code is not None else str(msg)
                        ),
                    )

                if not chunk.choices:
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                        apply_cache_usage(chunk.usage, usage)
                        self._trace(
                            f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} "
                            f"output={usage.output_tokens} "
                            f"cache_read={usage.cache_read_tokens or 0} "
                            f"cache_write={usage.cache_creation_tokens or 0}"
                        )
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            record_finish(choice)
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
                        record_finish(choice)

                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    )
                    apply_cache_usage(chunk.usage, usage)
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(
                f"{trace_prefix}_END chunks={chunk_count} "
                f"finish_reason={finish_reason}"
            )

            # Shape 2 of OpenRouter's mid-stream error (#766): the
            # sentinel choice arrived WITHOUT a top-level ``error``
            # object, so ``read_chunk_error`` above correctly saw
            # nothing and the only diagnosis is the native reason.
            # Reaching here with ``ERROR`` therefore means shape 2 by
            # construction — shape 1 has already raised.
            self._reject_error_finish(finish_reason, native_finish_reason)

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            elif guard.fired:
                # The watchdog tore the connection down; whatever the
                # parked read raised on the way out is a symptom, not the
                # cause.  Report the stall, and hand back a fresh client —
                # the old one's pool was closed from under it.
                self._rebuild_client_after_connection_error()
                raise self._stall_error(chunk_count) from e
            else:
                raise
        finally:
            guard.stop()
            self._close_stream_quietly(response_stream, trace_prefix)
            # SHAPE B (cancel-leak fix, 2026-06-09): close the openai
            # client's httpx pool when cancelled.  See vllm/provider.py
            # for the empirical evidence — ``response_stream.close()``
            # alone does NOT propagate TCP-FIN to the upstream.
            if was_cancelled:
                self._close_client_quietly()

        if guard.fired and not was_cancelled:
            # The iterator ended quietly once the watchdog closed the
            # connection under it.  That is a truncated turn, not a
            # finished one — returning here would hand the session a
            # response with no finish_reason and no error, which is the
            # silent half of #732.
            #
            # ``not was_cancelled`` because a cancel that lands in the
            # same instant as the deadline is still a cancel: the caller
            # asked for the turn to end, and reporting a stall instead
            # would turn a clean stop into a retryable error.
            self._rebuild_client_after_connection_error()
            raise self._stall_error(chunk_count)

        flush_text_block()
        flush_tool_calls()

        # TOOL_USE fills in an unreported or merely-``stop`` finish; it
        # must not displace a terminal one.  A turn that hit the output
        # cap mid-``arguments`` carries fragments, not a request — see
        # ``resolve_tool_use_finish`` and issue #745.
        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=bool(function_calls) and not was_cancelled,
        )

        thinking = "".join(accumulated_thinking) if accumulated_thinking else None

        # A stream that stopped arriving is not a turn that finished
        # (#687).  Raises rather than returning the fragment.
        return require_terminated_stream(
            ProviderResponse(
                parts=parts,
                usage=usage,
                finish_reason=finish_reason,
                raw=None,
                thinking=thinking,
            ),
            terminal_seen=terminal_seen,
            was_cancelled=was_cancelled,
            provider=self.name,
            model=self._model_name,
            chunks=chunk_count,
        )

    # ==================== Stream Teardown / Stall Handling ====================

    def _close_stream_quietly(self, response_stream: Any, trace_prefix: str) -> None:
        """Close a streaming response, swallowing teardown failures.

        Per the OpenRouter streaming spec ("Stream Cancellation"),
        aborting the connection immediately stops upstream model
        processing and billing on supported providers (OpenAI,
        Anthropic, DeepSeek, Together, ...).  Without this close, an
        abandoned turn keeps the upstream generating until the SDK
        ``Stream`` object is garbage-collected — which on cancellation
        can mean billing for the entire response the caller no longer
        wants.

        Called from two threads: the consumer thread (normal end,
        cancellation, error) and the watchdog thread (a stall, via
        :meth:`_abort_stalled_stream`).  ``None`` is accepted because
        the watchdog can fire before the stream object exists.
        """
        if response_stream is None:
            return
        try:
            response_stream.close()
        except Exception as close_exc:  # pragma: no cover - best effort
            self._trace(
                f"{trace_prefix}_CLOSE_ERROR "
                f"{type(close_exc).__name__}: {close_exc}"
            )

    def _close_client_quietly(self) -> None:
        """Close the OpenAI client's httpx pool, ignoring failures.

        Closing the *stream* is not enough to propagate TCP-FIN to the
        upstream (measured on vllm, 2026-06-09); closing the pool is.
        Used on cancellation and on a stall — in the latter case from the
        watchdog thread, which is what unparks the blocked read.
        """
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def _abort_stalled_stream(
        self, stream_holder: Dict[str, Any], trace_prefix: str,
    ) -> None:
        """Tear down a stalled exchange.  Runs on the watchdog thread.

        Bound to :class:`~.stall.StreamStallGuard` via ``partial`` in
        :meth:`_stream_response`.  Only touches the transport — the
        half-built response belongs to the consumer thread, which
        discovers ``guard.fired`` when its read comes back and raises
        :class:`StallTimeoutError` from there.

        ``stream_holder`` is a one-key dict rather than the stream
        itself because the guard is armed *before* the request is sent,
        so that a POST which never returns is bounded too; the stream
        appears in the holder only once ``create()`` has answered.
        """
        self._trace(
            f"{trace_prefix}_STALL idle_timeout={self._stream_idle_timeout:g}s "
            f"generation_id={self._last_generation_id!r} — closing transport"
        )
        self._close_stream_quietly(stream_holder.get("stream"), trace_prefix)
        self._close_client_quietly()

    def _stall_error(self, chunk_count: int) -> StallTimeoutError:
        """Build the typed error for a stall the watchdog detected."""
        return StallTimeoutError(
            self._stream_idle_timeout,
            chunks_received=chunk_count,
            generation_id=self._last_generation_id,
            model=self._model_name,
        )

    def _reject_error_finish(
        self,
        finish_reason: FinishReason,
        native_reason: Optional[str],
    ) -> None:
        """Raise a named, retryable error for a bare ``"error"`` finish.

        OpenRouter reports a mid-stream upstream failure in two shapes
        and only one of them used to survive (#766): a top-level
        ``error`` object is read by :func:`~.converters.read_chunk_error`
        and raised with the upstream's own words, while
        ``finish_reason: "error"`` *alone* travelled back as an ordinary
        response whose finish reason was ``ERROR``.  The session then
        turned that into a terminal ``RuntimeError("Provider returned an
        error")`` — one string for every cause, and fatal where the
        identical condition in the other shape was retried.

        This closes both halves.  The cause lives in the sibling
        ``native_finish_reason`` (Gemini's ``MALFORMED_FUNCTION_CALL``,
        say), which a :class:`FinishReason` has no room to carry, so it
        rides the exception instead; and
        :class:`~.errors.UpstreamFinishError` is an
        :class:`~.errors.InfrastructureError`, so ``classify_error``
        hands it to ``with_retry`` like any other transient.

        Called unconditionally by both the streaming and non-streamed
        paths — the decision lives here so neither call site grows a
        branch, and so the same condition reads the same way whichever
        one saw it.

        Args:
            finish_reason: The resolved finish reason for the turn.
            native_reason: The upstream's own word for why it stopped,
                or ``None`` when it reported nothing.

        Raises:
            UpstreamFinishError: When *finish_reason* is ``ERROR``.
        """
        if finish_reason is not FinishReason.ERROR:
            return
        self._trace(
            f"UPSTREAM_FINISH_ERROR native={native_reason!r} "
            f"generation_id={self._last_generation_id!r}"
        )
        raise UpstreamFinishError(
            native_reason,
            generation_id=self._last_generation_id,
            model=self._model_name,
        )

    # ==================== Error Handling ====================

    def _rebuild_client_after_connection_error(self) -> None:
        """Discard the HTTP client so the next attempt gets a fresh one.

        An ``APIConnectionError`` can leave the underlying ``httpx``
        transport permanently unusable.  ``with_retry`` retries the CALL,
        not the client, so without this every backoff re-runs against the
        same dead object: the ladder is guaranteed to exhaust, and the
        session stays broken for the rest of its life even after the
        network recovers.  Measured (jaato #705): a session died at 19:57
        and still failed at 20:01 while ``curl`` answered in 30ms, a direct
        provider call in another process succeeded, and a NEW session in
        the SAME daemon completed normally — the only difference being a
        freshly built client.

        Best-effort by design.  A failure to close or rebuild must not
        replace the caller's original error with a second one: the
        connection error is what the caller needs to see and classify, and
        a failed rebuild simply leaves the ladder no worse off than before
        this method existed.
        """
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # noqa: BLE001 - closing a dead transport may throw
            pass
        try:
            self._client = self._create_client()
            self._trace("[RECOVERY] client rebuilt after APIConnectionError")
        except Exception as exc:  # noqa: BLE001
            self._trace(
                f"[RECOVERY] client rebuild FAILED after APIConnectionError: "
                f"{type(exc).__name__}: {exc}"
            )

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to provider-specific exceptions.

        Returns without doing anything for an exception that is already
        one of ours — a mid-stream :class:`InfrastructureError`, a
        :class:`StallTimeoutError` from the idle watchdog, a
        ``StreamInterruptedError`` from a stream that ended without
        saying why (#687) — so the caller re-raises it unchanged.  That
        early exit also keeps the SDK import off a path that has
        nothing to map.
        """
        if isinstance(error, (OpenRouterError, StreamInterruptedError)):
            return

        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.RateLimitError):
            raise RateLimitError(
                retry_after=_retry_after_from_headers(error),
                original_error=str(error),
            ) from error

        if isinstance(error, openai.NotFoundError):
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIConnectionError):
            # Rebuild BEFORE raising: with_retry retries the call, not
            # the client, so a poisoned transport would make every
            # remaining backoff a guaranteed failure (#705).
            self._rebuild_client_after_connection_error()
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

    def get_max_output_tokens(self) -> Optional[int]:
        """Per-request output cap (``max_tokens``) configured via
        ``plugin_configs.openrouter.api_params.max_tokens``.

        Returns ``None`` when no cap is configured.  Used by
        ``JaatoSession``'s pre-flight refuse-send gate to compute
        ``prompt + max_tokens`` against the context window.
        """
        return self._max_tokens

    def get_token_usage(self) -> TokenUsage:
        """Return token usage from the last response."""
        return self._last_usage

    def get_last_generation_id(self) -> Optional[str]:
        """Return OpenRouter's ``X-Generation-Id`` from the most recent call.

        Useful for log correlation, support tickets, and ledger
        annotations — OpenRouter's dashboard indexes individual calls
        by this ID.  ``None`` if no call has been made or the upstream
        didn't expose the header (older SDK versions / unusual paths).
        """
        return self._last_generation_id

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
        """Extract a retry-after hint from an exception, if available.

        Consulted by ``with_retry`` BEFORE the shared
        ``retry_utils.get_retry_after``; returning ``None`` here falls
        through to it, so this only needs to cover what the shared reader
        cannot see — the hint OpenRouter puts in the response BODY rather
        than in an HTTP header (see :func:`_retry_after_from_body`).
        """
        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)
        return _retry_after_from_body(exc)


def create_provider() -> OpenRouterProvider:
    """Factory function for plugin discovery."""
    return OpenRouterProvider()
