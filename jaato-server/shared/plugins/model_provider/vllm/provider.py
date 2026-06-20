"""vLLM provider using the OpenAI-compatible API server.

vLLM exposes its inference engine through a faithful OpenAI-compatible
REST surface:

- Chat:    ``POST /v1/chat/completions``
- List:    ``GET  /v1/models``
- Health:  ``GET  /health``

Chat completions run through the OpenAI SDK exactly like the ``lmstudio``,
``nim``, and ``tensorrt_llm`` providers.  The provider is **passive** —
it talks to a vLLM server the user has already launched.  Model choice
(``--model``), context length (``--max-model-len``), tool-call parser
(``--enable-auto-tool-choice --tool-call-parser <name>``), tensor
parallelism, GPU memory utilization, quantization, and all other engine
knobs live at the server-launch boundary; there is no in-band load
endpoint analogous to LM Studio's ``/api/v1/models/load``.

Authentication: vLLM has a native ``--api-key <token>`` server flag.
When set, clients must send ``Authorization: Bearer <token>`` on every
request.  When unset, vLLM accepts any non-empty ``api_key`` value (the
OpenAI Python SDK requires one; the conventional placeholder is
``"EMPTY"``).  When ``VLLM_API_TOKEN`` (or
``plugin_configs.vllm.api_token``) is set, it is forwarded as the
bearer.

Researched against vLLM stable docs (https://docs.vllm.ai/en/stable/)
on 2026-06-07 via the context7 MCP — see the package ``__init__.py``
docstring for the per-endpoint findings.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

# Reuse NIM's OpenAI lazy imports and converters — identical SDK,
# identical wire format.
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
    resolve_context_window,
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
from shared.tool_id_map import name_to_id
from .env import (
    ENV_CONTEXT_LENGTH,
    ENV_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
    resolve_model,
)
from .errors import (
    VLLMAuthenticationError,
    VLLMConnectionError,
    VLLMMidStreamError,
    VLLMModelNotFoundError,
)


logger = logging.getLogger(__name__)


def _extract_cache_tokens(usage_obj) -> Optional[int]:
    """Extract cached_tokens from an OpenAI usage object, if present.

    vLLM emits prefix-cache reuse stats in the OpenAI-standard shape
    ``usage.prompt_tokens_details.cached_tokens`` when prefix caching is
    enabled (the default since vLLM 0.5.x).  Returns ``None`` when the
    server omits the field.
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


class VLLMProvider(ModalityCapabilityMixin):
    """vLLM provider talking to the OpenAI-compatible ``/v1`` endpoint.

    Lifecycle:
        1. ``__init__()``              — no state yet
        2. ``initialize(config)``      — resolve host / token / context override,
                                         create the OpenAI client, probe ``/health``
        3. ``connect(model)``          — verify the model in ``/v1/models``
                                         matches the requested model name
        4. ``complete(messages, ...)`` — stateless chat
        5. ``shutdown()``              — release the OpenAI client
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional["OpenAI"] = None
        self._model_name: Optional[str] = None

        # Host + context_length are required at session init time; the
        # empty-string / None sentinels here are placeholders until
        # ``initialize()`` validates them.  No localhost / 8K fallback.
        self._host: str = ""
        self._api_token: Optional[str] = None
        self._auth_info: str = ""

        self._last_usage: TokenUsage = TokenUsage()
        self._context_length_override: Optional[int] = None
        # Cap on the per-request output budget, forwarded as the OpenAI
        # Chat Completions ``max_tokens`` top-level field.  ``None`` ⇒
        # let vLLM apply its own default (which depends on the model's
        # generation_config and ``--max-model-len``).  Set this knob when
        # cascade workloads with large prompts would otherwise leave too
        # little budget under ``max_model_len``, surfacing as a
        # mid-stream connection drop.  See the symmetric handling in
        # ``TensorRTLLMProvider`` for the empirical pattern.
        self._max_tokens: Optional[int] = None

        # OpenAI Chat Completions ``parallel_tool_calls`` request-body field.
        # Per vLLM stable docs (verified 2026-06-09 via context7): when
        # ``false``, the server returns "only zero or one tool call"
        # regardless of whether the underlying tool-call parser
        # (hermes / llama3_json / etc.) would otherwise emit multiple.
        # Default ``None`` ⇒ omit the field, letting vLLM apply its own
        # default (``true``).  Closes the small-model parallel-batching
        # failure mode where qwen3-14b + hermes parser emits N readFiles
        # + signal_completion in one assistant message, bypassing any
        # persona-prose narration-between-tool-calls pattern.
        self._parallel_tool_calls: Optional[bool] = None

        # Quirk: coerce_typed_tool_args (server 0.6.194+).  When set
        # via ``profile.quirks.coerce_typed_tool_args``, ``flush_tool_calls``
        # walks each emitted function call's args and, for any string
        # value whose tool-schema property type is array / object /
        # integer / number / boolean, attempts ``ast.literal_eval``
        # (handles Python repr with single quotes) with ``json.loads``
        # fallback.  Workaround for Llama 3.1 on vLLM 0.22 with the
        # ``llama3_json`` parser under ``tool_choice: "auto"`` —
        # stringifies typed args because vLLM has not registered a
        # structural-tag enforcement for that parser.  See
        # ``feedback_llama31_vllm_auto_mode_stringifies_args``.
        self._coerce_typed_tool_args: bool = False

        # Quirk: force_tool_choice_for_lifecycle (server 0.6.195+).
        # When set via ``profile.quirks.force_tool_choice_for_lifecycle``,
        # ``complete()`` honors the per-call ``tool_choice`` kwarg the
        # session passes after a lifecycle-tool ``validation_failed``
        # return.  Wire shape forwarded verbatim:
        # ``{"type": "function", "function": {"name": <tool_name>}}``.
        # vLLM 0.22 engages xgrammar decoding when ``tool_choice`` is
        # a named function — constraining generation to the tool's
        # parameter JSON schema, which produces correctly-typed args
        # at the source.  Unlike ``coerce_typed_tool_args`` (parser-
        # tag-gated to ``deepseek_v4`` / ``qwen_3_5`` per peer's
        # vLLM-side diagnosis), named-function ``tool_choice`` engages
        # xgrammar universally — so Llama 3.1 8B AWQ benefits too.
        # When OFF (default), the session's ``tool_choice`` kwarg is
        # ignored and vLLM uses its auto-mode default.  See
        # ``project_backlog_vllm_provider_typed_tool_args`` for the
        # two-path prescription (this is Path 1).
        self._force_tool_choice_for_lifecycle: bool = False

        # Quirk: force_narration_between_tools (server 0.6.197+).  When
        # set via ``profile.quirks.force_narration_between_tools``, the
        # session injects a synthetic ``USER``-role prompt asking the
        # model to extract observations in 1-2 sentences after every
        # ``tool_result`` append (see
        # ``JaatoSession._send_tool_results_to_provider``).  Closes the
        # small-model narration-skipping failure class — qwen3-14b @
        # temp=0 in tool-mode skips narration regardless of persona
        # prose AND in-context examples (empirically falsified
        # 2026-06-09).  Provider-instance attribute so the session
        # reads it via ``getattr(self._provider,
        # 'force_narration_between_tools', False)`` symmetric with the
        # existing quirk-attribute pattern.  See
        # ``feedback_small_model_narration_skipping_is_structural``.
        self._force_narration_between_tools: bool = False

        # Quirk: auto_finalize_on_complete (server 0.6.199+).  When set
        # via ``profile.quirks.auto_finalize_on_complete``, the framework
        # auto-synthesizes ``signal_completion()`` server-side the
        # instant the COMPOSITE is_complete flips True (schema floor met
        # AND no ``phase: "completeness"`` processor reported
        # ``incomplete[]``).  Closes the context-overflow-at-finalize
        # death: the accumulator model fills the rich payload and then
        # over-runs the context window trying to take another turn —
        # synthesizing in-process (no model round-trip, see
        # ``LifecycleTools._execute_prepare_completion``) ends the turn
        # via the PR-255 termination BEFORE the oversized next request is
        # built.  Provider-instance attribute so ``LifecycleTools`` reads
        # it via ``getattr(self._provider, '_auto_finalize_on_complete',
        # False)`` — symmetric with ``force_narration_between_tools`` /
        # ``force_tool_choice_for_lifecycle``.  Decoupled from the
        # completeness gate itself: a profile can declare a
        # ``phase: "completeness"`` processor for is_complete GUIDANCE
        # alone (surfacing ``still_needed``) and leave this quirk off to
        # let the model self-finalize.
        self._auto_finalize_on_complete: bool = False

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
            prefix = "vllm:main"
        elif self._agent_name:
            prefix = f"vllm:subagent:{self._agent_name}"
        else:
            prefix = f"vllm:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "vllm"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        Reads host / optional token / context override from ``config.extra``
        (populated from the session profile's ``plugin_configs['vllm']``),
        then verifies the server is reachable via ``GET /health``.

        Args:
            config: ``ProviderConfig`` whose ``extra`` dict may contain:

                - ``host`` (str): Override ``VLLM_HOST``.
                - ``context_length`` (int): Manual context-window
                  override (tier-2).  Normally unnecessary — the window
                  is auto-detected from the server's ``/v1/models``
                  ``max_model_len`` (tier-1).  Set this only for older
                  vLLM builds that don't surface ``max_model_len``, or to
                  pin a value.  See ``resolve_context_window``.
                - ``api_token`` (str): Bearer token when vLLM was
                  launched with ``--api-key <token>`` or when an auth
                  proxy fronts the server.
                - ``max_tokens`` (int): Cap on per-request output budget;
                  forwarded as the OpenAI Chat Completions ``max_tokens``
                  field.  Omit to let vLLM apply its own default.

        Raises:
            VLLMConnectionError: Server not reachable.
            VLLMAuthenticationError: Server returned HTTP 401 (bad
                ``--api-key`` or auth-proxy rejection).
        """
        self._trace("[INIT] Starting initialization")

        if config is None:
            config = ProviderConfig()

        host_value = config.extra.get("host") or resolve_host()
        if not host_value:
            raise ValueError(
                f"vLLM provider: host is not configured.  Set {ENV_HOST} "
                f"in the environment, or plugin_configs.vllm.host in the "
                f"profile.  No hardcoded localhost fallback exists per "
                f"the project's no-fallback rule."
            )
        self._host = host_value.rstrip("/")
        self._api_token = config.extra.get("api_token") or resolve_api_token()

        # Context window resolution precedence (see ``resolve_context_window``):
        #   1. PRIMARY  — auto-detect ``max_model_len`` from the live server's
        #      GET /v1/models (current vLLM versions surface it; verified live).
        #   2. fallback — plugin_configs.vllm.context_length.
        #   3. fallback — VLLM_CONTEXT_LENGTH env var.
        # The server is the source of truth and self-updates with the engine's
        # launched ``--max-model-len`` (incl. rope/YARN scaling), so a stale
        # per-stage profile value can't silently under-declare the window.
        self._context_length_override = resolve_context_window(
            detect_capacity=self._detect_context_capacity,
            profile_value=config.extra.get("context_length"),
            env_value=resolve_context_length(),
        )
        if not self._context_length_override:
            raise ValueError(
                f"vLLM provider: context_length could not be resolved.  The "
                f"server's GET /v1/models did not report max_model_len (older "
                f"vLLM, or unreachable), and no manual override is set.  Set "
                f"plugin_configs.vllm.context_length in the profile, or "
                f"{ENV_CONTEXT_LENGTH} in the environment.  No hardcoded "
                f"fallback exists per the project's no-fallback rule."
            )

        max_tokens_extra = config.extra.get("max_tokens")
        if max_tokens_extra is not None:
            self._max_tokens = int(max_tokens_extra)

        # ``parallel_tool_calls`` request-body field — flat key, symmetric
        # with ``max_tokens``.  Migrates to ``api_params.parallel_tool_calls``
        # when the namespacing backlog is addressed (per
        # ``project_backlog_provider_config_namespacing``).
        parallel_extra = config.extra.get("parallel_tool_calls")
        if parallel_extra is not None:
            self._parallel_tool_calls = bool(parallel_extra)

        # Read profile-declared quirks (see ``self._coerce_typed_tool_args``
        # docstring).  ``quirks`` is a dict injected by ``core.py`` /
        # ``subagent/plugin.py`` at session bootstrap from the top-level
        # ``profile.quirks`` field.  Unknown keys log a warning and are
        # ignored — that surfaces profile typos.
        quirks = config.extra.get("quirks") or {}
        if not isinstance(quirks, dict):
            logger.warning(
                "vLLM provider: ignoring non-dict quirks (got %s)",
                type(quirks).__name__,
            )
            quirks = {}
        self._coerce_typed_tool_args = bool(
            quirks.get("coerce_typed_tool_args", False)
        )
        self._force_tool_choice_for_lifecycle = bool(
            quirks.get("force_tool_choice_for_lifecycle", False)
        )
        self._force_narration_between_tools = bool(
            quirks.get("force_narration_between_tools", False)
        )
        self._auto_finalize_on_complete = bool(
            quirks.get("auto_finalize_on_complete", False)
        )
        _KNOWN_QUIRKS = frozenset({
            "coerce_typed_tool_args",
            "force_tool_choice_for_lifecycle",
            "force_narration_between_tools",
            "auto_finalize_on_complete",
        })
        for unknown_quirk in set(quirks) - _KNOWN_QUIRKS:
            logger.warning(
                "vLLM provider: ignoring unknown quirk %r (known: %s)",
                unknown_quirk, sorted(_KNOWN_QUIRKS),
            )
        if self._coerce_typed_tool_args:
            self._trace(
                "[INIT] quirk coerce_typed_tool_args ENABLED — "
                "string args will be ast.literal_eval'd / json.loads'd "
                "before validation",
            )
        if self._force_tool_choice_for_lifecycle:
            self._trace(
                "[INIT] quirk force_tool_choice_for_lifecycle ENABLED — "
                "session-supplied tool_choice will be forwarded to "
                "vLLM so xgrammar constrains lifecycle-tool retries",
            )
        if self._auto_finalize_on_complete:
            self._trace(
                "[INIT] quirk auto_finalize_on_complete ENABLED — "
                "framework synthesizes signal_completion() server-side "
                "the instant the composite is_complete flips True "
                "(no model round-trip; avoids context-overflow-at-finalize)",
            )

        self._auth_info = (
            f"local ({self._host}, bearer)"
            if self._api_token
            else f"local ({self._host})"
        )
        self._trace(f"[INIT] host={self._host}")

        self._client = self._create_client()
        self._verify_connectivity()
        self._trace("[INIT] Initialization complete")

    def _create_client(self) -> "OpenAI":
        """Build the OpenAI client pointing at vLLM's /v1 endpoint.

        vLLM requires a non-empty ``api_key`` value at the OpenAI SDK
        layer.  Without ``--api-key`` at server launch it accepts any
        value (``"EMPTY"`` is the conventional placeholder per vLLM
        docs); when ``--api-key <token>`` is set, the SDK forwards the
        bearer on every request.
        """
        client_class = get_openai_client_class()
        return client_class(
            base_url=f"{self._host}/v1",
            api_key=self._api_token or "EMPTY",
        )

    def _auth_headers(self) -> Dict[str, str]:
        """Return auth headers for direct ``httpx`` calls."""
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _verify_connectivity(self) -> None:
        """Confirm vLLM is reachable via ``GET /health``.

        ``/health`` is vLLM's cheapest liveness probe — it returns 200
        when the engine is alive and 503 when the engine has died
        (verified against vLLM stable docs 2026-06-07).
        """
        try:
            response = httpx.get(
                f"{self._host}/health",
                headers=self._auth_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise VLLMConnectionError(self._host)
        except httpx.TimeoutException:
            raise VLLMConnectionError(self._host, "Connection timed out")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise VLLMAuthenticationError(original_error=str(exc))
            raise VLLMConnectionError(self._host, str(exc))
        except httpx.HTTPError as exc:
            raise VLLMConnectionError(self._host, str(exc))

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Credentials-only check — runs on an uninitialized instance.

        Per the provider-plugin contract (``shared/plugins/CLAUDE.md``),
        ``verify_auth`` runs on a *fresh, uninitialized* instance and
        must only check whether credentials are **available** — not
        whether they are valid, and not whether the remote service is
        reachable.  Reachability and engine validity are the job of
        ``initialize()`` and ``connect()``.

        For vLLM that means: when the server was launched without
        ``--api-key`` (and without a fronting auth proxy), there is
        nothing to authenticate, so we always return ``True``.  When
        ``--api-key`` IS set, the bearer must be read from
        ``plugin_configs['vllm']['api_token']`` (via ``config.extra``
        when supplied by the runtime) or from ``VLLM_API_TOKEN``.  Its
        presence is reported for operator visibility but is never a
        hard requirement at this stage — invalid bearers surface at
        ``initialize()`` time as ``VLLMAuthenticationError`` from the
        ``/health`` probe.
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
                on_message(f"vLLM bearer token configured ({masked})")
            else:
                on_message(
                    "vLLM: no authentication required "
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
        return self._auth_info or "vLLM (local)"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model for subsequent ``complete()`` calls.

        Verifies the requested model is one of the models vLLM is
        currently serving.  ``GET /v1/models`` typically returns one
        entry — the model name passed to ``vllm serve <model>`` — plus
        any LoRA adapters configured at server launch.

        Args:
            model: Model identifier as vLLM reports it in ``/v1/models``
                (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
            skip_model_test: Skip the GET-models validation call.

        Raises:
            VLLMModelNotFoundError: Server is not serving this model.
        """
        if not skip_model_test:
            catalog = self._fetch_catalog()
            if catalog and model not in {entry["id"] for entry in catalog}:
                raise VLLMModelNotFoundError(
                    model, available=[entry["id"] for entry in catalog],
                )

        self._model_name = model

        logger.info(
            "Connected to vLLM model: %s (context=%d)",
            model, self.get_context_limit(),
        )

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Query ``GET /v1/models`` and return the raw ``data`` array.

        Standard OpenAI shape per the vLLM stable docs (verified
        2026-06-07): ``{"object": "list", "data": [{"id": ...,
        "object": "model", "created": ..., "owned_by": "vllm",
        "root": ..., "parent": null|"<base-model>", "permission": [...]}]}``.

        Returns an empty list when the server is unreachable so callers
        can degrade gracefully (the connect-time test treats an empty
        catalog as "skip validation" rather than "model not found",
        which avoids spurious failures on transient blips).
        """
        try:
            response = httpx.get(
                f"{self._host}/v1/models",
                headers=self._auth_headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("data", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list vLLM models: %s", exc)
            return []

    def _detect_context_capacity(self) -> Optional[int]:
        """Tier-1 context-window auto-detection hook for vLLM.

        Reads ``max_model_len`` from the live server's ``GET /v1/models``.
        Current vLLM versions surface it in each model object (verified
        live against a running server); older versions that omit it make
        this return ``None`` so resolution falls back to the manual
        ``context_length`` knob / env var (see ``resolve_context_window``).

        vLLM hosts exactly one model per process, so the catalog normally
        has a single entry; we still prefer an ``id``-match against the
        selected model when one is known.  Failure-tolerant by
        construction: ``_fetch_catalog`` returns ``[]`` on an unreachable
        server, and a missing/zero ``max_model_len`` yields ``None``.
        """
        catalog = self._fetch_catalog()
        if not catalog:
            return None
        entry = None
        if self._model_name:
            entry = next(
                (m for m in catalog if m.get("id") == self._model_name), None
            )
        if entry is None:
            entry = catalog[0]
        max_len = entry.get("max_model_len")
        return int(max_len) if max_len else None

    @property
    def is_connected(self) -> bool:
        """True when both client and model are set."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Currently selected model name, or ``None`` before ``connect()``."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models served by this vLLM instance.

        Typically returns one entry (the model passed to
        ``vllm serve <model>``) plus any LoRA adapters; LoRA-adapted
        entries report the base model in ``parent`` and the adapter
        artifact in ``root``.
        """
        catalog = self._fetch_catalog()
        names = [entry["id"] for entry in catalog]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    # ==================== Stateless Completion ====================

    def _coerce_args_to_schema(
        self,
        args: Dict[str, Any],
        tool_schema: Optional[ToolSchema],
    ) -> Dict[str, Any]:
        """Coerce string-valued args to their tool-schema-declared type.

        Applies only when ``self._coerce_typed_tool_args`` is True
        (set via ``profile.quirks.coerce_typed_tool_args``).  For each
        arg whose tool-schema property type is array / object / integer
        / number / boolean, if a string arrived, attempts
        ``ast.literal_eval`` first (handles Python repr with single
        quotes — the actual wire shape from Llama 3.1 8B AWQ on vLLM
        0.22 under ``tool_choice: "auto"``) then ``json.loads`` as a
        fallback.  Coercion failures leave the string in place; the
        downstream schema validator will surface the type error as
        usual.

        ``ast.literal_eval`` is SAFE — it walks the AST for literal
        containers (str / num / list / dict / tuple / set / bool /
        None) only and never executes arbitrary code.

        No-op when ``tool_schema`` is None (unknown tool, e.g. one
        the framework hasn't registered) or when the schema has no
        ``properties`` map.

        Returns the (possibly modified) args dict.  Safe to call on
        a dict that wasn't mutated — returns the input as-is.
        """
        if not self._coerce_typed_tool_args:
            return args
        if not isinstance(args, dict) or not args:
            return args
        if tool_schema is None:
            return args
        properties = (
            (tool_schema.parameters or {}).get("properties") or {}
            if isinstance(tool_schema.parameters, dict) else {}
        )
        if not properties:
            return args

        non_string_types = {"array", "object", "integer", "number", "boolean"}
        coerced = dict(args)
        any_changed = False
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            prop_schema = properties.get(key)
            if not isinstance(prop_schema, dict):
                continue
            expected_type = prop_schema.get("type")
            # JSON Schema allows ``type`` to be a list (union of types).
            # Coerce when at least one expected type is non-string AND
            # ``string`` is NOT among them (otherwise a string is a
            # legitimate value and we'd be over-eager).
            if isinstance(expected_type, list):
                type_set = set(expected_type)
                if "string" in type_set:
                    continue
                if not (type_set & non_string_types):
                    continue
            elif expected_type not in non_string_types:
                continue
            new_value: Any = value
            for parser in (ast.literal_eval, json.loads):
                try:
                    new_value = parser(value)
                    break
                except (ValueError, SyntaxError):
                    continue
            if new_value is not value:
                coerced[key] = new_value
                any_changed = True
                self._trace(
                    f"QUIRK_COERCE tool={tool_schema.name} arg={key} "
                    f"from=str to={type(new_value).__name__}"
                )
        return coerced if any_changed else args

    def _coerce_response_function_calls(
        self,
        provider_response: ProviderResponse,
        tools: Optional[List[ToolSchema]],
    ) -> None:
        """Walk a ProviderResponse's parts and coerce stringified args
        on every function_call, in place.  No-op when the quirk is
        off or no function calls are present.  Used by the
        non-streaming path; the streaming path coerces inline in
        ``flush_tool_calls``."""
        if not self._coerce_typed_tool_args or not provider_response:
            return
        schemas_by_name = {t.name: t for t in (tools or [])}
        for part in (provider_response.parts or []):
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            tool_schema = schemas_by_name.get(fc.name)
            new_args = self._coerce_args_to_schema(fc.args or {}, tool_schema)
            if new_args is not fc.args:
                fc.args = new_args

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
        """Run one stateless chat completion through vLLM's /v1.

        Streaming and non-streaming paths mirror ``tensorrt_llm`` /
        ``lmstudio``'s implementation — vLLM speaks OpenAI's wire
        format faithfully, including ``tools`` / ``tool_choice`` for
        models whose tokenizer chat template + the server's
        ``--tool-call-parser`` (set at launch) support function
        calling.

        ``tool_choice`` is honored ONLY when the
        ``force_tool_choice_for_lifecycle`` quirk is enabled (default:
        off).  Without the quirk the kwarg is ignored and vLLM uses
        its auto-mode default; the session passes it generically and
        provider gates it via the quirk per
        ``project_backlog_vllm_provider_typed_tool_args``.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )

        # PR-256 PROBE INSTRUMENTATION (TEMPORARY, 2026-06-08).
        #
        # Companion to ``_maybe_stamp_lifecycle_retry_tool_choice`` /
        # ``_consume_pending_tool_choice`` probes in
        # ``jaato_session.py``.  PR-255 confirmed:
        #   - stamp fires on every val_failed (discovery AND context)
        #   - sentinel matches verbatim
        # But peer's empirical evidence shows xgrammar engages on
        # discovery retries (model converges in 3.2s) yet NOT on
        # context retries (5 loops without convergence, model emits
        # ``{}`` which xgrammar standalone proves is impossible under
        # the compiled grammar).  Between "stamp set" and "vLLM
        # receives named tool_choice on wire" something breaks for
        # context that doesn't break for discovery.
        #
        # This probe records what ``complete()`` actually receives:
        # tool_choice value (None / dict shape), quirk gate state,
        # tools count.  Triangulates three sub-hypotheses:
        #   B.1 — consume returned None (no tool_choice arrived)
        #   B.2 — tool_choice arrived but quirk False (profile gap)
        #   B.3 — both true but tools list empty (vllm drops kwarg)
        # Routed via ``logger.info`` (NOT ``self._trace``) so the
        # entry lands in the per-session log file alongside the
        # session-side MAYBE_STAMP_* probes, not the per-agent
        # provider_trace_subagent_<id>.log files which were the
        # gotcha behind peer's earlier "zero traces" finding.
        logger.info(
            "VLLM_COMPLETE_ENTRY tool_choice=%r force_quirk=%s tools_count=%d",
            tool_choice,
            self._force_tool_choice_for_lifecycle,
            len(tools) if tools else 0,
        )
        # PROBE (cancel-leak prod-vs-isolation diagnostic):
        # Log cancel_token identity at complete() entry — companion to
        # the _CT_ID trace in _stream_response.  Two distinct trace
        # sinks (logger vs self._trace) so we can correlate with the
        # MAYBE_STAMP probe entries (logger.info → per-session log)
        # AND the per-agent provider_trace_* files.  Same id() value
        # at both sites confirms the cancel_token plumbing is
        # consistent; divergent ids would surface H2 (instance
        # mismatch) immediately.
        logger.info(
            "VLLM_COMPLETE_ENTRY_CT id=%s cancelled=%s",
            id(cancel_token) if cancel_token else None,
            cancel_token.is_cancelled if cancel_token else None,
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
                # Diagnostic visibility (2026-06-09): emit per-tool wire
                # byte sizes so operators can correlate "tools_count=N"
                # with the actual bytes vLLM receives.  Closes one
                # residual uncertainty from the qwen3-14b empty-args
                # debug arc — Family (A) "framework strips schema before
                # sending to vLLM" can be falsified at-a-glance by
                # grepping VLLM_TOOLS_WIRE_DUMP, instead of code-reading
                # the converter every time the question recurs.  Routes
                # via ``logger.info`` so the trace lands in /tmp/jaato.log
                # regardless of apparmor confinement on self._trace.
                tools_json = json.dumps(openai_tools, separators=(",", ":"))
                per_tool_bytes = [
                    (
                        t.get("function", {}).get("name", "?"),
                        len(json.dumps(t, separators=(",", ":"))),
                    )
                    for t in openai_tools
                ]
                logger.info(
                    "VLLM_TOOLS_WIRE_DUMP total_bytes=%d count=%d per_tool=%s",
                    len(tools_json),
                    len(openai_tools),
                    per_tool_bytes,
                )
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        # Path 1 quirk: forward session-supplied ``tool_choice`` to
        # vLLM only when the quirk is enabled.  vLLM 0.22 engages
        # xgrammar decoding for named-function ``tool_choice``,
        # constraining generation to the tool's parameter JSON
        # schema → server-side correctly-typed args.  When the
        # quirk is OFF, the kwarg is dropped and vLLM uses its
        # auto-mode default.
        #
        # Tool-name translation (PR-251 fix, 2026-06-07): the wire
        # ``tools`` array carries HASHED ids (``t_<8-hex>`` per
        # ``shared/tool_id_map.py:name_to_id`` — same hashing
        # ``tool_schemas_to_openai`` applies at line 74), so
        # ``tool_choice.function.name`` MUST also be the hashed id
        # or vLLM 400s with ``The tool specified in tool_choice
        # does not match any of the specified tools``.  The
        # session passes the canonical name (it doesn't know about
        # hashing); the provider applies ``name_to_id`` here at
        # the wire boundary, symmetric to how
        # ``get_original_tool_name`` reverses the mapping on the
        # response path.  Empirically caught by peer's 2026-06-07
        # cascade after PR-250 landed — stamped name
        # "signal_completion" didn't resolve to any wire tool
        # because all entries were ``t_5ab8fa33`` etc.
        if (
            self._force_tool_choice_for_lifecycle
            and tool_choice is not None
            and tools
        ):
            forwarded_tool_choice = tool_choice
            canonical_name: Optional[str] = None
            if (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
                and isinstance(tool_choice["function"].get("name"), str)
            ):
                canonical_name = tool_choice["function"]["name"]
                hashed_name = name_to_id(canonical_name)
                forwarded_tool_choice = {
                    "type": "function",
                    "function": {"name": hashed_name},
                }
            kwargs["tool_choice"] = forwarded_tool_choice
            self._trace(
                f"QUIRK_FORCE_TOOL_CHOICE canonical={canonical_name} "
                f"wire={forwarded_tool_choice} — vLLM will engage "
                f"xgrammar for this call"
            )

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
                    tools=tools,
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
                self._coerce_response_function_calls(provider_response, tools)

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
        tools: Optional[List[ToolSchema]] = None,
    ) -> ProviderResponse:
        """Accumulate text, tool calls, and usage from a streaming response.

        vLLM emits the same OpenAI delta shape as LM Studio / NIM /
        trtllm-serve, including ``stream_options={include_usage: true}``
        support so usage arrives in the trailing chunk.

        ``tools`` is forwarded only to feed
        :meth:`_coerce_args_to_schema` inside ``flush_tool_calls`` when
        the ``coerce_typed_tool_args`` quirk is active.  Not used to
        build the request — that already happened in ``complete()``.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        # PROBE (cancel-leak prod-vs-isolation diagnostic):
        # Trace cancel_token identity at entry — settles H2 (right token
        # instance?  did is_cancelled ever return True for the token the
        # provider holds?).
        #
        # Routed via ``logger.info`` (NOT ``self._trace``).  Runner-tier
        # ``self._trace`` calls land in ``/tmp/provider_trace.log`` which
        # AppArmor SILENTLY DENIES under the per-WS confined-runner
        # profile (template grants ``rw`` only on
        # ``/tmp/jaato-<ws>-**``, not the bare ``/tmp/`` filename).
        # ``trace_write`` catches the PermissionError and ``pass``es,
        # so the trace lines vanish.  Empirically confirmed
        # 2026-06-09 by peer'\''s first probe run: only the
        # ``logger.info`` line at ``complete()`` entry landed in
        # ``/tmp/jaato.log``; all 4 ``self._trace`` lines absent
        # from every searchable log sink.  Backlog item filed for
        # the apparmor rule gap; for diagnostic purposes the trace
        # bumps to logger.info so traces are visible.
        logger.info(
            "VLLM_STREAM_CT_ID id=%s cancelled=%s",
            id(cancel_token) if cancel_token else None,
            cancel_token.is_cancelled if cancel_token else None,
        )

        schemas_by_name: Dict[str, ToolSchema] = {
            t.name: t for t in (tools or [])
        }

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
                    # Quirk: coerce stringified args BEFORE building the
                    # FunctionCall so downstream (schema validator,
                    # ledger, history) sees the typed shape.
                    args = self._coerce_args_to_schema(
                        args, schemas_by_name.get(original_name),
                    )
                    fc = FunctionCall(id=tool_id, name=original_name, args=args)
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
                if cancel_token and cancel_token.is_cancelled:
                    # PROBE (cancel-leak prod-vs-isolation diagnostic):
                    # Settles H3 (how many chunks elapsed before the
                    # provider's for-loop detected cancellation).
                    # Routed via logger.info per the apparmor-blocks-
                    # self._trace finding above.
                    logger.info(
                        "VLLM_STREAM_CT_CANCEL_DETECTED iter=%d ct_id=%s",
                        chunk_count,
                        id(cancel_token),
                    )
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
        finally:
            # Close the underlying HTTP connection so vLLM stops
            # generating immediately on cancel. Without this close,
            # vLLM continues filling output_tokens up to max_tokens
            # because the SDK ``Stream`` object only sends TCP-close
            # at garbage-collection time — which on a cancelled turn
            # can mean 60-120s of wasted GPU work per cancelled call.
            #
            # PROBE (cancel-leak prod-vs-isolation diagnostic):
            # The two _CLOSING_STREAM_NOW / _CLOSED_STREAM_OK traces
            # settle the central question — does close() actually FIRE
            # on prod cancel?  Peer'\''s 30-line isolation probe showed
            # close() stops vLLM in <50ms; prod 2026-06-09 shows 7 min
            # GPU-hold.  If both traces appear on a cancelled prod
            # turn AND vLLM still keeps generating, there'\''s a state
            # difference between probe and prod that close()
            # alone can'\''t fix.  If only _CLOSING appears, close
            # hangs.  If NEITHER appears, the finally never fires
            # (most likely if running inside a wrapper killed first).
            if response_stream is not None:
                # PROBE (cancel-leak prod-vs-isolation diagnostic):
                # Bumped to logger.info per apparmor-blocks-self._trace
                # finding above.  These two are the central traces for
                # the cancel-leak diagnostic — they answer whether
                # close() actually fires on prod cancel.
                logger.info(
                    "VLLM_STREAM_CLOSING_STREAM_NOW response_id=%s was_cancelled=%s",
                    id(response_stream),
                    was_cancelled,
                )
                try:
                    response_stream.close()
                    logger.info("VLLM_STREAM_CLOSED_STREAM_OK")
                except Exception as close_exc:  # pragma: no cover - best effort
                    logger.info(
                        "VLLM_STREAM_CLOSE_ERROR %s: %s",
                        type(close_exc).__name__,
                        close_exc,
                    )

            # SHAPE B (cancel-leak fix, 2026-06-09): close the
            # OpenAI client's httpx pool when the turn was cancelled.
            # Empirical evidence from peer's PR-260 cascade run:
            # ``response_stream.close()`` returns OK but does NOT
            # send TCP-FIN to vLLM — openai SDK ``Stream.close()``
            # releases the wrapper without aggressively closing
            # the underlying TCP socket, which httpx keeps in
            # its pool for keep-alive reuse.  vLLM kept generating
            # for 4+ minutes with GPU at 91 C climbing toward
            # thermal-throttle until ``kill -9`` on the runner.
            # ``self._client.close()`` dismantles the entire httpx
            # pool, forcing FIN on EVERY socket including the
            # in-flight one — peer's isolation probe variant B
            # confirmed this stops vLLM in <50ms.
            #
            # Safety: providers are per-session
            # (jaato_session.py:2096+3344 via
            # ``runtime.create_provider``), so closing this
            # client only affects the cancelled session.  Caveat:
            # if cancel reason is ``mid_turn_interrupt`` (TUI
            # mid-turn prompt injection), the session continues
            # but the next ``provider.complete()`` would fail
            # because ``self._client`` is closed.  That regression
            # is filed as a follow-up
            # ([[project_backlog_cancel_leak_mid_turn_interrupt_recreate_client]]).
            # For cascade workloads the cancel always ends the
            # session, so this fix is safe in the dominant case.
            if was_cancelled and self._client is not None:
                try:
                    self._client.close()
                    logger.info("VLLM_CLIENT_CLOSED_ON_CANCEL")
                except Exception as client_close_exc:  # pragma: no cover - best effort
                    logger.info(
                        "VLLM_CLIENT_CLOSE_ERROR %s: %s",
                        type(client_close_exc).__name__,
                        client_close_exc,
                    )

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
        """Map OpenAI SDK exceptions to vLLM-specific error types.

        Distinguishes connection errors from auth/model errors so the
        reliability layer can classify retryability correctly.  The
        mid-stream-vs-pre-flight distinction mirrors trtllm_llm /
        triton; per the rule in
        ``feedback_no_hardcoded_likely_cause_in_error_messages.md``,
        the mid-stream message points at the server log instead of
        guessing at the cause.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise VLLMAuthenticationError(original_error=str(error)) from error

        if isinstance(error, openai.NotFoundError):
            raise VLLMModelNotFoundError(
                model=self._model_name or "unknown"
            ) from error

        if isinstance(error, openai.APIConnectionError):
            # Distinguish mid-stream connection drops (engine error after
            # HTTP 200 was committed) from pre-flight connection failures
            # (host unreachable / firewall / DNS).  The first surfaces as
            # httpx.RemoteProtocolError wrapped by openai.APIConnectionError
            # AND the message contains "peer closed connection" or
            # "incomplete chunked"; the second is a clean
            # httpx.ConnectError / TimeoutException with no such phrasing.
            #
            # Misrouting the user toward firewall investigation when the
            # real issue is engine config burns forensics time — same
            # incident shape as the 2026-06-06 trtllm cascade hit that
            # motivated TensorRTLLMMidStreamError.
            error_msg = str(error).lower()
            is_mid_stream = (
                "peer closed connection" in error_msg
                or "incomplete chunked" in error_msg
                or "remoteprotocolerror" in error_msg
            )
            if is_mid_stream:
                raise VLLMMidStreamError(
                    self._host, original_error=str(error),
                ) from error
            raise VLLMConnectionError(self._host, str(error)) from error

        # Other APIStatusError subclasses fall through — the caller sees
        # the original exception, which already carries status/body.

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Heuristic token count (~4 chars/token).

        vLLM does not expose a tokenization endpoint via its
        OpenAI-compatible ``/v1`` surface.  (vLLM does have a
        ``/tokenize`` endpoint on its native surface, but that is not
        part of the OpenAI compatibility contract this provider
        targets.)
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Return the context window size for the currently connected model.

        Returns the value resolved at ``initialize()`` time via
        ``resolve_context_window`` (see that helper + the precedence
        comment in ``initialize()``):

        1. PRIMARY — auto-detected from the live server's ``GET
           /v1/models`` ``max_model_len`` (current vLLM versions
           surface it).
        2. fallback — ``plugin_configs.vllm.context_length``.
        3. fallback — ``VLLM_CONTEXT_LENGTH`` env var.

        The manual overrides exist only for older vLLM builds that
        don't surface ``max_model_len``, or to pin a value.
        ``initialize()`` raises only when auto-detect fails AND no
        override is set.

        Raises:
            RuntimeError: When called before ``initialize()`` has run.
        """
        if self._context_length_override is None:
            raise RuntimeError(
                "vLLM provider: get_context_limit() called before "
                "initialize() set the override.  This is a programmer "
                "error — the provider must be connected first."
            )
        return self._context_length_override

    def get_token_usage(self) -> TokenUsage:
        """Token usage from the most recent completion."""
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """vLLM accepts ``response_format={'type': 'json_object'}``.

        It also supports ``response_format={'type': 'json_schema',
        'json_schema': {...}}`` and guided decoding (``guided_json``,
        ``guided_choice``, ``guided_regex``, ``guided_grammar``) via
        ``extra_body``; this provider currently only wires the
        ``json_object`` variant through ``complete(response_schema=...)``
        — operators wanting strict guided decoding can compose
        ``extra_body`` themselves at the harness layer.
        """
        return True

    def supports_streaming(self) -> bool:
        """Streaming works through the OpenAI-compatible endpoint."""
        return True

    def supports_stop(self) -> bool:
        """Streaming responses can be interrupted via ``cancel_token``."""
        return True

    def supports_thinking(self) -> bool:
        """Most vLLM-served models don't surface a reasoning channel.

        vLLM supports ``--reasoning-parser <name>`` (e.g.
        ``deepseek_r1``) for reasoning models, which exposes a
        ``message.reasoning`` field in the response.  This provider
        does not currently extract that channel — operators running
        reasoning models can access it through the raw response
        directly.  Reported here as ``False`` so the framework does
        not silently drop reasoning content into the main text path
        for non-reasoning-aware sessions.
        """
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is not exposed through vLLM's /v1 by default."""
        pass

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify exceptions for the reliability layer's retry policy."""
        if isinstance(exc, VLLMConnectionError):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """vLLM does not emit retry-after hints."""
        return None

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — vLLM does not use interactive auth."""
        if on_message:
            on_message(
                "vLLM has no interactive auth flow.  If the server was "
                "launched with --api-key <token> or sits behind an auth "
                "proxy, set VLLM_API_TOKEN."
            )


def create_provider() -> VLLMProvider:
    """Factory function consumed by the provider discovery machinery."""
    return VLLMProvider()
