"""Amazon Bedrock provider — SigV4 auth, the Converse API.

One endpoint, many vendors: Bedrock serves Anthropic, Amazon Nova, Meta,
Mistral, Cohere, AI21 and DeepSeek behind a single message-shaped API
(``Converse`` / ``ConverseStream``), in the customer's own AWS account.
That is the whole reason this is a provider and not another OpenAI-compatible
gateway entry: the wire is Amazon's, the auth is SigV4, and neither is
reachable from ``_openai_compat``.

**jaato holds no credential here.**  Every other provider in this tree
resolves an API key; SigV4 signing is botocore's, and so is the resolution
chain behind it (env vars, a named profile, IAM Identity Center, container
and instance roles).  Re-implementing that chain would be a subtly wrong copy
of a moving target, so :meth:`initialize` builds a ``boto3.Session`` and asks
it what it found — which also means an EC2/EKS deployment needs no jaato
configuration at all beyond the region.

Two things Bedrock's catalog does NOT report, and what follows from each:

* **context window** — ``ListFoundationModels`` describes modalities and
  streaming support, never capacity.  So ``context_length`` is required in
  practice and :meth:`get_context_limit` fails loud without it, exactly as
  the native OpenAI and Azure providers do.  A guessed window truncates a
  session without saying so.
* **document / audio input** — the catalog's modality vocabulary is
  ``TEXT | IMAGE | EMBEDDING``, which cannot express the ``document`` and
  ``audio`` blocks Converse plainly carries.  An incomplete source wearing
  the authoritative "detect" tier would withhold every PDF from a model that
  reads PDFs, so input modalities resolve from a documented per-family table
  beneath the ``modalities`` knob instead — the same shape the Anthropic
  provider uses, and the tier :func:`resolve_modalities` documents for
  closed providers with no live modality endpoint.

**Extended thinking is extraction-only by default, and deliberately.**
Converse's ``reasoningContent`` block carries the upstream's ``signature``
next to the text, and Anthropic-family models REQUIRE that block back,
signature included, on the next request of a tool-call loop.  ``Part.thought``
has nowhere to put a signature, so replaying the text alone would be rejected
by the very models that produced it.  The provider therefore reads reasoning
out (it reaches the UI and ``ProviderResponse.thinking``) and does not write
it back; ``api_params.enable_thinking`` is available and WARNS at connect
that a thinking turn which also calls tools is not yet supported.  Wiring it
properly needs a signature-carrying part, which is the reasoning-replay seam's
next step rather than something to fake here.
"""

import json
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)

from shared.tool_id_map import wire_name_trace_fields
from ..base import (
    MODALITY_TEXT,
    FunctionCallDetectedCallback,
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
    ThinkingConfig,
    TokenUsage,
    ToolSchema,
    TurnResult,
    parse_tool_call_arguments,
    require_terminated_stream,
    resolve_tool_use_finish,
)
from .converters import (
    deserialize_history,
    finish_reason_from_bedrock,
    messages_to_bedrock,
    response_from_bedrock,
    serialize_history,
    system_to_bedrock,
    tool_config_to_bedrock,
    usage_from_bedrock,
)
from .env import (
    get_checked_credential_locations,
    resolve_context_length,
    resolve_endpoint_url,
    resolve_model,
    resolve_profile,
    resolve_region,
)
from .errors import (
    AccessDeniedError,
    BotocoreNotInstalledError,
    ContextLengthNotConfiguredError,
    ContextLimitError,
    CredentialsNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    RegionNotConfiguredError,
    ServiceUnavailableError,
    ThrottlingError,
)


#: Cross-region inference profiles prefix the model id with a geography
#: (``us.anthropic.claude-...``).  The prefix routes the request; it says
#: nothing about the model, so it is stripped before any table lookup.
_GEO_PREFIXES = ("us-gov.", "us.", "eu.", "apac.", "jp.", "au.", "ca.")

#: INPUT modalities per Bedrock model FAMILY, prefix-matched on the id with
#: any geography prefix stripped.  Documented constants, not detection: see
#: the module docstring for why the catalog cannot answer this question.
#: A family absent here resolves to the text-only floor — never a false claim.
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    # Claude reads images and documents (Converse carries PDFs as a
    # ``document`` block; Claude reads the pages, figures included).
    "anthropic.claude-3": frozenset({"text", "image", "file"}),
    "anthropic.claude-haiku-4": frozenset({"text", "image", "file"}),
    "anthropic.claude-opus-4": frozenset({"text", "image", "file"}),
    "anthropic.claude-sonnet-4": frozenset({"text", "image", "file"}),
    # Nova's multimodal tier takes images, documents and video; Nova Micro
    # is text-only and is deliberately absent rather than prefix-matched.
    "amazon.nova-lite": frozenset({"text", "image", "file", "video"}),
    "amazon.nova-pro": frozenset({"text", "image", "file", "video"}),
    "amazon.nova-premier": frozenset({"text", "image", "file", "video"}),
    # Llama's vision variants; the text-only sizes are absent.
    "meta.llama3-2-11b": frozenset({"text", "image"}),
    "meta.llama3-2-90b": frozenset({"text", "image"}),
    "meta.llama4": frozenset({"text", "image"}),
    "mistral.pixtral": frozenset({"text", "image"}),
}

#: botocore error codes that are worth retrying, and the domain error each
#: becomes.  Everything NOT listed here is terminal by construction — a
#: default of "retry" would spend the whole budget re-sending a 403.
_TRANSIENT_CODES = {
    "ThrottlingException": ThrottlingError,
    "TooManyRequestsException": ThrottlingError,
    "ServiceQuotaExceededException": ThrottlingError,
    "ServiceUnavailableException": ServiceUnavailableError,
    "InternalServerException": ServiceUnavailableError,
    "ModelTimeoutException": ServiceUnavailableError,
    "ModelErrorException": ServiceUnavailableError,
    "ModelNotReadyException": ModelNotReadyError,
}

#: The framework's default output cap.  Converse has no server-side default
#: worth relying on across seven vendors, so one is always sent.
DEFAULT_MAX_TOKENS = 8192

#: ``api_params`` keys forwarded into ``inferenceConfig``, under the names
#: Converse gives them.  An allow-list rather than a passthrough, because
#: Converse rejects an unknown member outright — vendor-specific parameters
#: go through ``additional_model_request_fields``, which is what it is for.
_INFERENCE_CONFIG_KEYS = {
    "max_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "stop_sequences": "stopSequences",
}


class BedrockProvider(ModalityCapabilityMixin):
    """Stateless Amazon Bedrock provider over the Converse API.

    Stateless in the framework's sense: the session owns the history and
    passes it whole to :meth:`complete` on every call.  The provider holds
    only the boto3 client, the resolved model id, and the profile knobs.
    """

    def __init__(self) -> None:
        """Construct an unconfigured provider (no client, no model)."""
        self._client: Optional[Any] = None      # bedrock-runtime client
        self._session: Optional[Any] = None     # boto3.Session
        self._model_name: Optional[str] = None
        self._region: Optional[str] = None
        self._endpoint_url: Optional[str] = None
        self._profile: Optional[str] = None
        self._auth_info: str = ""

        # Sizing.  ``None`` means "not configured", which is an error at the
        # point of use rather than a guess — Bedrock reports no capacity.
        self._context_length: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        self._output_modalities_knob: Optional[List[str]] = None

        # inferenceConfig knobs.  ``None`` means "omit and let the model's
        # own default stand", except max_tokens, which always ships.
        self._inference_config: Dict[str, Any] = {}
        self._max_tokens: int = DEFAULT_MAX_TOKENS
        self._tool_choice_default: Optional[Any] = None

        # Wire extensions.
        self._additional_request_fields: Dict[str, Any] = {}
        self._guardrail_config: Optional[Dict[str, Any]] = None
        self._performance_latency: Optional[str] = None
        self._enable_caching: bool = False
        self._cache_ttl: Optional[str] = None

        # Extended thinking (extraction always; request opt-in).
        self._enable_thinking: bool = False
        self._thinking_budget: int = 4096

        self._last_usage: TokenUsage = TokenUsage()

        # Agent context for tracing.
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "bedrock"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Build the ``bedrock-runtime`` client and read the profile knobs.

        Raises:
            BotocoreNotInstalledError: ``boto3`` is not installed.
            RegionNotConfiguredError: no region from any tier.
            CredentialsNotFoundError: boto3's chain found no credentials.
        """
        try:
            import boto3
        except ImportError as exc:
            raise BotocoreNotInstalledError() from exc

        config = config or ProviderConfig()
        extra = config.extra or {}
        api_params = extra.get("api_params") or {}

        self._profile = extra.get("profile") or resolve_profile()
        self._endpoint_url = extra.get("endpoint_url") or resolve_endpoint_url()

        # Region: knob -> jaato env -> ProviderConfig.location (the generic
        # "region" field) -> whatever boto3 itself resolves.  boto3 is asked
        # LAST rather than not at all, so an EC2 host with a region in its
        # instance profile needs no jaato configuration.
        self._session = boto3.Session(
            profile_name=self._profile,
            region_name=(extra.get("region") or resolve_region()
                         or config.location or None),
        )
        self._region = self._session.region_name
        if not self._region:
            raise RegionNotConfiguredError()

        credentials = self._session.get_credentials()
        if credentials is None:
            raise CredentialsNotFoundError(
                checked_locations=get_checked_credential_locations(),
                region=self._region,
            )
        self._auth_info = self._describe_credentials(credentials)

        self._read_knobs(extra, api_params)

        self._client = self._session.client(
            "bedrock-runtime",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        )
        self._trace(f"[INIT] region={self._region} profile={self._profile}")

    def _describe_credentials(self, credentials: Any) -> str:
        """Name the credential source boto3 resolved, for "Connected to".

        botocore stamps a ``method`` on the credentials it builds
        (``env``, ``shared-credentials-file``, ``sso``, ``iam-role``, ...),
        and that string is the answer to "why does this work on my laptop
        and not in the container" — the question this line exists for.
        """
        method = getattr(credentials, "method", None) or "unknown"
        if self._profile:
            return f"AWS credentials ({method}, profile {self._profile})"
        return f"AWS credentials ({method})"

    def _read_knobs(self, extra: Dict[str, Any],
                    api_params: Dict[str, Any]) -> None:
        """Populate the request-shaping state from the profile's knobs.

        Split out of :meth:`initialize` so each stays readable and under the
        complexity ceiling; it mutates ``self`` and returns nothing.
        """
        self._context_length = resolve_context_window(
            detect_capacity=None,                      # Bedrock reports none
            profile_value=extra.get("context_length"),
            env_value=resolve_context_length(),
        )

        modalities = extra.get("modalities")
        if modalities is not None:
            if not isinstance(modalities, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities
            ):
                raise TypeError(
                    "Bedrock 'modalities' config must be a list of strings "
                    f'(e.g. ["text", "image"]), got {type(modalities).__name__}'
                )
            self._modalities_knob = list(modalities)

        output_modalities = extra.get("output_modalities")
        if output_modalities is not None:
            self._output_modalities_knob = list(output_modalities)

        for knob, wire in _INFERENCE_CONFIG_KEYS.items():
            if api_params.get(knob) is not None:
                self._inference_config[wire] = api_params[knob]
        self._max_tokens = int(self._inference_config.pop(
            "maxTokens", DEFAULT_MAX_TOKENS))
        self._tool_choice_default = api_params.get("tool_choice")

        self._additional_request_fields = dict(
            extra.get("additional_model_request_fields") or {})
        self._guardrail_config = extra.get("guardrail_config")
        self._performance_latency = extra.get("performance_latency")
        self._enable_caching = bool(extra.get("enable_caching", False))
        self._cache_ttl = extra.get("cache_ttl")

        self._enable_thinking = bool(api_params.get("enable_thinking", False))
        if api_params.get("thinking_budget") is not None:
            self._thinking_budget = int(api_params["thinking_budget"])

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Any] = None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Check that boto3 can resolve credentials, without any network call.

        There is no interactive login to offer: AWS's own is ``aws sso login``
        / ``aws configure``, run outside this process, so ``allow_interactive``
        changes nothing here and the method stays cheap as the contract
        requires.

        Raises:
            CredentialsNotFoundError: boto3's chain resolved nothing.
        """
        try:
            import boto3
        except ImportError as exc:
            raise BotocoreNotInstalledError() from exc

        extra = (config.extra if config else {}) or {}
        session = boto3.Session(
            profile_name=extra.get("profile") or resolve_profile(),
            region_name=extra.get("region") or resolve_region(),
        )
        if session.get_credentials() is None:
            raise CredentialsNotFoundError(
                checked_locations=get_checked_credential_locations(),
                region=session.region_name,
            )
        return True

    def shutdown(self) -> None:
        """Drop the client. botocore clients hold pooled connections."""
        self._client = None
        self._session = None

    def get_auth_info(self) -> str:
        """Which credential source boto3 resolved."""
        return self._auth_info

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model (or inference profile) for this provider.

        No network call is made even when ``skip_model_test`` is False: the
        cheapest Bedrock probe is a billable generation, and the useful
        preflight — is there a context window for this model — is local.
        """
        self._model_name = model or resolve_model()
        if not self._model_name:
            raise ValueError(
                "No Bedrock model configured. Set the profile's `model:` to a "
                "Bedrock model id or inference-profile id (e.g. "
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0), or set "
                "JAATO_BEDROCK_MODEL."
            )
        if not self._context_length:
            raise ContextLengthNotConfiguredError(model=self._model_name)
        if self._enable_thinking:
            logger.warning(
                "bedrock: api_params.enable_thinking is on for %s. Reasoning "
                "is EXTRACTED and shown, but not replayed: Converse requires "
                "a reasoningContent block's upstream signature back on the "
                "next request of a tool-call loop, and Part.thought carries "
                "no signature. A thinking turn that also calls tools may be "
                "rejected by the model. Leave it off for tool-using sessions.",
                self._model_name,
            )
        self._trace(f"[CONNECT] model={self._model_name} region={self._region}")

    @property
    def is_connected(self) -> bool:
        """Whether a client and a model are both in place."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """The Bedrock model id / inference-profile id in use."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List model ids offered in this region.

        Reads BOTH catalogs, because they answer different questions and a
        caller wants one list: ``ListFoundationModels`` names the base models
        offered here, and ``ListInferenceProfiles`` names the cross-region
        profiles — which is how several current models must be addressed at
        all, so omitting them would hide the ids that actually work.

        Failure is empty, not fatal: listing is a convenience, and the
        ``bedrock:List*`` permissions are frequently withheld from an
        identity that can invoke perfectly well.
        """
        if self._session is None:
            return []
        ids: List[str] = []
        try:
            control = self._session.client(
                "bedrock", region_name=self._region)
        except Exception as exc:
            self._trace(f"[LIST_MODELS] client failed: {exc}")
            return []

        try:
            listed = control.list_foundation_models(byOutputModality="TEXT")
            ids.extend(
                summary["modelId"]
                for summary in listed.get("modelSummaries") or []
                if summary.get("modelId")
            )
        except Exception as exc:
            self._trace(f"[LIST_MODELS] foundation models failed: {exc}")

        try:
            profiles = control.list_inference_profiles()
            ids.extend(
                summary["inferenceProfileId"]
                for summary in profiles.get("inferenceProfileSummaries") or []
                if summary.get("inferenceProfileId")
            )
        except Exception as exc:
            self._trace(f"[LIST_MODELS] inference profiles failed: {exc}")

        unique = sorted(set(ids))
        if prefix:
            unique = [m for m in unique if m.startswith(prefix)]
        return unique

    # ==================== Token / capability reporting ====================

    def count_tokens(self, content: str) -> int:
        """Estimate token count for ``content``.

        An estimate rather than an API call, and that is a choice: Bedrock's
        ``CountTokens`` is a per-model, per-region network round trip, and
        this method is called on the GC hot path where a four-chars-per-token
        estimate is both fast and good enough to compare against a threshold.
        """
        return max(1, len(content) // 4)

    def get_context_limit(self) -> int:
        """The configured context window.

        Raises:
            ContextLengthNotConfiguredError: nothing configured one. Bedrock
                reports no capacity, so there is nothing to fall back to and
                a guess would silently truncate the session.
        """
        if self._context_length:
            return self._context_length
        raise ContextLengthNotConfiguredError(model=self._model_name)

    def get_token_usage(self) -> TokenUsage:
        """Usage from the most recent completion."""
        return self._last_usage

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities the named (or active) model accepts.

        Knob -> family table -> text floor. There is no detect tier: see the
        module docstring for why Bedrock's catalog cannot answer this.
        """
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=self._lookup_input_modalities(model),
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def _lookup_input_modalities(
        self, model: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Table-declared modalities for ``model``, geography prefix stripped."""
        name = model or self._model_name
        if not name:
            return None
        for geo in _GEO_PREFIXES:
            if name.startswith(geo):
                name = name[len(geo):]
                break
        for family, mods in MODEL_INPUT_MODALITIES.items():
            if name.startswith(family):
                return mods
        return None

    def supports_structured_output(self) -> bool:
        """Converse has no cross-vendor structured-output field.

        ``outputConfig.textFormat`` exists but is honoured by a subset of the
        hosted models, so claiming it here would promise something several
        of them silently ignore. Structured output is obtained the way every
        Anthropic-shaped wire obtains it: by asking, and parsing the answer.
        """
        return False

    def supports_streaming(self) -> bool:
        """``ConverseStream`` is available for every text model here."""
        return True

    def supports_stop(self) -> bool:
        """``inferenceConfig.stopSequences`` is part of the base Converse set."""
        return True

    def supports_thinking(self) -> bool:
        """Reasoning is extracted whenever the model emits it.

        True describes the EXTRACTION half only — see the module docstring
        on why the replay half is not wired.
        """
        return True

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Apply a session-level thinking config over the profile's."""
        self._enable_thinking = bool(getattr(config, "enabled", False))
        budget = getattr(config, "budget_tokens", None)
        if budget:
            self._thinking_budget = int(budget)

    def set_agent_context(self, agent_type: str = "main",
                          agent_name: Optional[str] = None,
                          agent_id: str = "main") -> None:
        """Set the agent identity trace lines are written under."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write a trace line to the per-agent provider log."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "bedrock:main"
        elif self._agent_name:
            prefix = f"bedrock:subagent:{self._agent_name}"
        else:
            prefix = f"bedrock:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    # ==================== Request assembly ====================

    def _build_request(
        self,
        messages: List[Message],
        system_instruction: Optional[str],
        tools: Optional[List[ToolSchema]],
        tool_choice: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the Converse request kwargs for one turn.

        Everything vendor-specific lands in ``additionalModelRequestFields``,
        which is Converse's own escape hatch: the profile's dict is applied
        LAST so an operator can override anything the framework put there,
        including the thinking config.
        """
        request: Dict[str, Any] = {
            "modelId": self._model_name,
            "messages": messages_to_bedrock(messages),
            "inferenceConfig": {"maxTokens": self._max_tokens,
                                **self._inference_config},
        }

        system = system_to_bedrock(
            system_instruction, cache=self._enable_caching,
            cache_ttl=self._cache_ttl)
        if system:
            request["system"] = system

        tool_config = tool_config_to_bedrock(
            tools, tool_choice if tool_choice is not None
            else self._tool_choice_default,
            cache=self._enable_caching, cache_ttl=self._cache_ttl)
        if tool_config:
            request["toolConfig"] = tool_config

        additional: Dict[str, Any] = {}
        if self._enable_thinking:
            # The Anthropic-family spelling. Other vendors name their
            # reasoning switch differently, which is precisely why the
            # profile's own dict wins below.
            additional["reasoning_config"] = {
                "type": "enabled", "budget_tokens": self._thinking_budget}
        additional.update(self._additional_request_fields)
        if additional:
            request["additionalModelRequestFields"] = additional

        if self._guardrail_config:
            request["guardrailConfig"] = self._guardrail_config
        if self._performance_latency:
            request["performanceConfig"] = {"latency": self._performance_latency}
        return request

    # ==================== Completion ====================

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
        """Stateless completion over ``Converse`` / ``ConverseStream``.

        Follows the provider contract in ``base.py``: a success returns
        ``TurnResult.from_provider_response``, a non-transient failure returns
        ``TurnResult.from_exception``, and a TRANSIENT failure is RAISED so
        ``with_retry`` can see it. ``_TRANSIENT_CODES`` is what decides which
        botocore error code is which.

        Args:
            messages: The whole conversation; the provider appends nothing.
            system_instruction: System prompt text.
            tools: Tool schemas for this turn.
            response_schema: Parsed out of the reply when the model returns
                JSON; Converse has no cross-vendor field to request it.
            cancel_token: Checked between stream events.
            on_chunk: Enables streaming when provided.
            on_usage_update: Called as usage arrives (streaming).
            on_function_call: Called when a tool call completes mid-stream.
            on_thinking: Called once with the reasoning text, before the
                first text chunk or the first tool call.
            tool_choice: Per-call override of ``api_params.tool_choice``.

        Returns:
            A :class:`TurnResult` classifying the outcome.
        """
        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first.")

        request = self._build_request(
            messages, system_instruction, tools, tool_choice)

        try:
            if on_chunk:
                provider_response = self._stream_response(
                    request, on_chunk=on_chunk, cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_function_call=on_function_call,
                    on_thinking=on_thinking,
                )
            else:
                provider_response = response_from_bedrock(
                    self._client.converse(**request))
        except Exception as exc:
            translated = self._translate_error(exc)
            if isinstance(translated, tuple(_TRANSIENT_CODES.values())):
                raise translated from exc
            if translated is not exc:
                return TurnResult.from_exception(translated)
            raise

        self._last_usage = provider_response.usage
        text = provider_response.get_text()
        if response_schema and text:
            try:
                provider_response.structured_output = json.loads(text)
            except json.JSONDecodeError:
                pass
        return TurnResult.from_provider_response(provider_response)

    def _translate_error(self, exc: Exception) -> Exception:
        """Map a botocore ``ClientError`` onto a domain error.

        botocore raises one class for every API failure and hides the
        diagnosis in ``response["Error"]["Code"]``, so without this every
        failure would be retried alike — or none would. Anything unrecognised
        is returned UNCHANGED, so the caller re-raises it rather than
        laundering an unknown failure into a terminal answer.
        """
        code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str((response.get("Error") or {}).get("Code") or "")
        if not code:
            code = type(exc).__name__
        detail = str(exc)

        transient = _TRANSIENT_CODES.get(code)
        if transient is ThrottlingError:
            return ThrottlingError(retry_after=self.get_retry_after(exc),
                                   original_error=detail)
        if transient is ModelNotReadyError:
            return ModelNotReadyError(model=self._model_name,
                                      original_error=detail)
        if transient is ServiceUnavailableError:
            return ServiceUnavailableError(original_error=detail)

        if code == "AccessDeniedException":
            return AccessDeniedError(model=self._model_name,
                                     region=self._region, original_error=detail)
        if code == "ResourceNotFoundException":
            return ModelNotFoundError(model=self._model_name,
                                      region=self._region, original_error=detail)
        if code == "ValidationException" and "too long" in detail.lower():
            return ContextLimitError(original_error=detail)
        if "NoCredentialsError" in code or "NoRegionError" in code:
            return CredentialsNotFoundError(
                checked_locations=get_checked_credential_locations(),
                region=self._region)
        return exc

    # ==================== Streaming ====================

    def _stream_response(
        self,
        request: Dict[str, Any],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Consume a ``ConverseStream`` event stream into a ProviderResponse.

        Converse streams one content block at a time, keyed by
        ``contentBlockIndex``: ``contentBlockStart`` opens a ``toolUse``
        block, ``contentBlockDelta`` carries text, tool-argument JSON
        fragments or reasoning, and ``contentBlockStop`` closes whichever
        block the index names. ``messageStop`` is the terminal event and
        ``metadata`` carries the final usage.

        Parts are appended in arrival order so text and tool calls stay
        interleaved as the model produced them.
        """
        state = _StreamState()
        try:
            self._trace(f"STREAM_START msgs={len(request.get('messages') or [])}")
            response = self._client.converse_stream(**request)
            for event in response.get("stream") or []:
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"STREAM_CANCELLED after {state.chunks} chunks")
                    state.was_cancelled = True
                    state.finish_reason = FinishReason.CANCELLED
                    break
                self._handle_stream_event(
                    event, state, on_chunk=on_chunk,
                    on_usage_update=on_usage_update,
                    on_function_call=on_function_call, on_thinking=on_thinking)
            self._trace(
                f"STREAM_END chunks={state.chunks} finish={state.finish_reason}")
        except Exception as exc:
            self._trace(f"STREAM_ERROR {type(exc).__name__}: {exc}")
            if cancel_token and cancel_token.is_cancelled:
                state.was_cancelled = True
                state.finish_reason = FinishReason.CANCELLED
            else:
                raise

        return self._finalise_stream(state)

    def _finalise_stream(self, state: "_StreamState") -> ProviderResponse:
        """Close the accumulated turn and decide whether it IS a turn.

        Split from the loop so each stays under the complexity ceiling.
        Three judgements live here, and each has an issue behind it:

        * a **cancelled** turn contributes no tool calls -- an unanswered
          ``toolUse`` poisons every later request in the session;
        * ``TOOL_USE`` fills in an unreported or merely-``stop`` finish but
          must not displace a terminal one, because a turn that hit the
          output cap mid-arguments carries fragments rather than a request
          (#745);
        * a stream that never said it ended did not finish, and
          :func:`require_terminated_stream` RAISES rather than returning the
          fragment (#687).
        """
        state.flush_text()
        if state.was_cancelled:
            state.parts = [p for p in state.parts if p.function_call is None]
        thinking = "".join(state.reasoning) or None
        if thinking and state.usage.thinking_tokens is None:
            state.usage.thinking_tokens = max(1, len(thinking) // 4)

        finish_reason = resolve_tool_use_finish(
            state.finish_reason,
            has_function_calls=(any(p.function_call for p in state.parts)
                                and not state.was_cancelled),
        )
        return require_terminated_stream(
            ProviderResponse(parts=state.parts, usage=state.usage,
                             finish_reason=finish_reason, raw=None,
                             thinking=thinking),
            terminal_seen=state.terminal_seen,
            was_cancelled=state.was_cancelled,
            provider=self.name,
            model=self._model_name,
            chunks=state.chunks,
        )

    def _handle_stream_event(
        self,
        event: Dict[str, Any],
        state: "_StreamState",
        *,
        on_chunk: StreamingCallback,
        on_usage_update: Optional[UsageUpdateCallback],
        on_function_call: Optional[FunctionCallDetectedCallback],
        on_thinking: Optional[ThinkingCallback],
    ) -> None:
        """Fold ONE stream event into ``state``.

        Split from the loop so each stays under the complexity ceiling, and
        so the event vocabulary reads as the table it is.
        """
        if "contentBlockStart" in event:
            self._open_block(event["contentBlockStart"], state, on_thinking)
        elif "contentBlockDelta" in event:
            self._apply_delta(event["contentBlockDelta"], state,
                              on_chunk=on_chunk, on_thinking=on_thinking)
        elif "contentBlockStop" in event:
            self._close_block(event["contentBlockStop"], state, on_function_call)
        elif "messageStop" in event:
            state.terminal_seen = True
            state.finish_reason = finish_reason_from_bedrock(
                (event["messageStop"] or {}).get("stopReason"))
        elif "metadata" in event:
            state.usage = usage_from_bedrock((event["metadata"] or {}).get("usage"))
            self._trace(
                f"STREAM_USAGE prompt={state.usage.prompt_tokens} "
                f"output={state.usage.output_tokens}")
            if on_usage_update and state.usage.total_tokens > 0:
                on_usage_update(state.usage)

    def _open_block(self, start_event: Dict[str, Any], state: "_StreamState",
                    on_thinking: Optional[ThinkingCallback]) -> None:
        """Begin accumulating a ``toolUse`` block."""
        start = (start_event or {}).get("start") or {}
        use = start.get("toolUse")
        if not use:
            return
        state.emit_thinking(on_thinking)
        index = (start_event or {}).get("contentBlockIndex", 0)
        self._trace(
            f"STREAM_TOOL_START idx={index} id={use.get('toolUseId')!r} "
            + wire_name_trace_fields(use.get("name"))
        )
        state.tool_calls[index] = {"id": use.get("toolUseId", ""),
                                   "name": use.get("name", ""),
                                   "json_chunks": []}

    def _apply_delta(self, delta_event: Dict[str, Any], state: "_StreamState",
                     *, on_chunk: StreamingCallback,
                     on_thinking: Optional[ThinkingCallback]) -> None:
        """Route one ``contentBlockDelta`` to text, tool args or reasoning."""
        delta = (delta_event or {}).get("delta") or {}
        index = (delta_event or {}).get("contentBlockIndex", 0)

        text = delta.get("text")
        if text:
            state.emit_thinking(on_thinking)
            state.chunks += 1
            state.text.append(text)
            on_chunk(text)

        tool_delta = delta.get("toolUse")
        if tool_delta is not None and index in state.tool_calls:
            state.tool_calls[index]["json_chunks"].append(
                tool_delta.get("input") or "")

        reasoning = delta.get("reasoningContent") or {}
        if reasoning.get("text"):
            state.reasoning.append(reasoning["text"])

    def _close_block(self, stop_event: Dict[str, Any], state: "_StreamState",
                     on_function_call: Optional[FunctionCallDetectedCallback]
                     ) -> None:
        """Finalise the tool call the stopped block was accumulating."""
        index = (stop_event or {}).get("contentBlockIndex", 0)
        pending = state.tool_calls.pop(index, None)
        if pending is None:
            return
        state.flush_text()
        state.parts.append(Part.from_function_call(
            self._finalise_call(pending, on_function_call)))

    def _finalise_call(
        self, pending: Dict[str, Any],
        on_function_call: Optional[FunctionCallDetectedCallback],
    ) -> FunctionCall:
        """Build the FunctionCall for an accumulated ``toolUse`` block.

        Converse streams the arguments as JSON text fragments, so a stream
        cut mid-object leaves a severed string: ``parse_tool_call_arguments``
        keeps that unreadable rather than presenting it as a zero-argument
        call (#750), which the session refuses instead of executing.
        """
        from shared.tool_id_map import id_to_name
        args, unreadable = parse_tool_call_arguments(
            "".join(pending["json_chunks"]))
        call = FunctionCall(id=pending["id"], name=id_to_name(pending["name"]),
                            args=args, unreadable_args=unreadable)
        self._trace(f"STREAM_FUNC_CALL name={call.name}")
        if on_function_call:
            on_function_call(call)
        return call

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize history to a JSON string for persistence."""
        return serialize_history(history)

    def deserialize_history(self, data: str) -> List[Message]:
        """Restore history from :meth:`serialize_history` output."""
        return deserialize_history(data)

    # ==================== Retry classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Tell the retry layer what kind of failure this was.

        ``None`` defers to the global classifier, which is the right answer
        for anything this provider did not translate.
        """
        if isinstance(exc, ThrottlingError):
            return {"transient": True, "rate_limit": True, "infra": False}
        if isinstance(exc, (ServiceUnavailableError, ModelNotReadyError)):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Seconds Bedrock asked us to wait, if it said.

        Bedrock returns the hint as a ``Retry-After`` response header rather
        than in the error body, so it is read off botocore's captured HTTP
        metadata.
        """
        if isinstance(exc, ThrottlingError) and exc.retry_after:
            return float(exc.retry_after)
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return None
        headers = (response.get("ResponseMetadata") or {}).get(
            "HTTPHeaders") or {}
        raw = headers.get("retry-after")
        try:
            return float(raw) if raw else None
        except (TypeError, ValueError):
            return None


class _StreamState:
    """Mutable accumulator for one ``ConverseStream`` turn.

    A class rather than a pile of ``nonlocal`` locals because the event
    handling is split across four methods, and passing one object beats
    threading eight names through each of them.

    Attributes:
        parts: Response parts in arrival order, so text and tool calls stay
            interleaved as the model produced them.
        text: Text fragments of the CURRENT text block, flushed into a single
            ``Part`` whenever a tool call interrupts it or the stream ends.
        reasoning: Reasoning fragments; joined into
            ``ProviderResponse.thinking`` and never replayed (see the module
            docstring).
        thinking_emitted: Whether ``on_thinking`` has already fired. The
            callback fires ONCE, before the first text chunk or tool call,
            because that is when the model stopped thinking and started
            answering.
        tool_calls: ``contentBlockIndex`` -> the ``toolUse`` block being
            accumulated. Keyed by index because Converse may interleave
            blocks, so "the last one" is not a safe assumption.
        terminal_seen: Whether ``messageStop`` arrived. Absent it, the stream
            was cut and ``require_terminated_stream`` refuses the fragment.
        was_cancelled: Whether the cancel token fired mid-stream.
    """

    def __init__(self) -> None:
        self.parts: List[Part] = []
        self.text: List[str] = []
        self.reasoning: List[str] = []
        self.thinking_emitted: bool = False
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.usage: TokenUsage = TokenUsage()
        self.finish_reason: FinishReason = FinishReason.UNKNOWN
        self.terminal_seen: bool = False
        self.was_cancelled: bool = False
        self.chunks: int = 0

    def flush_text(self) -> None:
        """Close the current text block into a ``Part``, if it has content."""
        if self.text:
            self.parts.append(Part.from_text("".join(self.text)))
            self.text = []

    def emit_thinking(self, on_thinking: Optional[ThinkingCallback]) -> None:
        """Deliver the accumulated reasoning once, at the first answer token."""
        if self.thinking_emitted or not self.reasoning or not on_thinking:
            return
        on_thinking("".join(self.reasoning))
        self.thinking_emitted = True


def create_provider() -> BedrockProvider:
    """Factory function for plugin discovery."""
    return BedrockProvider()
