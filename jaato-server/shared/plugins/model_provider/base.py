"""Base protocol for Model Provider plugins.

This module defines the interface that all model provider plugins must implement.
Model providers encapsulate all SDK-specific logic for interacting with AI models
(Google GenAI, Anthropic, OpenAI, etc.).

Providers are stateless with respect to conversation history. The session owns
the canonical message list and passes it to ``complete()`` on each call.
Providers hold only connection/auth state set by ``initialize()`` and
``connect()``.
"""

from dataclasses import dataclass, field
from shared.secret_repr import secret_safe_repr
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FunctionCall,
    MediaDelta,
    Message,
    Part,
    ProviderResponse,
    ThinkingConfig,
    ToolResult,
    ToolSchema,
    TokenUsage,
    TurnResult,
)


# Output callback type for real-time streaming
# Parameters: (source: str, text: str, mode: str)
#   source: "model" for model output, plugin name for plugin output
#   text: The output text
#   mode: "write" for new block, "append" to continue
OutputCallback = Callable[[str, str, str], None]

# Streaming callback type for token-level streaming.
# Parameters: (chunk: str | MediaDelta) - one delta from the model.
#
# ``str`` is a text token chunk, as it always was.  ``MediaDelta`` is a
# chunk of model-GENERATED binary output (an audio-capable model speaking
# its answer); it is delivered on the same callback rather than a parallel
# one so that ordering between a model's text and its audio is preserved
# by construction -- two callbacks would have to be re-interleaved by
# every consumer.
#
# A consumer MUST branch on the type.  ``MediaDelta`` is not text: adding
# it to a transcript, taking ``len()`` of it, or coercing it with ``str()``
# corrupts both the audio and the transcript.  Every in-tree consumer
# branches explicitly; text-only providers are unaffected, since a
# provider that never emits media never passes anything but ``str``.
StreamingCallback = Callable[[Union[str, MediaDelta]], None]

# Thinking callback type for extended thinking content
# Parameters: (thinking: str) - accumulated thinking content from the model
# Called BEFORE text streaming begins, when thinking is complete
ThinkingCallback = Callable[[str], None]

# Usage update callback for real-time token accounting
# Parameters: (usage: TokenUsage) - current token usage from streaming
UsageUpdateCallback = Callable[[TokenUsage], None]

# GC threshold callback for proactive garbage collection notifications
# Parameters: (percent_used: float, threshold: float) - current and threshold percentages
# Called when context usage crosses configured threshold during streaming
GCThresholdCallback = Callable[[float, float], None]

# Function call detected callback for streaming
# Parameters: (function_call: FunctionCall) - the function call detected mid-stream
# Called when a function call is detected during streaming, BEFORE any subsequent
# text chunks are emitted. This allows the caller to insert tool tree markers
# at the correct position between text blocks.
FunctionCallDetectedCallback = Callable[[FunctionCall], None]


# Authentication method type for Google GenAI provider
GoogleAuthMethod = Literal["auto", "api_key", "service_account_file", "adc", "impersonation"]


@dataclass
class ProviderConfig:
    """Configuration for model provider initialization.

    Providers may use different subsets of these fields depending on
    their authentication requirements.

    Attributes:
        project: Cloud project ID (GCP, AWS, etc.).
        location: Region/location for the service.
        api_key: API key for authentication (if applicable).
        credentials_path: Path to credentials file (if applicable).
        use_vertex_ai: If True, use Vertex AI endpoint (requires project/location).
            If False, use Google AI Studio endpoint (requires api_key).
            Default is True for backwards compatibility.
        auth_method: Authentication method to use. Options:
            - "auto": Automatically detect from available credentials (default)
            - "api_key": Use API key (Google AI Studio)
            - "service_account_file": Use service account JSON file
            - "adc": Use Application Default Credentials
            - "impersonation": Use service account impersonation
        target_service_account: Target service account email for impersonation.
            Required when auth_method is "impersonation".
        credentials: Pre-built credentials object (advanced usage).
            When provided, this takes precedence over other auth methods.
        extra: Provider-specific additional configuration.
    """
    project: Optional[str] = None
    location: Optional[str] = None
    api_key: Optional[str] = None
    credentials_path: Optional[str] = None
    use_vertex_ai: bool = True
    auth_method: GoogleAuthMethod = "auto"
    target_service_account: Optional[str] = None
    credentials: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Never print the key, and never print a secret that arrived
    # through ``extra`` (a provider's whole ``plugin_configs`` block
    # lands there, ``api_token`` included).  A bare dataclass repr put
    # a live key into a pytest failure message once already (#721);
    # this config object is passed through far more code than that
    # one was.  ``extra``'s non-secret keys still render, because
    # ``host`` / ``context_length`` / ``routing`` are what the repr is
    # read for.
    __repr__ = secret_safe_repr(
        "api_key", "credentials", mappings=("extra",),
    )


def profile_api_key_location(
    config: Optional["ProviderConfig"], provider_name: str
) -> str:
    """Describe the profile ``plugin_configs.<provider>.api_key`` knob for a
    provider's ``get_checked_credential_locations()`` list.

    The runtime promotes ``plugin_configs.<provider>.api_key`` to
    ``config.api_key`` (``jaato_runtime.py``) — the HIGHEST-precedence key
    source, consulted before env vars and stored credentials
    (``self._api_key = config.api_key or resolve_...``).  Env-only
    checked-locations helpers cannot see it, so an ``APIKeyNotFoundError``
    omitted the very first source it checked.  This returns a single
    precedence-ordered line (masked when set) so the profile knob is always
    surfaced.  ``config`` may be the initialize-time config (key promoted to
    ``config.api_key``) or the verify-time config (key still under
    ``config.extra['api_key']``) — both are inspected.

    Returns a ``"... : not set"`` line when ``config`` is None or carries no
    key, so callers can unconditionally prepend it.
    """
    key: Optional[str] = None
    if config is not None:
        key = getattr(config, "api_key", None)
        if not key:
            extra = getattr(config, "extra", None) or {}
            key = extra.get("api_key")
    label = f"plugin_configs.{provider_name}.api_key (profile)"
    if key:
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        return f"{label}: set ({masked})"
    return f"{label}: not set"


def resolve_context_window(
    *,
    detect_capacity: Optional[Callable[[], Optional[int]]] = None,
    profile_value: Optional[Any] = None,
    env_value: Optional[int] = None,
) -> Optional[int]:
    """Resolve a model's context-window size by a generic precedence.

    Shared by every provider so the resolution order is uniform; the
    *primary* tier is an optional per-provider hook.

    Precedence (2026-06-10 design):

    1. **PRIMARY — provider auto-detect.**  If ``detect_capacity`` is
       supplied — a provider whose backend exposes its own capacity
       endpoint, e.g. vLLM's ``GET /v1/models`` ``max_model_len`` — its
       non-falsy return wins.  The server is the source of truth and
       self-updates with engine config (rope/YARN scaling, re-launch
       with a different ``--max-model-len``).  Providers without such an
       endpoint simply omit the hook, so tier-1 is a no-op for them and
       resolution falls straight through to the manual tiers —
       preserving their existing behaviour.
    2. **fallback — per-profile knob.**  ``plugin_configs.<provider>.
       context_length`` (a per-model-safe manual override for backends
       whose capacity isn't auto-detectable, or to pin a value).
    3. **fallback — environment variable.**  e.g. ``VLLM_CONTEXT_LENGTH``.

    Returns the first non-falsy tier, or ``None`` so the caller raises a
    provider-specific "not configured" error.  No hardcoded fallback is
    ever substituted (project no-fallback rule).

    ``detect_capacity`` should itself be failure-tolerant (return
    ``None`` on an unreachable/blank endpoint) so a transient blip
    degrades to the manual tiers rather than raising here.
    """
    if detect_capacity is not None:
        detected = detect_capacity()
        if detected:
            return int(detected)
    if profile_value:
        return int(profile_value)
    if env_value:
        return int(env_value)
    return None


# Canonical input-modality tokens.  Lowercase, matching OpenRouter's
# ``architecture.input_modalities`` vocabulary so catalog detection needs
# no translation.  Every text-completion model accepts text, so ``"text"``
# is the floor that :func:`_normalise_modalities` always re-adds.
MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"
MODALITY_FILE = "file"  # PDFs / documents (OpenRouter's term)


def _normalise_modalities(raw: Iterable[str]) -> Set[str]:
    """Lower-case, strip, drop blanks, and re-add the ``"text"`` floor.

    A resolved modality set always contains ``"text"`` — every
    text-completion model accepts text; richer modalities only *add* to
    that floor.  This is a contract invariant, not a guessed fallback.
    """
    normalised = {str(m).strip().lower() for m in raw if str(m).strip()}
    normalised.add(MODALITY_TEXT)
    return normalised


def resolve_modalities(
    *,
    detect: Optional[Callable[[], Optional[Iterable[str]]]] = None,
    profile_value: Optional[Iterable[str]] = None,
    table_value: Optional[Iterable[str]] = None,
) -> Optional[Set[str]]:
    """Resolve a model's INPUT modality set ({"text","image",...}).

    Sibling of :func:`resolve_context_window` — the same per-model
    metadata problem (a gateway serves both vision and text-only models),
    so the same resolution discipline applies.

    Precedence:

    1. **detect — live catalog/endpoint (authoritative, self-updating).**
       A provider whose backend lists per-model modalities, e.g.
       OpenRouter's ``GET /api/v1/models`` ``architecture.input_modalities``.
       Its non-empty return wins and tracks the catalog automatically.
       Providers without such an endpoint omit the hook.
    2. **profile knob — explicit override / escape hatch.**
       ``plugin_configs.<provider>.modalities: ["text","image"]`` to assert
       a model the catalog doesn't list yet, or to correct it.
    3. **static per-model table — documented constants.** A provider's
       ``MODEL_*`` map (mirrors ``MODEL_CONTEXT_LIMITS``) for closed
       providers with no live modality endpoint.

    Returns the first resolved set (normalised, ``"text"``-floored), or
    ``None`` when nothing resolves — ``"unknown"``.  The caller (the
    capability mixin, the tier validator, the content gate) decides what
    ``None`` means; this helper never substitutes a guessed value.  No
    hardcoded fallback (project no-fallback rule).

    ``detect`` should be failure-tolerant (return ``None`` on an
    unreachable/blank endpoint) so a transient blip degrades to the
    manual tiers rather than raising here.
    """
    if detect is not None:
        detected = detect()
        if detected:
            return _normalise_modalities(detected)
    if profile_value:
        return _normalise_modalities(profile_value)
    if table_value:
        return _normalise_modalities(table_value)
    return None


class ModalityCapabilityMixin:
    """Default input-modality capability: **text-only**.

    Mix into a provider to give it the honest baseline answer — its
    adapter marshals text input only.  This is the accurate, complete
    statement for the providers whose adapters never convert richer
    content (the OpenAI-compat + local fleet), not a placeholder guess.

    Providers whose adapter DOES convert richer input (image, …) override
    :meth:`modalities` to resolve per-model via :func:`resolve_modalities`
    (catalog detect / profile knob / static table).  They inherit
    :meth:`supports_modality` unchanged.

    The mixin carries no state and no ``__init__``, so adding it to a
    provider's bases is a single-token change that never disturbs the
    provider's own constructor.
    """

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (or the active model) accepts.

        Text-only floor — this adapter marshals text only, so the answer
        is model-independent and ``model`` is ignored here.  The ``model``
        parameter lets callers ask about a model OTHER than the connected
        one (e.g. validating a vision-tier model before switching into it);
        image-capable overrides honor it.
        """
        return {MODALITY_TEXT}

    def supports_modality(self, kind: str, model: Optional[str] = None) -> bool:
        """Whether ``model`` (or the active model) accepts ``kind`` as input."""
        return kind.strip().lower() in self.modalities(model)

    def output_modalities(self, model: Optional[str] = None) -> Set[str]:
        """OUTPUT modalities ``model`` (or the active model) can EMIT.

        Deliberately NOT named ``modalities``: that name is framework-wide
        for *input* (see :meth:`modalities` and :meth:`supports_modality`),
        and reusing it for the opposite direction is the single easiest
        way to introduce a silent capability bug.  The naming collision is
        real and external too -- OpenAI's Chat Completions request field
        ``modalities`` means OUTPUT, while a jaato tier's ``modalities``
        key means INPUT.

        Text-only floor: every model emits text, and an adapter that does
        not parse response media cannot honestly claim more.

        A provider raises the floor either by setting
        ``_output_modalities_knob`` (the profile-level assertion, mirroring
        the input ``modalities`` knob -- the catalogs do not report output
        modalities, so an operator naming an audio model is the only
        source of truth available) or by overriding this method outright.
        ``text`` is always included: a model that speaks still writes.
        """
        knob = getattr(self, "_output_modalities_knob", None)
        if knob:
            return {str(m).strip().lower() for m in knob if m} | {MODALITY_TEXT}
        return {MODALITY_TEXT}

    def supports_output_modality(
        self, kind: str, model: Optional[str] = None
    ) -> bool:
        """Whether ``model`` (or the active model) can EMIT ``kind``.

        The startup tier check probes this method by name via ``getattr``
        and skips the check when it is absent, so defining it here turns
        outbound tier roles from unverified declarations into enforced
        ones.  That is the intended activation: a tier declaring
        ``audio: outbound`` against a provider that cannot emit audio is a
        configuration error worth catching at startup rather than at the
        first turn that tries to speak.
        """
        return kind.strip().lower() in self.output_modalities(model)

    def emit_media_delta(self, delta: Any, on_chunk: Any, sequence: int) -> int:
        """Decode model-generated media from ``delta`` and emit it.

        The no-op default IS the contract: a streaming loop calls this
        unconditionally, and a provider whose wire format carries no model
        media simply never overrides it, paying one call that returns
        ``sequence`` unchanged.  That is what lets media support be
        implemented per provider rather than assumed of all of them.

        A provider opts in either by adding
        ``_media_deltas.OpenAIMediaOutputMixin`` to its bases (OpenAI-shaped
        wire) or by overriding this with its own decoder.

        Args:
            delta: One streaming delta, in whatever shape the provider's
                SDK produces.
            on_chunk: The :data:`StreamingCallback`; receives a
                ``MediaDelta``, never a ``str``.
            sequence: Last media sequence issued for this turn.

        Returns:
            The new sequence -- unchanged when nothing was emitted.
        """
        return sequence

    def request_output_modalities(self, kinds: Any) -> None:
        """Ask the provider to make the model EMIT ``kinds`` on later turns.

        Called on every tier entry with that tier's declared outbound
        roles, which is what turns ``modalities: {audio: outbound}`` from a
        declaration into a request.  An empty set is meaningful and must
        be honoured: it is how leaving a speaking tier stops asking for
        audio.

        The no-op default is deliberate.  A provider that cannot emit media
        ignores the request rather than failing -- the startup capability
        check (against :meth:`supports_output_modality`) is what refuses a
        tier whose model genuinely cannot do the job, and it runs long
        before any request is built.
        """
        return None

    def capabilities(self) -> "ProviderCapabilities":
        """Declared wire-level capabilities (baseline default).

        Returns the honest minimal-adapter baseline.  Providers OVERRIDE
        this to return their ``PROVIDER_CAPABILITIES`` constant.  See
        :class:`ProviderCapabilities`.
        """
        return ProviderCapabilities()


# Canonical capability fields — the COLUMNS of the capability matrix.  Each is a
# concrete, wire-testable behavior (not a vague label) so the CI conformance
# guard can assert it.  Adding a field here is the one place a new capability is
# defined; the doc generator and the guard both read this tuple.
CAPABILITY_FIELDS = (
    "user_message_images",
    "tool_result_images",
    "pdf_input",
    "tool_choice_forwarding",
    "thinking",
    "prompt_caching",
    "streaming",
    "cancellation",
    "output_media",
)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared wire-level capabilities of a model provider.

    The SINGLE SOURCE OF TRUTH for both the capability doc table
    (``docs/model-provider-capabilities.md``) and the CI conformance guard.
    Each provider declares a module-level ``PROVIDER_CAPABILITIES`` constant in
    its ``__init__.py`` and returns it from :meth:`capabilities`.

    The guard asserts BOTH (1) every provider declares it (structural, AST) AND
    (2) every flag set ``True`` is actually delivered on the wire (behavioral) —
    closing the "declared ``image: yes`` but the converter silently drops it"
    gap this contract exists to prevent.

    Defaults are the honest baseline of a minimal OpenAI-compat adapter: it
    streams and cancels, but marshals text only and forwards no advanced knobs.
    Providers override per the behavior they actually implement.
    """

    user_message_images: bool = False   # Part.inline_data image -> wire image block
    tool_result_images: bool = False    # tool-result image attachment -> the model
    pdf_input: bool = False             # application/pdf attachment -> wire file/document block
    tool_choice_forwarding: bool = False  # complete(tool_choice=...) reaches the wire
    thinking: bool = False              # extended-reasoning request and/or extraction
    prompt_caching: bool = False        # cache_control breakpoints emitted on the wire
    streaming: bool = True              # on_chunk token streaming
    cancellation: bool = True           # cancel_token actually halts generation
    output_media: bool = False          # model-generated media -> MediaDelta on on_chunk

    def as_dict(self) -> Dict[str, bool]:
        return {f: bool(getattr(self, f)) for f in CAPABILITY_FIELDS}


# Canonical config-knob layers — the named sub-dicts under
# ``plugin_configs.<provider>`` an author can set.  ``top_level`` is the
# special layer for keys read directly off the provider config (not nested).
# Adding a layer name here is informational only; ProviderKnobs accepts any
# layer string, but keeping the vocabulary in one place lets the doc
# generator and the validator share a stable set of column names.
KNOB_LAYERS = (
    "top_level",            # keys directly under plugin_configs.<provider>
    "api_params",           # request-body sampling / generation params
    "routing",              # gateway routing extension (provider-specific)
    "load",                 # model-load passthrough (local runtimes)
    "framework_overrides",  # rare escape hatches (context_length, base_url)
)


@dataclass(frozen=True)
class KnobSpec:
    """One config knob the provider reads: key, type, default, description.

    ``type`` is a hint string — one of ``"str"``, ``"int"``, ``"float"``,
    ``"bool"``, ``"list"``, ``"dict"``.  ``default`` is the provider's
    STATIC default when the key is absent, or ``None`` when the provider
    RESOLVES it at runtime (env var → auth store → catalog).  ``None`` here
    honestly means "no static default; the provider has a resolution chain",
    NOT "defaults to null" — authored from the provider.py read site.
    """

    name: str
    type: str = "str"
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class KnobLayer:
    """One config layer of a provider and the knobs it recognizes.

    A layer maps to a nesting level under ``plugin_configs.<provider>``:
    ``top_level`` knobs sit directly under the provider; named layers
    (``api_params``, ``routing``, ``load``, ``framework_overrides``) are
    sub-dicts the provider reads.

    ``opaque=True`` marks a **pass-through** layer — the provider forwards
    every key verbatim to its upstream (e.g. OpenRouter's ``routing``
    provider-extension dict, LM Studio's ``load`` body).  For an opaque
    layer ``knobs`` is advisory documentation only; the validator MUST NOT
    reject unknown keys in it (there is no fixed key set — that's the
    point).  For a non-opaque layer ``knobs`` is the authoritative, closed
    set: the validator rejects anything outside it (these are the
    silently-ignored-typo failures the contract exists to catch).
    """

    layer: str
    knobs: Tuple[KnobSpec, ...] = ()
    opaque: bool = False
    description: str = ""

    @property
    def keys(self) -> FrozenSet[str]:
        """The recognized key names (for the validator's membership test)."""
        return frozenset(k.name for k in self.knobs)


@dataclass(frozen=True)
class ProviderKnobs:
    """Declared config knobs of a provider, grouped by layer.

    The SINGLE SOURCE OF TRUTH for ``jaato-scaffold explain provider <name>
    --knobs`` and for the ``validate`` linter's unknown-knob detection.
    Each provider declares a module-level ``PROVIDER_KNOBS`` constant in its
    ``__init__.py`` (sibling of ``PROVIDER_CAPABILITIES``), authored from the
    keys the provider's code actually reads — NOT from prose docs, which
    drift.

    **Framework-wiring keys are intentionally excluded.**  ``workspace_path``
    / ``config_root`` are set by the framework, never by a profile author, so
    they are not knobs (same exclusion rule as ``PluginSetting`` /
    ``get_config_schema``).  Auth/identity keys an author CAN set in a
    profile (``api_key``, ``oauth_token``) ARE included, in ``top_level``.

    A CI guard asserts every provider declares ``PROVIDER_KNOBS`` (+
    ``PROVIDER_QUIRKS``), mirroring the capability guard — so a new provider
    cannot silently ship prose-only knobs again.
    """

    layers: Tuple[KnobLayer, ...] = ()

    def get_layer(self, name: str) -> Optional[KnobLayer]:
        """Return the declared :class:`KnobLayer` named ``name``, or None."""
        for lyr in self.layers:
            if lyr.layer == name:
                return lyr
        return None

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Flatten to JSON-friendly ``{layer: {opaque, description, knobs}}``.

        Each knob carries ``{type, default, description}``.  Used by
        ``explain --json`` and the doc generator.
        """
        return {
            lyr.layer: {
                "opaque": lyr.opaque,
                "description": lyr.description,
                "knobs": {
                    k.name: {
                        "type": k.type,
                        "default": k.default,
                        "description": k.description,
                    }
                    for k in lyr.knobs
                },
            }
            for lyr in self.layers
        }

    def accepts(self, layer: str, key: str) -> bool:
        """Whether ``key`` is valid in ``layer`` (for the validator).

        Opaque layers accept any key (pass-through).  A non-opaque layer
        accepts only its declared keys.  An undeclared layer name is
        rejected (the author nested under a layer the provider never reads).
        """
        lyr = self.get_layer(layer)
        if lyr is None:
            return False
        return True if lyr.opaque else key in lyr.keys


# Canonical credential-source kinds for PROVIDER_AUTH_RESOLUTION steps.
AUTH_SOURCE_KINDS = (
    "api_key_param",  # api_key via ProviderConfig / plugin_configs.<provider>.api_key
    "env",            # an environment variable (name = the var)
    "stored",         # a credential persisted by an auth plugin (name = command/file)
    "oauth",          # an interactive OAuth token store (name = the store file)
    "adc",            # Google Application Default Credentials
    "cli",            # delegates to an external CLI's own session
    "none",           # no credential required (local servers)
)


@dataclass(frozen=True)
class AuthSource:
    """One step in a provider's credential-resolution chain.

    Providers try sources in order, highest priority first, until one yields a
    credential (the ``or``-chain in ``resolve_api_key`` / ``verify_auth``).
    Authored from the provider's actual auth code, NOT prose — so
    ``explain provider`` can show *why* a credential resolved the way it did
    (the #1 provider confusion source: "I set X but it used Y").

    Attributes:
        kind: one of :data:`AUTH_SOURCE_KINDS`.
        name: the concrete identifier — env var name, auth command, store file
            (empty for ``adc`` / ``none``).
        note: optional human note (e.g. "vendor var", "optional").
    """

    kind: str
    name: str = ""
    note: str = ""


# Default empty quirk set.  Providers that honor model quirks declare a
# module-level ``PROVIDER_QUIRKS = frozenset({...})`` in their ``__init__.py``
# enumerating the quirk names they read from ``profile.quirks`` /
# ``config.extra["quirks"]``.  Today only ``vllm`` is non-empty; the guard
# requires every provider to declare the constant (empty is the honest answer
# for the rest) so a new quirk consumer can't ship an unvalidated allow-list.
PROVIDER_QUIRKS_DEFAULT: FrozenSet[str] = frozenset()


@runtime_checkable
class ModelProviderPlugin(Protocol):
    """Protocol for Model Provider plugins.

    Model providers encapsulate all interactions with a specific AI SDK:
    - Connection and authentication
    - Stateless completion via ``complete()``
    - Token counting and context management
    - History serialization for persistence

    Providers are stateless with respect to conversation history. The session
    owns the canonical message list and passes it to ``complete()`` on each
    call. Providers hold only connection/auth state set during lifecycle
    methods.

    Example implementation:
        class MyProvider:
            @property
            def name(self) -> str:
                return "my_provider"

            def initialize(self, config: ProviderConfig) -> None:
                self._client = SomeSDK(api_key=config.api_key)

            def connect(self, model: str) -> None:
                self._model_name = model

            def complete(self, messages, system_instruction=None, tools=None, **kw):
                response = self._client.chat(messages=messages, model=self._model_name)
                return TurnResult.from_provider_response(convert_response(response))
    """

    @property
    def name(self) -> str:
        """Unique identifier for this provider (e.g., 'google_genai', 'anthropic')."""
        ...

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with configuration.

        This is called once when the provider is first set up.
        Establishes the SDK client connection.

        Args:
            config: Provider configuration with auth details.
        """
        ...

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Verify that authentication is configured and optionally trigger interactive login.

        This should be called BEFORE sending messages to ensure
        credentials are available. For providers that support interactive login
        (like Anthropic OAuth), this can trigger the login flow.

        Args:
            allow_interactive: If True and auth is not configured, attempt
                interactive login (e.g., browser-based OAuth). If False,
                only check if credentials exist without prompting.
            on_message: Optional callback for status messages during interactive
                login (e.g., "Opening browser...", "Waiting for auth...").
            config: Optional ``ProviderConfig`` with profile-level knobs
                already merged into ``extra`` by the runtime.  Providers
                that resolve credentials from non-environment sources
                (e.g. an API token in ``plugin_configs[provider]``) can
                read those values here.  Implementations that don't need
                the profile config simply ignore this kwarg.  Unlike
                ``initialize()`` — which also receives config — this
                call must remain cheap: read values, do not create
                clients or make network requests.

        Returns:
            True if authentication is configured and valid.
            False if authentication failed or was not completed.

        Raises:
            APIKeyNotFoundError: If allow_interactive=False and no credentials found.
        """
        ...

    def shutdown(self) -> None:
        """Clean up any resources held by the provider."""
        ...

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used.

        Called after ``initialize()`` to describe which auth method or
        credential file was resolved. Displayed in the "Connected to"
        message so users can see where credentials came from — critical
        for diagnosing fallback credential issues across workspaces.

        Returns:
            Human-readable string, e.g. ``"API key from ~/.jaato/zhipuai_auth.json"``,
            ``"PKCE OAuth"``, ``"ADC"``. Empty string if unknown.
        """
        ...

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the model to use for this provider.

        Args:
            model: Model name/ID (e.g., 'gemini-2.5-flash', 'claude-sonnet-4-5-20250929').
            skip_model_test: If True, skip the ``_verify_model_responds()``
                network call during connect.  The model will be validated on
                the first real message instead.  Used during bootstrap to
                reduce startup latency.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        ...

    @property
    def model_name(self) -> Optional[str]:
        """Get the currently configured model name."""
        ...

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from this provider.

        Args:
            prefix: Optional filter prefix (e.g., 'gemini', 'claude').

        Returns:
            List of model names/IDs.
        """
        ...

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
        on_usage_update: Optional['UsageUpdateCallback'] = None,
        on_function_call: Optional['FunctionCallDetectedCallback'] = None,
        on_thinking: Optional['ThinkingCallback'] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to provider format, call API, return response.

        This method does NOT modify any internal state. The caller (session)
        is responsible for maintaining the message list.

        When on_chunk is provided, the response is streamed token-by-token.
        When on_chunk is None, the response is returned in batch mode.

        Providers MUST:

        * Return ``TurnResult.from_provider_response(r)`` on success.
        * Return ``TurnResult.from_exception(exc)`` for **non-transient**
          errors (auth failures, context limits, safety blocks).
        * **Raise** transient errors (rate limits, overload) so the
          ``with_retry`` layer can retry transparently.

        Args:
            messages: Full conversation history in provider-agnostic Message format.
            system_instruction: System prompt text.
            tools: Available tool schemas.
            response_schema: Optional JSON Schema for structured output.
            cancel_token: Optional cancellation signal.
            on_chunk: If provided, enables streaming mode.
            on_usage_update: Real-time token usage callback (streaming).
            on_function_call: Callback when function call detected mid-stream.
            on_thinking: Callback for extended thinking content.
            tool_choice: Optional per-call ``tool_choice`` override.  When
                set, requests the model invoke a specific tool on this
                call (typically used by the session for one-shot retries
                after a lifecycle-tool ``validation_failed``).  Wire
                shape mirrors OpenAI / vLLM Chat Completions:
                ``{"type": "function", "function": {"name": <tool_name>}}``;
                Anthropic providers translate to
                ``{"type": "tool", "name": <tool_name>}``.  ``None``
                means provider default (``"auto"`` when tools are
                present).  Providers MAY ignore the kwarg when their
                wire-format quirk gate is off — the session passes it
                generically and lets each provider decide whether to
                honor it (vllm honors via
                ``force_tool_choice_for_lifecycle`` quirk; providers
                without the quirk no-op).

        Returns:
            A ``TurnResult`` classifying the outcome.
        """
        ...

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Args:
            content: Text to count tokens for.

        Returns:
            Token count.
        """
        ...

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        ...

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        ...

    # ==================== Serialization ====================
    # Used by SessionPlugin for persistence

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to a string.

        Used by SessionPlugin to persist conversations.
        The format should be provider-independent (e.g., JSON).

        Args:
            history: List of messages to serialize.

        Returns:
            Serialized string representation.
        """
        ...

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize conversation history from a string.

        Args:
            data: Previously serialized history string.

        Returns:
            List of Message objects.
        """
        ...

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if this provider supports structured output (response_schema).

        When True, the provider can accept response_schema in complete()
        to constrain the model's output to valid JSON matching the provided
        schema.

        Returns:
            True if structured output is supported.
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        When True, the provider supports the ``on_chunk`` callback in
        ``complete()`` for real-time token delivery.

        Returns:
            True if streaming is supported, False otherwise.
        """
        ...

    def supports_stop(self) -> bool:
        """Check if this provider supports mid-turn cancellation (stop).

        Stop capability requires streaming support since cancellation is
        implemented by breaking out of the streaming loop.

        Returns:
            True if stop/cancel is supported, False otherwise.
        """
        ...

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (default: the connected model) accepts.

        Returns a set of canonical lowercase tokens (``"text"`` at
        minimum, plus ``"image"`` / ``"audio"`` / … when the model and
        this provider's adapter support them).  Resolved per-model via
        :func:`resolve_modalities` (catalog detect → profile knob → static
        table); a gateway provider serves both vision and text-only
        models, so the answer is per-model.

        ``model`` lets callers ask about a model OTHER than the connected
        one — e.g. validating a ``vision`` tier's model before switching
        into it (V1 tiers are same-provider, so the tier's model lives on
        this provider).  When ``None``, answers for the active model.

        The default text-only floor is supplied by
        :class:`ModalityCapabilityMixin`; image-capable adapters override
        this.  Used by tier validation (a ``vision`` tier must map to an
        image-capable model) and the session content-boundary gate (an
        image ``Part`` must not be sent to a model that can't view it).

        Returns:
            The model's input-modality set (never empty — at least
            ``{"text"}``).
        """
        ...

    def supports_modality(self, kind: str, model: Optional[str] = None) -> bool:
        """Whether ``model`` (default: the connected model) accepts ``kind``.

        Convenience over :meth:`modalities`:
        ``kind in self.modalities(model)``.  Supplied by
        :class:`ModalityCapabilityMixin`.
        """
        ...

    # ==================== Agent Context ====================
    # Optional agent identification for tracing

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        This allows traces to identify which agent (main or subagent)
        produced the trace, useful for debugging multi-agent scenarios.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
            agent_id: Unique identifier for the agent instance.
        """
        ...

    # ==================== Thinking Mode ====================
    # Optional extended thinking/reasoning support

    def supports_thinking(self) -> bool:
        """Check if this provider/model supports extended thinking.

        Thinking mode enables extended reasoning capabilities:
        - Anthropic: Extended thinking with visible reasoning traces
        - Google Gemini: Thinking mode (Gemini 2.0+)

        Returns:
            True if thinking mode is supported, False otherwise.
        """
        ...

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set the thinking/reasoning mode configuration.

        Dynamically enables or disables extended thinking for subsequent
        API calls. Takes effect immediately for the next complete() call.

        Args:
            config: ThinkingConfig with enabled flag and budget.
                - enabled: Whether to use thinking mode
                - budget: Token budget for thinking (provider-specific)

        Note:
            If the provider/model doesn't support thinking, this is a no-op.
            Check supports_thinking() to verify capability first.
        """
        ...

    # ==================== Error Classification for Retry ====================
    # Optional methods for provider-specific error handling in retry logic

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Each provider knows its own error types and can provide precise
        classification. If not implemented, the global fallback in
        retry_utils.classify_error() is used.

        Args:
            exc: The exception to classify.

        Returns:
            Dict with keys:
                - transient: True if error is transient (should retry)
                - rate_limit: True if error is a rate limit (429)
                - infra: True if error is infrastructure (503, 500)
            Or None to use global fallback classification.

        Example implementation:
            def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
                if isinstance(exc, MyRateLimitError):
                    return {"transient": True, "rate_limit": True, "infra": False}
                if isinstance(exc, MyServerError):
                    return {"transient": True, "rate_limit": False, "infra": True}
                return None  # Use fallback
        """
        ...

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Many APIs include a Retry-After header or equivalent in their
        rate limit responses. Providers can extract this for better backoff.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.

        Example implementation:
            def get_retry_after(self, exc: Exception) -> Optional[float]:
                if hasattr(exc, 'retry_after'):
                    return float(exc.retry_after)
                return None
        """
        ...
