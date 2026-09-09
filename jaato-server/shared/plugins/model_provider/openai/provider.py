"""Native OpenAI model provider (Chat Completions **and** the Responses API).

jaato already reaches OpenAI's models through OpenRouter and through nine
OpenAI-*compatible* gateways.  What it did not have was a first-party
integration: the endpoint OpenAI itself serves, with the auth, headers and
error taxonomy that belong to it, and — the part no compatible gateway
offers — the **Responses API**, which is where OpenAI ships first and which
it names as the successor to Chat Completions.

This provider is one plugin speaking two wires:

``api: chat`` (default)
    Chat Completions, on the shared :class:`OpenAICompatProvider` transport
    — the same streaming loop, error mapping and media handling nine other
    providers use.  Two things are raised above the shared defaults:
    ``WIRE_PDF_AS_FILE`` and ``WIRE_AUDIO_AS_INPUT_AUDIO``, because
    OpenAI's own endpoint carries PDFs (``file`` blocks) and audio input
    (``input_audio`` blocks) where the gateways sharing that transport do
    not.

``api: responses``
    The Responses API, via :class:`ResponsesTransport` — a different
    request shape, a different event stream and different usage field
    names.  See ``responses.py`` and ``responses_converters.py``.

Both wires share one client, one credential resolution, one error
taxonomy and one trace log, which is the point of putting them in one
provider rather than two.

**The context window must be configured.**  OpenAI's ``GET /v1/models``
serves bare entries — ``{id, object, created, owned_by}`` and nothing
about capacity — so there is no tier to auto-detect from, and no
per-model table is hardcoded here: GC sizes the whole history against
that number, and a stale table would silently truncate a session or
over-fill a request.  Set ``plugin_configs.openai.context_length`` (or
``JAATO_OPENAI_CONTEXT_LENGTH``); ``connect()`` fails loud without one,
per the project's no-fallback rule.  Input *modalities* are a different
question and do carry a table — a wrong entry withholds an image, it
does not corrupt the history — with the ``modalities`` knob above it.

Authentication (API key, Bearer token):
- ``JAATO_OPENAI_API_KEY``, the vendor's own ``OPENAI_API_KEY``, the
  ``plugin_configs.openai.api_key`` profile knob (which may carry a
  ``pass://`` URI), or the stored ``openai_auth.json``.  Keys are created
  at https://platform.openai.com/api-keys.

Environment variables:
    JAATO_OPENAI_API_KEY / OPENAI_API_KEY: API key
    JAATO_OPENAI_BASE_URL / OPENAI_BASE_URL: Endpoint override
    JAATO_OPENAI_MODEL: Default model name
    JAATO_OPENAI_CONTEXT_LENGTH: Context window (required in practice)
    JAATO_OPENAI_ORG_ID / JAATO_OPENAI_PROJECT_ID: Billing attribution
    JAATO_OPENAI_API: Wire selector ("chat" or "responses")
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    Message,
    ToolSchema,
    TurnResult,
)
from .converters import AUDIO_AS_INPUT_AUDIO, PDF_AS_FILE
from .env import (
    API_CHAT,
    API_RESPONSES,
    DEFAULT_BASE_URL,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_CONTEXT_LENGTH,
    ENV_VENDOR_API_KEY,
    VALID_APIS,
    get_checked_credential_locations,
    is_self_hosted,
    resolve_api,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_organization,
    resolve_project,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)
from .responses import RESPONSES_API_PARAMS, ResponsesTransport

#: INPUT modalities by model-id prefix, longest match first.
#:
#: A table rather than a detected value because OpenAI's catalog reports
#: no modality field, and unlike the context window a wrong entry here is
#: recoverable: it withholds an attachment (visibly, with a note) rather
#: than mis-sizing the history.  ``plugin_configs.openai.modalities``
#: overrides it for a model the table has not caught up with.
#:
#: ``file`` is the PDF/document modality (OpenRouter's term, adopted
#: framework-wide); ``audio`` is claimed only by the audio-input models,
#: because a text model handed ``input_audio`` answers with a 400.
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    "gpt-4o-audio": frozenset({"text", "audio"}),
    "gpt-4o-mini-audio": frozenset({"text", "audio"}),
    "gpt-audio": frozenset({"text", "audio"}),
    "gpt-realtime": frozenset({"text", "audio"}),
    "gpt-4o": frozenset({"text", "image", "file"}),
    "gpt-4.1": frozenset({"text", "image", "file"}),
    "gpt-4-turbo": frozenset({"text", "image"}),
    "gpt-5": frozenset({"text", "image", "file"}),
    "o1": frozenset({"text", "image"}),
    "o3": frozenset({"text", "image"}),
    "o4": frozenset({"text", "image"}),
}

#: Model families that produce reasoning.  Only meaningful on the
#: Responses wire: Chat Completions returns no reasoning text for these
#: models at all (their thinking is billed and discarded), which is why
#: :meth:`OpenAIProvider.supports_thinking` also requires that wire.
REASONING_CAPABLE_MODELS = ["o1", "o3", "o4", "gpt-5"]


class OpenAIProvider(ResponsesTransport, OpenAICompatProvider):
    """Native OpenAI provider, speaking Chat Completions or Responses.

    Usage::

        provider = OpenAIProvider()
        provider.initialize(ProviderConfig(
            api_key='sk-...',
            extra={'context_length': 400000},
        ))
        provider.connect('gpt-5.1')
        result = provider.complete(messages, system_instruction="Be helpful.")

    The wire is chosen by ``plugin_configs.openai.api`` (``"chat"`` or
    ``"responses"``), or by ``JAATO_OPENAI_API``; the profile knob wins.
    """

    # Parameterize the base's shared error mapping with OpenAI's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    # OpenAI's own wire carries more than the gateways sharing the
    # transport do.  Declared from ``converters.py`` so the policy, the
    # class attributes and ``PROVIDER_CAPABILITIES`` have one source.
    WIRE_PDF_AS_FILE = PDF_AS_FILE
    WIRE_AUDIO_AS_INPUT_AUDIO = AUDIO_AS_INPUT_AUDIO

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        self._api_mode: str = API_CHAT
        self._organization: Optional[str] = None
        self._project: Optional[str] = None
        # Manual tiers; there is no catalog tier for either of these.
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        # Cached ``GET /v1/models`` ids, for list_models().
        self._catalog_cache: Optional[List[str]] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "openai"

    @property
    def api_mode(self) -> str:
        """Which wire this session speaks: ``"chat"`` or ``"responses"``."""
        return self._api_mode

    # ==================== Credential / config hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key, base URL and billing headers; validate.

        Pulls workspace_path / config_root from ``config.extra`` so
        credential lookup resolves under the session's explicit
        config_root rather than the unreliable ``JAATO_CONFIG_ROOT`` env
        var for headless sessions.

        A missing key is an error unless the endpoint is a local proxy
        (:func:`is_self_hosted`) — never a silent fallback.
        """
        extra = config.extra or {}
        self._api_key = config.api_key or extra.get("api_key") or resolve_api_key(
            workspace_path=extra.get("workspace_path"),
            config_root=extra.get("config_root"),
        )
        self._base_url = extra.get("base_url") or resolve_base_url()
        self._organization = extra.get("organization") or resolve_organization()
        self._project = extra.get("project") or resolve_project()

        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

    def _resolve_api_mode(self, config: ProviderConfig) -> str:
        """Resolve the wire selector: profile knob → env → ``chat``.

        Raises:
            ValueError: for an unrecognised selector.  Silently falling
                back to Chat Completions would answer a profile that
                asked for the Responses API with a session that never
                touches it, and the difference is invisible from outside.
        """
        raw = (config.extra or {}).get("api") or resolve_api() or API_CHAT
        mode = str(raw).strip().lower()
        if mode not in VALID_APIS:
            raise ValueError(
                f"openai 'api' must be one of {list(VALID_APIS)}, got {raw!r}. "
                f"Set plugin_configs.openai.api (or JAATO_OPENAI_API)."
            )
        return mode

    def _read_api_params(self, config: ProviderConfig) -> None:
        """Resolve the wire, then read ``api_params`` against ITS allow-list.

        The two wires disagree about the name of the most-used knob
        (``max_output_tokens`` vs ``max_tokens``), so one shared allow-list
        would forward a field the endpoint rejects and suppress the
        unsupported-key warning that makes the mistake visible.  The
        instance attribute shadows the class one for this session only.
        """
        self._api_mode = self._resolve_api_mode(config)
        if self._api_mode == API_RESPONSES:
            self._FORWARDED_API_PARAMS = RESPONSES_API_PARAMS
        super()._read_api_params(config)
        self._trace(f"[INIT] api={self._api_mode}")

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Stash the manual tiers; the window is resolved at ``connect()``.

        Deferred so the fail-loud message can name the model that has no
        window configured, which is the thing an operator has to look up.
        """
        self._context_length_knob = (
            (config.extra or {}).get("context_length") or resolve_context_length()
        )
        modalities_override = (config.extra or {}).get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "openai 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    def _create_client(self) -> Any:
        """Create the OpenAI client, carrying the billing headers.

        ``organization`` / ``project`` are constructor arguments rather
        than raw headers because the SDK also uses them when it renders
        error messages, and a project-scoped key with no project header is
        one of the two ways a valid key reads as invalid.
        """
        from .._openai_compat._lazy import get_openai_client_class
        client_class = get_openai_client_class()
        kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "api_key": self._api_key or "not-needed",
        }
        if self._organization:
            kwargs["organization"] = self._organization
        if self._project:
            kwargs["project"] = self._project
        return client_class(**kwargs)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and resolve its context window.

        Resolution is profile knob → ``JAATO_OPENAI_CONTEXT_LENGTH`` →
        fail loud.  There is deliberately no auto-detect tier: OpenAI's
        catalog reports no capacity for any model, so a ``detect`` hook
        here would be a function that always returns ``None`` pretending
        to be a source of truth.
        """
        self._model_name = model
        self._context_length = resolve_context_window(
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                f"OpenAI provider: context_length is not configured for model "
                f"{model!r}.  OpenAI's GET /v1/models reports no context "
                f"window for any model, so it cannot be detected.  Set "
                f"plugin_configs.openai.context_length in the profile, or "
                f"{ENV_OPENAI_CONTEXT_LENGTH} in the environment (per-model "
                f"windows are listed at https://platform.openai.com/docs/models"
                f").  No hardcoded fallback exists per the project's "
                f"no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} api={self._api_mode} "
            f"context_length={self._context_length}"
        )

    # ==================== Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[Any] = None,
        on_usage_update: Optional[Any] = None,
        on_function_call: Optional[Any] = None,
        on_thinking: Optional[Any] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Stateless completion on whichever wire this session selected.

        ``chat`` delegates to the shared OpenAI-compat implementation
        unchanged.  ``responses`` runs the Responses transport and then
        performs the same three post-steps the shared path does: parse
        structured output, record per-call usage, wrap in a
        ``TurnResult``.

        Raises:
            RuntimeError: If the provider is not initialized/connected.
        """
        if self._api_mode != API_RESPONSES:
            return super().complete(
                messages, system_instruction, tools,
                response_schema=response_schema,
                cancel_token=cancel_token,
                on_chunk=on_chunk,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
                on_thinking=on_thinking,
                tool_choice=tool_choice,
            )

        if not self._client or not self._model_name:
            raise RuntimeError(
                "Provider not connected. Call initialize() and connect() first."
            )
        try:
            response = self._complete_via_responses(
                messages, system_instruction, tools,
                response_schema=response_schema,
                cancel_token=cancel_token,
                on_chunk=on_chunk,
                on_usage_update=on_usage_update,
                on_thinking=on_thinking,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            self._handle_api_error(exc)
            raise

        self._last_usage = response.usage
        if response_schema:
            _parse_structured_output(response)
        return TurnResult.from_provider_response(response)

    # ==================== Catalog / capabilities ====================

    def _fetch_catalog(self) -> List[str]:
        """Fetch and cache the model ids from ``GET /v1/models``.

        Authenticated and account-scoped: the listing is what *this* key
        can reach, which is the only useful answer (model access is
        granted per organization).  Returns an empty list — not cached —
        on failure, so the next call retries.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        if self._project:
            headers["OpenAI-Project"] = self._project
        try:
            response = httpx.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._trace(f"[CATALOG] fetch failed: {type(exc).__name__}: {exc}")
            return []

        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            self._trace("[CATALOG] response missing 'data' list")
            return []

        self._catalog_cache = [
            e["id"] for e in entries
            if isinstance(e, dict) and isinstance(e.get("id"), str)
        ]
        return self._catalog_cache

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List the model ids this key can reach, sorted alphabetically."""
        ids = self._fetch_catalog()
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    def _lookup_input_modalities(
        self, model: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Table-declared input modalities for ``model`` (or the active one).

        Exact id first, then the longest matching prefix — ``gpt-4o-audio``
        must beat ``gpt-4o``, and a shortest-match walk would give the
        audio model a vision profile and no ears.
        """
        model = model or self._model_name
        if not model:
            return None
        if model in MODEL_INPUT_MODALITIES:
            return MODEL_INPUT_MODALITIES[model]
        matches = [p for p in MODEL_INPUT_MODALITIES if model.startswith(p)]
        if not matches:
            return None
        return MODEL_INPUT_MODALITIES[max(matches, key=len)]

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (default: the active model) accepts.

        ``modalities`` knob → :data:`MODEL_INPUT_MODALITIES` → text floor.
        """
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=self._lookup_input_modalities(model),
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def supports_thinking(self) -> bool:
        """Reasoning text is available on the Responses wire only.

        The reasoning families think on both wires and are billed for it
        on both; Chat Completions simply never returns any of it, so
        claiming support there would promise a channel that is always
        empty.
        """
        return self._api_mode == API_RESPONSES and self._is_reasoning_capable()

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key without
        ever touching ``self._client``.  A profile-supplied ``api_key``
        (the daemon expands ``pass://`` secrets into the verify-time
        ``ProviderConfig``) takes effect during this pre-init gate.  A
        stored credential file that exists but cannot be loaded surfaces
        its load error via ``on_message``.

        Raises:
            APIKeyNotFoundError: If no key is found and the endpoint is
                not a local proxy (unless ``allow_interactive``).
        """
        if _profile_api_key(config):
            _say(on_message, "Found OpenAI API key (profile config)")
            return True

        env_var = _env_api_key_var()
        if env_var:
            _say(on_message, f"Found OpenAI API key ({env_var})")
            return True

        if _stored_api_key_found(on_message):
            return True

        base_url = resolve_base_url()
        if is_self_hosted(base_url):
            _say(on_message,
                 f"Local OpenAI-compatible proxy ({base_url}), "
                 "no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config)
            )
        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os

        if is_self_hosted(self._base_url):
            return f"Local OpenAI-compatible proxy ({self._base_url})"

        for var in (ENV_OPENAI_API_KEY, ENV_VENDOR_API_KEY):
            if os.environ.get(var):
                return f"OpenAI API key ({var})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, "extra", None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get("workspace_path"),
                config_root=extra.get("config_root"),
            )
            if cred_path:
                return f"OpenAI API key ({cred_path})"
        except ImportError:
            pass

        return "OpenAI API key"


def _say(on_message, text: str) -> None:
    """Report a credential-search step, when the caller wants to hear it."""
    if on_message:
        on_message(text)


def _profile_api_key(config: Optional[ProviderConfig]) -> Optional[str]:
    """The profile's key, from either place the daemon may have put it.

    ``pass://`` secrets are expanded into the verify-time ``ProviderConfig``
    by the daemon, so this is a real source at the pre-init gate and not
    merely a later override.
    """
    if config is None:
        return None
    return config.api_key or (config.extra or {}).get("api_key")


def _env_api_key_var() -> Optional[str]:
    """The name of the env var holding a key, or ``None``.

    The NAME rather than the value: the caller reports which variable was
    used, and a credential must not pass through a log line to do it.
    """
    import os
    for var in (ENV_OPENAI_API_KEY, ENV_VENDOR_API_KEY):
        if os.environ.get(var):
            return var
    return None


def _stored_api_key_found(on_message) -> bool:
    """Whether the credential FILE yielded a key, reporting a broken one.

    "Present but corrupt" and "not configured" are different problems with
    different fixes, so a file that exists and will not load says so
    instead of being reported as an absence.
    """
    from .auth import try_load_credentials_with_reason

    credentials, load_error = try_load_credentials_with_reason()
    if credentials and credentials.api_key:
        _say(on_message, "Found OpenAI API key (stored credentials)")
        return True
    if load_error:
        _say(on_message,
             f"OpenAI credentials file found but could not be loaded: "
             f"{load_error}")
        _say(on_message,
             f"Repair or delete that file, or set {ENV_OPENAI_API_KEY} "
             f"instead.")
    return False


def _parse_structured_output(response) -> None:
    """Populate ``structured_output`` from the response text, if it parses.

    A model that answered prose when a schema was requested leaves the
    field ``None`` rather than raising: the caller sees the text and can
    say so, which is more useful than an exception with the answer
    discarded.
    """
    import json

    text = response.get_text()
    if not text:
        return
    try:
        response.structured_output = json.loads(text)
    except json.JSONDecodeError:
        pass


def create_provider() -> OpenAIProvider:
    """Factory function for plugin discovery.

    Returns:
        A new OpenAIProvider instance.
    """
    return OpenAIProvider()
