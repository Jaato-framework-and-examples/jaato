"""TypeSafe decision provider: Jev on TypeSafe's own API.

Jev (TypeSafe) is a decision model: it answers typed questions (``noul``,
``choice``, ``score``) about a state with calibrated probabilities and
writes no text.  This provider serves :meth:`TypeSafeProvider.decide`
against ``POST {base_url}/v1/systemone`` and refuses
:meth:`TypeSafeProvider.complete` (docs/design/decision-models.md §6.2).
The same model is reachable through OpenRouter (``openrouter``'s
``decide()``); this adapter is the vendor's own route, and the one TypeSafe
documents images and ``box`` on (§12, not built yet).

Binding (a ``decision_models:`` entry, never a tier)::

    decision_models:
      triage:
        provider: typesafe
        model: jev-latest
    plugin_configs:
      typesafe:
        api_key: pass://typesafe/key     # first; env vars are the fallback

The key: ``plugin_configs.typesafe.api_key`` (or ``ProviderConfig.api_key``),
then ``JAATO_TYPESAFE_API_KEY``, then ``TYPESAFE_API_KEY``, then the key
stored by ``typesafe-auth key``.  ``base_url`` and ``context_length`` read
their knob first and their ``JAATO_TYPESAFE_*`` variable second.  With no
window configured the pre-send size check is skipped, never guessed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionQuestionError,
    DecisionResult,
    build_decision_request,
    parse_decision_response,
)
from jaato_sdk.plugins.model_provider.types import CancelToken, TokenUsage

from .. import _decisions_http as _http
from ..base import (
    MODALITY_DECISIONS,
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    ProviderConfig,
)
from .auth import SYSTEMONE_PATH, get_credential_file_path, get_stored_api_key
from .env import (
    DEFAULT_BASE_URL,
    resolve_base_url,
    resolve_context_length,
    resolve_env_api_key_with_source,
)
from .errors import (
    AuthenticationError,
    CredentialsNotFoundError,
    DecisionRequestRejectedError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

#: TypeSafe's alias for the current Jev release.
DEFAULT_MODEL = "jev-latest"

#: Model ids the quick start and the Jev 1.14 post name.  Any other id is
#: passed through as written; this list only feeds ``list_models``.
KNOWN_MODELS = ("jev-latest", "jev-preview", "jev-1.13.0", "jev-1.14.0")

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 60.0


def _positive_float(value: Any, default: float) -> float:
    """A deadline knob: a number ``>= 0`` (``0`` = none), else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def resolve_key(config: Optional[ProviderConfig] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(key, where it came from)``, in the provider's order.

    The profile knob (``plugin_configs.typesafe.api_key``, or
    ``ProviderConfig.api_key``) first; the environment and the stored key
    are fallbacks.
    """
    extra = dict((config.extra if config else None) or {})
    knob = extra.get("api_key") or (config.api_key if config else None)
    if knob:
        return str(knob), "plugin_configs.typesafe.api_key"
    key, source = resolve_env_api_key_with_source()
    if key:
        return key, source
    stored = get_stored_api_key()
    if stored:
        return stored, get_credential_file_path() or "typesafe_auth.json"
    return None, None


class TypeSafeProvider(ModalityCapabilityMixin):
    """Decision-only provider for TypeSafe Jev.

    Lifecycle: ``initialize()`` resolves the key, API root, window and
    deadlines (nothing is contacted; a missing key is refused here);
    ``connect(model)`` records the model; ``decide()`` sends one request.
    ``complete()`` always raises :class:`DecisionModelOnlyError`.
    """

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._key_source: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL
        self._context_override: Optional[int] = None
        self._connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
        self._request_timeout: float = DEFAULT_REQUEST_TIMEOUT
        self._model_name: Optional[str] = None
        self._initialized = False
        self._last_usage = TokenUsage(reported=False)
        self._agent_type = "main"
        self._agent_name: Optional[str] = None
        self._agent_id = "main"
        self._errors = _http.DecisionHttpErrors(
            authentication=lambda text: AuthenticationError(
                original_error=text, source=self._key_source),
            not_found=lambda model, text: ModelNotFoundError(
                model=model, original_error=text),
            rate_limit=lambda retry_after, text: RateLimitError(
                retry_after=retry_after, original_error=text),
            infrastructure=lambda status, text: InfrastructureError(
                status_code=status, original_error=text),
            rejected=lambda status, model, body: DecisionRequestRejectedError(
                status_code=status, model=model, body=body),
        )

    # ------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        """Provider identifier, and the key under ``plugin_configs``."""
        return "typesafe"

    def set_agent_context(self, agent_type: str = "main",
                          agent_name: Optional[str] = None,
                          agent_id: str = "main") -> None:
        """Set agent context for trace identification."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        from jaato_server.shared.trace import provider_trace
        who = ("main" if self._agent_type == "main"
               else f"subagent:{self._agent_name or self._agent_id}")
        provider_trace(f"typesafe:{who}", msg)

    # ----------------------------------------------------------- lifecycle

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Resolve the configuration.  Contacts nothing.

        Raises:
            CredentialsNotFoundError: no key in any of the four places.
        """
        extra = dict((config.extra if config else None) or {})
        overrides = extra.get("framework_overrides") or {}
        self._api_key, self._key_source = resolve_key(config)
        if not self._api_key:
            raise CredentialsNotFoundError()
        base_url = extra.get("base_url") or resolve_base_url()
        self._base_url = str(base_url).rstrip("/")
        context = extra.get("context_length")
        self._context_override = (int(context) if context
                                  else resolve_context_length())
        self._connect_timeout = _positive_float(
            overrides.get("connect_timeout"), DEFAULT_CONNECT_TIMEOUT)
        self._request_timeout = _positive_float(
            overrides.get("request_timeout"), DEFAULT_REQUEST_TIMEOUT)
        self._initialized = True

    def verify_auth(self, allow_interactive: bool = False,
                    on_message: Any = None) -> bool:
        """Whether a key exists in the environment or the store.

        Works before ``initialize()`` and contacts nothing.  The profile
        knob is not visible here; ``initialize()`` reads it first.
        """
        key, source = resolve_env_api_key_with_source()
        if not key and get_stored_api_key():
            key, source = "stored", get_credential_file_path() or "typesafe_auth.json"
        if on_message:
            on_message(f"Found TypeSafe API key ({source})" if key else
                       "No TypeSafe API key in JAATO_TYPESAFE_API_KEY, "
                       "TYPESAFE_API_KEY or typesafe-auth; "
                       "plugin_configs.typesafe.api_key is read at initialize")
        return bool(key)

    def get_auth_info(self) -> str:
        """Where the key came from, for the status readout."""
        return f"API key from {self._key_source}" if self._key_source else "no API key"

    def shutdown(self) -> None:
        """Nothing is held between requests."""

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Record the model.  TypeSafe publishes no catalog to check it against."""
        self._model_name = model or DEFAULT_MODEL
        self._trace(f"CONNECT model={self._model_name} base_url={self._base_url} "
                    f"key_source={self._key_source} "
                    f"window={self.get_context_limit() or 'unchecked'}")

    @property
    def is_connected(self) -> bool:
        return self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """The ids TypeSafe's docs name; others are accepted as written."""
        return [m for m in KNOWN_MODELS if not prefix or m.startswith(prefix)]

    # ---------------------------------------------------------- modalities

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """Text only until images are built (design §12)."""
        return {MODALITY_TEXT}

    def output_modalities(self, model: Optional[str] = None) -> Set[str]:
        """Jev answers decisions and writes nothing else."""
        return {MODALITY_DECISIONS}

    # ---------------------------------------------------------- chat refusal

    def complete(self, messages: Any = None, system_instruction: Any = None,
                 tools: Any = None, **_kwargs: Any) -> Any:
        """Always raises: Jev cannot serve a chat turn.

        Raises:
            DecisionModelOnlyError: naming the model and pointing at ``decide``.
        """
        return self._refuse_chat()

    def serialize_history(self, history: Any) -> str:
        return self._refuse_chat()

    def deserialize_history(self, data: str) -> Any:
        return self._refuse_chat()

    def _refuse_chat(self) -> Any:
        """The one refusal every chat-shaped call makes."""
        raise DecisionModelOnlyError(self._model_name or DEFAULT_MODEL, self.name)

    def supports_structured_output(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False

    def supports_stop(self) -> bool:
        return False

    def supports_thinking(self) -> bool:
        return False

    def set_thinking_config(self, config: Any) -> None:
        """No reasoning to configure."""

    # ----------------------------------------------------------- accounting

    def count_tokens(self, content: str) -> int:
        """~4 characters a token, the framework's estimate."""
        return max(1, len(content) // 4)

    def get_context_limit(self) -> int:
        """The configured window, or ``0`` when none is (then unchecked)."""
        return self._context_override or 0

    def get_token_usage(self) -> TokenUsage:
        """Usage of the last decision (``reported=False`` when none came)."""
        return self._last_usage

    # ------------------------------------------------------------- retries

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """``429`` and server-side failures are transient; nothing else."""
        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}
        if isinstance(exc, InfrastructureError):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)
        return None

    # ------------------------------------------------------------ decisions

    def decide(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        *,
        model: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> DecisionResult:
        """Ask Jev typed questions about ``state``.

        Args:
            state: The text, object or array the questions are about.
            questions: ``{id: DecisionQuestion}``; ids come back as keys.
            model: The Jev release; defaults to the connected model.
            cancel_token: Stops the request in flight.

        Raises:
            DecisionQuestionError: A question is malformed, or the request
                is estimated over a configured window (nothing sent).
            DecisionResponseError: A question is left unanswered or
                answered in an unreadable shape.
            RuntimeError: ``initialize()`` has not run.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        model = model or self._model_name or DEFAULT_MODEL
        body = build_decision_request(state, questions, model)
        self._check_fits(body, model)
        self._trace(f"DECIDE model={model} questions={list(questions)}")
        raw = self._post(body, model, cancel_token)
        result = parse_decision_response(raw, questions, model)
        self._last_usage = result.usage
        self._trace(f"DECIDE_DONE model={result.model} answers={len(result.answers)} "
                    f"usage_reported={result.usage.reported}")
        return result

    def _check_fits(self, body: Dict[str, Any], model: str) -> None:
        """Refuse a request estimated over the configured window.

        The estimate is the state and the questions as JSON, at ~4
        characters a token.  No window configured: not checked.
        """
        limit = self.get_context_limit()
        if not limit:
            return
        text = json.dumps([body["state"], body["questions"]],
                          ensure_ascii=False, default=str)
        estimate = self.count_tokens(text)
        if estimate > limit:
            raise DecisionQuestionError(
                None,
                f"state and questions are ~{estimate} tokens, over the "
                f"{limit}-token window configured for {model!r}; shorten the state",
            )

    def _post(self, body: Dict[str, Any], model: str,
              cancel_token: Optional[CancelToken]) -> Dict[str, Any]:
        """One request to ``/v1/systemone``, retried by ``with_retry``."""
        from jaato_server.shared.retry_utils import with_retry

        url = self._base_url + SYSTEMONE_PATH
        headers = {"Authorization": f"Bearer {self._api_key}"}

        def send() -> Dict[str, Any]:
            return _http.post_decision(
                url, headers, body, model=model, errors=self._errors,
                connect_timeout=self._connect_timeout,
                request_timeout=self._request_timeout,
                cancel_token=cancel_token,
            )

        def on_retry(message: str, attempt: int, max_attempts: int, delay: float) -> None:
            self._trace(f"DECIDE_RETRY {attempt}/{max_attempts} in {delay:.1f}s: {message}")

        raw, _stats = with_retry(send, context="decide", on_retry=on_retry,
                                 cancel_token=cancel_token, provider=self)
        return raw


def create_provider() -> TypeSafeProvider:
    """Factory function for plugin discovery."""
    return TypeSafeProvider()
