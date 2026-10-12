"""Laya decision provider: ``laya-serve`` over HTTP, or the package in-process.

Laya (ConvAI Innovations, Apache 2.0) is a decision model: a BERT-family
encoder with a decision head that answers typed questions (``noul``,
``choice``, ``score``) about a state in one forward pass.  It writes no
text, so this provider serves :meth:`LayaProvider.decide` and refuses
:meth:`LayaProvider.complete` (docs/design/decision-models.md §3, §6.1).

Two modes, ``plugin_configs.laya.mode``:

``http`` (default)
    ``POST {host}/v1/systemone`` on a ``laya-serve`` process, the same
    request and response as TypeSafe Jev.  The transport is
    ``model_provider/_decisions_http.py``, shared with ``openrouter``.
    The weights live in the server, so the runner stays small.
``local``
    ``laya.load(model, subfolder=...)`` in the RUNNER process.  Torch and
    the weights then stay in that process after the session, so the
    first load notes a process retirement
    (``shared/process_retirement.py``) and the pool slot is closed at
    ``session.end`` rather than reused (#1565, design §9).

Binding (a ``decision_models:`` entry, never a tier)::

    decision_models:
      local:
        provider: laya
        model: convaiinnovations/laya
    plugin_configs:
      laya:
        host: http://gpu-box:8000

The window comes from the knob, then the environment, then a built-in
table of the published checkpoints (512 tokens English, 1,024
multilingual and typed-decisions).  A request estimated over it is
refused before anything is sent.  The vendor notes that the shipped
checkpoints are over-confident; jaato forwards the probabilities as given.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Set

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
from .env import (
    DEFAULT_HOST,
    resolve_api_key,
    resolve_context_length,
    resolve_host,
    resolve_mode,
)
from .errors import (
    AuthenticationError,
    DecisionRequestRejectedError,
    InfrastructureError,
    LayaNotInstalledError,
    ModelNotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

#: The published checkpoint, used when neither the binding nor the
#: environment names one.
DEFAULT_MODEL = "convaiinnovations/laya"

#: ``laya-serve``'s decision endpoint, relative to the host.
SYSTEMONE_PATH = "/v1/systemone"

MODES = ("http", "local")

#: ``(model, subfolder) -> window in tokens``, from the model card.  A
#: model the table does not know has an unknown window unless the knob
#: or ``JAATO_LAYA_CONTEXT_LENGTH`` says otherwise.
CONTEXT_WINDOWS: Dict[tuple, int] = {
    ("convaiinnovations/laya", None): 512,
    ("convaiinnovations/laya", "multilingual"): 1024,
    ("convaiinnovations/laya", "typed-decisions"): 1024,
    ("convaiinnovations/laya-multilingual", None): 1024,
    ("convaiinnovations/laya-typed-decisions", None): 1024,
}

#: What an in-process load leaves in the runner, reported at ``session.end``.
LOCAL_RETIRE_REASON = (
    "laya loaded a decision model into this process (mode: local); "
    "torch's heap is returned only by process exit"
)

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 60.0

_ERRORS = _http.DecisionHttpErrors(
    authentication=lambda text: AuthenticationError(original_error=text),
    not_found=lambda model, text: ModelNotFoundError(model=model, original_error=text),
    rate_limit=lambda retry_after, text: RateLimitError(
        retry_after=retry_after, original_error=text),
    infrastructure=lambda status, text: InfrastructureError(
        status_code=status, original_error=text),
    rejected=lambda status, model, body: DecisionRequestRejectedError(
        status_code=status, model=model, body=body),
)


def _positive_float(value: Any, default: float) -> float:
    """A deadline knob: a number ``>= 0`` (``0`` = none), else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def refuse_truncated(raw: Any, model: str) -> None:
    """Raise when Laya says it did not read the whole state or a question.

    Laya truncates a state over its window to one window and reports it in
    ``usage`` (``truncated``, ``state_tokens``, ``state_tokens_dropped``,
    ``truncated_questions``), both from ``laya-serve`` and in-process.  An
    answer about part of the state is not an answer about the state, so it
    is refused with the same error the pre-send window check raises.  A
    body without those keys (``LAYA_JEV_STRICT`` reduces ``usage`` to the
    two token counts) is not evidence either way, and passes.

    Raises:
        DecisionQuestionError: The state, or a question's head, was cut.
    """
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return
    cut_questions = usage.get("truncated_questions") or []
    if not usage.get("truncated") and not cut_questions:
        return
    detail = (f"{usage.get('state_tokens_dropped')} of "
              f"{usage.get('state_tokens')} state tokens dropped")
    if cut_questions:
        detail += f"; too little room for {', '.join(map(str, cut_questions))}"
    raise DecisionQuestionError(
        cut_questions[0] if len(cut_questions) == 1 else None,
        f"{model!r} truncated the request ({detail}); the answers did not "
        "see the whole state, so shorten it",
    )


class LayaProvider(ModalityCapabilityMixin):
    """Decision-only provider for Laya.

    Lifecycle: ``initialize()`` resolves the mode, host, key, subfolder,
    window and deadlines (nothing is contacted); ``connect(model)``
    records the model; ``decide()`` sends one request (``http``) or runs
    one forward pass (``local``, loading the model on first use).
    ``complete()`` always raises :class:`DecisionModelOnlyError`.
    """

    def __init__(self) -> None:
        self._mode: str = "http"
        self._host: str = DEFAULT_HOST
        self._api_key: Optional[str] = None
        self._subfolder: Optional[str] = None
        self._context_override: Optional[int] = None
        self._connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
        self._request_timeout: float = DEFAULT_REQUEST_TIMEOUT
        self._model_name: Optional[str] = None
        self._initialized = False
        self._last_usage = TokenUsage(reported=False)
        self._local_agent: Any = None
        self._local_lock = threading.Lock()
        self._agent_type = "main"
        self._agent_name: Optional[str] = None
        self._agent_id = "main"

    # ------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        """Provider identifier, and the key under ``plugin_configs``."""
        return "laya"

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
        provider_trace(f"laya:{who}", msg)

    # ----------------------------------------------------------- lifecycle

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Resolve the configuration.  Contacts nothing.

        Raises:
            ValueError: ``mode`` is neither ``http`` nor ``local``.
        """
        extra = dict((config.extra if config else None) or {})
        overrides = extra.get("framework_overrides") or {}
        mode = str(extra.get("mode") or resolve_mode() or "http").strip().lower()
        if mode not in MODES:
            raise ValueError(
                f"Laya provider: mode {mode!r} is not one of {', '.join(MODES)} "
                "(plugin_configs.laya.mode or JAATO_LAYA_MODE)."
            )
        self._mode = mode
        host = extra.get("host") or resolve_host() or DEFAULT_HOST
        self._host = str(host).rstrip("/")
        self._api_key = (
            extra.get("api_key") or (config.api_key if config else None)
            or resolve_api_key()
        )
        self._subfolder = extra.get("subfolder") or None
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
        """True: Laya needs no credential (a key is optional on ``laya-serve``).

        Works before ``initialize()``; touches no server.
        """
        if on_message:
            on_message(
                "Laya needs no credential"
                + (" (a bearer token is configured)" if resolve_api_key() else "")
            )
        return True

    def get_auth_info(self) -> str:
        """Where decisions are answered, for the status readout."""
        if self._mode == "local":
            return "in-process (laya package)"
        return f"laya-serve at {self._host}" + (", bearer" if self._api_key else "")

    def shutdown(self) -> None:
        """Drop the in-process model, if loaded.  The slot is still retired:
        the memory it held is not returned by dropping the reference."""
        self._local_agent = None

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Record the model.  No catalog exists to check it against."""
        self._model_name = model or DEFAULT_MODEL
        self._trace(f"CONNECT model={self._model_name} mode={self._mode} "
                    f"window={self.get_context_limit() or 'unknown'}")

    @property
    def is_connected(self) -> bool:
        return self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """The published checkpoints the window table knows."""
        names = sorted({model for model, _sub in CONTEXT_WINDOWS})
        return [n for n in names if not prefix or n.startswith(prefix)]

    # ---------------------------------------------------------- modalities

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """Laya reads text (a string, an object or an array)."""
        return {MODALITY_TEXT}

    def output_modalities(self, model: Optional[str] = None) -> Set[str]:
        """Laya answers decisions and writes nothing else."""
        return {MODALITY_DECISIONS}

    # ---------------------------------------------------------- chat refusal

    def complete(self, messages: Any = None, system_instruction: Any = None,
                 tools: Any = None, **_kwargs: Any) -> Any:
        """Always raises: Laya cannot serve a chat turn.

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
        """The window of the connected model, or ``0`` when unknown."""
        if self._context_override:
            return self._context_override
        return CONTEXT_WINDOWS.get((self._model_name, self._subfolder), 0)

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
        """Ask Laya typed questions about ``state``.

        Args:
            state: The text, object or array the questions are about.
            questions: ``{id: DecisionQuestion}``; ids come back as keys.
            model: The checkpoint; defaults to the connected model.
            cancel_token: Stops an HTTP request in flight.  An in-process
                forward pass (~33 ms) is checked before it starts and not
                interrupted.

        Raises:
            DecisionQuestionError: A question is malformed, or the request
                is estimated over the window (nothing sent), or Laya reports
                that it truncated the state (:func:`refuse_truncated`).
            DecisionResponseError: A question is left unanswered or
                answered in an unreadable shape.
            LayaNotInstalledError: ``mode: local`` without the package.
            RuntimeError: ``initialize()`` has not run.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        model = model or self._model_name or DEFAULT_MODEL
        body = build_decision_request(state, questions, model)
        self._check_fits(body, model)
        self._trace(f"DECIDE model={model} mode={self._mode} questions={list(questions)}")
        if self._mode == "local":
            raw = self._predict_local(body, model, cancel_token)
        else:
            raw = self._post(body, model, cancel_token)
        result = parse_decision_response(raw, questions, model)
        refuse_truncated(raw, model)
        self._last_usage = result.usage
        self._trace(f"DECIDE_DONE model={result.model} answers={len(result.answers)} "
                    f"usage_reported={result.usage.reported}")
        return result

    def _check_fits(self, body: Dict[str, Any], model: str) -> None:
        """Refuse a request estimated over the window.  Nothing is truncated.

        The estimate is the state and the questions as JSON, at ~4
        characters a token.  An unknown window is not checked.
        """
        limit = (self.get_context_limit() if model == self._model_name
                 else self._context_override
                 or CONTEXT_WINDOWS.get((model, self._subfolder), 0))
        if not limit:
            return
        text = json.dumps([body["state"], body["questions"]],
                          ensure_ascii=False, default=str)
        estimate = self.count_tokens(text)
        if estimate > limit:
            raise DecisionQuestionError(
                None,
                f"state and questions are ~{estimate} tokens, over the "
                f"{limit}-token window of {model!r}; shorten the state",
            )

    def _post(self, body: Dict[str, Any], model: str,
              cancel_token: Optional[CancelToken]) -> Dict[str, Any]:
        """One request to ``laya-serve``, retried by ``with_retry``."""
        from jaato_server.shared.retry_utils import with_retry

        url = self._host + SYSTEMONE_PATH
        headers = ({"Authorization": f"Bearer {self._api_key}"}
                   if self._api_key else {})

        def send() -> Dict[str, Any]:
            return _http.post_decision(
                url, headers, body, model=model, errors=_ERRORS,
                connect_timeout=self._connect_timeout,
                request_timeout=self._request_timeout,
                cancel_token=cancel_token,
            )

        def on_retry(message: str, attempt: int, max_attempts: int, delay: float) -> None:
            self._trace(f"DECIDE_RETRY {attempt}/{max_attempts} in {delay:.1f}s: {message}")

        raw, _stats = with_retry(send, context="decide", on_retry=on_retry,
                                 cancel_token=cancel_token, provider=self)
        return raw

    def _local_model(self, model: str) -> Any:
        """The in-process model, loaded once per provider.

        The retirement is noted BEFORE the load is attempted: a load that
        fails part-way has already imported torch.
        """
        with self._local_lock:
            if self._local_agent is not None:
                return self._local_agent
            from jaato_server.shared.process_retirement import note_process_retirement
            note_process_retirement(LOCAL_RETIRE_REASON)
            # The model card: a TensorFlow install can deadlock model
            # construction unless transformers is told not to use it.
            os.environ.setdefault("USE_TF", "0")
            # Imported by name: the package (and torch) is needed only in
            # this mode, so the import closure ``validate`` reads must not
            # report it missing for every laya profile.
            import importlib
            try:
                laya = importlib.import_module("laya")
            except ImportError as exc:
                raise LayaNotInstalledError(str(exc)) from exc
            kwargs = {"subfolder": self._subfolder} if self._subfolder else {}
            self._trace(f"LOCAL_LOAD model={model} {kwargs}")
            self._local_agent = laya.load(model, **kwargs)
            return self._local_agent

    def _predict_local(self, body: Dict[str, Any], model: str,
                       cancel_token: Optional[CancelToken]) -> Any:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        agent = self._local_model(model)
        return agent.predict(body["state"], body["questions"])


def create_provider() -> LayaProvider:
    """Factory function for plugin discovery."""
    return LayaProvider()
