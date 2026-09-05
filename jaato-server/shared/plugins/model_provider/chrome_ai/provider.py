"""Chrome built-in AI (Gemini Nano) model provider.

Drives the on-device model embedded in Google Chrome (and Microsoft Edge)
through the built-in AI **Prompt API** (``LanguageModel`` global), bridged
over the Chrome DevTools Protocol.  See the package ``__init__`` for the
user-facing overview and knobs.

Implementation notes:

- **Stateless per turn.**  Prompt API sessions are stateful with a tiny
  quota, but the ``ModelProviderPlugin`` contract makes the jaato session
  own history.  Each ``complete()`` therefore builds a fresh page-side
  session from ``initialPrompts`` (system + history) and destroys it after
  the turn.
- **Tool calling is prompt-injected** (``tooling.py``): the Prompt API's
  native ``tools`` option is not on stable Chrome and executes callbacks
  page-side, which inverts jaato's execution model.
- **Structured output is native**: ``response_schema`` maps directly to
  the Prompt API's ``responseConstraint`` (JSON Schema), one of the few
  places this small model is genuinely strong.
- **Context window is detected** from the session quota
  (``inputQuota``/``contextWindow``) via ``resolve_context_window``; the
  window is tiny (~6-9k tokens shared input+output), so aggressive GC is
  expected upstream.
- **Token accounting is estimated** (~4 chars/token): the API reports no
  token counts; ``inputUsage`` (whole-session usage, when reported) is
  used for ``total_tokens``.
"""

import json
import logging
import queue
import time
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ThinkingConfig,
    TokenUsage,
    ToolSchema,
    TurnResult,
)

from ..base import (
    ModalityCapabilityMixin,
    ProviderConfig,
    resolve_context_window,
)
from shared.tool_id_map import id_to_name

from .bridge import PromptApiBridge
from .converters import (
    deserialize_message,
    messages_to_prompt_api,
    serialize_message,
    tool_schemas_to_prompt,
)
from .env import (
    DEFAULT_MODEL,
    DEFAULT_PAGE_URL,
    default_user_data_dir,
    find_browser_binary,
    resolve_cdp_url,
    resolve_context_length,
    resolve_headless,
    resolve_page_url,
    resolve_user_data_dir,
)
from shared.cdp import CDPConnectionError
from .errors import (
    ChromeAIBinaryNotFoundError,
    ChromeAIConnectionError,
    ChromeAIResponseError,
    ChromeAIUnavailableError,
)
from .tooling import parse_tool_calls

logger = logging.getLogger(__name__)

#: Default seconds of event silence tolerated mid-turn before the turn is
#: declared dead.  Local inference emits chunks continuously; a long gap
#: means the page or model hung.
DEFAULT_TURN_TIMEOUT = 300.0

_ENABLE_INSTRUCTIONS = (
    "The built-in AI Prompt API (the 'LanguageModel' global) is not exposed "
    "on the page this provider drives. Requirements: Google Chrome 138+ "
    "(extension contexts) / 148+ (web pages), or recent Microsoft Edge — "
    "plain Chromium builds are NOT supported (the on-device model is "
    "Google-proprietary). On older/gated builds enable "
    "chrome://flags/#prompt-api-for-gemini-nano and "
    "chrome://flags/#optimization-guide-on-device-model "
    "('Enabled BypassPerfRequirement'), then restart the browser. If the "
    "API is exposed only to real origins on your build, point the "
    "'page_url' knob (JAATO_CHROME_AI_PAGE_URL) at an https page."
)

_DOWNLOAD_INSTRUCTIONS = (
    "The Gemini Nano on-device model is not downloaded into this browser "
    "profile yet. Either set plugin_configs.chrome_ai.auto_download: true "
    "to let the provider trigger the component download (several GB; "
    "needs ~22 GB free disk and an unmetered connection), or open the "
    "same profile in Chrome manually and run "
    "`await LanguageModel.create()` once in DevTools to download it."
)


class ChromeAIProvider(ModalityCapabilityMixin):
    """Model provider backed by Chrome's built-in on-device model.

    Lifecycle: ``initialize(config)`` reads knobs (no I/O) →
    ``connect(model)`` launches/attaches the browser, verifies model
    availability, and resolves the context window → ``complete()`` per
    turn → ``shutdown()`` tears the browser down.  ``verify_auth()`` is
    credential-free (local browser) and safe on an uninitialized
    instance, per the provider contract.
    """

    def __init__(self, bridge_factory: Optional[Callable[..., Any]] = None):
        """Construct an unconnected provider.

        Args:
            bridge_factory: Test seam — a callable with
                :class:`PromptApiBridge`'s constructor signature returning
                a bridge-compatible object.  Defaults to the real bridge.
        """
        self._bridge_factory = bridge_factory or PromptApiBridge
        self._bridge: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._connected = False
        self._context_limit: int = 0
        self._last_usage = TokenUsage()
        self._agent_id = "main"

        # Knobs (populated in initialize()).
        self._binary_knob: Optional[str] = None
        self._cdp_url: Optional[str] = None
        self._user_data_dir: Optional[str] = None
        self._headless: bool = True
        self._page_url: str = DEFAULT_PAGE_URL
        self._reuse_page: bool = False
        self._extra_args: List[str] = []
        self._auto_download: bool = False
        self._download_timeout: float = 1800.0
        self._connect_timeout: float = 30.0
        self._turn_timeout: float = DEFAULT_TURN_TIMEOUT
        self._context_length_knob: Optional[int] = None
        self._temperature: Optional[float] = None
        self._top_k: Optional[int] = None
        # Warmup knob (default ON): run one throwaway generation at the end
        # of connect() so the on-device model's cold-start compile/load
        # (~11s to first token, measured on Chrome 149 / Gemini Nano) lands
        # in setup rather than on the caller's first real turn — which also
        # eliminates the empty first-response the cold window produces.
        self._warmup: bool = True

    @property
    def name(self) -> str:
        """Provider identifier (= ``plugin_configs`` key, ``provider:`` value)."""
        return "chrome_ai"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Read knobs from ``config.extra`` (the merged ``plugin_configs.chrome_ai``).

        Performs no I/O — the browser is launched lazily in ``connect()``.
        Knob precedence for values that also have env vars is applied at
        the read site in ``connect()`` (knob → env → default).
        """
        extra = (config.extra if config else None) or {}
        self._binary_knob = extra.get("binary")
        self._cdp_url = extra.get("cdp_url") or resolve_cdp_url()
        self._user_data_dir = (extra.get("user_data_dir")
                               or resolve_user_data_dir()
                               or default_user_data_dir())
        headless = extra.get("headless")
        if headless is None:
            headless = resolve_headless()
        self._headless = True if headless is None else bool(headless)
        self._page_url = (extra.get("page_url") or resolve_page_url()
                          or DEFAULT_PAGE_URL)
        # Anchor onto a pre-existing tab whose URL == page_url (attach, and
        # leave it open on teardown) instead of creating — and closing — a
        # dedicated tab.  Lets a caller point the provider at a real https
        # tab the user already has open; falls back to creating a tab when
        # none matches.  Default off = the historical create-a-tab behavior.
        self._reuse_page = bool(extra.get("reuse_page", False))
        self._extra_args = list(extra.get("extra_args") or [])
        self._auto_download = bool(extra.get("auto_download", False))
        self._download_timeout = float(extra.get("download_timeout", 1800))
        self._connect_timeout = float(extra.get("connect_timeout", 30))
        self._turn_timeout = float(extra.get("turn_timeout",
                                             DEFAULT_TURN_TIMEOUT))
        self._context_length_knob = extra.get("context_length")
        self._warmup = bool(extra.get("warmup", True))
        api_params = extra.get("api_params") or {}
        self._temperature = api_params.get("temperature")
        self._top_k = api_params.get("top_k")

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Check that a browser is reachable — no credentials exist.

        Safe on a fresh, uninitialized instance (touches no state set by
        ``initialize()``): it only checks that either a CDP endpoint is
        configured or a Chrome/Edge binary can be located.
        """
        extra = (config.extra if config else None) or {}
        if extra.get("cdp_url") or resolve_cdp_url():
            if on_message:
                on_message("chrome_ai: attaching to an existing browser "
                           "(cdp_url configured); no credentials required")
            return True
        binary = find_browser_binary(extra.get("binary"))
        if binary:
            if on_message:
                on_message(f"chrome_ai: found browser binary at {binary}; "
                           "no credentials required")
            return True
        if on_message:
            on_message(
                "chrome_ai: no Google Chrome / Microsoft Edge binary found. "
                "Install Chrome, or set plugin_configs.chrome_ai.binary / "
                "JAATO_CHROME_AI_BINARY, or point cdp_url / "
                "JAATO_CHROME_AI_CDP_URL at a running browser.")
        return False

    def shutdown(self) -> None:
        """Close the page, the CDP link, and any provider-owned browser."""
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:
                logger.exception("chrome_ai: error during bridge shutdown")
        self._bridge = None
        self._connected = False

    def get_auth_info(self) -> str:
        """Describe the (absent) credential source."""
        return "none (local browser via Chrome DevTools Protocol)"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Start the browser bridge and verify the on-device model.

        Args:
            model: Nominal model name (the Prompt API has no model
                selection; ``"gemini-nano"`` by convention).
            skip_model_test: Skip the availability check and quota probe.
                The context window must then resolve from the
                ``context_length`` knob / ``JAATO_CHROME_AI_CONTEXT_LENGTH``
                or ``connect()`` fails loud (no hardcoded fallback).

        Raises:
            ChromeAIBinaryNotFoundError: no browser to launch/attach.
            ChromeAIUnavailableError: API missing, model unavailable, or
                model not downloaded (with state-specific instructions).
            ChromeAIConnectionError: launch/CDP failures.
        """
        self._start_bridge()
        if not skip_model_test:
            self._verify_model_available()
        detect = (self._bridge.probe_quota if not skip_model_test
                  else lambda: None)
        limit = resolve_context_window(
            detect_capacity=detect,
            profile_value=self._context_length_knob,
            env_value=resolve_context_length(),
        )
        if not limit:
            raise ChromeAIUnavailableError(
                "could not determine the model's context window: the "
                "Prompt API session reported no quota and no manual value "
                "is set. Set plugin_configs.chrome_ai.context_length or "
                "JAATO_CHROME_AI_CONTEXT_LENGTH.")
        self._context_limit = int(limit)
        self._model_name = model or DEFAULT_MODEL
        self._connected = True
        logger.info("chrome_ai: connected (model=%s, context=%d tokens)",
                    self._model_name, self._context_limit)
        if self._warmup and not skip_model_test:
            self._warmup_model()

    def _warmup_model(self) -> None:
        """Absorb the on-device model's cold-start on connect(), not turn 1.

        The first inference after the model is provisioned pays a one-time
        compile/load cost (~11s to first token on Chrome 149 / Gemini Nano)
        and can return an empty completion. Running one throwaway generation
        here moves that cost off the caller's first real turn and discards
        the empty-first artifact. Best-effort: a warmup failure never fails
        connect() (the model is already verified available) — it just logs,
        and the cost falls back onto turn 1. Skipped when ``warmup: false``
        or ``skip_model_test`` (expert path). ``_last_usage`` is reset so the
        warmup's tokens never surface via ``get_token_usage()``.
        """
        try:
            self._bridge.warmup(timeout=self._turn_timeout)
            logger.info("chrome_ai: model warmup complete")
        except Exception:
            logger.warning("chrome_ai: model warmup failed (non-fatal; the "
                           "cold-start cost will fall on the first turn)",
                           exc_info=True)
        finally:
            self._last_usage = TokenUsage()

    @property
    def is_connected(self) -> bool:
        """True after a successful ``connect()`` (until ``shutdown()``)."""
        return self._connected

    @property
    def model_name(self) -> Optional[str]:
        """The connected nominal model name, or None before ``connect()``."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """The single on-device model the Prompt API exposes."""
        models = [DEFAULT_MODEL]
        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return models

    # ==================== Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_usage_update: Optional[Callable[[TokenUsage], None]] = None,
        on_function_call: Optional[Callable[[FunctionCall], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Run one stateless generation turn against the Prompt API.

        Builds a fresh page-side session from ``system_instruction`` (plus
        the prompt-injected tool section when ``tools`` are given) and the
        full ``messages`` history, streams the response, then parses
        ``tool_call`` fenced blocks back into :class:`FunctionCall` parts.

        ``tool_choice`` is ignored (``tool_choice_forwarding=False`` — the
        Prompt API has no such control).  ``on_thinking`` never fires (no
        reasoning mode).

        Returns ``TurnResult.from_provider_response`` on success or
        cancellation; ``TurnResult.from_exception`` for non-transient
        generation errors.  Transport errors (browser/CDP death, turn
        timeout) RAISE :class:`ChromeAIConnectionError` so the framework
        retry layer re-enters — where ``_ensure_connected()`` relaunches
        the browser.
        """
        if cancel_token is not None and cancel_token.is_cancelled:
            return TurnResult.cancelled(context="before chrome_ai turn")
        self._ensure_connected()

        payload = self._build_payload(messages, system_instruction, tools,
                                      response_schema)
        run_id = self._bridge.start_turn(payload)
        events = self._bridge.events(run_id)
        if cancel_token is not None:
            cancel_token.on_cancel(lambda: self._bridge.abort(run_id))

        collected: List[str] = []
        final: Optional[Dict[str, Any]] = None
        try:
            deadline = time.monotonic() + self._turn_timeout
            while final is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._bridge.abort(run_id)
                    raise ChromeAIConnectionError(
                        f"turn produced no events for "
                        f"{self._turn_timeout:.0f}s; aborting")
                try:
                    event = events.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    if not self._bridge.is_alive:
                        raise ChromeAIConnectionError(
                            "browser died mid-turn")
                    continue
                deadline = time.monotonic() + self._turn_timeout
                etype = event.get("type")
                if etype == "chunk":
                    text = event.get("text") or ""
                    collected.append(text)
                    if on_chunk and text:
                        on_chunk(text)
                elif etype in ("done", "error", "cancelled"):
                    final = event
        finally:
            self._bridge.finish_turn(run_id)

        partial_text = "".join(collected)
        if final.get("type") == "cancelled":
            self._last_usage = self._estimate_usage(payload, partial_text, None)
            response = ProviderResponse(
                parts=[Part.from_text(partial_text)] if partial_text else [],
                usage=self._last_usage,
                finish_reason=FinishReason.CANCELLED,
            )
            return TurnResult.from_provider_response(response)
        if final.get("type") == "error":
            message = final.get("message") or "Prompt API generation failed"
            return TurnResult.from_exception(
                ChromeAIResponseError(message),
                error_message=f"chrome_ai generation failed: {message}")

        text = final.get("text") or partial_text
        clean_text, raw_calls = (parse_tool_calls(text) if tools
                                 else (text, []))
        calls: List[FunctionCall] = []
        for index, (wire_id, args) in enumerate(raw_calls):
            # The model calls tools by hashed wire id; the framework
            # (executor, permissions, UI) speaks human names.  Unknown /
            # hallucinated ids pass through unchanged and surface as a
            # structured unknown-tool error the model can recover from.
            call = FunctionCall(id=f"chrome_ai-{run_id[:8]}-{index}",
                                name=id_to_name(wire_id), args=args)
            calls.append(call)
            if on_function_call:
                on_function_call(call)

        self._last_usage = self._estimate_usage(payload, text,
                                                final.get("usage"))
        if on_usage_update:
            on_usage_update(self._last_usage)

        parts: List[Part] = []
        if clean_text:
            parts.append(Part.from_text(clean_text))
        parts.extend(Part.from_function_call(c) for c in calls)

        structured: Optional[Any] = None
        if response_schema and not calls:
            try:
                structured = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                logger.warning("chrome_ai: responseConstraint was set but "
                               "the output is not valid JSON")

        response = ProviderResponse(
            parts=parts,
            usage=self._last_usage,
            finish_reason=(FinishReason.TOOL_USE if calls
                           else FinishReason.STOP),
            structured_output=structured,
        )
        return TurnResult.from_provider_response(response)

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """~4-chars/token estimate — the Prompt API exposes no counter.

        (``session.measureInputUsage()`` exists page-side but a CDP round
        trip per accounting call is not worth it; the estimate matches
        ``claude_cli``'s approach.)
        """
        if not content:
            return 0
        return max(1, len(content) // 4)

    def get_context_limit(self) -> int:
        """The resolved context window (session quota / knob / env)."""
        return self._context_limit

    def get_token_usage(self) -> TokenUsage:
        """Usage estimated for the last ``complete()`` call."""
        return self._last_usage

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize history to JSON (standard Message round-trip)."""
        return json.dumps([serialize_message(m) for m in history])

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize a JSON string back to a list of Messages."""
        return [deserialize_message(m) for m in json.loads(data)]

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Yes — ``response_schema`` maps to ``responseConstraint``."""
        return True

    def supports_streaming(self) -> bool:
        """Yes — ``promptStreaming`` chunks flow through ``on_chunk``."""
        return True

    def supports_stop(self) -> bool:
        """Yes — the cancel token fires the run's AbortController."""
        return True

    def capabilities(self):
        """The declared wire capabilities (see package ``__init__``)."""
        from . import PROVIDER_CAPABILITIES
        return PROVIDER_CAPABILITIES

    # ==================== Agent Context / Thinking ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Record the owning agent id (used only for log correlation)."""
        self._agent_id = agent_id

    def supports_thinking(self) -> bool:
        """No reasoning mode in the Prompt API."""
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is unsupported."""
        return None

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Mark bridge/transport failures as transient infra errors.

        A dead browser or dropped CDP socket is recoverable: the retry
        layer re-enters ``complete()``, whose ``_ensure_connected()``
        relaunches the browser.  Everything else defers to the global
        classifier.
        """
        # Matches the SHARED base, so a raise from the transport
        # (shared.cdp) is classified transient just like this package's own
        # ChromeAIConnectionError, which subclasses it.
        if isinstance(exc, CDPConnectionError):
            return {"transient": True, "rate_limit": False, "infra": True}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """No rate limits locally — no retry-after hints."""
        return None

    # ==================== Internal ====================

    def _start_bridge(self) -> None:
        """Create and connect the browser bridge from the resolved knobs."""
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:
                pass
        binary: Optional[str] = None
        if not self._cdp_url:
            binary = find_browser_binary(self._binary_knob)
            if not binary:
                raise ChromeAIBinaryNotFoundError(
                    "no Google Chrome / Microsoft Edge binary found (plain "
                    "Chromium is not supported — the on-device model is "
                    "Google-proprietary). Install Chrome, set "
                    "plugin_configs.chrome_ai.binary / JAATO_CHROME_AI_BINARY, "
                    "or attach to a running browser via cdp_url / "
                    "JAATO_CHROME_AI_CDP_URL.")
        self._bridge = self._bridge_factory(
            binary=binary,
            cdp_url=self._cdp_url,
            user_data_dir=self._user_data_dir,
            headless=self._headless,
            page_url=self._page_url,
            extra_args=self._extra_args,
            connect_timeout=self._connect_timeout,
            reuse_page=self._reuse_page,
        )
        self._bridge.connect()

    def _verify_model_available(self) -> None:
        """Check availability, optionally triggering the model download."""
        state = self._bridge.availability()
        if state == "downloadable" and self._auto_download:
            logger.info("chrome_ai: triggering on-device model download "
                        "(this can take a while)")
            self._bridge.download(
                on_progress=lambda loaded: logger.info(
                    "chrome_ai: model download %.0f%%", loaded * 100),
                timeout=self._download_timeout)
            state = self._bridge.availability()
        if state == "available":
            return
        if state == "no-api":
            raise ChromeAIUnavailableError(_ENABLE_INSTRUCTIONS)
        if state in ("downloadable", "downloading"):
            raise ChromeAIUnavailableError(
                f"model availability is '{state}'. {_DOWNLOAD_INSTRUCTIONS}")
        raise ChromeAIUnavailableError(
            f"the on-device model reports availability '{state}'. This "
            "usually means unsupported hardware (needs a supported GPU or, "
            "since Chrome 140, 16 GB RAM + 4 cores for CPU inference) or "
            "server-side gating by Google. Check chrome://on-device-internals "
            "in the same profile for details.")

    def _ensure_connected(self) -> None:
        """Reconnect the bridge if the browser died between turns."""
        if not self._connected:
            raise ChromeAIConnectionError(
                "provider is not connected — call connect() first")
        if self._bridge is None or not self._bridge.is_alive:
            logger.warning("chrome_ai: browser bridge is down; reconnecting")
            self._start_bridge()

    def _build_payload(
        self,
        messages: List[Message],
        system_instruction: Optional[str],
        tools: Optional[List[ToolSchema]],
        response_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the page-side ``run`` payload for one turn.

        The final history message becomes the ``prompt`` argument; every
        earlier message (plus the system prompt, with the tool section
        appended) becomes ``initialPrompts``.  When the final message is a
        TOOL result, its rendered text is the prompt — the continuation
        turn after jaato executed the model's calls.
        """
        system_parts = []
        if system_instruction:
            system_parts.append(system_instruction)
        if tools:
            system_parts.append(tool_schemas_to_prompt(tools))
        initial: List[Dict[str, str]] = []
        if system_parts:
            initial.append({"role": "system",
                            "content": "\n\n".join(system_parts)})

        history = list(messages)
        prompt = ""
        if history:
            last_converted = messages_to_prompt_api([history[-1]])
            if last_converted and history[-1].role != Role.MODEL:
                prompt = "\n\n".join(m["content"] for m in last_converted)
                history = history[:-1]
        initial.extend(messages_to_prompt_api(history))
        if not prompt:
            # Degenerate histories (empty, or ending on a MODEL message):
            # everything stays in initialPrompts and the model is nudged on.
            prompt = "Continue."

        payload: Dict[str, Any] = {
            "initialPrompts": initial,
            "prompt": prompt,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._top_k is not None:
            payload["topK"] = self._top_k
        if response_schema:
            payload["responseConstraint"] = response_schema
        return payload

    def _estimate_usage(self, payload: Dict[str, Any], output_text: str,
                        usage_event: Optional[Dict[str, Any]]) -> TokenUsage:
        """Estimate token usage for the turn (the API reports no counts).

        ``inputUsage`` — the session's whole-context usage after the turn,
        when the browser reports it — is the best available
        ``total_tokens``; prompt/output are always the ~4-chars/token
        estimate.
        """
        prompt_chars = len(payload.get("prompt") or "")
        for msg in payload.get("initialPrompts") or []:
            prompt_chars += len(msg.get("content") or "")
        prompt_tokens = max(1, prompt_chars // 4) if prompt_chars else 0
        output_tokens = self.count_tokens(output_text)
        total = prompt_tokens + output_tokens
        if isinstance(usage_event, dict):
            reported = usage_event.get("inputUsage")
            if isinstance(reported, (int, float)) and reported > 0:
                total = int(reported)
        return TokenUsage(prompt_tokens=prompt_tokens,
                          output_tokens=output_tokens,
                          total_tokens=total)


def create_provider() -> ChromeAIProvider:
    """Factory function for provider discovery."""
    return ChromeAIProvider()
