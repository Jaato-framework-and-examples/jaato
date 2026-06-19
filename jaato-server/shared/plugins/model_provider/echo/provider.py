"""Echo provider — a deterministic, creds-free, network-free model provider.

This provider exists purely for end-to-end tests of the agentic loop and the
gossip remote-spawn path.  It needs NO credentials and makes NO network calls,
yet it drives the real chat loop because it returns genuine ``ProviderResponse``
objects through the standard ``complete()`` contract.

Behaviour (fully deterministic — zero variance across runs):

* **Tool-call mode (first turn).**  If a ``tool_call`` dict is configured AND no
  tool-result (function_response) is present anywhere in ``messages``, the
  provider emits exactly one function call, verbatim, with
  ``finish_reason=TOOL_USE``.  This lets a test drive the loop into executing a
  tool (e.g. ``spawn_subagent``) without a real model.

* **Text mode (otherwise).**  Either no ``tool_call`` is configured, OR a
  tool-result is already present (the continuation turn after the tool ran).
  The provider returns a single text part with ``finish_reason=STOP``:
  - the configured ``response`` string if set, else
  - the last USER message's concatenated text (echo the prompt verbatim).

Configuration arrives via ``ProviderConfig.extra`` — the runtime merges the
profile's ``plugin_configs.echo`` block into ``extra`` (same path ollama /
lmstudio read ``extra.get("host")`` etc.):

    plugin_configs:
      echo:
        response: "fixed reply"            # optional str
        tool_call:                          # optional dict
          name: spawn_subagent
          args: {profile: researcher, prompt: "go"}

The provider is stateless w.r.t. conversation history (the session owns the
message list and passes it to ``complete()`` each turn), matching the
``ModelProviderPlugin`` contract.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

from ..base import ModalityCapabilityMixin, ProviderConfig
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
    ToolResult,
    ToolSchema,
    TurnResult,
)


logger = logging.getLogger(__name__)


# A deliberately large constant so the GC denominator never trips during the
# tiny synthetic conversations this provider is used for.  Not a guessed model
# capacity — echo has no real context window, this is just "effectively
# unbounded" for test purposes.
ECHO_CONTEXT_LIMIT = 1_000_000

# The single model name this provider exposes.
ECHO_MODEL = "echo"


# ==================== History serialization ====================
# Standard Message JSON round-trip, mirrored from the anthropic converters so
# persistence works without importing the anthropic package (echo must stay
# dependency-free).

def _serialize_message(message: Message) -> Dict[str, Any]:
    """Serialize a Message to a JSON-safe dict (text / call / response parts)."""
    parts: List[Dict[str, Any]] = []
    for part in message.parts:
        if part.text is not None:
            parts.append({"type": "text", "text": part.text})
        elif part.function_call is not None:
            fc = part.function_call
            parts.append({
                "type": "function_call",
                "id": fc.id,
                "name": fc.name,
                "args": fc.args,
            })
        elif part.function_response is not None:
            fr = part.function_response
            parts.append({
                "type": "function_response",
                "call_id": fr.call_id,
                "name": fr.name,
                "result": fr.result,
                "is_error": fr.is_error,
            })
    return {"role": message.role.value, "parts": parts}


def _deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a dict produced by :func:`_serialize_message` to a Message."""
    parts: List[Part] = []
    for p in data.get("parts", []):
        ptype = p.get("type")
        if ptype == "text":
            parts.append(Part(text=p["text"]))
        elif ptype == "function_call":
            parts.append(Part(function_call=FunctionCall(
                id=p.get("id", ""),
                name=p["name"],
                args=p.get("args", {}),
            )))
        elif ptype == "function_response":
            parts.append(Part(function_response=ToolResult(
                call_id=p.get("call_id", ""),
                name=p.get("name", ""),
                result=p.get("result"),
                is_error=p.get("is_error", False),
            )))
    return Message(role=Role(data["role"]), parts=parts)


class EchoProvider(ModalityCapabilityMixin):
    """Deterministic echo provider for creds-free e2e tests.

    Lifecycle is trivial: ``initialize()`` reads the two config knobs,
    ``connect()`` records the model name and flips ``is_connected`` True.
    No client object, no auth, no network.
    """

    def __init__(self):
        """Construct an unconnected provider with no configured behaviour."""
        self._model_name: Optional[str] = None
        self._connected: bool = False
        # Config knobs, populated in initialize().
        self._tool_call: Optional[Dict[str, Any]] = None
        self._response: Optional[str] = None
        # Usage from the last complete() call (always zero — echo costs nothing).
        self._last_usage: TokenUsage = TokenUsage()

    @property
    def name(self) -> str:
        """Provider identifier used by discovery and profile ``provider:``."""
        return "echo"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Read the echo behaviour knobs from ``config.extra``.

        ``extra`` is where the runtime merges the profile's
        ``plugin_configs.echo`` block, the same path other local providers
        read their knobs from.

        Args:
            config: Optional provider config.  ``extra["tool_call"]`` (dict)
                and ``extra["response"]`` (str) drive ``complete()``.
        """
        if config is None:
            config = ProviderConfig()
        self._tool_call = config.extra.get("tool_call")
        self._response = config.extra.get("response")

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional[ProviderConfig] = None,
    ) -> bool:
        """Always authenticated — echo needs no credentials.

        Must work on a fresh, uninitialized instance (the runtime calls this on
        a throwaway provider before ``initialize()``), so it touches no state
        set by ``initialize()``.
        """
        if on_message:
            on_message("echo provider requires no authentication")
        return True

    def shutdown(self) -> None:
        """No resources to release."""
        self._connected = False

    def get_auth_info(self) -> str:
        """Describe the (absent) credential source."""
        return "none (echo provider)"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Record the model name and mark the provider ready.

        Args:
            model: Model name (always ``"echo"`` in practice).
            skip_model_test: Ignored — echo has nothing to verify.
        """
        self._model_name = model
        self._connected = True

    @property
    def is_connected(self) -> bool:
        """True once ``connect()`` has been called."""
        return self._connected

    @property
    def model_name(self) -> Optional[str]:
        """The connected model name, or None before connect()."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """Return the single model this provider exposes."""
        models = [ECHO_MODEL]
        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return models

    # ==================== Stateless Completion ====================

    @staticmethod
    def _has_tool_result(messages: List[Message]) -> bool:
        """Whether any part anywhere in ``messages`` is a tool-result.

        A ``function_response`` part is the wire shape of an executed tool's
        result being fed back to the model.  Its presence means we are on the
        continuation turn (the tool already ran), so echo must NOT emit another
        tool call — it terminates the loop with text instead.
        """
        for msg in messages:
            for part in msg.parts:
                if part.function_response is not None:
                    return True
        return False

    @staticmethod
    def _last_user_text(messages: List[Message]) -> str:
        """Concatenated text of the most recent USER message, or "".

        Walks ``messages`` in reverse and returns the first USER message's
        ``.text`` (which joins all of that message's text parts).  Used as the
        echo payload in text mode when no explicit ``response`` is configured.
        """
        for msg in reversed(messages):
            if msg.role == Role.USER:
                return msg.text or ""
        return ""

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk=None,
        on_usage_update=None,
        on_function_call=None,
        on_thinking=None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Return a deterministic response.

        First-turn tool-call mode (``tool_call`` configured AND no tool-result
        present) → a single verbatim function call with ``TOOL_USE``.
        Otherwise text mode → ``response`` or the echoed last-user-text with
        ``STOP``.

        The provider is stateless: it inspects ``messages`` but mutates nothing.
        """
        self._last_usage = TokenUsage()

        if self._tool_call is not None and not self._has_tool_result(messages):
            # First-turn tool-call mode: emit the configured call VERBATIM.
            call = FunctionCall(
                id="echo-call-0",
                name=self._tool_call["name"],
                args=self._tool_call.get("args", {}),
            )
            if on_function_call:
                on_function_call(call)
            response = ProviderResponse(
                parts=[Part.from_function_call(call)],
                usage=self._last_usage,
                finish_reason=FinishReason.TOOL_USE,
            )
            return TurnResult.from_provider_response(response)

        # Text mode: configured response, else echo the last user prompt.
        text = self._response if self._response is not None else self._last_user_text(messages)
        if on_chunk and text:
            on_chunk(text)
        response = ProviderResponse(
            parts=[Part.from_text(text)],
            usage=self._last_usage,
            finish_reason=FinishReason.STOP,
        )
        return TurnResult.from_provider_response(response)

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Cheap length-based token estimate (~4 chars/token)."""
        if not content:
            return 0
        return max(1, len(content) // 4)

    def get_context_limit(self) -> int:
        """Effectively-unbounded window for the tiny test conversations."""
        return ECHO_CONTEXT_LIMIT

    def get_token_usage(self) -> TokenUsage:
        """Usage from the last ``complete()`` — always zero (echo is free)."""
        return self._last_usage

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize history to a JSON string (standard Message round-trip)."""
        return json.dumps([_serialize_message(m) for m in history])

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize a JSON string back to a list of Messages."""
        items = json.loads(data)
        return [_deserialize_message(m) for m in items]

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Echo does not constrain output to a JSON schema."""
        return False

    def supports_streaming(self) -> bool:
        """Echo returns whole responses in one shot (no token streaming)."""
        return False

    def supports_stop(self) -> bool:
        """Echo is instantaneous; stop is trivially supported (no-op)."""
        return True

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """Text-only — echo never marshals richer input."""
        return {"text"}

    # ==================== Agent Context / Thinking ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """No-op — echo emits no traces that need agent identity."""
        return None

    def supports_thinking(self) -> bool:
        """Echo has no reasoning mode."""
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """No-op — thinking is unsupported."""
        return None

    # ==================== Error Classification ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Use the global fallback classifier — echo raises no provider errors."""
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """No retry-after hints — echo has no rate limits."""
        return None


def create_provider() -> EchoProvider:
    """Factory function for plugin discovery."""
    return EchoProvider()
