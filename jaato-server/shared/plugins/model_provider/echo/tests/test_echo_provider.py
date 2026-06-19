"""Tests for the echo provider — the deterministic, creds-free model used to
drive the agentic loop in gossip remote-spawn e2e tests.

Covers:
- text-echo mode (no tool_call): echoes the last user prompt verbatim, STOP
- ``response`` override returns the configured string
- tool-call mode FIRST turn (no tool-result present): one verbatim function
  call, TOOL_USE
- tool-call mode CONTINUATION (a function_response present): TEXT + STOP, so
  the loop terminates instead of looping forever
- verify_auth() on a fresh uninitialized instance returns True
"""

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.echo import EchoProvider, create_provider
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    TurnOutcome,
)


MARKER = "GOSSIP-E2E-ECHO-MARKER"


def _connected(extra=None) -> EchoProvider:
    """A provider initialized with ``extra`` and connected to model 'echo'."""
    provider = create_provider()
    provider.initialize(ProviderConfig(extra=extra or {}))
    provider.connect("echo")
    return provider


def test_text_echo_mode_echoes_prompt_verbatim():
    """No tool_call configured → echo the last user message, STOP, no calls."""
    provider = _connected()
    messages = [Message.from_text(Role.USER, MARKER)]

    result = provider.complete(messages)

    assert result.outcome == TurnOutcome.RESPONSE
    assert result.finish_reason == FinishReason.STOP
    assert result.text == MARKER
    assert result.response is not None
    assert result.response.get_function_calls() == []


def test_response_override_returns_configured_string():
    """``response`` knob is returned verbatim, ignoring the prompt."""
    provider = _connected({"response": "FIXED-REPLY"})
    messages = [Message.from_text(Role.USER, "anything at all")]

    result = provider.complete(messages)

    assert result.outcome == TurnOutcome.RESPONSE
    assert result.finish_reason == FinishReason.STOP
    assert result.text == "FIXED-REPLY"


def test_tool_call_mode_first_turn_emits_verbatim_call():
    """tool_call configured + no tool-result yet → one verbatim call, TOOL_USE."""
    tool_call = {"name": "spawn_subagent", "args": {"profile": "researcher", "prompt": "go"}}
    provider = _connected({"tool_call": tool_call})
    messages = [Message.from_text(Role.USER, "kick off the subagent")]

    result = provider.complete(messages)

    assert result.outcome == TurnOutcome.TOOL_USE
    assert result.finish_reason == FinishReason.TOOL_USE
    calls = result.response.get_function_calls()
    assert len(calls) == 1
    assert calls[0].name == "spawn_subagent"
    assert calls[0].args == {"profile": "researcher", "prompt": "go"}


def test_tool_call_mode_continuation_returns_text_and_stops():
    """A function_response present → TEXT + STOP, NOT another tool call."""
    tool_call = {"name": "spawn_subagent", "args": {"profile": "researcher"}}
    provider = _connected({"tool_call": tool_call})
    messages = [
        Message.from_text(Role.USER, "kick off the subagent"),
        # The model's turn that requested the tool.
        Message(
            role=Role.MODEL,
            parts=[Part.from_function_call(
                FunctionCall(id="echo-call-0", name="spawn_subagent", args={"profile": "researcher"})
            )],
        ),
        # The tool result fed back — this is what flips echo into text mode.
        Message(
            role=Role.TOOL,
            parts=[Part.from_function_response(
                ToolResult(call_id="echo-call-0", name="spawn_subagent", result="subagent done")
            )],
        ),
    ]

    result = provider.complete(messages)

    assert result.outcome == TurnOutcome.RESPONSE
    assert result.finish_reason == FinishReason.STOP
    assert result.response.get_function_calls() == []
    # Text mode with no `response` knob echoes the last USER message.
    assert result.text == "kick off the subagent"


def test_verify_auth_true_on_fresh_uninitialized_instance():
    """verify_auth() must work BEFORE initialize() and always return True."""
    provider = EchoProvider()  # NOT initialized, NOT connected
    assert provider.verify_auth() is True
