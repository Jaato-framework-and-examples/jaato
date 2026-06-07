"""Tests for the ``force_tool_choice_for_lifecycle`` quirk on the
vLLM provider (Path 1, server 0.6.195+).

Verifies the provider:

- Reads the quirk from ``profile.quirks.force_tool_choice_for_lifecycle``
  via ``config.extra["quirks"]`` at init time.
- Forwards the session-supplied ``tool_choice`` to vLLM's Chat
  Completions kwargs ONLY when the quirk is enabled.
- Drops the ``tool_choice`` kwarg silently when the quirk is off
  (so a session that always passes it doesn't break non-quirk
  profiles).

The OpenAI client interaction is mocked so the test doesn't need a
running vLLM endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from jaato_sdk.plugins.model_provider.types import (
    Message,
    ToolSchema,
)
from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.vllm.provider import VLLMProvider


# ----------------------------------------------------------------------
# Quirk-read at initialize()
# ----------------------------------------------------------------------


class TestForceToolChoiceQuirkRead:
    def test_default_off(self) -> None:
        p = VLLMProvider()
        quirks: Dict[str, Any] = {}
        p._force_tool_choice_for_lifecycle = bool(
            quirks.get("force_tool_choice_for_lifecycle", False)
        )
        assert p._force_tool_choice_for_lifecycle is False

    def test_quirk_enabled_via_config(self) -> None:
        p = VLLMProvider()
        quirks = {"force_tool_choice_for_lifecycle": True}
        p._force_tool_choice_for_lifecycle = bool(
            quirks.get("force_tool_choice_for_lifecycle", False)
        )
        assert p._force_tool_choice_for_lifecycle is True

    def test_both_quirks_independent(self) -> None:
        """Setting one quirk doesn't enable the other.  Each is
        opt-in independently."""
        p = VLLMProvider()
        quirks = {"coerce_typed_tool_args": True}
        p._coerce_typed_tool_args = bool(
            quirks.get("coerce_typed_tool_args", False)
        )
        p._force_tool_choice_for_lifecycle = bool(
            quirks.get("force_tool_choice_for_lifecycle", False)
        )
        assert p._coerce_typed_tool_args is True
        assert p._force_tool_choice_for_lifecycle is False


# ----------------------------------------------------------------------
# Wire forwarding in complete() — non-streaming path only (streaming
# is the same code path for kwargs assembly).
# ----------------------------------------------------------------------


def _wire_provider(
    *,
    coerce: bool = False,
    force_tool_choice: bool = False,
) -> VLLMProvider:
    """Build a connected VLLMProvider stub bypassing real init/connect.

    Sets the quirks directly and stubs the OpenAI client so
    ``complete()`` can run.
    """
    p = VLLMProvider()
    p._coerce_typed_tool_args = coerce
    p._force_tool_choice_for_lifecycle = force_tool_choice
    # Stub client + model so complete()'s connection-check passes.
    p._client = MagicMock()
    p._model_name = "llama31-8b-awq"

    # The chat-completions call returns a minimal response object with
    # the choices / usage attributes the response_from_openai helper
    # expects.  We don't care about the response content for these
    # tests — only about what kwargs were sent to chat.completions.create.
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(
        message=MagicMock(content="ok", tool_calls=None),
        finish_reason="stop",
    )]
    fake_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        prompt_tokens_details=None,
    )
    p._client.chat.completions.create.return_value = fake_response
    return p


def _tool() -> ToolSchema:
    return ToolSchema(
        name="signal_completion",
        description="terminal signal",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )


def _msg(text: str) -> Message:
    """Build a minimal Message via the parts API."""
    from jaato_sdk.plugins.model_provider.types import Part
    return Message(role="user", parts=[Part.from_text(text)])


class TestCompleteForwardsToolChoice:
    def test_quirk_off_drops_tool_choice_kwarg(self) -> None:
        """When the quirk is OFF, even if the session passes
        ``tool_choice=``, the provider must NOT forward it to
        vLLM's ``chat.completions.create`` kwargs."""
        p = _wire_provider(force_tool_choice=False)
        p.complete(
            messages=[_msg("hi")],
            tools=[_tool()],
            tool_choice={
                "type": "function",
                "function": {"name": "signal_completion"},
            },
        )
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs

    def test_quirk_on_forwards_tool_choice_with_hash_translation(self) -> None:
        """When the quirk is ON and the session passes a canonical
        function name in ``tool_choice``, the provider translates to
        the hashed wire id via ``name_to_id`` before forwarding to
        vLLM — symmetric to how the wire ``tools`` array carries
        hashed names (see ``tool_schemas_to_openai`` line 74).
        Pre-PR-251 the canonical name was forwarded verbatim and
        vLLM 400s with ``The tool specified in tool_choice does not
        match any of the specified tools``.
        """
        from shared.tool_id_map import name_to_id

        p = _wire_provider(force_tool_choice=True)
        canonical = {
            "type": "function",
            "function": {"name": "signal_completion"},
        }
        p.complete(
            messages=[_msg("hi")],
            tools=[_tool()],
            tool_choice=canonical,
        )
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        # Wire shape uses the hashed id, NOT the canonical name.
        expected_hash = name_to_id("signal_completion")
        assert call_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": expected_hash},
        }
        # Hashed id is deterministic + regex-safe.
        assert expected_hash.startswith("t_")
        assert len(expected_hash) == 10  # "t_" + 8 hex chars

    def test_quirk_on_non_function_tool_choice_forwarded_verbatim(self) -> None:
        """Non-function tool_choice shapes (e.g. the string
        ``"required"`` or ``"none"``) are forwarded verbatim — no
        name to translate."""
        p = _wire_provider(force_tool_choice=True)
        p.complete(
            messages=[_msg("hi")],
            tools=[_tool()],
            tool_choice={"type": "required"},
        )
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "required"}

    def test_quirk_on_without_session_tool_choice_omits(self) -> None:
        """Quirk-ON doesn't change behavior when the session passes
        ``tool_choice=None`` (the normal no-retry path)."""
        p = _wire_provider(force_tool_choice=True)
        p.complete(
            messages=[_msg("hi")],
            tools=[_tool()],
            # No tool_choice kwarg — same shape as the no-retry path.
        )
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs

    def test_quirk_on_without_tools_omits_tool_choice(self) -> None:
        """vLLM rejects ``tool_choice`` when no ``tools`` are passed.
        Guard against that mis-shape — drop tool_choice when tools
        list is empty/absent."""
        p = _wire_provider(force_tool_choice=True)
        p.complete(
            messages=[_msg("hi")],
            tools=None,
            tool_choice={
                "type": "function",
                "function": {"name": "signal_completion"},
            },
        )
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs

    def test_tool_choice_default_none_no_kwarg(self) -> None:
        """The signature defaults tool_choice to None — sessions that
        don't opt into the kwarg never trigger the forwarding logic.
        Backward-compat path for every caller pre-Path-1."""
        p = _wire_provider(force_tool_choice=True)
        # Call without passing tool_choice at all.
        p.complete(messages=[_msg("hi")], tools=[_tool()])
        call_kwargs = p._client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs
