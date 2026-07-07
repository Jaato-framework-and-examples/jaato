#!/usr/bin/env python3
"""Live smoke for the ``prose_tool_calls`` quirk (direct-provider, no daemon).

Unlike the sibling smokes (which drive a running daemon over IPC), this one
instantiates :class:`OpenRouterProvider` directly and calls ``complete()`` to
exercise the shared prose-emulated tool protocol
(``model_provider/_prose_tools.py``) end to end against a REAL model that
lacks native tool calling.  It is the wire test that proves a tool-less model
can still drive the harness through the quirk.

Default model: ``google/gemma-3-4b-it`` — a small model with NO tool-use
endpoint on OpenRouter (``supported_parameters`` lacks ``tools``), so the
prose path is the ONLY way it can emit a tool call.  With the quirk OFF the
gateway rejects a ``tools`` request outright (HTTP 404 "No endpoints found
that support tool use"), which ``--compare`` demonstrates.

What each mode checks:
  (default)   one turn, quirk ON  — model emits a fenced ``tool_call`` block
              (hashed wire id), provider parses it into a real FunctionCall.
  --loop      two turns — the model's own call + the tool result are replayed
              as prose history (``messages_to_prose_chat``) and the model
              produces a final natural-language answer grounded in the result.
  --compare   also runs one turn with the quirk OFF, showing the gateway
              reject the native ``tools`` array for this tool-less model.

Prerequisites:
    export JAATO_OPENROUTER_API_KEY=sk-or-...     # or: $(pass jaato/openrouter/api-key)
    # the provider resolves the key from the env var / stored credentials.

Run (from anywhere, with jaato-server importable):
    python smoke_prose_tool_calls.py
    python smoke_prose_tool_calls.py --loop
    python smoke_prose_tool_calls.py --compare
    python smoke_prose_tool_calls.py meta-llama/llama-3.2-3b-instruct --loop
"""
from __future__ import annotations

import argparse
import sys

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
from shared.tool_id_map import name_to_id
from jaato_sdk.plugins.model_provider.types import (
    FunctionCall, Message, Part, Role, ToolResult, ToolSchema,
)

DEFAULT_MODEL = "google/gemma-3-4b-it"  # no native tool-use endpoint

GET_WEATHER = ToolSchema(
    name="get_weather",
    description="Get the current weather for a city. Returns temperature in Celsius.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)
SYSTEM = "You are a helpful assistant. Use the available tools when needed."


def _connect(model: str, quirk: bool) -> OpenRouterProvider:
    """Build a provider (key resolved from env / stored creds) and connect."""
    provider = OpenRouterProvider()
    extra = {"quirks": {"prose_tool_calls": True}} if quirk else {}
    # No api_key passed: the provider resolves JAATO_OPENROUTER_API_KEY /
    # stored credentials itself, failing loud if neither is present.
    provider.initialize(ProviderConfig(extra=extra))
    provider.connect(model)
    return provider


def single_turn(model: str, quirk: bool) -> bool:
    """One turn.  Returns True if a tool call was produced (quirk ON path)."""
    print("=" * 72)
    print(f"MODEL={model}   prose_tool_calls={'ON' if quirk else 'OFF'}")
    print("=" * 72)
    provider = _connect(model, quirk)
    print(f"connected: context_limit={provider.get_context_limit()} "
          f"prose_mode={provider._prose_tool_calls}")
    print(f"(hashed wire id for get_weather = {name_to_id('get_weather')})\n")

    chunks: list[str] = []
    try:
        result = provider.complete(
            [Message.from_text(Role.USER,
                               "What's the weather in Oslo right now? Use the tool.")],
            system_instruction=SYSTEM,
            tools=[GET_WEATHER],
            on_chunk=chunks.append,
        )
    except Exception as exc:  # noqa: BLE001 — the OFF path is EXPECTED to fail
        print(f"REQUEST REJECTED (expected with quirk OFF for a tool-less "
              f"model):\n  {type(exc).__name__}: {exc}\n")
        provider.shutdown()
        return False

    print("--- RAW MODEL OUTPUT (streamed) ---")
    print("".join(chunks) or "(no streamed text)")
    print("\n--- PARSED RESULT ---")
    print(f"outcome       : {result.outcome.value}")
    print(f"finish_reason : {result.finish_reason.value}")
    calls = result.response.get_function_calls() if result.response else []
    if result.response:
        print(f"text          : {result.response.get_text()!r}")
    print(f"function_calls: {len(calls)}")
    for c in calls:
        print(f"   -> name={c.name!r} args={c.args} id={c.id!r}")
    print(f"usage         : {provider.get_token_usage()}\n")
    provider.shutdown()
    return bool(calls)


def full_loop(model: str) -> None:
    """call -> execute -> feed result -> final answer (history replay half)."""
    print("=" * 72)
    print(f"FULL LOOP   MODEL={model}   prose_tool_calls=ON")
    print("=" * 72)
    provider = _connect(model, quirk=True)

    history = [Message.from_text(
        Role.USER, "What's the weather in Oslo? Use the tool.")]
    r1 = provider.complete(history, system_instruction=SYSTEM,
                           tools=[GET_WEATHER])
    call = r1.response.get_function_calls()[0]
    print(f"TURN 1  outcome={r1.outcome.value}  call={call.name}({call.args})")

    tool_result = {"city": "Oslo", "temp_c": -3, "conditions": "light snow"}
    print(f"        executed -> {tool_result}")

    # Append the model's own call + the tool result; ask for the answer.
    history.append(Message(role=Role.MODEL, parts=[Part(function_call=call)]))
    history.append(Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id=call.id, name=call.name, result=tool_result))]))

    r2 = provider.complete(history, system_instruction=SYSTEM, tools=[GET_WEATHER])
    print(f"TURN 2  outcome={r2.outcome.value}")
    print(f"        final answer: {r2.response.get_text().strip()!r}\n")
    provider.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                    help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--loop", action="store_true",
                    help="run the full two-turn agentic loop")
    ap.add_argument("--compare", action="store_true",
                    help="also run one turn with the quirk OFF (expected 404)")
    args = ap.parse_args()

    produced = single_turn(args.model, quirk=True)
    if args.compare:
        single_turn(args.model, quirk=False)
    if args.loop:
        full_loop(args.model)
    # Exit non-zero if the quirk-ON turn produced no tool call (regression).
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
