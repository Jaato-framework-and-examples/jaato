#!/usr/bin/env python3
"""Live smoke test for the chrome_ai provider (manual, not run in CI).

Requires a host with Google Chrome installed and the Gemini Nano
component available (or downloadable) in the configured profile.

Usage:
    python smoke_chat.py                       # launch Chrome, simple chat
    JAATO_CHROME_AI_CDP_URL=http://localhost:9222 python smoke_chat.py
    python smoke_chat.py --tools               # exercise tool-call parsing
    python smoke_chat.py --download            # allow model download
"""

import sys

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.chrome_ai import ChromeAIProvider
from jaato_sdk.plugins.model_provider.types import (
    Message, Role, ToolSchema,
)


def main() -> int:
    extra = {}
    if "--download" in sys.argv:
        extra["auto_download"] = True
    if "--headed" in sys.argv:
        extra["headless"] = False

    provider = ChromeAIProvider()
    if not provider.verify_auth(on_message=lambda m: print(f"[auth] {m}")):
        return 1
    provider.initialize(ProviderConfig(extra=extra))
    print("[smoke] connecting (this launches Chrome)...")
    provider.connect("gemini-nano")
    print(f"[smoke] connected; context window = "
          f"{provider.get_context_limit()} tokens")

    tools = None
    if "--tools" in sys.argv:
        tools = [ToolSchema(
            name="get_time",
            description="Get the current time in a city",
            parameters={"type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]})]
        prompt = "What time is it in Tokyo? Use the tool."
    else:
        prompt = "Reply with one short sentence: what are you?"

    result = provider.complete(
        [Message.from_text(Role.USER, prompt)],
        system_instruction="You are a concise assistant.",
        tools=tools,
        on_chunk=lambda text: print(text, end="", flush=True),
    )
    print()
    print(f"[smoke] outcome={result.outcome.value} "
          f"finish={result.finish_reason.value}")
    if result.response and result.response.has_function_calls():
        for call in result.response.get_function_calls():
            print(f"[smoke] tool call: {call.name}({call.args})")
    print(f"[smoke] usage: {provider.get_token_usage()}")
    provider.shutdown()
    return 0 if not result.is_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
