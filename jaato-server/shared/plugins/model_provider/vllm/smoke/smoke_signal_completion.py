"""signal_completion smoke for the vllm provider.

Test 2 of 3 (smoke_chat / smoke_signal_completion / smoke_tools):
exercises the lifecycle ``signal_completion`` tool with a non-trivial
``completion_payload_schema``.  The persona instructs the model to
emit a brief acknowledgement THEN call ``signal_completion`` with a
structured payload (summary + status + word_count).  Success: the
model emits valid payload → schema validation passes → session
terminates → ``TURN_COMPLETED`` fires.

This is the **lifecycle test**: it validates the daemon's
schema-driven completion contract end-to-end against the vLLM
endpoint.  Weak models may fail to produce a schema-valid payload
on the first try; the framework returns a self-correction error
and lets them retry within ``max_turns``.

Sibling smokes:
    smoke_chat.py             — pure text round-trip, no signal_completion
    smoke_tools.py            — cli plugin + signal_completion (full tools shape)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import (
    AgentOutputEvent,
    ClientType,
    ErrorEvent,
    EventType,
    TurnCompletedEvent,
)

SOCKET = "/tmp/jaato.sock"
PROFILE = "vllm-signal_completion"
AGENT = "vllm-signal_completion"
PROMPT = (
    "Acknowledge briefly, then signal completion with a structured "
    "payload containing your summary and word count."
)
TURN_TIMEOUT_SECONDS = 120.0


async def main(workspace: str) -> int:
    client = IPCClient(
        socket_path=SOCKET,
        client_type=ClientType.API,
        workspace_path=workspace,
    )
    if not await client.connect(timeout=10.0):
        print("[smoke] connect failed", file=sys.stderr)
        return 2

    done = asyncio.Event()
    failure: list[str] = []

    def on_output(e: AgentOutputEvent) -> None:
        sys.stdout.write(e.text)
        sys.stdout.flush()

    def on_turn_complete(_: TurnCompletedEvent) -> None:
        sys.stdout.write("\n")
        done.set()

    def on_error(e: ErrorEvent) -> None:
        failure.append(f"{e.error_type}: {e.error}" if e.error_type else e.error)
        done.set()

    client.subscribe(EventType.AGENT_OUTPUT, on_output)
    client.subscribe(EventType.TURN_COMPLETED, on_turn_complete)
    client.subscribe(EventType.ERROR, on_error)

    await client.create_session(profile=PROFILE, agent=AGENT)
    await client.send_message(PROMPT)

    try:
        await asyncio.wait_for(done.wait(), timeout=TURN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        print(
            f"\n[smoke] timeout after {TURN_TIMEOUT_SECONDS}s waiting "
            "for TURN_COMPLETED",
            file=sys.stderr,
        )
        await client.disconnect()
        return 3

    await client.disconnect()

    if failure:
        print(f"[smoke] ERROR: {failure[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "Workspace path containing .jaato/profiles/ and .jaato/agents/. "
            "Defaults to the current directory."
        ),
    )
    args = parser.parse_args()
    workspace = args.workspace or os.getcwd()
    sys.exit(asyncio.run(main(workspace)))
