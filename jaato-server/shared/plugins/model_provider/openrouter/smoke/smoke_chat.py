"""Chat smoke for the openrouter provider — pure text round-trip.

Test 1 of 3 (smoke_chat / smoke_signal_completion / smoke_tools):
sends one user message, expects one text response, exits on
``TURN_COMPLETED``.  Profile declares **no**
``completion_payload_schema`` so ``signal_completion`` is HIDDEN
from the model's tool surface (per the 2026-06-07 schema gate);
the persona just emits text and the turn ends naturally when the
provider response carries no function calls.

This is the **wire test**: it validates the daemon can reach the
OpenRouter cloud gateway and round-trip a chat completion.
Tool-calling fidelity is exercised by the sibling
``smoke_signal_completion.py`` (lifecycle tool only) and
``smoke_tools.py`` (cli + signal_completion).

Prerequisites:
    1. A jaato daemon is running and listening on ``/tmp/jaato.sock``.
    2. The profile + agent templates from ``.jaato.example/`` are copied
       into the workspace's ``.jaato/profiles/`` and ``.jaato/agents/``.
    3. ``JAATO_OPENROUTER_API_KEY`` is set in the workspace ``.env`` so the
       profile's ``${JAATO_OPENROUTER_API_KEY}`` substitution resolves.

Run:
    .venv/bin/jaato-server --ipc-socket /tmp/jaato.sock --daemon
    .venv/bin/python smoke_chat.py --workspace /tmp/jaato-openrouter-smoke
    # or from inside the workspace:
    cd /tmp/jaato-openrouter-smoke && .venv/bin/python /path/to/smoke_chat.py
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
PROFILE = "openrouter-chat"
AGENT = "openrouter-chat"
PROMPT = "Reply with exactly one short sentence saying hello."
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
