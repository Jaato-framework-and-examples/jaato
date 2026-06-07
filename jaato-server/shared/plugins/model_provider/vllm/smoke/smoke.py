"""Minimal SDK harness for the vllm provider.

Connects to a running jaato daemon over IPC, spawns a session against the
``vllm-smoke`` profile + agent, sends one user message, prints model
output as it arrives, and exits on ``TURN_COMPLETED``.

This is a **wire test**: it validates the daemon can reach the remote
vLLM OpenAI-compatible server and round-trip a chat completion.  No
tools are exercised — function-calling is a separate smoke
(``smoke_tools.py``).

Prerequisites:
    1. A jaato daemon is running and listening on ``/tmp/jaato.sock``.
    2. The profile + agent templates from ``.jaato.example/`` are copied
       into the workspace's ``.jaato/profiles/`` and ``.jaato/agents/``.
    3. ``host`` and ``model`` in the profile are filled in to point at
       the remote endpoint.
    4. A vLLM server is running with the same model:
       ``vllm serve <model> --host 0.0.0.0 --port 8000``

Run:
    .venv/bin/jaato-server --ipc-socket /tmp/jaato.sock --daemon
    .venv/bin/python -m smoke --workspace /tmp/jaato-vllm-smoke
    # or from inside the workspace:
    cd /tmp/jaato-vllm-smoke && .venv/bin/python /path/to/smoke.py
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
PROFILE = "vllm-smoke"
AGENT = "vllm-smoke"
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
