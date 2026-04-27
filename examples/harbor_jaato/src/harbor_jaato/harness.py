"""In-container harness — drives the jaato daemon and writes result.json.

Lifecycle:

1. Connect via ``IPCRecoveryClient`` to ``/tmp/jaato.sock``. The SDK
   auto-starts the daemon (``python -m server --daemon``) if it isn't
   already running.
2. Create a session pinned to the requested profile (the BaseAgent
   writes ``.jaato/profiles/harbor.json`` during ``setup()`` so model,
   provider, and plugins are baked into the profile).
3. Send the instruction; drain events until a terminal status.
4. Auto-respond ``"a"`` (always-allow) to the first
   ``PermissionRequestedEvent`` — that promotes every subsequent tool
   call to whitelisted, so we don't bottleneck on per-call approvals.
5. Flush ``result.json`` atomically after every ``TurnCompletedEvent``
   so a kill mid-run still leaves usable token counts in
   ``AgentContext``.

CLI::

    python -m harbor_jaato.harness \\
        --instruction-file /workspace/.jaato/instruction.txt \\
        --profile harbor \\
        --result /workspace/.jaato/result.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from jaato_sdk.client import IPCRecoveryClient
from jaato_sdk.events import (
    AgentCompletedEvent,
    AgentOutputEvent,
    AgentStatusChangedEvent,
    ErrorEvent,
    PermissionRequestedEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TurnCompletedEvent,
)

from .result import HarnessResult

logger = logging.getLogger("harbor_jaato.harness")


async def _drive(args: argparse.Namespace) -> int:
    instruction = Path(args.instruction_file).read_text()
    out_path = Path(args.result)
    result = HarnessResult()
    result.write_atomic(out_path)

    client = IPCRecoveryClient(
        socket_path=args.socket,
        workspace_path=Path(args.workspace),
    )
    await client.connect()
    await client.create_session(profile=args.profile, agent=args.agent)
    await client.send_message(instruction)

    permission_promoted = False
    try:
        async for ev in client.events():
            if isinstance(ev, AgentOutputEvent):
                result.trajectory.append(
                    {"kind": "output", "source": ev.source, "text": ev.text}
                )
            elif isinstance(ev, ToolCallStartEvent):
                result.trajectory.append(
                    {
                        "kind": "tool_start",
                        "tool": ev.tool_name,
                        "args": ev.tool_args,
                        "call_id": ev.call_id,
                    }
                )
            elif isinstance(ev, ToolCallEndEvent):
                result.trajectory.append(
                    {
                        "kind": "tool_end",
                        "tool": ev.tool_name,
                        "call_id": ev.call_id,
                        "success": ev.success,
                        "error": ev.error_message,
                    }
                )
            elif isinstance(ev, PermissionRequestedEvent):
                # First prompt: "a" promotes the tool to the session
                # whitelist so we don't pay a round-trip per call.
                response = "a" if not permission_promoted else "y"
                permission_promoted = True
                await client.respond_to_permission(ev.request_id, response)
            elif isinstance(ev, TurnCompletedEvent):
                result.n_input_tokens += ev.prompt_tokens
                result.n_output_tokens += ev.output_tokens
                result.n_cache_read_tokens += ev.cache_read_tokens or 0
                result.n_cache_creation_tokens += ev.cache_creation_tokens or 0
                result.turns += 1
                result.write_atomic(out_path)
            elif isinstance(ev, AgentCompletedEvent):
                result.finished = True
                if not ev.success:
                    result.error = ev.error or "agent reported failure"
                break
            elif isinstance(ev, AgentStatusChangedEvent) and ev.status in (
                "done",
                "error",
            ):
                result.finished = True
                if ev.status == "error":
                    result.error = ev.error or "status=error"
                break
            elif isinstance(ev, ErrorEvent):
                result.error = getattr(ev, "message", "") or "ErrorEvent"
                break
    finally:
        result.write_atomic(out_path)
        try:
            await client.disconnect()
        except Exception:
            logger.warning("disconnect failed", exc_info=True)

    return 0 if result.error is None else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--profile", default="harbor")
    p.add_argument("--agent", default="harbor-shell")
    p.add_argument("--result", required=True)
    p.add_argument("--socket", default="/tmp/jaato.sock")
    p.add_argument(
        "--workspace",
        default="/workspace",
        help="Workspace path; pinned so the daemon doesn't fall back "
        "to CWD or $HOME, which can drift from where Harbor mounts "
        "the task files.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    sys.exit(asyncio.run(_drive(args)))


if __name__ == "__main__":
    main()
