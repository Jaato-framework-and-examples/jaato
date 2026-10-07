"""A minimal consumer of the jaato-eval driver contract (JAATO_EVAL_CONTRACT=1).

Two stages in a fixed order, each its own session, both stamped with the
arm's cascade id.  What a real driver adds — typed payloads between
stages, host tools in this process, a DAG — changes nothing below the
session-opening line.

Exit codes follow the vocabulary in ``jaato_eval/driver.py``: 0 ran to
its end, 75 the environment was unusable, anything else stopped short.
"""
from __future__ import annotations

import asyncio
import os
import sys

EX_TEMPFAIL = 75


def _contract() -> dict:
    if os.environ.get("JAATO_EVAL_CONTRACT") != "1":
        print("driver: not started under a jaato-eval contract I understand",
              file=sys.stderr)
        sys.exit(EX_TEMPFAIL)
    return {
        "workspace": os.environ["JAATO_EVAL_WORKSPACE"],
        "config_root": os.environ["JAATO_EVAL_CONFIG_ROOT"],
        "cascade_id": os.environ["JAATO_EVAL_CASCADE_ID"],
        # Absent means the SDK's default socket, as it does for the engine.
        "socket": os.environ.get("JAATO_EVAL_SOCKET"),
        "word": os.environ.get("JAATO_EVAL_PARAM_WORD", "READY"),
    }


async def _stage(contract: dict, prompt: str) -> None:
    """One session, in the arm's workspace, stamped with the arm's cid."""
    from jaato_sdk.client.convenience import Session
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType

    kwargs = {
        "client_type": ClientType.API,
        "workspace_path": contract["workspace"],
        "config_root": contract["config_root"],
        "env_file": ".env",            # carries JAATO_PROFILE_SET, the sweep's axis
    }
    if contract["socket"]:
        kwargs["socket_path"] = contract["socket"]
    client = IPCClient(**kwargs)
    if not await client.connect(timeout=120):
        print("driver: daemon unreachable", file=sys.stderr)
        sys.exit(EX_TEMPFAIL)
    try:
        sid = await client.create_session(
            profile="worker", cascade_driver_id=contract["cascade_id"])
        await Session(client, sid).complete(prompt)
    finally:
        await client.disconnect()


async def main() -> int:
    contract = _contract()
    word = contract["word"]
    stages = [
        f"Create a file called answer.txt in the workspace root whose entire "
        f"contents are the single word {word} (no trailing punctuation).",
        "Read answer.txt in the workspace root and create echo.txt with "
        "exactly the same contents.",
    ]
    for index, prompt in enumerate(stages, start=1):
        try:
            await _stage(contract, prompt)
        except Exception as exc:  # noqa: BLE001 — a stage that died stops the run short
            print(f"driver: stage {index} failed: {exc!r}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
