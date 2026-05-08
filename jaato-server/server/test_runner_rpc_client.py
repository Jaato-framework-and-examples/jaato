"""End-to-end tests for daemon-side ``RunnerSpawner`` + ``RunnerRPCClient``.

Spawns a real ``python -m server.runner`` subprocess with
``JAATO_RUNNER_DISABLE_CONFINE=1`` so the test exercises the full
fork+exec path on hosts without AppArmor (CI, dev workstations).
The §2.6 integration test covers the apparmor-on path.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Tuple

import pytest

from server.runner_spawner import (
    DaemonConfinementError,
    RunnerSpawner,
    SpawnedRunner,
)
from server.runner_rpc_client import RunnerCallError, RunnerRPCClient


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


async def _make_client(spawner: RunnerSpawner) -> Tuple[SpawnedRunner, RunnerRPCClient]:
    """Spawn a confine-disabled runner and start the daemon-side client."""
    spawned = spawner.spawn(
        profile_name="ignored-confine-disabled",
        session_id="test-session",
        workspace_path="/tmp",
        disable_confine=True,
    )
    client = RunnerRPCClient(
        spawned.parent_socket,
        runner_pid=spawned.pid,
    )
    await client.start()
    return spawned, client


# ----------------------------------------------------------------------
# Spawn-side
# ----------------------------------------------------------------------


def test_spawn_requires_profile_unless_disabled() -> None:
    spawner = RunnerSpawner()
    with pytest.raises(ValueError, match="profile_name required"):
        spawner.spawn(
            profile_name="",
            session_id="x",
            workspace_path=None,
        )


def test_spawn_assert_unconfined(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the daemon is somehow confined at fork time, refuse to spawn."""
    spawner = RunnerSpawner()
    # Forge /proc/self/attr/current to report a per-session profile.
    import io

    def _fake_open(path: str, *_a, **_k):
        if path == "/proc/self/attr/current":
            return io.StringIO("jaato-ws-someone-else (enforce)\n")
        return open(path)

    monkeypatch.setattr("builtins.open", _fake_open)

    with pytest.raises(DaemonConfinementError):
        spawner.spawn(
            profile_name="jaato-ws-test",
            session_id="x",
            workspace_path=None,
        )


# ----------------------------------------------------------------------
# End-to-end: spawn + echo + cli
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_round_trip_through_real_runner() -> None:
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    try:
        env = await client.call("echo", {"hello": "world"})
        assert env.ok is True
        assert env.result == {"hello": "world"}
        assert env.error is None
    finally:
        await client.close(timeout=2)


@pytest.mark.asyncio
async def test_cli_through_real_runner_streams_and_responds() -> None:
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    try:
        chunks: List[Tuple[str, str, Optional[str]]] = []

        def _on_output(source: str, text: str, mode: Optional[str]) -> None:
            chunks.append((source, text, mode))

        env = await client.call(
            "tool.execute",
            {"name": "cli_based_tool", "args": {"command": "echo hello"}},
            on_output=_on_output,
        )
        assert env.ok is True
        assert env.result["stdout"] == "hello\n"
        assert env.result["returncode"] == 0
        # The dispatcher should have streamed the line via on_output.
        assert ("stdout", "hello\n", None) in chunks
    finally:
        await client.close(timeout=2)


@pytest.mark.asyncio
async def test_runner_close_reaps_subprocess() -> None:
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    pid = spawned.pid
    await client.close(timeout=3)
    # After close, the child should be reaped — waitpid will raise
    # ChildProcessError because we already reaped it inside close().
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


@pytest.mark.asyncio
async def test_call_after_close_raises() -> None:
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    await client.close(timeout=2)
    with pytest.raises(RunnerCallError):
        await client.call("echo", {})


@pytest.mark.asyncio
async def test_unknown_method_returns_typed_error() -> None:
    """Domain-failure path: ``ok=False`` envelope, NOT an exception."""
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    try:
        env = await client.call("not.a.method")
        assert env.ok is False
        assert env.error is not None
        assert "unknown method" in env.error.message
    finally:
        await client.close(timeout=2)


@pytest.mark.asyncio
async def test_cancel_token_trips_runner_cancel() -> None:
    """A cancel-token registered on the call sends a cancel frame, and
    the cli runner short-circuits with CancelledException → ok=False."""
    from jaato_sdk.plugins.model_provider.types import CancelToken

    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    try:
        token = CancelToken()

        # Use a streaming cli command so the runner has cancel-check
        # opportunities (run_command only checks between output lines).
        async def _trip_after_a_bit() -> None:
            await asyncio.sleep(0.2)
            token.cancel()

        loop = asyncio.get_running_loop()
        loop.create_task(_trip_after_a_bit())

        env = await client.call(
            "tool.execute",
            {
                "name": "cli_based_tool",
                "args": {
                    "command": (
                        "for i in $(seq 1 200); do echo line-$i; "
                        "sleep 0.05; done"
                    ),
                },
            },
            cancel_token=token,
        )
        assert env.ok is False
        assert env.error is not None
        assert env.error.type == "CancelledException"
    finally:
        await client.close(timeout=3)
