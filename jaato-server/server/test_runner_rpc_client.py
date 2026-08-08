"""End-to-end tests for daemon-side ``RunnerSpawner`` + ``RunnerRPCClient``.

Spawns a real ``python -m server.runner`` subprocess with
``JAATO_RUNNER_DISABLE_CONFINE=1`` so the test exercises the full
fork+exec path on hosts without AppArmor (CI, dev workstations).
The §2.6 integration test covers the apparmor-on path.
"""

from __future__ import annotations

import asyncio
import os
import socket
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


# ----------------------------------------------------------------------
# #284/#280 — slot teardown sweeps the slot's process-group subtree
# (jdtls / mcp / PTY children) instead of orphaning it to the daemon.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_leads_own_session() -> None:
    """The runner calls ``os.setsid()`` at fork, so it leads its own
    session — pgid == its own pid, distinct from the daemon's group.
    This is the precondition that makes the killpg subtree-sweep safe
    (it can never hit the daemon's group)."""
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    try:
        assert os.getpgid(spawned.pid) == spawned.pid
        assert os.getpgid(spawned.pid) != os.getpgrp()
    finally:
        await client.close(timeout=3)


@pytest.mark.asyncio
async def test_close_sweeps_slot_subprocess_group() -> None:
    """A long-lived subprocess the slot spawned (stand-in for jdtls) is
    SIGKILLed by ``close()``'s process-group sweep — not left to
    re-parent to the daemon subreaper and leak (the #284 OOM)."""
    spawner = RunnerSpawner()
    spawned, client = await _make_client(spawner)
    # Runner spawns a detached, long-lived background process; ``echo
    # $!`` returns its pid.  fds redirected so the cli shell returns
    # immediately rather than blocking on the child.
    env = await client.call(
        "tool.execute",
        {
            "name": "cli_based_tool",
            "args": {"command": "sleep 300 >/dev/null 2>&1 & echo $!"},
        },
    )
    assert env.ok is True
    child_pid = int(env.result["stdout"].strip())
    # The stand-in inherited the slot's process group.
    assert os.getpgid(child_pid) == spawned.pid

    await client.close(timeout=3)

    # The group sweep SIGKILLed it; wait up to ~2s for kill + reap.
    for _ in range(40):
        try:
            os.getpgid(child_pid)
        except (ProcessLookupError, OSError):
            break
        await asyncio.sleep(0.05)
    else:
        # Belt-and-suspenders cleanup so a failing test never leaks the
        # stand-in process.
        try:
            os.kill(child_pid, 9)
        except OSError:
            pass
        pytest.fail(f"slot subprocess {child_pid} survived close() sweep")


@pytest.mark.asyncio
async def test_list_group_members_finds_self() -> None:
    """``_list_group_members`` enumerates live pids of a process group
    from /proc — the instrumentation that proves which children the
    sweep kills."""
    members = RunnerRPCClient._list_group_members(os.getpgrp())
    pids = {pid for pid, _comm in members}
    assert os.getpid() in pids


@pytest.mark.asyncio
async def test_capture_slot_pgid_refuses_daemon_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pgid guard returns None — skipping the sweep — when the slot
    appears to share the daemon's own process group (setsid regressed),
    so ``killpg`` can never SIGKILL the daemon itself."""
    a, b = socket.socketpair()
    try:
        client = RunnerRPCClient(a, runner_pid=os.getpid())
        # Force "slot shares our (daemon) group".
        monkeypatch.setattr(os, "getpgid", lambda _pid: os.getpgrp())
        assert client._capture_slot_pgid() is None
    finally:
        a.close()
        b.close()
