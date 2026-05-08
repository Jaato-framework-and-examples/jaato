"""§8.4 runner-crash detection.

Acceptance gate: runner crash → daemon emits ``SessionFailedEvent``
within 100 ms of socket EOF.

Phase 2 scope: this test verifies the proxy property — that the
daemon-side ``RunnerRPCClient`` detects EOF on the parent socket
within the budget after the runner is killed.  The
``SessionFailedEvent`` plumbing (SIGCHLD watcher → SessionManager
→ event emission) is risk §6.3 in the plan and lands as part of
the Phase 2 task 2.3 wiring; if it isn't fully wired yet, this
test catches the regression at the channel layer where the
detection actually originates.

Runs on any host (no AppArmor required) — uses
``JAATO_RUNNER_DISABLE_CONFINE=1``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time

import pytest

from server.runner_rpc import RunnerCallError, RunnerRPCClient
from server.runner_spawner import RunnerSpawner


PHASE2_CRASH_DETECT_BUDGET_SECONDS = 0.100  # 100 ms


@pytest.mark.asyncio
async def test_runner_kill_detected_within_budget() -> None:
    """SIGKILL the runner; assert in-flight calls fail within 100 ms."""
    spawner = RunnerSpawner()
    spawned = spawner.spawn(
        profile_name="phase2-crash",
        session_id="phase2-crash",
        workspace_path="/tmp",
        disable_confine=True,
    )
    client = RunnerRPCClient(spawned.parent_socket, runner_pid=spawned.pid)
    await client.start()

    try:
        # Sanity: a normal call works first.
        env = await client.call("echo", {"k": "v"})
        assert env.ok is True

        # Issue a deliberately-blocking call (long sleep), then
        # SIGKILL the runner.  The in-flight call's future should
        # resolve (with an error) within the budget.
        async def _long_call() -> None:
            await client.call(
                "tool.execute",
                {
                    "name": "cli_based_tool",
                    "args": {"command": "sleep 60"},
                },
            )

        task = asyncio.create_task(_long_call())
        # Let the call land and the runner pick it up.
        await asyncio.sleep(0.05)

        kill_t = time.perf_counter()
        os.kill(spawned.pid, signal.SIGKILL)

        with pytest.raises((RunnerCallError, Exception)):
            await asyncio.wait_for(
                task,
                timeout=PHASE2_CRASH_DETECT_BUDGET_SECONDS * 5,  # 500 ms hard cap
            )
        detected_at = time.perf_counter()
        elapsed = detected_at - kill_t
        print(
            f"\n[phase2-crash-detect] elapsed={elapsed*1000:.1f}ms "
            f"(budget {PHASE2_CRASH_DETECT_BUDGET_SECONDS*1000:.0f}ms)"
        )
        assert elapsed <= PHASE2_CRASH_DETECT_BUDGET_SECONDS, (
            f"runner-crash detection took {elapsed*1000:.1f}ms; "
            f"§8.4 budget is {PHASE2_CRASH_DETECT_BUDGET_SECONDS*1000:.0f}ms"
        )
    finally:
        # close() will reap whatever's left of the runner via
        # waitpid; SIGKILL'd process becomes a zombie until reaped.
        await client.close(timeout=2)


@pytest.mark.asyncio
async def test_calls_made_after_crash_raise_clean_error() -> None:
    """After the runner is dead and detected, new ``call()`` invocations
    raise ``RunnerCallError`` rather than hang."""
    spawner = RunnerSpawner()
    spawned = spawner.spawn(
        profile_name="phase2-crash-2",
        session_id="phase2-crash-2",
        workspace_path="/tmp",
        disable_confine=True,
    )
    client = RunnerRPCClient(spawned.parent_socket, runner_pid=spawned.pid)
    await client.start()

    try:
        env = await client.call("echo", {})
        assert env.ok is True

        os.kill(spawned.pid, signal.SIGKILL)

        # Give the read loop time to detect EOF and clean up.
        await asyncio.sleep(0.2)

        with pytest.raises(RunnerCallError):
            await client.call("echo", {})
    finally:
        await client.close(timeout=2)
