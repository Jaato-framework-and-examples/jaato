"""§8.3 RPC-overhead acceptance gate.

Asserts the runner-RPC tool-call overhead is ≤ 5 ms p50 measured
against the ``echo`` tool (the cheapest possible runner-side
executor: pure Python, no subprocess).  This isolates the RPC
machinery — the cli ``run_command`` path adds subprocess fork+exec
which dwarfs the channel cost.

Runs on any host (no AppArmor required) — uses
``JAATO_RUNNER_DISABLE_CONFINE=1`` so we measure the framing /
multiplex / asyncio overhead in isolation, not kernel transition cost.

The Phase 2 acceptance gate (per the prompt's "Phase 2 acceptance
gate" table) is ≤ 5 ms p50.  The 200 ms p99 spawn-latency target is
Phase 5; this test only measures per-call overhead on a warm runner.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import List

import pytest

from server.runner_rpc_client import RunnerRPCClient
from server.runner_spawner import RunnerSpawner


PHASE2_RPC_P50_BUDGET_SECONDS = 0.005  # 5 ms

WARMUP_CALLS = 20
MEASURE_CALLS = 200


@pytest.mark.asyncio
async def test_rpc_echo_p50_within_budget() -> None:
    """Round-trip 200 echo calls; assert p50 latency ≤ 5 ms."""
    spawner = RunnerSpawner()
    spawned = spawner.spawn(
        profile_name="phase2-rpc-overhead",
        session_id="phase2-rpc-overhead",
        workspace_path="/tmp",
        disable_confine=True,
    )
    client = RunnerRPCClient(spawned.parent_socket, runner_pid=spawned.pid)
    await client.start()

    try:
        # Warmup so JIT-style first-call costs (asyncio task creation,
        # ctypes module load on the runner side, etc.) don't pollute
        # the measurement.
        for _ in range(WARMUP_CALLS):
            await client.call("echo", {"k": "warmup"})

        latencies_s: List[float] = []
        for _ in range(MEASURE_CALLS):
            t0 = time.perf_counter()
            env = await client.call("echo", {"k": "v"})
            latencies_s.append(time.perf_counter() - t0)
            assert env.ok is True

        p50 = statistics.median(latencies_s)
        p99 = sorted(latencies_s)[int(0.99 * len(latencies_s))]
        mean = statistics.mean(latencies_s)

        # Surface the actual numbers so test failures explain
        # themselves without re-running.
        print(
            f"\n[phase2-rpc-overhead] echo n={MEASURE_CALLS} "
            f"p50={p50*1000:.2f}ms p99={p99*1000:.2f}ms "
            f"mean={mean*1000:.2f}ms (budget p50={PHASE2_RPC_P50_BUDGET_SECONDS*1000:.0f}ms)"
        )
        assert p50 <= PHASE2_RPC_P50_BUDGET_SECONDS, (
            f"RPC echo p50 of {p50*1000:.2f}ms exceeds Phase 2 budget "
            f"of {PHASE2_RPC_P50_BUDGET_SECONDS*1000:.0f}ms"
        )
    finally:
        await client.close(timeout=2)
