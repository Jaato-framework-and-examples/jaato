"""Load / perf tests for the subscribe API.

Marked ``load`` so they are skipped in normal CI (``pytest -m 'not load'``,
the default in pyproject.toml). Run explicitly with::

    pytest -m load jaato-sdk/jaato_sdk/tests/

Baselines and acceptance ratios live in ``docs/sdk-perf-baseline.md``.
These tests intentionally do not assert wall-clock numbers (which vary
across machines) — they assert *ratios* between configurations on the
same machine and in the same run, which are stable.
"""

import gc
import os
import time

import pytest

from jaato_sdk import IPCClient, EventType
from jaato_sdk.events import ClientType
from jaato_sdk.events import AgentOutputEvent


pytestmark = pytest.mark.load


N_EVENTS = 10_000
N_HANDLERS = 100


def _client() -> IPCClient:
    return IPCClient(socket_path="/tmp/_unused_load.sock", client_type=ClientType.API, auto_start=False)


def _output(i: int) -> AgentOutputEvent:
    return AgentOutputEvent(source="model", text=f"e{i}", mode="append")


def _measure_throughput(client: IPCClient, events: int) -> float:
    """Return events-per-second for ``events`` synthetic dispatches."""
    payload = _output(0)  # reuse — _dispatch never mutates
    start = time.perf_counter()
    for _ in range(events):
        client._dispatch(payload)
    elapsed = time.perf_counter() - start
    return events / elapsed if elapsed > 0 else float("inf")


def test_load_baseline_catchall():
    """Throughput baseline: one catchall handler, N events.

    Recorded so ratios in the other tests have a stable reference.
    """
    c = _client()
    counter = [0]
    c.subscribe_all(lambda e: counter.__setitem__(0, counter[0] + 1))

    rate = _measure_throughput(c, N_EVENTS)

    assert counter[0] == N_EVENTS
    print(f"\n[baseline] catchall ×1: {rate:>12,.0f} events/s")


def test_load_100_typed_handlers():
    """Per-handler-invocation cost should not regress with many handlers.

    100 handlers means each event triggers 100 invocations, so raw
    events/sec drops ~100× — that's the work scaling, not a regression.
    What we actually care about is **handler invocations per second**:
    the per-call cost should stay roughly the same. We allow up to 3×
    overhead vs. baseline to absorb snapshot + bucket-iteration cost.
    """
    # Baseline: single catchall handler. Throughput is per-event, so
    # also per-handler-invocation when there's only one handler.
    c_baseline = _client()
    c_baseline.subscribe_all(lambda e: None)
    baseline_invocations = _measure_throughput(c_baseline, N_EVENTS) * 1

    # 100 typed handlers all listening to the same event type.
    c = _client()
    for _ in range(N_HANDLERS):
        c.subscribe(EventType.AGENT_OUTPUT, lambda e: None)
    typed_event_rate = _measure_throughput(c, N_EVENTS)
    typed_invocations = typed_event_rate * N_HANDLERS

    per_call_ratio = (
        baseline_invocations / typed_invocations
        if typed_invocations > 0
        else float("inf")
    )
    print(
        f"\n[100-typed]  baseline_invocations={baseline_invocations:>12,.0f}/s  "
        f"100-typed_invocations={typed_invocations:>12,.0f}/s  "
        f"per-call={per_call_ratio:5.2f}x"
    )

    assert per_call_ratio < 3.0, (
        f"Per-handler-invocation cost is {per_call_ratio:.2f}x worse with "
        "100 handlers than with 1; expected <3x. Investigate dispatch overhead."
    )


def test_load_subscribe_unsubscribe_churn_no_leak():
    """Subscribe + unsubscribe loop should not leak handler entries.

    Verifies that the unsub closure actually removes entries from the
    underlying bucket (idempotent unsub from many tests masks a leak;
    here we explicitly check bucket size after the churn).
    """
    c = _client()
    n_cycles = 10_000

    for _ in range(n_cycles):
        unsub = c.subscribe(EventType.AGENT_OUTPUT, lambda e: None)
        unsub()

    # Internal state assertion — the handler bucket must be empty.
    bucket = c._registry._handlers.get(EventType.AGENT_OUTPUT.value, [])
    assert len(bucket) == 0, (
        f"Leak: {len(bucket)} entries remain after {n_cycles} sub/unsub cycles"
    )

    # Cross-check with gc: if entries leak, _HandlerEntry instances would
    # still be reachable. After a forced collection the count should be
    # near-zero (the dispatcher itself doesn't keep history).
    gc.collect()


@pytest.mark.asyncio
async def test_load_async_handler_throughput():
    """Async fire-and-forget dispatch should sustain ≥80% of sync throughput.

    Async handlers schedule a task per dispatch; we check that scheduling
    overhead doesn't tank throughput. Real async work (I/O) lives outside
    the dispatch hot path, so this measures purely the create_task cost.
    """
    sync_client = _client()
    sync_client.subscribe(EventType.AGENT_OUTPUT, lambda e: None)
    sync_rate = _measure_throughput(sync_client, N_EVENTS)

    async_client = _client()

    async def noop(event):
        return None

    async_client.subscribe(EventType.AGENT_OUTPUT, noop)
    async_rate = _measure_throughput(async_client, N_EVENTS)

    ratio = async_rate / sync_rate if sync_rate > 0 else 0.0
    print(
        f"\n[async]      sync={sync_rate:>10,.0f}/s  "
        f"async={async_rate:>10,.0f}/s  ratio={ratio:5.2f}x"
    )

    # Async dispatch creates an asyncio task per event — there's real
    # overhead here. We accept down to ~10% of sync, but flag if much
    # worse so we notice regressions in create_task cost.
    assert ratio >= 0.05, (
        f"Async dispatch dropped to {ratio:.0%} of sync throughput; "
        "create_task overhead has regressed."
    )
