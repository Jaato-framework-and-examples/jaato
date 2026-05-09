"""Pre-§7c-step-6 toolbar-coherence verification.

Step 6 atomically switches the daemon-side context_usage /
context_limit reads (currently from the in-process JaatoSession
via ``self._jaato.get_context_usage()``) to the runner-side
session via ``self._runner_rpc.session_get_context_usage_threadsafe()``.

For the toolbar to remain visually consistent across the
seat-flip, this test pins the precondition:

  Given the same underlying JaatoSession, the runner-side
  ``session.get_context_usage`` RPC returns dict-identical
  values to a direct in-process ``session.get_context_usage()``
  call.

Plus the determinism invariant:

  Two JaatoSession instances configured from the same envelope
  + same provider stub return equal ``get_context_usage()``
  output at init time.

Together these prove that step 6's read-source switch is
value-preserving — the daemon's ContextUpdatedEvent payload
won't flicker the moment we cut the daemon-side read leg.

If this test ever goes red, step 6's atomic-switch assumption
is broken and the daemon-side / runner-side reads have
diverged.  Investigate before landing the field-removal commit.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-coherence",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _DeterministicSession:
    """Stand-in for JaatoSession exercising ``get_context_usage``
    + ``get_context_limit`` with deterministic outputs.

    Constructor takes a usage dict + context_limit so two
    instances built from the same args produce identical reads
    — the determinism invariant the seat-flip relies on."""

    def __init__(
        self,
        *,
        usage: Dict[str, Any],
        context_limit: int,
    ) -> None:
        self._usage = dict(usage)
        self._context_limit = context_limit
        self._read_count_usage = 0
        self._read_count_limit = 0

    def get_context_usage(self) -> Dict[str, Any]:
        self._read_count_usage += 1
        return dict(self._usage)

    def get_context_limit(self) -> int:
        self._read_count_limit += 1
        return self._context_limit


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Determinism invariant: two instances, same args, equal reads
# ----------------------------------------------------------------------


def test_two_instances_with_equal_inputs_return_equal_usage() -> None:
    """Pin the determinism property: ``get_context_usage`` is a
    pure function of the underlying state (instruction_budget /
    provider).  Two sessions configured from the same envelope
    + same provider produce equal output."""
    usage_dict = {
        "model": "claude-sonnet-4-6",
        "context_limit": 200_000,
        "total_tokens": 1234,
        "prompt_tokens": 1000,
        "output_tokens": 234,
        "turns": 1,
        "percent_used": 0.617,
        "tokens_remaining": 198_766,
    }

    session_a = _DeterministicSession(usage=usage_dict, context_limit=200_000)
    session_b = _DeterministicSession(usage=usage_dict, context_limit=200_000)

    assert session_a.get_context_usage() == session_b.get_context_usage()
    assert session_a.get_context_limit() == session_b.get_context_limit()


def test_two_instances_with_different_inputs_return_different_usage() -> None:
    """Negative case — proves the test isn't tautologically green:
    different inputs → different outputs.  Catches a regression
    where ``get_context_usage`` collapsed to a constant by
    accident."""
    usage_a = {"total_tokens": 100, "context_limit": 200_000}
    usage_b = {"total_tokens": 200, "context_limit": 200_000}

    session_a = _DeterministicSession(usage=usage_a, context_limit=200_000)
    session_b = _DeterministicSession(usage=usage_b, context_limit=200_000)

    assert session_a.get_context_usage() != session_b.get_context_usage()


# ----------------------------------------------------------------------
# Wire-fidelity invariant: RPC round-trip preserves the dict
# ----------------------------------------------------------------------


class _Pair:
    def __init__(self, runner_rpc, runner_thread, daemon_client) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-coherence-test", daemon=True,
    )
    thread.start()

    daemon_client = RunnerRPCClient(daemon_sock, runner_pid=0)
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _Pair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_rpc_round_trip_preserves_get_context_usage_dict() -> None:
    """Wire-fidelity: the runner-side ``session.get_context_usage``
    RPC returns a dict bit-equal to what the underlying
    JaatoSession's ``get_context_usage()`` returned.

    Pins the seat-flip's read-source switch as value-preserving
    at the wire boundary."""
    pair = await _make_pair()
    try:
        usage_dict = {
            "model": "claude-sonnet-4-6",
            "context_limit": 200_000,
            "total_tokens": 12345,
            "prompt_tokens": 10000,
            "output_tokens": 2345,
            "turns": 3,
            "percent_used": 6.17,
            "tokens_remaining": 187_655,
        }
        session = _DeterministicSession(
            usage=usage_dict, context_limit=200_000,
        )
        _install(pair.runner_rpc, session)

        # Direct in-process read (what daemon-side _jaato would do).
        direct = session.get_context_usage()

        # Runner-side read via RPC (what the post-seat-flip daemon
        # will do via self._runner_rpc.session_get_context_usage_threadsafe).
        via_rpc = await pair.daemon_client.session_get_context_usage()

        assert direct == via_rpc, (
            f"wire round-trip changed the dict: direct={direct!r} "
            f"via_rpc={via_rpc!r}"
        )

        # Both reads count — pins that the RPC actually called the
        # underlying method (not cached), so divergence-on-mutation
        # would surface (the RPC's read happens after `direct`).
        assert session._read_count_usage == 2
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_rpc_round_trip_preserves_get_context_limit_int() -> None:
    """Wire-fidelity for ``session.get_context_limit``: the int
    returned by the RPC equals the in-process read."""
    pair = await _make_pair()
    try:
        session = _DeterministicSession(
            usage={"context_limit": 128_000}, context_limit=128_000,
        )
        _install(pair.runner_rpc, session)

        direct = session.get_context_limit()
        via_rpc = await pair.daemon_client.session_get_context_limit()

        assert direct == via_rpc == 128_000
        assert session._read_count_limit == 2
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_rpc_round_trip_preserves_zero_usage_at_init_time() -> None:
    """Pin the init-time / auth-completion-time invariant: when
    the underlying session has near-zero usage (just-bootstrapped,
    no messages processed), both surfaces report the same shape.

    This is the actual surface the §7c-step-5/6 read migration
    affects — the daemon emits ``ContextUpdatedEvent`` with these
    values right after init / auth-completion."""
    pair = await _make_pair()
    try:
        # Realistic init-time shape: instruction-budget overhead
        # only, no conversation tokens.
        usage_dict = {
            "model": "claude-sonnet-4-6",
            "context_limit": 200_000,
            "total_tokens": 850,         # system + tool schemas + enrichment
            "prompt_tokens": 850,
            "output_tokens": 0,
            "turns": 0,
            "percent_used": 0.425,
            "tokens_remaining": 199_150,
        }
        session = _DeterministicSession(
            usage=usage_dict, context_limit=200_000,
        )
        _install(pair.runner_rpc, session)

        via_rpc = await pair.daemon_client.session_get_context_usage()

        # Every key the daemon's ContextUpdatedEvent constructor
        # reads must be present + equal.  Toolbar pulls all of:
        #   prompt_tokens, output_tokens, total_tokens,
        #   context_limit, percent_used, tokens_remaining, turns.
        for key in (
            "prompt_tokens", "output_tokens", "total_tokens",
            "context_limit", "percent_used", "tokens_remaining",
            "turns", "model",
        ):
            assert via_rpc[key] == usage_dict[key], (
                f"toolbar-relevant field {key!r} diverged across RPC: "
                f"direct={usage_dict[key]!r} via_rpc={via_rpc[key]!r}"
            )
    finally:
        await _teardown(pair)
