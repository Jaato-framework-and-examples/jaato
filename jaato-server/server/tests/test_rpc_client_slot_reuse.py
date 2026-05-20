"""Pin test for the rpc-client-lives-on-slot fix.

PR #173 hotfix.  Phase 2/3 of cascade-sharing had a wiring bug:
``spawn_session_runner`` created a fresh ``RunnerRPCClient`` on
every session, including reused slots.  But the asyncio transport
adopted by ``RunnerRPCClient.start`` binds the slot's socket
exclusively.  Creating a second client on the same socket fails;
the session falls back to in-process and
``server._runner_rpc`` stays None.  Symptom (peer report
2026-05-20 v152-retry-4): ``AttributeError: 'NoneType' object has
no attribute 'session_send_message_threadsafe'`` on the
host_validator session (which reused the discovery slot).

Fix: the rpc lives on the ``PoolSlot``.  First session creates +
stashes it; subsequent sessions reuse it via
``reset_for_slot_reuse`` (clears per-session state, keeps the
transport bound).  Slot teardown (cascade-idle sweep) calls
``rpc.close()`` which closes the socket + reaps the runner.

These tests pin the lifecycle contract without spawning a real
runner subprocess.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.runner_pool import PoolSlot
from server.runner_rpc_client import RunnerRPCClient


# ======================================================================
# PoolSlot.rpc field
# ======================================================================


def test_pool_slot_has_rpc_field() -> None:
    """The new field is part of the PoolSlot dataclass."""
    slot = PoolSlot(pid=1, sock=MagicMock())
    assert slot.rpc is None  # default

    slot.rpc = MagicMock(spec=RunnerRPCClient)
    assert slot.rpc is not None


# ======================================================================
# RunnerRPCClient.reset_for_slot_reuse
# ======================================================================


def _make_rpc_with_mocked_socket() -> RunnerRPCClient:
    """RunnerRPCClient with a mock socket — start() not called, so
    no real asyncio transport is adopted."""
    fake_sock = MagicMock()
    rpc = RunnerRPCClient(
        sock=fake_sock,
        runner_pid=12345,
        loop=MagicMock(),
    )
    return rpc


def test_reset_for_slot_reuse_clears_in_flight() -> None:
    """In-flight RPC futures are failed with a clean error so any
    awaiter sees an exception (not a hang)."""
    import asyncio

    rpc = _make_rpc_with_mocked_socket()
    loop = asyncio.new_event_loop()
    try:
        fut = loop.create_future()
        rpc._in_flight[1] = fut
        rpc.reset_for_slot_reuse()
        assert rpc._in_flight == {}
        assert fut.done()
        with pytest.raises(Exception, match="reset for slot reuse"):
            fut.result()
    finally:
        loop.close()


def test_reset_for_slot_reuse_cancels_dispatch_tasks() -> None:
    """Daemon-side dispatch tasks (handlers for runner→daemon
    requests) are cancelled."""
    rpc = _make_rpc_with_mocked_socket()
    fake_task = MagicMock()
    fake_task.done.return_value = False
    rpc._dispatch_tasks[1] = fake_task
    rpc.reset_for_slot_reuse()
    fake_task.cancel.assert_called_once()
    assert rpc._dispatch_tasks == {}


def test_reset_for_slot_reuse_clears_callbacks() -> None:
    """Per-session stream + notification subscribers are dropped."""
    rpc = _make_rpc_with_mocked_socket()
    rpc._stream_cbs[1] = MagicMock()
    rpc._notification_cbs[1] = MagicMock()
    rpc.reset_for_slot_reuse()
    assert rpc._stream_cbs == {}
    assert rpc._notification_cbs == {}


def test_reset_for_slot_reuse_preserves_transport() -> None:
    """Reader / writer / read_task stay bound — the socket-level
    transport keeps the connection alive for the next session."""
    rpc = _make_rpc_with_mocked_socket()
    sentinel_reader = MagicMock(name="reader")
    sentinel_writer = MagicMock(name="writer")
    sentinel_task = MagicMock(name="read_task")
    rpc._reader = sentinel_reader
    rpc._writer = sentinel_writer
    rpc._read_task = sentinel_task

    rpc.reset_for_slot_reuse()

    assert rpc._reader is sentinel_reader
    assert rpc._writer is sentinel_writer
    assert rpc._read_task is sentinel_task

    # ``_closed`` flag stays False — the rpc is alive.
    assert rpc._closed is False


def test_reset_for_slot_reuse_does_not_close_socket() -> None:
    """The raw socket is NOT closed.  Verified by checking that
    ``close()`` on the writer (which would close the socket) is
    not invoked."""
    rpc = _make_rpc_with_mocked_socket()
    fake_writer = MagicMock()
    rpc._writer = fake_writer
    rpc.reset_for_slot_reuse()
    fake_writer.close.assert_not_called()
