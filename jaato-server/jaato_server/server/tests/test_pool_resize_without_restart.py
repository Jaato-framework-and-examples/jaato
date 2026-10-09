"""Resizing the runner pool on a running daemon (protocol 1.35).

The pool's floor and ceiling were read once, from
``JAATO_RUNNER_POOL_SIZE`` / ``JAATO_RUNNER_POOL_MAX_SIZE``, when the daemon
started; adding warm runners meant a restart, which unloads every session.
The replenish loop already re-reads both numbers on every pass, so the
mechanism is small.  What each test here holds:

* a larger floor is acted on NOW, not after the loop's next pause;
* a smaller one drops idle slots only, unreserved before reserved, and
  hands them to the replenish thread instead of blocking the caller;
* a shrink to 0 is still reaped, although startup never starts the thread
  for a disabled pool;
* only the daemon's own account or root may resize, decided from the
  kernel-reported peer and never from the request;
* the IPC transport routes the request for a client attached to no session
  (``--pool-size`` attaches to none);
* ``--restart`` keeps the new sizes.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    PoolStatusEvent,
    PoolStatusRequest,
    SystemMessageEvent,
)
from jaato_server.server.pool_admin import (
    INVALID_REQUEST,
    NOT_AUTHORIZED,
    PoolAdmin,
    parse_resize_args,
    pool_admin_refusal,
)
from jaato_server.server.runner_pool import PoolManager, PoolSlot
from jaato_server.shared.tests.reversion import Reversion

_POOL = "jaato-server/jaato_server/server/runner_pool.py"
_ADMIN = "jaato-server/jaato_server/server/pool_admin.py"
_IPC = "jaato-server/jaato_server/server/ipc.py"
_MAIN = "jaato-server/jaato_server/server/__main__.py"


def _slot(pid: int, cascade_id: Optional[str] = None,
          last_end_ts: Optional[float] = None) -> PoolSlot:
    return PoolSlot(pid=pid, sock=MagicMock(name=f"sock_{pid}"),
                    cascade_id=cascade_id, last_session_end_ts=last_end_ts)


def _template() -> MagicMock:
    tm = MagicMock(name="template")
    tm.is_alive.return_value = True
    pids = iter(range(5000, 6000))
    tm.request_fork_slot.side_effect = lambda: (next(pids), MagicMock())
    return tm


def _pool(target: int, max_size: Optional[int] = None,
          interval: float = 0.5) -> PoolManager:
    return PoolManager(template_manager=_template(), target_size=target,
                       max_size=max_size, replenish_interval=interval)


def _no_thread(pool: PoolManager) -> List[str]:
    """Record thread starts instead of starting one."""
    started: List[str] = []
    pool._start_replenish_thread = lambda: started.append("started")  # type: ignore[method-assign]
    return started


# --------------------------------------------------------------------------
# Growing
# --------------------------------------------------------------------------


def test_a_larger_target_is_forked_now_not_after_the_next_pause() -> None:
    """The loop sleeps ``replenish_interval`` once the pool is full.

    With a 60 s interval, a resize that only changed the number would wait
    out that pause before forking -- the operator asks for runners and sees
    none for a minute.  ``resize`` wakes the loop.
    """
    pool = _pool(target=1, interval=60.0)
    pool._idle_slots = [_slot(1)]
    pool.start_replenishment()
    try:
        time.sleep(0.3)        # the loop is now inside its 60 s pause
        pool.resize(3)
        deadline = time.monotonic() + 5.0
        while pool.unreserved_idle_count() < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pool.unreserved_idle_count() == 3
    finally:
        pool.stop_replenishment()


def test_resizing_a_disabled_pool_starts_the_thread() -> None:
    """A daemon booted with ``JAATO_RUNNER_POOL_SIZE=0`` has no thread.

    ``start_replenishment`` refuses a disabled pool, so a resize that went
    through it would set the number and fork nothing, ever.
    """
    pool = _pool(target=0)
    pool.start_replenishment()
    assert pool.snapshot()["replenishing"] is False
    try:
        pool.resize(2)
        assert pool.snapshot()["replenishing"] is True
    finally:
        pool.stop_replenishment()


# --------------------------------------------------------------------------
# Shrinking
# --------------------------------------------------------------------------


def test_a_shrink_drops_unreserved_idle_slots_and_keeps_reservations() -> None:
    pool = _pool(target=3, max_size=10)
    started = _no_thread(pool)
    pool._idle_slots = [_slot(1), _slot(2), _slot(3),
                        _slot(4, cascade_id="B", last_end_ts=time.monotonic())]

    change = pool.resize(1)

    assert change["queued_for_teardown"] == 2
    assert [s.pid for s in pool._idle_slots] == [1, 4]
    # Queued, not torn down here: the caller is usually the daemon loop,
    # and ``_teardown_slot`` blocks on it.
    assert sorted(s.pid for s in pool._pending_teardown) == [2, 3]
    assert all(s.teardown_reason == "pool-resize"
               for s in pool._pending_teardown)
    assert started, "the replenish thread is what reaps the queue"


def test_a_lower_ceiling_spends_the_stalest_reservation() -> None:
    pool = _pool(target=1, max_size=10)
    _no_thread(pool)
    now = time.monotonic()
    pool._idle_slots = [_slot(1),
                        _slot(2, cascade_id="old", last_end_ts=now - 100),
                        _slot(3, cascade_id="new", last_end_ts=now)]

    pool.resize(1, max_size=2)

    assert [s.pid for s in pool._idle_slots] == [1, 3]


def test_a_shrink_to_zero_is_still_reaped() -> None:
    """At target 0 the startup path would not start the thread; the queue
    of dropped slots would then never drain and the runners would live on
    with nothing in the pool to account for them."""
    pool = _pool(target=1)
    pool._idle_slots = [_slot(1)]
    try:
        pool.resize(0)
        assert pool.snapshot()["replenishing"] is True
    finally:
        pool.stop_replenishment()


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------


def test_a_derived_ceiling_follows_the_floor() -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    pool.resize(5)
    assert (pool.target_size, pool.max_size) == (5, 10)


def test_a_chosen_ceiling_is_kept_and_clamped_up() -> None:
    pool = _pool(target=2, max_size=6)
    _no_thread(pool)
    pool.resize(3)
    assert pool.max_size == 6
    pool.resize(8)
    assert pool.max_size == 8


@pytest.mark.parametrize("bad", [-1, True, "3", 2.5])
def test_a_size_that_is_not_a_non_negative_integer_is_refused(bad: Any) -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    with pytest.raises(ValueError):
        pool.resize(bad)
    assert pool.target_size == 2


# --------------------------------------------------------------------------
# Who may resize
# --------------------------------------------------------------------------


def _peer(uid: int) -> SimpleNamespace:
    return SimpleNamespace(uid=uid, identity=f"uid:{uid}")


def test_only_the_daemons_account_or_root_may_manage_the_pool() -> None:
    assert pool_admin_refusal(_peer(1000), daemon_uid=1000) is None
    assert pool_admin_refusal(_peer(0), daemon_uid=1000) is None
    assert pool_admin_refusal(_peer(1001), daemon_uid=1000)
    # No kernel-vouched account (WS, a ticket, a Windows pipe) is never
    # read as permission.
    assert pool_admin_refusal(None, daemon_uid=1000)


def test_a_refused_peer_changes_nothing() -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    other = _peer(os.getuid() + 4242)
    answer = PoolAdmin(pool, routing_enabled=lambda: True).answer(
        other, target_size=9)
    assert (answer.ok, answer.category) == (False, NOT_AUTHORIZED)
    assert pool.target_size == 2


def test_the_answer_reports_the_resize_and_whether_restart_keeps_it() -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    recorded: List[Any] = []
    admin = PoolAdmin(pool, routing_enabled=lambda: True,
                      on_resized=lambda t, m: recorded.append((t, m)) or True)

    answer = admin.answer(_peer(os.getuid()), request_id="r1", target_size=4)

    assert answer.ok and answer.changed and answer.persisted
    assert (answer.previous_target_size, answer.target_size) == (2, 4)
    assert answer.request_id == "r1"
    # A derived ceiling is recorded as None, so a restart derives it again.
    assert recorded == [(4, None)]


def test_typed_resize_arguments() -> None:
    assert parse_resize_args(["6"]) == (6, None)
    assert parse_resize_args(["2", "5"]) == (2, 5)
    for bad in ([], ["x"], ["-1"], ["1", "2", "3"]):
        with pytest.raises(ValueError):
            parse_resize_args(bad)


# --------------------------------------------------------------------------
# The router and the transport
# --------------------------------------------------------------------------


class _Sink:
    def __init__(self, peer: Any) -> None:
        self.peer = peer
        self.sent: List[Any] = []

    def send_event(self, client_id: str, event: Any) -> None:
        self.sent.append(event)

    def get_client_peer(self, client_id: str) -> Any:
        return self.peer


def _router(peer: Any, pool: PoolManager):
    from jaato_server.server.command_router import CommandRouter
    router = CommandRouter.__new__(CommandRouter)
    router._event_sink = _Sink(peer)
    router.set_pool_admin(PoolAdmin(pool, routing_enabled=lambda: True))
    return router


def test_the_request_is_answered_with_its_request_id() -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    router = _router(_peer(os.getuid()), pool)

    router._handle_pool_status_request(
        "ipc_1", PoolStatusRequest(request_id="pool_x", target_size=3), None)

    (answer,) = router._event_sink.sent
    assert isinstance(answer, PoolStatusEvent)
    assert (answer.request_id, answer.ok, answer.target_size) == ("pool_x", True, 3)


def test_the_typed_command_answers_with_the_event_and_a_line() -> None:
    pool = _pool(target=2)
    _no_thread(pool)
    router = _router(_peer(os.getuid()), pool)

    router._handle_pool_command("ipc_1", "pool.resize", ["nope"])

    answer, line = router._event_sink.sent
    assert (answer.ok, answer.category) == (False, INVALID_REQUEST)
    assert isinstance(line, SystemMessageEvent) and "usage" in line.message
    assert pool.target_size == 2


def test_the_ipc_transport_routes_it_for_a_sessionless_client() -> None:
    """``--pool-size`` attaches to no session.  The IPC server drops any
    request outside its sessionless allow-list for such a client, silently,
    and the caller waits out its timeout."""
    from jaato_server.server.ipc import _SESSIONLESS_REQUEST_TYPES
    assert isinstance(PoolStatusRequest(), _SESSIONLESS_REQUEST_TYPES)


# --------------------------------------------------------------------------
# --restart
# --------------------------------------------------------------------------


def test_a_resize_survives_restart(tmp_path, monkeypatch) -> None:
    from jaato_server.server.__main__ import JaatoDaemon
    monkeypatch.delenv("JAATO_RUNNER_POOL_SIZE", raising=False)
    monkeypatch.delenv("JAATO_RUNNER_POOL_MAX_SIZE", raising=False)
    config = tmp_path / "config.json"

    daemon = JaatoDaemon(config_file=str(config),
                         pid_file=str(tmp_path / "pid"))
    assert daemon._record_pool_size(5, None) is True
    saved = json.loads(config.read_text())
    assert (saved["pool_size"], saved["pool_max_size"]) == (5, None)

    restarted = JaatoDaemon(config_file=str(config),
                            pid_file=str(tmp_path / "pid"),
                            pool_size=saved["pool_size"],
                            pool_max_size=saved["pool_max_size"])
    assert (restarted._pool_manager.target_size,
            restarted._pool_manager.max_size) == (5, 10)


REVERSIONS = [
    Reversion(
        target=_POOL,
        find="""        self._start_replenish_thread()
        self._replenish_wake.set()
        return {"previous": previous, "current": current,""",
        replace="""        self._start_replenish_thread()
        return {"previous": previous, "current": current,""",
        test="test_a_larger_target_is_forked_now_not_after_the_next_pause",
        because=("a resize that only changes the number, leaving the loop "
                 "asleep for its whole interval before it forks"),
    ),
    Reversion(
        target=_POOL,
        find="""        self._start_replenish_thread()
        self._replenish_wake.set()
        return {"previous": previous, "current": current,""",
        replace="""        self.start_replenishment()
        self._replenish_wake.set()
        return {"previous": previous, "current": current,""",
        test="test_a_shrink_to_zero_is_still_reaped",
        because=("resize going through the startup gate, which refuses a "
                 "disabled pool, so slots dropped by a shrink to 0 are "
                 "never reaped"),
    ),
    Reversion(
        target=_POOL,
        find="""        unreserved = [s for s in self._idle_slots
                      if s.cascade_id is None
                      and (not s.has_served or self.target_size == 0)]""",
        replace="""        unreserved = list(self._idle_slots)""",
        test="test_a_shrink_drops_unreserved_idle_slots_and_keeps_reservations",
        because=("a shrink counting a cascade's reservation as spare "
                 "capacity and dropping it"),
    ),
    Reversion(
        target=_ADMIN,
        find="""    if uid in (own, 0):
        return None""",
        replace="""    if True:
        return None""",
        test="test_a_refused_peer_changes_nothing",
        because=("any local account on a shared socket resizing the "
                 "daemon's pool"),
    ),
    Reversion(
        target=_IPC,
        find="""    CommandRequest, ClientConfigRequest, PostAuthSetupResponse,
    PoolStatusRequest, PermissionResponseRequest,
) + MEMORY_REQUEST_TYPES""",
        replace="""    CommandRequest, ClientConfigRequest, PostAuthSetupResponse,
    PermissionResponseRequest,
) + MEMORY_REQUEST_TYPES""",
        test="test_the_ipc_transport_routes_it_for_a_sessionless_client",
        because=("the IPC server dropping --pool-size's request, which "
                 "carries no session, so the CLI times out"),
    ),
    Reversion(
        target=_MAIN,
        find="""            "pool_size": self._pool_size_override,
""",
        replace="",
        test="test_a_resize_survives_restart",
        because="--restart silently reverting a runtime resize",
    ),
]
