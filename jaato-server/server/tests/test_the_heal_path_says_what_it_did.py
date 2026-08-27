"""#633's heal path was unobservable, so its fix could not be PROVEN to work.

A consumer verified #633 from outside the daemon: forced the documented
cache-invalidation entrance (a live ``model select``), then counted
``percent_used=0`` readings across 111 progress events.  Zero sightings, zero
stalls.

That is a real result and it is not the one anybody wanted.  Zero
honest-unknown readings is consistent with TWO opposite things -- the heal
beat the first notification, or the miss path never executed at all -- and
nothing emitted from the daemon could separate them.  So the strongest
available claim was "no bad outcome occurred", not "the code under test ran
and behaved".

THAT WAS NOT A LIMIT OF EXTERNAL OBSERVATION.  It was this code declining to
testify: the miss branch logged nothing, ``_schedule_context_limit_fill`` had
three silent early returns, and its heal ended in a bare ``except Exception:
pass``.  Every outcome, including permanent failure, looked the same from
outside -- which is the defect class #633 itself was about, one notch quieter.

The tokens, all greppable:

    CONTEXT_LIMIT_MISS          the branch ran; emitting limit-unknown
    CONTEXT_LIMIT_HEALED        cache filled; next notification is fine
    CONTEXT_LIMIT_HEAL_EMPTY    provider reports no window (#541 honest-zero)
    CONTEXT_LIMIT_HEAL_FAILED   the cache stays COLD -- the state that used
                                to be permanent and silent
    CONTEXT_LIMIT_HEAL_SKIPPED  declined before scheduling, with the reason
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from server.core import JaatoServer


class _Loop:
    """A real, running event loop on its own thread.

    ``_schedule_context_limit_fill`` enqueues onto the runner RPC's loop and
    deliberately does NOT block on the future, so a fake loop object would
    never run the coroutine and every heal outcome would go untested.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)


class _RPC:
    def __init__(self, loop, limit=None, boom=None):
        self._loop = loop
        self._limit = limit
        self._boom = boom

    async def session_get_context_limit(self, timeout=None):
        if self._boom:
            raise self._boom
        return self._limit


def _server(rpc):
    s = JaatoServer.__new__(JaatoServer)
    s._session_id = "sess-probe"
    s._runner_rpc = rpc
    s._cached_context_limit = None
    s._context_limit_fill_inflight = False
    return s


@pytest.fixture
def caplog_warnings(caplog):
    caplog.set_level(logging.INFO, logger="server.core")
    return caplog


def _wait_for(caplog, token, timeout=2.0):
    """The fill is off-band, so poll rather than assert immediately."""
    deadline = asyncio.get_event_loop_policy().new_event_loop()
    deadline.close()
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(token in r.getMessage() for r in caplog.records):
            return True
        time.sleep(0.02)
    return False


def test_a_successful_heal_says_so_and_names_the_limit(caplog_warnings):
    lp = _Loop()
    try:
        server = _server(_RPC(lp.loop, limit=200000))
        assert server._schedule_context_limit_fill() is True
        assert _wait_for(caplog_warnings, "CONTEXT_LIMIT_HEALED")
        assert server._cached_context_limit == 200000
        assert "limit=200000" in "\n".join(
            r.getMessage() for r in caplog_warnings.records)
    finally:
        lp.stop()


def test_a_failed_heal_is_a_WARNING_that_names_the_exception_type(caplog_warnings):
    """The state that used to be permanent AND silent.

    A heal that keeps failing leaves the cache cold forever; the only symptom
    was ``percent_used=0`` on every event, with a bare ``except: pass`` where
    the reason should be.  ``str(TimeoutError())`` is the empty string, so the
    TYPE has to carry it.
    """
    lp = _Loop()
    try:
        server = _server(_RPC(lp.loop, boom=TimeoutError()))
        assert server._schedule_context_limit_fill() is True
        assert _wait_for(caplog_warnings, "CONTEXT_LIMIT_HEAL_FAILED")
        text = "\n".join(r.getMessage() for r in caplog_warnings.records)
        assert "TimeoutError" in text, (
            f"an empty str(exc) left the line naming nothing: {text!r}")
        assert server._cached_context_limit is None
        levels = {r.levelname for r in caplog_warnings.records
                  if "HEAL_FAILED" in r.getMessage()}
        assert levels == {"WARNING"}, (
            "the cache-stays-cold path must not be INFO — it is the one that "
            f"used to be permanent, got {levels}")
    finally:
        lp.stop()


def test_an_honest_zero_is_reported_as_honest_not_as_failure(caplog_warnings):
    """#541: a provider reporting 0 is saying it does not know.

    Caching that would turn an honest unknown into a wrong denominator, so the
    heal correctly declines — and must say it declined ON PURPOSE, or the next
    reader diagnoses a broken heal.
    """
    lp = _Loop()
    try:
        server = _server(_RPC(lp.loop, limit=0))
        assert server._schedule_context_limit_fill() is True
        assert _wait_for(caplog_warnings, "CONTEXT_LIMIT_HEAL_EMPTY")
        assert server._cached_context_limit is None
        text = "\n".join(r.getMessage() for r in caplog_warnings.records)
        assert "CONTEXT_LIMIT_HEAL_FAILED" not in text, (
            "an honest zero was reported as a failure")
    finally:
        lp.stop()


def test_declining_before_scheduling_names_the_reason(caplog_warnings):
    server = _server(None)                       # no runner rpc yet
    assert server._schedule_context_limit_fill() is False
    text = "\n".join(r.getMessage() for r in caplog_warnings.records)
    assert "CONTEXT_LIMIT_HEAL_SKIPPED" in text
    assert "reason=no_runner_rpc" in text


def test_single_flight_returns_False_so_the_caller_logs_once(caplog_warnings):
    """A stampede must produce ONE miss line, not one per notification.

    The caller logs only when this returns True, so the return value is the
    de-duplication — and a second concurrent miss must not report itself as a
    fresh one.
    """
    lp = _Loop()
    try:
        server = _server(_RPC(lp.loop, limit=1000))
        server._context_limit_fill_inflight = True      # a fill already in flight
        assert server._schedule_context_limit_fill() is False
        assert not any("CONTEXT_LIMIT" in r.getMessage()
                       for r in caplog_warnings.records), (
            "an already-in-flight heal announced itself as a new miss")
    finally:
        lp.stop()


def test_both_notification_hooks_log_the_miss():
    """Checked in the SOURCE: the two hooks are copies, and a fix applied to
    one of them is the way this regresses.

    Read as an AST rather than by grepping text, so a mention inside a comment
    cannot satisfy it — and every ``_schedule_context_limit_fill`` call is
    required to be inside an ``if``, which is what makes the logging
    conditional on a fill actually being scheduled.
    """
    import ast
    import pathlib

    src = pathlib.Path("jaato-server/server/core.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    guarded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = [n for n in ast.walk(node.test)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "_schedule_context_limit_fill"]
        if not calls:
            continue
        body_text = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        assert "CONTEXT_LIMIT_MISS" in body_text, (
            "a miss branch schedules a heal without announcing it; the branch "
            "becomes unobservable again and #633 goes back to unprovable")
        guarded += 1

    assert guarded == 2, (
        f"expected both notification hooks to guard-and-log, found {guarded}")

    # And no bare call survives outside an ``if`` — that would be a hook that
    # heals silently.
    bare = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_schedule_context_limit_fill"
    ]
    assert not bare, (
        f"{len(bare)} unguarded _schedule_context_limit_fill call(s): the miss "
        "would heal without saying it happened")
