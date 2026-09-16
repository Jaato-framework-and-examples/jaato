"""A parent that looks at a freshly-forked child is reading a race (#996).

``RunnerSpawner.spawn`` creates the socketpair BEFORE ``os.fork()``, so
both ends are connected the instant the parent returns -- and
``RunnerRPCClient.start()`` only adopts the parent end onto the loop and
begins reading.  Neither performs a handshake.  A "connected" client is
therefore connected to a child that may not have executed a single
instruction.

Everything the child does after the fork is concurrent with the parent:
``os.setsid()``, the cgroup migration, then ``exec()``.  So a parent that
reads the child's state immediately is reading a process caught
mid-startup, and under load it loses.  Measured on a 4-core host under
6-way CPU load, ``os.getpgid(child)`` returned **the daemon's own process
group**, settling to the child's pid 3.4 ms later:

    child=16633 daemon_pgrp=16624 first_read=16624  settled after 3.4ms

``test_runner_leads_own_session`` asserted exactly that pgid, which is
why it failed 4 of 6 runs under load on clean ``main`` while passing
when idle.

THE FIX IS AN ORDERING, NOT A MARGIN.  ``_make_client`` performs one
``echo`` round trip before returning.  The runner can only answer after
``exec()``, which is after ``setsid()`` and the cgroup attach, so a reply
proves -- by the wire, with no clock -- that the child is past its whole
pre-exec sequence.  #988 in this same file established what the
alternative buys: widening a deadline from 200 to 2000 iterations made
each failure ten times more expensive and caught nothing extra.

This module guards the ordering rather than re-running the race.  A
probabilistic failure makes a poor regression test -- it passes most of
the time by definition -- so what is pinned here is the *structure* that
removes the race: the helper does not hand back a client it has not
heard from.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

_CLIENT_TESTS = (
    Path(__file__).resolve().parents[1] / "test_runner_rpc_client.py"
)


def _make_client_body() -> List[ast.stmt]:
    """The statements of ``_make_client``, or fail saying it moved."""
    assert _CLIENT_TESTS.is_file(), (
        f"{_CLIENT_TESTS} does not exist. The helper this guards moved; "
        f"point this module at its new home rather than deleting it -- "
        f"the ordering it pins is what stops #996 recurring."
    )
    tree = ast.parse(_CLIENT_TESTS.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                and node.name == "_make_client"):
            return node.body
    raise AssertionError(
        "_make_client is gone from test_runner_rpc_client.py. Every test "
        "in that file gets its runner from it, so whatever replaced it "
        "must establish the same ordering: no assertion about a forked "
        "child before that child has answered something."
    )


def _index_of_call(body: List[ast.stmt], attr: str) -> Optional[int]:
    """Position of the first ``<something>.attr(...)`` in *body*."""
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == attr):
                return i
    return None


def test_make_client_hears_from_the_runner_before_returning() -> None:
    """The helper round-trips, and does it after ``start()``.

    Both halves matter.  A ``call`` BEFORE ``start()`` cannot work -- the
    read loop is not running -- and a ``call`` that is not there at all
    is the pre-#996 helper, which returned a client connected to a child
    that had possibly not run yet.
    """
    body = _make_client_body()
    start_at = _index_of_call(body, "start")
    call_at = _index_of_call(body, "call")

    assert start_at is not None, (
        "_make_client no longer calls client.start(); the read loop is "
        "what makes any round trip possible"
    )
    assert call_at is not None, (
        "_make_client returns a client without ever hearing from the "
        "runner.\n\n"
        "The socketpair is created before fork(), so 'connected' says "
        "nothing about whether the child has run. A test that then reads "
        "the child's state -- its pgid, its cgroup, that it exec'd -- is "
        "betting on the scheduler, and under load it loses (#996).\n\n"
        "Restore the round trip: `await client.call(\"echo\", {...})` "
        "after start(). The runner can only answer after exec(), which "
        "is after setsid() and the cgroup attach."
    )
    assert call_at > start_at, (
        "_make_client issues an RPC before client.start(), so no read "
        "loop is running to receive the reply"
    )


def test_the_returned_client_is_handed_back_after_the_round_trip() -> None:
    """The round trip precedes the ``return``, not merely exists.

    The control for the check above: a ``call`` placed after the return
    statement would satisfy "a call exists" and establish nothing.  This
    is what makes the pair non-vacuous rather than a search for a token
    anywhere in the function.
    """
    body = _make_client_body()
    call_at = _index_of_call(body, "call")
    returns = [i for i, s in enumerate(body) if isinstance(s, ast.Return)]

    assert returns, "_make_client does not return; it cannot be a helper"
    assert call_at is not None and call_at < min(returns), (
        "the RPC round trip in _make_client does not precede its return, "
        "so callers still receive a client that may not have heard from "
        "its runner"
    )


# ---------------------------------------------------------------------------
# Reversions -- the meta-suite (test_every_guard_detects_its_own_reversion)
# discovers this list by name and asserts each one makes the NAMED test
# fail.  `test` is the nodeid WITHIN this module, and `replace` must
# produce source that still COMPILES (#1065).
# ---------------------------------------------------------------------------
from shared.tests.test_every_guard_detects_its_own_reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target="jaato-server/server/test_runner_rpc_client.py",
        find='    await client.call("echo", {"ready": True})\n',
        replace="",
        because=(
            "the pre-#996 helper, which returned a 'connected' client "
            "for a child that may not have run os.setsid() yet -- so "
            "every test reading the child's state was betting on the "
            "scheduler, and test_runner_leads_own_session lost that bet "
            "4 of 6 runs under load"
        ),
        test="test_make_client_hears_from_the_runner_before_returning",
    ),
]
