"""Handler subscribers see the raw ``ErrorEvent`` before it becomes an exception.

``_drain_loop`` calls ``_dispatch(event)`` -- the ``subscribe()`` handler
fan-out -- BEFORE putting the event on the iterator/``_await_session_info``
queues (``ipc.py``, the two blocks are adjacent and in that order).  So a
consumer that both subscribes to ``ERROR`` and calls ``create_session`` sees
the daemon's own ``ErrorEvent`` first, and the ``SessionRefused`` second.

WHY THIS IS PINNED NOW.  #635 made ``create_session`` raise, which gave that
ordering a consequence it did not have before: a driver can reach a verdict
from the ErrorEvent (for instance "this refusal was a budget ceiling, so exit
2, not 1") and then let the exception unwind, knowing its handler already ran.
An out-of-tree driver was found depending on exactly that, and nothing in the
suite held the order in place -- ``test_subscribe_api`` pins ordering AMONG
handlers, not handlers versus the create wait.

Swapping the two blocks would still pass every other test and would silently
invert a decision in someone else's process.  Reported by the consumer that
depends on it, not discovered here; pinned so the next reader of
``_drain_loop`` learns it from a failure rather than from a bug report.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from jaato_sdk import IPCClient, SessionRefused
from jaato_sdk.events import ErrorEvent, EventType


def _client():
    c = IPCClient.__new__(IPCClient)
    c._buffered_events, c._event_subscribers = [], []
    c._session_id, c._protocol_version, c._server_info = None, "1.3", {}
    c._server_protocol_version = "1.3"
    c.sent = []
    # The real handler registry, not a stand-in: ``subscribe()`` delegates to
    # it and ``_dispatch`` reads it, so a hand-rolled dict would test a
    # different object than production uses.
    from jaato_sdk.client._handler_registry import _HandlerRegistry
    c._registry = _HandlerRegistry()

    async def _capture(data):
        c.sent.append(json.loads(
            data.decode() if isinstance(data, bytes) else data))

    c._write_message = _capture

    async def _disc():
        return None

    c.disconnect = _disc
    return c


def _request_id(client):
    for msg in client.sent:
        rid = (msg.get("payload") or {}).get("request_id")
        if rid:
            return rid
    return None


@pytest.mark.asyncio
async def test_the_handler_runs_before_create_session_raises():
    """The order a driver's exit-code verdict depends on."""
    c = _client()
    seen: list[str] = []

    c.subscribe(EventType.ERROR, lambda e: seen.append(f"handler:{e.error_type}"))

    async def _feed():
        await asyncio.sleep(0.02)
        event = ErrorEvent(
            error="cascade budget has no headroom left",
            error_type="CascadeExhaustedError",
            recoverable=True,
            request_id=_request_id(c),
        )
        # Mirror _drain_loop's own order: dispatch, THEN fan out.
        c._dispatch(event)
        for q in list(c._event_subscribers):
            q.put_nowait(event)

    asyncio.create_task(_feed())

    with pytest.raises(SessionRefused) as excinfo:
        await c.create_session(name="n", timeout=5.0)
    seen.append("raised")

    assert seen == ["handler:CascadeExhaustedError", "raised"], (
        "the ERROR handler must run BEFORE create_session raises — a driver "
        f"that reads the refusal to pick an exit code depends on it; got {seen}"
    )
    # And the exception carries the same fact, so a consumer that does NOT
    # subscribe is not forced to.
    assert excinfo.value.error_type == "CascadeExhaustedError"


@pytest.mark.asyncio
async def test_a_subscriber_is_not_required_to_learn_the_cause():
    """The ordering is a convenience, not the only channel.

    A driver should be able to drop its ErrorEvent subscription entirely and
    still reach the same verdict from the exception — otherwise the ordering
    above becomes load-bearing rather than merely useful, and every consumer
    is locked into the subscribe() path.
    """
    c = _client()

    async def _feed():
        await asyncio.sleep(0.02)
        for q in list(c._event_subscribers):
            q.put_nowait(ErrorEvent(
                error="cascade budget has no headroom left",
                error_type="CascadeExhaustedError",
                recoverable=True,
                request_id=_request_id(c),
            ))

    asyncio.create_task(_feed())

    with pytest.raises(SessionRefused) as excinfo:
        await c.create_session(name="n", timeout=5.0)

    assert excinfo.value.error_type == "CascadeExhaustedError"
    assert "headroom" in str(excinfo.value)


def test_drain_loop_dispatches_before_it_fans_out():
    """Read from the SOURCE, because the runtime tests above construct the
    order by hand rather than exercising ``_drain_loop`` itself.

    A runtime test that feeds events manually proves what the CONSUMER sees
    given an order; it cannot prove ``_drain_loop`` produces that order.  This
    one can, and it is the line that would actually be edited.
    """
    import ast
    import inspect

    import jaato_sdk.client.ipc as ipc_mod

    src = inspect.getsource(ipc_mod.IPCClient._drain_loop)
    tree = ast.parse(src.lstrip())

    # Collect ALL of each, then compare extremes.  A stateful single pass over
    # ``ast.walk`` does NOT work here: walk yields breadth-first, not in source
    # order, so "the first put_nowait seen after _dispatch" is whatever the
    # traversal happens to reach and is true no matter how the source is
    # arranged.  Written that way first, this guard PASSED with the two blocks
    # swapped -- a test that cannot fail, guarding the one thing it exists for.
    def _lines_of(attr):
        return sorted(
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr
        )

    dispatch_lines = _lines_of("_dispatch")
    fanout_lines = _lines_of("put_nowait")

    assert dispatch_lines, "_drain_loop no longer dispatches to handlers"
    assert fanout_lines, "_drain_loop no longer fans out to subscriber queues"
    assert max(dispatch_lines) < min(fanout_lines), (
        "_drain_loop fans out to the iterator/create-wait queues BEFORE "
        "dispatching to subscribe() handlers.  That inverts the order an "
        "out-of-tree driver depends on to pick its exit code, and every "
        "other test in this suite still passes.  Verified by swapping the "
        "blocks: nothing else in the SDK suite noticed."
        f"  (_dispatch at {dispatch_lines}, put_nowait at {fanout_lines})"
    )
