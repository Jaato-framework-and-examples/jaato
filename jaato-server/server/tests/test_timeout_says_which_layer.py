"""A stacked timeout must say WHICH of the two fired.

Every threadsafe runner call has two timeouts:

    inner   _await_runner      -> asyncio.wait_for; the RUNNER did not answer
    outer   _result_from_loop  -> future.result;    the DAEMON LOOP did not
                                                    deliver the result

In this Python ``asyncio.TimeoutError``, ``concurrent.futures.TimeoutError``
and ``TimeoutError`` are THE SAME CLASS.  So #622's log line, which names
``type(exc).__name__``, printed ``TimeoutError`` for both and could not
distinguish a busy runner from a saturated daemon loop -- which is exactly the
question the perpetual-monologue bench was left holding after #624 failed to
eliminate its ``unreachable`` deliveries.

The TYPE must stay ``TimeoutError``: ipc.py, command_router.py,
session_manager.py:4786 and apparmor.py all catch it, and a new type would
stop being caught.  What changes is that the exception carries a MESSAGE --
``str(TimeoutError())`` is the empty string, which is why these rendered blank
wherever they were logged.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

import pytest

from server.runner_rpc_client import _await_runner, _result_from_loop


def test_the_two_timeout_classes_really_are_one():
    """The premise.  If this ever stops being true, the fix is unnecessary
    and this whole file should be revisited rather than left passing."""
    assert asyncio.TimeoutError is TimeoutError
    assert concurrent.futures.TimeoutError is TimeoutError


@pytest.mark.asyncio
async def test_inner_timeout_names_the_runner():
    async def _never():
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError) as caught:
        await _await_runner(_never(), 0.01, "session.offer_message")

    msg = str(caught.value)
    assert msg, "a bare TimeoutError renders as the empty string"
    assert "runner did not answer" in msg
    assert "session.offer_message" in msg
    assert "daemon loop" not in msg, "named the wrong layer"


def test_outer_timeout_names_the_daemon_loop():
    never_set = concurrent.futures.Future()

    with pytest.raises(TimeoutError) as caught:
        _result_from_loop(never_set, 0.01, "session.offer_message")

    msg = str(caught.value)
    assert msg, "a bare TimeoutError renders as the empty string"
    assert "daemon loop did not deliver" in msg
    assert "session.offer_message" in msg
    assert "runner did not answer" not in msg, "named the wrong layer"


def test_the_two_messages_are_distinguishable():
    """The whole point: one grep separates them."""
    loop = asyncio.new_event_loop()
    try:
        async def _never():
            await asyncio.sleep(10)

        inner = None
        try:
            loop.run_until_complete(
                _await_runner(_never(), 0.01, "m"))
        except TimeoutError as exc:
            inner = str(exc)
    finally:
        loop.close()

    outer = None
    try:
        _result_from_loop(concurrent.futures.Future(), 0.01, "m")
    except TimeoutError as exc:
        outer = str(exc)

    assert inner and outer and inner != outer, (
        "the two layers must not render identically -- that identity is the "
        "defect this fixes"
    )


def test_the_type_is_unchanged_so_existing_catches_still_fire():
    """Several call sites catch TimeoutError; a new type would slip past."""
    caught = False
    try:
        _result_from_loop(concurrent.futures.Future(), 0.01, "m")
    except TimeoutError:
        caught = True
    assert caught, "existing `except TimeoutError` handlers would stop working"


@pytest.mark.asyncio
async def test_no_timeout_means_no_wrapper_in_the_way():
    """``timeout=None`` must await the coroutine directly."""
    async def _quick():
        return "done"

    assert await _await_runner(_quick(), None, "m") == "done"


def test_a_result_that_arrives_is_returned_untouched():
    fut = concurrent.futures.Future()
    threading.Thread(target=lambda: fut.set_result("payload"), daemon=True).start()
    assert _result_from_loop(fut, 5.0, "m") == "payload"
