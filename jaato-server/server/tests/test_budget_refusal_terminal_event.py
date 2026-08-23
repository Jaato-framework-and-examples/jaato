"""A budget refusal must reach the client as a TERMINAL, TYPED event.

The refusal short-circuits before any turn runs (``JaatoSession.send_message``
returns a string after a log line and a PROSE output chunk), so no
turn-completion notification fires and ``on_session_quiescent`` is never
reached.  A wake-driven driver had nothing to wait for: it waited out its full
300s timeout and then reported a generic failure, so a correct ceiling stop was
indistinguishable from a break (EXIT 1, not 2).

Reported live 2026-08-23 by a suspend/resume cascade, immediately after the
ceiling itself was fixed (#583) -- the ceiling held and the driver still could
not say why it stopped.
"""
from types import SimpleNamespace

import pytest

from server.core import JaatoServer
from jaato_sdk.events import EventType


RESULT = {
    "response": "[budget_exhausted (self-enforced: turns 100%)]",
    "budget_exhausted": True,
    "budget_exhausted_reason": "budget_exhausted (self-enforced: turns 100%)",
    "budget_usage": {"turns": 2.0, "usd": 0.26, "tokens": 246638.0},
}


def _server():
    """Minimal self -- the helper touches emit/session_id/_main_agent_id only."""
    emitted = []
    srv = SimpleNamespace(
        emit=emitted.append,
        session_id="20260823_142514",
        _main_agent_id="main",
        _terminal_reason=None,
    )
    return srv, emitted


def _call(result):
    srv, emitted = _server()
    fired = JaatoServer._emit_budget_refusal_if_exhausted(srv, result)
    return fired, emitted, srv


def test_refusal_emits_a_terminal_event():
    fired, emitted, _ = _call(RESULT)
    assert fired is True
    assert len(emitted) == 1
    assert emitted[0].type == EventType.SESSION_TERMINATED, (
        "the driver waits on a TERMINAL event; without one it sits out its "
        "whole timeout and reports a generic failure")


def test_reason_is_typed_not_prose():
    _, emitted, _ = _call(RESULT)
    assert emitted[0].reason == "budget_exhausted", (
        "a driver must branch on this instead of substring-matching the "
        "output stream -- the parse-the-log shape budgets exist to replace")


def test_details_carry_the_evidence():
    _, emitted, _ = _call(RESULT)
    details = emitted[0].details
    assert details["reason"] == RESULT["budget_exhausted_reason"]
    assert details["usage"]["turns"] == 2.0
    assert set(details["usage"]) == set(RESULT["budget_usage"])


def test_terminal_reason_is_stamped():
    _, _, srv = _call(RESULT)
    assert srv._terminal_reason == "budget_exhausted"


def test_a_normal_turn_emits_nothing():
    fired, emitted, srv = _call({"response": "all done"})
    assert fired is False
    assert emitted == []
    assert srv._terminal_reason is None


@pytest.mark.parametrize("result", [None, "text", {}, {"budget_exhausted": False}])
def test_non_refusal_results_are_inert(result):
    fired, emitted, _ = _call(result)
    assert fired is False
    assert emitted == []


def test_missing_reason_still_emits_a_usable_event():
    # budget_exhausted without prose: the terminal signal still matters.
    _, emitted, _ = _call({"budget_exhausted": True})
    assert emitted[0].reason == "budget_exhausted"
    assert emitted[0].details["reason"]
    assert emitted[0].details["usage"] == {}


# --------------------------------------------------------------- the relay

def test_send_wrapper_forwards_the_full_result_not_just_text():
    """The typed budget fields died at this boundary before ``on_result``.

    ``session_send_message`` returns ``str`` because every caller depends on
    it, so the runner's ``budget_exhausted*`` keys were dropped one function
    below the daemon that needed them.  Calls the REAL wrapper against a stub
    transport.
    """
    import asyncio
    from server.runner_rpc_client import RunnerRPCClient

    async def _go():
        seen = {}

        async def _call(method, args, **kw):
            return SimpleNamespace(ok=True, error=None, result=RESULT)

        stub = SimpleNamespace(call=_call)
        text = await RunnerRPCClient.session_send_message(
            stub, "hi", on_result=seen.update)
        return text, seen

    text, seen = asyncio.run(_go())
    assert text == RESULT["response"], "the str contract must be unchanged"
    assert seen.get("budget_exhausted") is True, (
        "the typed signal was dropped at the RPC boundary -- the daemon had "
        "the ceiling in hand and threw it away")
    assert seen["budget_usage"]["turns"] == 2.0


def test_relay_end_to_end_wrapper_into_emit():
    """Wrapper -> daemon helper, the two halves joined.

    Covers the shape of the production wiring (``on_result=_send_result.
    update`` then ``_emit_budget_refusal_if_exhausted(_send_result)``) without
    a live runner.  The single call-site line inside ``model_thread`` is not
    reachable from here -- that one was verified by a real suspend run.
    """
    import asyncio
    from server.runner_rpc_client import RunnerRPCClient

    srv, emitted = _server()
    collected = {}

    async def _go():
        async def _call(method, args, **kw):
            return SimpleNamespace(ok=True, error=None, result=RESULT)
        await RunnerRPCClient.session_send_message(
            SimpleNamespace(call=_call), "hi", on_result=collected.update)

    asyncio.run(_go())
    JaatoServer._emit_budget_refusal_if_exhausted(srv, collected)

    assert len(emitted) == 1
    assert emitted[0].reason == "budget_exhausted"
    assert emitted[0].details["usage"]["turns"] == 2.0
