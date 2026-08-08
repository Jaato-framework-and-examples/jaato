"""Tests for the daemon-side ClarificationRelayHandler (the
``client.request_clarification`` RPC relay).  Mirror of
``test_prompt_operator.py``."""

import asyncio

import pytest

from server.runner_rpc_handlers.clarification_relay import (
    ClarificationRelayHandler,
    register,
)


def _args(request_id="r1"):
    return {
        "request_id": request_id,
        "agent_id": "main",
        "tool_name": "request_clarification",
        "context": "why",
        "questions": [
            {
                "index": 1,
                "text": "Pick one",
                "question_type": "single_choice",
                "required": True,
                "choices": [{"text": "A"}, {"text": "B"}],
            }
        ],
    }


def test_handle_emits_batch_event_and_resolves():
    events = []
    h = ClarificationRelayHandler(emit_event=events.append)

    async def run():
        task = asyncio.ensure_future(h.handle(_args("r1")))
        await asyncio.sleep(0)  # let handle register + emit
        # The batch event was emitted to the connected client.
        assert len(events) == 1
        assert events[0].request_id == "r1"
        assert events[0].tool_name == "request_clarification"
        assert events[0].questions[0]["text"] == "Pick one"
        # Client answers arrive → resolve_response sets the future.
        assert h.resolve_response("r1", ["1"]) is True
        result = await task
        assert result == {"cancelled": False, "answers": ["1"]}

    asyncio.run(run())


def test_resolve_unknown_request_returns_false():
    h = ClarificationRelayHandler(emit_event=lambda e: None)
    assert h.resolve_response("nope", ["x"]) is False


def test_shutdown_fails_in_flight_with_clean_error():
    h = ClarificationRelayHandler(emit_event=lambda e: None)

    async def run():
        task = asyncio.ensure_future(h.handle(_args("r2")))
        await asyncio.sleep(0)
        h.shutdown()
        with pytest.raises(RuntimeError) as exc:
            await task
        assert "shutting down" in str(exc.value)

    asyncio.run(run())


def test_missing_request_id_raises():
    h = ClarificationRelayHandler(emit_event=lambda e: None)

    async def run():
        with pytest.raises(ValueError):
            await h.handle({"questions": []})

    asyncio.run(run())


def test_closed_handler_rejects():
    h = ClarificationRelayHandler(emit_event=lambda e: None)
    h.shutdown()

    async def run():
        with pytest.raises(RuntimeError):
            await h.handle(_args("r3"))

    asyncio.run(run())


def test_register_binds_method_name():
    calls = {}

    class FakeRPCServer:
        def register(self, method, fn):
            calls[method] = fn

    h = ClarificationRelayHandler(emit_event=lambda e: None)
    register(FakeRPCServer(), h)
    assert "client.request_clarification" in calls
    assert calls["client.request_clarification"] == h.handle
