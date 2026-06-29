"""Contract tests for the in-process facade (Shape 1, PR1 tracer-bullet).

Proves the existing ``jaato_sdk`` facade (``Session.ask``) rides UNCHANGED on
``InProcessClient`` for the ex01 ``ask`` round-trip — the AGENT_OUTPUT +
TURN_COMPLETED seams — using a fake embedded runtime, so no provider /
credentials / network are needed (CI-safe). The real-provider fidelity check
(facade-over-in-process == direct-in-process == IPC-facade) lives in the
dual-mode example suite.
"""

import asyncio

from jaato.in_process import InProcessClient, InProcessEventEmitter
from jaato_sdk.events import AgentOutputEvent, EventType, TurnCompletedEvent


class _FakeEmbedded:
    """Stand-in for ``jaato.JaatoClient`` — streams two model chunks and
    returns the joined text, exercising the ``send_message(on_output=...)``
    streaming path the dual-mode examples proved."""

    def __init__(self) -> None:
        self.connected_with = None
        self.closed = False

    def connect(self, project=None, location=None, model=None) -> None:
        self.connected_with = (project, location, model)

    def send_message(self, prompt, on_output=None, **_kwargs) -> str:
        if on_output is not None:
            on_output("model", "Hello ", "write")
            on_output("model", "world", "append")
        return "Hello world"

    def close_session(self) -> None:
        self.closed = True


class TestInProcessEventEmitter:
    def test_subscribe_emit_dispatch(self):
        emitter = InProcessEventEmitter()
        seen = []
        emitter.subscribe(EventType.AGENT_OUTPUT, seen.append)
        emitter.emit(AgentOutputEvent(source="model", text="x"))
        emitter.emit(TurnCompletedEvent())  # different type — not delivered
        assert len(seen) == 1
        assert seen[0].text == "x"

    def test_unsubscribe_stops_delivery(self):
        emitter = InProcessEventEmitter()
        seen = []
        unsub = emitter.subscribe(EventType.AGENT_OUTPUT, seen.append)
        unsub()
        emitter.emit(AgentOutputEvent(source="model", text="x"))
        assert seen == []

    def test_subscribe_once_fires_exactly_once(self):
        emitter = InProcessEventEmitter()
        seen = []
        emitter.subscribe_once(EventType.TURN_COMPLETED, seen.append)
        emitter.emit(TurnCompletedEvent())
        emitter.emit(TurnCompletedEvent())
        assert len(seen) == 1


class TestFacadeRidesOnInProcessClient:
    def test_ask_round_trip(self):
        """The facade ``ask`` collects the streamed AGENT_OUTPUT and returns
        on TURN_COMPLETED — proving the unchanged facade drives the embedded
        client."""

        async def _run():
            counters = {"output": 0, "turn": 0}
            async with InProcessClient.session(
                model="fake-model", embedded_factory=_FakeEmbedded,
            ) as s:
                s.client.subscribe(
                    EventType.AGENT_OUTPUT,
                    lambda ev: counters.__setitem__("output", counters["output"] + 1),
                )
                s.client.subscribe_once(
                    EventType.TURN_COMPLETED,
                    lambda ev: counters.__setitem__("turn", counters["turn"] + 1),
                )
                answer = await s.ask("Who are you?")

            # The facade returned the streamed model text, joined.
            assert answer == "Hello world"
            # Two model chunks -> two AGENT_OUTPUT events.
            assert counters["output"] == 2
            # Exactly one TURN_COMPLETED terminal.
            assert counters["turn"] == 1

        asyncio.run(_run())

    def test_source_filter_excludes_non_model_output(self):
        """``ask`` defaults to ``sources=("model",)`` — non-model chunks are
        not collected into the answer."""

        class _MixedEmbedded(_FakeEmbedded):
            def send_message(self, prompt, on_output=None, **_kwargs):
                if on_output is not None:
                    on_output("system", "[setup]", "write")  # filtered out
                    on_output("model", "answer", "write")
                return "answer"

        async def _run():
            async with InProcessClient.session(
                model="fake-model", embedded_factory=_MixedEmbedded,
            ) as s:
                answer = await s.ask("hi")
            assert answer == "answer"

        asyncio.run(_run())
