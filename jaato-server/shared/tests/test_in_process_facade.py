"""Contract tests for the in-process facade (Shape 1, PR1 + PR1.5).

Proves the existing ``jaato_sdk`` facade (``Session.ask``) rides UNCHANGED on
``InProcessClient`` for the ex01 ``ask`` round-trip — the AGENT_OUTPUT +
TURN_COMPLETED seams (PR1) — and that the real-provider wiring is in place:
``connect`` resolves credential secret URIs + ``verify_auth``s, and
``create_session`` ``configure_tools``s the embedded session (PR1.5).

A fake embedded runtime is used so no provider / credentials / network are
needed (CI-safe). The real-provider "it actually answers" proof
(facade-over-in-process == direct-in-process == IPC-facade) lives in the
dual-mode example suite.
"""

import asyncio

from jaato.in_process import InProcessClient, InProcessEventEmitter
from jaato_sdk.events import AgentOutputEvent, EventType, TurnCompletedEvent


class _FakeEmbedded:
    """Stand-in for ``jaato.JaatoClient`` — records the real-path lifecycle
    calls and streams two model chunks via ``send_message(on_output=...)``."""

    def __init__(self) -> None:
        self.connected_with = None
        self.verify_auth_plugin_configs = "<unset>"
        self.configure_tools_session_kwargs = "<unset>"
        self.closed = False

    def connect(self, project=None, location=None, model=None) -> None:
        self.connected_with = (project, location, model)

    def verify_auth(self, allow_interactive=False, on_message=None, plugin_configs=None):
        self.verify_auth_plugin_configs = plugin_configs
        return True

    def configure_tools(
        self, registry, permission_plugin=None, ledger=None,
        session_kwargs=None, skip_model_test=False,
    ) -> None:
        self.configure_tools_session_kwargs = session_kwargs

    def send_message(self, prompt, on_output=None, **_kwargs) -> str:
        if on_output is not None:
            on_output("model", "Hello ", "write")
            on_output("model", "world", "append")
        return "Hello world"

    def close_session(self) -> None:
        self.closed = True


def _factory_for(holder):
    """Build an ``embedded_factory(provider) -> client`` that stashes the
    constructed fake in ``holder`` for post-run assertions."""
    def _make(provider):
        client = _FakeEmbedded()
        holder["client"] = client
        holder["provider"] = provider
        return client
    return _make


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
        on TURN_COMPLETED — the unchanged facade driving the embedded client."""

        async def _run():
            holder = {}
            counters = {"output": 0, "turn": 0}
            async with InProcessClient.session(
                model="fake-model",
                provider="fakeprov",
                embedded_factory=_factory_for(holder),
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

            assert answer == "Hello world"
            assert counters["output"] == 2  # two model chunks
            assert counters["turn"] == 1    # one terminal
            assert holder["provider"] == "fakeprov"  # provider threaded through

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
                model="fake-model",
                embedded_factory=lambda provider: _MixedEmbedded(),
            ) as s:
                answer = await s.ask("hi")
            assert answer == "answer"

        asyncio.run(_run())


class TestRealPathWiring:
    """PR1.5: connect resolves creds + verify_auths; create_session
    configure_tools the embedded session with the resolved plugin_configs."""

    def test_verify_auth_and_configure_tools_wired(self):
        async def _run():
            holder = {}
            plugin_configs = {"openrouter": {"api_key": "sk-or-plain"}}
            async with InProcessClient.session(
                model="m",
                provider="openrouter",
                plugin_configs=plugin_configs,
                embedded_factory=_factory_for(holder),
            ) as s:
                await s.ask("hi")

            fake = holder["client"]
            # verify_auth was called with the (resolved) plugin_configs.
            assert fake.verify_auth_plugin_configs == {
                "openrouter": {"api_key": "sk-or-plain"}
            }
            # configure_tools received the resolved plugin_configs in session_kwargs.
            assert fake.configure_tools_session_kwargs == {
                "plugin_configs": {"openrouter": {"api_key": "sk-or-plain"}}
            }
            # A plain key passes through resolve_secret_uri unchanged; pass://
            # resolution requires jaato-premium and is exercised by the example
            # suite, not CI.

        asyncio.run(_run())

    def test_failed_auth_raises(self):
        async def _run():
            class _NoAuth(_FakeEmbedded):
                def verify_auth(self, allow_interactive=False, on_message=None,
                                plugin_configs=None):
                    return False

            raised = False
            try:
                async with InProcessClient.session(
                    model="m", provider="p",
                    embedded_factory=lambda provider: _NoAuth(),
                ):
                    pass
            except RuntimeError:
                raised = True
            assert raised

        asyncio.run(_run())
