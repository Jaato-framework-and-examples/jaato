"""A stopped completion-gated session is not re-driven by the nudge (#1433).

On a profile with a ``completion_payload_schema``, ``session.stop`` (#812),
``session.end`` and ``stop()`` each cancelled the in-flight turn, the first
two emitted ``SessionTerminatedEvent`` -- and about 0.3 s later the daemon
dispatched ``session.try_completion_nudge``, which answered "the agent ended
its loop without signal_completion" and re-prompted it.  The session then ran
to completion with no client attached, spending, 3 runs out of 3.

``try_completion_nudge`` asked two questions (did the agent signal; is the
budget left) and not the one that mattered: WHY did the loop end.  The cancel
token that could have said so is cleared at turn end, before the daemon asks.

The rule now: a loop that ended because it was cancelled is never nudged, and
it is enforced in ``try_completion_nudge`` itself -- the one gate the
daemon's guard (over the runner RPC), the embedded lead and the subagent loop
all pass through.  ``request_stop`` latches the stop (running turn or not,
because a stop can land between the turn's end and the nudge dispatch), a
turn whose token was tripped by anything else latches it at turn end, and
the next caller-originated turn clears it.

Every test drives a REAL ``JaatoSession`` through a real turn.  The provider
is echo with one change: its turn blocks until the session's cancel token
trips, which is the shape of the reported long generation.  The stop verbs
are the real ``SessionManager.stop_session`` and ``JaatoServer.stop``, and
they reach the session through the real runner handlers
(``RunnerRPC._handle_session_request_stop`` /
``_handle_session_try_completion_nudge``), as on the default path.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    Part,
    ProviderResponse,
    TokenUsage,
    TurnResult,
)
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.completion_nudge import COMPLETION_NUDGE_TEXT
from jaato_server.shared.jaato_runtime import JaatoRuntime
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.model_provider.base import ProviderConfig
from jaato_server.shared.plugins.model_provider.echo.provider import EchoProvider
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.session_context import isolated_current_session
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find=(
            "        if JaatoSession._completion_nudge_suppressed(self):\n"
            "            return False, getattr(self, \"_completion_nudges_fired\", 0)\n"
        ),
        replace="",
        test="TestAStopIsNotUndone::test_an_operator_stop_mid_turn_is_not_nudged",
        because="the gate no longer asks why the loop ended, so a stop is nudged",
    ),
    Reversion(
        target=_SESSION,
        find=(
            "        self._nudge_suppressed_reason = \"cancelled\"\n"
            "        if self._cancel_token and self._is_running:"
        ),
        replace="        if self._cancel_token and self._is_running:",
        test="TestAStopIsNotUndone::test_a_stop_between_turn_end_and_nudge_is_not_nudged",
        because="a stop that finds no turn running no longer counts",
    ),
    Reversion(
        target=_SESSION,
        find=(
            "        self._truncation_recovery_count = 0\n"
            "        # A stop suppressed the nudge of the loop that ended; this turn was\n"
            "        # started by a caller (a suppressed nudge starts no turn), so it is\n"
            "        # nudged normally again (#1433).\n"
            "        self._nudge_suppressed_reason = None\n"
        ),
        replace="        self._truncation_recovery_count = 0\n",
        test="TestAStopIsNotUndone::test_stop_cancels_the_turn_and_the_session_stays_drivable",
        because="one stop suppresses every later nudge for the life of the session",
    ),
    Reversion(
        target=_SESSION,
        find=(
            "            self._record_turn_ran(turn_data)\n"
            "            self._note_turn_cancellation()\n"
            "\n"
            "            # Update instruction budget with conversation tokens\n"
        ),
        replace=(
            "            self._record_turn_ran(turn_data)\n"
            "\n"
            "            # Update instruction budget with conversation tokens\n"
        ),
        test="TestAStopIsNotUndone::test_a_token_tripped_elsewhere_is_not_nudged",
        because="only request_stop suppresses; a token cancelled another way is nudged",
    ),
]

_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class _BlockingEcho(EchoProvider):
    """Echo whose turn runs until the session's cancel token trips.

    Counts every ``complete()``: after a stop, "no further provider call"
    is ``calls`` not moving.  Streaming is claimed so the session passes
    the token through (the batched path passes none).
    """

    def __init__(self) -> None:
        super().__init__()
        self.initialize(ProviderConfig(extra={}))
        self.calls = 0
        self.block = True
        self.started = threading.Event()

    def supports_streaming(self) -> bool:
        return True

    def complete(self, messages, *args, cancel_token=None, **kwargs):
        self.calls += 1
        if not self.block:
            return super().complete(
                messages, *args, cancel_token=cancel_token, **kwargs)
        self.started.set()
        deadline = time.monotonic() + 10
        while not (cancel_token is not None and cancel_token.is_cancelled):
            if time.monotonic() > deadline:
                raise AssertionError("the turn was never cancelled")
            time.sleep(0.005)
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part.from_text("an essay cut off part-way")],
            usage=TokenUsage(),
            finish_reason=FinishReason.CANCELLED,
        ))


class _DaemonRPC:
    """The daemon's ``_runner_rpc``, reduced to the two calls in play.

    Each forwards to the REAL runner handler, so the session is reached
    exactly as on the default (runner-served) path.
    """

    def __init__(self, runner: RunnerRPC) -> None:
        self._runner = runner

    def session_request_stop_threadsafe(self, reason: str = "",
                                        timeout: float = 2.0) -> bool:
        ok, out = self._runner._handle_session_request_stop({"reason": reason})
        assert ok, out
        return bool(out["cancelled"])

    def session_try_completion_nudge_threadsafe(self, max_nudges: int):
        ok, out = self._runner._handle_session_try_completion_nudge(
            {"max_nudges": max_nudges})
        assert ok, out
        return out["should_nudge"], out["nudges_fired"]


@pytest.fixture(autouse=True)
def _isolated():
    with isolated_current_session():
        yield


@pytest.fixture
def rig(tmp_path: Path):
    """A gated session behind a runner, and a daemon server stub over it."""
    rt = JaatoRuntime(provider_name="echo", workspace_path=tmp_path)
    reg = PluginRegistry()
    reg.set_workspace_path(str(tmp_path))
    rt.configure_plugins(reg)
    session = JaatoSession(rt, "echo")
    session.configure(skip_provider=True, plugins=[],
                      completion_payload_schema=_SCHEMA)
    provider = _BlockingEcho()
    session._provider = provider
    assert "signal_completion" in {
        t.name for t in session._get_tools_for_provider()}

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    runner = RunnerRPC(a, lambda name, args: (False, {"error": "none"}))
    runner._session_host = RunnerSessionHost(
        envelope=SessionInitEnvelope(
            session_id="s-1433", workspace_path=str(tmp_path),
            profile_name="gated", provider_name="echo", model_name="echo",
            plugins=[]),
        runtime=rt, session=session,
    )
    rpc = _DaemonRPC(runner)
    server = SimpleNamespace(_runner_rpc=rpc, _model_running=True,
                             _main_agent_id="main")
    yield SimpleNamespace(session=session, provider=provider, rpc=rpc,
                          server=server)
    a.close()


def _start_turn(rig) -> threading.Thread:
    thread = threading.Thread(
        target=rig.session.send_message, args=("write an essay",), daemon=True)
    thread.start()
    assert rig.provider.started.wait(5), "the turn never reached the provider"
    return thread


def _daemon_winds_down(rig) -> None:
    """What ``core.py``'s ``_finish_turn`` does after the turn: ask the gate,
    and re-prompt when it says so."""
    should_nudge, _ = rig.rpc.session_try_completion_nudge_threadsafe(2)
    if should_nudge:
        rig.session.send_message(COMPLETION_NUDGE_TEXT)


def _stop_session(rig) -> List[Any]:
    """The real ``SessionManager.stop_session`` on a one-session manager."""
    events: List[Any] = []
    manager = SimpleNamespace(
        _lock=threading.Lock(),
        _sessions={"s-1433": SimpleNamespace(
            server=SimpleNamespace(
                _model_running=rig.server._model_running,
                _main_agent_id="main",
                stop=lambda: JaatoServer.stop(rig.server)),
            runner_identity=None)},
        _emit_to_session=lambda sid, ev: events.append(ev),
    )
    out = SessionManager.stop_session(manager, "s-1433")
    assert out["found"]
    return events


class TestAStopIsNotUndone:
    def test_an_operator_stop_mid_turn_is_not_nudged(self, rig, caplog):
        thread = _start_turn(rig)
        events = _stop_session(rig)
        thread.join(5)
        assert not thread.is_alive()
        assert [type(e).__name__ for e in events] == ["SessionTerminatedEvent"]

        with caplog.at_level(logging.INFO):
            _daemon_winds_down(rig)

        assert rig.provider.calls == 1, (
            "a session the client was told had terminated was re-prompted")
        assert "COMPLETION_NUDGE: suppressed (cancelled)" in caplog.text
        # A refusal spends nothing: no token, no latch on the next turn.
        assert rig.session._completion_nudges_fired == 0
        assert rig.session._completion_nudge_turn_pending is False

    def test_a_stop_between_turn_end_and_nudge_is_not_nudged(self, rig):
        """The daemon dispatches the nudge after the turn wound down, so a
        stop can find nothing running and must still count."""
        rig.provider.block = False
        rig.session.send_message("write an essay")   # ends, no signal
        rig.server._model_running = False
        _stop_session(rig)
        _daemon_winds_down(rig)
        assert rig.provider.calls == 1

    def test_stop_cancels_the_turn_and_the_session_stays_drivable(self, rig):
        thread = _start_turn(rig)
        assert JaatoServer.stop(rig.server) is True
        thread.join(5)
        _daemon_winds_down(rig)
        assert rig.provider.calls == 1, "a cancelled turn was nudged"

        # The next caller-originated turn runs, and a natural end without
        # signal_completion is nudged again as usual.
        rig.provider.block = False
        rig.session.send_message("carry on")
        assert rig.provider.calls == 2
        assert rig.rpc.session_try_completion_nudge_threadsafe(2) == (True, 1)

    def test_a_token_tripped_elsewhere_is_not_nudged(self, rig):
        """The rule is "the loop was cancelled", not "request_stop ran"."""
        thread = _start_turn(rig)
        rig.session._cancel_token.cancel(reason="subagent_cancelled")
        thread.join(5)
        assert rig.rpc.session_try_completion_nudge_threadsafe(2) == (False, 0)


class TestANaturalEndIsStillNudged:
    def test_a_loop_that_ends_without_signalling_is_nudged(self, rig):
        rig.provider.block = False
        rig.session.send_message("write an essay")
        _daemon_winds_down(rig)
        assert rig.provider.calls == 2, "the nudge stopped firing altogether"
        assert rig.session._completion_nudges_fired == 1
