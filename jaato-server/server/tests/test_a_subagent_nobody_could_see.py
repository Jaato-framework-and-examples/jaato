"""A spawned subagent reaches the client on a runner-served session.

Reported from the web client: ``spawn_subagent`` succeeded, the agent
said *"Subagent spawned (id: subagent_1)"*, and the only tab on screen
stayed ``MAIN AGENT``.  The TUI has the same blind spot -- both render
``AgentCreatedEvent``, and on the default path nothing emitted one.

``subagent`` is runner-tier, so the plugin the model drives lives in the
RUNNER process, and ``SubagentPlugin._ui_hooks`` is the slot every
``if self._ui_hooks:`` in ``subagent/plugin.py`` reads.  The daemon arms
its OWN instance (``core.py``'s ``_setup_agent_hooks`` ->
``subagent_plugin.set_ui_hooks(hooks)``), which no runner-served session
calls.  The runner installs ``_AgentUIHooksNotificationShim`` -- but on
``session._ui_hooks``, a different object.  Measured against the real
class before the fix::

    session._ui_hooks      : _AgentUIHooksNotificationShim
    todo._reporter         : LivePlanReporter
    subagent._plan_reporter: LivePlanReporter
    subagent._ui_hooks     : NoneType

So the shim's own ``on_agent_created`` docstring -- *"Called by the
subagent plugin (PLUGIN_TIER='runner')"* -- named a caller that was
never handed it.  Same shape as the plan-reporter gap
(``test_plan_reporter_bridge``), one plugin over, and closed the same
way: install per turn on the runner, restore on exit, demux daemon-side
through the hooks the in-process path already uses.

The install is one change and unlocks a family, because the plugin
propagates its own hooks to each child session
(``session.set_ui_hooks(self._ui_hooks, agent_id)``): the subagent's
status, context, turn accounting and tool activity all travel that slot.

``on_agent_output`` is the second half.  It was a no-op on the reasoning
that the runner session uses the ``on_output`` kwarg path instead --
true of the ROOT session, false of a subagent, whose output has no other
route.  A stream frame could not have carried it either: it has no field
for an agent id, so a subagent's words would arrive attributed to
whoever owns the stream.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List

from jaato_sdk.events import AgentCreatedEvent, AgentOutputEvent
from server.runner.rpc import RunnerRPC
from shared.tests.reversion import Reversion


# ------------------------------------------------------------ fakes

class _FakeSubagent:
    """Only the two members the install touches."""

    def __init__(self, hooks: Any = None) -> None:
        self._ui_hooks = hooks
        self._plan_reporter = None

    def set_ui_hooks(self, hooks: Any) -> None:
        self._ui_hooks = hooks

    def set_plan_reporter(self, reporter: Any) -> None:
        self._plan_reporter = reporter


class _FakeTodo:
    def __init__(self) -> None:
        self._reporter = "memory-reporter"


def _session(subagent=None, todo=None):
    plugins = {"subagent": subagent, "todo": todo, "session": None}

    class _Registry:
        def get_plugin(self, name):
            return plugins.get(name)

    class _Runtime:
        registry = _Registry()

    class _Session:
        _ui_hooks = None
        _runtime = _Runtime()

    return _Session()


def _rpc(captured: List[Dict[str, Any]]) -> RunnerRPC:
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: captured.append(kw)
    return rpc


def _server():
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._agents = {}
    srv._session_id = "20260921_100000"
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)
    srv._get_agent_pipeline = lambda agent_id: None
    srv.registry = None          # skips the daemon-side subagent registration
    srv._setup_agent_hooks()     # the real ServerAgentHooks, not a stand-in
    return srv


# ------------------------------------------------------------ the install

def test_the_runner_hands_its_subagent_plugin_the_shim():
    """The slot ``plugin.py``'s ``if self._ui_hooks:`` guards read."""
    sub = _FakeSubagent()
    rpc = _rpc([])
    originals = rpc._install_session_notification_callbacks(
        _session(sub), request_id=1,
    )
    assert sub._ui_hooks is not None, (
        "SubagentPlugin._ui_hooks is still None -- every on_agent_* call "
        "in subagent/plugin.py is guarded on it and drops"
    )
    assert type(sub._ui_hooks).__name__ == "_AgentUIHooksNotificationShim"
    assert originals["subagent_ui_hooks"] is None


def test_the_session_slot_and_the_plugin_slot_are_different_objects():
    """Both are filled; neither install stands in for the other.

    The pre-fix tree filled only the first, which is why a session that
    streamed tool output perfectly still showed no subagent.
    """
    sub = _FakeSubagent()
    sess = _session(sub)
    _rpc([])._install_session_notification_callbacks(sess, request_id=1)
    assert sess._ui_hooks is not None and sub._ui_hooks is not None
    assert sess._ui_hooks is not sub._ui_hooks


def test_restore_puts_the_original_hooks_back():
    """The plugin is registry-scoped and outlives the call.

    A shim left behind keeps emitting frames under a request id that has
    already been answered.
    """
    sub = _FakeSubagent(hooks="daemon-hooks")
    rpc = _rpc([])
    sess = _session(sub)
    originals = rpc._install_session_notification_callbacks(sess, request_id=1)
    assert sub._ui_hooks != "daemon-hooks"
    rpc._restore_session_notification_callbacks(sess, originals)
    assert sub._ui_hooks == "daemon-hooks"


def test_a_session_with_no_subagent_plugin_installs_nothing():
    originals = _rpc([])._install_session_notification_callbacks(
        _session(None), request_id=1,
    )
    assert "subagent_ui_hooks" not in originals


def test_restoring_what_was_never_installed_is_a_no_op():
    sub = _FakeSubagent(hooks="untouched")
    rpc = _rpc([])
    rpc._restore_session_notification_callbacks(_session(sub), {})
    assert sub._ui_hooks == "untouched"


def test_a_plugin_that_cannot_take_hooks_is_skipped_not_raised():
    """A rolling upgrade, or a test double -- the turn must still run."""
    class _Older:
        pass

    originals = _rpc([])._install_session_notification_callbacks(
        _session(_Older()), request_id=1,
    )
    assert "subagent_ui_hooks" not in originals


# ------------------------------------------------------------ the frames

def _spawn(sub) -> None:
    """What ``subagent/plugin.py`` does at ``_execute_spawn_subagent``."""
    sub._ui_hooks.on_agent_created(
        agent_id="subagent_1",
        agent_name="analyst",
        agent_type="subagent",
        profile_name="analyst-codebase-documentation",
        parent_agent_id="main",
        created_at=datetime.datetime(2026, 9, 21, 10, 0, 0),
    )


def test_the_spawn_emits_an_agent_created_frame():
    captured: List[Dict[str, Any]] = []
    sub = _FakeSubagent()
    rpc = _rpc(captured)
    rpc._install_session_notification_callbacks(_session(sub), request_id=7)
    _spawn(sub)
    assert [c["event_type"] for c in captured] == ["agent_created"]
    assert captured[0]["request_id"] == 7
    assert captured[0]["payload"] == {
        "agent_id": "subagent_1",
        "agent_name": "analyst",
        "agent_type": "subagent",
        "profile_name": "analyst-codebase-documentation",
        "parent_agent_id": "main",
        "created_at": "2026-09-21T10:00:00",
    }


def test_a_subagents_output_is_forwarded_under_its_own_agent_id():
    captured: List[Dict[str, Any]] = []
    sub = _FakeSubagent()
    rpc = _rpc(captured)
    rpc._install_session_notification_callbacks(_session(sub), request_id=7)
    sub._ui_hooks.on_agent_output(
        agent_id="subagent_1", source="model",
        text="Reading the tree...", mode="append",
    )
    assert [c["event_type"] for c in captured] == ["agent_output"]
    assert captured[0]["payload"] == {
        "agent_id": "subagent_1",
        "source": "model",
        "text": "Reading the tree...",
        "mode": "append",
    }


def test_a_failing_emit_does_not_take_the_spawn_down():
    """A dead channel must not turn a working spawn into a raised tool."""
    sub = _FakeSubagent()
    rpc = RunnerRPC.__new__(RunnerRPC)

    def _boom(**_kw):
        raise RuntimeError("channel closed")

    rpc.emit_notification = _boom
    rpc._install_session_notification_callbacks(_session(sub), request_id=1)
    _spawn(sub)   # must not raise
    sub._ui_hooks.on_agent_output(agent_id="subagent_1", source="model",
                                  text="x", mode="append")


# ------------------------------------------------------------ the daemon

def test_the_demuxer_turns_agent_created_into_the_event_the_client_reads():
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("agent_created", {
        "agent_id": "subagent_1",
        "agent_name": "analyst",
        "agent_type": "subagent",
        "profile_name": "analyst-codebase-documentation",
        "parent_agent_id": "main",
        "created_at": "2026-09-21T10:00:00",
    })
    evs = [e for e in srv._emitted_events if isinstance(e, AgentCreatedEvent)]
    assert len(evs) == 1
    assert evs[0].agent_id == "subagent_1"
    assert evs[0].agent_type == "subagent"
    assert evs[0].parent_agent_id == "main"
    # ...and the daemon's own agent registry knows it, which is what the
    # attach replay reads.
    assert "subagent_1" in srv._agents


def test_the_demuxer_turns_agent_output_into_an_output_event():
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("agent_output", {
        "agent_id": "subagent_1", "source": "model",
        "text": "Reading the tree...", "mode": "append",
    })
    evs = [e for e in srv._emitted_events if isinstance(e, AgentOutputEvent)]
    assert len(evs) == 1
    assert evs[0].agent_id == "subagent_1"
    assert evs[0].text == "Reading the tree..."


# --------------------------------------------------------------------------
# Reversions
# --------------------------------------------------------------------------

_RPC = "jaato-server/server/runner/rpc.py"
_CORE = "jaato-server/server/core.py"

_INSTALL_CALL_FIXED = """        # The subagent plugin's ui-hooks slot (see _NOTIF_AGENT_CREATED).
        self._install_subagent_ui_hooks(session, originals, request_id)
"""

_RESTORE_CALL_FIXED = """        self._restore_subagent_ui_hooks(session, originals)
"""

_OUTPUT_FORWARD_FIXED = """            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_OUTPUT,"""
_OUTPUT_FORWARD_BROKEN = """            return
            self._rpc.emit_notification(
                request_id=self._request_id,
                event_type=self._rpc._NOTIF_AGENT_OUTPUT,"""

_OUTPUT_DEMUX_FIXED = """        if event_type == "agent_output":
            hooks.on_agent_output(
"""
_OUTPUT_DEMUX_BROKEN = """        if event_type == "agent_output_never_matches":
            hooks.on_agent_output(
"""

REVERSIONS = [
    Reversion(
        target=_RPC,
        find=_INSTALL_CALL_FIXED,
        replace="",
        test="test_the_runner_hands_its_subagent_plugin_the_shim",
        because=(
            "the runner's subagent plugin keeping a None ui-hooks slot, so "
            "a spawned subagent emits no AgentCreatedEvent and no client "
            "shows a tab for it"
        ),
    ),
    Reversion(
        target=_RPC,
        find=_INSTALL_CALL_FIXED,
        replace="",
        test="test_the_spawn_emits_an_agent_created_frame",
        because="a spawn that reaches the daemon as nothing at all",
    ),
    Reversion(
        target=_RPC,
        find=_RESTORE_CALL_FIXED,
        replace="",
        test="test_restore_puts_the_original_hooks_back",
        because=(
            "a shim left on a registry-scoped plugin after its call was "
            "answered, emitting frames under a dead request id"
        ),
    ),
    Reversion(
        target=_RPC,
        find=_OUTPUT_FORWARD_FIXED,
        replace=_OUTPUT_FORWARD_BROKEN,
        test="test_a_subagents_output_is_forwarded_under_its_own_agent_id",
        because=(
            "a subagent's output having no route to any client -- the "
            "stream-frame path it was said to use carries no agent id"
        ),
    ),
    Reversion(
        target=_CORE,
        find=_OUTPUT_DEMUX_FIXED,
        replace=_OUTPUT_DEMUX_BROKEN,
        test="test_the_demuxer_turns_agent_output_into_an_output_event",
        because=(
            "the daemon dropping an agent_output frame the runner emitted, "
            "so the tab appears and stays empty"
        ),
    ),
]
