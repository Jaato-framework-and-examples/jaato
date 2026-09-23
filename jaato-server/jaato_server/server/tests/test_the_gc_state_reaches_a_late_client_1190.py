"""The GC policy and the last pass reach a client that attaches late (#1190).

The Instructions panel is to show when GC last ran, what it freed, and which
policy is in force.  All three were already on the wire -- ``GCConfigEvent``
at initialisation, ``GCEvent`` for each phase of a pass -- and NEITHER was
replayed by ``emit_current_state``, the one door for "tell a client arriving
mid-session what the state is".  A browser that reconnected after a pass
would read "no GC has run" about a session that had collected an hour ago,
and would never learn the strategy at all.

The daemon now keeps the last completed pass as the event that announced it,
so a replay carries the ORIGINAL timestamp.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from jaato_sdk.events import EventType
from jaato_server.server.core import JaatoServer
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"

REVERSIONS = [
    Reversion(
        target=_CORE,
        find="""        self._emit_gc_state(emit)
""",
        replace="",
        test="test_emit_current_state_replays_the_gc_state",
        because="an attaching client never learns the policy or the last pass",
    ),
    Reversion(
        target=_CORE,
        find="""        if phase == "completed":
            self._last_gc_pass = event
""",
        replace="",
        test="test_a_completed_pass_is_remembered_and_replayed_with_its_own_time",
        because="nothing is kept, so a late client reads 'no GC yet'",
    ),
]


def _server(**agent_fields: Any) -> tuple:
    emitted: List[Any] = []
    agent = SimpleNamespace(gc_threshold=None, gc_strategy=None, gc_target_percent=None,
                            gc_continuous_mode=False, **{})
    for k, v in agent_fields.items():
        setattr(agent, k, v)
    srv = SimpleNamespace(emit=emitted.append, _main_agent_id="main",
                          _agents={"main": agent}, _last_gc_pass=None)
    return srv, emitted


def test_a_completed_pass_is_remembered_and_replayed_with_its_own_time():
    srv, _ = _server(gc_strategy="budget", gc_threshold=80.0)
    JaatoServer._emit_gc_phase_event(srv, {"phase": "started", "strategy": "budget"})
    assert srv._last_gc_pass is None, "a pass that has not finished is not 'the last pass'"
    JaatoServer._emit_gc_phase_event(srv, {"phase": "completed", "strategy": "budget",
                                           "success": True, "tokens_freed": 14200})
    live = srv._last_gc_pass
    replayed: List[Any] = []
    JaatoServer._emit_gc_state(srv, replayed.append)
    passes = [e for e in replayed if e.type == EventType.GC]
    assert len(passes) == 1
    assert passes[0].tokens_freed == 14200
    assert passes[0].timestamp == live.timestamp


def test_the_policy_is_replayed_even_when_there_is_none():
    """A session with no GC is the state an operator most needs to see."""
    srv, _ = _server()
    replayed: List[Any] = []
    JaatoServer._emit_gc_state(srv, replayed.append)
    (config,) = [e for e in replayed if e.type == EventType.GC_CONFIG]
    assert config.strategy is None
    assert not [e for e in replayed if e.type == EventType.GC]


def test_the_policy_replayed_is_the_one_in_force():
    srv, _ = _server(gc_strategy="hybrid", gc_threshold=75.0, gc_target_percent=50.0,
                     gc_continuous_mode=True)
    replayed: List[Any] = []
    JaatoServer._emit_gc_state(srv, replayed.append)
    (config,) = [e for e in replayed if e.type == EventType.GC_CONFIG]
    assert (config.strategy, config.threshold, config.target_percent, config.continuous_mode) == (
        "hybrid", 75.0, 50.0, True)


def test_emit_current_state_replays_the_gc_state():
    """Asserted on the call site: emit_current_state reaches a dozen
    subsystems, and what was missing is a call."""
    # Parsed from the file beside this test, not ``inspect.getsource``: the
    # reversion meta-guard runs in a sandbox copy, and getsource reads the
    # module the editable install pins -- the real checkout.
    core = Path(__file__).resolve().parents[1] / "core.py"
    method = next(
        n for n in ast.walk(ast.parse(core.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "emit_current_state"
    )
    called = {
        n.func.attr for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_emit_gc_state" in called
