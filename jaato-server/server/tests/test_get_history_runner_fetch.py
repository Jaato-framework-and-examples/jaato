"""JaatoServer.get_history reads the MAIN agent's transcript from the runner
post-seat-flip.

Root cause of the empty-transcript-on-cold-WS-reattach bug: the daemon-side
``_agents[*].history`` is not maintained for a runner-based session (empty
after a cold disk-restore), yet ``get_history`` / ``_emit_conversation_replay``
read it. The architecture mandates the runner-RPC surface for history reads
(core.py get_session-removal note). These pin that get_history now fetches
from the runner for the main agent, deserializes the canonical wire shape,
and falls back to the daemon-side copy on any runner failure.
"""
from __future__ import annotations

from typing import Any, List, Optional

from server.core import JaatoServer
from shared.plugins.session.serializer import serialize_message
from jaato_sdk.plugins.model_provider.types import Message, Part, Role


class _AgentState:
    def __init__(self, history: Optional[List[Any]] = None) -> None:
        self.history = history or []


class _FakeRPC:
    def __init__(self, history_dicts: Any = None, raise_exc: Any = None) -> None:
        self._history_dicts = history_dicts if history_dicts is not None else []
        self._raise = raise_exc
        self.calls = 0

    def session_get_history_threadsafe(self, *, raw: bool = False,
                                       timeout: float = 30.0) -> list:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return list(self._history_dicts)


def _server(rpc: Any, daemon_history: Optional[List[Any]] = None,
            main_id: str = "main", agents: Optional[dict] = None) -> JaatoServer:
    srv = JaatoServer.__new__(JaatoServer)
    srv._runner_rpc = rpc
    srv._main_agent_id = main_id
    srv._agents = agents if agents is not None else {
        main_id: _AgentState(daemon_history or [])}
    return srv


def _wire(role: Role, text: str) -> dict:
    return serialize_message(Message(role=role, parts=[Part.from_text(text)]))


def test_main_agent_history_comes_from_runner():
    # Runner has 2 turns; daemon-side is empty (the cold-restore state).
    rpc = _FakeRPC([_wire(Role.USER, "remember X"), _wire(Role.MODEL, "ok")])
    srv = _server(rpc, daemon_history=[])
    hist = srv.get_history("main")
    assert rpc.calls == 1
    assert [m.role for m in hist] == [Role.USER, Role.MODEL]
    assert hist[0].text == "remember X"
    assert hist[1].text == "ok"


def test_agent_id_none_resolves_to_main_and_fetches_runner():
    rpc = _FakeRPC([_wire(Role.USER, "hi")])
    srv = _server(rpc, daemon_history=[])
    assert [m.text for m in srv.get_history()] == ["hi"]
    assert rpc.calls == 1


def test_no_runner_reads_daemon_side():
    daemon = [Message(role=Role.USER, parts=[Part.from_text("local")])]
    srv = _server(rpc=None, daemon_history=daemon)
    hist = srv.get_history("main")
    assert hist is daemon  # daemon-side list, no fetch


def test_subagent_never_fetches_runner():
    # A non-main agent has no per-agent runner RPC → daemon-side only.
    rpc = _FakeRPC([_wire(Role.USER, "runner-main")])
    sub_hist = [Message(role=Role.MODEL, parts=[Part.from_text("sub")])]
    srv = _server(rpc, agents={"main": _AgentState([]),
                               "researcher": _AgentState(sub_hist)},
                  main_id="main")
    assert srv.get_history("researcher") is sub_hist
    assert rpc.calls == 0  # runner not consulted for a subagent


def test_runner_failure_falls_back_to_daemon_side():
    # Transient RPC failure must degrade to the daemon-side copy, not raise.
    daemon = [Message(role=Role.USER, parts=[Part.from_text("fallback")])]
    rpc = _FakeRPC(raise_exc=RuntimeError("rpc down"))
    srv = _server(rpc, daemon_history=daemon)
    hist = srv.get_history("main")
    assert [m.text for m in hist] == ["fallback"]
    assert rpc.calls == 1  # tried the runner, then fell back
