"""Tests for ``JaatoServer.clear_history`` forwarding to the
runner-side JaatoSession.

History (commit chronology):

  §7b.1 (commit 8cbb8ba2): introduced write-both — daemon-side
    ``_jaato.reset_session()`` + runner-RPC forward.
  §7c step 6.3: daemon-side leg dropped.  The runner-side
    ``session.reset`` RPC is the only source of truth for
    conversation-history state.
  #1573 (this version): ``clear_history`` became the ``reset``
    command's implementation and returns a result dict.  It no longer
    swallows a runner failure and reports success: the daemon-side
    mirror is cleared only after the runner confirms.

Tests pin the post-#1573 contract (the dispatch path itself is guarded
by ``test_reset_reaches_the_runner_1573.py``):

- with a runner attached, the method forwards to
  ``runner_rpc.session_reset_threadsafe(timeout=10)`` and answers
  ``{"result": "History cleared", ...}``
- a runner failure is ``stage="runner"`` and changes nothing daemon-side
- no runner is ``stage="no_runner"``
- a running turn is ``stage="busy"`` and never reaches the runner
"""

from __future__ import annotations

from typing import Any, List

from jaato_server.server.core import JaatoServer


class _FakeRPC:
    def __init__(self) -> None:
        self.reset_calls: List[float] = []
        self.raise_next: Any = None

    def session_reset_threadsafe(self, *, timeout: float = 5.0) -> Any:
        self.reset_calls.append(timeout)
        if self.raise_next is not None:
            raise self.raise_next
        return {"ok": True, "messages_cleared": 4}


def _make_server(rpc: Any = None) -> JaatoServer:
    """Minimal JaatoServer skeleton suitable for clear_history."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._runner_rpc = rpc
    srv._model_running = False
    srv._original_inputs = ["sample-input"]
    srv._main_agent_id = "main"
    srv._agents = {}
    srv._cached_context_limit = None
    srv.emit = lambda event: None  # type: ignore[method-assign]
    return srv


def test_clear_history_forwards_to_runner_when_attached() -> None:
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)

    result = srv.clear_history()

    assert rpc.reset_calls == [10.0]
    assert result["result"] == "History cleared"
    assert result["cleared"] is True
    assert result["messages_cleared"] == 4
    assert srv._original_inputs == []


def test_clear_history_runner_failure_is_reported_and_changes_nothing() -> None:
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("runner stuck")
    srv = _make_server(rpc=rpc)

    result = srv.clear_history()

    assert result["stage"] == "runner"
    assert "runner stuck" in result["error"]
    assert srv._original_inputs == ["sample-input"]


def test_clear_history_no_runner_attached_is_refused() -> None:
    srv = _make_server(rpc=None)

    result = srv.clear_history()

    assert result["stage"] == "no_runner"
    assert srv._original_inputs == ["sample-input"]


def test_clear_history_refuses_a_running_turn() -> None:
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)
    srv._model_running = True

    result = srv.clear_history()

    assert result["stage"] == "busy"
    assert rpc.reset_calls == []
    assert srv._original_inputs == ["sample-input"]
