"""A session's last turn is saved before its runner is released (#1542).

After #1506 the AppArmor Part C rerun logged, for 3 of 4 sessions::

    Not saving session <id>: its runner was already released, so its
    history cannot be read; keeping the record on disk.

The #1355 guard did its job (nothing was overwritten), but the turns since
the previous save were never persisted.  The unload ALREADY saved before
``server.shutdown()`` and the boundary release; it skipped that save because
``is_dirty`` read ``False``, and that was the defect:

* a tool call start spawns an async save, which reads the history from the
  runner while the turn is still running;
* the turn ends (``ToolCallEnd``, ``TurnCompleted``, the agent's ``done``),
  each marking the session dirty -- and ``done`` schedules the unload;
* the async save finishes and writes ``is_dirty = False``, erasing marks it
  never saw;
* the unload finds a clean session, skips its save and releases the runner;
* an async save queued behind the first then runs, finds the runner gone,
  and logs the warning.

``Session`` now counts dirty marks (``dirty_generation``) and a save clears
the flag only when no mark arrived after it read the history.  A late save
of a session that IS clean is superseded and skipped quietly, so the warning
again means a change was lost.

Not covered here: the kernel-gated check from the issue (N sequential IPC
sessions under an enforcing AppArmor host, grace 0).  No AppArmor kernel in
CI; the unit cases drive the real ``_save_session`` and
``_do_session_unload``.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Callable, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="""            if self.dirty_generation != generation:
                return False
""",
        replace="""            if False:
                return False
""",
        because="a save clears a dirty mark made after it read the history, "
                "so the unload skips its final save and the last turn is lost",
        test="test_unload_persists_the_turn_that_ended_during_an_async_save",
    ),
    Reversion(
        target=_SM,
        find="""            if self.dirty_generation != generation:
                return False
""",
        replace="""            if False:
                return False
""",
        because="a mark made while a save runs is erased by that save",
        test="test_a_mark_made_during_a_save_survives_it",
    ),
    Reversion(
        target=_SM,
        find="""        if session.is_dirty:
            self._save_session(session)

        # Close session-specific log handlers""",
        replace="""        # Close session-specific log handlers""",
        because="the unload releases the runner and the boundary without "
                "saving first",
        test="test_unload_order_is_save_then_runner_then_boundary",
    ),
    Reversion(
        target=_SM,
        find="""            if not session.is_dirty:
                # Superseded""",
        replace="""            if False:
                # Superseded""",
        because="an async save queued behind the unload's final save warns "
                "that a change was lost when nothing was",
        test="test_a_superseded_save_after_release_is_silent",
    ),
]


def _msg(text: str) -> Message:
    return Message(role=Role.USER, parts=[Part(text=text)])


class _Rpc:
    """The runner side: a history, and a hook run while it is being read."""

    def __init__(self, history: List[Message]) -> None:
        self.history = history
        self.during_fetch: Optional[Callable[[], None]] = None

    def session_get_history_threadsafe(self, *a: Any, **k: Any) -> list:
        snapshot = list(self.history)
        if self.during_fetch is not None:
            hook, self.during_fetch = self.during_fetch, None
            hook()
        return snapshot


class _Server:
    """Just what ``_save_session`` and ``_do_session_unload`` touch."""

    main_agent_id = "main"
    registry = None
    _model_running = False

    def __init__(self, rpc: _Rpc, calls: List[str]) -> None:
        self._runner_rpc = rpc
        self._runner_released = False
        self._agents: dict = {}
        self._calls = calls

    def _find_plugin_for_command(self, name: str) -> None:
        return None

    def shutdown(self) -> None:
        self._calls.append("runner_release")
        self._runner_released = True
        self._runner_rpc = None


class _Harness:
    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.calls: List[str] = []
        self.rpc = _Rpc([_msg("t1")])
        self.server = _Server(self.rpc, self.calls)
        self.manager = SessionManager()
        self.session = Session(
            session_id="s1", name="s1", server=self.server,  # type: ignore[arg-type]
            created_at="2026-10-05T00:00:00",
            workspace_path=str(self.workspace),
            is_dirty=True,
        )
        self.manager._sessions["s1"] = self.session
        real_save = self.manager._save_session

        def _save(session: Session) -> bool:
            self.calls.append("save")
            return real_save(session)

        self.manager._save_session = _save  # type: ignore[method-assign]
        self.manager._release_apparmor_boundary = (  # type: ignore[method-assign]
            lambda sid: self.calls.append("boundary_release"))

    def saved_history(self) -> list:
        record = (self.manager._session_storage_dir(str(self.workspace))
                  / "s1.json")
        return json.loads(record.read_text()).get("history", [])

    def turn_ends(self) -> None:
        """What the turn does while an async save is reading history."""
        self.rpc.history.append(_msg("t2"))
        self.session.is_dirty = True


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return _Harness(tmp_path)


def test_a_mark_made_during_a_save_survives_it(harness) -> None:
    harness.rpc.during_fetch = harness.turn_ends
    assert harness.manager._save_session(harness.session) is True
    assert harness.session.is_dirty, (
        "the save read the history before the turn ended and still cleared "
        "the dirty mark the turn's end made")


def test_a_save_that_saw_everything_clears_the_flag(harness) -> None:
    """The control: no mark during the save, so the flag clears."""
    assert harness.manager._save_session(harness.session) is True
    assert not harness.session.is_dirty


def test_unload_persists_the_turn_that_ended_during_an_async_save(
        harness) -> None:
    # The async save started at a tool call, racing the turn's end.
    harness.rpc.during_fetch = harness.turn_ends
    harness.manager._save_session(harness.session)
    harness.calls.clear()

    harness.manager._do_session_unload("s1")

    assert harness.calls[:1] == ["save"], (
        f"the unload skipped its final save: {harness.calls}")
    assert "t2" in json.dumps(harness.saved_history()), (
        "the last turn is not in the persisted history")


def test_unload_order_is_save_then_runner_then_boundary(harness) -> None:
    harness.manager._do_session_unload("s1")
    assert harness.calls == ["save", "runner_release", "boundary_release"]
    assert "t1" in json.dumps(harness.saved_history())


def test_a_superseded_save_after_release_is_silent(harness, caplog) -> None:
    """The unload's final save captured everything; an async save queued
    behind it has nothing to write and must not claim a loss."""
    harness.manager._do_session_unload("s1")
    with caplog.at_level(logging.DEBUG,
                         logger="jaato_server.server.session_manager"):
        assert harness.manager._save_session(harness.session) is False
    warnings = [r for r in caplog.records
                if r.levelno >= logging.WARNING and "Not saving" in r.message]
    assert warnings == []


def test_a_dirty_session_after_release_still_warns(harness, caplog) -> None:
    """The control: a change no save captured is reported, not hidden."""
    harness.manager._do_session_unload("s1")
    harness.session.is_dirty = True
    with caplog.at_level(logging.WARNING,
                         logger="jaato_server.server.session_manager"):
        assert harness.manager._save_session(harness.session) is False
    assert any("Not saving session s1" in r.message for r in caplog.records)


def test_the_generation_is_bumped_by_every_dirty_mark() -> None:
    s = Session(session_id="x", name="x", server=object(),  # type: ignore[arg-type]
                created_at="2026-10-05T00:00:00")
    before = s.dirty_generation
    s.is_dirty = True
    s.is_dirty = True
    s.is_dirty = False
    assert s.dirty_generation == before + 2
    assert s.clear_dirty_if_unchanged(before) is False
    s.is_dirty = True
    assert s.clear_dirty_if_unchanged(s.dirty_generation) is True
    assert s.is_dirty is False
