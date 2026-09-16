"""``workspace.ignore <path>`` on the daemon: the router verb (protocol 1.12).

The TUI toggles a ``.gitignore`` entry by writing the file itself; this verb
serves the same toggle to a client that has no file to write.  Pinned here:

- the entry is added, and the same press removes it again (the shared
  ``toggle_gitignore_pattern`` semantics — exact match, one line);
- the SESSION's workspace wins over the client's declared one, and having
  neither is a refusal, not a crash;
- a pattern that is not a workspace entry is refused with its reason;
- a write failure answers rather than raises;
- every outcome is exactly one ``WorkspaceIgnoreResultEvent`` on the
  caller's channel — a panel has to render *something* for the press;
- the prefixed dispatcher routes the verb and still routes ``cascade.*``.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import WorkspaceIgnoreResultEvent

from server.command_router import CommandRouter


def _make_router(session_workspace=None, client_workspace="/client-ws"):
    session_manager = MagicMock()
    session = MagicMock() if session_workspace is not None else None
    if session is not None:
        session.workspace_path = session_workspace
    session_manager.get_client_session.return_value = session
    event_sink = MagicMock()
    event_sink.get_client_workspace.return_value = client_workspace
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = event_sink
    return router, session_manager, event_sink


def _only_answer(event_sink) -> WorkspaceIgnoreResultEvent:
    sent = [call[0][1] for call in event_sink.send_event.call_args_list]
    assert len(sent) == 1, sent
    assert isinstance(sent[0], WorkspaceIgnoreResultEvent)
    return sent[0]


class TestToggle:
    def test_adds_the_entry_and_the_same_press_removes_it(self, tmp_path):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        gi = tmp_path / ".gitignore"
        gi.write_text("*.pyc")  # no trailing newline, deliberately

        router._handle_workspace_ignore("c1", ["build/"], None)
        first = _only_answer(sink)
        assert (first.ok, first.ignored, first.path) == (True, True, "build/")
        assert first.gitignore_path == str(gi)
        assert gi.read_text() == "*.pyc\nbuild/\n"

        sink.send_event.reset_mock()
        router._handle_workspace_ignore("c1", ["build/"], None)
        second = _only_answer(sink)
        assert (second.ok, second.ignored) == (True, False)
        assert gi.read_text() == "*.pyc\n"

    def test_creates_the_file_when_there_is_none(self, tmp_path):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_workspace_ignore("c1", ["src/app.py"], None)
        assert _only_answer(sink).ok is True
        assert (tmp_path / ".gitignore").read_text() == "src/app.py\n"


class TestWhichWorkspace:
    def test_the_sessions_workspace_wins_over_the_clients(self, tmp_path):
        session_ws = tmp_path / "session"
        client_ws = tmp_path / "client"
        session_ws.mkdir(); client_ws.mkdir()
        router, _, sink = _make_router(session_workspace=str(session_ws))
        router._handle_workspace_ignore("c1", ["x"], str(client_ws))
        assert _only_answer(sink).gitignore_path == str(session_ws / ".gitignore")
        assert not (client_ws / ".gitignore").exists()

    def test_falls_back_to_the_clients_declared_workspace(self, tmp_path):
        router, _, sink = _make_router(session_workspace=None)
        router._handle_workspace_ignore("c1", ["x"], str(tmp_path))
        assert _only_answer(sink).gitignore_path == str(tmp_path / ".gitignore")

    def test_no_workspace_at_all_is_a_refusal(self):
        router, _, sink = _make_router(session_workspace=None, client_workspace=None)
        router._handle_workspace_ignore("c1", ["x"], None)
        ans = _only_answer(sink)
        assert ans.ok is False and "no workspace" in ans.error


class TestRefusals:
    @pytest.mark.parametrize("bad", [[], [""], ["/etc/passwd"], ["a\nb"], ["#x"], ["!x"]])
    def test_not_a_workspace_entry(self, tmp_path, bad):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_workspace_ignore("c1", bad, None)
        ans = _only_answer(sink)
        assert ans.ok is False and ans.error.startswith("workspace.ignore:")
        assert not (tmp_path / ".gitignore").exists()

    def test_a_write_failure_answers_rather_than_raises(self, tmp_path):
        # A directory where the file should be: open(..., "w") raises IsADirectoryError.
        (tmp_path / ".gitignore").mkdir()
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_workspace_ignore("c1", ["x"], None)
        ans = _only_answer(sink)
        assert ans.ok is False and "could not write" in ans.error
        assert ans.gitignore_path == str(tmp_path / ".gitignore")


class TestDispatch:
    def test_the_prefixed_dispatcher_routes_the_verb(self, tmp_path):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        handled = router._dispatch_prefixed_command(
            "workspace.ignore", "c1", ["y"], None, None)
        assert handled is True
        assert _only_answer(sink).ok is True

    def test_cascade_still_reaches_its_own_dispatcher(self):
        router, _, _ = _make_router()
        router._dispatch_cascade_command = MagicMock(return_value=True)
        assert router._dispatch_prefixed_command(
            "cascade.cancel", "c1", ["k"], None, None) is True
        router._dispatch_cascade_command.assert_called_once_with(
            "cascade.cancel", "c1", ["k"], None)

    def test_an_unknown_verb_falls_through(self):
        router, _, sink = _make_router()
        assert router._dispatch_prefixed_command(
            "workspace.other", "c1", [], None, None) is False
        sink.send_event.assert_not_called()
