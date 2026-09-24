"""``scaffold.integration <name>`` on the daemon: the router verb (1.21).

The sibling of ``scaffold.explain`` (#1263).  ``explain`` renders a topic
from the daemon's install; this RUNS ``jaato-scaffold integration <name>
--refresh`` into the caller's own workspace on that install, so the
``jaato-sdk`` skill it writes carries the stamp of the ``jaato-server`` that
serves the session — the "which install?" answer the verb exists to keep with
the daemon.  Pinned here:

- a valid name applies the refresh and the event carries the four documented
  ``--refresh`` fields (``state_before`` / ``state_after`` / ``changed`` /
  ``skipped_reason``) plus the target path;
- a refresh the contract DECLINED (an edited copy) is ``ok=True`` with a
  ``skipped_reason`` — leaving a local edit alone is correct, not a failure;
- an unknown integration is refused with the daemon's own ``available`` list,
  and ``refresh`` is never reached;
- having no resolvable workspace is a refusal, not a crash, and never reaches
  ``refresh``;
- an empty name is refused, naming what the daemon ships;
- the target resolves under the caller's workspace at the manifest's own path;
- the prefixed dispatcher routes the verb and still routes ``scaffold.explain``.

The ``--refresh`` decision itself is #1261's (``integrations.refresh``); this
verb only reads its result, so these tests STUB ``integrations.refresh``
rather than re-deriving the safe-state rule the handler must not own.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import ScaffoldIntegrationEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.shared.scaffold import integrations


def _make_router(session_workspace=None):
    session_manager = MagicMock()
    session = MagicMock() if session_workspace is not None else None
    if session is not None:
        session.workspace_path = session_workspace
    session_manager.get_client_session.return_value = session
    session_manager.get_session.return_value = None
    event_sink = MagicMock()
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = event_sink
    return router, session_manager, event_sink


def _only_answer(event_sink) -> ScaffoldIntegrationEvent:
    sent = [call[0][1] for call in event_sink.send_event.call_args_list]
    assert len(sent) == 1, sent
    assert isinstance(sent[0], ScaffoldIntegrationEvent)
    return sent[0]


def _stub_refresh(monkeypatch, **result):
    calls = []

    def fake(name, dest, **kwargs):
        calls.append((name, dest, kwargs))
        return result

    # raising=False: `refresh` is #1261's addition and may not be present in
    # this checkout yet — the stub is what lets the handler be exercised now.
    monkeypatch.setattr(integrations, "refresh", fake, raising=False)
    return calls


class TestApplies:
    def test_a_valid_name_refreshes_and_reports_the_fields(self, tmp_path, monkeypatch):
        calls = _stub_refresh(
            monkeypatch,
            state_before="absent", state_after="current", changed=True,
            skipped_reason="", lines=["integrated Claude Code: X"],
        )
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_scaffold_integration("c1", ["claude-code"], None)

        ans = _only_answer(sink)
        assert ans.ok is True
        assert ans.integration == "claude-code"
        assert (ans.state_before, ans.state_after, ans.changed) == (
            "absent", "current", True)
        assert ans.skipped_reason == ""
        assert ans.text == "integrated Claude Code: X"
        assert ans.server_version  # the stamp's version, best-effort
        # The target is the caller's workspace joined onto the manifest path.
        assert ans.target == str(tmp_path / ".claude" / "skills" / "jaato-sdk")
        # refresh was called once, with that same destination.
        assert len(calls) == 1
        assert calls[0][0] == "claude-code"
        assert calls[0][1] == tmp_path / ".claude" / "skills" / "jaato-sdk"

    def test_a_skipped_refresh_is_ok_with_a_reason(self, tmp_path, monkeypatch):
        _stub_refresh(
            monkeypatch,
            state_before="edited", state_after="edited", changed=False,
            skipped_reason="the installed copy was changed since it was applied",
            lines=["left alone (edited): ..."],
        )
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_scaffold_integration("c1", ["claude-code"], None)

        ans = _only_answer(sink)
        assert ans.ok is True            # a skipped refresh is not a failure
        assert ans.changed is False
        assert "changed since it was applied" in ans.skipped_reason


class TestRealRefresh:
    """The handler over the REAL ``integrations.refresh`` (#1261), no stub:
    the skill lands in the workspace, and a second call finds it current."""

    def test_absent_then_current(self, tmp_path):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_scaffold_integration("c1", ["claude-code"], None)
        ans = _only_answer(sink)
        assert ans.ok is True and ans.changed is True
        assert (ans.state_before, ans.state_after) == ("absent", "current")
        skill = tmp_path / ".claude" / "skills" / "jaato-sdk" / "SKILL.md"
        assert skill.is_file(), "the daemon verb wrote the skill into the workspace"
        assert ans.target == str(tmp_path / ".claude" / "skills" / "jaato-sdk")

        # A second call finds it current: no change.  --refresh declines an
        # already-current copy, so state stays ``current`` and changed is
        # False (the web client shows no notice for the current state).
        sink.send_event.reset_mock()
        router._handle_scaffold_integration("c1", ["claude-code"], None)
        again = _only_answer(sink)
        assert again.ok is True and again.changed is False
        assert again.state_before == "current" and again.state_after == "current"


class TestRefusals:
    def test_unknown_integration_names_what_ships_and_never_refreshes(
            self, tmp_path, monkeypatch):
        calls = _stub_refresh(monkeypatch, changed=True)
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_scaffold_integration("c1", ["no-such-tool"], None)

        ans = _only_answer(sink)
        assert ans.ok is False
        assert "unknown integration" in ans.error
        assert "claude-code" in ", ".join(ans.available)
        assert calls == []               # refused before the payload copier

    def test_no_workspace_is_a_refusal_not_a_crash(self, monkeypatch):
        calls = _stub_refresh(monkeypatch, changed=True)
        router, _, sink = _make_router(session_workspace=None)
        router._handle_scaffold_integration("c1", ["claude-code"], None)

        ans = _only_answer(sink)
        assert ans.ok is False
        assert "no workspace" in ans.error
        assert calls == []

    def test_empty_name_names_what_ships(self, tmp_path):
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        router._handle_scaffold_integration("c1", [], None)
        ans = _only_answer(sink)
        assert ans.ok is False
        assert "name is required" in ans.error
        assert "claude-code" in ", ".join(ans.available)


class TestDispatch:
    def test_the_prefixed_dispatcher_routes_the_verb(self, tmp_path, monkeypatch):
        _stub_refresh(monkeypatch, state_before="absent", state_after="current",
                      changed=True, skipped_reason="", lines=[])
        router, _, sink = _make_router(session_workspace=str(tmp_path))
        handled = router._dispatch_prefixed_command(
            "scaffold.integration", "c1", ["claude-code"], None, str(tmp_path))
        assert handled is True
        assert _only_answer(sink).ok is True

    def test_scaffold_explain_still_routes(self):
        router, _, _ = _make_router()
        router._handle_scaffold_explain = MagicMock()
        assert router._dispatch_prefixed_command(
            "scaffold.explain", "c1", ["paths"], None, None) is True
        router._handle_scaffold_explain.assert_called_once()

    def test_an_unknown_verb_falls_through(self):
        router, _, sink = _make_router()
        assert router._dispatch_prefixed_command(
            "scaffold.other", "c1", [], None, None) is False
        sink.send_event.assert_not_called()
