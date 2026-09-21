"""A session you deleted came back, carrying a listing that was not yours.

Reported from a deployed web client: the rail's Sessions section said
``1 of 169 noted`` collapsed and ``1 of 2 noted`` expanded, and the one
"other" row was a session its owner had deleted -- still listed, still
carrying the note they had written about it.  Two independent defects,
one behind each number.

**The 2 that should have been 1.**  ``SessionManager.delete_session``
reads ``workspace_path`` off the in-memory ``Session`` it pops.  A COLD
(unloaded) session has no in-memory object, so ``workspace_path`` stayed
``None``, ``storage_dir`` stayed ``None``, and the delete landed on the
session plugin's own relative fallback directory instead of on
``<workspace>/.jaato/sessions/``.  The record survived, the call returned
``False``, the daemon answered ``Session '<id>' not found.`` -- and the
session was back in the next listing.

The daemon knew the workspace the whole time.
``_session_workspace_index`` exists for exactly this ("locating a COLD
(unloaded) session's record requires knowing its workspace", its own
module docstring) and was consulted here only to ``forget`` the entry,
twenty lines BELOW the delete that needed it.

**The 169 that should have been 2.**  ``SessionInfoEvent.sessions`` and
``SessionListEvent.sessions`` answer one question -- which sessions are
there -- and disagreed.  ``session.list`` renders
``CommandRouter._sessions_visible_to`` (#1113); the state snapshot was
built from ``list_sessions()``, every session on the daemon.  So the
WIDER answer was the one every client received at attach and at create,
each row naming another user's session id, workspace path,
provider/model and model-written description.

**And the verb that destroys was outside the boundary.**  #1113 wrote it
in terms of the two verbs that READ.  ``session.delete`` took no gate at
all, which mattered less while the snapshot handed every id to everybody
and cold deletes silently did nothing; with both fixed, an id -- a
second-granularity timestamp -- was the only thing between one user and
another user's record.

Every delete case here points at a record that EXISTS and asserts the
file is gone, not that the call returned True: the unfixed code returns
``False`` *and* leaves the record, so a test asserting only the return
value would pass against a delete that removed the wrong file.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import ErrorEvent
from server.command_router import CommandRouter
from server.session_manager import SessionManager
from shared.tests.reversion import Reversion

_SESSION_MANAGER = "jaato-server/server/session_manager.py"
_COMMAND_ROUTER = "jaato-server/server/command_router.py"


# --------------------------------------------------------------------------
# 1.  A cold session's record is where the index says it is
# --------------------------------------------------------------------------

@pytest.fixture()
def manager(tmp_path, monkeypatch):
    """A real ``SessionManager`` with its index pointed at a scratch home."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return SessionManager()


def _write_cold_record(sm: SessionManager, workspace: Path, session_id: str) -> Path:
    """Put a persisted record where a cold session's record really lives."""
    sessions_dir = sm._session_storage_dir(str(workspace))
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = sessions_dir / f"{session_id}.json"
    record.write_text(json.dumps({
        "version": "2.0",
        "session_id": session_id,
        "description": "",
        "created_at": "2026-09-17T12:06:57",
        "updated_at": "2026-09-17T12:06:57",
        "turn_count": 0,
        "workspace_path": str(workspace),
        "messages": [],
    }))
    return record


def test_deleting_a_cold_session_removes_its_record(manager, tmp_path):
    ws = tmp_path / "ws"
    record = _write_cold_record(manager, ws, "20260917_120657")
    manager._session_workspace_index.record("20260917_120657", str(ws))

    assert manager.delete_session("20260917_120657") is True
    # The file, not the return value: the unfixed code returns False AND
    # leaves this behind, so only the filesystem separates the two.
    assert not record.exists()


def test_the_record_is_gone_from_the_listing_the_client_reads(manager, tmp_path):
    ws = tmp_path / "ws"
    _write_cold_record(manager, ws, "20260917_120657")
    manager._session_workspace_index.record("20260917_120657", str(ws))

    before = {s.session_id for s in manager._get_persisted_sessions(workspace_path=str(ws))}
    assert "20260917_120657" in before

    manager.delete_session("20260917_120657")

    after = {s.session_id for s in manager._get_persisted_sessions(workspace_path=str(ws))}
    assert "20260917_120657" not in after


def test_an_unknown_id_still_reports_that_it_found_nothing(manager):
    # The honest answer, and the one the daemon turns into
    # ``Session '<id>' not found.``  Resolving nothing must not start
    # reading as a successful delete.
    assert manager.delete_session("20260101_000000") is False


def test_an_ambiguous_id_is_refused_rather_than_guessed(manager, tmp_path):
    # One timestamp, two workspaces: the index marks it AMBIGUOUS and
    # ``resolve`` returns None by design.  Deleting the wrong session is
    # unrecoverable, so failing is correct -- and both records survive.
    a, b = tmp_path / "a", tmp_path / "b"
    rec_a = _write_cold_record(manager, a, "20260917_120657")
    rec_b = _write_cold_record(manager, b, "20260917_120657")
    manager._session_workspace_index.record("20260917_120657", str(a))
    manager._session_workspace_index.record("20260917_120657", str(b))

    assert manager.delete_session("20260917_120657") is False
    assert rec_a.exists() and rec_b.exists()


# --------------------------------------------------------------------------
# 2.  The snapshot's listing is the listing ``session.list`` renders
# --------------------------------------------------------------------------

def _info(sid, workspace):
    return SimpleNamespace(
        session_id=sid, name=sid, description="", model_provider="", model_name="",
        is_loaded=False, client_count=0, turn_count=0, workspace_path=workspace,
        created_by=None, orphaned=False, runner=None,
        awaiting=None, awaiting_since=None,
    )


ALL_SESSIONS = [_info("mine", "/root/alices"), _info("bobs", "/root/bobs")]


def _manager_with_resolver(resolver):
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._visible_sessions_resolver = resolver
    sm.list_sessions = lambda: list(ALL_SESSIONS)          # type: ignore[assignment]
    return sm


def _snapshot_ids(sm, client_id):
    session = SimpleNamespace(session_id="mine", name="mine", server=None,
                              user_inputs=[], sandbox_mode=None)
    ev = SessionManager._build_session_info_event(sm, session, client_id=client_id)
    return {row["id"] for row in ev.sessions}


def test_the_snapshot_shows_only_what_this_client_may_see():
    sm = _manager_with_resolver(lambda cid: [_info("mine", "/root/alices")])
    assert _snapshot_ids(sm, "c1") == {"mine"}


def test_no_resolver_is_the_unscoped_snapshot_every_transport_had():
    sm = _manager_with_resolver(None)
    assert _snapshot_ids(sm, "c1") == {"mine", "bobs"}


def test_no_client_is_the_unscoped_snapshot_too():
    sm = _manager_with_resolver(lambda cid: [])
    assert _snapshot_ids(sm, None) == {"mine", "bobs"}


def test_a_resolver_that_raises_sends_none_rather_than_everything():
    def boom(_cid):
        raise RuntimeError("transport went away")

    sm = _manager_with_resolver(boom)
    # Failing closed: an empty listing costs completions until the next
    # ``session.list``; the unscoped one hands over somebody else's rows.
    assert _snapshot_ids(sm, "c1") == set()


def test_a_manager_without_the_hook_is_tolerated_and_announced(caplog):
    # This constructor is handed arbitrary objects (two in-tree test doubles
    # among them), so raising at one that is not a full ``SessionManager``
    # is the wrong failure.  Unscoped is what every transport had before --
    # but it must not be SILENT, because the snapshot is then the wide
    # listing this whole change exists to stop sending.
    class _NoHook:
        pass

    with caplog.at_level("WARNING"):
        CommandRouter(session_manager=_NoHook(), event_sink=MagicMock(), daemon_plugins={})
    assert any("cannot scope its state snapshot" in r.getMessage()
               for r in caplog.records)


def test_the_router_lends_the_manager_its_own_visibility_rule():
    # The wiring, which is what makes the mechanism reachable at all --
    # and it must be the ROUTER's method, so the two answers cannot drift.
    sm = MagicMock()
    router = CommandRouter(session_manager=sm, event_sink=MagicMock(), daemon_plugins={})
    sm.set_visible_sessions_resolver.assert_called_once_with(router._sessions_visible_to)


# --------------------------------------------------------------------------
# 3.  ``session.delete`` is inside the boundary
# --------------------------------------------------------------------------

def _router_for_delete(paths, user="app:alice"):
    sm = MagicMock()
    sm.list_sessions.return_value = ALL_SESSIONS
    sm.delete_session.return_value = True
    sink = MagicMock(spec=["send_event", "get_client_user", "get_client_workspace",
                           "set_client_session", "visible_workspace_paths"])
    sink.get_client_user.return_value = user
    sink.visible_workspace_paths.return_value = paths
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = sm
    router._event_sink = sink
    return router, sm, sink


def test_deleting_someone_elses_session_is_refused_without_touching_the_manager():
    router, sm, sink = _router_for_delete(["/root/alices"])
    router._handle_session_delete("c1", ["bobs"])

    sm.delete_session.assert_not_called()
    err = [c[0][1] for c in sink.send_event.call_args_list if isinstance(c[0][1], ErrorEvent)]
    assert len(err) == 1
    assert err[0].error_type == "SessionError"
    # Named for the verb that refused, not for ``session.attach``.
    assert err[0].error.startswith("session.delete: bobs")


def test_deleting_your_own_session_still_works():
    router, sm, _ = _router_for_delete(["/root/alices"])
    router._handle_session_delete("c1", ["mine"])
    sm.delete_session.assert_called_once_with("mine")


def test_an_unscoped_transport_deletes_exactly_as_before():
    router, sm, _ = _router_for_delete(None)
    router._handle_session_delete("c1", ["bobs"])
    sm.delete_session.assert_called_once_with("bobs")


# --------------------------------------------------------------------------
# Reversions
# --------------------------------------------------------------------------

_COLD_LOOKUP_FIXED = """        if workspace_path is None:
            workspace_path = self._session_workspace_index.resolve(session_id)
"""

_SNAPSHOT_SCOPED_FIXED = "for s in self._sessions_for_snapshot(client_id):"
_SNAPSHOT_SCOPED_BROKEN = "for s in self.list_sessions():"

_RESOLVER_FAILS_CLOSED_FIXED = """                client_id, exc_info=True)
            return []"""
_RESOLVER_FAILS_CLOSED_BROKEN = """                client_id, exc_info=True)
            return self.list_sessions()"""

_ROUTER_WIRING_FIXED = """        else:
            lend(self._sessions_visible_to)"""
_ROUTER_WIRING_BROKEN = """        else:
            pass"""

_DELETE_GATE_FIXED = """        if self._refuse_foreign_session(
                client_id, session_id_to_delete, verb="session.delete"):
            return
"""

REVERSIONS = [
    Reversion(
        target=_SESSION_MANAGER,
        find=_COLD_LOOKUP_FIXED,
        replace="",
        test="test_deleting_a_cold_session_removes_its_record",
        because=(
            "an unloaded session's record being deleted from the session "
            "plugin's relative fallback directory instead of from its "
            "workspace -- so the session comes back in the next listing"
        ),
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find=_COLD_LOOKUP_FIXED,
        replace="",
        test="test_the_record_is_gone_from_the_listing_the_client_reads",
        because="the deleted session still being listed to its owner",
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find=_SNAPSHOT_SCOPED_FIXED,
        replace=_SNAPSHOT_SCOPED_BROKEN,
        test="test_the_snapshot_shows_only_what_this_client_may_see",
        because=(
            "every session on the daemon riding the state snapshot to any "
            "client that attaches -- the #1113 boundary bypassed by the "
            "event that answers the same question as session.list"
        ),
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find=_RESOLVER_FAILS_CLOSED_FIXED,
        replace=_RESOLVER_FAILS_CLOSED_BROKEN,
        test="test_a_resolver_that_raises_sends_none_rather_than_everything",
        because=(
            "a transport hiccup degrading to the unscoped listing, which is "
            "failing OPEN on an entitlement decision"
        ),
    ),
    Reversion(
        target=_COMMAND_ROUTER,
        find=_ROUTER_WIRING_FIXED,
        replace=_ROUTER_WIRING_BROKEN,
        test="test_the_router_lends_the_manager_its_own_visibility_rule",
        because=(
            "the scoping existing and nothing installing it -- a rule "
            "resolved and applied to nobody"
        ),
    ),
    Reversion(
        target=_COMMAND_ROUTER,
        find=_DELETE_GATE_FIXED,
        replace="",
        test="test_deleting_someone_elses_session_is_refused_without_touching_the_manager",
        because=(
            "any authenticated client deleting any session on the daemon by "
            "id, with the id a second-granularity timestamp"
        ),
    ),
]
