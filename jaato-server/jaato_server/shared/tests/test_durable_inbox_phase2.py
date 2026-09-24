"""The durable inbox -- session group messaging, phase 2.

Phase 1 made ``send_to_session`` a receipt: ``accepted`` or ``queued`` said
the target HELD the message, and everything the target could not take right
now was a refusal the sender had to retry.  Two of those refusals were the
common case, not the edge: a busy target whose message carries bytes (bytes
ride the drive branch only, #845), and a cold target whose revive failed.
And ``queued`` was only as durable as the runner's in-memory queue -- an
unload between the queue and the turn that drains it lost the message with
no receipt saying so.

Phase 2 writes the message to the target's own record directory
(``server.session_inbox``) BEFORE the sender gets its receipt, and drains it
at the next point the target can take a turn.  The claims, each pinned
here with a reversion where the guard would otherwise be decorative:

- a busy target with bytes is SPOOLED whole, never queued stripped, never
  refused -- and the plugin reports ``spooled`` as a success;
- a ``queued`` message leaves a copy that the turn's end CONSUMES, and that
  a load re-arms when the turn never completed;
- the turn-end hook drives the oldest spooled message on the idle-only
  SIBLING tier with its bytes re-inflated;
- a failed revive is spooled and the lifetime watchdog revives with backoff;
- BACKPRESSURE (the pending cap) is still a refusal -- a spool would defeat
  the cap;
- a terminated target drops the entry (never woken, §4.7); a failed drive
  keeps it with the attempt counted;
- the ``event_id`` dedup is durable: a redelivery after the in-memory LRU
  is gone finds its spooled entry;
- ``delete_session`` removes the inbox with the record; the listing counts
  what is pending, and the wire row carries it.

Lives in ``shared/tests`` so the reversion meta-suite walks it.
"""

import pathlib
import threading
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from jaato_sdk.events import AgentStatusChangedEvent, SessionListEvent
from jaato_sdk.plugins.model_provider.types import (
    UNTRUSTED_OPEN, tool_result_is_error,
)
from jaato_server.server import session_inbox
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import RuntimeSessionInfo, SessionManager
from jaato_server.server.session_workspace_index import SessionWorkspaceIndex
from jaato_server.shared.message_queue import SourceType
from jaato_server.shared.tool_result_builder import split_executor_result
from jaato_server.shared.tests.reversion import Reversion
from .offer_double import wire_offer

_SM = "jaato-server/jaato_server/server/session_manager.py"

ATT = [{"mime_type": "audio/wav", "data": "QUJD", "display_name": "q.wav",
        "attachment_id": "sha256:beef"}]


class _Server:
    def __init__(self, running=False, main_agent_id="main"):
        self._model_running = running
        self._runner_rpc = None
        self._terminal_reason = None
        self._profile = NS(name="p", budget_control=None)
        self.main_agent_id = main_agent_id

    @property
    def is_processing(self):
        return self._model_running


def _session(sid, *, owner="app:u", running=False, ws, clients=()):
    return NS(session_id=sid, cascade_driver_id=None, created_by=owner,
              sibling_name=None, workspace_path=ws,
              attached_clients=set(clients), description=None,
              interrupted_turn=None, is_dirty=False,
              server=_Server(running))


def _sm(tmp_path, *sessions, cold=()):
    """A SessionManager skeleton with the group surface AND the inbox wired.

    Every workspace is under ``tmp_path`` because the inbox writes beside
    the session record; ``cold`` rows go into a real index.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {s.session_id: s for s in sessions}
    sm._lock = threading.RLock()
    sm._group_pending, sm._group_exchanges = {}, {}
    sm._inbox_dirty, sm._inbox_cold_retry = set(), {}
    sm._inbox_draining = set()
    sm._session_config = NS(storage_path=".jaato/sessions")
    sm._wake_seen_event_ids = OrderedDict()
    sm._session_workspace_index = SessionWorkspaceIndex(tmp_path / "index.json")
    for sid, ws, owner in cold:
        sm._session_workspace_index.record(sid, ws)
        sm._session_workspace_index.record_membership(
            sid, created_by=owner, cascade_driver_id=None, sibling_name=None)
    sm._get_persisted_sessions = lambda workspace_path=None: []
    sm.delivered, sm.revived = [], []
    sm.send_message_to_session = (
        lambda sid, text, attachments=None:
        sm.delivered.append((sid, text, "driven", attachments)) or True)

    def _resume(sid, workspace_path=None):
        sm.revived.append((sid, workspace_path))
        s = _session(sid, ws=workspace_path)
        wire_offer(s, sm.delivered)
        sm._sessions[sid] = s
        return sid
    sm.resume_session = _resume
    for s in sessions:
        wire_offer(s, sm.delivered)
    return sm


def _ws(tmp_path, name):
    p = tmp_path / name
    p.mkdir(exist_ok=True)
    return str(p)


def _inbox(sm, ws, sid):
    return session_inbox.pending(sm._session_storage_dir(ws), sid)


def _send(sm, sender="s-a", target="s-b", text="hello", **kw):
    return sm.deliver_group_message(sender, target, text, **kw)


# ------------------------------------------------------------- spooling

def test_a_busy_peer_with_bytes_is_spooled_whole(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             _session("s-b", ws=ws, running=True))
    r = _send(sm, text="", attachments=ATT)
    assert r["status"] == "spooled" and r["spooled"] is True
    assert r["message_id"] and r["attachments"] == 1
    assert sm.delivered == [], "nothing reached the runner-side queue"
    [entry] = _inbox(sm, ws, "s-b")
    assert entry.message_id == r["message_id"]
    assert entry.kind == "peer" and entry.source == "peer:s-a"
    assert entry.runner_queued is False
    assert session_inbox.load_attachments(sm._session_storage_dir(ws), entry) == ATT
    assert "s-b" in sm._inbox_dirty


def test_a_queued_message_leaves_a_copy_the_turn_end_consumes(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             _session("s-b", ws=ws, running=True))
    r = _send(sm)
    assert r["status"] == "queued" and r["spooled"] is True
    assert sm.delivered[0][3] is SourceType.SIBLING, "the runner has the text"
    [entry] = _inbox(sm, ws, "s-b")
    assert entry.runner_queued is True
    # Not driven by the turn's end -- the running turn consumed it -- and
    # not by anything else either: the copy exists for the unload case.
    assert sm.drain_session_inbox("s-b", trigger="sweep") is False
    assert len(_inbox(sm, ws, "s-b")) == 1
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is False
    assert _inbox(sm, ws, "s-b") == []
    assert len(sm.delivered) == 1, "the copy was never driven a second time"


def test_a_load_rearms_a_copy_whose_turn_never_completed(tmp_path):
    """The daemon died mid-turn: the runner's queue is gone with it, so the
    copy is what delivers the message.  A load clears the flag and puts the
    session on the drain schedule; nothing is driven by the load itself."""
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-b", ws=ws))
    sm._spool_inbox(sm._session_storage_dir(ws), session_id="s-b", kind="peer",
                    source="peer:s-a", source_id="s-a", text="hi", items=[],
                    runner_queued=True)
    sm._inbox_dirty.clear()
    assert sm._prepare_inbox_after_load("s-b") == 1
    [entry] = _inbox(sm, ws, "s-b")
    assert entry.runner_queued is False and sm.delivered == []
    assert "s-b" in sm._inbox_dirty
    assert sm.drain_session_inbox("s-b", trigger="sweep") is True
    assert sm.delivered[0][2] == "driven" and _inbox(sm, ws, "s-b") == []


# ------------------------------------------------------------- draining

def test_the_turn_end_drives_a_spooled_message_on_the_sibling_tier(tmp_path):
    ws = _ws(tmp_path, "b")
    b = _session("s-b", ws=ws, running=True)
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")), b)
    assert _send(sm, text="look", attachments=ATT)["status"] == "spooled"
    b.server._model_running = False          # the turn ended
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is True
    sid, text, how, atts = sm.delivered[0]
    assert (sid, how) == ("s-b", "driven")
    assert atts == ATT, "the bytes were re-inflated from the spool"
    assert text.startswith(UNTRUSTED_OPEN) and "peer:s-a" in text
    assert "q.wav" in text and "QUJD" not in text
    assert _inbox(sm, ws, "s-b") == [] and "s-b" not in sm._inbox_dirty


def test_one_entry_per_drain_so_a_backlog_drains_one_turn_at_a_time(tmp_path):
    ws = _ws(tmp_path, "b")
    b = _session("s-b", ws=ws, running=True)
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")), b)
    for i in range(3):
        assert _send(sm, text=f"m{i}", attachments=ATT)["status"] == "spooled"
    b.server._model_running = False
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is True
    assert len(sm.delivered) == 1 and "m0" in sm.delivered[0][1]
    assert len(_inbox(sm, ws, "s-b")) == 2
    assert "s-b" in sm._inbox_dirty, "still on the schedule for the next boundary"


def test_the_turn_end_hook_schedules_the_drain_before_the_unload_check(tmp_path):
    ws = _ws(tmp_path, "b")
    b = _session("s-b", ws=ws)
    sm = _sm(tmp_path, b)
    sm._inbox_dirty.add("s-b")
    calls = []
    sm._schedule_inbox_drain = lambda sid, *, trigger: calls.append(("drain", sid, trigger))
    sm._maybe_unload_session = lambda sid, **kw: calls.append(("unload", sid))
    sm._handle_turn_tracking_event(
        b, AgentStatusChangedEvent(agent_id="main", status="done"))
    assert calls == [("drain", "s-b", "turn_end"), ("unload", "s-b")]
    # ...and a session with nothing spooled costs the hook nothing.
    calls.clear()
    sm._inbox_dirty.clear()
    sm._handle_turn_tracking_event(
        b, AgentStatusChangedEvent(agent_id="main", status="done"))
    assert calls == [("unload", "s-b")]


def test_a_terminated_target_drops_the_entry_and_a_failed_drive_keeps_it(tmp_path):
    ws = _ws(tmp_path, "b")
    b = _session("s-b", ws=ws, running=True)
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")), b)
    assert _send(sm, text="x", attachments=ATT)["status"] == "spooled"
    b.server._model_running = False
    # a drive that fails: the entry stays, the attempt is counted
    sm.send_message_to_session = lambda sid, text, attachments=None: False
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is False
    [entry] = _inbox(sm, ws, "s-b")
    assert entry.attempts == 1 and entry.last_error == "unreachable"
    # a target that ENDED on an error is never woken: the entry is dropped
    b.server._terminal_reason = "error"
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is False
    assert _inbox(sm, ws, "s-b") == []


# ------------------------------------------------------------- cold targets

def test_a_failed_revive_is_spooled_and_the_sweep_retries_with_backoff(tmp_path):
    ws = _ws(tmp_path, "cold")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             cold=[("s-cold", ws, "app:u")])
    real_resume = sm.resume_session
    sm.resume_session = lambda sid, workspace_path=None: (
        sm.revived.append((sid, workspace_path)) or None)
    r = _send(sm, target="s-cold")
    assert r["status"] == "spooled" and len(_inbox(sm, ws, "s-cold")) == 1
    # the first attempt is due at once (the spool's own instant), then the
    # backoff doubles from 30s; the sweep takes ``now`` so the clock is ours
    t, delay = sm._inbox_cold_retry["s-cold"]
    assert delay == 30.0
    sm._sweep_inbox(now=t)                    # first attempt: the next sweep
    assert len(sm.revived) == 2 and sm._inbox_cold_retry["s-cold"] == (t + 30.0, 60.0)
    sm._sweep_inbox(now=t + 10)               # not due yet
    assert len(sm.revived) == 2
    sm._sweep_inbox(now=t + 31)               # due: fails again, backs off
    assert len(sm.revived) == 3 and sm._inbox_cold_retry["s-cold"] == (t + 91.0, 120.0)
    sm.resume_session = real_resume
    sm._sweep_inbox(now=t + 200)              # due: revives and drains
    assert sm.delivered[0][0] == "s-cold" and sm.delivered[0][2] == "driven"
    assert "peer:s-a" in sm.delivered[0][1]
    assert _inbox(sm, ws, "s-cold") == [] and "s-cold" not in sm._inbox_cold_retry


def test_the_backoff_is_capped_and_an_expired_entry_ends_the_retry(tmp_path):
    ws = _ws(tmp_path, "cold")
    sm = _sm(tmp_path, cold=[("s-cold", ws, "app:u")])
    sm.resume_session = lambda sid, workspace_path=None: None
    sm._spool_inbox(sm._session_storage_dir(ws), session_id="s-cold", kind="peer",
                    source="peer:s-a", source_id="s-a", text="hi", items=[],
                    expires_at=5000.0)
    sm._note_inbox_cold("s-cold", now=0.0)
    sm._inbox_cold_retry["s-cold"] = (0.0, 600.0)
    sm._sweep_inbox(now=0.0)
    assert sm._inbox_cold_retry["s-cold"] == (600.0, 600.0), "capped, not doubled"
    sm._sweep_inbox(now=6000.0)
    assert "s-cold" not in sm._inbox_cold_retry and _inbox(sm, ws, "s-cold") == []


def test_a_loaded_session_that_went_cold_is_handed_to_the_retry_path(tmp_path):
    ws = _ws(tmp_path, "b")
    b = _session("s-b", ws=ws, running=True)
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")), b)
    assert _send(sm, text="x", attachments=ATT)["status"] == "spooled"
    sm._sessions.pop("s-b")                   # unloaded before the drain
    sm._session_workspace_index.record("s-b", ws)
    sm._sweep_inbox(now=1.0)
    assert "s-b" in sm._inbox_cold_retry and "s-b" not in sm._inbox_dirty


# ------------------------------------------------------------- what is NOT spooled

def test_a_backpressure_refusal_is_never_spooled(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             _session("s-b", ws=ws, running=True))
    assert _send(sm, pending_cap=1)["status"] == "queued"
    r = _send(sm, pending_cap=1)
    assert r["status"] == "refused" and "waiting" in r["error"]
    assert [e.runner_queued for e in _inbox(sm, ws, "s-b")] == [True], (
        "only the queued copy; the refused message was not spooled")


def test_a_message_that_could_not_be_spooled_is_a_visible_refusal(tmp_path):
    """No inbox to write (the storage dir is a FILE): the outcome the
    delivery already had is reported, never a receipt for a message nobody
    holds."""
    ws = _ws(tmp_path, "b")
    (pathlib.Path(ws) / ".jaato").mkdir()
    (pathlib.Path(ws) / ".jaato" / "sessions").write_text("not a dir")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             _session("s-b", ws=ws, running=True))
    r = _send(sm, text="", attachments=ATT)
    assert r["status"] == "refused" and "spooled" not in r


# ------------------------------------------------------------- dedup

def test_a_redelivered_event_id_finds_its_spooled_entry_after_a_restart(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             _session("s-b", ws=ws, running=True))
    assert _send(sm, text="", attachments=ATT, event_id="e-1")["status"] == "spooled"
    sm._wake_seen_event_ids.clear()           # the in-memory LRU is gone
    r = _send(sm, text="", attachments=ATT, event_id="e-1")
    assert r["status"] == "duplicate"
    assert len(_inbox(sm, ws, "s-b")) == 1


# ------------------------------------------------------------- lifecycle

def test_delete_session_removes_the_inbox_with_the_record(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path, _session("s-b", ws=ws))
    sm._spool_inbox(sm._session_storage_dir(ws), session_id="s-b", kind="peer",
                    source="peer:s-a", source_id="s-a", text="hi", items=ATT)
    sm._inbox_cold_retry["s-b"] = (0.0, 30.0)
    sm._client_to_session = {}
    sm._session_plugin = NS(delete=lambda sid, storage_dir=None: True)
    sm._wake_binding_registry = NS(release_for_session=lambda sid: 0)
    sm._sessions["s-b"].server.shutdown = lambda: None
    assert sm.delete_session("s-b") is True
    assert not session_inbox.inbox_dir(sm._session_storage_dir(ws), "s-b").exists()
    assert "s-b" not in sm._inbox_dirty and "s-b" not in sm._inbox_cold_retry


def test_the_listing_counts_the_inbox(tmp_path):
    ws = _ws(tmp_path, "b")
    sm = _sm(tmp_path)
    for _ in range(2):
        sm._spool_inbox(sm._session_storage_dir(ws), session_id="s-b", kind="peer",
                        source="peer:s-a", source_id="s-a", text="hi", items=[])
    assert sm._inbox_pending_count("s-b", ws) == 2
    assert sm._inbox_pending_count("s-b", None) == 0
    # a persisted-only (cold) row carries the count too
    sm._client_config, sm._orphan_since, sm._ever_attached = {}, {}, set()
    sm._normalize_workspace = lambda p: p
    sm._get_persisted_sessions = lambda workspace_path=None: [NS(
        session_id="s-b", description=None, created_at=NS(isoformat=lambda: ""),
        updated_at=NS(isoformat=lambda: ""), turn_count=0, workspace_path=ws)]
    sm._session_workspace_index.record("s-b", ws)
    [row] = sm.list_sessions()
    assert row.inbox_pending == 2 and row.is_loaded is False


def test_the_listing_row_carries_inbox_pending():
    """The wire dict, not only the dataclass (#1133)."""
    row = RuntimeSessionInfo(
        session_id="s", name="s", description=None, created_at="",
        last_activity="", model_provider="", model_name="",
        is_processing=False, is_loaded=False, client_count=0, turn_count=0,
        inbox_pending=3)
    manager = MagicMock()
    manager.list_sessions.return_value = [row]
    sink = MagicMock(spec=["send_event", "get_client_user",
                           "get_client_workspace", "set_client_session"])
    sink.get_client_user.return_value = None
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = manager
    router._event_sink = sink
    router._handle_session_list("c1", None)
    [event] = [c[0][1] for c in sink.send_event.call_args_list
               if isinstance(c[0][1], SessionListEvent)]
    assert event.sessions[0]["inbox_pending"] == 3


# ------------------------------------------------------------- the plugin

def test_the_plugin_reports_spooled_as_success_on_both_signals(tmp_path):
    from jaato_server.shared.plugins.courier.plugin import CourierPlugin
    ws = _ws(tmp_path, "cold")
    sm = _sm(tmp_path, _session("s-a", ws=_ws(tmp_path, "a")),
             cold=[("s-cold", ws, "app:u")])
    sm.resume_session = lambda sid, workspace_path=None: None
    p = CourierPlugin()
    p.initialize({})
    p.set_plugin_registry(NS(session_id="s-a"))
    p.set_session_manager(sm)
    ok, data = split_executor_result(
        p._execute_send_to_session({"target": "s-cold", "message": "hi"}))
    assert ok is True and not tool_result_is_error(data)
    assert data["status"] == "spooled" and data["spooled"] is True


def test_the_result_event_reads_spooled_as_ok():
    from jaato_server.server.command_router import _message_result_event
    ev = _message_result_event("r1", "s-b", {"status": "spooled", "spooled": True,
                                              "message_id": "m"})
    assert ev.ok is True and ev.spooled is True and ev.status == "spooled"
    ev = _message_result_event("r1", "s-b", {"status": "queued", "spooled": True})
    assert ev.ok is True and ev.spooled is True
    ev = _message_result_event("r1", "s-b", {"status": "refused"})
    assert ev.ok is False and ev.spooled is False


# ------------------------------------------------------------- reversions

REVERSIONS = [
    Reversion(
        target=_SM,
        find="        if outcome not in spool_on:\n            return None\n",
        replace="        if outcome not in spool_on or True:\n            return None\n",
        because="nothing is ever spooled; a busy peer with bytes is refused again",
        test="test_a_busy_peer_with_bytes_is_spooled_whole",
    ),
    Reversion(
        target=_SM,
        find='                if trigger == "turn_end":\n',
        replace='                if trigger == "turn_end" and False:\n',
        because="the turn's end no longer consumes the queued copy; it leaks",
        test="test_a_queued_message_leaves_a_copy_the_turn_end_consumes",
    ),
    Reversion(
        target=_SM,
        find="        if at_cap:\n            return branch, outcome, False\n",
        replace="        if at_cap and False:\n            return branch, outcome, False\n",
        because="a backpressure refusal is spooled, which defeats the cap",
        test="test_a_backpressure_refusal_is_never_spooled",
    ),
    Reversion(
        target=_SM,
        find="        if has_inbox:\n            self._schedule_inbox_drain(",
        replace="        if has_inbox and False:\n            self._schedule_inbox_drain(",
        because="the turn-end hook never drains: a spooled message waits for the sweep or forever",
        test="test_the_turn_end_hook_schedules_the_drain_before_the_unload_check",
    ),
    Reversion(
        target=_SM,
        find="            if workspace is None or self.resume_session(\n",
        replace="            if workspace is None or True or self.resume_session(\n",
        because="the sweep never revives a cold target; a spooled message is never delivered",
        test="test_a_failed_revive_is_spooled_and_the_sweep_retries_with_backoff",
    ),
    Reversion(
        target=_SM,
        find="        if storage_dir is not None:\n            session_inbox.remove_all(storage_dir, session_id)\n",
        replace="        if storage_dir is not None and False:\n            session_inbox.remove_all(storage_dir, session_id)\n",
        because="a deleted session's inbox outlives its record",
        test="test_delete_session_removes_the_inbox_with_the_record",
    ),
]
