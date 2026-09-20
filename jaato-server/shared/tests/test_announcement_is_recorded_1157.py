"""The Art. 50(1) announcement is RECORDED, not only emitted (#1157).

``test_first_interaction_announcement.py`` pins that a profile declaring
``interacts_with_persons: true`` announces itself once, at creation.
What nothing pinned -- because nothing did it -- was writing down that it
happened.  ``jaato_sdk.audit.AUDIT_SCHEMA`` declared six events across
the five stores and the announcement was none of them, so a deployer
asked "was this person told, and what were they told?" had the
framework's intent, an event a client may or may not have rendered, and
nothing ``jaato-doctor --audit-verify`` could vouch for.

One event, ``announcement``, in the LEDGER (the store that chains),
binding the four facts that used to sit in four places: the text as
delivered, the channel, the locale, the model identity.  Six properties,
each attached to a way it could silently stop holding:

A. **one writer, named by the schema.**  ``announcement_record`` is what
   the schema's ``written_by`` points at, so the contract guard reads its
   source for every promised field;
B. **both outcomes are recorded.**  Emitted, and suppressed by the
   client's ``client_discloses_ai`` -- suppressed is a VALUE, absent is
   not a record of anything;
C. **a revive is recorded as a revive.**  Nothing is re-announced, and
   the ledger says so instead of stopping at the creation row;
D. **absent is not defaulted.**  A locale nobody declared is omitted, and
   a revive with no client attached carries no channel;
E. **a profile that declared nothing gets no row**, for the reason it
   gets no announcement: a row would be a determination nobody made;
F. **the daemon's row lands in the RUNNER's file, first.**  Two ledgers
   over one path continue one chain, and the announcement precedes the
   first ``response`` because the turn has not happened yet;
F'. **and in the session's workspace.**  A relative ``trace.ledger`` is
   resolved against the workspace root the session context publishes,
   never against the daemon's cwd -- which is where the row landed on the
   evidence harness while the conformance daemon, started from the
   workspace, passed on the same code;
G. **the flags reach the daemon at all.**  ``PresentationContext``
   declared ``client_discloses_ai`` and carried it through neither
   ``to_dict`` nor ``from_dict``, so the suppression #1116 shipped never
   held over the wire -- found by writing the suppressed record's test.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk import audit
from jaato_sdk.audit_chain import verify
from jaato_sdk.events import ClientType, PresentationContext
from shared.ai_disclosure import (
    ANNOUNCEMENT_AGENT_ID,
    CLIENT_DISCLOSES,
    announcement_record,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.token_accounting import TokenLedger

_MANAGER = "jaato-server/server/session_manager.py"
_CORE = "jaato-server/server/core.py"
_DISCLOSURE = "jaato-server/shared/ai_disclosure.py"
_EVENTS = "jaato-sdk/jaato_sdk/events.py"

REVERSIONS = [
    Reversion(
        target=_MANAGER,
        find="        self._record_ai_interaction(\n"
             "            session, text=text, suppressed=(reason == CLIENT_DISCLOSES))",
        replace="        pass  # reversion: the announcement is emitted and never recorded",
        because=(
            "an announcement nobody wrote down is the state #1157 reports: "
            "emitted, and unprovable afterwards"
        ),
        test="test_an_emitted_announcement_is_recorded_with_its_four_facts",
    ),
    Reversion(
        target=_MANAGER,
        find="            elif reason != CLIENT_DISCLOSES:\n"
             "                return      # the profile declared nothing, or declared false",
        replace="            else:\n"
                "                return      # reversion: a suppression leaves no record",
        because=(
            "a suppressed announcement must be recorded as suppressed, never "
            "as absent -- the client took the obligation, and the row is the "
            "only thing that says so"
        ),
        test="test_a_suppressed_announcement_is_recorded_as_suppressed",
    ),
    Reversion(
        target=_MANAGER,
        find="        self._record_revived_ai_interaction(session)",
        replace="        pass  # reversion: a wake leaves no record",
        because=(
            "a woken session's ledger must say why nothing was announced, "
            "or a wake is indistinguishable from a session that never "
            "continued"
        ),
        test="test_the_revive_path_records_and_the_create_path_emits",
    ),
    Reversion(
        target=_DISCLOSURE,
        find="        locale = getattr(presentation, \"locale\", None)\n"
             "        if locale:\n"
             "            record[\"locale\"] = str(locale)",
        replace="        record[\"locale\"] = str(getattr(presentation, \"locale\", None) or \"en\")",
        because=(
            "absent is recorded as absent, never defaulted: a locale nobody "
            "declared written as 'en' asserts a language the person may not "
            "have been addressed in"
        ),
        test="test_a_locale_nobody_declared_is_absent_not_defaulted",
    ),
    Reversion(
        target=_CORE,
        find="        with self._with_session_env(), self._in_workspace():\n"
             "            self.ledger._record(\"announcement\", record)",
        replace="        with self._with_session_env():\n"
                "            self.ledger._record(\"announcement\", record)",
        because=(
            "a relative trace.ledger resolves against the workspace root the "
            "session context publishes; without it the row lands under the "
            "daemon's cwd -- the evidence harness read NOT WRITTEN while the "
            "conformance daemon, started from the workspace, passed"
        ),
        test="test_the_row_lands_in_the_sessions_workspace_not_the_daemons_cwd",
    ),
    Reversion(
        target=_EVENTS,
        find='            client_discloses_ai=bool(data.get("client_discloses_ai", False)),',
        replace='            client_discloses_ai=False,  # reversion: dropped on ingest',
        because=(
            "the suppression #1116 shipped was declared on the model and "
            "carried by neither serializer, so a client that asserted it "
            "disclosed already was announced to anyway"
        ),
        test="test_the_two_disclosure_flags_and_the_locale_survive_the_wire",
    ),
]


# ------------------------------------------------------------- fixtures

class _Reg:
    def __init__(self, interacts=None, provider_name=None):
        self.interacts_with_persons = interacts
        self.provider_name = provider_name
        self.disclosure_text = None


def _server(tmp_path: Path, reg=None, presentation=None, *, ledger_name="ledger.jsonl",
            chained=True):
    """A real ``JaatoServer`` bound to a stand-in, the #1116 idiom.

    The three methods under test are the framework's
    (``disclosure_decision``, ``disclosure_announcement``,
    ``record_disclosure_announcement``); the state they read is
    fabricated because constructing a whole server needs a provider, a
    registry and a workspace, none of which this question involves.
    The ledger is real and writes to ``tmp_path`` through the session
    env, exactly as a daemon's would.
    """
    from server.core import JaatoServer

    class _Srv:
        pass

    srv = _Srv()
    srv._profile = type("P", (), {"regulatory": reg})()
    srv._presentation_context = presentation
    srv._model_provider = "echo"
    srv._model_name = "echo-1"
    srv._session_id = "20260920_120000"
    srv._client_user_id = None
    # What the daemon's resolved session env actually carries: the ledger
    # path and the integrity posture, NOT the workspace root -- that is
    # published by ``_in_workspace``, which is why the stand-in binds it.
    srv._session_env = {
        "LEDGER_PATH": ledger_name,
        **({"JAATO_LEDGER_INTEGRITY": "sha256-chain"} if chained else {}),
    }
    srv._workspace_path = str(tmp_path)
    srv._config_root = None
    srv.ledger = TokenLedger()
    srv.session_id = srv._session_id
    # The framework's own methods, bound to the stand-in -- including the
    # two context managers that route a relative ``trace.ledger`` to the
    # session's workspace rather than the daemon's cwd.
    for name in ("disclosure_decision", "disclosure_announcement",
                 "record_disclosure_announcement", "_with_session_env",
                 "_in_workspace"):
        setattr(srv, name, getattr(JaatoServer, name).__get__(srv, _Srv))
    return srv


def _manager(emitted: Optional[List[Any]] = None):
    from server.session_manager import SessionManager
    mgr = SessionManager.__new__(SessionManager)
    mgr._emit_to_client = lambda cid, ev: (emitted if emitted is not None else []).append(ev)
    return mgr


def _session(server, created_by: Optional[str] = None):
    s = type("S", (), {})()
    s.server = server
    s.session_id = server._session_id
    s.created_by = created_by
    return s


def _rows(tmp_path: Path, name="ledger.jsonl") -> List[Dict[str, Any]]:
    path = tmp_path / name
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------ A. one writer, named

def test_the_schema_names_the_event_in_the_ledger_and_its_writer():
    event = audit.event("announcement")
    assert event is not None, "AUDIT_SCHEMA declares no `announcement` event"
    assert event.store == "ledger", "the record must ride the store that chains"
    assert event.written_by == "shared/ai_disclosure.py::announcement_record"
    guaranteed = set(audit.guaranteed_fields("announcement"))
    assert {"session_id", "agent_id", "suppressed", "revived"} <= guaranteed
    # And the facts the issue asks to bind are declared, as measured-when-known.
    declared = {f.name for f in event.fields}
    assert {"text", "client_type", "client_discloses_ai", "locale",
            "provider", "model", "created_by"} <= declared
    assert audit.AUDIT_SCHEMA_VERSION == "2", (
        "adding an event bumps the schema version a reader pins against")


def test_the_writer_emits_every_guaranteed_field_and_no_null():
    rec = announcement_record("sid")
    for name in audit.guaranteed_fields("announcement"):
        if name in ("stage", "ts", "iso_ts", "event_index"):
            continue        # the ledger's own stamps
        assert name in rec, f"{name} is promised and not written"
    assert None not in rec.values(), "absent is omitted, never null"
    assert rec["agent_id"] == ANNOUNCEMENT_AGENT_ID == "main"


# --------------------------------------------- B. both outcomes recorded

def test_an_emitted_announcement_is_recorded_with_its_four_facts(tmp_path):
    from jaato_sdk.events import AgentOutputEvent
    from server.session_manager import SessionManager

    emitted: List[Any] = []
    srv = _server(tmp_path, _Reg(True, "Acme GmbH"),
                  PresentationContext(client_type=ClientType.WEB, locale="de-DE"))
    mgr = _manager(emitted)
    SessionManager._announce_ai_interaction(mgr, "client_1", _session(srv, "app:alice"))

    assert [type(e) for e in emitted] == [AgentOutputEvent]
    rows = _rows(tmp_path)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["stage"] == "announcement"
    assert row["text"] == emitted[0].text, "the record must carry the text AS DELIVERED"
    assert row["client_type"] == "web"
    assert row["client_discloses_ai"] is False
    assert row["locale"] == "de-DE"
    assert (row["provider"], row["model"]) == ("echo", "echo-1")
    assert row["session_id"] == "20260920_120000"
    assert row["agent_id"] == "main"
    assert row["created_by"] == "app:alice"
    assert row["suppressed"] is False and row["revived"] is False
    assert row.get("digest") and row.get("prev_digest") == "genesis", (
        "the row must ride the chain the profile declared")


def test_a_suppressed_announcement_is_recorded_as_suppressed(tmp_path):
    from server.session_manager import SessionManager

    emitted: List[Any] = []
    srv = _server(tmp_path, _Reg(True, "Acme GmbH"),
                  PresentationContext(client_type=ClientType.CHAT, client_discloses_ai=True))
    assert srv.disclosure_decision() == (None, CLIENT_DISCLOSES)
    SessionManager._announce_ai_interaction(_manager(emitted), "c", _session(srv))

    assert emitted == [], "the client took the obligation; the framework says nothing"
    rows = _rows(tmp_path)
    assert len(rows) == 1, "suppressed is a value; absent is not a record of anything"
    row = rows[0]
    assert row["suppressed"] is True
    assert row["client_discloses_ai"] is True
    assert row["client_type"] == "chat"
    assert "text" not in row, "nothing was delivered, so no text claims to have been"


# ---------------------------------------------- C. a revive is a revive

def test_the_revive_path_records_and_the_create_path_emits():
    """Source-level: exactly one emit site, and the revive path RECORDS.

    Behavioural for the recorder itself below; here the property is about
    call sites, which a behavioural test can only exercise once it knows
    about them -- and the failure guarded against is a wake nobody thought
    to record.
    """
    tree = ast.parse(Path(_MANAGER).read_text())

    def callers(attr):
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr]

    def enclosing(call):
        return [fn.name for fn in ast.walk(tree)
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(c is call for c in ast.walk(fn))]

    revive = callers("_record_revived_ai_interaction")
    assert len(revive) == 1, "the revive path must record exactly once"
    assert "_load_session_impl" in enclosing(revive[0]), enclosing(revive[0])

    # And #1116's property survives: one emit site, on the create path.
    announce = callers("_announce_ai_interaction")
    assert len(announce) == 1
    assert "_create_session_impl" in enclosing(announce[0])


def test_a_revived_session_is_recorded_as_revived_with_no_emit(tmp_path):
    from server.session_manager import SessionManager

    emitted: List[Any] = []
    srv = _server(tmp_path, _Reg(True, "Acme GmbH"), presentation=None)
    SessionManager._record_revived_ai_interaction(_manager(emitted), _session(srv))

    assert emitted == []
    [row] = _rows(tmp_path)
    assert row["revived"] is True and row["suppressed"] is False
    assert "text" not in row
    # D. no client attached yet: no channel is invented for it.
    assert "client_type" not in row and "client_discloses_ai" not in row
    assert (row["provider"], row["model"]) == ("echo", "echo-1")


# ------------------------------------------- D. absent is not defaulted

def test_a_locale_nobody_declared_is_absent_not_defaulted():
    rec = announcement_record(
        "sid", text="x", presentation=PresentationContext(client_type=ClientType.API))
    assert "locale" not in rec
    assert rec["client_type"] == "api"
    rec = announcement_record(
        "sid", text="x",
        presentation=PresentationContext(client_type=ClientType.API, locale="es"))
    assert rec["locale"] == "es"
    assert PresentationContext().locale is None


def test_a_binding_not_yet_known_is_absent():
    rec = announcement_record("sid", provider="", model=None)
    assert "provider" not in rec and "model" not in rec and "created_by" not in rec


# ------------------------------------- E. a silent profile gets no row

def test_a_profile_that_declared_nothing_gets_no_row(tmp_path):
    from server.session_manager import SessionManager

    for reg in (None, _Reg(), _Reg(False)):
        emitted: List[Any] = []
        srv = _server(tmp_path, reg, PresentationContext())
        SessionManager._announce_ai_interaction(_manager(emitted), "c", _session(srv))
        SessionManager._record_revived_ai_interaction(_manager(emitted), _session(srv))
        assert emitted == []
    assert _rows(tmp_path) == [], "a row for a profile that made no determination IS one"


# ------------------------------- F. the daemon's row precedes the runner's

def test_the_daemon_row_and_the_runner_rows_form_one_chain_in_order(tmp_path):
    """Two ``TokenLedger`` instances over one file: the daemon writes the
    announcement at creation, the runner writes the turns after.  The
    chain must be one chain and the announcement must come first."""
    srv = _server(tmp_path, _Reg(True), PresentationContext())
    srv.record_disclosure_announcement(text="You are talking to a bot.")

    runner = TokenLedger(path=str(tmp_path / "ledger.jsonl"), integrity="sha256-chain")
    runner._record("response", {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    runner._record("response", {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines]
    assert [r["stage"] for r in rows] == ["announcement", "response", "response"]
    intact, breaks = verify(lines)
    assert intact, breaks
    assert rows[1]["prev_digest"] == rows[0]["digest"], (
        "the runner's first record must link to the daemon's announcement")


def test_the_row_lands_in_the_sessions_workspace_not_the_daemons_cwd(tmp_path, monkeypatch):
    """A relative ``trace.ledger`` is one file per session, resolved against
    the SESSION's workspace.  The daemon's cwd is the server package
    directory on the evidence harness and the workspace on the conformance
    daemon -- so the first saw ``NOT WRITTEN`` and the second passed, on the
    same code.  The record must not depend on where the daemon was
    started from."""
    elsewhere = tmp_path / "daemon-cwd"
    elsewhere.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)

    srv = _server(workspace, _Reg(True), PresentationContext(),
                  ledger_name=".jaato/logs/ledger.jsonl")
    srv.record_disclosure_announcement(text="You are talking to a bot.")

    assert (workspace / ".jaato" / "logs" / "ledger.jsonl").is_file(), (
        "the row must land in the session's workspace")
    assert not (elsewhere / ".jaato").exists(), (
        "and never under the daemon's cwd")


def test_an_unchained_ledger_still_gets_the_row(tmp_path):
    srv = _server(tmp_path, _Reg(True), PresentationContext(), chained=False)
    srv.record_disclosure_announcement(text="You are talking to a bot.")
    [row] = _rows(tmp_path)
    assert row["stage"] == "announcement" and "digest" not in row


def test_a_record_that_cannot_land_costs_a_warning_not_the_session(tmp_path, caplog):
    from server.session_manager import SessionManager

    class Exploding:
        def disclosure_decision(self):
            return "You are talking to a bot.", None

        def record_disclosure_announcement(self, **kw):
            raise OSError("disk full")

    emitted: List[Any] = []
    s = type("S", (), {})()
    s.server = Exploding()
    s.session_id = "sid"
    s.created_by = None
    with caplog.at_level(logging.WARNING, logger="server.session_manager"):
        SessionManager._announce_ai_interaction(_manager(emitted), "c", s)
    assert len(emitted) == 1, "the person is still told"
    assert any("NOT recorded" in r.getMessage() for r in caplog.records), (
        "a disclosure that cannot be proved must be audible, not silent")


# --------------------------------------------- G. the flags cross the wire

def test_the_two_disclosure_flags_and_the_locale_survive_the_wire():
    """``client_discloses_ai`` was declared and carried by neither
    serializer, so a client asserting it disclosed already was announced
    to anyway -- the suppression guard held on the predicate and never on
    the wire.  ``renderable_media`` had the same gap (#824)."""
    sent = PresentationContext(client_type=ClientType.CHAT, client_discloses_ai=True,
                               locale="de-DE", renderable_media=["audio/*"])
    got = PresentationContext.from_dict(sent.to_dict())
    assert got.client_discloses_ai is True
    assert got.locale == "de-DE"
    assert got.renderable_media == ["audio/*"]
    assert got.client_type == ClientType.CHAT

    # An older client sends none of them: the model's own defaults, which
    # are the safe direction (announce; no locale claimed).
    old = PresentationContext.from_dict({"client_type": "terminal"})
    assert old.client_discloses_ai is False and old.locale is None
    assert old.renderable_media == []


def test_explain_oversight_names_the_record():
    from shared.scaffold.explain import _announcement_lines

    spoken = "\n".join(_announcement_lines(
        {"text": "You are talking to a bot.", "withheld_reason": None}))
    assert "`announcement` record" in spoken and "explain audit" in spoken
