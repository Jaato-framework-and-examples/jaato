"""The incident register -- EU AI Act, Arts. 72, 73, 26(5) (#1122).

Article 73 gives a provider **15 days** to report a serious incident from
the moment it becomes AWARE of it -- 10 for a death, 2 for a widespread
infringement.  All three clocks start from awareness, and the framework
already KNEW when the events that could be one happened.  It recorded
none of them as such: each was a log line in a different format, in a
different file, with no severity and no clock.

Six properties, each attached to a way it could silently stop holding:

A. **every named site raises**, checked by AST -- the failure this
   guards is a site nobody remembered;
B. one event, one incident -- a budget abort already emits a rung event
   and a terminal, and counting it twice would inflate the register that
   a reporting deadline is read off;
C. **it never fails the thing it reports on** -- a record written ABOUT
   a failure must not be able to add one;
D. the trace line is machine-readable by construction and round-trips;
E. the register does NOT classify: whether an entry is a serious
   incident under Art. 3(49) is a determination about consequences no
   log line carries;
F. **unreadable is not empty** -- answering "none" to a question it
   could not look at is the one thing a register must never do.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from jaato_sdk import incidents_view
from shared.incidents import (
    ARTICLE_73_WINDOWS,
    INCIDENT_KINDS,
    KIND_BUDGET_EXHAUSTED,
    KIND_CIRCUIT_OPENED,
    KIND_CONFINEMENT_REFUSED,
    KIND_DESCRIPTIONS,
    KIND_NUDGE_EXHAUSTED,
    KIND_SESSION_ERROR,
    Incident,
    clocks,
    parse_incident_trace,
    raise_incident,
    read_register,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_CORE = "jaato-server/server/core.py"
_BOOTSTRAP = "jaato-server/server/runner/bootstrap.py"
_RELIABILITY = "jaato-server/shared/plugins/reliability/plugin.py"
_SDK = "jaato-sdk/jaato_sdk/incidents.py"
_VIEW = "jaato-sdk/jaato_sdk/incidents_view.py"

_DAY = 86400.0

REVERSIONS = [
    Reversion(
        target=_CORE,
        # ``_note_incident`` is a module-level free function rather than a
        # method: every call site is on a terminal path, so a server double
        # lacking ``emit`` must get the pre-#1122 behaviour rather than an
        # AttributeError.  The anchor moved with it.
        find='''        _note_incident(
            self, kind, f"{error_type}: {error_summary}",
            site="server/core.py::_emit_error_termination", session_id=sid)''',
        replace="        pass  # reversion: the terminal raises no incident",
        because=(
            "a session that died on a terminal error is the first thing "
            "Art. 73's clock should be running on, and it was a log line "
            "in a format nothing read"
        ),
        test="test_every_named_site_raises",
    ),
    Reversion(
        target=_BOOTSTRAP,
        find="    _raise_confinement_incident(expected_profile, scan)\n",
        replace="",
        because=(
            "a bootstrap refused for thread divergence is evidence that "
            "code ran outside the boundary the session record claims -- "
            "and it happens before any session exists, so there is nobody "
            "above to notice it"
        ),
        test="test_every_named_site_raises",
    ),
    Reversion(
        target=_CORE,
        find='                       trace_path=env.get("JAATO_TRACE_LOG"),',
        replace="",
        because=(
            "the daemon raises from model_thread's `finally`, after the "
            "session-env overlay is popped -- so reading the ambient "
            "JAATO_TRACE_LOG put session_error and nudge_exhausted in the "
            "daemon's own trace and `--incidents <workspace>` said none"
        ),
        test="test_the_daemon_writes_where_the_profile_said",
    ),
    Reversion(
        target=_SDK,
        find='''        parts.append(f"cause={self.cause!r}")''',
        replace='''        parts.insert(1, f"cause={self.cause!r}")''',
        because=(
            "free text goes LAST or the line stops parsing: a cause "
            "containing a space or an '=' would otherwise corrupt every "
            "field after it, which is the #968 grammar this reuses"
        ),
        test="test_a_cause_containing_anything_does_not_corrupt_the_line",
    ),
    Reversion(
        target=_VIEW,
        find='''            out.append(check(f"incidents {path.name}", warn,
                             f"{path}: no such trace file — NOT read, which "
                             f"is not the same as 'no incidents'"))''',
        replace='''            out.append(check(f"incidents {path.name}", pass_,
                             "none recorded"))''',
        because=(
            "unreadable is not empty: answering 'none' to a question the "
            "register could not look at tells a reader something false "
            "about a reporting deadline"
        ),
        test="test_an_unreadable_file_is_not_reported_as_clean",
    ),
]


#: Every site the register claims to cover, and the function that must
#: call the raiser.  A LIST, so a site that stops raising fails rather
#: than quietly leaving a hole -- the failure #1122 is about is a site
#: nobody remembered, and a test that only checks the sites it knows
#: about cannot see one that was dropped.
_RAISING_SITES = (
    (_CORE, "_emit_error_termination", "_note_incident"),
    (_CORE, "_emit_budget_refusal_if_exhausted", "_note_incident"),
    (_CORE, "_note_incident", "raise_incident"),
    (_BOOTSTRAP, "verify_thread_confinement", "_raise_confinement_incident"),
    (_BOOTSTRAP, "_raise_confinement_incident", "raise_incident"),
    (_RELIABILITY, "_raise_block_incident", "raise_incident"),
)


def _calls_in(path: str, function: str):
    tree = ast.parse(Path(path).read_text())
    fn = next((f for f in ast.walk(tree)
               if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
               and f.name == function), None)
    assert fn is not None, f"{path} has no function {function!r}"
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            names.add(getattr(target, "attr", None) or getattr(target, "id", None))
    return names


# ------------------------------------------------------ A. every site raises

@pytest.mark.parametrize("path,function,callee", _RAISING_SITES,
                         ids=[f"{f}" for _p, f, _c in _RAISING_SITES])
def test_every_named_site_raises(path, function, callee):
    assert callee in _calls_in(path, function), (
        f"{path}::{function} no longer calls {callee}() — the register "
        f"claims to cover this site and would silently not")


def test_the_reliability_site_is_gated_on_the_transition():
    """One circuit opening is one incident, not one per later call.

    Gated on the same ``not was_blocked`` as the hook and the telemetry:
    a tool that keeps failing while already blocked would otherwise
    produce a row per call, and a register a reporting deadline is read
    off must not be inflated by repetition.
    """
    source = Path(_RELIABILITY).read_text()
    idx = source.index("_raise_block_incident(state.tool_name")
    preceding = source[:idx]
    guard = preceding.rindex("if not was_blocked:")
    assert "def " not in preceding[guard:], (
        "the raiser must sit inside the `not was_blocked` guard")


# --------------------------------------------------- B. one event, one incident

def test_the_budget_abort_raises_from_the_terminal_not_the_rung():
    """A rung fires for notify / finalize / escalate too.

    Only the terminal means the run was STOPPED, so raising from the rung
    would put three non-events in the register for every real one.
    """
    source = Path(_CORE).read_text()
    tree = ast.parse(source)
    raisers = [f.name for f in ast.walk(tree)
               if isinstance(f, ast.FunctionDef)
               and "_note_incident" in _calls_in(_CORE, f.name)]
    assert "_apply_budget_rungs" not in raisers
    assert "_emit_budget_refusal_if_exhausted" in raisers


def test_nudge_exhaustion_is_its_own_kind_from_the_same_site():
    # It reaches the terminal through _emit_error_termination, so a
    # second call site there would count one dying session twice.  The
    # kind is derived from the error type instead.
    source = Path(_CORE).read_text()
    assert "KIND_NUDGE_EXHAUSTED if error_type ==" in source
    assert source.count("site=\"server/core.py::_emit_error_termination\"") == 1


def test_every_kind_is_described():
    assert set(KIND_DESCRIPTIONS) == set(INCIDENT_KINDS)
    assert all(KIND_DESCRIPTIONS[k].strip() for k in INCIDENT_KINDS)


# ---------------------------------------------- C. it never fails its subject

def test_a_raiser_with_no_trace_sink_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setenv("JAATO_TRACE_LOG", str(tmp_path / "t.log"))
    assert raise_incident(KIND_SESSION_ERROR, "boom", site="x::y") is not None


def test_an_exploding_session_does_not_fail_the_raiser(monkeypatch, tmp_path):
    monkeypatch.setenv("JAATO_TRACE_LOG", str(tmp_path / "t.log"))

    class Exploding:
        def _trace(self, _msg):
            raise RuntimeError("the trace file is gone")

        @property
        def _model_name(self):
            raise RuntimeError("and so is the binding")

    # Neither the binding read nor the trace write may propagate: this is
    # called ON a terminal path, so a raiser that raised would turn a
    # reported failure into an unreported one.
    assert raise_incident(KIND_SESSION_ERROR, "boom", site="x::y",
                          session=Exploding()) is None


def test_an_exploding_emit_does_not_fail_the_raiser(monkeypatch, tmp_path):
    monkeypatch.setenv("JAATO_TRACE_LOG", str(tmp_path / "t.log"))

    def _boom(_incident):
        raise RuntimeError("no client")

    assert raise_incident(KIND_SESSION_ERROR, "boom", site="x::y",
                          emit=_boom) is not None


def test_an_unknown_kind_is_still_recorded(monkeypatch, tmp_path):
    # Refusing it would drop the report of something that DID happen
    # because the vocabulary was not updated first.
    monkeypatch.setenv("JAATO_TRACE_LOG", str(tmp_path / "t.log"))
    incident = raise_incident("something_new", "boom", site="x::y")
    assert incident is not None and incident.kind == "something_new"


# ------------------------------------------------------------- D. the grammar

def test_the_line_round_trips():
    original = Incident(
        kind=KIND_BUDGET_EXHAUSTED, at=time.time(),
        cause="stopped at its ceiling", session_id="s1",
        provider="anthropic", model="claude-sonnet-5", site="a.py::b")
    assert parse_incident_trace(original.to_trace()) == original


def test_a_cause_containing_anything_does_not_corrupt_the_line():
    nasty = "tool=x allowed=False  kind=forged  and an = sign"
    original = Incident(kind=KIND_SESSION_ERROR, at=1.0, cause=nasty,
                        session_id="s1")
    parsed = parse_incident_trace(original.to_trace())
    assert parsed.cause == nasty
    assert parsed.kind == KIND_SESSION_ERROR, (
        "a cause that looks like fields must not be read as fields")
    assert parsed.session_id == "s1"


def test_a_line_that_is_not_an_incident_is_skipped():
    assert parse_incident_trace("[PERMISSION] DECISION tool=x") is None
    assert read_register(["noise", "", "also noise"]) == []


def test_the_register_is_newest_first_and_windowed():
    now = time.time()
    lines = [Incident(kind=KIND_SESSION_ERROR, at=now - n * _DAY,
                      cause=f"e{n}").to_trace() for n in (1, 20, 5)]
    within = read_register(lines, since_days=15, now=now)
    assert [i.cause for i in within] == ["e1", "e5"]


def test_absent_fields_are_omitted_not_null():
    bare = Incident(kind=KIND_SESSION_ERROR, at=1.0, cause="c")
    assert "session_id" not in bare.to_trace()
    assert set(bare.to_dict()) == {"kind", "at", "cause"}


# ---------------------------------------------------- E. it does not classify

def test_all_three_windows_are_rendered_and_none_is_chosen():
    now = time.time()
    incident = Incident(kind=KIND_SESSION_ERROR, at=now - 3 * _DAY, cause="c")
    rendered = clocks(incident, now)
    assert len(rendered) == len(ARTICLE_73_WINDOWS) == 3
    assert any("PAST" in line for line in rendered)
    assert any("left" in line for line in rendered)


def test_the_event_carries_no_severity():
    """The absence is the decision, not an omission.

    Whether an entry is a serious incident under Art. 3(49) is a
    determination about CONSEQUENCES -- harm to a person, disruption of
    critical infrastructure -- that no log line carries.
    """
    from jaato_sdk.events import IncidentEvent

    assert "severity" not in IncidentEvent.model_fields
    assert set(IncidentEvent.model_fields) >= {
        "kind", "at", "cause", "site", "provider", "model", "tier"}


def test_the_register_header_says_so(tmp_path):
    assert "does NOT" in incidents_view.HEADER.replace("NOT", "NOT")
    assert "Art. 3(49)" in incidents_view.HEADER


# --------------------------------------------------- F. unreadable is not empty

def _render(paths, **kw):
    from jaato_sdk.doctor import PASS, WARN, Check
    return incidents_view.render(paths, check=Check, pass_=PASS, warn=WARN,
                                 **kw)


def test_an_unreadable_file_is_not_reported_as_clean(tmp_path):
    from jaato_sdk.doctor import WARN

    (check,) = _render([str(tmp_path / "absent.log")])
    assert check.status == WARN
    assert "not the same as 'no incidents'" in check.detail


def test_a_readable_file_with_nothing_in_it_passes(tmp_path):
    from jaato_sdk.doctor import PASS

    quiet = tmp_path / "quiet.log"
    quiet.write_text("[PERMISSION] DECISION tool=x allowed=True\n")
    (check,) = _render([str(quiet)])
    assert check.status == PASS
    assert "none recorded" in check.detail


def test_a_file_with_incidents_lists_them(tmp_path):
    from jaato_sdk.doctor import WARN

    now = time.time()
    busy = tmp_path / "busy.log"
    busy.write_text("\n".join(
        Incident(kind=k, at=now - 2 * _DAY, cause=f"{k} happened").to_trace()
        for k in (KIND_BUDGET_EXHAUSTED, KIND_CIRCUIT_OPENED)) + "\n")

    checks = _render([str(busy)], now=now)
    assert checks[0].status == WARN
    assert "2 recorded" in checks[0].detail
    assert {c.name.strip() for c in checks[1:]} == {
        KIND_BUDGET_EXHAUSTED, KIND_CIRCUIT_OPENED}
    assert "Art. 73 windows remaining" in checks[1].detail


def test_naming_no_file_is_a_warning_not_a_clean_bill():
    from jaato_sdk.doctor import WARN, check_incidents

    (check,) = check_incidents([])
    assert check.status == WARN


def test_an_unparseable_since_shows_everything_rather_than_nothing():
    from jaato_sdk.doctor import _since_days

    assert _since_days("15d") == 15.0
    assert _since_days("15") == 15.0
    assert _since_days("last tuesday") is None, (
        "a filter that did not parse must not silently hide every row")
    assert _since_days(None) is None


# --------------------------- G. the daemon writes into the PROFILE's trace

def test_the_daemon_writes_where_the_profile_said(tmp_path, monkeypatch):
    """Not into the daemon's own process-wide trace.

    ``_note_incident`` has no session to write through, and the fallback
    read ``JAATO_TRACE_LOG`` off the process environment -- but two of
    its three call sites run inside ``model_thread``'s ``finally``, AFTER
    ``_with_session_env`` has popped the overlay.  So ``session_error``
    and ``nudge_exhausted`` went to the daemon's trace (or ``/tmp``) and
    ``jaato-doctor --incidents <workspace>/.jaato/logs/...`` reported
    "none recorded" for exactly the kinds the daemon raises.
    ``budget_exhausted`` is raised while the overlay IS applied, which is
    what made the inconsistency hard to see.
    """
    from types import SimpleNamespace

    from server.core import _note_incident

    profile_trace = tmp_path / "ws" / ".jaato" / "logs" / "session_trace.jsonl"
    daemon_trace = tmp_path / "daemon.jsonl"
    # The overlay is GONE, exactly as it is on the terminal path.
    monkeypatch.setenv("JAATO_TRACE_LOG", str(daemon_trace))

    server = SimpleNamespace(
        session_id="20260101_x",
        emit=lambda _e: None,
        workspace_path=str(tmp_path / "ws"),
        _session_env={"JAATO_TRACE_LOG": ".jaato/logs/session_trace.jsonl",
                      "JAATO_WORKSPACE_ROOT": str(tmp_path / "ws")},
    )
    _note_incident(server, KIND_SESSION_ERROR, "boom",
                   site="server/core.py::_emit_error_termination")

    assert profile_trace.is_file(), (
        "the incident must land in the trace the PROFILE declared")
    assert "INCIDENT" in profile_trace.read_text()
    assert not daemon_trace.exists(), (
        "and not in the daemon's process-wide trace")


def test_an_agent_placeholder_is_substituted_not_written_literally(
        tmp_path, monkeypatch):
    """`{agent}` in a trace path is a token, not a directory name.

    ``resolve_trace_path`` does no substitution -- that lives in
    ``jaato_sdk.trace`` -- so this route created a literal ``{agent}``
    directory, which is the #775 shape one module over.
    """
    from types import SimpleNamespace

    from server.core import _note_incident

    ws = tmp_path / "ws"
    monkeypatch.delenv("JAATO_TRACE_LOG", raising=False)
    server = SimpleNamespace(
        session_id="s", emit=lambda _e: None, workspace_path=str(ws),
        _session_env={"JAATO_TRACE_LOG": ".jaato/logs/{agent}/trace.jsonl",
                      "JAATO_WORKSPACE_ROOT": str(ws)},
    )
    _note_incident(server, KIND_SESSION_ERROR, "boom", site="x::y")

    assert not (ws / ".jaato" / "logs" / "{agent}").exists(), (
        "an unsubstituted token became a real directory")
    assert (ws / ".jaato" / "logs" / "main" / "trace.jsonl").is_file()


def test_the_confinement_refusal_still_uses_the_environment():
    """It has no session, no profile and no workspace yet.

    The environment fallback is RIGHT for that caller -- which is why the
    fix is an explicit path from the caller that has one, rather than
    removing the fallback.
    """
    import inspect

    from server.runner import bootstrap

    source = inspect.getsource(bootstrap._raise_confinement_incident)
    assert "trace_path" not in source, (
        "the pre-session caller must not be made to invent a path it "
        "cannot know")


# --------------------------------------------- H. a row names its session

def test_the_circuit_incident_names_its_session():
    """A register row reading `no session` cannot be attributed.

    On a daemon serving two, the row is indistinguishable between them,
    so an operator asking "what happened in session B in the last 15
    days" gets an answer naming neither.  The plugin holds the id.
    """
    assert "session_id" in _calls_in(_RELIABILITY, "_raise_block_incident") or (
        "session_id=getattr(self, \"_session_id\", None)"
        in Path(_RELIABILITY).read_text()), (
        "the reliability raiser must pass the session id it holds")
