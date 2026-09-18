"""The audit record -- EU AI Act, Arts. 12, 13(3)(f), 19 (#1119).

#1109 made the ledger reach disk.  What it did not do is say what is
recorded: five stores record, none declared what is GUARANTEED, and so
Article 13(3)(f)'s "describe the mechanisms … to properly collect, store
and interpret the logs" would have had to be reverse-engineered from five
formats.  And nothing could say how long any of it is kept, which is
Arts. 19(1) and 26(6).

Six properties, each attached to a way it could silently stop holding:

A. **the schema is ENFORCED, not described** -- a guard walks the writers
   and fails when a field the schema promises stops being written.  A
   schema nothing checks is a wish, and a wish in a compliance document
   is worse than silence;
B. the block reaches EVERY ingress -- the #1113 lockstep: producer,
   consumer and the isolated-runner allow-list move together, or a
   retention policy is dropped at the one boundary that crosses a trust
   line with nothing said;
C. inheritance is most-restrictive-wins, and "most restrictive" is
   LONGER for a minimum -- with 0 unable to win it, so a child cannot
   disable a retention an ancestor set;
D. absent is not defaulted -- a profile with no block deletes exactly as
   it always has;
E. only POSITIVE evidence expires a record;
F. `workspace.delete` refuses rather than destroying a held record.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from jaato_sdk import audit
from server.record_retention import (
    describe_policy,
    expired_paths,
    judge,
    workspace_retention_hold,
)
from shared.plugins.subagent.config import (
    INTEGRITY_MODES,
    PROFILE_FILE_KEYS,
    RecordKeepingConfig,
    SubagentProfile,
    merge_record_keeping,
    profile_from_snapshot,
    profile_to_snapshot,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_CONFIG = "jaato-server/shared/plugins/subagent/config.py"
_RETENTION = "jaato-server/server/record_retention.py"
_WORKSPACE = "jaato-server/server/workspace_manager.py"
_SESSION = "jaato-server/shared/jaato_session.py"

_DAY = 86400.0

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="        record = {\n            'prompt_tokens': response.usage.prompt_tokens,",
        replace="        record = {\n            'prompt_tokens_RENAMED': response.usage.prompt_tokens,",
        because=(
            "the schema is enforced, not described: a field AUDIT_SCHEMA "
            "promises must still be written, or the contract a deployer "
            "reads is fiction"
        ),
        test="test_every_guaranteed_field_is_still_written",
    ),
    Reversion(
        target=_CONFIG,
        find="        bounded = [v for v in values if v > 0]\n        return max(bounded) if bounded else 0",
        replace="        return min(values)",
        because=(
            "a minimum inherits LONGEST-wins: min() would let a child "
            "shorten a retention its base declared, which is the one thing "
            "most-restrictive-wins exists to prevent"
        ),
        test="test_a_child_may_keep_longer_never_shorter",
    ),
    Reversion(
        target=_RETENTION,
        find='''    age = _age_days(target, stamp)
    if age is None:
        return RetentionVerdict(
            str(target), False, retention_days, None,''',
        replace='''    age = _age_days(target, stamp)
    if age is None:
        return RetentionVerdict(
            str(target), True, retention_days, None,''',
        because=(
            "only positive evidence expires a record: a path whose age "
            "cannot be read must be KEPT, because the alternative is "
            "deleting a record because its age could not be established"
        ),
        test="test_an_unreadable_age_never_expires_a_record",
    ),
    Reversion(
        target=_WORKSPACE,
        find="        hold = workspace_retention_hold(path)\n        if hold:",
        replace="        hold = workspace_retention_hold(path)\n        if False:",
        because=(
            "workspace.delete must refuse a held record rather than "
            "destroying it: a retention any delete silently defeats is "
            "not a retention"
        ),
        test="test_workspace_delete_refuses_a_held_record",
    ),
]


# ------------------------------------------------- A. enforced, not described

def _writer_source(written_by: str) -> str:
    """The source of the function an :class:`AuditEvent` names as its writer."""
    rel, _, func = written_by.partition("::")
    path = Path("jaato-server") / rel
    assert path.is_file(), f"schema names a writer file that is gone: {rel}"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func):
            return ast.unparse(node)
    raise AssertionError(
        f"schema names {written_by}, and no such function exists")


_LEDGER_EVENTS = [e for e in audit.AUDIT_SCHEMA if e.store == "ledger"]

#: The fields EVERY ledger event carries.  Stamped by ``TokenLedger``
#: itself (``_record`` / ``_enrich``), not by the caller -- so they are
#: checked against the ledger and the per-event fields against the
#: per-event writer.  Derived by intersection rather than listed, so a
#: field that stops being universal stops being checked here and starts
#: being checked where it is actually written.
_LEDGER_STAMPED = frozenset.intersection(*(
    frozenset(audit.guaranteed_fields(e.kind)) for e in _LEDGER_EVENTS))


def _mentions(source: str, name: str) -> bool:
    return f"'{name}'" in source or f'"{name}"' in source


@pytest.mark.parametrize("event", _LEDGER_EVENTS, ids=lambda e: e.kind)
def test_every_guaranteed_field_is_still_written(event):
    """A field the schema promises must appear in its writer's source.

    Source-level rather than behavioural, because the claim is about the
    CONTRACT: a deployer reading `explain audit` is told these fields are
    always there, and the only way that stays true is for the writer to
    keep writing them.  Scoped to the ledger events, whose writers build
    a literal dict -- the trace-line events assemble their text with a
    formatter, where a key-name scan would be guesswork.

    **Each field is checked against exactly ONE source.**  The first
    version of this test accepted a field found in the writer OR in the
    ledger, and the reversion meta-guard caught it: ``_enrich`` mentions
    ``prompt_tokens`` for its own arithmetic, so the fallback happily
    certified a ``_record_token_usage`` that had stopped writing it.  An
    OR across two sources is an assertion about neither.
    """
    writer = _writer_source(event.written_by)
    ledger = (_writer_source("shared/token_accounting.py::_record")
              + _writer_source("shared/token_accounting.py::_enrich"))

    for name in audit.guaranteed_fields(event.kind):
        source, where = ((ledger, "TokenLedger") if name in _LEDGER_STAMPED
                         else (writer, event.written_by))
        assert _mentions(source, name), (
            f"AUDIT_SCHEMA promises {event.kind}.{name} and "
            f"{where} no longer writes it")


def test_the_schema_names_writers_that_exist():
    # A schema pointing at a function that was renamed is a schema nobody
    # can check, which is how it stops being enforced without anything
    # failing.
    for event in audit.AUDIT_SCHEMA:
        _writer_source(event.written_by)


def test_every_event_lands_in_a_declared_store():
    keys = {s.key for s in audit.STORES}
    for event in audit.AUDIT_SCHEMA:
        assert event.store in keys, (
            f"{event.kind} lands in {event.store!r}, which is not a store")


def test_the_conversation_is_not_governed_by_the_audit_clock():
    """The session record is the CONVERSATION and has its own clock.

    Keeping the log while dropping the history is what satisfies
    Art. 19(1) and GDPR erasure at once, and it is not expressible if
    the two share a number.
    """
    assert audit.store("session_record").retained is False
    assert all(audit.store(k).retained
               for k in ("ledger", "session_trace", "provider_trace"))


def test_explain_audit_reads_the_schema():
    from shared.scaffold.explain import _audit_schema_view

    view = _audit_schema_view()
    assert view["schema_version"] == audit.AUDIT_SCHEMA_VERSION
    assert [e["kind"] for e in view["events"]] == [
        e.kind for e in audit.AUDIT_SCHEMA]
    # An unguaranteed field must be RENDERED as such -- `?` in the page --
    # or a reader takes "absent" for "zero".
    response = next(e for e in view["events"] if e["kind"] == "response")
    user_id = next(f for f in response["fields"] if f["name"] == "user_id")
    assert user_id["guaranteed"] is False


# ------------------------------------------------------ B. every ingress

def test_the_block_round_trips_through_a_snapshot():
    keeping = RecordKeepingConfig(retention_days=180,
                                  conversation_retention_days=30,
                                  integrity="sha256-chain")
    profile = SubagentProfile(name="p", description="d",
                              record_keeping=keeping)
    snapshot = profile_to_snapshot(profile)
    assert snapshot["record_keeping"] == {
        "retention_days": 180,
        "conversation_retention_days": 30,
        "integrity": "sha256-chain",
    }
    assert profile_from_snapshot(snapshot).record_keeping == keeping


def test_it_is_a_profile_file_key():
    assert "record_keeping" in PROFILE_FILE_KEYS


def test_it_crosses_the_isolated_runner_boundary():
    """Producer, consumer and allow-list in LOCKSTEP -- #1113's lesson.

    The isolated-subagent spawn is the one profile path that crosses a
    trust boundary, and the daemon REJECTS unknown keys by design.  A
    block the producer forgets is a retention policy silently not
    crossing.
    """
    from server.runner_rpc_handlers.profile_payload_schema import (
        PROFILE_PAYLOAD_ALLOWED_KEYS, validate_profile_payload,
    )
    from shared.plugins.subagent.plugin import _record_keeping_wire_shape

    assert "record_keeping" in PROFILE_PAYLOAD_ALLOWED_KEYS

    profile = SubagentProfile(
        name="p", description="d",
        record_keeping=RecordKeepingConfig(retention_days=180))
    wire = _record_keeping_wire_shape(profile)
    assert wire == {"record_keeping": {"retention_days": 180}}

    validate_profile_payload({"name": "p", **wire})   # must not raise

    # And the daemon re-runs the ONE rule rather than carrying a second
    # vocabulary that could drift.
    with pytest.raises(ValueError, match="record_keeping"):
        validate_profile_payload({"name": "p",
                                  "record_keeping": {"retention_days": -1}})


def test_a_profile_declaring_nothing_emits_nothing():
    from shared.plugins.subagent.plugin import _record_keeping_wire_shape

    profile = SubagentProfile(name="p", description="d")
    assert _record_keeping_wire_shape(profile) == {}
    assert profile_to_snapshot(profile)["record_keeping"] is None


# ------------------------------------------------------- C. inheritance

def _prof(**kw):
    return SubagentProfile(name="p", description="d",
                           record_keeping=RecordKeepingConfig(**kw))


def test_a_child_may_keep_longer_never_shorter():
    base = _prof(retention_days=180)
    shorter = _prof(retention_days=30)
    longer = _prof(retention_days=365)

    assert merge_record_keeping(shorter, [base]).retention_days == 180, (
        "a child must not be able to shorten its base's retention")
    assert merge_record_keeping(longer, [base]).retention_days == 365, (
        "a child may keep longer than its base")


def test_zero_cannot_disable_an_ancestors_retention():
    # 0 means "keep until something deletes it", which is the LEAST
    # restrictive thing a layer can say -- so it must lose to any real
    # minimum, and win only when every declaring layer said it.
    assert merge_record_keeping(
        _prof(retention_days=0), [_prof(retention_days=180)]
    ).retention_days == 180
    assert merge_record_keeping(
        _prof(retention_days=0), [_prof(retention_days=0)]
    ).retention_days == 0


def test_integrity_is_ranked_most_restrictive_wins():
    assert merge_record_keeping(
        _prof(integrity="none"), [_prof(integrity="sha256-chain")]
    ).integrity == "sha256-chain"
    assert INTEGRITY_MODES == ("none", "sha256-chain")


def test_no_layer_declaring_it_merges_to_nothing():
    plain = SubagentProfile(name="p", description="d")
    assert merge_record_keeping(plain, [plain]) is None


# ------------------------------------------------ D. absent is not defaulted

def test_a_profile_with_no_block_has_none():
    assert SubagentProfile(name="p", description="d").record_keeping is None


def test_an_empty_block_is_not_a_policy():
    # A knob that changes what DELETE MEANS has to be one somebody wrote.
    assert RecordKeepingConfig().declared is False
    assert RecordKeepingConfig(retention_days=0).declared is True
    assert describe_policy(None).startswith("record_keeping: undeclared")


def test_a_switch_written_into_a_day_count_is_refused():
    # bool is an int subclass; `retention_days: true` must not read as 1.
    with pytest.raises(ValueError, match="integer number of days"):
        RecordKeepingConfig.from_dict({"retention_days": True})
    with pytest.raises(ValueError, match=">= 0"):
        RecordKeepingConfig.from_dict({"retention_days": -1})
    with pytest.raises(ValueError, match="unknown key"):
        RecordKeepingConfig.from_dict({"retention_dayz": 5})


# ------------------------------------------------- E. positive evidence only

def test_an_unreadable_age_never_expires_a_record(tmp_path):
    verdict = judge(tmp_path / "never-existed.jsonl", 5, now=time.time())
    assert verdict.expired is False
    assert "absence of evidence" in verdict.reason


def test_no_policy_and_zero_both_keep(tmp_path):
    target = tmp_path / "ledger.jsonl"
    target.write_text("{}")
    ancient = time.time() + 10_000 * _DAY
    assert judge(target, None, now=ancient).expired is False
    assert judge(target, 0, now=ancient).expired is False


def test_a_record_past_its_minimum_expires(tmp_path):
    target = tmp_path / "ledger.jsonl"
    target.write_text("{}")
    now = time.time()
    assert judge(target, 180, now=now).expired is False
    assert judge(target, 180, now=now + 179 * _DAY).expired is False
    assert judge(target, 180, now=now + 181 * _DAY).expired is True


def test_both_halves_are_reported(tmp_path):
    """A pass that names only what it removed cannot answer 'why is this
    still here?'."""
    old = tmp_path / "old.jsonl"
    old.write_text("{}")
    expired, kept = expired_paths(
        [old, tmp_path / "gone.jsonl"], 30, now=time.time() + 60 * _DAY)
    assert [v.path for v in expired] == [str(old)]
    assert len(kept) == 1


# -------------------------------------------------------- F. delete refuses

def _workspace_with_retention(tmp_path, days=180):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".jaato" / "logs").mkdir(parents=True)
    (tmp_path / ".jaato" / "logs" / "ledger.jsonl").write_text("{}")
    (tmp_path / ".jaato" / "profiles" / "p.yaml").write_text(
        "name: p\ndescription: d\nmodel: m\nprovider: anthropic\n"
        "plugins: []\n"
        f"record_keeping:\n  retention_days: {days}\n"
        "trace:\n  ledger: .jaato/logs/ledger.jsonl\n")
    return tmp_path


def test_a_held_workspace_names_what_holds_it(tmp_path):
    ws = _workspace_with_retention(tmp_path)
    hold = workspace_retention_hold(ws)
    assert hold is not None
    assert "180" in hold and "Art. 19(1)" in hold
    assert "session.delete" in hold, (
        "the refusal must name the verb that DOES work")


def test_the_hold_lifts_once_the_minimum_elapses(tmp_path):
    ws = _workspace_with_retention(tmp_path)
    assert workspace_retention_hold(ws, now=time.time() + 200 * _DAY) is None


def test_a_workspace_declaring_nothing_is_never_held(tmp_path):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "p.yaml").write_text(
        "name: p\ndescription: d\nmodel: m\nprovider: anthropic\n"
        "plugins: []\n")
    assert workspace_retention_hold(tmp_path) is None


def test_an_unresolvable_workspace_is_not_held(tmp_path):
    """A workspace nobody can delete because its profiles stopped parsing
    is a worse outcome than one deleted without a retention check -- and
    a retention that cannot be established is not evidence of one."""
    assert workspace_retention_hold(tmp_path / "does-not-exist") is None


def test_workspace_delete_refuses_a_held_record(tmp_path):
    from server.workspace_manager import WorkspaceManager

    root = tmp_path / "root"
    root.mkdir()
    mgr = WorkspaceManager(str(root))
    ws = _workspace_with_retention(root / "held")
    mgr.create_workspace("held") if not ws.exists() else None

    with pytest.raises(ValueError, match="under retention"):
        mgr.delete_workspace("held")
    assert ws.exists(), "nothing may be removed before the refusal"
