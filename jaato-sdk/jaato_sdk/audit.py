"""The audit record -- Regulation (EU) 2024/1689, Articles 12, 13(3)(f), 19.

Article 12(1) asks a high-risk AI system to allow "the automatic
recording of events (logs) over the lifetime of the system"; 13(3)(f)
asks its instructions for use to describe "the mechanisms included
within the AI system that allows deployers to properly collect, store
and interpret the logs"; 19(1) asks the provider to keep those logs for
at least six months, and 26(6) asks the same of the deployer.

**This is not a sixth store.**  Five things in this framework already
record: the token ledger, the application trace, the per-agent provider
trace, the session record on disk, and the per-session logs.  What none
of them said was what is *guaranteed* to be recorded, or where -- so
13(3)(f)'s description would have had to be reverse-engineered from five
formats by whoever needed it.  This module is a CONTRACT over the stores
that exist: :data:`AUDIT_SCHEMA` enumerates the events, the fields each
carries, and the store each lands in.

Three things follow from it being a contract rather than a writer:

* **It is enforced, not described.**  A guard walks the writers and
  fails when a field named here stops being written
  (``shared/tests/test_audit_record_contract.py``).  A schema nothing
  checks is a wish, and a wish in a compliance document is worse than
  silence.
* **Absent is not zero.**  A dimension nothing reported is OMITTED from a
  record, never written as ``null`` or ``0`` -- the rule
  ``get_environment(aspect="consumption")`` already holds to.  A provider
  that reported no cache must not read as a cache that never hit.
* **It says where each field comes from**, so a deployer assembling the
  record knows which file to read and an auditor knows which file to ask
  for.

``jaato-scaffold explain audit [<profile>]`` renders this schema, and for
a named profile the concrete paths that profile writes to -- so
13(3)(f) is answered by the framework rather than written about it.

Stdlib only, and in the SDK, so a consumer can read the schema without
importing jaato-server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Bumped when an event or a guaranteed field is ADDED or REMOVED.  A
#: reader that pins this knows whether the record it is holding can
#: contain what it is looking for.
AUDIT_SCHEMA_VERSION = "1"


# --------------------------------------------------------------- stores

@dataclass(frozen=True)
class AuditStore:
    """One place records land, and how a deployer gets at it.

    Attributes:
        key: Stable identifier used by :class:`AuditEvent.store`.
        path_source: The profile key or env var that decides WHERE the
            store is, in the words an author would write.
        fmt: The on-disk shape.
        retained: Whether ``record_keeping.retention_days`` governs it.
            ``False`` for a store whose lifetime is somebody else's
            (the session record, which a deployer's own deletion
            policy governs through ``conversation_retention_days``).
    """

    key: str
    path_source: str
    fmt: str
    description: str
    retained: bool = True


LEDGER = AuditStore(
    key="ledger",
    path_source="trace.ledger  (env: LEDGER_PATH)",
    fmt="JSONL, one object per record, appended as it is recorded",
    description=(
        "Every model round trip and every permission verdict. Appended "
        "per record since #1109, so a process that dies mid-turn has "
        "written what it recorded"),
)

SESSION_TRACE = AuditStore(
    key="session_trace",
    path_source="trace.session_log  (env: JAATO_TRACE_LOG)",
    fmt="line-oriented; the DECISION lines are `key=value`, reason last",
    description=(
        "The application trace: permission DECISION lines (#951/#968), "
        "budget ceilings and rungs (#955), tool-runner resolve/result "
        "lines. The one artefact every deployment gets"),
)

PROVIDER_TRACE = AuditStore(
    key="provider_trace",
    path_source="trace.provider_log  (env: JAATO_PROVIDER_TRACE)",
    fmt="JSONL, one object per provider interaction",
    description=(
        "What went to the model and what came back, per agent. The "
        "largest store by far and the one a retention policy usually "
        "shortens first"),
)

SESSION_RECORD = AuditStore(
    key="session_record",
    path_source="<workspace>/.jaato/sessions/<session_id>/",
    fmt="JSON",
    description=(
        "The session's own state: history, profile snapshot, creator, "
        "runner identity. This is the CONVERSATION, not the audit log -- "
        "`conversation_retention_days` governs it, and it may be deleted "
        "while the audit record is kept"),
    retained=False,
)

STORES: Tuple[AuditStore, ...] = (
    LEDGER, SESSION_TRACE, PROVIDER_TRACE, SESSION_RECORD)

_STORES_BY_KEY: Dict[str, AuditStore] = {s.key: s for s in STORES}


def store(key: str) -> AuditStore:
    """The store registered under ``key``."""
    return _STORES_BY_KEY[key]


# --------------------------------------------------------------- fields

@dataclass(frozen=True)
class AuditField:
    """One field of one audit event.

    Attributes:
        name: The key as it appears in the record.
        description: What it means, in a deployer's vocabulary.
        guaranteed: ``True`` when the field is present on EVERY instance
            of the event.  A guaranteed field is what the guard checks
            the writer still writes; an unguaranteed one is present when
            it was measured and ABSENT otherwise, never ``null``.
    """

    name: str
    description: str
    guaranteed: bool = True


@dataclass(frozen=True)
class AuditEvent:
    """One kind of recorded event.

    Attributes:
        kind: The value the record carries (the ledger's ``stage``, or
            the trace line's leading token).
        article: The Article this event answers, for a reader mapping
            the framework onto the Regulation.
        store: :class:`AuditStore` key.
        fields: What the record carries.
        written_by: ``file.py::function`` -- where the record is
            produced.  Read by the guard, and by a deployer who wants to
            see the writer rather than trust this file.
        note: Anything a reader needs in order not to misread it.
    """

    kind: str
    article: str
    store: str
    written_by: str
    fields: Tuple[AuditField, ...] = field(default_factory=tuple)
    note: str = ""


#: Every field carried by every record in the ledger, whatever its stage.
_LEDGER_COMMON = (
    AuditField("stage", "which kind of record this is"),
    AuditField("ts", "unix timestamp"),
    AuditField("iso_ts", "the same instant, ISO-8601 UTC"),
    AuditField("event_index", "position in this ledger"),
)


AUDIT_SCHEMA: Tuple[AuditEvent, ...] = (
    AuditEvent(
        kind="response",
        article="Art. 12(1) -- each model round trip",
        store="ledger",
        written_by="shared/jaato_session.py::_record_token_usage",
        fields=_LEDGER_COMMON + (
            AuditField("prompt_tokens",
                       "NEW, uncached input tokens -- excludes both cache "
                       "counts (#758)"),
            AuditField("output_tokens", "tokens the model produced"),
            AuditField("total_tokens", "as the provider reported it"),
            AuditField("user_id",
                       "the authenticated user the session runs as (#859)",
                       guaranteed=False),
        ),
        note=(
            "A provider that reported NO usage writes the three token "
            "fields as null rather than omitting them, because a reported "
            "zero and an unreported one are different facts (#688) and "
            "`TokenUsage.reported` is what tells them apart"),
    ),
    AuditEvent(
        kind="permission-check",
        article="Art. 12(1), 14(4)(d) -- each tool call's verdict",
        store="ledger",
        written_by="shared/ai_tool_runner.py::_permission_gate_verdict",
        fields=_LEDGER_COMMON + (
            AuditField("tool", "the tool asked for"),
            AuditField("args", "the arguments it was asked with"),
            AuditField("allowed", "the verdict"),
            AuditField("reason", "the verdict in words"),
            AuditField("method",
                       "HOW it was reached: whitelist / default / "
                       "allow_all / a channel answer / an evaluator"),
            AuditField("agent", "which session asked (#951)",
                       guaranteed=False),
            AuditField("call_id", "correlates to the tool call",
                       guaranteed=False),
            AuditField("user_id", "who answered, when anybody did (#859)",
                       guaranteed=False),
            AuditField("approver",
                       "the name an external approval system attached "
                       "(#859); asserted, recorded as claimed",
                       guaranteed=False),
        ),
        note=(
            "`user_id` and `approver` are ABSENT for a policy decision. "
            "That is the point: 'nobody was asked' must stay "
            "distinguishable from 'somebody answered'"),
    ),
    AuditEvent(
        kind="DECISION",
        article="Art. 12(1), 14(4)(d) -- the same verdict, in the one "
                "artefact every deployment gets",
        store="session_trace",
        written_by="shared/plugins/permission/plugin.py::check_permission",
        fields=(
            AuditField("tool", "the tool asked for"),
            AuditField("call_id", "correlates to the tool call"),
            AuditField("agent", "which session asked"),
            AuditField("session", "the daemon session id"),
            AuditField("allowed", "the verdict"),
            AuditField("method", "how it was reached"),
            AuditField("asked",
                       "whether a HUMAN was consulted -- the fact `method` "
                       "could not supply, since allow_all and the two "
                       "suspensions are each produced both by a "
                       "pre-approval and by somebody typing a key (#968)"),
            AuditField("policy", "runtime or session-scoped (#957)"),
            AuditField("user_id", "who answered", guaranteed=False),
            AuditField("approver", "an external approver's claim",
                       guaranteed=False),
            AuditField("reason", "free text, LAST, so the line parses"),
        ),
        note=(
            "Machine-readable by construction: scalars first, free-text "
            "`reason` last. `parse_decision_trace` reads it back. The "
            "ledger's row needs a ledger and the event is opt-in; this "
            "line is what a deployment has without configuring anything"),
    ),
    AuditEvent(
        kind="BUDGET CEILING / BUDGET RUNG",
        article="Art. 12(1), 14(3)(a) -- a built-in constraint acting",
        store="session_trace",
        written_by="shared/jaato_session.py::_apply_budget_rungs",
        fields=(
            AuditField("dim", "which dimension crossed"),
            AuditField("used", "what had been spent"),
            AuditField("limit", "the declared ceiling"),
            AuditField("at", "the rung's percentage", guaranteed=False),
            AuditField("action", "what the rung did", guaranteed=False),
        ),
        note=(
            "A ceiling is traced whether or not a rung fires: a ladder "
            "that never logs is indistinguishable from one that is not "
            "wired (#955)"),
    ),
    AuditEvent(
        kind="session record header",
        article="Art. 12(1) -- the session's own lifecycle",
        store="session_record",
        written_by="server/session_manager.py::_save_session",
        fields=(
            AuditField("session_id", "the id"),
            AuditField("version", "record version"),
            AuditField("created_by",
                       "the authenticated creator (#859, record 2.9+)",
                       guaranteed=False),
            AuditField("runner_identity",
                       "which runner ran it (#812, record 2.10+); `stale` "
                       "after a reload, because the pid named is from a "
                       "previous process lifetime",
                       guaranteed=False),
            AuditField("profile_snapshot",
                       "the resolved recipe the session froze (#787)"),
            AuditField("sandbox_mode",
                       "apparmor / apparmor-complain / soft (#1014) -- "
                       "whether a boundary was ENFORCED, not merely "
                       "provisioned"),
        ),
    ),
)


def events_for_store(key: str) -> Tuple[AuditEvent, ...]:
    """Every event that lands in the store registered under ``key``."""
    return tuple(e for e in AUDIT_SCHEMA if e.store == key)


def guaranteed_fields(kind: str) -> Tuple[str, ...]:
    """The field names an event of ``kind`` always carries.

    What the enforcement guard checks the writer still writes.  A field
    that is conditionally present is deliberately absent from this list:
    promising it would make the guard demand a writer unconditionally
    write something the record is right to omit.
    """
    for event in AUDIT_SCHEMA:
        if event.kind == kind:
            return tuple(f.name for f in event.fields if f.guaranteed)
    return ()


def event(kind: str) -> Optional[AuditEvent]:
    """The event registered under ``kind``, or ``None``."""
    for candidate in AUDIT_SCHEMA:
        if candidate.kind == kind:
            return candidate
    return None
