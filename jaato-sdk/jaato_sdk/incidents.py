"""The incident register -- Regulation (EU) 2024/1689, Arts. 72, 73, 26(5).

Article 73 gives a provider **15 days** to report a serious incident from
the moment it becomes aware of it -- 10 for a death, 2 for a widespread
infringement or a critical-infrastructure disruption.  Article 72 asks
for post-market monitoring; 26(5) asks a deployer to inform the provider.
All three start from *becoming aware*, and the framework already KNOWS
when the events that could be one happen.

It recorded none of them AS SUCH.  Each was a log line in a different
format, in a different file, with no severity and no clock -- so
answering "what happened on this deployment in the last fifteen days"
meant grepping five stores for shapes nobody had written down.

**A query over the audit log, not a second store.**  An incident is one
more record in the application trace (``trace.session_log``), which is
the one artefact every deployment gets -- the ledger needs a ledger
configured and the event is opt-in.  ``jaato-doctor --incidents`` reads
the trace back through :func:`parse_incident_trace`.

**The tool does not classify.**  Whether an entry IS a "serious
incident" under Art. 3(49) is a human determination about consequences
the framework cannot see; what it can say is that something happened
that a person should look at, and when.  The register's output says so
in its header rather than labelling rows, and the clocks are rendered
against all three windows rather than the framework picking one.

Stdlib only, and it raises nothing: an incident record is written ABOUT
a failure, so it must not be able to add one.

**In the SDK, and ``shared.incidents`` imports it back** -- the
``jaato_sdk.session_env`` shape (#918).  The RAISER needs a session and
lives server-side; the RECORD, the vocabulary and the parser are what a
reader needs, and a reader is ``jaato-doctor``, which cannot import
``shared``.  One definition either way, so the writer and the register
cannot disagree about the line's grammar.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: The trace-line marker.  A prefix rather than a bare word so a grep for
#: it cannot match prose, and the same shape ``DECISION_TRACE_PREFIX``
#: uses -- the #968 precedent for a trace line that is a CONTRACT.
INCIDENT_TRACE_PREFIX = "INCIDENT: "

#: What the framework knows how to raise.  Each is a site that ALREADY
#: detected something and logged it in its own format; naming them here
#: is what turns five formats into one register.
#:
#: Ordered by how loudly the framework already complains about each, not
#: by severity -- severity is the deployment's call, and the register
#: says so.
KIND_SESSION_ERROR = "session_error"
KIND_BUDGET_EXHAUSTED = "budget_exhausted"
KIND_NUDGE_EXHAUSTED = "nudge_exhausted"
KIND_CONFINEMENT_REFUSED = "confinement_refused"
KIND_CIRCUIT_OPENED = "circuit_opened"

INCIDENT_KINDS: Tuple[str, ...] = (
    KIND_SESSION_ERROR,
    KIND_BUDGET_EXHAUSTED,
    KIND_NUDGE_EXHAUSTED,
    KIND_CONFINEMENT_REFUSED,
    KIND_CIRCUIT_OPENED,
)

#: What each kind means, in a reader's vocabulary.  Rendered by the
#: register and by ``explain audit``, so the register does not need a
#: reader who already knows the framework's internals.
KIND_DESCRIPTIONS: Dict[str, str] = {
    KIND_SESSION_ERROR:
        "a session ended on a terminal error the framework could not "
        "resolve",
    KIND_BUDGET_EXHAUSTED:
        "a session was stopped by its budget ceiling -- work was cut "
        "short, which for an agent acting on the world may leave a "
        "half-finished change",
    KIND_NUDGE_EXHAUSTED:
        "a session could not be made to complete: the model was "
        "re-prompted its whole budget of nudges and never signalled",
    KIND_CONFINEMENT_REFUSED:
        "a runner bootstrap was refused because its threads were not "
        "uniformly confined -- the guard working, and evidence that "
        "something ran outside the boundary the record claims (#1023)",
    KIND_CIRCUIT_OPENED:
        "the reliability plugin opened a circuit: one tool failed enough "
        "times in a row to be taken out of service",
}

#: Article 73's three reporting windows, in days, widest last.  Rendered
#: BESIDE each row rather than used to pick one: which window applies
#: depends on what the incident DID, which is the human determination
#: this tool does not make.
ARTICLE_73_WINDOWS: Tuple[Tuple[int, str], ...] = (
    (2, "widespread infringement / critical-infrastructure disruption"),
    (10, "death of a person"),
    (15, "any other serious incident"),
)

_SCALAR_RE = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class Incident:
    """One thing a person should look at.

    Attributes:
        kind: One of :data:`INCIDENT_KINDS`.
        at: Unix timestamp of when the framework became aware.
        session_id: The session it happened in, when there was one.
        provider / model / tier: The binding that was serving.
        cause: One line, free text, LAST on the trace line so it parses.
        site: What raised it -- ``file.py::function``.
    """

    kind: str
    at: float
    cause: str = ""
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tier: Optional[str] = None
    site: Optional[str] = None

    def age_days(self, now: Optional[float] = None) -> float:
        """Days since the framework became aware."""
        stamp = time.time() if now is None else now
        return max(0.0, (stamp - self.at) / 86400.0)

    def to_trace(self) -> str:
        """The trace line, machine-readable by construction.

        Scalars first, free-text ``cause`` LAST, exactly as the
        ``DECISION`` line is built (#968) -- so the line parses with a
        split rather than a regex over the whole of it, and a cause
        containing spaces or ``=`` cannot corrupt a field before it.
        """
        # ``repr`` rather than a fixed precision: the line IS the record,
        # so it must read back as the same object.  A truncated timestamp
        # round-trips to a DIFFERENT incident, which is a small wart on a
        # file whose whole job is to be evidence.
        parts = [f"{INCIDENT_TRACE_PREFIX}kind={self.kind}",
                 f"at={self.at!r}"]
        for name in ("session_id", "provider", "model", "tier", "site"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={value}")
        parts.append(f"cause={self.cause!r}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """The ledger/event shape, unset keys omitted.

        Omitted rather than ``null``: absent is "not measured here", the
        rule the whole audit record follows.
        """
        out: Dict[str, Any] = {"kind": self.kind, "at": self.at,
                               "cause": self.cause}
        for name in ("session_id", "provider", "model", "tier", "site"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out


def parse_incident_trace(msg: str) -> Optional[Incident]:
    """Read one ``INCIDENT:`` trace line back, or ``None``.

    The sibling of ``parse_decision_trace``, and provided for the same
    reason: an operator answering "what happened here" should not have to
    re-derive the line's grammar from a regex of their own.
    """
    if INCIDENT_TRACE_PREFIX not in msg:
        return None
    body = msg.split(INCIDENT_TRACE_PREFIX, 1)[1]
    head, _, cause = body.partition(" cause=")
    fields = dict(_SCALAR_RE.findall(head))
    kind = fields.get("kind")
    if not kind:
        return None
    try:
        at = float(fields.get("at", "0") or 0)
    except ValueError:
        at = 0.0
    text = cause.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return Incident(
        kind=kind, at=at, cause=text,
        session_id=fields.get("session_id"),
        provider=fields.get("provider"),
        model=fields.get("model"),
        tier=fields.get("tier"),
        site=fields.get("site"),
    )


def read_register(
    lines: Any,
    since_days: Optional[float] = None,
    now: Optional[float] = None,
) -> List[Incident]:
    """Every incident in a trace file, newest first.

    Args:
        lines: The trace file's lines, in any order.
        since_days: Keep only incidents this recent.  ``None`` = all.
        now: The instant to measure against (#996: a parameter, so a
            test states the moment it means).

    A line that does not parse is SKIPPED, not guessed at: a register
    that invented a row would be worse than one that is short, and the
    file it reads is written by a framework that may be older or newer
    than the reader.
    """
    stamp = time.time() if now is None else now
    found: List[Incident] = []
    for raw in lines:
        incident = parse_incident_trace(raw)
        if incident is None:
            continue
        if since_days is not None and incident.age_days(stamp) > since_days:
            continue
        found.append(incident)
    return sorted(found, key=lambda i: i.at, reverse=True)


def clocks(incident: Incident, now: Optional[float] = None) -> List[str]:
    """Article 73's three windows, rendered against one incident.

    All three, rather than the framework picking one: which applies
    depends on what the incident DID, and that is the determination
    Art. 3(49) leaves to a person.
    """
    age = incident.age_days(now)
    out = []
    for days, what in ARTICLE_73_WINDOWS:
        left = days - age
        state = f"{left:.1f}d left" if left > 0 else f"PAST by {-left:.1f}d"
        out.append(f"{days}d ({what}): {state}")
    return out
