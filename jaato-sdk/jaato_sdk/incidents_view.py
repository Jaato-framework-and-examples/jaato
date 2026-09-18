"""Rendering the incident register -- ``jaato-doctor --incidents`` (#1122).

Separate from :mod:`jaato_sdk.incidents` (the record and its grammar) and
from :mod:`jaato_sdk.doctor` (the CLI): this is the one place that turns
incidents into the lines a person reads, which keeps the doctor's
``check_incidents`` at three statements and keeps the record's module
free of presentation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from .incidents import ARTICLE_73_WINDOWS, KIND_DESCRIPTIONS, read_register

#: What the register says about itself, before any row.
#:
#: The framework can say that something happened and when; it cannot say
#: whether that something is a "serious incident" under Art. 3(49),
#: because that is a determination about CONSEQUENCES -- harm to a
#: person, disruption of critical infrastructure -- which no log line
#: carries.  Saying so in the header is what stops a reader taking an
#: empty register for a clean bill or a full one for a reportable event.
HEADER = (
    "what the framework noticed, with the Art. 73 clocks. It does NOT "
    "classify: whether any of these is a 'serious incident' under "
    "Art. 3(49) is a determination about consequences, and yours to make"
)


def render(
    paths: List[str],
    *,
    since_days: Optional[float] = None,
    check: Callable[..., Any],
    pass_: Any,
    warn: Any,
    now: Optional[float] = None,
) -> List[Any]:
    """One ``Check`` per file, plus one per incident.

    Args:
        paths: Application-trace files to read.
        since_days: Keep only entries this recent.
        check / pass_ / warn: The doctor's ``Check`` and its two
            statuses, injected so this module does not import the CLI it
            is rendering for.
        now: The instant (#996).

    Never returns a FAIL: an incident is news for a person, and
    ``jaato-doctor`` is documented as usable as a CI gate.
    """
    stamp = time.time() if now is None else now
    out: List[Any] = []
    total = 0

    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            out.append(check(f"incidents {path.name}", warn,
                             f"{path}: no such trace file — NOT read, which "
                             f"is not the same as 'no incidents'"))
            continue
        try:
            lines = path.read_text(encoding="utf-8",
                                   errors="replace").splitlines()
        except OSError as exc:
            out.append(check(f"incidents {path.name}", warn,
                             f"{path}: cannot read ({exc}) — NOT read, which "
                             f"is not the same as 'no incidents'"))
            continue

        found = read_register(lines, since_days=since_days, now=stamp)
        total += len(found)
        window = (f" in the last {since_days:g}d" if since_days else "")
        if not found:
            out.append(check(f"incidents {path.name}", pass_,
                             f"none recorded{window}.  {HEADER}"))
            continue
        out.append(check(f"incidents {path.name}", warn,
                         f"{len(found)} recorded{window}.  {HEADER}"))
        for incident in found:
            out.append(check(f"  {incident.kind}", warn, _row(incident, stamp)))
    return out


def _row(incident: Any, now: float) -> str:
    """One incident, with every Art. 73 window rendered beside it.

    All three, rather than the framework choosing: which applies depends
    on what the incident DID, and that is the determination above.
    """
    age = incident.age_days(now)
    where = incident.session_id or "no session"
    binding = "/".join(p for p in (incident.provider, incident.model) if p)
    parts = [f"{age:.1f}d ago", where]
    if binding:
        parts.append(binding)
    head = "  ".join(parts)
    clocks = "; ".join(
        f"{days}d: {left:+.1f}d"
        for days, _what in ARTICLE_73_WINDOWS
        for left in (days - age,)
    )
    what = KIND_DESCRIPTIONS.get(incident.kind, "")
    return (f"{head}\n      {incident.cause}\n"
            f"      {what}\n"
            f"      Art. 73 windows remaining: {clocks}")
