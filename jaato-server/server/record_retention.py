"""How long an audit record outlives the session it belongs to (#1119).

Regulation (EU) 2024/1689 Art. 19(1) asks a provider of a high-risk AI
system to keep the logs its system automatically generates for at least
six months; 26(6) asks the same of the deployer.  Neither was
expressible: ``session.delete`` and ``workspace.delete`` removed
everything, and the #812 lifetime sweep bounded how long a session may
RUN with no notion of how long its record must be KEPT.

The block is ``record_keeping:`` (``shared/plugins/subagent/config.py``).
This module is the clock that reads it.

**Two clocks, not one.**  The audit record answers "what did this system
do"; the conversation is personal data somebody may ask to have erased.
Keeping the first while dropping the second is what satisfies Art. 19(1)
and GDPR erasure at once, and one number cannot say it:

============================  =====================================
``retention_days``            the audit stores (ledger, traces)
``conversation_retention_days``  the session record -- the history
============================  =====================================

**A pure module, like ``session_lifetime``.**  Everything here decides;
nothing here deletes.  The caller does the I/O, which is what lets the
whole policy be exercised without a filesystem and what keeps a mistake
in the arithmetic from being a mistake that removed a file.

**Why it is a separate module from ``session_lifetime``.**  #1119 names
that sweep as the place, and the sweep is indeed what calls this -- but
the two answer different questions about different objects.
``session_lifetime`` bounds how long a LOADED session may run and judges
live sessions; retention bounds how long the RECORD of an ended session
is kept and judges files.  Folding the second into the first would have
put a filesystem walk inside a per-tick pass over the loaded-session
dict, on a module whose whole value is that it is pure arithmetic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Seconds in a day.  Named because the block is written in DAYS and every
#: clock here is in seconds, and a bare 86400 in an arithmetic expression
#: about record retention is the kind of thing that gets "fixed" wrongly.
SECONDS_PER_DAY = 86400.0

#: How often the retention pass runs, in seconds.  Far coarser than the
#: lifetime sweep's tick: the shortest retention anybody writes is a day,
#: so an hour of overshoot is noise, and the pass STATS FILES where the
#: lifetime sweep walks a dict.
DEFAULT_RETENTION_SWEEP_SECONDS = 3600.0


@dataclass(frozen=True)
class RetentionVerdict:
    """What the clock says about one path.

    Attributes:
        path: The file or directory judged.
        expired: Whether its minimum has elapsed.
        kept_days: The minimum the policy declared, in days.
        age_days: How old it is, in days.
        reason: One line, for the log and for an operator asking why.
    """

    path: str
    expired: bool
    kept_days: Optional[int]
    age_days: Optional[float]
    reason: str


def _age_days(path: Path, now: float) -> Optional[float]:
    """How many days since ``path`` was last modified, or ``None``.

    ``None`` means the age could not be read -- a path that vanished
    between the walk and the stat, a permission error.  Every caller
    treats that as NOT EXPIRED, which is the safe direction: the cost of
    being wrong is a record kept too long, and the alternative cost is a
    record deleted because its age could not be established.
    """
    try:
        return max(0.0, (now - path.stat().st_mtime) / SECONDS_PER_DAY)
    except OSError:
        return None


def judge(
    path: Any,
    retention_days: Optional[int],
    now: Optional[float] = None,
) -> RetentionVerdict:
    """Whether ``path`` has outlived its declared minimum.

    Args:
        path: The file or directory to judge.
        retention_days: The declared minimum.  ``None`` means no policy
            governs this path; ``0`` means "keep until something deletes
            it" -- the 0-disables spelling ``max_session_seconds`` uses.
        now: The instant to judge against.  A parameter rather than a
            call to ``time.time()`` so a test states the moment it means
            instead of betting on a clock (#996).

    Returns:
        A :class:`RetentionVerdict`.  ``expired`` is ``True`` only on
        POSITIVE evidence: a policy that declares a bounded minimum, and
        an age that could be read and exceeds it.  Absence of evidence
        never expires a record.
    """
    target = Path(path)
    stamp = time.time() if now is None else now

    if retention_days is None:
        return RetentionVerdict(
            str(target), False, None, None,
            "no record_keeping policy governs this path")
    if retention_days == 0:
        return RetentionVerdict(
            str(target), False, 0, None,
            "retention_days: 0 -- kept until something deletes it")

    age = _age_days(target, stamp)
    if age is None:
        return RetentionVerdict(
            str(target), False, retention_days, None,
            "age could not be read; kept, because absence of evidence is "
            "not an expiry")
    if age < retention_days:
        return RetentionVerdict(
            str(target), False, retention_days, age,
            f"{age:.1f}d old, minimum is {retention_days}d")
    return RetentionVerdict(
        str(target), True, retention_days, age,
        f"{age:.1f}d old, past the {retention_days}d minimum")


def expired_paths(
    paths: Iterable[Any],
    retention_days: Optional[int],
    now: Optional[float] = None,
) -> Tuple[List[RetentionVerdict], List[RetentionVerdict]]:
    """Split ``paths`` into (expired, kept) by :func:`judge`.

    Both halves are returned because both are reportable: an operator
    asking why a record is still there is asking about the second list,
    and a pass that only names what it removed cannot answer them.
    """
    expired: List[RetentionVerdict] = []
    kept: List[RetentionVerdict] = []
    for path in paths:
        verdict = judge(path, retention_days, now=now)
        (expired if verdict.expired else kept).append(verdict)
    return expired, kept


def describe_policy(keeping: Any) -> str:
    """One line naming what a ``record_keeping:`` block bounds.

    Logged when the retention pass arms, the rule #735 established: a cap
    that silently does not apply is worse than no cap, so the effective
    values are said out loud rather than inferred from the file.
    """
    if keeping is None or not getattr(keeping, "declared", False):
        return ("record_keeping: undeclared -- nothing is retained beyond "
                "what a delete leaves behind")
    retention = getattr(keeping, "retention_days", None)
    conv = getattr(keeping, "conversation_retention_days", None)

    def _days(value: Optional[int], what: str) -> str:
        if value is None:
            return f"{what}=unset"
        if value == 0:
            return f"{what}=0 (kept until deleted)"
        return f"{what}={value}d"

    return ("record_keeping armed: "
            + _days(retention, "audit")
            + " " + _days(conv, "conversation")
            + f" integrity={getattr(keeping, 'integrity', 'none')}")


def workspace_retention_hold(
    workspace: Any,
    now: Optional[float] = None,
) -> Optional[str]:
    """Why this workspace may not be deleted yet, or ``None``.

    Article 19(1) asks a provider to keep a high-risk system's logs for a
    minimum period.  ``workspace.delete`` removes the whole tree,
    including every audit store under it, so a workspace whose profiles
    declared a retention has records that a delete would destroy before
    that minimum elapsed.

    **It refuses rather than preserving or overriding**, and the choice
    is worth stating.  Preserving would leave orphan files in a directory
    an operator asked to be gone -- a surprise of its own, discovered
    later.  Overriding with a WARNING makes the policy something any
    delete silently defeats, which is not a policy.  A refusal is
    visible, is recoverable (wait, or drop the block), and cannot
    silently destroy a record somebody declared had to be kept.

    Returns:
        A message naming the files held, the minimum, and when the
        earliest of them expires -- or ``None`` when nothing holds.

    Never raises.  A workspace whose profiles cannot be resolved (a
    half-written file, an unreadable directory) is NOT held: the
    alternative is a workspace nobody can delete because its profiles no
    longer parse, and a retention that cannot be established is not
    evidence of one.
    """
    try:
        keeping, paths = _declared_retention(workspace)
    except Exception:  # noqa: BLE001 -- see the docstring
        logger.debug("retention hold: could not resolve %s", workspace,
                     exc_info=True)
        return None
    if keeping is None or not paths:
        return None
    retention = getattr(keeping, "retention_days", None)
    if not retention:          # None, or 0 = keep until something deletes it
        return None

    _expired, kept = expired_paths(paths, retention, now=now)
    if not kept:
        return None
    youngest = min((v.age_days for v in kept if v.age_days is not None),
                   default=0.0)
    lifts = retention - youngest
    return (
        f"{len(kept)} audit file(s) here are under a {retention}-day "
        f"record_keeping.retention_days minimum (EU AI Act Art. 19(1)); "
        f"the last of them expires in {lifts:.1f} day(s). Use "
        f"`session.delete` instead -- that removes the conversations and "
        f"leaves the record -- or drop record_keeping.retention_days from "
        f"the profiles here if the retention no longer applies."
    )


def _declared_retention(workspace: Any) -> Tuple[Any, List[Path]]:
    """The strictest ``record_keeping:`` in a workspace, and the files it governs.

    Strictest rather than first: a workspace may hold several profiles,
    and the one with the longest minimum is the one a delete would
    violate.  Only paths that EXIST are returned, so a declared-but-never
    written trace holds nothing.
    """
    from shared.plugins.subagent.config import discover_profiles

    root = Path(workspace)
    result = discover_profiles(
        profiles_dir=".jaato/profiles", base_path=str(root),
        config_root=str(root / ".jaato"),
    )
    strictest = None
    paths: List[Path] = []
    for profile in (result.profiles or {}).values():
        keeping = getattr(profile, "record_keeping", None)
        if keeping is None or not getattr(keeping, "declared", False):
            continue
        days = getattr(keeping, "retention_days", None)
        if days and (strictest is None
                     or days > getattr(strictest, "retention_days", 0)):
            strictest = keeping
        trace = getattr(profile, "trace", None)
        for value in (getattr(trace, "ledger", None),
                      getattr(trace, "session_log", None),
                      getattr(trace, "provider_log", None)):
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / value
            if candidate.exists():
                paths.append(candidate)
    return strictest, paths
