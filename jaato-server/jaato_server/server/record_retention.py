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


@dataclass(frozen=True)
class ProfileRetention:
    """One profile's declared policy and the paths it governs.

    A LIST of these rather than one "strictest" policy plus a pooled path
    set, because pooling violates the very declaration it is reading.
    Profile A keeping its ledger 30 days beside profile B keeping its own
    "until deleted" (``retention_days: 0``) yielded A's clock over BOTH
    sets of files, so the sweep unlinked B's record on day 31 -- against
    an explicit instruction not to.  A policy that silently governs
    somebody else's files is not a policy.

    Attributes:
        profile: Whose block this is, for the log line.
        keeping: The ``RecordKeepingConfig``.
        audit_paths: The files ``retention_days`` governs, expanded --
            see :func:`_expand_trace_paths`.
    """

    profile: str
    keeping: Any
    audit_paths: List[Path]


def _expand_trace_paths(root: Path, value: str) -> List[Path]:
    """Every file one declared trace path actually names.

    A trace path is not one file.  The PROVIDER channel splits per agent
    -- ``provider.jsonl`` becomes ``provider_subagent_1.jsonl`` with no
    placeholder at all, and ``provider{agent_suffix}.jsonl`` with one --
    so judging the literal string either found nothing or found one file
    of many, and the siblings accumulated forever while the pass reported
    nothing kept and nothing removed.  That is the one-way ratchet this
    module exists to prevent, arriving through the back door.

    Both forms are globbed:

    * a path naming ``{agent}`` / ``{agent_suffix}`` -> the token becomes
      ``*``, because at retention time there is no current agent and the
      question is *which files did this path ever produce*;
    * a path naming none -> the literal, plus the implicit
      ``<stem>_*<suffix>`` the provider channel appends.

    The glob is scoped to the directory the profile named, so the widest
    thing it can reach is a file whose name starts with the declared
    stem.  Stated rather than hidden: that is the price of the implicit
    suffix being a naming convention rather than a recorded fact.
    """
    from jaato_sdk.trace import TRACE_PATH_PLACEHOLDERS

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / value
    text = str(candidate)

    if any(token in text for token in TRACE_PATH_PLACEHOLDERS):
        for token in TRACE_PATH_PLACEHOLDERS:
            text = text.replace(token, "*")
        pattern = Path(text)
        try:
            return sorted(p for p in pattern.parent.glob(pattern.name)
                          if p.is_file())
        except (OSError, ValueError):       # an unglobbable pattern
            return []

    found = [candidate] if candidate.is_file() else []
    try:
        siblings = candidate.parent.glob(f"{candidate.stem}_*{candidate.suffix}")
        found.extend(p for p in siblings if p.is_file())
    except (OSError, ValueError):
        pass
    return sorted(set(found))


def declared_retentions(workspace: Any) -> List[ProfileRetention]:
    """Every profile in a workspace that declares ``record_keeping:``.

    Per profile, under ITS OWN policy.  See :class:`ProfileRetention` for
    why there is no "strictest" shortcut.  A profile declaring the block
    with no ``retention_days`` -- integrity only, or an explicit ``0`` --
    is still returned, so a caller can see that its files are governed
    and governed by *nothing expiring them*.
    """
    from jaato_server.shared.plugins.subagent.config import discover_profiles

    root = Path(workspace)
    result = discover_profiles(
        profiles_dir=".jaato/profiles", base_path=str(root),
        config_root=str(root / ".jaato"),
    )
    out: List[ProfileRetention] = []
    for name, profile in sorted((result.profiles or {}).items()):
        keeping = getattr(profile, "record_keeping", None)
        if keeping is None or not getattr(keeping, "declared", False):
            continue
        trace = getattr(profile, "trace", None)
        paths: List[Path] = []
        for value in (getattr(trace, "ledger", None),
                      getattr(trace, "session_log", None),
                      getattr(trace, "provider_log", None)):
            if value:
                paths.extend(_expand_trace_paths(root, value))
        out.append(ProfileRetention(profile=name, keeping=keeping,
                                    audit_paths=sorted(set(paths))))
    return out


def conversation_minimum(workspace: Any) -> Optional[int]:
    """The longest ``conversation_retention_days`` declared in a workspace.

    STRICTEST here, and per-profile above, because the two clocks govern
    different objects.  ``retention_days`` governs files a profile NAMES,
    so each profile's declaration reaches only its own.  A session record
    is not named by any profile -- it is one directory under
    ``<workspace>/.jaato/sessions/`` per session, whichever profile ran
    it -- so the whole workspace has one conversation clock, and the only
    safe reading of several is the longest.

    ``None`` when nothing declares one; ``0`` is a declaration and means
    *kept until something deletes it*, so it never expires anything.
    """
    longest: Optional[int] = None
    for entry in declared_retentions(workspace):
        days = getattr(entry.keeping, "conversation_retention_days", None)
        if days is None:
            continue
        longest = days if longest is None else max(longest, days)
    return longest


def expired_session_records(
    sessions_dir: Any,
    conversation_days: Optional[int],
    now: Optional[float] = None,
    keep_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[RetentionVerdict], List[RetentionVerdict]]:
    """Split a workspace's session records into (expired, kept).

    ``conversation_retention_days`` was parsed, validated, inherited,
    rendered by ``explain audit`` and described by ``jaato_sdk.audit`` as
    the field that governs when a session record is deleted -- and read
    by one log line.  A profile declaring it to satisfy GDPR storage
    limitation kept every conversation forever while the tools said
    otherwise, which is the #735 shape: a key that does everything except
    the thing it is for.

    The record is a DIRECTORY per session, judged by its own mtime, so a
    session written to recently is young however old its directory node
    is.  ``keep_ids`` are the sessions currently LOADED: removing one
    under a running daemon is a live failure with a delayed cause.
    """
    expired: List[RetentionVerdict] = []
    kept: List[RetentionVerdict] = []
    root = Path(sessions_dir)
    if not conversation_days or not root.is_dir():
        return expired, kept
    protected = set(keep_ids or ())
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in protected:
            continue
        # Judged on its newest FILE, reported as the DIRECTORY: the age
        # belongs to the content and the removal to the record.
        inner = judge(_newest_mtime(entry), conversation_days, now=now)
        verdict = RetentionVerdict(
            path=str(entry), expired=inner.expired,
            kept_days=inner.kept_days, age_days=inner.age_days,
            reason=inner.reason.replace(str(_newest_mtime(entry)), str(entry)))
        (expired if verdict.expired else kept).append(verdict)
    return expired, kept


def _newest_mtime(directory: Path) -> Path:
    """The most recently touched file in ``directory``, else the directory.

    A record's age is the age of its NEWEST part: a conversation appended
    to yesterday is a day old whatever its directory node says, and
    judging the container would expire a live conversation.
    """
    newest = directory
    best = -1.0
    try:
        for child in directory.rglob("*"):
            if not child.is_file():
                continue
            stamp = child.stat().st_mtime
            if stamp > best:
                best, newest = stamp, child
    except OSError:
        return directory
    return newest


def _declared_retention(workspace: Any) -> Tuple[Any, List[Path]]:
    """The strictest ``record_keeping:`` in a workspace, and its own files.

    Retained for :func:`workspace_retention_hold`, which asks a
    whole-workspace question -- *may this tree be deleted* -- where the
    strictest declaration is the right answer and pooling is not a
    problem, because the verdict is one refusal rather than a set of
    unlinks.  The SWEEP uses :func:`declared_retentions`, which keeps
    each profile's files under each profile's own clock.
    """
    strictest = None
    paths: List[Path] = []
    for entry in declared_retentions(workspace):
        days = getattr(entry.keeping, "retention_days", None)
        if days and (strictest is None
                     or days > getattr(strictest, "retention_days", 0)):
            strictest = entry.keeping
        if days:
            paths.extend(entry.audit_paths)
    return strictest, sorted(set(paths))
