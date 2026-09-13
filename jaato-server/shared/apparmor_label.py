"""One definition of "is this confined, and in which mode" (#1014).

Why this module exists
======================

Five places in the tree asked "is this session confined?" and four of them
answered by looking at the profile NAME and throwing the mode away::

    actual.startswith(f"{profile_name} ")   # "X (enforce)" and "X (complain)"

``JAATO_APPARMOR_COMPLAIN=1`` stamps ``flags=(complain)`` on the whole
profile chain, and a complain-mode profile enforces **nothing** — the
kernel logs the denial and allows the syscall.  So under that env var the
runner bootstrap said "confined", the idempotency check said "already
confined", the session record persisted ``sandbox_mode: "apparmor"``, and
``interactive_shell``'s ``require_confinement: true`` — the strictest
fail-closed knob in the repo — was satisfied while the kernel blocked
nothing.  Only the notebook kernel
(``shared.plugins.notebook.kernel_sandbox.apparmor_enforced_profile``) got
it right, which is why the sole visible symptom of a boundary-less session
was a notebook cell failing to ``import numpy``.

This module is that one right answer, lifted out so every caller shares it
rather than carrying a sixth opinion.

Two questions, deliberately kept apart
======================================

The label ``"jaato-ws-abc (enforce)"`` carries two independent facts, and
conflating them is what produced both #1023 and #1014:

* **which profile** a task is in — :func:`profile_name_ignoring_mode`.
  Mode-TOLERANT on purpose.  Comparing threads of one process against each
  other (#1023) is a question about identity: a thread in complain mode is
  not *divergent* from its complain-mode siblings, and a check that read it
  as divergence would make ``JAATO_APPARMOR_COMPLAIN`` unusable as the
  diagnostic it is.  The name says "ignoring mode" so it can never be
  mistaken for an enforcement assertion.
* **whether the kernel is enforcing** — :attr:`AppArmorLabel.enforced` /
  :func:`label_is_enforced`.  This is the only predicate that may stand
  behind a claim that a boundary exists.

Layering
========

Pure stdlib, zero jaato imports — the same reason
:mod:`shared.runtime_limits` is a free-standing module.  Both ``server/``
(including ``server.runner.bootstrap``, which must stay importable before
plugin discovery) and ``shared/plugins/`` need this answer, and
``shared/`` cannot import ``server/``.  ``shared/__init__.py`` is lazy, so
importing this module executes nothing but this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


#: The kernel's own spelling for "no profile attached".
UNCONFINED = "unconfined"

#: Enforcement modes AppArmor annotates a label with.  ``enforce`` is the
#: only one that blocks a syscall; ``complain`` logs and allows.
MODE_ENFORCE = "enforce"
MODE_COMPLAIN = "complain"

#: Process-level label — really the MAIN THREAD's, because ``/proc/self``
#: resolves to ``/proc/<pid>/``.  See :func:`read_thread_label` for the
#: calling thread's own, which is what a ``fork()`` inherits (#1023).
PROC_SELF_ATTR_CURRENT = "/proc/self/attr/current"

#: The CALLING thread's label.  ``/proc/thread-self`` resolves to
#: ``/proc/<pid>/task/<tid>/``.
PROC_THREAD_SELF_ATTR_CURRENT = "/proc/thread-self/attr/current"


@dataclass(frozen=True)
class AppArmorLabel:
    """A parsed ``attr/current`` value.

    Attributes:
        raw: The label exactly as read, after NUL/whitespace stripping.
            Empty when the file could not be read.
        profile: The profile name with the mode annotation removed, or
            ``""`` when the task is unconfined / the read failed.
        mode: ``"enforce"``, ``"complain"``, some other annotation the
            kernel produced, or ``None`` when the label carried no
            ``(mode)`` parenthetical at all.

    ``mode is None`` is treated as NOT enforcing, deliberately and
    conservatively: a bare name is not evidence the kernel is blocking
    anything, and this type's whole job is to stop absence of evidence
    reading as a boundary.
    """

    raw: str
    profile: str
    mode: Optional[str]

    @property
    def confined(self) -> bool:
        """Is a profile attached at all, whatever mode it is in?"""
        return bool(self.profile)

    @property
    def enforced(self) -> bool:
        """Is a profile attached AND is the kernel blocking on it?

        The only predicate in this module that may back a claim that a
        kernel boundary exists.
        """
        return self.confined and self.mode == MODE_ENFORCE

    @property
    def complaining(self) -> bool:
        """Is a profile attached in log-only (``complain``) mode?"""
        return self.confined and self.mode == MODE_COMPLAIN

    def describe(self) -> str:
        """A phrase for an operator log that never says "confined" alone.

        The #1014 incident turned on a log line reading ``runner confined
        to AppArmor profile X (kernel reports: X (complain))`` — the truth
        was in the parenthetical of a line whose leading words said the
        opposite, and nobody greps a parenthetical.
        """
        if not self.confined:
            return f"unconfined (kernel reports: {self.raw or '<unreadable>'})"
        if self.enforced:
            return f"{self.profile} (enforce)"
        if self.mode is None:
            return (
                f"{self.profile} (no enforcement mode reported — "
                f"NOT treated as a kernel boundary)"
            )
        return (
            f"{self.profile} ({self.mode} — NOT a kernel boundary; "
            f"the kernel logs denials and allows the syscall)"
        )


def _clean(raw: Optional[str]) -> str:
    """Strip what procfs appends to ``attr/current``.

    The kernel NUL-terminates this value, and the terminator does not
    always arrive alongside the newline — measured on a host whose active
    LSM reports a bare ``kernel\\x00``.  Stripping only ``\\n`` leaves the
    NUL inside the label, and every later comparison then fails against a
    profile name that looks identical when printed.
    """
    if not raw:
        return ""
    return raw.replace("\x00", "").strip()


def parse_label(raw: Optional[str]) -> AppArmorLabel:
    """Parse an ``attr/current`` value into its profile and its mode.

    Accepts every shape the kernel produces: ``"unconfined"``,
    ``"jaato-ws-a (enforce)"``, ``"jaato-ws-a//child (complain)"``, a bare
    ``"jaato-ws-a"`` with no annotation, and ``""`` / ``None`` for a read
    that failed.  Never raises.
    """
    cleaned = _clean(raw)
    if not cleaned or cleaned == UNCONFINED or cleaned.startswith(UNCONFINED + " "):
        return AppArmorLabel(raw=cleaned, profile="", mode=None)

    name, paren, rest = cleaned.partition(" (")
    name = name.strip()
    if not name:
        return AppArmorLabel(raw=cleaned, profile="", mode=None)
    if not paren:
        # A bare name with no ``(mode)``.  Not evidence of enforcement.
        return AppArmorLabel(raw=cleaned, profile=name, mode=None)

    mode = rest.rstrip(")").strip() or None
    return AppArmorLabel(raw=cleaned, profile=name, mode=mode)


def profile_name_ignoring_mode(raw: str) -> str:
    """Return the profile name, DISCARDING the enforcement mode.

    Mode-tolerant **on purpose**, and named so it cannot be mistaken for
    an enforcement assertion (#1014 ask 1).  The legitimate use is
    comparing tasks against each other — "is this thread in the same
    profile as its siblings" (#1023) — where a complain-mode label is not
    divergence.  ``"unconfined"`` is returned verbatim, because for that
    question "unconfined" IS the answer rather than a missing one.

    Never use this to decide whether a boundary exists; use
    :func:`label_is_enforced`.
    """
    cleaned = _clean(raw)
    return cleaned.split(" ", 1)[0].strip()


def label_is_enforced(raw: Optional[str]) -> bool:
    """Is *raw* a profile the kernel is actually blocking on?"""
    return parse_label(raw).enforced


def enforced_profile_name(raw: Optional[str]) -> Optional[str]:
    """Return the profile name iff it is enforced, else ``None``."""
    label = parse_label(raw)
    return label.profile if label.enforced else None


def read_label(path: str = PROC_SELF_ATTR_CURRENT) -> AppArmorLabel:
    """Read and parse one ``attr/current`` file.

    Raises:
        OSError: the file could not be read.  Propagated rather than
            swallowed because a caller that asked for a specific task's
            label needs to tell "unconfined" from "I could not look" —
            acting on absence of evidence is how a verifier takes a host
            down for a ``/proc`` it merely could not read.  Callers that
            genuinely want the lenient answer use :func:`try_read_label`.
    """
    with open(path, "r") as handle:
        return parse_label(handle.read())


def try_read_label(path: str = PROC_SELF_ATTR_CURRENT) -> AppArmorLabel:
    """:func:`read_label`, but an unreadable file yields an unconfined label.

    For the callers whose question is "may I claim a boundary here?", where
    the safe answer to "I could not look" is "no".
    """
    try:
        return read_label(path)
    except OSError:
        return AppArmorLabel(raw="", profile="", mode=None)


def read_thread_label() -> AppArmorLabel:
    """The CALLING thread's label, falling back to the process's.

    ``aa_change_profile`` is per-task, so a ``fork()`` inherits the cred of
    the thread that called it — which is the spawning worker, not the main
    thread ``/proc/self/attr/current`` reports (#1023).  Any caller asking
    "will the child I am about to spawn be confined?" wants this one.

    ``/proc/thread-self`` needs Linux 3.17+; on a kernel or a ``/proc``
    mount without it the process-level file is used instead, which is the
    pre-#1023 answer rather than a wrong one.
    """
    try:
        return read_label(PROC_THREAD_SELF_ATTR_CURRENT)
    except OSError:
        return try_read_label(PROC_SELF_ATTR_CURRENT)


# ---------------------------------------------------------------------
# The persisted vocabulary (#1014 ask 2)
# ---------------------------------------------------------------------

#: ``Session.sandbox_mode`` when a profile was provisioned AND the kernel
#: is enforcing it.
SANDBOX_MODE_APPARMOR = "apparmor"

#: ``Session.sandbox_mode`` when a profile was provisioned in complain
#: mode.  A distinct value rather than a flag, because the session record
#: is what an operator reads weeks later during a post-mortem and what an
#: auditor would read as evidence of enforcement: a record must not make a
#: positive claim about a boundary the kernel was not applying.
#:
#: An older reader comparing ``sandbox_mode == "apparmor"`` reads this as
#: "not confined", which is the true and safe direction — so this widens a
#: value vocabulary rather than adding a record field, and needs no record
#: version bump (the version string is written and never read back).
SANDBOX_MODE_APPARMOR_COMPLAIN = "apparmor-complain"

#: ``Session.sandbox_mode`` when AppArmor was requested and unavailable —
#: directory sandboxing only.
SANDBOX_MODE_SOFT = "soft"


def sandbox_mode_for_profile(*, complain: bool) -> str:
    """Pick the ``sandbox_mode`` value for a successfully loaded profile."""
    return SANDBOX_MODE_APPARMOR_COMPLAIN if complain else SANDBOX_MODE_APPARMOR


def sandbox_mode_is_apparmor(mode: Optional[str]) -> bool:
    """Did this session get an AppArmor profile, in ANY mode?

    The question every "should this session re-provision / re-confine on
    revive" site is really asking.  Distinct from
    :func:`sandbox_mode_is_enforced`, which is the question every "may I
    claim a boundary" site is asking, and which the tree had no way to ask
    at all before #1014.
    """
    return mode in (SANDBOX_MODE_APPARMOR, SANDBOX_MODE_APPARMOR_COMPLAIN)


def sandbox_mode_is_enforced(mode: Optional[str]) -> bool:
    """Did this session run behind a kernel boundary?"""
    return mode == SANDBOX_MODE_APPARMOR


# ---------------------------------------------------------------------
# The env knob that produces a complain-mode chain (#1014 ask 3)
# ---------------------------------------------------------------------

#: The documented diagnostic that puts the whole profile chain in complain
#: mode.  Named here, beside the vocabulary it produces, so the several
#: places that must announce it cannot disagree about its spelling.
COMPLAIN_ENV_VAR = "JAATO_APPARMOR_COMPLAIN"

_TRUTHY = ("1", "true", "yes", "on")


def complain_mode_requested(environ: Optional[dict] = None) -> bool:
    """Is :data:`COMPLAIN_ENV_VAR` asking for a log-only profile chain?

    Args:
        environ: Mapping to read instead of :data:`os.environ` (tests).
    """
    if environ is None:
        # env: generate AppArmor profiles in complain (log-only) mode;
        # confinement debugging aid.  Whole-daemon, so host-scoped: a
        # per-session value would be a lie about a kernel posture the
        # daemon applies to every profile it renders.
        raw = os.environ.get(COMPLAIN_ENV_VAR, "")
    else:
        raw = environ.get(COMPLAIN_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY
