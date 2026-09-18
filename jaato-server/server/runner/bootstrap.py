"""Runner self-confinement bootstrap (§4.6 steps 1-3).

The runner subprocess starts unconfined (inheriting the daemon's
posture per §4.6 daemon-apparmor-state constraint), then transitions
to its session's per-session AppArmor profile via libapparmor's
``aa_change_profile`` symbol.

This module deliberately avoids importing any of the rest of jaato:

- It runs BEFORE plugin discovery (§4.6 step 4 — "now import plugin
  code").  Any cross-import would force code to load unconfined and
  break the "every plugin runs confined" guarantee.
- It uses ``ctypes`` rather than a Python AppArmor binding so the
  runner has zero non-stdlib runtime dependencies for its own
  bootstrap step.

API:

- :func:`confine_to_profile` — the load-bearing call.  Raises on any
  failure; the caller (``__main__``) translates raises into
  ``os._exit(2)`` per the spec's no-fallback-to-unconfined contract.
- :func:`read_current_profile` — small helper exposed mostly for
  tests.
- :func:`current_confinement` — the readback as a parsed
  :class:`shared.apparmor_label.AppArmorLabel`, so callers can ask
  *which mode* rather than only *which profile* (#1014).
- :func:`verify_thread_confinement` — the #1023 check: every thread of
  this process, not just the one ``/proc/self/attr/current`` reports.
  See the section header below it for why the process-level readback
  provably cannot see the condition it is there to catch.

Failure modes are distinguished so the daemon-side error message can
explain the cause:

- ``RuntimeError("apparmor not installed")`` — libapparmor missing.
- ``RuntimeError("aa_change_profile failed: ...")`` — kernel refused
  the transition (profile not loaded, transition rule missing, etc.).
- :class:`ConfinementMismatchError` — transition apparently succeeded
  but ``/proc/self/attr/current`` doesn't match the requested profile
  (silent-failure case from §6.5).
- :class:`ConfinementModeError` — the kernel attached the right profile
  and is NOT enforcing it (``JAATO_APPARMOR_COMPLAIN=1``).  Raised only
  under ``require_enforce=True``; see :func:`confine_to_profile`.

The one import
--------------

The "no cross-imports" rule above has exactly one exception:
:mod:`shared.apparmor_label`.  It is a pure-stdlib leaf (the same shape
as :mod:`shared.runtime_limits`) carrying the single definition of "is
this confined, and in which mode", and ``shared/__init__.py`` is lazy —
so the import executes that one file and loads no plugin code, which is
what the rule protects.  The alternative was a sixth private copy of the
mode predicate inside the module whose readback is precisely the one
#1014 found mode-blind.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
)

from shared.apparmor_label import (
    AppArmorLabel,
    COMPLAIN_ENV_VAR,
    parse_label,
    profile_name_ignoring_mode,
)


logger = logging.getLogger(__name__)


# Default proc-attr path; override-able for tests.  Reads the label of the
# task whose tid == pid — the MAIN THREAD — because ``/proc/self``
# resolves to ``/proc/<pid>/``.
#
# This path is the right one for :func:`confine_to_profile`, which runs ON
# the main thread and is asking whether ITS OWN transition took.  It is the
# wrong one for "is this PROCESS confined", and a comment here used to say
# ``aa_change_profile`` was a process-level transition.  It is not:
# AppArmor confines TASKS, so a sibling thread created before the
# transition keeps its own cred and this path cannot see it.  That is
# #1023 — see :func:`verify_thread_confinement`, which asks the whole
# process.
DEFAULT_PROC_ATTR_PATH = "/proc/self/attr/current"

# Default libapparmor SONAME.
DEFAULT_LIBAPPARMOR_SONAME = "libapparmor.so.1"


class ConfinementMismatchError(RuntimeError):
    """Raised when ``aa_change_profile`` reported success but
    ``/proc/self/attr/current`` does not match the requested profile.

    Mitigation for §6.5 (silent confine-failure): the bootstrap
    cross-checks the kernel's view, NOT just the libapparmor return
    code.  This catches the case where the kernel-side profile lacks
    a ``change_profile -> jaato-ws-*`` transition rule and the call
    silently no-ops.
    """

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            f"AppArmor confinement mismatch: requested {expected!r} but "
            f"/proc/self/attr/current reports {actual!r}"
        )
        self.expected = expected
        self.actual = actual


class ConfinementModeError(RuntimeError):
    """Raised when the right profile is attached and the kernel is NOT
    enforcing it.

    The #1014 state: ``JAATO_APPARMOR_COMPLAIN=1`` stamps
    ``flags=(complain)`` on the whole profile chain, the transition
    succeeds, ``/proc/self/attr/current`` reports
    ``jaato-ws-<sid> (complain)`` — and the kernel logs every denial and
    allows the syscall.  There is a profile and there is no boundary.

    Raised ONLY under ``confine_to_profile(..., require_enforce=True)``.
    The default stays permissive because complain mode is a documented
    diagnostic (harvest the missing-rule set in one cascade run, then
    ship targeted grants) and refusing to start under it would delete the
    diagnostic.  What the default does instead is say so, loudly: see the
    WARNING in :func:`confine_to_profile`.

    ``require_enforce=True`` is the hook point #1013's
    ``JAATO_APPARMOR_BEHAVIOR=require`` needs.  Built on the mode-blind
    readback this replaces, ``require`` would have been satisfied by a
    complain-mode profile — the exact posture it exists to refuse.
    """

    def __init__(self, expected: str, label: AppArmorLabel) -> None:
        super().__init__(
            f"AppArmor profile {expected!r} is attached but NOT enforced: "
            f"the kernel reports {label.raw!r}.  A {label.mode or 'mode-less'} "
            f"profile logs denials and allows the syscall, so this session "
            f"has no kernel boundary.  Unset {COMPLAIN_ENV_VAR} (or stop "
            f"requiring enforcement) to proceed."
        )
        self.expected = expected
        self.label = label
        self.actual = label.raw


def _load_libapparmor(soname: str = DEFAULT_LIBAPPARMOR_SONAME) -> ctypes.CDLL:
    """Open ``libapparmor.so.1`` via ctypes.

    Tries the requested SONAME first (matches the spec's "we don't add
    pyspnego-style native bindings — the one libapparmor symbol we
    need is aa_change_profile"), then falls back to
    ``ctypes.util.find_library("apparmor")`` for distros that ship the
    library under a different name.

    Raises:
        RuntimeError: when neither lookup resolves.  Message is
            distinct from "kernel refused" so the daemon can tell
            "apparmor not installed" apart from "transition denied".
    """
    try:
        return ctypes.CDLL(soname, use_errno=True)
    except OSError:
        pass

    fallback = ctypes.util.find_library("apparmor")
    if fallback:
        try:
            return ctypes.CDLL(fallback, use_errno=True)
        except OSError:
            pass

    raise RuntimeError(
        f"libapparmor not installed: tried {soname!r} and "
        f"ctypes.util.find_library('apparmor'); install the "
        f"libapparmor1 (or distro-equivalent) package"
    )


def read_current_profile(proc_attr_path: str = DEFAULT_PROC_ATTR_PATH) -> str:
    """Return the current process's AppArmor profile string.

    Format on a confined process is ``<profile> (enforce)`` or
    ``<profile> (complain)``; an unconfined process reports
    ``unconfined``.  The trailing newline AND the NUL terminator procfs
    writes are stripped — the NUL does not always arrive alongside the
    newline, and a label carrying one compares unequal to a name that
    looks identical when printed.

    Returns the raw string for the callers that log it.  Callers deciding
    whether a BOUNDARY exists want :func:`current_confinement` instead:
    this string answers "which profile", never "is the kernel enforcing".
    """
    with open(proc_attr_path, "r") as f:
        return parse_label(f.read()).raw


def current_confinement(
    proc_attr_path: str = DEFAULT_PROC_ATTR_PATH,
) -> AppArmorLabel:
    """Read ``attr/current`` and return it PARSED — profile and mode.

    The mode-aware sibling of :func:`read_current_profile`, and the one
    #1014 asks callers to reach for.  Raises :class:`OSError` like its
    sibling, so a caller that cannot read ``/proc`` still tells "could not
    look" apart from "unconfined".
    """
    with open(proc_attr_path, "r") as f:
        return parse_label(f.read())


def confine_to_profile(
    profile_name: str,
    *,
    libapparmor: Optional[ctypes.CDLL] = None,
    proc_attr_path: str = DEFAULT_PROC_ATTR_PATH,
    require_enforce: bool = False,
) -> AppArmorLabel:
    """Self-confine the current process to *profile_name*.

    Implements §4.6 bootstrap steps 2-3:

    1. Call ``aa_change_profile(profile_name)`` via ctypes.
    2. Read ``/proc/self/attr/current`` and verify the kernel agrees
       the process is now confined to *profile_name*.

    Per the spec's no-fallback-to-unconfined contract, ANY failure
    raises.  The caller (``__main__``) is responsible for translating
    the raise into an explicit ``os._exit(2)`` so the daemon detects
    runner failure via socket EOF + non-zero exit, NOT a half-confined
    runner that survives.

    Args:
        profile_name: The AppArmor profile to enter.  Must already be
            loaded in the kernel by the daemon-side
            ``AppArmorManager.provision_profile`` BEFORE this runner
            was forked.
        libapparmor: Pre-opened ctypes handle (test injection point).
            When ``None``, :func:`_load_libapparmor` is called.
        proc_attr_path: Override for ``/proc/self/attr/current``
            (test injection point).
        require_enforce: Refuse a profile the kernel is not enforcing.
            Default ``False`` — see :class:`ConfinementModeError` for why
            complain mode is announced rather than refused by default,
            and why this flag is the hook #1013's ``require`` behaviour
            needs.

    Returns:
        The parsed post-transition label, so the caller can record the
        MODE rather than re-deriving it from a string.  This is the fact
        ``sandbox_mode`` was missing (#1014 ask 2).

    Raises:
        RuntimeError: libapparmor lookup failure ("apparmor not
            installed") or ``aa_change_profile`` returned non-zero
            ("kernel refused the transition").
        ConfinementMismatchError: ``aa_change_profile`` returned 0
            but ``/proc/self/attr/current`` reports a different
            profile (silent-failure case — §6.5).
        ConfinementModeError: the profile is attached but not enforced,
            and *require_enforce* is set.
    """
    if not profile_name:
        raise RuntimeError("confine_to_profile: profile_name is empty")

    lib = libapparmor if libapparmor is not None else _load_libapparmor()

    # ``int aa_change_profile(const char *profile)``.  Returns 0 on
    # success, -1 on error with errno set.
    aa_change_profile = lib.aa_change_profile
    aa_change_profile.argtypes = [ctypes.c_char_p]
    aa_change_profile.restype = ctypes.c_int

    rc = aa_change_profile(profile_name.encode("utf-8"))
    if rc != 0:
        err = ctypes.get_errno()
        raise RuntimeError(
            f"aa_change_profile({profile_name!r}) failed: "
            f"errno={err} ({os.strerror(err)}); "
            f"the kernel refused the transition — verify the profile "
            f"is loaded and grants change_profile from the runner's "
            f"current profile (likely unconfined)"
        )

    # Cross-check the kernel's view.  Catches the silent-success case
    # where libapparmor returned 0 but the kernel didn't actually
    # transition (rare but observed when the parent profile lacks the
    # required change_profile -> rule).
    #
    # TWO conditions, and until #1014 only the first was checked:
    #   1. WHICH profile the kernel attached -- a mismatch is the silent
    #      no-op above;
    #   2. WHICH MODE it is applying -- a ``(complain)`` profile logs
    #      every denial and allows the syscall, so there is a profile and
    #      no boundary.  The old comment on this line read
    #      ``# e.g. "jaato-ws-... (enforce)"`` while the match accepted
    #      ``(complain)`` just as happily.
    label = current_confinement(proc_attr_path)
    if profile_name_ignoring_mode(label.raw) != profile_name:
        raise ConfinementMismatchError(expected=profile_name, actual=label.raw)

    if label.enforced:
        logger.info(
            "runner confined to AppArmor profile %s (kernel reports: %s)",
            profile_name, label.raw,
        )
        return label

    if require_enforce:
        raise ConfinementModeError(profile_name, label)

    # Ask 3: the weakened posture is ANNOUNCED, the way
    # ``scrub_secret_env: none`` and ``--ws-unsafe-no-auth`` are.  The
    # leading words must not say "confined": the whole #1014 incident
    # turned on an operator reading a line that did, with the truth in
    # its parenthetical.
    remedy = (
        f"{COMPLAIN_ENV_VAR} is the only supported route to this posture "
        f"— unset it to enforce."
        if label.complaining
        else "The kernel reported no enforcement mode for this profile, "
             "which is not evidence of a boundary and is not treated as one."
    )
    logger.warning(
        "runner attached AppArmor profile %s WITHOUT a kernel boundary: "
        "%s.  This session's tools are NOT confined.  %s",
        profile_name, label.describe(), remedy,
    )
    return label


# ---------------------------------------------------------------------
# Per-thread confinement verification (#1023)
# ---------------------------------------------------------------------
#
# ``aa_change_profile`` is PER-TASK.  ``/proc/self/attr/current`` is
# not: ``/proc/self`` resolves to ``/proc/<pid>/``, whose ``attr/current``
# reports the label of the task whose tid == pid -- the MAIN THREAD.  So
# the post-transition readback in :func:`confine_to_profile`, the
# idempotency check in ``runner/session.py``, ``sandbox_mode`` in the
# session record and an operator's ``cat /proc/<pid>/attr/current`` all
# read exactly one thread's label and report the process confined.
#
# #1023 is the state that makes the difference observable: a worker
# thread spawned while the process was still ``unconfined`` keeps that
# cred for its whole life, and ``ThreadPoolExecutor`` spawns workers
# lazily on first submit -- so any RPC dispatched to a lane before that
# slot's first ``session.bootstrap`` leaves a durably unconfined thread
# behind while every reader above says ``(enforce)``.  Confirmed live:
# 2 of 5 runners on a fully enforcing host.
#
# Reading the CALLING thread's own label needs
# ``/proc/thread-self/attr/current``; reading EVERY thread's needs
# ``/proc/self/task/<tid>/attr/current``, which is what this section
# does.  Nothing in the tree did either before #1023.
#
# Note on the existing caution at ``server/apparmor.py`` (_run_unconfined):
# reading ``/proc/self/task/<tid>/attr/current`` to answer "am I
# confined?" is called fragile there, and correctly so -- that is a
# DAEMON worker asking about itself in a process where confinement is
# transient (``apparmor_confine`` enters a hat and restores), and a
# thread silently confined without the framework's knowledge would lie.
# The question here is a different one with a different failure mode:
# a runner process confines ONCE, for the life of a session, and the
# check compares every thread's label against the profile the framework
# just asked the kernel for.  It never asks a thread to classify itself,
# and it acts only on POSITIVE evidence -- see
# :func:`verify_thread_confinement`.

#: What a divergent tid is called when the interpreter does not know it.
#:
#: The ``task_dir`` route is COMPLETE -- it lists tids of threads created
#: by C extensions that :func:`threading.enumerate` never sees -- so the
#: name map is necessarily partial on that route.  A placeholder is the
#: honest rendering: "Python does not know this thread" is itself a fact
#: about the thread, and inventing a name from ``/proc/<tid>/comm`` is
#: not available (measured on CPython 3.11: every runner thread's
#: ``comm`` is ``"python"`` regardless of ``thread_name_prefix``).
UNKNOWN_THREAD_NAME = "(unknown)"

#: Default per-thread proc-attr directory; override-able for tests.
#: Shape is ``<task_dir>/<tid>/attr/current``, which a fabricated
#: temp tree reproduces exactly -- the only reason this is a
#: parameter.
DEFAULT_TASK_ATTR_DIR = "/proc/self/task"

#: How long :func:`verify_thread_confinement` keeps re-scanning while it
#: still sees divergence, before calling that divergence durable.
#:
#: NOT a tolerance for the defect -- a grace window for one benign race.
#: The pool recycle that precedes this check (``RunnerRPC``) swaps the
#: executors and shuts the old ones down withOUT waiting, deliberately:
#: waiting could deadlock against a pre-bootstrap task blocked on an
#: outgoing call that only the reader thread (us) can answer.  A worker
#: draining its last task is therefore briefly alive and still carrying
#: the old cred.  It exits within milliseconds; a leaked thread does not.
DEFAULT_VERIFY_GRACE_SECONDS = 2.0

#: Interval between re-scans inside the grace window.
DEFAULT_VERIFY_POLL_SECONDS = 0.05


class ThreadConfinementDivergence(RuntimeError):
    """Raised when a thread's AppArmor label is not the process's.

    Carries the evidence rather than a summary, because the operator's
    next question is always *which thread, and what does it hold* -- and
    because the answer names the exposure: a thread labelled
    ``unconfined`` runs in-process tools outside the kernel boundary
    entirely, while one labelled another session's ``jaato-ws-*``
    profile runs them against the wrong workspace (#1023 impact 3).

    The message names each divergent thread (#1100).  The first version
    printed ``tid=<n> label=<l>`` and nothing else, and a tid is not
    something anyone can look up once the process is gone: two separate
    live incidents were investigated -- one of them by correlating
    daemon-log timestamps against slot pids -- without ever establishing
    what the divergent threads were.  ``threading.enumerate`` had the
    names in hand at scan time and they were discarded.
    """

    def __init__(
        self,
        expected: str,
        divergent: Sequence[Tuple[int, str]],
        *,
        scanned: int,
        route: str,
        names: Optional[Mapping[int, str]] = None,
    ) -> None:
        names = dict(names or {})
        detail = ", ".join(
            f"tid={tid} "
            f"name={names.get(tid, UNKNOWN_THREAD_NAME)!r} "
            f"label={label!r}"
            for tid, label in divergent
        )
        super().__init__(
            f"AppArmor per-thread confinement divergence: expected every "
            f"thread of this process to be confined to {expected!r}, but "
            f"{len(divergent)} of {scanned} scanned threads report "
            f"otherwise ({detail}).  A thread created before the "
            f"aa_change_profile transition keeps the cred it was created "
            f"with -- aa_change_profile is per-task, and "
            f"/proc/self/attr/current reports only the main thread, which "
            f"is why every other check in the tree says this runner is "
            f"confined.  See issue #1023."
        )
        self.expected = expected
        self.divergent = tuple(divergent)
        self.scanned = scanned
        self.route = route
        #: ``{tid: thread name}`` for every thread the interpreter knows,
        #: divergent or not.  Kept whole rather than filtered to the
        #: divergent set so a caller re-rendering this evidence has the
        #: same map the message was built from.
        self.names = names


@dataclass(frozen=True)
class ThreadProfileScan:
    """One walk of every thread's AppArmor label.

    Attributes:
        expected: The profile name every thread should be inside.
        matched: tids whose label is *expected* or a sub-profile of it.
        divergent: ``(tid, label)`` for every thread whose label was
            read successfully and names something else.  This is the
            only field that is POSITIVE evidence of exposure.
        unreadable: ``(tid, reason)`` for every thread whose
            ``attr/current`` could not be read for a reason other than
            the thread having exited.  Absence of evidence -- never
            treated as divergence.
        gone: tids that vanished between enumeration and read.  Benign:
            threads exit.
        route: How tids were enumerated -- ``"task_dir"`` (complete) or
            ``"threading"`` (Python-visible threads only).  Recorded
            because it bounds what the scan could have seen.
        names: ``{tid: thread name}`` for every thread
            :func:`threading.enumerate` knows about, collected on BOTH
            routes (#1100).  A tid absent from this map is one the
            interpreter does not know -- always possible on the
            ``task_dir`` route -- and renders as
            :data:`UNKNOWN_THREAD_NAME`.

            Collected even when nothing diverges, because it is read at
            the moment the walk happens: a thread that exits between the
            scan and the message has already left
            :func:`threading.enumerate`, and a name resolved later would
            be missing exactly for the population that is churning.

            Advisory throughout.  A name never decides whether a thread
            is divergent -- that would be the allow-list of unconfined
            code #1100 rejects -- it only says what the divergent thread
            was, which is the question the original refusal could not
            answer.
    """

    expected: str
    matched: Tuple[int, ...]
    divergent: Tuple[Tuple[int, str], ...]
    unreadable: Tuple[Tuple[int, str], ...]
    gone: Tuple[int, ...]
    route: str
    names: Mapping[int, str] = field(default_factory=dict)

    def name_of(self, tid: int) -> str:
        """What *tid* is called, or :data:`UNKNOWN_THREAD_NAME`."""
        return self.names.get(tid, UNKNOWN_THREAD_NAME)

    @property
    def scanned(self) -> int:
        """How many threads the walk actually reached a verdict on."""
        return len(self.matched) + len(self.divergent) + len(self.unreadable)

    @property
    def uniform(self) -> bool:
        """True when nothing diverged and nothing was unreadable."""
        return not self.divergent and not self.unreadable

    def summary(self) -> str:
        """One line naming what the walk found, for the operator log."""
        return (
            f"threads={self.scanned} matched={len(self.matched)} "
            f"divergent={len(self.divergent)} "
            f"unreadable={len(self.unreadable)} gone={len(self.gone)} "
            f"route={self.route} expected={self.expected!r}"
        )


#: Deliberately mode-TOLERANT name extraction, kept as an alias so the
#: intent is legible at every call site.
#:
#: ``"jaato-ws-abc (enforce)"`` -> ``"jaato-ws-abc"``, and
#: ``"jaato-ws-abc (complain)"`` -> the same.  That is correct for the
#: question its callers ask — *which profile is this thread in, compared
#: with its siblings* (#1023) — where a complain-mode label is not
#: divergence, and a check that read it as divergence would make
#: ``JAATO_APPARMOR_COMPLAIN`` unusable as the diagnostic it is.
#:
#: It is NOT an enforcement assertion, and #1014 ask 1 is precisely that
#: the difference be impossible to mistake: the canonical spelling is
#: :func:`shared.apparmor_label.profile_name_ignoring_mode`, whose name
#: says so.  Use :func:`current_confinement` when the question is whether
#: a boundary exists.
profile_name_of = profile_name_ignoring_mode


def _label_is_inside(label: str, expected: str) -> bool:
    """Is *label* the expected profile, or something strictly narrower?

    ``expected`` itself matches.  So does a hat or sub-profile of it
    (``expected//child``): those DROP rules, so a thread wearing one is
    inside the boundary the session claims, which is the property being
    checked.  Anything else -- ``unconfined``, another session's
    ``jaato-ws-*``, an unrelated profile -- is divergence.

    Mode-tolerant, and that is the right tolerance HERE: this compares
    tasks of one process against each other, and every task of a
    complain-mode process is in complain mode.  It says nothing about
    whether the kernel is enforcing -- :func:`confine_to_profile` owns
    that question (#1014).
    """
    name = profile_name_ignoring_mode(label)
    return name == expected or name.startswith(expected + "//")


def _thread_names() -> Dict[int, str]:
    """``{tid: thread name}`` for every thread the interpreter knows.

    The map the refusal was missing (#1100).  ``threading.enumerate``
    hands back :class:`threading.Thread` objects carrying both
    ``native_id`` and ``name``; :func:`_tids_from_threading` read the
    first and discarded the second, so a divergence could name a tid and
    nothing else -- and a tid is not something an operator can look up
    after the process is gone.  Two live incidents were investigated
    without being able to say what ``82320`` and ``95982`` were.

    Best effort by construction.  A thread that has not started yet has
    ``native_id is None`` and is skipped; the map is only ever used to
    LABEL evidence, never to decide what counts as evidence, so a
    missing entry costs a placeholder and nothing else.
    """
    names: Dict[int, str] = {}
    try:
        main = threading.main_thread()
    except Exception:  # noqa: BLE001 -- diagnostic, never fatal
        main = None
    if main is not None:
        # ``os.getpid()`` is the main thread's tid on Linux, and is what
        # :func:`_tids_from_threading` adds unconditionally, so name it
        # from the same assumption rather than relying on the enumerate
        # pass below to have produced a ``native_id`` for it.
        names[os.getpid()] = getattr(main, "name", "MainThread")
    for thread in threading.enumerate():
        native_id = getattr(thread, "native_id", None)
        if native_id:
            names[int(native_id)] = getattr(thread, "name", "") or "(unnamed)"
    return names


def _tids_from_threading() -> List[int]:
    """Python-visible tids, including this one.

    ``os.getpid()`` is the main thread's tid on Linux and is added
    unconditionally: the main thread is the one thread whose label the
    rest of the tree already reads, so a walk that omitted it would
    lose the reference point.
    """
    tids = {os.getpid()}
    for thread in threading.enumerate():
        native_id = getattr(thread, "native_id", None)
        if native_id:
            tids.add(int(native_id))
    return sorted(tids)


def _enumerate_tids(task_dir: str) -> Tuple[List[int], str]:
    """List the tids of every thread in this process.

    Two routes, and the fallback is not cosmetic.  Listing
    ``/proc/<pid>/task/`` is COMPLETE -- it sees threads created by C
    extensions that Python knows nothing about -- but it needs a
    directory-read grant the per-session profile only carries from
    template v32 onward.  A runner confined by an older template gets
    ``EACCES`` on the listing (and one AVC), so the walk falls back to
    :func:`threading.enumerate`, which needs no grant at all because it
    reads no file: it sees every thread the interpreter created, which
    is every thread this defect is known to produce (the RPC lanes, the
    reader, the telemetry exporter).

    The route is reported rather than hidden, because it bounds what
    the scan could have seen.

    Returns:
        ``(tids, route)`` where route is ``"task_dir"`` or
        ``"threading"``.
    """
    try:
        entries = os.listdir(task_dir)
    except OSError:
        return _tids_from_threading(), "threading"
    tids: List[int] = []
    for entry in entries:
        try:
            tids.append(int(entry))
        except ValueError:
            continue
    if not tids:
        return _tids_from_threading(), "threading"
    return sorted(tids), "task_dir"


def _read_one_thread_label(task_dir: str, tid: int) -> Tuple[str, str]:
    """Read one thread's ``attr/current``.

    Returns:
        ``(kind, value)`` where kind is ``"label"`` (value is the
        label), ``"gone"`` (the thread exited; value is empty) or
        ``"unreadable"`` (value is the reason).
    """
    path = os.path.join(task_dir, str(tid), "attr", "current")
    try:
        with open(path, "r") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return "gone", ""
    except OSError as exc:
        return "unreadable", f"{type(exc).__name__}: {exc}"
    # procfs NUL-terminates this value, and the terminator does not always
    # arrive with the newline the AppArmor form carries -- measured on a
    # host whose active LSM reports a bare ``kernel\x00``.  Stripping only
    # ``\n`` (what :func:`read_current_profile` does) would leave the NUL
    # inside the label, and every later comparison then fails against a
    # profile name that looks identical when printed.
    label = raw.replace("\x00", "").strip()
    if not label:
        return "unreadable", "empty attr/current"
    return "label", label


def scan_thread_profiles(
    expected_profile: str,
    *,
    task_dir: str = DEFAULT_TASK_ATTR_DIR,
) -> ThreadProfileScan:
    """Read every thread's AppArmor label and classify it.

    Pure I/O plus classification -- no policy.  The caller decides what
    a divergent thread means; :func:`verify_thread_confinement` is that
    caller for the bootstrap path.

    Args:
        expected_profile: The profile name the process just entered.
        task_dir: ``/proc/self/task`` in production; a fabricated
            ``<dir>/<tid>/attr/current`` tree in tests.

    Returns:
        A :class:`ThreadProfileScan`.  Never raises for a thread that
        exited mid-walk or whose label could not be read -- those are
        recorded as ``gone`` / ``unreadable`` respectively, because
        acting on absence of evidence is how a verifier takes a host
        down for a ``/proc`` it merely could not read.
    """
    tids, route = _enumerate_tids(task_dir)
    # Read BEFORE the per-thread label walk: a thread that exits during
    # the walk is recorded as ``gone`` and never named, but one that
    # exits between the walk and the message would otherwise lose its
    # name at exactly the moment it is being reported.
    names = _thread_names()
    matched: List[int] = []
    divergent: List[Tuple[int, str]] = []
    unreadable: List[Tuple[int, str]] = []
    gone: List[int] = []

    for tid in tids:
        kind, value = _read_one_thread_label(task_dir, tid)
        if kind == "gone":
            gone.append(tid)
        elif kind == "unreadable":
            unreadable.append((tid, value))
        elif _label_is_inside(value, expected_profile):
            matched.append(tid)
        else:
            divergent.append((tid, value))

    return ThreadProfileScan(
        expected=expected_profile,
        matched=tuple(matched),
        divergent=tuple(divergent),
        unreadable=tuple(unreadable),
        gone=tuple(gone),
        route=route,
        names=names,
    )


def verify_thread_confinement(
    expected_profile: str,
    *,
    task_dir: str = DEFAULT_TASK_ATTR_DIR,
    grace_seconds: float = DEFAULT_VERIFY_GRACE_SECONDS,
    poll_seconds: float = DEFAULT_VERIFY_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ThreadProfileScan:
    """Assert that every thread of this process is inside *expected_profile*.

    Fails on POSITIVE EVIDENCE ONLY.  The distinction decides what this
    function can do to a deployment:

    - a label READ SUCCESSFULLY that names a different profile is proof
      that code in this process runs outside the boundary the session's
      record claims -- :class:`ThreadConfinementDivergence` is raised
      and the caller fails the bootstrap;
    - a label that could not be read proves nothing.  A restricted
      ``/proc`` (``hidepid``, an unusual container, an older profile
      template without the task-dir grant) must not take every session
      on the host down, so the scan is returned with ``unreadable``
      populated and the caller logs rather than raises.

    Divergence is re-checked across a short grace window before it is
    called durable: the pool recycle that precedes this check does not
    wait for in-flight work, so a worker draining its last task is
    briefly alive holding the old cred.  It exits in milliseconds.  A
    thread that is still divergent at the end of the window is not
    draining -- it is the #1023 population.

    Args:
        expected_profile: Profile the process just entered.
        task_dir: See :func:`scan_thread_profiles`.
        grace_seconds: Total time to keep re-scanning while divergence
            persists.
        poll_seconds: Interval between re-scans.
        sleep: Injection point so the grace window is testable without
            spending it.
        monotonic: Injection point for the same reason.

    Returns:
        The final :class:`ThreadProfileScan` when no divergence
        survived the window.

    Raises:
        ThreadConfinementDivergence: divergence persisted.
    """
    scan = scan_thread_profiles(expected_profile, task_dir=task_dir)
    if not scan.divergent:
        return scan

    deadline = monotonic() + max(0.0, grace_seconds)
    while monotonic() < deadline:
        sleep(poll_seconds)
        scan = scan_thread_profiles(expected_profile, task_dir=task_dir)
        if not scan.divergent:
            return scan

    # The incident register (#1122).  Raised BEFORE the exception, and
    # from here rather than from a caller, because this refusal happens
    # before any session exists -- there is nobody above to notice it, and
    # a bootstrap refused for thread divergence is evidence that code ran
    # outside the boundary the session record claims.  The guard working,
    # and exactly the thing Art. 72's post-market monitoring is about.
    _raise_confinement_incident(expected_profile, scan)
    raise ThreadConfinementDivergence(
        expected_profile,
        scan.divergent,
        scanned=scan.scanned,
        route=scan.route,
        names=scan.names,
    )


def _raise_confinement_incident(expected_profile: str, scan: Any) -> None:
    """Record a refused bootstrap in the incident register.  Never raises.

    Separate from the raise so the refusal is unaffected by anything that
    happens here: the whole point of the guard is that it fails CLOSED,
    and a register that could turn its exception into a different one
    would be the worst possible regression in it.
    """
    try:
        from shared.incidents import KIND_CONFINEMENT_REFUSED, raise_incident
        raise_incident(
            KIND_CONFINEMENT_REFUSED,
            f"{len(scan.divergent)} of {scan.scanned} scanned threads are "
            f"not in {expected_profile!r} (route={scan.route})",
            site="server/runner/bootstrap.py::verify_thread_confinement",
        )
    except Exception:  # noqa: BLE001 -- see the docstring
        pass
