"""Filesystem containment for notebook cell execution (issue #710).

Why this module exists
======================

``cli`` contains the paths a model names: every path token in a command is run
through :func:`~shared.plugins.sandbox_utils.check_path_with_jaato_containment`
and the command is refused when one falls outside the workspace.  The notebook
kernel had nothing equivalent — ``cwd=workspace_root`` makes *relative* paths
resolve in-workspace and says nothing about absolute ones — so
``open('/etc/hostname').read()`` succeeded in a cell after ``cli`` had refused
``cat /etc/hostname``, and ``subprocess.run(['cat', '/etc/hostname'])`` spawned
the very command ``cli`` had just refused.  Whatever ``cli`` contains, a model
could step around by asking the notebook to do the same thing in Python.

The boundary, in order of strength
==================================

1. **A kernel-enforced AppArmor profile.**  The runner's per-session profile
   grants ``ix`` on the interpreters it may exec, so the kernel subprocess runs
   under the *same* profile as its parent and its syscalls are bounded by the
   kernel whatever the cell attempts.  Nothing in this module is installed on
   that path — there is a real boundary already, and a second, weaker one in
   userspace would only add failure modes.
2. **This module's audit hook** (:func:`install`), on an unconfined host.  A
   :pep:`578` hook refuses the filesystem events a cell raises when the path
   falls outside the same boundary ``cli`` applies.  It sees *actual* I/O, so
   it does not care how the path was spelled, constructed or obfuscated — the
   weakness of the static :mod:`~shared.plugins.notebook.code_analyzer` scan.
3. **Nothing**, which is a refusal: a kernel that reaches neither of the above
   declines to run cells (see ``kernel_main``), and
   ``SubprocessKernelBackend.execution_boundary`` reports the same answer one
   layer up so the refusal is stated before a kernel is even spawned.

What this is NOT
================

**An in-process Python hook is not a kernel boundary**, and this docstring is
the honest version of that claim.  Two limits, both deliberate:

*A cell that sets out to escape can attack the hook itself* — rebind this
module's ``_POLICY``, reach the policy object through :mod:`gc`, or load a
shared library whose own ``open(2)`` calls are not audited (which is why
:func:`_check_dlopen` refuses ``ctypes`` loads of anything outside the read
roots).

*A spawned child process is not audited at all.*  Its **arguments** are
checked, which is what stops ``subprocess.run(['cat', '/etc/hostname'])``;
once it execs, the child is bounded by the OS and nothing else, exactly as a
command ``cli`` approved is.  So ``!pip install X`` writes wherever pip writes
— point ``workspace_venv`` at the workspace if that matters.  This boundary is
the one ``cli`` provides, brought to a surface that had none; it is not a
stronger one.

Together they close the accidental and the opportunistic case — which is what
#710 measured: a session asked to study a sibling checkout simply browsed it,
no jailbreak involved — and they make a deliberate escape an act that has to
defeat a declared boundary rather than walk through an open door.  For a
hostile-model threat model the answer is AppArmor (tier 1), not this.

Layering with the in-process gate
=================================

``backends/local.py`` answers a different question — *may model-authored code
run in the host interpreter at all?* — and answers it for the ``local``
backend, which shares the runner's memory (the tool executor, other plugins'
state, ``session_env`` secrets).  That gate stays exactly as it is.  This
module answers *where may cell code reach on the filesystem?* for the
``subprocess`` backend, which is the default.  Neither replaces the other: the
in-process gate does not bound filesystem reach, and containment does not make
in-process execution safe.

Threading and re-entrancy
=========================

An audit hook is called for *every* audited event in the process, including
events raised by the hook's own work (``os.stat`` from ``realpath``).  The
dispatch table is consulted first so an unhandled event costs one dict lookup,
and a thread-local re-entrancy flag makes the hook a no-op while it is already
running on that thread.

.. warning::
   :func:`install` is irreversible — CPython offers no way to remove an audit
   hook.  Never call it from a test process; drive the kernel subprocess
   instead, and unit-test :class:`ContainmentPolicy` directly.
"""

import logging
import os
import shlex
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shared.plugins.sandbox_utils import check_path_with_jaato_containment

logger = logging.getLogger(__name__)

# ---- boundary kinds reported to the runner (READY frame) and traced ---------

BOUNDARY_APPARMOR = "apparmor"
BOUNDARY_AUDIT = "audit"
BOUNDARY_OPT_OUT = "opt-out"
BOUNDARY_NONE = "none"

#: Operator opt-out: run notebook cells with NO filesystem boundary.  The
#: sibling of ``JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC`` (which governs whether
#: cells may run in the host process, not where they may reach), and announced
#: at WARNING for the same reason: the leaky posture must never be the quiet
#: path of least resistance.
UNCONTAINED_OPT_IN_ENV = "JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC"

#: Read-only system paths a stock interpreter needs for ordinary work
#: (timezone data, the trust store, cpu topology).  Narrow and non-sensitive
#: by construction: no ``/proc/self`` (``environ`` holds the session's
#: secrets), no ``/etc`` beyond the two timezone files.  A deployment that
#: needs more says so with ``plugin_configs.notebook.allow_read_paths``.
DEFAULT_SYSTEM_READ_PATHS: Tuple[str, ...] = (
    "/etc/localtime",
    "/etc/timezone",
    "/usr/share/zoneinfo",
    "/etc/ssl",
    "/usr/lib/ssl",
    "/etc/ca-certificates",
    "/usr/share/ca-certificates",
    "/etc/pki",
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/sys/devices/system/cpu",
)


class NotebookContainmentError(PermissionError):
    """A cell reached outside the notebook's filesystem boundary.

    Subclasses :class:`PermissionError` (hence :class:`OSError`) deliberately:
    an audit hook fires *inside* arbitrary library code, and the import
    machinery, :mod:`shutil` and friends already treat ``OSError`` from a
    filesystem call as "this path is not available" rather than crashing.  A
    bespoke exception type would turn a contained ``__pycache__`` write into a
    hard import failure.
    """


def apparmor_enforced_profile() -> Optional[str]:
    """Return the active AppArmor profile name iff it is *enforced*.

    Reads ``/proc/self/attr/current``.  Returns the profile label (e.g.
    ``jaato-ws-<id>//child``) only when an AppArmor profile is active in
    ``enforce`` mode — a real kernel boundary on this process's syscalls.
    Returns ``None`` when the process is unconfined, when AppArmor is
    unavailable, or when the profile is in ``complain`` mode (which logs but
    does not block, so it is not a boundary).

    Lives here rather than in ``backends/local.py`` (its original home)
    because both notebook execution paths need the same answer: the in-process
    gate asks it about the runner, the kernel asks it about itself.
    """
    try:
        with open("/proc/self/attr/current", "r") as fh:
            # The kernel terminates this value with a NUL byte (and a
            # newline); str.strip() alone leaves the NUL, which would
            # false-negative an enforced profile and break the confined
            # path. Match the convention in server/runner_spawner.py:251.
            raw = fh.read().strip("\x00 \t\r\n")
    except OSError:
        return None
    if not raw or raw.startswith("unconfined"):
        return None
    # Format is typically 'name (enforce)' or 'name (complain)'; a bare
    # 'name' with no mode annotation is treated conservatively as not a
    # boundary.
    if "(" not in raw:
        return None
    name, _, mode = raw.partition(" (")
    if mode.rstrip(")").strip() != "enforce":
        return None
    name = name.strip()
    return name or None


def env_truthy(name: str) -> bool:
    """Return True when env var ``name`` holds a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def interpreter_read_roots() -> List[str]:
    """Directories a cell may read from because the *interpreter* needs them.

    The kernel keeps importing after containment is installed (a cell's own
    ``import pandas`` is the normal case), and every import is an audited
    ``open``.  Denying the interpreter's own installation would not contain a
    model, it would break Python.

    Collected from the live interpreter rather than hardcoded: ``sys.prefix``
    and its ``base_``/``exec_`` variants (venv + its base), the executable's
    directory, and every existing directory currently on ``sys.path`` — which
    is what carries ``site-packages`` and the runner's own source tree.  The
    empty ``sys.path`` entry (the cwd, i.e. the workspace) is skipped: the
    workspace is judged by the containment check, not waved through.

    Returns:
        Absolute, de-duplicated, realpath-resolved directories.  Read-only:
        these roots grant reads, never writes.
    """
    candidates: List[str] = [
        sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
    ]
    if sys.executable:
        candidates.append(os.path.dirname(sys.executable))
    candidates.extend(p for p in sys.path if p)
    return _normalize_roots(candidates)


def _normalize_roots(paths: Iterable[str]) -> List[str]:
    """Realpath, expand and de-duplicate ``paths``, preserving order.

    Existence is deliberately NOT required.  Three of the four sources feeding
    this are legitimately allowed to name something that is not there yet or is
    not a directory: :data:`DEFAULT_SYSTEM_READ_PATHS` names plain files
    (``/etc/localtime``), ``sandbox add`` can grant an output directory a cell
    is about to create, and ``sys.path`` carries entries (a stdlib zip, a
    stale entry) that are not directories.  A root that does not exist grants
    nothing, because nothing resolves under it — filtering on ``isdir`` would
    quietly drop grants an operator made.
    """
    seen: Dict[str, None] = {}
    for raw in paths:
        if not raw:
            continue
        try:
            resolved = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
        except (OSError, ValueError):
            continue
        seen.setdefault(resolved, None)
    return list(seen)


def _under(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` itself or lies beneath it."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _spawn_path_like(token: str) -> bool:
    """True when a spawned command's word should be containment-checked.

    Mirrors ``cli.plugin._path_like``: absolute paths, ``..`` traversal and
    explicit ``./`` / ``~`` prefixes count; option flags and URLs do not.  A
    bare relative word (``README.md``, ``src/app.py``) is deliberately *not*
    checked — the child inherits the kernel's cwd, which is the workspace, so
    it resolves inside the boundary anyway.

    Copied rather than imported on purpose: ``cli.plugin`` pulls the whole
    tool surface (registry, subprocess runner, command analysis) into every
    kernel process, and the kernel is spawned per notebook.  The heuristic is
    five lines and is exercised on both sides.
    """
    if not token or token.startswith("-"):
        return False
    if "://" in token:
        return False
    return (token.startswith("/") or ".." in token
            or token.startswith("./") or token.startswith("~"))


class ContainmentPolicy:
    """The question "may cell code touch this path?", and its answer.

    One instance is built in the kernel at startup and handed to
    :func:`install`; it stays live for the kernel's lifetime because
    ``sandbox add`` / ``sandbox deny`` can change the answer mid-session (the
    runner refreshes :meth:`set_extra_allows` on every cell).

    Attributes:
        workspace_root: The session workspace.  Everything the boundary
            permits is expressed relative to it, and an unset root means the
            policy cannot answer — :meth:`allows` then refuses.
        allow_tmp: Whether ``/tmp`` is readable/writable, matching ``cli``'s
            default for the same reason (scratch files are ordinary work).
        read_roots: Directories readable but NOT writable — the interpreter
            installation plus :data:`DEFAULT_SYSTEM_READ_PATHS` and any
            operator additions.
        write_roots: Directories readable AND writable outside the workspace —
            today only a configured ``workspace_venv`` that lives elsewhere,
            which ``!pip install`` must be able to write.
    """

    def __init__(
        self,
        workspace_root: Optional[str],
        *,
        allow_tmp: bool = True,
        read_roots: Optional[Iterable[str]] = None,
        write_roots: Optional[Iterable[str]] = None,
    ) -> None:
        self.workspace_root = (
            os.path.realpath(workspace_root) if workspace_root else None)
        self.allow_tmp = allow_tmp
        self.read_roots: List[str] = _normalize_roots(read_roots or ())
        self.write_roots: List[str] = _normalize_roots(write_roots or ())
        # sandbox add / sandbox deny, refreshed per cell by the runner.
        self._extra_read: List[str] = []
        self._extra_write: List[str] = []
        self._denied: List[str] = []

    def set_extra_allows(
        self,
        read: Sequence[str] = (),
        write: Sequence[str] = (),
        deny: Sequence[str] = (),
    ) -> None:
        """Replace the session's ``sandbox add`` / ``sandbox deny`` paths.

        Called by the kernel loop from the ``allow`` block of each ``execute``
        frame, so a path authorized *after* the kernel spawned is honoured on
        the next cell rather than at the next kernel.  Replaces wholesale — a
        revoked authorization must actually disappear.

        Args:
            read: Paths authorized for reading (``readonly`` or ``readwrite``).
            write: Paths authorized for writing (``readwrite`` only).
            deny: Paths explicitly denied; these outrank every allowance,
                including the workspace itself.
        """
        self._extra_read = _normalize_roots(read)
        self._extra_write = _normalize_roots(write)
        self._denied = _normalize_roots(deny)

    def allows(self, path: str, mode: str) -> bool:
        """Decide one access.

        Args:
            path: The path the cell is reaching for; absolute or relative to
                the kernel's cwd (the workspace).
            mode: ``"read"`` or ``"write"``.

        Returns:
            True when the access is within the boundary.  A path that cannot
            be resolved at all is refused — the same fail-closed posture
            ``cli._is_path_within_workspace`` takes on ``OSError``.
        """
        if not self.workspace_root:
            return False
        try:
            absolute = os.path.abspath(os.path.expanduser(path))
            resolved = os.path.realpath(absolute)
        except (OSError, ValueError):
            return False
        if any(_under(resolved, root) for root in self._denied):
            return False
        if mode == "read" and any(_under(resolved, r) for r in self.read_roots):
            return True
        if any(_under(resolved, root) for root in self.write_roots):
            return True
        if any(_under(resolved, root) for root in self._extra_write):
            return True
        if mode == "read" and any(_under(resolved, r) for r in self._extra_read):
            return True
        return check_path_with_jaato_containment(
            absolute, self.workspace_root, None,
            allow_tmp=self.allow_tmp, mode=mode,
        )

    def refuse(self, path: str, mode: str, what: str) -> "NotebookContainmentError":
        """Build the error a refused access raises into the cell.

        The message names the boundary and the escape hatches, because the
        model reads it and its next move should be a workspace-relative path
        or a request to the operator — not a retry loop.
        """
        return NotebookContainmentError(
            f"notebook containment: {what} {mode} of {path!r} is outside the "
            f"workspace boundary ({self.workspace_root}). Notebook cells are "
            "contained to the workspace (and /tmp), the same boundary the cli "
            "tool applies. Use a workspace-relative path, ask the operator to "
            "run 'sandbox add <path>', or set "
            f"{UNCONTAINED_OPT_IN_ENV}=1 to accept an uncontained notebook."
        )


# --- audit-event dispatch ----------------------------------------------------
#
# Each table maps a PEP 578 event name to the positional indices of its
# path-bearing arguments.  Only events that name a path a cell chose are
# listed: metadata calls (``os.stat``) are deliberately absent — they leak
# existence, not content, and enforcing them breaks library probing for a
# boundary that is about reach, not reconnaissance.

_WRITE_PATH_EVENTS: Dict[str, Tuple[int, ...]] = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.link": (0, 1),
    "os.symlink": (1,),          # arg 0 is the target string, not an access
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.utime": (0,),
    "os.setxattr": (0,),
    "os.removexattr": (0,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
}

_READ_PATH_EVENTS: Dict[str, Tuple[int, ...]] = {
    "os.listdir": (0,),
    "os.scandir": (0,),
    "os.getxattr": (0,),
    "os.listxattr": (0,),
    "pathlib.Path.glob": (0,),
}

#: ``(executable_index, argv_index, cwd_index)`` per spawn event; ``None``
#: where the event does not carry that piece.
_SPAWN_EVENTS: Dict[str, Tuple[Optional[int], Optional[int], Optional[int]]] = {
    "subprocess.Popen": (0, 1, 2),
    "os.exec": (0, 1, None),
    "os.posix_spawn": (0, 1, None),
    "os.spawn": (1, 2, None),
}

_POLICY: Optional[ContainmentPolicy] = None
_INSTALLED = False
_reentry = threading.local()


def current_policy() -> Optional[ContainmentPolicy]:
    """The policy the installed hook is enforcing, if any (kernel + tests)."""
    return _POLICY


def _path_arg(value: Any) -> Optional[str]:
    """Coerce one audit-event argument to a checkable path, or ``None``.

    ``None`` means "nothing to check here": a file descriptor (``open(3)``),
    an omitted optional argument (``os.scandir()`` with no path), or a value
    that is not a path at all.
    """
    if isinstance(value, str):
        return value or None
    if isinstance(value, bytes):
        return value.decode("utf-8", "surrogateescape") or None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return None


def _write_intent(mode: Any, flags: Any) -> bool:
    """Whether an ``open`` audit event describes a write.

    ``io.open`` reports the mode string; ``os.open`` reports ``None`` and the
    raw flags.  Unknown shapes are read as writes — the stricter reading is
    the safe one when the event cannot be understood.
    """
    if isinstance(mode, str):
        return any(ch in mode for ch in ("w", "a", "x", "+"))
    if isinstance(flags, int):
        write_flags = (os.O_WRONLY | os.O_RDWR | getattr(os, "O_CREAT", 0)
                       | getattr(os, "O_TRUNC", 0) | getattr(os, "O_APPEND", 0))
        return bool(flags & write_flags)
    return mode is not None or flags is not None


def _check_open(policy: ContainmentPolicy, args: Sequence[Any]) -> None:
    """Enforce the ``open`` event (``open``/``io.open``/``os.open``)."""
    path = _path_arg(args[0]) if args else None
    if path is None:
        return
    mode = "write" if _write_intent(
        args[1] if len(args) > 1 else None,
        args[2] if len(args) > 2 else None) else "read"
    if not policy.allows(path, mode):
        raise policy.refuse(path, mode, "open")


def _check_indexed(policy: ContainmentPolicy, args: Sequence[Any],
                   indices: Sequence[int], mode: str, what: str) -> None:
    """Enforce ``mode`` on the path arguments of ``args`` named by ``indices``."""
    for index in indices:
        if index >= len(args):
            continue
        path = _path_arg(args[index])
        if path is not None and not policy.allows(path, mode):
            raise policy.refuse(path, mode, what)


def _check_spawn(policy: ContainmentPolicy, args: Sequence[Any],
                 layout: Tuple[Optional[int], Optional[int], Optional[int]]) -> None:
    """Enforce containment on a process spawn.

    This is the row #710 called "the point": ``cli`` had just refused
    ``cat /etc/hostname`` and the notebook spawned it.  A child process is not
    audited — its own ``open(2)`` calls happen in another interpreter, or in no
    interpreter at all — so the check has to happen here, on the arguments,
    exactly as ``cli`` checks the command it is asked to run.

    Coarser than ``cli`` in one respect, deliberately: ``cli`` derives
    read-vs-write for each token by parsing the shell command, which needs the
    full command grammar.  Here every path-shaped word is checked for *read*,
    which contains the escape (the path is refused outright when it is outside
    the boundary) while declining to guess intent.
    """
    exec_index, argv_index, cwd_index = layout
    if cwd_index is not None and cwd_index < len(args):
        cwd = _path_arg(args[cwd_index])
        if cwd is not None and not policy.allows(cwd, "read"):
            raise policy.refuse(cwd, "read", "spawn cwd")
    words: List[str] = []
    if exec_index is not None and exec_index < len(args):
        executable = _path_arg(args[exec_index])
        if executable is not None:
            words.append(executable)
    if argv_index is not None and argv_index < len(args):
        words.extend(_argv_words(args[argv_index]))
    for word in words:
        if _spawn_path_like(word) and not policy.allows(word, "read"):
            raise policy.refuse(word, "read", "spawn argument")


def _argv_words(argv: Any) -> List[str]:
    """Flatten a spawn event's argv into checkable words."""
    if isinstance(argv, (list, tuple)):
        return [w for w in (_path_arg(a) for a in argv) if w]
    word = _path_arg(argv)
    return [word] if word else []


def _check_system(policy: ContainmentPolicy, args: Sequence[Any]) -> None:
    """Enforce containment on ``os.system``'s shell string."""
    command = _path_arg(args[0]) if args else None
    if command is None:
        return
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    for word in words:
        if _spawn_path_like(word) and not policy.allows(word, "read"):
            raise policy.refuse(word, "read", "shell command argument")


def _check_dlopen(policy: ContainmentPolicy, args: Sequence[Any]) -> None:
    """Refuse a ``ctypes`` load of anything outside the read roots.

    A loaded shared library runs native code whose ``open(2)`` calls raise no
    audit event, so ``ctypes.CDLL("libc.so.6").open(b"/etc/shadow", 0)`` would
    walk straight past every other check in this module.  Libraries that ship
    inside the interpreter installation (a wheel's bundled ``.so``) resolve
    under the read roots and load normally; a bare soname or ``None`` (which
    means "this process's own symbols") does not, and is refused.
    """
    name = _path_arg(args[0]) if args else None
    if name is None:
        raise NotebookContainmentError(
            "notebook containment: ctypes.dlopen(None) is refused — loading "
            "native code would bypass the notebook's filesystem boundary.")
    if not os.path.isabs(name) or not policy.allows(name, "read"):
        raise NotebookContainmentError(
            f"notebook containment: loading the shared library {name!r} is "
            "refused — native code runs beneath the notebook's filesystem "
            "boundary. Only libraries inside the interpreter installation "
            "may be loaded.")


def _dispatch(policy: ContainmentPolicy, event: str, args: Sequence[Any]) -> None:
    """Route one audited event to its check.  Raises on refusal."""
    if event == "open":
        _check_open(policy, args)
    elif event in _WRITE_PATH_EVENTS:
        _check_indexed(policy, args, _WRITE_PATH_EVENTS[event], "write", event)
    elif event in _READ_PATH_EVENTS:
        _check_indexed(policy, args, _READ_PATH_EVENTS[event], "read", event)
    elif event in _SPAWN_EVENTS:
        _check_spawn(policy, args, _SPAWN_EVENTS[event])
    elif event == "os.system":
        _check_system(policy, args)
    elif event == "ctypes.dlopen":
        _check_dlopen(policy, args)


def _audit_hook(event: str, args: Sequence[Any]) -> None:
    """The installed :pep:`578` hook.

    Fast path first: an event with no entry in the dispatch tables costs one
    membership test, because this runs on *every* audited operation in the
    process.  The thread-local re-entrancy flag then makes the hook inert
    while it is already running on this thread — the checks themselves stat
    and resolve paths, which raises further events.

    An unexpected failure inside a check is converted into a refusal rather
    than swallowed: a containment check that cannot reach a verdict has not
    permitted anything, and ``cli`` fails closed on the same condition.
    """
    policy = _POLICY
    if policy is None or event not in _AUDITED_EVENTS:
        return
    if getattr(_reentry, "busy", False):
        return
    _reentry.busy = True
    try:
        _dispatch(policy, event, args)
    except NotebookContainmentError:
        raise
    except Exception as exc:  # noqa: BLE001 — a check that failed is a refusal
        raise NotebookContainmentError(
            f"notebook containment: refusing {event} — the containment check "
            f"failed ({type(exc).__name__}: {exc})") from exc
    finally:
        _reentry.busy = False


_AUDITED_EVENTS = frozenset(
    {"open", "os.system", "ctypes.dlopen"}
    | set(_WRITE_PATH_EVENTS) | set(_READ_PATH_EVENTS) | set(_SPAWN_EVENTS)
)


def install(policy: ContainmentPolicy) -> None:
    """Install ``policy`` as this process's notebook containment.

    Idempotent in the sense that matters: the hook is added to the interpreter
    once, and later calls only swap the policy it enforces.  That is all a
    kernel needs — it installs once at startup and refreshes the policy's
    authorized paths per cell.

    .. warning::
       There is no uninstall.  Call this only in a process whose entire
       remaining purpose is running model-authored cells (the notebook
       kernel), never in the runner and never in a test process.
    """
    global _POLICY, _INSTALLED
    _POLICY = policy
    if not _INSTALLED:
        sys.addaudithook(_audit_hook)
        _INSTALLED = True


def establish_containment(
    workspace_root: Optional[str],
    *,
    allow_tmp: bool = True,
    extra_read_paths: Sequence[str] = (),
    extra_write_paths: Sequence[str] = (),
    opt_out: bool = False,
) -> Tuple[str, str]:
    """Decide and install this process's boundary, strongest available first.

    Args:
        workspace_root: The session workspace the cells are contained to.
        allow_tmp: Whether ``/tmp`` is inside the boundary (matches ``cli``).
        extra_read_paths: Operator-configured readable paths
            (``plugin_configs.notebook.allow_read_paths``) plus the system
            defaults.
        extra_write_paths: Paths that must also be writable — a
            ``workspace_venv`` outside the workspace, which ``!pip`` writes.
        opt_out: The operator accepted an uncontained notebook.

    Returns:
        ``(kind, description)`` where ``kind`` is one of
        :data:`BOUNDARY_APPARMOR`, :data:`BOUNDARY_AUDIT`,
        :data:`BOUNDARY_OPT_OUT` or :data:`BOUNDARY_NONE`.  Only
        :data:`BOUNDARY_NONE` means "refuse to run cells"; the caller enforces
        that, because this function's job is to establish a boundary, not to
        decide policy about its absence.
    """
    profile = apparmor_enforced_profile()
    if profile:
        return BOUNDARY_APPARMOR, f"AppArmor-enforced profile {profile}"
    if opt_out:
        logger.warning(
            "Notebook kernel: running cells with NO filesystem boundary "
            "(enabled via %s / allow_uncontained_exec). Cell code can read "
            "and execute anything this user can.", UNCONTAINED_OPT_IN_ENV)
        return BOUNDARY_OPT_OUT, "operator opt-out (uncontained)"
    if not workspace_root:
        return BOUNDARY_NONE, "no workspace root resolved"
    if not hasattr(sys, "addaudithook"):
        return BOUNDARY_NONE, "interpreter provides no audit hooks"
    read_roots = list(interpreter_read_roots())
    read_roots.extend(DEFAULT_SYSTEM_READ_PATHS)
    read_roots.extend(extra_read_paths)
    install(ContainmentPolicy(
        workspace_root, allow_tmp=allow_tmp,
        read_roots=read_roots, write_roots=extra_write_paths,
    ))
    return BOUNDARY_AUDIT, f"audit-hook workspace containment ({workspace_root})"
