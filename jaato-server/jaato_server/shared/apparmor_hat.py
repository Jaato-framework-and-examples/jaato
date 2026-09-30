"""Per-thread ``tool_hat`` entry for in-process tool bodies (#1422).

The AppArmor template renders a ``^tool_hat`` hat inside every session
profile.  It mirrors the base profile and adds the read-denies on
user-authored ``.jaato/`` config plus the write-deny on
``.jaato/references/**`` that the BASE profile no longer carries.  Until
#1422 nothing entered it, so every model-called in-process tool
(``file_edit``, ``readFile``, ``glob_files``, memory, ``web_fetch``,
``proposeReference``, out-of-tree plugins) ran in base, and base had to
carry denies that also blocked the runner's own bookkeeping.

This module is the one place a thread enters and leaves the hat.

**Per task.**  ``change_hat`` confines the CALLING thread only, so the
context manager runs on the thread that runs the tool body, and restores
it in ``finally``.  Parallel tool execution (up to 8 workers) and the
auto-background pool each enter on their own thread.

**The token.**  ``change_hat`` takes a 64-bit token; only a write naming
the same token returns the thread to the parent profile.  A wrong token
makes the kernel kill the task, so this module never writes a
``changehat`` from a thread it already knows is in the hat
(:class:`ToolHatError` instead).  The token lives in a local variable of
the context manager for the length of one tool call.

**What this does NOT bound.**  Python code running in the same process
can find the token (frame inspection, ``gc``), return to base itself, and
from base write anything base allows.  The hat therefore stops accidents
and ordinary tool code paths, not deliberately hostile in-process code.
Only out-of-process execution (``//child``, where ``cli``,
``interactive_shell`` and the notebook kernel run) is a hard boundary.

**Nesting.**  A tool body that executes another tool on the same thread
does not re-enter (a hat cannot enter itself); a per-thread depth count
makes only the outermost call change the label.

**A thread already in the hat.**  A thread created by code running in
the hat is born in it (confinement is inherited at thread creation).
Such a thread runs a tool body where it is: it is already in the hat, and
writing a fresh token from inside a hat is a wrong token, which the
kernel answers by killing the task.

**Stuck threads.**  :func:`stuck_hat_tids` names the threads whose return
to base failed.  #1023's per-thread verification reads it: a worker
legitimately mid-tool, or a thread born in the hat, is inside the
boundary; a worker left in the hat by a failed return is divergence.

Pure stdlib plus :mod:`shared.apparmor_label`, which is itself pure
stdlib.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from contextlib import contextmanager
from typing import Callable, ContextManager, FrozenSet, Iterator, Optional

from jaato_server.shared.apparmor_label import (
    TOOL_HAT,
    AppArmorLabel,
    parse_label,
    profile_name_ignoring_mode,
)

logger = logging.getLogger(__name__)

_local = threading.local()
_stuck_lock = threading.Lock()
_stuck: set = set()


class ToolHatError(RuntimeError):
    """The tool body was not run because the thread could not enter the hat.

    Fail-closed: the base profile no longer write-denies
    ``.jaato/references/**``, so running a tool body in base because the
    hat was unavailable would widen what the tool may do.
    """


def hat_profile(base_profile: str) -> str:
    """The label profile a thread wears inside the hat."""
    return f"{base_profile}//{TOOL_HAT}"


def thread_attr_path() -> str:
    """The calling thread's own ``attr/current``.

    ``/proc/self/attr/current`` is the MAIN thread's (#1023); ``change_hat``
    must be written to the thread that is changing.
    """
    return f"/proc/self/task/{threading.get_native_id()}/attr/current"


def _write_attr(path: str, payload: str) -> None:
    """One ``write(2)`` to ``attr/current`` (no truncate, no buffering)."""
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, payload.encode("ascii"))
    finally:
        os.close(fd)


def _read_attr(path: str) -> Optional[AppArmorLabel]:
    """The label at *path*, or ``None`` when it could not be read."""
    try:
        with open(path, "r") as handle:
            return parse_label(handle.read())
    except OSError:
        return None


def _new_token() -> int:
    """A non-zero 64-bit token (zero means "no token" to the kernel)."""
    token = 0
    while token == 0:
        token = secrets.randbits(64)
    return token


def stuck_hat_tids() -> FrozenSet[int]:
    """Native ids of the threads whose return from the hat failed."""
    with _stuck_lock:
        return frozenset(_stuck)


def _entry_state(label: Optional[AppArmorLabel], base_profile: str) -> str:
    """Where the calling thread stands: ``"base"``, ``"hat"`` or ``"unknown"``.

    Raises:
        ToolHatError: the label names neither the session profile nor its
            hat (``unconfined`` included).  Only positive evidence
            refuses; an unreadable label is ``"unknown"`` and the write
            decides.
    """
    if label is None or not label.raw:
        return "unknown"
    name = profile_name_ignoring_mode(label.raw)
    if name == base_profile:
        return "base"
    if name == hat_profile(base_profile):
        return "hat"
    raise ToolHatError(
        f"thread {threading.get_native_id()} is in {label.raw!r}, not the "
        f"session profile {base_profile!r}; refusing to run a tool body "
        f"outside {hat_profile(base_profile)}"
    )


def _return(path: str, token: int, base_profile: str, write) -> bool:
    """Leave the hat.  Returns whether the write was accepted.

    A failed return records the thread in :func:`stuck_hat_tids`.
    """
    try:
        write(path, f"changehat {token:016x}^")
        return True
    except OSError as exc:
        with _stuck_lock:
            _stuck.add(threading.get_native_id())
        logger.error(
            "AppArmor: thread %d could not return from %s: %s.  The thread "
            "stays in the hat; #1023 verification reports it as divergent.",
            threading.get_native_id(), hat_profile(base_profile), exc,
        )
        return False


@contextmanager
def tool_hat(
    base_profile: str,
    *,
    attr_path: Optional[str] = None,
    write: Callable[[str, str], None] = _write_attr,
    read: Callable[[str], Optional[AppArmorLabel]] = _read_attr,
) -> Iterator[None]:
    """Run the body in ``<base_profile>//tool_hat`` on this thread.

    Args:
        base_profile: The session's base profile (``jaato-ws-...``).
        attr_path: Override for tests; defaults to the calling thread's
            ``/proc/self/task/<tid>/attr/current``.
        write: Injection point for the ``attr/current`` write.
        read: Injection point for the label read.

    Raises:
        ToolHatError: the thread could not enter the hat.  The body did
            not run.
    """
    depth = getattr(_local, "depth", 0)
    path = attr_path or thread_attr_path()
    if depth or _entry_state(read(path), base_profile) == "hat":
        # Nested call, or a thread born in the hat: run where it is.
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth -= 1
        return

    token = _new_token()
    try:
        write(path, f"changehat {token:016x}^{TOOL_HAT}")
    except OSError as exc:
        raise ToolHatError(
            f"could not enter {hat_profile(base_profile)} on thread "
            f"{threading.get_native_id()}: {exc}"
        ) from exc
    after = read(path)
    if after is not None and after.raw and (
        profile_name_ignoring_mode(after.raw) != hat_profile(base_profile)
    ):
        _return(path, token, base_profile, write)
        raise ToolHatError(
            f"change_hat was accepted but thread {threading.get_native_id()} "
            f"reports {after.raw!r}, not {hat_profile(base_profile)}"
        )

    _local.depth = 1
    try:
        yield
    finally:
        _local.depth = 0
        _return(path, token, base_profile, write)
        token = 0


def make_tool_hat_context(base_profile: str) -> Callable[[], ContextManager]:
    """A zero-argument factory for ``ToolExecutor.set_apparmor_context``."""
    def _enter() -> ContextManager:
        return tool_hat(base_profile)
    return _enter
