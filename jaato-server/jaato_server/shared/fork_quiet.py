"""Nothing is written to ``sys.stderr`` by a forked child before it is adopted (#1673).

A runner forks from a multi-threaded process (the RPC lanes, an OTel
export thread, log handlers).  ``subprocess`` with a ``preexec_fn`` runs
CPython's post-fork hooks in the child: every library's
``os.register_at_fork(after_in_child=...)`` callback.  When one of them
raises, CPython reports it through ``sys.unraisablehook``, whose default
writes ``Exception ignored in atfork callback ...`` to ``sys.stderr``.

That write takes the stream's buffer lock, which CPython does not reset
after ``fork()``.  If another thread of the parent held it at the moment
of the fork (writing a log line, ``OTLP export 401`` every few seconds),
the single-threaded child waits on it forever in ``futex_wait``, before
``//child``, before ``exec``, and the tool call never returns.  That is
the incident's signature exactly.

The callback that raises in practice: OpenTelemetry SDK up to 1.40
registers ``lambda: weak_reinit()()`` for each ``BatchSpanProcessor``.
The telemetry plugin builds one per session and shuts it down at
``session.end``; once it is collected, the weak reference is dead and the
lambda raises ``TypeError: 'NoneType' object is not callable`` in every
child forked afterwards.  An at-fork hook cannot be unregistered, so the
child must not write about it.

The rule here: a process that has forked from one that called
:func:`install`, and has not called :func:`adopt_child`, discards every
unraisable exception instead of writing it.  Such a child is on its way
to ``exec`` (or to dying), so the message has no reader worth the risk.
A forked child that goes on running Python as a process of its own (a
pool slot) calls :func:`adopt_child`, and from then on is reported as
usual.

Whether a process is a fresh child is answered with two syscalls and no
lock: a ``before`` at-fork hook records the forking pid in the parent,
and the hook compares ``os.getpid()`` with it.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional

# The pid of the process that last forked (set in the parent, by the
# ``before`` hook), or of a child that adopted itself.  ``None`` until
# the first fork after :func:`install`.
_owner_pid: Optional[int] = None

_previous_hook: Optional[Callable[[Any], Any]] = None
_installed = False


def _note_fork() -> None:
    """``before`` at-fork hook: record which process is forking."""
    global _owner_pid
    _owner_pid = os.getpid()


def in_unadopted_child() -> bool:
    """True in a forked child that has not called :func:`adopt_child`."""
    owner = _owner_pid
    return owner is not None and os.getpid() != owner


def _quiet_unraisablehook(unraisable: Any) -> None:
    """Discard an unraisable exception in an unadopted child, else pass it on.

    Reads nothing but a module global and ``os.getpid()``, so it is safe
    between ``fork()`` and ``exec()``.  Never formats or writes in that
    window.
    """
    if in_unadopted_child():
        return
    previous = _previous_hook or sys.__unraisablehook__
    previous(unraisable)


def install() -> None:
    """Install the quiet hook for this process.  Idempotent.

    Call it before anything forks: the daemon's ``start()``, the runner's
    ``main()``, the telemetry plugin's ``initialize()``.  A hook another
    component installs later replaces this one; nothing in the tree does.
    """
    global _previous_hook, _installed
    if _installed:
        return
    _installed = True
    current = sys.unraisablehook
    if current is not _quiet_unraisablehook:
        _previous_hook = current
    sys.unraisablehook = _quiet_unraisablehook
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(before=_note_fork)


def adopt_child() -> None:
    """Mark this forked child as a process of its own.

    Called by a child that keeps running Python instead of exec'ing (a
    pool slot forked from the template), so its later unraisable
    exceptions are reported again.  Call it once the child no longer
    shares a lock with its parent's threads, i.e. right after the fork's
    own setup.
    """
    global _owner_pid
    _owner_pid = os.getpid()
