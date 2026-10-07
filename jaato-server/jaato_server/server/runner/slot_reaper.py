"""The pool template reaps the slots it forks (#1572).

A pool slot is forked by the runner TEMPLATE
(``python -m jaato_server.server.runner --template-mode``), so the
template is its parent.  The daemon tears slots down (socket close, the
``RunnerRPCClient`` process-group sweep, ``PoolManager``'s teardowns),
but it is not their parent, so its ``waitpid(slot.pid)`` raises
``ChildProcessError`` and reaps nothing.  Before this module the
template never waited either, so every torn-down slot stayed a zombie
child of the template for the template's whole life: 43 in 17 minutes
on a live daemon.  Each one holds a process-table entry counted against
``RLIMIT_NPROC`` (eventually "can't fork"), and ``kill -0 <pid>`` kept
answering for a slot that was dead.

The template now reaps them itself:

- :func:`install_slot_reaper` installs a ``SIGCHLD`` handler in the
  template that drains every exited child with
  ``waitpid(-1, WNOHANG)``.  Python runs the handler on the main thread
  between bytecodes; the template's main thread spends its life in
  ``recvmsg`` on the control socket, which PEP 475 retries after the
  handler returns, so a reap never surfaces as an error to the control
  loop.  The handler starts no thread, so the template stays
  single-threaded at ``fork()``.
- :func:`reset_sigchld_in_child` restores ``SIG_DFL`` in a forked slot,
  first thing after ``fork()``.  A slot runs ``subprocess.run``,
  ``pexpect`` and notebook kernels, each of which waits for its own
  child; a slot that kept the template's handler would have those
  statuses stolen by the reap loop (``ChildProcessError`` /
  ``returncode`` lost).  A Python-level handler is not preserved across
  ``exec`` (the kernel resets caught signals), but the slot itself is
  a Python process that never execs, so the reset is required.

``SIG_IGN`` was the other option the issue named and is not used: it is
inherited across ``fork()`` AND preserved across ``exec``, and it makes
every ``waitpid`` in the inheriting process fail with ``ECHILD``, so a
single missed reset in a child would break every subprocess it, and
everything it execs, ever waits for.  A handler fails safe: a child that
somehow kept it loses statuses only in its own Python, never in what it
execs.

Rule: the template must never wait for a child of its own.  The reap
loop takes every exited child (``-1``), so a template-side
``subprocess.run`` started after :func:`install_slot_reaper` would race
it for its status.  The handler is installed only when the template
enters its control loop, after plugin discovery and the preload (the
only places the template could import something that starts a
process), and the control loop starts nothing but slots.

Nothing in the daemon depends on a slot zombie existing: slot liveness
is read off the RPC channel (``slot_rpc_death``, #1058), the slot's
process group is captured before teardown begins
(``RunnerRPCClient._capture_slot_pgid``), and the daemon's own
``waitpid(slot.pid)`` calls already tolerate ``ChildProcessError``.
After this change ``/proc/<pid>`` and ``kill -0`` stop answering for a
dead slot, which is what they were assumed to mean.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def reap_exited_children() -> List[int]:
    """Reap every exited child of this process, without blocking.

    Loops ``waitpid(-1, WNOHANG)`` until no exited child is left
    (``pid == 0``) or there are no children at all
    (``ChildProcessError``).  ``EINTR`` is retried by Python itself
    (PEP 475).  Returns the reaped pids, oldest first, for logging and
    for the guard test.
    """
    reaped: List[int] = []
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        except OSError as exc:  # pragma: no cover - defensive
            if exc.errno == errno.ECHILD:
                return reaped
            raise
        if pid == 0:
            return reaped
        reaped.append(pid)


def _on_sigchld(_signum: int, _frame: Optional[Any]) -> None:
    """``SIGCHLD`` handler: drain exited slots, log at DEBUG.

    Runs on the template's main thread between bytecodes, never in a
    C-level signal context, so logging here is safe.  Never raises: a
    handler that raised would surface inside ``recvmsg`` and end the
    control loop.
    """
    try:
        reaped = reap_exited_children()
    except Exception as exc:  # noqa: BLE001 - a handler must not raise
        logger.warning("runner template: slot reap failed: %s", exc)
        return
    if reaped:
        logger.debug("runner template: reaped exited slot(s) %s", reaped)


def install_slot_reaper() -> List[int]:
    """Install the template's ``SIGCHLD`` reaper and drain once.

    Called by the template when it enters its control loop.  The
    initial drain catches any child that exited before the handler was
    in place.  Returns the pids that drain reaped.
    """
    signal.signal(signal.SIGCHLD, _on_sigchld)
    return reap_exited_children()


def reset_sigchld_in_child() -> None:
    """Restore the default ``SIGCHLD`` disposition in a forked slot.

    Must be the first thing a slot does after ``fork()``, before any
    code that may start and wait for a subprocess.  See the module
    docstring for why a slot cannot keep the template's handler.
    """
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
