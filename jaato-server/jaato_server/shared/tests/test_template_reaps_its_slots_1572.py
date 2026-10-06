"""The pool template reaps the slots it forks (#1572).

A slot is forked by the runner template and torn down by the daemon,
which is not its parent, so nothing waited for it: every torn-down slot
stayed a zombie of the template.  The template now installs a
``SIGCHLD`` reaper when it enters its control loop, and a forked slot
restores ``SIG_DFL`` first thing so the reaper does not steal the
statuses of the slot's own subprocesses.

Every case runs in a forked child of the test process playing the
template, because the reaper takes ``waitpid(-1)``: run in the pytest
process itself it would reap children that belong to other tests.  The
template is driven through the real ``_template_control_loop`` and
``_handle_fork_slot``; only ``_run_slot_mode`` (the RPC server a slot
becomes) is replaced, by a probe that reports what the slot inherited.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
import traceback
from typing import Callable, List, Optional

import pytest

from jaato_server.shared.tests.reversion import Reversion

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or not os.path.isdir("/proc"),
    reason="needs fork() and /proc",
)

_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"
_REAPER = "jaato-server/jaato_server/server/runner/slot_reaper.py"

REVERSIONS = [
    Reversion(
        target=_MAIN,
        find="    install_slot_reaper()\n\n    while True:",
        replace="    pass\n\n    while True:",
        because="a template that never waits for its slots leaves each "
                "torn-down slot a zombie for the template's whole life",
        test="test_the_template_leaves_no_zombie_slot",
    ),
    Reversion(
        target=_MAIN,
        find="        reset_sigchld_in_child()\n",
        replace="        pass\n",
        because="a slot that keeps the template's SIGCHLD handler has its "
                "own subprocess statuses reaped out from under it",
        test="test_a_slot_starts_with_the_default_sigchld",
    ),
    Reversion(
        target=_REAPER,
        find="        reaped.append(pid)\n",
        replace="        reaped.append(pid)\n        return reaped\n",
        because="SIGCHLD is not queued: several slots exiting together "
                "raise one signal, so a reap that takes one child per "
                "signal leaves the rest as zombies",
        test="test_one_drain_reaps_every_exited_child",
    ),
]


# --------------------------------------------------------------- helpers


def _zombie_children() -> List[int]:
    """Pids of this process's children that are zombies, from /proc."""
    me = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        fields = data[data.rfind(b")") + 2:].split()
        if len(fields) >= 2 and fields[0] == b"Z" and int(fields[1]) == me:
            found.append(int(entry))
    return found


def _wait_for(predicate: Callable[[], bool], seconds: float = 5.0) -> bool:
    """Poll *predicate*; ``time.sleep`` lets a Python SIGCHLD handler run."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _in_child(body: Callable[[], Optional[str]]) -> Optional[str]:
    """Run *body* in a forked child; return its error string or ``None``.

    The child is the "template": whatever signal disposition and
    children it acquires die with it, never touching the test process.
    """
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(rfd)
        try:
            err = body()
        except BaseException:  # noqa: BLE001
            err = traceback.format_exc()
        os.write(wfd, (err or "").encode("utf-8", "replace"))
        os.close(wfd)
        os._exit(0)
    os.close(wfd)
    chunks = []
    while True:
        chunk = os.read(rfd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(rfd)
    os.waitpid(pid, 0)
    return b"".join(chunks).decode("utf-8", "replace") or None


def _drive_template_with_probe_slot() -> "tuple[str, List[int]]":
    """Run the real control loop for one FORK_SLOT, return what the slot saw.

    Returns ``(slot_report, zombies_after)``.  The slot's report is
    written to a pipe by the probe that replaces ``_run_slot_mode``.
    """
    from jaato_server.server.runner import __main__ as runner_main

    report_r, report_w = os.pipe()

    def probe(slot_fd: int, _log) -> None:
        os.close(report_r)
        os.close(slot_fd)
        handler = signal.getsignal(signal.SIGCHLD)
        msg = "default" if handler == signal.SIG_DFL else f"handler={handler!r}"
        os.write(report_w, msg.encode())
        os.close(report_w)
        os._exit(0)

    runner_main._run_slot_mode = probe
    daemon_end, template_end = socket.socketpair()
    slot_daemon, slot_child = socket.socketpair()
    daemon_end.sendmsg(
        [b"FORK_SLOT\n"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
          slot_child.fileno().to_bytes(4, sys.byteorder))],
    )
    slot_child.close()
    daemon_end.close()  # EOF after the command ends the loop
    import logging
    runner_main._template_control_loop(template_end, logging.getLogger("t"))
    os.close(report_w)
    report = os.read(report_r, 4096).decode()
    os.close(report_r)
    # The slot has exited (its pipe closed); give the reaper its chance.
    _wait_for(lambda: not _zombie_children())
    return report, _zombie_children()


# ----------------------------------------------------------------- tests


def test_the_template_leaves_no_zombie_slot():
    """A slot that exits is reaped by the template, not left a zombie."""
    def body() -> Optional[str]:
        _report, zombies = _drive_template_with_probe_slot()
        return f"zombie slot(s) left: {zombies}" if zombies else None

    assert _in_child(body) is None


def test_a_slot_starts_with_the_default_sigchld():
    """The forked slot restores SIG_DFL before anything else runs."""
    def body() -> Optional[str]:
        report, _zombies = _drive_template_with_probe_slot()
        return None if report == "default" else f"slot inherited {report}"

    assert _in_child(body) is None


def test_one_drain_reaps_every_exited_child():
    """Several children exited before the drain: one call takes them all."""
    from jaato_server.server.runner.slot_reaper import reap_exited_children

    def body() -> Optional[str]:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        pids = []
        for _ in range(5):
            pid = os.fork()
            if pid == 0:  # pragma: no cover
                os._exit(0)
            pids.append(pid)
        if not _wait_for(lambda: len(_zombie_children()) == 5):
            return f"setup: expected 5 zombies, saw {_zombie_children()}"
        reaped = reap_exited_children()
        if sorted(reaped) != sorted(pids):
            return f"reaped {reaped}, expected {pids}"
        left = _zombie_children()
        return f"zombies left: {left}" if left else None

    assert _in_child(body) is None


def test_a_drain_with_no_children_returns_nothing():
    """No children at all is ``[]``, not an exception."""
    from jaato_server.server.runner.slot_reaper import reap_exited_children

    assert _in_child(lambda: None if reap_exited_children() == []
                     else "expected []") is None
