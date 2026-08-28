"""Start a real daemon, wait until it can actually answer, tear it down.

The fixture is the load-bearing part of a live suite: a flaky daemon start
produces failures indistinguishable from the defects the suite is looking for,
which is worse than no suite.  Three rules follow from that, and each is here
because the alternative fails silently:

* **Readiness is a successful CONNECT, never a sleep and never the socket
  file's existence.**  A stale socket file is present and dead — that state is
  the single thing ``jaato_sdk.doctor`` exists to detect — so a fixture that
  waits for the path would proceed against a corpse.
* **Every wait is bounded and says what it was waiting for.**  A hung daemon
  must fail the job in seconds with a named cause, not hold a CI runner until
  the platform kills it with nothing to read.
* **Teardown is unconditional and escalates.**  A leaked daemon holds its
  socket, and the next run inherits a process that is not the one it thinks it
  is testing — the failure mode that makes a suite lie about which code it
  exercised.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

#: Seconds to wait for a cold daemon to accept a connection.  Generous
#: because CI runners are slow and plugin discovery is real work; bounded
#: because the alternative to a bound is a job that hangs.
STARTUP_TIMEOUT = 90.0

#: Seconds between connect attempts while waiting.
POLL_INTERVAL = 0.25


class DaemonStartupError(RuntimeError):
    """The daemon did not become answerable.  Carries what was captured.

    The daemon's own output is attached because a startup failure with no
    log is the least actionable failure a CI job can produce -- the reader
    knows only that something did not happen.
    """


def _can_connect(socket_path: str) -> bool:
    """Does something ANSWER on this socket right now?

    Not ``os.path.exists``: a stale socket file is present and dead, and
    treating presence as readiness is how a suite ends up asserting against a
    daemon that is not running.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(socket_path)
        return True
    except (OSError, socket.timeout):
        return False


def echo_workspace(root: Path, *, usage: Optional[dict] = None,
                   tool_call: Optional[dict] = None,
                   response: Optional[str] = None,
                   completion_schema: Optional[dict] = None,
                   name: str = "conformance") -> Path:
    """Write a workspace with one echo-backed profile and return its path.

    ``usage`` is what makes budget invariants possible: echo reports the spend
    it is told to, identically every turn, so "how many turns to the ceiling"
    is arithmetic rather than observation.

    ``completion_schema`` matters more than it looks.  A profile carrying one
    ends its run by calling ``signal_completion``, which terminates the
    session INSIDE a tool-use turn -- and that terminus is the condition under
    which a consumer measured event delivery silently stopping.  A suite whose
    profiles all end in prose never reaches it and reports everything healthy;
    that is not hypothetical, it is how the first repro of that defect
    exonerated the daemon.
    """
    profiles = root / ".jaato" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)

    echo_cfg: dict = {}
    if usage is not None:
        echo_cfg["usage"] = usage
    if tool_call is not None:
        echo_cfg["tool_call"] = tool_call
    if response is not None:
        echo_cfg["response"] = response

    profile: dict = {
        "name": name,
        "description": "echo-backed profile for live conformance",
        "model": "echo",
        "provider": "echo",
        "plugins": [],
    }
    if echo_cfg:
        profile["plugin_configs"] = {"echo": echo_cfg}
    if completion_schema is not None:
        profile["completion_payload_schema"] = completion_schema

    (profiles / f"{name}.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8")
    return root


class ConformanceDaemon:
    """A daemon owned by the test run, or one the operator supplied.

    ``JAATO_CONFORMANCE_SOCKET`` points the suite at an already-running daemon
    -- the consumer-facing mode, where the question is "does MY deployment
    conform?" rather than "did we regress?".  In that mode nothing is started
    and nothing is torn down, because a suite that kills an operator's daemon
    to tidy up is worse than one that never ran.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._external = os.environ.get("JAATO_CONFORMANCE_SOCKET")
        self.socket_path: str = self._external or ""
        self._proc: Optional[subprocess.Popen] = None
        self._tmpdir: Optional[str] = None
        self._log: Optional[Path] = None
        self._out: Optional[Path] = None
        self._out_handle = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "ConformanceDaemon":
        if self._external:
            if not _can_connect(self._external):
                raise DaemonStartupError(
                    f"JAATO_CONFORMANCE_SOCKET={self._external} but nothing "
                    "answers there. The suite does not start a daemon in "
                    "external mode -- start yours, or unset the variable to "
                    "have the suite run its own."
                )
            return self

        self._tmpdir = tempfile.mkdtemp(prefix="jaato-conformance-")
        self.socket_path = os.path.join(self._tmpdir, "d.sock")
        self._log = Path(self._tmpdir) / "daemon.log"

        # NOT --daemon: the fixture owns this process and must be able to kill
        # it deterministically.  A forked daemon outlives a failed test run and
        # the next one inherits it.
        # --pid-file IS NOT OPTIONAL HERE.  Without it the daemon uses the
        # DEFAULT pidfile, sees any other jaato daemon on the machine, and
        # refuses to start -- "Jaato server is already running (PID ...)".
        # A unique socket is not enough: the running-instance check is keyed
        # on the pidfile, not the socket.  Found the first time this fixture
        # met a machine that already had a daemon on it, which in CI is a
        # previous job's leftover and in development is the operator's own.
        cmd = [sys.executable, "-m", "server",
               "--ipc-socket", self.socket_path,
               "--pid-file", os.path.join(self._tmpdir, "d.pid"),
               "--log-file", str(self._log)]
        # OUTPUT GOES TO A FILE, NOT A PIPE.  The daemon is chatty at startup
        # (plugin discovery, pool warm-up, extension load).  With
        # ``stdout=PIPE`` and nobody reading, it fills the 64KB pipe buffer
        # and BLOCKS -- so the fixture's readiness wait times out on a daemon
        # that is healthy and merely gagged, and reports it as a startup
        # failure.  A file has no buffer limit and is readable after the
        # process dies, which is exactly when the error path needs it.
        #
        # ``--log-file`` is NOT sufficient: the daemon's startup diagnostics
        # and its refusal messages go to stdout, and that file stayed empty
        # through the failure that motivated this.
        self._out = Path(self._tmpdir) / "daemon.out"
        self._out_handle = open(self._out, "wb")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace),
            stdout=self._out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,          # so teardown can kill the group
        )
        self._await_ready()
        return self

    def _await_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise DaemonStartupError(
                    f"daemon exited with code {self._proc.returncode} before "
                    f"accepting a connection.\n--- daemon output ---\n"
                    f"{self._captured()}"
                )
            if _can_connect(self.socket_path):
                return
            time.sleep(POLL_INTERVAL)
        raise DaemonStartupError(
            f"daemon did not accept a connection on {self.socket_path} within "
            f"{STARTUP_TIMEOUT}s (process still alive: it started but never "
            f"answered).\n--- daemon output ---\n{self._captured()}"
        )

    def _captured(self) -> str:
        """Everything the daemon said, from BOTH channels.

        The refusal that motivated this went to the process's STDOUT and
        never reached the log file -- so the first version, which flushed
        the pipe without reading it, reported "(no daemon output captured)"
        while the answer sat unread in the pipe.  A startup failure with no
        log is the least actionable failure a CI job can produce, which this
        method's whole purpose is to prevent, and it was not delivering it.

        Reads non-blockingly: the daemon may still be alive (the timeout
        path), and a blocking read on a live process's pipe would hang the
        error path -- turning a legible failure into the hang it is
        diagnosing.
        """
        parts = []
        if self._out is not None and self._out.exists():
            try:
                self._out_handle.flush()
            except Exception:
                pass
            text = self._out.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                # Tail, not head: a startup failure announces itself at the
                # END of the output, after however much discovery chatter.
                parts.append("[stdout] " + "\n".join(text.splitlines()[-40:]))
        if self._log is not None and self._log.exists():
            text = self._log.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append("[log] " + text)
        return "\n".join(parts) or "(no daemon output captured)"

    def stop(self) -> None:
        """Unconditional, escalating teardown.

        A leaked daemon holds its socket and the next run inherits a process
        that is not the one it thinks it is testing.
        """
        if self._external:
            return
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 15)     # TERM the group
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                    proc.wait(timeout=10)
                except Exception:
                    pass
            except (ProcessLookupError, PermissionError):
                pass
        if self._out_handle is not None:
            try:
                self._out_handle.close()
            except Exception:
                pass
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> "ConformanceDaemon":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
