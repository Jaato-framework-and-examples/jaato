"""#1275: a crashed subprocess kernel is detected, its death reason captured,
and the notebook recovered — instead of a raw ``BrokenPipeError`` surfacing on
every later cell forever.

No real kernel is needed for the two failure-shape assertions: a fabricated
``Popen``-like stand-in exercises the write path (the confinement precedent for
kernel-free tests).  The recovery assertions DO spawn a real kernel — that is
the point of "respawns and succeeds" — so they need the module importable in
the kernel subprocess (the suite is run with ``jaato_server`` on the kernel's
``PYTHONPATH``).
"""
import signal

import pytest

from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
    SubprocessKernelBackend, _Kernel,
)
from jaato_server.shared.plugins.notebook.types import (
    ExecutionStatus, NotebookInfo, OutputType,
)


def _text(result):
    return "".join(o.content for o in result.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


class _FakeStream:
    """A stdin/stdout stand-in.  ``write`` raises if armed to, so a dead pipe
    is reproduced without a kernel; ``close`` records that it was closed."""

    def __init__(self, write_raises=None):
        self._write_raises = write_raises
        self.closed = False

    def write(self, _data):
        if self._write_raises is not None:
            raise self._write_raises
        return len(_data)

    def flush(self):
        pass

    def read(self, _n=-1):
        return b""

    def fileno(self):
        return -1

    def close(self):
        self.closed = True


class _FakeProc:
    """A ``subprocess.Popen`` stand-in for a kernel that has died or is dying.

    ``poll`` returns the current ``returncode`` (``None`` = still running).
    ``wait`` reaps: if the process had not exited it is treated as SIGKILL'd,
    so a kernel found alive at the write but dead by the reap gets a signal in
    its reason — the realistic between-cells-death shape.
    """

    def __init__(self, returncode=None, reap_signal=signal.SIGKILL):
        self.returncode = returncode
        self._reap_signal = reap_signal
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -int(self._reap_signal)
        return self.returncode

    def kill(self):
        if self.returncode is None:
            self.returncode = -int(signal.SIGKILL)


def _fake_kernel(returncode=None, write_raises=None, stderr_text=b"",
                 reap_signal=signal.SIGKILL):
    info = NotebookInfo(notebook_id="nb", name="nb", backend="subprocess")
    proc = _FakeProc(returncode=returncode, reap_signal=reap_signal)
    kernel = _Kernel(info, proc, _FakeStream(write_raises=write_raises),
                     _FakeStream())
    if stderr_text:
        kernel._stderr_chunks.append(stderr_text)
    return kernel


# --- the reported failure: a raw BrokenPipeError becomes a clean typed error ---

def test_write_to_dead_kernel_returns_clean_typed_error_not_brokenpipe(tmp_path):
    # Kernel looked alive at the pre-check (poll -> None) but its stdin is a
    # broken pipe; the write is what discovers the death.  Old code let the
    # BrokenPipeError escape to ai_tool_runner; now execute returns a clean,
    # typed 'kernel crashed' result naming the exit signal.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    dead = _fake_kernel(returncode=None,
                        write_raises=BrokenPipeError(32, "Broken pipe"),
                        stderr_text=b"Fatal Python error: Segmentation fault\n")
    be._kernels["nb"] = dead

    result = be.execute("nb", "print(1 + 1)")

    assert result.status == ExecutionStatus.FAILED
    assert result.error_name == "KernelDied"
    # names the signal (SIGKILL from the reap) and carries the stderr tail —
    # a diagnosable reason, not a transport traceback.
    assert "signal 9" in result.error_message
    assert "SIGKILL" in result.error_message
    assert "Segmentation fault" in result.error_message


def test_dead_kernel_found_at_precheck_is_reaped_and_respawned(tmp_path):
    # A kernel that already exited (poll -> nonzero) is reaped and a fresh one
    # spawned, so this innocent cell runs and succeeds — the notebook is not
    # bricked.  Old code returned the dead kernel from _get_or_create and the
    # write raised BrokenPipeError.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    dead = _fake_kernel(returncode=-int(signal.SIGKILL),
                        stderr_text=b"boom\n")
    be._kernels["nb"] = dead
    try:
        result = be.execute("nb", "print(1 + 1)")
        assert result.status == ExecutionStatus.COMPLETED, result.error_message
        assert "2" in _text(result)
        # the fake was dropped and a real kernel took its place
        assert be._kernels["nb"] is not dead
    finally:
        be.shutdown()


def test_precheck_death_reason_is_logged_and_stderr_streams_closed(tmp_path):
    # The reap closes the dead kernel's pipes (no fd leak) and its stderr tail
    # reaches the death reason.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    dead = _fake_kernel(returncode=None, write_raises=BrokenPipeError(),
                        stderr_text=b"kernel harness traceback here")
    be._kernels["nb"] = dead
    result = be.execute("nb", "1 + 1")
    assert dead.wstream.closed and dead.rstream.closed
    assert "kernel harness traceback here" in result.error_message


# --- describe_death: signal vs code vs unknown ---

def test_describe_death_names_signal():
    k = _fake_kernel(returncode=-int(signal.SIGSEGV), stderr_text=b"segfault")
    reason = k.describe_death()
    assert "signal 11" in reason and "SIGSEGV" in reason
    assert reason.endswith("segfault")


def test_describe_death_names_exit_code():
    k = _fake_kernel(returncode=1, stderr_text=b"")
    reason = k.describe_death()
    assert "code 1" in reason


def test_describe_death_unknown_when_still_unreaped():
    k = _fake_kernel(returncode=None)
    assert "unresponsive" in k.describe_death()


def test_stderr_tail_is_bounded():
    k = _fake_kernel(returncode=1)
    # more than the char cap; only the tail is rendered
    k._stderr_chunks.clear()
    k._stderr_chunks.append(b"A" * 100 + b"TAIL_MARK")
    tail = k.stderr_tail_text(limit=20)
    assert tail.startswith("…")
    assert "TAIL_MARK" in tail
    assert len(tail) <= 21


# --- end-to-end with a REAL kernel: a mid-execution crash is not fatal ---

def test_real_kernel_crash_midexecution_then_next_cell_succeeds(tmp_path):
    # A cell that hard-exits the process crashes the kernel mid-execution;
    # execute returns a clean FAILED (not a raw EOF/BrokenPipe), and the NEXT
    # cell spawns a fresh kernel and succeeds — the notebook recovers.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        crash = be.execute(nb.notebook_id, "import os\nos._exit(1)")
        assert crash.status == ExecutionStatus.FAILED
        assert crash.error_name == "KernelDied"
        # the dead kernel was dropped
        assert nb.notebook_id not in be._kernels
        # a subsequent cell works on a fresh kernel — not bricked
        ok = be.execute(nb.notebook_id, "print(6 * 7)")
        assert ok.status == ExecutionStatus.COMPLETED, ok.error_message
        assert "42" in _text(ok)
    finally:
        be.shutdown()


def test_real_kernel_crash_by_signal_reports_signal(tmp_path):
    # A cell that sends itself SIGKILL: the reason names the signal, proving
    # the exit-status capture works against a real process death.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        crash = be.execute(
            nb.notebook_id,
            "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)")
        assert crash.status == ExecutionStatus.FAILED
        assert crash.error_name == "KernelDied"
        assert "signal 9" in crash.error_message
    finally:
        be.shutdown()
