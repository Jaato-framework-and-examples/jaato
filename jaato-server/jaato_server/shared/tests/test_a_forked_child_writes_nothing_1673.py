"""A forked child writes nothing to stderr before exec (#1673).

A ``cli_based_tool`` call hung for good: the runner's forked child sat in
``futex_wait`` before ``//child`` and ``exec``.  The runner logged
``Exception ignored in atfork callback ... TypeError: 'NoneType' object
is not callable`` on every fork: an OpenTelemetry ``BatchSpanProcessor``
(one per session, shut down at ``session.end``) leaves an at-fork hook
that raises in every child once the processor is collected.  CPython
reports it by writing to ``sys.stderr`` IN THE CHILD, before
``preexec_fn``; if another runner thread held that stream's buffer lock
at fork time (the OTLP exporter logging its 401 every few seconds), the
child waits on it forever.

``jaato_server.shared.fork_quiet`` makes an unadopted forked child discard
unraisable reports.  The hang case reproduces the mechanism in a helper
interpreter: a thread blocked writing into a full stderr pipe holds the
buffer lock, an at-fork hook in the OTel <=1.40 shape raises, and
``subprocess.run(preexec_fn=...)`` must still return.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jaato_server.shared.tests.reversion import Reversion

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(os, "register_at_fork"),
    reason="needs fork() and os.register_at_fork",
)

_QUIET = "jaato-server/jaato_server/shared/fork_quiet.py"
_POSTURE = "jaato-server/jaato_server/server/process_posture.py"
_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"
_OTEL = "jaato-server/jaato_server/shared/plugins/telemetry/otel_plugin.py"

REVERSIONS = [
    Reversion(
        target=_QUIET,
        find="    if in_unadopted_child():\n        return\n",
        replace="    if in_unadopted_child():\n        pass\n",
        because="the child writes the at-fork TypeError to stderr and waits "
                "forever on a buffer lock another thread held at fork",
        test="test_a_raising_atfork_hook_does_not_hang_the_child",
    ),
    Reversion(
        target=_POSTURE,
        find="    fork_quiet.install()\n    apply_umask(",
        replace="    apply_umask(",
        because="the daemon (and so cold-spawned runners) forks with the "
                "default hook installed",
        test="test_every_entry_point_installs_it",
    ),
    Reversion(
        target=_MAIN,
        find="    fork_quiet.install()\n\n    # ----- 0. Mode dispatch",
        replace="    pass\n\n    # ----- 0. Mode dispatch",
        because="the template and every slot fork cli children with the "
                "default hook installed",
        test="test_every_entry_point_installs_it",
    ),
    Reversion(
        target=_MAIN,
        find="        adopt_child()\n        # Become a pool slot.",
        replace="        pass\n        # Become a pool slot.",
        because="a pool slot would discard its own unraisable reports for "
                "its whole life",
        test="test_a_pool_slot_adopts_itself",
    ),
]


def _run_helper(body: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run *body* in a fresh interpreter with this checkout importable."""
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


_HANG_HELPER = """
import gc, os, sys, threading, time, weakref, subprocess
from jaato_server.shared import fork_quiet
fork_quiet.install()

class _Processor:            # the OTel <=1.40 BatchProcessor at-fork shape
    def _at_fork_reinit(self):
        pass
p = _Processor()
weak_reinit = weakref.WeakMethod(p._at_fork_reinit)
os.register_at_fork(after_in_child=lambda: weak_reinit()())
del p
gc.collect()

out = os.dup(1)
r, w = os.pipe()             # stderr is a pipe nobody reads
os.dup2(w, 2)

def _writer():               # blocks mid-write, holding the buffer lock
    while True:
        sys.stderr.write("x" * 65536)
        sys.stderr.flush()
threading.Thread(target=_writer, daemon=True).start()
time.sleep(0.5)

result = []
def _spawn():
    subprocess.run(["true"], preexec_fn=lambda: None, timeout=10)
    result.append("returned")
t = threading.Thread(target=_spawn, daemon=True)
t.start()
t.join(5)
os.write(out, (result[0] if result else "HUNG").encode())
os._exit(0)
"""


def test_a_raising_atfork_hook_does_not_hang_the_child():
    proc = _run_helper(_HANG_HELPER)
    assert proc.stdout.strip() == "returned", (
        "a child forked while another thread held stderr's buffer lock "
        f"hung reporting a raising at-fork hook: {proc.stdout!r} {proc.stderr[-500:]!r}"
    )


def test_a_pool_slot_adopts_itself():
    """The template's slot fork marks the slot as a process of its own."""
    tree = ast.parse(Path(_repo_root(), _MAIN).read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_handle_fork_slot"
    )
    child = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and "child_pid == 0" in ast.unparse(n.test)
    )
    calls = [ast.unparse(c.func) for c in ast.walk(child) if isinstance(c, ast.Call)]
    assert "adopt_child" in calls, calls


def test_an_adopted_child_reports_again():
    """adopt_child() restores reporting; an unadopted child stays quiet."""
    body = """
    import os, sys
    from jaato_server.shared import fork_quiet
    fork_quiet.install()
    class _U:
        def __del__(self):
            raise RuntimeError("reported")
    def child(adopt):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.dup2(w, 2)
            if adopt:
                fork_quiet.adopt_child()
            _U()
            sys.stderr.flush()
            os._exit(0)
        os.close(w)
        os.waitpid(pid, 0)
        data = os.read(r, 65536)
        return b"reported" in data
    print(child(False), child(True))
    """
    proc = _run_helper(body)
    assert proc.stdout.split() == ["False", "True"], (proc.stdout, proc.stderr)


def test_the_parent_still_reports():
    body = """
    import sys, io
    from jaato_server.shared import fork_quiet
    fork_quiet.install()
    class _U:
        def __del__(self):
            raise RuntimeError("reported")
    _U()
    """
    proc = _run_helper(body)
    assert "reported" in proc.stderr


def test_every_entry_point_installs_it():
    """The daemon, the runner and the telemetry plugin each install it."""
    root = _repo_root()

    posture = ast.parse(Path(root, _POSTURE).read_text())
    fn = next(n for n in ast.walk(posture)
              if isinstance(n, ast.FunctionDef) and n.name == "apply_process_posture")
    assert "fork_quiet.install" in _calls(fn)

    main = ast.parse(Path(root, _MAIN).read_text())
    fn = next(n for n in ast.walk(main)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    first = [ast.unparse(s) for s in fn.body[:4]]
    assert any("fork_quiet.install()" in s for s in first), first

    otel = ast.parse(Path(root, _OTEL).read_text())
    fn = next(n for n in ast.walk(otel)
              if isinstance(n, ast.FunctionDef) and n.name == "initialize"
              and "BatchSpanProcessor" in ast.unparse(n))
    assert "fork_quiet.install" in _calls(fn)


def _calls(fn: ast.AST) -> list:
    return [ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)]


def _repo_root() -> Path:
    """The checkout root, found from the module under test (sandbox-safe)."""
    import jaato_server.shared.fork_quiet as m
    return Path(m.__file__).resolve().parents[3]
