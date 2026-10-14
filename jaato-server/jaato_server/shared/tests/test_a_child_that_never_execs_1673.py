"""A forked child that never reaches exec is bounded, and ours takes no lock (#1673).

A ``cli_based_tool`` call hung for good: the runner (a multi-threaded pool
slot) forked its child, and the child blocked in ``futex_wait`` before the
``//child`` transition, the seccomp filter and ``exec``.  ``subprocess.Popen``
returns only once the child has exec'd or failed (the parent reads the
child's exec-status pipe until it closes), so the parent's ``Popen`` blocked
too, before any timeout had started, and the turn never ended.

Two properties are guarded here, against the real code and with real forks:

1. **The post-fork chain takes no Python-level lock.**  The composed
   ``//child`` preexec (capability drop, LSM transition, seccomp filter,
   cgroup attach) runs in the child of a process whose other threads hold
   the locks that matter at fork time: a ``sys.stderr`` buffer lock and a
   logging handler lock (a thread blocked writing a log line into a full
   pipe, the incident's own signature), and a module import lock (a thread
   part-way through an import).  An audit hook records any symbol lookup
   (``ctypes.dlsym``), import or Python-level ``open`` the child makes.
   The chain must exec ``echo`` and record nothing.

2. **A child that never execs is bounded.**  A preexec step that never
   returns stands for any lock the child inherited; both ``cli`` spawn
   paths must kill the child at the exec deadline and report a failed
   call naming the cause, instead of blocking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from jaato_server.shared.tests.reversion import Reversion

_CD = "jaato-server/jaato_server/shared/capability_drop.py"
_CG = "jaato-server/jaato_server/server/cgroups.py"
_RUNNER = "jaato-server/jaato_server/shared/subprocess_runner.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_CD,
        find="            self._capset = self._libc.capset\n",
        replace=(
            "            self._capset = (\n"
            "                lambda *a: self._libc.__getitem__(\"capset\")(*a))\n"
        ),
        test="test_the_child_chain_execs_with_locks_held_and_takes_none",
        because="capset is looked up (dlsym) in the forked child",
    ),
    Reversion(
        target=_CG,
        find=(
            "    def _attach() -> None:\n"
            "        fd = os.open(path, flags, 0o644)\n"
            "        try:\n"
            "            os.write(fd, payload)\n"
            "        finally:\n"
            "            os.close(fd)\n"
        ),
        replace=(
            "    def _attach() -> None:\n"
            "        with open(path, \"w\") as f:\n"
            "            f.write(str(os.getpid()))\n"
        ),
        test="test_the_child_chain_execs_with_locks_held_and_takes_none",
        because="the cgroup attach builds a Python text stream in the child",
    ),
    Reversion(
        target=_RUNNER,
        find=(
            "    proc = popen_bounded(\n"
            "        cmd,\n"
            "        exec_deadline=exec_deadline,\n"
        ),
        replace=(
            "    proc = subprocess.Popen(\n"
            "        cmd,\n"
        ),
        test="test_the_sync_path_reports_a_child_that_never_execs",
        because="a child stuck before exec blocks Popen, and the call, forever",
    ),
    Reversion(
        target=_CLI,
        find=(
            "            proc = popen_bounded(\n"
            "                cmd,\n"
            "                exec_deadline=self._exec_deadline,\n"
        ),
        replace=(
            "            proc = subprocess.Popen(\n"
            "                cmd,\n"
        ),
        test="test_the_streaming_path_reports_a_child_that_never_execs",
        because="the streaming Popen is unbounded, so its tool_timeout never starts",
    ),
]


# ---------------------------------------------------------------------------
# 1. The post-fork chain takes no lock
# ---------------------------------------------------------------------------

#: Runs in its own interpreter: the audit hook it installs cannot be
#: removed, and the locks it holds must not be held in the test process.
_HELPER = textwrap.dedent(r'''
    import io, json, logging, os, struct, sys, threading, time

    events_path, work = sys.argv[1], sys.argv[2]
    sys.path.insert(0, work)

    from jaato_server.server.cgroups import procs_attach_callback
    from jaato_server.shared import capability_drop, seccomp_filter
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    marker = os.path.join(work, "transition")

    def transition():
        # Shaped like the AppArmor / SELinux callbacks: two syscalls on a
        # path and payload built in the parent.
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            os.write(fd, b"changeprofile x//child")
        finally:
            os.close(fd)

    caps = capability_drop.plan_for_session(None, boundary_active=True)
    chain = capability_drop.compose_child_preexec(transition, caps)
    installer = None
    if seccomp_filter.kernel_supports_seccomp():
        allow_all = struct.pack("HBBI", 0x06, 0, 0, 0x7FFF0000)
        installer = seccomp_filter.CompiledFilter(
            allowed=(), program=allow_all).install
    chain = seccomp_filter.compose_child_preexec(chain, installer)

    cli = CLIToolPlugin()
    cli.initialize({"workspace_root": work})
    cli.set_apparmor_child_transition_callback(chain)
    cli.set_runtime_limits(
        procs_attach_callback(os.path.join(work, "cgroup.procs")), None)
    cli._exec_deadline = 10.0

    # --- every lock below is held by another thread at fork time ---
    parent = os.getpid()
    ev_fd = os.open(events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    forbidden = {"import", "ctypes.dlsym", "ctypes.dlsym/handle", "ctypes.dlopen"}

    def audit(event, args):
        if os.getpid() == parent:
            return
        if event in forbidden or (event == "open" and len(args) > 1
                                  and args[1] is not None):
            os.write(ev_fd, (event + " " + repr(args)[:200] + "\n").encode())

    sys.addaudithook(audit)

    # A logging handler writing into a pipe nobody reads: the writing
    # thread blocks inside write() holding the handler lock and the
    # stream's buffer lock, as the runner's log did.
    r, w = os.pipe()
    stream = io.TextIOWrapper(io.FileIO(w, "w"), line_buffering=True)
    sys.stderr = stream
    log = logging.getLogger("holder")
    log.addHandler(logging.StreamHandler(stream))
    log.propagate = False
    threading.Thread(target=lambda: log.error("x" * (1 << 20)),
                     daemon=True).start()

    # A thread part-way through an import holds that module's lock.
    with open(os.path.join(work, "slow_import_1673.py"), "w") as f:
        f.write("import time\ntime.sleep(120)\n")
    threading.Thread(target=lambda: __import__("slow_import_1673"),
                     daemon=True).start()
    time.sleep(0.5)

    out = {"sync": cli._execute_sync({"command": "echo ok"})}
    out["sync"].pop("_telemetry", None)
    chunks = []
    out["stream"] = cli._execute_streaming_run(
        {"command": "echo ok"}, chunks.append, chunks.append,
        lambda rc: None)
    os.write(1, json.dumps(out).encode())
    os._exit(0)
''')


def _helper_env() -> Dict[str, str]:
    """Route the helper's imports to the tree this test was imported from."""
    import jaato_sdk
    import jaato_server

    env = dict(os.environ)
    roots = [str(Path(jaato_server.__file__).resolve().parents[1]),
             str(Path(jaato_sdk.__file__).resolve().parents[1])]
    env["PYTHONPATH"] = os.pathsep.join(
        roots + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    return env


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork semantics")
def test_the_child_chain_execs_with_locks_held_and_takes_none(tmp_path):
    events = tmp_path / "child_events"
    events.write_text("")
    script = tmp_path / "helper.py"
    script.write_text(_HELPER)
    proc = subprocess.run(
        [sys.executable, str(script), str(events), str(tmp_path)],
        capture_output=True, timeout=120, env=_helper_env(),
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]
    out = json.loads(proc.stdout)
    for path in ("sync", "stream"):
        result = out[path]
        assert result.get("returncode") == 0, (path, result)
        assert result.get("stdout", "").strip() == "ok", (path, result)
    # The chain really ran in the child: the transition stub and the
    # cgroup attach both wrote.
    assert (tmp_path / "transition").read_text() == "changeprofile x//child"
    assert (tmp_path / "cgroup.procs").read_text() == "0"
    assert events.read_text() == "", (
        "the forked child looked up a symbol, imported a module or built a "
        "Python file object between fork and exec:\n" + events.read_text())


# ---------------------------------------------------------------------------
# 2. A child that never execs is bounded
# ---------------------------------------------------------------------------


def _blocked_cli(tmp_path) -> Tuple[Any, int]:
    """A cli plugin whose child blocks before exec until *w* is closed."""
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    r, w = os.pipe()

    def stuck() -> None:
        os.read(r, 1)  # stands for a lock another thread held at fork

    cli = CLIToolPlugin()
    cli.initialize({"workspace_root": str(tmp_path)})
    cli.set_apparmor_child_transition_callback(stuck)
    cli._exec_deadline = 1.0
    cli._r_fd = r  # closed by the caller
    return cli, w


def _call_with_a_bound(fn, w: int, limit: float = 20.0) -> Tuple[Any, float]:
    """Run *fn* in a thread; a call still blocked after *limit* fails the test."""
    box: Dict[str, Any] = {}

    def run() -> None:
        box["result"] = fn()

    started = time.monotonic()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(limit)
    elapsed = time.monotonic() - started
    # Release a child still waiting either way: it holds its own copy of
    # *w* (fork keeps it open until exec), so closing ours is no EOF.
    try:
        os.write(w, b"x")
    finally:
        os.close(w)
    if t.is_alive():
        t.join(10)
        pytest.fail(f"the call was still blocked after {limit}s: a child "
                    "that never reaches exec hangs it")
    return box.get("result"), elapsed


def _assert_reported(result: Any, elapsed: float) -> None:
    assert isinstance(result, dict) and "error" in result, result
    assert "did not reach exec" in result["error"], result
    assert "returncode" not in result or result["returncode"] != 0
    assert elapsed < 15, elapsed


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork semantics")
def test_the_sync_path_reports_a_child_that_never_execs(tmp_path):
    cli, w = _blocked_cli(tmp_path)
    try:
        result, elapsed = _call_with_a_bound(
            lambda: cli._execute_sync({"command": "true"}), w)
    finally:
        os.close(cli._r_fd)
    _assert_reported(result, elapsed)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork semantics")
def test_the_streaming_path_reports_a_child_that_never_execs(tmp_path):
    cli, w = _blocked_cli(tmp_path)
    try:
        result, elapsed = _call_with_a_bound(
            lambda: cli._execute_streaming_run(
                {"command": "true"}, lambda b: None, lambda b: None,
                lambda rc: None),
            w)
    finally:
        os.close(cli._r_fd)
    _assert_reported(result, elapsed)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork semantics")
def test_a_healthy_spawn_is_not_killed():
    from jaato_server.shared.subprocess_runner import popen_bounded

    proc = popen_bounded(["/bin/sh", "-c", "sleep 1.5; echo done"],
                         exec_deadline=0.5, stdout=subprocess.PIPE)
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 0 and out.strip() == b"done"


def test_explain_runtime_names_steps_that_exist():
    """Every step ``explain runtime`` shows names code that still exists."""
    import importlib

    from jaato_server.shared import subprocess_runner as sr
    from jaato_server.shared.scaffold import explain

    for _step, where in sr.CHILD_PREEXEC_STEPS:
        if where.startswith("subprocess.Popen("):
            continue
        parts = where.split(".")
        for cut in range(len(parts), 0, -1):
            prefix = ".".join(parts[:cut])
            for base in ("jaato_server.shared.", "jaato_server."):
                try:
                    obj = importlib.import_module(base + prefix)
                    break
                except ImportError:
                    obj = None
            if obj is not None:
                for attr in parts[cut:]:
                    obj = getattr(obj, attr)
                break
        else:
            pytest.fail(f"{where} does not resolve")
    data, text = explain.runtime()
    assert data["child_spawn"]["exec_deadline_seconds"] == sr.EXEC_DEADLINE_SECONDS
    assert "CHILD SPAWN" in text and "//child" in text
