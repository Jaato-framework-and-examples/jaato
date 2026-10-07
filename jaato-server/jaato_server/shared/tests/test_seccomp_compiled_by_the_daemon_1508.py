"""The seccomp filter is compiled by the daemon and only installed by the runner (#1508).

Under SELinux every confined session reported posture ``absent``:
``seccomp_filter._export`` exported the program through a memfd, which is
``tmpfs_t``, and ``jaato_runner_t`` may not write it.  On both LSMs
``ctypes.util.find_library("seccomp")`` execed ``ldconfig`` at every runner
start, which both refuse.

So the daemon, which is unconfined, compiles the program per session (it
depends on ``runtime_limits.seccomp`` / ``seccomp_allow``) and ships the raw
BPF on ``SessionInitEnvelope.seccomp_program`` with the architecture it was
compiled for.  The runner checks the architecture and the program's shape
and installs the bytes in the forked child; it never loads libseccomp, never
creates a memfd and never calls ``find_library``.  A missing, foreign or
malformed program is treated like no filter, under the existing posture
rules: ``absent`` with a WARNING, or every spawn refused when confinement is
required.

No SELinux kernel is needed: the runner half is driven with libseccomp, the
memfd and ``find_library`` all made to fail, which is the confined runner's
situation.
"""

from __future__ import annotations

import ast
import base64
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from jaato_server.shared import seccomp_filter as sf
from jaato_server.shared.tests.reversion import Reversion

_SF = "jaato-server/jaato_server/shared/seccomp_filter.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_SF,
        find="        compiled = load_shipped(shipped, allowed)\n",
        replace="        compiled = compile_filter(allowed)\n",
        test="test_the_runner_installs_without_libseccomp_or_a_memfd",
        because="the runner compiles again, through the memfd SELinux "
                "denies it: posture absent on every confined session",
    ),
    Reversion(
        target=_SF,
        find="    names = list(LIBSECCOMP_NAMES)\n    for name in names:\n",
        replace="    import ctypes.util\n    names = list(LIBSECCOMP_NAMES)\n"
                "    names.append(ctypes.util.find_library(\"seccomp\") or \"x\")\n"
                "    for name in names:\n",
        test="test_the_daemon_loads_libseccomp_by_soname",
        because="find_library execs ldconfig, which both LSMs refuse",
    ),
    Reversion(
        target=_SPAWN,
        find="        seccomp_program=seccomp_program_of(\n",
        replace="        _unused_seccomp_program=seccomp_program_of(\n",
        test="test_the_envelope_builder_ships_the_program",
        because="the envelope carries no program: every confined session "
                "is absent",
    ),
    Reversion(
        target=_RUNNER,
        find="        shipped=getattr(envelope, \"seccomp_program\", None),\n",
        replace="",
        test="test_the_runner_reads_the_program_off_the_envelope",
        because="the runner ignores the shipped program",
    ),
    Reversion(
        target=_SF,
        find="    if arch != native_arch():\n",
        replace="    if False:\n",
        test="test_a_program_for_another_arch_is_no_filter",
        because="a program whose arch check is foreign would kill every "
                "syscall of every subprocess",
    ),
    Reversion(
        target=_SF,
        find="    if (first.code != _BPF_LD_W_ABS",
        replace="    if False and (first.code != _BPF_LD_W_ABS",
        test="test_malformed_bytes_are_no_filter",
        because="arbitrary bytes are installed as the filter",
    ),
]


def _noop() -> None:
    """Stands in for the LSM ``//child`` transition."""


def _wire_or_skip(allow=()) -> Dict[str, Any]:
    wire = sf.compile_for_envelope(None, allow)
    if wire is None or wire.get("unavailable"):
        pytest.skip(f"no seccomp filter can be built here: {wire}")
    return wire


def _confined_runner(monkeypatch) -> None:
    """Make this process look like a confined runner to seccomp_filter.

    No libseccomp, no memfd, no ``find_library``: each raises if reached.
    """
    import ctypes.util
    import os

    def _forbidden(*_a, **_k):
        raise AssertionError("a confined runner may not do this (#1508)")

    def _no_libseccomp():
        raise sf.SeccompUnavailable("libseccomp is not loadable in the runner")

    monkeypatch.setattr(sf, "_load_libseccomp", _no_libseccomp)
    monkeypatch.setattr(os, "memfd_create", _forbidden)
    monkeypatch.setattr(ctypes.util, "find_library", _forbidden)


# ------------------------------------------------------------- the runner


def test_the_runner_installs_without_libseccomp_or_a_memfd(monkeypatch):
    wire = _wire_or_skip(["ptrace"])  # the daemon's half, before confining
    _confined_runner(monkeypatch)
    plan = sf.plan_for_session("default", ["ptrace"], boundary_active=True,
                               required=True, shipped=wire)
    assert plan.posture == sf.POSTURE_FILTER, plan
    assert plan.allowed == ("ptrace",)
    out = subprocess.run(
        [sys.executable, "-c",
         "print([l.split()[1] for l in open('/proc/self/status') "
         "if l.startswith('NoNewPrivs:')])"],
        preexec_fn=sf.compose_child_preexec(_noop, plan.installer),
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "['1']" in out.stdout, out


def test_the_runner_reads_the_program_off_the_envelope(monkeypatch):
    from jaato_server.server.runner import lsm_confine
    from jaato_server.server.runner import session as runner_session

    wire = _wire_or_skip()
    _confined_runner(monkeypatch)
    monkeypatch.setattr(lsm_confine, "child_transition_callback",
                        lambda *a, **k: _noop)
    runner_session._CHILD_PREEXEC_CACHE.clear()
    envelope = SimpleNamespace(runtime_limits=None, seccomp_program=wire)
    confinement = SimpleNamespace(backend="selinux", label="x_t", child_label="c_t")
    runner_session._child_preexec(envelope, confinement)
    assert sf.current_posture()["posture"] == sf.POSTURE_FILTER


def test_no_program_on_the_envelope_is_absent_or_refused(caplog):
    with caplog.at_level(logging.WARNING, logger=sf.__name__):
        plan = sf.plan_for_session(None, None, boundary_active=True,
                                   required=False, shipped=None)
    assert plan.posture == sf.POSTURE_ABSENT and plan.installer is None
    assert "shipped no compiled filter" in plan.reason
    plan = sf.plan_for_session(None, None, boundary_active=True,
                               required=True, shipped=None)
    assert sf.current_posture()["spawns_refused"] is True
    with pytest.raises(sf.SeccompRefused):
        plan.installer()


def test_the_daemons_reason_reaches_the_runners_posture():
    plan = sf.plan_for_session(
        None, None, boundary_active=True, required=False,
        shipped={"format": sf.WIRE_FORMAT, "unavailable": "kernel has no seccomp",
                 "allowed": []})
    assert plan.posture == sf.POSTURE_ABSENT
    assert "kernel has no seccomp" in plan.reason


def _refused_both_ways(shipped: Dict[str, Any], expect: str) -> None:
    plan = sf.plan_for_session(None, None, boundary_active=True,
                               required=False, shipped=shipped)
    assert plan.posture == sf.POSTURE_ABSENT and plan.installer is None, plan
    assert expect in plan.reason, plan.reason
    plan = sf.plan_for_session(None, None, boundary_active=True,
                               required=True, shipped=shipped)
    assert plan.posture == sf.POSTURE_ABSENT
    with pytest.raises(sf.SeccompRefused):
        plan.installer()


def test_a_program_for_another_arch_is_no_filter():
    wire = dict(_wire_or_skip(), arch="sparc64/64")
    _refused_both_ways(wire, "compiled for 'sparc64/64'")


def test_malformed_bytes_are_no_filter():
    wire = dict(_wire_or_skip(),
                program_b64=base64.b64encode(b"\x06\0\0\0\0\0\xff\x7f" * 2).decode())
    _refused_both_ways(wire, "architecture check")
    bad_b64 = dict(_wire_or_skip(), program_b64="not base64!!")
    _refused_both_ways(bad_b64, "malformed")
    short = dict(_wire_or_skip(), program_b64=base64.b64encode(b"\0" * 7).decode())
    _refused_both_ways(short, "instructions of 8")
    _refused_both_ways({"format": "other/9"}, "format")


def test_a_program_allowing_other_families_is_no_filter():
    wire = _wire_or_skip(["ptrace"])
    plan = sf.plan_for_session("default", [], boundary_active=True,
                               required=False, shipped=wire)
    assert plan.posture == sf.POSTURE_ABSENT
    assert "allows back" in plan.reason


def test_the_runner_module_never_searches_or_memfds_on_its_own_path():
    """Static half: only the daemon-side functions may name them."""
    tree = ast.parse(Path(sf.__file__).read_text())
    daemon_side = {"_export", "_load_libseccomp", "compile_filter",
                   "filter_pseudocode", "compile_for_envelope", "_build_ctx",
                   "_libseccomp_version"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in daemon_side:
            names = {n.attr if isinstance(n, ast.Attribute) else n.id
                     for n in ast.walk(node)
                     if isinstance(n, (ast.Attribute, ast.Name))}
            for banned in ("memfd_create", "find_library", "_load_libseccomp",
                           "_export", "compile_filter"):
                assert banned not in names, (node.name, banned)
    assert "find_library(" not in Path(sf.__file__).read_text()


# ------------------------------------------------------------- the daemon


def test_the_daemon_loads_libseccomp_by_soname(monkeypatch):
    import ctypes.util

    def _forbidden(*_a, **_k):
        raise AssertionError("find_library execs ldconfig (#1508)")

    monkeypatch.setattr(ctypes.util, "find_library", _forbidden)
    if not sf.kernel_supports_seccomp():
        pytest.skip("kernel has no seccomp")
    try:
        sf.compile_filter()
    except sf.SeccompUnavailable as exc:
        assert "find_library" not in str(exc), exc
        pytest.skip(f"no libseccomp here: {exc}")


def test_the_wire_names_its_arch_and_round_trips_the_envelope():
    from jaato_server.shared.session_envelope import SessionInitEnvelope
    wire = _wire_or_skip(["perf"])
    assert wire["arch"] == sf.native_arch()
    assert wire["allowed"] == ["perf"]
    assert isinstance(wire["audit_arch"], int)
    env = SessionInitEnvelope(session_id="s", workspace_path=None,
                              profile_name="jaato-ws-x", provider_name="p",
                              model_name="m", seccomp_program=wire)
    back = SessionInitEnvelope.from_dict(env.to_dict())
    assert back.seccomp_program == wire


def test_seccomp_off_ships_nothing_and_no_boundary_ships_nothing():
    from jaato_server.server.runner_spawn import seccomp_program_of
    from jaato_server.shared.runtime_limits import RuntimeLimits
    assert sf.compile_for_envelope("off", ["ptrace"]) is None
    assert seccomp_program_of(None, None) is None
    assert seccomp_program_of({"label": "jaato-ws-x//sub"}, None) is None
    assert seccomp_program_of({"label": "jaato-ws-x"},
                              RuntimeLimits(seccomp="off")) is None


def test_the_envelope_builder_ships_the_program():
    """``build_session_envelope`` passes the compiled program to the
    envelope.  Asserted on the call site: building a real envelope needs a
    resolved server, and the defect would be a missing keyword."""
    from jaato_server.server import runner_spawn
    tree = ast.parse(Path(runner_spawn.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_session_envelope")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "SessionInitEnvelope"]
    assert calls
    kw = {k.arg: k.value for c in calls for k in c.keywords}
    assert "seccomp_program" in kw
    assert "seccomp_program_of" in ast.unparse(kw["seccomp_program"])


def test_seccomp_program_of_compiles_for_a_confined_session():
    from jaato_server.server.runner_spawn import seccomp_program_of
    from jaato_server.shared.runtime_limits import RuntimeLimits
    _wire_or_skip()
    wire = seccomp_program_of({"label": "system_u:system_r:jaato_runner_t:s0"},
                              RuntimeLimits(seccomp_allow=["bpf"]))
    assert wire["allowed"] == ["bpf"] and wire["program_b64"]
