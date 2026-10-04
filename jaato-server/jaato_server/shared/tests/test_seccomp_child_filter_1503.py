"""Model-driven subprocesses run under a seccomp-bpf filter (#1503).

The LSM a confined session wears decides what a tool subprocess may TOUCH;
it does not decide which kernel entry points the payload may REACH.  The
runner therefore composes a deny-list filter into the ``//child`` preexec
step every ``cli`` / ``interactive_shell`` / notebook subprocess receives:
LSM transition, then ``PR_SET_NO_NEW_PRIVS`` and ``PR_SET_SECCOMP``, then
``exec``.

These tests install the real filter in real forked children.  They skip
where this kernel or libseccomp cannot build one (the posture tests still
run, with the compiler substituted).  No AppArmor or SELinux kernel is
needed: the LSM transition is replaced by a no-op, which is exactly the part
the filter does not depend on.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from jaato_server.shared import seccomp_filter as sf
from jaato_server.shared.tests.reversion import Reversion

_SF = "jaato-server/jaato_server/shared/seccomp_filter.py"
_CFG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_SER = "jaato-server/jaato_server/shared/plugins/session/serializer.py"
_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_SF,
        find="        lsm_transition()\n        installer()\n",
        replace="        lsm_transition()\n",
        test="test_a_cli_subprocess_runs_under_the_filter",
        because="the //child preexec no longer installs the filter",
    ),
    Reversion(
        target=_SF,
        find='    Family("keyring", ("keyctl", "add_key", "request_key"),',
        replace='    Family("keyring", ("add_key", "request_key"),',
        test="test_inside_cli_the_denied_families_answer_eperm",
        because="keyctl reaches the kernel keyring from a confined subprocess",
    ),
    Reversion(
        target=_SF,
        find="SCMP_ACT_KILL_PROCESS = 0x80000000",
        replace="SCMP_ACT_KILL_PROCESS = 0x7FFF0000",
        test="test_a_foreign_arch_syscall_is_killed",
        because="an i386 int 0x80 syscall bypasses the x86_64 deny-list",
    ),
    Reversion(
        target=_SF,
        find="        if fam.name in allow:\n            continue\n",
        replace="",
        test="test_an_allowed_back_family_reaches_the_kernel",
        because="seccomp_allow: [ptrace] leaves ptrace denied",
    ),
    Reversion(
        target=_SF,
        find="        if required:\n            logger.error(",
        replace="        if False:\n            logger.error(",
        test="test_libseccomp_missing_when_required_refuses_every_spawn",
        because="a required boundary runs subprocesses with no filter",
    ),
    Reversion(
        target=_CFG,
        find='        mode = "default" if "default" in modes else "off"',
        replace="        mode = modes[-1]",
        test="test_inheritance_is_most_restrictive_wins",
        because="a child's seccomp: off reopens a parent's filter",
    ),
    Reversion(
        target=_SPAWN,
        find="        _note_seccomp_posture(server, result)\n",
        replace="",
        test="test_the_posture_reaches_the_daemon_record",
        because="the daemon never records what the runner installed",
    ),
    Reversion(
        target=_SER,
        find="        'seccomp': state.seccomp,  # #1503, additive\n",
        replace="",
        test="test_the_posture_survives_the_session_record",
        because="the posture is dropped when the record is saved",
    ),
    Reversion(
        target=_AA,
        find='                "    /usr/lib/cargo/bin/**    ix,"',
        replace='                "    /usr/lib/cargo/bin/**    Px,"',
        test="test_child_has_no_exec_rule_nnp_would_refuse",
        because="a Px exec rule in //child is refused under NO_NEW_PRIVS",
    ),
]


def _filter_or_skip(allow=()) -> sf.CompiledFilter:
    try:
        return sf.compile_filter(allow)
    except sf.SeccompUnavailable as exc:
        pytest.skip(f"no seccomp filter can be built here: {exc}")


def _noop() -> None:
    """Stands in for the LSM ``//child`` transition."""


def _run(argv, pre) -> subprocess.CompletedProcess:
    return subprocess.run(argv, preexec_fn=pre, capture_output=True,
                          text=True, timeout=60)


# ------------------------------------------------------------- the runner


def _composed_via_runner(monkeypatch, limits: Optional[Dict[str, Any]] = None):
    """The callable the runner hands every subprocess plugin."""
    from jaato_server.server.runner import lsm_confine
    from jaato_server.server.runner import session as runner_session

    _filter_or_skip()
    monkeypatch.setattr(lsm_confine, "child_transition_callback",
                        lambda *a, **k: _noop)
    envelope = SimpleNamespace(runtime_limits=limits)
    confinement = SimpleNamespace(backend="apparmor", label="jaato-ws-x",
                                  child_label="")
    return runner_session._child_preexec(envelope, confinement)


@pytest.fixture
def cli(tmp_path: Path):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    yield plugin
    plugin.shutdown()


def _cli(cli, command: str) -> Dict[str, Any]:
    return cli._execute({"command": command})


def test_a_cli_subprocess_runs_under_the_filter(cli, monkeypatch, tmp_path):
    cli.set_apparmor_child_transition_callback(_composed_via_runner(monkeypatch))
    # A script in the workspace: cli containment refuses a /proc path on
    # the command line, and the subprocess reads its own status anyway.
    (tmp_path / "status.py").write_text(
        "print(open('/proc/self/status').read())\n")
    result = _cli(cli, "python3 status.py")
    assert result.get("returncode") == 0, result
    assert re.search(r"Seccomp:\s*2", result["stdout"]), result
    assert sf.current_posture()["posture"] == sf.POSTURE_FILTER


def test_the_notebook_kernel_runs_under_the_filter(monkeypatch, tmp_path):
    """The kernel receives the same callable cli does, so it gets the filter.

    ``allow_uncontained_exec`` keeps the audit-tier hook (no AppArmor here)
    out of the way of the probes; it is not what is being tested.
    """
    from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
        SubprocessKernelBackend)
    composed = _composed_via_runner(monkeypatch)
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path),
                   "allow_uncontained_exec": True})
    be.set_apparmor_child_transition(composed)
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "print([l.split()[1] for l in "
                       "open('/proc/self/status') if l.startswith('Seccomp:')])")
        text = "".join(str(getattr(o, "content", "")) for o in r.outputs or [])
        assert "['2']" in text, (r.status, text, r.error_message)
    finally:
        be.shutdown()


def test_an_interactive_shell_runs_under_the_filter(monkeypatch, tmp_path):
    pytest.importorskip("pexpect")
    from jaato_server.shared.plugins.interactive_shell.plugin import (
        InteractiveShellPlugin)
    composed = _composed_via_runner(monkeypatch)
    plugin = InteractiveShellPlugin()
    plugin._start_reaper = lambda: None
    plugin.initialize({"workspace_root": str(tmp_path)})
    plugin.set_apparmor_child_transition_callback(composed)
    try:
        result = plugin._exec_spawn({"command": "unshare -U true",
                                     "session_name": "s1503"})
        assert "Operation not permitted" in result.get("output", ""), result
    finally:
        for s in list(plugin._sessions.values()):
            s.close()


_PROBES = textwrap.dedent("""
    import ctypes, sys
    libc = ctypes.CDLL(None, use_errno=True)
    nr = {"bpf": 321, "keyctl": 250}[sys.argv[1]]
    if sys.argv[1] == "bpf":
        rc = libc.syscall(nr, 0, 0, 0)
    else:
        rc = libc.syscall(nr, 0, ctypes.c_long(-3), 0)  # GET_KEYRING_ID
    print(rc, ctypes.get_errno())
""")


@pytest.mark.skipif(platform.machine() != "x86_64",
                    reason="syscall numbers in the probe are x86_64's")
def test_inside_cli_the_denied_families_answer_eperm(cli, monkeypatch, tmp_path):
    cli.set_apparmor_child_transition_callback(_composed_via_runner(monkeypatch))
    (tmp_path / "probe.py").write_text(_PROBES)
    for name in ("keyctl", "bpf"):
        result = _cli(cli, f"python3 probe.py {name}")
        assert result.get("returncode") == 0, result
        assert result["stdout"].split() == ["-1", "1"], (name, result)
    result = _cli(cli, "unshare -U true")
    assert result.get("returncode") != 0, result
    assert "Operation not permitted" in (result.get("stderr", "")
                                         + result.get("stdout", "")), result


_I386 = textwrap.dedent("""
    import ctypes, mmap
    code = bytes([0xb8, 20, 0, 0, 0, 0xcd, 0x80, 0xc3])  # i386 getpid; ret
    m = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    m.write(code)
    fn = ctypes.CFUNCTYPE(ctypes.c_long)(ctypes.addressof(ctypes.c_char.from_buffer(m)))
    print("ran", fn(), flush=True)
""")


@pytest.mark.skipif(platform.machine() != "x86_64",
                    reason="the i386 int 0x80 probe is x86_64-only")
def test_a_foreign_arch_syscall_is_killed(tmp_path):
    flt = _filter_or_skip()
    script = tmp_path / "i386.py"
    script.write_text(_I386)
    control = _run([sys.executable, str(script)], None)
    if control.returncode != 0:
        pytest.skip(f"no IA-32 emulation on this kernel: {control.stderr[-200:]}")
    filtered = _run([sys.executable, str(script)], flt.install)
    assert filtered.returncode == -31, (filtered.returncode, filtered.stdout)
    assert "ran" not in filtered.stdout


_PTRACE = "import ctypes;l=ctypes.CDLL(None,use_errno=True);print(l.ptrace(0,0,0,0),ctypes.get_errno())"


def test_an_allowed_back_family_reaches_the_kernel():
    closed = _filter_or_skip()
    opened = _filter_or_skip(["ptrace"])
    assert opened.allowed == ("ptrace",)
    assert _run([sys.executable, "-c", _PTRACE], closed.install).stdout.split() == ["-1", "1"]
    assert _run([sys.executable, "-c", _PTRACE], opened.install).stdout.split()[0] == "0"


def test_clone3_answers_enosys_and_threads_still_start():
    flt = _filter_or_skip()
    code = ("import ctypes,threading,os;l=ctypes.CDLL(None,use_errno=True);"
            "print(l.syscall(435,0,0),ctypes.get_errno());"
            "t=threading.Thread(target=lambda:None);t.start();t.join();"
            "pid=os.fork();os._exit(0) if pid==0 else print(os.waitpid(pid,0)[1])")
    if platform.machine() != "x86_64":
        pytest.skip("clone3's number in the probe is x86_64's")
    out = _run([sys.executable, "-c", code], flt.install)
    assert out.stdout.split() == ["-1", "38", "0"], out


# ------------------------------------------------------------- posture


def _fake_compiler(_allow):
    return SimpleNamespace(install=_noop, program=b"\0" * 8, libseccomp="t")


def _missing(_allow):
    raise sf.SeccompUnavailable("libseccomp not found (tried libseccomp.so.2)")


def test_off_is_announced_and_installs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=sf.__name__):
        plan = sf.plan_for_session("off", None, boundary_active=True,
                                   required=False, compiler=_fake_compiler)
    assert plan.posture == sf.POSTURE_OFF and plan.installer is None
    assert any("'off'" in r.getMessage() for r in caplog.records)
    assert sf.compose_child_preexec(_noop, plan.installer) is _noop


def test_allow_back_is_announced_and_unknown_names_stay_denied(caplog):
    with caplog.at_level(logging.WARNING, logger=sf.__name__):
        plan = sf.plan_for_session("default", ["ptrace", "ptarce"],
                                   boundary_active=True, required=False,
                                   compiler=_fake_compiler)
    assert plan.posture == sf.POSTURE_FILTER
    assert plan.allowed == ("ptrace",)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ptarce" in text and "allowed back" in text


def test_an_unconfined_session_gets_no_filter_and_says_so():
    plan = sf.plan_for_session(None, None, boundary_active=False, required=False,
                               compiler=_fake_compiler)
    assert plan.posture == sf.POSTURE_UNCONFINED and plan.installer is None
    assert "no kernel boundary" in plan.reason


def test_libseccomp_missing_best_effort_warns_and_records_absent(caplog):
    with caplog.at_level(logging.WARNING, logger=sf.__name__):
        plan = sf.plan_for_session(None, None, boundary_active=True,
                                   required=False, compiler=_missing)
    assert plan.posture == sf.POSTURE_ABSENT and plan.installer is None
    assert sf.current_posture() == {"posture": "absent",
                                    "reason": plan.reason}
    assert any("no syscall filter" in r.getMessage() for r in caplog.records)


def test_libseccomp_missing_when_required_refuses_every_spawn():
    plan = sf.plan_for_session(None, None, boundary_active=True,
                               required=True, compiler=_missing)
    assert plan.posture == sf.POSTURE_ABSENT
    assert sf.current_posture()["spawns_refused"] is True
    with pytest.raises(subprocess.SubprocessError):
        subprocess.run(["true"], preexec_fn=sf.compose_child_preexec(
            _noop, plan.installer), check=False)


def test_a_missing_library_is_found_missing(monkeypatch):
    monkeypatch.setattr(sf, "LIBSECCOMP_NAMES", ("libseccomp-absent.so.9",))
    monkeypatch.setattr(sf.ctypes.util, "find_library", lambda _n: None)
    if not sf.kernel_supports_seccomp():
        pytest.skip("kernel has no seccomp")
    with pytest.raises(sf.SeccompUnavailable, match="libseccomp not found"):
        sf.compile_filter()


def test_required_reads_both_env_vars():
    assert sf.confinement_required({"JAATO_REQUIRE_CONFINEMENT": "1"})
    assert sf.confinement_required({"JAATO_REQUIRE_APPARMOR": "true"})
    assert not sf.confinement_required({})


def test_the_posture_reaches_the_runtime_aspect():
    from jaato_server.shared.plugins.environment import runtime
    sf.plan_for_session("default", ["ptrace"], boundary_active=True,
                        required=False, compiler=_fake_compiler)
    report = runtime.seccomp_report()
    assert report["posture"] == "filter"
    assert "ptrace" not in report["denied_families"]
    assert "bpf" in report["denied_families"]
    summary = runtime.runtime_summary({
        "confinement": {"tier": "unconfined"},
        "subprocess": {"cli": "not loaded"},
        "toolchains": {"status": "absent"},
        "seccomp": report,
    })
    assert summary["seccomp"] == "filter"


def test_the_posture_reaches_the_daemon_record(monkeypatch):
    from jaato_server.server import runner_spawn
    from jaato_server.server.session_manager import _seccomp_for_record

    posture = {"posture": "filter", "libseccomp": "2.5.5"}

    class _Rpc:
        def bootstrap_session_threadsafe(self, envelope, timeout):
            return {"ok": True, "ready": True, "seccomp": posture}

    class _Server:
        runner_rpc = _Rpc()
        seccomp_posture = None

        def note_runner_bootstrap_outcome(self, error):
            pass

        def mark_runner_ready(self):
            pass

        def note_seccomp_posture(self, value):
            self.seccomp_posture = dict(value) if isinstance(value, dict) else None

    monkeypatch.setattr(runner_spawn, "build_session_envelope",
                        lambda **kw: object())
    server = _Server()
    runner_spawn.dispatch_bootstrap_envelope(
        server=server, session_id="s1", workspace_path=None, profile_name="")
    assert server.seccomp_posture == posture
    session = SimpleNamespace(server=server, seccomp={"posture": "absent"})
    assert _seccomp_for_record(session) == posture
    session.server = None
    assert _seccomp_for_record(session) == {"posture": "absent"}


def test_the_posture_survives_the_session_record():
    from datetime import datetime
    from jaato_server.shared.plugins.session.base import SessionState
    from jaato_server.shared.plugins.session.serializer import (
        deserialize_session_state, serialize_session_state)
    now = datetime.now()
    state = SessionState(session_id="s", history=[], created_at=now,
                         updated_at=now, seccomp={"posture": "filter"})
    back = deserialize_session_state(serialize_session_state(state))
    assert back.seccomp == {"posture": "filter"}


def test_diagnostics_carry_the_posture():
    from jaato_server.server.diagnostics_verbs import _compose_result
    session = SimpleNamespace(sandbox_mode="apparmor", runner_identity=None,
                              seccomp={"posture": "absent"})
    server = SimpleNamespace(seccomp_posture={"posture": "filter"})
    result = _compose_result("r", session, {"probe": {"ok": True}}, server)
    assert result.seccomp == {"posture": "filter"}


# ------------------------------------------------------------- profiles


def _profile(name, inherits=None, **limits):
    from jaato_server.shared.plugins.subagent.config import SubagentProfile
    from jaato_server.shared.runtime_limits import RuntimeLimits
    return SubagentProfile(name=name, description="d",
                           runtime_limits=RuntimeLimits(**limits) if limits else None)


def test_inheritance_is_most_restrictive_wins():
    from jaato_server.shared.plugins.subagent.config import _merge_runtime_limits
    parent = _profile("p", seccomp="default", seccomp_allow=["ptrace", "perf"])
    child = _profile("c", seccomp="off", seccomp_allow=["ptrace", "bpf"])
    merged, conflicts = _merge_runtime_limits([parent], child)
    assert conflicts == []
    assert merged.seccomp == "default"
    assert merged.seccomp_allow == ("ptrace",)
    # A layer that declares nothing has no opinion.
    merged, _ = _merge_runtime_limits([_profile("p2")],
                                      _profile("c2", seccomp="off"))
    assert merged.seccomp == "off"
    # Two parents differing only here do not conflict.
    merged, conflicts = _merge_runtime_limits(
        [_profile("a", seccomp="off"), _profile("b", seccomp="default")],
        _profile("c3"))
    assert conflicts == [] and merged.seccomp == "default"


def test_the_mode_is_a_closed_vocabulary():
    from jaato_server.shared.runtime_limits import SECCOMP_MODES, RuntimeLimits
    assert SECCOMP_MODES == sf.MODES
    with pytest.raises(ValueError):
        RuntimeLimits(seccomp="strict")
    with pytest.raises(ValueError):
        RuntimeLimits(seccomp_allow="ptrace")
    assert RuntimeLimits.from_dict({"seccomp_allow": ["ptrace"]}).seccomp_allow == ("ptrace",)


def test_validate_reports_unknown_families_and_off():
    from jaato_server.shared.scaffold.validate import _check_seccomp
    found = []
    profile = _profile("p", seccomp="off", seccomp_allow=["ptrace", "ptarce"])
    _check_seccomp(profile, lambda sev, code, msg, **kw: found.append((sev, code)))
    assert ("error", "seccomp_unknown_family") in found
    assert ("warn", "seccomp_disabled") in found
    assert len([f for f in found if f[1] == "seccomp_unknown_family"]) == 1


def test_explain_runtime_names_the_enforcer():
    from jaato_server.shared.scaffold.explain import _runtime_limits_report
    report = _runtime_limits_report()
    rows = {r["name"]: r for r in report["fields"]}
    assert rows["seccomp"]["layer"] == sf.ENFORCER
    assert rows["seccomp_allow"]["layer"] == sf.ENFORCER
    assert {f["name"] for f in report["seccomp_families"]} == set(sf.FAMILIES)


# ------------------------------------------------------------- AppArmor + NNP


def test_child_has_no_exec_rule_nnp_would_refuse(tmp_path):
    """``//child`` may exec only with ``ix``: a ``Px``/``Cx``/``Ux`` rule
    names a profile transition, which the kernel refuses once the seccomp
    step has set NO_NEW_PRIVS, and every such command would fail."""
    from jaato_server.server.apparmor import AppArmorManager
    workspace = tmp_path / "w"
    workspace.mkdir()
    for fragments in (None, []):
        text = AppArmorManager(workspace_root=str(tmp_path))._render_profile(
            "sid", str(workspace), requested_fragments=fragments,
            plugin_rules=['"/usr/bin/python3" ix,'])
        child = text[text.index("profile child {"):]
        for line in child.splitlines():
            rule = line.split("#", 1)[0].strip().rstrip(",")
            mode = rule.rsplit(None, 1)[-1] if " " in rule else ""
            if "x" in mode.lower() and not mode.startswith(("/", '"')):
                assert not re.search(r"[pPcCuU]", mode), (fragments, line)


# ------------------------------------------------------------- cost


def test_the_install_costs_microseconds():
    """Recorded, not a race: two ``prctl`` calls, well under a millisecond."""
    flt = _filter_or_skip()
    rounds = 20
    t0 = time.perf_counter()
    for _ in range(rounds):
        _run(["true"], _noop)
    plain = (time.perf_counter() - t0) / rounds
    t0 = time.perf_counter()
    for _ in range(rounds):
        _run(["true"], flt.install)
    filtered = (time.perf_counter() - t0) / rounds
    # Generous: spawn jitter dominates; a regression to compiling in the
    # child (tens of ms) would still show.
    assert filtered - plain < 0.010, (plain, filtered)
