"""SELinux phase 5: ``jaato-selinux`` installs the module; refusals get a hint.

``jaato-selinux install`` does what every kernel handoff did by hand:
builds the packaged module source with the host's selinux-policy-devel,
loads it, labels this venv ``lib_t`` and relabels ``~/.jaato``.  The
daemon's readiness reason names it when the module is missing or old.

A ``cli`` command SELinux refused reads ``Permission denied``, as a mode
bit does.  The runner holds its child context and asks the kernel
(``security_compute_av``, granted to ``jaato_runner_t`` by module v5)
whether that context may execute or read the refused file's context.

No kernel: the commands and the libselinux calls are substituted.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jaato_server.server.confinement import selinux_install
from jaato_server.shared import confinement_grants as cg
from jaato_server.shared import lsm_label
from jaato_server.shared.tests.reversion import Reversion

_INSTALL = "jaato-server/jaato_server/server/confinement/selinux_install.py"
_GRANTS = "jaato-server/jaato_server/shared/confinement_grants.py"
_SELINUX = "jaato-server/jaato_server/server/confinement/selinux.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"
_LSM = "jaato-server/jaato_server/shared/lsm_label.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_LSM,
        find="        return ctypes.CDLL(LIBSELINUX_SONAME, use_errno=True)\n",
        replace="        import ctypes.util\n"
                "        return ctypes.CDLL(ctypes.util.find_library(\"selinux\") "
                "or LIBSELINUX_SONAME, use_errno=True)\n",
        test="test_libselinux_is_loaded_by_its_soname",
        because="find_library runs ldconfig and gcc in a confined process "
                "(phase 5 kernel run: the probe crashed, every hint paid an AVC)",
    ),
    Reversion(
        target=_GRANTS,
        find="        missing = _SHELL_NOT_FOUND.match(line)\n        if missing:\n",
        replace="        missing = None\n        if missing:\n",
        test="test_a_program_the_child_cannot_see_gets_a_hint",
        because="a program SELinux hides reads as 'command not found' and the "
                "model goes looking for it (phase 5 kernel run)",
    ),
    Reversion(
        target=_CLI,
        find="        if (\"Permission denied\" not in output and \"not found\" not in output\n",
        replace="        if (\"Permission denied\" not in output\n",
        test="test_the_cli_gate_lets_not_found_through",
        because="the hint would never be asked for a hidden program",
    ),
    Reversion(
        target=_INSTALL,
        find="        return 0 if line.startswith(\"ready\") else 1\n",
        replace="        return 0\n",
        test="test_status_exits_one_when_not_ready",
        because="a script could not test readiness by exit code",
    ),
    Reversion(
        target=_INSTALL,
        find="    verb = \"-m\" if _fcontext_defined(run, spec) else \"-a\"\n",
        replace="    verb = \"-a\"\n",
        test="test_an_existing_venv_rule_is_modified_not_added",
        because="a second install would fail on the rule the first one added",
    ),
    Reversion(
        target=_INSTALL,
        find="    return sys.prefix if sys.prefix != sys.base_prefix else None\n",
        replace="    return sys.prefix\n",
        test="test_a_system_interpreter_is_not_relabelled",
        because="install would label /usr lib_t",
    ),
    Reversion(
        target=_SELINUX,
        find="                f\"is unknown); {INSTALL_REMEDY}\")\n",
        replace="                f\"is unknown)\")\n",
        test="test_readiness_names_the_install_command",
        because="an operator would be told the module is missing, not how to get it",
    ),
    Reversion(
        target=_GRANTS,
        find="        return explain_selinux_denial(\n",
        replace="        return None\n        return explain_selinux_denial(\n",
        test="test_a_policy_refusal_gets_a_hint",
        because="an SELinux refusal would read as a missing mode bit",
    ),
    Reversion(
        target=_GRANTS,
        find="        if refused != SEE and not _dac_allows(resolved, refused):\n            continue\n",
        replace="",
        test="test_a_mode_bit_refusal_gets_no_hint",
        because="a missing execute bit would be blamed on SELinux",
    ),
    Reversion(
        target=_GRANTS,
        find="        if None not in verdicts and False in verdicts:\n",
        replace="        if False in verdicts or None in verdicts:\n",
        test="test_a_question_the_kernel_will_not_answer_gets_no_hint",
        because="a runner that may not ask the policy would claim a refusal",
    ),
    Reversion(
        target=_GRANTS,
        find="                            which=lambda n, path: _which_by_mode(n, path))\n",
        replace=")\n",
        test="test_a_program_the_runner_may_not_access_still_resolves",
        because="access(2) asks the policy too, so a refused program would "
                "not resolve and its refusal would get no hint",
    ),
    Reversion(
        target=_RUNNER,
        find="        descriptor.get(\"child_label\") if descriptor.get(\"backend\") == \"selinux\" else None)\n",
        replace="        None)\n",
        test="test_bootstrap_installs_the_child_context",
        because="no SELinux refusal would ever get a hint",
    ),
]


# ------------------------------------------------------------------ install


class _Run:
    """Records each command; answers ``semanage fcontext -l -C`` from *rules*."""

    def __init__(self, rules="", fail=None):
        self.calls, self.rules, self.fail = [], rules, fail

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if self.fail and argv[0] == self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        if argv[:2] == ["make", "-f"]:
            Path(kw["cwd"], "jaato.pp").write_text("pp")
        out = self.rules if argv[:4] == ["semanage", "fcontext", "-l", "-C"] else ""
        return subprocess.CompletedProcess(argv, 0, out, "")


@pytest.fixture
def devel(tmp_path, monkeypatch):
    mk = tmp_path / "Makefile"
    mk.write_text("")
    monkeypatch.setattr(selinux_install, "DEVEL_MAKEFILE", str(mk))
    monkeypatch.setattr(selinux_install, "venv_prefix", lambda: "/opt/venv")
    return mk


def test_install_builds_loads_and_labels(devel, tmp_path):
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    run = _Run()
    selinux_install.install([str(home)], run=run)
    verbs = [c[:2] for c in run.calls]
    assert verbs[0] == ["make", "-f"]
    assert ["semodule", "-i"] in verbs
    assert ["semanage", "fcontext", "-a", "-t", "lib_t", "/opt/venv(/.*)?"] in run.calls
    assert ["restorecon", "-R", "/opt/venv"] in run.calls
    assert ["restorecon", "-R", str(home / ".jaato")] in run.calls


def test_an_existing_venv_rule_is_modified_not_added(devel):
    run = _Run(rules="/opt/venv(/.*)?   all files   system_u:object_r:lib_t:s0\n")
    selinux_install.install([], run=run)
    assert ["semanage", "fcontext", "-m", "-t", "lib_t", "/opt/venv(/.*)?"] in run.calls


def test_a_system_interpreter_is_not_relabelled(tmp_path, monkeypatch):
    mk = tmp_path / "Makefile"
    mk.write_text("")
    monkeypatch.setattr(selinux_install, "DEVEL_MAKEFILE", str(mk))
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    run = _Run()
    done = selinux_install.install([], run=run)
    assert not any(c[:2] == ["semanage", "fcontext"] for c in run.calls)
    assert any("system interpreter" in line for line in done)


def test_no_devel_tree_is_refused(monkeypatch):
    monkeypatch.setattr(selinux_install, "DEVEL_MAKEFILE", "/nonexistent/Makefile")
    with pytest.raises(selinux_install.InstallError, match="selinux-policy-devel"):
        selinux_install.install([], run=_Run())


def test_a_failing_step_stops_and_is_named(devel):
    run = _Run(fail="semodule")
    with pytest.raises(selinux_install.InstallError, match="semodule -i"):
        selinux_install.install([], run=run)
    assert not any(c[0] == "restorecon" for c in run.calls)


def test_uninstall_removes_the_module_and_the_rule(monkeypatch):
    monkeypatch.setattr(selinux_install, "venv_prefix", lambda: "/opt/venv")
    run = _Run(rules="/opt/venv(/.*)?   all files   system_u:object_r:lib_t:s0\n")
    selinux_install.uninstall(run=run)
    assert ["semodule", "-r", "jaato"] in run.calls
    assert ["semanage", "fcontext", "-d", "/opt/venv(/.*)?"] in run.calls


def test_install_and_uninstall_need_root(monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert selinux_install.main(["install"]) == 2
    assert "run as root" in capsys.readouterr().err


def test_the_module_source_ships_in_the_package():
    names = {p.name for p in selinux_install.policy_source_dir().iterdir()}
    assert set(selinux_install.SOURCES) <= names
    toml = Path(selinux_install.__file__).resolve().parents[3].joinpath(
        "pyproject.toml").read_text()
    for name in selinux_install.SOURCES:
        assert f'"selinux_policy/{name}"' in toml
    assert 'jaato-selinux = "jaato_server.server.confinement.selinux_install:main"' in toml


def test_readiness_names_the_install_command():
    from jaato_server.server.confinement.selinux import _policy_problem

    class K:
        def context_valid(self, _):
            return False

    assert "jaato-selinux install" in _policy_problem(K())


# ------------------------------------------------------------------ hint

_CHILD = "unconfined_u:unconfined_r:jaato_child_t:s0:c1,c2"
_FILE = "unconfined_u:object_r:jaato_workspace_t:s0:c1,c2"


@pytest.fixture
def policy(monkeypatch, tmp_path):
    """A refused program, and a policy whose answers the test chooses."""
    prog = tmp_path / "tool"
    prog.write_text("#!/bin/sh\n")
    prog.chmod(0o755)
    answers = {"execute": False, "execute_no_trans": False, "read": True, "getattr": True}
    monkeypatch.setattr(lsm_label, "load_libselinux", lambda: object())
    monkeypatch.setattr(lsm_label, "selinux_file_context", lambda lib, p: _FILE)
    monkeypatch.setattr(lsm_label, "selinux_allowed",
                        lambda lib, s, t, c, perm: answers[perm])
    cg.set_confinement_grants(None)
    cg.set_selinux_child_context(_CHILD)
    yield prog, answers
    cg.set_selinux_child_context(None)


def _explain(prog):
    return cg.explain_denial(
        command=str(prog), output=f"sh: 1: {prog}: Permission denied",
        returncode=126, search_path=None, cwd=None)


def test_a_policy_refusal_gets_a_hint(policy):
    prog, _ = policy
    hint = _explain(prog)
    assert hint and "SELinux refused this" in hint
    assert "jaato_child_t" in hint and "jaato_workspace_t" in hint


def test_a_program_the_runner_may_not_access_still_resolves(policy, monkeypatch):
    prog, _ = policy
    monkeypatch.setattr(os, "access", lambda *a, **k: False)  # the policy, via access(2)
    hint = cg.explain_denial(
        command=prog.name, output=f"sh: 1: {prog.name}: Permission denied",
        returncode=126, search_path=str(prog.parent), cwd=None)
    assert hint and "SELinux refused this" in hint


def test_a_policy_that_allows_it_gets_no_hint(policy):
    prog, answers = policy
    answers.update(execute=True, execute_no_trans=True)
    assert _explain(prog) is None


def test_a_mode_bit_refusal_gets_no_hint(policy, monkeypatch):
    prog, answers = policy
    answers["read"] = False
    prog.chmod(0o600)  # another uid may neither execute nor read it
    monkeypatch.setattr(os, "geteuid", lambda: 4242)
    assert _explain(prog) is None


def test_a_question_the_kernel_will_not_answer_gets_no_hint(policy):
    prog, answers = policy
    answers.update(execute=None, execute_no_trans=None)
    assert _explain(prog) is None


def test_no_child_context_means_no_hint(policy):
    prog, _ = policy
    cg.set_selinux_child_context(None)
    assert _explain(prog) is None


def test_bootstrap_installs_the_child_context():
    from jaato_server.server.runner.session import _install_confinement_grants
    from jaato_server.shared.session_envelope import SessionInitEnvelope

    def env(confinement):
        return SessionInitEnvelope(session_id="s1", workspace_path="/w", profile_name="",
                                   model_name="m", provider_name="echo", plugins=[],
                                   confinement=confinement)

    _install_confinement_grants(env({"backend": "selinux", "label": "x",
                                     "child_label": _CHILD}))
    assert cg.selinux_child_context() == _CHILD
    _install_confinement_grants(env(None))
    assert cg.selinux_child_context() is None


# ------------------------------------------------------------------ kernel run 1 (1f78ee65)


def test_libselinux_is_loaded_by_its_soname(monkeypatch):
    import ctypes
    import ctypes.util

    asked = []
    monkeypatch.setattr(ctypes.util, "find_library",
                        lambda name: (_ for _ in ()).throw(AssertionError("find_library")))
    monkeypatch.setattr(ctypes, "CDLL", lambda name, **kw: asked.append(name) or object())
    assert lsm_label.load_libselinux() is not None
    assert asked == ["libselinux.so.1"]


def test_a_program_the_child_cannot_see_gets_a_hint(policy):
    prog, answers = policy
    answers["getattr"] = False
    hint = cg.explain_denial(
        command=prog.name, output=f"/bin/sh: line 1: {prog.name}: command not found",
        returncode=127, search_path=str(prog.parent), cwd=None)
    assert hint and "even see" in hint and "cannot see it" in hint


def test_the_apparmor_path_does_not_judge_a_missing_program(policy):
    prog, _ = policy
    cg.set_confinement_grants({"profile_name": "p", "exec_scope": "unscoped", "rules": []})
    try:
        assert cg.explain_denial(
            command=prog.name, output=f"sh: 1: {prog.name}: not found",
            returncode=127, search_path=str(prog.parent), cwd=None) is None
    finally:
        cg.set_confinement_grants(None)


def test_the_cli_gate_lets_not_found_through(monkeypatch):
    from jaato_server.shared.plugins.cli import plugin as cli

    seen = []
    monkeypatch.setattr(cli, "explain_denial", lambda **kw: seen.append(kw) or "HINT")
    p = cli.CLIToolPlugin()
    p._apparmor_child_transition = lambda: None
    monkeypatch.setattr(p, "_build_subprocess_env", lambda: ({"PATH": "/bin"}, None))
    out = p._with_denial_hint({"returncode": 0, "stderr": "sh: 1: tool: not found"},
                              {"command": "tool; echo RC=$?"})
    assert seen and out["denial_hint"] == "HINT"


def test_status_exits_one_when_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(selinux_install, "readiness_line", lambda: "not ready: x")
    assert selinux_install.main(["status"]) == 1
    monkeypatch.setattr(selinux_install, "readiness_line", lambda: "ready: y")
    assert selinux_install.main(["status"]) == 0


def test_a_home_without_a_user_tier_is_named(tmp_path):
    assert selinux_install.relabel_homes(_Run(), [str(tmp_path)]) == [
        f"skipped {tmp_path}: no ~/.jaato yet (relabel it with restorecon -R once it exists)"]
