"""``jaato-doctor`` says which kernel confinement a daemon would use (design §9).

The check asks the daemon's own selection function, so it reports what a
daemon started now would pick, and FAILs exactly when that daemon would
refuse to start.  Under SELinux it also warns about the three host states
the kernel runs showed confine nothing or break a session quietly: a
permissive host or runner domain, and a ``~/.jaato`` that ``restorecon``
never relabelled.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk import doctor
from jaato_server.server import confinement
from jaato_server.server.confinement.selection import BackendChoice
from jaato_server.server.confinement.selinux import (
    REQUIRED_POLICY_VERSION, Readiness, SELinuxBackend,
)
from jaato_server.shared.tests.reversion import Reversion

_DOCTOR = "jaato-sdk/jaato_sdk/doctor.py"

REVERSIONS = [
    Reversion(
        target=_DOCTOR,
        find="    if choice.refuse:\n        return [Check(name, FAIL,\n",
        replace="    if choice.error:\n        return [Check(name, FAIL,\n",
        test="test_a_daemon_that_would_refuse_to_start_is_a_failure",
        because="JAATO_REQUIRE_CONFINEMENT with nothing available would read "
                "as a warning while the daemon refuses to start",
    ),
    Reversion(
        target=_DOCTOR,
        find='    if "permissive" in (facts["mode"], facts["runner_domain"]):\n',
        replace='    if facts["mode"] == "permissive":\n',
        test="test_a_permissive_runner_domain_is_a_warning",
        because="a host enforcing everything but jaato_runner_t would be "
                "reported as confining sessions",
    ),
    Reversion(
        target=_DOCTOR,
        find="            parsed is None or parsed.type != USER_DIR_TYPE):\n",
        replace="            parsed is None):\n",
        test="test_an_unlabelled_user_dir_is_a_warning",
        because="a ~/.jaato restorecon never relabelled would pass, and a "
                "confined runner could not reach it",
    ),
    Reversion(
        target=_DOCTOR,
        find="    checks += _guarded(lambda: check_confinement())\n",
        replace="",
        test="test_run_checks_runs_the_confinement_check",
        because="the check would exist and never run",
    ),
]


def _facts(**over):
    facts = {
        "mode": "enforcing", "runner_domain": "enforcing",
        "interpreter": "/opt/jaato/venv/bin/python3",
        "interpreter_label": "system_u:object_r:bin_t:s0",
        "user_dir": "/home/u/.jaato",
        "user_dir_label": "unconfined_u:object_r:jaato_user_dir_t:s0",
        "policy_version": "2",
    }
    facts.update(over)
    return facts


@pytest.fixture
def selinux_selected(monkeypatch):
    """Make the daemon's selection answer SELinux with the given facts."""
    def install(facts):
        backend = SimpleNamespace(host_facts=lambda home: facts)
        monkeypatch.setattr(
            confinement, "select_daemon_backend",
            lambda: BackendChoice("selinux", backend, "auto", False))
    return install


def _statuses(checks):
    return [c.status for c in checks]


def test_a_daemon_that_would_refuse_to_start_is_a_failure(monkeypatch):
    monkeypatch.setenv("JAATO_CONFINEMENT", "none")
    monkeypatch.setenv("JAATO_REQUIRE_CONFINEMENT", "1")
    [check] = doctor.check_confinement()
    assert check.status == doctor.FAIL
    assert "refuse to start" in check.detail


def test_an_unknown_backend_name_is_a_failure(monkeypatch):
    monkeypatch.setenv("JAATO_CONFINEMENT", "selinx")
    [check] = doctor.check_confinement()
    assert check.status == doctor.FAIL


def test_no_kernel_backend_is_a_warning(monkeypatch):
    monkeypatch.setenv("JAATO_CONFINEMENT", "none")
    monkeypatch.delenv("JAATO_REQUIRE_CONFINEMENT", raising=False)
    [check] = doctor.check_confinement()
    assert check.status == doctor.WARN
    assert "directory-sandbox" in check.detail


def test_a_ready_selinux_host_passes(selinux_selected):
    selinux_selected(_facts())
    checks = doctor.check_confinement()
    assert _statuses(checks) == [doctor.PASS]
    assert "selinux" in checks[0].detail and "enforcing" in checks[0].detail
    assert "policy module v2" in checks[0].detail


def test_a_permissive_runner_domain_is_a_warning(selinux_selected):
    selinux_selected(_facts(runner_domain="permissive"))
    assert _statuses(doctor.check_confinement()) == [doctor.PASS, doctor.WARN]


def test_a_permissive_host_is_a_warning(selinux_selected):
    selinux_selected(_facts(mode="permissive"))
    assert _statuses(doctor.check_confinement()) == [doctor.PASS, doctor.WARN]


def test_an_unlabelled_user_dir_is_a_warning(selinux_selected):
    selinux_selected(_facts(user_dir_label="unconfined_u:object_r:user_home_t:s0"))
    checks = doctor.check_confinement()
    assert _statuses(checks) == [doctor.PASS, doctor.WARN]
    assert "restorecon -Rv /home/u/.jaato" in checks[1].detail


def test_a_missing_user_dir_is_not_a_warning(selinux_selected):
    selinux_selected(_facts(user_dir_label=None))
    assert _statuses(doctor.check_confinement()) == [doctor.PASS]


def test_run_checks_runs_the_confinement_check():
    tree = ast.parse(Path(doctor.__file__).read_text())
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run_checks")
    called = {n.func.id for n in ast.walk(run)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "check_confinement" in called


class _Kernel:
    def file_context(self, path):
        return "system_u:object_r:bin_t:s0"

    def link_context(self, path):
        return "unconfined_u:object_r:jaato_user_dir_t:s0"


def test_host_facts_reads_the_kernel(tmp_path):
    (tmp_path / ".jaato").mkdir()
    backend = SELinuxBackend(kernel_factory=_Kernel,
                             interpreter=lambda: "/usr/bin/python3",
                             domain_permissive=lambda ctx: True)
    backend._readiness = Readiness(True, None, True)
    facts = backend.host_facts(str(tmp_path))
    assert facts == {
        "mode": "enforcing", "runner_domain": "permissive",
        "interpreter": "/usr/bin/python3",
        "interpreter_label": "system_u:object_r:bin_t:s0",
        "user_dir": str(tmp_path / ".jaato"),
        "user_dir_label": "unconfined_u:object_r:jaato_user_dir_t:s0",
        "policy_version": str(REQUIRED_POLICY_VERSION),
    }


def test_host_facts_has_no_label_for_a_missing_user_dir(tmp_path):
    backend = SELinuxBackend(kernel_factory=_Kernel, domain_permissive=lambda ctx: None)
    backend._readiness = Readiness(True, None, False)
    facts = backend.host_facts(str(tmp_path))
    assert facts["user_dir_label"] is None
    assert facts["mode"] == "permissive" and facts["runner_domain"] is None
