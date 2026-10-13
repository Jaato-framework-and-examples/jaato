"""The checkout-skew check tells two venvs over one system Python apart.

WHAT WENT WRONG (#1610).  ``check_checkout_skew`` decided whether it could
resolve the daemon's installed ``jaato_sdk`` by comparing
``realpath(/proc/<pid>/exe)`` against ``realpath(sys.executable)``.  A venv's
``bin/python`` is a symlink to the base binary, so a client venv and a daemon
venv over the same system Python compared EQUAL, the "cannot resolve it from
here" branch never fired, and the installed-package fallback answered with the
CLIENT's own site-packages — printed as the daemon's, graded PASS.  Two venvs
over one system Python is the recommended client shape, so the one check that
could detect a client/daemon framework mismatch could not fail there.

WHAT THESE DO.  They build two REAL venvs over the interpreter running the
suite, start a real "daemon" process from one of them, and drive the real
``probe_daemon`` against it: the environment is read from the process, not
declared by the test.  The daemon prints its own ``sys.prefix``, which is the
ground truth the probe must reproduce.  Linux only — ``/proc`` is the probe.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jaato_sdk import doctor as D
from jaato_server.shared.tests.reversion import Reversion

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/cmdline").exists(), reason="needs /proc")

_DOCTOR = "jaato-sdk/jaato_sdk/doctor.py"

REVERSIONS = [
    Reversion(
        target=_DOCTOR,
        find="""        return os.path.realpath(info.prefix) == os.path.realpath(sys.prefix)""",
        replace="""        return True""",
        test="test_a_daemon_in_another_venv_on_the_same_python_is_not_a_PASS",
        because="two venvs over one binary read as one environment, the #1610 false PASS",
    ),
    Reversion(
        target=_DOCTOR,
        find="""        if same_env is not True and _resolve_package_on(entries, pkg) is None:""",
        replace="""        if same_env is False and _resolve_package_on(entries, pkg) is None:""",
        test="test_an_unidentifiable_environment_says_it_cannot_tell",
        because="'could not establish the environment' read as 'the same environment'",
    ),
    Reversion(
        target=_DOCTOR,
        find="""    here = os.path.dirname(interpreter)
    for d in (here, os.path.dirname(here)):""",
        replace="""    here = os.path.dirname(os.path.realpath(interpreter))
    for d in (here, os.path.dirname(here)):""",
        test="test_the_probe_reproduces_the_daemons_own_sys_prefix",
        because="pyvenv.cfg looked up after resolving the symlink, landing on the base Python",
    ),
    Reversion(
        target=_DOCTOR,
        find="""            info.prefix = _daemon_prefix(info.pid, env, info.executable)
""",
        replace="",
        test="test_a_daemon_in_our_own_venv_is_compared",
        because="the environment derived and never observed by probe_daemon",
    ),
    Reversion(
        target=_DOCTOR,
        find="""        if os.path.realpath(invoked) != os.path.realpath(executable):
            return None""",
        replace="""        pass""",
        test="test_an_argv0_naming_another_binary_is_not_believed",
        because="a rewritten argv[0] trusted as the interpreter the daemon runs",
    ),
]


# --------------------------------------------------------------------------
# Fixtures: real venvs, real processes
# --------------------------------------------------------------------------

_PRINT_PREFIX_AND_WAIT = (
    "import sys, time; sys.stdout.write(sys.prefix + '\\n'); "
    "sys.stdout.flush(); time.sleep(120)")


@pytest.fixture(scope="module")
def other_venv(tmp_path_factory):
    """A venv over the SAME base interpreter as the one running the suite."""
    root = tmp_path_factory.mktemp("venv1610") / "daemon-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)],
                   check=True, capture_output=True)
    return root


def _start(args, executable=None, env=None):
    """Start a fake daemon; return ``(proc, its own sys.prefix)``."""
    proc = subprocess.Popen(args, executable=executable, env=env,
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    return proc, line


@pytest.fixture
def daemons():
    started = []

    def start(*a, **kw):
        proc, prefix = _start(*a, **kw)
        started.append(proc)
        return proc, prefix

    yield start
    for p in started:
        p.kill()
        p.wait()


def _probe(monkeypatch, pid):
    """The REAL ``probe_daemon`` against ``pid`` — only the socket is faked."""
    monkeypatch.setattr(D, "_socket_listening", lambda path: True)
    monkeypatch.setattr(D, "_read_pid", lambda pidfile: pid)
    return D.probe_daemon("/tmp/x1610.sock", "/tmp/x1610.pid")


def _real(p):
    return os.path.realpath(p)


# --------------------------------------------------------------------------
# The issue's shape
# --------------------------------------------------------------------------

def test_the_premise_two_venvs_share_one_binary(other_venv):
    """Why the old discriminator was blind: both resolve to one binary.

    If this ever stops holding the other cases still mean something, but the
    #1610 shape is no longer being reproduced, so say so loudly.
    """
    assert _real(other_venv / "bin" / "python") == _real(sys.executable)
    assert _real(other_venv) != _real(sys.prefix)


def test_the_probe_reproduces_the_daemons_own_sys_prefix(monkeypatch, daemons,
                                                        other_venv):
    proc, truth = daemons([str(other_venv / "bin" / "python"), "-c",
                           _PRINT_PREFIX_AND_WAIT])
    info = _probe(monkeypatch, proc.pid)
    assert info.prefix and _real(info.prefix) == _real(truth), \
        f"probe said {info.prefix!r}, the daemon's own sys.prefix is {truth!r}"


def test_a_daemon_in_another_venv_on_the_same_python_is_not_a_PASS(
        monkeypatch, daemons, other_venv):
    """The #1610 false PASS: no daemon PYTHONPATH, another venv, one binary."""
    proc, _ = daemons([str(other_venv / "bin" / "python"), "-c",
                       _PRINT_PREFIX_AND_WAIT], env={"PATH": "/usr/bin:/bin"})
    info = _probe(monkeypatch, proc.pid)
    assert D._same_environment(info) is False
    checks = D.check_checkout_skew(info)
    assert checks and all(c.status == D.WARN for c in checks), \
        [(c.name, c.status, c.detail) for c in checks]
    assert all("cannot resolve" in c.detail for c in checks)
    assert all(str(other_venv) in c.detail for c in checks), \
        "the daemon's environment belongs in the line, so a reader can see it"


def test_a_bare_argv0_is_found_on_the_daemons_own_path(monkeypatch, daemons,
                                                      other_venv):
    """``python`` with the daemon venv's bin/ first on ITS PATH is that venv."""
    proc, truth = daemons(["python", "-c", _PRINT_PREFIX_AND_WAIT],
                          executable=str(other_venv / "bin" / "python"),
                          env={"PATH": f"{other_venv / 'bin'}:/usr/bin:/bin"})
    info = _probe(monkeypatch, proc.pid)
    assert info.prefix and _real(info.prefix) == _real(truth) == _real(other_venv)


# --------------------------------------------------------------------------
# The control, and the cases where the honest answer is "cannot tell"
# --------------------------------------------------------------------------

def test_a_daemon_in_our_own_venv_is_compared(monkeypatch, daemons):
    """Same environment: the installed-package comparison is legitimate."""
    proc, _ = daemons([sys.executable, "-c", _PRINT_PREFIX_AND_WAIT])
    info = _probe(monkeypatch, proc.pid)
    assert D._same_environment(info) is True
    monkeypatch.setattr(D, "_SKEW_PACKAGES", ("jaato_sdk",))
    monkeypatch.setattr(D, "_my_pythonpath", list)
    checks = D.check_checkout_skew(info)
    assert [c.status for c in checks] == [D.PASS], \
        [(c.name, c.status, c.detail) for c in checks]


def test_an_unidentifiable_environment_says_it_cannot_tell(monkeypatch,
                                                          one_package_pass):
    """No environment established is not the same environment."""
    info = D.DaemonInfo(socket_path="/tmp/x.sock", socket_exists=True,
                        listening=True, pid=1, pid_alive=True, env={},
                        executable=sys.executable, prefix=None)
    checks = D.check_checkout_skew(info)
    assert [c.status for c in checks] == [D.WARN], \
        [(c.name, c.status, c.detail) for c in checks]
    assert "could not be identified" in checks[0].detail


@pytest.fixture
def one_package_pass(monkeypatch):
    """One package, which would PASS if the check compared at all."""
    monkeypatch.setattr(D, "_SKEW_PACKAGES", ("jaato_sdk",))
    monkeypatch.setattr(D, "_my_pythonpath", list)


def test_an_argv0_naming_another_binary_is_not_believed(monkeypatch, daemons):
    """``argv[0]`` is the process's to rewrite; it must match ``exe``."""
    sh = "/bin/sh"
    if not os.path.exists(sh):
        pytest.skip("no /bin/sh")
    proc, _ = daemons([sh, "-c", _PRINT_PREFIX_AND_WAIT],
                      executable=sys.executable)
    info = _probe(monkeypatch, proc.pid)
    assert info.prefix is None


def test_a_relative_argv0_is_not_guessed():
    """Relative to a cwd the daemon may have left: decline, do not guess."""
    assert D._locate_invoked("venv/bin/python", {"PATH": "/usr/bin"}) is None


def test_virtual_env_alone_decides_nothing(tmp_path):
    """The interpreter ignores VIRTUAL_ENV, so the probe does too."""
    assert D._locate_invoked("python-not-there", {"VIRTUAL_ENV": str(tmp_path),
                                                  "PATH": ""}) is None
