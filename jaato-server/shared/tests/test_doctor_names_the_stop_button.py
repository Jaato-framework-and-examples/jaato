"""``jaato-doctor`` names the stop button for the daemon that is running.

``jaato-server --stop`` is the framework's whole-deployment stop
(Regulation (EU) 2024/1689, Art. 14(4)(e)), and its exact invocation
depends on how THIS daemon was started -- a non-default ``--pid-file`` or
``--ipc-socket`` is precisely what a person who did not start it does not
know.  So the doctor reads the daemon's own argv and prints the command
that will work, rather than a default the person has to match up
(``docs/design/eu-ai-act.md`` §4.5).

Lives in ``shared/tests`` for the reason ``test_doctor_detects_checkout_
skew_823.py`` does: the reversion meta-guard walks only this package.
"""

from __future__ import annotations

import ast
from pathlib import Path

from jaato_sdk import doctor
from jaato_sdk.doctor import PASS, WARN, DaemonInfo, check_oversight
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_DOCTOR = "jaato-sdk/jaato_sdk/doctor.py"

REVERSIONS = [
    Reversion(
        target=_DOCTOR,
        find='    pid_flag = _daemon_flag_value(argv, "--pid-file") or pidfile\n'
             '    sock_flag = _daemon_flag_value(argv, "--ipc-socket") or socket_path',
        replace='    pid_flag = pidfile\n    sock_flag = socket_path',
        because=(
            "the invocation must be the RUNNING daemon's, read off its argv; "
            "printing the doctor's own defaults names a command that stops "
            "nothing on a daemon started with other paths"
        ),
        test="test_the_invocation_is_read_from_the_daemons_own_argv",
    ),
]


def _info(listening: bool, pid=4242) -> DaemonInfo:
    return DaemonInfo(socket_path="/tmp/jaato.sock", socket_exists=listening,
                      listening=listening, pid=pid if listening else None,
                      pid_alive=listening or None)


def test_the_invocation_is_read_from_the_daemons_own_argv(monkeypatch):
    monkeypatch.setattr(doctor, "_daemon_cmdline", lambda pid: [
        "python", "-m", "server", "--ipc-socket", "/srv/j/j.sock",
        "--pid-file", "/srv/j/j.pid", "--daemon"])
    [check] = check_oversight(_info(True), "/tmp/jaato.sock", "/tmp/jaato.pid")
    assert check.status == PASS
    assert "--pid-file /srv/j/j.pid" in check.detail
    assert "--ipc-socket /srv/j/j.sock" in check.detail
    assert "PID 4242" in check.detail


def test_a_daemon_started_with_defaults_gets_the_doctors_paths(monkeypatch):
    monkeypatch.setattr(doctor, "_daemon_cmdline", lambda pid: ["python", "-m", "server"])
    [check] = check_oversight(_info(True), "/tmp/x.sock", "/tmp/x.pid")
    assert "--pid-file /tmp/x.pid --ipc-socket /tmp/x.sock" in check.detail


def test_no_daemon_is_a_warning_that_still_names_the_command():
    [check] = check_oversight(_info(False), "/tmp/x.sock", "/tmp/x.pid")
    assert check.status == WARN
    assert "jaato-server --stop" in check.detail
    assert "session stop <id>" in check.detail


def test_the_line_names_the_per_session_verb_and_the_orphan_listing(monkeypatch):
    monkeypatch.setattr(doctor, "_daemon_cmdline", lambda pid: None)
    [check] = check_oversight(_info(True), "/tmp/x.sock", "/tmp/x.pid")
    assert "session stop <id>" in check.detail
    assert "session orphans" in check.detail


def test_run_checks_calls_it():
    """Wired, not merely defined -- an AST look at the driver."""
    src = Path(doctor.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run_checks")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "check_oversight" in called
