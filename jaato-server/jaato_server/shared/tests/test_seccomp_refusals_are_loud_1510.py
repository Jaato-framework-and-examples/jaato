"""Seccomp postures and refused spawns are said where they are read (#1510).

Two defects from the #1503 verification runs:

1. The daemon log carried postures ``absent`` and ``off`` at INFO only; the
   WARNINGs were in the runner log, and a required-mode refusal had no ERROR
   anywhere the operator looks first.  The daemon now logs WARNING for
   ``absent`` / ``off`` and ERROR when every spawn will be refused, each
   naming the session.
2. A refused or failed spawn was reported as a SUCCESS: ``shell_spawn`` and
   ``notebook_execute`` under a required filter that was absent, and
   ``shell_spawn``'s ``out of pty devices``, all came back ``success: true``
   because the plugins returned a bare ``{"error": ...}`` dict, which
   ``ToolExecutor`` reads as success.  Every executor of ``cli``,
   ``interactive_shell`` and ``notebook`` now goes through
   ``runner_forwarding.failures_explicit``, so an error dict is
   ``(False, payload)`` (#1053's rule).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from jaato_server.shared import seccomp_filter as sf
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.tests.reversion import Reversion

_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_FWD = "jaato-server/jaato_server/shared/plugins/runner_forwarding.py"
_SHELL = "jaato-server/jaato_server/shared/plugins/interactive_shell/plugin.py"
_NB = "jaato-server/jaato_server/shared/plugins/notebook/plugin.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_SPAWN,
        find='    if kind == sf.POSTURE_ABSENT and posture.get("spawns_refused"):\n'
             "        logger.error(",
        replace='    if kind == sf.POSTURE_ABSENT and posture.get("spawns_refused"):\n'
                "        logger.info(",
        test="test_a_required_refusal_is_an_error_in_the_daemon_log",
        because="every spawn is refused and the daemon log says so at INFO",
    ),
    Reversion(
        target=_SPAWN,
        find="    elif kind == sf.POSTURE_ABSENT:\n        logger.warning(",
        replace="    elif kind == sf.POSTURE_ABSENT:\n        logger.info(",
        test="test_absent_and_off_are_warnings_in_the_daemon_log",
        because="a session without its syscall filter is INFO in the daemon log",
    ),
    Reversion(
        target=_FWD,
        find="                return (False, _with_spawn_refusal(result))\n",
        replace="                return result\n",
        test="test_out_of_pty_devices_is_a_failure",
        because="an error dict is reported as success: true",
    ),
    Reversion(
        target=_SHELL,
        find="        return failures_explicit(self.wrap_executors_for_runner_forwarding({\n"
             "            'shell_spawn'",
        replace="        return (lambda d: d)(self.wrap_executors_for_runner_forwarding({\n"
                "            'shell_spawn'",
        test="test_a_refused_shell_spawn_is_a_failure",
        because="interactive_shell's refused spawn is reported as success",
    ),
    Reversion(
        target=_NB,
        find="        return failures_explicit(self.wrap_executors_for_runner_forwarding({\n"
             '            "notebook_execute"',
        replace="        return (lambda d: d)(self.wrap_executors_for_runner_forwarding({\n"
                '            "notebook_execute"',
        test="test_a_refused_notebook_kernel_is_a_failure",
        because="a notebook cell whose kernel was refused is reported as success",
    ),
    Reversion(
        target=_CLI,
        find="        return failures_explicit(self.wrap_executors_for_runner_forwarding({\n"
             "            'cli_based_tool'",
        replace="        return (lambda d: d)(self.wrap_executors_for_runner_forwarding({\n"
                "            'cli_based_tool'",
        test="test_a_refused_cli_command_is_a_failure",
        because="a cli command whose spawn was refused is reported as success",
    ),
    Reversion(
        target=_FWD,
        find="        result = dict(result, refused_by=reason)\n",
        replace="        pass\n",
        test="test_a_refused_cli_command_is_a_failure",
        because="the model is told 'Exception occurred in preexec_fn.' and "
                "not why",
    ),
]


# ------------------------------------------------------------- daemon log


def _note(posture: Dict[str, Any], caplog) -> list:
    from jaato_server.server import runner_spawn

    server = SimpleNamespace(note_seccomp_posture=lambda _p: None)
    with caplog.at_level(logging.DEBUG, logger=runner_spawn.logger.name):
        runner_spawn._note_seccomp_posture(server, {"seccomp": posture}, "s-1510")
    return [r for r in caplog.records if r.name == runner_spawn.logger.name]


def test_a_required_refusal_is_an_error_in_the_daemon_log(caplog):
    records = _note({"posture": "absent", "reason": "libseccomp missing",
                     "required": True, "spawns_refused": True}, caplog)
    errors = [r for r in records if r.levelno == logging.ERROR]
    assert errors, [(r.levelname, r.getMessage()) for r in records]
    text = errors[0].getMessage()
    assert "s-1510" in text and "refused" in text and "libseccomp missing" in text


@pytest.mark.parametrize("posture", [
    {"posture": "absent", "reason": "the daemon shipped no compiled filter"},
    {"posture": "off", "reason": "runtime_limits.seccomp: off"},
])
def test_absent_and_off_are_warnings_in_the_daemon_log(caplog, posture):
    records = _note(posture, caplog)
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert warnings, [(r.levelname, r.getMessage()) for r in records]
    assert "s-1510" in warnings[0].getMessage()
    assert posture["posture"] in warnings[0].getMessage()


def test_a_filter_stays_info(caplog):
    records = _note({"posture": "filter", "libseccomp": "2.5.5"}, caplog)
    assert records and all(r.levelno == logging.INFO for r in records)


# ------------------------------------------------------------- results


def _refuser():
    """The //child preexec of a session whose REQUIRED filter is absent."""
    plan = sf.plan_for_session(None, None, boundary_active=True,
                               required=True, shipped=None)
    return sf.compose_child_preexec(lambda: None, plan.installer)


def _through_executor(plugin, tool: str, args: Dict[str, Any]):
    ex = ToolExecutor()
    ex._map[tool] = plugin.get_executors()[tool]
    return ex.execute(tool, args)


def test_out_of_pty_devices_is_a_failure(monkeypatch, tmp_path):
    from jaato_server.shared.plugins.interactive_shell import plugin as shell_mod

    def _no_pty(**_kw):
        raise OSError("out of pty devices")

    monkeypatch.setattr(shell_mod, "ShellSession", _no_pty)
    plugin = shell_mod.InteractiveShellPlugin()
    plugin._start_reaper = lambda: None
    plugin.initialize({"workspace_root": str(tmp_path)})
    ok, result = _through_executor(plugin, "shell_spawn", {"command": "true"})
    assert ok is False, result
    assert "out of pty devices" in str(result)


def test_a_refused_shell_spawn_is_a_failure(tmp_path):
    pytest.importorskip("pexpect")
    from jaato_server.shared.plugins.interactive_shell.plugin import (
        InteractiveShellPlugin)
    plugin = InteractiveShellPlugin()
    plugin._start_reaper = lambda: None
    plugin.initialize({"workspace_root": str(tmp_path)})
    plugin.set_apparmor_child_transition_callback(_refuser())
    try:
        ok, result = _through_executor(plugin, "shell_spawn",
                                       {"command": "true"})
        assert ok is False, result
    finally:
        for s in list(plugin._sessions.values()):
            s.close()


def test_a_refused_notebook_kernel_is_a_failure(tmp_path):
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(tmp_path),
                       "allow_uncontained_exec": True})
    plugin.set_apparmor_child_transition_callback(_refuser())
    try:
        ok, result = _through_executor(plugin, "notebook_execute",
                                       {"code": "print(1)"})
        assert ok is False, result
        assert "seccomp filter required" in str(result), result
    finally:
        plugin.shutdown()


def test_a_refused_cli_command_is_a_failure(tmp_path):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    plugin.set_apparmor_child_transition_callback(_refuser())
    try:
        ok, result = _through_executor(plugin, "cli_based_tool",
                                       {"command": "true"})
        assert ok is False, result
        # subprocess says only "Exception occurred in preexec_fn."; the
        # result names the cause so the model sees why.
        assert "seccomp filter required" in str(result), result
    finally:
        plugin.shutdown()


def test_a_null_error_is_not_a_failure():
    from jaato_server.shared.plugins.runner_forwarding import failures_explicit
    wrapped = failures_explicit({
        "a": lambda _a: {"error": None, "ok": 1},
        "b": lambda _a: (True, {"error": "x"}),
        "c": lambda _a: "text",
    })
    assert wrapped["a"]({}) == {"error": None, "ok": 1}
    assert wrapped["b"]({}) == (True, {"error": "x"})
    assert wrapped["c"]({}) == "text"
