"""A command the AppArmor profile refused says so, naming the path (#1348).

``Permission denied`` from a confined command reads the same whether a
mode bit or the session's ``//child`` profile refused it.  The daemon
records what each profile grants (#1326); the ``//child`` rule lines now
ride ``SessionInitEnvelope.confinement_grants`` to the runner, and ``cli``
checks a failure against them.  A hint is added only on positive
evidence: the command ran confined, the named path resolves to a file,
ordinary permissions allow the access, and the rules do not.

CI has no AppArmor kernel, so the refusal is simulated: the tests hand
``_with_denial_hint`` the result a refused command produces.  What they
pin is the judgement, not the kernel.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_server.shared import confinement_grants as cg
from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"
_CG = "jaato-server/jaato_server/shared/confinement_grants.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"
_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_CLI,
        find="        if self._apparmor_child_transition is None or not isinstance(result, dict):\n",
        replace="        if not isinstance(result, dict):\n",
        test="test_no_hint_when_the_command_did_not_run_confined",
        because="a command that never entered //child is blamed on a profile",
    ),
    Reversion(
        target=_CG,
        find="    if kind in (EXEC, EXEC_OR_READ) and os.access(resolved, os.X_OK):\n",
        replace="    if kind in (EXEC, EXEC_OR_READ):\n",
        test="test_no_hint_when_ordinary_permissions_refuse_it",
        because="a missing execute bit is reported as an AppArmor refusal",
    ),
    Reversion(
        target=_CG,
        find="        if parsed.unreadable:\n            return None\n",
        replace="",
        test="test_a_rule_it_cannot_read_withholds_the_hint",
        because="a rule the matcher cannot read is treated as granting nothing",
    ),
    Reversion(
        target=_CG,
        find="            if maybe_authorized(resolved):\n                return False\n",
        replace="            if maybe_authorized(resolved) and False:\n                return False\n",
        test="test_no_read_hint_for_a_path_a_reference_authorized",
        because="a read a reference granted after provisioning is reported refused",
    ),
    Reversion(
        target=_CG,
        find="        shell = _SHELL_DENIED.match(line)\n",
        replace="        shell = re.search(r\"(?P<path>/[^\\s:'\\\"]+): Permission denied\", line)\n",
        test="test_a_program_failing_to_read_its_data_is_not_judged",
        because="cat failing to read a file is reported as an exec refusal",
    ),
    Reversion(
        target=_CG,
        find="            if not rule.deny:\n                allowed = True\n",
        replace="            if not rule.deny:\n                pass\n",
        test="test_no_hint_when_the_profile_grants_it",
        because="a path the profile grants is reported refused",
    ),
    Reversion(
        target=_RUNNER,
        find="    installed = set_confinement_grants(wire if envelope.profile_name else None)\n",
        replace="    installed = set_confinement_grants(wire)\n",
        test="test_an_unconfined_session_installs_no_grants",
        because="an unconfined session judges refusals by a profile it never ran in",
    ),
    Reversion(
        target=_AA,
        find='            "child_rules": profile_body_rules(profile_content, "child"),\n',
        replace="",
        test="test_the_record_carries_the_child_rules_to_the_envelope",
        because="the envelope never carries the //child rules, so no hint is ever given",
    ),
    Reversion(
        target=_CLI,
        find="        return self._with_denial_hint(self._execute_sync(args), args)\n",
        replace="        return self._execute_sync(args)\n",
        test="test_the_synchronous_path_attaches_the_hint",
        because="the synchronous cli path never explains a refusal",
    ),
    Reversion(
        target=_CLI,
        find=(
            "        return self._with_denial_hint(\n"
            "            self._execute_streaming_run(args, on_stdout, on_stderr, on_returncode),\n"
            "            args,\n"
            "        )\n"
        ),
        replace="        return self._execute_streaming_run(args, on_stdout, on_stderr, on_returncode)\n",
        test="test_the_streaming_path_attaches_the_hint",
        because="the auto-background cli path never explains a refusal",
    ),
]

_PROFILE = "jaato-ws-test-1348"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_grants_left_behind():
    cg.set_confinement_grants(None)
    yield
    cg.set_confinement_grants(None)


def _executable(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(mode)
    return path


@pytest.fixture
def layout(tmp_path: Path) -> Dict[str, Path]:
    """A workspace, a directory the rules grant exec+read on, one they grant
    exec only on (the script case), and one they grant nothing on."""
    return {
        "ws": (tmp_path / "ws").resolve(),
        "granted": (tmp_path / "granted").resolve(),
        "exec_only": (tmp_path / "exec-only").resolve(),
        "outside": (tmp_path / "outside").resolve(),
    }


def _install(layout: Dict[str, Path], extra: List[str] = (), scope: str = "scoped"):
    rules = [
        f'"{layout["ws"]}/**" rwkl,',
        f'"{layout["granted"]}/**" ix,',
        f'"{layout["granted"]}/**" r,',
        f'"{layout["exec_only"]}/*" ix,',
        "network inet stream,",
        "deny ptrace,",
        f'include if exists "/etc/apparmor.d/jaato/{_PROFILE}.refs.d/*"',
        *extra,
    ]
    installed = cg.set_confinement_grants(
        {"profile_name": _PROFILE, "exec_scope": scope, "rules": rules}
    )
    assert installed is not None
    return installed


@pytest.fixture
def cli(layout):
    layout["ws"].mkdir(parents=True, exist_ok=True)
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(layout["ws"])})
    plugin.set_apparmor_child_transition_callback(lambda: None)
    yield plugin
    plugin.shutdown()


def _refused(path: Path, shell: str = "bash: line 1") -> Dict[str, Any]:
    return {"stdout": "", "stderr": f"{shell}: {path}: Permission denied\n", "returncode": 126}


# ---------------------------------------------------------------------------
# The positive cases
# ---------------------------------------------------------------------------


def test_an_exec_the_profile_does_not_grant_is_named(cli, layout):
    tool = _executable(layout["outside"] / "tool")
    _install(layout)
    result = cli._with_denial_hint(_refused(tool), {"command": str(tool)})
    hint = result["denial_hint"]
    assert str(tool) in hint and _PROFILE in hint and "exec" in hint
    assert "do not retry" in hint
    assert "apparmor_fragments" in hint  # scoped exec: say where grants go


def test_a_symlinked_command_is_named_by_what_it_resolves_to(cli, layout):
    """The #1342 shape: ``/usr/bin/ls`` -> a path the rules do not cover."""
    real = _executable(layout["outside"] / "coreutils" / "ls")
    link = layout["granted"] / "ls"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    _install(layout)
    result = cli._with_denial_hint(_refused(link), {"command": "ls"})
    assert f"{real} (what {link} resolves to)" in result["denial_hint"]


def test_a_bare_name_resolves_through_the_command_path(layout):
    tool = _executable(layout["outside"] / "mytool")
    layout["ws"].mkdir(parents=True, exist_ok=True)
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(layout["ws"]),
                       "extra_paths": [str(tool.parent)]})
    plugin.set_apparmor_child_transition_callback(lambda: None)
    try:
        _install(layout)
        result = plugin._with_denial_hint(
            _refused(Path("mytool"), shell="sh: 1"), {"command": "mytool --x"},
        )
    finally:
        plugin.shutdown()
    assert str(tool) in result["denial_hint"]


def test_a_script_its_interpreter_cannot_read_is_named(cli, layout):
    script = _executable(layout["exec_only"] / "which")
    _install(layout)
    # Reference selections can widen reads, so the registry is asked.
    cli._plugin_registry = SimpleNamespace(is_path_authorized=lambda path, mode="read": False)
    result = cli._with_denial_hint(
        {"stdout": "", "stderr": f"/bin/sh: 0: cannot open {script}: Permission denied\n",
         "returncode": 2},
        {"command": str(script)},
    )
    hint = result["denial_hint"]
    assert "does not grant read" in hint and "script" in hint


def test_no_read_hint_when_references_cannot_be_asked_about(cli, layout):
    script = _executable(layout["exec_only"] / "which")
    _install(layout)
    result = cli._with_denial_hint(
        {"stdout": "", "stderr": f"/bin/sh: 0: cannot open {script}: Permission denied\n",
         "returncode": 2},
        {"command": str(script)},
    )
    assert "denial_hint" not in result


def test_an_exec_errno_names_the_path(cli, layout):
    tool = _executable(layout["outside"] / "tool")
    _install(layout)
    result = cli._with_denial_hint(
        {"error": f"[Errno 13] Permission denied: '{tool}'"}, {"command": str(tool)},
    )
    assert str(tool) in result["denial_hint"]


# ---------------------------------------------------------------------------
# No hint without positive evidence
# ---------------------------------------------------------------------------


def test_no_hint_when_the_command_did_not_run_confined(layout):
    tool = _executable(layout["outside"] / "tool")
    layout["ws"].mkdir(parents=True, exist_ok=True)
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(layout["ws"])})
    try:
        _install(layout)
        result = plugin._with_denial_hint(_refused(tool), {"command": str(tool)})
    finally:
        plugin.shutdown()
    assert "denial_hint" not in result


def test_no_hint_without_a_record(cli, layout):
    tool = _executable(layout["outside"] / "tool")
    result = cli._with_denial_hint(_refused(tool), {"command": str(tool)})
    assert "denial_hint" not in result


def test_no_hint_when_ordinary_permissions_refuse_it(cli, layout, monkeypatch):
    """A file without the execute bit.  ``os.access`` is patched because
    root passes every access(2) check, and CI does not run as root."""
    tool = _executable(layout["outside"] / "tool", mode=0o644)
    real_access = os.access
    monkeypatch.setattr(
        cg.os, "access",
        lambda path, mode: False if mode == os.X_OK else real_access(path, mode),
    )
    _install(layout)
    result = cli._with_denial_hint(_refused(tool), {"command": str(tool)})
    assert "denial_hint" not in result


def test_no_hint_when_the_profile_grants_it(cli, layout):
    tool = _executable(layout["granted"] / "bin" / "tool")
    _install(layout)
    result = cli._with_denial_hint(_refused(tool), {"command": str(tool)})
    assert "denial_hint" not in result


def test_a_rule_it_cannot_read_withholds_the_hint(cli, layout):
    tool = _executable(layout["outside"] / "tool")
    _install(layout, extra=["some future rule kind we do not know,"])
    result = cli._with_denial_hint(_refused(tool), {"command": str(tool)})
    assert "denial_hint" not in result


def test_no_read_hint_for_a_path_a_reference_authorized(cli, layout):
    script = _executable(layout["exec_only"] / "which")
    _install(layout)
    cli._plugin_registry = SimpleNamespace(
        is_path_authorized=lambda path, mode="read": path == str(script),
    )
    result = cli._with_denial_hint(
        {"stdout": "", "stderr": f"/bin/sh: 0: cannot open {script}: Permission denied\n",
         "returncode": 2},
        {"command": str(script)},
    )
    assert "denial_hint" not in result


def test_a_program_failing_to_read_its_data_is_not_judged(cli, layout):
    """``cat: <path>: Permission denied`` is cat, not a shell failing to exec.

    The file is executable and outside every grant, so reading the line as
    an exec refusal WOULD produce a hint; only the shell-line rule stops it.
    """
    data = _executable(layout["outside"] / "data")
    _install(layout)
    result = cli._with_denial_hint(
        {"stdout": "", "stderr": f"cat: {data}: Permission denied\n", "returncode": 1},
        {"command": f"cat {data}"},
    )
    assert "denial_hint" not in result


def test_a_success_is_left_alone(cli, layout):
    _install(layout)
    result = {"stdout": "Permission denied", "stderr": "", "returncode": 0}
    assert cli._with_denial_hint(dict(result), {"command": "echo"}) == result


# ---------------------------------------------------------------------------
# Both cli paths
# ---------------------------------------------------------------------------


def test_the_synchronous_path_attaches_the_hint(cli, layout, monkeypatch):
    tool = _executable(layout["outside"] / "tool")
    _install(layout)
    monkeypatch.setattr(cli, "_execute_sync", lambda args: _refused(tool))
    assert "denial_hint" in cli._execute({"command": str(tool)})


def test_the_streaming_path_attaches_the_hint(cli, layout, monkeypatch):
    tool = _executable(layout["outside"] / "tool")
    _install(layout)
    monkeypatch.setattr(cli, "_execute_streaming_run", lambda *a: _refused(tool))
    result = cli._execute_streaming(
        {"command": str(tool)}, lambda b: None, lambda b: None, lambda c: None,
    )
    assert "denial_hint" in result


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("glob,path,expected", [
    ("/usr/bin/*", "/usr/bin/ls", True),
    ("/usr/bin/*", "/usr/bin/x/ls", False),
    ("/usr/bin/**", "/usr/bin/x/ls", True),
    ("/usr/{bin,sbin}/ls", "/usr/sbin/ls", True),
    ("/usr/{bin,sbin}/ls", "/usr/lib/ls", False),
    ("/usr/bin/l?", "/usr/bin/ls", True),
    ("/usr/bin/[lm]s", "/usr/bin/ms", True),
    ("@{HOME}/.local/bin/*", "/home/u/.local/bin/tool", True),
    ("@{HOME}/.local/bin/*", "/usr/local/bin/tool", False),
])
def test_apparmor_globs(glob, path, expected):
    pattern = cg.glob_to_regex(glob)
    assert pattern is not None
    assert bool(pattern.fullmatch(path)) is expected


def test_an_unconditional_deny_refuses_and_an_owner_deny_is_unknown():
    grants = cg.ConfinementGrants("p", "unscoped", ["/usr/bin/** ix,",
                                                    "audit deny /usr/bin/su x,",
                                                    "owner deny /usr/bin/sudo x,"])
    assert grants.verdict("/usr/bin/ls", "x") is True
    assert grants.verdict("/usr/bin/su", "x") is False
    assert grants.verdict("/usr/bin/sudo", "x") is None  # depends on the owner
    assert grants.verdict("/opt/x", "x") is False


# ---------------------------------------------------------------------------
# Daemon record -> envelope -> runner
# ---------------------------------------------------------------------------


def test_the_record_carries_the_child_rules_to_the_envelope(tmp_path):
    from jaato_server.server import apparmor
    from jaato_server.server.apparmor import AppArmorManager, envelope_grants

    ws = tmp_path / "ws"
    ws.mkdir()
    manager = AppArmorManager(workspace_root=str(tmp_path))
    text = manager._render_profile("sid", str(ws), requested_fragments=[])
    name = "jaato-ws-test-1348-record"
    manager._record_grants("sid", "sid", name, [], None, {}, text)
    try:
        wire = envelope_grants(name)
        assert wire is not None and wire["exec_scope"] == "scoped"
        assert wire["rules"] == cg.profile_body_rules(text, "child")
        # The diagnostics wire does not grow by a rendered body.
        assert "child_rules" not in (apparmor.recorded_grants(name) or {})
        # A scoped //child grants no broad exec: the #1342 path is refused.
        grants = cg.ConfinementGrants.from_wire(wire)
        assert grants.verdict("/usr/lib/cargo/bin/coreutils/ls", "x") is False
    finally:
        with apparmor._GRANT_RECORDS_LOCK:
            apparmor._GRANT_RECORDS.pop(name, None)


def test_the_envelope_round_trips_the_grants():
    wire = {"profile_name": _PROFILE, "exec_scope": "unscoped", "rules": ["/usr/bin/** ix,"]}
    env = SessionInitEnvelope(
        session_id="s", workspace_path=None, profile_name=_PROFILE,
        provider_name="echo", model_name="m", confinement_grants=wire,
    )
    assert SessionInitEnvelope.from_dict(env.to_dict()).confinement_grants == wire
    older = env.to_dict()
    older.pop("confinement_grants")
    assert SessionInitEnvelope.from_dict(older).confinement_grants is None


def test_an_unconfined_session_installs_no_grants():
    from jaato_server.server.runner.session import _install_confinement_grants

    wire = {"profile_name": _PROFILE, "exec_scope": "unscoped", "rules": ["/usr/bin/** ix,"]}
    _install_confinement_grants(SimpleNamespace(profile_name=_PROFILE, confinement_grants=wire))
    assert cg.confinement_grants() is not None
    # The next session on the same slot is unconfined: the grants go.
    _install_confinement_grants(SimpleNamespace(profile_name="", confinement_grants=wire))
    assert cg.confinement_grants() is None
