"""On a confined host, only ``proposeReference`` can write a reference claim.

A claim's ``origin.witnessed_by`` says whether a person approved the
``proposeReference`` call at the permission prompt, and the curator's
Proposals panel shows it.  The claim file is written into
``<ws>/.jaato/references-claims/``, so that record is only as good as the
rule deciding who else may write there.  Two routes were open:

1. **The kernel.**  ``//child`` -- the profile every subprocess the model
   drives runs in (``cli``, ``interactive_shell``, a notebook kernel) --
   granted writes there: the workspace-wide ``rwkl`` covered it, and the
   references plugin's contributed grant lands in every body.  So
   ``python -c``, a script, or a notebook cell could write a claim with
   any ``witnessed_by``.  Template v43 write-denies the directory in
   ``//child`` only; base keeps the grant, because ``proposeReference``
   runs in-process there.
2. **``cli``'s own check.**  ``path_like`` examined only tokens starting
   with ``/``, ``./``, ``~`` or containing ``..``, so a bare
   ``.jaato/...`` was never run through the ``.jaato`` rule that
   ``file_edit`` and ``readFile`` already apply: ``echo > .jaato/x`` and
   ``cat .jaato/profiles/p.yaml`` passed.  A bare ``.jaato`` token is now a
   path, and the refusal says the directory is closed, not that it is
   "outside the workspace".

Stated limits: the flat isolated sub-runner profile keeps the grant (its
subprocesses and its in-process tools share one body), and an unconfined
host has no kernel boundary, so a script can still write a claim there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.confinement_grants import ConfinementGrants, profile_body_rules
from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.tests.reversion import Reversion

_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"
_CONTAINMENT = "jaato-server/jaato_server/shared/plugins/command_containment.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_APPARMOR,
        find='    audit deny "{workspace_path}/.jaato/references-claims/" wlk,\n'
             '    audit deny "{workspace_path}/.jaato/references-claims/**" wlk,\n',
        replace="",
        because=(
            "a subprocess the model drives could write a claim by hand and "
            "forge the witness the curator is shown"
        ),
        test="TestTheKernel::test_child_cannot_write_a_claim",
    ),
    Reversion(
        target=_CONTAINMENT,
        find="            token.startswith('./') or token.startswith('~') or\n"
             "            names_jaato_dir(token))\n",
        replace="            token.startswith('./') or token.startswith('~'))\n",
        because="a bare .jaato/... token would skip the .jaato rule again",
        test="TestCli::test_a_bare_relative_claim_write_is_refused",
    ),
    Reversion(
        target=_CLI,
        find="        elif boundary and is_jaato_path(os.path.join(boundary, path), boundary):\n",
        replace="        elif False:\n",
        because="the refusal would tell the model a .jaato path is outside the workspace",
        test="TestCli::test_the_refusal_names_the_jaato_rule",
    ),
]

WS = "/workspace"
CLAIM = f"{WS}/.jaato/references-claims/c.json"


def _bodies(tmp_path) -> str:
    (tmp_path / "w" / "sessions").mkdir(parents=True)
    (tmp_path / "p").mkdir()
    manager = AppArmorManager(workspace_root=str(tmp_path / "w"),
                              venv_path="/usr/local/venv",
                              profile_dir=str(tmp_path / "p"))
    rules = ReferencesPlugin.get_apparmor_rules(
        workspace_path=WS, session_id="s", config_root=None, plugin_config={})
    return manager._render_profile("s1", WS, plugin_rules=rules)


def _grants(rules) -> ConfinementGrants:
    return ConfinementGrants(profile_name="x", exec_scope=None, rules=list(rules))


def _base_rules(profile: str):
    # The base body: everything before the first sub-profile.
    head = profile.split("  hat tool_hat", 1)[0]
    return [line.strip() for line in head.splitlines()
            if line.strip().startswith(('"/', "/", "audit deny", "owner", "deny"))]


class TestTheKernel:
    def test_child_cannot_write_a_claim(self, tmp_path):
        child = _grants(profile_body_rules(_bodies(tmp_path), "child"))
        assert child.verdict(CLAIM, "w") is False
        assert child.verdict(f"{WS}/.jaato/references-claims/", "w") is False

    def test_child_still_writes_the_workspace(self, tmp_path):
        child = _grants(profile_body_rules(_bodies(tmp_path), "child"))
        assert child.verdict(f"{WS}/src/a.py", "w") is True

    def test_the_runner_still_writes_a_claim(self, tmp_path):
        assert _grants(_base_rules(_bodies(tmp_path))).verdict(CLAIM, "w") is True


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / ".jaato" / "references-claims").mkdir(parents=True)
    (root / ".jaato" / "profiles").mkdir()
    (root / ".jaato" / "profiles" / "p.yaml").write_text("name: p\n")
    return root


def _cli(ws: Path) -> CLIToolPlugin:
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(ws), "scrub_secret_env": "default"})
    return plugin


class TestCli:
    @pytest.mark.parametrize("streaming", [False, True], ids=["foreground", "streaming"])
    def test_a_bare_relative_claim_write_is_refused(self, ws, streaming):
        cmd = "echo '{}' > .jaato/references-claims/forged.json"
        plugin = _cli(ws)
        if streaming:
            result = plugin._execute_streaming(
                {"command": cmd}, lambda _b: None, lambda _b: None, lambda _rc: None)
        else:
            result = plugin._execute({"command": cmd})
        assert result.get("returncode") == 1, result
        assert not (ws / ".jaato" / "references-claims" / "forged.json").exists()

    def test_a_bare_relative_read_is_refused(self, ws):
        result = _cli(ws)._execute({"command": "cat .jaato/profiles/p.yaml"})
        assert result.get("returncode") == 1, result
        assert "name: p" not in result.get("stdout", "")

    def test_the_refusal_names_the_jaato_rule(self, ws):
        stderr = _cli(ws)._execute({"command": "ls .jaato"})["stderr"]
        assert "closed to tools" in stderr and "sandbox add" in stderr
        assert "outside the workspace" not in stderr

    def test_ordinary_relative_paths_still_run(self, ws):
        (ws / "notes.txt").write_text("hello\n")
        result = _cli(ws)._execute({"command": "cat notes.txt"})
        assert result.get("returncode") == 0 and "hello" in result["stdout"]

    def test_a_name_that_only_starts_with_jaato_is_not_the_directory(self, ws):
        (ws / ".jaatorc").write_text("x\n")
        assert _cli(ws)._execute({"command": "cat .jaatorc"}).get("returncode") == 0
