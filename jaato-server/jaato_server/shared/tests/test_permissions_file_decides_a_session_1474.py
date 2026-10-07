"""#1474: a ``permissions.json`` policy decides a daemon session.

Before #1474 ``PermissionPlugin.initialize`` read the file and then let an
inline ``policy`` key replace it, and both enforcer builders always passed
one (a hard-coded ``defaultPolicy: ask``, written out twice).  So the file
never decided anything: a workspace file saying ``deny`` got a prompt.

The decision, implemented once in ``policy_layers.resolve_effective_policy``
and called by both builders through ``initialize``:

* layers, lowest first: framework default (ask) < ``~/.jaato/permissions.json``
  < the project ``permissions.json`` < the profile's ``policy``;
* ``defaultPolicy``: the highest layer that sets it wins;
* white/blacklists: UNION across layers; blacklist beats whitelist;
* a #957 subagent's own block is layer 4 for that subagent, over the files;
* a file-sourced ``allow`` is announced at WARNING and reported by
  ``validate`` as ``permission_file_allow``;
* a confined session cannot write ``<ws>/.jaato/permissions.json``.

Every enforcer here is built by the real ``build_session_permission_plugin``
(the runner path, the default) or the real ``PermissionPlugin``.  HOME is
redirected so a developer's own ``~/.jaato/permissions.json`` cannot decide
the outcome.  Nothing here ran against an enforcing kernel.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_server.shared.tests.reversion import Reversion

_LAYERS = "jaato-server/jaato_server/shared/plugins/permission/policy_layers.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/permission/plugin.py"
_CORE = "jaato-server/jaato_server/server/core.py"
_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_LAYERS,
        find='        "workspace_path": workspace_path,\n    }\n',
        replace='        "workspace_path": workspace_path,\n'
                '        "policy": {"defaultPolicy": "ask"},\n    }\n',
        test="test_a_workspace_file_deny_decides_a_runner_session",
        because="the builder puts an inline default policy back, which "
                "outranks the file again: a file saying deny gets a prompt",
    ),
    Reversion(
        target=_PLUGIN,
        find="        self._policy = PermissionPolicy.from_config(effective.policy)\n",
        replace="        self._policy = PermissionPolicy.from_config(\n"
                "            config.get(\"policy\") or effective.policy)\n",
        test="test_file_and_profile_whitelists_are_unioned",
        because="an inline policy REPLACES the files again (the pre-#1474 "
                "initialize), so the file's whitelist is lost",
    ),
    Reversion(
        target=_LAYERS,
        find="            added |= _union(into.setdefault(sub, []), values)\n",
        replace="            into[sub] = list(values)\n            added = True\n",
        test="test_file_and_profile_whitelists_are_unioned",
        because="a higher layer's list replaces a lower one's instead of "
                "adding to it",
    ),
    Reversion(
        target=_LAYERS,
        find='    for layer, path in (("user", user_file), ("workspace", project_file)):\n',
        replace='    for layer, path in (("workspace", project_file), ("user", user_file)):\n',
        test="test_the_workspace_file_outranks_the_user_file",
        because="the user file outranks the workspace file",
    ),
    Reversion(
        target=_PLUGIN,
        find="        policy = PermissionPolicy.from_config(effective.policy)\n"
             "        evaluator_config = config.get(\"evaluators\")\n",
        replace="        policy = PermissionPolicy.from_config(config.get(\"policy\") or {})\n"
                "        evaluator_config = config.get(\"evaluators\")\n",
        test="test_a_subagent_block_is_layered_over_the_files",
        because="a #957 subagent's policy ignores the files again, so a "
                "file blacklist does not bind it",
    ),
    Reversion(
        target=_LAYERS,
        find="    if allow_file and _first_time(session_id, scope, \"allow:\" + allow_file):\n",
        replace="    if False and allow_file:\n",
        test="test_a_file_sourced_allow_is_announced_at_warning",
        because="a host whose file says allow starts auto-approving with "
                "nothing said",
    ),
    Reversion(
        target=_CORE,
        find="                            permission_init_config = enforcer_init_config(\n",
        replace="                            permission_init_config = dict(\n",
        test="test_both_builders_call_the_one_helper",
        because="the daemon-local builder assembles its own config again: a "
                "second place the default can come back",
    ),
    Reversion(
        target=_APPARMOR,
        find='    # and a policy file it rewrote would decide later sessions.\n'
             '    audit deny "{workspace_path}/.jaato/permissions.json"      wlk,\n',
        replace='    # and a policy file it rewrote would decide later sessions.\n',
        test="test_every_body_denies_writes_to_the_policy_file",
        because="the //child body (every model-driven subprocess) can "
                "rewrite the policy of later sessions",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find="    _check_permission_files(ws, config_root, out)\n",
        replace="",
        test="test_validate_reports_a_file_allow",
        because="a widened default is invisible until a session runs",
    ),
]


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A HOME of our own, so ~/.jaato/permissions.json is the test's."""
    h = tmp_path / "home"
    (h / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("PERMISSION_CONFIG_PATH", raising=False)
    from jaato_server.shared import user_tier
    monkeypatch.setattr(user_tier, "_installed_root", None)
    return h


@pytest.fixture
def ws(tmp_path, home):
    w = tmp_path / "ws"
    (w / ".jaato").mkdir(parents=True)
    return w


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _runner_enforcer(ws: Path, block=None, session_id="s1474"):
    """The runner's real enforcer builder (Step 8)."""
    from jaato_server.server.runner.session import build_session_permission_plugin
    env = SimpleNamespace(plugin_configs={"permission": block} if block else {},
                          config_root=None, session_id=session_id)
    return build_session_permission_plugin(env, str(ws))


def _decide(plugin, tool, context=None):
    from jaato_server.shared.plugins.permission.policy import PermissionDecision
    policy, _ = plugin._resolve_policy(context)
    d = policy.check(tool, {}).decision
    return {PermissionDecision.ALLOW: "allow", PermissionDecision.DENY: "deny",
            PermissionDecision.ASK_CHANNEL: "ask"}[d]


# ---------------------------------------------------------------- (a)-(e)

def test_a_workspace_file_deny_decides_a_runner_session(ws):
    """(a) The issue's own repro: file says deny, no profile block."""
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "deny"})
    p = _runner_enforcer(ws)
    assert _decide(p, "writeNewFile") == "deny"
    assert p._effective_policy.default_policy_source.layer == "workspace"


def test_no_file_and_no_block_is_the_framework_default(ws):
    """Control for (a): the same builder with no file asks."""
    p = _runner_enforcer(ws)
    assert _decide(p, "writeNewFile") == "ask"
    assert p._effective_policy.default_policy_source.layer == "framework"


def test_a_profile_default_overrides_the_file(ws):
    """(b)"""
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "deny"})
    p = _runner_enforcer(ws, {"policy": {"defaultPolicy": "allow"}})
    assert _decide(p, "writeNewFile") == "allow"
    assert p._effective_policy.default_policy_source.layer == "profile"


def test_file_and_profile_whitelists_are_unioned(ws):
    """(c)"""
    _write(ws / ".jaato" / "permissions.json",
           {"defaultPolicy": "deny", "whitelist": {"tools": ["readFile"]}})
    p = _runner_enforcer(ws, {"policy": {"whitelist": {"tools": ["glob_files"]}}})
    assert _decide(p, "readFile") == "allow"
    assert _decide(p, "glob_files") == "allow"
    assert _decide(p, "writeNewFile") == "deny"


def test_a_file_blacklist_survives_a_profile_whitelist(ws):
    """(d) Blacklist beats whitelist whichever layer wrote either."""
    _write(ws / ".jaato" / "permissions.json",
           {"blacklist": {"tools": ["cli_based_tool"]}})
    p = _runner_enforcer(ws, {"policy": {
        "defaultPolicy": "allow", "whitelist": {"tools": ["cli_based_tool"]}}})
    assert _decide(p, "cli_based_tool") == "deny"
    assert _decide(p, "readFile") == "allow"


def test_the_workspace_file_outranks_the_user_file(ws, home):
    """(e) user < workspace for the scalar; lists from both are kept."""
    _write(home / ".jaato" / "permissions.json",
           {"defaultPolicy": "deny", "whitelist": {"tools": ["readFile"]}})
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "ask"})
    p = _runner_enforcer(ws)
    assert _decide(p, "writeNewFile") == "ask"
    assert _decide(p, "readFile") == "allow"
    eff = p._effective_policy
    assert eff.default_policy_source.layer == "workspace"
    assert [s.layer for s in eff.list_sources] == ["user"]


def test_a_user_file_alone_decides(ws, home):
    """(e) control: the user layer applies when the project has no file."""
    _write(home / ".jaato" / "permissions.json", {"defaultPolicy": "deny"})
    assert _decide(_runner_enforcer(ws), "writeNewFile") == "deny"


# ---------------------------------------------------------------- (f)

def _calls(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and getattr(n.func, "id", getattr(n.func, "attr", None)) == name)


def test_both_builders_call_the_one_helper():
    """(f) Both enforcer builders build their config with
    ``enforcer_init_config``, and neither carries a default policy dict of
    its own -- the third copy #1474 forbids."""
    root = Path(__file__).resolve().parents[2]
    for rel in ("server/core.py", "server/runner/session.py"):
        src = root / rel
        assert _calls(src, "enforcer_init_config") >= 1, rel
        assert '"defaultPolicy": "ask"' not in src.read_text(encoding="utf-8"), rel


def test_the_helper_carries_no_policy(ws):
    """(f) What the helper hands ``initialize`` resolves identically to the
    runner builder: with no block there is no inline policy to outrank the
    files."""
    from jaato_server.shared.plugins.permission.plugin import PermissionPlugin
    from jaato_server.shared.plugins.permission.policy_layers import (
        enforcer_init_config)
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "deny"})
    cfg = enforcer_init_config(None, workspace_path=str(ws), session_id="d")
    assert "policy" not in cfg
    p = PermissionPlugin()
    p.initialize(cfg)
    assert p._effective_policy.policy == _runner_enforcer(ws)._effective_policy.policy


# ---------------------------------------------------------------- (g)

def test_a_subagent_block_is_layered_over_the_files(ws):
    """(g) A #957 subagent's own block is layer 4 for it, on top of the
    files: its allow cannot lift a file blacklist."""
    _write(ws / ".jaato" / "permissions.json",
           {"blacklist": {"tools": ["cli_based_tool"]}})
    p = _runner_enforcer(ws)
    p.set_scoped_policy("child", {"policy": {
        "defaultPolicy": "allow", "whitelist": {"tools": ["glob_files"]}}})
    ctx = {"permission_scope": "child"}
    assert _decide(p, "cli_based_tool", ctx) == "deny"
    assert _decide(p, "glob_files", ctx) == "allow"
    assert _decide(p, "writeNewFile", ctx) == "allow"
    # The root is untouched: the file's blacklist and the framework's ask.
    assert _decide(p, "writeNewFile") == "ask"


# ---------------------------------------------------------------- (h)

def test_a_file_sourced_allow_is_announced_at_warning(ws, caplog):
    """(h)"""
    path = ws / ".jaato" / "permissions.json"
    _write(path, {"defaultPolicy": "allow"})
    with caplog.at_level(logging.INFO):
        _runner_enforcer(ws, session_id="warn-1")
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "defaultPolicy=allow" in r.getMessage()]
    assert len(warnings) == 1 and str(path) in warnings[0], warnings
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO
             and "defaultPolicy=allow" in r.getMessage()]
    assert infos and "workspace file" in infos[0]


def test_a_profile_allow_is_not_a_file_warning(ws, caplog):
    """Control for (h): allow from the profile is the author's own choice."""
    with caplog.at_level(logging.WARNING):
        _runner_enforcer(ws, {"policy": {"defaultPolicy": "allow"}},
                         session_id="warn-2")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING
                and "defaultPolicy=allow comes from" in r.getMessage()]


# ---------------------------------------------------------------- the file

def test_every_body_denies_writes_to_the_policy_file(tmp_path):
    """A policy file the model could write would let a session widen the
    policy of later sessions.  Every body a confined process runs under
    write-denies it (sub-profiles do not inherit the base's rules)."""
    import re
    from jaato_server.server.apparmor import AppArmorManager
    from jaato_server.shared.tests.test_cache_fragment_tier_is_write_denied_1385 import (
        _bodies)
    (tmp_path / "w" / "sessions").mkdir(parents=True)
    (tmp_path / "p").mkdir()
    m = AppArmorManager(workspace_root=str(tmp_path / "w"),
                        venv_path="/usr/local/venv", profile_dir=str(tmp_path / "p"))
    deny = re.compile(r'audit deny "/workspace/\.jaato/permissions\.json"\s+wlk,')
    missing = [name for name, body in _bodies(m).items() if not deny.search(body)]
    assert not missing, f"permissions.json writable in: {missing}"


def test_file_tools_refuse_the_policy_file(ws):
    """The application layer: the .jaato/ containment rule refuses it."""
    from jaato_server.shared.plugins.sandbox_utils import (
        check_path_with_jaato_containment)
    target = str(ws / ".jaato" / "permissions.json")
    assert not check_path_with_jaato_containment(target, str(ws), mode="write")


def test_validate_reports_a_file_allow(ws):
    from jaato_server.shared.scaffold.validate import validate_workspace
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "allow"})
    codes = [d.code for d in validate_workspace(str(ws))]
    assert "permission_file_allow" in codes


def test_validate_is_quiet_for_a_file_that_asks(ws):
    from jaato_server.shared.scaffold.validate import validate_workspace
    _write(ws / ".jaato" / "permissions.json", {"defaultPolicy": "ask"})
    codes = [d.code for d in validate_workspace(str(ws))]
    assert "permission_file_allow" not in codes
