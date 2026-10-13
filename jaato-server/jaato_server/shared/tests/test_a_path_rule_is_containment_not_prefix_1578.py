"""#1578: a path argument rule is containment, not a string prefix.

``whitelist.arguments`` / ``blacklist.arguments`` are
``{"<tool>": {"<arg>": ["<value>", ...]}}``, and every argument was matched
by ``str.startswith``.  For a path that is not a containment test: a rule
``{"writeNewFile": {"path": ["out/"]}}`` allowed
``out/../../anywhere.py``, and ``out`` allowed the sibling ``out2``.

Now an argument whose name is a path name (``argument_paths.PATH_ARGUMENT_NAMES``)
is matched by containment, against the session's workspace:

* whitelist: inside the rule by path component, lexically (``..``
  collapsed) AND after symlink resolution;
* blacklist: the old string test, OR inside the rule lexically, OR after
  symlink resolution -- normalisation only widens a deny;
* any other argument keeps the prefix test (``command`` ``git``).

Every case drives the real ``PermissionPlugin.check_permission`` built by
the runner's real enforcer builder, with HOME redirected so a developer's
own ``~/.jaato/permissions.json`` cannot decide the outcome.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_server.shared.tests.reversion import Reversion

_PATHS = "jaato-server/jaato_server/shared/plugins/permission/argument_paths.py"
_POLICY = "jaato-server/jaato_server/shared/plugins/permission/policy.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/permission/plugin.py"

REVERSIONS = [
    Reversion(
        target=_POLICY,
        find="        if is_path_argument(arg_name):\n"
             "            return path_rule_allows(arg_value, allowed, self._path_rule_base)\n",
        replace="",
        test="test_parent_traversal_does_not_pass_a_path_whitelist",
        because="a path argument is matched by string prefix again, so "
                "'out/../../x' passes a rule on 'out/'",
    ),
    Reversion(
        target=_PATHS,
        find="    prefix = root if root.endswith(os.sep) else root + os.sep\n"
             "    return value.startswith(prefix)\n",
        replace="    return value.startswith(root)\n",
        test="test_a_sibling_with_a_shared_prefix_does_not_match",
        because="containment compares characters, not components: 'out2' is "
                "inside 'out'",
    ),
    Reversion(
        target=_PATHS,
        find="    return True if resolved is None else resolved\n",
        replace="    return True\n",
        test="test_a_symlink_out_of_the_rule_does_not_pass",
        because="the whitelist trusts the lexical path, so a link inside the "
                "rule that points outside it is allowed",
    ),
    Reversion(
        target=_POLICY,
        find="        return is_path_argument(arg_name) and path_rule_denies(\n"
             "            arg_value, blocked, self._path_rule_base)\n",
        replace="        return False\n",
        test="test_a_blacklist_catches_a_traversal_into_the_denied_dir",
        because="a deny on 'secret' is dodged by 'a/../secret/x'",
    ),
    Reversion(
        target=_PLUGIN,
        find="        self._policy.path_base = self._workspace_path\n",
        replace="",
        test="test_a_relative_rule_resolves_against_the_workspace",
        because="relative rules are compared against nothing, so an absolute "
                "path into the allowed workspace dir is not recognised",
    ),
]


@pytest.fixture
def home(tmp_path, monkeypatch):
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
    (w / "out").mkdir()
    (w / "secret").mkdir()
    return w


def _enforcer(ws: Path, policy: dict):
    """The runner's real enforcer builder (Step 8), with a profile policy."""
    from jaato_server.server.runner.session import build_session_permission_plugin
    env = SimpleNamespace(plugin_configs={"permission": {"policy": policy}},
                          config_root=None, session_id="s1578")
    return build_session_permission_plugin(env, str(ws))


def _allowed(plugin, tool: str, **args) -> bool:
    allowed, _info = plugin.check_permission(tool, args)
    return allowed


def _write_rule(ws, values):
    return _enforcer(ws, {"defaultPolicy": "deny",
                          "whitelist": {"arguments": {"writeNewFile": {"path": values}}}})


def test_a_path_inside_the_rule_is_allowed(ws):
    p = _write_rule(ws, ["out/"])
    assert _allowed(p, "writeNewFile", path="out/report.md", content="x")
    assert _allowed(p, "writeNewFile", path="out/a/b.md", content="x")
    assert _allowed(p, "writeNewFile", path=str(ws / "out" / "c.md"), content="x")


def test_parent_traversal_does_not_pass_a_path_whitelist(ws):
    p = _write_rule(ws, ["out/"])
    assert not _allowed(p, "writeNewFile", path="out/../../anywhere.py", content="x")
    assert not _allowed(p, "writeNewFile", path="out/../secret/x", content="x")


def test_a_sibling_with_a_shared_prefix_does_not_match(ws):
    p = _write_rule(ws, ["out"])
    assert _allowed(p, "writeNewFile", path="out/x", content="x")
    assert not _allowed(p, "writeNewFile", path="out2/x", content="x")
    assert not _allowed(p, "writeNewFile", path="out.md", content="x")


def test_a_symlink_out_of_the_rule_does_not_pass(ws):
    os.symlink(ws / "secret", ws / "out" / "link")
    p = _write_rule(ws, ["out"])
    assert not _allowed(p, "writeNewFile", path="out/link/x", content="x")


def test_a_relative_rule_resolves_against_the_workspace(ws):
    p = _write_rule(ws, ["out"])
    assert _allowed(p, "writeNewFile", path=str(ws / "out" / "y"), content="x")
    assert not _allowed(p, "writeNewFile", path=str(ws / "secret" / "y"), content="x")


def test_a_blacklist_catches_a_traversal_into_the_denied_dir(ws):
    p = _enforcer(ws, {"defaultPolicy": "allow",
                       "blacklist": {"arguments": {"readFile": {"path": ["secret"]}}}})
    assert not _allowed(p, "readFile", path="secret/k")
    assert not _allowed(p, "readFile", path="out/../secret/k")
    assert not _allowed(p, "readFile", path=str(ws / "secret" / "k"))
    assert _allowed(p, "readFile", path="out/k")


def test_a_blacklist_keeps_its_string_test(ws):
    """Normalisation only widens a deny: the old prefix still denies."""
    p = _enforcer(ws, {"defaultPolicy": "allow",
                       "blacklist": {"arguments": {"readFile": {"path": ["sec"]}}}})
    assert not _allowed(p, "readFile", path="secret/k")


def test_a_non_path_argument_keeps_its_prefix(ws):
    p = _enforcer(ws, {"defaultPolicy": "deny", "whitelist": {"arguments": {
        "cli_based_tool": {"command": ["git"]}}}})
    assert _allowed(p, "cli_based_tool", command="git status")
    assert not _allowed(p, "cli_based_tool", command="git status; rm -rf /")
