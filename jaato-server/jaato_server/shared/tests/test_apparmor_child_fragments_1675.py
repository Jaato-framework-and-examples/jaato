"""#1675: AppArmor fragments for the tool child alone.

``apparmor_fragments`` compose into the base profile, ``tool_hat`` and
``//child`` alike.  The runner wears the base profile and runs the model
loop, so a ``deny network inet,`` there cuts the session off from its
provider.  ``apparmor_child_fragments`` names fragments composed into
``//child`` only, where the commands ``cli`` / ``interactive_shell`` / the
notebook kernel run.

These tests read the rendered profile TEXT and the profile resolution; no
kernel.  Nothing here was run against an enforcing AppArmor host.
"""

import os
import sys
import types
from pathlib import Path

import pytest

from jaato_server.shared.tests.reversion import Reversion

if "jaato_server.server" not in sys.modules:
    _stub = types.ModuleType("jaato_server.server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["jaato_server.server"] = _stub

import jaato_server.server.apparmor as _apparmor_mod
from jaato_server.server.confinement.apparmor import AppArmorBackend
from jaato_server.server.confinement.base import Boundary
from jaato_server.shared.confinement_grants import profile_body_rules

AppArmorManager = _apparmor_mod.AppArmorManager

_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"
_BACKEND = "jaato-server/jaato_server/server/confinement/apparmor.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_APPARMOR,
        find="        child_only_inline = self._compose_child_fragments(\n"
             "            session_id, workspace_path, child_fragments, composition_out)\n",
        replace="        child_only_inline = \"\"\n",
        test="test_a_child_fragment_reaches_only_the_child",
        because="the named child-only fragments are never composed, so the "
                "session's commands keep the network the profile denied them",
    ),
    Reversion(
        target=_APPARMOR,
        find="            env_file_rule=env_file_rule,\n"
             "            refs_include_glob=f\"{self._refs_dir(session_id)}/*\",\n"
             "            extension_fragments_inline=extension_fragments_inline,\n"
             "            plugin_contributed_rules=plugin_rules_subprofile_inline,\n"
             "            session_id=session_id,\n"
             "            subprofile_flag_clause=subprofile_flag_clause,\n"
             "            private_tmp_rules=self._private_tmp_rules(private_tmp_dir, \"    \"),\n"
             "        )\n",
        replace="            env_file_rule=env_file_rule,\n"
                "            refs_include_glob=f\"{self._refs_dir(session_id)}/*\",\n"
                "            extension_fragments_inline=child_fragments_inline,\n"
                "            plugin_contributed_rules=plugin_rules_subprofile_inline,\n"
                "            session_id=session_id,\n"
                "            subprofile_flag_clause=subprofile_flag_clause,\n"
                "            private_tmp_rules=self._private_tmp_rules(private_tmp_dir, \"    \"),\n"
                "        )\n",
        test="test_a_child_fragment_reaches_only_the_child",
        because="tool_hat (in-process tools, on the runner's own thread "
                "pool) gets the child-only deny too",
    ),
    Reversion(
        target=_BACKEND,
        find="        if boundary.child_fragments:\n"
             "            kwargs[\"child_fragments\"] = list(boundary.child_fragments)\n",
        replace="",
        test="test_the_backend_hands_the_manager_the_child_fragments",
        because="the profile key reaches the Boundary and is dropped before "
                "the renderer",
    ),
    Reversion(
        target=_CONFIG,
        find="    for p in [child, *parents]:\n"
             "        value = getattr(p, 'apparmor_child_fragments', None)\n",
        replace="    for p in [child]:\n"
                "        value = getattr(p, 'apparmor_child_fragments', None)\n",
        test="test_the_key_is_inherited_child_replaces",
        because="a stage inheriting a base that denies its commands the "
                "network silently loses the deny",
    ),
    Reversion(
        target=_VALIDATE,
        find="    _check_apparmor_fragment_scopes(result.profiles, ws, out)\n",
        replace="",
        test="test_validate_warns_about_a_network_deny_in_the_runner",
        because="a deny network that cuts the model loop off validates clean",
    ),
]

_DENY = "deny network inet,"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    frags = ws / ".jaato" / "apparmor-fragments"
    frags.mkdir(parents=True)
    (frags / "no_network.rules").write_text(
        "# commands may not reach the network\n"
        f"{_DENY}\n"
        "deny network inet6,\n")
    (frags / "tools.rules").write_text("/usr/bin/git ix,\n")
    return ws


@pytest.fixture
def manager(tmp_path):
    root = tmp_path / "workspaces"
    (root / "sessions").mkdir(parents=True)
    pdir = tmp_path / "apparmor_profiles"
    pdir.mkdir()
    return AppArmorManager(
        workspace_root=str(root), venv_path="/usr/local/venv",
        profile_dir=str(pdir))


def _render(manager, ws, **kw):
    return manager._render_profile("s1", str(ws), **kw)


def _base_rules(text: str):
    """Base body rules, with the two sub-profiles cut out."""
    out, depth_skip = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if line in ("profile tool_hat {", "profile child {") or (
                line.startswith(("profile tool_hat ", "profile child ")) and line.endswith("{")):
            depth_skip = True
            continue
        if depth_skip:
            if line == "}":
                depth_skip = False
            continue
        if line and not line.startswith("#"):
            out.append(line)
    return out


def test_a_child_fragment_reaches_only_the_child(manager, workspace):
    text = _render(manager, workspace, requested_fragments=["tools"],
                   child_fragments=["no_network"])
    assert _DENY in profile_body_rules(text, "child")
    assert _DENY not in profile_body_rules(text, "tool_hat")
    assert _DENY not in _base_rules(text)
    # The ordinary fragment still reaches every body.
    assert "/usr/bin/git ix," in profile_body_rules(text, "child")
    assert "/usr/bin/git ix," in _base_rules(text)


def test_no_child_fragments_renders_as_before(manager, workspace):
    before = _render(manager, workspace, requested_fragments=["tools"])
    assert before == _render(manager, workspace, requested_fragments=["tools"],
                             child_fragments=[])
    assert before == _render(manager, workspace, requested_fragments=["tools"],
                             child_fragments=None)


def test_the_child_fragments_are_part_of_the_boundary(manager, workspace):
    plain = manager.confinement_id_for_boundary(str(workspace), requested_fragments=[])
    denied = manager.confinement_id_for_boundary(
        str(workspace), requested_fragments=[], child_fragments=["no_network"])
    assert plain != denied


def test_a_missing_child_fragment_is_recorded(manager, workspace):
    comp = {}
    text = manager._render_profile("s1", str(workspace), requested_fragments=[],
                                   child_fragments=["nope"], composition_out=comp)
    assert comp["missing_child_fragments"] == ["nope"]
    assert _DENY not in text


def test_the_backend_hands_the_manager_the_child_fragments():
    seen = {}

    class _Manager:
        def confinement_id_for_boundary(self, ws, **kw):
            seen["id"] = kw
            return "cid"

        def provision_profile(self, sid, ws, **kw):
            seen["provision"] = kw
            return True

        def get_profile_name(self, sid):
            return "jaato-ws-cid"

        def profile_is_complain_mode(self, sid):
            return False

    AppArmorBackend(_Manager()).provision("s1", Boundary(
        workspace_path="/ws", requested_fragments=(),
        child_fragments=("no_network",)))
    assert seen["id"]["child_fragments"] == ["no_network"]
    assert seen["provision"]["child_fragments"] == ["no_network"]


def _write_profiles(ws: Path, files: dict) -> None:
    d = ws / ".jaato" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (d / f"{name}.yaml").write_text(body)


def _discover(ws: Path):
    from jaato_server.shared.plugins.subagent.config import discover_profiles
    return discover_profiles(profiles_dir=".jaato/profiles", base_path=str(ws),
                             config_root=str(ws / ".jaato"), session_env={})


def test_the_key_is_inherited_child_replaces(workspace):
    _write_profiles(workspace, {
        "_base": "name: _base\ndescription: b\nplugins: [cli]\n"
                 "apparmor_child_fragments: [no_network]\n",
        "stage": "name: stage\ndescription: s\nplugins: [cli]\n"
                 "inherits: [_base]\nmodel: m\nprovider: echo\n",
        "open": "name: open\ndescription: o\nplugins: [cli]\n"
                "inherits: [_base]\napparmor_child_fragments: []\n",
    })
    profiles = _discover(workspace).profiles
    assert profiles["stage"].apparmor_child_fragments == ["no_network"]
    assert profiles["open"].apparmor_child_fragments == []


def test_the_key_survives_a_snapshot(workspace):
    from jaato_server.shared.plugins.subagent.config import (
        profile_from_snapshot, profile_to_snapshot,
    )
    _write_profiles(workspace, {
        "p": "name: p\ndescription: p\nplugins: [cli]\n"
             "apparmor_child_fragments: [no_network]\n",
    })
    prof = _discover(workspace).profiles["p"]
    assert profile_from_snapshot(profile_to_snapshot(prof)).apparmor_child_fragments == [
        "no_network"]


def test_a_malformed_value_is_refused(workspace):
    _write_profiles(workspace, {
        "p": "name: p\ndescription: p\nplugins: [cli]\n"
             "apparmor_child_fragments: no_network\n",
    })
    # The same posture as apparmor_fragments: a typo fails loud at load.
    with pytest.raises(ValueError, match="apparmor_child_fragments"):
        _discover(workspace)


def _codes(ws: Path):
    from jaato_server.shared.scaffold.validate import validate_workspace
    return {(d.code, d.profile) for d in validate_workspace(str(ws))}


def test_validate_warns_about_a_network_deny_in_the_runner(workspace):
    _write_profiles(workspace, {
        # unset apparmor_fragments composes every fragment: no_network too
        "loop": "name: loop\ndescription: l\nplugins: [cli]\nmodel: m\nprovider: echo\n",
        "scoped": "name: scoped\ndescription: s\nplugins: [cli]\nmodel: m\n"
                  "provider: echo\napparmor_fragments: [tools]\n"
                  "apparmor_child_fragments: [no_network]\n",
    })
    codes = _codes(workspace)
    assert ("apparmor_fragment_denies_model_loop", "loop") in codes
    assert ("apparmor_fragment_denies_model_loop", "scoped") not in codes


def test_validate_names_a_missing_child_fragment(workspace):
    _write_profiles(workspace, {
        "p": "name: p\ndescription: p\nplugins: [cli]\napparmor_fragments: []\n"
             "apparmor_child_fragments: [typo]\n",
    })
    assert ("apparmor_child_fragment_missing", "p") in _codes(workspace)


def test_denies_network_reads_rules_not_comments():
    from jaato_server.shared.apparmor_fragments import denies_network
    body = ("# deny network inet,\n"
            "deny network unix,\n"
            "  audit deny network inet stream,\n"
            "deny network,\n")
    assert denies_network(body) == ["audit deny network inet stream,", "deny network,"]


def test_oversight_shows_which_rules_reach_which_scope(workspace):
    from jaato_server.shared.scaffold.explain import oversight_profile
    _write_profiles(workspace, {
        "p": "name: p\ndescription: p\nplugins: [cli]\nmodel: m\nprovider: echo\n"
             "apparmor_fragments: [tools]\napparmor_child_fragments: [no_network]\n",
    })
    data, text = oversight_profile("p", str(workspace))
    assert data["apparmor_scopes"] == {
        "runner_and_child": ["tools"], "child_only": ["no_network"]}
    assert "child only (commands, never the model loop): no_network" in text
