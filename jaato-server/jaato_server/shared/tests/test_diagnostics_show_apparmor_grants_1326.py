"""The diagnostics panel shows what a session's AppArmor profile grants (#1326).

The panel (#1294) said a session was confined and never what the
confinement allowed.  When a confined command fails, the question is which
fragment, plugin or reference granted a rule, or why none did, and the only
answers were a daemon log line and the rendered profile on the host.

Now the profile's composition is recorded when it is loaded, keyed by the
profile name the runner identity carries, and the diagnostics verb returns
it to the workspace owner as ``apparmor_grants`` (protocol 1.26).  Pinned
here:

- provisioning records each fragment's tier, file and rules, fragments
  requested and not found, a fragment that shadows a lower tier, the exec
  scope and the rules each plugin contributed; a failed load records nothing;
- plugin rules keep which plugin contributed them;
- reference grants added after provisioning are listed live;
- the profile that declared ``apparmor_fragments`` survives inheritance and
  the snapshot round trip, and an older snapshot reads as unknown;
- the verb returns all of it to the owner, nothing to anyone else, ``None``
  for an unconfined session, and ``recorded: False`` rather than a guess.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from jaato_sdk.events import DiagnosticsRequest
from jaato_server.server import apparmor as aa
from jaato_server.server.apparmor import (
    AppArmorManager,
    record_grants,
    recorded_grants,
    resolve_plugin_apparmor_rules,
)
from jaato_server.server.diagnostics_verbs import answer_diagnostics_request
from jaato_server.shared.plugins.subagent.config import (
    SubagentProfile,
    profile_from_snapshot,
    profile_to_snapshot,
    resolve_profiles,
)
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_AA,
        find=(
            "        self._record_grants(session_id, render_id, profile_name,\n"
            "                            requested_fragments, plugin_rules, composition)\n"
        ),
        replace="",
        test="test_provisioning_records_what_the_profile_grants",
        because="nothing records what a loaded profile grants",
    ),
    Reversion(
        target=_AA,
        find="                by_plugin.setdefault(plugin_name, []).extend(contributed)\n",
        replace="",
        test="test_plugin_rules_keep_which_plugin_contributed_them",
        because="plugin rules arrive flat, with no way to tell who asked for one",
    ),
    Reversion(
        target=_AA,
        find='    record["references"] = references\n',
        replace='    record["references"] = []\n',
        test="test_reference_grants_are_listed_live",
        because="reference grants added mid-session never appear",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/subagent/config.py",
        find=(
            "            return list(parent_fragments), (\n"
            "                getattr(p, 'apparmor_fragments_source', None) or p.name)\n"
        ),
        replace="            return list(parent_fragments), p.name\n",
        test="test_the_declaring_profile_survives_inheritance",
        because="a grandchild names its parent rather than the ancestor that declared",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/diagnostics_verbs.py",
        find=(
            '            probe=probe_answer.get("probe"),\n'
            "            apparmor_grants=apparmor_grants,\n"
        ),
        replace='            probe=probe_answer.get("probe"),\n',
        test="test_the_owner_gets_the_grants",
        because="the verb never returns the recorded grants",
    ),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # The user-tier fragment dir is ~/.jaato/apparmor-fragments; point HOME
    # somewhere empty, and start each test from an empty grant registry.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(aa, "_GRANT_RECORDS", {})
    return home


@pytest.fixture
def manager(tmp_path):
    m = AppArmorManager(
        workspace_root=str(tmp_path / "workspaces"),
        venv_path="/usr/local/venv",
        profile_dir=str(tmp_path / "profiles"),
    )
    (tmp_path / "profiles").mkdir()
    m._available = True
    return m


def _fragment(dir_, name, body):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{name}.rules").write_text(body)


def _provision(manager, workspace, *, returncode=0, **kw):
    with patch("jaato_server.server.apparmor.subprocess.run") as run:
        run.return_value = MagicMock(returncode=returncode, stderr="")
        return manager.provision_profile("s1", str(workspace), **kw)


# ---- recorded at provisioning ------------------------------------------------

def test_provisioning_records_what_the_profile_grants(manager, tmp_path, _isolated):
    ws = tmp_path / "workspaces" / "ws"
    _fragment(_isolated / ".jaato" / "apparmor-fragments", "java", "# user copy\n/usr/bin/java ix,\n")
    _fragment(ws / ".jaato" / "apparmor-fragments", "java",
              "# java for this stage\n/usr/bin/java ix,\n/usr/bin/mvn ix,\n")
    plugin_rules = aa.PluginRules(["/etc/os-release r,"], {"cli": ["/etc/os-release r,"]})

    assert _provision(manager, ws, requested_fragments=["java", "curl"],
                      plugin_rules=plugin_rules) is True

    g = recorded_grants(manager.get_profile_name("s1"))
    assert g is not None
    assert g["exec_scope"] == "scoped"
    assert g["requested_fragments"] == ["java", "curl"]
    assert g["template_version"] == AppArmorManager._TEMPLATE_VERSION
    [frag] = g["fragments"]
    assert (frag["name"], frag["tier"]) == ("java", "workspace")
    assert frag["path"].endswith("/ws/.jaato/apparmor-fragments/java.rules")
    assert frag["rules"] == ["/usr/bin/java ix,", "/usr/bin/mvn ix,"]
    assert frag["shadows"] == ["user"]
    assert g["missing_fragments"] == ["curl"]
    assert g["plugin_rules"] == [{"plugin": "cli", "rules": ["/etc/os-release r,"]}]


def test_an_unscoped_session_is_recorded_as_unscoped(manager, tmp_path):
    assert _provision(manager, tmp_path / "workspaces" / "ws") is True
    g = recorded_grants(manager.get_profile_name("s1"))
    assert g["exec_scope"] == "unscoped"
    assert g["requested_fragments"] is None


def test_a_profile_the_kernel_refused_is_not_recorded(manager, tmp_path):
    assert _provision(manager, tmp_path / "workspaces" / "ws", returncode=1) is False
    assert recorded_grants(manager.get_profile_name("s1")) is None


def test_plugin_rules_keep_which_plugin_contributed_them():
    class _Plugin:
        def __init__(self, rules):
            self._rules = rules

        def get_apparmor_rules(self, **_):
            return list(self._rules)

    plugins = {"cli": _Plugin(["/etc/os-release r,"]),
               "notebook": _Plugin(["/etc/os-release r,", "/usr/bin/python3 ix,"])}

    class _Registry:
        def get_plugin(self, name):
            return plugins.get(name)

        def all_plugins(self):
            return plugins

    class _Server:
        registry = _Registry()

    class _Profile:
        plugins = ["cli", "notebook"]
        plugin_configs: Dict[str, Any] = {}

    rules = resolve_plugin_apparmor_rules(_Server(), _Profile(), "s", "/ws", None)
    # Still one flat, deduplicated list for the renderer...
    assert list(rules) == ["/etc/os-release r,", "/usr/bin/python3 ix,"]
    # ...and each plugin's own contribution for the panel.
    assert rules.by_plugin == {
        "cli": ["/etc/os-release r,"],
        "notebook": ["/etc/os-release r,", "/usr/bin/python3 ix,"],
    }


def test_reference_grants_are_listed_live(tmp_path):
    refs = tmp_path / "refs.d"
    record_grants("jaato-ws-x", {"exec_scope": "unscoped", "refs_dir": str(refs)})
    assert recorded_grants("jaato-ws-x")["references"] == []
    refs.mkdir()
    (refs / "spec").write_text("# ref\n/srv/spec/** r,\n")
    got = recorded_grants("jaato-ws-x")
    assert got["references"] == [{"ref_id": "spec", "rules": ["/srv/spec/** r,"]}]
    assert "refs_dir" not in got


# ---- which profile declared the fragments -------------------------------------

def _chain():
    base = SubagentProfile(name="base", description="b", apparmor_fragments=["java"])
    mid = SubagentProfile(name="mid", description="m", inherits=["base"])
    leaf = SubagentProfile(name="leaf", description="l", inherits=["mid"])
    own = SubagentProfile(name="own", description="o", inherits=["base"],
                          apparmor_fragments=[])
    plain = SubagentProfile(name="plain", description="p")
    resolved, errors = resolve_profiles({p.name: p for p in (base, mid, leaf, own, plain)})
    assert not errors
    return resolved


def test_the_declaring_profile_survives_inheritance():
    r = _chain()
    assert r["leaf"].apparmor_fragments_source == "base"
    assert r["mid"].apparmor_fragments_source == "base"
    assert r["own"].apparmor_fragments_source == "own"
    assert r["plain"].apparmor_fragments_source is None


def test_the_declaring_profile_survives_the_snapshot():
    leaf = _chain()["leaf"]
    snap = profile_to_snapshot(leaf)
    assert profile_from_snapshot(snap).apparmor_fragments_source == "base"
    # A snapshot written before #1326 carries the list, not its source.
    snap.pop("apparmor_fragments_source")
    assert profile_from_snapshot(snap).apparmor_fragments_source == ""


# ---- the verb -----------------------------------------------------------------

class _Identity:
    def __init__(self, profile):
        self.apparmor_profile = profile

    def to_dict(self):
        return {"apparmor_profile": self.apparmor_profile}


class _Session:
    def __init__(self, profile):
        self.runner_identity = _Identity(profile)
        self.sandbox_mode = "apparmor" if profile else None


class _Server:
    def __init__(self, profile=None):
        self._profile = profile

    def diagnostics_probe(self):
        return {"probe": {"ok": True}, "protocol_version": "1.26"}


def _ask(server, session, user="app:alice", owner="app:alice"):
    return answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r"),
        session_id="s1", user_id=user, owner=owner, session=session)


def test_the_owner_gets_the_grants():
    record_grants("jaato-ws-a", {"exec_scope": "scoped", "fragments": [], "refs_dir": ""})
    answer = _ask(_Server(_chain()["leaf"]), _Session("jaato-ws-a"))
    assert answer.apparmor_grants is not None
    assert answer.apparmor_grants["recorded"] is True
    assert answer.apparmor_grants["exec_scope"] == "scoped"
    assert answer.apparmor_grants["declared_by"] == "base"


def test_anyone_else_gets_nothing():
    record_grants("jaato-ws-a", {"exec_scope": "scoped", "refs_dir": ""})
    answer = _ask(_Server(), _Session("jaato-ws-a"), user="app:bob")
    assert answer.category == "not_owner"
    assert answer.apparmor_grants is None


def test_an_unconfined_session_has_no_grants():
    assert _ask(_Server(), _Session("")).apparmor_grants is None


def test_an_unrecorded_boundary_says_so():
    answer = _ask(_Server(_chain()["own"]), _Session("jaato-ws-never-recorded"))
    g = answer.apparmor_grants
    assert g["recorded"] is False
    assert g["declared_by"] == "own"
    assert g["requested_fragments"] == []
    assert "fragments" not in g
