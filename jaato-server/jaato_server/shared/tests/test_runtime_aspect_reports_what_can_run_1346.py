"""``get_environment(aspect="runtime")`` reports what a session can run (#1346).

An agent assessing a confined session probed for what it could run and
reached wrong conclusions.  The facts it needed exist runner-side; this
aspect reads them.  The properties guarded here are the issue's
acceptance criteria:

- the aspect is in the schema's enum;
- the PATH it reports is the one ``cli`` builds for its next command
  (``CLIToolPlugin._build_subprocess_env``), not a second assembly of it;
- an unconfined session says ``unconfined``, never ``unknown``, and a
  confined one reports its exec scope and exec roots from the grant
  record, or says the record is absent;
- ``aspect="all"`` carries only a one-line-per-field summary.

No AppArmor kernel is available in CI: labels and grant records are
fabricated, which is what routing both through one parser each allows.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest

from jaato_server.shared import confinement_grants as grants_mod
from jaato_server.shared.apparmor_label import parse_label
from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.plugins.environment import runtime as runtime_mod
from jaato_server.shared.plugins.environment.plugin import EnvironmentPlugin
from jaato_server.shared.tests.reversion import Reversion

_RUNTIME = "jaato-server/jaato_server/shared/plugins/environment/runtime.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/environment/plugin.py"

REVERSIONS = [
    Reversion(
        target=_RUNTIME,
        find="        env, venv_path = cli_plugin._build_subprocess_env()\n",
        replace="        env, venv_path = dict(os.environ), None\n",
        test="test_reported_path_is_the_one_cli_builds",
        because="a re-derived PATH disagrees with the one the command runs with",
    ),
    Reversion(
        target=_RUNTIME,
        find=(
            "    if not label.confined or label.mode is None:\n"
            "        return TIER_UNCONFINED\n"
        ),
        replace=(
            "    if not label.confined or label.mode is None:\n"
            "        return \"unknown\"\n"
        ),
        test="test_an_unconfined_session_says_unconfined",
        because="an unconfined session would report 'unknown'",
    ),
    Reversion(
        target=_PLUGIN,
        find='"consumption", "session", "datetime", "network", "runtime",\n',
        replace='"consumption", "session", "datetime", "network",\n',
        test="test_runtime_is_in_the_schema_enum",
        because="the model cannot ask for an aspect the enum does not list",
    ),
    Reversion(
        target=_RUNTIME,
        find='        out["grant_record"] = "absent"\n',
        replace="",
        test="test_a_confined_session_without_a_record_says_so",
        because="a missing grant record would read as nothing to report",
    ),
    Reversion(
        target=_RUNTIME,
        find='    roots = [r.glob for r in parsed.rules if r.grants("x") and not r.deny]\n',
        replace="    roots = [r.glob for r in parsed.rules if not r.deny]\n",
        test="test_a_confined_session_reports_its_exec_scope_and_roots",
        because="read-only rules would be reported as exec roots",
    ),
    Reversion(
        target=_RUNTIME,
        find=(
            '    if wanted is not None and "cli" not in wanted:\n'
            "        return None\n"
        ),
        replace="",
        test="test_a_session_without_cli_says_cli_is_not_loaded",
        because="a session whose model cannot run cli would be shown cli's PATH",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if summary:\n            return runtime_summary(report)\n",
        replace="",
        test="test_all_carries_only_a_summary",
        because="aspect='all' would carry the full report on every eager query",
    ),
]


class _Registry:
    """The two registry calls the aspect makes."""

    def __init__(self, cli: Optional[Any], exposed: Optional[List[str]] = None):
        self._cli = cli
        self._exposed = exposed if exposed is not None else ["cli"]

    def list_exposed(self) -> List[str]:
        return list(self._exposed)

    def get_plugin(self, name: str) -> Optional[Any]:
        return self._cli if name == "cli" else None


class _Session:
    def __init__(self, tool_plugins: Optional[List[str]]):
        self._tool_plugins = tool_plugins


@pytest.fixture(autouse=True)
def _no_grants():
    # The environment plugin keeps its session in a thread-local, so a
    # session one test set would otherwise be read by the next.
    EnvironmentPlugin().shutdown()
    grants_mod.set_confinement_grants(None)
    yield
    grants_mod.set_confinement_grants(None)
    EnvironmentPlugin().shutdown()


@pytest.fixture
def unconfined(monkeypatch):
    monkeypatch.setattr(runtime_mod, "read_thread_label",
                        lambda: parse_label("unconfined"))


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setattr(runtime_mod, "read_thread_label",
                        lambda: parse_label("jaato-ws-demo-abc (enforce)"))


def _cli(tmp_path) -> CLIToolPlugin:
    """A cli whose PATH differs from the runner's, so a re-derivation shows."""
    extra = tmp_path / "extra-bin"
    extra.mkdir()
    cli = CLIToolPlugin()
    cli.initialize({
        "extra_paths": [str(extra)],
        "workspace_venv": ".jaato/tool-venv",
        "workspace_home": ".home",
    })
    cli.set_workspace_path(str(tmp_path))
    return cli


def _query(plugin: EnvironmentPlugin, aspect: str) -> Dict[str, Any]:
    return json.loads(plugin._get_environment({"aspect": aspect}))


def _env_plugin(tmp_path, cli: Optional[Any], session=None) -> EnvironmentPlugin:
    plugin = EnvironmentPlugin()
    plugin.set_plugin_registry(_Registry(cli))
    plugin.set_workspace_path(str(tmp_path))
    if session is not None:
        plugin.set_session(session)
    return plugin


def test_runtime_is_in_the_schema_enum():
    schema = EnvironmentPlugin().get_tool_schemas()[0]
    aspect = schema.parameters["properties"]["aspect"]
    assert "runtime" in aspect["enum"]
    assert "'runtime' =" in aspect["description"]


def test_reported_path_is_the_one_cli_builds(tmp_path, unconfined):
    cli = _cli(tmp_path)
    report = _query(_env_plugin(tmp_path, cli), "runtime")
    env, venv_path = cli._build_subprocess_env()
    expected = [p for p in env["PATH"].split(os.pathsep) if p]
    sub = report["subprocess"]
    assert sub["path"] == expected
    assert str(tmp_path / "extra-bin") in sub["path"]
    assert sub["home"] == env["HOME"]
    assert sub["xdg"]["XDG_CONFIG_HOME"] == env["XDG_CONFIG_HOME"]
    assert sub["tool_venv"]["path"] == venv_path
    assert sub["tool_venv"]["created"] is False


def test_an_unconfined_session_says_unconfined(tmp_path, unconfined):
    report = _query(_env_plugin(tmp_path, None), "runtime")
    assert report["confinement"] == {"tier": "unconfined"}


def test_a_label_with_no_mode_claims_no_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_mod, "read_thread_label",
                        lambda: parse_label("kernel"))
    block = _query(_env_plugin(tmp_path, None), "runtime")["confinement"]
    assert block["tier"] == "unconfined"
    assert block["kernel_label"] == "kernel"


def test_a_complain_mode_profile_is_not_called_apparmor(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_mod, "read_thread_label",
                        lambda: parse_label("jaato-ws-x (complain)"))
    block = _query(_env_plugin(tmp_path, None), "runtime")["confinement"]
    assert block["tier"] == "apparmor-complain"


def test_a_confined_session_without_a_record_says_so(tmp_path, enforced):
    block = _query(_env_plugin(tmp_path, None), "runtime")["confinement"]
    assert block["tier"] == "apparmor"
    assert block["grant_record"] == "absent"
    assert "exec_scope" not in block


def test_a_confined_session_reports_its_exec_scope_and_roots(tmp_path, enforced):
    grants_mod.set_confinement_grants({
        "profile_name": "jaato-ws-demo-abc",
        "exec_scope": "scoped",
        "rules": [
            "/usr/bin/** rix,",
            "/usr/lib/cargo/bin/** ix,",
            "/etc/** r,",
            "deny /usr/bin/sudo x,",
        ],
    })
    block = _query(_env_plugin(tmp_path, None), "runtime")["confinement"]
    assert block["tier"] == "apparmor"
    assert block["profile"] == "jaato-ws-demo-abc"
    assert block["exec_scope"] == "scoped"
    assert block["exec_roots"] == ["/usr/bin/**", "/usr/lib/cargo/bin/**"]
    assert block["exec_denied"] == ["/usr/bin/sudo"]


def test_a_session_without_cli_says_cli_is_not_loaded(tmp_path, unconfined):
    plugin = _env_plugin(tmp_path, _cli(tmp_path), session=_Session(["file_edit"]))
    sub = _query(plugin, "runtime")["subprocess"]
    assert sub["cli"] == "not loaded"
    assert "path" not in sub


def test_all_carries_only_a_summary(tmp_path, unconfined):
    plugin = _env_plugin(tmp_path, _cli(tmp_path))
    summary = _query(plugin, "all")["runtime"]
    assert all(isinstance(v, str) for v in summary.values())
    assert summary["confinement"] == "unconfined"
    assert summary["toolchains"] == "absent"


def test_toolchain_manifest_is_read_tolerantly(tmp_path):
    assert runtime_mod.toolchains_report(str(tmp_path))["status"] == "absent"
    manifest = tmp_path / ".jaato" / "environment.json"
    manifest.parent.mkdir()
    manifest.write_text("{not json")
    assert runtime_mod.toolchains_report(str(tmp_path))["status"] == "unreadable"
    manifest.write_text(json.dumps({"toolchains": [{"name": "node"}]}))
    report = runtime_mod.toolchains_report(str(tmp_path))
    assert report["status"] == "present"
    assert report["toolchains"] == [{"name": "node"}]
