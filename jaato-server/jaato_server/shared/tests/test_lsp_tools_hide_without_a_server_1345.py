"""The LSP tools are hidden while no language server is configured (#1345).

In a workspace with no ``plugin_configs.lsp.languageServers`` and no
``.lsp.json``, every ``lsp_*`` call answers "No LSP servers configured".
The tools are deferred, so their cost was the catalog: ``list_tools``
listed them, ``get_tool_schemas`` activated them, and an agent spent
calls learning they could not work.

``LSPToolPlugin.is_tool_visible`` now hides them while the resolved server
table is empty, and ``list_tools`` / ``get_tool_schemas`` consult the same
predicates the provider tool array does (``shared/tool_visibility.py``);
before this they consulted none.

These tests drive a real ``PluginRegistry`` holding the real ``lsp`` and
``introspection`` plugins, and call ``list_tools`` / ``get_tool_schemas``
as the model would.  The lsp background thread is switched off so no
language server is spawned; the predicate reads configuration only, so
"configured but not connected" is exactly the state under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Set

import pytest

from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.introspection.plugin import IntrospectionPlugin
from jaato_server.shared.plugins.lsp.plugin import LSP_TOOL_NAMES, LSPToolPlugin
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_id_map import name_to_id

_LSP = "jaato-server/jaato_server/shared/plugins/lsp/plugin.py"
_INTRO = "jaato-server/jaato_server/shared/plugins/introspection/plugin.py"
_SESSION = "jaato-server/jaato_server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_LSP,
        find=(
            "        if tool_name not in LSP_TOOL_NAMES:\n"
            "            return True\n"
            "        return self._has_configured_servers()\n"
        ),
        replace="        return True\n",
        test="test_with_no_server_list_tools_names_no_lsp_tool",
        because="the predicate never hides, so the catalog lists tools that cannot work",
    ),
    Reversion(
        target=_LSP,
        find="        if signature != self._config_signature:\n",
        replace="        if self._config_signature is _UNLOADED:\n",
        test="test_a_lsp_json_written_mid_session_shows_the_tools",
        because="the table is resolved once, so a managed .lsp.json written at bind time is never seen",
    ),
    Reversion(
        target=_INTRO,
        find=(
            "        return filter_visible_tool_schemas(\n"
            "            self._registry, self._get_session_plugin_schemas()\n"
            "        )\n"
        ),
        replace="        return self._get_session_plugin_schemas()\n",
        test="test_with_no_server_get_tool_schemas_does_not_activate_them",
        because="get_tool_schemas activates a tool the provider call then withholds",
    ),
    Reversion(
        target=_INTRO,
        find=(
            "        return filter_visible_tool_schemas(\n"
            "            self._registry, self._registry.get_exposed_tool_schemas()\n"
            "        )\n"
        ),
        replace="        return self._registry.get_exposed_tool_schemas()\n",
        test="test_with_no_server_list_tools_names_no_lsp_tool",
        because="the category listing walks the unfiltered global set",
    ),
    Reversion(
        target=_SESSION,
        find="        return filter_visible_tool_schemas(registry, scoped, on_error=_on_error)\n",
        replace="        return scoped\n",
        test="test_the_provider_tool_array_hides_them_too",
        because="the wire and the catalog stop agreeing",
    ),
]


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A HOME with no ``~/.lsp.json``, so the developer's own file cannot leak in."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _make(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    lsp_config: Optional[Dict[str, Any]] = None,
):
    """A real registry with ``lsp`` and ``introspection`` exposed."""
    # No background thread: nothing is spawned, nothing connects.
    monkeypatch.setattr(LSPToolPlugin, "_ensure_thread", lambda self: None)
    registry = PluginRegistry()
    lsp = LSPToolPlugin()
    config = {"workspace_path": str(workspace)}
    config.update(lsp_config or {})
    registry.register_plugin(lsp, expose=True, config=config)
    intro = IntrospectionPlugin()
    registry.register_plugin(intro, expose=True)
    intro.set_plugin_registry(registry)
    return registry, lsp, intro


def _listed(intro: IntrospectionPlugin) -> Set[str]:
    """Every tool name ``list_tools`` names, across every category."""
    summary = intro._execute_list_tools({})
    names: Set[str] = set()
    for cat in summary["categories"]:
        detail = intro._execute_list_tools({"category_id": cat["id"]})
        names.update(t["name"] for t in detail.get("tools", []))
    return names


def _write_lsp_json(workspace: Path) -> None:
    (workspace / ".lsp.json").write_text(json.dumps(
        {"languageServers": {"python": {"command": "no-such-lsp-binary"}}}
    ))


# ── tests ──────────────────────────────────────────────────────────────


def test_the_hidden_set_is_exactly_the_plugins_tools(home, workspace, monkeypatch):
    _, lsp, _ = _make(workspace, monkeypatch)
    assert {s.name for s in lsp.get_tool_schemas()} == LSP_TOOL_NAMES


def test_with_no_server_list_tools_names_no_lsp_tool(home, workspace, monkeypatch):
    _, _, intro = _make(workspace, monkeypatch)
    listed = _listed(intro)
    assert "list_tools" in listed  # the catalog itself is not empty
    assert not (listed & LSP_TOOL_NAMES), sorted(listed & LSP_TOOL_NAMES)


def test_with_no_server_get_tool_schemas_does_not_activate_them(
    home, workspace, monkeypatch,
):
    _, _, intro = _make(workspace, monkeypatch)
    result = intro._execute_get_tool_schemas(
        {"tool_ids": [name_to_id("lsp_get_diagnostics")]}
    )
    assert not result.get("schemas"), result
    assert result.get("not_found"), result


def test_a_profile_declared_server_lists_the_tools(home, workspace, monkeypatch):
    # Configured, never connected: the tools stay visible so a server
    # that failed to start can still be reported by the model.
    _, lsp, intro = _make(workspace, monkeypatch, {
        "languageServers": {"python": {"command": "no-such-lsp-binary"}},
    })
    assert not lsp._connected_servers
    assert LSP_TOOL_NAMES <= _listed(intro)


def test_a_workspace_lsp_json_lists_the_tools(home, workspace, monkeypatch):
    _write_lsp_json(workspace)
    _, _, intro = _make(workspace, monkeypatch)
    assert LSP_TOOL_NAMES <= _listed(intro)


def test_an_empty_profile_table_hides_them(home, workspace, monkeypatch):
    # A present-but-empty ``languageServers`` suppresses .lsp.json.
    _write_lsp_json(workspace)
    _, _, intro = _make(workspace, monkeypatch, {"languageServers": {}})
    assert not (_listed(intro) & LSP_TOOL_NAMES)


def test_a_lsp_json_written_mid_session_shows_the_tools(
    home, workspace, monkeypatch,
):
    _, _, intro = _make(workspace, monkeypatch)
    assert not (_listed(intro) & LSP_TOOL_NAMES)
    _write_lsp_json(workspace)
    assert LSP_TOOL_NAMES <= _listed(intro)


def test_the_system_instructions_follow_the_tools(home, workspace, monkeypatch):
    _, lsp, _ = _make(workspace, monkeypatch)
    assert lsp.get_system_instructions() is None
    _write_lsp_json(workspace)
    assert "lsp_get_diagnostics" in lsp.get_system_instructions()


def test_the_user_command_stays_available(home, workspace, monkeypatch):
    _, lsp, _ = _make(workspace, monkeypatch)
    assert any(c.name == "lsp" for c in lsp.get_user_commands())


def test_the_provider_tool_array_hides_them_too(home, workspace, monkeypatch):
    registry, lsp, _ = _make(workspace, monkeypatch)
    schemas = registry.get_exposed_tool_schemas()
    fake = SimpleNamespace(
        _provider=SimpleNamespace(uses_external_tools=lambda: True),
        _tools=schemas,
        _apply_tool_scopes=lambda s: s,
        _runtime=SimpleNamespace(registry=registry),
        _trace=lambda msg: None,
    )
    wire = {s.name for s in JaatoSession._get_tools_for_provider(fake)}
    assert "list_tools" in wire
    assert not (wire & LSP_TOOL_NAMES)
