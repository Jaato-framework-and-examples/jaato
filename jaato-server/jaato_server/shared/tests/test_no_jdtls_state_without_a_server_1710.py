"""The jdtls state sibling is made only for a server that uses it (#1710).

``${jdtlsStateRoot}`` is ``<workspace.parent>/.<basename>-jdtls-state``: a
directory beside the workspace, in a folder the application owns.  The
daemon created it for every session with a workspace, whatever the session
could run, and nothing removed it, so an application listing its own
folder found entries it never made.

Now the daemon makes it only when the session's LSP configuration names
``${jdtlsStateRoot}``, the lsp plugin makes it at server start when it is
missing, and ``workspace.delete`` removes it with the workspace.
"""

from __future__ import annotations

import json
import os

import pytest

from jaato_server.server.core import JaatoServer
from jaato_server.server.workspace_manager import WorkspaceManager
from jaato_server.shared.jdtls_state import (
    jdtls_state_root_of,
    remove_jdtls_state_root,
)
from jaato_server.shared.plugins.lsp.plugin import LSPToolPlugin
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_MANAGER = "jaato-server/jaato_server/server/workspace_manager.py"
_LSP = "jaato-server/jaato_server/shared/plugins/lsp/plugin.py"

REVERSIONS = [
    Reversion(
        target=_CORE,
        find="        if not lsp_config_uses_jdtls_state(\n",
        replace="        if False and not lsp_config_uses_jdtls_state(\n",
        because="every session with a workspace would get the sibling again",
        test="TestProvisioning::test_no_lsp_configuration_makes_nothing",
    ),
    Reversion(
        target=_MANAGER,
        find="        remove_jdtls_state_root(str(path))\n",
        replace="",
        because="deleting a workspace would leave its sibling behind",
        test="TestDelete::test_workspace_delete_removes_the_sibling",
    ),
    Reversion(
        target=_LSP,
        find="        error = ensure_jdtls_state_root(root)\n",
        replace="        error = None\n",
        because="a server configured after session start would find no -data dir",
        test="TestLazy::test_server_start_makes_a_missing_sibling",
    ),
]


@pytest.fixture(autouse=True)
def _no_home_lsp_json(tmp_path, monkeypatch):
    """Keep a developer's ``~/.lsp.json`` out of the predicate."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _workspace(tmp_path, name="pipeline-1"):
    parent = tmp_path / "apps"
    parent.mkdir(exist_ok=True)
    ws = parent / name
    ws.mkdir()
    return ws


def _lsp_json(ws, args):
    (ws / ".lsp.json").write_text(json.dumps(
        {"languageServers": {"srv": {"command": "x", "args": args}}}))


def _resolve(ws, profile=None):
    server = JaatoServer(env_file=None, workspace_path=str(ws))
    server._profile = profile
    server._resolve_session_env()
    return server


class _Profile:
    """The attributes ``_resolve_session_env`` reads off a profile."""

    def __init__(self, plugin_configs):
        self.plugin_configs = plugin_configs
        self.env = None
        self.trace = None
        self.record_keeping = None


class TestProvisioning:
    def test_no_lsp_configuration_makes_nothing(self, tmp_path):
        ws = _workspace(tmp_path)
        server = _resolve(ws)
        assert sorted(os.listdir(ws.parent)) == [ws.name]
        assert "JAATO_JDTLS_STATE_DIR" not in server._session_env

    def test_a_server_that_does_not_name_it_makes_nothing(self, tmp_path):
        ws = _workspace(tmp_path)
        _lsp_json(ws, ["--stdio"])
        _resolve(ws)
        assert sorted(os.listdir(ws.parent)) == [ws.name]

    def test_lsp_json_naming_it_makes_it(self, tmp_path):
        ws = _workspace(tmp_path)
        _lsp_json(ws, ["-data", "${jdtlsStateRoot}"])
        server = _resolve(ws)
        root = jdtls_state_root_of(str(ws))
        assert os.path.isdir(root)
        assert server._session_env["JAATO_JDTLS_STATE_DIR"] == root

    def test_profile_table_naming_it_makes_it(self, tmp_path):
        ws = _workspace(tmp_path)
        profile = _Profile({"lsp": {"languageServers": {"java": {
            "command": "jdtls", "args": ["-data", "${jdtlsStateRoot}"]}}}})
        _resolve(ws, profile)
        assert os.path.isdir(jdtls_state_root_of(str(ws)))

    def test_profile_table_suppresses_the_file(self, tmp_path):
        """A declared table IS the configuration; ``.lsp.json`` is not read."""
        ws = _workspace(tmp_path)
        _lsp_json(ws, ["-data", "${jdtlsStateRoot}"])
        profile = _Profile({"lsp": {"languageServers": {}}})
        _resolve(ws, profile)
        assert sorted(os.listdir(ws.parent)) == [ws.name]


class TestDelete:
    def test_workspace_delete_removes_the_sibling(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        manager = WorkspaceManager(str(root),
                                   registry_path=tmp_path / "registry.json")
        ws = root / "mine"
        ws.mkdir()
        sibling = jdtls_state_root_of(str(ws))
        os.makedirs(os.path.join(sibling, "jdt.ls"))
        manager.delete_workspace("mine")
        assert not os.path.exists(ws)
        assert not os.path.exists(sibling)

    def test_a_link_at_the_name_is_left_alone(self, tmp_path):
        ws = _workspace(tmp_path)
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "keep").write_text("x")
        os.symlink(target, jdtls_state_root_of(str(ws)))
        assert remove_jdtls_state_root(str(ws)) is False
        assert (target / "keep").exists()

    def test_no_sibling_is_not_an_error(self, tmp_path):
        ws = _workspace(tmp_path)
        assert remove_jdtls_state_root(str(ws)) is False


class TestLazy:
    def _plugin(self, ws):
        plugin = LSPToolPlugin()
        plugin._workspace_path = str(ws)
        return plugin

    def test_server_start_makes_a_missing_sibling(self, tmp_path):
        ws = _workspace(tmp_path)
        self._plugin(ws)._ensure_jdtls_state_root(
            ["-data", "${jdtlsStateRoot}"])
        assert os.path.isdir(jdtls_state_root_of(str(ws)))

    def test_server_not_naming_it_makes_nothing(self, tmp_path):
        ws = _workspace(tmp_path)
        self._plugin(ws)._ensure_jdtls_state_root(["--stdio"])
        assert sorted(os.listdir(ws.parent)) == [ws.name]
