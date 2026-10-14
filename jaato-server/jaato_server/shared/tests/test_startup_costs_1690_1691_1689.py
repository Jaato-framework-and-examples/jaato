"""Three costs every new session or process paid for nothing (#1690, #1691, #1689).

Found by the WASM spike (#1688), and real on a native install too.

* **#1690.**  When discovery skips a plugin for a missing dependency
  (``interactive_shell`` without ``pexpect``), the skip message names the
  extra to install.  Building that hint ran
  ``importlib.metadata.packages_distributions()`` once per requirement
  name: 46-49 full metadata walks, ~12-18 s per cold process.  The walk is
  now taken once and inverted into a distribution -> import names map.
* **#1691.**  ``MCPToolPlugin.initialize`` started a thread with its own
  event loop and waited for discovery even with no ``.mcp.json``;
  ``LSPToolPlugin`` started its loop thread with no language server too.
  Neither starts one now until a server is configured, and each starts one
  later when a server appears (``mcp reload`` / ``mcp connect`` /
  ``set_workspace_path``; for lsp also a ``.lsp.json`` written mid-session,
  #1345).
* **#1689.**  ``permission/channels.py`` imported ``readline``
  unconditionally, so the embedded client could not be imported where
  ``readline`` does not exist (Pyodide).  The import is guarded, and kept
  where it exists so ``input()`` keeps its line editing.

Every thread start is observed by replacing ``_ensure_thread`` with a
recorder, so no case here starts an MCP or LSP loop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from jaato_server.shared.plugins.lsp.plugin import LSPToolPlugin
from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
from jaato_server.shared.plugins.registry import RUNNER_HOSTED_SESSION_KEY
from jaato_server.shared.scaffold import dependencies as D
from jaato_server.shared.tests.reversion import Reversion

_DEPS = "jaato-server/jaato_server/shared/scaffold/dependencies.py"
_MCP = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_LSP = "jaato-server/jaato_server/shared/plugins/lsp/plugin.py"
_CHANNELS = "jaato-server/jaato_server/shared/plugins/permission/channels.py"

REVERSIONS = [
    Reversion(
        target=_DEPS,
        find="    measured = _import_names_by_dist().get(key, ())\n",
        replace=("    measured = tuple(sorted(m for m, ds in packages_distributions().items()\n"
                 "                            if any(_dist_key(d) == key for d in ds)))\n"),
        test="TestTheInstallHint::test_the_index_walks_metadata_once",
        because="every requirement name walks every installed distribution again",
    ),
    Reversion(
        target=_DEPS,
        find="               _provided_import_names, _import_names_by_dist):\n",
        replace="               _provided_import_names):\n",
        test="TestTheInstallHint::test_reset_drops_the_measured_map",
        because="a test that fakes a distribution would read the previous map",
    ),
    Reversion(
        target=_MCP,
        find="        elif not self._read_startup_servers():\n",
        replace="        elif False:\n",
        test="TestMcpStartsNothingForNothing::test_no_config_starts_no_thread",
        because="every session starts a loop thread and waits on discovery of nothing",
    ),
    Reversion(
        target=_MCP,
        find="        self._ensure_thread()\n        return True\n",
        replace="        return True\n",
        test="TestMcpStartsNothingForNothing::test_reload_starts_it_once_a_server_is_configured",
        because="a server added after start could never be connected",
    ),
    Reversion(
        target=_MCP,
        find=("                    and self._read_startup_servers()):\n"
              "                self._ensure_thread()\n"),
        replace=("                    and self._read_startup_servers()):\n"
                 "                pass\n"),
        test="TestMcpStartsNothingForNothing::test_a_workspace_with_servers_starts_it_later",
        because="a session whose workspace arrives after initialize keeps no MCP tools",
    ),
    Reversion(
        target=_LSP,
        find="        elif not self._has_configured_servers():\n",
        replace="        elif False:\n",
        test="TestLspStartsNothingForNothing::test_no_config_starts_no_thread",
        because="every session starts an LSP loop thread with no server to serve",
    ),
    Reversion(
        target=_LSP,
        find="        if self._start_thread_if_configured():\n            return\n",
        replace="        if False:\n            return\n",
        test="TestLspStartsNothingForNothing::test_a_lsp_json_written_mid_session_starts_it",
        because="a .lsp.json written mid-session (#1345) would show tools nothing serves",
    ),
    Reversion(
        target=_CHANNELS,
        find="try:\n    import readline\nexcept ImportError:\n    readline = None\n",
        replace="import readline\n",
        test="TestReadlineIsOptional::test_the_embedded_client_imports_without_readline",
        because="the embedded client cannot be imported where readline does not exist",
    ),
    Reversion(
        target=_CHANNELS,
        find="try:\n    import readline\nexcept ImportError:\n",
        replace="try:\n    raise ImportError\nexcept ImportError:\n",
        test="TestReadlineIsOptional::test_readline_is_still_imported_where_it_exists",
        because="input() would lose the line editing importing readline turns on",
    ),
]


# ------------------------------------------------------------------ #1690


class TestTheInstallHint:
    """``packages_distributions()`` is walked once, and answers as before."""

    @pytest.fixture(autouse=True)
    def _fresh(self):
        D.reset_metadata_caches()
        yield
        D.reset_metadata_caches()

    def _count_walks(self, monkeypatch) -> List[int]:
        calls = [0]
        real = D.packages_distributions

        def counting():
            calls[0] += 1
            return real()

        monkeypatch.setattr(D, "packages_distributions", counting)
        return calls

    def test_the_index_walks_metadata_once(self, monkeypatch):
        calls = self._count_walks(monkeypatch)
        D._extras_index()
        D._core_index()
        assert calls[0] == 1

    def test_the_skip_message_walks_metadata_once(self, monkeypatch):
        from jaato_server.shared.plugins.registry import _install_advice

        calls = self._count_walks(monkeypatch)
        advice = _install_advice("pexpect")
        assert calls[0] <= 1
        assert advice.endswith(".")

    def test_names_match_the_per_name_walk(self, monkeypatch):
        """The inverted map answers each name as the per-name walk did."""
        fake: Dict[str, List[str]] = {
            "bs4": ["beautifulsoup4"], "yaml": ["PyYAML"],
            "google": ["google-genai", "google_api_core"],
            "_cffi_backend": ["cffi"], "cffi": ["cffi"],
        }
        monkeypatch.setattr(D, "packages_distributions", lambda: fake)

        def per_name_walk(dist: str) -> Tuple[str, ...]:
            key = dist.lower().replace("_", "-")
            return tuple(sorted(m for m, ds in fake.items()
                                if any(d.lower().replace("_", "-") == key for d in ds)))

        for dist in ("beautifulsoup4", "pyyaml", "google-genai",
                     "google-api-core", "cffi"):
            assert D._provided_import_names(dist) == per_name_walk(dist), dist
        # A name nothing measured still falls to the name-match guesses.
        assert "azure" in D._provided_import_names("azure-identity")

    def test_reset_drops_the_measured_map(self, monkeypatch):
        monkeypatch.setattr(D, "packages_distributions", lambda: {"one": ["pkg-x"]})
        assert D._provided_import_names("pkg-x") == ("one",)
        monkeypatch.setattr(D, "packages_distributions", lambda: {"two": ["pkg-x"]})
        D.reset_metadata_caches()
        assert D._provided_import_names("pkg-x") == ("two",)


# ------------------------------------------------------------------ #1691


def _record_starts(cls, monkeypatch) -> List[bool]:
    started: List[bool] = []
    monkeypatch.setattr(cls, "_ensure_thread", lambda self: started.append(True))
    return started


@pytest.fixture
def ws(tmp_path, monkeypatch) -> Path:
    """An empty workspace; HOME points into it so ``~/.mcp.json`` cannot answer."""
    monkeypatch.setenv("HOME", str(tmp_path))
    work = tmp_path / "ws"
    work.mkdir()
    return work


def _write_mcp(path: Path) -> None:
    (path / ".mcp.json").write_text(
        '{"mcpServers": {"s": {"type": "stdio", "command": "true"}}}')


def _write_lsp(path: Path) -> None:
    (path / ".lsp.json").write_text(
        '{"languageServers": {"s": {"command": "true", "extensions": [".x"]}}}')


class TestMcpStartsNothingForNothing:

    def test_no_config_starts_no_thread(self, ws, monkeypatch):
        started = _record_starts(MCPToolPlugin, monkeypatch)
        plugin = MCPToolPlugin()
        plugin.initialize({"workspace_path": str(ws)})
        assert started == [] and plugin._initialized
        assert plugin._thread is None
        assert plugin.get_tool_schemas() == []

    def test_a_configured_server_starts_it(self, ws, monkeypatch):
        """Control: with a server, initialize still starts the thread."""
        started = _record_starts(MCPToolPlugin, monkeypatch)
        _write_mcp(ws)
        MCPToolPlugin().initialize({"workspace_path": str(ws)})
        assert started == [True]

    def test_reload_starts_it_once_a_server_is_configured(self, ws, monkeypatch):
        started = _record_starts(MCPToolPlugin, monkeypatch)
        plugin = MCPToolPlugin()
        plugin.initialize({"workspace_path": str(ws)})
        _write_mcp(ws)
        message = plugin._cmd_reload()
        assert started == [True]
        assert message.startswith("Reloaded configuration")

    def test_a_workspace_with_servers_starts_it_later(self, ws, tmp_path, monkeypatch):
        started = _record_starts(MCPToolPlugin, monkeypatch)
        plugin = MCPToolPlugin()
        plugin.initialize({"workspace_path": str(ws)})
        other = tmp_path / "other"
        other.mkdir()
        _write_mcp(other)
        plugin.set_workspace_path(str(other))
        assert started == [True]

    def test_the_mirror_never_starts_it(self, ws, tmp_path, monkeypatch):
        started = _record_starts(MCPToolPlugin, monkeypatch)
        _write_mcp(ws)
        plugin = MCPToolPlugin()
        plugin.initialize({"workspace_path": str(ws), RUNNER_HOSTED_SESSION_KEY: True})
        assert plugin._start_for_command() is False
        other = tmp_path / "other"
        other.mkdir()
        _write_mcp(other)
        plugin.set_workspace_path(str(other))
        assert started == []


class TestLspStartsNothingForNothing:

    def test_no_config_starts_no_thread(self, ws, monkeypatch):
        started = _record_starts(LSPToolPlugin, monkeypatch)
        plugin = LSPToolPlugin()
        plugin.initialize({"workspace_path": str(ws)})
        assert started == [] and plugin._initialized

    def test_a_configured_server_starts_it(self, ws, monkeypatch):
        started = _record_starts(LSPToolPlugin, monkeypatch)
        _write_lsp(ws)
        LSPToolPlugin().initialize({"workspace_path": str(ws)})
        assert started == [True]

    def test_a_lsp_json_written_mid_session_starts_it(self, ws, monkeypatch):
        started = _record_starts(LSPToolPlugin, monkeypatch)
        plugin = LSPToolPlugin()
        plugin.initialize({"workspace_path": str(ws)})
        assert plugin.is_tool_visible("lsp_hover") is False
        _write_lsp(ws)
        assert plugin.is_tool_visible("lsp_hover") is True
        assert started == [True]

    def test_the_mirror_never_starts_it(self, ws, monkeypatch):
        started = _record_starts(LSPToolPlugin, monkeypatch)
        plugin = LSPToolPlugin()
        plugin.initialize({"workspace_path": str(ws), RUNNER_HOSTED_SESSION_KEY: True})
        _write_lsp(ws)
        assert plugin.is_tool_visible("lsp_hover") is True
        assert started == []


# ------------------------------------------------------------------ #1689


_BLOCK_READLINE = textwrap.dedent("""
    import importlib.abc, sys
    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "readline":
                raise ModuleNotFoundError("No module named 'readline'", name=name)
    sys.modules.pop("readline", None)
    sys.meta_path.insert(0, _Block())
""")


class TestReadlineIsOptional:

    def test_the_embedded_client_imports_without_readline(self):
        code = _BLOCK_READLINE + textwrap.dedent("""
            import jaato_embedded.client
            from jaato_server.shared.plugins.permission import channels
            assert channels.readline is None
            ch = channels.ConsoleChannel()
            ch.initialize({"input_func": lambda: "y"})
            assert ch._read_input() == "y"
            print("ok")
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=300, env=dict(os.environ))
        assert proc.returncode == 0 and "ok" in proc.stdout, proc.stderr[-2000:]

    def test_readline_is_still_imported_where_it_exists(self):
        real = pytest.importorskip("readline")
        from jaato_server.shared.plugins.permission import channels
        assert channels.readline is real
