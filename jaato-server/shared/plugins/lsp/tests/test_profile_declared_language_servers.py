"""The server table declared by the PROFILE, and what declaring it suppresses.

`plugin_configs.lsp.languageServers` carries the same mapping `.lsp.json`
carries under the same key.  The rule these tests pin — one rule, obeyed by
both readers, `_load_config_cache` (runtime) and `_load_lsp_config_static`
(apparmor composer):

    key absent  -> the file search runs, exactly as before
    key present -> the profile IS the configuration; NO file is read

The two readers agreeing is not a nicety: the composer turns each server's
`command` into an `ix` grant, so a server the composer cannot see is a
server the confined runner cannot exec.  Every "the file is not read" test
below therefore plants a DIFFERENT server in a real `.lsp.json` — if the
suppression regressed, the assertion sees that other server's name rather
than an empty result, which distinguishes "file ignored" from "nothing
loaded at all".
"""
import json
import logging
import os

import pytest
from unittest.mock import patch

from ..plugin import LSPToolPlugin


JDTLS = {"command": "jdtls", "args": ["--stdio"], "languageId": "java"}
PYRIGHT = {"command": "pyright-langserver", "args": ["--stdio"],
           "languageId": "python"}


def _plugin(config, *, home=None):
    """An initialized plugin with the background thread suppressed.

    `initialize` starts the LSP background thread, whose auto-connect would
    try to spawn real language servers; every knob this module tests is
    resolved before that point.
    """
    plugin = LSPToolPlugin()
    with patch.object(plugin, "_ensure_thread"):
        plugin.initialize(config)
    return plugin


def _write_lsp_json(directory, servers):
    path = os.path.join(str(directory), ".lsp.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"languageServers": servers}, f)
    return path


# =============================================================================
# Runtime reader
# =============================================================================

class TestRuntimeSourceSelection:

    def test_absent_key_still_reads_the_workspace_file(self, tmp_path):
        """The untouched path: a workspace that has always used .lsp.json."""
        _write_lsp_json(tmp_path, {"python": PYRIGHT})
        plugin = _plugin({"workspace_path": str(tmp_path)})

        plugin._load_config_cache()

        assert plugin._profile_config is None
        assert set(plugin._config_cache["languageServers"]) == {"python"}
        assert plugin._config_path == os.path.join(str(tmp_path), ".lsp.json")

    def test_a_declared_table_suppresses_the_workspace_file(self, tmp_path):
        _write_lsp_json(tmp_path, {"python": PYRIGHT})
        plugin = _plugin({
            "workspace_path": str(tmp_path),
            "languageServers": {"java": JDTLS},
        })

        plugin._load_config_cache()

        servers = plugin._config_cache["languageServers"]
        assert set(servers) == {"java"}, "the workspace .lsp.json was read"
        assert servers["java"]["command"] == "jdtls"
        # No file answered, so there is no path to report as the source.
        assert plugin._config_path is None

    def test_a_declared_table_suppresses_config_path_too(self, tmp_path):
        """`config_path` is the highest file tier; the profile outranks it."""
        elsewhere = tmp_path / "stack"
        elsewhere.mkdir()
        path = _write_lsp_json(elsewhere, {"python": PYRIGHT})
        plugin = _plugin({
            "workspace_path": str(tmp_path),
            "config_path": path,
            "languageServers": {"java": JDTLS},
        })

        plugin._load_config_cache()

        assert set(plugin._config_cache["languageServers"]) == {"java"}

    def test_a_declared_table_suppresses_the_home_file(self, tmp_path, monkeypatch):
        """~/.lsp.json is the last file tier — and is not consulted either."""
        home = tmp_path / "home"
        home.mkdir()
        _write_lsp_json(home, {"python": PYRIGHT})
        monkeypatch.setenv("HOME", str(home))

        plugin = _plugin({"languageServers": {"java": JDTLS}})
        plugin._load_config_cache()

        assert set(plugin._config_cache["languageServers"]) == {"java"}

    def test_present_and_empty_declares_no_servers_and_reads_no_file(self, tmp_path):
        """`{}` is a declaration, not an absence: this profile runs none.

        Distinguishing the two is the whole reason the rule keys on the
        PRESENCE of the key rather than on its truthiness.
        """
        _write_lsp_json(tmp_path, {"python": PYRIGHT})
        plugin = _plugin({
            "workspace_path": str(tmp_path),
            "languageServers": {},
        })

        plugin._load_config_cache()

        assert plugin._config_cache == {"languageServers": {}}
        assert plugin._profile_config is not None

    def test_a_non_mapping_value_is_an_authoring_error_not_a_file_fallback(
        self, tmp_path, caplog
    ):
        """Falling back to the file would make the declaration silently inert."""
        _write_lsp_json(tmp_path, {"python": PYRIGHT})
        with caplog.at_level(logging.ERROR):
            plugin = _plugin({
                "workspace_path": str(tmp_path),
                "languageServers": ["java"],
            })
            plugin._load_config_cache()

        assert plugin._config_cache == {"languageServers": {}}
        assert "must be a mapping" in caplog.text

    def test_a_non_mapping_entry_is_dropped_by_name_and_siblings_survive(
        self, caplog
    ):
        with caplog.at_level(logging.ERROR):
            plugin = _plugin({
                "languageServers": {"java": JDTLS, "broken": "jdtls"},
            })
            plugin._load_config_cache()

        assert set(plugin._config_cache["languageServers"]) == {"java"}
        assert "broken" in caplog.text

    def test_a_declared_spec_reaches_connect_unexpanded(self, tmp_path):
        """`${workspaceRoot}` must survive `initialize`.

        `initialize` runs BEFORE the framework's `set_workspace_path`
        broadcast, so expanding a spec there resolves ${workspaceRoot}
        against the daemon cwd — the asymmetry PR-157 closed for args
        loaded from the file.  `connect_server` is the one site that
        expands, with `workspace_root_override`.
        """
        plugin = _plugin({
            "languageServers": {
                "java": {"command": "jdtls",
                         "args": ["-data", "${workspaceRoot}/.jaato/jdtls-data"]},
            },
        })
        plugin._load_config_cache()

        args = plugin._config_cache["languageServers"]["java"]["args"]
        assert args == ["-data", "${workspaceRoot}/.jaato/jdtls-data"]

    def test_the_debug_log_names_the_profile_as_the_source(self, tmp_path):
        """The operator's only window into which source actually answered."""
        log = tmp_path / "logs" / "lsp_debug.log"
        plugin = _plugin({
            "workspace_path": str(tmp_path),
            "debug_log_path": str(log),
            "languageServers": {"java": JDTLS},
        })
        plugin._load_config_cache()

        text = log.read_text()
        assert "plugin_configs.lsp.languageServers" in text
        assert "Server 'java': command=jdtls" in text


# =============================================================================
# Apparmor composer — the reader that must agree with the one above
# =============================================================================

class TestApparmorComposerAgrees:

    def test_it_emits_an_exec_grant_for_a_profile_declared_command(self, tmp_path):
        """Without this the confined runner cannot exec the server at all."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="s1",
            config_root=None,
            plugin_config={"languageServers": {"sh": {"command": "/bin/sh"}}},
        )

        assert any(r.startswith(os.path.realpath("/bin/sh")) and r.endswith("ix,")
                   for r in rules), rules

    def test_it_ignores_the_file_when_the_profile_declares(self, tmp_path):
        _write_lsp_json(tmp_path, {"python": {"command": "/bin/cat"}})

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="s1",
            config_root=None,
            plugin_config={"languageServers": {"sh": {"command": "/bin/sh"}}},
        )

        cat = os.path.realpath("/bin/cat")
        assert not any(r.startswith(cat) for r in rules), rules

    def test_it_still_reads_the_file_when_the_profile_declares_nothing(self, tmp_path):
        _write_lsp_json(tmp_path, {"sh": {"command": "/bin/sh"}})

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="s1",
            config_root=None,
            plugin_config={},
        )

        assert any(r.startswith(os.path.realpath("/bin/sh")) and r.endswith("ix,")
                   for r in rules), rules

    def test_data_dir_grants_are_emitted_from_a_profile_declared_spec(self, tmp_path):
        """`-data <path>` grants come off the args of whichever source won."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="s1",
            config_root=None,
            plugin_config={"languageServers": {"sh": {
                "command": "/bin/sh",
                "args": ["-data", "${workspaceRoot}/.jaato/jdtls-data"],
            }}},
        )

        expected = os.path.join(str(tmp_path), ".jaato/jdtls-data")
        assert f"{expected}/**  rw," in rules, rules


# =============================================================================
# What the operator is told
# =============================================================================

class TestOperatorFacingText:

    def test_reload_does_not_claim_to_have_checked_a_file(self):
        plugin = _plugin({"languageServers": {"java": JDTLS}})
        plugin._initialized = True

        out = plugin._cmd_reload()

        assert "nothing to reload" in out
        assert "1 server(s) declared" in out

    def test_an_empty_profile_table_is_not_told_to_create_a_file(self, tmp_path):
        plugin = _plugin({
            "workspace_path": str(tmp_path),
            "languageServers": {},
        })

        out = plugin._cmd_list()

        assert "plugin_configs.lsp.languageServers" in out
        assert ".lsp.json" not in out, "advice points at a file this session ignores"

    def test_without_a_declaration_the_file_is_still_advertised(self, tmp_path):
        plugin = _plugin({"workspace_path": str(tmp_path)})

        out = plugin._cmd_list()

        assert ".lsp.json" in out

    def test_the_knob_is_declared_in_the_config_schema(self):
        """`jaato-scaffold explain` renders this; an undeclared knob is invisible."""
        names = [s.name for s in LSPToolPlugin().get_config_schema()]
        assert "languageServers" in names
