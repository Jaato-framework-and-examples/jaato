"""Tests for the file_edit plugin."""

import os
import tempfile
from pathlib import Path

import pytest

from ..plugin import FileEditPlugin, create_plugin


class TestFileEditPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, FileEditPlugin)

    def test_plugin_name(self):
        plugin = FileEditPlugin()
        assert plugin.name == "file_edit"

    def test_initialize_without_config_raises(self):
        """Server 0.6.130+ (PR-147): no config_root + no explicit
        backup_dir → RuntimeError.  Pre-PR-147 a workspace-CWD
        fallback masked this; per Daniel's "framework confined to
        config_root, workspace belongs to tenant" rule, missing
        config_root is now a configuration error (loud, not
        silent)."""
        plugin = FileEditPlugin()
        with pytest.raises(RuntimeError, match="cannot resolve backup base directory"):
            plugin.initialize()

    def test_initialize_with_custom_backup_dir(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "custom_backups"
        plugin.initialize({"backup_dir": str(backup_dir)})
        assert plugin._initialized is True
        assert plugin._backup_manager._base_dir == backup_dir

    def test_initialize_with_session_id_anchors_on_config_root(self, tmp_path):
        """Server 0.6.127+: session_id creates session-scoped backup
        directory anchored on ``config_root`` (NOT workspace_root).

        Backups are jaato-meta-state and belong with other meta-state
        (``.jaato/logs/``, ``.jaato/cache/`` etc.) under config_root,
        NOT polluting the workspace with a second ``.jaato/`` tree.
        PR-143 mistakenly anchored on workspace_root; PR-144 corrects.
        """
        config_root_dir = tmp_path / ".jaato"
        config_root_dir.mkdir()
        plugin = FileEditPlugin()
        plugin.initialize({
            "workspace_root": str(tmp_path),
            "config_root": str(config_root_dir),
            "session_id": "test-session-123",
        })
        assert plugin._initialized is True
        expected_path = (
            config_root_dir / "sessions" / "test-session-123" / "backups"
        ).resolve()
        assert plugin._backup_manager._base_dir == expected_path

    def test_initialize_session_id_ignores_cwd(self, tmp_path, monkeypatch):
        """Pin: when config_root is set, the daemon's CWD does NOT
        influence backup-path resolution.  Regression guard for the
        v122 wire-gap (CWD-relative resolution).
        """
        decoy_dir = tmp_path / "decoy_cwd"
        workspace_dir = tmp_path / "real_workspace"
        config_root_dir = tmp_path / "kb" / ".jaato"
        decoy_dir.mkdir()
        workspace_dir.mkdir()
        config_root_dir.mkdir(parents=True)
        monkeypatch.chdir(decoy_dir)

        plugin = FileEditPlugin()
        plugin.initialize({
            "workspace_root": str(workspace_dir),
            "config_root": str(config_root_dir),
            "session_id": "sess-1",
        })
        assert plugin._backup_manager._base_dir == (
            config_root_dir / "sessions" / "sess-1" / "backups"
        ).resolve()
        # Sanity: NOT under the decoy CWD or the workspace.
        assert decoy_dir not in plugin._backup_manager._base_dir.parents
        assert workspace_dir not in plugin._backup_manager._base_dir.parents

    def test_initialize_no_session_id_anchors_on_config_root(self, tmp_path):
        """Pin: even without session_id, the default ``backups``
        path anchors on config_root (not workspace, not CWD)."""
        config_root_dir = tmp_path / ".jaato"
        config_root_dir.mkdir()
        plugin = FileEditPlugin()
        plugin.initialize({
            "workspace_root": str(tmp_path),
            "config_root": str(config_root_dir),
        })
        assert plugin._backup_manager._base_dir == (
            config_root_dir / "backups"
        ).resolve()

    def test_initialize_no_config_root_raises(self, tmp_path):
        """Server 0.6.130+ (PR-147): no config_root → RuntimeError.

        Per the framework architectural rule (Daniel, 2026-05-19):
        framework + plugins are confined to config_root; workspace
        is tenant territory and must not receive framework-side
        writes.  Pre-PR-147 a workspace-fallback masked the missing
        config_root; PR-147 makes the configuration error loud.
        """
        plugin = FileEditPlugin()
        with pytest.raises(RuntimeError, match="cannot resolve backup base directory"):
            plugin.initialize({
                "workspace_root": str(tmp_path),
                "session_id": "sess-1",
            })

    def test_set_config_root_broadcast_reinit_backup_manager(self, tmp_path):
        """Pin: registry's set_config_root broadcast actually moves
        the backup root, not just updates the in-memory value.

        Mirrors how set_workspace_path already works on the 5 sibling
        plugins (references, subagent, prompt_library, template,
        service_connector).
        """
        # Initialize with config_root_A
        config_root_a = tmp_path / "kb_a" / ".jaato"
        config_root_a.mkdir(parents=True)
        plugin = FileEditPlugin()
        plugin.initialize({
            "config_root": str(config_root_a),
            "session_id": "sess-1",
        })
        assert plugin._backup_manager._base_dir == (
            config_root_a / "sessions" / "sess-1" / "backups"
        ).resolve()

        # Broadcast a different config_root — backup root must move.
        config_root_b = tmp_path / "kb_b" / ".jaato"
        config_root_b.mkdir(parents=True)
        plugin.set_config_root(str(config_root_b))
        assert plugin._backup_manager._base_dir == (
            config_root_b / "sessions" / "sess-1" / "backups"
        ).resolve()

    def test_set_workspace_path_does_not_affect_backup_anchor(self, tmp_path):
        """Server 0.6.130+ (PR-147): backup anchor is config_root only.

        Pre-PR-147 a ``set_workspace_path`` broadcast would re-anchor
        backups on the workspace fallback when config_root was unset.
        PR-147 drops the workspace-fallback per Daniel's
        "framework confined to config_root" rule.  workspace changes
        should NOT move the backup root.
        """
        config_root_a = tmp_path / "kb" / ".jaato"
        config_root_a.mkdir(parents=True)
        ws_a = tmp_path / "ws_a"
        ws_a.mkdir()
        plugin = FileEditPlugin()
        plugin.initialize({
            "config_root": str(config_root_a),
            "workspace_root": str(ws_a),
            "session_id": "sess-1",
        })
        assert plugin._backup_manager._base_dir == (
            config_root_a / "sessions" / "sess-1" / "backups"
        ).resolve()

        # Workspace change broadcast — backup root stays anchored on config_root.
        ws_b = tmp_path / "ws_b"
        ws_b.mkdir()
        plugin.set_workspace_path(str(ws_b))
        assert plugin._backup_manager._base_dir == (
            config_root_a / "sessions" / "sess-1" / "backups"
        ).resolve()

    def test_explicit_backup_dir_wins_over_anchors(self, tmp_path):
        """Operator-provided ``backup_dir`` wins regardless of
        config_root / workspace_root / set_*_path broadcasts."""
        explicit = tmp_path / "explicit"
        config_root_dir = tmp_path / ".jaato"
        config_root_dir.mkdir()
        plugin = FileEditPlugin()
        plugin.initialize({
            "backup_dir": str(explicit),
            "config_root": str(config_root_dir),
            "session_id": "sess-1",
        })
        assert plugin._backup_manager._base_dir == explicit
        # Broadcast a new config_root — operator's explicit override sticks.
        new_cr = tmp_path / "kb" / ".jaato"
        new_cr.mkdir(parents=True)
        plugin.set_config_root(str(new_cr))
        assert plugin._backup_manager._base_dir == explicit


class TestFileEditApparmorRules:
    """Pin: ``get_apparmor_rules`` declares the full sessions/ subtree
    grant for backup writes under config_root.

    Server 0.6.130+ (PR-147): rule shape grants the full
    ``<config_root>/sessions/`` subtree (parent + ``**`` descendants)
    via AppArmor specificity to override the framework template's
    read-only ``<config_root>/** r,`` baseline.  PR-145's leaf-only
    rules failed because they didn't grant mkdir of the ``sessions/``
    parent itself.  v126 evidence: PermissionError on
    ``<config_root>/sessions`` (the parent, not the leaf).

    Workspace branch dropped per Daniel's "framework + plugins
    confined to config_root; workspace is tenant territory" rule —
    file_edit's framework-side writes go in config_root only.
    """

    def test_config_root_grants_full_sessions_subtree(self, tmp_path):
        """Pin the structural fix: grant the sessions/ dir + all
        descendants.  Specificity wins over framework template's
        ``<config_root>/** r,``."""
        cr = str(tmp_path / "kb" / ".jaato")
        rules = FileEditPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path / "ws"),
            session_id="sess-42",
            config_root=cr,
            plugin_config={},
        )
        # Two rules: sessions/ dir entry + ** descendants
        assert f"{cr}/sessions/    rw," in rules
        assert f"{cr}/sessions/**  rw," in rules
        # NO workspace-anchored rules (workspace is tenant territory)
        assert not any("/ws/" in r for r in rules)

    def test_no_config_root_no_rules_emitted(self, tmp_path):
        """When config_root is unset, no rules emitted.  Plugin
        layer doesn't grant workspace writes — that's the framework
        template's territory and is for AGENT-mediated tenant
        operations, not framework-side backup writes."""
        rules = FileEditPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path / "ws"),
            session_id="sess-1",
            config_root=None,
            plugin_config={},
        )
        assert rules == []

    def test_explicit_backup_dir_grants_operator_path(self, tmp_path):
        """Operator-explicit ``backup_dir`` emits a grant for whatever
        path the operator chose; operator owns ensuring it falls
        under the active apparmor confinement."""
        explicit = str(tmp_path / "operator_chosen_backups")
        rules = FileEditPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path / "ws"),
            session_id="sess-1",
            config_root=str(tmp_path / "kb" / ".jaato"),
            plugin_config={"backup_dir": explicit},
        )
        assert f"{explicit}/    rw," in rules
        assert f"{explicit}/**  rw," in rules

    def test_rules_are_strings_with_trailing_comma(self, tmp_path):
        """Pin: rule entries are strings ending with ','.  The
        apparmor template concatenates them into the .rules file
        verbatim, so syntax must be valid AppArmor rule lines."""
        rules = FileEditPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="s",
            config_root=str(tmp_path / ".jaato"),
            plugin_config={},
        )
        assert all(isinstance(r, str) for r in rules)
        assert all(r.endswith(",") for r in rules)
        # Pin the EXACT rule shape — subtree grants, not leaf-only.
        cr = str(tmp_path / ".jaato")
        assert rules == [
            f"{cr}/sessions/    rw,",
            f"{cr}/sessions/**  rw,",
        ]


class TestFileEditConfigDictAnchors:
    """Pin: when initialize receives config_root in config dict (per
    PR-146 PluginRegistry pre-init injection), the BackupManager
    anchors directly on config_root without needing a post-init
    broadcast.  Server 0.6.130+ (PR-147): no workspace-fallback —
    missing config_root raises rather than masking with a CWD or
    workspace path."""

    def test_init_with_config_root_anchors_directly(self, tmp_path):
        """config_root in config → BackupManager anchored on it."""
        config_root_dir = tmp_path / ".jaato"
        config_root_dir.mkdir()
        plugin = FileEditPlugin()
        plugin.initialize({
            "config_root": str(config_root_dir),
            "session_id": "sess-1",
        })
        assert plugin._backup_manager._base_dir == (
            config_root_dir / "sessions" / "sess-1" / "backups"
        ).resolve()

    def test_initialize_backup_dir_takes_precedence_over_session_id(self, tmp_path):
        """Test that explicit backup_dir takes precedence over session_id."""
        plugin = FileEditPlugin()
        custom_dir = tmp_path / "explicit_backups"
        plugin.initialize({
            "backup_dir": str(custom_dir),
            "session_id": "should-be-ignored"
        })
        assert plugin._initialized is True
        # Explicit backup_dir should win
        assert plugin._backup_manager._base_dir == custom_dir

    def test_shutdown(self, tmp_path):
        plugin = FileEditPlugin()
        # Use explicit backup_dir so initialize() doesn't require
        # config_root (PR-147 makes config_root mandatory by default).
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})
        plugin.shutdown()
        assert plugin._initialized is False
        assert plugin._backup_manager is None


class TestFileEditPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_tool_schemas(self):
        plugin = FileEditPlugin()
        declarations = plugin.get_tool_schemas()

        # Pin the exact SET rather than a count: a bare number rots
        # invisibly (this read ``== 7`` while the plugin had grown to 11)
        # and, when it does trip, says nothing about what changed.
        tool_names = {d.name for d in declarations}
        assert tool_names == {
            "readFile", "updateFile", "writeNewFile", "removeFile",
            "moveFile", "renameFile", "undoFileChange",
            "findAndReplace", "multiFileEdit", "listBackups", "restoreFile",
        }

    def test_read_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        read_file = [s for s in schemas if s.name == "readFile"][0]
        schema = read_file.parameters

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "path" in schema["required"]

    def test_update_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        update_file = [s for s in schemas if s.name == "updateFile"][0]
        schema = update_file.parameters

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert "new_content" in schema["properties"]
        assert "old" in schema["properties"]
        assert "new" in schema["properties"]
        assert "prologue" in schema["properties"]
        assert "epilogue" in schema["properties"]
        assert schema["required"] == ["path"]


class TestFileEditPluginExecutors:
    """Tests for executor mapping."""

    def test_get_executors(self):
        plugin = FileEditPlugin()
        executors = plugin.get_executors()

        assert "readFile" in executors
        assert "updateFile" in executors
        assert "writeNewFile" in executors
        assert "removeFile" in executors
        assert "moveFile" in executors
        assert "renameFile" in executors
        assert "undoFileChange" in executors
        assert all(callable(e) for e in executors.values())


class TestFileEditPluginAutoApproval:
    """Tests for auto-approved tools."""

    def test_get_auto_approved_tools(self):
        plugin = FileEditPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "readFile" in auto_approved
        assert "undoFileChange" in auto_approved
        assert "updateFile" not in auto_approved
        assert "writeNewFile" not in auto_approved
        assert "removeFile" not in auto_approved


class TestFileEditPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions(self):
        plugin = FileEditPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "readFile" in instructions
        assert "updateFile" in instructions
        assert "moveFile" in instructions
        assert "renameFile" in instructions
        assert "backup" in instructions.lower()


class TestReadFileExecution:
    """Tests for readFile tool execution."""

    def test_read_existing_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = plugin._execute_read_file({"path": str(test_file)})

        # readFile returns a plain string (header + content) to avoid
        # JSON escaping in provider converters
        assert isinstance(result, str)
        assert "Hello, World!" in result
        assert str(test_file) in result
        assert "size: 13" in result
        assert "lines: 1" in result

    def test_read_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_read_file({"path": str(tmp_path / "nonexistent.txt")})

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_read_file_missing_path(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_read_file({})

        assert "error" in result
        assert "required" in result["error"].lower()


class TestUpdateFileExecution:
    """Tests for updateFile tool execution."""

    def test_update_existing_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Updated content"
        assert "backup" in result

    def test_update_creates_backup(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Original content"

    def test_update_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_update_file({
            "path": str(tmp_path / "nonexistent.txt"),
            "new_content": "Content"
        })

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_update_file_accepts_content_parameter(self, tmp_path):
        """Test that updateFile accepts 'content' as alias for 'new_content'.

        This ensures consistency with writeNewFile which uses 'content'.
        Models often use 'content' for both tools, so we accept either.
        """
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Use 'content' instead of 'new_content'
        result = plugin._execute_update_file({
            "path": str(test_file),
            "content": "Updated via content param"
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Updated via content param"
        assert result["size"] == len("Updated via content param")


    def test_update_file_path_only_preserves_file_and_errors(self, tmp_path):
        """updateFile with only 'path' must not truncate the file (#782).

        Absent content is a malformed call, not a request for an empty file.
        The guard is the file's BYTES: reverting the fix makes this test
        fail on the truncation even if the returned dict looks plausible.
        """
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "victim.txt"
        original = "line one\nline two\nline three\n"
        test_file.write_text(original)

        result = plugin._execute_update_file({"path": str(test_file)})

        assert "error" in result
        assert "success" not in result
        # The whole defect is this line: the bytes must be untouched.
        assert test_file.read_text() == original

    def test_update_file_explicit_empty_new_content_truncates(self, tmp_path):
        """An explicit new_content='' is a deliberate truncation and works.

        The fix must distinguish OMITTED from DELIBERATELY EMPTY, not
        forbid empty files.
        """
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("to be cleared\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "",
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == ""

    def test_update_file_explicit_empty_content_alias_truncates(self, tmp_path):
        """An explicit content='' (alias) also truncates deliberately."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("to be cleared\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "content": "",
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == ""

    def test_update_file_error_names_both_modes(self, tmp_path):
        """The guiding error must name both modes so the model can self-correct."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("content\n")

        result = plugin._execute_update_file({"path": str(test_file)})

        error = result["error"].lower()
        assert "old" in error and "new" in error
        assert "new_content" in error

    def test_format_update_file_path_only_pre_validates_error(self, tmp_path):
        """Permission preview for a path-only call surfaces the guiding error.

        Without this, an interactive session would preview a truncation the
        executor then refuses.
        """
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("content\n")

        display_info = plugin.format_permission_request(
            "updateFile", {"path": str(test_file)}, "console"
        )

        assert display_info is not None
        assert display_info.pre_validation_error
        assert "new_content" in display_info.pre_validation_error


class TestUpdateFileTargetedEdit:
    """Tests for updateFile targeted edit mode (old + new)."""

    def test_targeted_edit_replaces_fragment(self, tmp_path):
        """Test that old+new replaces only the matched fragment."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "def foo():\n    pass",
            "new": "def foo():\n    return 42",
        })

        assert "error" not in result
        assert result["success"] is True
        content = test_file.read_text()
        assert "def foo():\n    return 42" in content
        # bar should be untouched
        assert "def bar():\n    pass" in content

    def test_targeted_edit_with_prologue(self, tmp_path):
        """Test disambiguation via prologue."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("class A:\n    x = 1\n\nclass B:\n    x = 1\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "x = 1",
            "new": "x = 2",
            "prologue": "class B:\n    ",
        })

        assert "error" not in result
        content = test_file.read_text()
        assert "class A:\n    x = 1" in content
        assert "class B:\n    x = 2" in content

    def test_targeted_edit_not_found_returns_error(self, tmp_path):
        """Test error when old text is not in file."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "nonexistent text",
            "new": "replacement",
        })

        assert "error" in result
        assert "not found" in result["error"].lower()
        # File should be unchanged
        assert test_file.read_text() == "Hello, World!\n"

    def test_targeted_edit_ambiguous_returns_error(self, tmp_path):
        """Test error when old text appears multiple times."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("x = 1\nx = 1\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "x = 1",
            "new": "x = 2",
        })

        assert "error" in result
        assert "matched 2 times" in result["error"].lower()
        # File should be unchanged
        assert test_file.read_text() == "x = 1\nx = 1\n"

    def test_targeted_edit_requires_new_when_old_provided(self, tmp_path):
        """Test error when old is provided without new."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("content\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "old": "content",
        })

        assert "error" in result
        assert "'new' is required" in result["error"]

    def test_targeted_edit_creates_backup(self, tmp_path):
        """Test that targeted edit creates a backup."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("original content\n")

        plugin._execute_update_file({
            "path": str(test_file),
            "old": "original",
            "new": "modified",
        })

        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "original content\n"

    def test_full_replacement_still_works(self, tmp_path):
        """Test backward compatibility: new_content still works."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("old content\n")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "entirely new content\n",
        })

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "entirely new content\n"

    def test_format_targeted_edit_shows_diff(self, tmp_path):
        """Test that permission display shows correct diff for targeted mode."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {
                "path": str(test_file),
                "old": "def foo():\n    pass",
                "new": "def foo():\n    return 42",
            },
            "console"
        )

        assert display_info is not None
        assert display_info.format_hint == "diff"
        # The diff should show the change
        assert "-    pass" in display_info.details
        assert "+    return 42" in display_info.details

    def test_format_targeted_edit_error(self, tmp_path):
        """Test permission display when targeted edit can't find text."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {
                "path": str(test_file),
                "old": "nonexistent",
                "new": "replacement",
            },
            "console"
        )

        assert display_info is not None
        assert "error" in display_info.summary.lower()
        assert display_info.format_hint == "text"
        assert display_info.pre_validation_error is not None
        assert "not found" in display_info.pre_validation_error.lower()

    def test_format_targeted_edit_with_prologue_shows_diff(self, tmp_path):
        """Test that permission display shows diff when prologue disambiguates."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("class Foo:\n    x = 1\n\nclass Bar:\n    x = 1\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {
                "path": str(test_file),
                "old": "x = 1",
                "new": "x = 2",
                "prologue": "class Foo:\n    ",
            },
            "console"
        )

        assert display_info is not None
        assert display_info.format_hint == "diff"
        assert display_info.pre_validation_error is None
        assert "-    x = 1" in display_info.details
        assert "+    x = 2" in display_info.details

    def test_format_targeted_edit_with_bad_prologue_sets_pre_validation_error(self, tmp_path):
        """Test that bad prologue sets pre_validation_error to skip permission prompt."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.py"
        test_file.write_text("class Foo:\n    x = 1\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {
                "path": str(test_file),
                "old": "x = 1",
                "new": "x = 2",
                "prologue": "wrong prologue",
            },
            "console"
        )

        assert display_info is not None
        assert display_info.pre_validation_error is not None
        assert "not found" in display_info.pre_validation_error.lower()


class TestWriteNewFileExecution:
    """Tests for writeNewFile tool execution."""

    def test_write_new_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.txt"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": "New file content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert new_file.exists()
        assert new_file.read_text() == "New file content"

    def test_write_new_file_creates_directories(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "subdir" / "nested" / "new.txt"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": "Content"
        })

        assert "error" not in result
        assert result["success"] is True
        assert new_file.exists()

    def test_write_new_file_fails_if_exists(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("Existing content")

        result = plugin._execute_write_new_file({
            "path": str(existing_file),
            "content": "New content"
        })

        assert "error" in result
        assert "already exists" in result["error"].lower()


class TestRemoveFileExecution:
    """Tests for removeFile tool execution."""

    def test_remove_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        result = plugin._execute_remove_file({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert result["deleted"] is True
        assert not test_file.exists()

    def test_remove_file_creates_backup(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to backup")

        plugin._execute_remove_file({"path": str(test_file)})

        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Content to backup"

    def test_remove_nonexistent_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_remove_file({"path": str(tmp_path / "nonexistent.txt")})

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestUndoFileChangeExecution:
    """Tests for undoFileChange tool execution."""

    def test_undo_file_change(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Update the file (creates backup)
        plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert test_file.read_text() == "Updated content"

        # Undo the change
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert test_file.read_text() == "Original content"

    def test_undo_restores_deleted_file(self, tmp_path):
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Delete the file (creates backup)
        plugin._execute_remove_file({"path": str(test_file)})

        assert not test_file.exists()

        # Undo the deletion
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" not in result
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == "Original content"

    def test_undo_no_backup_available(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        # Try to undo without any backup
        result = plugin._execute_undo_file_change({"path": str(test_file)})

        assert "error" in result
        assert "no backup" in result["error"].lower()


class TestFormatPermissionRequest:
    """Tests for permission display formatting."""

    def test_format_update_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        display_info = plugin.format_permission_request(
            "updateFile",
            {"path": str(test_file), "new_content": "Line 1\nLine 2\nLine 3\n"},
            "console"
        )

        assert display_info is not None
        assert "Update" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+Line 3" in display_info.details

    def test_format_update_file_accepts_content_parameter(self, tmp_path):
        """Test that format_permission_request accepts 'content' for updateFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\n")

        # Use 'content' instead of 'new_content'
        display_info = plugin.format_permission_request(
            "updateFile",
            {"path": str(test_file), "content": "Line 1\nLine 2\nLine 3\n"},
            "console"
        )

        assert display_info is not None
        assert "Update" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+Line 3" in display_info.details

    def test_format_write_new_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.txt"

        display_info = plugin.format_permission_request(
            "writeNewFile",
            {"path": str(new_file), "content": "New content"},
            "console"
        )

        assert display_info is not None
        assert "Create" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "+New content" in display_info.details

    def test_format_remove_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        display_info = plugin.format_permission_request(
            "removeFile",
            {"path": str(test_file)},
            "console"
        )

        assert display_info is not None
        assert "Delete" in display_info.summary
        assert "backup" in display_info.summary.lower()
        assert display_info.format_hint == "diff"
        assert "-Content to delete" in display_info.details

    def test_format_unknown_tool_returns_none(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        display_info = plugin.format_permission_request(
            "unknownTool",
            {},
            "console"
        )

        assert display_info is None

    def test_format_move_file(self, tmp_path):
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("File content to move")
        dest_file = tmp_path / "subdir" / "dest.txt"

        display_info = plugin.format_permission_request(
            "moveFile",
            {
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            },
            "console"
        )

        assert display_info is not None
        assert "Move file" in display_info.summary
        assert display_info.format_hint == "diff"
        assert "-File content to move" in display_info.details
        assert "+File content to move" in display_info.details

    def test_format_rename_file(self, tmp_path):
        """Test that renameFile uses the same formatting as moveFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "old_name.txt"
        source_file.write_text("Content")
        dest_file = tmp_path / "new_name.txt"

        display_info = plugin.format_permission_request(
            "renameFile",
            {
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            },
            "console"
        )

        assert display_info is not None
        assert "Move file" in display_info.summary


class TestMoveFileExecution:
    """Tests for moveFile/renameFile tool execution."""

    def test_move_file_basic(self, tmp_path):
        """Test basic file move operation."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content to move")
        dest_file = tmp_path / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" not in result
        assert result["success"] is True
        assert result["source"] == str(source_file)
        assert result["destination"] == str(dest_file)
        assert not source_file.exists()
        assert dest_file.exists()
        assert dest_file.read_text() == "Content to move"

    def test_move_file_creates_directories(self, tmp_path):
        """Test that move creates destination directories."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")
        dest_file = tmp_path / "subdir" / "nested" / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" not in result
        assert result["success"] is True
        assert dest_file.exists()
        assert dest_file.read_text() == "Content"

    def test_move_file_creates_backup(self, tmp_path):
        """Test that move creates a backup of the source file."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Original content")
        dest_file = tmp_path / "dest.txt"

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "source_backup" in result
        # Check backup was created
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "Original content"

    def test_move_file_source_not_found(self, tmp_path):
        """Test error when source file doesn't exist."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_move_file({
            "source_path": str(tmp_path / "nonexistent.txt"),
            "destination_path": str(tmp_path / "dest.txt")
        })

        assert "error" in result
        assert "does not exist" in result["error"]
        assert "source" in result

    def test_move_file_destination_exists_no_overwrite(self, tmp_path):
        """Test error when destination exists without overwrite."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Source content")
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("Existing content")

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file)
        })

        assert "error" in result
        assert "already exists" in result["error"]
        assert "overwrite=True" in result["error"]
        # Source should still exist
        assert source_file.exists()
        # Destination should still have original content
        assert dest_file.read_text() == "Existing content"

    def test_move_file_destination_exists_with_overwrite(self, tmp_path):
        """Test successful overwrite when destination exists."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        source_file = tmp_path / "source.txt"
        source_file.write_text("New content")
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("Old content")

        result = plugin._execute_move_file({
            "source_path": str(source_file),
            "destination_path": str(dest_file),
            "overwrite": True
        })

        assert "error" not in result
        assert result["success"] is True
        assert not source_file.exists()
        assert dest_file.read_text() == "New content"
        # Should have backups for both source and destination
        assert "source_backup" in result
        assert "destination_backup" in result

    def test_move_file_missing_source_path(self, tmp_path):
        """Test error when source_path is missing."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        result = plugin._execute_move_file({
            "destination_path": str(tmp_path / "dest.txt")
        })

        assert "error" in result
        assert "source_path is required" in result["error"]

    def test_move_file_missing_destination_path(self, tmp_path):
        """Test error when destination_path is missing."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")

        result = plugin._execute_move_file({
            "source_path": str(source_file)
        })

        assert "error" in result
        assert "destination_path is required" in result["error"]

    def test_rename_file_uses_same_executor(self, tmp_path):
        """Test that renameFile uses the same executor as moveFile."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        executors = plugin.get_executors()
        # ``RunnerForwardingMixin`` now wraps every executor in a PER-TOOL
        # forwarder closure, so the two entries are distinct objects even
        # though they delegate to one implementation -- identity no longer
        # expresses "same executor".  The forwarder propagates the wrapped
        # function's ``__name__``, so compare that instead: both must resolve
        # to ``_execute_move_file``, which is what this test has always meant.
        assert (
            executors["moveFile"].__name__
            == executors["renameFile"].__name__
            == "_execute_move_file"
        )

    def test_move_file_source_is_directory(self, tmp_path):
        """Test error when source is a directory."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()

        result = plugin._execute_move_file({
            "source_path": str(source_dir),
            "destination_path": str(tmp_path / "dest")
        })

        assert "error" in result
        assert "not a file" in result["error"]


class TestMoveFileToolSchemas:
    """Tests for moveFile/renameFile tool schemas."""

    def test_move_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        move_file = [s for s in schemas if s.name == "moveFile"][0]
        schema = move_file.parameters

        assert schema["type"] == "object"
        assert "source_path" in schema["properties"]
        assert "destination_path" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "source_path" in schema["required"]
        assert "destination_path" in schema["required"]
        # overwrite is optional
        assert "overwrite" not in schema["required"]

    def test_rename_file_schema(self):
        plugin = FileEditPlugin()
        schemas = plugin.get_tool_schemas()
        rename_file = [s for s in schemas if s.name == "renameFile"][0]
        schema = rename_file.parameters

        assert schema["type"] == "object"
        assert "source_path" in schema["properties"]
        assert "destination_path" in schema["properties"]
        assert "overwrite" in schema["properties"]

    def test_tool_count_includes_move_and_rename(self):
        plugin = FileEditPlugin()
        declarations = plugin.get_tool_schemas()

        tool_names = [d.name for d in declarations]
        assert "moveFile" in tool_names
        assert "renameFile" in tool_names
        # The count this test used to pin was a proxy for "the schema list
        # has no duplicate entries" -- assert that directly, so adding a
        # tool doesn't fail a test about move/rename.
        assert len(tool_names) == len(set(tool_names))


class TestTelemetryEnrichment:
    """Tests for _telemetry convention dicts in tool results."""

    def test_update_file_result_includes_telemetry_dict(self, tmp_path):
        """updateFile result must include _telemetry with file operation metadata."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        result = plugin._execute_update_file({
            "path": str(test_file),
            "new_content": "Updated content"
        })

        assert "error" not in result
        assert "_telemetry" in result
        telemetry = result["_telemetry"]
        assert telemetry["jaato.file.operation"] == "update"
        assert "jaato.file.path" in telemetry
        assert telemetry["jaato.file.size_bytes"] == len("Updated content")
        assert telemetry["jaato.file.lines"] == len("Updated content".splitlines())
        assert "jaato.file.had_backup" in telemetry
        assert telemetry["jaato.file.had_backup"] is True

    def test_write_new_file_result_includes_telemetry_dict(self, tmp_path):
        """writeNewFile result must include _telemetry with file operation metadata."""
        plugin = FileEditPlugin()
        plugin.initialize({"backup_dir": str(tmp_path / "backups")})

        new_file = tmp_path / "new.txt"
        content = "New file content\nLine 2\n"

        result = plugin._execute_write_new_file({
            "path": str(new_file),
            "content": content
        })

        assert "error" not in result
        assert "_telemetry" in result
        telemetry = result["_telemetry"]
        assert telemetry["jaato.file.operation"] == "write_new"
        assert "jaato.file.path" in telemetry
        assert telemetry["jaato.file.size_bytes"] == len(content)
        assert telemetry["jaato.file.lines"] == len(content.splitlines())

    def test_remove_file_result_includes_telemetry_dict(self, tmp_path):
        """removeFile result must include _telemetry with file operation metadata."""
        plugin = FileEditPlugin()
        backup_dir = tmp_path / "backups"
        plugin.initialize({"backup_dir": str(backup_dir)})

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        result = plugin._execute_remove_file({"path": str(test_file)})

        assert "error" not in result
        assert "_telemetry" in result
        telemetry = result["_telemetry"]
        assert telemetry["jaato.file.operation"] == "remove"
        assert "jaato.file.path" in telemetry
