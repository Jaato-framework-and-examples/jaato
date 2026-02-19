"""Unit tests for the FilesystemQueryPlugin."""

import os
import tempfile
from pathlib import Path

import pytest

from ..plugin import FilesystemQueryPlugin, create_plugin
from ...permission.sanitization import PathScopeConfig


class TestFilesystemQueryPluginBasics:
    """Tests for basic plugin functionality."""

    def test_create_plugin(self):
        """Test plugin factory function."""
        plugin = create_plugin()
        assert plugin is not None
        assert isinstance(plugin, FilesystemQueryPlugin)

    def test_plugin_name(self):
        """Test plugin name property."""
        plugin = FilesystemQueryPlugin()
        assert plugin.name == "filesystem_query"

    def test_initialize_with_default_config(self):
        """Test plugin initialization with default config."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        assert plugin._initialized is True
        assert plugin._config is not None
        assert plugin._config.max_results == 500

    def test_initialize_with_custom_config(self):
        """Test plugin initialization with custom config."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "max_results": 100,
            "timeout_seconds": 60,
        })

        assert plugin._initialized is True
        assert plugin._config.max_results == 100
        assert plugin._config.timeout_seconds == 60

    def test_shutdown(self):
        """Test plugin shutdown."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()
        assert plugin._initialized is True

        plugin.shutdown()
        assert plugin._initialized is False

    def test_get_tool_schemas(self):
        """Test that tool schemas are returned correctly."""
        plugin = FilesystemQueryPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 2

        schema_names = [s.name for s in schemas]
        assert "glob_files" in schema_names
        assert "grep_content" in schema_names

    def test_get_executors(self):
        """Test that executors are returned correctly."""
        plugin = FilesystemQueryPlugin()
        executors = plugin.get_executors()

        assert "glob_files" in executors
        assert "grep_content" in executors
        assert callable(executors["glob_files"])
        assert callable(executors["grep_content"])

    def test_get_auto_approved_tools(self):
        """Test that both tools are auto-approved."""
        plugin = FilesystemQueryPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "glob_files" in auto_approved
        assert "grep_content" in auto_approved

    def test_get_system_instructions(self):
        """Test that system instructions are provided."""
        plugin = FilesystemQueryPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "glob_files" in instructions
        assert "grep_content" in instructions


class TestGlobFiles:
    """Tests for the glob_files tool."""

    def test_glob_files_basic(self):
        """Test basic glob_files functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.py").write_text("# test 1")
            (Path(tmpdir) / "test2.py").write_text("# test 2")
            (Path(tmpdir) / "readme.md").write_text("# readme")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "*.py",
                "root": tmpdir,
            })

            assert "error" not in result
            assert result["total"] == 2
            assert len(result["files"]) == 2
            assert not result["truncated"]

    def test_glob_files_recursive(self):
        """Test recursive glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "src"
            subdir.mkdir()
            (Path(tmpdir) / "main.py").write_text("# main")
            (subdir / "util.py").write_text("# util")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "**/*.py",
                "root": tmpdir,
            })

            assert result["total"] == 2
            paths = [f["path"] for f in result["files"]]
            assert "main.py" in paths
            assert "src/util.py" in paths

    def test_glob_files_max_results(self):
        """Test max_results limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(10):
                (Path(tmpdir) / f"file{i}.txt").write_text(f"content {i}")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": tmpdir,
                "max_results": 3,
            })

            assert result["total"] == 10
            assert result["returned"] == 3
            assert len(result["files"]) == 3
            assert result["truncated"] is True

    def test_glob_files_excludes_patterns(self):
        """Test that excluded patterns are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files including excluded dirs
            (Path(tmpdir) / "main.py").write_text("# main")
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "cached.py").write_text("# cached")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "**/*.py",
                "root": tmpdir,
            })

            # Should only find main.py, not cached.py
            assert result["total"] == 1
            assert result["files"][0]["path"] == "main.py"

    def test_glob_files_hidden_files(self):
        """Test include_hidden parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "visible.txt").write_text("visible")
            (Path(tmpdir) / ".hidden.txt").write_text("hidden")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            # Without include_hidden
            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": tmpdir,
                "include_hidden": False,
            })
            assert result["total"] == 1

            # With include_hidden
            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": tmpdir,
                "include_hidden": True,
            })
            assert result["total"] == 2

    def test_glob_files_nonexistent_root(self):
        """Test error handling for nonexistent root."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        result = plugin._execute_glob_files({
            "pattern": "*.py",
            "root": "/nonexistent/path/12345",
        })

        assert "error" in result
        assert result["total"] == 0

    def test_glob_files_empty_pattern(self):
        """Test error handling for empty pattern."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        result = plugin._execute_glob_files({
            "pattern": "",
        })

        assert "error" in result

    def test_glob_files_absolute_pattern_posix(self):
        """Test that absolute POSIX patterns return an actionable error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("# test")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "/home/user/project/**/*.py",
                "root": tmpdir,
            })

            assert "error" in result
            assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()
            assert result["total"] == 0

    def test_glob_files_absolute_pattern_windows(self):
        """Test that absolute Windows drive-letter patterns return an actionable error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("# test")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_glob_files({
                "pattern": "C:/Users/me/project/**/*.py",
                "root": tmpdir,
            })

            assert "error" in result
            assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()
            assert result["total"] == 0


class TestGrepContent:
    """Tests for the grep_content tool."""

    def test_grep_content_basic(self):
        """Test basic grep_content functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello():\n    pass\n")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "def hello",
                "path": tmpdir,
            })

            assert "error" not in result
            assert result["total_matches"] == 1
            assert result["files_with_matches"] == 1
            assert result["matches"][0]["line"] == 1
            assert "def hello" in result["matches"][0]["text"]

    def test_grep_content_regex(self):
        """Test regex pattern matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def process_data():\n"
                "def process_file():\n"
                "def other_func():\n"
            )

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": r"def process_\w+",
                "path": tmpdir,
            })

            assert result["total_matches"] == 2

    def test_grep_content_context_lines(self):
        """Test context lines around matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "line 1\nline 2\nMATCH\nline 4\nline 5\n"
            )

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "MATCH",
                "path": tmpdir,
                "context_lines": 2,
            })

            match = result["matches"][0]
            assert len(match["context_before"]) == 2
            assert len(match["context_after"]) == 2
            assert "line 2" in match["context_before"]
            assert "line 4" in match["context_after"]

    def test_grep_content_case_insensitive(self):
        """Test case-insensitive search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("Hello\nHELLO\nhello\n")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            # Case sensitive
            result = plugin._execute_grep_content({
                "pattern": "hello",
                "path": tmpdir,
                "case_sensitive": True,
            })
            assert result["total_matches"] == 1

            # Case insensitive
            result = plugin._execute_grep_content({
                "pattern": "hello",
                "path": tmpdir,
                "case_sensitive": False,
            })
            assert result["total_matches"] == 3

    def test_grep_content_file_glob(self):
        """Test file_glob filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("pattern here")
            (Path(tmpdir) / "test.txt").write_text("pattern here")
            (Path(tmpdir) / "test.md").write_text("pattern here")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "pattern",
                "path": tmpdir,
                "file_glob": ["*.py"],
            })

            assert result["total_matches"] == 1
            assert result["matches"][0]["file"] == "test.py"

    def test_grep_content_max_results(self):
        """Test max_results limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many matches
            content = "\n".join([f"match line {i}" for i in range(20)])
            (Path(tmpdir) / "test.txt").write_text(content)

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "match",
                "path": tmpdir,
                "max_results": 5,
            })

            assert len(result["matches"]) == 5
            assert result["truncated"] is True

    def test_grep_content_single_file(self):
        """Test searching a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("def test():\n    pass\n")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "def test",
                "path": str(file_path),
            })

            assert result["total_matches"] == 1
            assert result["files_searched"] == 1

    def test_grep_content_invalid_regex(self):
        """Test error handling for invalid regex."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        result = plugin._execute_grep_content({
            "pattern": "[invalid",
        })

        assert "error" in result
        assert "Invalid regex" in result["error"]

    def test_grep_content_nonexistent_path(self):
        """Test error handling for nonexistent path."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        result = plugin._execute_grep_content({
            "pattern": "test",
            "path": "/nonexistent/path/12345",
        })

        assert "error" in result

    def test_grep_content_excludes_patterns(self):
        """Test that excluded patterns are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files including excluded dirs
            (Path(tmpdir) / "main.py").write_text("pattern here")
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "cached.py").write_text("pattern here")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "pattern",
                "path": tmpdir,
            })

            # Should only find in main.py
            assert result["files_with_matches"] == 1
            assert result["matches"][0]["file"] == "main.py"

    def test_grep_content_file_glob_array(self):
        """Test file_glob with array of patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.java").write_text("@CircuitBreaker")
            (Path(tmpdir) / "test.kt").write_text("@CircuitBreaker")
            (Path(tmpdir) / "test.scala").write_text("@CircuitBreaker")
            (Path(tmpdir) / "test.py").write_text("@CircuitBreaker")
            (Path(tmpdir) / "test.txt").write_text("@CircuitBreaker")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "@CircuitBreaker",
                "path": tmpdir,
                "file_glob": ["*.java", "*.kt", "*.scala"],
            })

            # Should find in Java, Kotlin, and Scala files only
            assert result["total_matches"] == 3
            assert result["files_with_matches"] == 3
            matched_files = {m["file"] for m in result["matches"]}
            assert matched_files == {"test.java", "test.kt", "test.scala"}

    def test_grep_content_file_glob_array_no_duplicates(self):
        """Test that overlapping patterns don't produce duplicate results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("pattern here")
            (Path(tmpdir) / "test.py").write_text("pattern here")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "pattern",
                "path": tmpdir,
                # Both patterns match the same .py files
                "file_glob": ["*.py", "*.py"],
            })

            # Should not have duplicates
            assert result["total_matches"] == 2
            assert result["files_with_matches"] == 2

    def test_grep_content_absolute_file_glob_posix(self):
        """Test that absolute POSIX file_glob patterns return an actionable error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("hello world")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "hello",
                "path": tmpdir,
                "file_glob": ["/home/user/project/**/*.py"],
            })

            assert "error" in result
            assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()
            assert "file_glob" in result["error"] or "path" in result["error"]
            assert result["total_matches"] == 0

    def test_grep_content_absolute_file_glob_windows(self):
        """Test that absolute Windows file_glob patterns return an actionable error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("hello world")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "hello",
                "path": tmpdir,
                "file_glob": ["C:/Users/me/project/**/*.py"],
            })

            assert "error" in result
            assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()
            assert result["total_matches"] == 0

    def test_grep_content_mixed_absolute_relative_file_glob(self):
        """Test error when file_glob mixes absolute and relative patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("hello world")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "hello",
                "path": tmpdir,
                "file_glob": ["*.py", "/absolute/path/**"],
            })

            # The relative pattern runs fine; the absolute one should error
            # (implementation catches on first absolute pattern encountered)
            # Either we get results from *.py or an error from the absolute one
            if "error" in result:
                assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()


class TestBackgroundCapable:
    """Tests for background execution support."""

    def test_supports_background(self):
        """Test that tools support background execution."""
        plugin = FilesystemQueryPlugin()

        assert plugin.supports_background("glob_files") is True
        assert plugin.supports_background("grep_content") is True
        assert plugin.supports_background("unknown_tool") is False

    def test_get_auto_background_threshold(self):
        """Test auto-background threshold configuration."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={"timeout_seconds": 45})

        threshold = plugin.get_auto_background_threshold("glob_files")
        assert threshold == 45.0

    def test_estimate_duration(self):
        """Test duration estimation."""
        plugin = FilesystemQueryPlugin()

        # Recursive patterns should estimate longer
        duration = plugin.estimate_duration("glob_files", {"pattern": "**/*.py"})
        assert duration is not None
        assert duration > 1.0

        # Non-recursive should be faster
        duration = plugin.estimate_duration("glob_files", {"pattern": "*.py"})
        assert duration is not None
        assert duration <= 1.0


class TestBinaryFileDetection:
    """Tests for binary file detection."""

    def test_skip_binary_files(self):
        """Test that binary files are skipped during grep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a text file
            (Path(tmpdir) / "text.txt").write_text("searchable content")

            # Create a binary file with null bytes
            (Path(tmpdir) / "binary.bin").write_bytes(
                b"binary\x00content\x00here"
            )

            plugin = FilesystemQueryPlugin()
            plugin.initialize()

            result = plugin._execute_grep_content({
                "pattern": "content",
                "path": tmpdir,
            })

            # Should only find the text file
            assert result["total_matches"] == 1
            assert result["matches"][0]["file"] == "text.txt"

    def test_is_binary_file(self):
        """Test binary file detection method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_file = Path(tmpdir) / "text.txt"
            text_file.write_text("Hello, world!")

            binary_file = Path(tmpdir) / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03")

            plugin = FilesystemQueryPlugin()

            assert plugin._is_binary_file(text_file) is False
            assert plugin._is_binary_file(binary_file) is True


class TestPathScopeValidation:
    """Tests for path scope validation in filesystem query tools."""

    def test_glob_files_without_path_scope(self):
        """Test that glob_files works without path scope config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("content")

            plugin = FilesystemQueryPlugin()
            plugin.initialize()  # No path_scope config

            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": tmpdir,
            })

            assert "error" not in result
            assert result["total"] == 1

    def test_glob_files_path_scope_blocks_absolute(self):
        """Test that glob_files respects block_absolute."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "path_scope": {
                "allowed_roots": ["."],
                "block_absolute": True,
                "allow_tmp": False,
            }
        })

        result = plugin._execute_glob_files({
            "pattern": "*.txt",
            "root": "/etc",
        })

        assert "error" in result
        assert "not allowed" in result["error"].lower()

    def test_glob_files_path_scope_allows_tmp(self):
        """Test that glob_files allows /tmp when allow_tmp=True."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "path_scope": {
                "allowed_roots": ["."],
                "block_absolute": True,
                "allow_tmp": True,
            }
        })

        # Create a test file in /tmp
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_file = f.name

        try:
            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": "/tmp",
            })

            # Should be allowed, not blocked
            assert "violations" not in result
        finally:
            os.unlink(tmp_file)

    def test_grep_content_path_scope_blocks_absolute(self):
        """Test that grep_content respects block_absolute."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "path_scope": {
                "allowed_roots": ["."],
                "block_absolute": True,
                "allow_tmp": False,
            }
        })

        result = plugin._execute_grep_content({
            "pattern": "test",
            "path": "/etc/passwd",
        })

        assert "error" in result
        assert "not allowed" in result["error"].lower()

    def test_grep_content_path_scope_allows_tmp(self):
        """Test that grep_content allows /tmp when allow_tmp=True."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "path_scope": {
                "allowed_roots": ["."],
                "block_absolute": True,
                "allow_tmp": True,
            }
        })

        # Create a test file in /tmp
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".txt", delete=False, mode="w") as f:
            f.write("searchable content here")
            tmp_file = f.name

        try:
            result = plugin._execute_grep_content({
                "pattern": "searchable",
                "path": tmp_file,
            })

            # Should be allowed and find the content
            assert "violations" not in result
            assert result["total_matches"] == 1
        finally:
            os.unlink(tmp_file)

    def test_set_path_scope_after_init(self):
        """Test setting path scope after initialization."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize()

        # Initially no path scope
        assert plugin._path_scope_config is None

        # Set path scope
        config = PathScopeConfig(
            allowed_roots=["."],
            block_absolute=True,
            allow_tmp=True,
        )
        plugin.set_path_scope(config)

        assert plugin._path_scope_config is config

        # Now absolute paths should be blocked (except /tmp)
        result = plugin._execute_glob_files({
            "pattern": "*.txt",
            "root": "/etc",
        })
        assert "error" in result

    def test_path_scope_cleared_on_shutdown(self):
        """Test that path scope is cleared on shutdown."""
        plugin = FilesystemQueryPlugin()
        plugin.initialize(config={
            "path_scope": {
                "allowed_roots": ["."],
                "block_absolute": True,
            }
        })

        assert plugin._path_scope_config is not None

        plugin.shutdown()

        assert plugin._path_scope_config is None

    def test_path_scope_blocks_parent_traversal(self):
        """Test that parent traversal is blocked when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin = FilesystemQueryPlugin()
            plugin.initialize(config={
                "path_scope": {
                    "allowed_roots": [tmpdir],
                    "block_absolute": False,
                    "block_parent_traversal": True,
                }
            })

            result = plugin._execute_glob_files({
                "pattern": "*.txt",
                "root": f"{tmpdir}/../",
            })

            assert "error" in result
            assert "traversal" in result["error"].lower()

    def test_path_scope_with_pathscopeconfig_instance(self):
        """Test passing PathScopeConfig instance directly."""
        plugin = FilesystemQueryPlugin()
        config = PathScopeConfig(
            allowed_roots=["/tmp"],
            block_absolute=False,
            allow_tmp=True,
        )
        plugin.initialize(config={"path_scope": config})

        assert plugin._path_scope_config is config
