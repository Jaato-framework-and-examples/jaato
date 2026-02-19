"""Unit tests for the ASTSearchPlugin."""

import tempfile
from pathlib import Path

import pytest

from ..plugin import (
    ASTSearchPlugin,
    create_plugin,
    LANGUAGE_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    _check_ast_grep_available,
)


# Skip all tests if ast-grep-py is not installed
ast_grep_available = _check_ast_grep_available()
requires_ast_grep = pytest.mark.skipif(
    not ast_grep_available,
    reason="ast-grep-py not installed"
)


class TestASTSearchPluginBasics:
    """Tests for basic plugin functionality."""

    def test_create_plugin(self):
        """Test plugin factory function."""
        plugin = create_plugin()
        assert plugin is not None
        assert isinstance(plugin, ASTSearchPlugin)

    def test_plugin_name(self):
        """Test plugin name property."""
        plugin = ASTSearchPlugin()
        assert plugin.name == "ast_search"

    def test_initialize_with_default_config(self):
        """Test plugin initialization with default config."""
        plugin = ASTSearchPlugin()
        plugin.initialize()

        assert plugin._initialized is True
        assert plugin._max_results == 100
        assert plugin._context_lines == 2

    def test_initialize_with_custom_config(self):
        """Test plugin initialization with custom config."""
        plugin = ASTSearchPlugin()
        plugin.initialize(config={
            "max_results": 50,
            "context_lines": 5,
            "exclude_dirs": ["custom_exclude"],
        })

        assert plugin._initialized is True
        assert plugin._max_results == 50
        assert plugin._context_lines == 5
        assert "custom_exclude" in plugin._exclude_dirs

    def test_shutdown(self):
        """Test plugin shutdown."""
        plugin = ASTSearchPlugin()
        plugin.initialize()
        assert plugin._initialized is True

        plugin.shutdown()
        assert plugin._initialized is False

    def test_get_tool_schemas(self):
        """Test that tool schemas are returned correctly."""
        plugin = ASTSearchPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0].name == "ast_search"
        assert "pattern" in schemas[0].parameters["properties"]
        assert "language" in schemas[0].parameters["properties"]

    def test_tool_schema_category_and_discoverability(self):
        """Test that tool schema has correct category and discoverability."""
        plugin = ASTSearchPlugin()
        schemas = plugin.get_tool_schemas()

        schema = schemas[0]
        assert schema.category == "search"
        assert schema.discoverability == "discoverable"

    def test_get_executors(self):
        """Test that executors are returned correctly."""
        plugin = ASTSearchPlugin()
        executors = plugin.get_executors()

        assert "ast_search" in executors
        assert callable(executors["ast_search"])

    def test_get_auto_approved_tools(self):
        """Test that ast_search is auto-approved."""
        plugin = ASTSearchPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "ast_search" in auto_approved

    def test_get_system_instructions(self):
        """Test that system instructions are provided."""
        plugin = ASTSearchPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "ast_search" in instructions
        assert "$NAME" in instructions  # Metavariable documentation


class TestLanguageMappings:
    """Tests for language extension mappings."""

    def test_python_extensions(self):
        """Test Python file extensions."""
        assert ".py" in LANGUAGE_EXTENSIONS["python"]
        assert ".pyi" in LANGUAGE_EXTENSIONS["python"]

    def test_javascript_extensions(self):
        """Test JavaScript file extensions."""
        assert ".js" in LANGUAGE_EXTENSIONS["javascript"]
        assert ".mjs" in LANGUAGE_EXTENSIONS["javascript"]

    def test_extension_to_language_mapping(self):
        """Test reverse mapping from extension to language."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".go"] == "go"
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"


class TestASTSearchErrorHandling:
    """Tests for error handling when ast-grep is not available."""

    def test_error_when_ast_grep_unavailable(self):
        """Test graceful error when ast-grep-py is not installed."""
        plugin = ASTSearchPlugin()
        plugin.initialize()
        # Force unavailable state
        plugin._ast_grep_available = False

        result = plugin._execute_ast_search({
            "pattern": "def $FUNC($$$): $$$",
            "path": "/tmp",
        })

        assert "error" in result
        assert "ast-grep-py" in result["error"]

    def test_empty_pattern_error(self):
        """Test error handling for empty pattern."""
        plugin = ASTSearchPlugin()
        plugin.initialize()

        result = plugin._execute_ast_search({
            "pattern": "",
        })

        assert "error" in result
        assert "Pattern is required" in result["error"]

    def test_nonexistent_path_error(self):
        """Test error handling for nonexistent path."""
        plugin = ASTSearchPlugin()
        plugin.initialize()

        result = plugin._execute_ast_search({
            "pattern": "def $FUNC($$$): $$$",
            "path": "/nonexistent/path/12345",
        })

        assert "error" in result


@requires_ast_grep
class TestASTSearchPython:
    """Tests for AST search in Python files."""

    def test_find_function_definitions(self):
        """Test finding Python function definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def hello():\n"
                "    pass\n"
                "\n"
                "def world(x, y):\n"
                "    return x + y\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def $FUNC($$$): $$$",
                "path": tmpdir,
                "language": "python",
            })

            assert "error" not in result
            assert result["total_matches"] == 2
            assert result["files_with_matches"] == 1

    def test_find_class_definitions(self):
        """Test finding Python class definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "class MyClass:\n"
                "    pass\n"
                "\n"
                "class OtherClass(BaseClass):\n"
                "    def method(self):\n"
                "        pass\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "class $NAME: $$$",
                "path": tmpdir,
                "language": "python",
            })

            assert result["total_matches"] >= 1

    def test_find_import_statements(self):
        """Test finding Python import statements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "import $MODULE",
                "path": tmpdir,
                "language": "python",
            })

            assert result["total_matches"] == 2  # os and sys

    def test_context_lines(self):
        """Test context lines around matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "# Comment 1\n"
                "# Comment 2\n"
                "def target():\n"
                "    pass\n"
                "# Comment after\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def target(): $$$",
                "path": tmpdir,
                "language": "python",
                "context_lines": 2,
            })

            assert result["total_matches"] == 1
            match = result["matches"][0]
            assert match["context_before"] is not None
            assert match["context_after"] is not None

    def test_max_results_limit(self):
        """Test max_results limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many functions
            funcs = "\n".join([f"def func{i}():\n    pass\n" for i in range(20)])
            (Path(tmpdir) / "test.py").write_text(funcs)

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def $FUNC(): $$$",
                "path": tmpdir,
                "language": "python",
                "max_results": 5,
            })

            assert len(result["matches"]) == 5
            assert result["truncated"] is True

    def test_metavariable_capture(self):
        """Test that metavariables are captured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def my_function():\n"
                "    pass\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def $FUNC($$$): $$$",
                "path": tmpdir,
                "language": "python",
            })

            assert result["total_matches"] == 1
            match = result["matches"][0]
            if match["metavariables"]:
                assert "$FUNC" in match["metavariables"]


@requires_ast_grep
class TestASTSearchJavaScript:
    """Tests for AST search in JavaScript files."""

    def test_find_function_declarations(self):
        """Test finding JavaScript function declarations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.js").write_text(
                "function hello() {\n"
                "    console.log('hello');\n"
                "}\n"
                "\n"
                "function world(x, y) {\n"
                "    return x + y;\n"
                "}\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "function $NAME($$$) { $$$ }",
                "path": tmpdir,
                "language": "javascript",
            })

            assert result["total_matches"] == 2

    def test_find_arrow_functions(self):
        """Test finding JavaScript arrow functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.js").write_text(
                "const add = (a, b) => a + b;\n"
                "const multiply = (a, b) => {\n"
                "    return a * b;\n"
                "};\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "const $NAME = ($$$) => $$$",
                "path": tmpdir,
                "language": "javascript",
            })

            assert result["total_matches"] >= 1


@requires_ast_grep
class TestASTSearchAutoDetect:
    """Tests for automatic language detection."""

    def test_auto_detect_python(self):
        """Test automatic language detection for Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def hello():\n    pass\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            # No language specified
            result = plugin._execute_ast_search({
                "pattern": "def $FUNC($$$): $$$",
                "path": tmpdir,
            })

            assert result["total_matches"] == 1

    def test_auto_detect_single_file(self):
        """Test automatic language detection for single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def $FUNC($$$): $$$",
                "path": str(file_path),
            })

            assert result["total_matches"] == 1

    def test_unknown_extension_error(self):
        """Test error for unknown file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.unknown"
            file_path.write_text("some content")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def $FUNC($$$): $$$",
                "path": str(file_path),
            })

            assert "error" in result


@requires_ast_grep
class TestASTSearchExclusions:
    """Tests for directory exclusions."""

    def test_excludes_pycache(self):
        """Test that __pycache__ is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in main dir
            (Path(tmpdir) / "main.py").write_text("def hello():\n    pass\n")

            # Create file in __pycache__
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "cached.py").write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def hello(): $$$",
                "path": tmpdir,
                "language": "python",
            })

            # Should only find main.py
            assert result["total_matches"] == 1
            assert result["matches"][0]["file"] == "main.py"

    def test_excludes_node_modules(self):
        """Test that node_modules is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in main dir
            (Path(tmpdir) / "app.js").write_text(
                "function hello() { console.log('hi'); }\n"
            )

            # Create file in node_modules
            node_dir = Path(tmpdir) / "node_modules" / "some-package"
            node_dir.mkdir(parents=True)
            (node_dir / "index.js").write_text(
                "function hello() { console.log('hi'); }\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "function hello() { $$$ }",
                "path": tmpdir,
                "language": "javascript",
            })

            # Should only find app.js
            assert result["total_matches"] == 1
            assert result["matches"][0]["file"] == "app.js"

    def test_excludes_hidden_directories(self):
        """Test that hidden directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in main dir
            (Path(tmpdir) / "main.py").write_text("def hello():\n    pass\n")

            # Create file in hidden dir
            hidden_dir = Path(tmpdir) / ".hidden"
            hidden_dir.mkdir()
            (hidden_dir / "secret.py").write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def hello(): $$$",
                "path": tmpdir,
                "language": "python",
            })

            # Should only find main.py
            assert result["total_matches"] == 1


@requires_ast_grep
class TestASTSearchFilePattern:
    """Tests for file pattern filtering."""

    def test_file_pattern_glob(self):
        """Test file pattern glob filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "test.py").write_text("def hello():\n    pass\n")
            (Path(tmpdir) / "other.py").write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def hello(): $$$",
                "path": tmpdir,
                "file_pattern": "test*.py",
            })

            # Should only find test.py
            assert result["total_matches"] == 1
            assert result["matches"][0]["file"] == "test.py"

    def test_file_pattern_subdirectory(self):
        """Test file pattern in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "app.py").write_text("def hello():\n    pass\n")
            (Path(tmpdir) / "main.py").write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            result = plugin._execute_ast_search({
                "pattern": "def hello(): $$$",
                "path": tmpdir,
                "file_pattern": "src/**/*.py",
            })

            # Should only find src/app.py
            assert result["total_matches"] == 1
            assert "src" in result["matches"][0]["file"]


class TestBackgroundCapable:
    """Tests for background execution support."""

    def test_supports_background(self):
        """Test that ast_search supports background execution."""
        plugin = ASTSearchPlugin()

        assert plugin.supports_background("ast_search") is True
        assert plugin.supports_background("unknown_tool") is False

    def test_estimate_duration(self):
        """Test duration estimation."""
        plugin = ASTSearchPlugin()

        # Directory search should estimate longer
        duration = plugin.estimate_duration("ast_search", {"path": "/some/dir"})
        assert duration is not None
        assert duration > 5.0


class TestStreamingCapable:
    """Tests for streaming execution support."""

    def test_supports_streaming(self):
        """Test that ast_search supports streaming execution."""
        plugin = ASTSearchPlugin()

        assert plugin.supports_streaming("ast_search") is True
        assert plugin.supports_streaming("unknown_tool") is False

    def test_get_streaming_tool_names(self):
        """Test get_streaming_tool_names returns ast_search."""
        plugin = ASTSearchPlugin()

        streaming_tools = plugin.get_streaming_tool_names()
        assert "ast_search" in streaming_tools
        assert len(streaming_tools) == 1


@requires_ast_grep
class TestStreamingExecution:
    """Tests for streaming execution of ast_search."""

    @pytest.mark.asyncio
    async def test_streaming_basic(self):
        """Test basic streaming execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def hello():\n"
                "    pass\n"
                "\n"
                "def world():\n"
                "    pass\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            chunks = []
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {"pattern": "def $FUNC(): $$$", "path": tmpdir, "language": "python"},
            ):
                chunks.append(chunk)

            # Should have progress, 2 matches, and summary
            assert len(chunks) >= 3

            # First chunk should be progress
            assert chunks[0].chunk_type == "progress"

            # Should have match chunks
            match_chunks = [c for c in chunks if c.chunk_type == "match"]
            assert len(match_chunks) == 2

            # Last chunk should be summary
            assert chunks[-1].chunk_type == "summary"
            assert chunks[-1].metadata["total_matches"] == 2

    @pytest.mark.asyncio
    async def test_streaming_with_callback(self):
        """Test streaming execution with callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello():\n    pass\n")

            plugin = ASTSearchPlugin()
            plugin.initialize()

            callback_chunks = []

            def on_chunk(chunk):
                callback_chunks.append(chunk)

            chunks = []
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {"pattern": "def $FUNC(): $$$", "path": tmpdir, "language": "python"},
                on_chunk=on_chunk,
            ):
                chunks.append(chunk)

            # Callback should receive same chunks
            assert len(callback_chunks) == len(chunks)

    @pytest.mark.asyncio
    async def test_streaming_error_empty_pattern(self):
        """Test streaming error for empty pattern."""
        plugin = ASTSearchPlugin()
        plugin.initialize()

        chunks = []
        async for chunk in plugin.execute_streaming(
            "ast_search",
            {"pattern": "", "path": "/tmp"},
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "error"
        assert "Pattern is required" in chunks[0].content

    @pytest.mark.asyncio
    async def test_streaming_error_invalid_tool(self):
        """Test streaming error for unsupported tool."""
        plugin = ASTSearchPlugin()
        plugin.initialize()

        chunks = []
        async for chunk in plugin.execute_streaming(
            "unknown_tool",
            {"pattern": "test"},
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "error"
        assert "not supported" in chunks[0].content

    @pytest.mark.asyncio
    async def test_streaming_max_results(self):
        """Test streaming respects max_results limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many functions
            funcs = "\n".join([f"def func{i}():\n    pass\n" for i in range(10)])
            (Path(tmpdir) / "test.py").write_text(funcs)

            plugin = ASTSearchPlugin()
            plugin.initialize()

            chunks = []
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {
                    "pattern": "def $FUNC(): $$$",
                    "path": tmpdir,
                    "language": "python",
                    "max_results": 3,
                },
            ):
                chunks.append(chunk)

            # Should have progress, 3 matches (limited), and summary
            match_chunks = [c for c in chunks if c.chunk_type == "match"]
            assert len(match_chunks) == 3

            # Summary should indicate truncation
            summary = chunks[-1]
            assert summary.chunk_type == "summary"
            assert summary.metadata["truncated"] is True

    @pytest.mark.asyncio
    async def test_streaming_sequence_numbers(self):
        """Test that streaming chunks have correct sequence numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text(
                "def a():\n    pass\n"
                "def b():\n    pass\n"
            )

            plugin = ASTSearchPlugin()
            plugin.initialize()

            chunks = []
            async for chunk in plugin.execute_streaming(
                "ast_search",
                {"pattern": "def $FUNC(): $$$", "path": tmpdir, "language": "python"},
            ):
                chunks.append(chunk)

            # Match and summary chunks should have incrementing sequence numbers
            sequenced_chunks = [c for c in chunks if c.sequence > 0]
            sequences = [c.sequence for c in sequenced_chunks]

            # Sequences should be monotonically increasing
            assert sequences == sorted(sequences)
            assert len(set(sequences)) == len(sequences)  # All unique
