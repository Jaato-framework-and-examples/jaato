"""Tests for the prompt library plugin."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ..plugin import (
    PromptLibraryPlugin,
    create_plugin,
    PromptInfo,
    PromptParam,
    PROMPT_ENTRY_FILE,
    SKILL_ENTRY_FILE,
)


class TestPromptLibraryPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, PromptLibraryPlugin)

    def test_plugin_name(self):
        plugin = PromptLibraryPlugin()
        assert plugin.name == "prompt_library"

    def test_initialize_without_config(self):
        plugin = PromptLibraryPlugin()
        plugin.initialize()
        assert plugin._initialized is True

    def test_initialize_with_workspace_path(self):
        plugin = PromptLibraryPlugin()
        plugin.initialize({"workspace_path": "/test/path"})
        assert plugin._workspace_path == "/test/path"

    def test_initialize_discovers_prompts(self):
        """Initialize should pre-discover prompts."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a prompt before initializing
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "test.md").write_text("# Test")

            plugin.initialize({"workspace_path": tmpdir})

            # Cache should be populated
            assert "test" in plugin._prompt_cache

    def test_shutdown(self):
        plugin = PromptLibraryPlugin()
        plugin.initialize()
        plugin.shutdown()
        assert plugin._initialized is False
        assert plugin._prompt_cache == {}

    def test_set_workspace_path(self):
        plugin = PromptLibraryPlugin()
        plugin.set_workspace_path("/new/path")
        assert plugin._workspace_path == "/new/path"


class TestPromptLibraryToolSchemas:
    """Tests for tool schemas."""

    def test_get_tool_schemas_includes_savePrompt(self):
        """savePrompt should always be included."""
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()

        names = {s.name for s in schemas}
        assert "savePrompt" in names

    def test_get_tool_schemas_excludes_old_tools(self):
        """listPrompts and usePrompt should NOT be in schemas (replaced by prompt.* tools)."""
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()

        names = {s.name for s in schemas}
        assert "listPrompts" not in names
        assert "usePrompt" not in names

    def test_get_tool_schemas_includes_prompt_tools(self):
        """Each discovered prompt should have a prompt.name tool."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review code
---
Content.
""")
            (prompts_dir / "explain.md").write_text("""---
description: Explain code
---
Content.
""")

            schemas = plugin.get_tool_schemas()

            names = {s.name for s in schemas}
            assert "prompt.review" in names
            assert "prompt.explain" in names

    def test_savePrompt_schema(self):
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()
        save_prompt = next(s for s in schemas if s.name == "savePrompt")

        assert "name" in save_prompt.parameters["properties"]
        assert "content" in save_prompt.parameters["properties"]
        assert "description" in save_prompt.parameters["properties"]
        assert "tags" in save_prompt.parameters["properties"]
        assert "global" in save_prompt.parameters["properties"]
        assert save_prompt.category == "prompt"

    def test_prompt_tool_schema_structure(self):
        """Prompt tools should have correct structure."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "code-review.md").write_text("""---
description: Review code for issues
params:
  file:
    required: true
    description: File to review
  focus:
    default: all
    description: Focus area
---
Review {{file}} for {{focus}} issues.
""")

            schemas = plugin.get_tool_schemas()
            tool = next(s for s in schemas if s.name == "prompt.code-review")

            assert tool.description == "Review code for issues"
            assert tool.category == "prompt"
            assert tool.discoverability == "discoverable"
            assert "file" in tool.parameters["properties"]
            assert "focus" in tool.parameters["properties"]
            assert "file" in tool.parameters["required"]


class TestPromptToToolConversion:
    """Tests for prompt-to-tool schema conversion."""

    def test_params_to_json_schema_empty(self):
        """Empty params should produce empty schema."""
        plugin = PromptLibraryPlugin()
        schema = plugin._params_to_json_schema({})

        assert schema == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_params_to_json_schema_required_param(self):
        """Required params should be in required list."""
        plugin = PromptLibraryPlugin()
        params = {
            "file": PromptParam(name="file", required=True, description="The file")
        }
        schema = plugin._params_to_json_schema(params)

        assert "file" in schema["properties"]
        assert schema["properties"]["file"]["type"] == "string"
        assert schema["properties"]["file"]["description"] == "The file"
        assert "file" in schema["required"]

    def test_params_to_json_schema_optional_param(self):
        """Optional params should not be in required list."""
        plugin = PromptLibraryPlugin()
        params = {
            "format": PromptParam(
                name="format",
                required=False,
                default="json",
                description="Output format"
            )
        }
        schema = plugin._params_to_json_schema(params)

        assert "format" in schema["properties"]
        assert schema["properties"]["format"]["default"] == "json"
        assert "format" not in schema["required"]

    def test_params_to_json_schema_with_enum(self):
        """Params with enum should include enum in schema."""
        plugin = PromptLibraryPlugin()
        params = {
            "level": PromptParam(
                name="level",
                required=True,
                enum=["low", "medium", "high"],
                description="Priority level"
            )
        }
        schema = plugin._params_to_json_schema(params)

        assert schema["properties"]["level"]["enum"] == ["low", "medium", "high"]

    def test_prompt_to_tool_schema(self):
        """PromptInfo should convert to valid ToolSchema."""
        plugin = PromptLibraryPlugin()
        info = PromptInfo(
            name="my-prompt",
            description="Does something useful",
            source="project",
            path=Path("/test/path"),
            params={
                "file": PromptParam(name="file", required=True, description="Input file")
            }
        )

        schema = plugin._prompt_to_tool_schema(info)

        assert schema.name == "prompt.my-prompt"
        assert schema.description == "Does something useful"
        assert schema.category == "prompt"
        assert schema.discoverability == "discoverable"
        assert "file" in schema.parameters["properties"]


class TestPromptLibraryExecutors:
    """Tests for executor methods."""

    def test_get_executors_includes_savePrompt(self):
        plugin = PromptLibraryPlugin()
        executors = plugin.get_executors()

        assert "savePrompt" in executors
        assert "prompt" in executors
        assert callable(executors["savePrompt"])
        assert callable(executors["prompt"])

    def test_get_executors_excludes_old_tools(self):
        """listPrompts and usePrompt executors should be removed."""
        plugin = PromptLibraryPlugin()
        executors = plugin.get_executors()

        assert "listPrompts" not in executors
        assert "usePrompt" not in executors

    def test_get_executors_includes_prompt_tools(self):
        """Each discovered prompt should have an executor."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review")

            executors = plugin.get_executors()

            assert "prompt.review" in executors
            assert callable(executors["prompt.review"])


class TestPromptToolExecution:
    """Tests for prompt tool execution."""

    def test_execute_prompt_tool_success(self):
        """Calling prompt.name tool should return content."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review code
---
Review the code for bugs.
""")

            result = plugin._execute_prompt_tool("review", {})

            assert "content" in result
            assert "Review the code" in result["content"]
            assert result["source"] == "project"

    def test_execute_prompt_tool_includes_skill_path_and_instruction(self):
        """Prompt tool result should include skill_path and execution instruction."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review code
---
Review the code for bugs.
""")

            result = plugin._execute_prompt_tool("review", {})

            # skill_path should be present and point to the prompts directory
            assert "skill_path" in result
            assert result["skill_path"] == str(prompts_dir)

            # instruction should mention using tools and the skill path
            assert "instruction" in result
            assert "Execute" in result["instruction"] or "execute" in result["instruction"]
            assert str(prompts_dir) in result["instruction"]

    def test_execute_prompt_tool_directory_skill_path(self):
        """Directory-based prompt should have skill_path pointing to the skill directory."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            skill_dir = Path(tmpdir) / ".jaato" / "skills" / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("""---
description: My skill
---
Run ./scripts/generate.py
""")
            # Create the scripts directory too
            (skill_dir / "scripts").mkdir()
            (skill_dir / "scripts" / "generate.py").write_text("# script")

            result = plugin._execute_prompt_tool("my-skill", {})

            # skill_path should point to the skill directory, not its parent
            assert "skill_path" in result
            assert result["skill_path"] == str(skill_dir)

    def test_execute_prompt_tool_with_params(self):
        """Prompt tool should substitute parameters."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review file
---
Review {{file}} for {{focus}} issues.
""")

            result = plugin._execute_prompt_tool("review", {
                "file": "main.py",
                "focus": "security"
            })

            assert "main.py" in result["content"]
            assert "security" in result["content"]

    def test_execute_prompt_tool_not_found(self):
        """Non-existent prompt should return error."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._execute_prompt_tool("nonexistent", {})

            assert "error" in result
            assert "not found" in result["error"].lower()

    def test_execute_prompt_tool_missing_params(self):
        """Missing required params should be reported."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review
---
Review {{file}} for issues.
""")

            result = plugin._execute_prompt_tool("review", {})

            assert "missing_params" in result
            assert "file" in result["missing_params"]


class TestPromptLibraryUserCommands:
    """Tests for user commands."""

    def test_get_user_commands(self):
        plugin = PromptLibraryPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        assert commands[0].name == "prompt"
        assert commands[0].share_with_model is True

    def test_get_auto_approved_tools_includes_prompt_tools(self):
        """Prompt tools should be auto-approved (read-only)."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review")

            auto_approved = plugin.get_auto_approved_tools()

            assert "prompt" in auto_approved
            assert "prompt.review" in auto_approved
            # savePrompt should NOT be auto-approved (creates files)
            assert "savePrompt" not in auto_approved

    def test_get_auto_approved_tools_excludes_old_tools(self):
        """Old tools should not be in auto-approved list."""
        plugin = PromptLibraryPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "listPrompts" not in auto_approved
        assert "usePrompt" not in auto_approved


class TestFrontmatterParsing:
    """Tests for YAML frontmatter parsing."""

    def test_parse_frontmatter_simple(self):
        plugin = PromptLibraryPlugin()
        content = """---
description: Test prompt
tags: [test, example]
---

This is the prompt content.
"""
        frontmatter, body = plugin._parse_frontmatter(content)

        assert frontmatter["description"] == "Test prompt"
        assert frontmatter["tags"] == ["test", "example"]
        assert body.strip() == "This is the prompt content."

    def test_parse_frontmatter_with_params(self):
        plugin = PromptLibraryPlugin()
        content = """---
description: Parameterized prompt
params:
  file:
    required: true
    description: File to process
  format:
    default: json
    enum: [json, yaml, xml]
---

Process {{file}} as {{format}}.
"""
        frontmatter, body = plugin._parse_frontmatter(content)

        assert "params" in frontmatter
        assert "file" in frontmatter["params"]
        assert frontmatter["params"]["file"]["required"] is True
        assert frontmatter["params"]["format"]["default"] == "json"

    def test_parse_frontmatter_no_frontmatter(self):
        plugin = PromptLibraryPlugin()
        content = "Just plain content without frontmatter."

        frontmatter, body = plugin._parse_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_parse_frontmatter_invalid_yaml(self):
        plugin = PromptLibraryPlugin()
        content = """---
invalid: yaml: syntax: here
---

Content.
"""
        frontmatter, body = plugin._parse_frontmatter(content)

        # Should return empty frontmatter on parse error
        assert frontmatter == {}


class TestParameterSubstitution:
    """Tests for parameter substitution."""

    def test_substitute_named_params(self):
        plugin = PromptLibraryPlugin()
        content = "Review {{file}} for {{focus}} issues."
        substituted, missing = plugin._substitute_params(
            content,
            {"file": "main.py", "focus": "security"},
            []
        )

        assert substituted == "Review main.py for security issues."
        assert missing == []

    def test_substitute_named_params_with_defaults(self):
        plugin = PromptLibraryPlugin()
        content = "Review {{file}} for {{focus:all}} issues."
        substituted, missing = plugin._substitute_params(
            content,
            {"file": "main.py"},
            []
        )

        assert substituted == "Review main.py for all issues."
        assert missing == []

    def test_substitute_named_params_missing(self):
        plugin = PromptLibraryPlugin()
        content = "Review {{file}} for {{focus}} issues."
        substituted, missing = plugin._substitute_params(
            content,
            {"file": "main.py"},
            []
        )

        assert "main.py" in substituted
        assert "{{focus}}" in substituted  # Left unreplaced
        assert "focus" in missing

    def test_substitute_positional_params(self):
        plugin = PromptLibraryPlugin()
        content = "Review {{$1}} in directory {{$2}}."
        substituted, missing = plugin._substitute_params(
            content,
            {},
            ["main.py", "src/"]
        )

        assert substituted == "Review main.py in directory src/."
        assert missing == []

    def test_substitute_positional_params_all_args(self):
        plugin = PromptLibraryPlugin()
        content = "Process these files: {{$0}}"
        substituted, missing = plugin._substitute_params(
            content,
            {},
            ["a.py", "b.py", "c.py"]
        )

        assert substituted == "Process these files: a.py b.py c.py"
        assert missing == []

    def test_substitute_claude_arguments_placeholder(self):
        plugin = PromptLibraryPlugin()
        content = "Process: $ARGUMENTS"
        substituted, missing = plugin._substitute_params(
            content,
            {},
            ["file1.py", "file2.py"]
        )

        assert substituted == "Process: file1.py file2.py"

    def test_substitute_positional_with_defaults(self):
        plugin = PromptLibraryPlugin()
        content = "Review {{$1}} for {{$2:all}} issues."
        substituted, missing = plugin._substitute_params(
            content,
            {},
            ["main.py"]
        )

        assert substituted == "Review main.py for all issues."
        assert missing == []


class TestPromptDiscovery:
    """Tests for prompt discovery from filesystem."""

    def test_discover_prompts_empty(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock home to avoid finding global prompts
            with patch.object(Path, 'home', return_value=Path(tmpdir) / "fake_home"):
                plugin.set_workspace_path(tmpdir)
                prompts = plugin._discover_prompts()
                assert prompts == {}

    def test_discover_prompts_single_file(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create prompts directory and file
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review code
tags: [review]
---

Review the code for issues.
""")

            prompts = plugin._discover_prompts()

            assert "review" in prompts
            assert prompts["review"].description == "Review code"
            assert prompts["review"].source == "project"
            assert "review" in prompts["review"].tags

    def test_discover_prompts_directory_based(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create directory-based prompt
            prompt_dir = Path(tmpdir) / ".jaato" / "prompts" / "api-design"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / PROMPT_ENTRY_FILE).write_text("""---
description: Design API endpoints
---

Design the API.
""")
            (prompt_dir / "examples.md").write_text("Example content")

            prompts = plugin._discover_prompts()

            assert "api-design" in prompts
            assert prompts["api-design"].is_directory is True

    def test_discover_claude_skills(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create Claude skill
            skill_dir = Path(tmpdir) / ".claude" / "skills" / "explain-code"
            skill_dir.mkdir(parents=True)
            (skill_dir / SKILL_ENTRY_FILE).write_text("""---
name: explain-code
description: Explain code with diagrams
---

When explaining code, include diagrams.
""")

            prompts = plugin._discover_prompts()

            assert "explain-code" in prompts
            assert prompts["explain-code"].source == "claude-skills"

    def test_priority_jaato_over_claude(self):
        """Jaato prompts should take priority over Claude skills."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create both jaato prompt and claude skill with same name
            jaato_dir = Path(tmpdir) / ".jaato" / "prompts"
            jaato_dir.mkdir(parents=True)
            (jaato_dir / "review.md").write_text("""---
description: Jaato review prompt
---
Jaato content.
""")

            claude_dir = Path(tmpdir) / ".claude" / "skills" / "review"
            claude_dir.mkdir(parents=True)
            (claude_dir / SKILL_ENTRY_FILE).write_text("""---
description: Claude review skill
---
Claude content.
""")

            prompts = plugin._discover_prompts()

            assert "review" in prompts
            assert prompts["review"].source == "project"
            assert prompts["review"].description == "Jaato review prompt"


class TestExecuteSavePrompt:
    """Tests for savePrompt executor."""

    def test_save_prompt_success(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._execute_save_prompt({
                "name": "test-prompt",
                "content": "This is test content.",
                "description": "A test prompt",
                "tags": ["test"],
            })

            assert result.get("success") is True
            assert "test-prompt" in result["path"]

            # Verify file was created
            prompt_path = Path(tmpdir) / ".jaato" / "prompts" / "test-prompt.md"
            assert prompt_path.exists()
            content = prompt_path.read_text()
            assert "description: A test prompt" in content

    def test_save_prompt_invalid_name(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._execute_save_prompt({
                "name": "Invalid Name!",
                "content": "Content",
                "description": "Desc",
            })

            assert "error" in result
            assert "Invalid" in result["error"]

    def test_save_prompt_already_exists(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "existing.md").write_text("Existing content")

            result = plugin._execute_save_prompt({
                "name": "existing",
                "content": "New content",
                "description": "New desc",
            })

            assert "error" in result
            assert "already exists" in result["error"]

    def test_save_prompt_missing_fields(self):
        plugin = PromptLibraryPlugin()

        result = plugin._execute_save_prompt({"name": "test"})
        assert "error" in result

        result = plugin._execute_save_prompt({"name": "test", "content": "x"})
        assert "error" in result


class TestExecutePromptCommand:
    """Tests for the prompt user command."""

    def test_prompt_command_list(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review code")

            result = plugin._execute_prompt_command({"args": []})

            assert "Available prompts" in result
            assert "review" in result

    def test_prompt_command_use(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "greet.md").write_text("""---
description: Greet someone
---
Hello {{$1}}!
""")

            result = plugin._execute_prompt_command({"args": ["greet", "World"]})

            assert "Hello World!" in result

    def test_prompt_command_not_found(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._execute_prompt_command({"args": ["nonexistent"]})

            assert "not found" in result.lower()


class TestCommandCompletions:
    """Tests for command completions."""

    def test_completions_for_prompt_names(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review")
            (prompts_dir / "refactor.md").write_text("# Refactor")

            completions = plugin.get_command_completions("prompt", [])

            names = [c.value for c in completions]
            assert "review" in names
            assert "refactor" in names

    def test_completions_with_prefix(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review")
            (prompts_dir / "refactor.md").write_text("# Refactor")
            (prompts_dir / "explain.md").write_text("# Explain")

            completions = plugin.get_command_completions("prompt", ["re"])

            names = [c.value for c in completions]
            assert "review" in names
            assert "refactor" in names
            assert "explain" not in names

    def test_completions_wrong_command(self):
        plugin = PromptLibraryPlugin()
        completions = plugin.get_command_completions("other", [])
        assert completions == []


class TestSystemInstructions:
    """Tests for system instructions."""

    def test_system_instructions_with_prompts(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("# Review")

            instructions = plugin.get_system_instructions()

            # Should mention prompt tools, not old tools
            assert "prompt.review" in instructions
            assert "savePrompt" in instructions
            assert "list_tools" in instructions
            # Old tools should NOT be mentioned
            assert "listPrompts" not in instructions
            assert "usePrompt" not in instructions

    def test_system_instructions_no_prompts(self):
        """With no prompts, system instructions should be None."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock home to avoid finding global prompts
            with patch.object(Path, 'home', return_value=Path(tmpdir) / "fake_home"):
                plugin.set_workspace_path(tmpdir)

                instructions = plugin.get_system_instructions()

                assert instructions is None


class TestGitHubPathFetch:
    """Tests for GitHub path-specific fetch functionality."""

    def test_fetch_github_parses_path(self):
        """GitHub spec with path should parse correctly."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Mock gh not available - should return None and fall back to git
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = FileNotFoundError("gh not found")

                result = plugin._fetch_github_path("owner", "repo", "skill-path", Path(tmpdir))

                assert result is None  # Should return None to trigger fallback

    def test_fetch_github_path_extracts_components(self):
        """Verify path extraction from repo_spec."""
        plugin = PromptLibraryPlugin()

        # Test that _fetch_from_github correctly parses owner/repo/path
        parts = "mkdev-me/claude-skills/gemini-image-generator".split('/')
        assert len(parts) == 3
        assert parts[0] == "mkdev-me"
        assert parts[1] == "claude-skills"
        assert '/'.join(parts[2:]) == "gemini-image-generator"

    def test_fetch_github_path_with_nested_path(self):
        """Nested paths should be handled correctly."""
        parts = "owner/repo/path/to/skill".split('/')
        assert len(parts) == 5
        assert parts[0] == "owner"
        assert parts[1] == "repo"
        assert '/'.join(parts[2:]) == "path/to/skill"

    def test_fetch_github_directory_creates_skill(self):
        """Fetching a directory should create it in skills/ dir."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Mock the GitHub API response for a directory
            mock_contents = [
                {"name": "SKILL.md", "type": "file", "download_url": None,
                 "content": "IyBUZXN0IFNraWxs"},  # base64 of "# Test Skill"
            ]

            result = plugin._fetch_github_directory(
                owner="owner",
                repo="repo",
                path="test-skill",
                contents=mock_contents,
                dest_dir=Path(tmpdir) / ".jaato" / "skills"
            )

            assert result.success is True
            assert "test-skill" in result.prompts_fetched

    def test_fetch_github_file_validates_md_extension(self):
        """Fetching a non-md file should fail."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._fetch_github_file(
                file_info={"name": "file.txt"},
                dest_dir=Path(tmpdir),
                source_ref="owner/repo/file.txt"
            )

            assert result.success is False
            assert "Expected .md file" in result.error

    def test_get_github_file_content_from_base64(self):
        """Should decode base64 content correctly."""
        import base64
        plugin = PromptLibraryPlugin()

        content_text = "# Test Prompt\nThis is a test."
        encoded = base64.b64encode(content_text.encode()).decode()

        file_info = {"content": encoded}
        result = plugin._get_github_file_content(file_info)

        assert result == content_text

    def test_fetch_help_shows_github_path_syntax(self):
        """Help text should document GitHub path syntax."""
        plugin = PromptLibraryPlugin()

        result = plugin._handle_fetch_subcommand([])

        assert "github <owner/repo/path>" in result
        assert "mkdev-me/claude-skills/gemini-image-generator" in result


class TestToolsChangedNotification:
    """Tests for the tools changed notification mechanism."""

    def test_set_on_tools_changed_callback(self):
        """Setting the callback should store it."""
        plugin = PromptLibraryPlugin()
        called_with = []

        def callback(new_tools):
            called_with.append(new_tools)

        plugin.set_on_tools_changed(callback)
        assert plugin._on_tools_changed is callback

    def test_notify_tools_changed_calls_callback(self):
        """Notification should call the callback with new tool names."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            called_with = []
            def callback(new_tools):
                called_with.append(new_tools)

            plugin.set_on_tools_changed(callback)
            plugin._notify_tools_changed(["prompt.new-skill"])

            assert len(called_with) == 1
            assert called_with[0] == ["prompt.new-skill"]

    def test_notify_tools_changed_refreshes_cache(self):
        """Notification should refresh the prompt cache."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create initial state
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "existing.md").write_text("# Existing prompt")

            # Discover prompts
            prompts = plugin._discover_prompts()
            assert "existing" in prompts

            # Add new prompt to disk
            (prompts_dir / "new-skill.md").write_text("# New skill")

            # Cache should not have the new prompt yet (still has old cache)
            assert "new-skill" not in plugin._prompt_cache

            # Notify triggers refresh
            plugin._notify_tools_changed(["prompt.new-skill"])

            # Now cache should have the new prompt
            assert "new-skill" in plugin._prompt_cache

    def test_set_plugin_registry_stores_reference(self):
        """Setting the registry should store a reference."""
        plugin = PromptLibraryPlugin()

        mock_registry = object()
        plugin.set_plugin_registry(mock_registry)

        assert plugin._plugin_registry is mock_registry


class TestRemoveSubcommand:
    """Tests for the prompt remove subcommand."""

    def test_remove_shows_help_without_args(self):
        """Remove without args should show help."""
        plugin = PromptLibraryPlugin()

        result = plugin._handle_remove_subcommand([])

        assert "Usage: prompt remove <name>" in result
        assert "Examples:" in result

    def test_remove_prompt_not_found(self):
        """Removing non-existent prompt should fail."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            result = plugin._handle_remove_subcommand(["nonexistent"])

            assert "Prompt not found: nonexistent" in result

    def test_remove_prompt_success(self):
        """Removing a writable prompt should succeed."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a prompt
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            prompt_file = prompts_dir / "test-prompt.md"
            prompt_file.write_text("# Test prompt")

            # Mock confirmation to return "yes"
            plugin.set_confirm_callback(lambda msg, opts: "yes")

            result = plugin._handle_remove_subcommand(["test-prompt"])

            assert "Removed prompt 'test-prompt'" in result
            assert not prompt_file.exists()

    def test_remove_prompt_cancelled(self):
        """Cancelling removal should not delete the file."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a prompt
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            prompt_file = prompts_dir / "test-prompt.md"
            prompt_file.write_text("# Test prompt")

            # Mock confirmation to return "no"
            plugin.set_confirm_callback(lambda msg, opts: "no")

            result = plugin._handle_remove_subcommand(["test-prompt"])

            assert "Removal cancelled" in result
            assert prompt_file.exists()

    def test_remove_directory_prompt(self):
        """Removing a directory-based prompt should remove the entire directory."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a directory-based prompt
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            skill_dir = prompts_dir / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "PROMPT.md").write_text("# My Skill")
            (skill_dir / "helper.py").write_text("# Helper file")

            # Mock confirmation to return "yes"
            plugin.set_confirm_callback(lambda msg, opts: "yes")

            result = plugin._handle_remove_subcommand(["my-skill"])

            assert "Removed prompt 'my-skill'" in result
            assert not skill_dir.exists()

    def test_remove_readonly_prompt_fails(self):
        """Cannot remove prompts from read-only locations."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a prompt in a read-only location (.claude/skills)
            claude_skills = Path(tmpdir) / ".claude" / "skills" / "readonly-skill"
            claude_skills.mkdir(parents=True)
            (claude_skills / "SKILL.md").write_text("# Read-only skill")

            result = plugin._handle_remove_subcommand(["readonly-skill"])

            assert "read-only location" in result
            assert claude_skills.exists()

    def test_remove_notifies_tools_changed(self):
        """Removing a prompt should trigger tools changed notification."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a prompt
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "test-prompt.md").write_text("# Test prompt")

            # Track notification
            notified_tools = []
            plugin.set_on_tools_changed(lambda tools: notified_tools.extend(tools))
            plugin.set_confirm_callback(lambda msg, opts: "yes")

            plugin._handle_remove_subcommand(["test-prompt"])

            assert "prompt.test-prompt" in notified_tools

    def test_remove_completions_only_writable(self):
        """Remove completions should only show writable prompts."""
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            # Create a writable prompt
            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "writable.md").write_text("# Writable")

            # Create a read-only prompt
            claude_skills = Path(tmpdir) / ".claude" / "skills" / "readonly"
            claude_skills.mkdir(parents=True)
            (claude_skills / "SKILL.md").write_text("# Read-only")

            completions = plugin.get_command_completions("prompt", ["remove", ""])

            completion_names = [c.value for c in completions]
            assert "writable" in completion_names
            assert "readonly" not in completion_names
