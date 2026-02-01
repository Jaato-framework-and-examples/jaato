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

    def test_get_tool_schemas(self):
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 3
        names = {s.name for s in schemas}
        assert "listPrompts" in names
        assert "usePrompt" in names
        assert "savePrompt" in names

    def test_listPrompts_schema(self):
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()
        list_prompts = next(s for s in schemas if s.name == "listPrompts")

        assert list_prompts.category == "introspection"
        assert "tag" in list_prompts.parameters["properties"]
        assert "search" in list_prompts.parameters["properties"]

    def test_usePrompt_schema(self):
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()
        use_prompt = next(s for s in schemas if s.name == "usePrompt")

        assert "name" in use_prompt.parameters["properties"]
        assert "params" in use_prompt.parameters["properties"]
        assert "args" in use_prompt.parameters["properties"]
        assert "name" in use_prompt.parameters["required"]

    def test_savePrompt_schema(self):
        plugin = PromptLibraryPlugin()
        schemas = plugin.get_tool_schemas()
        save_prompt = next(s for s in schemas if s.name == "savePrompt")

        assert "name" in save_prompt.parameters["properties"]
        assert "content" in save_prompt.parameters["properties"]
        assert "description" in save_prompt.parameters["properties"]
        assert "tags" in save_prompt.parameters["properties"]
        assert "global" in save_prompt.parameters["properties"]


class TestPromptLibraryExecutors:
    """Tests for executor methods."""

    def test_get_executors(self):
        plugin = PromptLibraryPlugin()
        executors = plugin.get_executors()

        assert "listPrompts" in executors
        assert "usePrompt" in executors
        assert "savePrompt" in executors
        assert "prompt" in executors
        for executor in executors.values():
            assert callable(executor)


class TestPromptLibraryUserCommands:
    """Tests for user commands."""

    def test_get_user_commands(self):
        plugin = PromptLibraryPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        assert commands[0].name == "prompt"
        assert commands[0].share_with_model is True

    def test_get_auto_approved_tools(self):
        plugin = PromptLibraryPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "listPrompts" in auto_approved
        assert "usePrompt" in auto_approved
        assert "prompt" in auto_approved
        # savePrompt should NOT be auto-approved
        assert "savePrompt" not in auto_approved


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


class TestExecuteListPrompts:
    """Tests for listPrompts executor."""

    def test_list_prompts_empty(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)
            result = plugin._execute_list_prompts({})

            assert result["total"] == 0
            assert result["prompts"] == []

    def test_list_prompts_with_filter(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "review.md").write_text("""---
description: Review code
tags: [review, quality]
---
Content.
""")
            (prompts_dir / "explain.md").write_text("""---
description: Explain code
tags: [docs]
---
Content.
""")

            # Filter by tag
            result = plugin._execute_list_prompts({"tag": "review"})
            assert result["total"] == 1
            assert result["prompts"][0]["name"] == "review"

    def test_list_prompts_with_search(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            prompts_dir = Path(tmpdir) / ".jaato" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "security-review.md").write_text("# Security focused review")
            (prompts_dir / "code-style.md").write_text("# Check code style")

            result = plugin._execute_list_prompts({"search": "security"})
            assert result["total"] == 1
            assert "security" in result["prompts"][0]["name"]


class TestExecuteUsePrompt:
    """Tests for usePrompt executor."""

    def test_use_prompt_not_found(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)
            result = plugin._execute_use_prompt({"name": "nonexistent"})

            assert "error" in result
            assert "not found" in result["error"].lower()

    def test_use_prompt_success(self):
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

            result = plugin._execute_use_prompt({"name": "review"})

            assert "content" in result
            assert "Review the code" in result["content"]
            assert result["source"] == "project"

    def test_use_prompt_with_params(self):
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

            result = plugin._execute_use_prompt({
                "name": "review",
                "params": {"file": "main.py", "focus": "security"}
            })

            assert "main.py" in result["content"]
            assert "security" in result["content"]

    def test_use_prompt_no_name(self):
        plugin = PromptLibraryPlugin()
        result = plugin._execute_use_prompt({})

        assert "error" in result
        assert "No prompt name" in result["error"]


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

            assert "listPrompts" in instructions
            assert "usePrompt" in instructions
            assert "savePrompt" in instructions
            assert "review" in instructions

    def test_system_instructions_no_prompts(self):
        plugin = PromptLibraryPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin.set_workspace_path(tmpdir)

            instructions = plugin.get_system_instructions()

            assert "no prompts available" in instructions.lower()
