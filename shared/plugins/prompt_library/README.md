# Prompt Library Plugin

A plugin for managing reusable prompts and skills, with full Claude Code interoperability.

## Overview

The Prompt Library plugin provides:
- **User commands**: `prompt` for listing, using, and fetching prompts
- **Model tools**: `listPrompts`, `usePrompt`, `savePrompt` for programmatic access
- **Claude Code compatibility**: Reads Claude's skills and supports ClawdHub installation

## Storage Locations

Prompts and skills are discovered from multiple locations in priority order:

| Location | Source Name | Format | Writable | Description |
|----------|-------------|--------|----------|-------------|
| `.jaato/prompts/` | project | PROMPT.md | ✓ | Jaato native prompts (project-scoped) |
| `~/.jaato/prompts/` | global | PROMPT.md | ✓ | Jaato native prompts (user-scoped) |
| `.jaato/skills/` | project-skills | SKILL.md | ✓ | Claude-format skills installed via ClawdHub (project) |
| `~/.jaato/skills/` | global-skills | SKILL.md | ✓ | Claude-format skills installed via ClawdHub (user) |
| `.claude/skills/` | claude-skills | SKILL.md | ✗ | Claude Code's own skills (read-only interop) |
| `.claude/commands/` | claude-commands | SKILL.md | ✗ | Claude Code legacy commands (read-only interop) |
| `~/.claude/skills/` | claude-global | SKILL.md | ✗ | Claude Code's global skills (read-only interop) |

### Priority

When prompts share the same name across locations, higher-priority (earlier in list) locations win.

### Directory Structure

**Single-file prompt:**
```
.jaato/prompts/review.md
```

**Directory-based prompt (with supporting files):**
```
.jaato/prompts/api-design/
├── PROMPT.md           # Entry point (required)
├── template.md         # Supporting template
└── examples/
    └── sample.md       # Example output
```

**Claude-format skill:**
```
.jaato/skills/explain-code/
├── SKILL.md            # Entry point (required)
└── scripts/
    └── helper.py       # Supporting script
```

## User Commands

### `prompt` - List and use prompts

```bash
# List all available prompts
prompt

# List prompts filtered by tag
prompt --tag security

# Use a prompt with arguments
prompt review src/main.py

# Use a prompt with named parameters
prompt api-design endpoint=/users method=POST
```

### `prompt fetch` - Fetch prompts from external sources

```bash
# Install skill via ClawdHub (recommended for Claude-compatible skills)
prompt fetch npx clawdhub@latest install some-skill

# Install to user directory instead of project
prompt fetch npx clawdhub@latest install some-skill user

# Fetch from git repository
prompt fetch git https://github.com/user/prompts

# Fetch from GitHub (shorthand)
prompt fetch github user/prompts

# Fetch single prompt from URL
prompt fetch url https://example.com/review.md
```

**Source types:**

| Source | Destination | Description |
|--------|-------------|-------------|
| `npx <package> [args...]` | `.jaato/skills/` | Run npx command (ClawdHub, etc.) |
| `git <repo_url>` | `.jaato/prompts/` | Clone git repository |
| `github <owner/repo>` | `.jaato/prompts/` | GitHub shorthand |
| `url <url>` | `.jaato/prompts/` | Single file download |

## Model Tools

### `listPrompts`

Discover available prompts.

```json
{
  "tag": "security",
  "search": "review"
}
```

### `usePrompt`

Retrieve and expand a prompt with parameters.

```json
{
  "name": "code-review",
  "params": {
    "file": "src/auth.py",
    "focus": "security"
  }
}
```

### `savePrompt`

Create a new prompt.

```json
{
  "name": "security-review",
  "description": "Review code for security vulnerabilities",
  "content": "Review {{file}} for security issues:\n\n1. Check for injection vulnerabilities\n2. Look for auth issues",
  "tags": ["security", "review"]
}
```

## Template Syntax

Prompts support parameter substitution:

| Pattern | Description | Example |
|---------|-------------|---------|
| `{{name}}` | Named parameter (required) | `{{file}}` |
| `{{name:default}}` | Named parameter with default | `{{focus:general}}` |
| `{{$1}}`, `{{$2}}` | Positional parameters | `{{$1}}` for first arg |
| `{{$0}}` | All arguments joined | `{{$0}}` |
| `$ARGUMENTS` | Claude Code compatibility | Becomes `{{$0}}` |

**Example prompt:**

```markdown
---
name: code-review
description: Review code with specific focus
tags: [review, code-quality]
---

Review {{file}} with focus on {{focus:general issues}}.

Provide:
1. Summary of findings
2. Specific line numbers
3. Suggested fixes
```

## Frontmatter

Prompts support YAML frontmatter for metadata:

```yaml
---
name: my-prompt
description: What this prompt does
tags: [tag1, tag2]
fetched_from: npx clawdhub@latest install my-prompt
fetched_at: 2024-01-15T10:30:00
---
```

**Fields:**

| Field | Description |
|-------|-------------|
| `name` | Display name (defaults to filename) |
| `description` | What the prompt does |
| `tags` | List of tags for filtering |
| `fetched_from` | Provenance (auto-added by fetch) |
| `fetched_at` | Fetch timestamp (auto-added) |

## Claude Code Interoperability

### Reading Claude's Skills

The plugin automatically discovers skills from Claude Code's directories:
- `.claude/skills/` (project)
- `.claude/commands/` (legacy)
- `~/.claude/skills/` (global)

These are read-only - use Claude Code to manage them directly.

### Installing Claude-Compatible Skills

Use ClawdHub to install skills that work with both Jaato and Claude Code:

```bash
# Install to Jaato's skills directory
prompt fetch npx clawdhub@latest install some-skill

# The skill is installed to .jaato/skills/some-skill/SKILL.md
# and is immediately available via `prompt some-skill`
```

### Skill Format Compatibility

Both `PROMPT.md` (Jaato native) and `SKILL.md` (Claude format) are supported:
- Claude's frontmatter fields (`disable-model-invocation`, `allowed-tools`, etc.) are preserved
- Template syntax is compatible (`$ARGUMENTS` → `{{$0}}`)

## System Instructions

When prompts are available, the plugin injects system instructions that:
1. List available prompts
2. Guide the model to use `listPrompts`/`usePrompt` tools
3. Encourage proactive prompt creation when patterns are detected

## Testing

```bash
# Run all tests
.venv/bin/pytest shared/plugins/prompt_library/tests/ -v

# Run specific test class
.venv/bin/pytest shared/plugins/prompt_library/tests/test_plugin.py::TestPromptDiscovery -v
```

## Configuration

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.expose_plugin("prompt_library", config={
    "workspace_path": "/path/to/project"  # Optional, defaults to cwd
})
```

## License

Same as jaato project.
