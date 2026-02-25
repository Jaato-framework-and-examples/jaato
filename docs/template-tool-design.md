# Template Rendering Tool Design

> **Status: Implemented** - See `shared/plugins/template/` for the implementation.
> The plugin includes both the `writeFileFromTemplate` tool and automatic template extraction
> via prompt enrichment.

## Overview

This document proposes a template rendering capability for code generation tasks within jaato.
The goal is to replace error-prone manual string substitution with reliable, reusable templates.

## Implementation Status

The template plugin has been implemented with two key features:

1. **Template Rendering (`writeFileFromTemplate`)**: Jinja2-based template rendering with variable
   substitution, writing results to files.

2. **Automatic Template Extraction (Prompt Enrichment)**: The plugin subscribes to the prompt
   enrichment pipeline (priority 40) to detect embedded templates in documentation (MODULE.md,
   SKILL.md, etc.) and automatically extract them to `.jaato/templates/` for later use.

See `shared/plugins/template/README.md` for usage documentation.

## Where Should It Live?

**Recommendation: New standalone `template` plugin**

### Analysis of Options

| Option | Pros | Cons |
|--------|------|------|
| Part of `file_edit` | Shares permission model, fewer plugins | Bloats a focused plugin, mixing concerns |
| New `template` plugin | Clean separation, independent evolution, consistent with architecture | One more plugin to manage |
| Extension to existing tool | N/A | No natural fit |

### Rationale

A standalone `template` plugin is the best fit because:

1. **Single Responsibility**: The `file_edit` plugin handles CRUD operations on files. Template rendering has distinct concerns (parsing, variable substitution, potentially template discovery/caching).

2. **Independent Evolution**: Template syntax and features can evolve without affecting file operations. Future enhancements (partials, includes, caching) can be added cleanly.

3. **Consistent with Architecture**: Looking at existing plugins (`cli`, `todo`, `web_search`, `file_edit`), each addresses a focused domain. The pattern is clear: one plugin per capability area.

4. **Enable/Disable Flexibility**: Some agents may not need templates. Having it as a separate plugin allows selective exposure.

5. **Permission Integration**: Like `file_edit`, the template plugin can implement `format_permission_request()` to show a preview of the rendered content before writing.

### Proposed Location

```
shared/plugins/template/
├── __init__.py          # PLUGIN_KIND = "tool", create_plugin()
├── plugin.py            # TemplatePlugin implementation
├── renderer.py          # Template engine abstraction
└── README.md
```

### Prerequisite: Extract Shared Diff Utilities

**Problem**: The `file_edit` plugin contains `diff_utils.py` with functions for generating
unified diffs (`generate_unified_diff`, `generate_new_file_diff`, `summarize_diff`). The
template plugin needs these same utilities for permission preview. Options:

| Option | Trade-off |
|--------|-----------|
| Import from `file_edit` | Creates coupling; template depends on file_edit |
| Duplicate code | Maintenance burden; inconsistency risk |
| Extract to shared location | Clean; one-time refactor |

**Recommendation**: Extract `diff_utils.py` to `shared/utils/` before implementing template plugin.

```
shared/
├── plugins/
│   ├── file_edit/
│   │   └── plugin.py        # imports from shared/utils/diff_utils
│   └── template/
│       └── plugin.py        # imports from shared/utils/diff_utils
└── utils/
    └── diff_utils.py        # Extracted: generate_unified_diff, etc.
```

**Refactor steps**:
1. Create `shared/utils/diff_utils.py` (move existing code)
2. Update `file_edit` imports: `from shared.utils.diff_utils import ...`
3. Template plugin uses same import path

This is a small, low-risk refactor that enables clean code sharing without plugin coupling.

## Tool Interface

### Recommended Design: Single Unified Tool

```python
{% raw %}
ToolSchema(
    name="writeFileFromTemplate",
    description="Render a template with variable substitution and write the result to a file. "
                "Templates support variable substitution ({{name}}), conditionals "
                "({% if condition %}...{% endif %}), and loops ({% for item in items %}...{% endfor %}).",
{% endraw %}
    parameters={
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "description": "Inline template content. Mutually exclusive with template_path."
            },
            "template_path": {
                "type": "string",
                "description": "Path to template file. Mutually exclusive with template."
            },
            "variables": {
                "type": "object",
                "description": "Key-value pairs for template substitution.",
                "additionalProperties": True
            },
            "output_path": {
                "type": "string",
                "description": "Path where rendered content should be written."
            }
        },
        "required": ["variables", "output_path"]
    }
)
```

### Design Rationale

**Why a single tool with mutually exclusive parameters?**

- Agents can choose inline templates for one-off generation or file-based for reusable templates
- Cleaner than separate `renderInlineTemplate` / `renderTemplateFile` tools
- Follows precedent: CLI plugins accept various input modes through a single tool
- The executor validates that exactly one of `template` or `template_path` is provided

**Why require `output_path`?**

- The primary use case is generating files (not just rendering to memory)
- Integrates naturally with the permission system (shows what will be written)
- If agents need just the rendered string (rare), they can use a temp file or we add a `dry_run` option later

### Secondary Tool: Template Discovery (Optional)

```python
ToolSchema(
    name="listTemplates",
    description="List available templates from the templates directory (.templates/).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter templates (e.g., '*.java.tmpl')."
            }
        }
    }
)
```

**Recommendation**: Start without this tool. Agents can use `readFile` or file search tools to discover templates. Add `listTemplates` later if a clear need emerges.

## Template Syntax

### Recommendation: Jinja2-style

{% raw %}
```
Hello {{ name }}!

{% if admin %}
You have admin privileges.
{% endif %}

{% for service in services %}
- {{ service.name }}: {{ service.port }}
{% endfor %}
```
{% endraw %}

### Syntax Options Analyzed

| Syntax | Complexity | Power | Familiarity |
|--------|------------|-------|-------------|
| {% raw %}`{{var}}`{% endraw %} only | Very Low | Low | High |
| Python `string.Template` (`$var`) | Low | Low | Medium |
| Mustache/Handlebars | Medium | Medium | Medium |
| Jinja2 | Medium | High | High |

### Rationale for Jinja2

1. **Code Generation Needs Loops**: Generating service classes, API endpoints, or configuration files almost always requires iteration. {% raw %}`{% for field in fields %}`{% endraw %} is essential.

2. **Conditionals are Common**: Different code for different environments, optional features, platform-specific sections. {% raw %}`{% if %}`{% endraw %} blocks are frequently needed.

3. **Industry Standard**: Jinja2 is widely used (Ansible, Flask, dbt, Airflow). LLM agents already understand its syntax from training data.

4. **Python Native**: We're in a Python codebase. Jinja2 is the natural choice. It's a single pip dependency (`Jinja2`).

5. **Graceful Degradation**: Simple templates ({% raw %}`Hello {{name}}!`{% endraw %}) work without needing any Jinja2 features. Complexity is opt-in.

### Subset for Safety

Consider a restricted Jinja2 configuration:
- **Enabled**: Variables, if/elif/else, for loops, filters (e.g., {% raw %}`{{ name | upper }}`{% endraw %})
- **Disabled**: {% raw %}`{% include %}`, `{% import %}`, `{% macro %}`{% endraw %} (potential security concerns with arbitrary file access)
- **Consider Later**: Template inheritance ({% raw %}`{% extends %}`{% endraw %}) if patterns emerge

### Example Templates

**Simple service class:**
{% raw %}
```java
// {{ output_path }}
package {{ package }};

public class {{ class_name }} {
{% for field in fields %}
    private {{ field.type }} {{ field.name }};
{% endfor %}

{% for field in fields %}
    public {{ field.type }} get{{ field.name | capitalize }}() {
        return this.{{ field.name }};
    }
{% endfor %}
}
```
{% endraw %}

**Configuration file:**
{% raw %}
```yaml
# Generated configuration for {{ environment }}
server:
  port: {{ port | default(8080) }}
{% if enable_ssl %}
  ssl:
    enabled: true
    cert_path: {{ ssl_cert_path }}
{% endif %}

logging:
  level: {{ log_level | default('INFO') }}
```
{% endraw %}

## Template Storage and Discovery

### Recommendation: Convention with Flexibility

1. **Convention**: Templates in `.templates/` directory (project root)
2. **Flexibility**: `template_path` accepts any valid path

### Why `.templates/`?

- Consistent location for agents to look for reusable templates
- Follows pattern of `.github/`, `.vscode/`, `.jaato/`
- Easy to gitignore if templates contain sensitive placeholders
- Agent system instructions can reference it: "Check .templates/ for reusable templates"

### Template Naming Convention

Suggest but don't enforce:
- `<name>.<extension>.tmpl` - e.g., `service.java.tmpl`, `config.yaml.tmpl`
- This makes the output type clear

### No `listTemplates` Initially

Agents already have:
- `readFile` to inspect template contents
- Glob/grep tools to find templates by pattern

Add `listTemplates` only if agents frequently struggle with discovery.

## Permission Integration

### File Writing Requires Permission

Template rendering writes files, so it should integrate with the permission system like `file_edit`.

```python
def format_permission_request(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    channel_type: str
) -> Optional[PermissionDisplayInfo]:
    """Show rendered template preview for approval."""

    if tool_name == "writeFileFromTemplate":
        output_path = arguments.get("output_path", "")

        # Render the template (same logic as executor)
        rendered_content = self._render(arguments)

        if isinstance(rendered_content, dict) and "error" in rendered_content:
            # Render error - skip permission, let executor report
            return None

        # Check if output file exists (update vs create)
        if Path(output_path).exists():
            old_content = Path(output_path).read_text()
            diff_text, truncated, total = generate_unified_diff(
                old_content, rendered_content, output_path
            )
            summary = f"Update file from template: {output_path}"
        else:
            diff_text, truncated, total = generate_new_file_diff(
                rendered_content, output_path
            )
            summary = f"Create file from template: {output_path}"

        return PermissionDisplayInfo(
            summary=summary,
            details=diff_text,
            format_hint="diff",
            truncated=truncated
        )

    return None
```

### Auto-Approved: None

All template operations write files, so none should be auto-approved. This matches the file_edit pattern where `writeNewFile` and `updateFile` require approval.

## Return Value

```python
# Success
{
    "success": True,
    "output_path": "/path/to/output.java",
    "size": 1234,
    "lines": 42,
    "variables_used": ["class_name", "package", "fields"]
}

# Error - missing template
{
    "error": "Must provide either 'template' or 'template_path'",
}

# Error - template not found
{
    "error": "Template file not found: /path/to/missing.tmpl"
}

# Error - render failure
{
    "error": "Template render error: undefined variable 'clazz_name' (did you mean 'class_name'?)"
}

# Error - output write failure
{
    "error": "Failed to write output: Permission denied: /etc/protected.txt"
}
```

### Error Handling Notes

- Use helpful error messages (suggest typos, show line numbers for syntax errors)
- Jinja2's `UndefinedError` can be caught and enhanced with suggestions
- Consider `strict_undefined=True` to catch typos rather than silently rendering empty

## System Instructions

{% raw %}
```python
def get_system_instructions(self) -> Optional[str]:
    return """You have access to template rendering tools.

Use `writeFileFromTemplate` to generate files from templates with variable substitution.

Templates use Jinja2 syntax:
- Variables: {{ variable_name }}
- Conditionals: {% if condition %}...{% endif %}
- Loops: {% for item in items %}...{% endfor %}
- Filters: {{ name | upper }}, {{ value | default('fallback') }}

Example:
  writeFileFromTemplate(
    template="Hello {{ name }}, welcome to {{ project }}!",
    variables={"name": "Alice", "project": "jaato"},
    output_path="greeting.txt"
  )

For reusable templates, check .templates/ directory:
  writeFileFromTemplate(
    template_path=".templates/service.java.tmpl",
    variables={"class_name": "OrderService", "package": "com.example.orders"},
    output_path="src/main/java/com/example/orders/OrderService.java"
  )

Template rendering requires approval since it writes files."""
```
{% endraw %}

## Concerns and Trade-offs

### 1. Template Security

**Concern**: Jinja2 is powerful. Sandbox mode prevents arbitrary Python execution, but we should still restrict file access.

**Mitigation**:
- Disable {% raw %}`{% include %}`{% endraw %} and {% raw %}`{% import %}`{% endraw %} to prevent arbitrary file reads
- Use `SandboxedEnvironment` from `jinja2.sandbox`
- Review: Do we need {% raw %}`{% extends %}`{% endraw %}? (Template inheritance adds complexity)

### 2. Dependency Addition

**Concern**: Adding Jinja2 as a dependency.

**Mitigation**:
- Jinja2 is lightweight, widely used, and well-maintained
- Only loaded when template plugin is exposed
- Alternative: Use Python's `string.Template` for simple cases only, but this limits utility

### 3. Two Ways to Write Files

**Concern**: Now agents have `writeNewFile` and `writeFileFromTemplate` for creating files.

**Clarification**:
- `writeNewFile`: Direct content, agent has already composed the final text
- `writeFileFromTemplate`: Templated content, separating structure from data
- System instructions should clarify when to use each
- They complement rather than compete

### 4. Template Discovery

**Concern**: How do agents know what templates exist?

**Mitigation**:
- Start with convention (`.templates/` directory)
- Agents use existing file tools for discovery
- Add `listTemplates` tool if patterns show agents struggling
- Consider: Should subagents inherit a template context?

### 5. Large Templates

**Concern**: Large inline templates in function calls.

**Mitigation**:
- File-based templates (`template_path`) for large/reusable templates
- Inline templates for simple one-off generation
- Potential future: Template caching/registry

## Summary

| Aspect | Decision |
|--------|----------|
| Location | New `shared/plugins/template/` plugin |
| Tool Name | `writeFileFromTemplate` |
| Template Sources | Inline (`template`) or file (`template_path`) |
| Syntax | Jinja2 (restricted: no include/import) |
| Template Storage | `.templates/` convention, any path allowed |
| Permissions | Requires approval (shows rendered diff) |
| Discovery Tool | Not initially; use file tools |
| Shared Utilities | Extract `diff_utils.py` to `shared/utils/` |

## Next Steps

1. **Extract shared utilities**: Move `diff_utils.py` from `file_edit` to `shared/utils/`
2. Update `file_edit` imports to use new location
3. Create `shared/plugins/template/` directory structure
4. Implement `TemplatePlugin` following existing patterns
5. Add Jinja2 dependency to `pyproject.toml` dependencies
6. **Register in pyproject.toml**: Add entry point under `[project.entry-points."jaato.plugins"]`:
   ```toml
   template = "shared.plugins.template:create_plugin"
   ```
7. Write tests for variable substitution, conditionals, loops
8. Add integration test with permission system
9. **Register in README.md**: Add to "Available Plugins" section under "File & Content Management"
10. Document in CLAUDE.md
