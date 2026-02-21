"""Template rendering plugin implementation.

Provides tools for rendering templates with variable substitution
and writing the results to files.

Key features:
1. Standalone template discovery: Detects .tpl/.tmpl files in directories
   referenced by the references plugin and indexes them without copying.
   Runs during system instruction enrichment.
2. Tool result enrichment: Detects embedded templates in tool outputs
   (e.g., from readFile, cat) and extracts them to .jaato/templates/.
3. Template rendering: Renders templates with variable substitution.
   Supports BOTH Jinja2 and Mustache/Handlebars syntax (auto-detected).

Note: System instruction code blocks are NOT scanned for templates.
Instructions contain documentation and examples that may use template
syntax illustratively; extracting those produces false positives. Only
actual file content (via tool results) triggers embedded extraction.

Template Index:
All templates (embedded and standalone) are registered in a unified index
that maps template names to their source paths. The model refers to templates
by name only; the system resolves actual paths via the index. The index is
persisted to .jaato/templates/index.json for inspectability.

Template Syntax Support:
- Jinja2: {{ variable }}, {% if %}, {% for %}, {{ var | filter }}
- Mustache: {{variable}}, {{#section}}...{{/section}}, {{^inverted}}, {{.}}

The template engine is auto-detected based on syntax patterns.

See docs/template-tool-design.md for the design specification.
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jaato_sdk.plugins.base import UserCommand, SystemInstructionEnrichmentResult, ToolResultEnrichmentResult, PermissionDisplayInfo
from jaato_sdk.plugins.model_provider.types import EditableContent, ToolSchema, TRAIT_FILE_WRITER
from shared.trace import trace as _trace_write


# File extensions recognized as standalone template files
TEMPLATE_FILE_EXTENSIONS = {'.tpl', '.tmpl'}


@dataclass
class TemplateIndexEntry:
    """Entry in the unified template index.

    Maps a template name to its actual location on disk, along with
    metadata about syntax and required variables. Covers both embedded
    templates (extracted to .jaato/templates/) and standalone templates
    (left in their original location, not copied).

    Attributes:
        name: Template name used for lookup (e.g., "Entity.java.tpl").
        source_path: Absolute path to the actual template file on disk.
        syntax: Detected template syntax ("jinja2" or "mustache").
        variables: Sorted list of variable names required by the template.
        origin: How the template was discovered ("embedded" or "standalone").
    """
    name: str
    source_path: str  # String for JSON serialization; resolved to Path at lookup
    syntax: str
    variables: List[str] = field(default_factory=list)
    origin: str = "embedded"  # "embedded" or "standalone"


# Regex patterns for detecting Jinja2 template syntax in code blocks
# Matches {{ variable }}, {% control %}, or {# comment #}
JINJA2_VARIABLE_PATTERN = re.compile(r'\{\{\s*\w+')
JINJA2_CONTROL_PATTERN = re.compile(r'\{%\s*\w+')
JINJA2_COMMENT_PATTERN = re.compile(r'\{#.*#\}')

# Mustache/Handlebars specific patterns (distinguish from Jinja2)
# These patterns identify Mustache syntax that wouldn't appear in Jinja2
# Allow whitespace/newlines between {{ and control character (common in formatted templates)
# Note: For DETECTION we don't require closing }} - just detect the start of constructs
MUSTACHE_SECTION_PATTERN = re.compile(r'\{\{\s*#\s*\w+')  # {{#section or {{#if condition
MUSTACHE_END_SECTION_PATTERN = re.compile(r'\{\{\s*/\s*\w+')  # {{/section or {{ /section
MUSTACHE_INVERTED_PATTERN = re.compile(r'\{\{\s*\^\s*\w+')  # {{^inverted or {{ ^inverted
MUSTACHE_CURRENT_ITEM_PATTERN = re.compile(r'\{\{\s*\.\s*\}\}')  # {{.}} or {{ . }}

# Jinja2 specific patterns (distinguish from Mustache)
JINJA2_FILTER_PATTERN = re.compile(r'\{\{.*\|.*\}\}')  # {{ var | filter }}

# Regex to find fenced code blocks in markdown
# Captures: language (group 1), content (group 2)
CODE_BLOCK_PATTERN = re.compile(
    r'```(\w*)\n(.*?)```',
    re.DOTALL
)

# Regex to extract template ID from surrounding context
# Looks for "## Template N:" or "### Template:" style headings
TEMPLATE_HEADING_PATTERN = re.compile(
    r'##\s*Template\s*(?:\d+)?:?\s*(.+)',
    re.IGNORECASE
)

# Frontmatter ID pattern (e.g., "id: mod-code-001")
FRONTMATTER_ID_PATTERN = re.compile(r'^id:\s*(.+)$', re.MULTILINE)


class TemplatePlugin:
    """Plugin for template-based file generation.

    Maintains a unified template index that maps template names to their actual
    locations on disk. Templates come from two sources:
    - Standalone: .tpl/.tmpl files found in referenced directories (not copied)
    - Embedded: Code blocks with template syntax extracted to .jaato/templates/

    The model refers to templates by name only (e.g., "Entity.java.tpl"). The
    system resolves actual paths via the index. The index is persisted to
    .jaato/templates/index.json for inspectability.

    Tools provided:
    - renderTemplate: Render a template with variables and write to file
    - renderTemplateToFile: Same as renderTemplate with overwrite option
    - listAvailableTemplates: List all templates in the unified index
    - listTemplateVariables: List all variables required by a template

    Enrichment:
    - System instruction enrichment: Extracts embedded templates from code blocks
      and discovers standalone templates from referenced directories
    - Tool result enrichment: Extracts embedded templates from tool outputs

    Supported template syntaxes (auto-detected):

    Jinja2:
    - Variables: {{ variable_name }}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in items %}...{% endfor %}
    - Filters: {{ name | upper }}

    Mustache/Handlebars:
    - Variables: {{variable_name}}
    - Sections/loops: {{#items}}...{{/items}}
    - Conditionals: {{#hasValue}}...{{/hasValue}}
    - Inverted sections: {{^isEmpty}}...{{/isEmpty}}
    - Current item: {{.}}
    """

    def __init__(self):
        self._initialized = False
        self._agent_name: Optional[str] = None
        self._base_path: Optional[Path] = None
        self._templates_dir: Optional[Path] = None
        # Track extracted templates in this session: hash -> path
        self._extracted_templates: Dict[str, Path] = {}
        # Unified template index: name -> TemplateIndexEntry
        # Covers both embedded (extracted to .jaato/templates/) and standalone
        # templates (left in original location). The model refers to templates
        # by name; the system resolves actual paths via this index.
        self._template_index: Dict[str, TemplateIndexEntry] = {}
        # Plugin registry for cross-plugin communication (e.g., querying
        # the references plugin for selected directory sources).
        self._plugin_registry = None

    @property
    def name(self) -> str:
        return "template"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        prefix = f"TEMPLATE@{self._agent_name}" if self._agent_name else "TEMPLATE"
        _trace_write(prefix, msg)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the template plugin."""
        config = config or {}
        self._agent_name = config.get("agent_name")

        # Allow custom base path
        if "base_path" in config:
            self._base_path = Path(config["base_path"])

        # Templates directory under .jaato
        if self._base_path is not None:
            self._templates_dir = self._base_path / ".jaato" / "templates"

        self._initialized = True
        self._trace(f"initialized: base_path={self._base_path}, templates_dir={self._templates_dir}")

    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry for cross-plugin communication.

        Called automatically during expose_tool() by the PluginRegistry.
        Used to query the references plugin for selected directory sources
        during standalone template discovery.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        self._trace("set_plugin_registry: wired with registry")

    def set_workspace_path(self, path: str) -> None:
        """Update the base path to the client's workspace directory.

        Called by PluginRegistry.set_workspace_path() when a session binds
        to a specific workspace.  Re-resolves _base_path and _templates_dir
        so template resolution uses the workspace, not the server CWD.
        """
        self._base_path = Path(path)
        self._templates_dir = self._base_path / ".jaato" / "templates"
        self._trace(f"set_workspace_path: base_path={self._base_path}, templates_dir={self._templates_dir}")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False
        self._extracted_templates.clear()
        self._template_index.clear()

    def get_prerequisite_policies(self):
        """Declare template-first file creation policy for reliability enforcement.

        Returns a PrerequisitePolicy that requires ``listAvailableTemplates``
        to have been called before any file-writing tool. The reliability
        plugin's PatternDetector generically enforces this policy — the
        template plugin owns the policy declaration and nudge messages,
        while the reliability plugin owns the enforcement mechanism.

        Returns:
            List containing the template check prerequisite policy.
        """
        from shared.plugins.reliability.types import (
            NudgeType,
            PatternSeverity,
            PrerequisitePolicy,
        )

        return [
            PrerequisitePolicy(
                policy_id="template_check",
                prerequisite_tool="listAvailableTemplates",
                gated_tools={
                    "writeNewFile", "updateFile", "multiFileEdit", "findAndReplace",
                },
                lookback_turns=2,
                nudge_templates={
                    PatternSeverity.MINOR: (
                        NudgeType.DIRECT_INSTRUCTION,
                        "NOTICE: You called {tool_name} without checking templates first. "
                        "Call listAvailableTemplates before writing files to check if a template "
                        "can produce or contribute to the target file (directly via renderTemplateToFile "
                        "or indirectly as a patch source)."
                    ),
                    PatternSeverity.MODERATE: (
                        NudgeType.DIRECT_INSTRUCTION,
                        "NOTICE: Repeated file writes without template check (#{count}). "
                        "You MUST call listAvailableTemplates before using {tool_name}. "
                        "Templates may exist that produce this file directly or provide "
                        "the code pattern you need to patch in. Check templates NOW."
                    ),
                    PatternSeverity.SEVERE: (
                        NudgeType.INTERRUPT,
                        "BLOCKED: {count} file-writing tool calls without checking templates. "
                        "This violates the Template-First File Creation policy. "
                        "Call listAvailableTemplates immediately before any further file operations."
                    ),
                },
                expected_action_template=(
                    "Call {prerequisite_tool} before using {tool_name} "
                    "to check if a template can produce or contribute to the target file"
                ),
            )
        ]

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for template tools."""
        return [
            ToolSchema(
                name="renderTemplate",
                description=(
                    "**PREFERRED OVER MANUAL CODING**: Render a template with variable substitution "
                    "and write the result to a file. When a template exists for your task (check "
                    ".jaato/templates/ or use listAvailableTemplates), you MUST use this tool instead "
                    "of writing code manually. Templates ensure consistency and reduce errors. "
                    "Supports BOTH Jinja2 and Mustache/Handlebars syntax (auto-detected). "
                    "Jinja2: {{name}}, {% if %}, {% for %}. "
                    "Mustache: {{name}}, {{#items}}...{{/items}}, {{^empty}}...{{/empty}}, {{.}}. "
                    "Provide either 'template' for inline content or 'template_name' for a registered template."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "template": {
                            "type": "string",
                            "description": "Inline template content. Mutually exclusive with template_name."
                        },
                        "template_name": {
                            "type": "string",
                            "description": "Template name from the annotation (e.g., 'Entity.java.tpl'). Resolved via the template index. Mutually exclusive with template."
                        },
                        "variables": {
                            "type": "object",
                            "description": "Key-value pairs for template variable substitution.",
                            "additionalProperties": True
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path where rendered content should be written."
                        }
                    },
                    "required": ["variables", "output_path"]
                },
                category="code",
                discoverability="discoverable",
                traits=frozenset({TRAIT_FILE_WRITER}),
            ),
            ToolSchema(
                name="listAvailableTemplates",
                description=(
                    "**CHECK THIS BEFORE WRITING CODE**: List all templates available in this "
                    "session. If a template exists for your task, you MUST use renderTemplate "
                    "instead of writing code manually. Shows both standalone templates (from "
                    "referenced directories) and embedded templates (extracted from documentation)."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="renderTemplateToFile",
                description=(
                    "Render a template and write directly to a file. "
                    "Supports BOTH Jinja2 and Mustache/Handlebars syntax (auto-detected). "
                    "Jinja2: {{name}}, {% if %}, {% for %}, {{ name | filter }}. "
                    "Mustache: {{name}}, {{#items}}...{{/items}}, {{^empty}}...{{/empty}}, {{.}}. "
                    "Provide either 'template' for inline content or 'template_name' for a registered template."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "Path where rendered content will be written."
                        },
                        "template_name": {
                            "type": "string",
                            "description": "Template name from the annotation (e.g., 'Entity.java.tpl'). Resolved via the template index. Mutually exclusive with 'template'."
                        },
                        "template": {
                            "type": "string",
                            "description": "Inline template string. Mutually exclusive with 'template_name'."
                        },
                        "variables": {
                            "type": "object",
                            "description": "Key-value pairs for {{variable}} substitution.",
                            "additionalProperties": True
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Allow overwriting existing file. Default is false."
                        }
                    },
                    "required": ["output_path", "variables"]
                },
                category="code",
                discoverability="discoverable",
                editable=EditableContent(
                    parameters=["template", "variables"],
                    format="yaml",
                    template="# Edit the template content and/or variables below. Save and exit to continue.\n",
                ),
                traits=frozenset({TRAIT_FILE_WRITER}),
            ),
            ToolSchema(
                name="validateTemplateIndex",
                description=(
                    "Validate a template index JSON file against the expected schema. "
                    "Checks for required top-level fields, per-entry required fields, "
                    "valid syntax and origin values, variable format, and optionally "
                    "whether source paths exist on disk."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to a template index JSON file to validate."
                        }
                    },
                    "required": ["path"]
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="listTemplateVariables",
                description=(
                    "List all variables required by a template. Call this before renderTemplateToFile "
                    "to know exactly what variables to provide. Analyzes the template and returns "
                    "all variable names that need to be substituted."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "template_name": {
                            "type": "string",
                            "description": "Template name from the annotation (e.g., 'Entity.java.tpl')"
                        }
                    },
                    "required": ["template_name"]
                },
                category="code",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor functions for each tool."""
        return {
            "renderTemplate": self._execute_render_template,
            "listAvailableTemplates": self._execute_list_available,
            "renderTemplateToFile": self._execute_render_template_to_file,
            "listTemplateVariables": self._execute_list_template_variables,
            "validateTemplateIndex": self._execute_validate_template_index,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for template tools."""
        return """## Template Rendering (MANDATORY USAGE)

**CRITICAL**: When templates exist for a task, you MUST use template tools instead of
manually writing code. Templates ensure consistency, reduce errors, and follow established
patterns. Manual coding when a template exists is NOT acceptable.

### IMPORTANT: Variable Names Are Provided Automatically

When a template is detected, the system automatically injects an annotation
showing the **exact variable names** required. Look for annotations like:

```
[!] **TEMPLATE AVAILABLE - MANDATORY USAGE**: Entity.java.tpl
  Syntax: mustache
  Required variables: [Entity, basePackage, entityFields]
  ...
```

**USE THESE EXACT VARIABLE NAMES** when calling renderTemplateToFile. Do NOT guess or
invent variable names - use the ones shown in the annotation.

### TEMPLATE TOOLS:

**renderTemplateToFile(output_path, template_name, variables)** - PREFERRED tool for file generation
  - template_name: Use the template **name** from the annotation (e.g., "Entity.java.tpl")
  - The system resolves the name to the actual file location via the template index
  - Use the EXACT variable names from the template annotation
  - Automatically creates parent directories - NO mkdir needed!
  - Supports both Jinja2 and Mustache/Handlebars syntax (auto-detected)
  - Checks if file exists (use overwrite=true to replace)
  - Returns: {"success": true, "path": "...", "bytes_written": 1234, "template_syntax": "jinja2|mustache"}

**renderTemplate(template_name, variables, output_path)** - Alternative (same functionality)
  - Also creates parent directories automatically
  - Returns: {"success": true, "path": "...", "size": 1234, "template_syntax": "jinja2|mustache"}

**listAvailableTemplates()** - List all available templates
  - Shows all templates discovered in this session (embedded + standalone)
  - Each entry shows: name, origin, syntax, variables, source path
  - Auto-approved (no permission required)

**listTemplateVariables(template_name)** - Get required variables for a template (OPTIONAL)
  - Use this if you need to re-check the variables for a template
  - Helpful if the original annotation is no longer visible in context
  - Auto-approved (no permission required)
  - Returns: {"variables": ["var1", "var2", ...], "syntax": "jinja2|mustache", "count": N}

### CRITICAL: Directory Creation Rules

**DO NOT use `mkdir` to create directory structures!** The template tools automatically
create all necessary parent directories when writing files.

**WRONG approach (causes malformed directories):**
```
# NEVER DO THIS - mkdir with template notation creates literal garbage directories
cli_based_tool: mkdir -p src/main/java/{{package}}/domain/{model,service}
renderTemplate: ...
```

**CORRECT approach:**
```
# Just call renderTemplateToFile for each file - directories are created automatically
renderTemplateToFile(
    output_path="customer-service/src/main/java/com/bank/customer/domain/model/Customer.java",
    template_name="Entity.java.tpl",
    variables={"Entity": "Customer", "basePackage": "com.bank.customer"}
)
```

### File Path Rules

1. **output_path must be a CONCRETE path** - all variables must be substituted BEFORE calling the tool
2. **NEVER include `{` or `}` in output_path** - these are for template CONTENT only, not file paths
3. **NEVER use shell brace expansion** like `{model,service,repository}` in paths
4. **Generate ONE file at a time** - call renderTemplateToFile once per output file

**Example - Generating multiple files:**
```
# For each entity, call renderTemplateToFile with concrete paths:
renderTemplateToFile(output_path="src/main/java/com/bank/customer/domain/model/Customer.java", ...)
renderTemplateToFile(output_path="src/main/java/com/bank/customer/domain/model/CustomerId.java", ...)
renderTemplateToFile(output_path="src/main/java/com/bank/customer/domain/service/CustomerDomainService.java", ...)
renderTemplateToFile(output_path="src/main/java/com/bank/customer/domain/repository/CustomerRepository.java", ...)
```

### Template Priority Rule (PREREQUISITE FOR FILE TOOLS)

**MANDATORY PREREQUISITE**: Before calling ANY of these file-writing tools, you MUST
call `listAvailableTemplates` at least once in the current or recent turns:
- `writeNewFile` — creating new files
- `updateFile` — modifying existing files
- `multiFileEdit` — batch file operations
- `findAndReplace` — regex-based replacements across files

**The workflow is always:**
1. Call `listAvailableTemplates` to check what templates are available
2. If a template matches your task **directly** → use `renderTemplateToFile`
3. If a template matches your task **indirectly** (the template provides content
   that should be layered onto an existing file) → render it mentally, then apply
   the relevant sections via `updateFile` or `multiFileEdit` as a patch
4. If NO template matches → proceed freely with file-writing tools

**Direct vs. Indirect Template Usage:**
- **Direct**: Template produces a complete new file → `renderTemplateToFile`
- **Indirect**: Template provides a pattern or code fragment that must be merged
  into an existing file (e.g., adding resilience annotations to a Java class).
  The template is the **source of truth** for the new code — render it to
  understand the pattern, then apply the relevant portions as a patch using
  `updateFile` or `multiFileEdit`.

**Enforcement:** The reliability plugin monitors for file-writing tool calls without
a recent `listAvailableTemplates` check and will inject a nudge. Treat nudges as
mandatory corrections — call `listAvailableTemplates` and re-evaluate before proceeding.

### Non-Compliance Policy

WARNING: Code generated without using available templates will be:
- REJECTED as non-compliant during validation
- Considered INVALID output that does not meet quality standards
- Flagged for rework requiring template usage

If no suitable template exists for your task:
1. Report the missing template as a blocker
2. Do NOT generate code directly as a workaround
3. Request that the template be created first

This policy ensures consistent, maintainable code across all generated projects.

### Templates Discovery

Templates come from two sources, unified under a single index:

1. **Standalone templates**: .tpl/.tmpl files found in referenced directories
   (e.g., knowledge module template folders). These stay in their original
   location — not copied.

2. **Embedded templates**: Code blocks with template syntax found in documentation
   (MODULE.md, etc.). These are extracted to `.jaato/templates/`.

Both types are registered in the index and can be referenced by name only.
The index is persisted to `.jaato/templates/index.json` for inspection.

### Template Syntax (both supported, auto-detected)

**Jinja2 style:**
- Variables: {{ variable_name }}
- Conditionals: {% if condition %}...{% endif %}
- Loops: {% for item in items %}...{% endfor %}
- Filters: {{ name | upper }}, {{ value | default('fallback') }}

**Mustache/Handlebars style:**
- Variables: {{variable_name}}
- Sections/loops: {{#items}}...{{/items}}
- Conditionals: {{#hasValue}}...{{/hasValue}}
- Inverted sections: {{^isEmpty}}...{{/isEmpty}}
- Current item in loop: {{.}}

The template engine is auto-detected based on syntax patterns. Mustache patterns
({{#section}}, {{/section}}, {{^inverted}}, {{.}}) trigger Mustache rendering.
Jinja2 patterns ({% %}, {{ | filter }}) trigger Jinja2 rendering.
Simple {{variable}} works in both and defaults to Jinja2.

Template rendering writes files to the workspace."""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        return ["listAvailableTemplates", "listTemplateVariables", "validateTemplateIndex"]

    def format_permission_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        channel_type: str
    ) -> Optional[PermissionDisplayInfo]:
        """Format permission request for file writing tools.

        Provides custom display formatting for renderTemplateToFile to show
        the user what file will be created and with what content.

        Args:
            tool_name: Name of the tool being executed
            arguments: Arguments passed to the tool
            channel_type: Type of channel requesting approval

        Returns:
            PermissionDisplayInfo with formatted content, or None to use default.
        """
        if tool_name != "renderTemplateToFile":
            return None

        output_path = arguments.get("output_path", "")
        template = arguments.get("template")
        template_name = arguments.get("template_name")
        variables = arguments.get("variables", {})
        overwrite = arguments.get("overwrite", False)

        # Build summary
        action = "Overwrite" if overwrite else "Create"
        source = template_name if template_name else "(inline template)"
        summary = f"{action} file: {output_path} from {source}"

        # Build details showing the template and variables
        details_lines = []

        if template_name:
            details_lines.append(f"Template: {template_name}")
        else:
            details_lines.append("Template: (inline)")

        details_lines.append(f"Output: {output_path}")

        if variables:
            details_lines.append("")
            details_lines.append("Variables:")
            for key, value in variables.items():
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                details_lines.append(f"  {key}: {val_str}")

        # Show template content preview if inline
        if template:
            details_lines.append("")
            details_lines.append("Template content:")
            template_preview = template
            truncated = False
            if len(template) > 500:
                template_preview = template[:500]
                truncated = True
            for line in template_preview.split('\n'):
                details_lines.append(f"  {line}")
            if truncated:
                details_lines.append("  ... (truncated)")

        return PermissionDisplayInfo(
            summary=summary,
            details="\n".join(details_lines),
            format_hint="text"
        )

    def get_user_commands(self) -> List[UserCommand]:
        """Template plugin provides model tools only."""
        return []

    # ==================== System Instruction Enrichment ====================

    def get_system_instruction_enrichment_priority(self) -> int:
        """Return system instruction enrichment priority (lower = earlier).

        Template extraction runs at priority 40 - after references plugin
        has already contributed its content to system instructions.
        """
        return 40

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        """Subscribe to system instruction enrichment for template extraction."""
        return True

    def enrich_system_instructions(
        self,
        instructions: str
    ) -> SystemInstructionEnrichmentResult:
        """Discover standalone templates and annotate system instructions.

        Queries the references plugin for selected LOCAL directory sources and
        scans for .tpl/.tmpl files (left in original location). Discovered
        templates are registered in the unified index and annotated in the
        instructions so the model knows what templates are available.

        Note: Code blocks in system instructions are NOT scanned for embedded
        templates. System instructions contain documentation and examples that
        may use template syntax illustratively — extracting those would produce
        false positives. Embedded template extraction only happens via
        enrich_tool_result(), where the content is actual file data.

        Args:
            instructions: Combined system instructions (includes MODULE.md content
                from references plugin).

        Returns:
            SystemInstructionEnrichmentResult with annotated instructions and
            discovery metadata.
        """
        instructions_preview = instructions[:100].replace('\n', '\\n') + ('...' if len(instructions) > 100 else '')
        self._trace(f"enrich_system_instructions called: {len(instructions)} chars, preview: {instructions_preview}")

        annotations: List[str] = []

        # Discover standalone templates from referenced directories
        standalone_entries = self._discover_from_references()
        for entry in standalone_entries:
            self._template_index[entry.name] = entry

            # Build annotation for each standalone template
            if entry.variables:
                var_list = ", ".join(entry.variables)
                var_dict_example = ", ".join(f'"{v}": <value>' for v in entry.variables[:3])
                if len(entry.variables) > 3:
                    var_dict_example += ", ..."
            else:
                var_list = "(none detected)"
                var_dict_example = ""

            annotations.append(
                f"[!] **TEMPLATE AVAILABLE - MANDATORY USAGE**: {entry.name}\n"
                f"  Syntax: {entry.syntax}\n"
                f"  Required variables: [{var_list}]\n"
                f"  **YOU MUST USE THIS TEMPLATE** instead of writing code manually.\n"
                f"  Call: renderTemplateToFile(\n"
                f"      template_name=\"{entry.name}\",\n"
                f"      variables={{{var_dict_example}}},\n"
                f"      output_path=\"<your-output-file>\"\n"
                f"  )"
            )

        # Persist the unified index to disk
        self._persist_index()

        if not annotations:
            return SystemInstructionEnrichmentResult(instructions=instructions)

        # Append annotations to instructions
        annotation_block = "\n\n---\n[!] **MANDATORY TEMPLATES AVAILABLE - USE THESE INSTEAD OF MANUAL CODING:**\n" + "\n\n".join(annotations) + "\n---"
        enriched_instructions = instructions + annotation_block

        return SystemInstructionEnrichmentResult(
            instructions=enriched_instructions,
            metadata={
                "standalone_count": len(standalone_entries),
                "standalone_templates": [
                    {"name": e.name, "path": e.source_path, "variables": e.variables}
                    for e in standalone_entries
                ]
            }
        )

    def _discover_from_references(self) -> List[TemplateIndexEntry]:
        """Discover standalone templates from all referenced directories.

        Queries the references plugin for selected LOCAL directory sources,
        scans each for .tpl/.tmpl files, and returns new index entries
        (skipping any already in the index).

        Returns:
            List of newly discovered TemplateIndexEntry instances.
        """
        directories = self._get_reference_directories()
        if not directories:
            return []

        all_entries: List[TemplateIndexEntry] = []
        for directory in directories:
            entries = self._discover_standalone_templates(directory)
            all_entries.extend(entries)

        if all_entries:
            self._trace(f"_discover_from_references: discovered {len(all_entries)} standalone templates")

        return all_entries

    # ==================== Tool Result Enrichment ====================

    def get_tool_result_enrichment_priority(self) -> int:
        """Return tool result enrichment priority (lower = earlier)."""
        return 40

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment for template extraction."""
        return True

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str
    ) -> ToolResultEnrichmentResult:
        """Detect embedded templates in tool results and extract them.

        Scans tool output for fenced code blocks containing Jinja2 template
        syntax. When found, extracts them to .jaato/templates/ and annotates
        the result.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.

        Returns:
            ToolResultEnrichmentResult with annotated result and extraction metadata.
        """
        result_preview = result[:100].replace('\n', '\\n') + ('...' if len(result) > 100 else '')
        self._trace(f"enrich_tool_result [{tool_name}]: {len(result)} chars, preview: {result_preview}")

        # Find all code blocks in the result
        code_blocks = self._find_code_blocks(result)

        # If no code blocks found, check if the raw content itself is a template
        # This handles cases like readFile on a .tpl file
        if not code_blocks:
            if self._is_template(result):
                self._trace("  no code blocks, but raw content is a template")
                # Treat the entire result as a single template block
                # Use empty lang, full content, position 0
                code_blocks = [("", result, 0, len(result))]
            else:
                self._trace("  no code blocks found in tool result")
                return ToolResultEnrichmentResult(result=result)

        # Filter to blocks that contain template syntax
        template_blocks = [
            (lang, content, start, end)
            for lang, content, start, end in code_blocks
            if self._is_template(content)
        ]

        if not template_blocks:
            # Debug: show what's in each code block
            for i, (lang, content, start, end) in enumerate(code_blocks):
                preview = content[:100].replace('\n', '\\n') + ('...' if len(content) > 100 else '')
                has_var = bool(JINJA2_VARIABLE_PATTERN.search(content))
                has_section = bool(MUSTACHE_SECTION_PATTERN.search(content))
                self._trace(f"  block {i+1}/{len(code_blocks)} lang={lang!r}: var={has_var} section={has_section} preview={preview}")
            self._trace(f"  found {len(code_blocks)} code blocks but none with template syntax")
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: found {len(template_blocks)} template blocks")

        # Extract each template and collect annotations
        extracted: List[Tuple[str, Path, List[str]]] = []
        annotations: List[str] = []

        for lang, content, start, end in template_blocks:
            content_hash = self._hash_content(content)

            # Check if already processed this content in this session
            if content_hash in self._extracted_templates:
                template_path = self._extracted_templates[content_hash]
                self._trace(f"  reusing already-extracted: {template_path.name}")
            else:
                # Determine template filename and extract
                template_name = self._generate_template_name(result, content, lang, start)
                template_path, is_new = self._extract_template(template_name, content, lang)

                if template_path:
                    self._extracted_templates[content_hash] = template_path
                    if is_new:
                        self._trace(f"  extracted new: {template_path.name}")
                    else:
                        self._trace(f"  found existing on disk: {template_path.name}")

            # Always add annotation for available templates (new or existing)
            if template_path:
                variables = self._extract_variables(content)
                syntax = self._detect_template_syntax(content)
                extracted.append((content_hash, template_path, variables))

                # Build annotation with COMPLETE variable list
                rel_path = template_path.relative_to(self._base_path) if self._base_path and template_path.is_relative_to(self._base_path) else template_path

                # Show ALL variables so the model knows exactly what to provide
                if variables:
                    var_list = ", ".join(variables)
                    var_dict_example = ", ".join(f'"{v}": <value>' for v in variables[:3])
                    if len(variables) > 3:
                        var_dict_example += ", ..."
                else:
                    var_list = "(none detected)"
                    var_dict_example = ""

                annotations.append(
                    f"[!] **TEMPLATE AVAILABLE - MANDATORY USAGE**: {rel_path}\n"
                    f"  Syntax: {syntax}\n"
                    f"  Required variables: [{var_list}]\n"
                    f"  **YOU MUST USE THIS TEMPLATE** instead of writing code manually.\n"
                    f"  Call: renderTemplateToFile(\n"
                    f"      template_name=\"{rel_path}\",\n"
                    f"      variables={{{var_dict_example}}},\n"
                    f"      output_path=\"<your-output-file>\"\n"
                    f"  )"
                )

        # Register extracted templates in the unified index
        for content_hash, template_path, variables in extracted:
            index_name = template_path.name
            if index_name not in self._template_index:
                syntax = self._detect_template_syntax(
                    template_path.read_text(encoding="utf-8") if template_path.exists() else ""
                )
                self._template_index[index_name] = TemplateIndexEntry(
                    name=index_name,
                    source_path=str(template_path),
                    syntax=syntax,
                    variables=variables,
                    origin="embedded",
                )

        # Persist the unified index to disk
        self._persist_index()

        if not annotations:
            return ToolResultEnrichmentResult(result=result)

        annotation_block = "\n\n---\n[!] **MANDATORY TEMPLATES AVAILABLE - USE THESE INSTEAD OF MANUAL CODING:**\n" + "\n\n".join(annotations) + "\n---"
        enriched_result = result + annotation_block

        return ToolResultEnrichmentResult(
            result=enriched_result,
            metadata={
                "extracted_count": len(extracted),
                "templates": [
                    {"hash": h, "path": str(p), "variables": v}
                    for h, p, v in extracted
                ]
            }
        )

    # ==================== Standalone Template Discovery ====================

    def _discover_standalone_templates(self, directory: Path) -> List[TemplateIndexEntry]:
        """Scan a directory for standalone template files (.tpl/.tmpl).

        Discovers template files recursively, reads each to extract metadata
        (syntax, variables), and returns index entries. Does NOT copy files —
        entries point to the original location on disk.

        Name collision handling: if two files have the same filename, the
        immediate parent folder is prepended (e.g., "domain/Entity.java.tpl").
        If still ambiguous, the full relative path from the scanned directory
        is used.

        Args:
            directory: Absolute path to the directory to scan.

        Returns:
            List of TemplateIndexEntry for each discovered template file.
        """
        if not directory.is_dir():
            self._trace(f"_discover_standalone: not a directory: {directory}")
            return []

        # Collect all template files with their relative paths
        template_files: List[Tuple[Path, Path]] = []  # (absolute_path, relative_path)
        try:
            for item in sorted(directory.rglob("*")):
                if item.is_file() and item.suffix in TEMPLATE_FILE_EXTENSIONS:
                    rel = item.relative_to(directory)
                    template_files.append((item, rel))
        except (PermissionError, OSError) as e:
            self._trace(f"_discover_standalone: scan error in {directory}: {e}")
            return []

        if not template_files:
            self._trace(f"_discover_standalone: no template files found in {directory}")
            return []

        self._trace(f"_discover_standalone: found {len(template_files)} template files in {directory}")

        # Detect name collisions among filenames
        name_counts: Dict[str, int] = {}
        for _, rel in template_files:
            name = rel.name
            name_counts[name] = name_counts.get(name, 0) + 1

        # Build index entries
        entries: List[TemplateIndexEntry] = []
        for abs_path, rel_path in template_files:
            filename = rel_path.name

            # Determine the index name, handling collisions
            if name_counts[filename] > 1:
                # Prepend parent folder to disambiguate
                parent_prefixed = str(rel_path.parent / filename) if rel_path.parent != Path('.') else filename
                # If still not unique (unlikely), use full relative path
                index_name = str(rel_path)
            else:
                index_name = filename

            # Skip if already in index (e.g., from a previous directory scan)
            if index_name in self._template_index:
                self._trace(f"  skip already-indexed: {index_name}")
                continue

            # Read content and extract metadata
            try:
                content = abs_path.read_text(encoding="utf-8")
            except (IOError, OSError) as e:
                self._trace(f"  error reading {abs_path}: {e}")
                continue

            syntax = self._detect_template_syntax(content)
            variables = self._extract_variables(content)

            entry = TemplateIndexEntry(
                name=index_name,
                source_path=str(abs_path),
                syntax=syntax,
                variables=variables,
                origin="standalone",
            )
            entries.append(entry)
            self._trace(f"  discovered: {index_name} ({syntax}, {len(variables)} vars)")

        return entries

    def _get_reference_directories(self) -> List[Path]:
        """Query the references plugin for selected LOCAL directory sources.

        Uses the plugin registry to access the references plugin and find
        directories from selected reference sources. This enables standalone
        template discovery without the references plugin needing any changes.

        Returns:
            List of absolute Paths to selected reference directories.
        """
        if not self._plugin_registry:
            return []

        try:
            ref_plugin = self._plugin_registry.get_plugin("references")
        except Exception:
            return []

        if ref_plugin is None:
            return []

        try:
            selected_ids = set(ref_plugin.get_selected_ids())
            sources = ref_plugin.get_sources()
        except Exception as e:
            self._trace(f"_get_reference_directories: error querying references: {e}")
            return []

        directories: List[Path] = []
        for source in sources:
            if source.id not in selected_ids:
                continue
            # Only LOCAL type sources with resolved paths
            if source.type.value != "local":
                continue
            path_str = source.resolved_path or source.path
            if not path_str:
                continue
            path = Path(path_str)
            if path.is_dir():
                directories.append(path)

        if directories:
            self._trace(f"_get_reference_directories: found {len(directories)} dirs")

        return directories

    def _persist_index(self) -> None:
        """Write the template index to .jaato/templates/index.json.

        Persists the in-memory index for inspectability and debugging.
        The runtime uses the in-memory _template_index; this file is
        informational only.
        """
        if not self._template_index:
            return

        try:
            self._templates_dir.mkdir(parents=True, exist_ok=True)
            index_path = self._templates_dir / "index.json"

            index_data = {
                "generated_at": datetime.now().isoformat(),
                "template_count": len(self._template_index),
                "templates": {
                    name: asdict(entry)
                    for name, entry in self._template_index.items()
                }
            }

            index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
            self._trace(f"_persist_index: wrote {len(self._template_index)} entries to {index_path}")
        except (IOError, OSError) as e:
            self._trace(f"_persist_index: error writing index: {e}")

    # ==================== Code Block Detection ====================

    def _find_code_blocks(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Find all fenced code blocks in text.

        Returns:
            List of (language, content, start_pos, end_pos) tuples.
        """
        blocks = []
        for match in CODE_BLOCK_PATTERN.finditer(text):
            lang = match.group(1) or ""
            content = match.group(2)
            blocks.append((lang, content, match.start(), match.end()))
        return blocks

    def _is_template(self, content: str) -> bool:
        """Check if content contains template syntax (Jinja2 or Mustache)."""
        return bool(
            JINJA2_VARIABLE_PATTERN.search(content) or
            JINJA2_CONTROL_PATTERN.search(content) or
            MUSTACHE_SECTION_PATTERN.search(content) or
            MUSTACHE_END_SECTION_PATTERN.search(content) or
            MUSTACHE_INVERTED_PATTERN.search(content) or
            MUSTACHE_CURRENT_ITEM_PATTERN.search(content)
        )

    def _detect_template_syntax(self, template: str) -> str:
        """Detect whether template uses Jinja2 or Mustache syntax.

        Mustache indicators: {{#section}}, {{/section}}, {{^inverted}}, {{.}}
        Jinja2 indicators: {% tag %}, {{ var | filter }}

        Args:
            template: Template content string.

        Returns:
            'mustache' or 'jinja2'
        """
        # Check for Mustache-specific patterns first
        mustache_patterns = [
            MUSTACHE_SECTION_PATTERN,     # {{#section}}
            MUSTACHE_END_SECTION_PATTERN,  # {{/section}}
            MUSTACHE_INVERTED_PATTERN,     # {{^inverted}}
            MUSTACHE_CURRENT_ITEM_PATTERN,  # {{.}}
        ]

        for pattern in mustache_patterns:
            if pattern.search(template):
                return 'mustache'

        # Check for Jinja2-specific patterns
        if JINJA2_CONTROL_PATTERN.search(template):  # {% for/if/etc %}
            return 'jinja2'
        if JINJA2_FILTER_PATTERN.search(template):  # {{ var | filter }}
            return 'jinja2'

        # Default to jinja2 for simple {{variable}} (works in both)
        return 'jinja2'

    def _hash_content(self, content: str) -> str:
        """Generate a short hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _generate_template_name(
        self,
        prompt: str,
        content: str,
        lang: str,
        position: int
    ) -> str:
        """Generate a meaningful template filename.

        Tries to extract context from:
        1. Frontmatter ID (e.g., "id: mod-code-001")
        2. Nearby heading (e.g., "## Template 1: Basic Fallback")
        3. Fallback to hash-based name
        """
        # Try to find frontmatter ID in the prompt
        frontmatter_match = FRONTMATTER_ID_PATTERN.search(prompt)
        base_id = frontmatter_match.group(1).strip() if frontmatter_match else None

        # Try to find a template heading near this code block
        # Look in the 500 chars before the code block
        context_before = prompt[max(0, position - 500):position]
        heading_matches = list(TEMPLATE_HEADING_PATTERN.finditer(context_before))

        if heading_matches:
            # Use the closest heading
            heading_name = heading_matches[-1].group(1).strip()
            # Sanitize for filename
            heading_slug = re.sub(r'[^\w\-]', '-', heading_name.lower())
            heading_slug = re.sub(r'-+', '-', heading_slug).strip('-')[:30]
        else:
            heading_slug = None

        # Build filename
        parts = []
        if base_id:
            parts.append(base_id)
        if heading_slug:
            parts.append(heading_slug)
        if not parts:
            # Fallback to hash
            parts.append(f"template-{self._hash_content(content)[:8]}")

        # Add language extension
        ext = self._get_template_extension(lang)
        filename = "-".join(parts) + ext

        return filename

    def _get_template_extension(self, lang: str) -> str:
        """Get appropriate file extension for a template."""
        lang_lower = lang.lower()
        extensions = {
            "java": ".java.tmpl",
            "python": ".py.tmpl",
            "py": ".py.tmpl",
            "javascript": ".js.tmpl",
            "js": ".js.tmpl",
            "typescript": ".ts.tmpl",
            "ts": ".ts.tmpl",
            "yaml": ".yaml.tmpl",
            "yml": ".yaml.tmpl",
            "json": ".json.tmpl",
            "xml": ".xml.tmpl",
            "html": ".html.tmpl",
            "css": ".css.tmpl",
            "sql": ".sql.tmpl",
            "sh": ".sh.tmpl",
            "bash": ".sh.tmpl",
            "go": ".go.tmpl",
            "rust": ".rs.tmpl",
            "kotlin": ".kt.tmpl",
            "scala": ".scala.tmpl",
            "groovy": ".groovy.tmpl",
        }
        return extensions.get(lang_lower, ".tmpl")

    def _extract_template(self, name: str, content: str, lang: str) -> Tuple[Optional[Path], bool]:
        """Extract template content to .jaato/templates/ directory.

        Args:
            name: Template filename.
            content: Template content.
            lang: Source language (for header comment).

        Returns:
            Tuple of (path, is_new) where:
            - path: Path to extracted template, or None on failure
            - is_new: True if newly created, False if reusing existing file
        """
        try:
            # Ensure templates directory exists
            self._templates_dir.mkdir(parents=True, exist_ok=True)

            template_path = self._templates_dir / name

            # Handle name collisions by appending counter
            counter = 1
            base_name = template_path.stem
            suffix = template_path.suffix
            while template_path.exists():
                # Check if existing file has same content
                if template_path.read_text(encoding="utf-8") == content:
                    return template_path, False  # Reuse existing (not new)
                template_path = self._templates_dir / f"{base_name}-{counter}{suffix}"
                counter += 1

            # Write template
            template_path.write_text(content, encoding="utf-8")
            self._trace(f"wrote template: {template_path}")
            return template_path, True  # Newly created

        except (IOError, OSError) as e:
            self._trace(f"error extracting template {name}: {e}")
            return None, False

    def _extract_variables(self, content: str) -> List[str]:
        """Extract variable names from template content.

        Uses Jinja2's AST parser for accurate extraction from Jinja2 templates,
        or regex for Mustache templates. This ensures the model knows exactly
        which variables are required before rendering.

        Args:
            content: Template content string.

        Returns:
            Sorted list of variable names required by the template.
        """
        syntax = self._detect_template_syntax(content)

        if syntax == "jinja2":
            # Use Jinja2's AST parser for accurate variable extraction
            try:
                from jinja2 import Environment, meta
                env = Environment()
                ast = env.parse(content)
                variables = meta.find_undeclared_variables(ast)
                return sorted(list(variables))
            except Exception:
                # Fall back to regex if Jinja2 parsing fails
                pass

        # Regex fallback for Mustache or if Jinja2 parsing failed
        if syntax == "mustache":
            # Match simple variables {{var}}, excluding section markers and comments
            matches = re.findall(r'\{\{([^#/^!}]+)\}\}', content)
            variables = set()
            for m in matches:
                var = m.strip()
                if var and var not in ('.', 'this'):
                    variables.add(var)
            return sorted(list(variables))

        # Default regex for unknown syntax
        var_pattern = re.compile(r'\{\{\s*(\w+)')
        variables = set()
        for match in var_pattern.finditer(content):
            var_name = match.group(1)
            if var_name not in ('if', 'else', 'elif', 'endif', 'for', 'endfor', 'loop', 'true', 'false', 'none'):
                variables.add(var_name)
        return sorted(variables)

    # ==================== Template Rendering ====================

    def _render_template(self, template: str, variables: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """Render template using detected syntax (Jinja2 or Mustache).

        Automatically detects which template syntax is used and renders with
        the appropriate engine.

        Args:
            template: Template content string.
            variables: Key-value pairs for template variable substitution.

        Returns:
            Tuple of (rendered_content, error_dict).
            If error_dict is not None, rendering failed.
        """
        syntax = self._detect_template_syntax(template)

        if syntax == 'mustache':
            return self._render_mustache(template, variables)
        else:
            return self._render_jinja2(template, variables)

    # Regex patterns for the dotted-path preprocessor.
    # Match section/inverted/closing tags whose name contains at least one dot
    # but is NOT a helper call (helpers have a space between name and argument).
    _MUSTACHE_TAG_RE = re.compile(r'\{\{(.*?)\}\}')

    def _preprocess_mustache_dotted_paths(self, template: str) -> str:
        """Rewrite dotted paths in Mustache section/inverted tags for pybars3.

        pybars3 does not support dotted paths in raw section tags
        (``{{#a.b}}``) or inverted section tags (``{{^a.b}}``), even though
        the Mustache spec requires it.  It *does* support dots in:

        * Variable interpolation: ``{{a.b}}``
        * Built-in helper arguments: ``{{#if a.b}}``, ``{{#each a.b}}``,
          ``{{#with a.b}}``, ``{{#unless a.b}}``

        This preprocessor rewrites the unsupported forms into equivalent
        helper-based constructs that pybars3 can compile:

        * ``{{#a.b.c}}…{{/a.b.c}}``  →  ``{{#if a.b.c}}…{{/if}}``
          Uses ``if`` because it preserves the current context, keeping
          dotted variable references inside the block working correctly.
          (pybars3 does not traverse the context stack, so nested-section
          rewrites like ``{{#a}}{{#b}}{{#c}}`` would break inner
          ``{{a.b.c}}`` variable references.)
        * ``{{^a.b.c}}…{{/a.b.c}}``  →  ``{{#unless a.b.c}}…{{/unless}}``

        Limitation: the ``{{#if}}`` rewrite handles conditional checks (the
        dominant use case for dotted section tags) but does not support
        context switching or list iteration.  Templates that need those
        semantics should use explicit ``{{#with a.b}}`` or ``{{#each a.b}}``
        helpers, which pybars3 supports natively.

        The method is idempotent: templates without dotted section/inverted
        tags pass through unchanged.
        """
        result: list[str] = []
        # Stack tracks ('section', 'a.b.c') or ('inverted', 'a.b.c') or
        # ('other', 'name') for non-dotted / helper openings.
        stack: list[tuple[str, str]] = []
        last_end = 0

        for m in self._MUSTACHE_TAG_RE.finditer(template):
            result.append(template[last_end:m.start()])
            tag_content = m.group(1).strip()

            if tag_content.startswith('#'):
                rest = tag_content[1:].strip()
                # A dotted section has no spaces (helpers like {{#if a.b}} do)
                if '.' in rest and ' ' not in rest:
                    result.append('{{#if ' + rest + '}}')
                    stack.append(('section', rest))
                else:
                    result.append(m.group(0))
                    name = rest.split()[0] if rest else rest
                    stack.append(('other', name))

            elif tag_content.startswith('^'):
                rest = tag_content[1:].strip()
                if '.' in rest and ' ' not in rest:
                    result.append('{{#unless ' + rest + '}}')
                    stack.append(('inverted', rest))
                else:
                    result.append(m.group(0))
                    stack.append(('other', rest))

            elif tag_content.startswith('/'):
                rest = tag_content[1:].strip()
                if '.' in rest and stack and stack[-1][1] == rest:
                    kind, _ = stack.pop()
                    if kind == 'section':
                        result.append('{{/if}}')
                    elif kind == 'inverted':
                        result.append('{{/unless}}')
                    else:
                        # Shouldn't happen, but be safe
                        result.append(m.group(0))
                else:
                    # Non-dotted close, or unmatched dotted close — pass through
                    if stack and stack[-1][1] == rest:
                        stack.pop()
                    result.append(m.group(0))

            else:
                # Variable or other tag — pass through unchanged
                result.append(m.group(0))

            last_end = m.end()

        result.append(template[last_end:])
        return ''.join(result)

    def _render_mustache(self, template: str, variables: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """Render template using Handlebars syntax.

        Supports full Handlebars syntax including dotted paths:
        - Variables: ``{{variable_name}}``, ``{{a.b.c}}``
        - Sections/loops: ``{{#items}}…{{/items}}``, ``{{#a.b}}…{{/a.b}}``
        - Conditionals: ``{{#if condition}}``, ``{{#if a.b}}``
        - Each loops: ``{{#each items}}``, ``{{#each a.b}}``
        - Inverted sections: ``{{^isEmpty}}…{{/isEmpty}}``, ``{{^a.b}}…{{/a.b}}``
        - Current item: ``{{.}}``, ``{{this}}``

        Dotted paths in section/inverted tags are preprocessed into
        equivalent pybars3-compatible constructs before compilation
        (see ``_preprocess_mustache_dotted_paths``).

        Args:
            template: Template content string with Handlebars syntax.
            variables: Key-value pairs for template variable substitution.

        Returns:
            Tuple of (rendered_content, error_dict).
            If error_dict is not None, rendering failed.
        """
        try:
            from pybars import Compiler
        except ImportError:
            return "", {
                "error": "Handlebars template detected but pybars3 not installed. Install with: pip install pybars3",
                "status": "dependency_missing"
            }

        try:
            preprocessed = self._preprocess_mustache_dotted_paths(template)
            compiler = Compiler()
            compiled_template = compiler.compile(preprocessed)
            rendered = compiled_template(variables)
            return rendered, None
        except Exception as e:
            return "", {
                "error": f"Handlebars render error: {str(e)}",
                "status": "render_error"
            }

    def _render_jinja2(self, template: str, variables: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """Render template using Jinja2.

        Args:
            template: Template content string.
            variables: Key-value pairs for template variable substitution.

        Returns:
            Tuple of (rendered_content, error_dict).
            If error_dict is not None, rendering failed.
        """
        # Try to import Jinja2
        try:
            from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            return "", {
                "error": "Jinja2 is not installed. Install with: pip install Jinja2",
                "status": "dependency_missing"
            }

        # Create sandboxed environment (safer execution)
        env = SandboxedEnvironment(undefined=StrictUndefined)

        # Disable dangerous features
        env.globals = {}  # No built-in globals

        try:
            # Compile and render template
            jinja_template = env.from_string(template)
            rendered = jinja_template.render(**variables)
            return rendered, None
        except TemplateSyntaxError as e:
            return "", {
                "error": f"Template syntax error at line {e.lineno}: {e.message}",
                "status": "syntax_error"
            }
        except UndefinedError as e:
            # Try to suggest similar variable names
            return "", {
                "error": f"Undefined variable: {e.message}",
                "available_variables": list(variables.keys()),
                "status": "undefined_variable"
            }
        except Exception as e:
            return "", {
                "error": f"Template render error: {str(e)}",
                "status": "render_error"
            }

    # ==================== Path Resolution ====================

    def _resolve_template_path(self, template_path: str) -> Tuple[Optional[Path], List[str]]:
        """Resolve template path, supporting index lookup and multiple base locations.

        Tries paths in order:
        1. Template index lookup by name (exact match on full name or filename)
        2. Absolute path (if absolute)
        3. Relative to current working directory
        4. Relative to base_path (configured path)
        5. Relative to .jaato/templates/
        6. Resolved path (handles .. components)

        The index lookup (step 1) enables the model to refer to templates by
        name only (e.g., "Entity.java.tpl") regardless of where the file
        actually lives on disk.

        Args:
            template_path: Path to template, or template name for index lookup.

        Returns:
            Tuple of (resolved_path, paths_tried).
            resolved_path is None if file not found.
        """
        path = Path(template_path)
        paths_tried = []

        # 1. Check template index by exact name
        if template_path in self._template_index:
            entry = self._template_index[template_path]
            resolved = Path(entry.source_path)
            paths_tried.append(f"index:{template_path} -> {entry.source_path}")
            if resolved.exists():
                return resolved, paths_tried

        # 2. Check template index by filename (strip any path prefix)
        filename = path.name
        if filename != template_path and filename in self._template_index:
            entry = self._template_index[filename]
            resolved = Path(entry.source_path)
            paths_tried.append(f"index:{filename} -> {entry.source_path}")
            if resolved.exists():
                return resolved, paths_tried

        # 3. If absolute, use as-is
        if path.is_absolute():
            paths_tried.append(str(path))
            if path.exists():
                return path, paths_tried
            return None, paths_tried

        # 4. Try relative to base_path (workspace)
        if self._base_path is not None:
            base_path = self._base_path / path
            paths_tried.append(str(base_path))
            if base_path.exists():
                return base_path, paths_tried

        # 5. Try relative to .jaato/templates/
        if self._templates_dir is not None:
            templates_path = self._templates_dir / path
            paths_tried.append(str(templates_path))
            if templates_path.exists():
                return templates_path, paths_tried

        # 6. Try resolving .. components from base_path
        if self._base_path is not None:
            try:
                resolved = (self._base_path / path).resolve()
                if str(resolved) not in paths_tried:
                    paths_tried.append(str(resolved))
                if resolved.exists():
                    return resolved, paths_tried
            except (OSError, ValueError):
                pass

        return None, paths_tried

    # ==================== Tool Executors ====================

    @staticmethod
    def _coerce_variables(raw: Any) -> Dict[str, Any]:
        """Coerce a ``variables`` argument into a dict.

        LLMs sometimes serialise the JSON object as a string instead of
        passing a proper dict.  This helper transparently handles that
        (and other common mis-shapes) so tool executors never crash on
        ``variables.keys()``.

        Coercion rules:
        - ``dict`` → returned as-is.
        - ``str``  → decoded via ``json.loads``; must produce a dict.
        - ``None`` / missing → empty dict.
        - Anything else        → empty dict (best-effort).
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return {}
        return {} if raw is None else {}

    def _execute_render_template(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute renderTemplate tool.

        Supports both Jinja2 and Mustache template syntax (auto-detected).
        """
        template = args.get("template")
        template_name_arg = args.get("template_name")
        variables = self._coerce_variables(args.get("variables"))
        output_path = args.get("output_path", "")

        # Validation
        if not output_path:
            return {"error": "output_path is required"}

        if not template and not template_name_arg:
            return {"error": "Either 'template' or 'template_name' must be provided"}

        if template and template_name_arg:
            return {"error": "Provide either 'template' or 'template_name', not both"}

        # Get template content
        template_source = "inline"
        if template_name_arg:
            resolved_path, paths_tried = self._resolve_template_path(template_name_arg)
            if resolved_path is None:
                return {
                    "error": f"Template not found: {template_name_arg}",
                    "paths_tried": paths_tried
                }
            try:
                template = resolved_path.read_text(encoding="utf-8")
                template_source = str(resolved_path)
            except IOError as e:
                return {
                    "error": f"Failed to read template: {e}",
                    "resolved_path": str(resolved_path)
                }

        # Detect syntax and render using appropriate engine
        syntax = self._detect_template_syntax(template)
        rendered, error = self._render_template(template, variables)
        if error:
            return error

        # Write output
        try:
            out_path = Path(output_path)
            if not out_path.is_absolute():
                if self._base_path is None:
                    return {
                        "error": "No workspace path configured — cannot resolve relative output path",
                        "status": "no_workspace"
                    }
                out_path = self._base_path / out_path

            # Create parent directories
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path.write_text(rendered, encoding="utf-8")
        except IOError as e:
            return {
                "error": f"Failed to write output: {e}",
                "status": "write_error"
            }

        return {
            "success": True,
            "path": str(out_path),
            "size": len(rendered),
            "lines": rendered.count('\n') + 1,
            "variables_used": list(variables.keys()),
            "template_syntax": syntax,
            "template_source": template_source
        }

    def _execute_list_available(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all templates available in this session.

        Returns templates from the unified index, covering both embedded
        templates (extracted from code blocks to .jaato/templates/) and
        standalone templates (discovered in referenced directories, left
        in their original location).

        Each entry includes the template name (used for renderTemplateToFile),
        its origin, syntax, required variables, and source path.
        """
        if not self._template_index:
            return {
                "templates": [],
                "message": "No templates have been discovered in this session."
            }

        templates = []
        for name, entry in self._template_index.items():
            source_path = Path(entry.source_path)
            exists = source_path.exists()

            # Show relative path for display when inside base_path
            try:
                display_path = str(source_path.relative_to(self._base_path)) if self._base_path and source_path.is_relative_to(self._base_path) else str(source_path)
            except ValueError:
                display_path = str(source_path)

            templates.append({
                "name": name,
                "origin": entry.origin,
                "syntax": entry.syntax,
                "variables": entry.variables,
                "source_path": display_path,
                "exists": exists,
            })

        # Sort: standalone first (they're the primary templates), then embedded
        templates.sort(key=lambda t: (0 if t["origin"] == "standalone" else 1, t["name"]))

        return {
            "templates": templates,
            "count": len(templates)
        }

    def _execute_render_template_to_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute renderTemplateToFile tool.

        Renders a template and writes the result to a file.
        Supports both Jinja2 and Mustache template syntax (auto-detected).
        """
        output_path = args.get("output_path", "")
        template = args.get("template")
        template_name_arg = args.get("template_name")
        variables = self._coerce_variables(args.get("variables"))
        overwrite = args.get("overwrite", False)

        # Validation
        if not output_path:
            return {"error": "output_path is required"}

        if not template and not template_name_arg:
            return {
                "error": "Exactly one of 'template' or 'template_name' must be provided"
            }

        if template and template_name_arg:
            return {
                "error": "Provide either 'template' or 'template_name', not both"
            }

        # Determine template source
        template_source = "inline" if template else "file"

        # Load template from file if template_name provided
        if template_name_arg:
            resolved_path, paths_tried = self._resolve_template_path(template_name_arg)
            if resolved_path is None:
                return {
                    "error": f"Template not found: {template_name_arg}",
                    "paths_tried": paths_tried
                }
            try:
                template = resolved_path.read_text(encoding="utf-8")
                template_source = str(resolved_path)
            except IOError as e:
                return {
                    "error": f"Failed to read template: {e}",
                    "resolved_path": str(resolved_path),
                    "template_name": template_name_arg
                }

        # Check if output path already exists
        out_path = Path(output_path)
        if not out_path.is_absolute():
            if self._base_path is None:
                return {
                    "error": "No workspace path configured — cannot resolve relative output path",
                    "status": "no_workspace"
                }
            out_path = self._base_path / out_path

        if out_path.exists() and not overwrite:
            return {
                "error": f"Output file already exists: {output_path}. Set overwrite=true to replace.",
                "output_path": str(out_path)
            }

        # Detect syntax and render using appropriate engine
        syntax = self._detect_template_syntax(template)
        rendered, error = self._render_template(template, variables)
        if error:
            # Add template_name to error response if applicable
            if template_name_arg:
                error["template_name"] = template_name_arg
            return error

        # Create parent directories if needed
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return {
                "error": f"Failed to create parent directories: {e}",
                "output_path": str(out_path)
            }

        # Write rendered content to file
        try:
            out_path.write_text(rendered, encoding="utf-8")
            bytes_written = len(rendered.encode('utf-8'))
        except IOError as e:
            return {
                "error": f"Failed to write output file: {e}",
                "output_path": str(out_path)
            }
        except PermissionError as e:
            return {
                "error": f"Permission denied: {e}",
                "output_path": str(out_path)
            }

        self._trace(f"renderTemplateToFile: wrote {bytes_written} bytes to {out_path} (syntax: {syntax})")

        return {
            "success": True,
            "path": str(out_path),
            "bytes_written": bytes_written,
            "variables_used": sorted(variables.keys()),
            "template_source": template_source,
            "template_syntax": syntax
        }

    def _validate_template_index(self, data: Any) -> Tuple[bool, List[str], List[str]]:
        """Validate a template index JSON structure.

        Checks the top-level structure and each template entry for required
        fields, valid enum values, and correct types.

        Args:
            data: Parsed JSON data from a template index file.

        Returns:
            Tuple of (is_valid, errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return False, ["File must contain a JSON object"], []

        # Top-level fields
        if "generated_at" not in data:
            errors.append("'generated_at' is required")
        elif not isinstance(data["generated_at"], str):
            errors.append("'generated_at' must be a string")

        if "template_count" not in data:
            errors.append("'template_count' is required")
        elif not isinstance(data["template_count"], int):
            errors.append("'template_count' must be an integer")

        if "templates" not in data:
            errors.append("'templates' is required")
            return len(errors) == 0, errors, warnings

        templates = data["templates"]
        if not isinstance(templates, dict):
            errors.append("'templates' must be an object")
            return len(errors) == 0, errors, warnings

        # Warn if template_count doesn't match actual count
        if isinstance(data.get("template_count"), int):
            if data["template_count"] != len(templates):
                warnings.append(
                    f"template_count ({data['template_count']}) does not match "
                    f"actual number of templates ({len(templates)})"
                )

        valid_syntaxes = ("jinja2", "mustache")
        valid_origins = ("standalone", "embedded")

        for name, entry in templates.items():
            prefix = f"templates['{name}']"

            if not isinstance(entry, dict):
                errors.append(f"{prefix}: must be an object")
                continue

            # Required fields
            if not entry.get("name"):
                errors.append(f"{prefix}: 'name' is required")
            if not entry.get("source_path"):
                errors.append(f"{prefix}: 'source_path' is required")

            # Validate syntax
            syntax = entry.get("syntax")
            if not syntax:
                errors.append(f"{prefix}: 'syntax' is required")
            elif syntax not in valid_syntaxes:
                errors.append(f"{prefix}: invalid syntax '{syntax}'. Must be one of: {', '.join(valid_syntaxes)}")

            # Validate origin
            origin = entry.get("origin")
            if not origin:
                errors.append(f"{prefix}: 'origin' is required")
            elif origin not in valid_origins:
                errors.append(f"{prefix}: invalid origin '{origin}'. Must be one of: {', '.join(valid_origins)}")

            # Validate variables
            variables = entry.get("variables")
            if variables is not None:
                if not isinstance(variables, list):
                    errors.append(f"{prefix}: 'variables' must be an array")
                elif not all(isinstance(v, str) for v in variables):
                    errors.append(f"{prefix}: 'variables' must contain only strings")

            # Warn if source_path doesn't exist for standalone entries
            source_path = entry.get("source_path", "")
            if origin == "standalone" and source_path and os.path.isabs(source_path):
                if not os.path.exists(source_path):
                    warnings.append(f"{prefix}: source_path does not exist: {source_path}")

        return len(errors) == 0, errors, warnings

    def _execute_validate_template_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a template index JSON file against the expected schema.

        Reads the file, parses it as JSON, and runs _validate_template_index()
        to check structure, entry fields, syntax/origin values, and variable format.

        Args:
            args: Tool arguments with 'path' (string, required).

        Returns:
            Dict with 'valid', 'path', 'errors', and 'warnings' fields.
        """
        file_path = args.get("path", "")
        if not file_path:
            return {"valid": False, "path": "", "errors": ["'path' is required"], "warnings": []}

        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            if self._base_path is None:
                return {"valid": False, "path": file_path, "errors": ["No workspace path configured — cannot resolve relative path"], "warnings": []}
            path_obj = self._base_path / path_obj

        if not path_obj.exists():
            return {"valid": False, "path": str(path_obj), "errors": [f"File not found: {path_obj}"], "warnings": []}

        try:
            content = path_obj.read_text(encoding='utf-8')
        except (IOError, OSError) as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Cannot read file: {e}"], "warnings": []}

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Invalid JSON: {e}"], "warnings": []}

        is_valid, errors, warnings = self._validate_template_index(data)
        return {
            "valid": is_valid,
            "path": str(path_obj),
            "errors": errors,
            "warnings": warnings,
        }

    def _execute_list_template_variables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all undeclared variables from a template.

        Uses Jinja2's AST parser for Jinja2 templates to find undeclared variables,
        or regex for Mustache templates.

        Args:
            args: Tool arguments containing 'template_name'.

        Returns:
            Dict with 'variables' list and 'syntax' type, or 'error' on failure.
        """
        template_name = args.get("template_name", "")

        if not template_name:
            return {"error": "template_name is required"}

        # Resolve the template name via index or filesystem
        resolved_path, paths_tried = self._resolve_template_path(template_name)
        if not resolved_path or not resolved_path.exists():
            return {
                "error": f"Template not found: {template_name}",
                "paths_tried": paths_tried
            }

        # Read template content
        try:
            template_content = resolved_path.read_text(encoding="utf-8")
        except IOError as e:
            return {
                "error": f"Failed to read template: {e}",
                "resolved_path": str(resolved_path)
            }

        # Detect template syntax
        syntax = self._detect_template_syntax(template_content)

        if syntax == "jinja2":
            # Use Jinja2's AST parser for accurate variable extraction
            try:
                from jinja2 import Environment, meta
            except ImportError:
                return {
                    "error": "Jinja2 is not installed. Install with: pip install Jinja2",
                    "status": "dependency_missing"
                }

            try:
                env = Environment()
                ast = env.parse(template_content)
                variables = meta.find_undeclared_variables(ast)
                return {
                    "variables": sorted(list(variables)),
                    "syntax": "jinja2",
                    "template_name": template_name,
                    "count": len(variables)
                }
            except Exception as e:
                return {
                    "error": f"Failed to parse Jinja2 template: {e}",
                    "syntax": "jinja2",
                    "template_name": template_name
                }

        elif syntax == "mustache":
            # Use regex to find {{variable}} patterns for Mustache
            # Match simple variables {{var}}, but not section markers {{#...}}, {{/...}}, {{^...}}
            # Also exclude comments {{!...}}
            matches = re.findall(r'\{\{([^#/^!}]+)\}\}', template_content)
            variables = set()
            for m in matches:
                var = m.strip()
                # Skip special Mustache markers like {{.}} (current context) and {{this}}
                if var and var not in ('.', 'this'):
                    variables.add(var)

            return {
                "variables": sorted(list(variables)),
                "syntax": "mustache",
                "template_name": template_name,
                "count": len(variables)
            }

        else:
            return {
                "error": f"Unknown template syntax",
                "syntax": syntax,
                "template_name": template_name
            }


def create_plugin() -> TemplatePlugin:
    """Factory function to create the template plugin instance."""
    return TemplatePlugin()
