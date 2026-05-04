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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import (
    PermissionDisplayInfo,
    PromptEnrichmentResult,
    SystemInstructionEnrichmentResult,
    ToolResultEnrichmentResult,
    UserCommand,
)
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
        tags: Topical tags driving prompt/tool-result enrichment matching.
            Produced upstream by the gen-references agent and persisted in
            ``index.json``; runtime-discovered entries have an empty list
            and remain accessible via ``listAvailableTemplates``/``renderTemplateToFile``
            tools but do not surface contextually.
        description: Optional human-readable description (used in enrichment
            hints when present).
    """
    name: str
    source_path: str  # String for JSON serialization; resolved to Path at lookup
    syntax: str
    variables: List[str] = field(default_factory=list)
    origin: str = "embedded"  # "embedded" or "standalone"
    tags: List[str] = field(default_factory=list)
    description: str = ""


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

# Spring Boot property placeholder collision protection.
#
# Spring Boot uses ${ENV_VAR:default} syntax for property placeholders.
# When a Mustache variable appears inside a Spring placeholder, e.g.:
#   ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}
# the sequence $+{+{{VAR}} creates a collision with Handlebars triple-brace
# unescaped syntax ({{{var}}}). pybars3 consumes the opening brace from
# Spring's ${, breaking the output.
#
# The sentinel replaces the Spring opening brace before Mustache rendering
# and is restored afterwards, keeping both syntaxes intact.
_SPRING_BRACE_SENTINEL = "__JAATO_SPRING_BRACE__"
_SPRING_COLLISION_RE = re.compile(r'\$\{\{\{')  # Matches ${{{ - the collision point

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

# Pattern matching @generated annotation lines (Java, YAML, shell comments).
# These lines contain metadata placeholders (e.g., {{skillId}}, {{skillVersion}})
# that are NOT template variables and must be excluded from variable extraction
# and rendering to avoid undefined-variable errors.
# Matches lines like:
#   * @generated {{skillId}} v{{skillVersion}}
#   // @generated {{skillId}} v{{skillVersion}}
#   # @generated {{skillId}} v{{skillVersion}}
_GENERATED_ANNOTATION_RE = re.compile(r'^[/*#\s]*@generated\b.*$', re.MULTILINE)


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
    - renderTemplateToFile: Render a template with variables and write to file
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
        # Optional override for the read-only framework-config root,
        # set by ``PluginRegistry.set_config_root``.  When non-None,
        # ``_templates_dir`` is resolved as ``<config_root>/templates``
        # rather than ``<workspace>/.jaato/templates`` — supporting
        # the sandbox + config_root pattern where the workspace is the
        # ephemeral runtime sandbox but the framework config (incl.
        # the template index) lives at the repo root.  Mirrors the
        # references plugin's ``_config_root`` field.
        self._config_root: Optional[str] = None
        self._templates_dir: Optional[Path] = None
        # Track extracted templates in this session: hash -> path
        self._extracted_templates: Dict[str, Path] = {}
        # Unified template index: name -> TemplateIndexEntry
        # Covers both embedded (extracted to .jaato/templates/) and standalone
        # templates (left in original location). The model refers to templates
        # by name; the system resolves actual paths via this index.
        self._template_index: Dict[str, TemplateIndexEntry] = {}
        # Tag-based indexer over the template catalog.  Drives prompt and
        # tool-result enrichment via shared.tag_coherence (same matcher
        # used by memory and references plugins).  Re-built whenever the
        # catalog mutates; see ``_rebuild_indexer``.
        from .indexer import TemplateIndexer
        self._indexer = TemplateIndexer()
        # Plugin registry for cross-plugin communication (e.g., querying
        # the references plugin for selected directory sources).
        self._plugin_registry = None
        # Template names whose contextual "📦 Available Templates" hint
        # bullet has already been injected into the model's context
        # during this session.  Prevents the same block from being
        # re-surfaced on every tool call.  Cleared by
        # on_history_cleared() when the session history is wiped.
        # Note: this only guards the *contextual surfacing* pass — the
        # embedded-template extraction pass in enrich_tool_result() has
        # its own content-hash dedup via ``_extracted_templates``.
        self._surfaced_template_names: Set[str] = set()

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

        # Templates directory: prefer config_root when set, else fall
        # back to <workspace>/.jaato/templates.
        self._templates_dir = self._compute_templates_dir()

        self._initialized = True
        self._load_persisted_index()
        self._indexer.build_index(list(self._template_index.values()))
        self._trace(f"initialized: base_path={self._base_path}, config_root={self._config_root}, templates_dir={self._templates_dir}")

    def _compute_templates_dir(self) -> Optional[Path]:
        """Resolve where the template index + standalone templates live.

        Priority chain (mirrors the references plugin's resolution):

        1. ``self._config_root`` if set (via ``set_config_root``) — the
           sandbox + config_root pattern where the workspace is the
           ephemeral sandbox but the framework's `.jaato/templates/`
           lives at the repo root.  Resolves to
           ``<config_root>/templates``.
        2. ``self._base_path`` (workspace) — single-dir workspaces.
           Resolves to ``<workspace>/.jaato/templates`` (legacy default).
        3. None when neither is set — returned so callers can guard.
        """
        if self._config_root is not None:
            return Path(self._config_root) / "templates"
        if self._base_path is not None:
            return self._base_path / ".jaato" / "templates"
        return None

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
        Also loads the persisted template index from disk so templates are
        available immediately — without depending on the references plugin.

        When ``set_config_root`` has already been called with a non-None
        value, the templates_dir resolution uses config_root instead of
        the workspace path (see ``_compute_templates_dir``).
        """
        self._base_path = Path(path)
        self._templates_dir = self._compute_templates_dir()
        self._load_persisted_index()
        self._indexer.build_index(list(self._template_index.values()))
        self._trace(f"set_workspace_path: base_path={self._base_path}, config_root={self._config_root}, templates_dir={self._templates_dir}")

    def set_config_root(self, path: Optional[str]) -> None:
        """Adopt the registry-broadcast config_root override.

        Called by :meth:`PluginRegistry.set_config_root` whenever the
        session's ``config_root`` changes.  Re-resolves ``_templates_dir``
        to use the new override (or fall back to the workspace tier
        when ``path`` is ``None``) and reloads the persisted template
        index from the new location.

        This enables the sandbox + config_root pattern (handoff_test +
        kb-enablement-2.0): the workspace is the ephemeral runtime
        sandbox, but the framework's ``.jaato/templates/index.json``
        lives at the repo root.  Without this method, the template
        plugin would be pinned to the sandbox and miss the committed
        template catalog.

        Mirrors the references plugin's ``set_config_root`` (lines
        241-253 of ``shared/plugins/references/plugin.py``).

        Args:
            path: The config_root to adopt, or ``None`` to fall back
                to the workspace tier.
        """
        self._config_root = path
        self._templates_dir = self._compute_templates_dir()
        # Reload the index from the (potentially) new location so
        # listAvailableTemplates / renderTemplateToFile pick up the
        # right catalog without requiring a session restart.
        self._load_persisted_index()
        self._indexer.build_index(list(self._template_index.values()))
        self._trace(f"set_config_root: {path}, templates_dir={self._templates_dir}")

    def _load_persisted_index(self) -> None:
        """Load the template index from .jaato/templates/index.json if it exists.

        Seeds ``_template_index`` so that ``listAvailableTemplates`` and
        ``renderTemplateToFile`` work immediately, even before the
        references plugin discovers template directories.  Entries loaded
        here are overwritten if the references plugin later discovers the
        same template name (runtime discovery takes precedence).
        """
        if not self._templates_dir:
            return
        index_path = self._templates_dir / "index.json"
        if not index_path.exists():
            return
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            # Two on-disk schemas exist:
            # 1. Runtime persist: ``{"templates": {name: entry, ...}}``
            # 2. gen-references:  ``{"entries": [entry, ...], "schema": ...}``
            # Normalise both into an iterable of (name, entry_dict) pairs.
            if "entries" in data:
                pairs = ((e.get("name", ""), e) for e in data.get("entries", []))
            else:
                pairs = data.get("templates", {}).items()

            loaded = 0
            for name, entry_data in pairs:
                if not name or name in self._template_index:
                    continue
                self._template_index[name] = TemplateIndexEntry(
                    name=entry_data.get("name", name),
                    # gen-references writes "source"; runtime persist writes
                    # "source_path".  Accept either so both schemas load.
                    source_path=entry_data.get("source_path") or entry_data.get("source", ""),
                    syntax=entry_data.get("syntax", "jinja2"),
                    variables=entry_data.get("variables", []),
                    origin=entry_data.get("origin", "standalone"),
                    tags=entry_data.get("tags", []),
                    description=entry_data.get("description") or entry_data.get("display_name", ""),
                )
                loaded += 1
            if loaded:
                self._trace(f"_load_persisted_index: loaded {loaded} templates from {index_path}")
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            self._trace(f"_load_persisted_index: failed to load {index_path}: {exc}")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False
        self._extracted_templates.clear()
        self._template_index.clear()
        self._surfaced_template_names.clear()

    def on_history_cleared(self) -> None:
        """Reset per-session enrichment tracking when history is wiped.

        Called by ``JaatoSession.reset_session()`` on a true history clear.
        Clears the ``_surfaced_template_names`` set so previously hinted
        templates can surface again in the fresh conversation.

        Does NOT clear ``_extracted_templates`` (content-hash → path) or
        ``_template_index`` — those are durable on-disk artifacts whose
        lifetime is orthogonal to the conversation.
        """
        self._surfaced_template_names.clear()
        self._trace("on_history_cleared: cleared surfaced template tracking")

    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {},
        }

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
                name="renderTemplateToFile",
                description=(
                    "**PREFERRED OVER MANUAL CODING**: Render a template with variable substitution "
                    "and write the result to a file. When a template exists for your task (check "
                    ".jaato/templates/ or use listAvailableTemplates), you MUST use this tool instead "
                    "of writing code manually. Templates ensure consistency and reduce errors. "
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
                name="listAvailableTemplates",
                description=(
                    "**CHECK THIS BEFORE WRITING CODE**: List all templates available in this "
                    "session. If a template exists for your task, you MUST use renderTemplateToFile "
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
                    "List all variables required by a template, with structural type info "
                    "for each one. Call this before renderTemplateToFile to know exactly "
                    "what variables to provide AND what shape each variable needs.\n\n"
                    "Returns ``variables: list[{name, kind, item_keys?}]`` where:\n"
                    "  - ``kind == 'scalar'``: provide a string/number/bool — used as ``{{name}}``.\n"
                    "  - ``kind == 'section'``: provide a list of dicts — used as "
                    "    ``{{#name}}...{{/name}}``. Each dict in the list MUST contain the "
                    "    keys listed in ``item_keys`` (these are the field names the "
                    "    template's body references inside the section). Example: "
                    "    ``apiEndpoints`` with ``item_keys: ['methodName', 'path']`` means "
                    "    pass ``[{'methodName': 'getCustomer', 'path': '/customers/{id}'}, ...]``.\n"
                    "  - ``kind == 'inverted_section'``: provide a falsy value (empty list, "
                    "    None, False) — block renders only when the value is empty/missing. "
                    "    Used as ``{{^name}}...{{/name}}``.\n\n"
                    "If you guess the wrong kind (e.g. pass a section as flat scalar, or a list "
                    "of strings instead of list of dicts), pybars3 will raise a render error. "
                    "Read the structural info from this tool's output rather than guessing."
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
            "renderTemplateToFile": self._execute_render_template_to_file,
            "listAvailableTemplates": self._execute_list_available,
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
renderTemplateToFile: ...
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

### CRITICAL: Never Read Templates Manually

Do NOT read `.tpl`/`.tmpl` template files with file-reading tools and then pass the content
to `writeNewFile`. This bypasses the template engine's variable substitution, syntax
detection, and validation. The ONLY correct way to use a template is:
- `renderTemplateToFile(template_name="...", variables={...}, output_path="...")`

The template engine resolves the file location, detects syntax (Jinja2/Mustache),
substitutes variables, and writes the result. Manual reading and writing skips all of this.

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

        # Discover standalone templates from referenced directories and
        # merge them into the unified index.
        standalone_entries = self._discover_from_references()
        for entry in standalone_entries:
            self._template_index[entry.name] = entry

        # Persist the unified index to disk and rebuild the tag indexer
        # so contextual surfacing reflects the latest catalog.
        self._persist_index()
        self._indexer.build_index(list(self._template_index.values()))

        total = len(self._template_index)
        if total == 0:
            return SystemInstructionEnrichmentResult(instructions=instructions)

        # Compact pointer in lieu of per-template MANDATORY-USAGE blocks.
        # Per-template enumeration scaled linearly with the catalog size,
        # nagged the model about templates unrelated to the task, and
        # invalidated the cacheable system-instruction prefix on every
        # catalog change.  Relevant templates now surface contextually
        # via enrich_prompt / enrich_tool_result using tag-coherence
        # matching (see TemplateIndexer); the catalog stays discoverable
        # via listAvailableTemplates.
        pointer = (
            f"\n\n---\n"
            f"📦 {total} template{'s' if total != 1 else ''} available "
            f"in the unified index.  Relevant ones surface in-context per "
            f"prompt; call `listAvailableTemplates` for the full catalog "
            f"or `renderTemplateToFile(template_name=..., variables={{...}}, "
            f"output_path=...)` to use one.\n---"
        )
        enriched_instructions = instructions + pointer

        return SystemInstructionEnrichmentResult(
            instructions=enriched_instructions,
            metadata={
                "template_count": total,
                "standalone_count": len(standalone_entries),
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

    # ==================== Prompt Enrichment ====================

    def get_enrichment_priority(self) -> int:
        """Return prompt enrichment priority (lower = earlier).

        Templates run after memory/references contributions so the
        relevance hints sit close to the bottom of the assembled prompt.
        """
        return 60

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe so user prompts are scanned for tag-coherent template matches."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Surface templates whose tags are coherent with the user prompt.

        Mirrors the memory and references plugins' ``enrich_prompt``
        contract: scans *prompt* with :class:`TemplateIndexer`, formats
        a compact hint listing matched templates, and appends it to the
        prompt the model receives.  Untagged templates never surface here
        — they remain reachable via ``listAvailableTemplates``.
        """
        enriched, metadata = self._enrich_text_with_template_hints(prompt)
        return PromptEnrichmentResult(prompt=enriched, metadata=metadata)

    # ==================== Shared enrichment core ====================

    def _enrich_text_with_template_hints(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Compute template hints for arbitrary text (prompt or tool result).

        Returns ``(enriched_text, metadata)``.  The same hint format is
        used on both surfaces so the model sees a consistent block
        whether the trigger was a user prompt or a tool's output.
        """
        if not self._indexer or self._indexer.get_template_count() == 0:
            return text, {"template_matches": 0}

        matches = self._indexer.find_matches_in_text(text, limit=5)
        if not matches:
            return text, {"template_matches": 0}

        # Dedup: drop templates whose hint bullet was already injected in
        # this session — otherwise every tool call re-appends the same
        # "📦 Available Templates" block for the same matches.  When every
        # match has already surfaced, return the text unchanged so no
        # "surfaced N templates" notification fires either.
        new_matches = [e for e in matches if e.name not in self._surfaced_template_names]
        if not new_matches:
            return text, {
                "template_matches": 0,
                "suppressed_duplicates": [e.name for e in matches],
            }
        matches = new_matches

        triggering_tags = self._indexer.triggering_tags(text, matches)

        # Build the hint block.  Each bullet shows the template name,
        # its description (when known), and the literal call form so the
        # model can copy-paste rather than reconstruct argument shape.
        hint_lines = ["", "📦 **Available Templates** — call by name:"]
        for entry in matches:
            desc = entry.description.strip() if entry.description else ""
            tail = f" — {desc}" if desc else ""
            hint_lines.append(f"  - `{entry.name}`{tail}")
            if entry.variables:
                var_preview = ", ".join(entry.variables[:5])
                if len(entry.variables) > 5:
                    var_preview += f", … (+{len(entry.variables) - 5} more)"
                hint_lines.append(f"    variables: [{var_preview}]")
        hint_lines.append(
            "  Use: `renderTemplateToFile(template_name=<name>, "
            "variables={...}, output_path=...)`"
        )

        enriched_text = text + "\n" + "\n".join(hint_lines)

        tag_summary = ", ".join(f'"{t}"' for t in triggering_tags[:3])
        if len(triggering_tags) > 3:
            tag_summary += f" +{len(triggering_tags) - 3} more"

        metadata = {
            "template_matches": len(matches),
            "matched_names": [e.name for e in matches],
            "trigger_tags": triggering_tags,
            "notification": {
                "message": (
                    f"surfaced {len(matches)} relevant template"
                    f"{'s' if len(matches) != 1 else ''}"
                    + (f" (tags: {tag_summary})" if tag_summary else "")
                ),
            },
            "_telemetry": {
                "jaato.enrichment.template.contextual_matches": len(matches),
            },
        }
        # Remember what we injected so the same bullet doesn't reappear
        # on the next tool call within this session.
        self._surfaced_template_names.update(e.name for e in matches)
        return enriched_text, metadata

    # ==================== Tool Result Enrichment ====================

    def get_tool_result_enrichment_priority(self) -> int:
        """Return tool result enrichment priority (lower = earlier)."""
        return 40

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment for template extraction
        and for contextual template surfacing on tool output."""
        return True

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> ToolResultEnrichmentResult:
        """Detect embedded templates in tool results and extract them.

        Scans tool output for fenced code blocks containing Jinja2 template
        syntax. When found, extracts them to .jaato/templates/ and annotates
        the result.

        Extraction is **suppressed** when the file being read belongs to a
        reference that declares ``contents.templates`` — the standalone
        templates in the reference folder are authoritative and should be
        used instead of extracting embedded snippets from documentation.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.
            tool_args: Optional tool call arguments for path detection.

        Returns:
            ToolResultEnrichmentResult with annotated result and extraction metadata.
        """
        result_preview = result[:100].replace('\n', '\\n') + ('...' if len(result) > 100 else '')
        self._trace(f"enrich_tool_result [{tool_name}]: {len(result)} chars, preview: {result_preview}")

        # Suppress extraction when file belongs to a reference with standalone templates
        if tool_args and self._should_suppress_extraction(tool_args):
            self._trace("  suppressed: file belongs to reference with contents.templates")
            return self._finalize_tool_result(result)

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
                return self._finalize_tool_result(result)

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
            return self._finalize_tool_result(result)

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
            return self._finalize_tool_result(result)

        annotation_block = "\n\n---\n[!] **MANDATORY TEMPLATES AVAILABLE - USE THESE INSTEAD OF MANUAL CODING:**\n" + "\n\n".join(annotations) + "\n---"
        enriched_result = result + annotation_block

        # Embedded extraction added new entries to the catalog — refresh
        # the tag indexer so subsequent prompts see them.
        if extracted:
            self._indexer.build_index(list(self._template_index.values()))

        return self._finalize_tool_result(
            enriched_result,
            base_metadata={
                "extracted_count": len(extracted),
                "templates": [
                    {"hash": h, "path": str(p), "variables": v}
                    for h, p, v in extracted
                ],
                "_telemetry": {
                    "jaato.enrichment.template.extracted_count": len(extracted),
                },
            },
        )

    def _finalize_tool_result(
        self,
        result_text: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResultEnrichmentResult:
        """Layer contextual template hints on top of any extraction-stage output.

        Every return path of :meth:`enrich_tool_result` routes through here
        so contextual surfacing (driven by the tag indexer) is consistent
        across the early-exit paths and the extraction success path.

        ``base_metadata`` carries any extraction-stage telemetry the caller
        wants to preserve; contextual hint metadata is merged on top
        (contextual keys never overwrite extraction keys).
        """
        enriched, ctx_meta = self._enrich_text_with_template_hints(result_text)
        metadata: Dict[str, Any] = dict(base_metadata or {})
        for k, v in ctx_meta.items():
            metadata.setdefault(k, v)
        return ToolResultEnrichmentResult(result=enriched, metadata=metadata)

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

    def _should_suppress_extraction(self, tool_args: Dict[str, Any]) -> bool:
        """Check if embedded template extraction should be suppressed.

        Returns True when the file being read belongs to a selected reference
        that declares ``contents.templates``. In that case, the reference's
        standalone templates are authoritative and embedded extraction would
        produce duplicates.

        Args:
            tool_args: The tool call arguments dict (looks for ``path`` key).

        Returns:
            True if extraction should be suppressed.
        """
        if not self._plugin_registry:
            return False

        # Extract file path from tool args
        file_path = tool_args.get("path") or tool_args.get("file_path")
        if not file_path or not isinstance(file_path, str):
            return False

        try:
            ref_plugin = self._plugin_registry.get_plugin("references")
        except Exception:
            return False

        if ref_plugin is None:
            return False

        try:
            return ref_plugin.file_belongs_to_reference_with_templates(file_path)
        except Exception as e:
            self._trace(f"_should_suppress_extraction: error querying references: {e}")
            return False

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

    @staticmethod
    def _strip_generated_annotations(content: str) -> str:
        """Remove ``@generated`` annotation lines from template content.

        Imported templates often contain ``@generated {{skillId}} v{{skillVersion}}``
        in JavaDoc / comment lines. These are metadata annotations — their
        placeholders are **not** template variables and should be excluded from
        variable extraction and rendering. Stripping these lines prevents
        undefined-variable errors when the template is rendered without
        ``skillId`` / ``skillVersion`` in the variable context.

        Args:
            content: Raw template content string.

        Returns:
            Content with ``@generated`` lines removed.
        """
        return _GENERATED_ANNOTATION_RE.sub('', content)

    def _extract_variables(self, content: str) -> List[str]:
        """Extract variable names from template content.

        Uses Jinja2's AST parser for accurate extraction from Jinja2 templates,
        or regex for Mustache templates. This ensures the model knows exactly
        which variables are required before rendering.

        ``@generated`` annotation lines are stripped before extraction so that
        metadata placeholders (e.g. ``{{skillId}}``) are never reported as
        required template variables.

        Spring Boot ``${...}`` placeholders containing Mustache variables are
        protected before extraction so the regex does not produce bogus
        variable names with a leading brace (see
        ``_protect_spring_placeholders``).

        Args:
            content: Template content string.

        Returns:
            Sorted list of variable names required by the template.
        """
        # Strip @generated annotation lines so their metadata placeholders
        # (e.g. {{skillId}}, {{skillVersion}}) are not treated as variables.
        content = self._strip_generated_annotations(content)

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
            # Protect Spring Boot ${{{VAR}} patterns so the regex extracts
            # the variable name cleanly (without a leading brace).
            protected = self._protect_spring_placeholders(content)
            # Match simple variables {{var}}, excluding section markers and comments
            matches = re.findall(r'\{\{([^#/^!}]+)\}\}', protected)
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
        the appropriate engine. ``@generated`` annotation lines are stripped
        before rendering so that their metadata placeholders do not cause
        undefined-variable errors.

        Args:
            template: Template content string.
            variables: Key-value pairs for template variable substitution.

        Returns:
            Tuple of (rendered_content, error_dict).
            If error_dict is not None, rendering failed.
        """
        # Strip @generated annotation lines before rendering so metadata
        # placeholders (e.g. {{skillId}}) don't cause undefined-variable errors.
        template = self._strip_generated_annotations(template)

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

    @staticmethod
    def _inject_list_metadata(variables: Dict[str, Any]) -> Dict[str, Any]:
        """Inject ``first``, ``last``, and ``@index`` metadata into list items.

        Mustache templates commonly use ``{{^last}}, {{/last}}`` to produce
        comma-separated lists without a trailing comma.  This pattern requires
        each list item to carry a boolean ``last`` property — but callers
        rarely provide it, causing ``{{^last}}`` (inverted section) to render
        for *every* item (undefined is falsy) and produce trailing commas.

        This method walks the variables dict recursively and, for every list
        of dicts, injects:

        * ``first`` — ``True`` on the first item, ``False`` on the rest.
        * ``last``  — ``True`` on the last item, ``False`` on the rest.
        * ``@index`` — 0-based position within the list.

        Existing keys are **never** overwritten, so callers can still supply
        their own values when needed.

        The method returns a shallow copy of the top-level dict; nested dicts
        inside list items are copied only when metadata is actually injected.

        Args:
            variables: Original template variables dict.

        Returns:
            A (possibly copied) variables dict with metadata injected.
        """

        def _process_value(value: Any) -> Any:
            """Recursively process a value, injecting metadata into lists of dicts."""
            if isinstance(value, list) and value and isinstance(value[0], dict):
                last_idx = len(value) - 1
                new_list = []
                for idx, item in enumerate(value):
                    # Recurse into nested dicts first
                    new_item = _process_dict(item)
                    # Inject metadata only when the key is absent
                    if 'first' not in new_item:
                        new_item['first'] = (idx == 0)
                    if 'last' not in new_item:
                        new_item['last'] = (idx == last_idx)
                    if '@index' not in new_item:
                        new_item['@index'] = idx
                    new_list.append(new_item)
                return new_list
            if isinstance(value, dict):
                return _process_dict(value)
            return value

        def _process_dict(d: dict) -> dict:
            """Shallow-copy a dict and recurse into its values."""
            result = {}
            for k, v in d.items():
                result[k] = _process_value(v)
            return result

        return _process_dict(variables)

    @staticmethod
    def _protect_spring_placeholders(template: str) -> str:
        """Replace ``${{{`` with ``$SENTINEL{{`` to prevent Handlebars collision.

        Spring Boot property placeholders like ``${VAR:default}`` can collide
        with Handlebars triple-brace unescaped syntax when a Mustache variable
        appears immediately inside, e.g.::

            ${{{SERVICE_NAME}}_SYSTEM_API_URL:http://localhost:8081}

        pybars3 would interpret ``{{{SERVICE_NAME}}}`` as an unescaped
        variable, consuming the opening brace that belongs to Spring's ``${``.

        This method replaces the ``${`` before the Mustache ``{{`` with a
        sentinel so that pybars3 sees a normal double-brace variable instead.
        Call ``_restore_spring_placeholders`` after rendering to put the
        ``${`` back.

        Args:
            template: Raw template content.

        Returns:
            Template with ``${{{`` sequences protected by the sentinel.
        """
        return _SPRING_COLLISION_RE.sub('$' + _SPRING_BRACE_SENTINEL + '{{', template)

    @staticmethod
    def _restore_spring_placeholders(rendered: str) -> str:
        """Restore ``${`` from the sentinel inserted by ``_protect_spring_placeholders``.

        Args:
            rendered: Rendered output containing sentinel markers.

        Returns:
            Output with sentinels replaced by ``${``.
        """
        return rendered.replace('$' + _SPRING_BRACE_SENTINEL, '${')

    def _render_mustache(self, template: str, variables: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """Render template using Handlebars syntax.

        Supports full Handlebars syntax including dotted paths:
        - Variables: ``{{variable_name}}``, ``{{a.b.c}}``
        - Sections/loops: ``{{#items}}…{{/items}}``, ``{{#a.b}}…{{/a.b}}``
        - Conditionals: ``{{#if condition}}``, ``{{#if a.b}}``
        - Each loops: ``{{#each items}}``, ``{{#each a.b}}``
        - Inverted sections: ``{{^isEmpty}}…{{/isEmpty}}``, ``{{^a.b}}…{{/a.b}}``
        - Current item: ``{{.}}``, ``{{this}}``

        Before rendering, ``first``, ``last``, and ``@index`` metadata are
        automatically injected into list-of-dict items so that templates can
        use ``{{^last}}, {{/last}}`` for comma-separated lists without the
        caller having to provide those flags manually (see
        ``_inject_list_metadata``).

        Dotted paths in section/inverted tags are preprocessed into
        equivalent pybars3-compatible constructs before compilation
        (see ``_preprocess_mustache_dotted_paths``).

        Spring Boot property placeholders (``${...}``) that contain Mustache
        variables are protected before compilation to prevent a triple-brace
        collision with Handlebars unescaped syntax (see
        ``_protect_spring_placeholders``).

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
            variables = self._inject_list_metadata(variables)
            protected = self._protect_spring_placeholders(template)
            preprocessed = self._preprocess_mustache_dotted_paths(protected)
            compiler = Compiler()
            compiled_template = compiler.compile(preprocessed)
            rendered = compiled_template(variables)
            rendered = self._restore_spring_placeholders(rendered)
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
            "count": len(templates),
            "_telemetry": {
                "jaato.template.operation": "list",
                "jaato.template.count": len(templates),
            },
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
            "template_syntax": syntax,
            "_telemetry": {
                "jaato.template.operation": "write",
                "jaato.template.path": str(out_path),
                "jaato.template.bytes_written": bytes_written,
                "jaato.template.syntax": syntax,
            },
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

    def _parse_mustache_structure(self, content: str) -> List[Dict[str, Any]]:
        """Walk a Mustache template, classify each variable by kind.

        Returns a list of variable descriptors:

            [
              {"name": "Entity", "kind": "scalar"},
              {"name": "apiEndpoints", "kind": "section",
               "item_keys": ["methodName", "path", "returnType",
                             "isVoid", ...],
               "has_inverted_branch": false},
              {"name": "isEmpty", "kind": "inverted_section"},
            ]

        Kinds:
        - ``"scalar"``: ``{{name}}`` or ``{{{name}}}`` — replaced
          verbatim with the value (triple-brace == unescaped output;
          same variable shape, normalised here so the model isn't
          misled by escaping syntax).
        - ``"section"``: ``{{#name}}...{{/name}}`` — when ``name`` is
          a list, the body renders once per item with each item as
          context; ``item_keys`` collects the field names referenced
          ANYWHERE inside the iteration (incl. through nested boolean
          sections like ``{{#item.flag}}...{{/item.flag}}``) so the
          agent knows the full inner shape required.  When the same
          identifier ALSO appears as ``{{^name}}`` (Mustache if/else
          idiom), ``has_inverted_branch`` is set so the agent knows
          there's an else-branch that fires when the value is
          falsy/empty.
        - ``"inverted_section"``: ``{{^name}}...{{/name}}`` —
          standalone, body renders only when ``name`` is falsy/empty
          (no iteration); inner references aren't item-keys.

        Top-level scalars are emitted ONLY for references that occur
        OUTSIDE any section; references inside sections become
        item_keys of the OUTERMOST iteration section, never top-level.
        Without this rule, deeply-nested per-item references (typical
        ``{{#apiEndpoints}}...{{#isVoid}}...{{/isVoid}}{{^isVoid}}...
        {{/isVoid}}...{{/apiEndpoints}}`` patterns) leaked to top
        level, polluting the agent's variable dict.

        Triple-brace ``{{{x}}}`` is normalised to ``{{x}}`` before
        parsing — they're the same variable from a structural
        perspective; only the render-time escaping differs.
        """
        # Mustache triple-brace ``{{{x}}}`` is the unescaped-output
        # form.  Structurally it's identical to ``{{x}}`` — same
        # variable, same kind.  Normalise here so the regex below
        # doesn't include the inner ``{`` in the captured name.
        content = re.sub(r'\{\{\{([^}]+)\}\}\}', r'{{\1}}', content)

        # Match all {{...}} constructs in order so we can build a
        # section stack and attribute scalar references to their
        # enclosing sections.  Capture the optional prefix
        # (``#`` / ``^`` / ``/`` / ``!``) and the name.
        pattern = re.compile(r'\{\{([#^/!]?)([^}]+)\}\}')

        variables: Dict[str, Dict[str, Any]] = {}
        # Stack of (kind, name) for currently-open sections.
        section_stack: List[tuple[str, str]] = []

        def _outermost_iteration_section() -> Optional[str]:
            """Return the name of the outermost ``section`` (i.e. the
            iteration boundary), or None if not inside any section.
            Used to attribute scalar references and nested-section
            names to the iteration that produces them — never to the
            inner boolean-check sections.
            """
            for kind, name in section_stack:
                if kind == "section":
                    return name
            return None

        for match in pattern.finditer(content):
            prefix = match.group(1)
            name = match.group(2).strip()

            # Skip Mustache comments and current-context markers.
            if prefix == '!':
                continue
            if name in ('.', 'this'):
                continue

            # Helper: when we encounter ANY identifier inside a
            # section (whether scalar reference or nested section
            # marker), credit it to the OUTERMOST iteration section's
            # item_keys.  Without this, a nested ``{{#isVoid}}`` body
            # attributes its inner refs to ``isVoid`` instead of to
            # ``apiEndpoints``, and the inner refs leak to top-level.
            outer_iter = _outermost_iteration_section()

            if prefix == '#':
                # Opening section.  Two cases:
                # - At top level (no outer iteration): register as a
                #   top-level section variable.
                # - Nested inside an outer iteration: this section's
                #   identifier is a per-item field of the outer
                #   iteration (e.g. ``isVoid`` inside ``apiEndpoints``).
                #   Credit to outer.item_keys; do NOT create a
                #   top-level entry.  Nested-section identifiers are
                #   provided as fields on each list-item dict, never
                #   at the top of the variables dict.
                if outer_iter is None or outer_iter == name:
                    entry = variables.get(name)
                    if entry is None:
                        variables[name] = {
                            "name": name, "kind": "section", "item_keys": set(),
                        }
                    elif entry["kind"] == "scalar":
                        # Promote: section is more constrained.
                        entry["kind"] = "section"
                        entry.setdefault("item_keys", set())
                    elif entry["kind"] == "inverted_section":
                        # Mustache if/else with the same identifier,
                        # encountered ^ first then # — promote to
                        # section AND mark inverted branch exists.
                        entry["kind"] = "section"
                        entry.setdefault("item_keys", set())
                        entry["has_inverted_branch"] = True
                    # else: already a section; idempotent.
                else:
                    # Nested inside outer iteration: credit only as
                    # an item-key of outer.  No top-level entry.
                    outer_entry = variables.get(outer_iter)
                    if outer_entry and outer_entry["kind"] == "section":
                        outer_entry["item_keys"].add(name)
                section_stack.append(("section", name))
            elif prefix == '^':
                # Opening inverted section.  Same nested-vs-top-level
                # split as ``#`` — nested inverted sections credit
                # only as item_keys of outer.
                if outer_iter is None or outer_iter == name:
                    entry = variables.get(name)
                    if entry is None:
                        variables[name] = {
                            "name": name, "kind": "inverted_section",
                        }
                    elif entry["kind"] == "section":
                        # Mustache if/else, # encountered first then
                        # ^ — mark inverted branch alongside section.
                        entry["has_inverted_branch"] = True
                    # else: already inverted_section or scalar; leave.
                else:
                    # Nested inverted: credit only as item-key.
                    outer_entry = variables.get(outer_iter)
                    if outer_entry and outer_entry["kind"] == "section":
                        outer_entry["item_keys"].add(name)
                section_stack.append(("inverted_section", name))
            elif prefix == '/':
                # Closing marker — pop the matching open section.
                if section_stack and section_stack[-1][1] == name:
                    section_stack.pop()
            else:
                # Scalar reference.  Three cases:
                # - Inside any section: credit the outermost
                #   iteration section's item_keys; do NOT add as
                #   top-level scalar.
                # - Outside all sections: top-level scalar.
                # - Inside an inverted-only section (no outer
                #   iteration): top-level scalar (the inverted
                #   block runs in the parent context, so refs are
                #   parent-scope variables).
                if section_stack:
                    if outer_iter and outer_iter != name:
                        outer_entry = variables.get(outer_iter)
                        if outer_entry and outer_entry["kind"] == "section":
                            # Leftmost token of dotted paths.
                            item_key = name.split(".")[0] if "." in name else name
                            outer_entry["item_keys"].add(item_key)
                    else:
                        # Inside inverted-only sections: parent-scope
                        # scalar.
                        variables.setdefault(
                            name, {"name": name, "kind": "scalar"},
                        )
                else:
                    variables.setdefault(name, {"name": name, "kind": "scalar"})

        # Convert sets to sorted lists for stable output.
        result = []
        for name in sorted(variables.keys()):
            entry = variables[name].copy()
            if "item_keys" in entry:
                entry["item_keys"] = sorted(list(entry["item_keys"]))
            # has_inverted_branch defaults to False on sections that
            # don't have an inverted form — explicit False rather
            # than missing key keeps the schema predictable.
            if entry["kind"] == "section":
                entry.setdefault("has_inverted_branch", False)
            result.append(entry)
        return result

    def _execute_list_template_variables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all undeclared variables from a template.

        Uses Jinja2's AST parser for Jinja2 templates to find undeclared variables,
        or structural Mustache parsing for Mustache templates.

        For Mustache (server 0.6.28+), each variable carries its kind
        (``scalar`` / ``section`` / ``inverted_section``) and sections
        also carry ``item_keys`` — the fields each list item must
        provide when rendering.  This eliminates a non-determinism
        source where agents would guess wrong shape on first attempt
        (passing ``apiEndpoints`` as flat scalars or ``updateableFields``
        as ``list[str]`` instead of ``list[dict]``), trigger a
        ``pybars3`` render error, and self-correct on retry — with
        the retry path producing different content across runs.

        Args:
            args: Tool arguments containing 'template_name'.

        Returns:
            Dict with 'variables' list and 'syntax' type, or 'error' on failure.
            For Mustache: ``variables`` is ``list[{name, kind, item_keys?}]``.
            For Jinja2: ``variables`` is ``list[{name, kind: "scalar"}]``
            (kind detection for Jinja2 is a follow-up — current shape
            is consistent in API but flat in semantics).
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
                # Wrap in the same structured shape Mustache returns
                # so consumers don't need to branch on syntax.  Jinja2
                # AST-based kind detection (for / if blocks) is a
                # follow-up; for now mark all as "scalar" — agents
                # using Jinja2 templates today have less determinism
                # surface anyway.
                return {
                    "variables": [
                        {"name": v, "kind": "scalar"}
                        for v in sorted(variables)
                    ],
                    "syntax": "jinja2",
                    "template_name": template_name,
                    "count": len(variables),
                }
            except Exception as e:
                return {
                    "error": f"Failed to parse Jinja2 template: {e}",
                    "syntax": "jinja2",
                    "template_name": template_name
                }

        elif syntax == "mustache":
            # Structural parse: classify each variable by kind
            # (scalar / section / inverted_section) and collect
            # ``item_keys`` for sections — the inner field names the
            # agent must provide on each list-item dict.  See
            # ``_parse_mustache_structure`` docstring for the full
            # rules.
            structured = self._parse_mustache_structure(template_content)
            return {
                "variables": structured,
                "syntax": "mustache",
                "template_name": template_name,
                "count": len(structured),
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
