"""Template rendering plugin implementation.

Provides tools for rendering Jinja2 templates with variable substitution
and writing the results to files.

Key features:
1. System instruction enrichment: Detects embedded templates in system
   instructions (from MODULE.md injected by references plugin) and extracts
   them to .jaato/templates/ for later use via renderTemplate tool.
2. Tool result enrichment: Detects embedded templates in tool outputs
   (e.g., from cat, readFile) and extracts them similarly.
3. Template rendering: Renders Jinja2 templates with variable substitution.

See docs/template-tool-design.md for the design specification.
"""

import hashlib
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import UserCommand, SystemInstructionEnrichmentResult, ToolResultEnrichmentResult, PermissionDisplayInfo
from ..model_provider.types import ToolSchema


# Regex patterns for detecting Jinja2 template syntax in code blocks
# Matches {{ variable }}, {% control %}, or {# comment #}
JINJA2_VARIABLE_PATTERN = re.compile(r'\{\{\s*\w+')
JINJA2_CONTROL_PATTERN = re.compile(r'\{%\s*\w+')
JINJA2_COMMENT_PATTERN = re.compile(r'\{#.*#\}')

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

    Tools provided:
    - renderTemplate: Render a template with variables and write to file

    Prompt enrichment:
    - Detects code blocks containing Jinja2 template syntax
    - Extracts them to .jaato/templates/ directory
    - Annotates the prompt with the extracted template paths

    This plugin uses Jinja2 syntax for templates:
    - Variables: {{ variable_name }}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in items %}...{% endfor %}
    """

    def __init__(self):
        self._initialized = False
        self._agent_name: Optional[str] = None
        self._base_path: Path = Path.cwd()
        self._templates_dir: Path = Path.cwd() / ".jaato" / "templates"
        # Track extracted templates in this session: hash -> path
        self._extracted_templates: Dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "template"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [TEMPLATE{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the template plugin."""
        config = config or {}
        self._agent_name = config.get("agent_name")

        # Allow custom base path
        if "base_path" in config:
            self._base_path = Path(config["base_path"])

        # Templates directory under .jaato
        self._templates_dir = self._base_path / ".jaato" / "templates"

        self._initialized = True
        self._trace(f"initialized: base_path={self._base_path}, templates_dir={self._templates_dir}")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False
        self._extracted_templates.clear()

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for template tools."""
        return [
            ToolSchema(
                name="renderTemplate",
                description=(
                    "**PREFERRED OVER MANUAL CODING**: Render a template with variable substitution "
                    "and write the result to a file. When a template exists for your task (check "
                    ".jaato/templates/ or use listExtractedTemplates), you MUST use this tool instead "
                    "of writing code manually. Templates ensure consistency and reduce errors. "
                    "Templates support Jinja2 syntax: variables ({{name}}), conditionals "
                    "({% if condition %}...{% endif %}), and loops ({% for item in items %}...{% endfor %}). "
                    "Provide either 'template' for inline content or 'template_path' for extracted templates."
                ),
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
                            "description": "Key-value pairs for template variable substitution.",
                            "additionalProperties": True
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path where rendered content should be written."
                        }
                    },
                    "required": ["variables", "output_path"]
                }
            ),
            ToolSchema(
                name="listExtractedTemplates",
                description=(
                    "**CHECK THIS BEFORE WRITING CODE**: List templates extracted from documentation "
                    "in this session. If a template exists for your task, you MUST use renderTemplate "
                    "instead of writing code manually. These templates were detected in code blocks "
                    "with Jinja2 syntax and extracted to .jaato/templates/."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            ToolSchema(
                name="renderTemplateToFile",
                description=(
                    "Render a template with variable substitution and write the result directly to a file. "
                    "This is a convenience tool that combines template rendering and file writing in one operation. "
                    "Use simple {{variable_name}} syntax for variable substitution. "
                    "Provide either 'template' for inline content or 'template_path' for a template file."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "Path where rendered content will be written."
                        },
                        "template_path": {
                            "type": "string",
                            "description": "Path to template file. Mutually exclusive with 'template'."
                        },
                        "template": {
                            "type": "string",
                            "description": "Inline template string. Mutually exclusive with 'template_path'."
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
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor functions for each tool."""
        return {
            "renderTemplate": self._execute_render_template,
            "listExtractedTemplates": self._execute_list_extracted,
            "renderTemplateToFile": self._execute_render_template_to_file,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for template tools."""
        return """## Template Rendering (MANDATORY USAGE)

**CRITICAL**: When templates exist for a task, you MUST use template tools instead of
manually writing code. Templates ensure consistency, reduce errors, and follow established
patterns. Manual coding when a template exists is NOT acceptable.

### TEMPLATE TOOLS:

**renderTemplate(template_path, variables, output_path)** - Render with full Jinja2 syntax
  - Supports Jinja2 features: variables, conditionals, loops, filters
  - Returns: {"success": true, "output_path": "...", "size": 1234, "lines": 42}

**renderTemplateToFile(output_path, template_path, variables)** - Simple render and write
  - Uses simple {{variable}} substitution only
  - Checks if file exists (use overwrite=true to replace)
  - Returns: {"success": true, "output_path": "...", "bytes_written": 1234}

**listExtractedTemplates()** - List templates extracted from documentation
  - Shows all templates extracted in this session
  - Auto-approved (no permission required)

### Template Priority Rule
1. ALWAYS check if a template exists before writing code
2. If a template matches your task, USE IT via template tools
3. ONLY write code manually if NO suitable template exists
4. When in doubt, use `listExtractedTemplates` to see available templates

### Templates Location
Templates are stored in `.jaato/templates/`. When you read documentation files
(like MODULE.md) containing embedded templates, they are automatically extracted
to this directory. Watch for "[Template extracted: ...]" annotations.

### Example - Generate a Java service from template:
```
renderTemplateToFile(
    output_path="/src/main/java/com/bank/CustomerService.java",
    template_path="/templates/service.java.tmpl",
    variables={"class_name": "CustomerService", "package": "com.bank.customer"}
)
```

### Example - Using inline template:
```
renderTemplateToFile(
    output_path="/src/CustomerService.java",
    template="package {{package}};\\n\\npublic class {{class_name}} {}",
    variables={"class_name": "CustomerService", "package": "com.bank"}
)
```

### Template Syntax
**For renderTemplateToFile (simple):**
- Variables: {{variable_name}} - replaced with value
- Escape literal {{: \\{{

**For renderTemplate (full Jinja2):**
- Variables: {{ variable_name }}
- Conditionals: {% if condition %}...{% endif %}
- Loops: {% for item in items %}...{% endfor %}
- Filters: {{ name | upper }}, {{ value | default('fallback') }}

Template rendering requires approval since it writes files."""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        return ["listExtractedTemplates"]

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
        template_path = arguments.get("template_path")
        variables = arguments.get("variables", {})
        overwrite = arguments.get("overwrite", False)

        # Build summary
        action = "Overwrite" if overwrite else "Create"
        source = template_path if template_path else "(inline template)"
        summary = f"{action} file: {output_path} from {source}"

        # Build details showing the template and variables
        details_lines = []

        if template_path:
            details_lines.append(f"Template: {template_path}")
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
        """Detect embedded templates in system instructions and extract them.

        Scans the system instructions for fenced code blocks containing Jinja2
        template syntax ({{ }}, {% %}, {# #}). When found, extracts them to
        .jaato/templates/ and annotates the instructions with the extracted paths.

        Args:
            instructions: Combined system instructions (includes MODULE.md content
                from references plugin).

        Returns:
            SystemInstructionEnrichmentResult with annotated instructions and
            extraction metadata.
        """
        instructions_preview = instructions[:100].replace('\n', '\\n') + ('...' if len(instructions) > 100 else '')
        self._trace(f"enrich_system_instructions called: {len(instructions)} chars, preview: {instructions_preview}")

        # Find all code blocks in the instructions
        code_blocks = self._find_code_blocks(instructions)

        if not code_blocks:
            self._trace("  no code blocks found in instructions")
            return SystemInstructionEnrichmentResult(instructions=instructions)

        # Filter to blocks that contain template syntax
        template_blocks = [
            (lang, content, start, end)
            for lang, content, start, end in code_blocks
            if self._is_template(content)
        ]

        if not template_blocks:
            # Log each code block for debugging
            for i, (lang, content, start, end) in enumerate(code_blocks):
                preview = content[:80].replace('\n', '\\n') + ('...' if len(content) > 80 else '')
                self._trace(f"  code block {i+1}: lang={lang!r}, {len(content)} chars: {preview}")
            self._trace(f"  found {len(code_blocks)} code blocks but none with template syntax")
            return SystemInstructionEnrichmentResult(instructions=instructions)

        self._trace(f"enrich_system_instructions: found {len(template_blocks)} template blocks")

        # Extract each template and collect annotations
        extracted: List[Tuple[str, Path, List[str]]] = []  # (content_hash, path, variables)
        annotations: List[str] = []

        for lang, content, start, end in template_blocks:
            # Generate template ID and path
            content_hash = self._hash_content(content)

            # Skip if already extracted this content
            if content_hash in self._extracted_templates:
                template_path = self._extracted_templates[content_hash]
                self._trace(f"  skipping already-extracted: {template_path.name}")
                continue

            # Determine template filename
            template_name = self._generate_template_name(instructions, content, lang, start)
            template_path, is_new = self._extract_template(template_name, content, lang)

            if template_path:
                self._extracted_templates[content_hash] = template_path

                # Only add annotation for newly created templates
                if is_new:
                    variables = self._extract_variables(content)
                    extracted.append((content_hash, template_path, variables))

                    # Build annotation
                    rel_path = template_path.relative_to(self._base_path) if template_path.is_relative_to(self._base_path) else template_path
                    var_list = ", ".join(variables[:5])
                    if len(variables) > 5:
                        var_list += f", ... ({len(variables)} total)"

                    annotations.append(
                        f"[!] **TEMPLATE AVAILABLE - MANDATORY USAGE**: {rel_path}\n"
                        f"  Variables: {var_list or '(none detected)'}\n"
                        f"  **YOU MUST USE THIS TEMPLATE** instead of writing code manually.\n"
                        f"  Call: renderTemplate(template_path=\"{rel_path}\", variables={{...}}, output_path=\"...\")"
                    )
                    self._trace(f"  extracted: {template_path.name} with {len(variables)} variables")
                else:
                    self._trace(f"  reusing existing: {template_path.name}")

        if not annotations:
            return SystemInstructionEnrichmentResult(instructions=instructions)

        # Append annotations to instructions
        annotation_block = "\n\n---\n[!] **MANDATORY TEMPLATES EXTRACTED - USE THESE INSTEAD OF MANUAL CODING:**\n" + "\n\n".join(annotations) + "\n---"
        enriched_instructions = instructions + annotation_block

        return SystemInstructionEnrichmentResult(
            instructions=enriched_instructions,
            metadata={
                "extracted_count": len(extracted),
                "templates": [
                    {"hash": h, "path": str(p), "variables": v}
                    for h, p, v in extracted
                ]
            }
        )

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

        if not code_blocks:
            self._trace("  no code blocks found in tool result")
            return ToolResultEnrichmentResult(result=result)

        # Filter to blocks that contain template syntax
        template_blocks = [
            (lang, content, start, end)
            for lang, content, start, end in code_blocks
            if self._is_template(content)
        ]

        if not template_blocks:
            self._trace(f"  found {len(code_blocks)} code blocks but none with template syntax")
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: found {len(template_blocks)} template blocks")

        # Extract each template and collect annotations
        extracted: List[Tuple[str, Path, List[str]]] = []
        annotations: List[str] = []

        for lang, content, start, end in template_blocks:
            content_hash = self._hash_content(content)

            if content_hash in self._extracted_templates:
                template_path = self._extracted_templates[content_hash]
                self._trace(f"  skipping already-extracted: {template_path.name}")
                continue

            template_name = self._generate_template_name(result, content, lang, start)
            template_path, is_new = self._extract_template(template_name, content, lang)

            if template_path:
                self._extracted_templates[content_hash] = template_path

                # Only add annotation for newly created templates
                if is_new:
                    variables = self._extract_variables(content)
                    extracted.append((content_hash, template_path, variables))

                    rel_path = template_path.relative_to(self._base_path) if template_path.is_relative_to(self._base_path) else template_path
                    var_list = ", ".join(variables[:5])
                    if len(variables) > 5:
                        var_list += f", ... ({len(variables)} total)"

                    annotations.append(
                        f"[!] **TEMPLATE AVAILABLE - MANDATORY USAGE**: {rel_path}\n"
                        f"  Variables: {var_list or '(none detected)'}\n"
                        f"  **YOU MUST USE THIS TEMPLATE** instead of writing code manually.\n"
                        f"  Call: renderTemplate(template_path=\"{rel_path}\", variables={{...}}, output_path=\"...\")"
                    )
                    self._trace(f"  extracted: {template_path.name} with {len(variables)} variables")
                else:
                    self._trace(f"  reusing existing: {template_path.name}")

        if not annotations:
            return ToolResultEnrichmentResult(result=result)

        annotation_block = "\n\n---\n[!] **MANDATORY TEMPLATES EXTRACTED - USE THESE INSTEAD OF MANUAL CODING:**\n" + "\n\n".join(annotations) + "\n---"
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
        """Check if content contains Jinja2 template syntax."""
        return bool(
            JINJA2_VARIABLE_PATTERN.search(content) or
            JINJA2_CONTROL_PATTERN.search(content)
        )

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
                if template_path.read_text() == content:
                    return template_path, False  # Reuse existing (not new)
                template_path = self._templates_dir / f"{base_name}-{counter}{suffix}"
                counter += 1

            # Write template
            template_path.write_text(content)
            self._trace(f"wrote template: {template_path}")
            return template_path, True  # Newly created

        except (IOError, OSError) as e:
            self._trace(f"error extracting template {name}: {e}")
            return None, False

    def _extract_variables(self, content: str) -> List[str]:
        """Extract variable names from template content."""
        # Find all {{ variable }} patterns
        var_pattern = re.compile(r'\{\{\s*(\w+)')
        variables = set()

        for match in var_pattern.finditer(content):
            var_name = match.group(1)
            # Filter out common Jinja2 built-ins
            if var_name not in ('if', 'else', 'elif', 'endif', 'for', 'endfor', 'loop', 'true', 'false', 'none'):
                variables.add(var_name)

        return sorted(variables)

    # ==================== Tool Executors ====================

    def _execute_render_template(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute renderTemplate tool."""
        template = args.get("template")
        template_path = args.get("template_path")
        variables = args.get("variables", {})
        output_path = args.get("output_path", "")

        # Validation
        if not output_path:
            return {"error": "output_path is required"}

        if not template and not template_path:
            return {"error": "Either 'template' or 'template_path' must be provided"}

        if template and template_path:
            return {"error": "Provide either 'template' or 'template_path', not both"}

        # Get template content
        if template_path:
            try:
                path = Path(template_path)
                if not path.is_absolute():
                    path = self._base_path / path
                template = path.read_text()
            except FileNotFoundError:
                return {"error": f"Template file not found: {template_path}"}
            except IOError as e:
                return {"error": f"Failed to read template: {e}"}

        # Try to import Jinja2
        try:
            from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            return {
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
        except TemplateSyntaxError as e:
            return {
                "error": f"Template syntax error at line {e.lineno}: {e.message}",
                "status": "syntax_error"
            }
        except UndefinedError as e:
            # Try to suggest similar variable names
            return {
                "error": f"Undefined variable: {e.message}",
                "available_variables": list(variables.keys()),
                "status": "undefined_variable"
            }
        except Exception as e:
            return {
                "error": f"Template render error: {str(e)}",
                "status": "render_error"
            }

        # Write output
        try:
            out_path = Path(output_path)
            if not out_path.is_absolute():
                out_path = self._base_path / out_path

            # Create parent directories
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path.write_text(rendered)
        except IOError as e:
            return {
                "error": f"Failed to write output: {e}",
                "status": "write_error"
            }

        return {
            "success": True,
            "output_path": str(out_path),
            "size": len(rendered),
            "lines": rendered.count('\n') + 1,
            "variables_used": list(variables.keys())
        }

    def _execute_list_extracted(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List templates extracted in this session."""
        if not self._extracted_templates:
            return {
                "templates": [],
                "message": "No templates have been extracted in this session."
            }

        templates = []
        for content_hash, path in self._extracted_templates.items():
            try:
                rel_path = path.relative_to(self._base_path) if path.is_relative_to(self._base_path) else path
                content = path.read_text() if path.exists() else "(file not found)"
                variables = self._extract_variables(content) if path.exists() else []
                templates.append({
                    "path": str(rel_path),
                    "hash": content_hash,
                    "variables": variables,
                    "exists": path.exists()
                })
            except Exception:
                templates.append({
                    "path": str(path),
                    "hash": content_hash,
                    "error": "Could not read template"
                })

        return {
            "templates": templates,
            "count": len(templates)
        }

    def _execute_render_template_to_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute renderTemplateToFile tool.

        Renders a template using simple {{variable}} substitution and writes
        the result to a file.
        """
        output_path = args.get("output_path", "")
        template = args.get("template")
        template_path = args.get("template_path")
        variables = args.get("variables", {})
        overwrite = args.get("overwrite", False)

        # Validation
        if not output_path:
            return {"error": "output_path is required"}

        if not template and not template_path:
            return {
                "error": "Exactly one of 'template' or 'template_path' must be provided"
            }

        if template and template_path:
            return {
                "error": "Provide either 'template' or 'template_path', not both"
            }

        # Determine template source
        template_source = "inline" if template else "file"

        # Load template from file if template_path provided
        if template_path:
            try:
                path = Path(template_path)
                if not path.is_absolute():
                    path = self._base_path / path
                if not path.exists():
                    return {
                        "error": f"Template file not found: {template_path}",
                        "template_path": template_path
                    }
                template = path.read_text()
            except IOError as e:
                return {
                    "error": f"Failed to read template: {e}",
                    "template_path": template_path
                }

        # Check if output path already exists
        out_path = Path(output_path)
        if not out_path.is_absolute():
            out_path = self._base_path / out_path

        if out_path.exists() and not overwrite:
            return {
                "error": f"Output file already exists: {output_path}. Set overwrite=true to replace.",
                "output_path": str(out_path)
            }

        # Perform simple {{variable}} substitution
        rendered = template
        variables_used = []
        undefined_vars = []

        # Find all {{variable}} patterns
        var_pattern = re.compile(r'\{\{(\w+)\}\}')
        matches = var_pattern.findall(template)
        unique_vars = set(matches)

        for var_name in unique_vars:
            if var_name in variables:
                # Replace all occurrences of {{var_name}} with the value
                pattern = r'\{\{' + re.escape(var_name) + r'\}\}'
                rendered = re.sub(pattern, str(variables[var_name]), rendered)
                variables_used.append(var_name)
            else:
                undefined_vars.append(var_name)

        # Handle escaped {{ (literal \{{)
        rendered = rendered.replace(r'\{{', '{{')

        # Check for undefined variables
        if undefined_vars:
            return {
                "error": f"Undefined variable: {undefined_vars[0]}",
                "template_path": template_path if template_path else None,
                "variables_provided": list(variables.keys())
            }

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
            out_path.write_text(rendered)
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

        self._trace(f"renderTemplateToFile: wrote {bytes_written} bytes to {out_path}")

        return {
            "success": True,
            "output_path": str(out_path),
            "bytes_written": bytes_written,
            "variables_used": sorted(variables_used),
            "template_source": template_source
        }


def create_plugin() -> TemplatePlugin:
    """Factory function to create the template plugin instance."""
    return TemplatePlugin()
