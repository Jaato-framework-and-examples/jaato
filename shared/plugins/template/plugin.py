"""Template rendering plugin implementation.

Provides tools for rendering Jinja2 templates with variable substitution
and writing the results to files.

NOTE: This is a stub implementation. Full implementation pending.
See docs/template-tool-design.md for the design specification.
"""

from typing import Any, Callable, Dict, List, Optional

from ..base import UserCommand
from ..model_provider.types import ToolSchema


class TemplatePlugin:
    """Plugin for template-based file generation.

    Tools provided:
    - renderTemplate: Render a template with variables and write to file

    This plugin uses Jinja2 syntax for templates:
    - Variables: {{ variable_name }}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in items %}...{% endfor %}
    """

    def __init__(self):
        self._initialized = False
        self._agent_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "template"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the template plugin."""
        config = config or {}
        self._agent_name = config.get("agent_name")
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for template tools."""
        return [
            ToolSchema(
                name="renderTemplate",
                description=(
                    "Render a template with variable substitution and write the result to a file. "
                    "Templates support Jinja2 syntax: variables ({{name}}), conditionals "
                    "({% if condition %}...{% endif %}), and loops ({% for item in items %}...{% endfor %}). "
                    "Provide either 'template' for inline content or 'template_path' for a file."
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
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor functions for each tool."""
        return {
            "renderTemplate": self._execute_render_template,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for template tools."""
        return """You have access to template rendering tools.

Use `renderTemplate` to generate files from templates with variable substitution.

Templates use Jinja2 syntax:
- Variables: {{ variable_name }}
- Conditionals: {% if condition %}...{% endif %}
- Loops: {% for item in items %}...{% endfor %}
- Filters: {{ name | upper }}, {{ value | default('fallback') }}

Example:
  renderTemplate(
    template="Hello {{ name }}, welcome to {{ project }}!",
    variables={"name": "Alice", "project": "jaato"},
    output_path="greeting.txt"
  )

For reusable templates, check .templates/ directory:
  renderTemplate(
    template_path=".templates/service.java.tmpl",
    variables={"class_name": "OrderService", "package": "com.example.orders"},
    output_path="src/main/java/com/example/orders/OrderService.java"
  )

Template rendering requires approval since it writes files."""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved.

        Template rendering writes files, so none are auto-approved.
        """
        return []

    def get_user_commands(self) -> List[UserCommand]:
        """Template plugin provides model tools only."""
        return []

    def _execute_render_template(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute renderTemplate tool.

        NOTE: Stub implementation - returns error until fully implemented.
        """
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

        # Stub: Return not-implemented error
        return {
            "error": "Template plugin not yet implemented. See docs/template-tool-design.md for design.",
            "status": "stub"
        }
