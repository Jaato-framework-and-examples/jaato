"""Template rendering plugin for code generation.

Provides Jinja2-based template rendering with variable substitution,
conditionals, and loops for generating files from templates.
"""

PLUGIN_KIND = "tool"

from .plugin import TemplatePlugin

def create_plugin() -> TemplatePlugin:
    """Factory function to create the template plugin instance."""
    return TemplatePlugin()

__all__ = ['TemplatePlugin', 'create_plugin', 'PLUGIN_KIND']
