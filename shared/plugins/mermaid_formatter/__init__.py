# shared/plugins/mermaid_formatter/__init__.py
"""Mermaid diagram formatter plugin.

Detects ```mermaid code blocks in model output and renders them as
inline terminal graphics using the best available protocol, or as
Unicode half-block art as a universal fallback.
"""

from .plugin import MermaidFormatterPlugin, create_plugin

__all__ = ["MermaidFormatterPlugin", "create_plugin"]
