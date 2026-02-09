# shared/plugins/mermaid_formatter/__init__.py
"""Mermaid diagram formatter plugin.

Detects ```mermaid code blocks in model output and renders them as
inline terminal graphics using the best available protocol (kitty,
iTerm2, sixel). When no image protocol is available, falls back to
showing truncated mermaid source with a path to the rendered PNG.
"""

from .plugin import MermaidFormatterPlugin, create_plugin

__all__ = ["MermaidFormatterPlugin", "create_plugin"]
