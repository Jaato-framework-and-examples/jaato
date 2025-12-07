"""Multimodal plugin for image handling via @file references.

This plugin enables multimodal interactions by:
1. Detecting @file.ext references to image files in prompts
2. Enriching prompts with viewImage tool availability info
3. Providing a viewImage tool that returns images as multimodal function responses

Requires Gemini 3 Pro or later for multimodal function responses.
"""

from .plugin import MultimodalPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'MultimodalPlugin',
    'create_plugin',
]
