"""Vision capture plugin for TUI screenshots.

This plugin captures the TUI state as images (PNG/SVG) for debugging
and AI vision analysis.

Components:
- VisionCapture: Core capture utility using Rich Console recording
- VisionCaptureFormatter: Pipeline plugin that observes output and triggers captures
"""

from .protocol import (
    CaptureConfig,
    CaptureContext,
    CaptureFormat,
    CaptureResult,
    VisionCapturePlugin,
)
from .capture import VisionCapture, create_plugin
from .formatter import VisionCaptureFormatter, create_formatter

__all__ = [
    # Protocol types
    "CaptureConfig",
    "CaptureContext",
    "CaptureFormat",
    "CaptureResult",
    "VisionCapturePlugin",
    # Core implementation
    "VisionCapture",
    "create_plugin",
    # Formatter pipeline plugin
    "VisionCaptureFormatter",
    "create_formatter",
]
