# shared/plugins/vision_capture/__init__.py
"""Vision capture plugin for TUI screenshots.

This plugin captures the TUI state as images (PNG/SVG) for debugging
and AI vision analysis. Captures can be triggered manually via the
`screenshot` command or automatically on turn/tool completion.

Components:
- VisionCapture: Core capture utility using Rich Console recording
- VisionCaptureFormatter: Pipeline plugin that observes output and triggers captures

Example (manual capture):
    from shared.plugins.vision_capture import create_plugin, CaptureConfig

    vision = create_plugin()

    from rich.panel import Panel
    panel = Panel("Hello, World!")
    result = vision.capture(panel)

    if result.success:
        message = result.to_system_message()
        # <tui-screenshot path="/tmp/..." format="png" context="user_requested" .../>

Example (pipeline integration):
    from shared.plugins.vision_capture import create_formatter, CaptureContext

    def on_capture(context: CaptureContext, turn_index: int):
        # Perform actual capture using VisionCapture
        result = vision.capture(output_buffer.render_panel())
        ...

    formatter = create_formatter(
        capture_callback=on_capture,
        auto_capture_on_turn_end=True,
    )
    pipeline.register(formatter)
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
