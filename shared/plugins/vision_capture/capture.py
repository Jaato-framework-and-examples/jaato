# shared/plugins/vision_capture/capture.py
"""Vision capture implementation using Rich Console."""

import os
import glob
import time
from datetime import datetime, timedelta
from typing import Optional, Any

from rich.console import Console
from rich.text import Text

from .protocol import (
    CaptureConfig,
    CaptureContext,
    CaptureFormat,
    CaptureResult,
    VisionCapturePlugin,
)


class VisionCapture:
    """Captures TUI state as images using Rich Console recording."""

    def __init__(self):
        self._config: CaptureConfig = CaptureConfig()
        self._last_capture: Optional[CaptureResult] = None
        self._capture_count: int = 0

    @property
    def name(self) -> str:
        return "vision_capture"

    def initialize(self, config: Optional[CaptureConfig] = None) -> None:
        """Initialize with configuration."""
        if config:
            self._config = config

        # Ensure output directory exists
        os.makedirs(self._config.output_dir, exist_ok=True)

    def _generate_filename(self) -> str:
        """Generate a unique filename for the capture."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._capture_count += 1
        ext = self._config.format.value
        return f"capture_{timestamp}_{self._capture_count:04d}.{ext}"

    def _create_console(self) -> Console:
        """Create a recording console with configured dimensions."""
        return Console(
            record=True,
            force_terminal=True,
            width=self._config.width,
            color_system="truecolor",
        )

    def capture(
        self,
        renderable: Any,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult:
        """Capture a Rich renderable to an image file."""
        console = self._create_console()
        timestamp = datetime.now()

        try:
            # Render the content to the recording console
            console.print(renderable)

            # Generate output path
            filename = self._generate_filename()
            path = os.path.join(self._config.output_dir, filename)

            # Export based on format
            if self._config.format == CaptureFormat.SVG:
                console.save_svg(path, title=self._config.title)
            elif self._config.format == CaptureFormat.HTML:
                console.save_html(path)
            elif self._config.format == CaptureFormat.PNG:
                # PNG requires SVG intermediate + cairosvg
                svg_path = path.replace(".png", ".svg")
                console.save_svg(svg_path, title=self._config.title)
                path = self._convert_svg_to_png(svg_path, path)

            result = CaptureResult(
                path=path,
                format=self._config.format,
                timestamp=timestamp,
                context=context,
                width=self._config.width,
                height=self._config.height,
                turn_index=turn_index,
                agent_id=agent_id,
            )
            self._last_capture = result
            return result

        except Exception as e:
            return CaptureResult(
                path="",
                format=self._config.format,
                timestamp=timestamp,
                context=context,
                width=self._config.width,
                height=self._config.height,
                turn_index=turn_index,
                agent_id=agent_id,
                error=str(e),
            )

    def capture_ansi(
        self,
        ansi_text: str,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult:
        """Capture ANSI-escaped text to an image file."""
        # Convert ANSI text to Rich Text object
        text = Text.from_ansi(ansi_text)
        return self.capture(text, context, turn_index, agent_id)

    def _convert_svg_to_png(self, svg_path: str, png_path: str) -> str:
        """Convert SVG to PNG using cairosvg if available."""
        try:
            import cairosvg

            # Read SVG and fix font-family for better Unicode support
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()

            # Replace the default monospace font with fonts that have better Unicode coverage
            # Rich's SVG uses "Fira Code" or similar, but we need fallbacks
            better_fonts = (
                '"DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono", '
                '"Fira Code", "JetBrains Mono", "Source Code Pro", '
                '"Cascadia Code", monospace'
            )
            # Rich SVG template uses font-family in CSS and inline styles
            svg_content = svg_content.replace(
                'font-family: ',
                f'font-family: {better_fonts}, '
            )

            # Write fixed SVG
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            cairosvg.svg2png(url=svg_path, write_to=png_path)
            # Remove intermediate SVG
            os.remove(svg_path)
            return png_path
        except ImportError:
            # cairosvg not available, fall back to SVG
            # Rename .png back to .svg
            actual_path = svg_path
            return actual_path

    def get_last_capture(self) -> Optional[CaptureResult]:
        """Get the most recent capture result."""
        return self._last_capture

    def cleanup_old_captures(self) -> int:
        """Remove captures older than configured threshold."""
        if not os.path.exists(self._config.output_dir):
            return 0

        cutoff = datetime.now() - timedelta(hours=self._config.auto_cleanup_hours)
        removed = 0

        for pattern in ["*.svg", "*.png", "*.html"]:
            for path in glob.glob(os.path.join(self._config.output_dir, pattern)):
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(path))
                    if mtime < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass

        return removed


def create_plugin(config: Optional[CaptureConfig] = None) -> VisionCapture:
    """Create and initialize a vision capture plugin."""
    plugin = VisionCapture()
    plugin.initialize(config)
    return plugin
