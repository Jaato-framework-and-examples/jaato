# shared/plugins/vision_capture/capture.py
"""Vision capture implementation using Rich Console."""

import os
import glob
import stat
import time
from datetime import datetime, timedelta
from typing import Optional, Any

from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from .protocol import (
    CaptureConfig,
    CaptureContext,
    CaptureFormat,
    CaptureResult,
    VisionCapturePlugin,
)

try:
    # Canonical implementation.  jaato-tui declares no dependency on
    # jaato-server, so this import is best-effort in a standalone TUI install;
    # _refuse_preplanted_dir below enforces the same refusals when it is
    # absent.  Keep the two in step.
    from shared.plugins.path_safety import ensure_private_dir
except ImportError:  # pragma: no cover - TUI installed without jaato-server
    ensure_private_dir = None


def _refuse_preplanted_dir(path: str) -> None:
    """Create or adopt ``path``, refusing a directory someone else planted.

    Fallback for :func:`shared.plugins.path_safety.ensure_private_dir` when
    that module is not importable.  The capture directory defaults to a
    predictable location under a shared, world-writable ``/tmp``, which is the
    classic pre-planting target: another user creates it first, as a symlink
    to somewhere of their choosing or as a directory they can read, and every
    screenshot afterwards lands where they decided.  ``makedirs(exist_ok=True)``
    adopts both without complaint.

    Args:
        path: Directory to create or adopt.

    Raises:
        OSError: if the path is a symlink, is not a directory, or is owned by
            another user.
    """
    try:
        os.makedirs(path, mode=0o700, exist_ok=False)
        return
    except FileExistsError:
        pass

    st = os.lstat(path)  # lstat: a symlink must be seen as a symlink
    if stat.S_ISLNK(st.st_mode):
        raise OSError(f"Refusing pre-planted symlink as capture directory: {path}")
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"Refusing non-directory as capture directory: {path}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise OSError(
            f"Refusing capture directory owned by another user (uid {st.st_uid}): {path}"
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
        """Initialize with configuration.

        The output directory is created owner-only and refused outright if
        something is already sitting at that path that we did not put there —
        see :func:`_refuse_preplanted_dir` for why that matters at the
        shared-``/tmp`` default.

        Args:
            config: Capture configuration; the existing config is kept if None.

        Raises:
            OSError: if the configured output directory cannot be used safely.
        """
        if config:
            self._config = config

        # Ensure output directory exists — and is ours.
        if ensure_private_dir is not None:
            ensure_private_dir(self._config.output_dir)
        else:  # pragma: no cover - TUI installed without jaato-server
            _refuse_preplanted_dir(self._config.output_dir)

    def _generate_filename(self) -> str:
        """Generate a unique filename for the capture."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._capture_count += 1
        ext = self._config.format.value
        return f"capture_{timestamp}_{self._capture_count:04d}.{ext}"

    def _create_console(self) -> Console:
        """Create a recording console with configured dimensions.

        Uses a StringIO buffer to prevent output to the actual terminal,
        which would corrupt the TUI display.
        """
        import io
        return Console(
            record=True,
            force_terminal=True,
            width=self._config.width,
            color_system="truecolor",
            file=io.StringIO(),  # Don't output to terminal
        )

    def capture(
        self,
        renderable: Any,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
        terminal_theme: Optional[TerminalTheme] = None,
    ) -> CaptureResult:
        """Capture a Rich renderable to an image file.

        Args:
            renderable: Rich renderable to capture.
            context: What triggered the capture.
            turn_index: Current turn index.
            agent_id: Agent identifier.
            terminal_theme: Optional Rich TerminalTheme for export styling.
        """
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
                console.save_svg(path, title=self._config.title, theme=terminal_theme)
            elif self._config.format == CaptureFormat.HTML:
                console.save_html(path)
            elif self._config.format == CaptureFormat.PNG:
                # PNG requires SVG intermediate + cairosvg
                svg_path = path.replace(".png", ".svg")
                console.save_svg(svg_path, title=self._config.title, theme=terminal_theme)
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
        except (ImportError, OSError):
            # cairosvg not available or native Cairo library not found.
            # On Windows, cairocffi raises OSError when libcairo DLL is missing.
            # Fall back to SVG.
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
