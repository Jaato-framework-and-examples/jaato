# shared/plugins/mermaid_formatter/plugin.py
"""Mermaid diagram formatter plugin.

Detects ```mermaid code blocks in streaming model output, renders them
to PNG via the Mermaid renderer, and outputs them using the best
available terminal graphics backend.

Priority 28: After diff_formatter (20), before code_block_formatter (40).
This ensures mermaid blocks are intercepted before the generic code block
formatter adds syntax highlighting to them.
"""

import logging
import os
import re
from typing import Any, Dict, Iterator, Optional

from . import renderer
from .backends import select_backend

logger = logging.getLogger(__name__)


class MermaidFormatterPlugin:
    """Formatter plugin that renders Mermaid diagrams inline."""

    def __init__(self):
        self._priority = 28
        self._console_width = 80
        self._theme = "default"
        self._scale = 2
        self._background = "white"
        self._enabled = True
        self._artifact_dir: Optional[str] = None
        self._artifact_counter = 0

        # Streaming state
        self._buffer = ""
        self._in_mermaid_block = False

    @property
    def name(self) -> str:
        return "mermaid_formatter"

    @property
    def priority(self) -> int:
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Buffer mermaid blocks, pass through everything else.

        Detects ```mermaid ... ``` patterns in the stream. Content
        before and after mermaid blocks is yielded immediately.
        Mermaid block content is buffered until the closing ``` is seen.
        """
        if not self._enabled:
            yield chunk
            return

        self._buffer += chunk

        while self._buffer:
            if not self._in_mermaid_block:
                # Look for mermaid block start: ```mermaid\n
                match = re.search(r'```mermaid\s*\n', self._buffer)
                if match:
                    # Yield text before the mermaid block
                    before = self._buffer[:match.start()]
                    if before:
                        yield before

                    # Enter mermaid block mode
                    self._buffer = self._buffer[match.end():]
                    self._in_mermaid_block = True
                else:
                    # Check for partial match at end of buffer
                    # Could be: `, ``, ```, ```m, ```me, ```mer, etc.
                    partial = re.search(r'`{1,3}(?:m(?:e(?:r(?:m(?:a(?:i(?:d)?)?)?)?)?)?)?$',
                                        self._buffer)
                    if partial:
                        to_yield = self._buffer[:partial.start()]
                        self._buffer = self._buffer[partial.start():]
                        if to_yield:
                            yield to_yield
                        return  # Wait for more chunks
                    # No mermaid block, yield everything
                    yield self._buffer
                    self._buffer = ""
            else:
                # In mermaid block, look for closing ```
                end_match = re.search(r'\n```', self._buffer)
                if end_match:
                    # Extract mermaid source
                    mermaid_source = self._buffer[:end_match.start()]

                    # Render the diagram
                    rendered = self._render_diagram(mermaid_source)
                    yield rendered

                    # Continue with text after the closing ```
                    self._buffer = self._buffer[end_match.end():]
                    self._in_mermaid_block = False
                else:
                    # Block not complete, keep buffering
                    return

    def flush(self) -> Iterator[str]:
        """Flush any buffered content at turn end."""
        if self._buffer:
            if self._in_mermaid_block:
                # Incomplete mermaid block - try to render what we have
                rendered = self._render_diagram(self._buffer)
                yield rendered
            else:
                yield self._buffer
            self._buffer = ""
            self._in_mermaid_block = False

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_mermaid_block = False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with configuration."""
        config = config or {}
        self._priority = config.get("priority", 28)
        self._console_width = config.get("console_width", 80)
        self._theme = config.get("theme", "default")
        self._scale = config.get("scale", 2)
        self._background = config.get("background", "white")

        # Check env vars
        env_theme = os.environ.get("JAATO_MERMAID_THEME")
        if env_theme:
            self._theme = env_theme

        env_scale = os.environ.get("JAATO_MERMAID_SCALE")
        if env_scale:
            try:
                self._scale = int(env_scale)
            except ValueError:
                pass

        # Check if rendering is disabled
        env_backend = os.environ.get("JAATO_MERMAID_BACKEND", "").lower()
        if env_backend == "off":
            self._enabled = False

        # Artifact directory for saving rendered diagrams
        self._artifact_dir = os.environ.get(
            "JAATO_VISION_DIR", "/tmp/jaato_vision"
        )

    def set_console_width(self, width: int) -> None:
        """Update console width for rendering."""
        self._console_width = max(20, width)

    def get_system_instructions(self) -> Optional[str]:
        """Inform the model about mermaid diagram rendering capability.

        Returns instructions only when rendering is actually available
        (plugin enabled and a renderer is installed). This avoids
        encouraging mermaid usage when diagrams would just pass through
        as raw code blocks.
        """
        if not self._enabled:
            return None

        if not renderer.is_renderer_available():
            return None

        return (
            "The output pipeline renders ```mermaid code blocks as graphical "
            "diagrams in the terminal. When a visual diagram would aid "
            "understanding (architecture, flows, state machines, sequences, "
            "ER diagrams), prefer using mermaid syntax over ASCII art."
        )

    def _render_diagram(self, source: str) -> str:
        """Render a mermaid diagram and return terminal output.

        Tries to render via mmdc/mermaid-py → PNG → terminal backend.
        Falls back to showing the source as a formatted code block
        with a hint about installing mermaid-cli.

        Args:
            source: Mermaid diagram source text.

        Returns:
            Formatted string ready for terminal display.
        """
        source = source.strip()
        if not source:
            return ""

        # Try rendering to PNG
        png_data = renderer.render(
            source,
            theme=self._theme,
            scale=self._scale,
            background=self._background,
        )

        if png_data is not None:
            # Save artifact if directory configured
            artifact_path = self._save_artifact(png_data)

            # Select backend and render for terminal
            backend = select_backend(max_width=self._console_width)
            rendered = backend.render(png_data, max_width=self._console_width)

            # Add artifact path reference
            if artifact_path:
                rendered += f"    \x1b[2m[saved: {artifact_path}]\x1b[0m\n"

            return "\n" + rendered

        # No renderer available - fall back to code block passthrough
        # Return as ```mermaid block so code_block_formatter can handle it
        return self._fallback_code_block(source)

    def _fallback_code_block(self, source: str) -> str:
        """Format source as a code block with an install hint.

        Returns the source wrapped in ```mermaid fences so the
        downstream code_block_formatter will syntax-highlight it.
        Prepends a dim hint about installing mermaid-cli.
        """
        hint = (
            "\x1b[2m[mermaid diagram - install mermaid-cli for rendered output: "
            "npm install -g @mermaid-js/mermaid-cli]\x1b[0m\n"
        )
        return hint + f"```mermaid\n{source}\n```"

    def _save_artifact(self, png_data: bytes) -> Optional[str]:
        """Save rendered PNG to the artifact directory.

        Returns:
            Path to saved file, or None if saving failed.
        """
        if not self._artifact_dir:
            return None

        try:
            os.makedirs(self._artifact_dir, exist_ok=True)
            self._artifact_counter += 1
            filename = f"mermaid_{self._artifact_counter:03d}.png"
            path = os.path.join(self._artifact_dir, filename)
            with open(path, "wb") as f:
                f.write(png_data)
            return path
        except OSError as e:
            logger.debug("Failed to save mermaid artifact: %s", e)
            return None


def create_plugin() -> MermaidFormatterPlugin:
    """Factory function to create the mermaid formatter plugin."""
    return MermaidFormatterPlugin()
