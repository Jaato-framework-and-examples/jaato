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
from shared.plugins.formatter_pipeline.pipeline import PRERENDERED_LINE_PREFIX

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

        # Turn feedback for model self-correction
        self._turn_feedback: Optional[str] = None

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
        self._turn_feedback = None

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
        # JAATO_VISION_DIR env var takes priority; otherwise set_workspace_path()
        # will resolve to <workspace>/.jaato/vision/ when called by the pipeline.
        env_vision_dir = os.environ.get("JAATO_VISION_DIR")
        if env_vision_dir:
            self._artifact_dir = env_vision_dir
        # else: remains None until set_workspace_path() is called

    def set_console_width(self, width: int) -> None:
        """Update console width for rendering."""
        self._console_width = max(20, width)

    def set_workspace_path(self, path: str) -> None:
        """Set workspace path for artifact directory resolution.

        Resolves to <workspace>/.jaato/vision/ unless JAATO_VISION_DIR
        env var is set (env var takes priority).
        """
        if not os.environ.get("JAATO_VISION_DIR"):
            self._artifact_dir = os.path.join(path, ".jaato", "vision")

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
            "ER diagrams), prefer using mermaid syntax over ASCII art. "
            "If a mermaid block has a syntax error, the error will be shown "
            "inline — fix the diagram and re-emit it."
        )

    def get_turn_feedback(self) -> Optional[str]:
        """Return feedback about syntax errors detected this turn.

        Called by the pipeline after flush(). Returns and clears any
        accumulated feedback so the model can self-correct on the next turn.
        """
        feedback = self._turn_feedback
        self._turn_feedback = None
        return feedback

    def _render_diagram(self, source: str) -> str:
        """Render a mermaid diagram and return terminal output.

        Tries to render via mmdc/kroki → PNG → terminal backend.
        On syntax error, shows the source with the diagnostic appended
        (matching code_validation_formatter visual style).
        When no renderer is available, shows a passthrough code block.

        Args:
            source: Mermaid diagram source text.

        Returns:
            Formatted string ready for terminal display.
        """
        source = source.strip()
        if not source:
            return ""

        result = renderer.render(
            source,
            theme=self._theme,
            scale=self._scale,
            background=self._background,
        )

        if result.png is not None:
            # Save artifact if directory configured
            artifact_path = self._save_artifact(result.png)

            render_width = max(20, self._console_width)

            # Select backend and render for terminal
            backend = select_backend(max_width=render_width)
            rendered = backend.render(result.png, max_width=render_width)

            # Add artifact path reference
            if artifact_path:
                rendered += f"    \x1b[2m[saved: {artifact_path}]\x1b[0m\n"

            # Prefix each non-empty rendered line so the output buffer
            # skips wrapping (preserves pixel-aligned half-block art)
            lines = rendered.split('\n')
            rendered = '\n'.join(
                PRERENDERED_LINE_PREFIX + line if line.strip() else line
                for line in lines
            )

            return "\n" + rendered

        if result.error is not None:
            # Syntax error — show source with diagnostic appended
            # Store feedback for model self-correction on next turn
            self._turn_feedback = (
                "[Mermaid Validation Feedback]\n"
                "A mermaid diagram you produced has a syntax error:\n"
                f"{result.error}\n\n"
                "Please fix the diagram and re-emit it."
            )
            return self._fallback_with_diagnostic(source, result.error)

        # No renderer available — passthrough code block
        return self._fallback_code_block(source)

    def _fallback_with_diagnostic(self, source: str, error: str) -> str:
        """Show truncated mermaid source around the error with a diagnostic.

        Parses the error for a line number and shows a context window
        of CONTEXT_RADIUS lines above/below. When no line number is
        found, shows the first few lines.  The diagnostic box follows
        the code_validation_formatter visual style.
        """
        CONTEXT_RADIUS = 2
        MAX_LINES_NO_LINENUM = 5
        source_lines = source.splitlines()

        line_match = re.search(r'[Ll]ine\s+(\d+)', error)

        if line_match:
            error_idx = int(line_match.group(1)) - 1  # 0-indexed
            error_idx = max(0, min(error_idx, len(source_lines) - 1))

            start = max(0, error_idx - CONTEXT_RADIUS)
            end = min(len(source_lines), error_idx + CONTEXT_RADIUS + 1)

            parts = []
            if start > 0:
                parts.append(f"  ... ({start} lines above)")
            for i in range(start, end):
                marker = f"       \u2190 line {i + 1}" if i == error_idx else ""
                parts.append(f"{source_lines[i]}{marker}")
            if end < len(source_lines):
                parts.append(f"  ... ({len(source_lines) - end} lines below)")
            truncated = "\n".join(parts)
        elif len(source_lines) > MAX_LINES_NO_LINENUM:
            head = "\n".join(source_lines[:MAX_LINES_NO_LINENUM])
            truncated = f"{head}\n  ... ({len(source_lines) - MAX_LINES_NO_LINENUM} more lines)"
        else:
            truncated = source

        block = f"```mermaid\n{truncated}\n```\n"
        indent = "    \u2502 "
        diag = ["    \u250c\u2500 Mermaid Validation \u2500"]
        for err_line in error.splitlines():
            diag.append(f"{indent}{err_line}")
        diag.append("    \u2514\u2500")
        return block + "\n".join(diag) + "\n"

    def _fallback_code_block(self, source: str) -> str:
        """Show the source as a passthrough code block.

        Used when no renderer is available at all (not installed / unreachable).
        """
        hint = (
            "\x1b[2m[mermaid diagram - rendering unavailable; "
            "install @mermaid-js/mermaid-cli or set JAATO_KROKI_URL]\x1b[0m\n"
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
