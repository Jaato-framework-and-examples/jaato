# shared/plugins/notebook_output_formatter/plugin.py
"""Notebook output formatter plugin.

Transforms <notebook-cell> markers into a structured 2-column table layout
for Jupyter-style notebook cell presentation.

The output format is a borderless table where:
- Column 1: Cell label (In[n]:, Out[n]:, Err[n]:, etc.)
- Column 2: Cell content (code with fences, or plain text)

The borders are presentation layer - clients decide how to render them.
Code content keeps its fenced blocks so code_block_formatter can highlight.

Detection patterns:
    <notebook-cell type="input" exec="3">
    ```python
    code here
    ```
    </notebook-cell>

Output format (conceptual 2-column table):
    | In [3]:  | ```python                    |
    |          | code here                    |
    |          | ```                          |
    | Out [3]: | result here                  |

Usage:
    from shared.plugins.formatter_pipeline import create_pipeline
    from shared.plugins.notebook_output_formatter import create_plugin

    pipeline = create_pipeline()
    pipeline.register(create_plugin())  # priority 22
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple


# Priority for pipeline ordering (before code_block_formatter at 40)
# After hidden_content_filter (10) and diff_formatter (20)
DEFAULT_PRIORITY = 22

# Pattern to match notebook cell markers
NOTEBOOK_CELL_PATTERN = re.compile(
    r'<notebook-cell\s+type="([^"]+)"\s+exec="(\d+)">\s*\n?(.*?)\n?</notebook-cell>',
    re.DOTALL
)

# Cell type to label mapping
CELL_LABELS = {
    "input": "In",
    "stdout": "",       # No label for stdout, just show output
    "stderr": "Err",
    "result": "Out",
    "display": "Out",
    "error": "Err",
}


class NotebookOutputFormatterPlugin:
    """Plugin that formats notebook cells into a structured table layout.

    Implements the FormatterPlugin protocol for use in a formatter pipeline.
    Detects <notebook-cell> markers and renders them as a 2-column table
    without borders (borders are client's responsibility).

    Features:
    - Transforms notebook cell markers into table rows
    - Preserves code fences for syntax highlighting by code_block_formatter
    - Supports input, output, error, stdout, stderr, and display cells
    - Handles multi-line content properly
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._buffer: str = ""
        self._in_cell = False

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "notebook_output_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (22 = after diff, before table/code formatting)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, transforming notebook cell markers.

        Buffers content until complete cell markers are found, then
        transforms them into table format.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Formatted output when appropriate.
        """
        self._buffer += chunk

        # Process complete cells in the buffer
        while True:
            # Look for a complete notebook-cell in the buffer
            match = NOTEBOOK_CELL_PATTERN.search(self._buffer)
            if not match:
                # Check if we might have an incomplete cell marker
                if "<notebook-cell" in self._buffer:
                    # Check if we have the closing tag
                    start_idx = self._buffer.find("<notebook-cell")
                    remaining = self._buffer[start_idx:]
                    if "</notebook-cell>" not in remaining:
                        # Incomplete cell, yield content before the marker
                        if start_idx > 0:
                            yield self._buffer[:start_idx]
                            self._buffer = self._buffer[start_idx:]
                        return
                # No notebook cells, yield everything
                if self._buffer:
                    yield self._buffer
                    self._buffer = ""
                return

            # Found a complete cell
            start, end = match.span()

            # Yield content before the cell
            if start > 0:
                yield self._buffer[:start]

            # Transform the cell
            cell_type = match.group(1)
            exec_count = match.group(2)
            content = match.group(3)

            formatted = self._format_cell(cell_type, exec_count, content)
            yield formatted

            # Update buffer to remaining content
            self._buffer = self._buffer[end:]

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content."""
        if self._buffer:
            # Try one more time to process any complete cells
            for output in self.process_chunk(""):
                yield output
            # Yield any remaining content as-is
            if self._buffer:
                yield self._buffer
                self._buffer = ""

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_cell = False

    # ==================== Cell Formatting ====================

    def _format_cell(self, cell_type: str, exec_count: str, content: str) -> str:
        """Format a notebook cell with semantic markers for client rendering.

        The output preserves semantic structure using <nb-row> markers that
        the client can interpret to render as a 2-column table:
        - Column 1: Label (In[n]:, Out[n]:, etc.)
        - Column 2: Content (code fences preserved for syntax highlighting)

        Code fences flow through to code_block_formatter for highlighting.
        The client adds borders and table styling based on the markers.

        Args:
            cell_type: Type of cell (input, result, error, etc.)
            exec_count: Execution count for the cell
            content: Cell content (may include code fences)

        Returns:
            Formatted cell string with semantic markers.
        """
        label = self._get_label(cell_type, exec_count)
        content = content.strip()

        # Wrap in semantic markers for client-side table rendering:
        # <nb-row type="input" label="In [1]:">
        # ```python
        # code
        # ```
        # </nb-row>
        #
        # The client interprets these as table rows:
        # | In [1]: | <highlighted code> |

        return f'<nb-row type="{cell_type}" label="{label}">\n{content}\n</nb-row>\n'

    def _get_label(self, cell_type: str, exec_count: str) -> str:
        """Get the label for a cell type.

        Args:
            cell_type: Type of cell
            exec_count: Execution count

        Returns:
            Label string like "In [3]:" or "Out [3]:"
        """
        prefix = CELL_LABELS.get(cell_type, "")
        if not prefix:
            # No label for stdout - just show content
            return ""
        return f"{prefix} [{exec_count}]:"

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - priority: Pipeline priority (default: 22)
        """
        config = config or {}
        self._priority = config.get("priority", DEFAULT_PRIORITY)

    def set_console_width(self, width: int) -> None:
        """Update console width for rendering.

        Args:
            width: Terminal width in columns.
        """
        # Not used for this formatter
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass


def create_plugin() -> NotebookOutputFormatterPlugin:
    """Factory function to create a NotebookOutputFormatterPlugin instance."""
    return NotebookOutputFormatterPlugin()
