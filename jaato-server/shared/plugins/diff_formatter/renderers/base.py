# shared/plugins/diff_formatter/renderers/base.py
"""Base protocol and types for diff renderers."""

from dataclasses import dataclass
from typing import Protocol

from ..parser import ParsedDiff


@dataclass
class ColorScheme:
    """Color scheme for diff rendering.

    All values are ANSI escape codes.
    """
    reset: str = "\033[0m"
    dim: str = "\033[2m"
    bold: str = "\033[1m"

    # Diff-specific colors
    added: str = "\033[102;30m"       # Pale green background for additions
    deleted: str = "\033[101;30m"     # Pale red background for deletions
    added_bold: str = "\033[102;30m"  # Pale green background for word-level highlight
    deleted_bold: str = "\033[101;30m"  # Pale red background for word-level highlight

    # Structure colors
    header_path: str = "\033[1;36m"   # Bold cyan for file path
    header_old: str = "\033[2;31m"    # Dim red for "OLD" header
    header_new: str = "\033[2;32m"    # Dim green for "NEW" header
    hunk_header: str = "\033[36m"     # Cyan for @@ lines
    line_numbers: str = "\033[2m"     # Dim for line numbers
    box: str = "\033[2m"              # Dim for box drawing
    stats: str = "\033[2m"            # Dim for summary stats


# Default color scheme
DEFAULT_COLOR_SCHEME = ColorScheme()

# No-color scheme for testing or non-ANSI output
NO_COLOR_SCHEME = ColorScheme(
    reset="",
    dim="",
    bold="",
    added="",
    deleted="",
    added_bold="",
    deleted_bold="",
    header_path="",
    header_old="",
    header_new="",
    hunk_header="",
    line_numbers="",
    box="",
    stats="",
)


class DiffRenderer(Protocol):
    """Protocol for diff rendering strategies.

    Each renderer handles a specific terminal width range and
    produces formatted output appropriate for that width.
    """

    def min_width(self) -> int:
        """Minimum terminal width for this renderer.

        Returns:
            Minimum width in columns. Renderer will be skipped
            if terminal is narrower than this.
        """
        ...

    def render(self, diff: ParsedDiff, width: int, colors: ColorScheme) -> str:
        """Render the diff to a formatted string.

        Args:
            diff: Parsed diff structure.
            width: Current terminal width.
            colors: Color scheme to use.

        Returns:
            Formatted diff string with ANSI colors.
        """
        ...
