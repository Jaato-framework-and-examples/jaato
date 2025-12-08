"""Terminal UI utilities for console output formatting.

Provides reusable primitives for ANSI colors, text wrapping, and progress bars.
"""

import shutil
import textwrap
from typing import Optional


class TerminalUI:
    """Utility class for terminal formatting operations."""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'green': '\033[32m',
        'cyan': '\033[36m',
        'red': '\033[31m',
        'yellow': '\033[33m',
    }

    def colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text.

        Args:
            text: The text to colorize.
            color: Color name from COLORS dict.

        Returns:
            Text wrapped with ANSI color codes, or original text if color not found.
        """
        code = self.COLORS.get(color, '')
        return f"{code}{text}{self.COLORS['reset']}" if code else text

    def get_terminal_width(self) -> int:
        """Get current terminal width with margin.

        Returns:
            Terminal width minus margin, minimum 40 columns.
        """
        terminal_width = shutil.get_terminal_size().columns
        return max(40, terminal_width - 2)

    def wrap_text(
        self,
        text: str,
        prefix: str = "",
        initial_prefix: Optional[str] = None,
        width: Optional[int] = None
    ) -> str:
        """Wrap text to fit terminal width with word boundaries.

        Args:
            text: The text to wrap.
            prefix: Prefix for continuation lines (e.g., spaces for indentation).
            initial_prefix: Prefix for first line (defaults to prefix if not specified).
            width: Custom width (defaults to terminal width).

        Returns:
            Word-wrapped text that fits the terminal width.
        """
        if width is None:
            width = self.get_terminal_width()

        if initial_prefix is None:
            initial_prefix = prefix

        # Handle multi-paragraph text by wrapping each paragraph separately
        paragraphs = text.split('\n')
        wrapped_paragraphs = []

        for i, para in enumerate(paragraphs):
            if not para.strip():
                # Preserve empty lines
                wrapped_paragraphs.append('')
                continue

            # Use initial_prefix only for the very first paragraph
            if i == 0:
                wrapper = textwrap.TextWrapper(
                    width=width,
                    initial_indent=initial_prefix,
                    subsequent_indent=prefix,
                    break_long_words=True,
                    break_on_hyphens=True,
                )
            else:
                wrapper = textwrap.TextWrapper(
                    width=width,
                    initial_indent=prefix,
                    subsequent_indent=prefix,
                    break_long_words=True,
                    break_on_hyphens=True,
                )
            wrapped_paragraphs.append(wrapper.fill(para))

        return '\n'.join(wrapped_paragraphs)

    def progress_bar(
        self,
        percent: float,
        width: int = 40,
        filled_char: str = '█',
        empty_char: str = '░'
    ) -> str:
        """Generate a text progress bar.

        Args:
            percent: Progress percentage (0-100).
            width: Bar width in characters.
            filled_char: Character for filled portion.
            empty_char: Character for empty portion.

        Returns:
            Progress bar string like [████░░░░░░] 40%
        """
        filled = int(width * percent / 100)
        bar = filled_char * filled + empty_char * (width - filled)
        return f"[{bar}] {percent:.0f}%"

    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix.

        Args:
            text: Text to truncate.
            max_length: Maximum length including suffix.
            suffix: Suffix to append when truncating.

        Returns:
            Truncated text or original if within limit.
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
