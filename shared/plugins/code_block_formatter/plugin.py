# shared/plugins/code_block_formatter/plugin.py
"""Streaming code block formatter plugin for syntax highlighting.

This plugin transforms model output text containing markdown code blocks
into ANSI-escaped text with syntax highlighting. It buffers content inside
code blocks until they're complete, while passing through regular text
immediately.

Usage:
    from shared.plugins.code_block_formatter import create_plugin

    formatter = create_plugin()
    formatter.initialize({"theme": "monokai", "line_numbers": True})

    # Streaming mode
    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import re
from typing import Any, Dict, Iterator, List, Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from shared.plugins.table_formatter.plugin import _display_width


# Common language aliases mapping
LANGUAGE_ALIASES = {
    'js': 'javascript',
    'ts': 'typescript',
    'py': 'python',
    'rb': 'ruby',
    'yml': 'yaml',
    'sh': 'bash',
    'shell': 'bash',
    'zsh': 'bash',
    'dockerfile': 'docker',
    'md': 'markdown',
    'cs': 'csharp',
    'c++': 'cpp',
    'objective-c': 'objectivec',
    'ipython3': 'ipython',  # IPython with !shell, %magic support
    'jupyter': 'ipython',  # Jupyter notebooks use IPython kernel
}

# Priority for pipeline ordering (40-59 = syntax highlighting)
DEFAULT_PRIORITY = 40


def _get_content_width(line: str) -> int:
    """Get the display width of actual content in a line (excluding trailing whitespace).

    Args:
        line: The line (may contain ANSI codes).

    Returns:
        Display width of content (excluding trailing spaces).
    """
    if not line:
        return 0

    # Parse ANSI codes to get plain text
    if '\x1b[' in line:
        text = Text.from_ansi(line)
        plain = text.plain
    else:
        plain = line

    # Strip trailing spaces and measure content width
    content = plain.rstrip()
    if not content:
        return 0

    return _display_width(content)


def _trim_line_to_width(line: str, target_width: int) -> str:
    """Trim a line to target_width, preserving ANSI codes and styling.

    This removes trailing background padding while keeping the actual content
    and its styling intact.

    Args:
        line: The line to trim (may contain ANSI codes).
        target_width: The width to trim to.

    Returns:
        The line trimmed to target_width, as an ANSI string.
    """
    if not line:
        return ""

    # Parse ANSI codes to get styled text
    if '\x1b[' in line:
        text = Text.from_ansi(line)
    else:
        text = Text(line)

    plain = text.plain

    if target_width <= 0:
        return ""

    # Find the character index that corresponds to target_width display columns
    current_width = 0
    char_count = 0
    for char in plain:
        char_width = _display_width(char)
        if current_width + char_width > target_width:
            break
        current_width += char_width
        char_count += 1

    # Slice the styled text to preserve styling
    if char_count > 0:
        result = text[:char_count]
    else:
        result = Text()

    # Convert back to ANSI string
    console = Console(force_terminal=True, no_color=False, highlight=False)
    with console.capture() as capture:
        console.print(result, end="")
    return capture.get()


class CodeBlockFormatterPlugin:
    """Streaming plugin that formats code blocks with syntax highlighting.

    Implements the FormatterPlugin protocol. Buffers content inside code
    blocks (```...```) until complete, passes through other text immediately.
    """

    def __init__(self):
        self._theme = "monokai"
        self._line_numbers = False
        self._word_wrap = True
        self._background_color: Optional[str] = None
        self._console_width = 80
        self._priority = DEFAULT_PRIORITY

        # Streaming state
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "code_block_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (40 = syntax highlighting range)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering code blocks, passing through text.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Output chunks - immediate for regular text, formatted for complete code blocks.
        """
        self._buffer += chunk

        while self._buffer:
            if not self._in_code_block:
                # Look for code block start: ```lang or ```
                match = re.search(r'```(\w*)\n', self._buffer)
                if match:
                    # Yield text before the code block
                    before = self._buffer[:match.start()]
                    if before:
                        yield before

                    # Enter code block mode
                    self._code_block_lang = match.group(1) or "text"
                    self._buffer = self._buffer[match.end():]
                    self._in_code_block = True
                else:
                    # Check if we might have a partial code block start at the end
                    # This includes: `, ``, ```, ```lang (without trailing \n)
                    partial_match = re.search(r'`{1,3}\w*$', self._buffer)
                    if partial_match:
                        # Hold back the potential code block start
                        to_yield = self._buffer[:partial_match.start()]
                        self._buffer = self._buffer[partial_match.start():]
                        if to_yield:
                            yield to_yield
                        return
                    # No code block start, yield everything
                    yield self._buffer
                    self._buffer = ""
            else:
                # In code block, look for closing ```
                end_match = re.search(r'\n```', self._buffer)
                if end_match:
                    # Extract code block content
                    code = self._buffer[:end_match.start()]

                    # Format and yield the complete code block
                    formatted = self._render_code_block(code, self._code_block_lang)
                    yield formatted

                    # Exit code block mode, continue with remaining text
                    self._buffer = self._buffer[end_match.end():]
                    self._in_code_block = False
                    self._code_block_lang = ""
                else:
                    # Code block not complete yet, keep buffering
                    return

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        Yields:
            Any remaining content, formatted if it was a code block.
        """
        if self._buffer:
            if self._in_code_block:
                # Incomplete code block - format what we have
                formatted = self._render_code_block(self._buffer, self._code_block_lang)
                yield formatted
            else:
                # Regular text
                yield self._buffer
            self._buffer = ""
            self._in_code_block = False
            self._code_block_lang = ""

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - theme: Syntax highlighting theme (default: "monokai")
                - line_numbers: Show line numbers (default: False)
                - word_wrap: Wrap long lines (default: True)
                - background_color: Background color or None (default: None)
                - console_width: Width for rendering (default: 80)
                - priority: Pipeline priority (default: 40)
        """
        config = config or {}
        self._theme = config.get("theme", "monokai")
        self._line_numbers = config.get("line_numbers", False)
        self._word_wrap = config.get("word_wrap", True)
        self._background_color = config.get("background_color", None)
        self._console_width = config.get("console_width", 80)
        self._priority = config.get("priority", DEFAULT_PRIORITY)

    def set_console_width(self, width: int) -> None:
        """Update the console width for rendering."""
        self._console_width = max(20, width)

    def set_syntax_theme(self, theme_name: str) -> None:
        """Set the syntax highlighting theme based on UI theme.

        Maps UI theme names to appropriate Pygments syntax themes.

        Args:
            theme_name: UI theme name ("dark", "light", "high-contrast", etc.)
        """
        # Map UI themes to Pygments syntax themes
        theme_mapping = {
            "dark": "monokai",
            "light": "solarized-light",  # Light background, good contrast
            "high-contrast": "native",  # High contrast dark theme
        }
        self._theme = theme_mapping.get(theme_name, "monokai")

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self.reset()

    # ==================== Internal Methods ====================

    def _render_code_block(self, code: str, language: str) -> str:
        """Render a code block with syntax highlighting and block indent.

        Args:
            code: The code content (without ``` markers).
            language: The language identifier.

        Returns:
            ANSI-escaped string with syntax highlighting, indented as a block.
        """
        # Map language aliases
        lang = LANGUAGE_ALIASES.get(language.lower(), language.lower())

        # Block indent for visual distinction
        indent = "    "  # 4 spaces

        try:
            # Calculate natural width from raw code to prevent wrapping
            code_lines = code.split('\n')
            max_code_width = max((_display_width(line) for line in code_lines), default=0)

            # Account for line numbers if enabled (Rich format: " NUM │ ")
            if self._line_numbers and code_lines:
                num_lines = len(code_lines)
                line_number_width = len(str(num_lines)) + 4  # padding + separator
                max_code_width += line_number_width

            # Use content width as console width (prevents wrapping)
            render_width = max(40, max_code_width + 1)

            syntax = Syntax(
                code,
                lang,
                theme=self._theme,
                line_numbers=self._line_numbers,
                word_wrap=self._word_wrap,
                background_color=self._background_color,
            )

            # Render to ANSI string using a temporary console
            console = Console(
                width=render_width,
                force_terminal=True,
                no_color=False,
                highlight=False,
            )
            with console.capture() as capture:
                console.print(syntax, end="")

            rendered = capture.get()
            lines = rendered.split('\n')

            # Find the natural width of content (max content width across all lines)
            # This ensures background styling only extends to the widest content line
            natural_width = max((_get_content_width(line) for line in lines), default=0)

            # Trim each line to the natural width, then add indent
            trimmed_lines = []
            for line in lines:
                trimmed = _trim_line_to_width(line, natural_width)
                trimmed_lines.append(indent + trimmed)

            return '\n' + '\n'.join(trimmed_lines) + '\n'

        except Exception:
            # Fallback: return code as-is with indent if highlighting fails
            indented_lines = [indent + line for line in code.split('\n')]
            return '\n' + '\n'.join(indented_lines) + '\n'


def create_plugin() -> CodeBlockFormatterPlugin:
    """Factory function to create a CodeBlockFormatterPlugin instance."""
    return CodeBlockFormatterPlugin()
