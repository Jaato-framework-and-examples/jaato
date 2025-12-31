# shared/plugins/code_block_formatter/plugin.py
"""Code block formatter plugin for syntax highlighting code blocks.

This plugin transforms model output text containing markdown code blocks
into ANSI-escaped text with syntax highlighting. It can be used by any
client that wants to render code with colors.

Usage:
    from shared.plugins.code_block_formatter import create_plugin

    formatter = create_plugin()
    formatter.initialize({"theme": "monokai", "line_numbers": False})

    # Transform text with code blocks
    formatted = formatter.format_output(text)
    # formatted now contains ANSI escape codes for syntax highlighting
"""

import re
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text


# Regex pattern for detecting code blocks with optional language specifier
# Matches: ```lang\ncode\n``` or ```\ncode\n```
CODE_BLOCK_PATTERN = re.compile(
    r'```(\w*)\n(.*?)```',
    re.DOTALL
)

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
}


class CodeBlockFormatterPlugin:
    """Plugin that formats model output with syntax highlighting for code blocks.

    This is a client-side formatter that transforms text containing markdown
    code blocks into ANSI-escaped text with syntax highlighting.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "code_block_formatter"

    def __init__(self):
        self._theme = "monokai"
        self._line_numbers = False
        self._word_wrap = True
        self._background_color: Optional[str] = None  # None = no background
        self._console_width = 80

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - theme: Syntax highlighting theme (default: "monokai")
                - line_numbers: Show line numbers (default: False)
                - word_wrap: Wrap long lines (default: True)
                - background_color: Background color or None (default: None)
                - console_width: Width for rendering (default: 80)
        """
        config = config or {}
        self._theme = config.get("theme", "monokai")
        self._line_numbers = config.get("line_numbers", False)
        self._word_wrap = config.get("word_wrap", True)
        self._background_color = config.get("background_color", None)
        self._console_width = config.get("console_width", 80)

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass

    def set_console_width(self, width: int) -> None:
        """Update the console width for rendering.

        Args:
            width: New console width in characters.
        """
        self._console_width = max(20, width)

    def has_code_blocks(self, text: str) -> bool:
        """Check if text contains any code blocks.

        Args:
            text: Text to check.

        Returns:
            True if text contains markdown code blocks.
        """
        return CODE_BLOCK_PATTERN.search(text) is not None

    def format_output(self, text: str) -> str:
        """Transform text with code blocks into ANSI-highlighted text.

        Parses the text for markdown code blocks (```lang...```) and renders
        them with syntax highlighting. Plain text segments are preserved.

        Args:
            text: Text potentially containing markdown code blocks.

        Returns:
            Text with ANSI escape codes for syntax highlighting.
            If no code blocks are found, returns the original text unchanged.
        """
        if not self.has_code_blocks(text):
            return text

        # Handle literal \n in text (escaped newlines) - convert to actual newlines
        text = text.replace('\\n', '\n')

        result_parts: List[str] = []
        last_end = 0

        for match in CODE_BLOCK_PATTERN.finditer(text):
            # Add any plain text before this code block
            plain_text = text[last_end:match.start()]
            if plain_text:
                result_parts.append(plain_text)

            # Extract language and code from the match
            language = match.group(1) or "text"
            code = match.group(2)

            # Remove trailing newline from code if present
            if code.endswith('\n'):
                code = code[:-1]

            # Render the code block with syntax highlighting
            highlighted = self._render_code_block(code, language)
            result_parts.append(highlighted)

            last_end = match.end()

        # Add any remaining plain text after the last code block
        remaining = text[last_end:]
        if remaining:
            result_parts.append(remaining)

        return ''.join(result_parts)

    def _render_code_block(self, code: str, language: str) -> str:
        """Render a single code block with syntax highlighting.

        Args:
            code: The code content (without ``` markers).
            language: The language identifier.

        Returns:
            ANSI-escaped string with syntax highlighting.
        """
        # Map language aliases
        lang = LANGUAGE_ALIASES.get(language.lower(), language.lower())

        try:
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
                width=self._console_width,
                force_terminal=True,
                no_color=False,
                highlight=False,
            )
            with console.capture() as capture:
                console.print(syntax, end="")

            return capture.get()

        except Exception:
            # Fallback: return code as-is if highlighting fails
            return code

    def format_code(self, code: str, language: str = "text") -> str:
        """Format a single code snippet with syntax highlighting.

        Convenience method for formatting code without markdown wrappers.

        Args:
            code: Code to highlight.
            language: Programming language (default: "text").

        Returns:
            ANSI-escaped string with syntax highlighting.
        """
        return self._render_code_block(code, language)

    def extract_code_blocks(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract all code blocks from text with their positions.

        Args:
            text: Text containing markdown code blocks.

        Returns:
            List of tuples: (language, code, start_pos, end_pos)
        """
        blocks = []
        for match in CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1) or "text"
            code = match.group(2)
            if code.endswith('\n'):
                code = code[:-1]
            blocks.append((language, code, match.start(), match.end()))
        return blocks


def create_plugin() -> CodeBlockFormatterPlugin:
    """Factory function to create a CodeBlockFormatterPlugin instance."""
    return CodeBlockFormatterPlugin()
