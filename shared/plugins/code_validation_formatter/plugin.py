# shared/plugins/code_validation_formatter/plugin.py
"""Code validation formatter plugin for LSP diagnostics on output code blocks.

This plugin intercepts code blocks in model output, validates them via LSP,
and appends diagnostic warnings/errors. This enables the model to see issues
before code is written to files.

The plugin runs AFTER the code_block_formatter (priority 40) so that syntax
highlighting is applied first, then validation warnings are appended.

Usage (standalone):
    from shared.plugins.code_validation_formatter import create_plugin

    formatter = create_plugin()
    formatter.set_lsp_plugin(lsp_plugin)
    formatter.initialize({"enabled": True})
    formatted = formatter.format_output(text)

Usage (pipeline):
    from shared.plugins.formatter_pipeline import create_pipeline
    from shared.plugins.code_validation_formatter import create_plugin

    pipeline = create_pipeline()
    code_validator = create_plugin()
    code_validator.set_lsp_plugin(lsp_plugin)
    pipeline.register(code_validator)
    formatted = pipeline.format(text)
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Try to import Rich for colored output (optional)
try:
    from rich.console import Console
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Regex pattern for detecting code blocks with optional language specifier
# Matches: ```lang\ncode\n``` or ```\ncode\n```
CODE_BLOCK_PATTERN = re.compile(
    r'```(\w*)\n(.*?)```',
    re.DOTALL
)

# Map of language identifiers to file extensions for temp file creation
LANGUAGE_EXTENSIONS = {
    'python': '.py',
    'py': '.py',
    'javascript': '.js',
    'js': '.js',
    'typescript': '.ts',
    'ts': '.ts',
    'tsx': '.tsx',
    'jsx': '.jsx',
    'go': '.go',
    'rust': '.rs',
    'java': '.java',
    'kotlin': '.kt',
    'c': '.c',
    'cpp': '.cpp',
    'c++': '.cpp',
    'csharp': '.cs',
    'cs': '.cs',
    'ruby': '.rb',
    'rb': '.rb',
    'php': '.php',
    'swift': '.swift',
    'scala': '.scala',
    'lua': '.lua',
    'r': '.r',
    'zig': '.zig',
}

# Priority for pipeline ordering (45 = after code_block_formatter at 40)
DEFAULT_PRIORITY = 45

# Severity icons for diagnostic output
SEVERITY_ICONS = {
    'Error': '❌',
    'Warning': '⚠️',
    'Information': 'ℹ️',
    'Hint': '💡',
}


def _trace(msg: str) -> None:
    """Write trace message to log file for debugging."""
    trace_path = os.environ.get(
        'JAATO_TRACE_LOG',
        os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
    )
    if trace_path:
        try:
            with open(trace_path, "a") as f:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] [CodeValidator] {msg}\n")
                f.flush()
        except (IOError, OSError):
            pass


class CodeValidationFormatterPlugin:
    """Plugin that validates code blocks via LSP and appends diagnostics.

    Implements the FormatterPlugin protocol for use in a formatter pipeline.
    Detects markdown code blocks, runs LSP diagnostics, and appends warnings.
    """

    def __init__(self):
        self._enabled = True
        self._lsp_plugin: Optional[Any] = None  # LSPToolPlugin instance
        self._temp_dir: Optional[Path] = None
        self._console_width = 80
        self._priority = DEFAULT_PRIORITY
        self._max_errors_per_block = 5  # Limit errors shown per code block
        self._max_warnings_per_block = 3  # Limit warnings shown per code block
        # Callback for injecting feedback into conversation
        self._feedback_callback: Optional[Callable[[str], None]] = None
        # Accumulated validation issues for feedback injection
        self._accumulated_issues: List[Dict[str, Any]] = []

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "code_validation_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (45 = after syntax highlighting)."""
        return self._priority

    def should_format(self, text: str, format_hint: Optional[str] = None) -> bool:
        """Check if this formatter should process the text.

        Args:
            text: Text to check.
            format_hint: Optional hint (e.g., "model" for model output).

        Returns:
            True if enabled and text contains code blocks we can validate.
        """
        if not self._enabled:
            return False

        if not self._lsp_plugin:
            return False

        # Only process if there are code blocks with supported languages
        for match in CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1).lower()
            if language in LANGUAGE_EXTENSIONS:
                return True

        return False

    def format_output(self, text: str) -> str:
        """Validate code blocks and append diagnostics.

        Args:
            text: Text potentially containing markdown code blocks.

        Returns:
            Text with validation warnings appended after code blocks.
        """
        if not self.should_format(text):
            return text

        _trace(f"format_output: processing text with {len(list(CODE_BLOCK_PATTERN.finditer(text)))} code blocks")

        # Clear accumulated issues for this output
        self._accumulated_issues = []

        # Process code blocks in reverse order to preserve positions
        matches = list(CODE_BLOCK_PATTERN.finditer(text))
        result = text

        for match in reversed(matches):
            language = match.group(1).lower()
            code = match.group(2)

            if language not in LANGUAGE_EXTENSIONS:
                continue

            # Remove trailing newline from code if present
            if code.endswith('\n'):
                code = code[:-1]

            # Validate the code block
            diagnostics = self._validate_code(code, language)

            if diagnostics:
                # Build warning text to append after the code block
                warning_text = self._format_diagnostics(diagnostics, language)

                # Insert warning after the code block
                insert_pos = match.end()
                result = result[:insert_pos] + warning_text + result[insert_pos:]

                # Accumulate for feedback injection
                self._accumulated_issues.append({
                    'language': language,
                    'code_snippet': code[:100] + '...' if len(code) > 100 else code,
                    'diagnostics': diagnostics
                })

        # Trigger feedback callback if there are issues
        if self._accumulated_issues and self._feedback_callback:
            feedback = self._build_feedback_message()
            self._feedback_callback(feedback)

        return result

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration.

        Args:
            config: Dict with optional settings:
                - enabled: Enable/disable validation (default: True)
                - max_errors_per_block: Max errors to show (default: 5)
                - max_warnings_per_block: Max warnings to show (default: 3)
                - console_width: Width for rendering (default: 80)
                - priority: Pipeline priority (default: 45)
        """
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._max_errors_per_block = config.get("max_errors_per_block", 5)
        self._max_warnings_per_block = config.get("max_warnings_per_block", 3)
        self._console_width = config.get("console_width", 80)
        self._priority = config.get("priority", DEFAULT_PRIORITY)

        # Create temp directory for validation files
        self._temp_dir = Path(tempfile.mkdtemp(prefix="jaato_code_validation_"))
        _trace(f"initialize: temp_dir={self._temp_dir}, enabled={self._enabled}")

    def set_console_width(self, width: int) -> None:
        """Update the console width for rendering.

        Args:
            width: New console width in characters.
        """
        self._console_width = max(20, width)

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
            except OSError:
                pass
        self._temp_dir = None
        _trace("shutdown: cleaned up temp directory")

    # ==================== LSP Integration ====================

    def set_lsp_plugin(self, lsp_plugin: Any) -> None:
        """Set the LSP plugin for validation.

        Args:
            lsp_plugin: An LSPToolPlugin instance.
        """
        self._lsp_plugin = lsp_plugin
        _trace(f"set_lsp_plugin: plugin set, connected_servers={getattr(lsp_plugin, '_connected_servers', set())}")

    def set_feedback_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for injecting feedback into conversation.

        The callback will be called with a formatted message when validation
        issues are found. This message can be injected into the conversation
        to trigger model self-correction.

        Args:
            callback: Function that takes a feedback message string.
        """
        self._feedback_callback = callback

    def get_accumulated_issues(self) -> List[Dict[str, Any]]:
        """Get accumulated validation issues from the last format_output call.

        Returns:
            List of issue dicts with 'language', 'code_snippet', 'diagnostics'.
        """
        return self._accumulated_issues

    def clear_accumulated_issues(self) -> None:
        """Clear accumulated validation issues."""
        self._accumulated_issues = []

    # ==================== Internal Methods ====================

    def _validate_code(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Validate a code snippet via LSP.

        Args:
            code: The code content.
            language: Programming language identifier.

        Returns:
            List of diagnostic dicts with severity, message, line, etc.
        """
        if not self._lsp_plugin or not self._temp_dir:
            return []

        # Get file extension for this language
        extension = LANGUAGE_EXTENSIONS.get(language)
        if not extension:
            return []

        # Check if LSP has a server for this language
        if not self._has_server_for_language(language):
            _trace(f"_validate_code: no LSP server for {language}")
            return []

        # Create temp file with the code
        temp_file = self._temp_dir / f"validate_{language}{extension}"
        try:
            temp_file.write_text(code)
        except OSError as e:
            _trace(f"_validate_code: failed to write temp file: {e}")
            return []

        try:
            # Call LSP diagnostics
            result = self._lsp_plugin._exec_get_diagnostics({
                'file_path': str(temp_file)
            })

            if isinstance(result, dict) and result.get('error'):
                _trace(f"_validate_code: LSP error: {result.get('error')}")
                return []

            if isinstance(result, list):
                _trace(f"_validate_code: found {len(result)} diagnostics for {language}")
                return result

            return []

        except Exception as e:
            _trace(f"_validate_code: exception: {e}")
            return []

        finally:
            # Clean up temp file
            try:
                temp_file.unlink()
            except OSError:
                pass

    def _has_server_for_language(self, language: str) -> bool:
        """Check if LSP has a server for the given language.

        Args:
            language: Language identifier (e.g., 'python', 'javascript').

        Returns:
            True if a server is available.
        """
        if not self._lsp_plugin:
            return False

        # Check connected servers
        connected = getattr(self._lsp_plugin, '_connected_servers', set())
        if not connected:
            return False

        # Map language to what LSP servers might be named
        lang_variants = {language}
        if language == 'py':
            lang_variants.add('python')
        elif language == 'js':
            lang_variants.add('javascript')
        elif language == 'ts':
            lang_variants.add('typescript')

        for server_name in connected:
            server_lower = server_name.lower()
            for variant in lang_variants:
                if variant in server_lower:
                    return True

        # Also check by languageId in server config
        clients = getattr(self._lsp_plugin, '_clients', {})
        for name, client in clients.items():
            if hasattr(client, 'config') and hasattr(client.config, 'language_id'):
                if client.config.language_id in lang_variants:
                    return True

        return False

    def _format_diagnostics(
        self,
        diagnostics: List[Dict[str, Any]],
        language: str
    ) -> str:
        """Format diagnostics as a warning block.

        Args:
            diagnostics: List of diagnostic dicts.
            language: The language of the code block.

        Returns:
            Formatted warning text to append after the code block.
        """
        # Separate by severity
        errors = [d for d in diagnostics if d.get('severity') == 'Error']
        warnings = [d for d in diagnostics if d.get('severity') == 'Warning']
        others = [d for d in diagnostics if d.get('severity') not in ('Error', 'Warning')]

        lines = ["\n"]  # Start with newline after code block

        # Format errors
        if errors:
            lines.append(f"⚠️ **Code Validation ({language}):** {len(errors)} error(s) found")
            for i, err in enumerate(errors[:self._max_errors_per_block]):
                line_num = err.get('line', '?')
                msg = err.get('message', 'Unknown error')
                lines.append(f"  {SEVERITY_ICONS['Error']} Line {line_num}: {msg}")
            if len(errors) > self._max_errors_per_block:
                lines.append(f"  ... and {len(errors) - self._max_errors_per_block} more errors")

        # Format warnings (only if no errors, to reduce noise)
        if warnings and not errors:
            lines.append(f"⚠️ **Code Validation ({language}):** {len(warnings)} warning(s)")
            for i, warn in enumerate(warnings[:self._max_warnings_per_block]):
                line_num = warn.get('line', '?')
                msg = warn.get('message', 'Unknown warning')
                lines.append(f"  {SEVERITY_ICONS['Warning']} Line {line_num}: {msg}")
            if len(warnings) > self._max_warnings_per_block:
                lines.append(f"  ... and {len(warnings) - self._max_warnings_per_block} more warnings")

        if not errors and not warnings:
            # Only info/hints - skip unless verbose
            return ""

        lines.append("")  # Blank line after warnings
        return "\n".join(lines)

    def _build_feedback_message(self) -> str:
        """Build a feedback message for conversation injection.

        Returns:
            Formatted feedback message for the model.
        """
        if not self._accumulated_issues:
            return ""

        lines = [
            "[Code Validation Feedback]",
            "The following issues were detected in your code output:",
            ""
        ]

        for i, issue in enumerate(self._accumulated_issues, 1):
            lang = issue['language']
            diagnostics = issue['diagnostics']
            errors = [d for d in diagnostics if d.get('severity') == 'Error']
            warnings = [d for d in diagnostics if d.get('severity') == 'Warning']

            lines.append(f"Code block {i} ({lang}):")
            if errors:
                lines.append(f"  - {len(errors)} error(s):")
                for err in errors[:3]:
                    lines.append(f"    Line {err.get('line', '?')}: {err.get('message', '?')}")
            if warnings:
                lines.append(f"  - {len(warnings)} warning(s)")
            lines.append("")

        lines.append("Please review and correct these issues before writing to files.")

        return "\n".join(lines)


def create_plugin() -> CodeValidationFormatterPlugin:
    """Factory function to create a CodeValidationFormatterPlugin instance."""
    return CodeValidationFormatterPlugin()
