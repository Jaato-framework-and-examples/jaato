# shared/plugins/code_validation_formatter/plugin.py
"""Streaming code validation formatter plugin for LSP diagnostics on output code blocks.

This plugin intercepts code blocks in model output, validates them via LSP,
and appends diagnostic warnings/errors. It buffers code blocks until complete,
while passing through regular text immediately.

The plugin runs BEFORE the code_block_formatter (priority 35 vs 40) because
the syntax highlighter strips the ``` markers. Validation warnings are inserted
after each code block, then syntax highlighting is applied to everything.

Usage:
    from shared.plugins.code_validation_formatter import create_plugin

    formatter = create_plugin()
    formatter.set_lsp_plugin(lsp_plugin)
    formatter.initialize({"enabled": True})

    # Streaming mode
    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional


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

# Priority for pipeline ordering (35 = BEFORE code_block_formatter at 40)
DEFAULT_PRIORITY = 35

# Severity icons for diagnostic output
SEVERITY_ICONS = {
    'Error': '❌',
    'Warning': '⚠️',
    'Information': 'ℹ️',
    'Hint': '💡',
}


def _trace(msg: str) -> None:
    """Write trace message to log file (only if JAATO_TRACE_LOG is set)."""
    trace_path = os.environ.get('JAATO_TRACE_LOG')
    if not trace_path:
        return
    try:
        with open(trace_path, "a") as f:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] [CodeValidator] {msg}\n")
            f.flush()
    except (IOError, OSError):
        pass


class CodeValidationFormatterPlugin:
    """Streaming plugin that validates code blocks via LSP and appends diagnostics.

    Implements the FormatterPlugin protocol. Buffers code blocks until complete,
    validates them, and appends warnings. Regular text passes through immediately.
    """

    def __init__(self):
        self._enabled = True
        self._lsp_plugin: Optional[Any] = None
        self._console_width = 80
        self._priority = DEFAULT_PRIORITY
        self._max_errors_per_block = 5
        self._max_warnings_per_block = 3
        self._feedback_callback: Optional[Callable[[str], None]] = None
        self._accumulated_issues: List[Dict[str, Any]] = []

        # Streaming state
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "code_validation_formatter"

    @property
    def priority(self) -> int:
        """Execution priority (35 = before syntax highlighting at 40)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, buffering code blocks for validation.

        Args:
            chunk: Incoming text chunk.

        Yields:
            Output chunks - immediate for regular text, validated code blocks
            with diagnostics appended.
        """
        if not self._enabled:
            yield chunk
            return

        self._buffer += chunk

        while self._buffer:
            if not self._in_code_block:
                # Look for code block start: ```lang\n
                match = re.search(r'```(\w*)\n', self._buffer)
                if match:
                    # Yield text before the code block
                    before = self._buffer[:match.start()]
                    if before:
                        yield before

                    # Enter code block mode, keep the ``` marker in buffer
                    self._code_block_lang = match.group(1) or ""
                    self._buffer = self._buffer[match.start():]
                    self._in_code_block = True
                else:
                    # Check if we might have a partial ``` at the end
                    if self._buffer.endswith('`'):
                        for i in range(min(3, len(self._buffer)), 0, -1):
                            if self._buffer[-i:] == '`' * i:
                                to_yield = self._buffer[:-i]
                                self._buffer = self._buffer[-i:]
                                if to_yield:
                                    yield to_yield
                                return
                    # No code block, yield everything
                    yield self._buffer
                    self._buffer = ""
            else:
                # In code block, look for closing \n```
                # Need to find ``` that's not part of the opening
                search_start = self._buffer.find('\n') + 1  # Skip opening line
                if search_start > 0:
                    end_match = re.search(r'\n```(?:\s|$)', self._buffer[search_start:])
                    if end_match:
                        # Found complete code block
                        end_pos = search_start + end_match.end()
                        code_block_text = self._buffer[:end_pos]

                        # Validate and possibly add diagnostics
                        validated = self._validate_and_annotate(code_block_text)
                        yield validated

                        # Exit code block mode
                        self._buffer = self._buffer[end_pos:]
                        self._in_code_block = False
                        self._code_block_lang = ""
                    else:
                        # Code block not complete yet
                        return
                else:
                    # Haven't even seen the opening newline yet
                    return

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content."""
        if self._buffer:
            if self._in_code_block:
                # Incomplete code block - still validate what we have
                validated = self._validate_and_annotate(self._buffer)
                yield validated
            else:
                yield self._buffer
            self._buffer = ""
            self._in_code_block = False
            self._code_block_lang = ""

        # Trigger feedback callback if there are accumulated issues
        if self._accumulated_issues and self._feedback_callback:
            feedback = self._build_feedback_message()
            self._feedback_callback(feedback)

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""
        self._accumulated_issues = []

    # ==================== ConfigurableFormatter Protocol ====================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the formatter with configuration."""
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._max_errors_per_block = config.get("max_errors_per_block", 5)
        self._max_warnings_per_block = config.get("max_warnings_per_block", 3)
        self._console_width = config.get("console_width", 80)
        self._priority = config.get("priority", DEFAULT_PRIORITY)
        _trace(f"initialize: enabled={self._enabled}")

    def set_console_width(self, width: int) -> None:
        """Update the console width for rendering."""
        self._console_width = max(20, width)

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self._enabled = False
        self.reset()
        _trace("shutdown: disabled")

    # ==================== LSP Integration ====================

    def set_lsp_plugin(self, lsp_plugin: Any) -> None:
        """Set the LSP plugin for validation."""
        self._lsp_plugin = lsp_plugin
        _trace(f"set_lsp_plugin: plugin set")

    def set_feedback_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for injecting feedback into conversation."""
        self._feedback_callback = callback

    def get_accumulated_issues(self) -> List[Dict[str, Any]]:
        """Get accumulated validation issues from the last format_output call."""
        return self._accumulated_issues

    def clear_accumulated_issues(self) -> None:
        """Clear accumulated validation issues."""
        self._accumulated_issues = []

    # ==================== Internal Methods ====================

    def _validate_and_annotate(self, code_block_text: str) -> str:
        """Validate a code block and append diagnostics if any.

        Args:
            code_block_text: Complete code block including ``` markers.

        Returns:
            Code block text with diagnostics appended after closing ```.
        """
        # Extract language and code from the block
        match = re.match(r'```(\w*)\n(.*?)(\n```)', code_block_text, re.DOTALL)
        if not match:
            return code_block_text

        language = match.group(1).lower()
        code = match.group(2)

        if language not in LANGUAGE_EXTENSIONS:
            return code_block_text

        # Validate the code
        diagnostics = self._validate_code(code, language)

        if not diagnostics:
            return code_block_text

        # Build warning text
        warning_text = self._format_diagnostics(diagnostics, language)

        # Accumulate for feedback
        self._accumulated_issues.append({
            'language': language,
            'code_snippet': code[:100] + '...' if len(code) > 100 else code,
            'diagnostics': diagnostics
        })

        # Insert warning after the closing ```
        return code_block_text + warning_text

    def _validate_code(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Validate a code snippet via syntax check and LSP."""
        diagnostics = []

        # First pass: syntax validation for supported languages
        syntax_errors = self._validate_syntax(code, language)
        if syntax_errors:
            diagnostics.extend(syntax_errors)
            _trace(f"_validate_code: found {len(syntax_errors)} syntax errors for {language}")

        if not self._lsp_plugin:
            return diagnostics

        extension = LANGUAGE_EXTENSIONS.get(language)
        if not extension:
            return diagnostics

        if not self._has_server_for_language(language):
            return diagnostics

        try:
            _trace(f"_validate_code: calling validate_snippet for {language}")
            result = self._lsp_plugin._exec_validate_snippet({
                'code': code,
                'language': language,
                'extension': extension
            })

            if isinstance(result, dict) and result.get('error'):
                _trace(f"_validate_code: LSP error: {result.get('error')}")
                return diagnostics

            if isinstance(result, list):
                _trace(f"_validate_code: found {len(result)} LSP diagnostics for {language}")
                diagnostics.extend(result)

            return diagnostics

        except Exception as e:
            _trace(f"_validate_code: exception: {e}")
            return diagnostics

    def _validate_syntax(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Validate syntax using language-specific compile/parse."""
        diagnostics = []

        if language in ('python', 'py'):
            try:
                compile(code, '<code_block>', 'exec')
            except SyntaxError as e:
                _trace(f"_validate_syntax: Python SyntaxError: {e}")
                diagnostics.append({
                    'severity': 'Error',
                    'message': str(e.msg) if e.msg else str(e),
                    'line': e.lineno or 1,
                    'character': e.offset or 0,
                    'source': 'syntax_check'
                })
            except Exception as e:
                _trace(f"_validate_syntax: Python compile error: {e}")
                diagnostics.append({
                    'severity': 'Error',
                    'message': str(e),
                    'line': 1,
                    'character': 0,
                    'source': 'syntax_check'
                })

        return diagnostics

    def _has_server_for_language(self, language: str) -> bool:
        """Check if LSP has a server for the given language."""
        if not self._lsp_plugin:
            return False

        connected = getattr(self._lsp_plugin, '_connected_servers', set())
        if not connected:
            return False

        lang_variants = {language}
        if language == 'py':
            lang_variants.add('python')
        elif language == 'python':
            lang_variants.add('py')
        elif language == 'js':
            lang_variants.add('javascript')
        elif language == 'ts':
            lang_variants.add('typescript')

        for server_name in connected:
            server_lower = server_name.lower()
            for variant in lang_variants:
                if variant in server_lower:
                    return True

        clients = getattr(self._lsp_plugin, '_clients', {})
        for name, client in clients.items():
            if hasattr(client, 'config') and hasattr(client.config, 'language_id'):
                if client.config.language_id in lang_variants:
                    return True

        return False

    def _format_diagnostics(self, diagnostics: List[Dict[str, Any]], language: str) -> str:
        """Format diagnostics as a visually distinct warning block."""
        errors = [d for d in diagnostics if d.get('severity') == 'Error']
        warnings = [d for d in diagnostics if d.get('severity') == 'Warning']

        if not errors and not warnings:
            return ""

        indent = "    │ "
        lines = [f"    ┌─ Code Validation ({language}) ─"]

        if errors:
            lines.append(f"{indent}{SEVERITY_ICONS['Error']} {len(errors)} error(s) found:")
            for err in errors[:self._max_errors_per_block]:
                line_num = err.get('line', '?')
                msg = err.get('message', 'Unknown error')
                lines.append(f"{indent}  Line {line_num}: {msg}")
            if len(errors) > self._max_errors_per_block:
                lines.append(f"{indent}  ... and {len(errors) - self._max_errors_per_block} more")

        if warnings:
            lines.append(f"{indent}{SEVERITY_ICONS['Warning']} {len(warnings)} warning(s):")
            for warn in warnings[:self._max_warnings_per_block]:
                line_num = warn.get('line', '?')
                msg = warn.get('message', 'Unknown warning')
                lines.append(f"{indent}  Line {line_num}: {msg}")
            if len(warnings) > self._max_warnings_per_block:
                lines.append(f"{indent}  ... and {len(warnings) - self._max_warnings_per_block} more")

        lines.append("    └─")
        # No leading \n (code block ends with one), add trailing \n for spacing
        return "\n".join(lines) + "\n"

    def _build_feedback_message(self) -> str:
        """Build a feedback message for conversation injection."""
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

    # ==================== Legacy Methods ====================

    def should_format(self, text: str, format_hint: Optional[str] = None) -> bool:
        """Legacy method for backwards compatibility."""
        if not self._enabled or not self._lsp_plugin:
            return False
        return '```' in text

    def format_output(self, text: str) -> str:
        """Legacy method for backwards compatibility."""
        self.reset()
        result_parts = []
        for output in self.process_chunk(text):
            result_parts.append(output)
        for output in self.flush():
            result_parts.append(output)
        return ''.join(result_parts)


def create_plugin() -> CodeValidationFormatterPlugin:
    """Factory function to create a CodeValidationFormatterPlugin instance."""
    return CodeValidationFormatterPlugin()
