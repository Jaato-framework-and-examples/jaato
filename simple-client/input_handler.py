"""Input handler for interactive client.

Abstracts input collection with support for prompt_toolkit (with completion)
or fallback to standard readline input.
"""

import shutil
import sys
from typing import Callable, List, Optional, Tuple

# Try to import prompt_toolkit for enhanced completion
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    PromptSession = None
    InMemoryHistory = None
    ANSI = None

# Try to import file completer
try:
    from file_completer import CombinedCompleter, FileReferenceProcessor
    HAS_FILE_COMPLETER = True
except ImportError:
    HAS_FILE_COMPLETER = False
    CombinedCompleter = None
    FileReferenceProcessor = None


class InputHandler:
    """Handles user input with optional completion support.

    Provides a unified interface for input collection, whether using
    prompt_toolkit with rich completions or standard readline.
    """

    def __init__(self):
        """Initialize the input handler."""
        self._pt_history = InMemoryHistory() if HAS_PROMPT_TOOLKIT else None
        self._completer = CombinedCompleter() if (HAS_PROMPT_TOOLKIT and HAS_FILE_COMPLETER) else None
        self._file_processor = FileReferenceProcessor() if HAS_FILE_COMPLETER else None

        # Prompt style for completion menu and status bar
        self._pt_style = Style.from_dict({
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#00aa00 #ffffff',
            'completion-menu.meta.completion': 'bg:#333333 #888888',
            'completion-menu.meta.completion.current': 'bg:#00aa00 #ffffff',
            # Status bar styles
            'status-bar': 'bg:#333333 #aaaaaa',
            'status-bar.label': 'bg:#333333 #888888',
            'status-bar.value': 'bg:#333333 #ffffff bold',
            'status-bar.separator': 'bg:#333333 #555555',
        }) if HAS_PROMPT_TOOLKIT else None

    @property
    def has_completion(self) -> bool:
        """Check if completion support is available."""
        return HAS_PROMPT_TOOLKIT and self._completer is not None

    @property
    def has_file_processor(self) -> bool:
        """Check if file reference processing is available."""
        return self._file_processor is not None

    def get_input(self, prompt_str: str) -> str:
        """Get user input with completion support if available.

        Args:
            prompt_str: The prompt string to display.

        Returns:
            User input string (stripped).

        Raises:
            EOFError: On end of input (Ctrl+D).
            KeyboardInterrupt: On interrupt (Ctrl+C).
        """
        if HAS_PROMPT_TOOLKIT and self._completer:
            return self._get_prompt_toolkit_input(prompt_str)
        else:
            return input(prompt_str).strip()

    def _get_prompt_toolkit_input(self, prompt_str: str) -> str:
        """Get input using prompt_toolkit with completions.

        Args:
            prompt_str: The prompt string to display.

        Returns:
            User input string (stripped).
        """
        # Create ANSI-formatted prompt
        formatted_prompt = ANSI(prompt_str) if ANSI else prompt_str

        def get_size():
            cols, rows = shutil.get_terminal_size()
            return Size(rows=rows, columns=cols)

        # Create output with enable_cpr=False to avoid CPR queries in PTY environments
        output = Vt100_Output(sys.stdout, get_size=get_size, enable_cpr=False)

        session = PromptSession(
            completer=self._completer,
            history=self._pt_history,
            auto_suggest=AutoSuggestFromHistory(),
            style=self._pt_style,
            complete_while_typing=True,
            complete_in_thread=True,
            output=output,
            reserve_space_for_menu=0,  # Don't reserve space above - completions appear below
        )
        return session.prompt(formatted_prompt).strip()

    def expand_file_references(self, text: str) -> str:
        """Expand @file references to include file contents.

        Args:
            text: Input text that may contain @file references.

        Returns:
            Text with file contents appended, or original text if no references.
        """
        if not self._file_processor:
            return text
        return self._file_processor.expand_references(text)

    def add_to_history(self, text: str) -> None:
        """Add text to input history.

        Args:
            text: Text to add to history.
        """
        if HAS_PROMPT_TOOLKIT and self._pt_history:
            self._pt_history.append_string(text)

    def restore_history(self, inputs: List[str]) -> int:
        """Restore history from a list of inputs.

        Args:
            inputs: List of input strings to restore.

        Returns:
            Number of inputs restored.
        """
        if not HAS_PROMPT_TOOLKIT:
            return 0

        # Clear and restore
        self._pt_history = InMemoryHistory()
        for user_input in inputs:
            self._pt_history.append_string(user_input)

        return len(inputs)

    def add_commands(self, commands: List[Tuple[str, str]]) -> None:
        """Add commands to the completer.

        Args:
            commands: List of (command_name, description) tuples.
        """
        if self._completer:
            self._completer.add_commands(commands)

    def set_session_provider(self, provider: Callable[[], List]) -> None:
        """Set the session ID provider for completion.

        Args:
            provider: Callable that returns list of session objects.
        """
        if self._completer:
            self._completer.set_session_provider(provider)

    def set_command_completion_provider(
        self,
        provider: Callable[[str, List], List],
        commands: set
    ) -> None:
        """Set the provider for plugin command argument completions.

        Args:
            provider: Callable that takes (command, args) and returns completions.
            commands: Set of command names that support argument completion.
        """
        if self._completer:
            self._completer.set_command_completion_provider(provider, commands)
