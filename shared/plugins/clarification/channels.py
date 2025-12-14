"""Channels for handling user interaction in the clarification plugin."""

import sys
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .models import (
    Answer,
    ClarificationRequest,
    ClarificationResponse,
    Question,
    QuestionType,
)


class ClarificationChannel(ABC):
    """Base class for channels that handle user interaction for clarifications."""

    @abstractmethod
    def request_clarification(
        self,
        request: ClarificationRequest,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> ClarificationResponse:
        """
        Present the clarification request to the user and collect responses.

        Args:
            request: The clarification request containing questions
            on_question_displayed: Hook called when each question is shown.
                Signature: (tool_name, question_index, total_questions, question_lines) -> None
            on_question_answered: Hook called when user answers a question.
                Signature: (tool_name, question_index, answer_summary) -> None

        Returns:
            ClarificationResponse with user's answers
        """
        pass


class ConsoleChannel(ClarificationChannel):
    """Console-based channel for interactive terminal sessions."""

    def __init__(
        self,
        input_stream=None,
        output_stream=None,
        use_colors: bool = True,
    ):
        """
        Initialize the console channel.

        Args:
            input_stream: Input stream (defaults to sys.stdin)
            output_stream: Output stream (defaults to sys.stdout)
            use_colors: Whether to use ANSI colors in output
        """
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._use_colors = use_colors and hasattr(self._output, "isatty") and self._output.isatty()

    def _color(self, text: str, code: str) -> str:
        """Apply ANSI color code to text if colors are enabled."""
        if self._use_colors:
            return f"\033[{code}m{text}\033[0m"
        return text

    def _cyan(self, text: str) -> str:
        return self._color(text, "36")

    def _yellow(self, text: str) -> str:
        return self._color(text, "33")

    def _green(self, text: str) -> str:
        return self._color(text, "32")

    def _red(self, text: str) -> str:
        return self._color(text, "31")

    def _dim(self, text: str) -> str:
        return self._color(text, "2")

    def _bold(self, text: str) -> str:
        return self._color(text, "1")

    def _write(self, text: str = "", end: str = "\n") -> None:
        """Write text to output stream."""
        self._output.write(text + end)
        self._output.flush()

    def _read_line(self, prompt: str = "") -> str:
        """Read a line from input stream."""
        if prompt:
            self._write(prompt, end="")
        return self._input.readline().strip()

    def request_clarification(
        self,
        request: ClarificationRequest,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> ClarificationResponse:
        """Present questions to user via console and collect responses."""
        self._write()
        self._write(self._bold("═" * 60))
        self._write(self._bold(self._cyan("  Clarification Needed")))
        self._write(self._bold("═" * 60))
        self._write()

        if request.context:
            self._write(self._dim(request.context))
            self._write()

        self._write(
            self._dim(f"Please answer the following {len(request.questions)} question(s).")
        )
        self._write(self._dim("Type 'cancel' at any prompt to cancel all questions."))
        self._write()

        answers = []
        for i, question in enumerate(request.questions, 1):
            # Show question number and required/optional status
            req_status = self._red("*required") if question.required else self._dim("optional")
            self._write(self._bold(f"Question {i}/{len(request.questions)}") + f" [{req_status}]")
            answer = self._ask_question(i, question)

            if answer is None:  # User cancelled
                return ClarificationResponse(cancelled=True)

            answers.append(answer)
            self._write()

        self._write(self._green("✓ All questions answered."))
        self._write()

        return ClarificationResponse(answers=answers)

    def _ask_question(self, question_index: int, question: Question) -> Optional[Answer]:
        """Ask a single question and return the answer, or None if cancelled."""
        self._write(f"  {self._yellow(question.text)}")

        if question.question_type == QuestionType.FREE_TEXT:
            return self._ask_free_text(question_index, question)
        elif question.question_type == QuestionType.SINGLE_CHOICE:
            return self._ask_single_choice(question_index, question)
        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            return self._ask_multiple_choice(question_index, question)
        else:
            # Fallback to free text for unknown types
            return self._ask_free_text(question_index, question)

    def _ask_free_text(self, question_index: int, question: Question) -> Optional[Answer]:
        """Ask a free text question."""
        if not question.required:
            self._write(self._dim("  (press Enter to skip)"))

        while True:
            response = self._read_line("  > ")

            if response.lower() == "cancel":
                return None

            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)

            if not response and question.required:
                self._write(self._yellow("  Please provide an answer."))
                continue

            return Answer(question_index=question_index, free_text=response)

    def _ask_single_choice(self, question_index: int, question: Question) -> Optional[Answer]:
        """Ask a single choice question."""
        # Display choices with 1-based indices
        for i, choice in enumerate(question.choices, 1):
            default_marker = ""
            if question.default_choice == i:
                default_marker = self._dim(" (default)")
            self._write(f"    {self._cyan(str(i))}. {choice.text}{default_marker}")

        # Build prompt
        num_choices = len(question.choices)
        if num_choices == 1:
            prompt_hint = "1"
        else:
            prompt_hint = f"1-{num_choices}"
        prompt = f"  Enter choice [{prompt_hint}]: "

        while True:
            response = self._read_line(prompt)

            if response.lower() == "cancel":
                return None

            # Use default if available and no input
            if not response and question.default_choice:
                return Answer(
                    question_index=question_index,
                    selected_choices=[question.default_choice],
                )

            # Skip if optional and no input
            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)

            # Required but no input
            if not response:
                self._write(self._yellow("  Please select an option."))
                continue

            # Validate choice
            try:
                choice_num = int(response)
                if 1 <= choice_num <= num_choices:
                    return Answer(
                        question_index=question_index, selected_choices=[choice_num]
                    )
            except ValueError:
                pass

            self._write(
                self._yellow(f"  Invalid choice. Please enter a number from {prompt_hint}")
            )

    def _ask_multiple_choice(self, question_index: int, question: Question) -> Optional[Answer]:
        """Ask a multiple choice question."""
        # Display choices with 1-based indices
        self._write(self._dim("  (Enter comma-separated numbers, e.g., 1,3)"))
        default_indices: List[int] = []
        for i, choice in enumerate(question.choices, 1):
            default_marker = ""
            if question.default_choice and i == question.default_choice:
                default_marker = self._dim(" (default)")
                default_indices.append(i)
            self._write(f"    {self._cyan(str(i))}. {choice.text}{default_marker}")

        # Build prompt
        num_choices = len(question.choices)
        prompt = "  Enter choices: "

        while True:
            response = self._read_line(prompt)

            if response.lower() == "cancel":
                return None

            # Use default if available and no input
            if not response and default_indices:
                return Answer(
                    question_index=question_index, selected_choices=default_indices
                )

            # Skip if optional and no input
            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)

            # Required but no input
            if not response and question.required:
                self._write(self._yellow("  Please select at least one option."))
                continue

            # Parse and validate choices
            try:
                selected = []
                invalid = []
                for part in response.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    num = int(part)
                    if 1 <= num <= num_choices:
                        if num not in selected:
                            selected.append(num)
                    else:
                        invalid.append(part)

                if invalid:
                    self._write(
                        self._yellow(f"  Invalid choice(s): {', '.join(invalid)}. Use 1-{num_choices}.")
                    )
                    continue

                if not selected and question.required:
                    self._write(self._yellow("  Please select at least one option."))
                    continue

                return Answer(question_index=question_index, selected_choices=selected)

            except ValueError:
                self._write(
                    self._yellow(f"  Invalid input. Enter numbers from 1-{num_choices}, separated by commas.")
                )


class AutoChannel(ClarificationChannel):
    """Channel that automatically selects defaults or first available choices.

    Useful for non-interactive/automated scenarios or testing.
    """

    def __init__(self, default_free_text: str = "auto-response"):
        """
        Initialize the auto channel.

        Args:
            default_free_text: Default response for free text questions
        """
        self._default_free_text = default_free_text

    def request_clarification(
        self,
        request: ClarificationRequest,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> ClarificationResponse:
        """Automatically answer all questions with defaults."""
        answers = []

        for i, question in enumerate(request.questions, 1):
            answer = self._auto_answer(i, question)
            answers.append(answer)

        return ClarificationResponse(answers=answers)

    def _auto_answer(self, question_index: int, question: Question) -> Answer:
        """Generate an automatic answer for a question."""
        if question.question_type == QuestionType.FREE_TEXT:
            return Answer(question_index=question_index, free_text=self._default_free_text)

        # For choice questions, use default or first choice (1-based)
        if question.default_choice:
            return Answer(question_index=question_index, selected_choices=[question.default_choice])

        # Use first choice if available (1-based index)
        if question.choices:
            return Answer(question_index=question_index, selected_choices=[1])

        # Fallback for edge cases
        if not question.required:
            return Answer(question_index=question_index, skipped=True)

        return Answer(question_index=question_index, free_text=self._default_free_text)


class QueueChannel(ClarificationChannel):
    """Channel that displays prompts via callback and receives input via queue.

    Designed for TUI integration where:
    - Clarification prompts are shown in an output panel
    - User input comes through a shared queue from the main input handler
    - No direct stdin access needed (works with full-screen terminal UIs)
    """

    def __init__(self, **kwargs):
        self._output_callback: Optional[callable] = None
        self._input_queue: Optional['queue.Queue[str]'] = None
        self._waiting_for_input: bool = False
        self._prompt_callback: Optional[callable] = None

    def set_callbacks(
        self,
        output_callback: Optional[callable] = None,
        input_queue: Optional['queue.Queue[str]'] = None,
        prompt_callback: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """Set the callbacks and queue for TUI integration.

        Args:
            output_callback: Called with (source, text, mode) to display output.
            input_queue: Queue to receive user input from the main input handler.
            prompt_callback: Called with True when waiting for input, False when done.
        """
        self._output_callback = output_callback
        self._input_queue = input_queue
        self._prompt_callback = prompt_callback

    @property
    def waiting_for_input(self) -> bool:
        """Check if channel is waiting for user input."""
        return self._waiting_for_input

    def _output(self, text: str, mode: str = "append") -> None:
        """Output text via callback."""
        if self._output_callback:
            self._output_callback("clarification", text, mode)

    def _read_input(self, timeout: float = 60.0) -> Optional[str]:
        """Read input from the queue with timeout."""
        import queue as queue_module

        if not self._input_queue:
            return None

        try:
            return self._input_queue.get(timeout=timeout)
        except queue_module.Empty:
            return None

    def request_clarification(
        self,
        request: ClarificationRequest,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> ClarificationResponse:
        """Present questions via output panel and collect responses via queue."""
        total_questions = len(request.questions)

        answers = []
        for i, question in enumerate(request.questions, 1):
            # Build question lines for this single question
            question_lines = []
            if i == 1 and request.context:
                question_lines.append(f"Context: {request.context}")
                question_lines.append("")

            req_status = "*required" if question.required else "optional"
            question_lines.append(f"Question {i}/{total_questions} [{req_status}]")
            question_lines.append(f"  {question.text}")

            # Show choices if any
            if question.choices:
                for j, choice in enumerate(question.choices, 1):
                    default_marker = " (default)" if question.default_choice == j else ""
                    question_lines.append(f"    {j}. {choice.text}{default_marker}")

            # Show input hint
            if question.question_type == QuestionType.FREE_TEXT:
                if not question.required:
                    question_lines.append("  (press Enter to skip, or type 'cancel' to cancel)")
                else:
                    question_lines.append("  (type 'cancel' to cancel)")
            elif question.question_type == QuestionType.SINGLE_CHOICE:
                question_lines.append(f"  Enter choice [1-{len(question.choices)}]:")
            elif question.question_type == QuestionType.MULTIPLE_CHOICE:
                question_lines.append("  Enter choices (comma-separated, e.g., 1,3):")

            # Notify UI about this question
            if on_question_displayed:
                on_question_displayed("request_clarification", i, total_questions, question_lines)

            # Signal waiting for input
            self._waiting_for_input = True
            if self._prompt_callback:
                self._prompt_callback(True)

            try:
                response = self._read_input(timeout=60.0)

                if response is None or response.lower() == 'cancel':
                    return ClarificationResponse(cancelled=True)

                answer = self._parse_answer(i, question, response)
                answers.append(answer)

                # Notify UI that question was answered
                if on_question_answered:
                    answer_summary = self._format_answer_summary(answer, question)
                    on_question_answered("request_clarification", i, answer_summary)

            finally:
                self._waiting_for_input = False
                if self._prompt_callback:
                    self._prompt_callback(False)

        return ClarificationResponse(answers=answers)

    def _format_answer_summary(self, answer: Answer, question: Question) -> str:
        """Format a brief summary of the answer for display."""
        if answer.skipped:
            return "skipped"
        if answer.free_text is not None:
            text = answer.free_text
            if len(text) > 30:
                text = text[:27] + "..."
            return f'"{text}"'
        if answer.selected_choices:
            if len(answer.selected_choices) == 1:
                idx = answer.selected_choices[0]
                if question.choices and idx <= len(question.choices):
                    return question.choices[idx - 1].text
                return f"choice {idx}"
            else:
                return f"{len(answer.selected_choices)} choices"
        return "answered"

    def _parse_answer(self, question_index: int, question: Question, response: str) -> Answer:
        """Parse user response into an Answer."""
        response = response.strip()

        if question.question_type == QuestionType.FREE_TEXT:
            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)
            return Answer(question_index=question_index, free_text=response)

        elif question.question_type == QuestionType.SINGLE_CHOICE:
            if not response and question.default_choice:
                return Answer(question_index=question_index, selected_choices=[question.default_choice])
            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)
            try:
                choice_num = int(response)
                if 1 <= choice_num <= len(question.choices):
                    return Answer(question_index=question_index, selected_choices=[choice_num])
            except ValueError:
                pass
            # Invalid - return first choice as fallback
            return Answer(question_index=question_index, selected_choices=[1])

        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            if not response and not question.required:
                return Answer(question_index=question_index, skipped=True)
            selected = []
            for part in response.split(','):
                part = part.strip()
                if part:
                    try:
                        num = int(part)
                        if 1 <= num <= len(question.choices) and num not in selected:
                            selected.append(num)
                    except ValueError:
                        pass
            if selected:
                return Answer(question_index=question_index, selected_choices=selected)
            # Fallback
            return Answer(question_index=question_index, selected_choices=[1])

        # Unknown type - treat as free text
        return Answer(question_index=question_index, free_text=response)


def create_channel(channel_type: str = "console", **kwargs) -> ClarificationChannel:
    """Factory function to create a clarification channel.

    Args:
        channel_type: Type of channel ("console", "queue", or "auto")
        **kwargs: Additional arguments for the specific channel type

    Returns:
        A ClarificationChannel instance
    """
    if channel_type == "console":
        return ConsoleChannel(**kwargs)
    elif channel_type == "queue":
        return QueueChannel(**kwargs)
    elif channel_type == "auto":
        return AutoChannel(**kwargs)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
