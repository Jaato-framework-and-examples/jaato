"""Actors for handling user interaction in the clarification plugin."""

import sys
from abc import ABC, abstractmethod
from typing import List, Optional

from .models import (
    Answer,
    ClarificationRequest,
    ClarificationResponse,
    Question,
    QuestionType,
)


class ClarificationActor(ABC):
    """Base class for actors that handle user interaction for clarifications."""

    @abstractmethod
    def request_clarification(
        self, request: ClarificationRequest
    ) -> ClarificationResponse:
        """
        Present the clarification request to the user and collect responses.

        Args:
            request: The clarification request containing questions

        Returns:
            ClarificationResponse with user's answers
        """
        pass


class ConsoleActor(ClarificationActor):
    """Console-based actor for interactive terminal sessions."""

    def __init__(
        self,
        input_stream=None,
        output_stream=None,
        use_colors: bool = True,
    ):
        """
        Initialize the console actor.

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
        self, request: ClarificationRequest
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


class AutoActor(ClarificationActor):
    """Actor that automatically selects defaults or first available choices.

    Useful for non-interactive/automated scenarios or testing.
    """

    def __init__(self, default_free_text: str = "auto-response"):
        """
        Initialize the auto actor.

        Args:
            default_free_text: Default response for free text questions
        """
        self._default_free_text = default_free_text

    def request_clarification(
        self, request: ClarificationRequest
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


class CallbackConsoleActor(ConsoleActor):
    """Console actor with pause/resume callbacks for TUI integration.

    Extends ConsoleActor to call pause/resume callbacks before and after
    user interaction, allowing TUI frameworks to temporarily yield control.
    """

    def __init__(self, **kwargs):
        """Initialize the callback console actor.

        Args:
            **kwargs: Additional arguments passed to ConsoleActor.
        """
        super().__init__(**kwargs)
        self._pause_callback = None
        self._resume_callback = None
        self._output_callback = None
        self._run_in_terminal = None

    def set_callbacks(
        self,
        pause_callback=None,
        resume_callback=None,
        output_callback=None,
        run_in_terminal=None,
    ) -> None:
        """Set the pause/resume/output callbacks.

        Args:
            pause_callback: Called before requesting user input (legacy).
            resume_callback: Called after user input is complete (legacy).
            output_callback: Called with (source, text, mode) to log the Q&A summary.
            run_in_terminal: Callable that wraps console I/O for TUI frameworks.
                            Signature: run_in_terminal(func) -> None
                            If provided, pause/resume callbacks are ignored.
        """
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback
        self._output_callback = output_callback
        self._run_in_terminal = run_in_terminal

    def request_clarification(
        self, request: ClarificationRequest
    ) -> ClarificationResponse:
        """Present questions to user via console with pause/resume callbacks.

        If a run_in_terminal callable is provided via set_callbacks, the
        console interaction runs inside it to properly suspend TUI frameworks.
        """
        response = None

        def do_clarification():
            nonlocal response
            response = super(CallbackConsoleActor, self).request_clarification(request)

        # If we have a run_in_terminal wrapper, use it
        if self._run_in_terminal:
            self._run_in_terminal(do_clarification)
        else:
            # Fallback to simple pause/resume
            if self._pause_callback:
                self._pause_callback()
            try:
                do_clarification()
            finally:
                if self._resume_callback:
                    self._resume_callback()

        # Log the Q&A summary to output after interaction
        if self._output_callback and response and not response.cancelled:
            self._log_qa_summary(request, response)

        return response

    def _log_qa_summary(
        self, request: ClarificationRequest, response: ClarificationResponse
    ) -> None:
        """Log a summary of the Q&A to the output callback."""
        if not self._output_callback:
            return

        # Header
        self._output_callback("clarification", "Clarification Answered:", "write")

        if request.context:
            self._output_callback("clarification", f"  Context: {request.context}", "append")

        # Each Q&A pair
        for i, (question, answer) in enumerate(zip(request.questions, response.answers), 1):
            self._output_callback("clarification", f"  Q{i}: {question.text}", "append")

            # Show available choices if any
            if question.choices:
                self._output_callback("clarification", "      Options:", "append")
                for j, choice in enumerate(question.choices, 1):
                    self._output_callback("clarification", f"        {j}. {choice.text}", "append")

            if answer.skipped:
                self._output_callback("clarification", f"  A{i}: (skipped)", "append")
            elif answer.free_text:
                self._output_callback("clarification", f"  A{i}: {answer.free_text}", "append")
            elif answer.selected_choices:
                # Map choice indices to choice text
                choices_text = []
                for choice_idx in answer.selected_choices:
                    if 1 <= choice_idx <= len(question.choices):
                        choices_text.append(question.choices[choice_idx - 1].text)
                self._output_callback("clarification", f"  A{i}: {', '.join(choices_text)}", "append")


def create_actor(actor_type: str = "console", **kwargs) -> ClarificationActor:
    """Factory function to create a clarification actor.

    Args:
        actor_type: Type of actor ("console", "callback_console", or "auto")
        **kwargs: Additional arguments for the specific actor type

    Returns:
        A ClarificationActor instance
    """
    if actor_type == "console":
        return ConsoleActor(**kwargs)
    elif actor_type == "callback_console":
        return CallbackConsoleActor(**kwargs)
    elif actor_type == "auto":
        return AutoActor(**kwargs)
    else:
        raise ValueError(f"Unknown actor type: {actor_type}")
