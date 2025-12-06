"""Actors for handling user interaction in the clarification plugin."""

import sys
from abc import ABC, abstractmethod
from typing import Optional

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
            self._write(self._bold(f"Question {i}/{len(request.questions)}:"))
            answer = self._ask_question(question)

            if answer is None:  # User cancelled
                return ClarificationResponse(cancelled=True)

            answers.append(answer)
            self._write()

        self._write(self._green("✓ All questions answered."))
        self._write()

        return ClarificationResponse(answers=answers)

    def _ask_question(self, question: Question) -> Optional[Answer]:
        """Ask a single question and return the answer, or None if cancelled."""
        self._write(f"  {self._yellow(question.text)}")

        if question.question_type == QuestionType.FREE_TEXT:
            return self._ask_free_text(question)
        elif question.question_type == QuestionType.SINGLE_CHOICE:
            return self._ask_single_choice(question)
        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            return self._ask_multiple_choice(question)
        else:
            # Fallback to free text for unknown types
            return self._ask_free_text(question)

    def _ask_free_text(self, question: Question) -> Optional[Answer]:
        """Ask a free text question."""
        prompt_parts = ["  > "]
        if not question.required:
            prompt_parts.append(self._dim("(optional, press Enter to skip) "))

        while True:
            response = self._read_line("".join(prompt_parts))

            if response.lower() == "cancel":
                return None

            if not response and not question.required:
                return Answer(question_id=question.id, skipped=True)

            if not response and question.required:
                self._write(self._yellow("  Please provide an answer."))
                continue

            return Answer(question_id=question.id, free_text=response)

    def _ask_single_choice(self, question: Question) -> Optional[Answer]:
        """Ask a single choice question."""
        # Display choices
        for choice in question.choices:
            default_marker = ""
            if choice.id == question.default_choice_id:
                default_marker = self._dim(" (default)")
            self._write(f"    [{self._cyan(choice.id)}] {choice.text}{default_marker}")

        # Build prompt
        valid_ids = [c.id for c in question.choices]
        prompt_hint = "/".join(valid_ids)
        prompt = f"  Enter choice [{prompt_hint}]: "

        while True:
            response = self._read_line(prompt)

            if response.lower() == "cancel":
                return None

            # Use default if available and no input
            if not response and question.default_choice_id:
                return Answer(
                    question_id=question.id,
                    selected_choice_ids=[question.default_choice_id],
                )

            # Skip if optional and no input
            if not response and not question.required:
                return Answer(question_id=question.id, skipped=True)

            # Validate choice
            if response in valid_ids:
                return Answer(
                    question_id=question.id, selected_choice_ids=[response]
                )

            self._write(
                self._yellow(f"  Invalid choice. Please enter one of: {prompt_hint}")
            )

    def _ask_multiple_choice(self, question: Question) -> Optional[Answer]:
        """Ask a multiple choice question."""
        # Display choices
        self._write(self._dim("  (Select multiple by entering comma-separated values)"))
        for choice in question.choices:
            default_marker = ""
            if (
                question.default_choice_id
                and choice.id in question.default_choice_id.split(",")
            ):
                default_marker = self._dim(" (default)")
            self._write(f"    [{self._cyan(choice.id)}] {choice.text}{default_marker}")

        # Build prompt
        valid_ids = [c.id for c in question.choices]
        prompt = f"  Enter choices (comma-separated): "

        while True:
            response = self._read_line(prompt)

            if response.lower() == "cancel":
                return None

            # Use default if available and no input
            if not response and question.default_choice_id:
                default_ids = [
                    id.strip() for id in question.default_choice_id.split(",")
                ]
                return Answer(
                    question_id=question.id, selected_choice_ids=default_ids
                )

            # Skip if optional and no input
            if not response and not question.required:
                return Answer(question_id=question.id, skipped=True)

            # Parse and validate choices
            selected = [id.strip() for id in response.split(",") if id.strip()]

            if not selected and question.required:
                self._write(self._yellow("  Please select at least one option."))
                continue

            invalid = [id for id in selected if id not in valid_ids]
            if invalid:
                self._write(
                    self._yellow(f"  Invalid choice(s): {', '.join(invalid)}")
                )
                continue

            return Answer(question_id=question.id, selected_choice_ids=selected)


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

        for question in request.questions:
            answer = self._auto_answer(question)
            answers.append(answer)

        return ClarificationResponse(answers=answers)

    def _auto_answer(self, question: Question) -> Answer:
        """Generate an automatic answer for a question."""
        if question.question_type == QuestionType.FREE_TEXT:
            return Answer(question_id=question.id, free_text=self._default_free_text)

        # For choice questions, use default or first choice
        if question.default_choice_id:
            if question.question_type == QuestionType.MULTIPLE_CHOICE:
                selected = [
                    id.strip() for id in question.default_choice_id.split(",")
                ]
            else:
                selected = [question.default_choice_id]
            return Answer(question_id=question.id, selected_choice_ids=selected)

        # Use first choice if available
        if question.choices:
            return Answer(
                question_id=question.id, selected_choice_ids=[question.choices[0].id]
            )

        # Fallback for edge cases
        if not question.required:
            return Answer(question_id=question.id, skipped=True)

        return Answer(question_id=question.id, free_text=self._default_free_text)


def create_actor(actor_type: str = "console", **kwargs) -> ClarificationActor:
    """Factory function to create a clarification actor.

    Args:
        actor_type: Type of actor ("console" or "auto")
        **kwargs: Additional arguments for the specific actor type

    Returns:
        A ClarificationActor instance
    """
    if actor_type == "console":
        return ConsoleActor(**kwargs)
    elif actor_type == "auto":
        return AutoActor(**kwargs)
    else:
        raise ValueError(f"Unknown actor type: {actor_type}")
