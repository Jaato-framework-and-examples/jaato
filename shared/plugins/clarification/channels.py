"""Channels for handling user interaction in the clarification plugin."""

import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from .models import (
    Answer,
    ClarificationRequest,
    ClarificationResponse,
    Question,
    QuestionType,
)
from ...message_queue import SourceType


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
        self._cancel_token: Optional[Any] = None  # CancelToken for interruption

    def set_callbacks(
        self,
        output_callback: Optional[callable] = None,
        input_queue: Optional['queue.Queue[str]'] = None,
        prompt_callback: Optional[callable] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Set the callbacks and queue for TUI integration.

        Args:
            output_callback: Called with (source, text, mode) to display output.
            input_queue: Queue to receive user input from the main input handler.
            prompt_callback: Called with True when waiting for input, False when done.
            cancel_token: Optional CancelToken to check for cancellation requests.
        """
        self._output_callback = output_callback
        self._input_queue = input_queue
        self._prompt_callback = prompt_callback
        self._cancel_token = cancel_token

    @property
    def waiting_for_input(self) -> bool:
        """Check if channel is waiting for user input."""
        return self._waiting_for_input

    def _output(self, text: str, mode: str = "append") -> None:
        """Output text via callback."""
        if self._output_callback:
            self._output_callback("clarification", text, mode)

    def _read_input(self, timeout: float = 60.0) -> Optional[str]:
        """Read input from the queue with timeout, checking for cancellation.

        Returns:
            User input string, None on timeout, or "__CANCELLED__" if cancelled.
        """
        import queue as queue_module
        import time

        if not self._input_queue:
            return None

        # Poll in short intervals to check for cancellation
        poll_interval = 0.1  # Check every 100ms
        elapsed = 0.0

        while elapsed < timeout:
            # Check for cancellation
            if self._cancel_token and hasattr(self._cancel_token, 'is_cancelled'):
                if self._cancel_token.is_cancelled:
                    return "__CANCELLED__"

            try:
                return self._input_queue.get(timeout=poll_interval)
            except queue_module.Empty:
                elapsed += poll_interval

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

                # Handle cancellation (via cancel token or typing 'cancel')
                if response is None or response == "__CANCELLED__" or response.lower() == 'cancel':
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


class ParentBridgedChannel(ClarificationChannel):
    """Channel for subagents that routes clarification requests through parent.

    Instead of blocking on a local queue waiting for user input, this channel:
    1. Forwards the clarification request to the parent agent
    2. Waits for the parent's response on the session's injection queue
    3. Parses the response and returns it

    This unifies the communication model - all subagent input comes through
    the same injection queue mechanism.
    """

    def __init__(self, **kwargs):
        """Initialize the parent-bridged channel.

        The session reference is set later via set_session().
        """
        self._session: Optional[Any] = None  # JaatoSession reference
        self._pending_request_id: Optional[str] = None
        self._timeout: float = 300.0  # 5 minute timeout for parent response

    def set_session(self, session: Any) -> None:
        """Set the session reference for forwarding to parent.

        Args:
            session: JaatoSession instance that has parent reference.
        """
        self._session = session

    def _format_request_for_parent(
        self,
        request_id: str,
        request: ClarificationRequest
    ) -> str:
        """Format clarification request as structured message for parent.

        Args:
            request_id: Unique ID for tracking request/response.
            request: The clarification request.

        Returns:
            Formatted string for parent to parse.
        """
        lines = [f'<clarification_request request_id="{request_id}">']

        if request.context:
            lines.append(f'  <context>{request.context}</context>')

        for i, question in enumerate(request.questions, 1):
            q_type = question.question_type.value
            required = "true" if question.required else "false"
            lines.append(f'  <question index="{i}" type="{q_type}" required="{required}">')
            lines.append(f'    <text>{question.text}</text>')

            if question.choices:
                lines.append('    <choices>')
                for j, choice in enumerate(question.choices, 1):
                    default = ' default="true"' if question.default_choice == j else ''
                    lines.append(f'      <choice index="{j}"{default}>{choice.text}</choice>')
                lines.append('    </choices>')

            lines.append('  </question>')

        lines.append('</clarification_request>')
        return '\n'.join(lines)

    def _parse_response_from_parent(self, response: str, request: ClarificationRequest) -> ClarificationResponse:
        """Parse parent's response into ClarificationResponse.

        Args:
            response: Raw response string from parent.
            request: Original request (to match questions).

        Returns:
            Parsed ClarificationResponse.
        """
        import re

        # Check for cancellation
        if '<cancelled' in response.lower() or response.strip().lower() == 'cancel':
            return ClarificationResponse(cancelled=True)

        answers = []

        # Try to parse structured response
        # Format: <clarification_response request_id="...">
        #           <answer index="1">response text</answer>
        #         </clarification_response>
        answer_pattern = r'<answer\s+index="(\d+)"[^>]*>(.*?)</answer>'
        matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for index_str, answer_text in matches:
                question_index = int(index_str)
                answer_text = answer_text.strip()

                if question_index <= len(request.questions):
                    question = request.questions[question_index - 1]
                    answer = self._parse_single_answer(question_index, answer_text, question)
                    answers.append(answer)
        else:
            # Fallback: treat entire response as answer to first question
            if request.questions:
                question = request.questions[0]
                answer = self._parse_single_answer(1, response.strip(), question)
                answers.append(answer)

        return ClarificationResponse(answers=answers)

    def _parse_single_answer(self, question_index: int, text: str, question: Question) -> Answer:
        """Parse a single answer based on question type."""
        if question.question_type == QuestionType.FREE_TEXT:
            return Answer(question_index=question_index, free_text=text)

        elif question.question_type == QuestionType.SINGLE_CHOICE:
            # Try to parse as number
            try:
                choice = int(text)
                if 1 <= choice <= len(question.choices):
                    return Answer(question_index=question_index, selected_choices=[choice])
            except ValueError:
                pass
            # Try to match text to choice
            for i, choice in enumerate(question.choices, 1):
                if choice.text.lower() == text.lower():
                    return Answer(question_index=question_index, selected_choices=[i])
            # Default to free text
            return Answer(question_index=question_index, free_text=text)

        elif question.question_type == QuestionType.MULTIPLE_CHOICE:
            # Parse comma-separated choices
            choices = []
            for part in text.split(','):
                try:
                    choice = int(part.strip())
                    if 1 <= choice <= len(question.choices):
                        choices.append(choice)
                except ValueError:
                    pass
            return Answer(question_index=question_index, selected_choices=choices)

        return Answer(question_index=question_index, free_text=text)

    def _wait_for_response(self, request_id: str) -> Optional[str]:
        """Wait for parent's response on injection queue.

        Args:
            request_id: The request ID to match.

        Returns:
            Response string or None on timeout.
        """
        import queue as queue_module
        import time

        if not self._session:
            return None

        # Access the session's injection queue
        injection_queue = getattr(self._session, '_injection_queue', None)
        if not injection_queue:
            return None

        poll_interval = 0.1
        elapsed = 0.0
        held_messages = []  # Messages that aren't our response

        while elapsed < self._timeout:
            # Check for cancellation
            cancel_token = getattr(self._session, '_cancel_token', None)
            if cancel_token and hasattr(cancel_token, 'is_cancelled'):
                if cancel_token.is_cancelled:
                    # Put held messages back
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return None

            try:
                message = injection_queue.get(timeout=poll_interval)

                # Check if this is our response
                if f'request_id="{request_id}"' in message or f"request_id='{request_id}'" in message:
                    # Put held messages back
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return message

                # Check if it's a clarification response without explicit request_id
                # (simple response from parent for single pending request)
                if '<clarification_response' in message.lower() and self._pending_request_id == request_id:
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return message

                # Not our response, hold it
                held_messages.append(message)

            except queue_module.Empty:
                elapsed += poll_interval

        # Timeout - put held messages back
        for msg in held_messages:
            injection_queue.put(msg)
        return None

    def request_clarification(
        self,
        request: ClarificationRequest,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> ClarificationResponse:
        """Request clarification by forwarding to parent agent.

        Args:
            request: The clarification request.
            on_question_displayed: Hook for UI (called but parent sees formatted request).
            on_question_answered: Hook for UI (called when response received).

        Returns:
            ClarificationResponse from parent.
        """
        import uuid

        if not self._session:
            # No session - return cancelled
            return ClarificationResponse(cancelled=True)

        # Check if we have a parent
        parent_session = getattr(self._session, '_parent_session', None)
        if not parent_session:
            # No parent (this is main agent) - fall back to cancelled
            # In practice, main agent should use QueueChannel, not this
            return ClarificationResponse(cancelled=True)

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        self._pending_request_id = request_id

        # Format request for parent
        formatted_request = self._format_request_for_parent(request_id, request)

        # Notify hooks (for UI)
        if on_question_displayed:
            for i, question in enumerate(request.questions, 1):
                question_lines = [question.text]
                if question.choices:
                    for j, choice in enumerate(question.choices, 1):
                        question_lines.append(f"  {j}. {choice.text}")
                on_question_displayed(
                    "request_clarification",
                    i,
                    len(request.questions),
                    question_lines
                )

        # Forward to parent via session's mechanism
        forward_method = getattr(self._session, '_forward_to_parent', None)
        if forward_method:
            forward_method("CLARIFICATION_REQUESTED", formatted_request)
        else:
            # Fallback: direct inject to parent (CHILD source - from subagent)
            agent_id = getattr(self._session, '_agent_id', 'unknown')
            parent_session.inject_prompt(
                f"[SUBAGENT agent_id={agent_id} event=CLARIFICATION_REQUESTED]\n{formatted_request}",
                source_id=agent_id,
                source_type=SourceType.CHILD
            )

        # Wait for response
        response = self._wait_for_response(request_id)
        self._pending_request_id = None

        if response is None:
            # Timeout
            return ClarificationResponse(cancelled=True)

        # Parse response
        result = self._parse_response_from_parent(response, request)

        # Notify answer hooks
        if on_question_answered and result.answers:
            for answer in result.answers:
                summary = answer.free_text or str(answer.selected_choices)
                on_question_answered(
                    "request_clarification",
                    answer.question_index,
                    summary
                )

        return result


def create_channel(channel_type: str = "console", **kwargs) -> ClarificationChannel:
    """Factory function to create a clarification channel.

    Args:
        channel_type: Type of channel ("console", "queue", "auto", or "parent_bridged")
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
    elif channel_type == "parent_bridged":
        return ParentBridgedChannel(**kwargs)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
