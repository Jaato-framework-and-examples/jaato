"""Clarification plugin for requesting user input with multiple questions and choices."""

from typing import Any, Callable, Dict, List, Optional

from ..model_provider.types import ToolSchema

from ..base import UserCommand
from .channels import ClarificationChannel, create_channel
from .models import (
    Choice,
    ClarificationRequest,
    Question,
    QuestionType,
)


class ClarificationPlugin:
    """Plugin that allows the model to request clarifications from the user.

    This plugin provides a tool for the AI model to ask the user multiple
    questions, each potentially with multiple choices, when it needs more
    information to proceed with a task.

    Features:
    - Multiple questions per request
    - Multiple choice types: single choice, multiple choice, or free text
    - Optional questions with defaults
    - Context explanation for why clarification is needed
    """

    def __init__(self):
        self._initialized = False
        self._channel: Optional[ClarificationChannel] = None
        # Clarification lifecycle hooks for UI integration
        self._on_clarification_requested: Optional[Callable[[str, List[str]], None]] = None
        self._on_clarification_resolved: Optional[Callable[[str], None]] = None
        # Per-question hooks for progressive display
        self._on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None
        self._on_question_answered: Optional[Callable[[str, int, str], None]] = None

    @property
    def name(self) -> str:
        return "clarification"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dictionary with:
                - channel_type: "console" (default) or "auto"
                - channel_config: Dict of config for the channel
        """
        config = config or {}
        channel_type = config.get("channel_type", "console")
        channel_config = config.get("channel_config", {})

        self._channel = create_channel(channel_type, **channel_config)
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        self._channel = None
        self._initialized = False

    def set_clarification_hooks(
        self,
        on_requested: Optional[Callable[[str, List[str]], None]] = None,
        on_resolved: Optional[Callable[[str], None]] = None,
        on_question_displayed: Optional[Callable[[str, int, int, List[str]], None]] = None,
        on_question_answered: Optional[Callable[[str, int, str], None]] = None
    ) -> None:
        """Set hooks for clarification lifecycle events.

        These hooks enable UI integration by notifying when clarification
        requests start and complete.

        Args:
            on_requested: Called when clarification session starts.
                Signature: (tool_name, prompt_lines) -> None
            on_resolved: Called when clarification is resolved.
                Signature: (tool_name) -> None
            on_question_displayed: Called when each question is shown.
                Signature: (tool_name, question_index, total_questions, question_lines) -> None
            on_question_answered: Called when user answers a question.
                Signature: (tool_name, question_index, answer_summary) -> None
        """
        self._on_clarification_requested = on_requested
        self._on_clarification_resolved = on_resolved
        self._on_question_displayed = on_question_displayed
        self._on_question_answered = on_question_answered

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the tool declarations for Vertex AI."""
        return [
            ToolSchema(
                name="request_clarification",
                description=(
                    "Request clarification from the user by asking one or more questions. "
                    "Use this when you need more information to proceed with a task. "
                    "Each question can be single-choice (pick one), multiple-choice (pick many), "
                    "or free-text (open response). Provide context explaining why you need "
                    "this information. Questions and choices are identified by their ordinal "
                    "position (1-based)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": (
                                "Brief explanation of why you need clarification. "
                                "This helps the user understand the purpose of the questions."
                            ),
                        },
                        "questions": {
                            "type": "array",
                            "description": (
                                "List of questions to ask. Questions are numbered automatically "
                                "(1, 2, 3...). Each question's choices are also numbered."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "The question text to display",
                                    },
                                    "question_type": {
                                        "type": "string",
                                        "enum": [
                                            "single_choice",
                                            "multiple_choice",
                                            "free_text",
                                        ],
                                        "description": (
                                            "Type of question: 'single_choice' (pick one), "
                                            "'multiple_choice' (pick many), or 'free_text' "
                                            "(open response). Defaults to 'single_choice'."
                                        ),
                                    },
                                    "choices": {
                                        "type": "array",
                                        "description": (
                                            "Available choices (for single/multiple choice). "
                                            "Choices are numbered 1, 2, 3... automatically."
                                        ),
                                        "items": {
                                            "type": "string",
                                            "description": "Choice text",
                                        },
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": (
                                            "Whether an answer is required (default: true). "
                                            "Optional questions can be skipped."
                                        ),
                                    },
                                    "default_choice": {
                                        "type": "integer",
                                        "description": (
                                            "1-based index of the default choice. "
                                            "Used if the user doesn't select anything."
                                        ),
                                    },
                                },
                                "required": ["text"],
                            },
                            "minItems": 1,
                        },
                    },
                    "required": ["context", "questions"],
                },
            )
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the mapping of tool names to executor functions."""
        return {"request_clarification": self._execute_clarification}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the AI model."""
        return """## Clarification Tool

You have access to a `request_clarification` tool that allows you to ask the user questions when you need more information.

**IMPORTANT**: When you need to ask the user a question, you MUST use the `request_clarification` tool. Do NOT ask questions directly in your text response - always use the tool instead. This ensures a consistent user experience and proper input handling.

### When to use:
- When the user's request is ambiguous
- When you need to choose between multiple valid approaches
- When configuration or preferences affect the outcome
- When you're missing critical information

### Best practices:
1. Group related questions together in a single request
2. Provide clear context explaining why you need the information
3. Use appropriate question types:
   - `single_choice`: When exactly one option must be selected
   - `multiple_choice`: When zero or more options can be selected
   - `free_text`: When you need an open-ended response
4. Set reasonable defaults when possible using `default_choice` (1-based index)
5. Mark questions as optional (`required: false`) when appropriate

### Example usage:
```json
{
  "context": "I need to know your deployment preferences to set up the configuration correctly.",
  "questions": [
    {
      "text": "Which environment should I configure?",
      "question_type": "single_choice",
      "choices": ["Development", "Staging", "Production"],
      "default_choice": 1
    },
    {
      "text": "Which optional features would you like enabled?",
      "question_type": "multiple_choice",
      "choices": ["Verbose logging", "Metrics collection", "Distributed tracing"],
      "required": false
    },
    {
      "text": "Any additional notes for the deployment?",
      "question_type": "free_text",
      "required": false
    }
  ]
}
```

### Response format:
The tool returns responses keyed by question number (1-based):
```json
{
  "responses": {
    "1": {"selected": 1, "text": "Development", "type": "single_choice"},
    "2": {"selected": [1, 3], "texts": ["Verbose logging", "Distributed tracing"], "type": "multiple_choice"},
    "3": {"value": "Please enable debug mode", "type": "free_text"}
  }
}
```
"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of tools that don't require permission approval.

        Clarification requests are inherently user-interactive, so they
        don't need additional permission checks.
        """
        return ["request_clarification"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands.

        This plugin only provides model tools, no direct user commands.
        """
        return []

    def _execute_clarification(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a clarification request.

        Args:
            args: Tool arguments containing context and questions

        Returns:
            Dict with either 'responses' (answers) or 'error'
        """
        if not self._initialized or not self._channel:
            return {"error": "Plugin not initialized"}

        try:
            # Parse the request
            request = self._parse_request(args)

            # Emit clarification requested hook (for initial context, not all questions)
            if self._on_clarification_requested:
                # Just send context, not all questions - questions come one by one
                context_lines = []
                if request.context:
                    context_lines.append(f"Context: {request.context}")
                self._on_clarification_requested("request_clarification", context_lines)

            # Get user responses via the channel, passing per-question hooks
            response = self._channel.request_clarification(
                request,
                on_question_displayed=self._on_question_displayed,
                on_question_answered=self._on_question_answered
            )

            # Emit clarification resolved hook
            if self._on_clarification_resolved:
                self._on_clarification_resolved("request_clarification")

            # Format the response for the model
            if response.cancelled:
                return {
                    "cancelled": True,
                    "message": "User cancelled the clarification request.",
                }

            # Build a structured response keyed by question number
            result = {"responses": {}}

            for answer in response.answers:
                q_key = str(answer.question_index)
                question = request.questions[answer.question_index - 1] if answer.question_index <= len(request.questions) else None

                if answer.skipped:
                    result["responses"][q_key] = {"skipped": True}
                elif answer.free_text is not None:
                    result["responses"][q_key] = {
                        "value": answer.free_text,
                        "type": "free_text",
                    }
                else:
                    # Get choice texts for selected indices
                    choice_texts = []
                    if question:
                        for idx in answer.selected_choices:
                            if 1 <= idx <= len(question.choices):
                                choice_texts.append(question.choices[idx - 1].text)

                    is_multiple_choice = (
                        question
                        and question.question_type == QuestionType.MULTIPLE_CHOICE
                    )

                    if is_multiple_choice:
                        result["responses"][q_key] = {
                            "selected": answer.selected_choices,
                            "texts": choice_texts,
                            "type": "multiple_choice",
                        }
                    else:
                        result["responses"][q_key] = {
                            "selected": answer.selected_choices[0] if answer.selected_choices else None,
                            "text": choice_texts[0] if choice_texts else None,
                            "type": "single_choice",
                        }

            return result

        except Exception as e:
            return {"error": f"Failed to process clarification request: {str(e)}"}

    def _build_prompt_lines(self, request: ClarificationRequest) -> List[str]:
        """Build prompt lines for UI display from request info.

        Args:
            request: The clarification request.

        Returns:
            List of strings representing the clarification prompt.
        """
        lines = []

        if request.context:
            lines.append(f"Context: {request.context}")
            lines.append("")

        for i, question in enumerate(request.questions, 1):
            req_marker = "*" if question.required else ""
            lines.append(f"Q{i}{req_marker}: {question.text}")
            if question.choices:
                for j, choice in enumerate(question.choices, 1):
                    lines.append(f"  {j}. {choice.text}")

        return lines

    def _parse_request(self, args: Dict[str, Any]) -> ClarificationRequest:
        """Parse tool arguments into a ClarificationRequest."""
        context = args.get("context", "")
        questions_data = args.get("questions", [])

        questions = []
        for q_data in questions_data:
            # Choices can be either strings or dicts with text
            choices_raw = q_data.get("choices", [])
            choices = []
            for c in choices_raw:
                if isinstance(c, str):
                    choices.append(Choice(text=c))
                elif isinstance(c, dict):
                    choices.append(Choice(text=c.get("text", "")))

            question_type_str = q_data.get("question_type", "single_choice")
            try:
                question_type = QuestionType(question_type_str)
            except ValueError:
                question_type = QuestionType.SINGLE_CHOICE

            question = Question(
                text=q_data.get("text", ""),
                question_type=question_type,
                choices=choices,
                required=q_data.get("required", True),
                default_choice=q_data.get("default_choice"),
            )
            questions.append(question)

        return ClarificationRequest(context=context, questions=questions)

    # Interactivity protocol methods

    def supports_interactivity(self) -> bool:
        """Clarification plugin requires user interaction for answering questions.

        Returns:
            True - clarification plugin has interactive question prompts.
        """
        return True

    def get_supported_channels(self) -> List[str]:
        """Return list of channel types supported by clarification plugin.

        Returns:
            List of supported channel types: console, queue, auto.
        """
        return ["console", "queue", "auto"]

    def set_channel(
        self,
        channel_type: str,
        channel_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the interaction channel for clarification prompts.

        Args:
            channel_type: One of: console, queue, auto
            channel_config: Optional channel-specific configuration

        Raises:
            ValueError: If channel_type is not supported
        """
        if channel_type not in self.get_supported_channels():
            raise ValueError(
                f"Channel type '{channel_type}' not supported. "
                f"Supported: {self.get_supported_channels()}"
            )

        # Create the channel with config
        from .channels import create_channel
        self._channel = create_channel(channel_type, **(channel_config or {}))


def create_plugin() -> ClarificationPlugin:
    """Factory function to create a ClarificationPlugin instance."""
    return ClarificationPlugin()
