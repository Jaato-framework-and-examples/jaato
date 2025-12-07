"""Clarification plugin for requesting user input with multiple questions and choices."""

from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from ..base import UserCommand
from .actors import ClarificationActor, create_actor
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
        self._actor: Optional[ClarificationActor] = None

    @property
    def name(self) -> str:
        return "clarification"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dictionary with:
                - actor_type: "console" (default) or "auto"
                - actor_config: Dict of config for the actor
        """
        config = config or {}
        actor_type = config.get("actor_type", "console")
        actor_config = config.get("actor_config", {})

        self._actor = create_actor(actor_type, **actor_config)
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        self._actor = None
        self._initialized = False

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return the tool declarations for Vertex AI."""
        return [
            types.FunctionDeclaration(
                name="request_clarification",
                description=(
                    "Request clarification from the user by asking one or more questions. "
                    "Use this when you need more information to proceed with a task. "
                    "Each question can be single-choice (pick one), multiple-choice (pick many), "
                    "or free-text (open response). Provide context explaining why you need "
                    "this information. Questions and choices are identified by their ordinal "
                    "position (1-based)."
                ),
                parameters_json_schema={
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
        if not self._initialized or not self._actor:
            return {"error": "Plugin not initialized"}

        try:
            # Parse the request
            request = self._parse_request(args)

            # Get user responses via the actor
            response = self._actor.request_clarification(request)

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


def create_plugin() -> ClarificationPlugin:
    """Factory function to create a ClarificationPlugin instance."""
    return ClarificationPlugin()
