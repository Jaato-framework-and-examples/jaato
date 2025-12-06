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
                    "this information."
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
                            "description": "List of questions to ask the user",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": (
                                            "Unique identifier for this question "
                                            "(e.g., 'q1', 'deployment_env')"
                                        ),
                                    },
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
                                            "(open response)"
                                        ),
                                        "default": "single_choice",
                                    },
                                    "choices": {
                                        "type": "array",
                                        "description": (
                                            "Available choices for single/multiple choice "
                                            "questions. Not needed for free_text."
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {
                                                    "type": "string",
                                                    "description": (
                                                        "Short identifier for the choice "
                                                        "(e.g., 'a', 'b', '1', '2')"
                                                    ),
                                                },
                                                "text": {
                                                    "type": "string",
                                                    "description": "Display text for this choice",
                                                },
                                            },
                                            "required": ["id", "text"],
                                        },
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": (
                                            "Whether an answer is required. "
                                            "Optional questions can be skipped."
                                        ),
                                        "default": True,
                                    },
                                    "default_choice_id": {
                                        "type": "string",
                                        "description": (
                                            "ID of default choice if user doesn't select one. "
                                            "For multiple_choice, use comma-separated IDs."
                                        ),
                                    },
                                },
                                "required": ["id", "text"],
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
4. Provide meaningful choice IDs (e.g., 'a', 'b' or descriptive like 'dev', 'prod')
5. Set reasonable defaults when possible
6. Mark questions as optional (`required: false`) when appropriate

### Example usage:
```json
{
  "context": "I need to know your deployment preferences to set up the configuration correctly.",
  "questions": [
    {
      "id": "env",
      "text": "Which environment should I configure?",
      "question_type": "single_choice",
      "choices": [
        {"id": "dev", "text": "Development"},
        {"id": "staging", "text": "Staging"},
        {"id": "prod", "text": "Production"}
      ],
      "default_choice_id": "dev"
    },
    {
      "id": "features",
      "text": "Which optional features would you like enabled?",
      "question_type": "multiple_choice",
      "choices": [
        {"id": "logging", "text": "Verbose logging"},
        {"id": "metrics", "text": "Metrics collection"},
        {"id": "tracing", "text": "Distributed tracing"}
      ],
      "required": false
    }
  ]
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

            # Build a structured response
            result = {"responses": {}}

            for answer in response.answers:
                if answer.skipped:
                    result["responses"][answer.question_id] = {
                        "skipped": True,
                    }
                elif answer.free_text is not None:
                    result["responses"][answer.question_id] = {
                        "value": answer.free_text,
                        "type": "free_text",
                    }
                else:
                    # Find the question to get choice texts and type
                    question = next(
                        (q for q in request.questions if q.id == answer.question_id),
                        None,
                    )
                    choice_texts = []
                    if question:
                        choice_map = {c.id: c.text for c in question.choices}
                        choice_texts = [
                            choice_map.get(cid, cid)
                            for cid in answer.selected_choice_ids
                        ]

                    # Use question type to determine response format
                    is_multiple_choice = (
                        question
                        and question.question_type == QuestionType.MULTIPLE_CHOICE
                    )

                    if is_multiple_choice:
                        result["responses"][answer.question_id] = {
                            "values": answer.selected_choice_ids,
                            "texts": choice_texts,
                            "type": "multiple_choice",
                        }
                    else:
                        result["responses"][answer.question_id] = {
                            "value": answer.selected_choice_ids[0] if answer.selected_choice_ids else None,
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
            choices = [
                Choice(id=c["id"], text=c["text"])
                for c in q_data.get("choices", [])
            ]

            question_type_str = q_data.get("question_type", "single_choice")
            try:
                question_type = QuestionType(question_type_str)
            except ValueError:
                question_type = QuestionType.SINGLE_CHOICE

            question = Question(
                id=q_data["id"],
                text=q_data["text"],
                question_type=question_type,
                choices=choices,
                required=q_data.get("required", True),
                default_choice_id=q_data.get("default_choice_id"),
            )
            questions.append(question)

        return ClarificationRequest(context=context, questions=questions)


def create_plugin() -> ClarificationPlugin:
    """Factory function to create a ClarificationPlugin instance."""
    return ClarificationPlugin()
