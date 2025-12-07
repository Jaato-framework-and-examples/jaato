"""Clarification plugin for requesting user input with multiple questions and choices.

This plugin allows the AI model to request clarifications from the user when it needs
more information to proceed with a task. It supports:
- Multiple questions per request
- Multiple choice types: single choice, multiple choice, or free text
- Optional questions with defaults
- Context explanation for why clarification is needed
"""

PLUGIN_KIND = "tool"

from .actors import AutoActor, ClarificationActor, ConsoleActor, create_actor
from .models import (
    Answer,
    Choice,
    ClarificationRequest,
    ClarificationResponse,
    Question,
    QuestionType,
)
from .plugin import ClarificationPlugin, create_plugin

__all__ = [
    # Plugin
    "ClarificationPlugin",
    "create_plugin",
    "PLUGIN_KIND",
    # Models
    "Choice",
    "Question",
    "QuestionType",
    "ClarificationRequest",
    "Answer",
    "ClarificationResponse",
    # Actors
    "ClarificationActor",
    "ConsoleActor",
    "AutoActor",
    "create_actor",
]
