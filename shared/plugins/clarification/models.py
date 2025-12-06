"""Data models for the clarification plugin."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QuestionType(str, Enum):
    """Type of question determining how choices are presented and answered."""

    SINGLE_CHOICE = "single_choice"  # User selects exactly one option
    MULTIPLE_CHOICE = "multiple_choice"  # User can select multiple options
    FREE_TEXT = "free_text"  # User provides free-form text response


@dataclass
class Choice:
    """A single choice option for a question.

    Choices are identified by their ordinal position (1-based index).
    """

    text: str  # Display text for the choice

    def to_dict(self) -> dict:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "Choice":
        # Support both new format (text only) and legacy format (with id)
        if isinstance(data, str):
            return cls(text=data)
        return cls(text=data.get("text", ""))


@dataclass
class Question:
    """A question that can be asked to the user for clarification.

    Questions are identified by their ordinal position (1-based index).
    """

    text: str  # The question text
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    choices: List[Choice] = field(default_factory=list)
    required: bool = True  # Whether an answer is required
    default_choice: Optional[int] = None  # 1-based index of default choice

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "question_type": self.question_type.value,
            "choices": [c.to_dict() for c in self.choices],
            "required": self.required,
            "default_choice": self.default_choice,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        choices_data = data.get("choices", [])
        choices = [Choice.from_dict(c) for c in choices_data]

        question_type_str = data.get("question_type", "single_choice")
        try:
            question_type = QuestionType(question_type_str)
        except ValueError:
            question_type = QuestionType.SINGLE_CHOICE

        return cls(
            text=data.get("text", ""),
            question_type=question_type,
            choices=choices,
            required=data.get("required", True),
            default_choice=data.get("default_choice"),
        )


@dataclass
class ClarificationRequest:
    """A request containing one or more questions for the user."""

    context: str  # Brief context explaining why clarification is needed
    questions: List[Question] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClarificationRequest":
        return cls(
            context=data.get("context", ""),
            questions=[Question.from_dict(q) for q in data.get("questions", [])],
        )


@dataclass
class Answer:
    """User's answer to a single question.

    question_index: 1-based index of the question being answered
    selected_choices: List of 1-based indices of selected choices
    """

    question_index: int  # 1-based index of the question
    selected_choices: List[int] = field(default_factory=list)  # 1-based indices
    free_text: Optional[str] = None  # For free text questions
    skipped: bool = False  # True if user skipped an optional question

    def to_dict(self) -> dict:
        return {
            "question_index": self.question_index,
            "selected_choices": self.selected_choices,
            "free_text": self.free_text,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Answer":
        return cls(
            question_index=data.get("question_index", 0),
            selected_choices=data.get("selected_choices", []),
            free_text=data.get("free_text"),
            skipped=data.get("skipped", False),
        )


@dataclass
class ClarificationResponse:
    """Collection of user's answers to a clarification request."""

    answers: List[Answer] = field(default_factory=list)
    cancelled: bool = False  # True if user cancelled the entire clarification

    def to_dict(self) -> dict:
        return {
            "answers": [a.to_dict() for a in self.answers],
            "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClarificationResponse":
        return cls(
            answers=[Answer.from_dict(a) for a in data.get("answers", [])],
            cancelled=data.get("cancelled", False),
        )

    def get_answer(self, question_index: int) -> Optional[Answer]:
        """Get the answer for a specific question by 1-based index."""
        for answer in self.answers:
            if answer.question_index == question_index:
                return answer
        return None
