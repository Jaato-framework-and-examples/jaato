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
    """A single choice option for a question."""

    id: str  # Unique identifier for this choice (e.g., "a", "b", "1", "2")
    text: str  # Display text for the choice

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "Choice":
        return cls(id=data["id"], text=data["text"])


@dataclass
class Question:
    """A question that can be asked to the user for clarification."""

    id: str  # Unique identifier for this question
    text: str  # The question text
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    choices: List[Choice] = field(default_factory=list)
    required: bool = True  # Whether an answer is required
    default_choice_id: Optional[str] = None  # Default choice if not answered

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "question_type": self.question_type.value,
            "choices": [c.to_dict() for c in self.choices],
            "required": self.required,
            "default_choice_id": self.default_choice_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(
            id=data["id"],
            text=data["text"],
            question_type=QuestionType(data.get("question_type", "single_choice")),
            choices=[Choice.from_dict(c) for c in data.get("choices", [])],
            required=data.get("required", True),
            default_choice_id=data.get("default_choice_id"),
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
    """User's answer to a single question."""

    question_id: str  # ID of the question being answered
    selected_choice_ids: List[str] = field(default_factory=list)  # For choice questions
    free_text: Optional[str] = None  # For free text questions
    skipped: bool = False  # True if user skipped a non-required question

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "selected_choice_ids": self.selected_choice_ids,
            "free_text": self.free_text,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Answer":
        return cls(
            question_id=data["question_id"],
            selected_choice_ids=data.get("selected_choice_ids", []),
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

    def get_answer(self, question_id: str) -> Optional[Answer]:
        """Get the answer for a specific question by ID."""
        for answer in self.answers:
            if answer.question_id == question_id:
                return answer
        return None
