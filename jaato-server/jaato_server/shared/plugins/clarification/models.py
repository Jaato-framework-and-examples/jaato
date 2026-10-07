"""Data models for the clarification plugin."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import Attachment

from .attachments import (
    attachment_from_wire,
    attachment_to_wire,
)


class QuestionType(str, Enum):
    """Type of question determining how choices are presented and answered."""

    SINGLE_CHOICE = "single_choice"  # User selects exactly one option
    MULTIPLE_CHOICE = "multiple_choice"  # User can select multiple options
    FREE_TEXT = "free_text"  # User provides free-form text response


@dataclass
class Choice:
    """A single choice option for a question.

    Choices are identified by their ordinal position (1-based index).

    Attributes:
        text: Display text for the choice.
        expects_attachment: This branch of the question expects the user
            to attach a file (#989).  It is a per-CHOICE declaration, not
            a per-question one, because the case that motivates it splits
            within a question::

                How should we design this?
                  1. You attach a screenshot of the design   <- expects one
                  2. We discuss the design                   <- does not

            Purely declarative, and deliberately so: nothing refuses an
            answer that selects such a choice without attaching, and
            nothing refuses an attachment on a choice that does not
            declare it -- ``Answer.attachments`` is orthogonal to
            ``selected_choices``, and the ordinal says which branch while
            the attachment says what content.  What this field buys is a
            client that can render an attach affordance at the right
            place, instead of the agent writing "attach a screenshot" as
            prose that nothing downstream can act on.  Reaches a client on
            ``ClarificationBatchEvent.questions[].choices[]``.
    """

    text: str  # Display text for the choice
    expects_attachment: bool = False

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {"text": self.text}
        if self.expects_attachment:
            # Omitted when false so an existing consumer's dict is
            # byte-identical to what it has always received.
            data["expects_attachment"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Choice":
        # Support both new format (text only) and legacy format (with id)
        if isinstance(data, str):
            return cls(text=data)
        return cls(
            text=data.get("text", ""),
            expects_attachment=bool(data.get("expects_attachment", False)),
        )


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

    Attributes:
        question_index: 1-based index of the question being answered.
        selected_choices: 1-based indices of the selected choices.
        free_text: The typed answer, for a ``free_text`` question.  ``""``
            is a real (empty) answer; ``None`` means this question was not
            answered with text.
        skipped: The user declined an OPTIONAL question.  Never true for
            an answer that carries attachments -- an attachment IS content
            (#838), so a voice note with no typed text is an answer, not a
            skip.  ``ClarificationChannel._parse_answer_with_attachments``
            is what enforces that when it builds the answer.
        attachments: Media the user attached to THIS answer (#989) --
            audio, an image, a document.  Set by the channel that received
            them and consumed by ``ClarificationPlugin``, which folds them
            onto the tool result so they reach the model through
            ``ToolResult.attachments``.  Empty on every channel that has
            no way to receive media (``ConsoleChannel``, ``QueueChannel``,
            ``ParentBridgedChannel`` -- a subagent's answer arrives as
            injected TEXT and has no representation for bytes at all).

            Deliberately a field of the ANSWER rather than of the
            free-text branch: choosing "1. you attach a screenshot" and
            attaching the screenshot is ``selected_choices=[1]`` **plus**
            an image.  The two are different axes, not alternatives.
    """

    question_index: int  # 1-based index of the question
    selected_choices: List[int] = field(default_factory=list)  # 1-based indices
    free_text: Optional[str] = None  # For free text questions
    skipped: bool = False  # True if user skipped an optional question
    attachments: List[Attachment] = field(default_factory=list)

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {
            "question_index": self.question_index,
            "selected_choices": self.selected_choices,
            "free_text": self.free_text,
            "skipped": self.skipped,
        }
        if self.attachments:
            # Rendered as the canonical wire dicts (base64 payload), so the
            # whole answer stays JSON-serialisable the way every other
            # field here already is.  Omitted when empty: an existing
            # consumer's dict is unchanged.
            data["attachments"] = [
                attachment_to_wire(a) for a in self.attachments
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Answer":
        raw = data.get("attachments") or []
        restored = [attachment_from_wire(a) for a in raw]
        return cls(
            question_index=data.get("question_index", 0),
            selected_choices=data.get("selected_choices", []),
            free_text=data.get("free_text"),
            skipped=data.get("skipped", False),
            attachments=[a for a in restored if a is not None],
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
