"""Tests for clarification plugin data models."""

import pytest

from ..models import (
    Answer,
    Choice,
    ClarificationRequest,
    ClarificationResponse,
    Question,
    QuestionType,
)


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_type_values(self):
        assert QuestionType.SINGLE_CHOICE.value == "single_choice"
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.FREE_TEXT.value == "free_text"


class TestChoice:
    """Tests for Choice dataclass."""

    def test_create_choice(self):
        choice = Choice(id="a", text="Option A")
        assert choice.id == "a"
        assert choice.text == "Option A"

    def test_choice_to_dict(self):
        choice = Choice(id="1", text="First option")
        data = choice.to_dict()

        assert data["id"] == "1"
        assert data["text"] == "First option"

    def test_choice_from_dict(self):
        data = {"id": "b", "text": "Option B"}
        choice = Choice.from_dict(data)

        assert choice.id == "b"
        assert choice.text == "Option B"


class TestQuestion:
    """Tests for Question dataclass."""

    def test_create_question_defaults(self):
        question = Question(id="q1", text="What is your preference?")

        assert question.id == "q1"
        assert question.text == "What is your preference?"
        assert question.question_type == QuestionType.SINGLE_CHOICE
        assert question.choices == []
        assert question.required is True
        assert question.default_choice_id is None

    def test_create_question_with_choices(self):
        choices = [
            Choice(id="a", text="Option A"),
            Choice(id="b", text="Option B"),
        ]
        question = Question(
            id="q1",
            text="Pick one",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=choices,
        )

        assert len(question.choices) == 2
        assert question.choices[0].id == "a"

    def test_create_question_multiple_choice(self):
        question = Question(
            id="q1",
            text="Select all that apply",
            question_type=QuestionType.MULTIPLE_CHOICE,
            choices=[
                Choice(id="1", text="Feature 1"),
                Choice(id="2", text="Feature 2"),
            ],
        )

        assert question.question_type == QuestionType.MULTIPLE_CHOICE

    def test_create_question_free_text(self):
        question = Question(
            id="q1",
            text="Describe your requirements",
            question_type=QuestionType.FREE_TEXT,
        )

        assert question.question_type == QuestionType.FREE_TEXT

    def test_create_optional_question(self):
        question = Question(
            id="q1",
            text="Any comments?",
            required=False,
        )

        assert question.required is False

    def test_create_question_with_default(self):
        question = Question(
            id="q1",
            text="Environment",
            choices=[
                Choice(id="dev", text="Development"),
                Choice(id="prod", text="Production"),
            ],
            default_choice_id="dev",
        )

        assert question.default_choice_id == "dev"

    def test_question_to_dict(self):
        question = Question(
            id="q1",
            text="Pick one",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=[Choice(id="a", text="A")],
            required=True,
            default_choice_id="a",
        )

        data = question.to_dict()

        assert data["id"] == "q1"
        assert data["text"] == "Pick one"
        assert data["question_type"] == "single_choice"
        assert len(data["choices"]) == 1
        assert data["required"] is True
        assert data["default_choice_id"] == "a"

    def test_question_from_dict(self):
        data = {
            "id": "q2",
            "text": "Select features",
            "question_type": "multiple_choice",
            "choices": [
                {"id": "1", "text": "Feature 1"},
                {"id": "2", "text": "Feature 2"},
            ],
            "required": False,
            "default_choice_id": "1,2",
        }

        question = Question.from_dict(data)

        assert question.id == "q2"
        assert question.text == "Select features"
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert len(question.choices) == 2
        assert question.required is False
        assert question.default_choice_id == "1,2"

    def test_question_from_dict_defaults(self):
        data = {"id": "q1", "text": "Simple question"}

        question = Question.from_dict(data)

        assert question.question_type == QuestionType.SINGLE_CHOICE
        assert question.choices == []
        assert question.required is True


class TestClarificationRequest:
    """Tests for ClarificationRequest dataclass."""

    def test_create_request(self):
        request = ClarificationRequest(
            context="I need more info",
            questions=[
                Question(id="q1", text="Question 1"),
                Question(id="q2", text="Question 2"),
            ],
        )

        assert request.context == "I need more info"
        assert len(request.questions) == 2

    def test_create_request_empty(self):
        request = ClarificationRequest(context="")

        assert request.context == ""
        assert request.questions == []

    def test_request_to_dict(self):
        request = ClarificationRequest(
            context="Context here",
            questions=[Question(id="q1", text="Q1")],
        )

        data = request.to_dict()

        assert data["context"] == "Context here"
        assert len(data["questions"]) == 1

    def test_request_from_dict(self):
        data = {
            "context": "Need info",
            "questions": [
                {"id": "q1", "text": "First question"},
                {"id": "q2", "text": "Second question"},
            ],
        }

        request = ClarificationRequest.from_dict(data)

        assert request.context == "Need info"
        assert len(request.questions) == 2
        assert request.questions[0].id == "q1"


class TestAnswer:
    """Tests for Answer dataclass."""

    def test_create_answer_single_choice(self):
        answer = Answer(
            question_id="q1",
            selected_choice_ids=["a"],
        )

        assert answer.question_id == "q1"
        assert answer.selected_choice_ids == ["a"]
        assert answer.free_text is None
        assert answer.skipped is False

    def test_create_answer_multiple_choice(self):
        answer = Answer(
            question_id="q1",
            selected_choice_ids=["1", "3", "5"],
        )

        assert len(answer.selected_choice_ids) == 3

    def test_create_answer_free_text(self):
        answer = Answer(
            question_id="q1",
            free_text="This is my answer",
        )

        assert answer.free_text == "This is my answer"
        assert answer.selected_choice_ids == []

    def test_create_answer_skipped(self):
        answer = Answer(
            question_id="q1",
            skipped=True,
        )

        assert answer.skipped is True

    def test_answer_to_dict(self):
        answer = Answer(
            question_id="q1",
            selected_choice_ids=["a", "b"],
        )

        data = answer.to_dict()

        assert data["question_id"] == "q1"
        assert data["selected_choice_ids"] == ["a", "b"]
        assert data["free_text"] is None
        assert data["skipped"] is False

    def test_answer_from_dict(self):
        data = {
            "question_id": "q2",
            "selected_choice_ids": ["x"],
            "free_text": None,
            "skipped": False,
        }

        answer = Answer.from_dict(data)

        assert answer.question_id == "q2"
        assert answer.selected_choice_ids == ["x"]


class TestClarificationResponse:
    """Tests for ClarificationResponse dataclass."""

    def test_create_response(self):
        response = ClarificationResponse(
            answers=[
                Answer(question_id="q1", selected_choice_ids=["a"]),
                Answer(question_id="q2", free_text="my answer"),
            ],
        )

        assert len(response.answers) == 2
        assert response.cancelled is False

    def test_create_cancelled_response(self):
        response = ClarificationResponse(cancelled=True)

        assert response.cancelled is True
        assert response.answers == []

    def test_response_to_dict(self):
        response = ClarificationResponse(
            answers=[Answer(question_id="q1", selected_choice_ids=["a"])],
        )

        data = response.to_dict()

        assert len(data["answers"]) == 1
        assert data["cancelled"] is False

    def test_response_from_dict(self):
        data = {
            "answers": [
                {"question_id": "q1", "selected_choice_ids": ["b"]},
            ],
            "cancelled": False,
        }

        response = ClarificationResponse.from_dict(data)

        assert len(response.answers) == 1
        assert response.answers[0].question_id == "q1"

    def test_get_answer(self):
        response = ClarificationResponse(
            answers=[
                Answer(question_id="q1", selected_choice_ids=["a"]),
                Answer(question_id="q2", free_text="text"),
                Answer(question_id="q3", skipped=True),
            ],
        )

        answer1 = response.get_answer("q1")
        assert answer1 is not None
        assert answer1.selected_choice_ids == ["a"]

        answer2 = response.get_answer("q2")
        assert answer2 is not None
        assert answer2.free_text == "text"

        answer3 = response.get_answer("q3")
        assert answer3 is not None
        assert answer3.skipped is True

    def test_get_answer_not_found(self):
        response = ClarificationResponse(
            answers=[Answer(question_id="q1", selected_choice_ids=["a"])],
        )

        answer = response.get_answer("nonexistent")

        assert answer is None
