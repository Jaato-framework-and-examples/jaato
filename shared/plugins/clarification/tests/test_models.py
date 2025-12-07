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
        choice = Choice(text="Option A")
        assert choice.text == "Option A"

    def test_choice_to_dict(self):
        choice = Choice(text="First option")
        data = choice.to_dict()

        assert data["text"] == "First option"

    def test_choice_from_dict(self):
        data = {"text": "Option B"}
        choice = Choice.from_dict(data)

        assert choice.text == "Option B"

    def test_choice_from_string(self):
        choice = Choice.from_dict("Option C")

        assert choice.text == "Option C"


class TestQuestion:
    """Tests for Question dataclass."""

    def test_create_question_defaults(self):
        question = Question(text="What is your preference?")

        assert question.text == "What is your preference?"
        assert question.question_type == QuestionType.SINGLE_CHOICE
        assert question.choices == []
        assert question.required is True
        assert question.default_choice is None

    def test_create_question_with_choices(self):
        choices = [
            Choice(text="Option A"),
            Choice(text="Option B"),
        ]
        question = Question(
            text="Pick one",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=choices,
        )

        assert len(question.choices) == 2
        assert question.choices[0].text == "Option A"

    def test_create_question_multiple_choice(self):
        question = Question(
            text="Select all that apply",
            question_type=QuestionType.MULTIPLE_CHOICE,
            choices=[
                Choice(text="Feature 1"),
                Choice(text="Feature 2"),
            ],
        )

        assert question.question_type == QuestionType.MULTIPLE_CHOICE

    def test_create_question_free_text(self):
        question = Question(
            text="Describe your requirements",
            question_type=QuestionType.FREE_TEXT,
        )

        assert question.question_type == QuestionType.FREE_TEXT

    def test_create_optional_question(self):
        question = Question(
            text="Any comments?",
            required=False,
        )

        assert question.required is False

    def test_create_question_with_default(self):
        question = Question(
            text="Environment",
            choices=[
                Choice(text="Development"),
                Choice(text="Production"),
            ],
            default_choice=1,
        )

        assert question.default_choice == 1

    def test_question_to_dict(self):
        question = Question(
            text="Pick one",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=[Choice(text="A")],
            required=True,
            default_choice=1,
        )

        data = question.to_dict()

        assert data["text"] == "Pick one"
        assert data["question_type"] == "single_choice"
        assert len(data["choices"]) == 1
        assert data["required"] is True
        assert data["default_choice"] == 1

    def test_question_from_dict(self):
        data = {
            "text": "Select features",
            "question_type": "multiple_choice",
            "choices": [
                {"text": "Feature 1"},
                {"text": "Feature 2"},
            ],
            "required": False,
            "default_choice": 2,
        }

        question = Question.from_dict(data)

        assert question.text == "Select features"
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert len(question.choices) == 2
        assert question.required is False
        assert question.default_choice == 2

    def test_question_from_dict_defaults(self):
        data = {"text": "Simple question"}

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
                Question(text="Question 1"),
                Question(text="Question 2"),
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
            questions=[Question(text="Q1")],
        )

        data = request.to_dict()

        assert data["context"] == "Context here"
        assert len(data["questions"]) == 1

    def test_request_from_dict(self):
        data = {
            "context": "Need info",
            "questions": [
                {"text": "First question"},
                {"text": "Second question"},
            ],
        }

        request = ClarificationRequest.from_dict(data)

        assert request.context == "Need info"
        assert len(request.questions) == 2
        assert request.questions[0].text == "First question"


class TestAnswer:
    """Tests for Answer dataclass."""

    def test_create_answer_single_choice(self):
        answer = Answer(
            question_index=1,
            selected_choices=[1],
        )

        assert answer.question_index == 1
        assert answer.selected_choices == [1]
        assert answer.free_text is None
        assert answer.skipped is False

    def test_create_answer_multiple_choice(self):
        answer = Answer(
            question_index=1,
            selected_choices=[1, 3, 5],
        )

        assert len(answer.selected_choices) == 3

    def test_create_answer_free_text(self):
        answer = Answer(
            question_index=1,
            free_text="This is my answer",
        )

        assert answer.free_text == "This is my answer"
        assert answer.selected_choices == []

    def test_create_answer_skipped(self):
        answer = Answer(
            question_index=1,
            skipped=True,
        )

        assert answer.skipped is True

    def test_answer_to_dict(self):
        answer = Answer(
            question_index=1,
            selected_choices=[1, 2],
        )

        data = answer.to_dict()

        assert data["question_index"] == 1
        assert data["selected_choices"] == [1, 2]
        assert data["free_text"] is None
        assert data["skipped"] is False

    def test_answer_from_dict(self):
        data = {
            "question_index": 2,
            "selected_choices": [3],
            "free_text": None,
            "skipped": False,
        }

        answer = Answer.from_dict(data)

        assert answer.question_index == 2
        assert answer.selected_choices == [3]


class TestClarificationResponse:
    """Tests for ClarificationResponse dataclass."""

    def test_create_response(self):
        response = ClarificationResponse(
            answers=[
                Answer(question_index=1, selected_choices=[1]),
                Answer(question_index=2, free_text="my answer"),
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
            answers=[Answer(question_index=1, selected_choices=[1])],
        )

        data = response.to_dict()

        assert len(data["answers"]) == 1
        assert data["cancelled"] is False

    def test_response_from_dict(self):
        data = {
            "answers": [
                {"question_index": 1, "selected_choices": [2]},
            ],
            "cancelled": False,
        }

        response = ClarificationResponse.from_dict(data)

        assert len(response.answers) == 1
        assert response.answers[0].question_index == 1

    def test_get_answer(self):
        response = ClarificationResponse(
            answers=[
                Answer(question_index=1, selected_choices=[1]),
                Answer(question_index=2, free_text="text"),
                Answer(question_index=3, skipped=True),
            ],
        )

        answer1 = response.get_answer(1)
        assert answer1 is not None
        assert answer1.selected_choices == [1]

        answer2 = response.get_answer(2)
        assert answer2 is not None
        assert answer2.free_text == "text"

        answer3 = response.get_answer(3)
        assert answer3 is not None
        assert answer3.skipped is True

    def test_get_answer_not_found(self):
        response = ClarificationResponse(
            answers=[Answer(question_index=1, selected_choices=[1])],
        )

        answer = response.get_answer(99)

        assert answer is None
