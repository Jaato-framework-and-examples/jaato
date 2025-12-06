"""Tests for clarification plugin actors."""

import io
import pytest

from ..actors import AutoActor, ConsoleActor, create_actor
from ..models import (
    Choice,
    ClarificationRequest,
    Question,
    QuestionType,
)


class TestConsoleActor:
    """Tests for ConsoleActor."""

    def test_create_console_actor(self):
        actor = ConsoleActor()
        assert actor is not None

    def test_single_choice_question(self):
        input_stream = io.StringIO("1\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="Option A"),
                        Choice(text="Option B"),
                    ],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert response.answers[0].question_index == 1
        assert response.answers[0].selected_choices == [1]

    def test_multiple_choice_question(self):
        input_stream = io.StringIO("1, 3\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Select all that apply",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                        Choice(text="C"),
                    ],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert set(response.answers[0].selected_choices) == {1, 3}

    def test_free_text_question(self):
        input_stream = io.StringIO("This is my answer\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Describe your needs",
                    question_type=QuestionType.FREE_TEXT,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert response.answers[0].free_text == "This is my answer"

    def test_multiple_questions(self):
        input_stream = io.StringIO("1\nHello world\n1,2\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Need multiple inputs",
            questions=[
                Question(
                    text="Single choice",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                    ],
                ),
                Question(
                    text="Free text",
                    question_type=QuestionType.FREE_TEXT,
                ),
                Question(
                    text="Multiple choice",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    choices=[
                        Choice(text="One"),
                        Choice(text="Two"),
                        Choice(text="Three"),
                    ],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 3
        assert response.answers[0].selected_choices == [1]
        assert response.answers[1].free_text == "Hello world"
        assert set(response.answers[2].selected_choices) == {1, 2}

    def test_cancel_request(self):
        input_stream = io.StringIO("cancel\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[Choice(text="A")],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert response.cancelled

    def test_default_choice(self):
        input_stream = io.StringIO("\n")  # Empty input uses default
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                    ],
                    default_choice=2,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [2]

    def test_optional_question_skipped(self):
        input_stream = io.StringIO("\n")  # Empty input skips optional
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Optional question",
                    question_type=QuestionType.FREE_TEXT,
                    required=False,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].skipped is True

    def test_invalid_choice_retry(self):
        # First input is invalid (99), second is valid (1)
        input_stream = io.StringIO("99\n1\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                    ],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [1]

        output = output_stream.getvalue()
        assert "Invalid choice" in output

    def test_shows_required_status(self):
        input_stream = io.StringIO("1\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Required question",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[Choice(text="A")],
                    required=True,
                ),
            ],
        )

        actor.request_clarification(request)

        output = output_stream.getvalue()
        assert "*required" in output or "required" in output

    def test_shows_optional_status(self):
        input_stream = io.StringIO("\n")
        output_stream = io.StringIO()

        actor = ConsoleActor(
            input_stream=input_stream,
            output_stream=output_stream,
            use_colors=False,
        )

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Optional question",
                    question_type=QuestionType.FREE_TEXT,
                    required=False,
                ),
            ],
        )

        actor.request_clarification(request)

        output = output_stream.getvalue()
        assert "optional" in output


class TestAutoActor:
    """Tests for AutoActor."""

    def test_create_auto_actor(self):
        actor = AutoActor()
        assert actor is not None

    def test_auto_single_choice_with_default(self):
        actor = AutoActor()

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                    ],
                    default_choice=2,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [2]

    def test_auto_single_choice_first_option(self):
        actor = AutoActor()

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Pick one",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[
                        Choice(text="A"),
                        Choice(text="B"),
                    ],
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert response.answers[0].selected_choices == [1]

    def test_auto_multiple_choice_with_default(self):
        actor = AutoActor()

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Select features",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    choices=[
                        Choice(text="One"),
                        Choice(text="Two"),
                        Choice(text="Three"),
                    ],
                    default_choice=2,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert response.answers[0].selected_choices == [2]

    def test_auto_free_text(self):
        actor = AutoActor(default_free_text="custom response")

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Describe",
                    question_type=QuestionType.FREE_TEXT,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert response.answers[0].free_text == "custom response"

    def test_auto_multiple_questions(self):
        actor = AutoActor()

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Q1",
                    question_type=QuestionType.SINGLE_CHOICE,
                    choices=[Choice(text="A")],
                ),
                Question(
                    text="Q2",
                    question_type=QuestionType.FREE_TEXT,
                ),
            ],
        )

        response = actor.request_clarification(request)

        assert len(response.answers) == 2
        assert response.answers[0].selected_choices == [1]
        assert response.answers[1].free_text == "auto-response"


class TestCreateActor:
    """Tests for create_actor factory function."""

    def test_create_console_actor(self):
        actor = create_actor("console")
        assert isinstance(actor, ConsoleActor)

    def test_create_auto_actor(self):
        actor = create_actor("auto")
        assert isinstance(actor, AutoActor)

    def test_create_auto_actor_with_config(self):
        actor = create_actor("auto", default_free_text="test")
        assert isinstance(actor, AutoActor)

    def test_create_unknown_actor(self):
        with pytest.raises(ValueError, match="Unknown actor type"):
            create_actor("unknown")
