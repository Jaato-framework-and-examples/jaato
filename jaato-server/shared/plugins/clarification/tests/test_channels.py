"""Tests for clarification plugin channels."""

import io
import pytest

from ..channels import AutoChannel, ConsoleChannel, create_channel
from ..models import (
    Choice,
    ClarificationRequest,
    Question,
    QuestionType,
)


class TestConsoleChannel:
    """Tests for ConsoleChannel."""

    def test_create_console_channel(self):
        channel = ConsoleChannel()
        assert channel is not None

    def test_single_choice_question(self):
        input_stream = io.StringIO("1\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert response.answers[0].question_index == 1
        assert response.answers[0].selected_choices == [1]

    def test_multiple_choice_question(self):
        input_stream = io.StringIO("1, 3\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert set(response.answers[0].selected_choices) == {1, 3}

    def test_free_text_question(self):
        input_stream = io.StringIO("This is my answer\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 1
        assert response.answers[0].free_text == "This is my answer"

    def test_multiple_questions(self):
        input_stream = io.StringIO("1\nHello world\n1,2\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert len(response.answers) == 3
        assert response.answers[0].selected_choices == [1]
        assert response.answers[1].free_text == "Hello world"
        assert set(response.answers[2].selected_choices) == {1, 2}

    def test_cancel_request(self):
        input_stream = io.StringIO("cancel\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert response.cancelled

    def test_default_choice(self):
        input_stream = io.StringIO("\n")  # Empty input uses default
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [2]

    def test_optional_question_skipped(self):
        input_stream = io.StringIO("\n")  # Empty input skips optional
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].skipped is True

    def test_invalid_choice_retry(self):
        # First input is invalid (99), second is valid (1)
        input_stream = io.StringIO("99\n1\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [1]

        output = output_stream.getvalue()
        assert "Invalid choice" in output

    def test_shows_required_status(self):
        input_stream = io.StringIO("1\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        channel.request_clarification(request)

        output = output_stream.getvalue()
        assert "*required" in output or "required" in output

    def test_shows_optional_status(self):
        input_stream = io.StringIO("\n")
        output_stream = io.StringIO()

        channel = ConsoleChannel(
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

        channel.request_clarification(request)

        output = output_stream.getvalue()
        assert "optional" in output


class TestAutoChannel:
    """Tests for AutoChannel."""

    def test_create_auto_channel(self):
        channel = AutoChannel()
        assert channel is not None

    def test_auto_single_choice_with_default(self):
        channel = AutoChannel()

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

        response = channel.request_clarification(request)

        assert not response.cancelled
        assert response.answers[0].selected_choices == [2]

    def test_auto_single_choice_first_option(self):
        channel = AutoChannel()

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

        response = channel.request_clarification(request)

        assert response.answers[0].selected_choices == [1]

    def test_auto_multiple_choice_with_default(self):
        channel = AutoChannel()

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

        response = channel.request_clarification(request)

        assert response.answers[0].selected_choices == [2]

    def test_auto_free_text(self):
        channel = AutoChannel(default_free_text="custom response")

        request = ClarificationRequest(
            context="Testing",
            questions=[
                Question(
                    text="Describe",
                    question_type=QuestionType.FREE_TEXT,
                ),
            ],
        )

        response = channel.request_clarification(request)

        assert response.answers[0].free_text == "custom response"

    def test_auto_multiple_questions(self):
        channel = AutoChannel()

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

        response = channel.request_clarification(request)

        assert len(response.answers) == 2
        assert response.answers[0].selected_choices == [1]
        assert response.answers[1].free_text == "auto-response"


class TestCreateChannel:
    """Tests for create_channel factory function."""

    def test_create_console_channel(self):
        channel = create_channel("console")
        assert isinstance(channel, ConsoleChannel)

    def test_create_auto_channel(self):
        channel = create_channel("auto")
        assert isinstance(channel, AutoChannel)

    def test_create_auto_channel_with_config(self):
        channel = create_channel("auto", default_free_text="test")
        assert isinstance(channel, AutoChannel)

    def test_create_unknown_channel(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            create_channel("unknown")
