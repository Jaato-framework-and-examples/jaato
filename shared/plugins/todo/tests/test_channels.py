"""Tests for TODO plugin reporter channels."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ..models import TodoPlan, StepStatus
from ..channels import (
    ConsoleReporter,
    WebhookReporter,
    FileReporter,
    MultiReporter,
    create_reporter,
)


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_reporter_name(self):
        reporter = ConsoleReporter()
        assert reporter.name == "console"

    def test_report_plan_created(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test Plan", ["Step 1", "Step 2"])
        reporter.report_plan_created(plan)

        output = "\n".join(output_lines)
        assert "Test Plan" in output
        assert "Step 1" in output
        assert "Step 2" in output

    def test_report_step_update(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test", ["Step A"])
        step = plan.steps[0]
        step.complete("Done")

        reporter.report_step_update(plan, step)

        output = "\n".join(output_lines)
        assert "Step A" in output
        assert "COMPLETED" in output

    def test_report_step_with_result(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test", ["Step A"])
        step = plan.steps[0]
        step.complete("Task finished successfully")

        reporter.report_step_update(plan, step)

        output = "\n".join(output_lines)
        assert "Task finished successfully" in output

    def test_report_step_with_error(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test", ["Step A"])
        step = plan.steps[0]
        step.fail("Connection timeout")

        reporter.report_step_update(plan, step)

        output = "\n".join(output_lines)
        assert "Connection timeout" in output

    def test_report_plan_completed_success(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test", ["A"])
        plan.complete_plan("All done")

        reporter.report_plan_completed(plan)

        output = "\n".join(output_lines)
        assert "COMPLETED" in output
        assert "All done" in output

    def test_report_plan_completed_failed(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({"output_func": output_lines.append})

        plan = TodoPlan.create("Test", ["A"])
        plan.fail_plan("Something broke")

        reporter.report_plan_completed(plan)

        output = "\n".join(output_lines)
        assert "FAILED" in output

    def test_colors_disabled(self):
        reporter = ConsoleReporter()
        output_lines = []
        reporter.initialize({
            "output_func": output_lines.append,
            "colors": False
        })

        plan = TodoPlan.create("Test", ["A"])
        reporter.report_plan_created(plan)

        # Should not contain ANSI escape codes
        output = "\n".join(output_lines)
        assert "\033[" not in output


class TestFileReporter:
    """Tests for FileReporter."""

    def test_reporter_name(self):
        reporter = FileReporter()
        assert reporter.name == "file"

    def test_report_plan_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = FileReporter()
            reporter.initialize({"base_path": tmpdir})

            plan = TodoPlan.create("Test Plan", ["A", "B"])
            reporter.report_plan_created(plan)

            # Check files were created
            plan_dir = Path(tmpdir) / "plans" / plan.plan_id
            assert plan_dir.exists()
            assert (plan_dir / "plan.json").exists()
            assert (plan_dir / "progress.json").exists()
            assert (plan_dir / "events").exists()

            # Check event file
            events = list((plan_dir / "events").glob("*_plan_created.json"))
            assert len(events) == 1

    def test_report_step_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = FileReporter()
            reporter.initialize({"base_path": tmpdir})

            plan = TodoPlan.create("Test", ["A"])
            reporter.report_plan_created(plan)

            plan.steps[0].complete()
            reporter.report_step_update(plan, plan.steps[0])

            # Check event file
            plan_dir = Path(tmpdir) / "plans" / plan.plan_id
            events = list((plan_dir / "events").glob("*_step_completed.json"))
            assert len(events) == 1

    def test_report_plan_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = FileReporter()
            reporter.initialize({"base_path": tmpdir})

            plan = TodoPlan.create("Test", ["A"])
            reporter.report_plan_created(plan)

            plan.complete_plan()
            reporter.report_plan_completed(plan)

            # Check event file
            plan_dir = Path(tmpdir) / "plans" / plan.plan_id
            events = list((plan_dir / "events").glob("*_plan_completed.json"))
            assert len(events) == 1

    def test_latest_file_updated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = FileReporter()
            reporter.initialize({"base_path": tmpdir})

            plan = TodoPlan.create("Test Plan", ["A"])
            reporter.report_plan_created(plan)

            latest_file = Path(tmpdir) / "latest.json"
            assert latest_file.exists()

            with open(latest_file, 'r') as f:
                data = json.load(f)
            assert data["plan_id"] == plan.plan_id
            assert data["title"] == "Test Plan"

    def test_requires_base_path(self):
        reporter = FileReporter()
        with pytest.raises(ValueError, match="base_path"):
            reporter.initialize({})


class TestWebhookReporter:
    """Tests for WebhookReporter."""

    def test_reporter_name(self):
        reporter = WebhookReporter()
        assert reporter.name == "webhook"

    def test_requires_endpoint(self):
        reporter = WebhookReporter()
        with pytest.raises(ValueError, match="endpoint"):
            reporter.initialize({})

    @patch('shared.plugins.todo.channels.requests')
    def test_report_plan_created(self, mock_requests):
        mock_requests.post.return_value.status_code = 200

        reporter = WebhookReporter()
        reporter.initialize({
            "endpoint": "https://example.com/webhook",
            "timeout": 5
        })

        plan = TodoPlan.create("Test", ["A"])
        reporter.report_plan_created(plan)

        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "https://example.com/webhook"

        # Check JSON payload
        payload = call_args[1]["json"]
        assert payload["event_type"] == "plan_created"
        assert payload["plan_id"] == plan.plan_id

    @patch('shared.plugins.todo.channels.requests')
    def test_report_with_auth_token(self, mock_requests):
        mock_requests.post.return_value.status_code = 200

        reporter = WebhookReporter()
        reporter.initialize({
            "endpoint": "https://example.com/webhook",
            "auth_token": "secret-token"
        })

        plan = TodoPlan.create("Test", ["A"])
        reporter.report_plan_created(plan)

        headers = mock_requests.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret-token"

    @patch('shared.plugins.todo.channels.requests')
    def test_report_with_custom_headers(self, mock_requests):
        mock_requests.post.return_value.status_code = 200

        reporter = WebhookReporter()
        reporter.initialize({
            "endpoint": "https://example.com/webhook",
            "headers": {"X-Custom": "value"}
        })

        plan = TodoPlan.create("Test", ["A"])
        reporter.report_plan_created(plan)

        headers = mock_requests.post.call_args[1]["headers"]
        assert headers["X-Custom"] == "value"


class TestMultiReporter:
    """Tests for MultiReporter."""

    def test_reporter_name(self):
        reporter = MultiReporter()
        assert reporter.name == "multi"

    def test_broadcasts_to_all_reporters(self):
        mock_reporter1 = Mock()
        mock_reporter2 = Mock()

        multi = MultiReporter([mock_reporter1, mock_reporter2])

        plan = TodoPlan.create("Test", ["A"])
        multi.report_plan_created(plan)

        mock_reporter1.report_plan_created.assert_called_once_with(plan)
        mock_reporter2.report_plan_created.assert_called_once_with(plan)

    def test_add_reporter(self):
        multi = MultiReporter()
        mock_reporter = Mock()

        multi.add_reporter(mock_reporter)

        plan = TodoPlan.create("Test", ["A"])
        multi.report_plan_created(plan)

        mock_reporter.report_plan_created.assert_called_once()

    def test_continues_on_error(self):
        failing_reporter = Mock()
        failing_reporter.report_plan_created.side_effect = Exception("Fail")

        working_reporter = Mock()

        multi = MultiReporter([failing_reporter, working_reporter])

        plan = TodoPlan.create("Test", ["A"])
        # Should not raise exception
        multi.report_plan_created(plan)

        # Working reporter should still be called
        working_reporter.report_plan_created.assert_called_once()


class TestCreateReporter:
    """Tests for create_reporter factory function."""

    def test_create_console_reporter(self):
        reporter = create_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_create_file_reporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = create_reporter("file", {"base_path": tmpdir})
            assert isinstance(reporter, FileReporter)

    def test_create_unknown_reporter(self):
        with pytest.raises(ValueError, match="Unknown reporter type"):
            create_reporter("invalid_type")
