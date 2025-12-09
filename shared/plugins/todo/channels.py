"""Reporter channels for the TODO plugin.

Channels handle progress reporting through different transport protocols:
- ConsoleReporter: Renders progress to terminal (with in-place updates via rich)
- WebhookReporter: Sends progress to HTTP endpoints
- FileReporter: Writes progress to filesystem
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rich.console import Console
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .models import ProgressEvent, StepStatus, TodoPlan, TodoStep


class TodoReporter(ABC):
    """Base class for progress reporters.

    Reporters handle different transport protocols for reporting
    plan progress to external systems or users.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this reporter type."""
        ...

    @abstractmethod
    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report that a new plan was created."""
        ...

    @abstractmethod
    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report that a step's status changed."""
        ...

    @abstractmethod
    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report that a plan was completed/failed/cancelled."""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the reporter with optional configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the reporter."""
        pass


class ConsoleReporter(TodoReporter):
    """Reporter that renders progress to the console.

    Displays formatted progress updates with visual indicators
    for step status and overall plan progress.

    In compact mode (default), only final statuses (COMPLETED, FAILED, SKIPPED)
    are shown - IN_PROGRESS updates are suppressed to reduce output noise.
    Set compact=False in config to see all status transitions.

    Uses the `rich` library when available for better terminal handling.
    """

    def __init__(self):
        self._output_func: Optional[Callable[[str], None]] = None
        self._show_timestamps: bool = True
        self._progress_bar: bool = True
        self._use_colors: bool = True
        self._width: int = 60
        self._compact_mode: bool = True  # Skip IN_PROGRESS, only show final status
        # Rich console for portable terminal output
        self._console: Optional["Console"] = None

    def _init_console(self) -> None:
        """Initialize rich Console if available and appropriate."""
        if HAS_RICH and self._output_func is None and self._use_colors:
            self._console = Console(force_terminal=None)  # Auto-detect TTY
        else:
            self._console = None


    @property
    def name(self) -> str:
        return "console"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize console reporter.

        Config options:
            output_func: Custom output function (for testing, disables rich)
            show_timestamps: Show timestamps in output
            progress_bar: Show ASCII progress bar
            colors: Use ANSI colors (default True)
            width: Output width for progress bar
            compact: Compact mode - skip IN_PROGRESS, only show final status (default True)
        """
        if config:
            if "output_func" in config:
                self._output_func = config["output_func"]
            self._show_timestamps = config.get("show_timestamps", True)
            self._progress_bar = config.get("progress_bar", True)
            self._use_colors = config.get("colors", True)
            self._width = config.get("width", 60)
            self._compact_mode = config.get("compact", True)
        # Initialize rich console after config is set
        self._init_console()

    def _print(self, text: str) -> None:
        """Print text using either custom output_func, rich console, or print."""
        if self._output_func is not None:
            self._output_func(text)
        elif self._console is not None:
            self._console.print(text, highlight=False, markup=False)
        else:
            print(text)

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors are enabled."""
        if not self._use_colors:
            return text

        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "gray": "\033[90m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _status_symbol(self, status: StepStatus) -> str:
        """Get visual symbol for step status."""
        symbols = {
            StepStatus.PENDING: self._color("○", "gray"),
            StepStatus.IN_PROGRESS: self._color("●", "blue"),
            StepStatus.COMPLETED: self._color("✓", "green"),
            StepStatus.FAILED: self._color("✗", "red"),
            StepStatus.SKIPPED: self._color("⊘", "yellow"),
        }
        return symbols.get(status, "?")

    def _render_progress_bar(self, progress: Dict[str, Any]) -> str:
        """Render ASCII progress bar."""
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        percent = progress.get("percent", 0)

        bar_width = 30
        filled = int(bar_width * percent / 100) if total > 0 else 0
        empty = bar_width - filled

        bar = self._color("[", "gray")
        bar += self._color("=" * filled, "green")
        if filled < bar_width:
            bar += self._color(">", "blue") if filled > 0 else ""
            bar += " " * (empty - 1 if filled > 0 else empty)
        bar += self._color("]", "gray")
        bar += f" {completed}/{total} ({percent:.0f}%)"

        return bar

    def _timestamp(self) -> str:
        """Get formatted timestamp."""
        if self._show_timestamps:
            return self._color(f"[{datetime.now().strftime('%H:%M:%S')}] ", "gray")
        return ""

    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report new plan creation."""
        self._print("")
        self._print("=" * self._width)
        self._print(self._color(f"📋 PLAN: {plan.title}", "bold"))
        self._print("=" * self._width)
        self._print("")

        for step in sorted(plan.steps, key=lambda s: s.sequence):
            symbol = self._status_symbol(step.status)
            self._print(f"  {symbol} {step.sequence}. {step.description}")

        self._print("")
        if self._progress_bar:
            self._print(self._render_progress_bar(plan.get_progress()))
        self._print("")

    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report step status change.

        When compact mode is enabled (inplace_updates=True), only final statuses
        (COMPLETED, FAILED, SKIPPED) are shown - IN_PROGRESS is suppressed to
        reduce noise. This is simpler and more robust than cursor manipulation
        which breaks when other output occurs between status changes.
        """
        # In compact mode, skip IN_PROGRESS - only show final status
        if self._compact_mode and step.status == StepStatus.IN_PROGRESS:
            return

        symbol = self._status_symbol(step.status)
        status_text = step.status.value.upper()

        if step.status == StepStatus.IN_PROGRESS:
            status_color = "blue"
        elif step.status == StepStatus.COMPLETED:
            status_color = "green"
        elif step.status == StepStatus.FAILED:
            status_color = "red"
        elif step.status == StepStatus.SKIPPED:
            status_color = "yellow"
        else:
            status_color = "gray"

        # Output the status line
        status_line = (
            f"{self._timestamp()}{symbol} "
            f"[{step.sequence}/{len(plan.steps)}] "
            f"{self._color(status_text, status_color)}: {step.description}"
        )
        self._print(status_line)

        # Show result or error for completed/failed/skipped steps
        if step.result and step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
            self._print(f"    → {step.result}")

        if step.error and step.status == StepStatus.FAILED:
            self._print(self._color(f"    ✗ Error: {step.error}", "red"))

        # Show progress bar
        if self._progress_bar:
            progress = plan.get_progress()
            progress_line = f"    {self._render_progress_bar(progress)}"
            self._print(progress_line)

    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report plan completion."""
        self._print("")
        self._print("=" * self._width)

        progress = plan.get_progress()

        if plan.status.value == "completed":
            emoji = "✅"
            status_text = self._color("PLAN COMPLETED", "green")
        elif plan.status.value == "failed":
            emoji = "❌"
            status_text = self._color("PLAN FAILED", "red")
        else:
            emoji = "⚠️"
            status_text = self._color("PLAN CANCELLED", "yellow")

        self._print(f"{emoji} {status_text}: {plan.title}")

        # Summary stats
        self._print(
            f"   Steps: {progress['completed']} completed, "
            f"{progress['failed']} failed, "
            f"{progress['skipped']} skipped"
        )

        if plan.summary:
            self._print(f"   Summary: {plan.summary}")

        self._print("=" * self._width)
        self._print("")


class WebhookReporter(TodoReporter):
    """Reporter that sends progress to an HTTP webhook.

    Designed for integration with external systems like dashboards,
    notification services, or workflow automation.
    """

    def __init__(self):
        self._endpoint: Optional[str] = None
        self._timeout: int = 10
        self._headers: Dict[str, str] = {}
        self._auth_token: Optional[str] = None

    @property
    def name(self) -> str:
        return "webhook"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize webhook reporter.

        Config options:
            endpoint: URL to send progress events (required)
            timeout: Request timeout in seconds
            headers: Additional headers to include
            auth_token: Bearer token for authorization
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for WebhookReporter")

        if not config:
            raise ValueError("WebhookReporter requires configuration with 'endpoint'")

        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("WebhookReporter requires 'endpoint' in config")

        self._timeout = config.get("timeout", 10)
        self._headers = config.get("headers", {})
        self._auth_token = config.get("auth_token") or os.environ.get("TODO_WEBHOOK_TOKEN")

    def _send_event(self, event: ProgressEvent) -> bool:
        """Send event to webhook endpoint."""
        if not self._endpoint:
            return False

        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        try:
            response = requests.post(
                self._endpoint,
                json=event.to_dict(),
                headers=headers,
                timeout=self._timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report new plan via webhook."""
        event = ProgressEvent.create("plan_created", plan)
        self._send_event(event)

    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report step update via webhook."""
        event_type = f"step_{step.status.value}"
        event = ProgressEvent.create(event_type, plan, step)
        self._send_event(event)

    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report plan completion via webhook."""
        event = ProgressEvent.create(f"plan_{plan.status.value}", plan)
        self._send_event(event)


class FileReporter(TodoReporter):
    """Reporter that writes progress to the filesystem.

    Designed for scenarios where a separate process monitors progress,
    or for creating persistent logs of plan execution.

    Directory structure:
        {base_path}/
        ├── plans/{plan_id}/
        │   ├── plan.json       (full plan state)
        │   ├── progress.json   (current progress stats)
        │   └── events/         (individual event files)
        │       ├── 001_plan_created.json
        │       ├── 002_step_started.json
        │       └── ...
        └── latest.json         (pointer to most recent plan)
    """

    def __init__(self):
        self._base_path: Optional[Path] = None
        self._event_counter: Dict[str, int] = {}  # plan_id -> event count

    @property
    def name(self) -> str:
        return "file"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize file reporter.

        Config options:
            base_path: Directory for progress files (required)
        """
        if not config:
            raise ValueError("FileReporter requires configuration with 'base_path'")

        base_path = config.get("base_path")
        if not base_path:
            raise ValueError("FileReporter requires 'base_path' in config")

        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        (self._base_path / "plans").mkdir(exist_ok=True)

    def _get_plan_dir(self, plan_id: str) -> Path:
        """Get or create plan directory."""
        plan_dir = self._base_path / "plans" / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "events").mkdir(exist_ok=True)
        return plan_dir

    def _write_event(self, plan: TodoPlan, event: ProgressEvent) -> None:
        """Write event to file."""
        plan_dir = self._get_plan_dir(plan.plan_id)

        # Increment event counter
        if plan.plan_id not in self._event_counter:
            # Count existing events
            events_dir = plan_dir / "events"
            existing = list(events_dir.glob("*.json"))
            self._event_counter[plan.plan_id] = len(existing)

        self._event_counter[plan.plan_id] += 1
        count = self._event_counter[plan.plan_id]

        # Write event file
        event_file = plan_dir / "events" / f"{count:03d}_{event.event_type}.json"
        with open(event_file, 'w', encoding='utf-8') as f:
            json.dump(event.to_dict(), f, indent=2)

        # Update plan state
        plan_file = plan_dir / "plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan.to_dict(), f, indent=2)

        # Update progress
        progress_file = plan_dir / "progress.json"
        progress = plan.get_progress()
        progress["status"] = plan.status.value
        progress["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)

        # Update latest pointer
        latest_file = self._base_path / "latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump({
                "plan_id": plan.plan_id,
                "title": plan.title,
                "status": plan.status.value,
                "progress": progress,
            }, f, indent=2)

    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report new plan to filesystem."""
        event = ProgressEvent.create("plan_created", plan)
        self._write_event(plan, event)

    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report step update to filesystem."""
        event_type = f"step_{step.status.value}"
        event = ProgressEvent.create(event_type, plan, step)
        self._write_event(plan, event)

    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report plan completion to filesystem."""
        event = ProgressEvent.create(f"plan_{plan.status.value}", plan)
        self._write_event(plan, event)

    def shutdown(self) -> None:
        """Clean up event counters."""
        self._event_counter.clear()


class MultiReporter(TodoReporter):
    """Reporter that broadcasts to multiple underlying reporters.

    Allows combining reporters for multi-channel reporting,
    e.g., console + webhook simultaneously.
    """

    def __init__(self, reporters: Optional[List[TodoReporter]] = None):
        self._reporters: List[TodoReporter] = reporters or []

    @property
    def name(self) -> str:
        return "multi"

    def add_reporter(self, reporter: TodoReporter) -> None:
        """Add a reporter to the broadcast list."""
        self._reporters.append(reporter)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize all reporters."""
        for reporter in self._reporters:
            reporter.initialize(config)

    def shutdown(self) -> None:
        """Shutdown all reporters."""
        for reporter in self._reporters:
            reporter.shutdown()

    def report_plan_created(self, plan: TodoPlan) -> None:
        """Report to all underlying reporters."""
        for reporter in self._reporters:
            try:
                reporter.report_plan_created(plan)
            except Exception:
                pass  # Don't let one reporter failure stop others

    def report_step_update(self, plan: TodoPlan, step: TodoStep) -> None:
        """Report to all underlying reporters."""
        for reporter in self._reporters:
            try:
                reporter.report_step_update(plan, step)
            except Exception:
                pass

    def report_plan_completed(self, plan: TodoPlan) -> None:
        """Report to all underlying reporters."""
        for reporter in self._reporters:
            try:
                reporter.report_plan_completed(plan)
            except Exception:
                pass


def create_reporter(
    reporter_type: str,
    config: Optional[Dict[str, Any]] = None
) -> TodoReporter:
    """Factory function to create a reporter by type.

    Args:
        reporter_type: One of "console", "webhook", "file"
        config: Optional configuration for the reporter

    Returns:
        Initialized TodoReporter instance

    Raises:
        ValueError: If reporter_type is unknown
    """
    reporters = {
        "console": ConsoleReporter,
        "webhook": WebhookReporter,
        "file": FileReporter,
    }

    if reporter_type not in reporters:
        raise ValueError(
            f"Unknown reporter type: {reporter_type}. "
            f"Available: {list(reporters.keys())}"
        )

    reporter = reporters[reporter_type]()
    reporter.initialize(config)
    return reporter
