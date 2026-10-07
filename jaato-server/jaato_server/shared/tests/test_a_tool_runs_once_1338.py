"""A backgroundable tool call runs exactly once (#1338).

``ToolExecutor._execute_with_auto_background`` starts a BackgroundCapable
tool (``cli_based_tool`` is the one that matters) as a background task and
polls it.  The poll called ``cancel_token.is_cancelled()``, but
``CancelToken.is_cancelled`` is a property, so every poll raised
``TypeError``.  One broad ``except`` read that as "the background start
failed" and ran the tool again synchronously, while the first copy was
still running.  Every session hands the executor a cancel token, so every
``cli_based_tool`` call ran its command twice.

The pre-existing auto-background tests never passed a cancel token, which
is how the defect stayed invisible for three weeks.  Every test here passes
a real :class:`CancelToken`, and the last one drives the real cli plugin
and counts lines in a file its command appends to.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import CancelToken
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.plugins.background.mixin import BackgroundCapableMixin
from jaato_server.shared.tests.reversion import Reversion

_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"

REVERSIONS = [
    Reversion(
        target=_RUNNER,
        find="if cancel_token is not None and cancel_token.is_cancelled:",
        replace="if cancel_token is not None and cancel_token.is_cancelled():",
        test="test_a_backgrounded_call_runs_once_with_a_cancel_token",
        because="the poll calls a property, and the fallback re-runs the tool",
    ),
    Reversion(
        target=_RUNNER,
        find=(
            "            return False, {\n"
            "                'error': (\n"
            "                    f\"{name} was started (task {handle.task_id}) but its \"\n"
        ),
        replace=(
            "            return self._execute_sync(name, args)\n"
            "            return False, {\n"
            "                'error': (\n"
            "                    f\"{name} was started (task {handle.task_id}) but its \"\n"
        ),
        test="test_an_error_after_the_start_is_reported_not_rerun",
        because="a failure while waiting re-runs a tool that is already running",
    ),
    Reversion(
        target=_RUNNER,
        find="            if cancel_token is not None and cancel_token.is_cancelled:",
        replace="            if False:",
        test="test_a_stop_request_cancels_the_background_task",
        because="the wait ignores the cancel token and holds a stop request",
    ),
]


class _CountingPlugin(BackgroundCapableMixin):
    """A backgroundable tool that records every time its body runs."""

    def __init__(self, duration: float = 0.3) -> None:
        super().__init__(max_workers=2)
        self.duration = duration
        self.runs: List[float] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "counting"

    def supports_background(self, tool_name: str) -> bool:
        return tool_name == "count_tool"

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        return 0.05 if tool_name == "count_tool" else None

    def estimate_duration(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[float]:
        return None

    def get_executors(self) -> Dict[str, Any]:
        return {"count_tool": self._run}

    def _run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.runs.append(time.monotonic())
        time.sleep(self.duration)
        return {"status": "done"}


class _Registry:
    """The registry surface ToolExecutor reads, and no more."""

    def __init__(self, *plugins: Any) -> None:
        self._by_tool = {t: p for p in plugins for t in p.get_executors()}

    def get_plugin_for_tool(self, tool_name: str) -> Optional[Any]:
        return self._by_tool.get(tool_name)

    def get_plugin(self, name: str) -> Optional[Any]:
        return None

    def get_tool_traits(self, tool_name: str) -> frozenset:
        return frozenset()


def _executor(plugin: Any) -> ToolExecutor:
    ex = ToolExecutor(auto_background_enabled=True)
    ex.set_registry(_Registry(plugin))
    for tool, fn in plugin.get_executors().items():
        ex.register(tool, fn)
    return ex


def _settle(plugin: _CountingPlugin) -> None:
    """Give a stray second copy time to start, so a count of 1 means 1."""
    time.sleep(plugin.duration + 0.3)


def test_a_backgrounded_call_runs_once_with_a_cancel_token():
    plugin = _CountingPlugin()
    ok, result = _executor(plugin).execute(
        "count_tool", {}, cancel_token=CancelToken(),
    )
    _settle(plugin)
    assert ok is True, result
    assert result == {"status": "done"}
    assert len(plugin.runs) == 1, f"the tool body ran {len(plugin.runs)} times"


def test_an_error_after_the_start_is_reported_not_rerun(monkeypatch):
    plugin = _CountingPlugin()
    ex = _executor(plugin)

    def broken_wait(*_a, **_k):
        raise RuntimeError("poll failed")

    monkeypatch.setattr(ex, "_wait_for_background_task", broken_wait)
    ok, result = ex.execute("count_tool", {}, cancel_token=CancelToken())
    _settle(plugin)
    assert ok is False
    assert "not run a second time" in result["error"]
    assert result["task_id"]
    assert len(plugin.runs) == 1, f"the tool body ran {len(plugin.runs)} times"


def test_a_failed_start_still_falls_back_to_one_synchronous_run(monkeypatch):
    plugin = _CountingPlugin(duration=0.0)
    ex = _executor(plugin)

    def broken_start(*_a, **_k):
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(plugin, "start_background", broken_start)
    ok, result = ex.execute("count_tool", {}, cancel_token=CancelToken())
    assert ok is True, result
    assert len(plugin.runs) == 1


def test_a_stop_request_cancels_the_background_task():
    # Long enough that only the cancel token can end the wait in time.
    plugin = _CountingPlugin(duration=3.0)
    token = CancelToken()
    threading.Timer(0.3, token.cancel).start()
    started = time.monotonic()
    _executor(plugin).execute("count_tool", {}, cancel_token=token)
    assert time.monotonic() - started < 2.0, "the stop request was not honoured"
    assert len(plugin.runs) == 1


def test_a_cli_command_runs_once(tmp_path: Path):
    """The live shape: the real cli plugin, one command, one line."""
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    cli = CLIToolPlugin()
    cli.initialize({"workspace_root": str(tmp_path)})
    try:
        ex = _executor(cli)
        ok, result = ex.execute(
            "cli_based_tool",
            {"command": "echo run >> marker.txt; echo ok"},
            cancel_token=CancelToken(),
        )
        time.sleep(0.5)
        assert ok is True, result
        lines = (tmp_path / "marker.txt").read_text().splitlines()
        assert lines == ["run"], f"the command ran {len(lines)} times"
    finally:
        cli.shutdown()
