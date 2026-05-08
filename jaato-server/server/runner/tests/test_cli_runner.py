"""Tests for the runner-side cli executor — §4.1.1 streaming +
cancellation contract.

These run in-process (no fork, no real apparmor) — they exercise
:func:`server.runner.cli_runner.execute_cli_based_tool` directly with
the runner's thread-local state set up to mimic the dispatcher's
behavior.

The §3 test matrix from the Phase 2 plan:

| Scenario | Verified by |
|----------|-------------|
| Single-shot success | ``test_cli_single_shot_echo_returns_stdout`` |
| Streaming chunks in order | ``test_cli_streaming_chunks_arrive_in_order`` |
| stderr/stdout per-source ordering (NOT cross-source) | ``test_cli_per_source_ordering_preserved`` |
| Mid-stream cancellation | ``test_cli_cancel_mid_stream_kills_subprocess`` |
| Output truncation | ``test_cli_output_truncation_flag`` |
| Timeout | ``test_cli_timeout_flag`` |

The "subprocess SIGTERM on parent shutdown" scenario from §3 is
covered by the §2.6 sister test ``test_phase2_runner_crash.py`` —
it requires actually killing the runner process, which the
in-process unit tests can't do.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import pytest

from server.runner import cli_runner, rpc as runner_rpc


@pytest.fixture
def thread_local_dispatcher():
    """Install per-test runner-side thread-local state.

    Mimics the work :class:`RunnerRPC._handle_request` does for an
    in-flight call.  Returns ``(captured_chunks, cancel_token,
    install)`` where ``install()`` sets the thread-locals on the
    CALLING thread — pulled out of the fixture body so cross-thread
    tests (e.g. cancel-from-main-while-cli-runs-in-worker) can call
    it from the worker thread, since thread-locals are per-thread.
    """
    captured: List[Tuple[str, str, Optional[str]]] = []
    token = runner_rpc.CancelToken()

    def _on_output(source: str, text: str, mode: Optional[str] = None) -> None:
        captured.append((source, text, mode))

    def _install() -> None:
        runner_rpc._thread_local.cancel_token = token
        runner_rpc._thread_local.on_output = _on_output

    # Install on the test's own thread by default so single-threaded
    # tests don't have to call _install() explicitly.
    _install()
    try:
        yield captured, token, _install
    finally:
        runner_rpc._thread_local.cancel_token = None
        runner_rpc._thread_local.on_output = None


# ----------------------------------------------------------------------
# Single-shot
# ----------------------------------------------------------------------


def test_cli_single_shot_echo_returns_stdout(thread_local_dispatcher) -> None:
    captured, _, _install = thread_local_dispatcher
    ok, result = cli_runner.execute_cli_based_tool(
        {"command": "echo hi"},
    )
    assert ok is True
    assert result["stdout"] == "hi\n"
    assert result["returncode"] == 0
    assert "_telemetry" in result
    # Streaming callback fired for the one stdout line.
    assert [c for c in captured if c[0] == "stdout"] == [("stdout", "hi\n", None)]


def test_cli_missing_command_returns_error(thread_local_dispatcher) -> None:
    ok, result = cli_runner.execute_cli_based_tool({})
    assert ok is False
    assert "command must be provided" in result["error"]


def test_cli_executable_not_found_surfaces_clean_error(thread_local_dispatcher) -> None:
    ok, result = cli_runner.execute_cli_based_tool(
        {"command": "this-command-does-not-exist-anywhere"},
    )
    assert ok is False
    assert "not found in PATH" in result["error"]
    assert "hint" in result


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def test_cli_streaming_chunks_arrive_in_order(thread_local_dispatcher) -> None:
    captured, _, _install = thread_local_dispatcher
    ok, result = cli_runner.execute_cli_based_tool(
        {
            "command": "for i in 1 2 3; do echo line-$i; done",
        },
    )
    assert ok is True
    assert result["returncode"] == 0
    # The streaming callback fired for each line in order.
    stdout_chunks = [text for src, text, _ in captured if src == "stdout"]
    assert stdout_chunks == ["line-1\n", "line-2\n", "line-3\n"]


def test_cli_per_source_ordering_preserved(thread_local_dispatcher) -> None:
    """Per the §3 sharpened assertion (review edit E):
    per-source ordering preserved within each source; cross-source
    ordering NOT asserted.

    Note: today's run_command only streams stdout via on_stdout_line —
    stderr only appears in the final result.  That's the contract this
    test pins down.
    """
    captured, _, _install = thread_local_dispatcher
    ok, result = cli_runner.execute_cli_based_tool(
        {
            "command": "echo to-stdout; echo to-stderr 1>&2",
        },
    )
    assert ok is True
    # stdout streamed live, in order:
    stdout_chunks = [text for src, text, _ in captured if src == "stdout"]
    assert stdout_chunks == ["to-stdout\n"]
    # stderr captured in the final result:
    assert "to-stderr" in result["stderr"]


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------


def test_cli_cancel_mid_stream_kills_subprocess(thread_local_dispatcher) -> None:
    """While streaming output, a cancel token kills the subprocess
    between output lines.

    The run_command contract polls the cancel token between stdout
    lines (see shared/subprocess_runner.py:237), so the test command
    has to produce output for the cancel to fire — that matches the
    real cli use case.  A no-output command would only be cancelled
    via the wall-clock timeout (covered by ``test_cli_timeout_flag``).
    """
    _captured, token, install = thread_local_dispatcher

    exc_holder: List[BaseException] = []

    def _runner() -> None:
        # Thread-locals are per-thread; install on the worker thread.
        install()
        try:
            cli_runner.execute_cli_based_tool(
                {
                    # Emits a line every 50ms so the streaming reader
                    # has frequent cancel-check opportunities.
                    "command": (
                        "for i in $(seq 1 600); do "
                        "echo line-$i; sleep 0.05; "
                        "done"
                    ),
                },
            )
        except BaseException as exc:  # noqa: BLE001
            exc_holder.append(exc)

    t = threading.Thread(target=_runner, name="cli-cancel-test")
    started = time.monotonic()
    t.start()
    # Let a few lines stream so the loop is past `for line in proc.stdout:`.
    time.sleep(0.2)
    token.cancel()
    t.join(timeout=5)
    elapsed = time.monotonic() - started
    assert not t.is_alive(), "cli_runner did not honor cancel within 5s"
    assert elapsed < 5, f"took {elapsed:.2f}s — should be sub-second"

    # The dispatcher would catch CancelledException and serialize it
    # into the typed envelope's error; here we just verify the type.
    assert exc_holder, "expected CancelledException to propagate"
    from jaato_sdk.plugins.model_provider.types import CancelledException
    assert isinstance(exc_holder[0], CancelledException)


# ----------------------------------------------------------------------
# Truncation + timeout
# ----------------------------------------------------------------------


def test_cli_output_truncation_flag(thread_local_dispatcher) -> None:
    """A command emitting more than max_output_chars is truncated and
    the result carries the truncation flag + message."""
    ok, result = cli_runner.execute_cli_based_tool(
        {"command": "yes a | head -c 5000"},
        max_output_chars=100,
    )
    assert ok is True
    assert result.get("truncated") is True
    assert "Output truncated" in result.get("truncation_message", "")
    assert len(result["stdout"]) <= 100 + len("\n")


def test_cli_timeout_flag(thread_local_dispatcher) -> None:
    """When tool_timeout_seconds fires, the result carries timed_out=True."""
    ok, result = cli_runner.execute_cli_based_tool(
        {"command": "sleep 5"},
        tool_timeout_seconds=0.5,
    )
    assert ok is True  # timeout is a soft failure — process did run
    assert result.get("timed_out") is True
    assert result.get("timeout_seconds") == 0.5
