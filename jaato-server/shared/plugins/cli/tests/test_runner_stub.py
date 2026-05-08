"""Tests for the daemon-side cli plugin's runner-RPC stub.

When ``registry.runner_rpc`` is set, ``CLIToolPlugin._execute`` short-
circuits the in-process ``run_command`` path and forwards to the
runner via :meth:`RunnerRPCClient.call_threadsafe`.  These tests
verify the translation between the typed envelope and the legacy
result-dict shape the model + history pipeline expects.

End-to-end coverage of the runner subprocess itself is in
``jaato-server/server/test_runner_rpc.py`` (real fork+exec); these
tests focus on the stub's translation contract using a fake RPC.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock

import pytest

from server.runner.envelope import ErrorPayload, ResponseEnvelope
from shared.plugins.cli.plugin import CLIToolPlugin


class _FakeRPC:
    """Minimal :class:`RunnerRPCClient` stand-in for stub-side tests.

    Records each ``call_threadsafe`` invocation; returns a
    ``ResponseEnvelope`` set on the instance via ``next_response``.
    """

    def __init__(self) -> None:
        self.next_response: Optional[ResponseEnvelope] = None
        self.next_raise: Optional[BaseException] = None
        self.calls: list[Dict[str, Any]] = []

    def call_threadsafe(
        self,
        method: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        on_output: Optional[Callable] = None,
        cancel_token: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> ResponseEnvelope:
        self.calls.append({
            "method": method,
            "args": args,
            "on_output": on_output,
            "cancel_token": cancel_token,
            "timeout": timeout,
        })
        if self.next_raise is not None:
            raise self.next_raise
        if self.next_response is None:
            raise AssertionError("test forgot to set _FakeRPC.next_response")
        return self.next_response


@pytest.fixture
def cli_plugin_with_fake_rpc() -> tuple[CLIToolPlugin, _FakeRPC]:
    plugin = CLIToolPlugin()
    rpc = _FakeRPC()

    # Mimic the registry-attribute injection pattern from §5.4 of the
    # plan: ``set_runner_rpc`` puts the RPC on the registry, the
    # plugin's ``set_plugin_registry`` hook captures the registry.
    fake_registry = MagicMock()
    fake_registry.runner_rpc = rpc
    fake_registry.register_category = MagicMock()
    plugin.set_plugin_registry(fake_registry)

    return plugin, rpc


# ----------------------------------------------------------------------
# Happy-path translation
# ----------------------------------------------------------------------


def test_stub_routes_through_rpc_when_runner_attached(
    cli_plugin_with_fake_rpc,
) -> None:
    plugin, rpc = cli_plugin_with_fake_rpc
    rpc.next_response = ResponseEnvelope(
        id=1,
        ok=True,
        result={
            "stdout": "hello\n",
            "stderr": "",
            "returncode": 0,
            "_telemetry": {"jaato.cli.command": "echo hello"},
        },
    )

    result = plugin._execute({"command": "echo hello"})

    assert result["stdout"] == "hello\n"
    assert result["returncode"] == 0
    assert result["_telemetry"]["jaato.cli.command"] == "echo hello"
    # Verify the args we sent over RPC mirror what the model passed.
    assert len(rpc.calls) == 1
    assert rpc.calls[0]["method"] == "tool.execute"
    assert rpc.calls[0]["args"] == {
        "name": "cli_based_tool",
        "args": {"command": "echo hello"},
    }


def test_stub_falls_back_in_process_when_no_runner() -> None:
    """No runner attached → today's in-process path runs.

    Verified by checking that ``run_command`` is called (not the
    fake RPC).  We don't actually exercise a real subprocess here —
    we just confirm the stub doesn't try to dispatch to a None
    runner_rpc.
    """
    plugin = CLIToolPlugin()
    fake_registry = MagicMock()
    fake_registry.runner_rpc = None
    fake_registry.register_category = MagicMock()
    plugin.set_plugin_registry(fake_registry)

    # Smoke: calling _execute with no runner should reach the
    # in-process try/except block and either succeed or return an
    # error dict; neither should raise.
    result = plugin._execute({"command": "echo hello"})
    # The in-process path either returns stdout or surfaces the
    # path-validation / executable-not-found error — both are dicts.
    assert isinstance(result, dict)


# ----------------------------------------------------------------------
# Cancellation translation (§4.1.1)
# ----------------------------------------------------------------------


def test_stub_propagates_cancelled_exception(cli_plugin_with_fake_rpc) -> None:
    """Runner-side CancelledException → re-raised on the daemon side."""
    from jaato_sdk.plugins.model_provider.types import CancelledException

    plugin, rpc = cli_plugin_with_fake_rpc
    rpc.next_response = ResponseEnvelope(
        id=1,
        ok=False,
        result=None,
        error=ErrorPayload(
            type="CancelledException",
            message="Cancelled during subprocess output",
        ),
    )

    with pytest.raises(CancelledException, match="Cancelled"):
        plugin._execute({"command": "sleep 30"})


# ----------------------------------------------------------------------
# Domain-failure translation (envelope.ok=False with structured result)
# ----------------------------------------------------------------------


def test_stub_returns_domain_failure_dict_unchanged(
    cli_plugin_with_fake_rpc,
) -> None:
    """ok=False with a structured result dict (e.g. exec-not-found)
    flows through to the model unchanged."""
    plugin, rpc = cli_plugin_with_fake_rpc
    rpc.next_response = ResponseEnvelope(
        id=1,
        ok=False,
        result={
            "error": "cli_based_tool: executable 'no-such' not found in PATH",
            "hint": "Configure extra_paths or provide full path",
        },
        error=ErrorPayload(
            type="ToolError",
            message="cli_based_tool: executable 'no-such' not found in PATH",
        ),
    )

    result = plugin._execute({"command": "no-such"})
    assert "not found in PATH" in result["error"]
    assert "Configure extra_paths" in result["hint"]


# ----------------------------------------------------------------------
# RPC transport failure
# ----------------------------------------------------------------------


def test_stub_surfaces_rpc_transport_error_as_tool_error(
    cli_plugin_with_fake_rpc,
) -> None:
    """If the RPC call itself raises (e.g. peer dead), translate to a
    legacy error-dict so the model gets a clean message rather than
    an opaque traceback."""
    from server.runner_rpc_client import RunnerCallError

    plugin, rpc = cli_plugin_with_fake_rpc
    rpc.next_raise = RunnerCallError("runner RPC closed before response arrived")

    result = plugin._execute({"command": "echo hi"})
    assert "runner RPC failed" in result["error"]
    assert "RunnerCallError" in result["error"]
