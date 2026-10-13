"""An MCP server's ``isError: true`` answer is a failed tool call (#1654).

WHAT WAS WRONG.  An MCP server refuses a call by ANSWERING it with
``isError: true`` rather than with a JSON-RPC error.  ``MCPToolPlugin._execute``
returned that answer as a bare dict::

    {"result": {"tool": "kb_get_node", "isError": True,
                "content": ["no node 'x'"]},
     "_telemetry": {...}}

and ``ToolExecutor._normalize_executor_return`` reads a bare value as a
success.  So ``ok=True`` reached the tool-call events, the persisted
``is_error``, the reliability plugin and the telemetry span.  The ledger a
completion processor reads did not help either: it computed ``success`` as
``"error" not in result`` and the payload has no ``error`` key.  Live, a wiki
panel offered a link to a page the server had just said did not exist.  The
#1053 / #1614 class, one plugin over.

AND ON mcp 2.x THE FLAG WAS NEVER READ AT ALL.  The SDK's result model names
the field ``is_error`` there (``isError`` is only the JSON alias), and the
plugin read ``getattr(result, 'isError', False)`` -- ``False`` for every
refusal, so even the payload's own ``isError`` said ``False``.
``mcp_result_is_error`` reads both spellings.

THE FIX, at the three places the verdict could be lost:

- ``mcp_result_is_error`` reads the flag on both SDK generations;
- ``_execute`` returns ``(False, payload)`` for an ``isError`` answer (and for
  its transport / timeout / lookup failures), payload unchanged;
- ``runner_forwarding._forward_via_runner`` turns a runner answer with
  ``ok=False`` into ``(False, payload)``, so a call forwarded from the daemon
  keeps the verdict the runner's executor reached;
- ``build_ledger`` reads the recorded ``is_error`` flag beside the ``error``
  key, on both carriers.

WHAT THE MODEL READS.  For the ``isError`` answer: the same dict (it has a
``result`` key, so ``normalize_result_dict`` does not collapse it), now beside
the wire's own error flag where the wire has one.  The transport failures are
``error``-only dicts and collapse to the bare error string, as #1614's did.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_result_builder import normalize_result_dict

_PLUGIN = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_FORWARDING = "jaato-server/jaato_server/shared/plugins/runner_forwarding.py"
_LEDGER = "jaato-sdk/jaato_sdk/completion_processors.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="    for name in ('isError', 'is_error'):\n",
        replace="    for name in ('isError',):\n",
        test="test_the_flag_is_read_on_both_sdk_generations",
        because=(
            "on mcp 2.x the result's field is is_error, so reading only "
            "isError answers False for every refusal"
        ),
    ),
    Reversion(
        target=_PLUGIN,
        find="        if out['isError'] is True:\n            return False, payload\n",
        replace="",
        test="test_a_refused_mcp_call_is_a_failure",
        because=(
            "the server's isError answer is a bare dict again, so the "
            "refused call is reported as ok=True everywhere"
        ),
    ),
    Reversion(
        target=_FORWARDING,
        find=("        return (False, _reinject_envelope_telemetry(\n"
              "            envelope.result, envelope.telemetry))\n"),
        replace=("        return _reinject_envelope_telemetry(\n"
                 "            envelope.result, envelope.telemetry)\n"),
        test="test_a_failure_forwarded_from_the_runner_stays_a_failure",
        because=(
            "the runner said ok=False and the daemon-side forwarder hands "
            "back a bare dict, which reads as a success"
        ),
    ),
    Reversion(
        target=_LEDGER,
        find='                "success": not recorded_error and "error" not in result,\n',
        replace='                "success": "error" not in result,\n',
        test="test_the_ledger_reads_the_recorded_flag",
        because=(
            "the completion-processor ledger ignores is_error, and an "
            "isError payload has no error key"
        ),
    ),
]

_FIXTURE = Path(__file__).with_name("mcp_http_fixture_server.py")


# --------------------------------------------------------------- a real server

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def mcp_plugin(tmp_path):
    """A real MCPToolPlugin connected to a real MCP server."""
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(_FIXTURE), str(port), str(tmp_path / "h.log")],
        env={**os.environ, "MCP_FIXTURE_REFUSING_TOOL": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    plugin = None
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"fixture": {
            "type": "http", "url": f"http://127.0.0.1:{port}/mcp"}}}))
        from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
        plugin = MCPToolPlugin()
        plugin.initialize({"config_path": str(cfg)})
        deadline = time.time() + 20
        while time.time() < deadline and "fixture" not in plugin._connected_servers:
            time.sleep(0.1)
        assert "fixture" in plugin._connected_servers, plugin._failed_servers
        plugin._load_config_cache()
        yield plugin
    finally:
        if plugin is not None:
            plugin.shutdown()
        proc.terminate()
        proc.wait(10)


class _Registry:
    """The registry surface ToolExecutor reads, and no more."""

    def __init__(self, plugin: Any) -> None:
        self._by_tool = {t: plugin for t in plugin.get_executors()}

    def get_plugin_for_tool(self, tool_name: str) -> Optional[Any]:
        return self._by_tool.get(tool_name)

    def get_plugin(self, name: str) -> Optional[Any]:
        return None

    def get_tool_traits(self, tool_name: str) -> frozenset:
        return frozenset()


def _run(plugin: Any, suffix: str, args: dict):
    """Run one MCP tool through the real ToolExecutor."""
    from jaato_server.shared.ai_tool_runner import ToolExecutor
    executors = plugin.get_executors()
    name = next(n for n in executors if n.endswith(suffix))
    ex = ToolExecutor()
    ex.set_registry(_Registry(plugin))
    for tool, fn in executors.items():
        ex.register(tool, fn)
    return ex.execute(name, args)


def test_a_refused_mcp_call_is_a_failure(mcp_plugin):
    """THE GUARD: the server's ``isError: true`` reaches ToolExecutor as ok=False."""
    ok, payload = _run(mcp_plugin, "get_node", {"node_id": "nope"})
    assert ok is False, payload
    # The payload is the server's answer, unchanged.
    assert payload["result"]["isError"] is True
    assert any("no node 'nope'" in (c or "")
               for c in payload["result"]["content"]), payload


def test_an_answered_mcp_call_is_still_a_success(mcp_plugin):
    """The control: a call the server answered normally stays a success."""
    ok, payload = _run(mcp_plugin, "echo", {"text": "hi"})
    assert ok is True, payload
    assert payload["result"]["isError"] is False
    assert payload["result"]["content"] == ["echo: hi"]


def test_the_flag_is_read_on_both_sdk_generations():
    from types import SimpleNamespace
    from jaato_server.shared.plugins.mcp.plugin import mcp_result_is_error

    assert mcp_result_is_error(SimpleNamespace(isError=True))    # mcp 1.x
    assert mcp_result_is_error(SimpleNamespace(is_error=True))   # mcp 2.x
    assert not mcp_result_is_error(SimpleNamespace(is_error=False))
    assert not mcp_result_is_error(SimpleNamespace())
    # Only a real True: a truthy non-bool is not a refusal.
    assert not mcp_result_is_error(SimpleNamespace(isError="yes"))


# ----------------------------------------------- the daemon-forwarding path

def test_a_failure_forwarded_from_the_runner_stays_a_failure():
    """The runner's ``ok=False`` survives the trip back to the daemon."""
    from jaato_server.server.runner.envelope import ResponseEnvelope
    from jaato_server.shared.plugins.runner_forwarding import _forward_via_runner

    body = {"result": {"tool": "get_node", "isError": True,
                       "content": ["no node 'nope'"]}}
    rpc = MagicMock()
    rpc.call_threadsafe.return_value = ResponseEnvelope(
        id=1, ok=False, result=dict(body))
    assert _forward_via_runner(rpc, "mcp__fixture__get_node", {}) == (False, body)


def test_a_success_forwarded_from_the_runner_stays_bare():
    from jaato_server.server.runner.envelope import ResponseEnvelope
    from jaato_server.shared.plugins.runner_forwarding import _forward_via_runner

    rpc = MagicMock()
    rpc.call_threadsafe.return_value = ResponseEnvelope(
        id=1, ok=True, result={"result": {"isError": False}})
    assert _forward_via_runner(rpc, "t", {}) == {"result": {"isError": False}}


# -------------------------------------------------------------- the ledger

def test_the_ledger_reads_the_recorded_flag():
    """A completion processor's ledger sees the failure on both carriers."""
    from jaato_sdk.completion_processors import build_ledger
    from jaato_sdk.plugins.model_provider.types import (
        FunctionCall, Message, Part, Role, ToolResult,
    )

    body = {"result": {"tool": "get_node", "isError": True,
                       "content": ["no node 'nope'"]}}
    obj = [Message(role=Role.MODEL, parts=[
               Part(function_call=FunctionCall(id="c", name="t", args={}))]),
           Message(role=Role.USER, parts=[
               Part(function_response=ToolResult(
                   call_id="c", name="t", result=body, is_error=True))])]
    wire = [{"role": "model", "parts": [
                {"type": "function_call", "name": "t", "args": {},
                 "call_id": "c"}]},
            {"role": "user", "parts": [
                {"type": "function_response", "name": "t", "call_id": "c",
                 "response": body, "is_error": True}]}]
    assert build_ledger(obj)[0]["success"] is False
    assert build_ledger(wire)[0]["success"] is False


def test_the_model_facing_payload_is_unchanged():
    """The flag flipped; what the model reads for an isError answer did not."""
    payload = {"result": {"tool": "get_node", "isError": True,
                          "content": ["no node 'nope'"]},
               "_telemetry": {"jaato.mcp.is_error": True}}
    assert (normalize_result_dict(dict(payload), ok=False)
            == normalize_result_dict(dict(payload), ok=True))
