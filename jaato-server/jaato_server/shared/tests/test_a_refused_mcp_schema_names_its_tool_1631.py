"""A provider-refused MCP tool schema degrades the agent, and is named (#1631).

One tool of a remote MCP server expressed "exactly one of A, B or C" as a
TOP-LEVEL ``oneOf``: valid JSON Schema, refused by OpenAI as a function
parameter schema.  Every tool's schema rides every request, so every turn of
the agent failed, the greeting included, with

    Invalid schema for function 't_d3b93b33': ... not have 'oneOf' ...

-- a wire id only the issuing runner can resolve.  Two fixes, two halves:

* the MCP plugin leaves such a tool out of everything it offers (schemas,
  executors, streaming names, the system-instruction listing) and announces
  it once at WARNING by name, the bargain it already makes for a server it
  cannot connect;
* the runner's send-stage error resolves tool ids back to names with
  ``scrub_tool_ids``, in the process whose reverse map issued them.

The runner half drives the real ``_handle_session_send_message`` with a
session that raises the provider's own message.
"""

from __future__ import annotations

import logging
import socket
from types import SimpleNamespace
from typing import Any

from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
from jaato_server.shared.plugins.mcp.schema_compat import (
    top_level_schema_refusal,
)
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_id_map import name_to_id

_PLUGIN = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_COMPAT = "jaato-server/jaato_server/shared/plugins/mcp/schema_compat.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="            for tool in self._usable_tools(server_name, tools):\n"
             "                try:\n",
        replace="            for tool in tools:\n"
                "                try:\n",
        test="test_a_refused_schema_is_not_offered",
        because="the refused schema reaches the provider and fails every turn",
    ),
    Reversion(
        target=_PLUGIN,
        find="            for tool in self._usable_tools(server_name, tools):\n"
             "                # Normalize the tool name for registration",
        replace="            for tool in tools:\n"
                "                # Normalize the tool name for registration",
        test="test_executors_and_listing_agree_with_the_schemas",
        because="an excluded tool would stay callable by a guessed name",
    ),
    Reversion(
        target=_COMPAT,
        find='TOP_LEVEL_REFUSED_KEYWORDS = ("oneOf", "anyOf", "allOf", "enum", "const", "not")',
        replace='TOP_LEVEL_REFUSED_KEYWORDS = ()',
        test="test_the_refused_shapes",
        because="no top-level combinator would be recognised",
    ),
    Reversion(
        target=_PLUGIN,
        find="                self._excluded_announced.add(key)\n"
             "                self._log_event(\n",
        replace="                self._excluded_announced.add(key)\n"
                "                if False: self._log_event(\n",
        test="test_the_exclusion_is_announced_once_by_name",
        because="the tool would vanish with nothing saying why",
    ),
    Reversion(
        target=_RPC,
        find='                    "error": scrub_tool_ids(\n',
        replace='                    "error": (\n',
        test="test_the_send_error_names_the_tool_not_its_wire_id",
        because="the provider's 400 names a hash only this process can resolve",
    ),
]


def _tool(name: str, schema: Any) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=f"{name} does things",
                           inputSchema=schema)


_GOOD = {"type": "object", "properties": {"q": {"type": "string"}}}
_ONE_OF = {
    "oneOf": [
        {"type": "object", "properties": {"a": {"type": "string"}},
         "required": ["a"]},
        {"type": "object", "properties": {"b": {"type": "string"}},
         "required": ["b"]},
    ],
}


def _plugin() -> MCPToolPlugin:
    plugin = MCPToolPlugin()
    plugin._initialized = True
    plugin._tool_cache = {
        "kbwiki": [_tool("search", _GOOD), _tool("write_page", _ONE_OF)],
    }
    plugin._streaming_servers = {"kbwiki"}
    return plugin


def test_the_refused_shapes():
    assert top_level_schema_refusal(_GOOD) is None
    assert top_level_schema_refusal(None) is None
    # A nested combinator is accepted by providers and left alone.
    assert top_level_schema_refusal({
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    }) is None
    for kw in ("oneOf", "anyOf", "allOf", "enum", "const", "not"):
        assert top_level_schema_refusal({"type": "object", kw: []}), kw
    assert "'array'" in top_level_schema_refusal({"type": "array"})
    assert top_level_schema_refusal(["not", "a", "dict"])


def test_a_refused_schema_is_not_offered():
    names = [s.name for s in _plugin().get_tool_schemas()]
    assert "mcp__kbwiki__search" in names
    assert "mcp__kbwiki__write_page" not in names


def test_executors_and_listing_agree_with_the_schemas():
    plugin = _plugin()
    executors = plugin.get_executors()
    assert "mcp__kbwiki__search" in executors
    assert "mcp__kbwiki__write_page" not in executors
    assert plugin.get_streaming_tool_names() == ["mcp__kbwiki__search"]
    listing = plugin.get_system_instructions() or ""
    assert "mcp__kbwiki__search" in listing
    assert "write_page" not in listing


def test_the_exclusion_is_announced_once_by_name(caplog):
    plugin = _plugin()
    with caplog.at_level(logging.WARNING):
        plugin.get_tool_schemas()
        plugin.get_executors()
        plugin.get_system_instructions()
    hits = [r for r in caplog.records
            if "mcp__kbwiki__write_page" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert hits[0].levelno == logging.WARNING
    assert "'oneOf'" in hits[0].getMessage()


class _RefusingSession:
    def __init__(self, message: str) -> None:
        self._message = message

    def send_message(self, *_a: Any, **_kw: Any) -> str:
        raise RuntimeError(self._message)

    def request_stop(self, reason: str = "") -> bool:
        return True


def test_the_send_error_names_the_tool_not_its_wire_id():
    real = "mcp__kbwiki__write_page"
    wire = name_to_id(real)  # what the provider was sent, and quotes back
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    rpc = RunnerRPC(a, lambda name, args: (False, {"error": "no executor"}))
    rpc._session_host = RunnerSessionHost(
        envelope=SessionInitEnvelope(
            session_id="sess-1631", workspace_path="/tmp/ws",
            profile_name="p", provider_name="openrouter",
            model_name="openai/gpt-4o", plugins=[],
        ),
        runtime=None,
        session=_RefusingSession(
            f"Error code: 400 - Invalid schema for function '{wire}': schema "
            f"must have type 'object' and not have 'oneOf' at the top level."
        ),
    )
    try:
        ok, result = rpc._handle_session_send_message(
            {"prompt": "hi"}, request_id=1,
        )
    finally:
        a.close()
    assert ok is False and result["stage"] == "send"
    assert real in result["error"], result["error"]
    assert wire not in result["error"]
