"""A string argument the schema types otherwise is converted once (#1358).

Some models send ``"operations": "[{...}]"`` or ``"max_results": "5"``.
Each plugin met the string in its own way: ``multiFileEdit`` refused it,
``web_search`` passed it to a library that divided by it, and
``store_memory`` iterated it character by character.  ``ToolExecutor`` now
converts a top-level argument before the permission gate, only when the
conversion is unambiguous and yields the declared type.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from jaato_server.shared import ai_tool_runner
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.plugins.memory.plugin import MemoryPlugin
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_arg_coercion import coerce_args_to_schema, coerce_value

_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"
_COERCE = "jaato-server/jaato_server/shared/tool_arg_coercion.py"
_MEMORY = "jaato-server/jaato_server/shared/plugins/memory/plugin.py"

REVERSIONS = [
    Reversion(
        target=_RUNNER,
        find="        args = self._coerce_to_schema(name, args, call_id)\n",
        replace="",
        test="test_the_executor_hands_the_tool_the_declared_types",
        because="the executor passes a JSON string through to a tool that declared an array",
    ),
    Reversion(
        target=_COERCE,
        find="        if isinstance(value, wanted):\n",
        replace="        if True:\n",
        test="test_json_of_the_wrong_shape_is_left_alone",
        because="an array parameter is handed an object because the string parsed as JSON",
    ),
    Reversion(
        target=_COERCE,
        find='    if not declared or "string" in declared:\n',
        replace="    if not declared:\n",
        test="test_a_parameter_that_accepts_strings_is_left_alone",
        because="a string that is a legitimate value is rewritten",
    ),
    Reversion(
        target=_MEMORY,
        find="        if isinstance(raw_tags, str):\n",
        replace="        if False:\n",
        test="test_store_memory_names_a_string_tags_value",
        because="a string of tags is split into characters and reported as too short",
    ),
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {"type": "array"},
        "options": {"type": "object"},
        "max_results": {"type": "integer"},
        "ratio": {"type": "number"},
        "dry_run": {"type": "boolean"},
        "query": {"type": "string"},
        "either": {"type": ["string", "integer"]},
    },
}


# The rule ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prop, text, expected", [
    ({"type": "array"}, '[{"path": "a"}]', [{"path": "a"}]),
    ({"type": "object"}, '{"k": 1}', {"k": 1}),
    ({"type": "integer"}, " 5 ", 5),
    ({"type": "number"}, "0.5", 0.5),
    ({"type": "number"}, "3", 3),
    ({"type": "boolean"}, "True", True),
    ({"type": "boolean"}, "false", False),
    ({"type": ["integer", "null"]}, "7", 7),
])
def test_an_unambiguous_string_takes_the_declared_type(prop, text, expected):
    assert coerce_value(text, prop) == (True, expected)


@pytest.mark.parametrize("prop, text", [
    ({"type": "array"}, '{"k": 1}'),
    ({"type": "object"}, "[1, 2]"),
    ({"type": "integer"}, "5.5"),
])
def test_json_of_the_wrong_shape_is_left_alone(prop, text):
    assert coerce_value(text, prop) == (False, text)


@pytest.mark.parametrize("prop, text", [
    ({"type": "array"}, "a b c"),
    ({"type": "integer"}, "five"),
    ({"type": "boolean"}, "yes"),
    ({}, "5"),
])
def test_a_string_that_does_not_convert_is_left_alone(prop, text):
    assert coerce_value(text, prop) == (False, text)


def test_a_parameter_that_accepts_strings_is_left_alone():
    assert coerce_value("5", {"type": ["string", "integer"]}) == (False, "5")
    assert coerce_value("[1]", {"type": "string"}) == (False, "[1]")


def test_nothing_changed_returns_the_same_dict():
    args = {"query": "x", "max_results": 3}
    out, changes = coerce_args_to_schema(args, _SCHEMA)
    assert out is args and changes == []


def test_unknown_schema_changes_nothing():
    args = {"max_results": "5"}
    assert coerce_args_to_schema(args, None) == (args, [])


# The executor ─────────────────────────────────────────────────────────


class _Plugin:
    name = "stub"

    def __init__(self) -> None:
        self.schema_reads = 0

    def get_tool_schemas(self):
        self.schema_reads += 1
        return [SimpleNamespace(name="stub_tool", parameters=_SCHEMA)]


class _Registry:
    def __init__(self) -> None:
        self.plugin = _Plugin()

    def get_plugin_for_tool(self, name):
        return self.plugin if name == "stub_tool" else None

    def get_tool_traits(self, name):
        return frozenset()


class _Recording:
    def __init__(self) -> None:
        self.seen: List[Dict[str, Any]] = []

    def check_permission(self, name, args, context=None, call_id=None):
        self.seen.append(dict(args))
        return True, {"method": "whitelist", "reason": "test"}


def _executor(received: List[Dict[str, Any]]) -> ToolExecutor:
    ex = ToolExecutor(auto_background_enabled=False)
    ex._registry = _Registry()
    ex.register("stub_tool", lambda args: received.append(args) or {"ok": True})
    return ex


def test_the_executor_hands_the_tool_the_declared_types():
    received: List[Dict[str, Any]] = []
    ex = _executor(received)
    ok, _ = ex.execute("stub_tool", {
        "operations": '[{"action": "edit"}]',
        "max_results": "5",
        "query": "5",
    })
    assert ok
    assert received == [{
        "operations": [{"action": "edit"}], "max_results": 5, "query": "5",
    }]


def test_the_permission_gate_sees_the_converted_arguments():
    received: List[Dict[str, Any]] = []
    ex = _executor(received)
    gate = _Recording()
    ex.set_permission_plugin(gate)
    ex.execute("stub_tool", {"max_results": "5"})
    assert gate.seen == [{"max_results": 5}]


def test_each_conversion_is_traced(monkeypatch):
    lines: List[str] = []
    monkeypatch.setattr(ai_tool_runner, "_trace_runner", lines.append)
    ex = _executor([])
    ex.execute("stub_tool", {"dry_run": "true"}, call_id="c1")
    assert any(
        line.startswith("coerce: tool=stub_tool call_id=c1 args=dry_run:str->bool")
        for line in lines
    ), lines


def test_the_schema_is_looked_up_once_per_tool():
    ex = _executor([])
    for _ in range(3):
        ex.execute("stub_tool", {"max_results": "1"})
    assert ex._registry.plugin.schema_reads == 1


# store_memory's own error for plain text ─────────────────────────────


def test_store_memory_names_a_string_tags_value():
    p = MemoryPlugin()
    p._storage = MagicMock()
    p._indexer = MagicMock()
    p._get_session_id = lambda: None
    res = p._execute_store({"content": "c", "description": "d",
                            "tags": "jaato sandbox"})
    assert res["status"] == "error"
    assert "array of strings" in res["error"]
    p._storage.save.assert_not_called()
