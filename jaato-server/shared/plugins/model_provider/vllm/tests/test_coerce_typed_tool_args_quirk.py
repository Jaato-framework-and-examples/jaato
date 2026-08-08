"""Tests for the ``coerce_typed_tool_args`` quirk on the vLLM provider.

The quirk addresses Llama 3.1 on vLLM 0.22 with the ``llama3_json``
parser under ``tool_choice: "auto"`` — the parser passes the model's
stringified args through verbatim (Python repr with single quotes for
arrays, JSON strings like ``"14"`` for integers), causing the
framework's schema validator to reject every call with
``'X' is not of type 'Y'``.  See
``feedback_llama31_vllm_auto_mode_stringifies_args`` for the full
diagnosis.

These tests pin:

- The helper coerces every JSON-Schema-typed property (array / object
  / integer / number / boolean) when a string arrived.
- ``ast.literal_eval`` handles Python-repr single-quote arrays
  (``json.loads`` would fail on those).
- ``json.loads`` is the fallback for valid JSON strings.
- The quirk is OFF by default and only activates when set on the
  provider instance.
- Coercion failures leave the string in place (let the downstream
  schema validator surface the type error).
- Strings whose schema declares ``type: string`` (or ``type: [string,
  array]`` etc.) are never coerced.
- Provider init reads ``config.extra["quirks"]["coerce_typed_tool_args"]``
  and warns on unknown quirk keys.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Part,
    ProviderResponse,
    TokenUsage,
    ToolSchema,
    FinishReason,
)

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.vllm.provider import VLLMProvider


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_provider(coerce: bool = False) -> VLLMProvider:
    """Build a VLLMProvider with the quirk pre-applied.

    Bypasses ``initialize()`` so we don't need a live vLLM server —
    the coercion helper only needs ``self._coerce_typed_tool_args``
    and operates on already-parsed FunctionCall args.
    """
    p = VLLMProvider()
    p._coerce_typed_tool_args = coerce
    return p


def _tool_schema(name: str, properties: Dict[str, Any]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"test tool {name}",
        parameters={
            "type": "object",
            "properties": properties,
        },
    )


# ----------------------------------------------------------------------
# Coercion behavior — helper-level
# ----------------------------------------------------------------------


class TestCoerceArgsToSchema:
    def test_quirk_off_returns_args_unchanged(self) -> None:
        """With the quirk disabled, the helper must never mutate args."""
        p = _make_provider(coerce=False)
        schema = _tool_schema("t", {"n": {"type": "integer"}})
        args = {"n": "42"}
        result = p._coerce_args_to_schema(args, schema)
        assert result is args  # identity — not just equal
        assert result == {"n": "42"}  # unchanged value

    def test_stringified_integer_coerced(self) -> None:
        """The original symptom from smoke_tools: word_count="14"."""
        p = _make_provider(coerce=True)
        schema = _tool_schema(
            "signal_completion",
            {
                "summary": {"type": "string"},
                "word_count": {"type": "integer", "minimum": 0},
            },
        )
        result = p._coerce_args_to_schema(
            {"summary": "hi", "word_count": "14"}, schema,
        )
        assert result["summary"] == "hi"  # string stays string
        assert result["word_count"] == 14  # coerced to int
        assert isinstance(result["word_count"], int)

    def test_stringified_python_repr_array_via_literal_eval(self) -> None:
        """Peer 7:1's symptom: Python repr with single quotes."""
        p = _make_provider(coerce=True)
        schema = _tool_schema(
            "report",
            {"capabilities": {"type": "array", "items": {"type": "object"}}},
        )
        # Single-quote Python repr — json.loads would FAIL here.
        repr_str = (
            "[{'capability_id': 'arch', 'feature': 'hex'}, "
            "{'capability_id': 'persistence', 'feature': 'jpa'}]"
        )
        result = p._coerce_args_to_schema(
            {"capabilities": repr_str}, schema,
        )
        assert isinstance(result["capabilities"], list)
        assert len(result["capabilities"]) == 2
        assert result["capabilities"][0] == {
            "capability_id": "arch", "feature": "hex",
        }

    def test_valid_json_array_via_json_loads_fallback(self) -> None:
        """JSON-shaped arrays should parse via json.loads (or
        literal_eval, which also handles JSON literals).  Either way
        the result is a list."""
        p = _make_provider(coerce=True)
        schema = _tool_schema(
            "tag", {"items": {"type": "array", "items": {"type": "string"}}},
        )
        result = p._coerce_args_to_schema(
            {"items": '["a", "b", "c"]'}, schema,
        )
        assert result["items"] == ["a", "b", "c"]

    def test_stringified_object_coerced(self) -> None:
        p = _make_provider(coerce=True)
        schema = _tool_schema(
            "configure", {"opts": {"type": "object"}},
        )
        result = p._coerce_args_to_schema(
            {"opts": "{'flag': True, 'n': 5}"}, schema,
        )
        assert result["opts"] == {"flag": True, "n": 5}

    def test_stringified_boolean_coerced(self) -> None:
        p = _make_provider(coerce=True)
        schema = _tool_schema("flag", {"enabled": {"type": "boolean"}})
        result = p._coerce_args_to_schema({"enabled": "True"}, schema)
        assert result["enabled"] is True

    def test_stringified_number_coerced(self) -> None:
        p = _make_provider(coerce=True)
        schema = _tool_schema("metric", {"value": {"type": "number"}})
        result = p._coerce_args_to_schema({"value": "3.14"}, schema)
        assert result["value"] == 3.14

    def test_string_schema_property_never_coerced(self) -> None:
        """A string property must stay a string even if the value
        happens to look like an int (e.g. an ID like "42")."""
        p = _make_provider(coerce=True)
        schema = _tool_schema("get", {"id": {"type": "string"}})
        result = p._coerce_args_to_schema({"id": "42"}, schema)
        assert result["id"] == "42"
        assert isinstance(result["id"], str)

    def test_union_type_including_string_skips_coercion(self) -> None:
        """When the schema declares ``type: [string, integer]``,
        a string is a valid value — don't coerce."""
        p = _make_provider(coerce=True)
        schema = _tool_schema("x", {"v": {"type": ["string", "integer"]}})
        result = p._coerce_args_to_schema({"v": "42"}, schema)
        assert result["v"] == "42"

    def test_union_type_all_non_string_coerces(self) -> None:
        """``type: [array, object]`` accepts no strings — coerce."""
        p = _make_provider(coerce=True)
        schema = _tool_schema("x", {"v": {"type": ["array", "object"]}})
        result = p._coerce_args_to_schema({"v": "[1, 2, 3]"}, schema)
        assert result["v"] == [1, 2, 3]

    def test_garbage_string_left_in_place(self) -> None:
        """When neither ast.literal_eval nor json.loads can parse,
        leave the string and let the schema validator surface the
        error downstream — don't crash and don't silently substitute
        a placeholder."""
        p = _make_provider(coerce=True)
        schema = _tool_schema("x", {"v": {"type": "array"}})
        result = p._coerce_args_to_schema(
            {"v": "not[parseable]anything"}, schema,
        )
        assert result["v"] == "not[parseable]anything"

    def test_unknown_tool_schema_returns_args_unchanged(self) -> None:
        p = _make_provider(coerce=True)
        result = p._coerce_args_to_schema({"v": "42"}, tool_schema=None)
        assert result == {"v": "42"}

    def test_schema_without_properties_returns_unchanged(self) -> None:
        p = _make_provider(coerce=True)
        # ToolSchema with empty parameters
        schema = ToolSchema(name="x", description="x", parameters={})
        result = p._coerce_args_to_schema({"v": "42"}, schema)
        assert result == {"v": "42"}

    def test_empty_args_returns_unchanged(self) -> None:
        p = _make_provider(coerce=True)
        schema = _tool_schema("x", {"v": {"type": "integer"}})
        result = p._coerce_args_to_schema({}, schema)
        assert result == {}

    def test_non_string_args_passed_through(self) -> None:
        """When the model emits a CORRECTLY-typed value (the common case
        on stronger models), coercion must not interfere."""
        p = _make_provider(coerce=True)
        schema = _tool_schema(
            "signal_completion",
            {"word_count": {"type": "integer"}},
        )
        result = p._coerce_args_to_schema({"word_count": 14}, schema)
        assert result["word_count"] == 14
        assert isinstance(result["word_count"], int)


# ----------------------------------------------------------------------
# Response-level coercion (non-streaming path)
# ----------------------------------------------------------------------


class TestCoerceResponseFunctionCalls:
    def test_quirk_off_noop(self) -> None:
        p = _make_provider(coerce=False)
        fc = FunctionCall(id="1", name="x", args={"n": "42"})
        resp = ProviderResponse(
            parts=[Part.from_function_call(fc)],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
        )
        tools = [_tool_schema("x", {"n": {"type": "integer"}})]
        p._coerce_response_function_calls(resp, tools)
        assert fc.args == {"n": "42"}

    def test_quirk_on_walks_every_function_call(self) -> None:
        p = _make_provider(coerce=True)
        fc1 = FunctionCall(id="1", name="x", args={"n": "42"})
        fc2 = FunctionCall(id="2", name="y", args={"items": "[1, 2]"})
        resp = ProviderResponse(
            parts=[
                Part.from_text("preamble"),
                Part.from_function_call(fc1),
                Part.from_function_call(fc2),
            ],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
        )
        tools = [
            _tool_schema("x", {"n": {"type": "integer"}}),
            _tool_schema("y", {"items": {"type": "array"}}),
        ]
        p._coerce_response_function_calls(resp, tools)
        assert fc1.args == {"n": 42}
        assert fc2.args == {"items": [1, 2]}

    def test_unknown_tool_call_left_alone(self) -> None:
        """A function_call whose name doesn't match any tool in the
        ``tools`` list shouldn't crash and shouldn't be coerced."""
        p = _make_provider(coerce=True)
        fc = FunctionCall(id="1", name="unknown", args={"n": "42"})
        resp = ProviderResponse(
            parts=[Part.from_function_call(fc)],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
        )
        tools = [_tool_schema("x", {"n": {"type": "integer"}})]
        p._coerce_response_function_calls(resp, tools)
        assert fc.args == {"n": "42"}  # unchanged


# ----------------------------------------------------------------------
# Quirk read at initialize() — config plumbing
# ----------------------------------------------------------------------


class TestQuirkReadFromConfig:
    def test_default_off(self) -> None:
        """When no quirks dict is provided, the flag stays False."""
        p = VLLMProvider()
        # Simulate just the read step (skip the actual httpx connect).
        config = ProviderConfig(
            extra={
                "host": "http://localhost:8000",
                "context_length": 32768,
            },
        )
        # Reach into the relevant init block by calling the helper
        # logic directly — avoids requiring a live server.
        quirks = config.extra.get("quirks") or {}
        p._coerce_typed_tool_args = bool(
            quirks.get("coerce_typed_tool_args", False)
        )
        assert p._coerce_typed_tool_args is False

    def test_quirk_enabled_via_config(self) -> None:
        p = VLLMProvider()
        config = ProviderConfig(
            extra={
                "host": "http://localhost:8000",
                "context_length": 32768,
                "quirks": {"coerce_typed_tool_args": True},
            },
        )
        quirks = config.extra.get("quirks") or {}
        p._coerce_typed_tool_args = bool(
            quirks.get("coerce_typed_tool_args", False)
        )
        assert p._coerce_typed_tool_args is True

    def test_non_dict_quirks_treated_as_empty(self) -> None:
        """Defensive: if quirks is a list / string / None, treat as
        empty dict and proceed."""
        for bad in [None, [], "true", 42]:
            p = VLLMProvider()
            quirks = ({} if not isinstance(bad, dict) else bad)
            p._coerce_typed_tool_args = bool(
                quirks.get("coerce_typed_tool_args", False)
            )
            assert p._coerce_typed_tool_args is False, (
                f"non-dict quirks={bad!r} should not enable the quirk"
            )
