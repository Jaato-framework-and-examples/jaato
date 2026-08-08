"""Tests for shared/rewind.py — truncated tool-call detection."""

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Part,
    ProviderResponse,
    ToolSchema,
)

from shared.rewind import (
    REASON_MAX_TOKENS_EMPTY_ARGS,
    REASON_MAX_TOKENS_MISSING_REQUIRED,
    detect_truncated_tool_call,
    find_truncated_call_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema(
    name: str,
    required: list = None,
    properties: dict = None,
) -> ToolSchema:
    """Build a ToolSchema with a JSON-schema parameters object."""
    params = {"type": "object", "properties": properties or {}}
    if required is not None:
        params["required"] = required
    return ToolSchema(name=name, description=f"{name} tool", parameters=params)


def _response(
    finish_reason: FinishReason,
    calls: list = None,
    text: str = "",
) -> ProviderResponse:
    """Build a ProviderResponse with optional text + function calls."""
    parts = []
    if text:
        parts.append(Part(text=text))
    for call in calls or []:
        parts.append(Part(function_call=call))
    return ProviderResponse(parts=parts, finish_reason=finish_reason)


def _fc(name: str, args: dict = None, call_id: str = "c1") -> FunctionCall:
    return FunctionCall(id=call_id, name=name, args=args or {})


WRITE_FILE_SCHEMA = _schema(
    "writeNewFile",
    required=["path", "content"],
    properties={"path": {"type": "string"}, "content": {"type": "string"}},
)

SHELL_LIST_SCHEMA = _schema("shell_list")  # no required params

LIST_DIRECTORY_SCHEMA = _schema(
    "listDirectory",
    required=["path"],
    properties={"path": {"type": "string"}},
)


# ---------------------------------------------------------------------------
# detect_truncated_tool_call
# ---------------------------------------------------------------------------


class TestDetector:

    # Positive cases — detector fires

    def test_max_tokens_plus_empty_args_on_required_tool_fires(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("writeNewFile", args={})],
            text="I am about to write a large file.",
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) == (
            REASON_MAX_TOKENS_EMPTY_ARGS
        )

    def test_max_tokens_plus_missing_required_key_fires(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("writeNewFile", args={"path": "a.txt"})],  # content missing
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) == (
            REASON_MAX_TOKENS_MISSING_REQUIRED
        )

    def test_empty_args_takes_precedence_over_missing_required(self):
        """When both shapes appear in the same response, report the empty-args one."""
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[
                _fc("writeNewFile", args={}),
                _fc("listDirectory", args={"extra": "x"}),  # missing 'path'
            ],
        )
        assert detect_truncated_tool_call(
            resp, [WRITE_FILE_SCHEMA, LIST_DIRECTORY_SCHEMA]
        ) == REASON_MAX_TOKENS_EMPTY_ARGS

    # Negative cases — detector does NOT fire

    def test_stop_finish_reason_does_not_fire(self):
        resp = _response(
            FinishReason.STOP,
            calls=[_fc("writeNewFile", args={})],
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) is None

    def test_tool_use_finish_reason_does_not_fire(self):
        resp = _response(
            FinishReason.TOOL_USE,
            calls=[_fc("writeNewFile", args={})],
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) is None

    def test_max_tokens_without_function_calls_does_not_fire(self):
        """Text-only response that hit the cap is a normal MAX_TOKENS case
        the existing abnormal-termination classifier should handle."""
        resp = _response(FinishReason.MAX_TOKENS, text="I was going to say...")
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) is None

    def test_max_tokens_with_legitimate_empty_args_tool_does_not_fire(self):
        """shell_list has no required params — {} is a valid call."""
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("shell_list", args={})],
        )
        assert detect_truncated_tool_call(resp, [SHELL_LIST_SCHEMA]) is None

    def test_max_tokens_with_fully_populated_args_does_not_fire(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("writeNewFile", args={"path": "a", "content": "b"})],
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) is None

    def test_unknown_tool_is_conservatively_skipped(self):
        """If we can't find a schema for the tool, don't false-positive —
        the executor will surface 'no executor registered' on its own."""
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("mystery_tool", args={})],
        )
        assert detect_truncated_tool_call(resp, [WRITE_FILE_SCHEMA]) is None

    def test_malformed_schema_does_not_crash(self):
        """required key being non-list should be tolerated."""
        weird = ToolSchema(
            name="weird",
            description="weird",
            parameters={"required": "not-a-list"},  # intentionally malformed
        )
        resp = _response(FinishReason.MAX_TOKENS, calls=[_fc("weird", args={})])
        # required normalizes to []; no false positive
        assert detect_truncated_tool_call(resp, [weird]) is None

    def test_empty_parameters_dict_treated_as_no_required(self):
        """A schema with parameters={} (no 'required' key) → no flag."""
        bare = ToolSchema(name="bare", description="bare", parameters={})
        resp = _response(FinishReason.MAX_TOKENS, calls=[_fc("bare", args={})])
        assert detect_truncated_tool_call(resp, [bare]) is None


# ---------------------------------------------------------------------------
# find_truncated_call_name
# ---------------------------------------------------------------------------


class TestFindName:

    def test_returns_first_truncated_call_name(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("writeNewFile", args={})],
        )
        assert find_truncated_call_name(resp, [WRITE_FILE_SCHEMA]) == "writeNewFile"

    def test_skips_legitimate_empty_args_tool(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[
                _fc("shell_list", args={}),
                _fc("writeNewFile", args={}),
            ],
        )
        schemas = [SHELL_LIST_SCHEMA, WRITE_FILE_SCHEMA]
        assert find_truncated_call_name(resp, schemas) == "writeNewFile"

    def test_returns_none_when_no_truncation(self):
        resp = _response(FinishReason.STOP, calls=[_fc("writeNewFile", args={})])
        assert find_truncated_call_name(resp, [WRITE_FILE_SCHEMA]) is None

    def test_returns_none_when_no_function_calls(self):
        resp = _response(FinishReason.MAX_TOKENS, text="hi")
        assert find_truncated_call_name(resp, [WRITE_FILE_SCHEMA]) is None

    def test_returns_none_for_unknown_tool(self):
        resp = _response(
            FinishReason.MAX_TOKENS,
            calls=[_fc("mystery", args={})],
        )
        assert find_truncated_call_name(resp, [WRITE_FILE_SCHEMA]) is None
