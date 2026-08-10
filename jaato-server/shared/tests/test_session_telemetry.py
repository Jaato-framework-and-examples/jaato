"""Unit tests for the pure telemetry helpers in shared.session_telemetry.

These functions were extracted from JaatoSession; the tests pin their
data-mapping and classification behavior so the conversions can evolve
independently of the session God object.
"""

from __future__ import annotations

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
)

from shared.session_telemetry import (
    build_input_messages,
    classify_cache_outcome,
    history_to_openinference,
    response_to_openinference,
)


class TestResponseToOpenInference:
    def test_empty_response_maps_to_empty_list(self):
        assert response_to_openinference(ProviderResponse(parts=[])) == []

    def test_text_only(self):
        resp = ProviderResponse(parts=[Part.from_text("hello world")])
        out = response_to_openinference(resp)
        assert out == [{"role": "assistant", "content": "hello world"}]

    def test_with_tool_calls(self):
        resp = ProviderResponse(parts=[
            Part.from_text("calling"),
            Part.from_function_call(
                FunctionCall(id="1", name="ls", args={"path": "/tmp"})
            ),
        ])
        out = response_to_openinference(resp)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "calling"
        assert out[0]["tool_calls"] == [
            {"name": "ls", "arguments": '{"path": "/tmp"}'}
        ]

    def test_tool_call_without_args_serializes_empty_object(self):
        resp = ProviderResponse(parts=[
            Part.from_function_call(FunctionCall(id="1", name="now", args={})),
        ])
        out = response_to_openinference(resp)
        assert out[0]["content"] == ""
        assert out[0]["tool_calls"] == [{"name": "now", "arguments": "{}"}]


class TestHistoryToOpenInference:
    def test_model_role_normalized_to_assistant(self):
        history = [
            Message(role=Role.USER, parts=[Part.from_text("hi")]),
            Message(role=Role.MODEL, parts=[Part.from_text("hello")]),
        ]
        out = history_to_openinference(history)
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_model_message_includes_tool_calls(self):
        history = [
            Message(role=Role.MODEL, parts=[
                Part.from_text("running"),
                Part.from_function_call(
                    FunctionCall(id="1", name="grep", args={"q": "x"})
                ),
            ]),
        ]
        out = history_to_openinference(history)
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "running"
        assert out[0]["tool_calls"] == [
            {"name": "grep", "arguments": '{"q": "x"}'}
        ]

    def test_empty_parts_yield_empty_content(self):
        history = [Message(role=Role.USER, parts=[])]
        out = history_to_openinference(history)
        assert out == [{"role": "user", "content": ""}]

    def test_empty_history(self):
        assert history_to_openinference([]) == []


class TestBuildInputMessages:
    def test_prepends_system_prompt_as_first_message(self):
        # Regression: the system prompt reaches the provider as a separate API
        # param, so without this it is dropped from the span entirely.
        history = [Message(role=Role.USER, parts=[Part.from_text("hi")])]
        out = build_input_messages("You are helpful.", history)
        assert out[0] == {"role": "system", "content": "You are helpful."}
        assert out[1] == {"role": "user", "content": "hi"}

    def test_no_system_prompt_leaves_history_unchanged(self):
        history = [Message(role=Role.USER, parts=[Part.from_text("hi")])]
        assert build_input_messages(None, history) == history_to_openinference(history)
        assert build_input_messages("", history) == history_to_openinference(history)

    def test_system_only_when_history_empty(self):
        assert build_input_messages("SYS", []) == [{"role": "system", "content": "SYS"}]


class TestClassifyCacheOutcome:
    def test_unknown_when_no_input(self):
        assert classify_cache_outcome(0, None, None) == "unknown"

    def test_hit_when_mostly_cached(self):
        # read=900, new=100 -> ratio 0.9
        assert classify_cache_outcome(100, 900, 0) == "hit"

    def test_partial_when_some_cached(self):
        # read=300, new=700 -> ratio 0.3
        assert classify_cache_outcome(700, 300, 0) == "partial"

    def test_warm_when_creation_only(self):
        # no reads, but creation > 0
        assert classify_cache_outcome(500, 0, 500) == "warm"

    def test_miss_when_no_cache_activity(self):
        assert classify_cache_outcome(500, 0, 0) == "miss"

    def test_ratio_caps_at_one_on_cache_warm_turn(self):
        # Regression: dividing by prompt_tokens alone produced ratios > 1.0
        # (e.g. 26580/719); the read/(read+new) form caps at 1.0 -> "hit".
        assert classify_cache_outcome(719, 26580, 0) == "hit"
