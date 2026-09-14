"""Tests for the Bedrock provider itself.

The wire conversion has its own suite; what is covered here is everything
around the request — the two things Bedrock's catalog does not report and the
provider therefore refuses to guess, the stream fold, and the error
translation that decides what ``with_retry`` ever sees.

``boto3`` is imported lazily inside ``initialize()``, so every case here runs
without it: the client is a stub the test injects.
"""

import pytest

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    Message,
    Part,
    Role,
    StreamInterruptedError,
    ToolSchema,
)
from shared.tool_id_map import name_to_id

from ..errors import (
    AccessDeniedError,
    ContextLengthNotConfiguredError,
    ModelNotFoundError,
    ModelNotReadyError,
    ServiceUnavailableError,
    ThrottlingError,
)
from ..provider import BedrockProvider, create_provider


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("JAATO_BEDROCK_REGION", "JAATO_BEDROCK_MODEL",
                "JAATO_BEDROCK_CONTEXT_LENGTH", "JAATO_BEDROCK_PROFILE",
                "JAATO_BEDROCK_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)


class _StubClient:
    """A ``bedrock-runtime`` stand-in that records what it was sent."""

    def __init__(self, stream=None, batch=None, raises=None):
        self.stream = stream or []
        self.batch = batch
        self.raises = raises
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        if self.raises:
            raise self.raises
        return self.batch or {"output": {"message": {"content": []}},
                              "stopReason": "end_turn"}

    def converse_stream(self, **kwargs):
        self.requests.append(kwargs)
        if self.raises:
            raise self.raises
        return {"stream": iter(self.stream)}


def _connected(stream=None, batch=None, raises=None, **knobs):
    """A provider with a stub client, past ``initialize()``/``connect()``."""
    provider = BedrockProvider()
    provider._client = _StubClient(stream=stream, batch=batch, raises=raises)
    provider._model_name = knobs.pop(
        "model", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    provider._region = "us-east-1"
    provider._context_length = 200_000
    for key, value in knobs.items():
        setattr(provider, f"_{key}", value)
    return provider


class _ClientError(Exception):
    """Shaped like botocore's ``ClientError`` without importing botocore."""

    def __init__(self, code, message="boom", retry_after=None):
        super().__init__(f"An error occurred ({code}): {message}")
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = {"Error": {"Code": code, "Message": message},
                         "ResponseMetadata": {"HTTPHeaders": headers}}


# ------------------------------------------------- what Bedrock does not report

def test_connect_refuses_a_model_with_no_context_window():
    """Bedrock's catalog reports no capacity, so an unconfigured window is a
    configuration error — a guessed one truncates a session silently."""
    provider = BedrockProvider()
    provider._client = _StubClient()
    with pytest.raises(ContextLengthNotConfiguredError) as exc:
        provider.connect("anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert "context_length" in str(exc.value)


def test_the_configured_window_is_what_get_context_limit_returns():
    assert _connected().get_context_limit() == 200_000


def test_modalities_come_from_the_family_table_not_the_catalog():
    """The catalog's vocabulary (TEXT|IMAGE|EMBEDDING) cannot express the
    ``document`` block Converse plainly carries, so the table answers."""
    assert _connected().modalities() == {"text", "image", "file"}


def test_a_geography_prefix_does_not_hide_the_family():
    """Cross-region inference profiles prefix the id; the prefix routes the
    request and says nothing about the model."""
    provider = _connected()
    assert provider.modalities("eu.anthropic.claude-3-5-sonnet-20241022-v2:0") == {
        "text", "image", "file"}
    assert provider.modalities("apac.amazon.nova-pro-v1:0") == {
        "text", "image", "file", "video"}


def test_a_model_outside_the_table_is_text_only_never_a_false_claim():
    assert _connected().modalities("cohere.command-r-v1:0") == {"text"}


def test_the_modalities_knob_outranks_the_table():
    provider = _connected(modalities_knob=["text", "image", "audio"])
    assert provider.modalities() == {"text", "image", "audio"}


def test_a_non_list_modalities_knob_is_refused_at_initialize():
    provider = BedrockProvider()
    with pytest.raises(TypeError):
        provider._read_knobs({"modalities": "image"}, {})


# ------------------------------------------------------------ request assembly

def test_max_tokens_always_ships():
    """Converse has no server-side default worth relying on across the seven
    vendors behind one endpoint."""
    provider = _connected()
    request = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, None, None)
    assert request["inferenceConfig"]["maxTokens"] == 8192


def test_api_params_reach_inference_config_under_bedrocks_names():
    provider = BedrockProvider()
    provider._read_knobs({}, {"temperature": 0.2, "top_p": 0.9,
                              "max_tokens": 512, "stop_sequences": ["END"]})
    provider._client = _StubClient()
    provider._model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    provider._context_length = 200_000
    config = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, None, None
    )["inferenceConfig"]
    assert config == {"maxTokens": 512, "temperature": 0.2, "topP": 0.9,
                      "stopSequences": ["END"]}


def test_the_profiles_own_additional_fields_win_over_the_frameworks():
    """``additionalModelRequestFields`` is Converse's escape hatch, and an
    escape hatch that the framework can overrule is not one."""
    provider = _connected(
        enable_thinking=True,
        additional_request_fields={"reasoning_config": {"type": "disabled"}})
    request = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, None, None)
    assert request["additionalModelRequestFields"]["reasoning_config"] == {
        "type": "disabled"}


def test_thinking_when_asked_for_uses_the_anthropic_family_spelling():
    provider = _connected(enable_thinking=True, thinking_budget=2048)
    request = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, None, None)
    assert request["additionalModelRequestFields"]["reasoning_config"] == {
        "type": "enabled", "budget_tokens": 2048}


def test_no_tools_means_no_tool_config():
    provider = _connected()
    request = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, None, "required")
    assert "toolConfig" not in request


def test_a_per_call_tool_choice_overrides_the_profiles():
    provider = _connected(tool_choice_default="auto")
    tools = [ToolSchema(name="readFile", description="d", parameters={})]
    request = provider._build_request(
        [Message.from_text(Role.USER, "hi")], None, tools,
        {"type": "function", "function": {"name": "readFile"}})
    assert request["toolConfig"]["toolChoice"] == {
        "tool": {"name": name_to_id("readFile")}}


# ------------------------------------------------------------------- streaming

def _text_stream():
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "he"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "llo"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 2,
                                "totalTokens": 9}}},
    ]


def test_a_streamed_text_turn_is_assembled():
    chunks = []
    provider = _connected(stream=_text_stream())
    result = provider.complete([Message.from_text(Role.USER, "hi")],
                               on_chunk=chunks.append)
    assert chunks == ["he", "llo"]
    assert result.text == "hello"
    assert result.finish_reason == FinishReason.STOP
    assert provider.get_token_usage().prompt_tokens == 7


def test_a_streamed_tool_call_is_assembled_from_its_json_fragments():
    """Converse streams tool arguments as text fragments, keyed by block index."""
    seen = []
    provider = _connected(stream=[
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"text": "let me look"}}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {
            "toolUseId": "tu-1", "name": name_to_id("readFile")}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1,
                               "delta": {"toolUse": {"input": '{"pa'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1,
                               "delta": {"toolUse": {"input": 'th": "x"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ])
    result = provider.complete([Message.from_text(Role.USER, "hi")],
                               on_chunk=lambda _c: None,
                               on_function_call=seen.append)
    assert [c.name for c in seen] == ["readFile"]
    assert seen[0].args == {"path": "x"}
    assert seen[0].unreadable_args is None
    assert result.finish_reason == FinishReason.TOOL_USE


def test_severed_tool_arguments_stay_unreadable():
    """A stream cut mid-object leaves a severed string, and an empty dict
    would present that as a zero-argument call (#750)."""
    seen = []
    provider = _connected(stream=[
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {
            "toolUseId": "tu-1", "name": name_to_id("readFile")}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"toolUse": {"input": '{"pa'}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ])
    provider.complete([Message.from_text(Role.USER, "hi")],
                      on_chunk=lambda _c: None, on_function_call=seen.append)
    assert seen[0].unreadable_args == '{"pa'


def test_reasoning_is_delivered_once_before_the_first_answer_token():
    thoughts = []
    provider = _connected(stream=[
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"reasoningContent": {"text": "hm"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"reasoningContent": {"text": "mm"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "a"}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "b"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ])
    provider.complete([Message.from_text(Role.USER, "hi")],
                      on_chunk=lambda _c: None, on_thinking=thoughts.append)
    assert thoughts == ["hmmm"]


def test_a_stream_that_never_says_it_ended_is_not_a_finished_turn():
    """#687: no ``messageStop`` means the stream was cut, and the fragment
    must not be returned as an answer."""
    provider = _connected(stream=[
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "par"}}},
    ])
    with pytest.raises(StreamInterruptedError):
        provider.complete([Message.from_text(Role.USER, "hi")],
                          on_chunk=lambda _c: None)


def test_a_cancelled_turn_contributes_no_tool_calls():
    """An unanswered ``toolUse`` block poisons the next request."""
    token = CancelToken()

    def cancel_on_first(_chunk):
        token.cancel("user")

    provider = _connected(stream=[
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {
            "toolUseId": "tu-1", "name": name_to_id("readFile")}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "x"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ])
    result = provider.complete([Message.from_text(Role.USER, "hi")],
                               on_chunk=cancel_on_first, cancel_token=token)
    assert result.finish_reason == FinishReason.CANCELLED


def test_a_batch_turn_needs_no_streaming_callback():
    provider = _connected(batch={
        "output": {"message": {"role": "assistant",
                               "content": [{"text": "hi there"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
    })
    result = provider.complete([Message.from_text(Role.USER, "hi")])
    assert result.text == "hi there"


# ------------------------------------------------------------ error handling

@pytest.mark.parametrize("code,expected", [
    ("ThrottlingException", ThrottlingError),
    ("TooManyRequestsException", ThrottlingError),
    ("ServiceUnavailableException", ServiceUnavailableError),
    ("InternalServerException", ServiceUnavailableError),
    ("ModelNotReadyException", ModelNotReadyError),
])
def test_transient_failures_are_raised_so_with_retry_sees_them(code, expected):
    provider = _connected(raises=_ClientError(code))
    with pytest.raises(expected):
        provider.complete([Message.from_text(Role.USER, "hi")])


@pytest.mark.parametrize("code,expected_type", [
    ("AccessDeniedException", AccessDeniedError),
    ("ResourceNotFoundException", ModelNotFoundError),
])
def test_terminal_failures_become_a_turn_result_not_a_retry(code, expected_type):
    """Retrying a 403 spends the whole budget on the same failure."""
    provider = _connected(raises=_ClientError(code))
    result = provider.complete([Message.from_text(Role.USER, "hi")])
    assert result.is_error
    assert isinstance(provider._translate_error(_ClientError(code)),
                      expected_type)


def test_an_access_denial_names_model_access_not_only_iam():
    """On Bedrock the common cause is the model not being enabled for the
    account, which looks identical to a policy problem."""
    message = str(AccessDeniedError(model="m", region="us-east-1"))
    assert "Model access" in message


def test_an_unrecognised_failure_is_re_raised_unchanged():
    """Laundering an unknown failure into a terminal answer hides it."""
    boom = _ClientError("SomeFutureException")
    provider = _connected(raises=boom)
    with pytest.raises(_ClientError):
        provider.complete([Message.from_text(Role.USER, "hi")])


def test_retry_after_is_read_off_the_response_header():
    """Bedrock returns the hint as a header, not in the error body."""
    provider = _connected()
    assert provider.get_retry_after(
        _ClientError("ThrottlingException", retry_after="12")) == 12.0


def test_classify_error_splits_rate_limit_from_infrastructure():
    provider = _connected()
    assert provider.classify_error(ThrottlingError())["rate_limit"] is True
    assert provider.classify_error(ServiceUnavailableError())["infra"] is True
    assert provider.classify_error(ValueError("unrelated")) is None


# ------------------------------------------------------------------- plumbing

def test_the_factory_produces_the_provider():
    assert create_provider().name == "bedrock"


def test_completing_before_connect_is_a_programming_error():
    with pytest.raises(RuntimeError):
        BedrockProvider().complete([Message.from_text(Role.USER, "hi")])


def test_history_round_trips_through_the_provider():
    provider = _connected()
    history = [Message(role=Role.USER, parts=[Part(text="hi")])]
    assert provider.deserialize_history(
        provider.serialize_history(history))[0].text == "hi"
