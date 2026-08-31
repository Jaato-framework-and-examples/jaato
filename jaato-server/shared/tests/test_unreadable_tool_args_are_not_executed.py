"""Arguments that could not be read must not become a call that runs.

THE GAP (#750).  Every provider decoded a tool call's wire
``arguments`` with the same three lines::

    try:
        args = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError:
        args = {}

and then built a well-formed ``FunctionCall`` out of the result.  "I
could not read the arguments" became "the model called this tool with
no arguments", and the session executed it.  The model never made that
call; absence and emptiness shared one representation, so nothing
downstream could tell them apart.  A handful of sites wrote
``{"raw": ...}`` or ``{"_partial": ...}`` instead, which is the same
fabrication wearing a label -- an argument mapping with a key no tool
declares.

It was load-bearing in a real incident.  An eval arm hit the output cap
mid-``arguments``; the partial JSON failed to parse, ``args = {}``
produced a plausible zero-argument write call, and because the
truncated turn also reported ``TOOL_USE`` the loop continued.  46
requests in an hour, nothing committed.  #745 closed the truncation
route by fixing the finish reason, but truncation was never the only
way in: a weaker model emitting malformed JSON, a prose-tool-call parse
failure, any provider-side encoding bug all arrive with an ordinary
``tool_calls`` finish, and the turn continues.

For a read-only tool that is a wasted turn.  For a writer, a delete, or
a shell invocation, "no arguments" is not obviously safe -- and the
required-argument check that would have caught it lives *downstream* of
a call that now looks valid.

THE CONTRACT, in two halves.

  * ``parse_tool_call_arguments`` never invents a value: a parse
    failure returns ``({}, raw_text)`` and the raw text lands on
    ``FunctionCall.unreadable_args``.  Every provider calls it instead
    of catching ``JSONDecodeError`` itself.
  * ``JaatoSession`` refuses to execute a call carrying
    ``unreadable_args`` and reports the failure back as a tool error the
    model can see.  An error result is still a tool *output*, so the
    call stays paired in history (a dropped call would leave a dangling
    ``tool_use`` block and the next request would be rejected) and the
    model learns its call was unreadable and can re-emit it.

WHY A GUARD AND NOT JUST THE PROVIDER TESTS.  The trap here is a test
that feeds *valid* JSON and checks the args round-trip: it passes
today, and it would keep passing after any regression, because the
defect only shows on input that fails to parse.  The assertion has to
be on malformed input.  And the cross-provider half has to be checked
at the source -- 17 sites across seven providers acquired this by
copy-paste, exactly as the finish-reason overwrite in #745 did, and
there is no shared entry point to drive them all through.

NOT COVERED, deliberately: ``_prose_tools.parse_tool_calls`` already
complies -- a block whose JSON does not parse is left in the text
verbatim and yields no call at all.  Its ``{"value": <scalar>}``
coercion is a *documented feature* of the prose protocol for a call
whose arguments parsed fine but were not an object, not a parse
failure, so it is out of scope here.
"""

import ast
import json
import pathlib
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    UNREADABLE_ARGS_EXCERPT_CHARS,
    parse_tool_call_arguments,
    unreadable_arguments_error,
)

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


PROVIDER_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "plugins" / "model_provider"
)

#: Provider modules that decode wire ``arguments`` into a
#: ``FunctionCall``, and the number of decode sites each one has.
#: Pinned so that deleting a call site fails here instead of silently
#: reintroducing a hand-rolled ``try/except``; a new decode path is
#: expected to bump the count along with its call.
#:
#: ``github_models`` carries four because it speaks three wire dialects
#: (Azure inference, Copilot chat, Copilot Responses) and its
#: converters carry a fourth for streaming-delta reassembly.  The
#: ``converters`` entries are the history-rebuild direction, which had
#: the ``{"raw": ...}`` / ``{"_partial": ...}`` variant of the defect.
PARSING_PROVIDERS = {
    "_openai_compat/base.py": 1,
    "_openai_compat/converters.py": 2,
    "anthropic/provider.py": 2,
    "github_models/converters.py": 4,
    "github_models/provider.py": 4,
    "nebius/converters.py": 2,
    "openrouter/converters.py": 2,
    "openrouter/provider.py": 1,
    "vllm/provider.py": 1,
}


#: The defect, put back.  ``_openai_compat/base.py`` stands in for the
#: whole OpenAI-compatible family (``nim``, ``nebius``, ``ovhcloud``,
#: ``lmstudio``, ``tensorrt_llm``, ``doubleword``, ``triton``,
#: ``zhipuai_openai`` all inherit its streaming path).
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, raw""",
        replace="""    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, None""",
        test="test_unparseable_arguments_never_decode_to_a_value",
        because="a parse failure being turned back into an empty "
                "argument dict, which is indistinguishable from a "
                "genuine zero-argument call",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            if fc.unreadable_args is not None:
                # The provider could not decode this call's arguments, so""",
        replace="""            if False:
                # The provider could not decode this call's arguments, so""",
        test="test_the_session_refuses_a_call_it_could_not_read",
        because="the session executing a call whose arguments never "
                "arrived, against the workspace",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "_openai_compat/base.py"
        ),
        find="""                    args, unreadable_args = parse_tool_call_arguments(
                        tc.get("function", {}).get("arguments")
                    )""",
        replace="""                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    unreadable_args = None""",
        test=(
            "test_no_provider_swallows_a_decode_failure_into_arguments"
        ),
        because="a provider going back to catching JSONDecodeError "
                "itself, which is how seven providers acquired this "
                "defect at once",
    ),
]


# ==================== The contract itself ====================


UNPARSEABLE = [
    pytest.param('{"path": "a.py", "content": "----------',
                 id="severed-mid-string"),
    pytest.param('{"path": "a.py", ', id="severed-mid-object"),
    pytest.param("not json at all", id="prose"),
    pytest.param("{'path': 'a.py'}", id="python-repr"),
    pytest.param('{"path": "a.py"}}', id="trailing-garbage"),
    pytest.param("null", id="json-null"),
    pytest.param("[1, 2]", id="json-array"),
    pytest.param("42", id="json-number"),
    pytest.param('"a string"', id="json-string"),
]


@pytest.mark.parametrize("raw", UNPARSEABLE)
def test_unparseable_arguments_never_decode_to_a_value(raw):
    """No input that cannot be read yields arguments to run with."""
    args, unreadable = parse_tool_call_arguments(raw)
    assert args == {}
    assert unreadable == raw, (
        f"{raw!r} was discarded rather than preserved. Without the raw "
        f"text the session cannot tell the model what was unreadable, "
        f"and cannot tell a failed decode from a zero-argument call."
    )


@pytest.mark.parametrize("raw", UNPARSEABLE)
def test_an_unreadable_call_is_distinguishable_from_a_real_empty_one(raw):
    """The whole defect in one assertion.

    A genuine zero-argument call and a call whose arguments could not be
    read both end up with ``args == {}``.  The only thing separating
    them is ``unreadable_args`` -- which is what the session keys on.
    """
    genuine = FunctionCall(id="c1", name="list_files",
                           args=parse_tool_call_arguments("{}")[0])
    args, unreadable = parse_tool_call_arguments(raw)
    fabricated = FunctionCall(id="c2", name="list_files", args=args,
                              unreadable_args=unreadable)

    assert genuine.args == fabricated.args  # the trap: these are equal
    assert genuine.unreadable_args is None
    assert fabricated.unreadable_args is not None


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_an_absent_arguments_slot_is_a_genuine_zero_argument_call(raw):
    """The case the ``{}`` default existed to serve is preserved.

    Without this the guard above could be satisfied by refusing every
    call, which would break every zero-argument tool in the fleet.
    """
    assert parse_tool_call_arguments(raw) == ({}, None)


def test_well_formed_arguments_still_decode():
    assert parse_tool_call_arguments('{"path": "a.py", "n": 1}') == (
        {"path": "a.py", "n": 1}, None,
    )


def test_a_pre_decoded_dict_passes_through():
    """Some SDKs hand us the object already decoded."""
    assert parse_tool_call_arguments({"path": "a.py"}) == (
        {"path": "a.py"}, None,
    )


def test_the_error_quotes_a_bounded_excerpt_back_to_the_model():
    """The model is told what was unreadable, within a budget.

    Quoting the raw text is what #749's truncation nudge needs; quoting
    all of it would re-admit the 60k-token blob that blew the cap in the
    first place.
    """
    raw = '{"content": "' + "-" * 50_000
    call = FunctionCall(id="c1", name="write_file", args={},
                        unreadable_args=raw)
    payload = unreadable_arguments_error(call)

    assert "write_file" in payload["error"]
    excerpt = payload["unreadable_arguments"]
    assert len(excerpt) <= UNREADABLE_ARGS_EXCERPT_CHARS + 3  # + ellipsis
    assert excerpt.startswith('{"content": "')
    assert payload["unreadable_arguments_length"] == len(raw)
    # Serializable: it travels to the model as a tool result.
    json.dumps(payload)


# ==================== The session half ====================


def _session_with_stub_executor():
    """A ``JaatoSession`` shell wired with just enough to run one tool.

    Built with ``__new__`` rather than ``__init__`` for the same reason
    the other session guards do: constructing a real session needs a
    provider, a runtime and a registry, none of which this contract
    touches.
    """
    from shared.jaato_session import JaatoSession

    session = JaatoSession.__new__(JaatoSession)
    session._agent_id = "test-agent"
    session._runtime = MagicMock()
    session._runtime.registry = None
    session._ui_hooks = None
    session._executor = MagicMock()
    session._executor.execute = MagicMock(
        side_effect=AssertionError(
            "the executor was reached for a call whose arguments could "
            "not be read (#750)"
        )
    )
    session._cancel_token = None
    # ``_telemetry`` is a property reading ``_runtime.telemetry``; the
    # MagicMock runtime supplies a context-manager-shaped stub.
    session._trace = lambda msg: None
    session._forward_to_parent = lambda *a, **k: None
    session._is_streaming_tool = lambda name: False
    return session


def _unreadable_call():
    args, unreadable = parse_tool_call_arguments(
        '{"path": "notes.md", "content": "----------'
    )
    return FunctionCall(id="call_abc", name="write_file", args=args,
                        unreadable_args=unreadable)


def test_the_session_refuses_a_call_it_could_not_read():
    """The executor is never reached, and the model is told why."""
    session = _session_with_stub_executor()

    result = session._execute_single_tool(_unreadable_call(), None)

    session._executor.execute.assert_not_called()
    success, payload = result.executor_result
    assert success is False
    assert "could not be parsed" in payload["error"]
    assert payload["unreadable_arguments"].startswith('{"path"')


def test_the_parallel_path_refuses_it_too():
    """Both execution paths, or the refusal is a coin flip on batch size.

    ``_execute_function_call_group`` routes a single call to the
    sequential path and two or more to the thread pool, so a fix applied
    to only one of them holds exactly until the model batches its calls.
    """
    session = _session_with_stub_executor()

    result = session._execute_single_tool_for_parallel(_unreadable_call())

    session._executor.execute.assert_not_called()
    success, payload = result.executor_result
    assert success is False
    assert "could not be parsed" in payload["error"]


def test_a_readable_call_still_reaches_the_executor():
    """The refusal is narrow.

    Without this the session half could be satisfied by refusing
    everything, which would be a far louder outage than the one being
    fixed.
    """
    session = _session_with_stub_executor()
    session._executor.execute = MagicMock(return_value=(True, {"ok": True}))

    args, unreadable = parse_tool_call_arguments('{"path": "notes.md"}')
    assert unreadable is None
    fc = FunctionCall(id="call_ok", name="write_file", args=args,
                      unreadable_args=unreadable)

    result = session._execute_single_tool(fc, None)

    session._executor.execute.assert_called_once()
    assert result.executor_result == (True, {"ok": True})


# ==================== The cross-provider source guard ====================


def _calls_to_parser(tree: ast.AST) -> list:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_tool_call_arguments"
    ]


@pytest.mark.parametrize("relpath", sorted(PARSING_PROVIDERS))
def test_every_provider_decodes_through_the_shared_parser(relpath):
    """Each pinned provider still routes its argument decode."""
    path = PROVIDER_DIR / relpath
    assert path.is_file(), f"{relpath} moved; this guard now checks nothing"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    expected = PARSING_PROVIDERS[relpath]
    actual = len(_calls_to_parser(tree))
    assert actual == expected, (
        f"{relpath} calls parse_tool_call_arguments {actual} time(s), "
        f"expected {expected}. A removed call means that path decodes "
        f"tool arguments on its own again; an added one means a new "
        f"decode path landed and this pin needs updating."
    )


def _handles_json_decode_error(handler: ast.ExceptHandler) -> bool:
    """Whether *handler* catches ``json.JSONDecodeError``."""
    node = handler.type
    if node is None:
        return True  # bare except
    candidates = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(
        isinstance(c, ast.Attribute) and c.attr in {
            "JSONDecodeError", "ValueError",
        }
        for c in candidates
    ) or any(
        isinstance(c, ast.Name) and c.id == "ValueError" for c in candidates
    )


def test_no_provider_swallows_a_decode_failure_into_arguments():
    """The reverted shape must not reappear under ``model_provider``.

    Matched on the *handler body*: what makes the defect is not catching
    the error, it is producing an argument value from having caught it.
    A handler that logs, re-raises, or leaves the text alone is fine --
    the offending shape is ``args = <anything>`` inside it.
    """
    offenders = []
    for path in sorted(PROVIDER_DIR.rglob("*.py")):
        if "/tests/" in path.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - not our business here
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _handles_json_decode_error(node):
                continue
            for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if not isinstance(stmt, ast.Assign):
                    continue
                names = {
                    t.id for t in stmt.targets if isinstance(t, ast.Name)
                }
                if names & {"args", "arguments", "tool_args"}:
                    offenders.append(
                        f"{path.relative_to(PROVIDER_DIR)}:{stmt.lineno}"
                    )

    assert not offenders, (
        "these providers build a tool call's arguments inside a "
        "JSON-decode error handler, so a call the model never made is "
        "handed to the session as a valid one (#750): "
        + ", ".join(offenders) +
        ". Call parse_tool_call_arguments(raw) instead and pass the "
        "second return value as FunctionCall(unreadable_args=...)."
    )
