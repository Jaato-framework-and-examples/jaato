"""A stream that stops arriving must not be accepted as a completed turn.

THE GAP (#687).  Every streaming provider accumulates into a
``finish_reason`` that starts at ``FinishReason.UNKNOWN`` and is
overwritten only when the wire delivers a terminal event -- Anthropic's
``message_stop`` / ``message_delta.stop_reason``, an OpenAI-compatible
chunk carrying ``choice.finish_reason``, a Google candidate carrying
``finish_reason``, the Claude CLI's ``ResultMessage``.  When the stream
simply stops -- a proxy drops the connection, a gateway times out, an
upstream 5xx lands mid-body, TLS resets -- none of those arrive, the
iterator ends quietly, and the accumulator still holds ``UNKNOWN`` plus
whatever text got through.

``UNKNOWN`` was then grouped with the two SUCCESS outcomes at every
consumer::

    if response.finish_reason not in (STOP, UNKNOWN, TOOL_USE):   # abnormal

so the half-finished turn was accepted as a finished one.  Three
consequences, in increasing severity:

1.  **Silent truncation.**  The user gets half an answer with no
    indication it was cut.
2.  **Orphaned tool calls.**  A call the stream died in the middle of
    was handed downstream as a request -- #674's shape, arriving from
    the wire rather than from GC, and the same failure #750 closed for
    arguments that would not parse.
3.  **No retry.**  A dropped connection is the textbook retryable
    failure, and it was the one failure that never reached the retry
    path, because no exception was ever raised for a classifier to see.

There was already partial handling for the EMPTY case
(``_nudge_for_tool_use`` special-cases ``UNKNOWN and is_empty``), which
is the tell: an empty truncation was caught, a half-full one was not.

The upstream fix (pi #3936) classified the same condition as an error in
two providers independently.

THE CONTRACT, in three parts:

*   ``UNKNOWN`` and ``INCOMPLETE`` are opposites, not synonyms.
    ``UNKNOWN`` means "the turn ended, with a label we do not map" and
    stays a success.  ``INCOMPLETE`` means "the turn never ended" and is
    terminal.  Providers therefore track whether a terminal event was
    SEEN, separately from what it said -- an unmapped label must not
    read as an interruption, and an interruption must not read as an
    unmapped label.
*   ``require_terminated_stream`` is the single implementation.  Every
    streaming accumulator ends with it; it drops the accumulated
    function calls, marks the response ``INCOMPLETE``, and raises
    ``StreamInterruptedError``.
*   ``StreamInterruptedError`` is retryable through the SHARED
    classifier, because every provider's own ``classify_error`` returns
    ``None`` for types it does not know.  One entry covers all of them
    and cannot drift apart between them.

WHY A GUARD AND NOT JUST THE PROVIDER TESTS.  A test that streams a
clean finish passes both before and after the defect -- which is how
this survived in eight providers at once, exactly as #745 did.  The
discriminating case is a stream that ends WITHOUT its terminal event,
and the cross-provider half has to be checked at the source, because
there is no shared streaming entry point to drive.
"""

import ast
import pathlib

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Part,
    ProviderResponse,
    StreamInterruptedError,
    TERMINAL_FINISH_REASONS,
    TurnOutcome,
    TurnResult,
    require_terminated_stream,
    resolve_tool_use_finish,
)

from shared.retry_utils import classify_error
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


PROVIDER_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "plugins" / "model_provider"
)

SESSION_PY = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"


#: Provider modules whose streaming path accumulates a response and
#: therefore has to prove the stream TERMINATED before returning it.
#: Pinned so that deleting a call site is a failure here and not a
#: silent loss of coverage; a new streaming provider is expected to be
#: added to this list along with its call.
#:
#: The counts match ``RESOLVING_PROVIDERS`` in
#: ``test_truncation_is_not_reported_as_tool_use`` -- deliberately, since
#: both guards attach to the same end-of-stream decision point.
#: ``github_models`` carries three because it speaks three wire dialects
#: (Azure inference, Copilot chat, Copilot Responses); ``claude_cli``
#: carries two, one per CLI transport.
#:
#: ``_openai_compat/base.py`` stands in for eight providers: ``nim``,
#: ``nebius``, ``ovhcloud``, ``lmstudio``, ``tensorrt_llm``,
#: ``doubleword``, ``triton`` and ``zhipuai_openai`` all inherit that
#: one streaming loop.
TERMINATING_PROVIDERS = {
    "_openai_compat/base.py": 1,
    "anthropic/provider.py": 1,
    "antigravity/provider.py": 1,
    "claude_cli/provider.py": 2,
    "github_models/provider.py": 3,
    "google_genai/provider.py": 1,
    "openrouter/provider.py": 1,
    "vllm/provider.py": 1,
}


#: How many places each provider SETS ``terminal_seen`` -- i.e. how many
#: distinct wire signals it accepts as "the upstream ended the turn".
#:
#: Pinned exactly, not as a lower bound, because the interesting
#: reversion is a provider that keeps the guard call and loses ONE of
#: its reads: the healthy path that happened to be covered by the other
#: read still passes, and the uncovered one starts raising on turns that
#: finished perfectly well.  A lower bound cannot see that.
#:
#: Where the counts exceed the accumulator count, the extra sites are:
#:
#: * ``_openai_compat`` / ``vllm``: a choice carries ``finish_reason``
#:   with a delta and without one -- two branches, one meaning.
#: * ``anthropic``: ``message_stop``, ``message_delta.stop_reason``
#:   (several Anthropic-compatible endpoints send the reason and close
#:   without a separate ``message_stop``), and the malformed-SSE
#:   recovery, which substitutes its own finish and its own notice to
#:   the model and so has already accounted for the turn.
#: * ``antigravity``: a chunk's finish reason, and the ``done`` sentinel.
#: * ``claude_cli``: a ``ResultMessage`` per transport, plus the
#:   streaming transport's exception branch, which names a terminal
#:   ``ERROR`` on its own.
TERMINAL_EVENT_SITES = {
    "_openai_compat/base.py": 2,
    "anthropic/provider.py": 3,
    "antigravity/provider.py": 2,
    "claude_cli/provider.py": 3,
    "github_models/provider.py": 3,
    "google_genai/provider.py": 1,
    "openrouter/provider.py": 1,
    "vllm/provider.py": 2,
}


#: The defect, put back.
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    if was_cancelled or terminal_seen:
        return response

    dropped = [p for p in response.parts if p.function_call is not None]""",
        replace="""    if True:
        return response

    dropped = [p for p in response.parts if p.function_call is not None]""",
        test="test_a_stream_with_no_terminal_event_raises",
        because="a stream that stopped arriving being handed back as a "
                "finished turn, which is silent truncation with no retry",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    FinishReason.CANCELLED,
    FinishReason.INCOMPLETE,
})""",
        replace="""    FinishReason.CANCELLED,
})""",
        test="test_an_incomplete_turn_is_terminal",
        because="an interrupted stream's accumulated fragments being "
                "relabelled TOOL_USE, so a call the stream never finished "
                "presents downstream as a request",
    ),
    Reversion(
        target="jaato-server/shared/retry_utils.py",
        find="""    if isinstance(exc, STREAM_INTERRUPTED_CLASSES):
        return {"transient": True, "rate_limit": False, "infra": True}
""",
        replace="",
        test="test_an_interrupted_stream_is_retryable",
        because="the one failure that most deserves a retry -- a dropped "
                "connection -- not being retried",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "_openai_compat/base.py"
        ),
        find="""                    # Extract finish reason
                    if choice.finish_reason:
                        terminal_seen = True
                        finish_reason = map_finish_reason(choice.finish_reason)""",
        replace="""                    # Extract finish reason
                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)""",
        test=(
            "test_every_streaming_provider_records_the_terminal_event"
            "[_openai_compat/base.py]"
        ),
        because="a provider forgetting to record that the wire ended the "
                "turn, so every healthy stream would be reported as "
                "interrupted -- the false-positive half of the same seam",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "google_genai/converters.py"
        ),
        find='    return _GOOGLE_FINISH_REASONS.get(name, FinishReason.UNKNOWN)',
        replace="    if 'STOP' in name:\n        return FinishReason.STOP\n    elif 'MAX' in name or 'LENGTH' in name:\n        return FinishReason.MAX_TOKENS\n    elif 'SAFETY' in name:\n        return FinishReason.SAFETY\n    elif 'TOOL' in name or 'FUNCTION' in name:\n        return FinishReason.TOOL_USE\n\n    return FinishReason.UNKNOWN",
        test="test_a_google_error_stop_is_not_read_as_tool_use"
             "[MALFORMED_FUNCTION_CALL]",
        because="Google's MALFORMED_FUNCTION_CALL -- the model emitted a "
                "call its own serialiser rejected -- being read as a turn "
                "that wants a tool run, and its filtered stops "
                "(RECITATION, BLOCKLIST, SPII) as clean ones",
    ),
]


# ==================== UNKNOWN and INCOMPLETE are opposites ====================


def test_incomplete_exists_and_is_distinct_from_unknown():
    """The sentinel and the failure need separate names.

    Overloading ``UNKNOWN`` to mean both is the defect: it is the value
    every accumulator STARTS at, so any reading of it as a failure would
    also condemn a turn whose finish label simply wasn't in the mapping.
    """
    assert FinishReason.INCOMPLETE != FinishReason.UNKNOWN


def test_an_incomplete_turn_is_terminal():
    """So its accumulated fragments are never re-read as a request."""
    assert FinishReason.INCOMPLETE in TERMINAL_FINISH_REASONS
    assert resolve_tool_use_finish(
        FinishReason.INCOMPLETE, has_function_calls=True,
    ) == FinishReason.INCOMPLETE, (
        "an interrupted stream's half-built calls were relabelled "
        "TOOL_USE, which is how a severed turn becomes an executed one"
    )


def test_unknown_is_still_a_success():
    """The other half of the distinction, stated so it cannot drift.

    Without this the guard above could be satisfied by condemning
    ``UNKNOWN`` too, which would fail every turn whose upstream reports a
    finish label the mapping does not carry.
    """
    result = TurnResult.from_provider_response(
        ProviderResponse(finish_reason=FinishReason.UNKNOWN),
    )
    assert result.outcome == TurnOutcome.RESPONSE


def test_an_incomplete_turn_is_an_error_outcome():
    result = TurnResult.from_provider_response(
        ProviderResponse(finish_reason=FinishReason.INCOMPLETE),
    )
    assert result.outcome == TurnOutcome.ERROR
    assert result.is_error


def test_an_unmapped_finish_reason_is_not_a_success_either():
    """``from_provider_response``'s default used to be ``RESPONSE``.

    That is the same shape as the ``UNKNOWN``-is-success defect, waiting
    for the next enum member: a reason the table has not heard of read
    as a clean stop.
    """
    assert TurnResult.from_provider_response(
        ProviderResponse(finish_reason=FinishReason.INCOMPLETE),
    ).outcome is TurnOutcome.ERROR

    mapped = {
        FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE,
        FinishReason.CANCELLED, FinishReason.MAX_TOKENS,
        FinishReason.SAFETY, FinishReason.ERROR, FinishReason.INCOMPLETE,
    }
    assert set(FinishReason) == mapped, (
        "a FinishReason member landed without an entry in "
        "TurnResult.from_provider_response's outcome_map. It now "
        "defaults to ERROR, which is the safe direction -- but decide "
        "its outcome deliberately and add it here."
    )


# ==================== The contract itself ====================


def _response_with_a_call():
    return ProviderResponse(
        parts=[
            Part.from_text("Half an ans"),
            Part.from_function_call(
                FunctionCall(id="c1", name="write_file", args={}),
            ),
        ],
    )


def test_a_stream_with_no_terminal_event_raises():
    """The claim, at its narrowest."""
    with pytest.raises(StreamInterruptedError):
        require_terminated_stream(
            _response_with_a_call(),
            terminal_seen=False,
            was_cancelled=False,
            provider="anthropic",
        )


def test_a_terminated_stream_is_returned_untouched():
    """The case the guard must stay invisible for.

    Without this, the contract above could be satisfied by raising
    always, which would fail every healthy turn.
    """
    response = _response_with_a_call()
    returned = require_terminated_stream(
        response,
        terminal_seen=True,
        was_cancelled=False,
        provider="anthropic",
    )
    assert returned is response
    assert len(returned.parts) == 2
    assert returned.finish_reason == FinishReason.UNKNOWN


def test_a_cancelled_turn_is_not_an_interrupted_one():
    """A cancel has no terminal event BY CONSTRUCTION.

    The caller asked for the turn to stop; reporting a retryable
    infrastructure failure would turn a clean stop into an error and
    then retry the very work that was cancelled.
    """
    response = _response_with_a_call()
    assert require_terminated_stream(
        response,
        terminal_seen=False,
        was_cancelled=True,
        provider="anthropic",
    ) is response


def test_the_partial_is_marked_incomplete_and_stripped_of_calls():
    """What the error carries, and what it refuses to pass on."""
    with pytest.raises(StreamInterruptedError) as exc:
        require_terminated_stream(
            _response_with_a_call(),
            terminal_seen=False,
            was_cancelled=False,
            provider="openrouter",
            model="openai/gpt-5-mini",
            chunks=7,
        )

    partial = exc.value.partial
    assert partial.finish_reason == FinishReason.INCOMPLETE, (
        "a value that escapes by some other route must not read as a "
        "success"
    )
    assert not [p for p in partial.parts if p.function_call], (
        "a call accumulated by a stream that then died may be missing "
        "its arguments, its name, or its closing brace; passing it on "
        "is how a severed turn becomes an executed one (#687, #750)"
    )
    assert partial.get_text() == "Half an ans", (
        "the text is kept: it is the record of what the turn was when "
        "it died, and it is what the caller already streamed out"
    )
    assert exc.value.dropped_calls == 1
    assert exc.value.chunks == 7
    assert exc.value.provider == "openrouter"
    assert "openai/gpt-5-mini" in str(exc.value)


def test_the_message_names_the_condition_not_a_guess():
    """An operator reading a log must not have to infer the diagnosis."""
    with pytest.raises(StreamInterruptedError) as exc:
        require_terminated_stream(
            ProviderResponse(),
            terminal_seen=False,
            was_cancelled=False,
            provider="anthropic",
        )
    message = str(exc.value)
    assert "without a terminal event" in message
    assert "before any content arrived" in message
    assert "retried" in message


# ==================== Reaching the retry path ====================


def test_an_interrupted_stream_is_retryable():
    """#687's third consequence, closed at the shared classifier.

    It lives in ``retry_utils`` rather than in each provider's
    ``classify_error`` because every provider's own classifier returns
    ``None`` for a type it does not recognise -- so one entry covers all
    of them and cannot drift apart between them.
    """
    verdict = classify_error(StreamInterruptedError("anthropic"))
    assert verdict["transient"] is True, (
        "a dropped connection is the textbook retryable failure and it "
        "was the one failure that never got retried"
    )
    assert verdict["infra"] is True
    assert verdict["rate_limit"] is False


def test_no_provider_classifier_can_intercept_it():
    """Every provider's ``classify_error`` must fall through to the shared one.

    A provider that answered a dict for an unrecognised exception would
    silently take this error out of the retry path for itself alone --
    the drift the shared entry exists to prevent.
    """
    offenders = []
    for path in sorted(PROVIDER_DIR.rglob("provider.py")):
        if "/tests/" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name == "classify_error"):
                continue
            # The last statement must be ``return None`` (possibly bare),
            # i.e. "I do not know this one, ask the shared classifier".
            last = node.body[-1]
            falls_through = (
                isinstance(last, ast.Return)
                and (last.value is None
                     or (isinstance(last.value, ast.Constant)
                         and last.value.value is None))
            )
            if not falls_through:
                offenders.append(f"{path.relative_to(PROVIDER_DIR)}:{node.lineno}")

    assert not offenders, (
        "these providers' classify_error does not end by falling through "
        "to the shared classifier, so StreamInterruptedError may never "
        "reach retry_utils.classify_error for them: "
        + ", ".join(offenders)
    )


# ==================== The session must not accept INCOMPLETE ====================


def _session_success_tuples():
    """Every ``FinishReason`` membership test in ``jaato_session.py``.

    Returns the attribute-name sets of each ``in (...)`` / ``not in
    (...)`` comparison whose elements are ``FinishReason`` members --
    the places a finish reason is waved through as "keep going".
    """
    tree = ast.parse(SESSION_PY.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            continue
        for comparator in node.comparators:
            if not isinstance(comparator, (ast.Tuple, ast.Set, ast.List)):
                continue
            names = set()
            for elt in comparator.elts:
                if (isinstance(elt, ast.Attribute)
                        and isinstance(elt.value, ast.Name)
                        and elt.value.id == "FinishReason"):
                    names.add(elt.attr)
            if names:
                found.append((node.lineno, names))
    return found


def test_the_session_never_waves_incomplete_through():
    """The backstop, checked at the source.

    Providers RAISE rather than return an ``INCOMPLETE`` response, so
    this branch should be unreachable -- which is exactly why it needs a
    guard: an unreachable branch is where a wrong default survives.  Any
    continue-set that lists ``STOP`` is a "this turn is fine" set, and
    ``INCOMPLETE`` must never join one.
    """
    offenders = [
        f"jaato_session.py:{lineno}"
        for lineno, names in _session_success_tuples()
        if "STOP" in names and "INCOMPLETE" in names
    ]
    assert not offenders, (
        "these finish-reason continue-sets include INCOMPLETE alongside "
        "STOP, so a stream that never ended is processed as one that "
        "did (#687): " + ", ".join(offenders)
    )


def test_the_session_guard_is_actually_looking_at_something():
    """Anchor: the sets above must exist, or the guard checks nothing."""
    sets_with_stop = [
        (lineno, names) for lineno, names in _session_success_tuples()
        if "STOP" in names
    ]
    assert sets_with_stop, (
        "no finish-reason continue-set found in jaato_session.py. Either "
        "they moved or the parse is wrong; both make the guard above "
        "vacuous."
    )
    # And each one must still carry UNKNOWN -- the value that IS a
    # success. A set that lost it would fail healthy turns instead.
    assert all("UNKNOWN" in names for _l, names in sets_with_stop)


# ============ The opposite error: a real stop read as an ordinary one =====
#
# #687 is one half of a symmetry.  The other half is a turn that DID end
# -- and ended badly -- being read as an ordinary one.  Google carried
# both variants inside a single substring mapping, so they are guarded
# together here; the exhaustive per-name table lives in
# ``google_genai/tests/test_finish_reason_mapping.py``.


def _google_finish(name):
    from shared.plugins.model_provider.google_genai.converters import (
        finish_reason_from_sdk,
    )
    return finish_reason_from_sdk(name)


@pytest.mark.parametrize("name", [
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
])
def test_a_google_error_stop_is_not_read_as_tool_use(name):
    """Gemini has no tool-use finish reason; it reports ``STOP``.

    So ``'TOOL' in name or 'FUNCTION' in name -> TOOL_USE`` never
    matched a tool-use turn.  It matched only the errors whose names
    mention tools, and turned each into a request the session would then
    execute or nudge on.
    """
    assert _google_finish(name) is FinishReason.ERROR, (
        f"Google's {name} is a generation FAILURE, not a request for a "
        f"tool run"
    )


@pytest.mark.parametrize("name", [
    "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
])
def test_a_google_filtered_stop_is_not_read_as_success(name):
    """These matched nothing and fell through to ``UNKNOWN``."""
    assert _google_finish(name) is FinishReason.SAFETY


def test_google_still_maps_the_ordinary_reasons():
    """Without this, the two above could be satisfied by mapping nothing."""
    assert _google_finish("STOP") is FinishReason.STOP
    assert _google_finish("MAX_TOKENS") is FinishReason.MAX_TOKENS
    assert _google_finish("SOME_FUTURE_REASON") is FinishReason.UNKNOWN


# ==================== The cross-provider source guard ====================


def _calls_to(tree: ast.AST, func_name: str) -> list:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    ]


@pytest.mark.parametrize("relpath", sorted(TERMINATING_PROVIDERS))
def test_every_streaming_provider_requires_a_terminated_stream(relpath):
    """Each pinned provider still proves its stream ended."""
    path = PROVIDER_DIR / relpath
    assert path.is_file(), f"{relpath} moved; this guard now checks nothing"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    expected = TERMINATING_PROVIDERS[relpath]
    actual = len(_calls_to(tree, "require_terminated_stream"))
    assert actual == expected, (
        f"{relpath} calls require_terminated_stream {actual} time(s), "
        f"expected {expected}. A removed call means that streaming path "
        f"accepts a dead stream as a finished turn again; an added one "
        f"means a new path landed and this pin needs updating."
    )


@pytest.mark.parametrize("relpath", sorted(TERMINATING_PROVIDERS))
def test_every_streaming_provider_records_the_terminal_event(relpath):
    """``terminal_seen`` must be both initialised AND set from the wire.

    Half of this is the false-negative (never set to ``True`` from
    anywhere the wire is read, so nothing raises); the other half is the
    false-positive (never initialised ``False``, so a ``NameError`` or a
    stale truthy value hides the check).  Both leave the call above in
    place and doing nothing, which is the shape a source-count guard
    alone cannot see.
    """
    path = PROVIDER_DIR / relpath
    tree = ast.parse(path.read_text(encoding="utf-8"))

    inits, sets = 0, 0
    for node in ast.walk(tree):
        # ``terminal_seen = True`` and ``terminal_seen |= <cond>`` both
        # record the event; the augmented form is used where the
        # complexity ratchet holds a function at its frozen size.
        if isinstance(node, ast.AugAssign):
            target, value = node.target, None
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == "terminal_seen"):
            continue
        if isinstance(value, ast.Constant) and value.value is False:
            inits += 1
        else:
            sets += 1

    assert inits >= 1, (
        f"{relpath} never initialises terminal_seen = False, so the "
        f"guard call reads an undefined or stale name."
    )
    expected_sites = TERMINAL_EVENT_SITES[relpath]
    assert sets == expected_sites, (
        f"{relpath} sets terminal_seen = True at {sets} site(s), "
        f"expected {expected_sites}. A LOST site means the wire signal "
        f"it read no longer counts as the turn ending, so healthy turns "
        f"that end that way now raise; an ADDED one means a new signal "
        f"landed and this pin needs updating. Both are decisions, not "
        f"details."
    )
