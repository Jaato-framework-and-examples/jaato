"""A turn severed at the output cap must not present as a tool-use turn.

THE GAP (#745).  Every streaming provider ended its turn with the same
unconditional line::

    if function_calls and not was_cancelled:
        finish_reason = FinishReason.TOOL_USE

The line exists for a real case: several upstreams report ``stop``, or
report nothing at all, on a turn that did emit tool calls, and the
accumulated calls are then the only evidence of what the turn wants.
But it ran unconditionally, so it also overwrote the reason on a turn
the provider had *correctly* read as truncated -- and truncation is
exactly the case where the accumulated calls are least trustworthy,
because the thing the cap severed may be the ``arguments`` object of
the call itself.

The framework has two mechanisms for that, and the overwrite made both
unreachable on streamed turns:

  * ``shared.rewind.detect_truncated_tool_call`` keys on ``MAX_TOKENS``
    **together with** function calls.  With the reason rewritten to
    ``TOOL_USE`` it never fires, so rewind-with-hint recovery -- the
    whole of ``docs/design/rewind-with-hint.md`` -- was dead code for
    every streaming provider.
  * ``JaatoSession._classify_finish_reason`` raises the abnormal-finish
    banner (#544) for ``MAX_TOKENS`` / ``SAFETY`` / ``ERROR``.  It saw
    ``TOOL_USE`` and continued the loop.

What that looks like from outside is an agent executing or failing on a
fragment and going round again.  An OpenRouter activity export for one
eval arm (gpt-5-mini, 2026-08-31) has two turns that ran to a
65,536-token cap over 12-14 minutes each, reported as ``tool_calls``;
they were 44% of the arm's spend and the arm committed nothing.

THE CONTRACT.  ``TOOL_USE`` is a fallback, not an override: it fills in
an unreported or merely-``stop`` finish and never displaces a reason in
``TERMINAL_FINISH_REASONS``.  ``resolve_tool_use_finish`` is the single
implementation; each provider calls it instead of assigning.

WHY A GUARD AND NOT JUST THE PROVIDER TESTS.  A test that streams a
clean ``tool_calls`` finish passes both before and after the defect --
which is how it survived in seven providers at once.  Discriminating
cases have to end the stream on a truncation *while* calls have been
accumulated, and the cross-provider half has to be checked at the
source, because there is no shared streaming entry point to drive.

NOT COVERED, deliberately: ``github_models``'
``_responses_api_response_to_provider`` sets ``TOOL_USE`` per emitted
``function_call`` item on a *batch* Responses-API turn.  It is the same
shape, but the truncation signal there is ``status`` /
``incomplete_details.reason``, which ``copilot_client``'s
``ResponsesAPIResponse`` does not parse at all -- fixing it means
extending that client's wire model, with no captured trace to check it
against.  Left open rather than guessed at.
"""

import ast
import pathlib

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    TERMINAL_FINISH_REASONS,
    resolve_tool_use_finish,
)

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


PROVIDER_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "plugins" / "model_provider"
)

#: Provider modules whose streaming path accumulates function calls and
#: therefore has to *resolve* the finish reason at end-of-stream rather
#: than assign one.  Pinned so that deleting a call site is a failure
#: here and not a silent loss of coverage; a new streaming provider is
#: expected to be added to this list along with its call.
#:
#: ``github_models`` carries three because it speaks three wire dialects
#: (Azure inference, Copilot chat, Copilot Responses) with a separate
#: streaming accumulator for each; ``claude_cli`` carries two, one per
#: CLI transport.  ``claude_cli``'s pair were the worst of the set --
#: they overwrote a ``FinishReason.ERROR`` that the CLI had explicitly
#: reported one block earlier.
RESOLVING_PROVIDERS = {
    "_openai_compat/base.py": 1,
    "anthropic/provider.py": 1,
    "antigravity/provider.py": 1,
    "claude_cli/provider.py": 2,
    "github_models/provider.py": 3,
    "google_genai/provider.py": 1,
    "openrouter/provider.py": 1,
    "vllm/provider.py": 1,
}


#: The defect, put back.  ``_openai_compat/base.py`` stands in for the
#: whole family: it is the streaming path ``nim``, ``nebius``,
#: ``ovhcloud``, ``lmstudio``, ``tensorrt_llm``, ``doubleword``,
#: ``triton`` and ``zhipuai_openai`` all inherit, so one reversion there
#: is eight providers' worth of regression.
REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/plugins/model_provider/types.py",
        find="""    if not has_function_calls:
        return observed
    if observed in TERMINAL_FINISH_REASONS:
        return observed
    return FinishReason.TOOL_USE""",
        replace="""    if not has_function_calls:
        return observed
    return FinishReason.TOOL_USE""",
        test="test_tool_use_never_displaces_a_terminal_reason[max_tokens]",
        because="a turn severed at the output cap being relabelled as a "
                "turn that wants a tool executed, which hides it from "
                "both the rewind detector and the abnormal-finish banner",
    ),
    Reversion(
        target=(
            "jaato-server/shared/plugins/model_provider/"
            "_openai_compat/base.py"
        ),
        find="""        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=bool(function_calls) and not was_cancelled,
        )""",
        replace="""        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE""",
        test=(
            "test_every_streaming_provider_resolves_instead_of_assigning"
            "[_openai_compat/base.py]"
        ),
        because="a provider going back to overwriting the finish reason "
                "at end-of-stream, which is how eight OpenAI-compatible "
                "providers acquired the defect at once",
    ),
]


# ==================== The contract itself ====================


@pytest.mark.parametrize(
    "terminal", sorted(TERMINAL_FINISH_REASONS), ids=lambda r: r.value,
)
def test_tool_use_never_displaces_a_terminal_reason(terminal):
    """The fragments of a severed call are not a request for a tool."""
    assert resolve_tool_use_finish(
        terminal, has_function_calls=True,
    ) == terminal, (
        f"a turn that ended for {terminal.value!r} was relabelled "
        f"{FinishReason.TOOL_USE.value!r} because it happened to carry "
        f"function-call fragments. Downstream cannot then tell a "
        f"complete call from a severed one."
    )


def test_tool_use_still_fills_in_a_non_terminal_finish():
    """The case the override existed to serve is preserved.

    Without this the guard above could be satisfied by never returning
    ``TOOL_USE`` at all, which would break every upstream that reports
    ``stop`` (or nothing) on a turn that did emit calls.
    """
    for observed in (FinishReason.UNKNOWN, FinishReason.STOP):
        assert resolve_tool_use_finish(
            observed, has_function_calls=True,
        ) == FinishReason.TOOL_USE


def test_a_turn_with_no_calls_keeps_its_reason():
    for observed in FinishReason:
        assert resolve_tool_use_finish(
            observed, has_function_calls=False,
        ) == observed


def test_the_rewind_detector_can_actually_be_reached():
    """The end-to-end claim, stated where it can't drift.

    ``shared.rewind`` requires ``MAX_TOKENS`` *and* function calls in
    the same response.  If ``resolve_tool_use_finish`` ever rewrites
    that combination, the detector's precondition becomes unsatisfiable
    and rewind-with-hint silently stops existing.
    """
    from shared.rewind import detect_truncated_tool_call
    from jaato_sdk.plugins.model_provider.types import (
        FunctionCall, Part, ProviderResponse, ToolSchema,
    )

    resolved = resolve_tool_use_finish(
        FinishReason.MAX_TOKENS, has_function_calls=True,
    )
    response = ProviderResponse(
        parts=[Part.from_function_call(
            FunctionCall(id="c1", name="write_file", args={}),
        )],
        usage=None,
        finish_reason=resolved,
        raw=None,
    )
    schema = ToolSchema(
        name="write_file",
        description="write a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    assert detect_truncated_tool_call(response, [schema]) is not None


# ==================== The cross-provider source guard ====================


def _assignments_of_tool_use(tree: ast.AST) -> list:
    """Every ``finish_reason = FinishReason.TOOL_USE`` in *tree*."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute)
                and value.attr == "TOOL_USE"
                and isinstance(value.value, ast.Name)
                and value.value.id == "FinishReason"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "finish_reason":
                found.append(node)
                break
    return found


def _calls_to_resolver(tree: ast.AST) -> list:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_tool_use_finish"
    ]


@pytest.mark.parametrize("relpath", sorted(RESOLVING_PROVIDERS))
def test_every_streaming_provider_resolves_instead_of_assigning(relpath):
    """Each pinned provider still routes its end-of-stream decision."""
    path = PROVIDER_DIR / relpath
    assert path.is_file(), f"{relpath} moved; this guard now checks nothing"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    expected = RESOLVING_PROVIDERS[relpath]
    actual = len(_calls_to_resolver(tree))
    assert actual == expected, (
        f"{relpath} calls resolve_tool_use_finish {actual} time(s), "
        f"expected {expected}. A removed call means that streaming path "
        f"is deciding the finish reason on its own again; an added one "
        f"means a new path landed and this pin needs updating."
    )


def test_no_provider_overwrites_the_finish_reason_at_end_of_stream():
    """The reverted shape must not reappear anywhere under model_provider.

    Matched on the guarding ``if`` rather than the assignment alone,
    because assigning ``TOOL_USE`` while *building* a response from
    parsed parts is legitimate -- what is not is deciding it from the
    presence of accumulated calls after the stream has ended.
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
            if not isinstance(node, ast.If):
                continue
            names = {
                n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
            }
            if "function_calls" not in names:
                continue
            if "finish_reason" in names:
                # Reads the current reason, so it is not unconditional.
                continue
            if _assignments_of_tool_use(ast.Module(body=node.body,
                                                   type_ignores=[])):
                offenders.append(
                    f"{path.relative_to(PROVIDER_DIR)}:{node.lineno}"
                )

    assert not offenders, (
        "these providers decide TOOL_USE from the presence of "
        "accumulated function calls without consulting the finish reason "
        "the wire reported, so a truncated turn is relabelled as a "
        "tool-use turn (#745): " + ", ".join(offenders) +
        ". Call resolve_tool_use_finish(finish_reason, "
        "has_function_calls=...) instead."
    )
