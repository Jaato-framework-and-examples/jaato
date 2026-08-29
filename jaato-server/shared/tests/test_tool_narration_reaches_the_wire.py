"""The operator must be told WHY a tool is being called, while it happens.

Before this, the only related constant was ``_TURN_SUMMARY_INSTRUCTION``,
and its own comment says what it is for: *"Turn-end summary guidance —
needed for GC to work effectively"*.  It exists for the garbage collector.
It arrives at the END of a turn, and a capable model makes dozens or
hundreds of calls inside one turn, so the person watching the tool tree
sees a wall of calls with no stated reason for any of them — and if the
turn dies mid-flight (provider error, hung tool) they never learn why any
of it happened.  Reported from live use 2026-08-29.

These tests pin the constant AND its delivery.  A prompt constant that
quietly stops reaching the wire fails silently: output just gets a little
worse, and nobody can point at a broken test.
"""

from shared import jaato_runtime as R


def test_narration_guidance_exists_and_is_non_empty() -> None:
    assert getattr(R, "_TOOL_NARRATION_GUIDANCE", ""), (
        "_TOOL_NARRATION_GUIDANCE is missing or empty; the operator has no "
        "window into the model's reasoning during a long turn"
    )


def test_narration_is_distinct_from_the_turn_summary() -> None:
    """They serve different consumers and must not be merged.

    The summary serves GC and is retrospective; narration serves the human
    and is prospective.  Folding one into the other loses whichever
    consumer the surviving wording was not written for.
    """
    assert R._TOOL_NARRATION_GUIDANCE != R._TURN_SUMMARY_INSTRUCTION
    lowered = R._TOOL_NARRATION_GUIDANCE.lower()
    assert "before" in lowered, (
        "narration must be prospective — 'before' a call, not after the turn"
    )


def test_narration_asks_per_batch_not_per_call() -> None:
    """Per-call narration would fight the parallel-batching guidance.

    One instruction asking for several calls in a single response, and
    another asking for a sentence before each call, cannot both be
    satisfied.  The narration is deliberately scoped to the batch.
    """
    lowered = R._TOOL_NARRATION_GUIDANCE.lower()
    assert "batch" in lowered, (
        "narration must be scoped per batch; a per-call rule contradicts "
        "_PARALLEL_TOOL_GUIDANCE, which wants independent calls issued together"
    )


def test_both_guidances_reach_the_assembled_instructions() -> None:
    """Delivery, not just declaration.

    A constant can exist, be well worded, and never be delivered — which is
    indistinguishable from good output getting slightly worse.  Checked by
    CALLING the assembler rather than grepping its source, so the guard
    survives the block being refactored (it already was, once).
    """
    live = R._framework_prompt_constants()
    assert R._TOOL_NARRATION_GUIDANCE in live, (
        "_TOOL_NARRATION_GUIDANCE is defined but not among the live prompt "
        "constants, so it never reaches the model"
    )
    assert R._PARALLEL_TOOL_GUIDANCE in live, (
        "_PARALLEL_TOOL_GUIDANCE is defined but not among the live prompt "
        "constants (is parallel execution disabled in this environment?)"
    )


def test_parallel_guidance_is_withheld_when_parallelism_is_off(monkeypatch) -> None:
    """Do not promise batching the runtime will not honour.

    With JAATO_PARALLEL_TOOLS=false the calls are serialised anyway, so the
    guidance would be advice the runtime contradicts.
    """
    monkeypatch.setenv("JAATO_PARALLEL_TOOLS", "false")
    live = R._framework_prompt_constants()
    assert R._PARALLEL_TOOL_GUIDANCE not in live
    assert R._TOOL_NARRATION_GUIDANCE in live, (
        "narration is independent of parallelism and must survive"
    )


def test_parallel_guidance_states_the_dependency_test() -> None:
    """The rewrite must give a decision rule, not just an aspiration.

    The previous wording ("when you need to perform multiple independent
    operations...") described the situation without telling the model how
    to recognise it, and was measurably under-used in live sessions.
    """
    lowered = R._PARALLEL_TOOL_GUIDANCE.lower()
    assert "depends on" in lowered, (
        "parallel guidance should give the model a test to apply — whether "
        "the next call depends on this one's result — not just a description"
    )


def test_narration_is_premium_overridable_like_its_siblings() -> None:
    """Every other prompt constant can be replaced by jaato-premium."""
    import inspect
    src = inspect.getsource(R._apply_premium_prompt_overrides)
    assert '"tool_narration"' in src, (
        "narration cannot be overridden by a premium prompt provider, unlike "
        "task_completion / parallel_tool_guidance / turn_summary"
    )
    assert "_TOOL_NARRATION_GUIDANCE" in src
