"""BLOCKED means "we learned nothing", not "this was free".

An arm cut mid-turn has usually spent real money.  One observed
2026-08-30 ran its full 900s ceiling across **467 billed
`chat/completions` calls** and reported `cost=$0.0000`, because
`result.usage` was assigned only on the success path — the timeout and
session-failure paths set `blocked_reason` and returned without it.

Worse, the accumulator itself was created *inside* the coroutine and
returned by it, so `asyncio.wait_for` cancelling on timeout meant the
caller never received it.  Even usage from turns that HAD completed was
unreachable.

This is not merely a wrong number.  The task pool's `usd` ceiling is
evaluated against reported spend (`manifest.py`: arms "are clamped at
spawn, degraded mid-flight at a rung, and refused once it is empty"), so
an arm that never completes a turn could burn without bound while the
pool read zero — the control absent exactly in the runaway case it exists
for.
"""

import pytest

from jaato_eval.runner import _TurnAccumulator, _record_partial_usage
from jaato_eval.results import ArmResult


class _Usage:
    """Shape of the usage object carried on a TurnCompletedEvent."""

    def __init__(self, prompt=0, output=0, cost=0.0):
        self.prompt_tokens = prompt
        self.output_tokens = output
        self.cost_usd = cost


class _Event:
    def __init__(self, usage=None, finish_reason=None):
        self.usage = usage
        self.finish_reason = finish_reason
        self.completion_gap = None


_TASK_YAML = """
id: t/blocked-spend
description: an arm that is cut mid-flight
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: work
harness:
  profile: worker
graders:
  - kind: script
    run: "true"
"""


def _arm_result() -> ArmResult:
    """A bare result, as `run_arm` builds before the session starts."""
    return ArmResult(spec=None, verdicts=[], usage={})


def test_a_blocked_arm_reports_what_it_spent() -> None:
    """The regression, at the level that actually cost money."""
    acc = _TurnAccumulator()
    acc.on_turn(_Event(_Usage(prompt=1000, output=200, cost=0.0184)))
    acc.on_turn(_Event(_Usage(prompt=900, output=150, cost=0.0121)))

    result = _arm_result()
    result.blocked_reason = "arm exceeded the harness ceiling of 900s"
    _record_partial_usage(result, acc)

    assert result.usage["cost_usd"] == pytest.approx(0.0305), (
        "a cut-short arm reported less than it spent; the pool's usd "
        "ceiling is evaluated against this number, so under-reporting it "
        "makes the ceiling unenforceable"
    )
    assert result.usage["prompt_tokens"] == 1900
    assert result.turns == 2, "turns completed before the cut must survive it"


def test_an_arm_cut_before_any_turn_reports_zero_honestly() -> None:
    """Zero is the right answer when the accumulator genuinely saw nothing.

    Usage rides on turn-completion events, so an arm cut inside its FIRST
    turn still reports nothing.  That residual gap is real and needs
    per-response usage to close; what must not happen is discarding usage
    the accumulator DID see.
    """
    result = _arm_result()
    _record_partial_usage(result, _TurnAccumulator())
    assert result.usage["cost_usd"] in (0, 0.0, None)
    assert result.turns == 0


def test_usage_is_never_left_unset_on_a_blocked_result() -> None:
    """A missing key reads as 'unknown'; the report renders it as zero.

    Downstream consumers (report pivot, comparative judge) index into this
    dict, so it must exist even when empty.
    """
    result = _arm_result()
    _record_partial_usage(result, _TurnAccumulator())
    assert isinstance(result.usage, dict)
    for key in ("prompt_tokens", "output_tokens", "cost_usd"):
        assert key in result.usage, f"{key} missing from a blocked arm's usage"


def test_the_caller_owns_the_accumulator() -> None:
    """`_run_session` must accept one, or a timeout loses it again.

    The accumulator used to be created inside the coroutine and returned;
    `asyncio.wait_for` cancels on timeout, so the caller never saw it.
    Accepting one is what makes the fix possible at all.
    """
    import inspect
    from jaato_eval.runner import _run_session
    params = inspect.signature(_run_session).parameters
    assert "accumulator" in params, (
        "_run_session no longer accepts a caller-owned accumulator; a "
        "timed-out arm's usage becomes unreachable again"
    )
    assert params["accumulator"].default is None, (
        "the parameter must stay optional so existing callers are unaffected"
    )


def test_a_TIMED_OUT_arm_keeps_its_spend_end_to_end(tmp_path, monkeypatch) -> None:
    """The wiring, not just the helper.

    An earlier version of this file only called ``_record_partial_usage``
    directly.  Removing the call from ``run_arm``'s timeout path left every
    test green while restoring the exact bug — a guard that cannot fail is
    not a guard.  This drives the real ``run_arm`` timeout path instead.

    The stub records a completed turn (so there IS spend to lose) and then
    hangs, forcing ``asyncio.wait_for`` to cancel it exactly as a runaway
    arm does.
    """
    import asyncio
    from pathlib import Path
    import jaato_eval.runner as R

    async def _hang(spec, workspace, *, socket_path, cascade_driver_id=None,
                    accumulator=None):
        assert accumulator is not None, (
            "run_arm must supply an accumulator it owns; one created inside "
            "this coroutine is destroyed with it on cancellation"
        )
        accumulator.on_turn(_Event(_Usage(prompt=1234, output=56, cost=0.0184)))
        await asyncio.sleep(3600)

    monkeypatch.setattr(R, "_run_session", _hang)

    from jaato_eval.arm import ArmSpec
    from jaato_eval.manifest import load_manifest

    (tmp_path / "fixture").mkdir()
    (tmp_path / "cfg").mkdir()
    (tmp_path / "task.yaml").write_text(_TASK_YAML)
    task = load_manifest(tmp_path / "task.yaml")
    spec = ArmSpec(task=task, profile_set="cheap", repeat=0)

    result = asyncio.run(R.run_arm(spec, workspace_root=tmp_path / "ws",
                                   arm_timeout_seconds=0.25))

    assert result.blocked_reason, "the arm should have been cut short"
    assert result.usage.get("cost_usd") == pytest.approx(0.0184), (
        f"a timed-out arm lost its spend: usage={result.usage!r}. The pool's "
        f"usd ceiling is evaluated against this, so zero here means a "
        f"runaway arm can never trip it."
    )
    assert result.turns == 1
