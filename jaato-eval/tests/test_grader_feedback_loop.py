"""A failing arm is sent back deterministically, not by its own judgement.

Six arm-attempts across three sweeps committed a fix whose feature was
never reachable — every one of them passed "did it commit" and "does it
compile", and every one failed the acceptance check.  None had run the
command that would have shown them, because nothing made them: the worker
persona said read, edit, commit, and the tool that would have revealed the
problem was available but optional.

Asking the model to verify its own work would put a deterministic
criterion behind a judgement call.  The grader already knows, exactly, so
the loop belongs in the harness: grade between turns and hand the arm its
own FAILED verdicts.
"""

import pytest

from jaato_eval.runner import _retry_feedback


class _V:
    def __init__(self, state, claim="", detail=""):
        self.state = state
        self.claim = claim
        self.detail = detail


def test_all_passing_produces_no_feedback() -> None:
    """A passing arm must not be sent back — that would burn budget and
    invite it to churn a correct answer."""
    assert _retry_feedback([_V("PASS"), _V("PASS")]) is None
    assert _retry_feedback([]) is None


def test_a_failure_produces_actionable_feedback() -> None:
    fb = _retry_feedback([
        _V("PASS", "commit check"),
        _V("FAIL", "cd repo && python3 -m shared.scaffold explain commands",
           "exit 1 (expected 0)"),
    ])
    assert fb
    assert "explain commands" in fb, (
        "the arm must be told the exact command that was run, so it can run "
        "it itself rather than guess at the criterion"
    )
    assert "exit 1" in fb, "the observed result must be quoted, not paraphrased"
    assert "commit check" not in fb, "passing checks are noise here"


def test_feedback_forbids_editing_the_check() -> None:
    """The obvious cheat is to change the test.

    The acceptance check is the task's definition of done; an arm that
    edits it has not fixed anything.  Saying so is cheap and the failure
    mode is otherwise invisible — a passing verdict on a weakened check
    looks exactly like a passing verdict.
    """
    fb = _retry_feedback([_V("FAIL", "some check", "exit 1")])
    low = fb.lower()
    assert "do not change the check" in low
    assert "say so plainly" in low, (
        "an arm that believes the check is wrong needs a sanctioned way to "
        "say so, or its only options are to cheat or to loop"
    )


def test_only_failures_are_reported_back() -> None:
    fb = _retry_feedback([_V("FAIL", "A", "exit 1"), _V("FAIL", "B", "exit 2")])
    assert "A" in fb and "B" in fb
    assert fb.count("FAILED:") == 2


@pytest.mark.parametrize("state", ["PASS", "BLOCKED", "SKIPPED", None])
def test_non_FAIL_states_never_trigger_a_retry(state) -> None:
    """BLOCKED means the arm was never exercised, so there is nothing to
    tell it to fix; retrying on it would spend budget on a run that was cut
    short for reasons the arm cannot address."""
    assert _retry_feedback([_V(state)]) is None


# ----------------------------------------------------------------------
# The WIRING, not just the builder.
#
# A previous guard in this suite tested only its helper, and deleting the
# call site from run_arm left every test green while restoring the bug.
# These drive run_arm itself.
# ----------------------------------------------------------------------

_TASK_YAML = """
id: t/loop
description: retry loop
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: do the thing
harness:
  profile: worker
graders:
  - kind: script
    run: "true"
"""


def _task(tmp_path):
    from jaato_eval.manifest import load_manifest
    (tmp_path / "fixture").mkdir(exist_ok=True)
    (tmp_path / "cfg").mkdir(exist_ok=True)
    (tmp_path / "task.yaml").write_text(_TASK_YAML)
    return load_manifest(tmp_path / "task.yaml")


def _run(tmp_path, monkeypatch, grade_states, max_attempts):
    """Drive run_arm with a stub session and scripted grader outcomes."""
    import asyncio
    from jaato_eval.arm import ArmSpec
    import jaato_eval.runner as R

    prompts = []

    async def _fake_session(spec, workspace, *, socket_path,
                            cascade_driver_id=None, accumulator=None,
                            session_ref=None, retry_hook=None):
        prompts.append("initial")
        while retry_hook is not None:
            fb = await retry_hook()
            if not fb:
                break
            prompts.append(fb)
        return None, accumulator, []

    calls = {"n": 0}

    async def _fake_grade(task, ctx):
        i = min(calls["n"], len(grade_states) - 1)
        calls["n"] += 1
        return [_V(grade_states[i], "the check", "exit 1")]

    monkeypatch.setattr(R, "_run_session", _fake_session)
    monkeypatch.setattr(R, "_grade", _fake_grade)

    spec = ArmSpec(task=_task(tmp_path), profile_set="s", repeat=0)
    result = asyncio.run(R.run_arm(spec, workspace_root=tmp_path / "ws",
                                   max_attempts=max_attempts))
    return result, prompts


def test_a_failing_arm_is_actually_sent_back(tmp_path, monkeypatch) -> None:
    result, prompts = _run(tmp_path, monkeypatch, ["FAIL", "PASS"], max_attempts=3)
    assert len(prompts) == 2, (
        f"the arm was not re-prompted after a FAIL; prompts={prompts!r}. "
        f"The loop exists so a failing arm is corrected deterministically."
    )
    assert "the check" in prompts[1]
    assert result.attempts == 2, "attempts must record the extra completion"


def test_a_passing_arm_is_not_sent_back(tmp_path, monkeypatch) -> None:
    result, prompts = _run(tmp_path, monkeypatch, ["PASS"], max_attempts=3)
    assert len(prompts) == 1
    assert result.attempts == 1


def test_the_attempt_budget_is_respected(tmp_path, monkeypatch) -> None:
    """A permanently failing arm must stop, not loop until the budget dies."""
    result, prompts = _run(tmp_path, monkeypatch, ["FAIL"], max_attempts=3)
    assert len(prompts) == 3, f"expected 3 completions, got {len(prompts)}"
    assert result.attempts == 3


def test_default_is_single_shot(tmp_path, monkeypatch) -> None:
    """Without --max-attempts the behaviour is unchanged, so existing
    sweeps stay comparable with ones run before this feature."""
    result, prompts = _run(tmp_path, monkeypatch, ["FAIL", "FAIL"], max_attempts=1)
    assert len(prompts) == 1
    assert result.attempts == 1
