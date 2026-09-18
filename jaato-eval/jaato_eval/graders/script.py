"""Script grader — run a command against the mutated workspace.

The classic programmatic rubric: ``mvn clean compile``, ``pytest -q``,
``cargo build``.  Exit 0 is PASS.

The distinction this adapter is careful about is *whose* fault a non-zero
exit is.  A compiler that rejects the agent's code is a FAIL — the agent
was exercised and produced something wrong.  A command that does not
exist on the runner is a BLOCKED — nothing about the agent was
established.  Conflating them is how a benchmark ends up reporting that
every model fails a task whose toolchain was never installed.

The second thing it is careful about is *what* is being graded.  A shell
command cannot read ``GraderContext``, so before this adapter exported
them a script grader was blind to the task's own ``agent_params`` — the
inputs that decide what "correct" means for this arm.  Any input-dependent
check therefore had to hardcode the input, and a hardcoded value cannot
notice when the input changes: re-point a task at a different issue id and
every arm is graded against the previous one's criteria, reported as FAIL
with no error anywhere (jaato #762).  Exporting the parameters lets a
grader follow the input by construction instead of by remembering.

The third is *which* arm.  The parameters were exported here first and
lifted into :mod:`jaato_eval.params` when a driver arm became the second
consumer, on the argument that a grader and the thing it grades must read
the SAME variable for the same input.  The rest of the contract did not
follow, and the row that cost something was ``JAATO_EVAL_PYTHON``: a
driver task is told to write ``run: '"$JAATO_EVAL_PYTHON" -m pkg.driver'``
rather than bet on the host's ``PATH``, and a scorer written the same way
got an empty variable, asked the shell to execute ``""``, and was BLOCKED
as a missing toolchain on every arm — while the arms themselves exited 0
(jaato #1127).  The whole table now comes from one builder
(:func:`jaato_eval.contract.arm_environment`), so the interpreter, the
workspace, the config root, the socket and the cascade id say the same
thing on both sides of the run.
"""
from __future__ import annotations

import os
import subprocess
from typing import List

from ..contract import arm_environment
from ..manifest import GraderSpec
from ..sign_off import describe_unsigned
from ..verdict import FAIL, PASS, Verdict
from .base import GraderContext, blocked

#: Exit code a POSIX shell returns when the command itself was not found.
#: Distinct from any exit code the command could have chosen, so it is a
#: reliable "the harness is missing something" signal.
_COMMAND_NOT_FOUND = 127

#: Cap on captured output kept as evidence, in lines from the tail.  A
#: build log can be tens of thousands of lines; the failure is at the end.
_EVIDENCE_LINES = 20


class ScriptGrader:
    """Run ``config['run']`` in the workspace; exit 0 is PASS.

    Config keys:
        run: The command line, executed through the shell so ordinary
            pipelines and ``&&`` work.  Required.
        timeout_seconds: Wall-clock cap (default 600).  A timeout is
            BLOCKED, not FAIL — a command that never finished did not
            establish anything, and treating it as failure would make a
            slow runner look like a bad model.
        expect_exit: Exit code counted as PASS (default 0).  For tasks
            whose success condition is a command *failing*.

    Environment the command inherits, on top of ``os.environ``: the arm
    contract, from :func:`jaato_eval.contract.arm_environment` — the same
    builder a driver arm is handed, so the two cannot disagree about a
    name or a value.  ``JAATO_EVAL=1`` says this is a graded run;
    ``JAATO_EVAL_PARAM_<KEY>`` and ``JAATO_EVAL_PARAMS`` carry the arm's
    inputs; ``JAATO_EVAL_PYTHON``, ``JAATO_EVAL_WORKSPACE``,
    ``JAATO_EVAL_CONFIG_ROOT``, ``JAATO_EVAL_SOCKET`` and
    ``JAATO_EVAL_CASCADE_ID`` are the rest of the table, the last three
    ABSENT rather than empty when the arm has none.

    So a grader that depends on an input says so in the manifest::

        - kind: script
          run: bash acceptance.sh compliant "$JAATO_EVAL_PARAM_ISSUE_ID"

    rather than baking the value into ``acceptance.sh``, where nothing
    can notice when the task's input moves on without it — and a grader
    that has to import the package it grades names the interpreter that
    has it::

        - kind: script
          run: '"$JAATO_EVAL_PYTHON" -m ta_cascade.score'

    rather than ``python``, which is a bet on the runner's ``PATH``.
    """

    def __init__(self, spec: GraderSpec) -> None:
        self.spec = spec

    def grade(self, context: GraderContext) -> Verdict:
        command = self.spec.config.get("run")
        claim = f"`{command}` succeeds in the workspace"

        if not command:
            return blocked(self.spec, "script grader runs",
                           "manifest grader has no 'run' key")

        # AN UNSIGNED ARM IS NOT A TRUNCATED ONE.  ``truncation_reason``
        # answers "did the session end where it meant to", and for an agent
        # that spent the completion-nudge budget the answer is honestly no —
        # but the workspace it left is a tree it worked on to a stop of its
        # own, not one interrupted mid-edit.  Blocking here recorded such an
        # arm as unmeasured with a passing (or failing) tree on disk, which
        # is what this gate is for on every OTHER terminal and exactly wrong
        # on this one (jaato #773).  What the sign-off's absence invalidates
        # is the graders that read the sign-off; this is not one of them.
        truncated = context.truncation_reason
        if truncated and not context.missing_sign_off:
            return blocked(self.spec, claim,
                           f"arm {truncated}; the workspace reflects a "
                           "truncated run")

        if not context.workspace_path.is_dir():
            return blocked(self.spec, claim,
                           f"workspace does not exist: {context.workspace_path}")

        timeout = float(self.spec.config.get("timeout_seconds", 600))
        expect_exit = int(self.spec.config.get("expect_exit", 0))

        contract_env, collision = arm_environment(
            workspace=context.workspace_path,
            config_root=context.config_root,
            params=context.agent_params,
            cascade_id=context.cascade_id,
            socket_path=context.socket_path,
        )
        if collision:
            return blocked(self.spec, claim, collision)

        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(context.workspace_path),
                capture_output=True, text=True, timeout=timeout,
                # The contract overlays the inherited environment rather
                # than the other way round: a sweep started from inside
                # another graded run must not have its parent's arm
                # described to this one's grader.
                env={**os.environ, **contract_env},
            )
        except subprocess.TimeoutExpired:
            return blocked(self.spec, claim,
                           f"command exceeded {timeout:g}s and was killed; "
                           "no signal about the agent")
        except OSError as exc:
            return blocked(self.spec, claim, f"could not execute command: {exc}")

        if proc.returncode == _COMMAND_NOT_FOUND and expect_exit != _COMMAND_NOT_FOUND:
            return blocked(self.spec, claim,
                           f"command not found on this runner (exit 127): {command!r}")

        state = PASS if proc.returncode == expect_exit else FAIL
        verdict = Verdict(
            grader_id=f"script:{self.spec.identifier}",
            claim=claim,
            state=state,
            detail=f"exit {proc.returncode} (expected {expect_exit})",
        )
        if context.missing_sign_off:
            # Carried as evidence, not as a caveat on the state: the
            # command ran against the real tree and its exit code means
            # what it always means.  But a reader comparing this arm with
            # its siblings should know the agent never declared itself
            # done, because that is a real difference in how it behaved.
            verdict.note(
                f"graded without a completion payload — "
                f"{describe_unsigned(context.termination_error_type)}")
        for line in _tail(proc.stdout, proc.stderr):
            verdict.note(line)
        return verdict


def _tail(stdout: str, stderr: str) -> List[str]:
    """Last lines of combined output, for verdict evidence."""
    combined = (stdout or "") + (stderr or "")
    lines = [ln.rstrip() for ln in combined.splitlines() if ln.strip()]
    return lines[-_EVIDENCE_LINES:]
