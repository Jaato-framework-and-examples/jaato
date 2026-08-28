"""Script grader — run a command against the mutated workspace.

The classic programmatic rubric: ``mvn clean compile``, ``pytest -q``,
``cargo build``.  Exit 0 is PASS.

The distinction this adapter is careful about is *whose* fault a non-zero
exit is.  A compiler that rejects the agent's code is a FAIL — the agent
was exercised and produced something wrong.  A command that does not
exist on the runner is a BLOCKED — nothing about the agent was
established.  Conflating them is how a benchmark ends up reporting that
every model fails a task whose toolchain was never installed.
"""
from __future__ import annotations

import os
import subprocess
from typing import List

from ..manifest import GraderSpec
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
    """

    def __init__(self, spec: GraderSpec) -> None:
        self.spec = spec

    def grade(self, context: GraderContext) -> Verdict:
        command = self.spec.config.get("run")
        claim = f"`{command}` succeeds in the workspace"

        if not command:
            return blocked(self.spec, "script grader runs",
                           "manifest grader has no 'run' key")

        truncated = context.truncation_reason
        if truncated:
            return blocked(self.spec, claim,
                           f"arm {truncated}; the workspace reflects a "
                           "truncated run")

        if not context.workspace_path.is_dir():
            return blocked(self.spec, claim,
                           f"workspace does not exist: {context.workspace_path}")

        timeout = float(self.spec.config.get("timeout_seconds", 600))
        expect_exit = int(self.spec.config.get("expect_exit", 0))

        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(context.workspace_path),
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "JAATO_EVAL": "1"},
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
        for line in _tail(proc.stdout, proc.stderr):
            verdict.note(line)
        return verdict


def _tail(stdout: str, stderr: str) -> List[str]:
    """Last lines of combined output, for verdict evidence."""
    combined = (stdout or "") + (stderr or "")
    lines = [ln.rstrip() for ln in combined.splitlines() if ln.strip()]
    return lines[-_EVIDENCE_LINES:]
