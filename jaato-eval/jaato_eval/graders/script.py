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
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Tuple

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

#: Prefix for the per-parameter variables exported from ``agent_params``.
#: Namespaced so a task's inputs cannot shadow anything the surrounding
#: environment already means.
_PARAM_PREFIX = "JAATO_EVAL_PARAM_"

#: Variable carrying the whole ``agent_params`` mapping as JSON, under the
#: authors' own key spellings.  It is what distinguishes "this task has no
#: such parameter" from "the parameter is there and empty" — the per-key
#: variables cannot, because an unset variable and an empty one read alike
#: in a shell.  A grader that must be strict about a parameter it depends
#: on can assert against this rather than passing vacuously.
_PARAMS_JSON = "JAATO_EVAL_PARAMS"

#: Characters an environment variable name cannot carry.  Parameter keys
#: are YAML identifiers by convention but nothing enforces it, so
#: ``issue-id`` and ``issue.id`` have to become something a shell can name.
_NON_IDENTIFIER = re.compile(r"[^A-Z0-9_]")


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

    Environment the command inherits, on top of ``os.environ``:
        ``JAATO_EVAL=1``: this is a graded run, not an interactive one.
        ``JAATO_EVAL_PARAM_<KEY>``: one variable per ``agent_params``
            entry, key upper-cased with non-identifier characters
            replaced by ``_``.
        ``JAATO_EVAL_PARAMS``: the whole mapping as JSON.

    So a grader that depends on an input says so in the manifest::

        - kind: script
          run: bash acceptance.sh compliant "$JAATO_EVAL_PARAM_ISSUE_ID"

    rather than baking the value into ``acceptance.sh``, where nothing
    can notice when the task's input moves on without it.
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

        param_env, collision = _param_env(context.agent_params)
        if collision:
            return blocked(self.spec, claim, collision)

        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(context.workspace_path),
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "JAATO_EVAL": "1", **param_env},
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
                f"graded without a completion payload — the agent never "
                f"called signal_completion ({context.termination_error_type})")
        for line in _tail(proc.stdout, proc.stderr):
            verdict.note(line)
        return verdict


def _env_name(key: str) -> str:
    """Environment variable name for one ``agent_params`` key."""
    return _PARAM_PREFIX + _NON_IDENTIFIER.sub("_", key.upper())


def _encode(value: Any) -> str:
    """Render one parameter value for a shell.

    Strings pass through verbatim — quoting them would mean every grader
    had to unquote, and the common case is a bare identifier.  Everything
    else is JSON, which is the only encoding that survives the round trip
    for the values ``agent_params`` actually holds: a nested dict or list
    has no other textual form a grader could parse back, ``True`` becomes
    the ``true`` the manifest author wrote rather than Python's ``True``,
    and ``None`` becomes ``null`` instead of the empty string that would
    make an explicit null indistinguishable from an absent key.

    ``default=str`` keeps an exotic value (a ``Path``, say) from raising
    out of a grader — the adapter's contract is to describe what it can
    do, never to explode inside the sweep driver.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _param_env(agent_params: Mapping[str, Any]) -> Tuple[Dict[str, str],
                                                         Optional[str]]:
    """Exported environment for ``agent_params``, or a reason it cannot be.

    Returns ``(env, None)`` normally, and ``({}, reason)`` when two
    parameter keys collapse onto one variable name (``issue-id`` and
    ``issue_id`` both want ``$JAATO_EVAL_PARAM_ISSUE_ID``).  Picking a
    winner there would hand the grader one input while the arm ran with
    the other — precisely the silent disagreement this export exists to
    remove — so the ambiguity is surfaced as BLOCKED and the task author
    renames one key.
    """
    env = {_PARAMS_JSON: json.dumps(dict(agent_params), sort_keys=True,
                                    default=str)}
    origin: Dict[str, str] = {}
    for key, value in agent_params.items():
        name = _env_name(key)
        if name in origin:
            return {}, (f"agent_params keys {origin[name]!r} and {key!r} "
                        f"both map to ${name}; rename one — grading "
                        "against whichever won would be arbitrary")
        origin[name] = key
        env[name] = _encode(value)
    return env, None


def _tail(stdout: str, stderr: str) -> List[str]:
    """Last lines of combined output, for verdict evidence."""
    combined = (stdout or "") + (stderr or "")
    lines = [ln.rstrip() for ln in combined.splitlines() if ln.strip()]
    return lines[-_EVIDENCE_LINES:]
