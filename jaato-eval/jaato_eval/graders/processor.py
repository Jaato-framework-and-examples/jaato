"""Processor grader — run a completion processor post-hoc.

The framework's completion processors already have a grader's signature::

    def validate(payload, context) -> list[str]   # empty = OK

In-band they gate ``signal_completion`` and drive a retry.  Here the same
module is run *after* the arm, and its return value becomes a verdict
instead of a correction.  An unmodified processor works — the context
object exposes ``tool_calls``, ``workspace_path``, ``config_root`` and
``agent_params`` under the names the framework uses.

THE LEDGER GATE
===============

Most processors worth reusing cross-reference the agent's claims against
``context.tool_calls``.  That ledger now comes from
``jaato_sdk.completion_processors.build_ledger`` (jaato #640), so on a
current daemon these processors run here exactly as they do in-band.

The gate remains for one case: a daemon predating jaato #639 emits call
Parts with no identifier, and nothing can be paired.  Rather than hand a
processor a ledger that may mis-attribute a retry's success to the call
that failed, this adapter detects that the processor reads ``tool_calls``
and returns BLOCKED.  A grader that cannot be run correctly must say so;
a grader that runs on bad data and reports PASS is the vacuous pass this
whole engine is built to refuse.

Processors that only inspect the payload and the filesystem are
unaffected and run regardless of daemon version.

THE PAYLOAD GATE
================

There is no post-hoc substitute for the payload: this adapter's whole
contract is ``validate(payload, context)``.  So an arm the engine graded
through an unsigned terminal — the agent worked, left a workspace, and
never called ``signal_completion`` (jaato #773) — BLOCKS here even though
its script graders returned verdicts.  That is the per-grader split
working: the missing sign-off invalidates exactly the graders that read
the sign-off.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import List

from ..manifest import GraderSpec
from ..verdict import FAIL, PASS, Verdict
from .base import GraderContext, blocked

#: Attribute whose use forces the ledger-faithfulness gate.
_LEDGER_ATTR = "tool_calls"


class ProcessorGrader:
    """Load ``config['script']`` and call its ``validate``.

    Config keys:
        script: Path to the processor module, relative to the task's
            ``config_root`` (or absolute).  Required.
    """

    def __init__(self, spec: GraderSpec) -> None:
        self.spec = spec

    def grade(self, context: GraderContext) -> Verdict:
        script = self.spec.config.get("script")
        claim = f"completion processor {script} accepts the payload"

        if not script:
            return blocked(self.spec, "processor grader runs",
                           "manifest grader has no 'script' key")

        path = Path(script)
        if not path.is_absolute():
            path = context.config_root / path
        if not path.is_file():
            return blocked(self.spec, claim, f"processor not found: {path}")

        if context.payload is None:
            # Name the CAUSE when the engine knows it.  An arm graded
            # through an unsigned terminal (jaato #773) reaches here with a
            # real workspace and no payload, and the generic wording below
            # sends its reader to check a schema that is fine — the same
            # misdirection ``JudgeGrader`` records against its own guess.
            if context.missing_sign_off:
                return blocked(
                    self.spec, claim,
                    f"the agent never called signal_completion "
                    f"({context.termination_error_type}), so there is no "
                    "payload to validate — its workspace was still graded "
                    "by the graders that read the workspace, and the "
                    "profile's completion_payload_schema is not implicated")
            return blocked(self.spec, claim,
                           "arm produced no signal_completion payload "
                           "(profile declares no completion_payload_schema, "
                           "or the agent never completed)")

        source = path.read_text()
        if _reads_ledger(source) and not context.ledger.faithful:
            return blocked(
                self.spec, claim,
                f"{path.name} reads context.{_LEDGER_ATTR}, but the ledger "
                f"could not be reconstructed faithfully: {context.ledger.reason}. "
                "Grading on a best-effort pairing could attribute a retry's "
                "success to the call that failed.")

        try:
            module = _load_module(path)
        except Exception as exc:  # noqa: BLE001 — any import error is BLOCKED
            return blocked(self.spec, claim, f"could not import {path.name}: {exc!r}")

        validate = getattr(module, "validate", None)
        if not callable(validate):
            return blocked(self.spec, claim,
                           f"{path.name} defines no callable validate()")

        try:
            errors = validate(context.payload, context)
        except Exception as exc:  # noqa: BLE001
            return blocked(self.spec, claim,
                           f"{path.name}.validate() raised {exc!r}")

        if not isinstance(errors, list):
            return blocked(self.spec, claim,
                           f"{path.name}.validate() returned "
                           f"{type(errors).__name__}, expected list")

        state = PASS if not errors else FAIL
        verdict = Verdict(
            grader_id=f"processor:{path.name}",
            claim=claim,
            state=state,
            detail="no errors" if not errors else f"{len(errors)} error(s)",
        )
        for err in errors[:10]:
            verdict.note(str(err))
        return verdict


def _reads_ledger(source: str) -> bool:
    """Does this processor read ``context.tool_calls``?

    AST rather than a substring search: the string ``tool_calls`` appears
    in most of these files' docstrings describing the contract, and a
    substring match would gate every processor including the ones that
    only touch the filesystem.  Matching an actual attribute access is
    the difference between witnessing the use and guessing at it.

    Falls back to ``True`` (gate on) when the source will not parse — a
    processor that cannot be analysed is one whose data dependencies are
    unknown, and the safe reading of unknown is "might need the ledger".
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == _LEDGER_ATTR:
            return True
    return False


def _load_module(path: Path):
    """Import a processor module from an arbitrary path.

    Each load gets a unique module name derived from the full path so two
    processors that share a basename (``_audit.py`` in two task trees) do
    not collide in ``sys.modules``.
    """
    name = "jaato_eval_processor_" + "".join(
        c if c.isalnum() else "_" for c in str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
