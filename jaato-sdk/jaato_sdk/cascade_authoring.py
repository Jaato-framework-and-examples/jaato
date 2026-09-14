"""Typed surface for cascade authors — accumulating umbrella module.

Cascade authors touch a handful of framework contracts when writing
completion processors, reactor handlers, and orchestrators.  Each
contract has historically been documented only in framework
docstrings, discoverable by grepping server-side code.  This module
is the **explicit accumulating destination** for typed surfaces
those authors need.

Currently exposed (server 0.6.160+ / SDK 0.14.2+):

- :class:`ProcessorResult`: return type for
  ``completion_processors[*].validate()`` — separates fatal errors
  from advisory warnings, semantic "not done yet" signals, and
  environment faults (this module).
- :class:`ToolCallEntry`: per-entry shape of
  ``context.tool_calls`` handed to completion processors
  (re-exported from :mod:`jaato_sdk.completion_processors`).

Open gaps (will accumulate here as they bite — see
``project_backlog`` memory):

- ``ReactorHandlerContext`` (premium-owned; coord with jaato-premium)
- Reactor event per-EventType payload typing (extending existing
  :mod:`jaato_sdk.event_payloads`)
- Reactor rule "where" DSL spec (string-based; JSON schema + prose)

Authors should prefer importing from this module:
    from jaato_sdk.cascade_authoring import ProcessorResult, ToolCallEntry

over reaching into individual submodules — the umbrella surface
stays stable across SDK versions even as new types land in
specialised submodules.
"""

from __future__ import annotations

from typing import List, TypedDict

from jaato_sdk.completion_processors import ToolCallEntry


class ProcessorResult(TypedDict, total=False):
    """Return value from ``completion_processors[*].validate(payload, context)``.

    Authors emit fatal errors and advisory warnings in separate lists.
    The framework's ``invoke_processors`` accepts BOTH this TypedDict
    AND the legacy plain ``List[str]`` shape (treated as all-errors)
    so existing processors continue to work unchanged.

    Fields:
        errors: Rejection-worthy issues.  Each entry triggers
            re-prompting the agent with the joined errors as a
            self-correction prompt.  Subject to the processor's
            ``on_error`` policy: if the processor declares
            ``on_error="warn"``, ALL ``errors`` for that processor
            get downgraded to advisory (the proc-level policy wins
            for fatal entries).  Default ``on_error`` is
            ``fail_completion``.
        warnings: Advisory issues.  Logged to the audit trail
            (``cascade_state/processor_audit.jsonl``) but NEVER
            block completion.  Independent of ``on_error`` policy —
            warnings stay warnings regardless of the proc-level
            downgrade knob.

    Authoring rule of thumb:
        - "agent should retry to fix this" → ``errors[]``
        - "operator should see this in the audit but completion is
          fine" → ``warnings[]``

    The semantic split mirrors the schema-side ``warnings[]/errors[]``
    escape hatches documented in the cascade-authoring SKILL.md
    under "Schema-first design" — same naming, same intent, one
    layer down (processor return vs. agent payload).

    Example::

        from jaato_sdk.cascade_authoring import ProcessorResult

        def validate(payload, context) -> ProcessorResult:
            errors: list[str] = []
            warnings: list[str] = []
            for tc in context.tool_calls or []:
                if tc.get("name") != "store_memory":
                    continue
                if not tc.get("success"):
                    errors.append(
                        f"store_memory call at turn {tc['turn_index']} "
                        f"failed — agent must retry."
                    )
                    continue
                result = tc.get("result") or {}
                if not result.get("memory_id"):
                    warnings.append(
                        f"store_memory call at turn {tc['turn_index']} "
                        f"succeeded but didn't return memory_id."
                    )
            return {"errors": errors, "warnings": warnings}

    Backwards-compatible legacy shape (still accepted indefinitely)::

        def validate(payload, context) -> list[str]:
            errors: list[str] = []
            # ... populate errors only ...
            return errors  # treated as ProcessorResult(errors=..., warnings=[])

    Completeness channel (server 0.6.199+):
        incomplete: SEMANTIC "not done yet — keep going" signals,
            used ONLY by processors declared ``phase: "completeness"``
            in the profile.  Such a processor runs DURING
            ``prepare_completion`` (gated on the schema-required floor
            being met) and its ``incomplete[]`` return gates the
            COMPOSITE ``is_complete`` verdict:

            - ``incomplete`` non-empty  → ``is_complete`` stays False;
              the entries surface to the model as neutral "still
              needed" guidance (like pending-field descriptors) — NOT
              errors, NO retry penalty, NO completion block.
            - ``incomplete`` empty/absent → no completeness objection;
              ``is_complete`` follows the schema floor (and, when True,
              the framework may auto-finalize — see the
              ``auto_finalize_on_complete`` quirk).

            ``incomplete`` is distinct from ``errors`` (reject + retry)
            and ``warnings`` (advisory, never blocks).  It expresses a
            THIRD state — "valid so far, but the downstream cascade
            needs more before this run is genuinely complete."  On a
            ``phase: "finalization"`` processor ``incomplete`` is
            ignored (a completeness verdict is meaningless once the
            agent has already chosen to finalize).

    Example completeness processor::

        def validate(payload, context) -> ProcessorResult:
            incomplete: list[str] = []
            caps = context.agent_params.get("detected_capabilities", {})
            if caps.get("rest_surface") and not payload.get("api"):
                incomplete.append(
                    "api is absent but discovery detected a REST "
                    "surface — keep setting api.endpoints[*]."
                )
            return {"errors": [], "warnings": [], "incomplete": incomplete}

    Fault channel (issue #768):
        faults: ENVIRONMENT faults — conditions the agent's next
            attempt cannot clear.  A missing acceptance script, an
            absent ``agent_params`` key, a checks harness that timed
            out: none of these is a wrong answer, and phrasing one as a
            retryable error burns the whole retry budget without ever
            producing a verdict (issue #768 rule 6).  A fault:

            - BLOCKS completion exactly ONCE per session (per
              processor), which is the single round-trip the agent
              needs to record it in the payload's own ``errors[]``, and
              is advisory from then on — blocking repeatedly on
              something no retry can clear is the non-terminating loop
              that ``max_refusals`` exists to prevent;
            - never consumes a refusal, so the ``max_refusals`` budget
              stays available for genuine check failures.

            Write a fault as an instruction, like an error: say what is
            broken, that retrying will not fix it, and what to do
            instead (typically: record it in the payload and signal
            again).

    Example with the fault split::

        def validate(payload, context) -> ProcessorResult:
            script = Path(context.workspace_path) / "acceptance.sh"
            if not script.is_file():
                return {"faults": [
                    "acceptance.sh is missing from the workspace — this is "
                    "an environment fault, not something your fix can "
                    "address. Record it in errors[] and signal completion "
                    "again; do not retry the checks."
                ]}
            failures = run_checks(script)          # one line per failure
            return {"errors": failures}
    """

    errors: List[str]
    warnings: List[str]
    incomplete: List[str]
    faults: List[str]


__all__ = ["ProcessorResult", "ToolCallEntry"]
