"""Judge grader — score the arm with another jaato session.

The prototype's ``build_judge`` stage, generalised: a session whose
profile declares a rubric ``completion_payload_schema``, handed the arm's
output and asked to score it.  Because the rubric is a completion schema,
the score comes back *typed* — the provider enforces the shape at
sampling time and the framework validates it before emitting the
payload.  There is no free-text score to parse.

The profile owns the rubric.  This adapter only reads a numeric field out
of the returned payload and compares it to a threshold, so changing what
"good" means is a schema edit, not a code change.

COST WARNING
============

A judge runs a full session per arm.  On a six-backend sweep with three
repeats that is eighteen judge sessions per task, which can cost more
than the arms being measured.  ``gate_on`` exists for this: name the
graders that must pass first, and the judge is skipped (BLOCKED, with the
reason recorded) when they did not.

BLOCKING IS SYNCHRONOUS
=======================

``grade()`` is synchronous for every adapter, so the sweep driver can run
graders in a worker thread without adapters needing to know whether they
are on an event loop.  This one opens its own loop internally; it must
therefore not be called from inside a running loop.  The driver
(:mod:`jaato_eval.runner`) satisfies that by dispatching graders through
``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from ..manifest import GraderSpec
from ..verdict import FAIL, PASS, Verdict
from .base import GraderContext, blocked

#: How many workspace paths to show the judge.  Enough to orient it; not
#: so many that a node_modules tree eats the context window.
_MAX_LISTING = 200


class JudgeGrader:
    """Open a session on ``config['profile']`` and score the arm.

    Config keys:
        profile: Rubric profile name.  Required.
        agent: Persona name (``.jaato/agents/<name>.md``).  Optional, but
            required for any rubric using a PREFETCH — the ``{{!py:}}``
            placeholder lives in the persona, so without an agent the
            script never runs and the judge silently reverts to having
            nothing but a file listing.
        threshold: Minimum ``score`` counted as PASS (default 0.7).
        score_field: Payload key holding the score (default ``"score"``).
        socket_path: Daemon socket.  Defaults to the client's own default.
        gate_on: Grader ids that must have passed before this judge runs.
            When any named grader did not pass, the judge is BLOCKED
            rather than spending a session on an arm already known bad.
    """

    def __init__(self, spec: GraderSpec) -> None:
        self.spec = spec

    def grade(self, context: GraderContext) -> Verdict:
        profile = self.spec.config.get("profile")
        claim = f"judge {profile!r} scores the arm at or above threshold"

        if not profile:
            return blocked(self.spec, "judge grader runs",
                           "manifest grader has no 'profile' key")

        truncated = context.truncation_reason
        if truncated:
            return blocked(self.spec, claim,
                           f"arm {truncated}; judging a truncated run "
                           "would score the interruption, not the work")

        unmet = self._unmet_gates(context)
        if unmet:
            return blocked(self.spec, claim,
                           "gate_on graders did not pass: "
                           + ", ".join(f"{g}={context.prior_verdicts.get(g, 'not run')}"
                                       for g in unmet)
                           + " — judge skipped to avoid spending a session on an "
                             "arm already known bad")

        try:
            payload = asyncio.run(self._ask_judge(profile, context))
        except RuntimeError as exc:
            # Raised when there is already a running loop in this thread.
            return blocked(self.spec, claim,
                           f"judge could not run: {exc}. Graders must be "
                           "dispatched off the event loop (see module docstring).")
        except Exception as exc:  # noqa: BLE001 — a judge that errored graded nothing
            return blocked(self.spec, claim, f"judge session failed: {exc!r}")

        if payload is None:
            # Prefer the framework's own account over this adapter's guess.
            # jaato #654 added completion_gap for exactly this: before it,
            # the "asked and refused" path emitted no terminal event at all,
            # so the guess below was the best anyone could do and it named
            # the schema — the one thing that was fine.
            if context.completion_gap:
                return blocked(
                    self.spec, claim,
                    f"the judge was asked to signal completion and never did "
                    f"({context.completion_gap}); its rubric schema is not "
                    "implicated")
            return blocked(
                self.spec, claim,
                f"judge profile {profile!r} returned no typed payload — either "
                "it declares no completion_payload_schema, or it answered in "
                "prose without calling signal_completion. Check the profile's "
                "schema first, then whether its persona suppresses the "
                "instruction this adapter sends.")

        field = str(self.spec.config.get("score_field", "score"))
        if field not in payload:
            return blocked(self.spec, claim,
                           f"judge payload has no {field!r} key: "
                           f"{sorted(payload)}")

        try:
            score = float(payload[field])
        except (TypeError, ValueError):
            return blocked(self.spec, claim,
                           f"judge {field!r} is not numeric: {payload[field]!r}")

        # A JUDGE THAT COULD NOT JUDGE MUST BLOCK, NOT FAIL.  `errors` is
        # the standing escape hatch on a completion schema: it is where a
        # judge says "I could not carry out the assessment", as distinct
        # from "I assessed it and it was bad".  Scoring the first as FAIL
        # blames the arm for the judge's own broken tooling — measured:
        # a rubric reported `errors: ["Attempted to read answer.txt but
        # file-read tool returned a path-not-found error"]` with score 0.0,
        # and the arm was recorded as a failure although its artefact was
        # correct and on disk.
        #
        # The CAUSE of that flake is still unknown — 3 of 4 bare probes on
        # identical fresh workspaces scored 1.0 and one scored 0.0, with no
        # arm involved.  It is NOT runner-tier workspace resolution: every
        # runner-tier filesystem_query init in that log resolved a real
        # path (39 of 39).  An earlier note here blamed one; that compared
        # a DAEMON-tier init against a runner-tier one and was withdrawn.
        # This guard does not depend on the cause: whatever makes a judge
        # unable to judge, the arm is not the evidence for it.
        reported = self.spec.config.get("errors_field", "errors")
        judge_errors = payload.get(reported) or []
        if judge_errors:
            return blocked(
                self.spec, claim,
                f"the judge reported it could not complete the assessment: "
                f"{_brief(judge_errors)} — BLOCKED rather than FAIL, because "
                "this is the judge's failure and says nothing about the arm")

        threshold = float(self.spec.config.get("threshold", 0.7))
        state = PASS if score >= threshold else FAIL
        verdict = Verdict(
            grader_id=f"judge:{profile}",
            claim=claim,
            state=state,
            detail=f"{field}={score:.3f} threshold={threshold:.3f}",
        )
        for key in ("reasoning", "criteria_met", "errors"):
            if key in payload:
                verdict.note(f"{key}: {_brief(payload[key])}")
        return verdict

    def _unmet_gates(self, context: GraderContext) -> List[str]:
        """Gate ids from ``gate_on`` that did not record a PASS.

        A gate naming a grader that has not run yet counts as unmet — the
        alternative (treating "not run" as satisfied) would let a manifest
        ordering mistake silently disable the gate.
        """
        gates = self.spec.config.get("gate_on") or []
        if isinstance(gates, str):
            gates = [gates]
        return [g for g in gates if context.prior_verdicts.get(str(g)) != PASS]

    async def _ask_judge(self, profile: str, context: GraderContext) -> Optional[Dict[str, Any]]:
        """Run the judge session and return its typed payload."""
        from jaato_sdk.client.ipc import IPCClient  # lazy: SDK not needed to import this package

        kwargs: Dict[str, Any] = {
            "profile": profile,
            "workspace_path": str(context.workspace_path),
            "config_root": str(context.config_root),
        }
        agent = self.spec.config.get("agent")
        if agent:
            kwargs["agent"] = str(agent)
        # The RUN's socket wins over a manifest override: the judge must
        # score on the same daemon the arm ran on, and only the sweep
        # knows which that was.  A manifest `socket_path` remains as an
        # escape hatch for a judge deliberately hosted elsewhere.
        socket_path = context.socket_path or self.spec.config.get("socket_path")
        if socket_path:
            kwargs["socket_path"] = socket_path

        async with IPCClient.session(**kwargs) as session:
            return await session.complete(_render_prompt(context))


def _render_prompt(context: GraderContext) -> str:
    """Build the judge's input: the arm's payload plus a workspace listing.

    The judge profile decides how much further to look — if it carries
    ``filesystem_query`` or ``cli`` it can read the files itself.  This
    prompt only guarantees it knows what the agent claimed and what
    exists.
    """
    listing = _workspace_listing(context)
    payload_json = json.dumps(context.payload, indent=2, sort_keys=True, default=str) \
        if context.payload is not None else "(none — profile declared no completion schema)"
    return (
        "Score the agent run described below against your rubric.\n\n"
        # NO ABSOLUTE PATH.  The filesystem tools resolve a relative path
        # against the workspace root themselves (_resolve_path), so an
        # absolute one buys nothing — and handing a model a 150-character
        # path to reproduce inside a tool argument invites transcription
        # error.  Observed: a judge reported
        #   Path does not exist: .../jaato-eval-tests-e1050a88-.../example_echo-a-fileopenrouter_gpt5mini_0
        # against a real
        #   .../jaato-eval-tests/e1050a88-.../example_echo-a-file@openrouter_gpt5mini_0
        # — a '/' become '-' AND an '@' dropped, two corruptions no single
        # sanitiser produces, in a path the model had been asked to copy.
        # Arm workspace names come from arm_id (task@set#repeat), so they
        # are long and punctuated by construction: this engine supplies the
        # hazard, so this engine removes it.
        "## Files\nPaths below are RELATIVE TO THE WORKSPACE ROOT. Pass them "
        "to your tools exactly as written — do not prefix them with a "
        "directory, and do not construct an absolute path.\n"
        f"({len(listing)} shown)\n"
        + "\n".join(f"- {p}" for p in listing)
        + "\n\n## The agent's completion payload\n```json\n"
        + payload_json
        + "\n```\n\n## Run facts\n"
        f"- turns: {context.turns}\n"
        f"- finish_reason: {context.finish_reason}\n"
        f"- tool calls: {len(context.ledger.entries)}\n"
        # WITHOUT THIS THE JUDGE ANSWERS IN PROSE AND NEVER SIGNALS.
        # Measured: every arm came back BLOCKED with "returned no typed
        # payload" while the rubric's completion_payload_schema was
        # present and correct.  A profile with a schema is OFFERED
        # signal_completion; it is not compelled to call it, and a prompt
        # that only says "score this" reads as a request for an answer.
        # The instruction belongs here rather than in each rubric profile:
        # every judge needs it, and leaving it to the profile means each
        # author rediscovers this failure — with an error message that
        # points at the schema, which is the one thing that was fine.
        "\n## Required\nWhen you have finished scoring, call "
        "`signal_completion` with your rubric's fields. Do not reply in "
        "prose — the score is only read from that payload.\n"
    )


def _workspace_listing(context: GraderContext) -> List[str]:
    """Workspace-relative paths, capped, sorted, directories excluded."""
    root = context.workspace_path
    if not root.is_dir():
        return []
    paths = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() or ".git" in p.parts:
            continue
        paths.append(str(p.relative_to(root)))
        if len(paths) >= _MAX_LISTING:
            break
    return paths


def _brief(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"
