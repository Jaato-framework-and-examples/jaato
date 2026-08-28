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
        f"## Workspace\n`{context.workspace_path}`\n\n"
        f"## Files present ({len(listing)} shown)\n"
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
