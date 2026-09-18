"""Per-turn accounting — what one session's event stream adds up to.

Lifted out of :mod:`jaato_eval.runner` so a second consumer could share
it without importing the runner: a driver arm (``harness.kind: driver``,
:mod:`jaato_eval.driver`) opens MANY sessions, and its observer keeps one
of these per session id and sums them at the end.  The runner re-exports
the two names, so a reader of the session path finds them where it always
did.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

#: Usage keys summed across turns rather than taken from the last turn.
#: ``total_tokens`` is deliberately excluded: for a prompt-inclusive
#: provider it is the end-of-turn CONTEXT SIZE, not spend, so summing it
#: across turns overcounts.  ``spend_total_tokens`` is the billed figure.
#:
#: The cache pair follows the same rule, and used to break it.
#: ``cache_read_tokens`` / ``cache_creation_tokens`` are the turn's LAST
#: RESPONSE's figures — a level, not spend — so adding them across turns
#: produced neither.  The SDK documents the distinction as load-bearing
#: (``jaato_sdk.events``): under ``model_tiers`` a mid-turn tier switch
#: re-reads the whole prefix cold at the new model, and the last-response
#: figures hide exactly that miss.  The fingerprint of the bug was visible
#: in the archived corpus — three of four Gemini arms reported
#: ``cache_creation`` equal to ``cache_read`` to the token, which is one
#: level reading copied into two fields, not two independent billed sums
#: (jaato #800).  ``spend_cache_read_tokens`` /
#: ``spend_cache_creation_tokens`` are already summed over the turn's
#: responses, the same shape as ``spend_total_tokens``, so summing them
#: across turns is the right operation.
#:
#: ``prompt_tokens`` / ``output_tokens`` were the last pair here with the
#: same defect, and could not be fixed with the cache pair because no spend
#: counterpart reached the wire: the session accumulated ``spend_prompt`` /
#: ``spend_output`` per response and dropped both at the boundary.
#: jaato #802 carries them, so every member of this tuple is now a billed
#: figure and the tuple's name is true of all of it.
_SUMMED_USAGE = ("spend_prompt_tokens", "spend_output_tokens",
                 "spend_total_tokens",
                 "spend_cache_read_tokens", "spend_cache_creation_tokens",
                 "reasoning_tokens", "thinking_tokens")


class _TurnAccumulator:
    """Collects per-turn facts as ``TurnCompletedEvent``s arrive.

    Usage arrives per turn, not once at the end, so an arm's real spend is
    only knowable by accumulating.  ``cost_usd`` stays ``None`` unless at
    least one turn reported a cost — a zero would be indistinguishable
    from "free", which it is not.
    """

    def __init__(self) -> None:
        self.turns = 0
        self.finish_reason = "stop"
        self.usage: Dict[str, Any] = {k: 0 for k in _SUMMED_USAGE}
        self.cost_usd: Optional[float] = None
        self.termination_reason = ""
        self.termination_detail = ""
        self.termination_error_type = ""
        self.agent_error: Optional[str] = None
        self.completion_gap: Optional[str] = None
        # PROVIDER-SIDE FACTS THE WIRE DOES NOT CARRY YET.  Both stay None
        # on every arm today: the OpenRouter provider reads
        # ``native_finish_reason`` off the choice and the routed upstream
        # off the response, and neither reaches TurnCompletedEvent (jaato
        # #766).  Read here anyway, by name, so the per-arm report fills
        # these columns the day the framework reports them — the
        # alternative is a report that keeps printing "—" for a fact the
        # daemon has started sending.
        self.native_finish_reason: Optional[str] = None
        self.upstream_provider: Optional[str] = None

    def on_terminated(self, event: Any) -> None:
        """Record why the session wound down.

        ``SessionTerminatedEvent.reason`` is the only place an abnormal
        stop names ITSELF.  A budget ceiling in particular short-circuits
        BEFORE any turn runs, so no ``TurnCompletedEvent`` fires and the
        per-turn ``finish_reason`` never mentions it — the SDK's own
        docstring warns that a driver reading only turns reports "a
        generic failure ... a ceiling stop indistinguishable from a
        break".  That is exactly what this engine did until it subscribed
        here.

        ``natural`` / ``client_request`` / ``stopped`` are ordinary
        wind-downs and say nothing about completeness; only
        ``budget_exhausted`` and ``error`` name a stop.
        """
        self.termination_reason = getattr(event, "reason", "") or ""
        detail = (getattr(event, "details", None)
                  or getattr(event, "error_summary", None) or "")
        self.termination_detail = str(detail)
        # The terminal's TYPE, alongside its prose.  ``reason="error"``
        # says only that something failed; the type is what separates a
        # daemon that died mid-turn from an agent that finished its work
        # and never called signal_completion (see :mod:`jaato_eval.sign_off`).
        self.termination_error_type = str(
            getattr(event, "error_type", "") or "")

    def note_unsigned(self, exc: Exception) -> None:
        """Record an error terminal the arm is being graded through anyway.

        Called by :func:`_run_session` when ``complete()`` raises a
        terminal :mod:`jaato_eval.sign_off` classifies as *unsigned* — the
        agent worked and left a workspace, it just never called
        ``signal_completion``.  The facts land here rather than on a local
        so that the arm's result and every grader see the same account of
        why no payload arrived, and so a session whose
        ``SessionTerminatedEvent`` never reached us (the exception carries
        the same two fields) is described just as fully.

        Never overwrites what the terminal event already said: the event is
        the daemon's own account, the exception is the SDK's relay of it.
        """
        error_type = str(getattr(exc, "error_type", "") or "")
        if error_type and not self.termination_error_type:
            self.termination_error_type = error_type
        summary = str(getattr(exc, "error_summary", "") or "")
        if summary and not self.termination_detail:
            self.termination_detail = summary
        if not self.termination_reason:
            self.termination_reason = "error"
        self.agent_error = str(exc)

    def on_turn(self, event: Any) -> None:
        self.turns += 1
        reason = getattr(event, "finish_reason", None)
        if reason:
            self.finish_reason = reason
        # LATCHED PER TURN, not read off the last one.  completion_gap
        # rides EXACTLY ONE event and is read-and-cleared, so a session
        # that gave up and then received more work stops reporting it —
        # sampling only the final turn would miss the very turn that
        # carried the fact.  It means "asked twice and refused", not
        # "did not signal on this turn", so a legitimately multi-turn
        # session never sets it.
        gap = getattr(event, "completion_gap", None)
        if gap:
            self.completion_gap = str(gap)
        # LATCHED, not overwritten by a later turn that omits them: a
        # gateway reports the upstream once per response and a normalised
        # finish reason has no native twin on most turns, so "the last turn
        # did not say" must not erase what an earlier one did.
        native = getattr(event, "native_finish_reason", None)
        if native:
            self.native_finish_reason = str(native)
        upstream = getattr(event, "upstream_provider", None)
        if upstream:
            self.upstream_provider = str(upstream)
        usage = getattr(event, "usage", None)
        if usage is None:
            return
        for key in _SUMMED_USAGE:
            value = getattr(usage, key, None)
            if isinstance(value, (int, float)):
                self.usage[key] += value
        cost = getattr(usage, "cost_usd", None)
        if isinstance(cost, (int, float)):
            self.cost_usd = (self.cost_usd or 0.0) + float(cost)

    def snapshot(self) -> Dict[str, Any]:
        out = dict(self.usage)
        out["cost_usd"] = self.cost_usd
        return out
