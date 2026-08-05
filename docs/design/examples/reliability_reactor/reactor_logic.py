"""Example reliability reactor — event routing + steer (Phase 1).

REFERENCE EXAMPLE — see this directory's README.  The event-driven successor to
the in-process reliability plugin: detect behavioral/tool-trust drift from the
event stream, STEER with a non-blocking ``ctx.inject_prompt`` nudge (both
session types), and EMIT ``reliability.*`` observability.  No enforcement —
that is Phase 2 (presentation-aware), see
docs/design/reliability-event-driven-migration.md.

State survives between turns the same way drift_monitor's does: this MODULE is
import-cached in ``sys.modules`` across the framework's per-dispatch *script*
reload, so ``_session_states`` persists.  The action shim (registration.py)
only imports ``handle_event``.
"""

from __future__ import annotations

from typing import Any, Dict

from shared.plugins.reliability.types import BehavioralPattern, FailureKey

from .state import ReliabilityReactorState

# Import-cache-survived per-session state.  Keyed by ctx.session_id (the session
# that fired the event) — Phase-1 decision: the trust ledger is PER-SESSION
# (reactor-native), not a global/persisted tier.
_session_states: Dict[str, ReliabilityReactorState] = {}


def reset_state() -> None:
    """Drop all per-session state (tests / explicit teardown)."""
    _session_states.clear()


def _get_state(ctx: Any) -> ReliabilityReactorState:
    """Get-or-create this session's state and wire the pattern hook once."""
    sid = ctx.session_id
    st = _session_states.get(sid)
    if st is None:
        st = ReliabilityReactorState(session_id=sid)
        # When the detector flags a behavioral pattern, route it through the
        # reactor's emit+nudge path.  Bind ctx via a closure captured per
        # session — patterns are evaluated synchronously inside the same
        # handle_event dispatch that fed the detector, so ctx is current.
        st.detector.set_pattern_hook(
            on_pattern_detected=lambda p: _on_pattern(st, ctx, p)
        )
    _session_states[sid] = st
    return st


# ---- steer + observe --------------------------------------------------------

def _nudge(ctx: Any, message: str) -> None:
    """Steer: write a corrective hint into the session's NEXT turn.  Always
    non-blocking — safe in headless/cascade sessions by construction (the §7c
    invariant).  Phase-2 adds presentation-aware enforcement on top."""
    ctx.inject_prompt(message)


def _on_escalation(st: ReliabilityReactorState, ctx: Any, tool_name: str, count: int) -> None:
    key = f"{tool_name}#{count}"
    ctx.emit_event(
        "reliability.escalated",
        {
            "session_id": st.session_id,
            "tool_name": tool_name,
            "consecutive_failures": count,
            "turn_index": st.turn_index,
        },
    )
    if st.should_nudge(key):
        _nudge(
            ctx,
            f"[reliability] '{tool_name}' has failed {count}x in a row. "
            f"Stop and reconsider: check inputs/preconditions, try a different "
            f"approach, or confirm this tool is the right one before retrying.",
        )


def _on_pattern(st: ReliabilityReactorState, ctx: Any, pattern: BehavioralPattern) -> None:
    ptype = pattern.pattern_type.value
    ctx.emit_event(
        "reliability.pattern_detected",
        {
            "session_id": st.session_id,
            "pattern_type": ptype,
            "tool_sequence": list(pattern.tool_sequence),
            "repetition_count": pattern.repetition_count,
            "turn_index": pattern.turn_index,
        },
    )
    if st.should_nudge(f"pattern:{ptype}"):
        _nudge(
            ctx,
            f"[reliability] behavioral pattern detected: {ptype} "
            f"(seq={pattern.tool_sequence[-4:]}). Break the loop — change "
            f"strategy or take a concrete next action toward the goal.",
        )


# ---- event routing ----------------------------------------------------------

def handle_event(params: Dict[str, Any], event: Dict[str, Any], ctx: Any) -> None:
    """Reactor entry point.  Routes the subscribed bus events into the
    per-session trust ledger + behavioral detector, then steers/observes.

    ``event`` is the flat merged view (envelope + payload hoisted); read fields
    with ``event.get(...)``.  Subscribed (see registration.py): tool.call_started,
    tool.call_completed, agent.output, turn.completed, plan.step_updated.
    """
    et = event.get("event_type", "")
    st = _get_state(ctx)

    if et == "tool.call_started":
        # Pre-execution: feed the detector (repetitive / read-only / prereq
        # checks fire here in the full plugin).
        st.detector.on_tool_called(
            event.get("tool_name", ""), event.get("tool_args", {}) or {}
        )

    elif et == "tool.call_completed":
        tool = event.get("tool_name", "")
        # NOTE: tool.call_completed carries tool_name+success but NOT tool_args
        # (and not the raw result).  A production reactor correlates by call_id
        # with the tool.call_started it saw to rebuild the FailureKey params; for
        # this example we key on the tool name alone (args={}).
        success = bool(event.get("success", True))
        is_err = bool(event.get("is_error_result", False))  # PR #319 deeper check
        key = FailureKey.from_invocation(tool, {}).to_string()
        if st.record_result(tool, {}, success, is_err):
            _on_escalation(st, ctx, tool, st.failures.get(key, 0))
        # Feed the detector's result stream (drives ERROR_RETRY_LOOP).
        st.detector.on_tool_result(tool, success, {"error": True} if is_err else {})

    elif et == "agent.output":
        st.detector.on_model_text(event.get("text", "") or "")

    elif et == "turn.completed":
        # Evaluate end-of-turn patterns (e.g. ANNOUNCE_NO_ACTION), then open the
        # next turn's window.  No turn.started bus event is subscribed, so we
        # advance the boundary here (matches the plugin's on_turn_end/start).
        st.detector.on_turn_end()
        st.turn_index += 1
        st.detector.on_turn_start(st.turn_index)

    elif et == "plan.step_updated":
        # Plan context (prerequisite-policy gating consults this in the full
        # plugin). Out of scope for the Phase-1 example — left as an extension
        # point.
        pass
