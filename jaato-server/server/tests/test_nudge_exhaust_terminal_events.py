"""Pin test for the nudge-exhaust terminal-event emission path.

PR #179 (Finding D, 2026-05-21).  When the completion-nudge guard
exhausts ``MAX_COMPLETION_NUDGES`` without the agent calling
``signal_completion``, the model thread must emit:

  - ``ErrorEvent(error_type="NudgeExhausted")``
  - ``SessionTerminatedEvent(reason="error")``

BEFORE the existing ``AgentStatusChangedEvent(status="done")``.

Pre-fix the only event was ``AgentStatusChangedEvent(status="done")``
— which fires on NORMAL COMPLETION TOO — so cascade observers
couldn't distinguish success from nudge-exhaust failure.  Surfaced
by peer's kb-orchestrator v152-retry-12 (transform step 5 looped
201 turns + exhausted nudges + observer saw no terminal signal +
90-min poll timeout).

This pin is source-level (the alternative — exercising the full
model_thread path — would require mocking ~20 dependencies).
Verifies the load-bearing structural elements are present so a
future refactor that removes them fails loudly.

Composes with PR #178 (Phase 1 cascade-as-client): the emitted
``SessionTerminatedEvent(reason="error")`` triggers
``_apply_default_cascade_policy`` → headless/cascade-owned
sessions auto-unload.  Observers' ``watch_session_failures`` fire
sub-second.
"""

from __future__ import annotations

import re
from pathlib import Path


def _core_py_source() -> str:
    """Read the server/core.py source for source-pin assertions."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # jaato-server/ → ../..
    core_path = repo_root / "jaato-server" / "server" / "core.py"
    assert core_path.is_file(), f"core.py not found at {core_path}"
    return core_path.read_text()


def test_core_py_emits_error_event_on_nudge_exhaust():
    """The nudge-exhaust path must emit ``ErrorEvent`` with
    ``error_type="NudgeExhausted"``.  Cascade observers + clients
    subscribed to ErrorEvent get a typed signal."""
    src = _core_py_source()
    # Tolerate minor formatting variance: look for the literal
    # quoted token + an ErrorEvent emit somewhere.
    assert 'error_type="NudgeExhausted"' in src, (
        "core.py is missing ``ErrorEvent(error_type=\"NudgeExhausted\")`` "
        "emission in the nudge-exhaust path.  Without this, cascade "
        "observers can't distinguish nudge-exhaust failure from "
        "normal completion.  See PR #179."
    )


def _assert_chokepoint_emits_terminal(src: str) -> None:
    """The single ``_emit_error_termination`` chokepoint constructs
    ``SessionTerminatedEvent(reason="error")`` carrying ``error_summary`` +
    ``error_type`` (so observers surface the cause without log-grep).  Both
    error paths route through it, so this is where the terminal-event contract
    now lives."""
    marker = "def _emit_error_termination("
    assert marker in src, "core.py missing the _emit_error_termination chokepoint"
    # Window spans the (long) docstring + body so it reaches the actual emit.
    body = src[src.index(marker):src.index(marker) + 3000]
    assert "SessionTerminatedEvent(" in body and 'reason="error"' in body, (
        "_emit_error_termination must emit SessionTerminatedEvent(reason=error)."
    )
    assert "error_summary=error_summary" in body and "error_type=error_type" in body, (
        "_emit_error_termination must carry error_summary + error_type so "
        "observers distinguish failure classes without log-grep."
    )


def test_core_py_emits_session_terminated_on_nudge_exhaust():
    """The nudge-exhaust path routes through the error-termination chokepoint,
    which emits ``SessionTerminatedEvent(reason="error")`` so Phase 1's default
    cascade policy fires — and emits ``AgentErrorEvent`` first (recovery offer),
    making the invariant structural."""
    src = _core_py_source()
    trace_marker = "NUDGE_EXHAUSTED"
    assert trace_marker in src, (
        f"core.py missing nudge-exhaust trace marker {trace_marker!r}"
    )
    window = src[src.index(trace_marker):src.index(trace_marker) + 2000]
    # The nudge path delegates to the single chokepoint with the NudgeExhausted type.
    assert "_emit_error_termination(" in window, (
        "core.py nudge-exhaust path must route through _emit_error_termination "
        "(the single error-terminal chokepoint that emits AgentErrorEvent first)."
    )
    assert '"NudgeExhausted"' in window, (
        "core.py nudge-exhaust path must pass error_type=\"NudgeExhausted\" to "
        "the chokepoint so observers can match on the class string."
    )
    # The chokepoint carries the terminal-event contract.
    _assert_chokepoint_emits_terminal(src)


def test_core_py_nudge_exhaust_distinguishing_condition():
    """The detection condition must combine ``nudges_fired >=
    MAX_COMPLETION_NUDGES`` AND ``signal_completion_in_surface``.
    The latter prevents firing the terminal events for sessions
    where signal_completion isn't even available (interactive
    sessions where the tool is filtered)."""
    src = _core_py_source()
    # Look for the distinguishing condition.  Whitespace-tolerant
    # via regex.
    cond_pattern = re.compile(
        r"nudges_fired\s*>=\s*MAX_COMPLETION_NUDGES"
        r"[^)]*?signal_completion_in_surface",
        re.DOTALL,
    )
    assert cond_pattern.search(src), (
        "core.py nudge-exhaust condition must combine "
        "`nudges_fired >= MAX_COMPLETION_NUDGES` AND "
        "`signal_completion_in_surface`.  Without both, terminal "
        "events would fire on interactive sessions (where "
        "signal_completion is filtered) OR on normal completions "
        "where nudges_fired stayed 0.  See PR #179."
    )


def test_core_py_emits_agent_status_done_for_backward_compat():
    """AgentStatusChangedEvent(status="done") still fires on the
    nudge-exhaust path AFTER the new terminal events, for backward
    compat with consumers that don't watch SessionTerminatedEvent.
    """
    src = _core_py_source()
    marker_idx = src.index("NUDGE_EXHAUSTED")
    window = src[marker_idx:marker_idx + 2500]
    assert "AgentStatusChangedEvent" in window, (
        "core.py must still emit AgentStatusChangedEvent on the "
        "nudge-exhaust fall-through for back-compat.  See PR #179."
    )
    # Order check: the terminal-event chokepoint call must appear BEFORE
    # AgentStatusChangedEvent in the window (terminal events first, then the
    # back-compat status change).
    st_idx = window.find("_emit_error_termination(")
    asc_idx = window.find("AgentStatusChangedEvent")
    assert st_idx >= 0 and asc_idx >= 0, "both the chokepoint call and the status event must be present"
    assert st_idx < asc_idx, (
        "the _emit_error_termination chokepoint (terminal signal) must be "
        "invoked BEFORE AgentStatusChangedEvent on the nudge-exhaust path so "
        "observers receive the terminal signal first."
    )


def test_core_py_provider_error_path_carries_error_context():
    """Server 0.6.159+ (Q2): the provider-error fallthrough path
    (the ``finally`` block guarded by ``terminal_error is not None``)
    must populate ``error_summary`` + ``error_type`` from the live
    Exception so cascade observers can surface the failure cause
    without grepping the daemon log.

    Together with the nudge-exhaust assertion above, this covers
    BOTH error-path SessionTerminatedEvent emit sites.
    """
    src = _core_py_source()
    # The provider-error path routes the live Exception through the chokepoint.
    marker = "if terminal_error is not None:"
    assert marker in src, (
        f"core.py missing terminal_error guard {marker!r}.  Was the "
        f"provider-error path refactored?  Update this test."
    )
    window = src[src.index(marker):src.index(marker) + 1500]
    assert "_emit_error_termination_from_exc(terminal_error)" in window, (
        "core.py provider-error path must route the live Exception through "
        "_emit_error_termination_from_exc (the chokepoint), not emit directly."
    )
    # The from_exc wrapper derives error_summary/error_type from the Exception.
    fe = "def _emit_error_termination_from_exc("
    assert fe in src, "core.py missing _emit_error_termination_from_exc"
    fe_body = src[src.index(fe):src.index(fe) + 1500]
    assert "error_type=type(exc).__name__" in fe_body, (
        "_emit_error_termination_from_exc must derive error_type=type(exc).__name__."
    )
    assert "error_summary=str(exc)" in fe_body, (
        "_emit_error_termination_from_exc must derive error_summary=str(exc)."
    )
    # And the chokepoint emits them on SessionTerminatedEvent(reason=error).
    _assert_chokepoint_emits_terminal(src)
