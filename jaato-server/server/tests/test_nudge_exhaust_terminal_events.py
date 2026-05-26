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


def test_core_py_emits_session_terminated_on_nudge_exhaust():
    """The nudge-exhaust path must emit ``SessionTerminatedEvent``
    with ``reason="error"`` so Phase 1's
    ``_apply_default_cascade_policy`` triggers session unload for
    headless / cascade-owned sessions (closes Finding B-equivalent
    stall on nudge-exhaust)."""
    src = _core_py_source()
    # Find the nudge-exhaust block by its trace string + check that
    # a SessionTerminatedEvent emission appears in proximity.
    trace_marker = "NUDGE_EXHAUSTED"
    assert trace_marker in src, (
        f"core.py missing nudge-exhaust trace marker {trace_marker!r}"
    )
    marker_idx = src.index(trace_marker)
    # Examine the ~2KB window after the marker — the emit block
    # should be within this range (current code has it ~30 lines
    # below; ample buffer).
    window = src[marker_idx:marker_idx + 2000]
    assert "SessionTerminatedEvent" in window, (
        "core.py nudge-exhaust path doesn't emit SessionTerminatedEvent. "
        "Without this Phase 1's default cascade policy can't fire — "
        "cascade-owned sessions stall on nudge-exhaust.  See PR #179."
    )
    assert 'reason="error"' in window, (
        "core.py nudge-exhaust SessionTerminatedEvent emission must "
        "use reason=\"error\" so the Phase 1 default policy gate "
        "(reason==error → unload) fires.  See PR #179."
    )
    # Server 0.6.159+ (Q2): the nudge-exhaust SessionTerminatedEvent
    # must also carry error_summary + error_type so cascade observers
    # can distinguish nudge-exhaust from a provider error without
    # grepping the daemon log.
    assert "error_summary=" in window and "error_type=" in window, (
        "core.py nudge-exhaust SessionTerminatedEvent emission must "
        "carry error_summary + error_type (server 0.6.159+).  Without "
        "these, cascade observers can't distinguish NudgeExhausted "
        "from a provider error class without log-grep.  See PR for Q2."
    )
    assert '"NudgeExhausted"' in window, (
        "core.py nudge-exhaust SessionTerminatedEvent emission must "
        "carry error_type=\"NudgeExhausted\" so observers can match "
        "on the class string.  See PR for Q2."
    )


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
    # Order check: SessionTerminatedEvent must appear BEFORE
    # AgentStatusChangedEvent in the window (terminal events first,
    # then the back-compat status change).
    st_idx = window.find("SessionTerminatedEvent")
    asc_idx = window.find("AgentStatusChangedEvent")
    assert st_idx >= 0 and asc_idx >= 0, "both events must be present"
    assert st_idx < asc_idx, (
        "SessionTerminatedEvent must be emitted BEFORE "
        "AgentStatusChangedEvent on the nudge-exhaust path so "
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
    # Locate the terminal_error guard block — emits
    # SessionTerminatedEvent in the finally clause after the model
    # thread's terminal Exception escaped with_retry.
    marker = "if terminal_error is not None:"
    assert marker in src, (
        f"core.py missing terminal_error guard {marker!r}.  Was the "
        f"provider-error path refactored?  Update this test."
    )
    marker_idx = src.index(marker)
    window = src[marker_idx:marker_idx + 1500]
    # The emit must reference the Exception variable for both fields.
    assert "error_summary=str(terminal_error)" in window, (
        "core.py provider-error path must populate "
        "error_summary=str(terminal_error) on SessionTerminatedEvent "
        "(server 0.6.159+).  Cascade observers depend on this for "
        "log-grep-free error surfacing."
    )
    assert "error_type=type(terminal_error).__name__" in window, (
        "core.py provider-error path must populate "
        "error_type=type(terminal_error).__name__ on "
        "SessionTerminatedEvent (server 0.6.159+)."
    )
