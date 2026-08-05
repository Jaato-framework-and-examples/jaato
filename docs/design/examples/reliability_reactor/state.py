"""Per-session state for the example reliability reactor (Phase 1).

REFERENCE EXAMPLE — see this directory's README.  This is the state a
tenant-authored reliability reactor holds per session in the import cache.  It
REUSES the public reliability primitives (``FailureKey``, ``EscalationRule``,
``PatternDetector``) rather than re-implementing them — the migration moves the
*wiring* to a reactor, the *logic* stays in ``shared.plugins.reliability``.

Phase 1 scope: the trust ledger + behavioral pattern detector.  No enforcement
(forced-approval / headless gate) — that is Phase 2 (presentation-aware), see
docs/design/reliability-event-driven-migration.md §5–§6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

# Public reliability primitives (jaato-server) — reused, not reimplemented.
from shared.plugins.reliability.types import (
    EscalationRule,
    FailureKey,
    TrustState,
)
from shared.plugins.reliability.patterns import PatternDetector


@dataclass
class ReliabilityReactorState:
    """One session's reliability state, kept alive in the import cache.

    Lives in ``reactor_logic._session_states[session_id]`` so it survives the
    framework's per-dispatch script reload (the *script* reloads; this module's
    globals are cached in ``sys.modules`` — same mechanism drift_monitor uses).

    Holds:
      * ``failures`` — consecutive-failure count per ``FailureKey`` (the
        adaptive-trust ledger, compact form).
      * ``escalated`` — keys currently in ``TrustState.ESCALATED`` (so we emit
        ``reliability.escalated`` + nudge exactly ONCE per transition, not every
        subsequent failure).
      * ``detector`` — the public ``PatternDetector`` (behavioral patterns:
        repetitive / read-only-stall / announce-no-action / error-retry …).
      * ``last_nudge_turn`` — per-key cooldown so we don't nudge every turn.
    """

    session_id: str
    rule: EscalationRule = field(default_factory=EscalationRule)
    failures: Dict[str, int] = field(default_factory=dict)
    escalated: Set[str] = field(default_factory=set)
    last_nudge_turn: Dict[str, int] = field(default_factory=dict)
    turn_index: int = 0
    detector: PatternDetector = field(init=False)

    def __post_init__(self) -> None:
        # The pattern hook is wired by reactor_logic after construction so the
        # detector can call back into the reactor's emit+nudge path.
        self.detector = PatternDetector(session_id=self.session_id)

    # ---- adaptive tool-trust (the FailureKey ledger) ------------------------

    def record_result(
        self, tool_name: str, args: dict, success: bool, is_error_result: bool
    ) -> bool:
        """Update the trust ledger for one tool result.

        ``is_error_result`` is the framework-computed deeper check delivered on
        ``tool.call_completed`` (PR #319): a tool that "succeeded" but returned
        an error body still counts as a failure — the executor's ``success``
        flag alone misses it.

        Returns True IFF this result newly escalates the key (caller emits
        ``reliability.escalated`` + nudges on a True return; subsequent failures
        of an already-escalated key return False to avoid nudge spam).
        """
        key = FailureKey.from_invocation(tool_name, args).to_string()
        failed = (not success) or is_error_result
        if not failed:
            # A clean success resets the consecutive counter (compact recovery;
            # the full plugin tracks a RECOVERING window — out of scope here).
            self.failures.pop(key, None)
            self.escalated.discard(key)
            return False

        self.failures[key] = self.failures.get(key, 0) + 1
        threshold = self.rule.count_threshold or 3
        if self.failures[key] >= threshold and key not in self.escalated:
            self.escalated.add(key)  # TrustState.TRUSTED -> ESCALATED
            return True
        return False

    def is_escalated(self, tool_name: str, args: dict) -> bool:
        """Whether this tool+params is currently escalated (Phase-2 enforcement
        reads this; in Phase 1 it is observability only)."""
        return FailureKey.from_invocation(tool_name, args).to_string() in self.escalated

    # ---- nudge cooldown -----------------------------------------------------

    def should_nudge(self, key: str, min_turns_between: int = 1) -> bool:
        """Rate-limit nudges per key to at most once per ``min_turns_between``
        turns, so steering stays a hint, not a flood."""
        last = self.last_nudge_turn.get(key)
        if last is not None and (self.turn_index - last) < min_turns_between:
            return False
        self.last_nudge_turn[key] = self.turn_index
        return True
