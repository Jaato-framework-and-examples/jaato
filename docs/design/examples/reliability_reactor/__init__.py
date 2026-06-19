"""Example reliability reactor (Phase 1) — REFERENCE, see README.md.

The event-driven successor to the in-process reliability plugin: a
tenant-authored, premium-gated reactor that detects tool-trust / behavioral
drift from the event stream and steers with a non-blocking nudge.  Reuses the
public ``shared.plugins.reliability`` primitives + the SDK substrate (jaato PRs
#318 event types / #319 is_error_result).  Not wired/registered in this repo —
copy/adapt into a tenant package's ``jaato.premium_reactors`` entry point.
"""

from .reactor_logic import handle_event, reset_state
from .registration import get_reactor_definition
from .state import ReliabilityReactorState

__all__ = [
    "handle_event",
    "reset_state",
    "get_reactor_definition",
    "ReliabilityReactorState",
]
