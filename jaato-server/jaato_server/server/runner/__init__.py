"""Per-session confined runner subprocess (Phase 2 onwards).

The runner hosts the runner-tier plugins for ONE top-level session
(plus any subagent sessions sharing the parent's runner per design
§4.3) and self-confines via AppArmor at startup.

See:
- ``docs/design/per_session_confined_runner.md`` — design spec.
- ``docs/design/per_session_confined_runner_phase2_plan.md`` — Phase 2
  implementation plan.

Phase 2 ships only:
- the bootstrap dance (libapparmor self-confine + /proc verify),
- the bidirectional RPC dispatcher (length-prefixed JSON over fd 3),
- a minimal ``ToolExecutor`` registering ``echo`` + ``cli_based_tool``,
- the cli plugin's executor body migrated runner-side.

Plugin discovery, subagent hosting, full session-state migration —
all Phase 3.
"""
