# Project Backlog

Tracked backlog items. Each entry links to a design doc or implementation
plan. Promote to a feature branch / ticket when work is ready to start.

> Some older backlog items live in the jaato memory store as
> `project_backlog_*.md` (e.g. `project_backlog_fork_replay.md`,
> `project_backlog_conversation_fork.md`). New items are tracked here;
> migrate to the memory store if/when that becomes the canonical location.

## Open

### Rewind-with-hint for truncated tool calls

- **Design**: [docs/design/rewind-with-hint.md](design/rewind-with-hint.md)
- **Status**: Design drafted, not scheduled.
- **Summary**: Detect when an LLM emits a tool call with empty/truncated
  arguments (typically because `max_tokens` cut off mid-serialization of a
  large `content` parameter), rewind the last assistant message to keep its
  narration but drop the failed `tool_use`, and inject a synthetic user-role
  hint naming the specific tool and suggesting a chunked-write strategy.
- **Why it matters**: Replaces the current workaround of baking large-payload
  guidance into every session's system prompt. Reactive, targeted, bounded.
- **Entry points**: `jaato-server/shared/jaato_session.py:3962` (hook after
  `_add_model_response_to_history`); new `shared/rewind.py` module for the
  detector.
- **Open questions**: Cache invalidation cost on Anthropic provider; detector
  scope v1 conservatism; budget-reset semantics.

### Clarification + References plugins lack RunnerRPCChannel equivalents

- **Design**: [docs/design/project_backlog_clarification_references_runner_rpc_gap.md](design/project_backlog_clarification_references_runner_rpc_gap.md)
- **Status**: Pre-existing gap surfaced by §7c Step 7 disposition audit +
  confirmed by Step 7.4 investigation.  **Latent regression** post-§7c
  seat-flip — runner-fired clarification / references ASKs orphan
  (no daemon-side relay).  Existing tests pass because no automated
  test exercises the regression path.
- **Summary**: The permission plugin ships `runner_rpc_channel.py`
  routing ASKs through `client.prompt_operator` daemon-RPC (wired
  end-to-end in §7c Steps 7.1+7.2+7.3).  Clarification + references
  both run runner-side (`PLUGIN_TIER = "runner"`) but their channels
  return the in-process queue directly — no runner-side execution
  detection, no RPC bridge.  Same orphan pattern §7c Step 7 closed
  for permission.
- **Why it matters**: User-visible impact: model invokes
  `request_clarification` → ASK fires runner-side → queue write the
  daemon never reads → tool hangs.  Same for `selectReferences`.
- **Likely fix shape**: Mirror permission's pattern per plugin —
  new `types.py` + `runner_rpc_channel.py` + plugin
  `_get_channel` extension + daemon-side handler in
  `server/runner_rpc_handlers/`.  Plus `set_runner_rpc` registration
  + `respond_to_*` dual-path routing extension (same pattern as
  Step 7.3).  ~4-6 hours implementation for both plugins.

### Runner-side `_ui_hooks` is None — tool lifecycle events silently no-op

- **Design**: [docs/design/project_backlog_runner_ui_hooks_gap.md](design/project_backlog_runner_ui_hooks_gap.md)
- **Status**: Pre-existing gap surfaced by §7c step 6.6.4.5 audit (Finding 3).
  Not in scope for the §7c series.
- **Summary**: Post-§7c step 6.6.4.3b seat-flip, the runner-side `JaatoSession`
  is the live session for tool execution but its `_ui_hooks` attribute is
  never set (cross-grep of `server/runner/` confirms zero references).  All
  10 `if self._ui_hooks: self._ui_hooks.on_*(...)` callsites in
  `jaato_session.py` (`on_tool_call_start`, `on_tool_call_end`,
  `on_tool_output`, `on_turn_progress`,
  `on_agent_instruction_budget_updated`) are null-guarded → silently no-op
  runner-side.
- **Why it matters**: Whether this matters depends on whether something else
  routes those events daemon-side via a different path (e.g., executor-
  boundary wrapping, NotificationFrame extension).  Needs investigation
  before deciding on a fix.
- **Likely fix shapes**: (a) extend the §7c step 6.6.4.1 NotificationFrame
  protocol with new event_types for the 5 ui_hooks methods (10th–14th
  events alongside the current 8); (b) install a runner-side hooks shim at
  bootstrap that emits notification frames matching the AgentUIHooks
  protocol surface.

### §3.11 isolated-subagent opt-in (`agent_params.isolated: true` → fresh runner)

- **Design**: [docs/design/per_session_confined_runner.md §3.11](design/per_session_confined_runner.md)
- **Status**: Termination-hook portion shipped (subagent termination hook +
  reliability cleanup at `subagent/plugin.py:118` + M4 plumbing).  Isolation-
  opt-in portion deferred — was gated on the §7c seat-flip, which shipped
  at commit `a922082f`.  Not in scope for §7d.
- **Summary**: Subagents currently share the parent's runner subprocess
  per §4.3 default.  The §3.11 spec adds an opt-in via `agent_params.isolated:
  true` that spawns a fresh runner subprocess with its own sub-profile
  (AppArmor + cgroup).  Cross-grep at §7d audit time (commit `b16d31f3`):
  no production sites found for `agent_params.*isolated` /
  `isolated_subagent` — the architectural prerequisite (seat-flip) is
  satisfied but the opt-in path isn't wired.
- **Why it matters**: Enables stronger isolation for untrusted subagent
  workloads (e.g., model-generated code execution under a tighter sandbox
  than the parent's profile permits).  Important for jaato-as-a-service
  deployments where subagents may run partially-trusted plugins.
- **Likely fix shape**: Extend `SubagentPlugin._spawn_subagent` to detect
  the `agent_params.isolated: true` flag; route through
  `SessionManager.create_session(...)` with a fresh `RunnerSpawner.spawn`
  invocation (the §7d cgroup-attach plumbing is already in place); wire a
  sub-AppArmor-profile / sub-cgroup naming scheme (e.g.,
  `jaato-ws-<parent>-sub-<n>`).
- **Out of scope**: Default-share vs opt-in-isolation policy semantics —
  the spec already settled "default-share for parity with pre-§3.11
  behavior; isolation is opt-in only."

### Daemon-side description-callback hook is silently broken post-6.6.4.3b

- **Design**: [docs/design/project_backlog_description_callback_gap.md](design/project_backlog_description_callback_gap.md)
- **Status**: Pre-existing gap surfaced by §7c step 6.6.4.4 audit (Finding 2).
  Not in scope for the §7c series.
- **Summary**: `_setup_session_plugin` wires `on_description_changed` on the
  daemon-side `session_plugin` instance.  The runner has its own
  `session_plugin` instance constructed from the bootstrap envelope's plugin
  list.  When the model invokes the `set_description` tool, it fires the
  runner-side instance's callback → daemon never sees it →
  `SessionDescriptionUpdatedEvent` no longer emits.
- **Why it matters**: Session description updates (used by the UI's session
  picker and persistence layer's auto-titling) silently stop flowing once a
  session is past first-message setup.  Sessions retain whatever description
  was set pre-seat-flip, or none at all.
- **Likely fix shapes**: (a) new `description_updated` NotificationFrame
  event_type extending the §7c step 6.6.4.1 protocol (would expand the
  daemon-side demuxer from 8 branches to 9); (b) runner-side
  `set_description_callback` install hook in
  `_install_session_notification_callbacks` (parallels the 6 callbacks
  already wired there).
