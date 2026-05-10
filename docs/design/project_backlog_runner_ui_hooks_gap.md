# Backlog: runner-side `_ui_hooks` is None — tool lifecycle events silently no-op

> **Status**: Pre-existing gap surfaced by §7c step 6.6.4.5 audit (Finding 3).
> Not in scope for the §7c series.  See
> [`per_session_confined_runner_phase3_3c_rpc_surface.md`](per_session_confined_runner_phase3_3c_rpc_surface.md)
> for full context.

## Problem

Post-§7c step 6.6.4.3b seat-flip, the runner-side `JaatoSession` is the live
session for the model loop and tool execution.  Its `_ui_hooks` attribute is
**never set** — cross-grep of `jaato-server/server/runner/` confirms zero
references to `ui_hooks`, `set_ui_hooks`, or `AgentUIHooks`.

The 10 callsites in `jaato_session.py` that consume `_ui_hooks`:

| Site (line) | Method called | Context |
|---|---|---|
| 2502 | `on_agent_instruction_budget_updated` | Per-budget-update (now also emitted via NotificationFrame) |
| 5000, 5217 | `on_tool_call_start` | Tool-call boundary |
| 5078, 5268, 5349, 5532 | `on_tool_call_end` | Tool-call completion / failure |
| 5251, 5518 | `on_tool_output` | Streaming tool output chunks |
| 6909 | `on_turn_progress` | Per-turn progress updates |

All are null-guarded (`if self._ui_hooks:`) so they silently no-op when
`_ui_hooks is None` — no crash, no log, just quiet event drop.

## Audit context

Surfaced during the §7c step 6.6.4.5 implementation-review audit while
checking whether `set_ui_hooks` needed a new RPC handler.  The audit found
that the `AgentUIHooks` callable object isn't wire-serializable (it holds
daemon-callable references) AND the runner-side session has no `ui_hooks`
infrastructure at all.  The seat-flip was behavior-preserving for *most*
runner-side flows because the rest of the lifecycle events route through
the §7c step 6.6.4.1 NotificationFrame protocol (8 event_types as of
6.6.4.3b).  The 5 `AgentUIHooks` methods listed above are the surfaces
that don't have NotificationFrame coverage.

## Open questions

1. **Are these events observed by anyone post-seat-flip?**  Some are likely
   redundant with daemon-side observation:
   - Tool start/end events — daemon may observe via the executor-wrapping
     path (verify).
   - `on_agent_instruction_budget_updated` — already covered by the
     `instruction_budget_updated` NotificationFrame event_type.
   - `on_turn_progress` — used by client UIs (TUI tab-bar progress).  Not
     obviously covered elsewhere.
2. **What client UIs depend on each event?**  jaato-tui consumes
   `on_tool_call_start` / `on_tool_call_end` / `on_tool_output` for
   per-tool widget rendering.  Are those UI surfaces still functional
   post-seat-flip?  If yes, events route through some other path
   (verify).  If no, the regression is real but undetected.
3. **Selective vs full coverage?**  If only `on_turn_progress` is
   genuinely missing, a minimal fix is one new event_type.  If all 5 are
   missing, the fix scope grows.

## Likely fix shapes

### Option A — extend NotificationFrame protocol

Add 5 new event_types alongside the current 8, mapping each
`AgentUIHooks` method to a NotificationFrame:

| Event_type | Method | Payload |
|---|---|---|
| `tool_call_start` | `on_tool_call_start` | `{agent_id, tool_name, tool_args, call_id}` |
| `tool_call_end` | `on_tool_call_end` | `{agent_id, tool_name, call_id, result, error?}` |
| `tool_output` | `on_tool_output` | `{agent_id, call_id, chunk}` |
| `turn_progress` | `on_turn_progress` | `{agent_id, ...turn_progress_fields}` |
| (`agent_instruction_budget_updated` — already covered by `instruction_budget_updated`) | | |

Wire a runner-side `_install_ui_hooks_notification_shim(session, request_id)`
helper that installs a shim implementing the `AgentUIHooks` protocol and
emits notification frames.  Daemon-side demuxer extends to 12-13 branches
(8 existing + 4-5 new).

Pros: matches the established §7c step 6.6.4.3b pattern; per-event
granular routing; no new wire infrastructure.

Cons: per-tool-output frame chatter for streaming-heavy tools could be
high-volume.  Existing `StreamFrame` channel already handles output
chunks for the `on_output` callback — `on_tool_output` may overlap.

### Option B — runner-side hooks shim at bootstrap

At runner-side bootstrap (after `JaatoSession.configure`), install an
auto-generated `AgentUIHooks` impl that proxies all 5 methods through
NotificationFrame emission.  Same wire surface as Option A but factored
differently — one helper class instead of per-method shims.

Pros: cleaner separation; the proxy class lives runner-side and can
batch / dedupe events.

Cons: the proxy needs the per-call `request_id` — but `set_ui_hooks` is
called once at session construction, not per-call.  Need a different
threading model (e.g., a context-var or active-call lookup).

## Out of scope for this backlog item

- Adding `set_ui_hooks` as a runner-side RPC handler.  The hook itself
  isn't serializable; the migration path is event-emission, not
  state-transfer.

## Files to touch (when scheduled)

- `jaato-server/server/runner/envelope.py` — new event_type constants
- `jaato-server/server/runner/rpc.py` — install/restore machinery
- `jaato-server/server/runner_rpc_client.py` — daemon demuxer extension
- `jaato-server/server/core.py` — `_build_send_message_notification_handler`
  branches
- `jaato-server/shared/jaato_session.py` — possible setter-call additions
  if the per-call shim approach is chosen
