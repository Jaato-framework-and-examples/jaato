# Phase 3 closure recap — confined runner + seat-flip

**Status**: Critical path complete.  This document is the audit-of-record
for Phase 3's structural completion and the seed of the eventual
Phase 3 PR description.

Mirror of Phase 2 plan doc's §8 "post-rebase review fixes" closure
structure.

## What Phase 3 was about

The per-session confined runner architecture, per the parent design
[`per_session_confined_runner.md`](per_session_confined_runner.md):

- Daemon process runs unconfined; holds provider clients, OAuth tokens,
  EventBus, plugin registry, ledger.
- Per-session child process — the **runner** — takes the AppArmor
  profile.  Hosts `JaatoSession`.  Owns workspace FS access.  Spawns
  subprocess-launching plugins (cli, lsp, mcp, interactive_shell,
  notebook).
- Everything session-touching crosses **runner-RPC**.  The daemon-side
  `_jaato` field (the in-process `JaatoSession` reference) is gone.
- Per-session cgroup placement at runner-spawn time; child processes
  inherit by default per the cgroup-v2 kernel contract.

Every word of that was hypothetical when the §7c series started.
It's now structural reality.

## Structural inventory at closure

| Component | Pre-§7c state | Post-§7c+§7d state |
|---|---|---|
| `JaatoServer._jaato: Optional[JaatoClient]` | Live; held the in-process session | **Removed** (`a922082f`) |
| Daemon-side `JaatoSession` | The live session for the model loop | **Gone** — runner-side session is the live one |
| `_start_model_thread` model loop | Called `_jaato.send_message(...)` | Calls `_runner_rpc.session_send_message_threadsafe(...)` (`ae34dc0f`) |
| Daemon-side 7 callback wirings | `set_instruction_budget_callback`, `set_prompt_injected_callback`, `set_continuation_callback`, `set_retry_callback`, `set_mid_turn_interrupt_callback`, `_event_bus_tools._on_subscribed`, plus auth-completion mirror | **Deleted** — collapsed into runner-side `NotificationFrame` emissions consumed by the daemon's 8-event demuxer (`ae34dc0f`) |
| Runner subprocess cgroup placement | Inherits the daemon's cgroup (host's session.scope) | **Per-session cgroup** at spawn time; subprocess children inherit (`6bd31540`) |
| Plugin-level `_cgroup_attach` preexec_fn | Per-Popen explicit migrate | **No-op via inheritance** (peer-review v2 obs #2 realized; `6bd31540`) |
| `JaatoClient` | Daemon-side facade + SDK facade | **SDK facade only**; daemon constructs `JaatoRuntime` directly (`998b4a83` → `a922082f`) |
| Runner-side `JaatoSession` | None | Bootstrapped from `SessionInitEnvelope` |
| Runner-RPC handler surface | Phase 2 echo + tool.execute only | **35+ session RPCs** spanning history, budget, replay, fork, send_message, session methods (5b + 5c.1-5c.5 + earlier) |

## §7c sub-commit ledger (the seat-flip)

The §7c series ran 14 implementation commits + audit docs.  Notable
milestones:

| Commit | Step | What it did |
|---|---|---|
| `6e31d375` | 6.6.4.1 | `NotificationFrame` protocol on the `session.send_message` stream channel |
| `973923c6` | 6.6.4.2 | Runner-side notification-emission machinery (6 session callbacks) |
| `68abe7c8` | 6.6.4.3a | `session.try_completion_nudge` RPC (atomic check-and-increment) |
| `ae34dc0f` | 6.6.4.3b | **Atomic seat-flip** — send_message → runner-RPC + 9-callback collapse + 7-wiring delete + daemon-side demuxer install |
| `9ea2f827` | 6.6.4.4 | Narrow WIRING deletions (3 safe-only sites; 3 unsafe deferred to 6.6.4.5) |
| `977d69db` | 6.6.4.5a | 4 `get_runtime()` → `self._runtime` reads migrated |
| `250f3d43` | 6.6.4.5b | 15 existing-RPC reads migrated (in-flight scope narrowed) |
| `bff782a4` | 6.6.4.5c.1 | `session.get_auth_info` RPC + missing-method add |
| `592cbf94` | 6.6.4.5c.2 | `session.get_user_commands` RPC (dict-shape) |
| `175e9220` | 6.6.4.5c.3 | `session.execute_user_command` RPC (Path A per-type reconstruction) |
| `68572a92` | 6.6.4.5c.4 | `session.get_model_completions` RPC (+ command_router site discovered mid-flight) |
| `ad774a54` | 6.6.4.5c.5 | `session.get_tool_schemas` RPC (Path D finale) |
| `998b4a83` | 6.6.4.5d | Daemon-direct `JaatoRuntime` construction (Path A transitional) |
| `a922082f` | 6.6.4.5e | **Atomic `_jaato` field removal** (seat-flip closure) |
| `6bd31540` | §7d | Runner-subprocess cgroup attach (peer-review v2 obs #2 realized) |

## Process metrics

### Audit-discipline statistics

**20 audits across the §7c+§7d arc.  20 actionable findings.  Zero
false positives.**

The audit-discipline pattern emerged organically around audit #5
(`cd3ecf20`'s §6.6 boundary redraw) and held through closure.
Every audit before an architectural pivot earned its keep.

### The 3 process properties that explain the rate

**1. Audits forced grep-before-design.**
Every audit started with "let me actually check what's in the code"
rather than "let me reason about what should be in the code."
Reviewer-reasoning lost to code-grep evidence three times explicitly:

- **Refinement 2 (runtime-cache)**: §6.6.4.5 audit recommended a
  daemon-side `_runtime.get_tool_schemas()` cache.  In-flight 5b
  grep revealed `JaatoRuntime.get_tool_schemas()` returns the
  registry's *full* set, NOT the session-resolved subset.  Cache
  approach would have broken `signal_completion_in_surface`
  filtering.  Pivoted to new `session.get_tool_schemas` RPC.

- **Path B (stringify pre-wire) for execute_user_command**: 5c.3
  audit assumed Path B (value-only stringify).  Pre-implementation
  grep at core.py:4044-4099 revealed 3 daemon-side structured-access
  sites (`isinstance(result, HelpLines)` + `.lines`, `result.get("success")`,
  IPC-return fallback).  Path B would have broken all 3.  Pivoted
  to Path A (per-type tagged reconstruction).

- **"4 sites" callback miscount**: 6.6.2 audit identified 4 callback
  sites.  Re-grep at 6.6.4.2 found 3 additional sites in
  `_start_model_thread` plus 2 per-call kwargs the original audit
  hadn't categorized (`on_usage_update`, `on_gc_threshold`).
  9-callback collapse, not 7.

Grep-first epistemics scaled.  Three named instances; many more
unrecorded similar shifts.

**2. The inverse-virtue pattern produced as much value as
bug-catching.**

Audits cancelled work that was about to ship:

- **§7b.3 withdrawal** (2 commits cancelled): empirical investigation
  determined the proposed work would parallel an existing path.
- **5c elimination** (24-30 tests + 5 commits + 2 RPC handlers + 2
  daemon-side wrappers cancelled): missing-method audit found
  `set_agent_identity` / `set_ui_hooks` are daemon-side state
  mutations with no runner-side propagation.  Both calls just
  deleted with field removal in 5e.
- **§7d `_cgroup_attach` field preservation**: instinct was to
  delete the daemon-side `ToolExecutor._cgroup_attach` field (a
  no-op since §7c step 6.2).  Audit Q4 preserved it as the §3.12
  disk-restore fallback path.

That's a class of audit value distinct from regression-prevention.

**3. Backlog discipline preserved adjacent findings.**

Audits routinely surfaced scope-adjacent issues.  Capturing them
as backlog entries — not folding them in mid-stream — kept the
per-commit scope tight while preventing the "I know we found
something" drift.

Three backlog entries captured during the §7c+§7d arc:

- [Runner-side `_ui_hooks` is None — tool lifecycle events silently no-op](project_backlog_runner_ui_hooks_gap.md)
  (§6.6.4.5c.0 audit Finding 3)
- [Daemon-side description-callback hook is silently broken post-6.6.4.3b](project_backlog_description_callback_gap.md)
  (§6.6.4.4 audit Finding 2)
- §3.11 isolated-subagent opt-in (deferred per §7d Q6) — entry in
  [`project-backlog.md`](../project-backlog.md)

### Test count progression

- Pre-§7c series: ~750 server tests
- Post-§7d: 903 server tests
- **+153 net regression-pin tests** across the series
- 3 additional `TestRealKernel` integration tests gated on cgroup-v2
  writable parent (skip on hosts without cgroup-v2 access)

### Commit count

| Bucket | Count |
|---|---|
| §7c implementation sub-commits | 14 |
| §7c audit-doc commits | 5 (6.6.2, 6.6.3, 6.6.4, 6.6.4.3, 6.6.4.5) |
| §7c mid-implementation audit findings (in commit messages) | 4 (5b in-flight, 5c.2 wire-shape, 5c.3 path pivot, 5c.4 inventory miss) |
| §7d implementation | 1 |
| §7d audit doc | 1 |
| Backlog entries | 3 commits |
| Phase 3 closure recap (this doc) | 1 |
| **Total Phase 3 series** | **~29 commits** |

## Deferred sub-tracks (out of critical-path closure)

These are explicitly out of Phase 3 closure criteria per their
respective audits:

### §3.11 isolated-subagent opt-in

**Status**: Backlog entry shipped (`c018db53`).
**Gate**: §7c seat-flip (shipped).  Architectural prerequisite satisfied;
implementation work is the runner-spawn for `agent_params.isolated: true`
subagents with sub-AppArmor/sub-cgroup naming.

### Step 7 — Wire PromptOperatorHandler (and the bidirectional runner-RPC)

**Status**: Step 7 disposition audit complete (in the §3c rpc-surface
design doc).  Scope discovered larger than initially framed:
PromptOperatorHandler is one of 4 wiring gaps on both ends of the
runner→daemon RPC.  4 sub-commits planned (7.0 audit done; 7.1-7.3 +
conditional 7.4).
**Gate**: permission-plugin runner-side activation.
**Critical-path independence**: Step 7 closure isn't required for
Phase 3's structural completion.  The seat-flip + cgroup migration
ARE Phase 3's architectural milestone.

### Description-callback regression (Finding 2)

**Status**: Backlog entry at `project_backlog_description_callback_gap.md`.
**Gate**: NotificationFrame protocol extension (event_type
`description_updated`) OR runner-side
`set_description_callback` install hook.

### Runner-side `_ui_hooks` gap (Finding 3)

**Status**: ✅ **CLOSED** by Path F (cycle 7).  Option A from the
backlog (extend NotificationFrame protocol) shipped: runner-side
`_AgentUIHooksNotificationShim` emits 4 new event_types
(`tool_call_start`, `tool_call_end`, `tool_output`, `turn_progress`).
Subsequent cycles surfaced two follow-on layers (per-callback-SCOPE
mismatch for bootstrap events → Path I closed at cycle 11;
PermissionInputModeEvent companion → Path J closed at cycle 12).
See "Integration-test verification arc" below.

## Integration-test verification arc (cycles 1-13)

After §7c+§7d closed the structural seat-flip, integration testing
against a real provider surfaced 10 architectural layers across 6
distinct chains.  Each chain was a separately-migrated subsystem
that the structural-completion audits didn't anticipate.  Closure
arrived at cycle 13 with Test #3 (the load-bearing ASK round-trip)
passing end-to-end for the first time: 🔒 rendered, Answer> engaged,
'y' routed, tool ran (✓ with output "hello").

### Six chains, 10 layers, 6 fix paths

| Chain | Layer | Path | Cycle closed | Commit |
|---|---|---|---|---|
| Bootstrap (session-init) | 1 — snapshot wrap | A | 2 | `20b16326` |
| Bootstrap | 2 — IPC bootstrap dispatch | B | 3 | `c68151a8` |
| Bootstrap | 3 — runtime.connect envelope | C | 4 | `4d9cf8b7` |
| Bootstrap | 4 — configure_plugins + prereqs | D | 5 | `00fa6d86` |
| Send-message in-band | 5 — usage_update race | E.1-3 | 6 | `e66414c7` |
| Send-message in-band | 6 — save/replay .role wire format | E.4-5 | 6 | `e66414c7` |
| Streaming response (send_message scope) | 7 — ui_hooks bridge | F | 7 | `dc759a94` |
| Save re-entrancy | 8 — incremental save synchronous | H | 10 | `1192dd5f` |
| Agent-lifecycle (bootstrap scope) | 9 — on_agent_created fires before shim | I | 11 | `956014c7` |
| Permission-emit completeness | 10 — PermissionInputModeEvent companion missing | J | 12 | `1a8a37e4` |

Plus probe-driven localization scaffolding (cycles 8-10) committed as
`25bf6b59` + `7687cb8f`, stripped at cycle 13 (`97613261`).

### Architectural patterns that repeated across chains

**Pattern 1 — daemon-side emit path orphaned by seat-flip.**  Paths I
and J close instances of the same shape: a daemon-side emit that
pre-§7c was driven in-process by a daemon plugin became dead post-§7c
because the runner-side plugin is now the one in the loop.  The fix
in both cases: synthesize the emit at a daemon-side entry point that
receives the runner's signal (`_create_main_agent` for Path I,
`PromptOperatorHandler.handle()` for Path J).

**Pattern 2 — re-entrancy race in notification handlers.**  Paths E
and H close instances: a daemon-side handler invoked from the
runner-RPC notification path makes blocking RPCs back into the same
runner that's still processing the original request.  Path E
(cycle 6) closed `usage_update`'s 2 in-band RPCs via payload-batching
+ daemon-side caching.  Path H (cycle 10) closed `_save_session`'s
2 in-band RPCs via daemon-thread deferral.

**Pattern 3 — per-RPC-request shim scope mismatch.**  Path F's shim
install runs inside `_handle_session_send_message` (per-RPC-scoped).
Path I surfaced when `on_agent_created` fired BEFORE the shim was
installed (bootstrap scope).  Recorded as the per-callback-SCOPE
dimension orthogonal to per-callback-TYPE.

### Worker-correction discipline (3 instances)

The user's verdicts diagnosed the gap at the symptom layer, but in
three cases the actual gap was one layer upstream:

| Cycle | User diagnosis | Worker correction |
|---|---|---|
| 4 | "Runtime not connected — fix connect call" | Envelope schema gap; `project` + `location` not threaded through (Path C) |
| 9 | "Missing `server.emit(ToolCallStartEvent)` inside `ServerAgentHooks.on_tool_call_start`" | All 12 hooks already emit; gap is downstream (eventually closed as Path H) |
| 12 | "TUI permission-prompt rendering gap (client-side)" | TUI grep showed zero subscribers for `PermissionRequestedEvent`; TUI listens for `PermissionInputModeEvent`; producer never emits it post-§7c (Path J) |

**Meta-lesson recorded**: when verdict diagnoses "missing X in layer
Y", grep layer Y for the relevant handler registration FIRST — zero
matches = bug is upstream.  5-second check supersedes probe
instrumentation when the verdict's hypothesis is testable by grep.

### Methodology evolution: static-audit → probe-driven

Cycles 1-7 used pre-implementation static-audit + integration-test.
Each chain took 1-3 cycles to close.  Cycle 8 introduced probe
instrumentation (10 probes inserted at suspected handoff points).
Cycle 10 named the exact offending line in `session_manager.py:1611`
in ONE cycle.  Probes were stripped at cycle 13 once their purpose
was served.

**Phase 4+ guidance**: when integration tests surface silent failures
(no errors logged), reach for probe instrumentation immediately
rather than continuing static-audit iteration.  Each probe is a
binary signal that bisects the call graph.

### Final regression-pin inventory

Across the 6 fix paths, 70 new regression-pin tests landed:

| Path | Test file | Pin count |
|---|---|---|
| D | `test_runner_configure_plugins_path_d.py` | 12 |
| E | `test_send_message_active_turn_path_e.py` | 14 |
| F | `test_streaming_response_path_f.py` | 20 |
| H | `test_save_reentrancy_path_h.py` | 8 |
| I | `test_agent_created_bootstrap_path_i.py` | 8 |
| J | `test_permission_input_mode_emit_path_j.py` | 8 |

Each pin documents not just "this works" but "this layer's regression
shape" so a future refactor that re-introduces the same gap fails
fast with the regression's own commit reference in the failure
message.

### Known sub-gaps deferred post-cycle-13

- **Path J.A — `call_id` propagation**: `PromptPayload` doesn't carry
  `call_id`; `PermissionInputModeEvent.call_id` is `None`.  TUI's
  per-tool-block correlation degrades for parallel-tool flows.
  Filed: `project_backlog_path_j_sub_gaps_call_id_editable_metadata.md`.
- **Path J.B — `editable_metadata` schema lookup**: requires
  `permission_plugin` reference inside `PromptOperatorHandler` or
  enriched `PromptPayload`.  Edit-and-approve flow can't activate.
  Filed: same doc as J.A.
- **Cycle-13 multi-turn UX hint**: "cumulative 16 PermissionRequestedEvents
  + Error: 1bcd84... rendering" — investigate whether J.A or J.B
  explains it before closing the backlog.

## What the next phase should preserve

The audit-discipline pattern that ran 20-for-20 across Phase 3's
critical path is the recommended discipline for Phase 4+ work.  The
13-cycle integration-test arc layered two additional disciplines on
top:

1. **Pre-implementation grep before architectural commits** — cross-grep
   to verify reviewer's design assumptions match what's in the code.
2. **Audit-doc commits before implementation commits** at architectural
   decision points — gives reviewer a chance to pre-correct scope.
3. **In-flight scope narrowing** when implementation cross-grep
   surfaces audit gaps — pre-greenlit by reviewer per the "always
   split" policy.
4. **Backlog discipline** for adjacent findings — capture, don't fold.
5. **Inverse-virtue activations** — when audit reveals the proposed
   work targets state with no consumer, the proposed work shouldn't
   exist either.
6. **Worker-correction over reviewer-deference** (cycles 4/9/12) — when
   user's diagnosis points to a symptom layer, grep that layer for
   the implied handler/emit/wiring; zero matches = bug is upstream.
   Bias toward upstream localization saved 3 cycles across the arc.
7. **Probe-driven localization for silent failures** (cycles 8-10) — when
   no errors fire, blanket instrumentation at suspected handoff
   points bisects the call graph in 1 cycle.  Strip probes once
   localization closes.
8. **Per-callback-TYPE + per-callback-SCOPE enumeration** for
   in-process-to-RPC migrations — Path F caught per-callback-TYPE
   gaps; Path I revealed the orthogonal per-callback-SCOPE axis.
   Both dimensions must be audited.

## Audit anchors for future bisect

- Seat-flip closure: `a922082f` (§7c step 6.6.4.5e)
- Cgroup migration: `6bd31540` (§7d)
- First audit-discipline commit (boundary redraw): `cd3ecf20` (§6.6)
- Path D 5-handler completion: `ad774a54` (§7c step 6.6.4.5c.5)
- Inverse virtue precedents: `a88676ca` (5c.0 missing-method audit) and
  the §7d disposition audit at `b16d31f3`
- Integration-test arc chain-closure commits: see "Six chains" table above
- Probe-driven cycle-10 localization: `7687cb8f`
- Cycle-13 cleanup (probe strip): `97613261`

Phase 3 is **structurally AND functionally complete**.  Test #3
empirically passes end-to-end.
