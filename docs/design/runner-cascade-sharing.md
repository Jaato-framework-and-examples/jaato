# Runner Cascade Sharing — Design Document

**Status**: Phase 0 (design) — locked decisions, awaiting Phase 1 implementation
**Author**: Advisor + Daniel (locked decisions)
**Date**: 2026-05-20
**Origin**: v141-v151 LSP-arc debugging surfaced architectural limit captured in
[`project_backlog_lsp_server_persistence_across_sessions`](../../README.md);
runner-tier subprocess + LSP server per-session lifecycle blocks cascade
enrichment.  This document specifies the cascade-sharing model that closes
the class.

---

## 1. Executive Summary

Today the daemon spawns a fresh **runner subprocess** per session, and that
runner instantiates fresh **plugin instances** including the lsp plugin that
spawns fresh **LSP server subprocesses** (e.g. jdtls).  Every cascade stage
pays a 30-60s cold-start tax for any heavy-init plugin or LSP server, and —
critically — each session's plugin instance has its own state (e.g.
`_connected_servers={"java"}` on one instance but empty on another), so
enrichment running on session B never sees the jdtls that session A
connected.

**This design replaces the per-session lifecycle with a per-cascade
lifecycle** for runner subprocesses, plugin instances, and LSP servers.  A
**cascade-driver ID** identifies a logical multi-session execution (e.g. a
codegen pipeline running discovery → context → codegen-step-1 →
codegen-step-2 → build_descriptor as one cascade).  All sessions sharing
that ID claim the SAME pool slot, which keeps its warm imports + plugin
state + LSP server connection alive across the cascade.

Side benefits:

- **One LSP server per cascade** (vs one per session) → enrichment markdown
  surfaces on stage-2+ renders within the cascade because jdtls's
  `_connected_servers` is preserved across the cascade's sessions.
- Warm imports survive (already true with pool slots; this extends to
  plugin-instance state).
- `~30-60s × N stages` of cold-start tax → paid once per cascade.

---

## 2. Background

### 2.1 Current architecture (server 0.6.141, post 9-PR LSP arc)

```
daemon (unconfined)
├── PoolManager
│   ├── template subprocess (warm imports for 30 runner-tier plugins)
│   └── pool slots (default 2): forked from template, idle
│
├── session.new request arrives:
│   ├── spawn_session_runner claims a pool slot (PoolManager.acquire_slot)
│   ├── compose per-session apparmor profile (jaato-ws-{session_id})
│   ├── dispatch session.bootstrap RPC to slot
│   │   └── slot self-confines to jaato-ws-{session_id} (PR-5a)
│   ├── slot instantiates plugins, calls initialize() on each
│   │   └── lsp plugin starts background thread, spawns jdtls subprocess
│   └── slot serves session.message RPCs for this session
│
└── session teardown:
    ├── slot shuts down plugins (lsp tears down jdtls)
    ├── slot exits (NOT returned to pool)
    └── PoolManager replenishes pool by forking a new slot from template
```

### 2.2 Why per-session lifecycle is the wrong shape

**v141-v151 evidence** (kb-enablement-2.0 cascade):

- 7 cascade stages, each spawns a fresh session → fresh runner → fresh
  lsp plugin → fresh jdtls
- jdtls Maven workspace import: 30-60s cold-start per stage
- Result: 3-6 minutes of pure cold-start waste per cascade
- Plus: **multi-instance state isolation** — v151 trace showed 3 lsp plugin
  instances initializing during one cascade with `_connected_servers=[]` on
  each fresh instance.  Even when one instance connected to jdtls, OTHER
  instances handling enrichment had no visibility → enrichment markdown
  never appeared

The 9-PR LSP arc (PR-150 through PR-158) closed every framework-addressable
layer of the per-session-LSP class:

| PR | Closed |
|---|---|
| #150-#157 | Connect timeout, bounded poll, debug-log scoping, server-binary `ix` grants, `-data` rw grants, TMPDIR auto-inject, workspace_path symmetric resolution |
| #158 | `apparmor_extra_rules` operator knob (closed chained-exec layer for java/JVM) |

Everything in `_load_config_cache`, `connect_server`, `enrich_tool_result`,
and the apparmor composer now resolves correctly.  The remaining barrier is
not a bug at any layer — it's the **architectural choice** to give each
session its own runner subprocess.  Cascade sharing fixes this at the root.

---

## 3. Locked Decisions

These four decisions are inputs to the design (Daniel call, 2026-05-20):

1. **Tenant identifier**: `cascade_driver_id` (NOT `workspace_path`).  The
   tenant is the cascade itself, not the workspace.  Two cascades running
   sequentially against the same workspace get different runner subprocesses;
   one cascade's multiple stages share one runner.

2. **Reuse scope**: within ONE cascade only.  A pool slot serves one
   cascade's sessions, then tears down when the cascade ends (or goes idle).
   It does NOT persist across cascade re-runs.

3. **Apparmor**: ONE profile per cascade, shared across all sessions of
   that cascade.  Profile name shifts from `jaato-ws-{session_id}` to
   `jaato-cascade-{cascade_driver_id}`.  Composed once at cascade start,
   reused for every session of that cascade, torn down at cascade end.

4. **Cascade-driver ID propagation**: IPC client supplies it at the FIRST
   session of the cascade via a new field on `session.new`.  Subsequent
   sessions of the same cascade supply the same ID.  Subagent sessions
   inherit their parent's `cascade_driver_id` automatically.

5. **Cascade end signaling**: idle timeout per `cascade_driver_id`.  No
   explicit `cascade.end` IPC verb.  A configurable timeout
   (`cascade_idle_timeout_seconds`, default 300s = 5 min) — if no session
   activity for that cascade_id during the idle window, PoolManager tears
   down the slot.

6. **Slot allocation when no idle slot has matching `cascade_driver_id`**:
   spawn fresh from pool.  No waiting.  This may temporarily exceed pool
   size; pool manager handles the overflow + replenishes naturally.

---

## 4. Concepts

### 4.1 Cascade-driver ID

- **Type**: opaque UTF-8 string, client-generated, recommended UUID format.
- **Source**: IPC client supplies via new optional field
  `session_new_request.cascade_driver_id: Optional[str]`.
- **Default**: `None` — treated as standalone session (no slot reuse).
- **Propagation**: same client passes same ID for subsequent sessions of
  the cascade.  Subagent sessions auto-inherit parent's cascade_driver_id
  via the existing subagent-spawn path (`runtime.create_session()` adds
  the parent's ID to the child's bootstrap envelope).
- **Collision policy**: no collision detection.  If two unrelated clients
  generate the same ID, they JOIN the same cascade-slot.  UUID convention
  makes this negligible.  Mitigation: log a warning if a cascade_id is
  seen from a different client_id than the first.

### 4.2 Pool slot lifecycle (cascade-aware)

```
slot states:
- IDLE: no session, available for claim
- ACTIVE_FOR_CASCADE(cascade_id): serving a session, locked to cascade_id
- IDLE_FOR_CASCADE(cascade_id): no current session, but reserved for cascade
- TEARDOWN: idle timeout fired, shutting down

claim flow:
  request: session.new with cascade_driver_id=C

  1. PoolManager scans:
     - any slot in IDLE_FOR_CASCADE(C)? → claim it, transition to ACTIVE
     - any slot in IDLE (no cascade affinity)? → claim it, transition to
       ACTIVE_FOR_CASCADE(C), set cascade_id
     - none of above? → spawn fresh slot, claim, transition to ACTIVE
  2. dispatch session.bootstrap to claimed slot
  3. slot calls plugin.initialize() ONLY on plugins not already initialized
     (first session of cascade), OR plugin.reset_for_next_session() (subsequent
     sessions of same cascade)
  4. session runs

session teardown:
  - slot calls plugin.reset_for_next_session() on plugins that need it
  - slot transitions ACTIVE_FOR_CASCADE(C) → IDLE_FOR_CASCADE(C)
  - slot is now available for next session of cascade C

cascade end (idle timeout):
  - PoolManager replenishment thread (every 0.5s) scans IDLE_FOR_CASCADE slots
  - if (now - slot.last_session_end_time) > cascade_idle_timeout_seconds:
    - call plugin.shutdown() on every plugin (full teardown)
    - slot transitions IDLE_FOR_CASCADE(C) → TEARDOWN → process exit
    - PoolManager replenishes pool by forking a new slot from template
```

### 4.3 Plugin reset protocol (`reset_for_next_session()`)

New method on the plugin base class:

```python
def reset_for_next_session(self) -> None:
    """Clear per-session state in preparation for the next session of
    the same cascade.  Default: no-op.

    Override if the plugin holds per-session state that MUST be cleared
    before the next session runs.  Examples of per-session state:
    - conversation history
    - active todo list
    - pending clarifications
    - active session_id reference
    - per-session telemetry counters

    Do NOT clear state that MUST survive across the cascade.  Examples:
    - LSP server connections (the whole point of cascade sharing)
    - warm imports
    - cached config files
    - plugin registry membership
    - workspace_path (constant within a cascade)

    The runner subprocess calls this method on each plugin AFTER a
    session ends + BEFORE the slot returns to IDLE_FOR_CASCADE state.
    """
```

**Audit categories** (cursory — Phase 1 deepens):

| Category | Plugins | Reset needed? |
|---|---|---|
| Stateless / warm-import-only | `cli`, `web_fetch`, `web_search`, `filesystem_query`, `calculator`, `multimodal`, `ast_search`, `bundle`, `environment`, `notebook`, `prompt_library`, `references`, `reliability`, `service_connector`, `webhook` | NO-OP |
| Holds per-session state | `session`, `todo`, `clarification`, `subagent`, `waypoint`, `thinking`, `memory`, `artifact_tracker`, `permission` | YES — needs explicit clear |
| Holds across-session state by design (cascade-sharing PR target) | `lsp`, `mcp`, `interactive_shell`, `sandbox_manager` | NO-OP (state SHOULD survive) |
| Holds workspace-scoped state (constant within cascade) | `file_edit`, `template`, `introspection` | NO-OP (workspace doesn't change within cascade) |

Phase 1 deliverable: deep audit of each plugin's `_self.*` attributes, marking which need reset_for_next_session implementation.

### 4.4 Per-cascade apparmor profile

**Profile naming shift**: `jaato-ws-{session_id}` → `jaato-cascade-{cascade_driver_id}`.

**Composition lifecycle**:
- First session of cascade: compose profile + load via apparmor_parser
- Subsequent sessions of same cascade: REUSE the loaded profile, no
  recompose
- Cascade end (after final teardown): unload profile

**Rule-shape implications**:
- Session-specific paths in rules (e.g. file_edit's
  `<config_root>/sessions/<session_id>/`) need to become cascade-specific or
  session-globbed:
  - Option A: `<config_root>/sessions/<cascade_id>/<session_id>/`
    rw, — cascade tier in path, session per subdirectory
  - Option B: `<config_root>/sessions/*/` rw, — wildcard session
    (less specific, broader grant within cascade)
- Plugin `get_apparmor_rules` classmethods receive `cascade_driver_id`
  in addition to `session_id` so they can choose the right shape.
- Lsp plugin's per-session rules (debug log path, jdtls data dir) become
  per-cascade rules (one set of grants per cascade, all sessions share).

### 4.5 Daemon-side IPC additions

New optional field on `session.new` request:
```python
@dataclass
class SessionNewRequest:
    name: Optional[str] = None
    profile: Optional[str] = None
    cascade_driver_id: Optional[str] = None  # NEW
    # ... existing fields
```

No new IPC verbs.  Cascade end is implicit (idle timeout).

---

## 5. Edge Cases

### 5.1 Cascade ID collision

Two clients independently generate the same UUID.  The second client's
session joins the first's cascade-slot.  Detection: PoolManager logs a
warning when an existing cascade_slot is claimed by a different client_id.

**Decision**: no collision detection / dedup.  UUID convention makes
this negligible in practice.  The warning is for operator visibility.

### 5.2 Partial cascade failure

A session within a cascade crashes (e.g. runner subprocess SIGKILL).
PoolManager detects via existing slot-watchdog mechanism.  Slot is torn
down, replenished.  Next session for that cascade_id falls into "no
matching idle slot" path → spawns fresh slot → pays cold-start tax for
that one session.  Acceptable.

### 5.3 Slot exhaustion under concurrent cascades

N cascades each request sessions; pool has M slots (M < N).

**Decision** (Daniel call): spawn fresh from pool.  This temporarily
exceeds the pool size.  PoolManager's replenishment thread refills the
pool to its target asynchronously.  No queuing, no blocking.

Risk: under heavy concurrent load, runaway slot creation.  Mitigation:
operator config `JAATO_RUNNER_POOL_MAX_OVERFLOW` (future knob, not in
Phase 1) caps the temporary excess.  Phase 0 defers this knob; ship
without limit + observe behavior.

### 5.4 Idle teardown firing during active session

Timer-based teardown could race with an in-flight session.

**Mitigation**: PoolManager's idle-sweep ONLY teardown slots in
`IDLE_FOR_CASCADE` state (not `ACTIVE_FOR_CASCADE`).  Active slots are
exempt by design.

### 5.5 Mixed-stack cascades

Cascade A is Java/Spring (jdtls); Cascade B is Python (pyright).
Slot reuse is per-cascade, so slots are naturally segregated by stack.
No cross-contamination.

### 5.6 Long-running cascade exceeding idle timeout

Cascade has gaps between stages (e.g. operator review pauses).  If a gap
exceeds `cascade_idle_timeout_seconds`, slot tears down.  Next session
spawns fresh → cold-start tax for one stage.

**Mitigation**: operator increases `cascade_idle_timeout_seconds` if their
cascade has known long gaps.  Default 300s (5 min) should cover typical
agent-driven gaps.  Cascade-aware drivers can send a keep-alive ping
(future IPC verb) if needed.  Phase 0 defers the ping verb; ship without
it.

---

## 6. Implementation Phases

### Phase 0: Design doc (this document)
- Lock decisions
- Plugin audit (cursory)
- Edge case enumeration
- Phase plan

### Phase 1: Plugin reset protocol + audit
**Scope**: ~800 LOC
- Add `reset_for_next_session()` method to `ToolPlugin` base class
  (default no-op)
- Deep audit of 30 runner-tier plugins:
  - Classify per the §4.3 table
  - For "holds per-session state" plugins: implement reset_for_next_session
- Pin tests: each plugin needing reset has a unit test confirming the
  reset clears the right state
- NO behavior change (slot reuse not yet active; reset method is just
  defined + tested)
- **Sign-off gate**: reset protocol is correct for every plugin BEFORE
  Phase 2 enables actual reuse

### Phase 2: Pool manager slot reuse mechanism
**Scope**: ~400 LOC
- Add `cascade_driver_id` field to `SlotHandle`
- Add `IDLE_FOR_CASCADE(C)` state to slot state machine
- `PoolManager.acquire_slot(cascade_driver_id)` accepts the ID
- Slot affinity routing: matching cascade_id wins, then IDLE, then fresh-spawn
- Add `PoolManager.return_slot_after_session(slot)` API
- Idle teardown sweep checks `cascade_idle_timeout_seconds`
- `spawn_session_runner` passes cascade_driver_id from envelope
- IPC: add `cascade_driver_id` to `session.new` request schema
- Subagent spawn propagation: child inherits parent's cascade_driver_id
- Pin tests: slot reuse fires only for matching cascade_id; cold-spawn
  for non-matching; idle teardown after timeout
- **Sign-off gate**: behavior validated end-to-end with a stub plugin
  that tracks reuse hits

### Phase 3: Per-cascade apparmor profile
**Scope**: ~500 LOC
- Profile name format: `jaato-cascade-{cascade_driver_id}`
- `get_apparmor_rules` signature gains `cascade_driver_id` kwarg
- Audit existing plugins for session-id-tied paths → migrate to
  cascade-id-tied OR session-globbed
- Profile load/unload lifecycle moves from session-end to cascade-end
- Profile composition runs ONCE per cascade
- Pin tests: first session of cascade composes + loads; subsequent
  sessions skip; cascade end unloads
- **Sign-off gate**: kernel apparmor logs show one load per cascade,
  one unload per cascade

### Phase 4: lsp plugin lifecycle adjustment
**Scope**: ~150 LOC
- `reset_for_next_session()`: NO-OP — keep `_connected_servers`,
  `_clients`, `_config_cache` intact
- `shutdown()`: still tears down on slot-teardown (cascade-end), full
  cleanup
- **Side effect**: most of PR-154/155/156/158's lsp-specific apparmor
  grants stay relevant (jdtls still lives runner-side under the
  per-cascade profile).  No major rewrite.
- Pin tests: `_connected_servers` survives `reset_for_next_session()`;
  cleared on `shutdown()`
- **Sign-off gate**: cascade-sharing demo shows ONE jdtls subprocess
  for an entire cascade (vs N today)

### Phase 5: Validation cascade + telemetry
**Scope**: ~300 LOC
- New `PoolManager.get_telemetry()` counters:
  - `cascade_slot_reuse_hits_total` (slot acquired for matching cascade_id)
  - `cascade_slot_reuse_misses_total` (no matching slot, spawned fresh)
  - `cascade_slots_idle_tornDown_total`
  - `cascade_idle_duration_seconds` histogram per slot
- Validation cascade: 6-stage kb-enablement-2.0 codegen pipeline
  - Expected: 1 slot, 1 jdtls process, 5+ enrichment markdown blocks
    (renders 2..N within cascade)
  - vs baseline (current): 6 slots, 6 jdtls processes, 0 enrichment markdown
- Acceptance: closes
  `project_backlog_lsp_server_persistence_across_sessions` + leaves
  positive telemetry on the cascade workload

---

## 7. Open Questions Resolved

All Phase 0 decisions are locked.  Phase 1 can begin without further input.

| Question | Locked answer |
|---|---|
| Tenant identifier | `cascade_driver_id` |
| Reuse scope | within ONE cascade only |
| Apparmor model | one profile per cascade, shared across sessions |
| Cascade ID propagation | IPC client supplies at first session, auto-inherits to subagents |
| Cascade end signaling | idle timeout (`cascade_idle_timeout_seconds`, default 300s) |
| Slot allocation when no match | spawn fresh from pool (no waiting) |

---

## 8. Backlog Cross-References

- [`project_backlog_pool_slot_reuse_for_cascades`](../../../../.claude/projects/-home-apanoia-Sources-Jaato-framework-and-examples-jaato/memory/project_backlog_pool_slot_reuse_for_cascades.md)
  — this design IS the implementation of that backlog item
- [`project_backlog_lsp_server_persistence_across_sessions`](../../../../.claude/projects/-home-apanoia-Sources-Jaato-framework-and-examples-jaato/memory/project_backlog_lsp_server_persistence_across_sessions.md)
  — closed as side effect when Phase 4 lands
- [`project_backlog_pool_reuse_plus_gated_init_interaction`](../../../../.claude/projects/-home-apanoia-Sources-Jaato-framework-and-examples-jaato/memory/project_backlog_pool_reuse_plus_gated_init_interaction.md)
  — the gated-init constraint Phase 1's reset protocol must honor
- 9-PR LSP arc (PR-150 through PR-158): closed all framework-addressable
  layers of the per-session LSP class.  The remaining barrier was always
  this cascade-sharing architecture.

---

## 9. Out of Scope (deferred)

- Cross-cascade slot reuse (workspace-tier reuse across cascade runs)
- Operator knob for `cascade_idle_timeout_seconds` (uses default 300s)
- Pool overflow cap (`JAATO_RUNNER_POOL_MAX_OVERFLOW`)
- Cascade keep-alive IPC ping verb (for known-long-gap cascades)
- Web client (telegram, etc.) cascade_driver_id flows — Phase 2 IPC change
  is the touchpoint; WS clients adopt as needed

---

## 10. Risks

- **Plugin reset protocol completeness**: missing a per-session state field
  in some plugin → next session sees stale state from previous session.
  Mitigation: Phase 1 audit + comprehensive pin tests.

- **Cascade-id propagation through subagents**: subagent code must explicitly
  read parent's cascade_id + pass it.  Easy to miss.  Mitigation: framework
  enforces in `runtime.create_session()` — subagent spawn API doesn't expose
  a cascade_id kwarg; it's always inherited.

- **Apparmor reload semantics**: kernel apparmor_parser behavior under
  same-profile-name reload differs from first-load.  Mitigation: Phase 3
  uses `apparmor_parser --replace` for the per-cascade profile;
  validate on Ubuntu 22.04 + 24.04 + AppArmor 3.x/4.x.

- **The reset protocol is itself the source of new bugs**: a plugin's
  `reset_for_next_session()` could miss state OR clear too much.
  Mitigation: per-plugin pin tests confirm "state X is cleared, state Y
  survives" with specific assertions.

---

End of Phase 0 design document.  Awaiting sign-off to begin Phase 1.
