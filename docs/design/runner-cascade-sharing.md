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

3. **Apparmor**: per-session profiles preserved; slot transitions
   between profiles via `aa_change_profile` on each session boundary.
   ~~Original decision (withdrawn 2026-05-20 — see Amendment below): ONE
   profile per cascade, shared across all sessions of that cascade.~~
   Profile naming stays `jaato-ws-{session_id}` (NOT `jaato-cascade-{id}`).
   See §4.4 + the Amendment at the head of §4 for the revised model.

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

### Amendment 2026-05-20 — per-session apparmor preserved

The original §3 decision 3 ("one apparmor profile per cascade") assumed
all sessions of a cascade share the same apparmor requirements.  In
practice this is wrong: a multi-stage cascade uses DIFFERENT profiles
per stage (discovery agent, codegen agent, build agent, ...), each
with its own plugin set, `apparmor_fragments` override, and
plugin-contributed `get_apparmor_rules` output.  A single per-cascade
profile is either too narrow (session N+1 lacks rules N+2 needed) or
too broad (UNION-of-all-stages widens the security boundary
monotonically).

**Revised model (Phase 3 Shape ε, sign-off 2026-05-20):**

- Per-session apparmor profile preserved: each session keeps its own
  `jaato-ws-{session_id}` profile, freshly composed against its
  profile's plugin set + fragments.
- Slot transitions via `aa_change_profile` on each
  `session.bootstrap` (initial AND re-entry for subsequent sessions
  of the same cascade-reused slot).  Today bootstrap step 1c calls
  `aa_change_profile` exactly once; the new path detects re-entry
  + transitions to the next session's profile.
- Per-session profile template carries a
  `change_profile -> jaato-ws-*,` rule so the runner's main thread
  can transition between any two `jaato-ws-*` profiles.  All
  `jaato-ws-*` profiles are framework-composed or operator-authored
  — there's no untrusted profile in the transition space.
- Old session's profile is unloaded via `apparmor_parser --remove`
  after the change_profile succeeds (or at cascade-idle teardown if
  the slot tears down without a successor).

§4.4 below describes the lifecycle in detail.  §6 Phase 3 carries
the revised implementation scope (~250 LoC vs the original ~500
LoC — no per-cascade composition logic, no plugin
`get_apparmor_rules` signature change).

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

**Litmus test for the categorization** (Daniel rule 2026-05-20):

> **A plugin's state should survive `reset_for_next_session()` if a
> subsequent session within the SAME cascade might benefit from it.**

This rule prevents the framework from silently discarding context that
the next cascade stage was meant to consume.  Examples:
- `memory` — model writes memories in session A; the next cascade stage
  reads them.  Survive.
- `artifact_tracker` — artifacts produced by session A become inputs for
  session B's validation / dependency tracking.  Survive.
- `todo` — each cascade stage typically has its own task focus; carrying
  todos across stages confuses the agent.  Reset.

**Audit categories** (cursory — Phase 1 deepens by applying the litmus
test to every `_self.*` attribute of every plugin):

| Category | Plugins | Reset needed? |
|---|---|---|
| Stateless / warm-import-only | `cli`, `web_fetch`, `web_search`, `filesystem_query`, `calculator`, `multimodal`, `ast_search`, `bundle`, `environment`, `notebook`, `prompt_library`, `references`, `reliability`, `service_connector`, `webhook` | NO-OP |
| Holds per-session state that next session does NOT benefit from | `session` (conversation history), `todo` (per-stage tasks), `clarification` (per-session Q&A), `thinking` (per-turn ephemeral), `permission` (per-session approvals — see Phase 1 review) | YES — needs explicit clear |
| Holds across-session state by design (cascade-sharing PR target — next session BENEFITS from carry-over per Daniel's litmus test) | `lsp` (LSP server connections), `mcp` (MCP server connections + tool catalogs), `interactive_shell` (PTY session map — Phase 1 review whether SHOULD survive), `sandbox_manager`, **`memory`** (cross-session model memories), **`artifact_tracker`** (cross-session artifact dependency graph) | NO-OP (state SHOULD survive) |
| Holds workspace-scoped state (constant within cascade) | `file_edit`, `template`, `introspection` | NO-OP (workspace doesn't change within cascade) |
| **Phase 1 review required** — could go either way depending on workload | `subagent` (subagent registry — cross-session reuse?), `waypoint` (cross-session debug/replay?), `permission` (session-scoped vs cascade-scoped approvals?) | TBD per Phase 1 audit |

**Daniel's correction (2026-05-20)**: `memory` and `artifact_tracker` were
initially in the "reset needed" bucket — moved to "across-session by
design" because subsequent cascade stages clearly benefit from prior
stages' memories and tracked artifacts.  The litmus test surfaced via
this correction generalises: apply it to every plugin in Phase 1 audit.

Phase 1 deliverable: deep audit of each plugin's `_self.*` attributes,
marking which need reset_for_next_session implementation.  Per-attribute
litmus-test application; "TBD" plugins above resolved by attribute-level
classification, not whole-plugin categorisation.

### 4.4 Per-session apparmor with cross-session transitions

(Revised 2026-05-20 — see Amendment at head of §4.)

**Profile naming**: stays `jaato-ws-{session_id}`.  Each session has
its own apparmor profile, composed against that session's profile's
plugins + fragments — identical to the pre-cascade-sharing model.
No `get_apparmor_rules` signature change.

**Slot lifecycle vs apparmor profile lifecycle:**

```
slot states              apparmor state
─────────────────        ──────────────────────────────
[fresh from template]    unconfined
                          │
                          ▼
session 1 acquires       compose jaato-ws-S1; load via apparmor_parser
                          │
slot.bootstrap_session   step 1c: aa_change_profile(jaato-ws-S1)
                          │ runner main thread now confined to S1
session 1 runs           │
                          │
session 1 ends           runner stays under jaato-ws-S1
slot returns to pool     (transition deferred until session 2 acquires)
                          │
session 2 acquires       compose jaato-ws-S2; load via apparmor_parser
                          │
slot.bootstrap_session   step 1c: aa_change_profile(jaato-ws-S2)
                          │ — allowed because S1's profile has
                          │   `change_profile -> jaato-ws-*,`
                          │
                          ▼
                         apparmor_parser --remove jaato-ws-S1
                         (kernel unloads the old profile; no longer
                         referenced by any running process)
```

**Per-session profile template addition:**

Every `jaato-ws-{session_id}` profile carries a single new rule in
the runner-main scope:

```
change_profile -> jaato-ws-*,
```

This permits the runner's main thread to transition between any two
`jaato-ws-*` profiles.  The transition space is closed — all profiles
matching the pattern are framework-composed (no operator-untrusted
profile in scope).  The `//child` sub-profile (the LLM-driven scope)
does NOT inherit this rule: the LLM cannot trigger `change_profile`.

**Re-entry detection in bootstrap step 1c:**

The runner-side `session.bootstrap` handler today calls
`aa_change_profile(envelope.profile_name)` exactly once on first
bootstrap.  Phase 3 modifies it to detect re-entry (a subsequent
session of the same slot):

```python
# bootstrap_session step 1c (Phase 3 revised)
if self._current_apparmor_profile is None:
    # First bootstrap — transition from unconfined to S1.
    aa_change_profile(envelope.profile_name)
    self._current_apparmor_profile = envelope.profile_name
elif self._current_apparmor_profile != envelope.profile_name:
    # Re-entry — transition from S(N) to S(N+1).
    aa_change_profile(envelope.profile_name)
    old = self._current_apparmor_profile
    self._current_apparmor_profile = envelope.profile_name
    # Schedule kernel unload of `old` once we know no other thread
    # is in it (today: runner is single-process so this is safe to
    # request immediately; tomorrow: §4.x will revisit if //child
    # sub-profile transitions need coordination).
# else: same profile (re-bootstrap with identical session_id —
# shouldn't happen in cascade reuse since session_ids are unique).
```

**Unload-old-profile lifecycle:**

After `aa_change_profile` succeeds:

1. Slot returns control to daemon (bootstrap completes).
2. Daemon's `apparmor` helper calls `apparmor_parser --remove
   jaato-ws-{old_session_id}` to free the kernel slot.
3. If the unload fails (kernel busy, profile still referenced), log
   a warning and continue.  The kernel will eventually evict stale
   profiles; the unload is best-effort cleanup.

This unload is daemon-side because `apparmor_parser` is unconfined
+ requires `CAP_MAC_ADMIN` — runner-side execution would fail.

**Existing per-session security model preserved:**

The §4.4 revision does NOT change:

- Per-session profile composition (today's `_provision_apparmor_for_session`)
- Plugin `get_apparmor_rules` signatures or contributions
- The `//child` sub-profile model (LLM-driven tool exec under tightening)
- The `tool_hat` sub-profile (runner main during tool-exec)

What changes:

- A new template rule (`change_profile -> jaato-ws-*,`) in the
  per-session profile's main scope.
- Bootstrap step 1c gains re-entry detection.
- Daemon's session-teardown path adds an old-profile unload step
  (after change_profile succeeds for session N+1).

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

### Phase 3: Per-session apparmor with cross-session transitions

(Revised 2026-05-20 — see Amendment at head of §4 + the revised §4.4.)

**Scope**: ~250 LOC (down from the original ~500 LOC — no
per-cascade composition logic, no plugin `get_apparmor_rules`
signature change).

- Per-session profile template gains
  `change_profile -> jaato-ws-*,` rule in the runner-main scope.
- Runner-side `session.bootstrap` step 1c gains re-entry detection
  + `aa_change_profile` for subsequent sessions of the same slot.
- Daemon-side: after `aa_change_profile` succeeds for session N+1,
  call `apparmor_parser --remove jaato-ws-{old_session_id}` to free
  the kernel slot.  Best-effort (kernel may report busy if
  references linger; log + continue).
- No plugin `get_apparmor_rules` signature change — per-session
  profiles continue to receive `session_id` only.  Plugins do NOT
  need cascade-awareness.
- Pin tests: rendered profile carries the change_profile rule;
  bootstrap step 1c re-entry path fires aa_change_profile + logs
  the transition; --remove called daemon-side after successful
  transition.
- **Sign-off gate**: kernel apparmor audit logs show clean
  PROFILE_LOAD on each new session AND clean PROFILE_REMOVE on
  the prior session.  Cascade-sharing demo (kb-enablement-2.0
  workload) runs cleanly with N profiles loaded + N-1 removed
  across a single slot's lifetime.

**What this does NOT change** (preserved):

- Per-session profile composition (today's `_provision_apparmor_for_session`)
- Plugin `get_apparmor_rules` signatures or contributions
- The `//child` sub-profile model (LLM-driven tool-exec scope)
- The `tool_hat` sub-profile (runner main during tool-exec)

### Phase 4: lsp plugin lifecycle adjustment
**Scope**: ~150 LOC
- `reset_for_next_session()`: NO-OP — keep `_connected_servers`,
  `_clients`, `_config_cache` intact
- `shutdown()`: still tears down on slot-teardown (cascade-end), full
  cleanup
- **Side effect**: PR-154/155/156/158's lsp-specific apparmor
  grants stay relevant (jdtls lives runner-side under the FIRST
  session's `jaato-ws-{session_id}` profile; child processes
  inherit that profile when forked, and `aa_change_profile` on
  the parent runner thread does NOT propagate to existing
  children — jdtls keeps the rules it was spawned with).  No
  rewrite needed.
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
| Apparmor model | per-session profiles preserved; slot transitions via `aa_change_profile` on each session boundary (revised 2026-05-20 — see §4 Amendment) |
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

- **Apparmor change_profile rejection**: kernel may reject the
  `aa_change_profile` transition if the calling thread's current
  profile doesn't permit it (rule absent OR malformed).  Mitigation:
  Phase 3 adds `change_profile -> jaato-ws-*,` to the per-session
  template; pin tests verify the rule is present in every rendered
  profile.  Validate on Ubuntu 22.04 + 24.04 + AppArmor 3.x/4.x.

- **Old-profile reference linger**: `apparmor_parser --remove` may
  return EBUSY if any process (or future thread) still references the
  profile.  Mitigation: best-effort unload; log + continue.  Kernel
  eventually evicts stale profiles via LRU.

- **The reset protocol is itself the source of new bugs**: a plugin's
  `reset_for_next_session()` could miss state OR clear too much.
  Mitigation: per-plugin pin tests confirm "state X is cleared, state Y
  survives" with specific assertions.

---

End of Phase 0 design document.  Awaiting sign-off to begin Phase 1.
