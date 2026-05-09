# Phase 3 §3.3c — Session RPC Dispatch Surface

Status: **dispatch surface complete; daemon-shell rewrite (the actual seat-flip) is the remaining work.**

This doc inventories the daemon→runner RPC handlers built up across
the §3.3c precursor commits, plus the daemon-side wrapper API +
migration call sites the seat-flip will progressively migrate.

## 1. Why the surface exists

The §3.3c seat-flip removes the daemon-side `JaatoSession` instance
from `JaatoServer` (`self._jaato`) and dispatches every session-
level operation to the runner-side `JaatoSession` via runner-RPC
instead.  That migration touches ~95 daemon-side `self._jaato.X`
call sites in `core.py` alone.  The §3.3c plan calls this **one
commit** but it's realistically multi-PR work.

The precursor surface lets each daemon-side migration be a local
change against a frozen RPC contract — the wire shape, error
envelopes, and edge cases are pinned by ~85+ tests before the
first migration touches `self._jaato`.

## 2. Runner-side handlers

All handlers live in `jaato-server/server/runner/rpc.py`,
dispatched from `RunnerRPC._dispatch_method`.  Each follows the
same shape:

- Args decoded from `env.args` (a JSON-friendly dict).
- Common precondition via `_require_ready_session()` returning
  clean `stage="no_host"` / `stage="no_session"` errors when the
  runner isn't bootstrapped or the session is None.
- Method-specific arg validation surfacing as `stage="decode"`
  errors.
- Underlying call wrapped in try/except returning
  `stage="<read|set|write>"` errors on failure (defensive: the
  probe must never crash the runner process).
- Success returns `(True, <result_dict>)`.

| Method | Direction | Body | Wrapper |
|---|---|---|---|
| `session.health_check` | read | `{has_host, ready, session_id, tool_count}` | `session_health_check()` |
| `session.get_session_state` | read | `{value: Any}` | `session_get_state(key, default)` |
| `session.set_session_state` | write | `{ok: True}` | `session_set_state(key, value)` |
| `session.get_all_session_state` | read | `{state: dict}` | `session_get_all_state()` |
| `session.is_running` | read | `{running: bool}` | `session_is_running()` |
| `session.request_stop` | write | `{cancelled: bool}` | `session_request_stop(reason)` |
| `session.get_history` | read | `{history: [dict, ...]}` | `session_get_history(raw=False)` |
| `session.get_context_usage` | read | `{usage: dict}` | `session_get_context_usage()` |
| `session.get_turn_accounting` | read | `{turns: [dict, ...]}` | `session_get_turn_accounting()` |
| `session.set_terminal_width` | write | `{ok: True}` | `session_set_terminal_width(width)` |
| `session.set_streaming_enabled` | write | `{ok: True}` | `session_set_streaming_enabled(enabled)` |
| `session.set_presentation_context` | write | `{ok: True}` | `session_set_presentation_context(ctx)` |
| `session.reset` | write | `{ok: True}` | `session_reset()` |
| `session.shutdown` | write | `{shutdown_session_id: str}` | `session_shutdown()` |

Each wrapper has a `_threadsafe` variant for synchronous worker-
thread callers (mirroring the `bootstrap_session_threadsafe`
template).

## 3. Daemon-side wrapper contract

All wrappers live in
`jaato-server/server/runner_rpc_client.py:RunnerRPCClient`.

- Async + threadsafe variants per method.
- Uses `_call_named(method, args, timeout)` internal helper
  (mirrors the `bootstrap_session` template).
- Raises `RunnerCallError` on transport failure or
  protocol-level error (`response.ok=False`).
- Returns the unwrapped value (bool / dict / list / str / None)
  rather than the raw result envelope — daemon-side callers get
  a typed Python API.

## 4. Defensive contracts (cross-handler)

| Concern | Behavior |
|---|---|
| Runner not bootstrapped (no host) | `stage="no_host"` error envelope — daemon distinguishes from "session present" |
| Host bootstrapped but session=None (test-stub mode / configure failed) | `stage="no_session"` error |
| Underlying setter / reader raises | `stage="set"` / `stage="read"` error envelope; runner process never crashes |
| Non-dict / non-list / non-bool from custom session subclass | `stage="read"` error rather than letting JSON encoder choke |
| Idempotent re-call (e.g. shutdown twice) | success no-op, not error |
| Per-message serialization failure in `get_history` | placeholder substituted; count preserved; the buggy message doesn't drop the whole history |

## 5. Test coverage

~85+ tests across:

- **Per-handler unit tests** — `test_session_<area>_rpc.py` files;
  exercise direct `_handle_session_X` calls + dispatch routing.
- **Daemon-side wrapper e2e tests** —
  `test_session_method_wrappers_e2e.py`; drive each wrapper over
  a real `socketpair()` connecting `RunnerRPC` to
  `RunnerRPCClient`.
- **Lifecycle composition test** —
  `test_session_dispatch_lifecycle_e2e.py`; 13-step session arc
  proves handlers compose correctly across a realistic
  bootstrap → state ops → config → shutdown sequence.

## 6. Daemon-side migrations done (vanguard)

| Daemon site | Old behavior | New behavior |
|---|---|---|
| `JaatoServer.shutdown` | closes RPC transport directly (SIGTERM races plugin teardown) | calls `runner_rpc.session_shutdown_threadsafe(timeout=5)` first; then transport close.  Best-effort: failures log + proceed |
| `JaatoServer.terminal_width` setter | propagates only to in-process `_jaato` + formatter pipelines | also forwards via `runner_rpc.session_set_terminal_width_threadsafe(width, timeout=2)` |
| `JaatoServer.set_presentation_context` | propagates only to in-process `_jaato` | also forwards via `runner_rpc.session_set_presentation_context_threadsafe(ctx, timeout=2)` |

These three are the seat-flip's **vanguard** — proof the dispatch
pattern works end-to-end against real call sites, not exhaustive.
The remaining ~47 `self._jaato.X` sites in `core.py` (50 total —
3 vanguard) follow the same shape per bucket §7b below.

All three follow the same pattern:

1. Update daemon-side state (unchanged).
2. If runner attached, forward via the matching wrapper with a
   short timeout.
3. Wrap the forward in try/except logging at DEBUG — failures
   never block the daemon-side flow.
4. Use `getattr` + `callable` to skip gracefully when the
   wrapper method is absent (forward-compat with rolling
   upgrades).

## 7. Remaining work (the seat-flip itself)

The dispatch surface is comprehensive enough that the seat-flip
can proceed handler-by-handler.

**Recommended bucket order** (per peer-review v2 + v3 §7b.3
withdrawal):

```
§7a (always-spawn)                                 ← shipped (55ae4ba0 / 20526f4d)
  → §7b.1 (stateless setters / readers)            ← shipped (3 vanguard + 3 iteration:
                                                       8cbb8ba2 + fafe90a6 + 7f8be0cb;
                                                       remaining "NOW" sites re-bucketed
                                                       to DEFER-§7c per §10 wrap-up table)
    → §7b.2 (session.send_message)                 ← shipped (3ca3c14d)
      → §7c (remove in-process JaatoSession + flag,
             absorbs §7b.3 response-handler rewiring
             + introspection collapse + DEFER-§7c
             read-site migrations)
        → §7d (cgroup attach migration; depends on
               7a's stable cgroup placement)
```

**§7b.3 was withdrawn after empirical investigation** (commit
`fafe90a6`'s ahead-of-time §7b.3 audit).  The proposed scope
(4 new `session.respond_to_*` RPCs + a `session_get_provider_info`
introspection RPC) duplicated existing infrastructure or
front-ran architecture not yet built.  See §7b.3 below for the
detailed reasoning; the short version: response-handler wiring
already lives in `PromptOperatorHandler.resolve_response()`,
and provider/model introspection reads daemon-side `JaatoRuntime`
directly per §4.2 (model_provider plugins are daemon-tier).
Both sets of sites collapse during §7c without new RPCs.

§7b.1 + §7b.2 share a common dependency on §7a (always-spawn) —
until the runner is unconditionally available, the daemon-side
migrations would need a fallback shim.  Order within 7b is
bottom-up by complexity.

Daemon-side migration scope (the canonical count, replacing the
original "~95"):

| File | `self._jaato.X` access count | Notes |
|---|---|---|
| `server/core.py` | 50 | Bulk of `_jaato.send_message`, `respond_to_*`, state setters / getters; the seat-flip's main migration target. |
| `server/websocket.py` | 18 (named `_jaato_server.X`) | WS-standalone direct API.  Same patterns as core but without the daemon-event-emit glue. |
| `server/command_router.py` | 1 | `session.server._jaato.get_runtime` — single fork-ask call site. |

Total: **~69 daemon-side dispatch sites**.  Reading wider call
sites (`server/__main__.py`, plugin-extension hooks) adds ~5 more.

### 7a. Always-spawn the runner

**Status: design pending.**

Today the runner spawns only when the IPC client opts into
apparmor (`SessionManager._provision_ipc_apparmor_and_spawn_runner`).
For the seat-flip to work, every IPC + WS-standalone session
needs a runner so daemon-side `_jaato.X` calls have a dispatch
target.

Open design questions (peer-review M1):

1. **Profile schema.** Today profiles can opt-in to apparmor
   independently.  After 7a, every IPC + WS-standalone session
   has a runner regardless of the apparmor flag.  Two modes the
   doc must pick:
   - **Mandatory** (chosen): runner always spawned for IPC/WS.
     Apparmor becomes a confinement-mode flag inside the runner-
     spawn path.  Eliminates the `spawn_runner: bool` knob.  Per
     peer-review v2 confirmation: "a `spawn_runner: bool` knob
     would just defer the seat-flip's complexity into a permanent
     config-mode bifurcation.  Mandatory = single posture =
     simpler maintenance."
   - **Configurable** (rejected): new `spawn_runner: bool`
     profile field independent of `apparmor`.  Adds a feature
     surface that must stay maintained.  Rejected.
2. **Performance.** Subprocess spawn + RPC handshake + plugin
   discovery is non-trivial latency at session-create time.
   Daemons hosting many short-lived sessions (CI test harnesses,
   serverless adapters) feel this most.  Mitigation: amortize
   plugin-discovery cost across pre-spawned warm runners (a pool)
   — defer to Phase 4+ if measured baseline is acceptable.
3. **Test fixtures.** A large fraction of existing tests don't
   have a runner attached.  After 7a they all do — fixture
   surface changes significantly.  Some integration tests may
   break or need re-baseline.  Specifically:
   - All `_FakeJaatoServer` based tests (currently ~40 in the
     server suite) gain a runner side; either the fake exposes
     an in-process JaatoSession-equivalent, or the test
     constructs a real `socketpair`-driven runner stub.
   - The §3.3c precursor tests already use the
     `_FakeSession` + `RunnerSessionHost` pattern that survives
     the seat-flip; those tests stay green.

Files touched:
- `server/session_manager.py` — split
  `_provision_ipc_apparmor_and_spawn_runner` into
  `_spawn_session_runner_unconditional` (always called) +
  `_provision_apparmor_for_session` (opt-in only).  The IPC
  bootstrap path always invokes the spawn helper; apparmor is
  layered atop iff the client opted in.
- `server/runner_spawn.py` — already independent; no changes
  needed but verify the helper handles the no-apparmor path
  cleanly (the runner's `aa_change_profile` becomes a no-op
  when no profile is attached — should fall through to the
  `JAATO_RUNNER_DISABLE_CONFINE=1` mode, which Phase 2's
  `runner/__main__.py` already supports).
- `server/websocket.py` — apply the same split for the WS-
  standalone bootstrap.

Tests:
- New `tests/integration/test_runner_always_spawned.py` —
  exercises an IPC `session.new` without apparmor opt-in;
  asserts `server.runner_rpc is not None` post-init.
- Existing apparmor-opt-in path tests still pass (apparmor is
  layered, not gated).
- A performance-baseline test recording session-create p95
  latency — ratchet to detect regressions after 7a lands.

One commit; lands BEFORE 7b.

### 7b. Migrate remaining daemon-side dispatch sites

Split into three sub-buckets per peer-review M2 — each has
comparable scope to §3.7 and warrants its own commit.

#### 7b.1. Stateless setters / readers (the easy bulk) — **shipped**

**Status: shipped (6 commits).**  Final scope smaller than the
original ~10-site estimate after the §10 audit appendix's
NOW-bucket re-classification (5 read-side sites deferred to §7c).

Patterns the existing dispatch surface covers:

- `_jaato.set_*` → `runner_rpc.session_set_*_threadsafe`
- `_jaato.get_session().get_session_state(...)` → `runner_rpc.session_get_state`
- `_jaato.is_processing` → `runner_rpc.session_is_running`
- `_jaato.stop()` → `runner_rpc.session_request_stop`
- `_jaato.get_history()` → `runner_rpc.session_get_history`
- `_jaato.get_context_usage()` → `runner_rpc.session_get_context_usage`

Each migration followed the vanguard write-both pattern (call
daemon-side AND forward to runner during transition); no new RPC
handlers required.

Shipped commits (chronological):

| Commit | Site | Method |
|---|---|---|
| `45a2dbd8` | `JaatoServer.shutdown` | `session.shutdown` |
| `b2c0772d` | `JaatoServer.terminal_width` setter | `session.set_terminal_width` |
| `c3d5ec08` | `JaatoServer.set_presentation_context` | `session.set_presentation_context` |
| `8cbb8ba2` | `JaatoServer.clear_history` | `session.reset` |
| `fafe90a6` | `JaatoServer.stop` | `session.request_stop` |
| `7f8be0cb` | `initialize()` post-init terminal_width sync | `session.set_terminal_width` (direct call, bypasses property) |

Five originally-NOW READ sites (context_usage / context_limit at
init + auth-completion) plus `emit_current_state`'s
`instruction_budget` read fold into §7c — see §10 wrap-up table
for the per-site re-bucketing rationale.

#### 7b.2. `session.send_message` (the big one)

The biggest single handler.  Open design questions:

1. **Streaming/multi-turn pattern.** The existing `tool.execute`
   handler streams output via `_make_on_output(request_id)`
   bridging into the daemon's stream-frame channel.
   `session.send_message` follows the same pattern but with a
   larger per-call surface:
   - Model API request (provider plugin invocation runs daemon-
     tier per §4.2; the runner-side session hands off to
     daemon-side runtime via `client.complete` RPC — yet another
     primitive).
   - Plugin enrichment pipeline: prompt enrichment, system-
     instruction enrichment, response enrichment.  All run
     runner-side post-seat-flip.
   - Function-calling loop (multi-turn until model stops calling
     tools).  Each turn round-trips through the model API
     daemon-side.
   - Permission ASK during tool execution: relays through the
     existing `client.prompt_operator` primitive (§3.2.1), not a
     new channel.  The §3.7 `RunnerRPCChannel` is the same
     plumbing, *NOT* a sibling.
2. **Cancellation.** `tool.execute` handles cancellation via the
   `cancel` frame routed by `RunnerRPC._handle_cancel`.
   `session.send_message`'s cancel propagates through the same
   mechanism: the runner-side session checks the per-call
   cancel token at each turn boundary.
3. **Wire shape.** Args = `{"prompt": str, "agent_id": str?, ...}`.
   Streams chunks via the stream-frame channel.  Response = the
   final model response dict (matches today's
   `JaatoSession.send_message` return).

Files touched:
- `server/runner/rpc.py` — add `session.send_message` handler;
  reuses the streaming + cancel infrastructure already proven
  for `tool.execute`.
- `server/runner_rpc_client.py` — async wrapper +
  `session_send_message_threadsafe` (note: long-running, so
  `timeout=None` default with explicit caller-side cancellation
  via the cancel frame).
- `server/core.py` — migrate `JaatoServer.send_message` to call
  the wrapper.  This is the LAST big migration before the
  seat-flip's final flag-removal commit.

Tests: integration test driving a real provider stub through the
RPC; cancellation mid-stream; multi-turn with permission ASK.

One commit; depends on 7a.

#### 7b.3. Response handlers + `get_runtime` — **WITHDRAWN**

**Status: withdrawn.** Empirical investigation (worker commit
`fafe90a6`'s pre-implementation audit; reviewer-confirmed) found
that the originally-scoped work either duplicates existing
infrastructure or front-runs architecture that doesn't need to
be built.  The audit-of-record:

##### Response-handler half

The original framing assumed `_jaato.respond_to_permission(...)` /
`respond_to_clarification(...)` / `respond_to_clarification_batch(...)` /
`respond_to_reference_selection(...)` were internal `_jaato.X` call
sites needing forwarding to the runner.  They are not.

Tracing the actual call sites:

```
WS client → server/websocket.py:1387-1406
              → JaatoServer.respond_to_X(request_id, response, ...)
                  → daemon-side response queue

IPC client → server/session_manager.py:4146-4158
              → JaatoServer.respond_to_X(...)
                  → daemon-side response queue
```

`respond_to_X` are **client-facing entry points on JaatoServer**,
not internal `_jaato.X` consumers.  They flow client responses
INTO the daemon's response-queue infrastructure.

The runner-side ASK round-trip already has a working primitive:
`server/runner_rpc_handlers/prompt_operator.py:PromptOperatorHandler`
(class at line 58) — the runner emits ASK via `client.prompt_operator`
RPC, awaits the response as the RPC return value, and the daemon's
transport layer calls `PromptOperatorHandler.resolve_response()`
(line 162) to set the matching future when the client's response
arrives.  Future-keyed-by-request-id, complete + tested.

Adding `session.respond_to_*` RPC handlers would create a parallel
daemon→runner pathway that **duplicates** what `PromptOperatorHandler`
already does.  The **actual** remaining work is wiring the
daemon-side `respond_to_X` methods to call into
`PromptOperatorHandler.resolve_response()` — which only makes sense
once the runner-side permission plugin is the source of truth for
ASK state.  That gating is **§7c**, not §7b.3.

##### `get_runtime` half

Audit of `_jaato.get_runtime` call sites in `core.py` (verified
post-§7b.1-shipped state):

| Line | Use | Migration path |
|---|---|---|
| 637 | `runtime.set_confine_context_factory(...)` | **Stays daemon-side** — init-time apparmor wiring, daemon-tier per §4.7 |
| 996 | `runtime.event_bus` | **Stays daemon-side** — event_bus is daemon-tier per §4.2 |
| 1465 | `runtime.set_confine_context_factory(...)` | **Stays daemon-side** — same as 637 |
| 1810 | `runtime.event_bus` | **Stays daemon-side** — same as 996 |
| 3575 | `runtime.registry` walk for command lookup | **Refactor in §7c** — registry split decisions live there |
| 3607 | `runtime.registry` walk for command lookup | **Refactor in §7c** — same as 3575 |

All `get_runtime` call sites are either daemon-tier wiring (the 4
init-time / event_bus cases) or registry walks that refactor as
part of §7c's `_jaato`-removal pass.  None need a new RPC.

The `session_get_provider_info` introspection handler proposed in
the original §7b.3 was supposed to cover sparse read-only sites
like `_jaato.provider_name` / `_jaato.model_name` / `_jaato.is_connected` /
`_jaato.auth_info`.  But these properties read from the
`JaatoRuntime` instance, which **stays daemon-side per §4.2**
(model_provider plugins are daemon-tier).  Post-§7c, the daemon
reads `self._runtime.provider_name` directly without going
through the runner — same parallel-pathway problem the response-
handler RPCs would have created.

##### What §7b.3 leaves to §7c

Two threads roll forward into §7c:

1. **Response-handler wiring**: when §7c activates the runner-side
   permission plugin as authoritative, `respond_to_X` on
   `JaatoServer` is rewired to call
   `prompt_operator_handler.resolve_response(request_id, ...)`
   (already-existing primitive) instead of the daemon-side
   response queue.  Small refactor inside §7c, not a new RPC.

2. **Introspection collapse**: when §7c removes the `_jaato` field,
   the read-only callers (`provider_name`, `model_name`,
   `is_connected`, `auth_info`) become direct daemon-side
   `JaatoRuntime` reads.  Zero RPC, zero handler, zero wrapper.

##### Files touched (this withdrawal)

- This doc — §7b.3 marked WITHDRAWN with the audit-of-record above.
- `server/runner/rpc.py` — no new handlers added for §7b.3.
- `server/runner_rpc_client.py` — no new wrappers added for §7b.3.
- `server/core.py` — no new migrations for §7b.3; affected sites
  collapse during §7c.

##### Commit budget

Original §7b.3: 2 commits (response handlers + runtime primitive).
Revised §7b.3: **0 commits**.  §7c absorbs the response-handler
rewiring + introspection collapse.

### 7c. Remove the in-process `JaatoSession`

After all 7b migrations land, the `_jaato` field can be removed
from `JaatoServer.__init__`.  The runner-side host becomes the
single source of truth.

**`JAATO_RUNNER_HOSTS_SESSION` flag lifetime** (peer-review M4):
the v5 plan §3.3b N4 specified the flag "lands and is removed
within the same PR."  As implemented, the flag has been live
across ~24 §3.3c precursor commits.  This is a **scope expansion
from v5 N4** — the design has shifted from "single-PR
transitional flag" to "multi-PR transitional flag with explicit
removal commit."  Both readings are defensible; this doc picks
the latter:

- The flag is a **multi-PR transitional shape**.
- It exists for the duration of §3.3c precursor → §7c rollout.
- Removed in §7c's final commit (alongside the
  `_jaato`-field removal from JaatoServer).
- Operators never see the flag on a released server version
  (still upholds N4's intent — the user-facing concern was
  preventing operator-visible feature-flag accumulation, which
  this doesn't violate).

Files touched:
- `server/core.py` — remove `self._jaato` field + all None-
  guarded fallbacks (the `if self._jaato:` checks become
  unconditional `if self._runner_rpc:` checks).
- `shared/jaato_runtime.py` — `create_session` no longer
  instantiates `JaatoSession`; builds envelope, dispatches to
  runner, returns runner-RPC handle.
- `server/__main__.py` — remove `JAATO_RUNNER_HOSTS_SESSION`
  env-var read + the conditional bootstrap-envelope dispatch.
- `server/runner/__main__.py` — flag check goes away;
  runner-side JaatoSession host is unconditional.

One commit (logically — the §7c change-set absorbs the
response-handler rewiring + introspection collapse withdrawn from
§7b.3, so the actual landing may be 2-3 reviewable commits).
Depends on 7a + 7b.1 + 7b.2.

### 7d. Dependent migrations

- **§3.11 default-share + isolation knob**: ephemeral subagents
  share the parent's runner via `BootstrapEnvelope.parent_runner_handle`.
  Unblocks once 7a lands (parent has a runner unconditionally).
- **§3.12 ASK queue + drain**: runner-side permission plugin
  buffers ASK prompts when no client is attached
  (`Session.restored_pending_attach`); flushes on attach via
  the §7b.3 response handlers.
- **Cgroup attach migration**: per peer-review M3 — currently
  daemon-side via `shared/ai_tool_runner.py:_cgroup_attach`
  (verified at line 211).  §3.5 (commit 03a5166d) migrated
  subprocess-spawning plugins to forward via runner-RPC but
  **did NOT migrate the cgroup-attach mechanism itself** — the
  daemon-side `set_runtime_limits` callback still threads
  through.  Runner-side cgroup attach lands in §7d as a
  follow-on:
  - Files: `server/runner/cli_runner.py` gains
    `_cgroup_attach_to_session_cgroup()` invoked at Popen time;
    the runner subprocess is itself in the cgroup so child
    processes inherit by default — `_cgroup_attach` becomes
    an inherit-check rather than an explicit move.  This is
    materially simpler than today's per-process explicit
    migrate (per peer-review v2 observation #2).
  - Test: existing `test_runtime_limits_e2e.py:312/354` gates
    the migration via `_can_migrate_to(_find_writable_cgroup_parent())`.
  - **Additional test required** (per peer-review v2 observation
    #2): grandchild cgroup inheritance — a process spawned by a
    runner-tier plugin (e.g., `interactive_shell`'s pexpect-spawned
    PTY child, or `cli`'s subprocess child) MUST inherit the
    runner's cgroup placement.  Inheritance vs explicit-move is
    the stress case for this approach: today's per-process
    `_cgroup_attach` callback runs once per Popen and is
    explicit; inheritance relies on the kernel's default child-
    inherits-parent behavior, which is correct under cgroup v2
    but worth pinning with an integration test that:
    1. Spawns a runner attached to a specific cgroup.
    2. Has the runner spawn an interactive_shell PTY (or cli
       subprocess).
    3. Reads `/proc/<pty_child_pid>/cgroup` from the daemon side
       and asserts it matches the runner's expected cgroup
       placement.
    4. Repeats for a grandchild (the PTY's child shell).
    Catches the regression where a future kernel / cgroup-driver
    change breaks inheritance silently.
  - One commit; depends on 7a (always-spawn) so the runner
    has a stable cgroup placement.

## 8. Test invariant for the next contributor

If a daemon-side migration commits to the runner-RPC surface,
the matching handler MUST have:

- A unit test in `jaato-server/server/runner/tests/test_session_<area>_rpc.py`.
- A daemon-side wrapper test in
  `jaato-server/server/runner/tests/test_session_method_wrappers_e2e.py`
  (or a dedicated e2e file).
- The lifecycle composition test
  (`test_session_dispatch_lifecycle_e2e.py`) extended with the
  new step if the migration introduces a new ordering invariant.

This ensures the wire contract stays frozen while the daemon-
side code churns.

## 9. Bisect anchors

The §3.3c precursor work is bounded by the following commit range:

- **First**: `5796ba93` — `runner: add session.health_check RPC handler`
- **Last** (precursors only): `395b71f8` — `runner: add session.get_turn_accounting RPC handler`
- **Design doc**: `37a9500e` — this doc's initial landing.

Daemon-side migrations done within the precursor window:

- `45a2dbd8` — `JaatoServer.shutdown` → `session.shutdown`
- `b2c0772d` — `JaatoServer.terminal_width` setter forwards
- `c3d5ec08` — `JaatoServer.set_presentation_context` forwards

A future bisect that fingers a session-RPC regression should
look in this range first; the lifecycle composition test
(`test_session_dispatch_lifecycle_e2e.py`) is the most
load-bearing single test for cross-handler ordering.

## 10. §7b.1 audit appendix — per-site classification

Per peer-review request after the worker's §7b.1 scope correction
(commit `fafe90a6`).  The 50 `self._jaato.X` call sites in
`core.py` plus 25 plain truthiness checks (`if self._jaato:`,
total 75 references) are classified below by **enclosing method**
+ **bucket**.  The classification is grounded in init ordering:

### Init ordering (the load-bearing fact)

```
SessionManager._bootstrap_session(envelope):
  1. _run_pre_initialize_hooks(server, session_id, workspace_path, client_id)
  2. _provision_ipc_apparmor_and_spawn_runner(...)  ← inline per §3.13
        ├── apparmor.add_session_profile(...)        (daemon-side)
        ├── runner_spawn.spawn_session_runner(...)   (forks runner)
        │     └── server.set_runner_rpc(rpc, spawned)  ← runner_rpc set HERE
        └── (auto-sends session.bootstrap envelope)   ← runner-side session ready
  3. server.initialize()                              ← every _jaato.X site here
                                                       has runner_rpc available
  4. Build Session record
```

So **every `self._jaato.X` site that runs from inside or after
`server.initialize()` has `self._runner_rpc` available.** Only
sites in `JaatoServer.__init__` itself are truly pre-runner.

### Available `session.*` RPC handlers (15)

`bootstrap`, `health_check`, `get_session_state`,
`set_session_state`, `get_all_session_state`, `is_running`,
`request_stop`, `get_history`, `get_context_usage`,
`get_turn_accounting`, `set_terminal_width`,
`set_streaming_enabled`, `set_presentation_context`, `reset`,
`shutdown`.

**Notably absent:** `send_message` (the §7b.2 task; not in
surface yet); `get_context_limit` (today's
`get_context_usage` returns a dict that *may* include
`context_limit`, with a fallback to a separate `get_context_limit`
call — adding the dict field would make the fallback
unnecessary).

### Bucket key

| Bucket | Meaning |
|---|---|
| **DONE** | Already migrated (vanguard) |
| **NOW** | Migratable today: post-runner-spawn AND a matching RPC handler exists |
| **DAEMON** | Stays daemon-side per §4.2 (provider/auth/runtime/event_bus/UI) |
| **INTERNAL** | Reads/writes `JaatoSession`'s private state (`_executor`, `_tools`, `_agent_id`, `_agent_name`, etc.) — refactored away during §7c, not migrated |
| **WIRING** | Wires daemon-side state INTO the in-process JaatoSession (`configure_tools`, `set_session_plugin`, `set_gc_plugin`) — migrates when the runner-side equivalents exist |
| **§7b.2** | Belongs to the `send_message` migration task |
| **TRULY-PRE** | Runs before runner-spawn — defer to post-§7c |
| **TRUTHINESS** | `if self._jaato:` truthiness check; becomes `if self._runner_rpc:` post-§7c, no individual migration |

### Per-method classification

| Method (line range) | Sites | Bucket |
|---|---|---|
| `__init__` (215-418) | 309 (truthiness) | **TRUTHINESS** (collapses post-§7c) |
| `event_bus` getter (430-443) | 436, 438 | **DAEMON** (event_bus is daemon-tier) |
| `terminal_width` setter (490-523) | 498, 499 | **DONE** (vanguard `b2c0772d`) |
| `set_presentation_context` (525-561) | 543, 544 | **DONE** (vanguard `c3d5ec08`) |
| `set_apparmor_confinement` (563-588) | 583, 586 | **INTERNAL** (sets `session._apparmor_context`) |
| `set_pre_init_confine_context` (590-628) | 625, 626 | **DAEMON** (`get_runtime`) |
| `set_runtime_limits` (630-675) | 668, 671 | **INTERNAL** (sets `session._executor` runtime limits) |
| `set_reference_authorizer` (677-701) | 694, 699 | **INTERNAL** (sets `session._reference_plugin._authorizer`) |
| `_get_event_bus` (978-988) | 984, 985 | **DAEMON** (`get_runtime` → `event_bus`) |
| `emit_current_state` (1002-1081) | 1061, 1062 | **NOW** (could use `session.get_all_session_state` RPC) |
| `_build_tool_id_mappings` (1281-1302) | 1290, 1291 | **INTERNAL** (reads `session._tools`) |
| `initialize` (1340-1969) | 14 sites | **mixed**, see breakdown below |
| `_run_connect_provider` (1426-1460) | 1436, 1444, 1454 | **DAEMON** (constructs `_jaato`, calls `connect`, reads runtime) |
| `_setup_session_plugin` (2197-2246) | 2203, 2215 | **WIRING** (`set_session_plugin` wires daemon-side) |
| `_setup_agent_hooks` (2248-2639) | 2251, 2622, 2624, 2626, 2628 | **INTERNAL** (`_agent_id`, `_agent_name`) + **DAEMON** (`set_ui_hooks`) |
| `send_message` (3147-3190) | 3154, 3163, 3164 | **§7b.2** |
| `_start_model_thread` (3192-3474) | 3198, 3199 | **§7b.2** |
| `_find_plugin_for_command` (3552-3584) | 3561, 3564 | **DAEMON** (`get_runtime`) |
| `_get_sandbox_paths` (3586-3621) | 3593, 3596 | **DAEMON** (`get_runtime`) |
| `stop` (3623-3631) | 3629, 3630 | **NOW** (`session.is_running` + `session.request_stop`) |
| `execute_command` (3633-3732) | 3643, 3646, 3669 | **DAEMON** (user-commands UI surface) |
| `clear_history` (3734-3748) | 3736, 3737 | **NOW** (`session.reset`) |
| `get_session` (3759-3769) | 3767, 3769 | **INTERNAL** (returns `JaatoSession` reference) — refactor as part of §7c |
| `get_available_commands` (3824-3829) | 3826, 3828 | **DAEMON** (UI commands) |
| `get_available_models` (3847-3860) | 3852, 3857 | **DAEMON** (model completion list) |
| `_check_auth_completion` (4006-4133) | 4017, 4029, 4038, 4047, 4086, 4087, 4088, 4115 | **mixed** — same shape as `initialize` |

#### `initialize()` and `_check_auth_completion()` — the 14+8 mixed clusters

Both methods do the daemon-side wiring sequence after auth.
Same shape, same bucket distribution:

| Site (initialize / _check_auth_completion) | Bucket |
|---|---|
| 1627 `model_name` | **DAEMON** (provider read) |
| 1628 `provider_name` | **DAEMON** |
| 1629 `set_terminal_width` | **NOW** (handler exists; this is the per-site insight the worker flagged) |
| 1665 / 4017 `verify_auth` | **DAEMON** (auth) |
| 1729 `configure_plugins_only` | **WIRING** |
| 1748 / 4029 `configure_tools` | **WIRING** |
| 1799 `get_runtime` | **DAEMON** |
| 1818 / 4038 `set_gc_plugin` | **WIRING** (gc is daemon-tier per §4.2 but the call wires it INTO `_jaato`) |
| 1829, 1845 / 4047 `get_session` | **INTERNAL** |
| 1899 / 4087 `get_context_usage` | **NOW** (handler exists) |
| 1900 / 4088 `get_context_limit` | **NOW-with-caveat** (handler missing; either add a sibling handler OR extend `get_context_usage` dict to always include `context_limit`, removing the fallback) |
| 1928 / 4115 `auth_info` | **DAEMON** (auth) |
| 1898 / 4086 (truthiness) | **TRUTHINESS** |

### Bucket totals

| Bucket | Site count | Status |
|---|---|---|
| **DONE** (vanguard + §7b.1 iteration) | 6 | All shipped.  Vanguard (3): `shutdown`, `terminal_width` setter, `set_presentation_context`.  §7b.1 iteration (3): `clear_history` (8cbb8ba2), `stop` (fafe90a6), init-time `set_terminal_width` direct call (7f8be0cb). |
| **DEFER-§7c** (was originally tagged NOW) | ~5 | Re-bucketed per 7f8be0cb's audit-correction: read-time sites that the seat-flip naturally collapses.  See "§7b.1 wrap-up: NOW-bucket re-classification" below. |
| **DAEMON** | ~22 | Stay daemon-side per §4.2 |
| **INTERNAL** | ~12 | Refactor away during §7c |
| **WIRING** | ~7 | Migrate when runner-side counterparts exist (§7b.1 phase 2 or post-§7c) |
| **§7b.2** | 5 | `send_message` + `_start_model_thread` cluster — shipped (3ca3c14d) |
| **TRULY-PRE** | 0 | None — `__init__` only has a truthiness check |
| **TRUTHINESS** | ~15 | Collapse post-§7c |

**Total: 75** (50 attribute calls + 25 truthiness checks).

### §7b.1 wrap-up: NOW-bucket re-classification

The audit's original "NOW" bucket included 5 sites that subsequent
implementation work re-bucketed.  Recorded for posterity:

| Audit-line | Code today | Original bucket | Final bucket | Reason |
|---|---|---|---|---|
| 1061-1062 | 1072-1078 (`emit_current_state` reads `session.instruction_budget`) | NOW | **DAEMON** | The audit assumed `session.get_all_session_state` would cover this.  It doesn't — `instruction_budget` is a private attribute exposed via property, not a registered state provider.  Migration would require a dedicated `session.get_instruction_budget` RPC, which is daemon-tier (instruction-budget tracking lives daemon-side per §4.2). |
| 1899 / 4087 | 1935 / 4171 (`get_context_usage` reads at init / auth-completion) | NOW | **DEFER-§7c** | READS at initialize / auth-completion time.  Daemon-side `_jaato` is the source of truth pre-§7c (just-initialized; runner-side session has zero usage at this moment).  Migrating reads NOW would break the toolbar's initial-usage display.  Defer to post-§7c when the runner becomes the source of truth. |
| 1900 / 4088 | 1936 / 4172 (`get_context_limit` reads at init / auth-completion) | NOW | **DEFER-§7c** | Same shape as `get_context_usage` reads above.  The precursor handler `session.get_context_limit` (added at 34ecbe0a) is in place for the eventual flip. |

**Net §7b.1 closed scope:** 6 WRITE sites migrated across 6 commits (3
vanguard + 3 iteration).  All read-side sites + emit_current_state
collapse during §7c — no further §7b.1 work is required.

The 7f8be0cb commit message contains the canonical audit correction;
this table is its design-doc-resident counterpart.

### What this audit does NOT decide

- **§7a (always-spawn) impact on this audit:** if §7a lands first, every IPC + WS session has a runner regardless of apparmor opt-in.  The audit's classifications don't change (init ordering is unchanged — the runner spawns at the same step), but the PROPORTION of sessions where the migrated sites actually take the runner path goes from "apparmor-opt-in subset" to "all sessions."  This is the correctness amplification §7a delivers.
- **§7c flag-removal sequencing:** the **TRUTHINESS** + **INTERNAL** + **WIRING** buckets all collapse during §7c (when `_jaato` field is removed).  This audit doesn't sequence those collapses; that's §7c's task.
