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
      → §7c (remove in-process JaatoSession + flag) ← steps 1-2 shipped
                                                       (always-bootstrap +
                                                       JAATO_RUNNER_HOSTS_SESSION
                                                       flag removed; WS
                                                       bootstrap parity);
                                                       steps 3-7 pending. See
                                                       §7c sequencing table.
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

**Sequencing.**  The original "one logical commit" estimate has
been refined into discrete steps as the absorbed scope (response-
handler rewiring + introspection collapse + DEFER-§7c read
migrations + the actual field removal) was enumerated:

| Step | Focus | Status |
|---|---|---|
| **§7c step 1** | Always-bootstrap the runner-side session (remove `JAATO_RUNNER_HOSTS_SESSION` flag from the IPC spawn path; bootstrap dispatches unconditionally; failure-tolerant). | **Shipped.**  Files: `server/__main__.py`, `server/runner_spawn.py`, `server/runner/__main__.py`, `server/runner/session.py` (doc-comments).  New regression test: `server/tests/test_spawn_session_runner_always_bootstraps.py` (8 tests pinning the unconditional dispatch + failure tolerance + flag-value irrelevance). |
| **§7c step 2** | WS-side bootstrap parity (WS spawn lacked the bootstrap dispatch; refactored the IPC envelope-build + dispatch into a shared `dispatch_bootstrap_envelope` helper in `server/runner_spawn.py` and wired both IPC + WS callers through it). | **Shipped.**  Files: `server/runner_spawn.py` (new `build_session_envelope` + `dispatch_bootstrap_envelope` helpers — relocated from `__main__.py`), `server/__main__.py` (call-site rewritten through the helper; legacy `_build_session_envelope` re-exported under the old private name for back-compat), `server/websocket.py` (WS hook calls `dispatch_bootstrap_envelope` after `spawn_session_runner`).  New tests: `server/tests/test_dispatch_bootstrap_envelope.py` (4 unit tests pinning happy-path + None-rpc no-op + failure-swallow + timeout-threading), plus 2 new tests in `test_ws_always_spawn_runner.py` covering the WS bootstrap dispatch on confined + unconfined paths. |
| **§7c step 3** | INTERNAL + WIRING bucket refactors — replace `_jaato.get_session()._executor` / `set_session_plugin` / `set_gc_plugin` with daemon-side direct accessors or runner-RPC equivalents.  Shrinks `_jaato.X` site count to the DAEMON-only residual.  Decomposed into sub-steps below. | In progress. |
|     **§7c step 3a** | Encapsulation cleanup — replace daemon-side reaches into private `_jaato._agent_id` / `_jaato._agent_name` attributes with the new public `JaatoClient.set_agent_identity(agent_id, agent_name)` setter. | **Shipped.**  Files: `shared/jaato_client.py` (new `set_agent_identity` method), `server/core.py` (call-site `_setup_agent_hooks` updated).  New tests: 4 in `shared/tests/test_jaato_client.py` (`TestJaatoClientSetAgentIdentity`). |
|     **§7c step 3b** | Remaining INTERNAL bucket — refactor `_build_tool_id_mappings` (1290) to use the new public `JaatoSession.get_tool_schemas` / `JaatoClient.get_tool_schemas` accessors instead of reading the private `session._tools`.  Other INTERNAL sites (`set_apparmor_confinement`, `set_runtime_limits` — `session._executor` reaches; `set_reference_authorizer` — already public; `send_message`/`_start_model_thread` `get_session()` reads — daemon-tier executor accesses) defer to step 6 when `_jaato` removal forces the rework. | **Shipped.**  Files: `shared/jaato_session.py` (new public `get_tool_schemas`), `shared/jaato_client.py` (forwarding `get_tool_schemas`), `server/core.py` (`_build_tool_id_mappings` updated).  New tests: 3 in `shared/tests/test_jaato_session.py` (`TestGetToolSchemas`) + 2 in `shared/tests/test_jaato_client.py` (`TestJaatoClientGetToolSchemas`). |
|     **§7c step 3c** | WIRING bucket — `set_session_plugin` (2251), `configure_plugins_only` (1765), `configure_tools` (1784, 4113), `set_gc_plugin` (1854, 4122).  **Doc-only clarification:** these sites pass rich Python plugin INSTANCES (GCPlugin, SessionPlugin, registry, permission_plugin) that cannot cross an RPC boundary as-is.  But the runner-side session ALREADY receives the equivalent configuration via the `SessionInitEnvelope` (the `plugins` + `gc` fields), and `bootstrap_session()` → `runtime.create_session()` → `session.configure()` does the runner-side wiring during boot.  So daemon-side WIRING calls today configure the *daemon-side* JaatoClient/JaatoSession; post-step-6 (when the daemon-side session is removed) they collapse into either no-ops or daemon-side runtime configuration that does NOT need a runner-RPC forward.  No active migration in step 3c — the WIRING bucket sunsets alongside step 6's `_jaato`-removal. | **Shipped (clarification).**  No code changes; this row records the architectural decision. |
| **§7c step 4** | DAEMON bucket migration — split `JaatoClient` so `JaatoRuntime` lives daemon-side under a new `self._runtime` field; convert `_jaato.provider_name` / `_jaato.model_name` / `_jaato.is_connected` / `_jaato.auth_info` / `get_runtime` to direct `self._runtime.X` reads.  Introspection collapse from §7b.3 lands here.  **Step 4 first pass shipped:** added the `self._runtime` field + aliased to `self._jaato.get_runtime()` at connect; migrated all 5 active `get_runtime()` read sites (`set_pre_init_confine_context`, `_get_event_bus`, formatter-pipeline init wiring, `_find_plugin_for_command`, `_get_sandbox_paths`) to read through the new field.  `model_name` / `provider_name` / `auth_info` reads (4 sites at lines 1665-1666, 1990, 4225) batch with step 5 since they need a similar but distinct migration shape (JaatoRuntime exposes `provider_name` natively but `model_name` lives on JaatoClient and `auth_info` reads through the session-tier provider). | First pass shipped (5 sites). |
| **§7c step 5** | DEFER-§7c read migrations — switch `get_context_usage` / `get_context_limit` reads at `initialize()` + `_check_auth_completion()` to runner-RPC reads. | **Folded into step 6.**  Per the path-A architectural decision: shipping step 5 as a standalone read-source switch creates a transitional window where `_jaato` still exists but reads come from the runner — a coherence/divergence risk for the toolbar.  Step 6 atomically removes `_jaato` and switches all reads simultaneously, eliminating the transitional state.  Pre-step-6 verification: see "Pre-step-6 toolbar-coherence test" below. |
| **Pre-step-6** | Toolbar-coherence verification — pin the wire-fidelity + determinism invariants the atomic seat-flip relies on.  Catches divergence between daemon-side and runner-side `get_context_usage()` reads BEFORE step 6 lands. | **Shipped.**  File: `server/runner/tests/test_pre_step6_toolbar_coherence.py` (5 tests pinning: 2-instance determinism with equal inputs, negative determinism with different inputs, RPC round-trip preserves dict + int, init-time zero-usage shape preservation across all toolbar-relevant fields). |
| **§7c step 6** | Atomic seat-flip — absorbs the read migrations folded from step 5.  **Pre-implementation audit (post-step-4-first-pass) revised the original 6a/6b/6c shape:** the `_jaato.get_session()` readers — originally classified as a single "INTERNAL refactor" sub-step (6b) — actually have 4 distinct dispositions that each need their own treatment.  Updated decomposition below.  Removes the `JAATO_RUNNER_HOSTS_SESSION` env-var doc references along the way. | Pending. |
|     **§7c step 6 disposition audit** | Per-`get_session()`-reader disposition table built from the actual call sites in `server/core.py`.  Recorded for the next implementer.  See "Step 6 disposition audit" below. | **Shipped (audit-only).** |
|     **§7c step 6.1** | Add 3 new runner-RPC handlers (each §7b.2-scale: handler + daemon-side wrapper + unit + e2e tests): `session.set_reference_authorizer`, `session.snapshot_instruction_budget`, `session.inject_prompt`.  Prerequisites for the daemon-side reader migrations in 6.2. | **All 3 shipped:** (a) `session.set_reference_authorizer` (12 tests; bool flag — ReferenceAuthorizer Python object can't cross RPC, holds daemon-side AppArmorManager reference; runner-side references plugin post-migration reads `JaatoSession.is_reference_authorization_enabled()` + uses the existing `apparmor.add_reference_fragment` runner→daemon RPC §3.2.2 to authorize paths). (b) `session.snapshot_instruction_budget` (11 tests; reads `JaatoSession.instruction_budget.snapshot()` for the daemon's `emit_current_state` site at core.py:1091; returns `{"snapshot": <dict|None>}` with deep-copy isolation; None when budget pre-configure). (c) `session.inject_prompt` (20 tests; for the `execute_command` site at core.py:3238; SourceType enum serialized as its lowercase string value across the wire; runner-side handler maps back to enum; unknown enum values rejected with the valid-list hint to catch typos that would silently misroute message priority).  **Step 6.1 closed — 6.2 unblocked.** |
|     **§7c step 6.2** | Migrate the **6 straightforward `_jaato.get_session()` readers** per the disposition audit below: 2 deletes (lines 610, 697 — daemon-tier executor reaches), 3 new-RPC forwards (lines 725, 1091, 3238 — using the handlers from 6.1), 1 trivial migration to `self._runtime.event_bus` (line 462).  Self-contained commit — does not depend on the architectural callback decisions in 6.2.5. | **Shipped.**  Per-site disposition: (a) line 462 `event_bus` property — migrated to `self._runtime.event_bus` (mirrors `_get_event_bus` from §7c step 4 first pass).  (b) line 610 `set_apparmor_confinement` — body gutted to no-op (zero live callers in OSS tree; the runner subprocess is process-confined at spawn time, so daemon-side thread-level wiring this method installed had no effect post-§7b.2).  (c) line 697 `set_runtime_limits` — body gutted to no-op (WS still calls it; runner-side gets cgroup attach + limits via env vars at spawn time).  (d) line 725 `set_reference_authorizer` — already write-both since §7c step 6.1 (1/3); daemon-side leg drops naturally with the references-plugin runner-side migration in a separate sub-track.  (e) line 1091 `emit_current_state` instruction_budget read — REPLACED with `session_snapshot_instruction_budget_threadsafe` RPC forward; runner-side becomes source of truth for the snapshot.  (f) line 3238 `inject_prompt` — added write-both: forwards via `session_inject_prompt_threadsafe` RPC AND keeps the daemon-side leg (drops in step 6.3 alongside other write-both daemon legs).  Tests: 600/600 server suite green. |
|     **§7c step 6.2.5** | Migrate the **3 architectural callback-rewire `get_session()` readers** (lines 1891, 1907, 3273 — `_event_bus_tools` + `instruction_budget` callback wiring + `set_prompt_injected_callback`).  Each gates on runner-side event-bus access plumbing which the Phase 3 plan parked as out-of-scope.  Decoupled from 6.2 so that step's progress doesn't get tied to an architectural decision that may not be ready.  May itself fan out into multiple commits depending on the event-bus plumbing decisions. | Pending. |
|     **§7c step 6.3** | Drop write-both daemon-side legs (clear_history, stop, terminal_width property setter, set_presentation_context, init's set_terminal_width direct call) — safe ONLY after 6.2 completes (until then, daemon-side `get_session()` readers still depend on the daemon-side session having current state). | **Shipped (6 of 7 legs):** dropped daemon-side legs of `clear_history` (was §7b.1 8cbb8ba2), `stop` (was §7b.1 fafe90a6), `terminal_width` property setter (was vanguard b2c0772d), `set_presentation_context` (was vanguard c3d5ec08), init's `set_terminal_width` direct call (was §7b.1 7f8be0cb), and `inject_prompt` (was §7c step 6.2).  **Split-out per reviewer's clause:** `set_reference_authorizer` daemon-side leg KEPT for now — the daemon-side references plugin (still active until tier-filtered runner-side migration lands as a separate sub-track) consumes the Python authorizer object via `session.get_reference_authorizer()`.  Dropping it would break kernel-layer authorization for WS sessions with confined references; will drop alongside the references-plugin runner-side migration.  Test churn: 5 write-both-specific assertions collapsed in `test_jaato_server_clear_history_forwards.py` + `test_jaato_server_stop_forwards.py`; the surviving runner-only assertions prove the post-§7c-step-6.3 contract holds.  `_jaato` count 66 → 55 (-11; 6 legs yielded 11 reductions because some sites had paired truthiness checks).  Tests: 597/597 server suite green. |
|     **§7c step 6.4** | Delete WIRING calls per the §7c step 3c clarification: `set_session_plugin`, `configure_plugins_only`, `configure_tools` (×2), `set_gc_plugin` (×2).  Runner already gets equivalent config via the bootstrap envelope. | **Deferred — collapses into step 6.6** per the disposition audit below.  The 6 WIRING calls are load-bearing for the daemon-side seat during the §7c rollout window (specifically: `configure_tools` is what *creates* the daemon-side `JaatoClient._session` instance; many surviving daemon-side `_jaato` consumers — architectural callbacks 1969/1985/3374/4264 + `auth_info` reads + the public `get_session()` method called from session_manager + websocket + `set_ui_hooks` — depend on that instance existing).  Dropping the WIRING calls in isolation would silently null out the daemon-side session and cascade.  Step 6.6's `_jaato`-field removal eliminates them naturally; no separate step 6.4 commit needed. |

### Step 6.4 disposition audit

Mirroring cd3ecf20's audit shape for the 9 `get_session()`
readers.  The 6 WIRING call sites + their downstream
dependencies:

| Site | Wires what onto JaatoClient | Side effect — what does the daemon-side seat lose if dropped? | Disposition |
|---|---|---|---|
| 1870 (`init` `configure_plugins_only`) | runtime.registry / runtime.permission_plugin / runtime.ledger | Runtime-side plugin discovery for the daemon's runtime instance; same registry instance is also wired runner-side via the bootstrap envelope.  Daemon's runtime can still respond to `provider_name` / `is_connected` reads (those don't depend on `configure_plugins`). | Stays — drops with step 6.6. |
| 1889 (`init` `configure_tools`) | runtime.registry/permission/ledger + **creates `_jaato._session`** | **Highest impact:** without this call, `_jaato._session` is None.  Cascades to: callback wiring at 1969/1985 (None-safe — silently no-ops), 3374's `set_prompt_injected_callback` (NOT None-safe — AttributeError), `auth_info` reads at 2068/4332 (returns ``""``), public `get_session()` method at 3986 (returns None to many external consumers including session_manager fork/journal paths + websocket persistence). | Stays — drops with step 6.6. |
| 1958 (`init` `set_gc_plugin`) | gc_plugin onto session | Session's GC trigger.  Daemon-side `send_message` is gone post-§7b.2, so the GC's trigger path is itself dormant; but session-state save points may still trigger GC daemon-side via the `_get_event_bus` path.  Even if dormant, the WIRING call's failure mode is "AttributeError because `_jaato._session` is None" (cascades from 1889). | Stays — drops with step 6.6. |
| 2355 (`_setup_session_plugin` `set_session_plugin`) | session_plugin (persistence) | Persistence plugin is still consumed daemon-side for save/resume/sessions/delete-session/backtoturn user commands — those flow through `JaatoServer.execute_user_command` → daemon-tier path.  Dropping this WIRING call silently disables the persistence plugin's daemon-side hooks. | Stays — drops with step 6.6. |
| 4246 (`_check_auth_completion` `configure_tools`) | Mirror of 1889 | Same as 1889 — load-bearing creation of `_jaato._session` for the auth-deferred init path. | Stays — drops with step 6.6. |
| 4255 (`_check_auth_completion` `set_gc_plugin`) | Mirror of 1958 | Same as 1958. | Stays — drops with step 6.6. |

#### Reviewer's symmetric question, answered:

> Post-6.2/6.3, do any of the 4 architectural callbacks (1970/1986/3382/4282) or the connect/runtime path read `_jaato.permission_plugin` / `_jaato.gc_plugin` / `_jaato.session_plugin` / `_jaato.registry`?

**Direct attribute reads:** No.  The surviving `_jaato.X` sites
read public methods/properties (`model_name`, `provider_name`,
`auth_info`, `get_session`, `verify_auth`, `get_user_commands`,
`execute_user_command`, `get_model_completions`,
`get_tool_schemas`, `set_agent_identity`, `set_ui_hooks`).  None
read the wired plugin instances directly.

**Transitive reads through `get_session()`:** Yes — extensively.
The wired plugins' instances are *consumed* by the daemon-side
JaatoSession that `configure_tools` creates.  Dropping WIRING
makes `_jaato._session = None`, breaking every site that calls
`_jaato.get_session()` non-None-safely + degrading several that
are None-safe (callback wiring silently no-ops; `auth_info`
returns ``""``; public `get_session()` returns None to external
consumers in session_manager + websocket).

**Conclusion:** the WIRING calls are not *directly* load-bearing
(no `_jaato.permission_plugin` reads exist) but they are
*transitively* load-bearing via `_jaato._session`.  The same
discipline that produced the §7c step 6.3 split-out for
`set_reference_authorizer` applies here: do not drop calls
whose absence cascades through indirect consumers.

#### Sequencing implication:

Step 6.4 as originally framed ("delete the 6 WIRING calls in
their own commit") is **incompatible** with the rollout window
where steps 6.2.5 / 6.5 / 6.6 haven't landed yet.  Two options:

  - **Option A (chosen):** fold step 6.4 into step 6.6.  When
    `_jaato` is removed, the WIRING calls are mechanically
    deleted alongside (they reference `self._jaato` which no
    longer exists); the cascading consumers also rewrite/remove
    in the same commit.  Single coherent diff.

  - **Option B (rejected):** insert a step 6.4-prereq doing the
    cascade-rewrites first — migrate 1969/1985/3374/4264 to
    runner-RPCs (= step 6.2.5), migrate `auth_info` /
    `get_session` consumers (= part of step 6.5), THEN delete
    WIRING.  Adds a multi-commit chain that step 6.6 absorbs
    anyway; doesn't reduce review surface.

#### What this audit does NOT decide:

The audit assumes the §7c rollout window's daemon-side seat is
still functional (per the path-A "atomic seat-flip in step 6.6"
framing).  If a future review decides to ship a transitional
"daemon-side seat half-broken" window, step 6.4 could land
in isolation; that's a deliberate scope choice the audit is
not asked to make.
|     **§7c step 6.5** | Migrate the remaining ~5 introspection reads not covered by step 4: `model_name` / `provider_name` (lines 1665-1666), `auth_info` (1990, 4241), `verify_auth` (1728, 4143), `get_user_commands` / `execute_user_command` (3755, 3778, 3954), `get_model_completions` (3983).  Routes through `self._runtime` for runtime-tier reads + `self._jaato.get_session()._provider` reaches for session-tier reads (which themselves collapse if 6.2 lands first). | **Shipped** (4 of 9 sites; rest transitively-load-bearing per the §7c step 6.6 disposition audit and fold into 6.6.4).  Migrated: line 1747 `model_name` (collapsed to direct param read — JaatoClient's `model_name` was always equal to the passed-in arg post-connect, the `or` was dead defensive code), line 1748 `provider_name` (now reads `self._runtime.provider_name`), lines 1817 + 4250 `verify_auth` ×2 (now call `self._runtime.verify_auth(...)` directly — JaatoClient.verify_auth was a thin forwarder).  Tests: 597/597 server suite green.  `_jaato` count 55 → 54 (−1; the 4 site migrations collapsed into 3 lines with one site folded into the param-read shortening). |
|     **§7c step 6.6** | Remove `self._jaato` field + collapse the ~15 `if self._jaato:` truthiness checks to `if self._runner_rpc:`.  This is the actual seat-flip's "the daemon-side session is gone" moment.  Mechanical diff; safe ONLY after 6.1 → 6.5 land. | **Pending — sub-commit decomposition recorded in the §7c step 6.6 disposition audit below.**  Original "mechanical diff" framing was over-simplified; the audit reveals 4 sub-commits worth of cross-cutting work absorbed by 6.6 alongside the field removal: (6.6.1) 3 new RPC handlers for external `JaatoServer.get_session()` consumers' persistence-restore paths in `session_manager.py`; (6.6.2) architectural callbacks rewire (the original step 6.2.5 work, now folded in); (6.6.3) external consumer migrations across `session_manager.py` + `websocket.py` (8 callers of the public `JaatoServer.get_session()`); (6.6.4) the actual atomic field removal + WIRING absorbs + transitively-load-bearing introspection cleanups + truthiness collapses.  See sub-commit table below. |

### Step 6.6 disposition audit

Mirroring cd3ecf20's audit shape and ac088e67's audit shape.
The §7c step 6.4 audit established that WIRING calls are
transitively load-bearing via `_jaato._session`.  This audit
extends the same discipline to the broader set of remaining
`_jaato` sites + the external `JaatoServer.get_session()`
consumers.

#### Per-site classification (35 active sites in `core.py`)

Every non-comment `self._jaato.X` site post-§7c-step-6.3,
classified by what it actually reads:

| Site | Operation | Read source | Classification |
|---|---|---|---|
| 767 | `get_session()` daemon leg of `set_reference_authorizer` (split out per 6.3) | `_jaato._session` | Stays — drops with references-plugin runner-side migration sub-track |
| 1407 | `get_tool_schemas()` for `_build_tool_id_mappings` | `_jaato._session._tools` (or empty list when `_session` None) | Transitively load-bearing; drops with 6.6.4 |
| 1558 | `JaatoClient(...)` constructor + `connect()` call | Daemon-side construction | Stays — refactor to daemon-direct `JaatoRuntime` construction in 6.6.4 |
| 1565 | `self._runtime = self._jaato.get_runtime()` (alias-write) | `_jaato._runtime` | Refactor — split JaatoClient construction so daemon constructs JaatoRuntime directly in 6.6.4 |
| 1747 | `self._model_name = self._jaato.model_name or model_name` | `_jaato._model_name` (pure JaatoClient state, not session-dependent) | **CLEAN — step 6.5** |
| 1748 | `self._model_provider = self._jaato.provider_name` | `_jaato._provider_name` (pure JaatoClient state) — also exposed as `_runtime.provider_name` | **CLEAN — step 6.5** |
| 1806 | `self._jaato.verify_auth(...)` (init path) | `_jaato._runtime.verify_auth(...)` — runtime-tier, not session-dependent | **CLEAN — step 6.5** |
| 1870 | `configure_plugins_only(...)` WIRING | runtime-tier wiring | Drops with 6.6.4 (per 6.4 audit) |
| 1889 | `configure_tools(...)` WIRING — **creates `_jaato._session`** | Highest-impact WIRING call | Drops with 6.6.4 (per 6.4 audit) |
| 1958 | `set_gc_plugin(...)` WIRING | `_jaato._session.set_gc_plugin(...)` | Drops with 6.6.4 (per 6.4 audit) |
| 1969 | `_event_bus_tools` callback wiring | `_jaato._session._event_bus_tools` | **ARCHITECTURAL — step 6.6.2** |
| 1985 | `instruction_budget` callback wiring (init path) | `_jaato._session._on_instruction_budget_updated` | **ARCHITECTURAL — step 6.6.2** |
| 2039, 2040 | `get_context_usage` / `get_context_limit` (init path) | `_jaato._session.get_context_usage()` / `.get_context_limit()` | DEFER-§7c reads — drop in 6.6.4; runner-RPC handlers exist (`session.snapshot_instruction_budget` covers usage; `session.get_context_limit` added at 34ecbe0a) |
| 2068 | `auth_info` (init path) | `_jaato._session._provider.get_auth_info()` | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |
| 2355 | `set_session_plugin(...)` WIRING | `_jaato._session.set_session_plugin(...)` | Drops with 6.6.4 (per 6.4 audit) |
| 2775 | `self._jaato.set_agent_identity(...)` (in `_setup_agent_hooks`) | `_jaato._agent_id` / `_jaato._agent_name` (pure JaatoClient state since §7c step 3a) | **TRANSITIVELY-USED — step 6.6.4** (the daemon-side `_jaato.set_ui_hooks(hooks)` at 2780 forwards hooks to `_session` if it exists; setting agent_id without a session is benign but the cascade through 2780 needs care) |
| 2780 | `self._jaato.set_ui_hooks(hooks)` | Forwards to `_jaato._session.set_ui_hooks(hooks, self._agent_id)` | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |
| 3374 | `set_prompt_injected_callback` (NOT None-safe) | `_jaato._session.set_prompt_injected_callback(...)` | **ARCHITECTURAL — step 6.6.2** (the AttributeError-on-None site flagged in 6.4 audit) |
| 3844, 4045 | `get_user_commands()` | `_jaato._session.get_user_commands()` (returns `{}` when no session) | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |
| 3867 | `execute_user_command(command, parsed_args)` | `_jaato._session.execute_user_command(...)` (raises RuntimeError when no session) | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |
| 3986 | `get_session()` (public method) returns the session | Returns `_jaato._session` to 8 external consumers | **EXTERNAL-FACING — step 6.6.3** (8 callers in session_manager + websocket; see external-consumer table below) |
| 4074 | `get_model_completions(["select"])` | `_jaato._session.get_model_completions(...)` (returns `[]` when no session) | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |
| 4234 | `verify_auth(allow_interactive=False)` (auth-completion path) | `_jaato._runtime.verify_auth(...)` — runtime-tier | **CLEAN — step 6.5** |
| 4246 | `configure_tools(...)` WIRING (mirror of 1889) | Same as 1889 | Drops with 6.6.4 |
| 4255 | `set_gc_plugin(...)` WIRING (mirror of 1958) | Same as 1958 | Drops with 6.6.4 |
| 4264 | `instruction_budget` callback (auth-completion mirror of 1985) | Same as 1985 | **ARCHITECTURAL — step 6.6.2** |
| 4304, 4305 | `get_context_usage` / `get_context_limit` (auth-completion mirrors of 2039/2040) | Same as 2039/2040 | DEFER-§7c reads — drop in 6.6.4 |
| 4332 | `auth_info` (auth-completion mirror of 2068) | Same as 2068 | **TRANSITIVELY LOAD-BEARING — step 6.6.4** |

**Bucket totals:**

| Bucket | Sites | Disposition |
|---|---|---|
| **CLEAN** (step 6.5) | 1747, 1748, 1806, 4234 | 4 sites; ship as small standalone pre-6.6 commit |
| **TRANSITIVELY LOAD-BEARING** (step 6.6.4) | 1407, 2068, 2775, 2780, 3844, 3867, 4045, 4074, 4332 | 9 sites; cleanup folds into 6.6.4 alongside field removal |
| **ARCHITECTURAL** (step 6.6.2) | 1969, 1985, 3374, 4264 | 4 sites; were the original step 6.2.5 work — now folded into 6.6.2 |
| **EXTERNAL-FACING** (step 6.6.3) | 3986 | 1 site; 8 external consumers (see table below) |
| **WIRING** (step 6.6.4 absorbs from §7c step 6.4) | 1870, 1889, 1958, 2355, 4246, 4255 | 6 sites; per 6.4 audit, drops with field removal |
| **DEFER-§7c reads** (step 6.6.4 absorbs from path-A §7c step 5 fold) | 2039, 2040, 4304, 4305 | 4 sites; drops with field removal — runner-RPC handlers exist |
| **Construction sites** (step 6.6.4 refactor) | 1558, 1565 | 2 sites; refactor JaatoClient → daemon-direct JaatoRuntime construction |
| **Split-out** (separate sub-track) | 767 | 1 site; references-plugin runner-side migration |

**Total: 35 active code sites + ~15 truthiness checks → all converge in step 6.6.**

#### External consumer audit for `JaatoServer.get_session()`

The public method at line 3986 returns `_jaato._session` to
external consumers.  Inventory + per-caller migration target:

| File:Line | Operation | What it does with the session | Migration target |
|---|---|---|---|
| `core.py:3213` | `get_cancel_token` closure | Reads `session._cancel_token` (private) | **Delete** — the legacy in-process cancel-token is dead post-§7b.2.  Cancellation already routes through `_runner_rpc.session_request_stop_threadsafe`. |
| `core.py:3601` | `signal_completion` filter | Reads `session._tools` (filters for tool name) | **Refactor** to use existing `JaatoClient.get_tool_schemas()` (added §7c step 3b at 7b30c237) OR direct `self.registry.get_exposed_tool_schemas()` walk. |
| `websocket.py:1481` | event-bus access | Reads `jaato_session._runtime.event_bus` | **Trivial migration** — `event_bus` lives daemon-side per §4.2.  Use `server.event_bus` property (already migrated to `self._runtime.event_bus` in §7c step 6.2). |
| `websocket.py:1485` | event-bus access (alternate path) | Same as 1481 | Same migration as 1481 |
| `session_manager.py:1968` | initial state injection | Calls `jaato_session.set_session_state(key, value)` | **Use existing runner-RPC** — `session.set_session_state` handler shipped pre-§7c precursor.  Daemon-side wrapper: `session_set_session_state_threadsafe`. |
| `session_manager.py:2130` | initial-history seeding | Calls `jaato_session.set_initial_history(initial_history)` | **NEW runner-RPC needed** — `session.set_initial_history` handler.  Wire shape: list of Message dicts; runner-side reconstructs the Message instances.  ~§7b.2-scale (Messages have provider-specific structure but the JaatoSession.set_initial_history method itself is well-bounded). |
| `session_manager.py:2185` | cross-session prompt injection | Calls `jaato_session.inject_prompt(text, source_id, source_type)` | **Use existing runner-RPC** — `session.inject_prompt` handler shipped at §7c step 6.1 (3/3) commit 14e57709. |
| `session_manager.py:2558` | turn_accounting restore | Reads `server._jaato.get_context_usage()` + writes `jaato_session._turn_accounting = list(...)` (private attr assignment) | **NEW runner-RPC needed** — `session.restore_turn_accounting` handler.  Wire shape: list of dicts.  Used during session-restore from disk persistence; ~§7b.2-scale. |
| `session_manager.py:2591` | conversation budget restore | Calls `jaato_session.instruction_budget.restore_conversation_from_snapshot(state.budget_state)` + reads `instruction_budget.snapshot()` | **NEW runner-RPC needed** — `session.restore_conversation_budget` handler.  Mirrors the existing `session.snapshot_instruction_budget` (§7c step 6.1 at 1043bfde) but in the inverse direction.  ~§7b.2-scale. |

**Total: 8 callers; 1 delete, 4 reuse-existing-RPC, 3 NEW runner-RPC handlers needed.**

#### Sub-commit decomposition for step 6.6

Decomposed into 4 sub-commits + 1 standalone pre-6.6 commit
(6.5).  Per the audit's findings:

| Sub-commit | Scope | Estimated tests |
|---|---|---|
| **§7c step 6.5** (standalone, pre-6.6) | 4 CLEAN introspection migrations: `model_name` / `provider_name` (lines 1747-1748) + `verify_auth` ×2 (1806, 4234).  Read directly from `self._runtime` instead of `self._jaato.X`. | Existing tests cover; minimal new tests. |
| **§7c step 6.6.1** | Add 3 new runner-RPC handlers consumed by the external `get_session()` migration in 6.6.3: `session.set_initial_history`, `session.restore_turn_accounting`, `session.restore_conversation_budget`.  **Sub-decomposed per the missing-method finding below.**  Prerequisites for 6.6.3. | See sub-table. |
| **§7c step 6.6.1.0** | **JaatoSession public-method additions + session_manager private-attr-write migration.**  Two of the proposed RPC handlers don't have underlying public methods — daemon `session_manager.py` reaches into private state (`_turn_accounting` direct assignment + `instruction_budget.restore_conversation_from_snapshot` via the private accessor).  Add public methods on JaatoSession (`restore_turn_accounting(turns)` + `restore_conversation_budget(snapshot)`), migrate session_manager to use them.  Same encapsulation discipline as §7c step 3a (private `_agent_id` reach → `set_agent_identity` public method) + §7c step 3b (private `_tools` read → `get_tool_schemas` public method). | **Shipped.**  5 new tests in `shared/tests/test_jaato_session.py` (`TestRestoreTurnAccounting` + `TestRestoreConversationBudget`).  Tests: 659 passing (was 654). |
| **§7c step 6.6.1.1** | Add `session.set_initial_history` RPC handler + daemon wrapper + unit + e2e tests.  Underlying method `JaatoSession.set_initial_history` already exists (line 8252).  Per 6.1 trio cadence. | **Shipped.**  13 new tests in `server/runner/tests/test_session_set_initial_history_rpc.py`.  Wire shape: `{"messages": [<dict>, ...]}` reusing the existing `shared.plugins.session.serializer.serialize_history` / `deserialize_history` round-trip the disk-persistence path already exercises.  Tests pin: happy path 2-message round-trip, empty-list seed accepted, provenance-fields (model + provider) preservation across the wire, missing/non-list/malformed-dict args → `stage="decode"`, no_host / setter-raises / missing-method error paths, dispatch routing, e2e wrapper for 3 scenarios. |
| **§7c step 6.6.1.2** | Add `session.restore_turn_accounting` RPC handler + daemon wrapper + unit + e2e tests.  Underlying method `JaatoSession.restore_turn_accounting` added in 6.6.1.0.  Per 6.1 trio cadence. | **Shipped.**  13 new tests in `server/runner/tests/test_session_restore_turn_accounting_rpc.py`.  Wire shape: `{"turns": [<dict>, ...]}` direct (turn entries are already JSON-native dicts in the persistence serializer at `serializer.py:215`; no special encode/decode).  Tests pin: happy path round-trip, empty-list accepted, arbitrary dict keys preserved (cache-token / thinking-token / provenance fields), missing/non-list/non-dict-element args → `stage="decode"` (per-element validation catches wire-corruption / version-skew at the boundary), no_host / setter-raises / missing-method error paths, dispatch routing, e2e wrapper round-trip + caller-mutation-isolation invariant. |
| **§7c step 6.6.1.3** | Add `session.restore_conversation_budget` RPC handler + daemon wrapper + unit + e2e tests.  Underlying method `JaatoSession.restore_conversation_budget` added in 6.6.1.0.  Per 6.1 trio cadence. | **Shipped.**  13 new tests in `server/runner/tests/test_session_restore_conversation_budget_rpc.py`.  Wire shape: `{"snapshot": <dict>}` direct (the snapshot is a JSON-native dict produced by `InstructionBudget.get_conversation_snapshot()` / `SourceEntry.to_dict()`; same wire-shape-reuse rationale as 6.6.1.1 + 6.6.1.2 — persistence shape IS wire shape).  Tests pin: happy path, empty-dict accepted (matches underlying method's no-op-on-empty contract), nested SourceEntry children preserved, missing/non-dict args → `stage="decode"`, no_host / setter-raises (e.g. invalid gc_policy enum) / missing-method error paths, no-op-when-no-budget contract preserved, dispatch routing, e2e wrapper round-trip + caller-mutation-isolation invariant.  **§7c step 6.6.1 trio CLOSED — 6.6.3 unblocked.** |
| **§7c step 6.6.2** | Architectural callbacks rewire (the original §7c step 6.2.5 work, now folded in).  4 sites: 1969 / 1985 / 3374 / 4264.  Gates on runner-side event-bus access plumbing — may itself fan out. | **Audit revised** in §7c step 6.6.2 disposition audit below.  Audit reveals the original "4 sites" was incomplete (7 callback wiring sites total; 3 were missed: `set_continuation_callback` at 3415, `set_retry_callback` at 3430, `set_mid_turn_interrupt_callback` at 3440).  6 of 7 are pure emit-to-client; 5 of those don't even hit the daemon EventBus (unmapped event types).  Continuation_callback (3415) is the only daemon-logic-driven site.  **Step 6.6.2 collapses into 6.6.4** — the callback wiring naturally disappears alongside `_start_model_thread`'s migration to `session_send_message_threadsafe`.  No event-bus plumbing required.  No new RPC handlers needed (extend the existing `session.send_message` stream channel with notification frames). |
| **§7c step 6.6.3** | External consumer migrations: 8 `get_session()` callers in `session_manager.py` + `websocket.py` + `core.py`.  Migrate each to its target per the external-consumer table above (1 delete, 4 reuse-existing-RPC, 3 use-new-RPCs from 6.6.1).  Also drops the public `JaatoServer.get_session()` method. | **Audit revised** in §7c step 6.6.3 disposition audit below — original "8 callers" was a 9-site undercount.  Cross-grep of ALL `_jaato.get_session()` reach patterns (private + public) reveals **17 sites total** (8 missing in `session_manager.py`).  5 of the 17 need NEW RPC handlers (`session.append_history_message`, `session.snapshot_conversation_budget`, `session.set_parallel_tools_override`, `session.replay_messages`, `session.resolve_fork_point`).  Sub-decomposition: 6.6.3.0 audit-correction (THIS) → 6.6.3.1-.5 (5 new RPC handlers per 6.1 trio cadence) → 6.6.3.6 (full migration + public method drop). |
| **§7c step 6.6.4** | Atomic seat-flip moment.  Removes `self._jaato` field; absorbs WIRING deletions; migrates remaining transitively-load-bearing sites; migrates DEFER-§7c read sites; folds in 6.6.2 (architectural callbacks via send_message stream channel); deletes the public `JaatoServer.get_session()` method (already done in 6.6.3.6); collapses truthiness checks. | **Audit revised** in §7c step 6.6.4 disposition audit below.  Cross-grep reveals additional inventory not in prior audits: 4 `_jaato.get_runtime()` reaches in session_manager + websocket (post-step-4-first-pass migration; daemon-side runtime is daemon-tier per §4.2 — these collapse to `self._runtime` reads).  Sub-decomposition matches the reviewer's pre-laid 6.6.4.1-6.6.4.5 split with one addition (6.6.4.0 audit).  Each sub-commit independently reviewable. |

### Step 6.6.4 disposition audit

Mirroring cd3ecf20 / ac088e67 / 875e48bd / 4d53fd49 / 9f28f96d
/ 2752fd46 — the seventh audit in the §7c chain.  Per the
reviewer's "expect a 6.6.4 disposition audit alongside" framing
(parallel to 6.6.3 work stream).

#### Audit Step 1 — site inventory (post-§7c-step-6.6.3.6)

Cross-grep for `self._jaato\b` + `server._jaato\b` (per the §10
audit-discipline note 1) reveals the full remaining surface:

  - `core.py`: 66 references (counting truthiness + comments)
  - `session_manager.py`: 13 references
  - `websocket.py`: 4 references

  Total: ~83 references.

Active code (excluding doc comments):

**core.py** (35 active sites):

| Category | Sites | Disposition |
|---|---|---|
| **Field declaration** | 309 (``self._jaato: Optional[JaatoClient] = None``) | Delete in 6.6.4.5 |
| **Construction + connect** | 1558 (``self._jaato = JaatoClient(...)``), 1565 (``self._runtime = self._jaato.get_runtime()`` aliasing) | Refactor in 6.6.4.5 — daemon constructs JaatoRuntime directly |
| **Truthiness checks** | 766, 1406, 2054, 2359, 2407, 3327, 3394, 3872, 4091, 4117, 4352, 766, 2084 (×2), 4381 (×2) | ~14 truthiness checks — collapse in 6.6.4.5 |
| **WIRING calls** (per 6.4 audit) | 1886 (configure_plugins_only), 1905 (configure_tools), 1974 (set_gc_plugin), 2371 (set_session_plugin), 4295 (configure_tools mirror), 4304 (set_gc_plugin mirror) | Delete in 6.6.4.4 |
| **Architectural callbacks** (6.6.2 fold) | 1985 (_event_bus_tools), 2001 (instruction_budget init), 3395 (set_prompt_injected_callback), 4313 (instruction_budget auth-completion) | Notification-frame stream extension in 6.6.4.1 + 6.6.4.2 |
| **set_agent_identity / set_ui_hooks** | 2791, 2796 | Migrate via existing RPC OR drop (set_ui_hooks is daemon→runner state push; new RPC may be needed) — **flag for 6.6.4.2 review** |
| **DEFER-§7c reads** | 2055/2056 (get_context_usage/limit), 2084 (auth_info), 4353/4354 (get_context_usage/limit auth-mirror), 4381 (auth_info auth-mirror) | Migrate via existing snapshot_instruction_budget RPC + new auth_info RPC OR daemon-side auth_info read.  **Flag for 6.6.4.5 review.** |
| **Transitively load-bearing reads** | 1407 (get_tool_schemas — already migrated to public method, but the wrapper still goes through `_jaato._session`), 3631 (signal_completion private-attr read of `_signal_completion_called`), 3875/4093 (get_user_commands ×2), 3898 (execute_user_command), 4122 (get_model_completions) | Migrate in 6.6.4.5 — get_user_commands / execute_user_command / get_model_completions go through `self.registry` directly daemon-side OR new RPCs |
| **References-plugin split-out** | 767 (set_reference_authorizer daemon-side leg) | Stays — references-plugin runner-side migration sub-track |
| **send_message daemon-side leg** (§7c step 6.3 didn't drop) | 3539 (auto-continuation send_message), 3562 (formatter-feedback send_message; line offsets approximate) | Migrate to `session_send_message_threadsafe` in 6.6.4.3 |

**session_manager.py** (13 active sites):

| Category | Sites | Disposition |
|---|---|---|
| **Truthiness checks** | 2574, 2697, 3013, 3361, 3449 | Collapse in 6.6.4.5 |
| **runtime access** (§4.2 daemon-tier) | 2826 (get_runtime()), 3362 (get_runtime()), 3450 (get_runtime()) | Migrate to `server._runtime` — same pattern as core.py event_bus migration in §7c step 6.2 |
| **session reset / history reads** | 2575 (reset_session via existing RPC), 3014 (get_history via existing RPC), 2619 (get_context_usage via existing RPC) | Migrate via existing `session.reset` / `session.get_history` / `session.get_context_usage` RPCs in 6.6.4.5 |

**websocket.py** (1 active site post-6.6.3.6):

| Category | Sites | Disposition |
|---|---|---|
| **runtime access** | 2092 (get_runtime() in `_jaato_server._jaato.get_runtime()`) | Migrate to `_jaato_server._runtime` — same pattern |

#### Audit Step 2 — bucket totals

| Bucket | Site count | Sub-commit |
|---|---|---|
| Notification-frame protocol (NEW wire format) | 1 (the protocol itself) | **6.6.4.1** |
| Architectural callback collapse using new protocol | 4 callbacks + supporting refactor | **6.6.4.2** |
| send_message migration (daemon→runner) | 2 daemon-side `_jaato.send_message()` legs | **6.6.4.3** |
| WIRING deletions | 6 calls (configure_tools ×2 + configure_plugins_only + set_gc_plugin ×2 + set_session_plugin) | **6.6.4.4** |
| `_jaato`-field removal + truthiness collapse + cleanup | ~14 truthiness checks + ~10 transitively-load-bearing reads + DEFER-§7c reads + construction sites + runtime-access migrations + remaining 1 references-plugin split-out (kept) | **6.6.4.5** |

#### Audit Step 3 — new prerequisite RPC handlers needed?

Cross-check against the §7c step 6.6.3 audit's missing-method
discipline:

  - **Notification-frame protocol** (6.6.4.1): NOT a new RPC.
    Extends the existing `session.send_message` stream channel
    with a frame-type discriminator.  Per §7c step 6.6.2 audit
    (commit 9f28f96d): "the existing session.send_message stream
    channel already provides" the runner→daemon notification
    surface.  Wire format addition only; no new dispatch route.

  - **send_message migration** (6.6.4.3): NOT a new RPC.
    `session.send_message` already exists from §7b.2 (commit
    3ca3c14d); 6.6.4.3 just switches the daemon caller's leg
    from `_jaato.send_message()` to
    `session_send_message_threadsafe()`.

  - **WIRING deletions** (6.6.4.4): NO new RPCs needed.  Per
    §7c step 6.4 audit (commit ac088e67): "These sites pass
    rich Python plugin instances that can't cross an RPC
    boundary, but the runner-side session ALREADY receives the
    equivalent configuration via the SessionInitEnvelope."
    Sites just delete daemon-side; no runner equivalents
    needed.

  - **Field removal + cleanup** (6.6.4.5): mostly mechanical.
    But potential new RPCs for the `auth_info` reads and
    `get_user_commands` / `execute_user_command` /
    `get_model_completions` — see flagged dispositions above.

**Potential 6.6.4.5 new RPC needs** (audit can't fully resolve
without implementation):

  - `session.get_auth_info` (str return) — auth_info reads at
    sites 2084, 4381.  Daemon-tier alternative: read from
    `self._runtime._provider.get_auth_info()` directly.  Decide
    at 6.6.4.5 implementation.
  - `session.get_user_commands` / `session.execute_user_command`
    / `session.get_model_completions` — at sites 3875/4093,
    3898, 4122.  Daemon-tier alternative: walk `self.registry`
    directly for user-commands; that's the canonical surface.
    Decide at 6.6.4.5 implementation.
  - `session.set_ui_hooks` — at 2796.  Push hooks state to the
    runner-side session.  May need a new handler.

Three potential new prereq handlers; final count decided at
6.6.4.5 implementation (similar to how 6.6.3.0 audit refined
its "5 new handlers" estimate vs the original "3 new handlers"
in 6.6.1).

#### Audit Step 4 — sub-commit decomposition

Per the reviewer's pre-laid framing + this audit's findings:

| Sub-commit | Scope | Estimated tests |
|---|---|---|
| **§7c step 6.6.4.0** | Audit doc update (THIS commit). | 0 |
| **§7c step 6.6.4.1** | Notification-frame protocol on the `session.send_message` stream channel.  Frame-type discriminator (`output` vs `notification`); daemon-side wrapper grows a notification-frame demuxer.  No new dispatch route; wire format extension only. | **Shipped.** 15 new tests in `server/runner/tests/test_notification_frame_protocol.py`.  Promoted the pre-built `KIND_EVENT = "event"` scaffolding (in `envelope.py:32` since Phase 2; daemon read loop dispatched as a debug-log no-op) into a typed wire surface.  Zero new dispatch route; new `NotificationFrame` dataclass + `RunnerRPC.emit_notification()` runner-side helper + `OnNotificationCb` daemon-side callback registry threaded through `call()` + `session_send_message()`.  Wire-shape-reuse rationale honored — same JSON-line socket as StreamFrame, different `kind` discriminator. |
| **§7c step 6.6.4.2** | 7-callback collapse using the new notification protocol.  Daemon-side `_start_model_thread` callback wirings (4 + 3 = 7 sites flagged in §7c step 6.6.2 audit) all delete; runner-side session emits notification frames; daemon's `session_send_message` wrapper demuxes + invokes `server.emit(<Event>)` or `server._start_model_thread(...)` for continuation. | **Shipped — runner-side leg only (Option E split).**  15 new tests in `server/runner/tests/test_session_send_message_notification_emit.py`.  The daemon-side leg-drop + handler-install lands atomically in 6.6.4.3.  This commit lands the runner-side install/restore machinery in `RunnerRPC._handle_session_send_message`: install hook wires 6 session-callback shims (`set_instruction_budget_callback`, `set_prompt_injected_callback`, `set_continuation_callback`, `set_retry_callback`, `set_mid_turn_interrupt_callback`, plus `_event_bus_tools._on_subscribed` direct-attr write), each emitting a `NotificationFrame` with a well-known event_type (`instruction_budget_updated`, `prompt_injected`, `continuation_needed`, `retry`, `mid_turn_interrupt`, `events_subscribed`); finally-block restores originals.  Behavior-preserving: until 6.6.4.3 switches the daemon to `session_send_message_threadsafe`, the runner-side session never processes a turn so callbacks are dormant.  Defensive invariants: `hasattr`-gated install (rolling-upgrade safe); per-callback try/except around setter calls; per-callback try/except inside emission shims; per-key try/except in restore.  Audit-discipline tally: 8 audits, 8 silent-regression catches — Option E split caught the 6.6.4.2/6.6.4.3 coupling that would have orphaned-emit broken the daemon if landed atomically.  6 callback design choice (vs. 7 in original count) reflects that the §7c step 6.6.2 audit's "set_running_state_changed" callback is NOT part of `send_message`'s lifetime — it's set once at session-construction time, so it stays as a one-time install in 6.6.4.4 WIRING refactor, not the per-call install path. |
| **§7c step 6.6.4.3** | send_message daemon-side leg drop.  `_start_model_thread` switches from `server._jaato.send_message(...)` to `server._runner_rpc.session_send_message_threadsafe(...)`.  Couples tightly with 6.6.4.1+6.6.4.2 (the notification stream is what makes daemon-side _start_model_thread's local state mutations work post-migration).  **Split into 6.6.4.3a + 6.6.4.3b** per implementation-review audit (see below). | ~10-15 |
| **§7c step 6.6.4.3a** | Prerequisite `session.try_completion_nudge` RPC.  Single round-trip read+inc operation on `_signal_completion_called` / `_completion_nudges_fired` private state — collapses 3 daemon-side reaches (`core.py:3646/3647/3649`) into one RPC.  JaatoSession gains a public `try_completion_nudge(max_nudges)` method returning `(should_nudge: bool, nudges_fired: int)`; runner-side handler + daemon-side wrapper land alongside.  Matches the 6.6.3 missing-method cadence (prerequisite RPC ships before daemon-side migration). | **Shipped.** 17 new tests in `server/runner/tests/test_session_try_completion_nudge_rpc.py`.  Public method `JaatoSession.try_completion_nudge(max_nudges)` added (atomic check-and-increment, returns `(should_nudge, nudges_fired)`); runner handler `_handle_session_try_completion_nudge` with stage codes `decode`/`no_host`/`no_session`/`missing_method`/`call`; daemon wrapper `session_try_completion_nudge[_threadsafe]`.  Defensive contract: rejects bool `max_nudges` (Python int subclass blind spot); rolling-upgrade safe (missing-method on the session class surfaces as typed stage code, not crash). |
| **§7c step 6.6.4.3b** | Atomic seat-flip: leg drop + 9-callback collapse + 7-wiring delete + handler install.  Switches `_start_model_thread` to `session_send_message_threadsafe(...)`; deletes the 7 daemon-side `set_*_callback` wirings (4 init-time + 3 per-call); installs the daemon-side `on_notification` demuxer fanning to `server.emit(<Event>)` / `server._start_model_thread(...)`; adopts the 6.6.4.3a `try_completion_nudge` RPC for the completion-nudge guard.  Runner-side `_handle_session_send_message` extends to also wire `on_usage_update` + `on_gc_threshold` per-call kwargs as notification-emitting shims (closing the audit-caught kwargs-drop gap). | **Shipped.** 21 new tests in `server/tests/test_send_message_seat_flip_643b.py`.  Runner-side: 2 new event-type constants (`usage_update`, `gc_threshold`) + per-call kwarg shims `_make_usage_update_notification_shim` / `_make_gc_threshold_notification_shim` (no install/restore — kwargs only live for one call).  Daemon-side: `_build_send_message_notification_handler` factory returns 8-branch demuxer (instruction_budget_updated / prompt_injected / continuation_needed / retry / mid_turn_interrupt / events_subscribed / usage_update / gc_threshold + unknown-type forward-compat drop).  Atomic deletions: 4 init-time + 3 per-call setter wirings, both `_jaato.send_message(...)` legs in `_start_model_thread`, completion-nudge private-attr reaches.  AST-based regression-pin tests guard against re-introduction. |
| **§7c step 6.6.4.4** | WIRING deletions (per 6.4 audit).  6 daemon-side calls delete: `configure_plugins_only`, `configure_tools` ×2, `set_gc_plugin` ×2, `set_session_plugin`.  Daemon-side `_setup_session_plugin` may need a refactor — the `set_session_plugin` site's daemon-side hook (description-callback emission) might need preservation.  **Flag for 6.6.4.4 implementation review.**  **Narrowed to safe-only per 6.6.4.4 implementation-review audit (see below)**: scope reduced to 3 sites (`set_gc_plugin` ×2 + `set_session_plugin`).  The other 3 (`configure_*` calls) collapse with 6.6.4.5's atomic field removal because they have cascading downstream daemon-side read dependencies. | ~5-10 |
| **§7c step 6.6.4.5** | Atomic field removal + cleanup.  Removes `self._jaato` field; collapses ~14 truthiness checks; migrates the runtime-access reads (3 sites in session_manager + 1 in websocket) to `server._runtime`; migrates the DEFER-§7c reads; migrates auth_info / get_user_commands / execute_user_command / get_model_completions (with potential new RPC handlers OR daemon-side alternatives); refactors construction sites (1558, 1565); deletes the JaatoClient construction entirely (the daemon constructs JaatoRuntime directly).  **Now also absorbs the 3 deferred WIRING calls** (`configure_plugins_only`, `configure_tools` ×2) per the 6.6.4.4 audit's narrowing.  **Split into 4 sub-commits per 6.6.4.5 implementation-review audit (G3 + Refinement 1)**: 5a (truthiness + runtime reads), 5b (existing-RPC reads + daemon-runtime reads + get_tool_schemas cache), 5d (construction refactor), 5e (atomic field removal).  5c (set_agent_identity/set_ui_hooks RPCs) **eliminated** by missing-method audit — both are daemon-side state mutations that disappear with field removal, no new RPCs needed. | ~20-30 (test churn from removed write-both-specific tests) |

**Total: 9 implementation sub-commits + 1 audit (post-6.6.4.0 audit count).**

#### §7c step 6.6.4.5 implementation-review audit (mid-commit)

Pre-6.6.4.5-implementation cross-grep + dependency analysis caught
**~77 touch sites across 3 files** — by far the largest scope in
the §7c series.

**Site inventory:**

| File | Count | Categories |
|---|---|---|
| `core.py` | ~60 | Construction (3), WIRING (3 deferred from 6.6.4.4), reads (~25), truthiness checks (~14), comments (~15) |
| `session_manager.py` | 13 | `get_runtime()` ×4, `get_context_usage()` ×1, `reset_session()` ×1, `get_history()` ×1, etc. |
| `websocket.py` | 4 | All `get_runtime()` reads |

**G3 split decision (per "always split" policy):**

| Sub-commit | Scope | Risk |
|---|---|---|
| **5a** | Truthiness collapses + 5 `get_runtime()` → `self._runtime` reads | **Shipped.** 5 new tests in `server/tests/test_get_runtime_migration_645a.py`.  Migrated 4 call sites: `session_manager.py` ×3 (lines 2826, 3361-3364, 3449-3452), `websocket.py` ×1 (line 2092).  The 5th site (`core.py:1565`) is the populator — stays until 5d's construction refactor.  Truthiness collapses **deferred** to 5e (the `self._jaato` truthiness checks remain defensive for pre-init paths until the field itself is removed).  Behavior-preserving migration: `_runtime` is non-None iff `_jaato` was successfully connected. |
| **5b** | Existing-RPC reads (~10 `get_context_usage`/`get_context_limit`, 1 `get_turn_accounting`, 1 `reset_session`) + 8 daemon-side runtime reads (`auth_info`, `user_commands`, `model_completions`, `execute_user_command`) + `get_tool_schemas` via daemon-side `_runtime` cache (Refinement 2) | **Shipped — narrowed to 15 mechanical sites only.**  12 new tests in `server/tests/test_existing_rpc_migration_645b.py`.  Pre-implementation cross-grep verifying RPC/method existence caught a scope-mismatch in the 6.6.4.5 audit: `auth_info`, `get_user_commands`, `execute_user_command`, `get_model_completions` are session-tier methods (not runtime-tier as labeled) — need new RPCs, deferred to **5c (re-introduced)**.  `get_tool_schemas` cache also deferred to 5c after verifying `JaatoRuntime.get_tool_schemas()` returns the registry's full set (not the session-resolved subset that `JaatoSession.get_tool_schemas()` returns) — semantically different and would cause `signal_completion_in_surface` filter regressions.  Migrations: 4× `get_context_usage`, 6× `get_context_limit`, 1× `get_turn_accounting`, 1× `reset_session` → `set_initial_history` (semantically equivalent at the restore-from-disk site), 1× `get_history`, 2× `get_session().instruction_budget.snapshot()` → `session_snapshot_instruction_budget` RPC.  `_jaato.get_session()` count drops from 4 → 1 (the references-plugin deferred site at core.py:767 stays). |
| **5c** | (re-introduced) 4 user-command/auth_info migrations + `get_tool_schemas` migration via new RPCs.  Sub-decomposes per the 6.6.1/6.6.3 cadence: one new RPC handler per sub-commit with missing-method audit + ~12-15 tests each.  Candidates: `session.get_auth_info`, `session.get_user_commands`, `session.execute_user_command`, `session.get_model_completions`, `session.get_tool_schemas`.  **Path D adopted** (5 individual commits + 5c.0 audit prereq) — preserves per-handler bisectability + per-commit reviewability. |

#### §7c step 6.6.4.5c missing-method audit (pre-implementation)

Mirrors the 6.6.1.0 / 6.6.3 / 6.6.4.3 missing-method audit
discipline.  Verifies each proposed RPC handler has an actual
underlying public method on `JaatoSession` before committing to
the handler's existence in a plan.

| Proposed RPC | Underlying JaatoSession method | Status |
|---|---|---|
| `session.get_auth_info` | `JaatoSession.get_auth_info` | ❌ **MISSING.**  `JaatoClient.auth_info` (jaato_client.py:165) reaches `self._session._provider.get_auth_info()` (provider's method, accessed via session's private `_provider` attr).  Needs a public wrapper on JaatoSession that returns `self._provider.get_auth_info()` (or `""` when no provider).  Lands in 5c.1 alongside the RPC handler. |
| `session.get_user_commands` | `JaatoSession.get_user_commands` (jaato_session.py:7544) | ✅ EXISTS.  Returns `Dict[str, UserCommand]`.  Wire-shape note: `UserCommand` is a dataclass — needs serialization for transport (similar to `Message`/`Part` round-trip in 6.6.1). |
| `session.execute_user_command` | `JaatoSession.execute_user_command` (jaato_session.py:7548) | ✅ EXISTS.  Returns `tuple[Any, bool]` — `(result, share_with_model)`.  Wire-shape concern: `result` may be `HelpLines` (display-only, not JSON-serializable) for some commands; needs wire-shape decision at 5c.3 implementation.  Side effect: when `share_with_model=True`, the call injects into conversation history — already handled session-tier so the RPC is a single round-trip. |
| `session.get_model_completions` | `JaatoSession.get_model_completions` (jaato_session.py:3299) | ✅ EXISTS.  Takes `args: List[str]`, returns `List[CommandCompletion]`.  Wire-shape note: `CommandCompletion` is a small dataclass — straightforward serialization. |
| `session.get_tool_schemas` | `JaatoSession.get_tool_schemas` (jaato_session.py:2708) | ✅ EXISTS (confirmed earlier).  Returns `List[ToolSchema]` — the session-resolved subset (not `JaatoRuntime`'s full registry set).  Wire-shape note: `ToolSchema` may carry frozen-set traits + nested schemas; serialization shape needs verification at 5c.5 implementation. |

**Findings:**

1. **1 of 5 needs a missing-method add** — `JaatoSession.get_auth_info()`.
   Lands in 5c.1 alongside its RPC handler (matches 6.6.3.1 / 6.6.3.2 /
   6.6.3.3 / 6.6.4.3a cadence: "RPC + JaatoSession public method"
   in the same commit).

2. **3 wire-shape concerns** for follow-up decisions during
   per-handler implementation:
   - `UserCommand` dataclass serialization (5c.2)
   - `HelpLines`-or-other-non-JSON `result` shapes (5c.3)
   - `ToolSchema` with traits/nested schemas (5c.5)

3. **No false-positive RPCs** — all 5 candidates target real
   session-tier behavior daemon-side callers exercise today.

**Audit-discipline tally: 14 audits, 14 silent-regression catches.**
Today's audit caught the 1 missing public method (would have
failed at 5c.1's missing-method check otherwise) AND surfaced 3
wire-shape decisions per-handler reviewers will need.

**Path D sub-decomposition (5 commits + audit prereq):**

| Sub-commit | Scope |
|---|---|
| **5c.0** | This audit doc (no code) |
| **5c.1** | Add `JaatoSession.get_auth_info()` public method + `session.get_auth_info` RPC handler + daemon wrapper + tests + migrate 2 daemon callsites (core.py:2073, 4481).  **Shipped.** 14 new tests in `server/runner/tests/test_session_get_auth_info_rpc.py`.  Public method `JaatoSession.get_auth_info()` added (returns `_provider.get_auth_info()` with try/except defensive wrap; returns `""` when no provider).  Runner handler `_handle_session_get_auth_info` with stage codes `no_host`/`no_session`/`missing_method`/`call`.  Daemon wrapper `session_get_auth_info[_threadsafe]`.  2 daemon callsites migrated; both wrapped in try/except (display-only, fall back to `""` on transport error). |
| **5c.2** | `session.get_user_commands` RPC + wrapper + tests + migrate 2 daemon callsites (core.py:3977, 4195).  Wire-shape: serialize `UserCommand` dataclass.  **Shipped.**  13 new tests in `server/runner/tests/test_session_get_user_commands_rpc.py`.  Wire decision per the §7c step 6.6.4.5c.2 audit: dict-shape-only (Path B), not Message/Part round-trip (Path A) — pre-implementation grep verified `UserCommand` and `CommandParameter` are NamedTuples with primitive fields only (str/bool), no callables/Type[X]/class refs to strip.  Daemon-side wrapper reconstructs `UserCommand`/`CommandParameter` NamedTuple instances on receipt so existing `parse_command_args(cmd, raw_args)` works unmodified.  Handler callable stays runner-side; daemon invokes via `session.execute_user_command` (5c.3) — no callable crosses the wire either direction. |
| **5c.3** | `session.execute_user_command` RPC + wrapper + tests + migrate 1 daemon callsite (core.py:4000).  Wire-shape: handle `HelpLines`-or-other non-JSON result.  **Shipped.**  21 new tests in `server/runner/tests/test_session_execute_user_command_rpc.py`.  Wire decision per the §7c step 6.6.4.5c.3 mid-implementation audit: **Path A (per-type reconstruction) bounded to 3 shapes** — pre-implementation grep killed Path B (stringify-pre-wire) by surfacing 3 daemon-side structured-access sites: `isinstance(result, HelpLines)` + `result.lines` (4073), `isinstance(result, dict)` + `result.get("success")` / `result["current_model"]` (4078), `isinstance(result, dict)` IPC-return fallback (4099).  Wire format: `{"_kind": "HelpLines", "lines": [[text, style], ...]}` / `{"_kind": "dict", "value": <json-dict>}` / `{"_kind": "str", "value": <str>}` (everything-else coerced).  Daemon wrapper reconstructs HelpLines NamedTuple instances + re-tuples the lines (wire flattens to lists; HelpLines.lines contract is `List[tuple]`). |
| **5c.4** | `session.get_model_completions` RPC + wrapper + tests + migrate 1 daemon callsite (core.py:4224).  Wire-shape: serialize `CommandCompletion`.  **Shipped — scope expanded to 2 callsites.**  16 new tests in `server/runner/tests/test_session_get_model_completions_rpc.py`.  Mid-implementation audit caught a 5c.0 inventory miss: `command_router.py:1149` is a second daemon-side consumer of `_jaato.get_model_completions`, reads both `.value` AND `.description` for the model-subcommand-expansion autocomplete catalog.  Without 5c.4 migration of this site, model-subcommand expansion would silently stop working post-5e field removal.  Wire format mirrors 5c.2's UserCommand pattern: dict-shape-only (Path A) with daemon-side `CommandCompletion` NamedTuple reconstruction.  Audit-discipline tally: 16 audits, 16 silent-regression catches. |
| **5c.5** | `session.get_tool_schemas` RPC + wrapper + tests + migrate 2 daemon callsites (core.py:1407, 3721).  Wire-shape: serialize `ToolSchema` with traits/nested schemas.  **Shipped — Path D finale.**  17 new tests in `server/runner/tests/test_session_get_tool_schemas_rpc.py`.  Pre-implementation grep confirmed all 7 ToolSchema fields + nested EditableContent fields are JSON-encodable; `traits: FrozenSet[str]` round-trips as `traits: List[str]` on wire and back to FrozenSet on receipt.  Daemon wrapper reconstructs ToolSchema + EditableContent dataclass instances; daemon callsites at core.py:1407 (`for schema in ... .name + .category`) and core.py:3759 (`signal_completion_in_surface` filter via `getattr(t, 'name', ...)`) work unmodified.  Both wrapped in try/except for best-effort fallback to `[]`.  All 5c sub-commits complete; Path D's 5-handler decomposition closes. |
| **5d** | Construction refactor — daemon constructs `JaatoRuntime` directly; remove `self._jaato.get_runtime()` indirection at construction site (lines 1550-1565).  Pre-audit per Refinement 3: verify `JaatoRuntime.__init__()` signature can be called daemon-direct.  **Shipped — Path A transitional.**  9 new tests in `server/tests/test_construction_refactor_645d.py`.  Daemon now constructs `JaatoRuntime` directly inside `_run_connect_provider` (preserves the ThreadPoolExecutor concurrency with plugin loading).  Three new substages: `create_runtime` + `runtime_connect` (inside the threadpool task) + a post-join `self._runtime.configure_plugins(self.registry, self.permission_plugin, self.ledger)` (so daemon-side reads on `self._runtime.registry` work against the daemon-direct runtime, not JaatoClient's internal one).  JaatoClient construction stays transitionally — 5 unguarded daemon-side calls (`configure_plugins_only`, `configure_tools` ×2, `set_agent_identity`, `set_ui_hooks`) still operate on it; 5e drops the JaatoClient + dependent calls atomically.  `JaatoRuntime` added to the `from shared import ...` block.  Confine-context-factory propagation now operates on the daemon-direct runtime. |

#### §7c step 6.6.4.5d disposition audit (pre-implementation)

Mirrors the 5c.0 audit shape per the audit-discipline pattern.
The construction refactor is architecturally distinct from 5a-5c
(which were all uniform RPC-handler-and-wrapper migrations); audit
catches what mechanical-instinct would miss.

**Q1 — `JaatoRuntime.__init__()` signature widening needed?**

Current signature (jaato_runtime.py:228):

```python
def __init__(self, provider_name: str = "google_genai",
             workspace_path: Optional[Path] = None,
             config_root: Optional[str] = None,
             instruction_token_cache: Optional[InstructionTokenCache] = None):
```

4 args, all primitives + Optional types.  Zero `JaatoClient`
references inside `__init__` body — fields set are local
(`_provider_name`, `_workspace_path`, `_config_root`,
`_provider_config = None`, etc.).  No state-set-by-Client
expectations. **Daemon-direct construction is feasible.**
**Refinement 3 closes**: no 5d.0 prereq commit needed.

**Q2 — `JaatoClient.connect()` side-effects beyond runtime
construction + auth?**

Body of `connect()` (jaato_client.py:399-438) — 5 steps:

1. Resolves `model` (with `MODEL_NAME` env-var fallback).
2. Validates `_provider_name` is set.
3. Constructs `JaatoRuntime(provider_name, ws, config_root, instruction_token_cache)`.
4. Calls `self._runtime.connect(project, location)` — sets the
   provider config + `_connected = True` on the runtime.
5. Stores `_model_name`, `_project`, `_location` on `JaatoClient`
   for SDK-side properties (`is_connected`, `model_name`).

Steps 1-2 are arg validation, replicable daemon-side trivially.
Steps 3-4 are runtime construction + connect, directly replicable.
**Step 5 is JaatoClient-state only — daemon doesn't need it**
(`JaatoServer` already has `self._model_name`/`self._model_provider`
on its own state).

**Q3 — Auth flow re-wiring?**

`JaatoClient.verify_auth()` is a thin wrapper over
`JaatoRuntime.verify_auth(allow_interactive, on_message,
provider_name, plugin_configs)`.  Auth implementation **already
lives on JaatoRuntime** — daemon can call
`self._runtime.verify_auth(...)` directly post-construction.
**No auth re-wiring needed.**

**Q4 — SDK consumers (Path B contract preservation)?**

Cross-grep of `JaatoClient(...)` constructor calls across the
codebase (excluding tests + docs):

| Site | Disposition |
|---|---|
| `core.py:1557` | The daemon-side site being migrated in 5d |
| (none else) | No other non-test production callers |

Docs in plugin READMEs (`web_search`, `subagent`,
`filesystem_query`, etc.) demonstrate `JaatoClient()` for
**external SDK callers** — Path A doesn't touch them since
JaatoClient itself stays in place for those external use cases.
**JaatoClient remains the SDK facade.**

**Q5 — Actual construction site (re-grep per 5c.4 inventory-miss
lesson)?**

`_run_connect_provider()` (core.py:1547-1587) runs in a
ThreadPoolExecutor thread, concurrent with `_run_load_plugins()`
(saves ~100-200ms during bootstrap).  4 substages:

1. `create_client` — constructs `JaatoClient(...)` (line 1557)
2. `client_connect` — `self._jaato.connect(project, location, model)` (line 1565)
3. Implicit alias — `self._runtime = self._jaato.get_runtime()` (line 1572)
4. `_pre_init_confine_context_factory` propagation (lines 1578-1585)

**No adjacent state-setup found that requires co-migration.**
The confine-context-factory propagation operates on `self._runtime`
(post-alias) — unchanged by the refactor.

### Path A vs Path B verdict

**Path A wins.** Direct `JaatoRuntime(...)` construction
daemon-side with `self._runtime.connect(project, location)`
invoked separately.  Reasons:

1. `JaatoRuntime.__init__` is already fully decoupled from
   JaatoClient (zero client references in init body) — no helper
   extraction needed.
2. `JaatoClient.connect()` does only 5 things; 2 are runtime
   construction + connect (cleanly replicable daemon-side); 3 are
   JaatoClient-side state (not needed daemon-side, JaatoServer has
   its own).
3. `JaatoRuntime.verify_auth` exists as a runtime-tier method —
   daemon can call it directly.
4. SDK consumers don't touch the daemon path; JaatoClient stays
   as the external-SDK facade.
5. **Path B over-engineers**: extracting a `_build_runtime` helper
   benefits only if JaatoClient and the daemon share more code,
   but the audit shows they share only the trivial 2-step
   "construct + connect" pattern.

### Audit findings

| # | Finding | Disposition |
|---|---|---|
| 1 | `JaatoRuntime.__init__` daemon-direct-callable as-is | Refinement 3 closes; no 5d.0 prereq |
| 2 | `JaatoClient.connect()` body decomposes cleanly: 2 runtime ops + 3 client-state stores | Daemon replicates only the 2 runtime ops |
| 3 | Auth flow already on JaatoRuntime | Daemon calls `self._runtime.verify_auth()` directly |
| 4 | Only 1 production callsite (core.py:1557) — JaatoClient stays SDK facade | Path A safe |
| 5 | Concurrent ThreadPoolExecutor stage — must preserve concurrency | Swap construction inside same `_run_connect_provider` thread function |

**Audit-discipline tally: 18 audits, 18 silent-regression catches.**
Today's audit disposed Refinement 3 cleanly (no 5d.0 prereq) AND
validated Path A's safety over Path B's helper-extraction.

5d ships as a single-commit migration; no further split needed.
Next concrete step: 5d implementation.
| **5e** | Atomic `_jaato`-field removal + truthiness collapses + 3 deferred WIRING drops + drop `set_agent_identity` / `set_ui_hooks` calls (per Refinement 1's missing-method audit).  **Shipped — seat-flip closure.**  9 new tests in `server/tests/test_seat_flip_complete_645e.py`.  Every deletion independently justified by a prior audit (audit-cross-reference table in commit message).  Field removed from JaatoServer; `JaatoClient` import dropped; 4 truthiness-check guards collapsed to unconditional bodies; `set_reference_authorizer` daemon-side leg dropped (runner-RPC forwarder already handles it); session_manager.py's `if state.metadata.get('subagents') and server._jaato:` truthiness pivoted to `server._runtime`.  §7c step 6.6.4 closes; seat-flip complete. |

**Refinement 1 — Missing-method audit for 5c (eliminated):**

| Site | `JaatoSession` underlying method | `JaatoClient` body | Verdict |
|---|---|---|---|
| `core.py:2793` `_jaato.set_agent_identity(...)` | ❌ doesn't exist (`set_agent_context` exists but for runner-side) | Mutates daemon-side `_agent_id` / `_agent_name`; doesn't propagate to session | **No new RPC needed** — daemon-side state mutation; equivalent state already on `JaatoServer._main_agent_id`.  Call drops with field removal. |
| `core.py:2798` `_jaato.set_ui_hooks(hooks)` | ✅ exists (`set_ui_hooks(hooks, agent_id)` at jaato_session.py:778) | Sets daemon-side state + propagates to session | **No new RPC needed** — runner-side `_ui_hooks` is already None post-6.6.4.3b (no runner-side wiring exists; cross-grep confirmed); all `if self._ui_hooks:` callsites null-guard.  The `AgentUIHooks` object isn't serializable across the wire anyway. Daemon-side state lives on `JaatoServer` / `subagent_plugin` (registered separately). |

5c eliminated; G3 collapses from 7 sub-commits to 4.

**Refinement 2 — `get_tool_schemas()` strategy: daemon-side `_runtime` cache.**

Per §4.2 tier classification, the plugin registry is daemon-tier.
Daemon already has the resolved plugin list at bootstrap-envelope-
construction time.  Cache populated at session-init from the
envelope's resolved plugin list; read-only, stable across session
lifetime, single populate-point.  Zero new RPC handlers; smaller
blast radius for 5b.  Cache invalidation: not needed today (registry
doesn't mutate mid-session); flag if it ever does.

**Refinement 3 — 5d construction-refactor pre-audit (deferred to 5d implementation):**

Verify `JaatoRuntime.__init__()` can be called daemon-direct
(currently invoked via `JaatoClient`).  If `JaatoRuntime` expects
construction-time `JaatoClient` state, 5d needs a refactor of
`JaatoRuntime.__init__` first.  Likely small but unverified —
flag for 5d implementation review.

**Audit Findings:**

| # | Finding | Disposition |
|---|---|---|
| 1 | Scope is ~10× any prior sub-commit (~77 touch sites) | Split via G3 (4 sub-commits) |
| 2 | `set_agent_identity` / `set_ui_hooks` don't need new RPCs (daemon-side state only) | 5c eliminated |
| 3 | Runner-side `_ui_hooks` is already None post-6.6.4.3b — runner-side tool lifecycle events silently no-op via that path.  Pre-existing gap. | Orthogonal to 6.6.4.5; flag for follow-up alongside Finding 2's description-callback fix |
| 4 | `_runtime` lifecycle untangling: post-removal, daemon constructs `JaatoRuntime` directly (was `self._jaato.get_runtime()`).  Runtime persists daemon-side for event_bus, plugin registry, user commands, auth state. | Handled in 5d |

**Audit-discipline tally: 12 audits, 12 silent-regression catches.**
Today's audit caught the scope-explosion (G3 split) AND the
"RPCs not actually needed" simplification (5c eliminated, would
have over-engineered with 2 unnecessary RPC handlers + 24-30 tests)
AND surfaced Findings 3-4.

#### §7c step 6.6.4.4 implementation-review audit (mid-commit)

Pre-6.6.4.4-implementation cross-grep of the 6 WIRING sites +
dependency analysis caught **2 silent-regression risks**:

| Finding | Evidence | Disposition |
|---|---|---|
| **4 of 6 sites have cascading daemon-side read deps.** Dropping `configure_*` calls removes `_runtime.configure_plugins(...)` + `_session` creation; ~15 downstream daemon-side reads (`_jaato.get_session()`, `get_history()`, `get_context_*()`, `get_turn_accounting()`, etc.) would fail. | `core.py:767, 1407, 1993, 2035, 2036, 2577, 2731, 3470, 3475, 3601, 3602, 3955, 3978, 4173, 4202, 4393, 4428, 4429`; `session_manager.py:2619, 3014` | Defer 4 unsafe sites (`configure_plugins_only`, `configure_tools` ×2) to 6.6.4.5 alongside the read-site migration that's already planned there.  6.6.4.4 narrows to 3 safe-only sites (`set_gc_plugin` ×2 + `set_session_plugin`). |
| **Daemon-side description-callback hook silently broken post-6.6.4.3b** (pre-existing, surfaced by seat-flip).  `_setup_session_plugin` wires `on_description_changed` on the daemon-side `session_plugin` instance, but the model invokes `set_description` runner-side → fires runner-side instance's callback → daemon never sees it. | `core.py:2360-2367` (description-callback wiring) + `shared/plugins/session/file_session.py:743-747` (callback fires from inside tool execution) | Orthogonal to 6.6.4.4 — pre-existing regression from 6.6.4.3b.  Fix needs either a new `description_updated` notification-frame event_type (extends 8-event protocol to 9) OR a runner-side `set_description_callback` wired through `_install_session_notification_callbacks`.  Recommend deferring to a follow-up step (e.g., 6.6.4.6 or fold into 6.6.4.5). |

Audit-discipline tally: **10 audits, 10 silent-regression catches.**
Today's audit caught the cascade-deps issue (would have broken ~15
daemon-side reads with a unilateral 6-site WIRING-only deletion)
and surfaced Finding 2 (pre-existing description-callback
regression hidden by the 6.6.4.3b seat-flip).

Decision: **F2** per user's "always split" policy.  6.6.4.4 narrows
to 3 safe-only WIRING deletions; the 3 unsafe ones collapse with
6.6.4.5's atomic field removal where their dependent reads are
already planned for migration.

#### §7c step 6.6.4.3 implementation-review audit (mid-commit)

Pre-6.6.4.3-implementation cross-grep of `_start_model_thread`
+ `_jaato.send_message(...)` legs caught **3 silent-regression
risks** beyond §7c step 6.6.2's "7 callback" inventory:

| Finding | Evidence | Disposition |
|---|---|---|
| **9-callback collapse, not 7.** `on_usage_update` and `on_gc_threshold` are passed as per-call kwargs, not setters; missed by 6.6.2's setter-shaped grep. | `core.py:3511-3512`, `3534-3535` | Rolled into 6.6.4.3b — runner-side `_handle_session_send_message` wires both as notification-emitting shims; daemon demuxer fans to `ContextUpdatedEvent` and `SystemMessageEvent`. |
| **Runner-side handler currently drops these kwargs.** Pre-existing latent gap — no caller exercises send_message via runner-RPC yet. | `rpc.py:_handle_session_send_message` calls `session.send_message(prompt, on_output=on_output)` only | Closed in 6.6.4.3b (extends call-through with new shims). |
| **`_signal_completion_called` / `_completion_nudges_fired` reaches.** 3 sites, including a daemon-side `+= 1` mutation of runner-side state. Pre-flagged in design doc. | `core.py:3646`, `3647`, `3649` | New RPC `session.try_completion_nudge` lands in 6.6.4.3a. Runner-side counterpart in `subagent/plugin.py:2881-2888` stays runner-tier (subagent's own loop). |

Audit-discipline tally: **9 audits, 9 silent-regression catches.**
Today's audit caught the 7→9 callback miss (setter-shape blind
spot) and the runner-side kwargs-drop gap.

Decision: split per user policy ("always split — better to
manage").  Matches the 6.6.3 cadence (prerequisite RPC ships
first, daemon migration follows).  Each sub-commit independently
reviewable.

#### Audit Step 5 — what stays in place (split-outs)

| Site | Reason kept |
|---|---|
| `core.py:767` (`set_reference_authorizer` daemon-side leg) | §7c step 6.3 split-out: references-plugin runner-side migration is a separate sub-track.  Daemon-side leg drops with that migration, not 6.6.4. |
| `core.py:3631` (`_signal_completion_called` private-attr read) | Daemon-side counter in `_start_model_thread` recovery path.  Either gets migrated alongside _start_model_thread refactor in 6.6.4.3 (move counter to JaatoServer state) or migrated to a new RPC.  **Flag for 6.6.4.3 implementation review.** |

#### What this audit decides

  - §7c step 6.6.4 expands from 1 commit to 6 (audit + 5
    implementations).  Same pattern as prior audits.

  - The 6.6.4.1-6.6.4.5 reviewer-pre-laid split holds, plus
    audit-required 6.6.4.0 (this commit).

  - Notification-frame protocol is wire-format-only (no new
    dispatch route).  Per §7c step 6.6.2 audit's
    inverse-virtue: reuse `session.send_message`'s existing
    stream channel.

  - Up to 3 potential new prerequisite RPC handlers in
    6.6.4.5 (auth_info / get_user_commands /
    set_ui_hooks).  Audit-of-record discipline applied at
    implementation time per 6.6.3 missing-method pattern.

  - Audit-discipline tally: 7 audits, 7 silent-regression
    catches.  This audit caught the `_jaato.get_runtime()`
    inventory miss (4 sites in session_manager + websocket)
    that prior audits missed.

#### What this audit does NOT decide

  - Notification-frame wire format (frame-type discriminator
    encoding).  Decide at 6.6.4.1 implementation.

  - Whether `_signal_completion_called` counter migrates to
    JaatoServer state or new RPC.  Decide at 6.6.4.3
    implementation.

  - Whether `_setup_session_plugin`'s daemon-side
    description-callback hook is preserved or rewired.  Decide
    at 6.6.4.4 implementation.

  - Specific 6.6.4.5 new-handler count (auth_info /
    get_user_commands / set_ui_hooks): 0-3 depending on
    implementation choices.  Apply 6.6.3 missing-method
    pattern per-handler.

#### Missing-method finding (§7c step 6.6.1 prerequisite check)

Per the §7b.3 audit lesson + the §7c step 6.4 audit lesson —
"verify that proposed RPC handlers have actual underlying
methods before committing to their existence in a plan" — the
3 proposed RPC handlers in step 6.6.1 were spot-checked against
JaatoSession's public surface.  Findings:

| Proposed RPC | Underlying method | Status |
|---|---|---|
| `session.set_initial_history` | `JaatoSession.set_initial_history(messages: List[Message])` | ✅ **EXISTS** (jaato_session.py:8252).  Public.  RPC handler can be built directly. |
| `session.restore_turn_accounting` | `JaatoSession.restore_turn_accounting(turns)` | ❌ **MISSING.**  Daemon's `session_manager.py:2558-2559` does direct private-attr write: `jaato_session._turn_accounting = list(state.turn_accounting)`.  No public method on JaatoSession. |
| `session.restore_conversation_budget` | `JaatoSession.restore_conversation_budget(snapshot)` | ❌ **MISSING.**  Daemon's `session_manager.py:2592-2593` reaches `jaato_session.instruction_budget.restore_conversation_from_snapshot(state.budget_state)` — the underlying `restore_conversation_from_snapshot` exists on `InstructionBudget` (instruction_budget.py:401), but no public wrapper exists on JaatoSession. |

**Implication:** step 6.6.1 cannot ship as 3 RPC-handler
commits in sequence.  The 2 missing public methods must land
first, otherwise the RPC handlers would either:

  - Reach into private state from the runner side (an
    encapsulation violation that would fail the same audit
    discipline that produced §7c step 3a / 3b's public-method
    additions), OR

  - Ship a parallel-daemon-side path with its own private-attr
    write while the RPC handler does the same on the runner
    side (duplicate-implementation risk that exactly mirrors
    the §7b.3 withdrawal's "duplicates existing infrastructure"
    failure mode — except inverted, with no existing
    infrastructure to duplicate).

**Resolution:** sub-decompose 6.6.1 into 4 commits per the
sub-decomposition table above:

  - 6.6.1.0  encapsulation-cleanup commit (analogous to step
             3a + 3b): add public methods, migrate the daemon
             session_manager call sites to use them.
  - 6.6.1.1  `session.set_initial_history` RPC (no
             prerequisite needed; method already exists).
  - 6.6.1.2  `session.restore_turn_accounting` RPC (gated on
             6.6.1.0).
  - 6.6.1.3  `session.restore_conversation_budget` RPC (gated
             on 6.6.1.0).

Total: 4 commits for what was originally 1.  Worker tally for
this finding: 4 audits, 4 silent-regression catches.

#### What this audit decides:

  - The 6.5/6.6 boundary is fuzzy — only 4 sites are truly
    "clean" (read from `_runtime` directly, no `_session`
    reach).  The other 5 originally-scoped 6.5 sites
    (`auth_info` ×2, `get_user_commands` ×2, plus
    `execute_user_command`, `get_model_completions`) are
    transitively load-bearing and fold into 6.6.4.

  - Step 6.6 is materially larger than the original
    "mechanical diff" framing — it absorbs WIRING (6.4),
    architectural callbacks (was 6.2.5), external consumer
    migrations (this audit found 8), DEFER-§7c reads (path-A
    fold of step 5), 5 transitively-load-bearing 6.5 sites,
    and the construction-site refactor.

  - **3 new runner-RPC handlers must land first** (step 6.6.1)
    before the external consumer migrations in 6.6.3 can
    proceed.  This is the prerequisite chain.  Sub-decomposed
    into 6.6.1.0/.1/.2/.3 per the missing-method finding above.

  - The architectural callbacks (6.6.2) and external consumer
    migrations (6.6.3) can land independently AFTER 6.6.1, in
    either order.  6.6.4 is the atomic seat-flip and lands
    last.

#### What this audit does NOT decide:

  - Whether `set_initial_history`'s wire shape can serialize
    Messages cleanly across the RPC boundary.  Provider-
    specific Message subtypes may need a `Message.to_dict()` /
    `from_dict()` round-trip.  Decide at step 6.6.1
    implementation time.

  - Whether the references-plugin runner-side migration (which
    enables dropping the `set_reference_authorizer` daemon leg
    at site 767 — split out per §7c step 6.3) lands during
    §7c or post-§7c.  Out of audit scope; flagged for a
    separate sub-track.

### Step 6 disposition audit

Per-`get_session()`-reader audit (post-step-4-first-pass; line
numbers as of commit `7c34f218`):

| Site | Operation | Disposition | New runner-RPC needed? |
|---|---|---|---|
| 462 | reads `session._runtime.event_bus` | Migrate to `self._runtime.event_bus` | No — already on runtime |
| 610 | calls `session._executor.set_apparmor_context(...)` | **Delete** — daemon doesn't execute tools post-§7c | No |
| 697 | calls `session._executor.set_runtime_limits(...)` | **Delete** — same reason | No |
| 725 | calls `session.set_reference_authorizer(authorizer)` | **Forward via new RPC** | **Yes** — `session.set_reference_authorizer` |
| 1091 | reads `session.instruction_budget.snapshot()` for `emit_current_state` | **Forward via new RPC** | **Yes** — `session.snapshot_instruction_budget` |
| 1891 | configures `_event_bus_tools` callback wiring | **Architectural** — move runner-side; gates on runner-side event-bus access | Possibly |
| 1907 | configures `instruction_budget` callback wiring | **Architectural** — same as 1891 | Possibly |
| 3238 | calls `session.inject_prompt(text, ...)` | **Forward via new RPC** | **Yes** — `session.inject_prompt` |
| 3273 | calls `session.set_prompt_injected_callback(...)` | **Architectural** — callback registration runner-side | Possibly |
| **§7c step 7** | Wire `PromptOperatorHandler` into the daemon's session bootstrap + thread `respond_to_X` through `prompt_operator_handler.resolve_response()`.  Response-handler rewiring from §7b.3 lands here.  Gated on permission-plugin runner-side activation (a separate sub-track that doesn't block steps 1-6). | Pending. |

**`JAATO_RUNNER_HOSTS_SESSION` flag lifetime** (peer-review M4):
the v5 plan §3.3b N4 specified the flag "lands and is removed
within the same PR."  As implemented, the flag was live across
~24 §3.3c precursor commits — a **scope expansion from v5 N4**
(single-PR transitional flag → multi-PR transitional flag with
explicit removal commit).  The flag was removed in **§7c step 1**
(this doc-revision's commit).  Operators never see the flag on a
released server version (still upholds N4's intent — the
user-facing concern was preventing operator-visible feature-flag
accumulation, which this doesn't violate).

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

### Step 6.6.2 disposition audit

Mirroring cd3ecf20 / ac088e67 / 875e48bd / 4d53fd49 — the
fifth audit in the §7c chain.  Per the reviewer's framing
("§7c step 6.6.2 ... is the architectural decision point I
called for ... 6 architectural shapes ... narrowed to the best
two ... worth a 30-minute audit before any code commits").

#### Audit Step 1 — site inventory (the "4 sites" was incomplete)

The §7c step 6.6 disposition audit (commit 875e48bd) flagged 4
callback rewire sites: 1969 / 1985 / 3374 / 4264.  Re-survey
post-§7c-step-6.5 turned up **3 additional sites in
`_start_model_thread`** that the original audit missed:

| Site | Callback | What it triggers |
|---|---|---|
| 1996 | `_event_bus_tools._on_subscribed = ...` | `server.emit(EventsSubscribedEvent)` |
| 2011 | `session.set_instruction_budget_callback(cb)` (init path) | `server.emit(InstructionBudgetEvent)` |
| 3391 | `session.set_prompt_injected_callback(cb)` | `server.emit(MidTurnPromptInjectedEvent)` |
| 3415 | `session.set_continuation_callback(cb)` | `server._start_model_thread(child_messages)` + `server.emit(AgentStatusChangedEvent)` + `server._pending_continuation = ...` |
| 3430 | `session.set_retry_callback(cb)` | `server.emit(RetryEvent)` |
| 3440 | `session.set_mid_turn_interrupt_callback(cb)` | `server.emit(MidTurnInterruptEvent)` + tracing |
| 4291 | `session.set_instruction_budget_callback(cb)` (auth-completion mirror of 2011) | `server.emit(InstructionBudgetEvent)` |

**Total: 7 callback sites** (not 4).  Missing 3 sites would
have produced silent callback gaps post-seat-flip — the same
class of issue the prior 4 audits caught.  Audit-of-record
discipline catches this kind of inventory error.

#### Audit Step 2 — E filter (vestigial / daemon-side consumer survey)

Per the reviewer's recommendation: "audit each site for
vestigial-ness; for genuinely-load-bearing sites, push callback
runner-side ... sites that can't push runner-side (daemon-side
consumers like reactor) get the Path C stream."

Three questions per site:

1. Does the daemon-side `JaatoServer.emit(<Event>)` route the
   event through the daemon's EventBus (for plugin / reactor
   subscribers)?
2. Are there any actual daemon-side subscribers for these
   event types (jaato-premium reactor, plugin
   `subscribe_to_bus`, etc.)?
3. Does the callback do any non-emit daemon-side work?

**Per-event-type bus mapping** (read from `_SERVER_TO_BUS` at
`server/core.py:127`):

| Event type | In `_SERVER_TO_BUS`? | If yes, bus type |
|---|---|---|
| `EventsSubscribedEvent` | **No** — unmapped | Goes directly to client; never touches bus |
| `InstructionBudgetEvent` | **No** — unmapped | Direct to client |
| `MidTurnPromptInjectedEvent` | **No** — unmapped | Direct to client |
| `RetryEvent` | **No** — unmapped | Direct to client |
| `MidTurnInterruptEvent` | **No** — unmapped | Direct to client |
| `AgentStatusChangedEvent` (within continuation_callback) | **Yes** | `BusEventType.AGENT_STATUS_CHANGED` |

**Daemon-side subscribers for these event types** (grep
`subscribe.*<EventName>` across server/ + shared/):

  - `EventsSubscribedEvent`: 0 subscribers
  - `InstructionBudgetEvent`: 0 subscribers
  - `MidTurnPromptInjectedEvent`: 0 subscribers
  - `RetryEvent`: 0 subscribers
  - `MidTurnInterruptEvent`: 0 subscribers
  - `AgentStatusChangedEvent` (bus): subscribers TBD (reactor
    in jaato-premium per CLAUDE.md; not in OSS tree)

**Per-site classification:**

| Site | Pure emit? | Daemon-side bus consumer? | Daemon-side logic? | E-filter verdict |
|---|---|---|---|---|
| 1996 | ✅ Yes (EventsSubscribedEvent) | No (unmapped) | None | **Vestigial** for daemon — pure client notification |
| 2011 | ✅ Yes (InstructionBudgetEvent) | No (unmapped) | None | **Vestigial** for daemon — pure client notification |
| 3391 | ✅ Yes (MidTurnPromptInjectedEvent) | No (unmapped) | None | **Vestigial** for daemon — pure client notification |
| 3415 | ❌ No — emits AgentStatusChanged AND calls `_start_model_thread` AND writes `_pending_continuation` | Possibly (AgentStatusChanged is mapped) | **Yes** — restarts model thread daemon-side | **Load-bearing** — triggers daemon-side action |
| 3430 | ✅ Yes (RetryEvent) | No (unmapped) | None | **Vestigial** for daemon — pure client notification |
| 3440 | ✅ Yes (MidTurnInterruptEvent) | No (unmapped) | None | **Vestigial** for daemon — pure client notification |
| 4291 | ✅ Yes (InstructionBudgetEvent) | No (unmapped) | None | **Vestigial** for daemon (mirror of 2011) |

**6 of 7 callbacks are vestigial-for-daemon.**  Their entire
purpose is "fire `server.emit(<Event>)` to fan out to clients."
Zero daemon-side reactors / plugin subscribers / jaato-premium
hooks consume any of these 5 unmapped event types.

The 1 load-bearing site (continuation_callback at 3415) needs
its own treatment — see Step 4 below.

#### Audit Step 3 — pre-existing context: `_start_model_thread` is daemon-side

The 7 callback wiring sites all live inside `_start_model_thread`
(except 2011 + 4291 which run during `initialize()` /
`_check_auth_completion`).  This method runs daemon-side.  It:

  1. Wires the callbacks onto `_jaato.get_session()`.
  2. Calls `server._jaato.send_message(prompt, on_output=..., ...)`
     at line 3503 (and 3526 for the formatter-feedback
     continuation loop).

**Critical pre-existing finding**: §7b.2 (commit 3ca3c14d)
shipped the `session.send_message` runner-RPC handler + daemon
wrapper, BUT did not migrate the daemon-side
`_jaato.send_message()` callers.  The daemon's
`_start_model_thread` still uses the in-process daemon-side
session for actual model execution.  This was the write-both-
without-leg-drop pattern §7b.1 used (with daemon-side leg
intact); the leg-drop is part of step 6.6.4.

When step 6.6.4 removes `_jaato`, `_start_model_thread` MUST
switch to `session_send_message_threadsafe(...)`.  At that
point:

  - The daemon no longer holds the session; can't install
    in-process callbacks on it.
  - The runner-side session emits its callbacks runner-side.
  - The daemon needs to receive notifications somehow to do its
    follow-up actions (emit-to-client; restart model thread).

This re-frames the question: **the callback rewires aren't an
"event-bus plumbing" decision; they're a "how does runner-side
send_message notify the daemon of intra-session events" decision.**
The existing `session.send_message` RPC already has a streaming
output channel (per the §7b.2 commit message: "Streams output
chunks via the existing stream-frame channel").  Extending it to
multiplex notification frames is the obvious shape.

#### Audit Step 4 — architectural decision per site

Surveyed 6 alternatives (A bidirectional event RPC; B daemon-
side callback registry with runner→daemon push; C stream-based
subscription; D inverted runner-side callbacks; E vestigial-
audit; F event-bus mirror).  Per-site decision:

**Sites 1996, 2011, 3391, 3430, 3440, 4291 (6 vestigial-for-daemon)**:

  Path: extend the existing `session.send_message` RPC's stream
  channel with **notification frames** alongside output frames.
  Daemon-side wrapper demuxes notification frames and invokes
  `server.emit(<Event>)`.  This is **Path C (stream-based
  subscription)** with the **E-filter applied first**:

  - Runner-side session installs its OWN callbacks (in-process
    runner-side); each callback writes a notification frame
    onto the existing stream channel.
  - Daemon-side wrapper for `session_send_message_threadsafe`
    grows a notification-frame demuxer that reconstructs the
    appropriate `Event` instance and calls `server.emit(...)`.
  - **No new RPC handlers needed.**  Zero new wire surface;
    extends the existing stream-frame channel.

**Site 3415 (continuation_callback — load-bearing)**:

  Path: same notification-frame channel, but with a different
  daemon-side handler.  Continuation triggers "start another
  model thread" daemon-side, which post-seat-flip means "call
  `session_send_message_threadsafe(...)` again."

  - Runner-side session emits a `continuation_needed` notification
    frame with the child_messages text.
  - Daemon-side wrapper for `session_send_message_threadsafe`'s
    notification-demuxer recognizes the frame and either:
    (a) Stashes into `server._pending_continuation` if a
        send_message is currently in flight (for the existing
        "pick up after current turn" semantic), OR
    (b) Calls `server._start_model_thread(child_messages)`
        directly if the daemon is idle.
  - The daemon-side action is identical to today's
    continuation_callback body; only the trigger source moves.

**Pre-audit critical question answered**:

> Is event-bus plumbing in-scope for Phase 3, or a Phase 4+
> concern that 6.6.2 should defer entirely?

**Mostly DEFER.**  Of the 7 callback sites, only 1
(`AgentStatusChangedEvent` within continuation_callback) hits
the daemon's EventBus today.  The other 6 events are unmapped
in `_SERVER_TO_BUS` — they go directly to clients without
touching the bus.  No event-bus plumbing required for them.

The `AgentStatusChangedEvent` bus emission still happens
post-seat-flip — it's emitted by `server.emit(...)` daemon-side
when the daemon receives the continuation notification frame,
same as today.  No runner-side bus access needed.

**The original "event-bus plumbing" framing was a misnomer.**
The actual plumbing is "runner→daemon notification stream" —
which the existing `session.send_message` stream channel
already provides.

#### Audit Step 5 — sub-commit decomposition

**Step 6.6.2 collapses into step 6.6.4** alongside the
`_jaato.send_message()` migration.  The callbacks naturally
disappear because they're wired by daemon-side
`_start_model_thread`, and `_start_model_thread` itself
simplifies dramatically when send_message moves to RPC:

  - Pre-§7c step 6.6.4: `_start_model_thread` wires 7 callbacks
    on `_jaato.get_session()` + calls `_jaato.send_message(...)`.
  - Post-§7c step 6.6.4: `_start_model_thread` simply calls
    `session_send_message_threadsafe(prompt, ...)` with a
    notification-handler kwarg.  The notification handler
    invokes `server.emit(...)` for the 5 vestigial events and
    invokes `_start_model_thread(child_messages)` for the
    continuation case.
  - The 7 callback wiring sites + `_pending_continuation`
    machinery + the continuation-flow logic ALL collapse
    around the new notification-stream shape.

**Required prerequisite (NEW finding from this audit)**:
extending the runner-side `session.send_message` handler +
daemon-side wrapper to support notification frames.  This is
NOT a new RPC handler (no new dispatch route, no new method
name); it's an extension to the existing stream-frame protocol.

Step 6.6.4's scope grows by:

  - Notification-frame protocol extension on the
    `session.send_message` RPC (~2 commits if split: protocol +
    consumer-side wiring).
  - The 7 callback wirings deletion + replacement with
    notification-handler kwarg.
  - The daemon-side `_jaato.send_message()` → RPC migration.
  - The continuation-flow refactor.

These all land together because they're tightly coupled —
splitting them would create transitional broken states.

#### What this audit decides

  - **§7c step 6.6.2 is REMOVED as a separate sub-commit.**
    Folded into 6.6.4 alongside the daemon-side `_jaato.send_message`
    migration that triggers the natural callback-disappearance.

  - **No new RPC handlers needed.**  The 6 vestigial-for-daemon
    callbacks + the continuation_callback all flow through an
    extension to the existing `session.send_message` stream
    channel.

  - **No event-bus plumbing required.**  5 of the 6 emitted
    event types are unmapped in `_SERVER_TO_BUS` (direct-to-client);
    the 1 mapped one (`AgentStatusChangedEvent`) still goes
    through `server.emit(...)` daemon-side.

  - **Audit caught a 3-site inventory miss.**  The original
    "4 sites" was off by 3 — `set_continuation_callback`,
    `set_retry_callback`, `set_mid_turn_interrupt_callback` were
    missing.  Audit-discipline tally: 5 audits, 5 silent-
    regression catches.

#### What this audit does NOT decide

  - Notification-frame wire format.  The existing stream-frame
    protocol carries `(stream_id, data)` for output chunks;
    notification frames need a discriminator (frame_type:
    "output" | "notification").  Decide at 6.6.4 implementation
    time.

  - Whether to ship the notification-frame protocol extension
    as a separate commit before 6.6.4 (cleaner review boundary)
    or inline in 6.6.4 (single coherent diff).  Decide at
    implementation time once the wire-format question is
    settled.

### Step 6.6.3 disposition audit

Mirroring cd3ecf20 / ac088e67 / 875e48bd / 4d53fd49 / 9f28f96d
— the sixth audit in the §7c chain.  Per the 6.6.2 audit's
inventory-miss lesson and the reviewer's recommendation to
"cross-grep for ALL `_jaato.get_session()` reach patterns
rather than relying on prior classifications," this audit
re-runs the inventory with the corrected pattern.

#### Audit Step 1 — corrected site inventory

The §7c step 6.6 disposition audit (commit 875e48bd) listed 8
external `JaatoServer.get_session()` callers based on the
PUBLIC method.  Cross-grep of `(server\.|jaato\.|_jaato\.)get_session()`
patterns (catching both public-method calls and private-attr
reaches) reveals **17 sites total** — **9 sites missed by the
original audit** (8 in `session_manager.py`, 1 in
`websocket.py` was correct).

Full inventory + per-site disposition:

| # | Site | Operation | Migration target |
|---|---|---|---|
| 1 | `core.py:3229` | `get_cancel_token` closure → `session._cancel_token` | **Delete** — legacy in-process cancel-token dead post-§7b.2 |
| 2 | `core.py:3617` | `signal_completion` filter → `session._tools` walk | **Use existing** `JaatoClient.get_tool_schemas()` (added §7c step 3b at 7b30c237) |
| 3 | `websocket.py:1481` | event-bus access → `jaato_session._runtime.event_bus` | **Use existing** `server.event_bus` property (migrated to `self._runtime.event_bus` in §7c step 6.2) |
| 4 | `websocket.py:1485` | event-bus access (alternate path) | Same as 1481 |
| 5 | `session_manager.py:1968` | `set_session_state(key, value)` (initial-state injection) | **Use existing** `session_set_session_state_threadsafe` (§3.3c precursor) |
| 6 | `session_manager.py:2130` | `set_initial_history(initial_history)` | **Use existing** `session_set_initial_history_threadsafe` (§7c step 6.6.1.1, commit 3f859e3a) |
| 7 | `session_manager.py:2185` | `inject_prompt(text, source_id, source_type)` | **Use existing** `session_inject_prompt_threadsafe` (§7c step 6.1 (3/3), commit 14e57709) |
| 8 | `session_manager.py:2564` | `restore_turn_accounting(turns)` | **Use existing** `session_restore_turn_accounting_threadsafe` (§7c step 6.6.1.2, commit 82b8da29) |
| 9 | `session_manager.py:2607` | `restore_conversation_budget(snapshot)` | **Use existing** `session_restore_conversation_budget_threadsafe` (§7c step 6.6.1.3, commit b40d2439) |
| 10 | `session_manager.py:2855` | Append synthetic tool message: `session.get_history()` + `session.reset_session(modified_history)` | **NEW RPC needed** — `session.append_history_message` (mirrors the §7b.1 lifecycle straggler pattern; pure write).  Alternatively reuse `session.set_initial_history` after `session.reset` round-trip — but that's a multi-RPC dance for what should be one operation.  Single dedicated handler is cleaner. |
| 11 | `session_manager.py:2986` | `instruction_budget.get_conversation_snapshot()` (persistence-save path) | **NEW RPC needed** — `session.snapshot_conversation_budget`.  Inverse of `session.restore_conversation_budget` (6.6.1.3); same wire-shape (`SourceEntry.to_dict()` JSON-native dict). |
| 12 | `session_manager.py:4096` | `_parallel_tools_override` private write (per-turn override) | **NEW RPC needed** — `session.set_parallel_tools_override`.  Simple bool flag setter; ~§7b.2-precursor scale. |
| 13 | `session_manager.py:4243` | `instruction_budget.snapshot()` for `InstructionBudgetEvent` emit | **Use existing** `session_snapshot_instruction_budget_threadsafe` (§7c step 6.1 (2/3), commit 1043bfde) |
| 14 | `session_manager.py:4290` | `inject_prompt` via SDK request handler (mirror of 2185) | Same as 7 — **use existing** `session_inject_prompt_threadsafe` |
| 15 | `session_manager.py:4318` | `ReplayMessagesRequest` → `session.replay_messages(messages, timeout)` (provider-blocking) | **NEW RPC needed** — `session.replay_messages`.  Streaming + cancel-aware shape similar to `session.send_message` (§7b.2 commit 3ca3c14d).  Returns response_text after blocking on provider call. |
| 16 | `session_manager.py:4353` | `ResolveForkPointRequest` → `session.resolve_fork_point(history, after_message, after_tool_call, after_timestamp)` (pure read) | **NEW RPC needed** — `session.resolve_fork_point`.  Pure read; returns int (fork_index).  ~§7b.2-precursor scale. |
| 17 | `core.py:3986` | Public `JaatoServer.get_session()` method itself | **Delete** — drops alongside `_jaato`-field removal in step 6.6.4 (no consumers remain after migrations 1-16). |

#### Audit Step 2 — bucket totals

| Bucket | Sites | Disposition |
|---|---|---|
| **Delete outright** | 1 (cancel_token) + 17 (public method) | 2 sites |
| **Use existing accessor / RPC** | 2, 3, 4, 5, 6, 7, 8, 9, 13, 14 | 10 sites; covered by §7c step 3b's `get_tool_schemas`, §7c step 6.2's `event_bus` property, §3.3c precursor `set_session_state`, §7c step 6.1 trio's `inject_prompt` + `snapshot_instruction_budget`, §7c step 6.6.1 trio's `set_initial_history` + `restore_turn_accounting` + `restore_conversation_budget` |
| **NEW RPC handlers** | 10, 11, 12, 15, 16 | 5 sites; require new prerequisite handlers before migration can land |

**Inventory miss diagnosis**: the original §7c step 6.6 audit
focused on `JaatoServer.get_session()` (the PUBLIC method on
the server class).  Cross-grepping `_jaato.get_session()`
(the underlying JaatoClient call, reachable via private-attr
access) was missed — `session_manager.py` reaches into the
server's `_jaato` field directly in 9 sites.  The §7c step
6.6.2 audit caught a similar inventory miss (4 sites → 7).
Pattern is consistent.

#### Audit Step 3 — sub-commit decomposition

Step 6.6.3 cannot ship as a single commit per the corrected
scope.  Decomposition mirrors the §7c step 6.6.1 sub-pattern
(prerequisites first, then migration):

| Sub-commit | Scope |
|---|---|
| **§7c step 6.6.3.0** | Audit doc update + corrected inventory + the 3 reviewer-flagged audit-discipline notes added to §10 (THIS commit). |
| **§7c step 6.6.3.1** | Add `session.append_history_message` RPC + handler + wrapper + tests.  Required new `JaatoSession.append_history_message(message)` public method first (missing-method finding confirmed; method added inline as 6.6.3.0 encapsulation pattern).  **Shipped.** 12 new tests (including provenance-round-trip + multiple-appends e2e).  Wire shape reuses `serialize_message` / `deserialize_message` (same wire-shape-reuse rationale as 6.6.1.1). |
| **§7c step 6.6.3.2** | Add `session.snapshot_conversation_budget` RPC.  Inverse of `restore_conversation_budget` from 6.6.1.3.  Required new `JaatoSession.snapshot_conversation_budget()` public wrapper (audit-of-record: `InstructionBudget.get_conversation_snapshot` exists but JaatoSession had no public wrapper — same shape as 6.6.1.0's `restore_conversation_budget` addition).  **Shipped.** 12 new tests including a save→restore round-trip pinning the inverse-pair invariant. |
| **§7c step 6.6.3.3** | Add `session.set_parallel_tools_override` RPC.  Public method addition: `JaatoSession.set_parallel_tools_override(enabled)` wrapping the private `_parallel_tools_override` field.  **Shipped.** 12 new tests (True/False round-trip + bool coercion + per-turn toggle e2e). |
| **§7c step 6.6.3.4** | Add `session.replay_messages` RPC.  Originally framed as streaming + cancel-aware (similar to `session.send_message`), but on-implementation found the existing `JaatoSession.replay_messages` is a blocking call returning `str` — no cancel-token surface and no streaming output (the daemon caller already runs it in a worker thread).  Simpler than expected.  **Shipped.** 16 new tests including non-str response coercion + empty-list acceptance + provider-error → `stage="replay"`.  Wire shape: `{"messages": [...], "timeout": float}` → `{"response_text": str}`.  No missing-method gap (replay_messages exists at jaato_session.py:8252; this is the ONLY one of the 5 prerequisites without an encapsulation cleanup). |
| **§7c step 6.6.3.5** | Add `session.resolve_fork_point` RPC.  Pure-read shape.  **Shipped.** 19 new tests covering all 3 specifiers + session.get_history() defaulting + explicit-history override + per-arg validation + new `stage="resolve"` error + non-int return rejection.  No missing-method gap.  Step 6.6.3 prerequisite chain CLOSED — 6.6.3.6 (the actual 17-site migration) unblocked. |
| **§7c step 6.6.3.6** | Migrate all 17 sites + drop the public `JaatoServer.get_session()` method.  Each site uses the appropriate existing-or-new RPC per the inventory table above.  **Shipped.**  Per-site outcome: 2 deletes (cancel_token + public method), 5 sites using existing RPCs (`session.set_session_state`, `session.set_initial_history`, `session.inject_prompt` ×2 + `session.snapshot_instruction_budget`), 5 sites using new 6.6.3 RPCs (`session.append_history_message`, `session.snapshot_conversation_budget`, `session.set_parallel_tools_override`, `session.replay_messages`, `session.resolve_fork_point`), 3 trivial migrations (event_bus property reads + signal_completion using `get_tool_schemas`), 2 test-fake updates (test_session_manager_routing.py + test_sdk_parity_handlers.py).  707/707 server suite green.  Remaining `_jaato.get_session()` sites (7): 4 architectural callbacks (step 6.6.4), 1 references-plugin split-out from 6.3, 1 signal_completion private-attr read, 1 doc comment — all collapse in 6.6.4 alongside `_jaato`-field removal. |

Total: 7 sub-commits (1 audit + 5 new handlers + 1 migration).
Each handler-sub-commit is ~12-15 tests per the 6.1 trio
cadence.  The migration sub-commit's test churn likely
includes deleting tests that pinned the daemon-side write-both
pattern for these sites.

#### What this audit decides

  - **§7c step 6.6.3 expands from 1 commit to 7.**  Same
    pattern as the §7c step 6.6.1 audit (1 → 4) caught.

  - **5 NEW prerequisite RPC handlers required** before site
    migration can land: `session.append_history_message`,
    `session.snapshot_conversation_budget`,
    `session.set_parallel_tools_override`,
    `session.replay_messages`, `session.resolve_fork_point`.

  - **Audit-discipline tally: 6 audits, 6 silent-regression
    catches.**  This audit's miss class is consistent with the
    6.6.2 audit's (inventory shortfall when prior audit
    classified a subset).

#### What this audit does NOT decide

  - The wire shape for `session.replay_messages` — does it
    stream output via the §7b.2 stream channel like
    `session.send_message`, or block-and-return?  Decide at
    6.6.3.4 implementation; likely streaming for parity.

  - Whether the 5 new handlers can ship in parallel (they're
    independent prerequisites for site migration) or must
    serialize.  Independent in principle; serializing per the
    6.1 trio cadence keeps reviewability tight.

### §7d disposition audit (pre-implementation)

Mirrors the audit-discipline pattern that ran 18-for-18 across §7c.
Prerequisites for §7d: §7a (always-spawn the runner) shipped;
seat-flip complete (§7c step 6.6.4.5e at commit a922082f).  §7d
adds runner-subprocess-side cgroup placement so child processes
(cli, interactive_shell PTY children) inherit the per-session
cgroup automatically — the architectural simplification peer-
review v2 observation #2 called out.

**Q1 — Cgroup attach mechanics: where + how is the runner placed
into the per-session cgroup?**

Current state (pre-§7d):

- `server/cgroups.py:CgroupsManager.provision_cgroup(session_id,
  config)` creates the cgroup at `/sys/fs/cgroup/<root>/jaato-
  <session>/` and applies limits.  WS sessions trigger this at
  line `websocket.py:703`.
- `CgroupsManager.make_attach_callback(session_id)` returns a
  zero-arg callable that does `open("cgroup.procs", "w").write(
  str(os.getpid()))` from within the forked child.  Designed
  for `Popen(preexec_fn=...)`.
- Today's daemon-side `set_runtime_limits` (core.py:674-736) is
  a documented no-op since §7c step 6.2 (commit message: "runner
  subprocess gets cgroup attach + RuntimeLimits at spawn time,
  not via this daemon-side method").  Comment foresaw §7d as
  the implementation step.

`RunnerSpawner.spawn` (runner_spawner.py:83-166):
- Forks via `os.fork()` (line 143).
- Child: `_exec_runner` dup's the socket fd → 3, optionally
  redirects stdout/stderr, closes inherited fds, exec's runner.
- **No cgroup attach today** — the runner ends up in the
  daemon's cgroup (the host's session.scope or the daemon's
  systemd unit).

Implementation shape for §7d:

Between `os.fork()` and `_exec_runner` in the child branch, call
the attach callback returned by `CgroupsManager.make_attach_callback(session_id)`.
Concretely: pass an optional `attach_callback` to
`RunnerSpawner.spawn(...)` and invoke it right after the fork.
Same mechanism as today's plugin-level `Popen(preexec_fn=...)`,
just one level higher (the runner's own pid migrates, not each
plugin subprocess).

**Q2 — Inherit verification: explicit `/proc/<child>/cgroup`
check vs trust-by-default?**

cgroup v2 default behavior: children inherit their parent's
cgroup placement.  `fork()` puts the child in the parent's
cgroup; `exec()` doesn't change cgroup membership.  Linux
kernel ≥4.5 (cgroup v2 stable since 4.5) treats this as
guaranteed.

Verdict: **trust-by-default + one-shot integration test**.
Per-spawn `/proc/<pid>/cgroup` checks would be a paranoia tax;
inheritance is a kernel contract.  The integration test
(below) pins inheritance under realistic conditions so a future
kernel-driver change breaking inheritance fails loudly.

**Q3 — Existing `_can_migrate_to(_find_writable_cgroup_parent())`
gate at test_runtime_limits_e2e.py: integration-test entry
point?**

The test file already exists at
`jaato-server/shared/tests/test_runtime_limits_e2e.py` with
`TestRealKernel` skipif-gated on a writable cgroup parent
(lines 460-485).  §7d adds new test cases inside the existing
`TestRealKernel` class:

1. **Runner-spawn-into-cgroup**: spawn a real runner under
   `RunnerSpawner.spawn(session_id, ..., cgroup_attach=...)`;
   read `/proc/<runner_pid>/cgroup` from the daemon; assert
   the runner's cgroup ancestry matches the session cgroup.
2. **Child-inherit**: have the runner execute a cli tool that
   forks a subprocess (e.g., `bash -c "sleep 1 & echo $!"`);
   read `/proc/<child_pid>/cgroup`; assert matches the
   runner's cgroup.
3. **Grandchild-inherit (PTY stress case)**: have the runner
   spawn an `interactive_shell` PTY; assert the PTY's child
   shell (pexpect-spawned grandchild) inherits the same
   cgroup.  This is peer-review v2 observation #2's
   stress-case test.

**Q4 — `shared/ai_tool_runner.py:211` daemon-side
`_cgroup_attach`: stay or delete?**

Today this field is set via `ToolExecutor.set_runtime_limits`,
which the daemon's `set_runtime_limits` (core.py:674) ALREADY
no-ops since §7c step 6.2.  Daemon-side `ToolExecutor` is dead
post-seat-flip (tool execution flows through the runner
subprocess); the field is set but never read.

**Verdict: leave the field declaration + setter in place for
now.**  Two reasons:

1. **Disk-restored sessions fallback (§3.12 plan)**: until
   Phase 4+ ships its own session-restore path, disk-restored
   sessions could theoretically fall through to in-process
   daemon-side tool execution (defensive against the runner-
   spawn failure path).  Today's `set_runtime_limits` no-op is
   benign; keeping it preserves the fallback's correctness if
   any restore-path regression surfaces.
2. **Runner-side `_cgroup_attach` in plugins (cli,
   interactive_shell)**: post-§7d, the runner subprocess is
   itself in the cgroup, so child Popen's inherit by default
   → `_cgroup_attach` becomes a **no-op preexec_fn**
   structurally (existing `_noop` callable in CgroupsManager
   already serves this when cgroup is unavailable).  Plugin-
   side code stays unchanged — same Popen pattern, no-op
   callback.  Future cleanup could delete the plugin-side field
   entirely; out of §7d scope.

**Q5 — PTY grandchildren inheritance (peer-review v2 obs #2's
stress case)?**

`interactive_shell/plugin.py:576` does
`Popen(..., preexec_fn=self._cgroup_attach)`.  Post-§7d the
runner subprocess is in the cgroup; the PTY (pexpect-spawned
child) inherits; the PTY's child shell (grandchild) inherits.
Three levels of inheritance for one cgroup decision — kernel
contract holds for cgroup v2.  The integration test pins this
explicitly.

**Q6 — §3.11 isolated-subagent opt-in status?**

Cross-grep of `agent_params.*isolated` / `isolated_subagent`:
no production sites found.  The comment at
`session_manager.py:4889` describes the planned semantics
("default-share / opt-in-isolation come with §3.11 + the
seat-flip") but the actual `agent_params.isolated: true` →
fresh-runner spawn path **doesn't exist yet**.  The seat-flip
(§7c) just enabled the architectural prerequisite.  The
termination-hook portion of §3.11 shipped (subagent
termination hook + reliability cleanup); the isolation-opt-in
portion is **separate work, post-§7d**.

Disposition: §3.11 isolated-subagent opt-in is OUT OF SCOPE for
§7d.  Track as a follow-up or backlog entry depending on
prioritization.

#### §7d audit findings

| # | Finding | Disposition |
|---|---|---|
| 1 | Runner-spawn cgroup attach is a single preexec_fn call between fork() and _exec_runner | Single-commit migration; ~5 lines in RunnerSpawner |
| 2 | Inheritance is a cgroup-v2 kernel contract; no per-spawn verification needed | Trust-by-default + integration tests pin the contract |
| 3 | Integration tests gate on existing `_can_migrate_to` infrastructure (test_runtime_limits_e2e.py) | 3 new TestRealKernel tests: runner-spawn / child-inherit / grandchild-inherit |
| 4 | Daemon-side `ToolExecutor._cgroup_attach` field stays (defensive against disk-restore fallback) | Out of §7d scope; cleanup deferred to Phase 4+ |
| 5 | WS session: `set_runtime_limits(...)` call (websocket.py:721) is already a documented no-op since §7c step 6.2 | Leave-as-is; plan deletion for a separate cleanup commit |
| 6 | Plugin-level `_cgroup_attach` becomes a no-op preexec_fn post-§7d (peer-review v2 obs #2 realized) | Mechanical realization; no immediate plugin code changes needed |
| 7 | §3.11 isolated-subagent opt-in NOT shipped; out of §7d scope | Defer to follow-up per prioritization |

#### §7d sub-decomposition

Single-commit migration per the audit's findings (mechanical
implementation, contract pinned by tests).

| Sub-step | Scope | Test pin |
|---|---|---|
| **7d** | `RunnerSpawner.spawn` accepts optional `cgroup_attach: Callable[[], None]` arg; SessionManager / WS handlers pass `cgroups.make_attach_callback(session_id)` at spawn time; child invokes the callback between `os.fork()` and `_exec_runner`.  **Shipped.**  6 unit-level regression pins in `server/tests/test_runner_cgroup_attach_7d.py` (signature pins, fork-then-attach-then-exec order, WS pre-init hook provisions cgroup before spawn, IPC-style optionality, daemon-side `_cgroup_attach` field preservation per Q4).  3 integration tests in `shared/tests/test_runtime_limits_e2e.py::TestRealKernel` (runner-spawn-into-cgroup, child-inherit, PTY grandchild-inherit stress case).  WS pre-init hook reordered: cgroup-provision moved from post-init session_hook to pre-init so the attach_cb is available at spawn time.  Provisioning is idempotent — post-init hook's redundant re-provisioning is no-op'd by `mkdir(exist_ok=True)` and limit-file overwrites; the post-init hook's `set_runtime_limits(...)` call stays as a documented no-op (preserves §3.12 disk-restore fallback). |

**Audit-discipline tally: 19 audits, 19 silent-regression
catches** (today's catch: the daemon-side `_cgroup_attach`
field-deletion question — keeping it was the safer call given
the §3.12 disk-restore fallback path).

§7d ships as a single-commit migration; no further split needed.

### Step 7 disposition audit (pre-implementation, post-§7c/§7d)

20th audit in the §7c+§7d arc.  Verifies the gate ("permission-
plugin runner-side activation") is genuinely satisfied post-
seat-flip + cgroup migration and identifies the actual scope
of Step 7's wiring work.

**Q1 — Post-§7c state of `JaatoServer.respond_to_*` methods?**

All 4 still exist at `core.py:3808-3882`:
- `respond_to_permission(request_id, response, edited_arguments)` — pushes to `_channel_input_queue`
- `respond_to_clarification(request_id, response)` — pushes to `_channel_input_queue`
- `respond_to_clarification_batch(request_id, answers)` — pushes multiple
- `respond_to_reference_selection(request_id, response)` — pushes to `_channel_input_queue`

The daemon-side `_channel_input_queue` is consumed by **daemon-
side** plugin `_channel.set_callbacks(input_queue=...)` wirings
at `core.py:3279/3291/3301` — but those are wired to
**daemon-side plugin instances**, NOT runner-side ones.
Post-seat-flip, the runner-side plugin is the one firing
ASKs; the daemon-side queue is orphaned for runner-fired ASKs.

**Q2 — `PromptOperatorHandler.resolve_response()` integration?**

Class exists at `server/runner_rpc_handlers/prompt_operator.py:58`.
`register()` helper exists at line 215.  Cross-grep:
**`PromptOperatorHandler` is NEVER instantiated in production
code** — only tests construct it.  The wiring slot is empty.

**Q3 — Permission plugin's runner-side ASK path?**

`permission/plugin.py:135` `_get_runner_rpc_channel()` does
`rpc_client = getattr(registry, 'runner_rpc_client', None)`.
Cross-grep: **`registry.runner_rpc_client` is never assigned
in production code** — only daemon-side `setattr(self.registry,
"runner_rpc", rpc_client)` at `core.py:4315` (different
attribute name + daemon-side registry).  The runner-side
permission plugin's lookup always returns None → falls back
to the in-process `_channel` → reads from a queue no one fills
(runner-side `_channel_input_queue` is daemon-instance-bound).

**Q4 — Runner-internal `RunnerRPCClient`?**

Class exists at `server/runner/rpc_client.py:48` with
`prompt_operator()` method.  Cross-grep:
**`RunnerRPCClient` is never instantiated** — defined but unwired.

**Q5 — Daemon-side `RunnerRPCServer`?**

Class exists at `server/runner_rpc_server.py:68`.  Cross-grep:
**never instantiated** in production code.  The runner→daemon
RPC dispatch infrastructure (apparmor fragment, prompt
operator, telemetry publish) is defined but unwired on both
ends.

**Q6 — Clarification + References runner-RPC channels?**

`shared/plugins/clarification/channels.py` and
`shared/plugins/references/channels.py` exist as in-process
channels.  Neither has a RunnerRPCChannel equivalent.  Only
the permission plugin shipped a runner-RPC channel
(`permission/runner_rpc_channel.py`); clarification + references
remained in-process-only.

**Q7 — Test coverage for ASK round-trip?**

`permission/tests/test_plugin_runner_rpc_wiring.py` exists and
tests the runner-RPC channel + plugin lookup, but with mocked
`prompt_operator` callables — not an integration test through
real daemon emit → client response → handler resolution.

#### Step 7 audit findings

| # | Finding | Disposition |
|---|---|---|
| 1 | Step 7's original scope ("wire PromptOperatorHandler") was framed assuming only the handler-registration was missing.  Actual scope: BOTH ends of the runner→daemon RPC infrastructure are unwired. | Sub-track expands; single commit is insufficient. |
| 2 | `PromptOperatorHandler`, `RunnerRPCServer`, and the runner-internal `RunnerRPCClient` are all defined-but-never-instantiated. | Step 7.1 instantiates the daemon-side trio; Step 7.2 instantiates the runner-internal client. |
| 3 | `registry.runner_rpc_client` attribute on the runner-side registry is never assigned.  Permission plugin's runner-side detection always returns None. | Step 7.2 wires the assignment in the runner bootstrap. |
| 4 | Daemon-side `respond_to_permission` writes to `_channel_input_queue` — orphaned for runner-fired ASKs.  Needs to route through `PromptOperatorHandler.resolve_response(request_id, response)` instead. | Step 7.3 rewires `respond_to_permission`. |
| 5 | Clarification + References plugins lack RunnerRPCChannel equivalents.  Open question: are they currently fired runner-side at all, or do their ASKs only flow daemon-side via the in-process channel? | **Flag for Step 7 follow-up audit** — investigation needed before committing scope. |
| 6 | `RunnerRPCServer` instances need to be plumbed into the `RunnerRPCClient` read loop so incoming runner→daemon RPCs get dispatched.  Today's read loop handles response/stream/event frames; needs a request-frame path too. | Step 7.1 adds the bidirectional dispatch. |
| 7 | No e2e integration test exists for the full ASK round-trip. | Step 7.3 adds one. |
| 8 | Audit-discipline catch: original "wire PromptOperatorHandler" framing would have missed the runner-side RPC client instantiation gap (finding #3) and the bidirectional read-loop dispatch gap (finding #6).  | Both surfaced by pre-implementation grep — confirms the discipline's value at sub-track boundaries. |

#### Step 7 sub-decomposition

**This is a multi-commit sub-track, not a single commit.**

| Sub-commit | Scope | Test pin |
|---|---|---|
| **Step 7.0** | This disposition audit (no code). | 0 |
| **Step 7.1** | Daemon-side: instantiate `RunnerRPCServer` + `PromptOperatorHandler` per session; bind handler to `JaatoServer.emit`; register handler via `register()` helper; plumb the server into the daemon-side `RunnerRPCClient` read loop's request-frame dispatch path.  **Shipped — scope narrower than Step 7 audit's finding #6 framed.**  9 new tests in `server/tests/test_prompt_operator_wiring_step7_1.py`.  Implementation revealed the bidirectional dispatch IS already wired: `RunnerRPCClient.__init__` lazy-constructs `RunnerRPCServer` (line 152); read loop dispatches `KIND_REQUEST` frames at line 355.  Step 7.1's actual scope was just handler instantiation + registration inside `set_runner_rpc`, plus shutdown teardown.  Corrects Step 7 audit finding #6 (bidirectional dispatch gap was already closed). |
| **Step 7.2** | Runner-side: instantiate runner-internal `RunnerRPCClient` from the bootstrap socket; attach to runner registry as `registry.runner_rpc_client`; verify the permission plugin picks it up.  **Shipped.**  6 tests in `server/runner/tests/test_runner_rpc_client_wiring_step7_2.py`.  Wiring lives in `_handle_session_bootstrap` inside `server/runner/rpc.py` — after `bootstrap_session(envelope)` returns a healthy host, construct `RunnerRPCClient(self)` (where self is the runner-side `RunnerRPC` dispatcher) and `setattr(host.runtime._registry, "runner_rpc_client", client)`.  Pre-§7.2 the attribute was never assigned; permission plugin's `_get_runner_rpc_channel` always returned None and fell back to the orphaned in-process channel. |
| **Step 7.3** | Rewire `JaatoServer.respond_to_permission` to call `PromptOperatorHandler.resolve_response(request_id, response)` instead of pushing to `_channel_input_queue` (keep queue path as fallback for daemon-fired ASKs if any remain).  Add e2e integration test for full ASK round-trip.  **Shipped.**  9 tests in `server/tests/test_respond_to_permission_routing_step7_3.py`.  Dual-path routing: Path 1 (runner-fired ASKs) calls `handler.resolve_response(request_id, response, edited_arguments=...)` first.  When that returns False (no pending future), falls through to Path 2 (legacy daemon-fired ASK path: check `_pending_permission_request_id` + push to `_channel_input_queue`).  Unknown requests (neither path resolves) emit `ErrorEvent`.  Path 1 wins over Path 2 in the collision case.  E2E test exercises the full daemon-half round-trip: runner-payload → handler.handle → emit `PermissionRequestedEvent` → `respond_to_permission` → handler's future resolves → result dict deserializes back via `PromptResponse.from_dict`. |
| **Step 7.4 (conditional)** | Clarification + References RunnerRPCChannel equivalents IF investigation confirms they're fired runner-side.  **Investigation complete; deferred to backlog.**  Both plugins are `PLUGIN_TIER = "runner"` and use in-process channels with no RPC bridge — **latent regression** post-seat-flip.  Existing test suite doesn't exercise the regression path (no automated e2e test for clarification/references ASKs).  Per-plugin fix is straightforward but non-trivial (~4-6 hours each plugin mirror of permission's `runner_rpc_channel.py` pattern).  Captured in [`project_backlog_clarification_references_runner_rpc_gap.md`](project_backlog_clarification_references_runner_rpc_gap.md) per the audit-discipline pattern: adjacent gaps surfaced during audit get captured in the backlog, not folded in mid-stream. |

**Audit-discipline tally: 20 audits, 20 silent-regression
catches** (today's catch: original "wire PromptOperatorHandler"
framing missed the bidirectional RPC infrastructure gaps —
both ends of the wire were unwired, not just the handler).

Step 7 is a **deferred sub-track** per the user's earlier
classification.  Phase 3 critical path closure does NOT depend
on Step 7 shipping; the seat-flip (§7c) + cgroup migration
(§7d) constitute the architectural milestone.

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

> **Cross-reference:** this appendix's INTERNAL bucket lumps every
> `self._jaato.get_session()` reader under a single classification.
> Step 6's pre-implementation audit (post-step-4-first-pass) showed
> those readers actually have **four distinct dispositions** (delete,
> new-RPC-handler, architectural callback rewire, trivial migration).
> For the per-`get_session()`-reader sub-classification, see the
> **"Step 6 disposition audit"** in §7c's sequencing table above.

### Audit-discipline notes (reusable primitives)

Recorded across the §7c audit chain (cd3ecf20 / ac088e67 /
875e48bd / 4d53fd49 / 9f28f96d / this commit's 6.6.3 audit).
The audit-of-record discipline produced re-usable primitives
that future audits can apply mechanically:

#### Note 1 — Cross-grep ALL reach patterns, not just the public surface

The §7c step 6.6.2 audit (commit 9f28f96d) found 3 callback
sites the prior 6.6 audit missed.  The §7c step 6.6.3 audit
found 9 `_jaato.get_session()` reaches the prior 6.6 audit
missed (8 in `session_manager.py`, 1 in `websocket.py`).  Both
inventory misses had the same root cause: prior audits
classified a subset (the public method's call sites) without
cross-grepping the underlying private-attr reach pattern.

**Reusable primitive**: when auditing a public surface (e.g.
`JaatoServer.get_session()`), ALSO cross-grep:

  - The underlying object's private-attr access pattern
    (`_jaato.get_session()` here)
  - Any forwarding wrappers / properties that expose the same
    object via a different name
  - The implementation method itself (in case it has internal
    callers)

Failing this catches inventory misses that propagate as
silent regressions when the audited surface is removed.

#### Note 2 — `_SERVER_TO_BUS` mapping is a re-usable diagnostic

The §7c step 6.6.2 audit (commit 9f28f96d) discovered that 5
of 6 callback-emitted event types are UNMAPPED in
`server/core.py:127`'s `_SERVER_TO_BUS` dict — they go
directly to clients and never touch the daemon's EventBus.
This was the load-bearing diagnostic for the audit's
"event-bus plumbing is mostly DEFER" conclusion.

**Reusable primitive**: any future "is this Event wired to
the bus or does it go direct-to-client?" question can be
answered by grepping `_SERVER_TO_BUS` in `server/core.py`:

  - Mapped event types → published to bus + forwarded to
    client (potential daemon-side reactor consumers)
  - Unmapped event types → direct to client only (no
    daemon-side consumer possible)

If an audit needs to determine whether a callback's emitted
events have daemon-side consumers, this dict is the
authoritative source.  Combined with `grep -rn
"subscribe.*<EventName>"` for the mapped types, the survey is
mechanical.

#### Note 3 — The §7b.2 stream channel multiplex pattern (inverse-virtue at the streaming layer)

The §7c step 6.6.2 audit (commit 9f28f96d) recommended
extending the existing `session.send_message` stream channel
with notification frames rather than inventing a new
runner→daemon notification primitive.  Same inverse-virtue
as §7c step 6.6.1.1's reuse of `serialize_history` /
`deserialize_history` (commit 3f859e3a) — except applied to
the streaming layer instead of the serialization layer.

**Reusable primitive**: when an audit identifies a need for a
new runner→daemon notification channel, check first whether
the existing `session.send_message` stream channel can carry
it.  Adding a frame-type discriminator to multiplex
notification frames alongside output frames is cheaper than:

  - A new RPC method (new dispatch route + new wire surface)
  - A new long-lived stream subscription (new lifecycle
    management + new test surface)
  - A daemon-side event-bus mirror (two buses to keep
    coherent)

The pattern compresses test surface (no new wire format to
exhaustively test) and inherits the §7b.2 stream channel's
already-battle-tested cancellation + ordering guarantees.

#### Audit-discipline tally (across §7c)

| Audit | Catch |
|---|---|
| cd3ecf20 (§7c step 6) | 9 `get_session()` readers had 4 distinct dispositions; original 6a/6b/6c framing was wrong |
| ac088e67 (§7c step 6.4) | All 6 WIRING calls transitively load-bearing via `_jaato._session`; 6.4 collapses into 6.6 |
| 875e48bd (§7c step 6.6) | 35 sites + 8 external consumers + sub-commit decomposition |
| 4d53fd49 (§7c step 6.6.1) | 2 of 3 proposed RPC handlers had missing underlying methods |
| 9f28f96d (§7c step 6.6.2) | 3-site inventory miss (4 → 7); event-bus plumbing was a misnomer |
| THIS (§7c step 6.6.3) | 9-site inventory miss (8 → 17); 5 new RPC handlers needed |

**6 audits, 6 silent-regression catches.**  Each audit
recorded a re-usable primitive (cross-grep, bus-mapping, or
inverse-virtue pattern).

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
