# Runner Pre-warm Pool + Deferred Provider INIT — Bootstrap Latency Reduction

**Status:** PRs 1-4 shipped (server 0.6.76). PR 5 (operationalization) pending v63 cascade validation.
**Origin:** 2026-05-13 cascade step-6 stall diagnosed as 30s RPC timeout (`runner_rpc_client.py:704`). Root cause: §7c seat-flip (`6406fe35`, 2026-05-09) moved JaatoSession into per-session subprocesses, introducing ~16s of per-session Python startup + runner-tier-plugin import cost that didn't exist pre-§7c. Workspace state growth (37 generated Java files at step 6) tipped step 6's bootstrap over the 30s line.
**Decision (2026-05-13):** address the regression structurally rather than via band-aids (raising timeout, partial deferral). Reduce per-session bootstrap from ~30s to ~3s while preserving §7c's per-session isolation properties.

**Ship log:**
- PR 1 → #85 (`1a7665e2`): defer provider INIT to first model use
- PR 2 → #87 (`a401047f` line, see merge `16d17741` parent): template subprocess + lifecycle
- PR 3 → #88: pool slot fork-slot RPC + PoolManager
- PR 4 → #89 (`6b2a768c`, merged in `fef96997`): route through pool + replenishment thread (combined; replenishment originally PR 5 scope)

## 1. Principle

> *Per-session subprocess cost is paid ONCE, amortized across all sessions, with per-session bootstrap doing only per-session work.*

§7c chose per-session subprocess isolation for legitimate reasons (kernel-enforced AppArmor + cgroup attach, crash isolation, workspace boundary cleanliness). What §7c didn't address is that EACH session now repays the import + plugin discovery cost. This design pays that cost ONCE at daemon startup via a dedicated template subprocess, then forks pool slots from the template that inherit the warm imports.

A parallel optimization: provider plugin INIT (zhipuai's ~9s, anthropic's ~few s) runs inside `JaatoSession.configure()` today. It's not session-bootstrap-critical — the first model call could trigger provider construction lazily, deferring the network handshake off the bootstrap RPC's critical path.

Combined, per-session bootstrap drops from ~30s → ~3s.

## 2. What's wrong today

**The 16s pre-INIT runner cost:**

Per 2026-05-13 measurement on v60 step 6 session 20260513_132358:

```
13:23:58  runner subprocess spawn (fork + execvpe)
13:23:58 → 13:24:14  16s of pre-INIT work (Python startup + runner-tier plugin imports + plugin discovery)
13:24:14 → 13:24:23  9s for zhipuai INIT (provider.initialize → Anthropic client construction)
13:24:24 → 13:24:26  ~3s for configure() rest (plugin re-init with session config, system instructions render)
13:24:26 → 13:24:29  3s background BUDGET_BG refining 9 token counts (non-blocking, daemon-thread)
13:24:32  daemon-side RPC timeout fires (30s + 5s buffer at runner_rpc_client.py:704)
```

Steps 1-5 of the same cascade booted in <2s because their workspace state was smaller and they presumably hit a different combination of plugin profile / cache state. Step 6 was the first to exceed 30s.

**The cost is structural,** not workload-specific. As workspaces grow, more sessions will tip over the line. Raising the timeout (Shape α in the discussion) is a band-aid that pushes the wall back without addressing the architectural cost.

**Module-global audit findings (2026-05-13):** the runner-tier plugin codebase is overwhelmingly fork-safe by construction:

- Zero module-level HTTP clients / asyncio loops / subprocess spawns / threading.Thread / atexit / signal handlers
- Module-level state is regex compilations + frozensets + constants (read-only) plus a small number of `threading.local()` containers (empty per-thread on the child side) and one module-level `threading.Lock()` (the pybars3 lock in the `template` plugin, only acquired transiently during template render — unheld at template-subprocess fork time)

This audit makes the fork-and-inherit approach viable across the whole runner-tier plugin set with no per-plugin opt-in required.

## 3. Target state

### 3.1 Template subprocess

Daemon spawns a **dedicated** "template runner" subprocess at startup:

```
python3 -m server.runner --template-mode
```

The template:
- Imports `server.runner` and transitively the runner-tier plugin modules
- Walks `PluginRegistry._discover_via_directory()` so the registry is populated
- Does **NOT** call `plugin.initialize(config)` for any plugin (no per-session state yet, no network, no thread spawns)
- Does **NOT** call any provider INIT
- Sits idle on a control pipe to the daemon, waiting for `fork-slot` requests

The template never serves sessions itself. Its only role: be a warm fork source.

### 3.2 Pool slots

Daemon maintains N idle pool slots (configurable, default 2-4). To create a slot:

1. Daemon sends `fork-slot` request over template's control pipe
2. Template `os.fork()`s itself (no exec). Child inherits all warm imports.
3. Template returns the child's PID + a fresh socket pair end to the daemon
4. The child slot sits idle on its own pipe, waiting for `bootstrap-envelope` from the daemon

Pool replenishment: a background daemon thread fires `fork-slot` whenever idle count drops below N.

### 3.3 Session arrival

When `session.new` lands:

1. Daemon picks an idle pool slot
2. Daemon sends the `SessionInitEnvelope` over the slot's pipe
3. Slot does the per-session work:
   - Calls `plugin.initialize(config)` for each plugin in the envelope
   - Constructs the session: `runtime.create_session(...)` → `JaatoSession.configure(...)`
   - Provider INIT runs (or is deferred — see §4)
   - System instructions render, prefetch scripts, etc.
4. Slot signals `bootstrap-ready` to daemon
5. Daemon hands the client FD to the slot via `SCM_RIGHTS` (existing pattern — see daemon-as-pure-factory backlog)
6. Slot owns the session for its lifetime

### 3.4 Session end

Slot exits when session ends. No state reuse — clean isolation. Pool manager spawns a replacement slot from the template.

### 3.5 Deferred provider INIT (companion optimization)

`JaatoSession.configure()` today calls `runtime.create_provider(...)` at line 1650, which triggers `provider.initialize()` — the 9s zhipuai INIT or equivalent for other providers. This is on the bootstrap critical path.

Change: defer to first model call.

- `JaatoSession.configure()` stores the provider name + config, does NOT create the provider object
- First `JaatoSession.send_message()` checks `self._provider is None` and lazily constructs+initializes it
- Symmetric to existing lazy-init patterns (cache plugin attaches lazily, GC plugin attaches lazily)

User-visible impact: first model call has ~9s extra latency. But that's INSIDE the streaming response anyway — the user sees the "thinking..." spinner regardless. Cumulative latency budget unchanged; just shifted.

## 4. PR plan (5 PRs)

Each PR independently reviewable, rollback-safe. Each requires daemon restart; coordinate with peer per `feedback_cascade_aware_daemon_restart_coordination`.

### PR 1 — Defer provider INIT (β / Y from earlier discussion) — **SHIPPED 0.6.74 → PR #85**

**Scope:**
- `JaatoSession.configure()` removes the `runtime.create_provider()` call at line 1650
- `JaatoSession.send_message()` (and any other model-call entry points) lazy-init the provider on first use via a new `_ensure_provider()` helper
- Cache plugin attach moves from `_wire_cache_plugin` (called in configure) to inside `_ensure_provider` so caching is wired before the first call
- Tests pin: configure() doesn't call provider.initialize; first send_message does

**Touchpoints:**
- `jaato-server/shared/jaato_session.py:1647-1663` — remove eager provider create, add `_ensure_provider`
- `jaato-server/shared/jaato_session.py:_wire_cache_plugin` — call site relocates
- `jaato-server/shared/jaato_runtime.py:create_session` — pass provider config through, don't create yet

**Effort:** ~150 LoC + tests. ~3-4 hours.

**User-facing impact:** per-session bootstrap drops by ~9s (zhipuai) or ~2-3s (anthropic/openrouter). Cascade smoke v60+ steps that previously timed out at 30s now complete bootstrap in ~20s.

**Cascade unblock:** YES. Ships independently of PRs 2-5. After PR 1 lands, the existing v60 cascade SHOULD succeed because every session's bootstrap drops below the 30s timeout.

**Standalone-rollback-safe:** yes (no wire-format change, no daemon-restart-required interface change beyond the code change itself).

### PR 2 — Template subprocess — **SHIPPED 0.6.75 → PR #87**

**Scope:**
- New entry point: `python -m server.runner --template-mode`
- Template subprocess imports runner-tier plugins + walks plugin discovery + sits idle on a control pipe
- Daemon spawns template at daemon startup
- Daemon manages template lifecycle (restart if template dies, etc.)
- No pool slots yet — pool work is PR 3

**Touchpoints:**
- `jaato-server/server/runner/__main__.py` — add `--template-mode` flag + idle loop
- `jaato-server/server/runner_spawner.py` — new `spawn_template()` method
- `jaato-server/server/__main__.py` — daemon startup spawns template
- New `jaato-server/server/runner_template.py` — template lifecycle manager

**Effort:** ~300 LoC + tests. ~1-2 days.

**User-facing impact:** none yet (no sessions use the template).

**Standalone-rollback-safe:** yes (template subprocess sits idle; no other code paths consume it yet).

### PR 3 — Pool slot management — **SHIPPED 0.6.75 → PR #88**

**Scope:**
- Daemon's pool manager: spawn N slots from template at startup
- Each slot communicates with daemon over its own pipe
- Slot lifecycle: idle → serving → exited → replenished
- Pool size knob via `--runner-pool-size` env var, default 2
- Idle slots track via dict on the daemon
- Background thread keeps pool replenished

**Touchpoints:**
- New `jaato-server/server/runner_pool.py` — pool manager
- `jaato-server/server/__main__.py` — daemon startup creates pool
- `jaato-server/server/runner_template.py` — extend to support `fork-slot` request

**Effort:** ~400 LoC + tests. ~2-3 days.

**User-facing impact:** none yet (session manager still spawns fresh runners; pool slots sit idle).

**Standalone-rollback-safe:** yes (pool exists but unused).

### PR 4 — Session bootstrap via pool + replenishment thread — **SHIPPED 0.6.76 → PR #89**

**Final scope (combined with replenishment thread per 2026-05-13 user authorization "PR 4 combined"):**
- Slot mode no longer runs PR 3's line-delimited command loop. After fork from template, slot adopts fd 3 as RPC socket and runs `RunnerRPC.serve()` — same body session-mode runners use after AppArmor self-confine.
- `spawn_session_runner` accepts `pool_manager` parameter. Pool path is gated to sessions that satisfy ALL four conditions: pool_manager wired AND `JAATO_RUNNER_POOL_ENABLED=true` AND `disable_confine=True` AND no `cgroup_attach`. Any gate failure → today's cold-spawn path unchanged.
- `PoolManager.start_replenishment()` spawns a daemon thread that watches `idle_count() < target_size` and asks the template for a fork-slot to refill. Daemon's `start()` invokes it after `spawn_initial_slots()`.
- Feature flag default stays **opt-in (`false`)** — PR 5 flips after soak.

**Touchpoints (actual):**
- `jaato-server/server/runner/__main__.py` — `_run_slot_mode` replaced command loop with `RunnerRPC.serve()`
- `jaato-server/server/runner_spawn.py` — `_pool_enabled()` env reader + four-gate routing in `spawn_session_runner`
- `jaato-server/server/runner_pool.py` — `start_replenishment` / `stop_replenishment` / `_replenish_loop`
- `jaato-server/server/session_manager.py` — `set_apparmor_dependencies(pool_manager=...)`; threaded into `_spawn_session_runner_unconditional`
- `jaato-server/server/websocket.py` — apparmor pre-init hook reads `getattr(ws_server, "_pool_manager_ref", None)`
- `jaato-server/server/__main__.py` — sets `ws_server._pool_manager_ref` + calls `pool_manager.start_replenishment()`

**Tests shipped:** 5 new template/pool cases (slot serves echo-RPC + 4 replenishment cases) + 6 routing-gate cases (one per branch).  411 server tests + 604 runner tests green.

**User-facing impact:** when flag enabled, per-session bootstrap drops to ~3-5s for sessions that meet the four gates. When flag disabled (default), no change.

**Standalone-rollback-safe:** yes (flag gates the change; default off).

**Deferred to PR 5:**
- Per-slot AppArmor self-confinement (slots run unconfined in PR 4 — same posture as `JAATO_RUNNER_DISABLE_CONFINE=true`)
- Subreaper fix (`prctl(PR_SET_CHILD_SUBREAPER)`) so daemon can `waitpid` slot PIDs cleanly (slots are template-children; current code catches `ChildProcessError` and ignores — operationally fine)
- Template watchdog auto-respawn on template death
- Proper template-ready handshake replacing the 2s sleep in daemon startup
- Flag default flip to `true`

### PR 5 — Operationalization + cleanup — **PENDING (post-v63 validation)**

Scope evolved as PR 4 absorbed replenishment.  Net PR 5 work now:

**Hardening (correctness):**
- **Per-slot AppArmor self-confinement.**  Slots accept the bootstrap envelope's `profile_name`, call `aa_change_profile()` BEFORE plugin.initialize() runs.  Restores per-session confinement parity with cold-spawned session-mode runners.  Requires an extra RPC (e.g., `session.self_confine`) the daemon calls before `session.bootstrap`, OR extending the bootstrap RPC handler to honor a `profile_name` field and self-confine when set + not yet confined.
- **Subreaper fix.**  Daemon calls `prctl(PR_SET_CHILD_SUBREAPER, 1)` at startup so it re-parents the template's children (i.e., pool slots) on template death.  Today, slots are template-children and the daemon's `waitpid(slot_pid)` returns `ChildProcessError`.  Caught + ignored as a non-blocking issue in PR 4; PR 5 closes it.

**Operationalization:**
- **Template watchdog auto-respawn.**  If template dies (OOM, signal), daemon detects + respawns the template + drains and refills the pool from the new template.  Today, template death silently breaks all future fork-slot requests.
- **Template-ready handshake.**  Replace the 2s `time.sleep` in daemon `start()` with a proper "template-ready" message the template sends after `discover()` completes.  Removes a startup race window where pool spawn would fail if discover takes longer than 2s.
- **Pool telemetry.**  Counters: `pool_slot_acquired_total`, `pool_slot_cold_spawn_fallback_total`, `pool_replenish_failures_total`.  Surface in daemon log + optionally OpenTelemetry counters.
- **Graceful pool drain on shutdown.**  Daemon shutdown currently runs `shutdown_all()` which closes slot sockets.  PR 5 audits whether active sessions on pool slots get a chance to flush + the cleanup ordering is right.

**Rollout:**
- Flip `JAATO_RUNNER_POOL_ENABLED` default to `true` after a 1-week soak window.
- Document architecture in `docs/architecture.md` + add a note in CLAUDE.md.
- Eventually remove the feature flag (one or two releases after default flip).

**Effort:** ~250 LoC + tests + docs.  ~2 days focused work.

**Standalone-rollback-safe:** Individually yes — AppArmor self-confine + subreaper + watchdog are each independent landings.  Flag default flip is the only piece with measurable cascade impact at flip time.

## 5. Back-compat + rollback story

| Stage | Status | Rollback action |
|---|---|---|
| PR 1 (defer provider INIT) | shipped 0.6.74 | Code revert. No wire-format change. |
| PR 2 (template subprocess) | shipped 0.6.75 | Stop spawning template at daemon startup. Template exits cleanly when daemon dies. |
| PR 3 (pool slots) | shipped 0.6.75 | Stop spawning slots. Existing fallback path (cold spawn) handles all sessions. |
| PR 4 (route through pool + replenishment) | shipped 0.6.76 | Flag flip: `JAATO_RUNNER_POOL_ENABLED=false`. Sessions fall back to cold spawn. |
| PR 5 (self-confine + subreaper + watchdog + default-on) | pending | Flag flip default. Future revert is `git revert` + daemon restart. |

Throughout the migration, the cold-spawn path remains functional. Every PR can be independently shipped, run for days, and rolled back without state corruption.

## 6. Open questions

1. **Pool size default.** 2-4 slots reasonable for typical workstation; cascade harnesses spawning 6 concurrent sessions need ≥4 to avoid pool-empty fallback. Operator-tunable via env var.

2. **Template subprocess crash recovery.** If template dies (OOM, signal), daemon needs to detect and respawn. Watchdog thread polls template's pipe; restart on EOF.

3. **AppArmor profile attach for pool slots.** Today, profile is provisioned per-session before runner spawn. With pool slots pre-spawned (no session yet), the profile isn't ready. Options:
   - Slot inherits template's profile (unconfined) at fork; daemon provisions session profile after envelope arrives, slot self-confines to it before plugin.initialize() runs
   - Pool of N slots × N pre-provisioned generic profiles, daemon assigns at session arrival (complex)
   - Slot stays unconfined during pool-idle; per-session confinement happens just before envelope processing (preferred)

4. **Cgroup attach timing.** Same as AppArmor — cgroup is per-session. Slot must attach to its cgroup before plugin.initialize() runs.

5. **Template restart on plugin code change.** Today, daemon restart picks up plugin code changes. With pre-imported template, code changes don't propagate to existing slots until they cycle. Acceptable — dev iteration restarts daemon anyway, and prod doesn't hot-reload plugins.

6. **First-call latency for deferred provider INIT (PR 1).** Provider construction in send_message means the FIRST message's response latency includes the 9s INIT. The TUI shows a thinking spinner so user UX is fine, but reactor cascade timings may differ. Document this. Optionally allow a config flag `eager_provider_init: true` for cascades that prefer slow bootstrap over slow first-message.

## 7. Non-goals

- **Pool slot reuse across sessions.** Slots die after one session. Reuse would require state cleanup between sessions which is fragile. Trade clean isolation > 50-200ms slot creation savings.
- **Cross-daemon pool sharing.** Pool is per-daemon-process.
- **Pool size autoscaling.** Static size for now. Autoscaling is future work.
- **Pre-warmed providers.** Provider INIT is per-session because each provider has session-specific config. Could be optimized further but out of scope.

## 8. Composition with other backlog items

- **`feedback_cascade_aware_daemon_restart_coordination`** — every PR's daemon restart requires peer coordination
- **`project_backlog_daemon_as_pure_factory`** — pool implementation is the foundation for daemon thinning; FD-pass handoff (SCM_RIGHTS) layers on top
- **`docs/design/shape3_workspace_state_relocation_plan.md`** — independent from pool work; both can land in parallel
- **`feedback_no_jaato_changes_without_authorization`** — every PR's framework code change requires explicit user OK

## 9. Acceptance gates

Each PR lands behind:
- Differential test sweep: zero new failures vs main (same pattern as Phase 5 §5.x reviews)
- New regression tests pinning the new wire/state contract
- For PR 1: manual cascade smoke against an inflated workspace verifies bootstrap drops below 20s
- For PR 4: end-to-end test verifies pool slot is consumed (not a cold spawn) when flag enabled
- For PR 5: telemetry surface visible in `dmesg` or daemon log

## 10. Estimated total effort

| PR | LoC | Tests | Time | Risk | Status |
|---|---|---|---|---|---|
| PR 1 | ~150 | ~30 | 3-4 hours | LOW | shipped 0.6.74 |
| PR 2 | ~300 | ~40 | 1-2 days | MEDIUM | shipped 0.6.75 |
| PR 3 | ~400 | ~50 | 2-3 days | MEDIUM | shipped 0.6.75 |
| PR 4 (combined w/ replenishment) | ~760 line-delta | ~11 new | 1 day | MEDIUM | shipped 0.6.76 |
| PR 5 (self-confine + subreaper + watchdog + flag default flip) | ~250 | ~30 | ~2 days | MEDIUM | pending |
| **Total** | **~1900 line-delta shipped + ~250 pending** | **~170** | **~1 week focused work, ~80% done** | MEDIUM | |

PR 1 alone unblocked the cascade.  PR 4 is the structural fix — once the flag flips in PR 5, every session that opts out of AppArmor automatically benefits from warm imports.

## 11. Decision log

- **2026-05-13** — Pre-warm pool with daemon-as-fork-source rejected after audit revealed daemon doesn't import runner-tier plugins. Dedicated template subprocess design chosen instead — daemon imports daemon-tier only; template imports runner-tier; pool slots fork from template.
- **2026-05-13** — Module-global audit (`re.compile`, `frozenset`, `threading.local`, `threading.Lock` × 1) confirmed runner-tier plugins are fork-safe by construction. No per-plugin opt-in hint required.
- **2026-05-13** — Deferred provider INIT chosen as PR 1 (independent of pool, smallest cascade-unblock).
- **2026-05-13** — Stage 3.1 (5-LoC `os.environ` fallback) and Shape α (raise timeout to 120s) both rejected as band-aids that mask the structural cost.
- **2026-05-13** — PR 4 combined with replenishment thread (originally PR 5 scope) per user authorization "PR 4 combined".  Without replenishment, target_size=2 only amortizes the first 2 of an N-step cascade — defeating the pool for the very workload that motivated the multi-PR project.  Combined PR delivers measurable cascade unblock on first ship.
- **2026-05-13** — PR 4 pool path gated to sessions with `disable_confine=True` AND no `cgroup_attach`.  Per-slot AppArmor self-confine + cgroup migration both deferred to PR 5 because they require new post-fork RPC handshakes the existing wire surface doesn't support.  Trade-off: PR 4 ships faster + does the structural work; PR 5 brings the kernel-isolation properties back.
