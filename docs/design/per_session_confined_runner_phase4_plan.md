# Phase 4 Implementation Plan — Sub-track Closure + Performance Pass

**Status:** Draft, awaiting peer review.

**Parent design:** `per_session_confined_runner.md` (Phase 1 design).
**Predecessor:** `phase3_closure_recap.md` (Phase 3 merged 2026-05-11, PR
#54, merge commit `405d7f00`).

Phase 3 delivered the per-session confined runner's structural
architecture: seat-flip complete, cgroup migration, full plugin
migration, 6 integration-test-cycles-discovered fix chains closed.
Phase 4 picks up the **deferred sub-tracks** from Phase 3 (carryover
backlog) **plus the performance-pass work** the parent design
identified as the natural Phase 4 scope.

Phase 4 is **NOT** Phase 5 (production hardening) or Phase 6
(cleanup + cross-platform).  Those have their own scope per the
parent design.

Section references (§3 / §4.x) point into the parent design doc
unless noted.

---

## 1. Goal + scope

**Goal.** After Phase 4 lands:

- All Phase 3 deferred sub-tracks have either shipped or been
  explicitly re-deferred to Phase 5+ with a concrete reason.
- Performance baseline is established: runner spawn latency, per-RPC
  overhead, RAM-per-runner.  Measured with `pytest-benchmark` or
  equivalent; recorded as regression-ratchet thresholds.
- Idle-runner-shutdown knob is wired (operator-side cost regression
  hatch for deployments hosting many sessions).
- The 8 audit-disciplines codified in Phase 3 closure recap §"What
  the next phase should preserve" are applied to every Phase 4
  architectural commit.

**Out of scope:**
- Phase 5: production hardening (multi-tenant acceptance gate,
  per-operator policy store, capability-based fragment loading).
- Phase 6: cleanup (daemon AppArmor profile, remove vestigial
  machinery, cross-platform runner).
- Daemon-side AppArmor profile.  Parent §6.3 explicitly defers this
  to Phase 6.

## 2. Critical-path constraint

Per Phase 3's audit-discipline lesson, **audit before code at every
architectural decision point**.  Phase 4 tasks split into two
buckets by audit requirement:

**Carryover sub-tracks** (audit largely pre-existing in Phase 3
closure recap + their respective backlog entries):

- §4.1 J.A (`call_id` propagation in PromptPayload)
- §4.2 J.B (`editable_metadata` schema lookup)
- §4.3 §3.11 isolated-subagent opt-in
- §4.4 Finding 2 (description-callback regression)

**New Phase 4 tasks** (need pre-implementation audit before code):

- §4.5 Performance baseline (audit which metrics, which test fixtures)
- §4.6 Idle-runner-shutdown knob
- §4.7 Multi-turn UX investigation (cycle-13 "16 PermissionRequestedEvents
  + Error: 1bcd84..." — verify whether J.A/J.B resolve it; if not,
  audit the actual root cause)

Carryover tasks (§4.1–§4.4) can ship in any order; they're independent.
New tasks (§4.5–§4.7) get their own audit commits per the discipline.

---

## 3. Task-by-task breakdown

### 3.1 — §4.1 `call_id` propagation in PromptPayload (J.A)

**Status:** backlog entry shipped at
`docs/design/project_backlog_path_j_sub_gaps_call_id_editable_metadata.md`.
~30 LoC change.

**Files touched:**
- `shared/plugins/permission/types.py` (or wherever `PromptPayload`
  is defined) — add `call_id: Optional[str] = None` field.
- `shared/plugins/permission/runner_rpc_channel.py` — populate
  `call_id` when constructing the `PromptPayload` for the ASK relay.
- `server/runner_rpc_handlers/prompt_operator.py` — pass `call_id`
  through to the emitted `PermissionInputModeEvent` (already-shipped
  Path J emit, just needs the field wired).

**Tests:** ~5 regression pins (field declaration + round-trip +
parallel-tool-block correlation case + backward-compat empty-call_id).

**One commit.**

### 3.2 — §4.2 `editable_metadata` schema lookup (J.B)

**Status:** backlog entry shipped (same doc as J.A).  ~40-60 LoC.
Requires `permission_plugin` reference inside `PromptOperatorHandler`
OR enriched `PromptPayload`.

**Open clarification (audit needed first):** which approach?

- **Option A** — `PromptPayload` carries `editable_metadata: Optional[dict]`
  populated at the runner-side ASK site (the runner has the
  permission plugin instance already).  Smaller surface.
- **Option B** — `PromptOperatorHandler` gets a daemon-side
  `permission_plugin` reference at session-bootstrap, looks up the
  schema at handler invocation.  Daemon-side coupling.

**Lean (subject to audit):** Option A — keeps the daemon-side
handler stateless re permission details.

**Tests:** ~6-8 regression pins (schema field present, edit-and-approve
flow works end-to-end, backward-compat with empty metadata).

**Audit commit + 1-2 implementation commits.**

### 3.3 — §4.3 §3.11 isolated-subagent opt-in

**Status:** backlog entry shipped at
`project_backlog_3_11_isolated_subagent_opt_in.md`.

**Scope:** wire `agent_params.isolated: true` to spawn a fresh
runner with a sub-AppArmor profile (`jaato-ws-{session_id}//{subagent_id}`)
+ sub-cgroup.  Default-share path already shipped in Phase 3.

**Files touched:**
- `shared/plugins/subagent/plugin.py` — branch on
  `agent_params.isolated`; default → share parent's runner;
  opt-in → call `SessionManager._spawn_isolated_runner(...)`.
- `server/session_manager.py` — add `_spawn_isolated_runner` helper
  reusing `_spawn_session_runner_unconditional` machinery with a
  sub-profile name.
- `server/apparmor.py` — sub-profile generation (likely already
  supports `parent_name//{child}` syntax; verify).

**Tests:** integration test gated on `_can_migrate_to(_find_writable_cgroup_parent())`
exercising both default-share AND opt-in isolated flows.

**Audit commit + 1-2 implementation commits.**

### 3.4 — §4.4 Finding 2 (description-callback regression)

**Status:** backlog entry shipped at
`project_backlog_description_callback_gap.md`.

**Scope:** `description_updated` event flows daemon-side pre-§7c but
post-§7c the runner-side session has no install hook for the
description callback.  Same architectural shape as Finding 3 (which
Path F closed via NotificationFrame extension).

**Files touched:**
- `runner/rpc.py` — extend `_install_session_notification_callbacks`
  to also install a description-callback shim; emit a new
  `description_updated` event_type.
- `core.py` — extend the demuxer's Path F.3 branch table with the
  new event_type → ServerAgentHooks dispatch.

**Tests:** ~5-7 regression pins matching Path F's cadence.

**Single commit (likely sub-§7b.2 size — mechanical extension of
Path F's pattern).**

### 3.5 — §4.5 Performance baseline

**Pre-implementation audit prerequisite** (mirror Phase 3 audit-disciplines):
- WHICH metrics to baseline (spawn latency, RPC overhead, RAM, etc.)
- WHICH test fixtures (real-provider end-to-end? offline mock?
  unit-level synthetic load?)
- WHICH regression-ratchet thresholds (p50/p95/p99)

**Lean (subject to audit):**

- Session-create p95: target ≤500ms (spawn + bootstrap + RPC handshake)
- `session.send_message` per-RPC overhead: target ≤5% of model-API wall time
- Idle-runner RSS: target ≤30 MB

**Files touched:**
- New `tests/integration/test_perf_baseline.py` with parametrized
  micro-benchmarks gated behind `pytest.mark.perf`.
- New `docs/perf/baseline.md` recording current measurements +
  regression thresholds.

**Audit commit + 1 implementation commit.**

### 3.6 — §4.6 Idle-runner-shutdown knob

**Status:** parent design §4.6 line 575 mentions "operators with high
session counts can flip a daemon-level idle-runner-shutdown knob (out
of scope for v1)" — v1 was Phase 3.  Phase 4 lands the knob.

**Scope:** daemon-side configuration option that, when enabled,
issues `runner.shutdown` to runners that have been idle for N
seconds.  Saves ~30 MB RSS per stale session.

**Files touched:**
- `server/__main__.py` — new daemon config field (env var
  `JAATO_RUNNER_IDLE_SHUTDOWN_SECONDS`, default disabled).
- `server/session_manager.py` — periodic idle-scan task; tracks
  per-runner last-activity timestamp; issues shutdown when idle
  exceeds threshold.
- `server/runner_spawn.py` — runners restart-on-demand for
  shutdown sessions (next session.send_message re-spawns).

**Open clarification (audit):** restart-on-demand vs session-failure
on shutdown?  The former is more user-friendly but adds spawn
latency on next call; the latter is simpler but surfaces the idle
shutdown as a visible failure.

**Tests:** integration test with low idle-threshold (e.g., 5s);
verify runner shut down + next call respawns successfully.

**Audit commit + 1-2 implementation commits.**

### 3.7 — §4.7 Multi-turn UX investigation

**Status:** flagged in Phase 3 closure recap "Cycle-13 multi-turn UX
hint" — investigate whether J.A or J.B explains the cumulative 16
PermissionRequestedEvents + "Error: 1bcd84..." rendering.

**Scope:** read-only investigation (no code changes initially).
Determine if the multi-turn loop is fixed by §4.1 / §4.2 OR if
there's a separate gap.

**Acceptance:** either (a) verified to be fixed by §4.1+§4.2 (closes
the investigation), or (b) audit doc filed for a separate fix
(promotes to §4.8 or follow-up).

**Single audit commit.**

---

## 4. Test plan + acceptance gate

**Acceptance gate.** Phase 4 is done when:

1. Test #3 (ASK round-trip) extended to multi-turn workflow:
   model makes 5+ tool calls in sequence; operator approves each;
   all complete without spurious re-prompts.  Catches J.A/J.B
   regressions if present.
2. §3.11 isolated-subagent integration test passes (both
   default-share and opt-in paths).
3. Performance baseline test passes the ratchet thresholds; baseline
   doc recorded.
4. Idle-runner-shutdown integration test (with low threshold)
   passes: runner shuts down, next call respawns.
5. All Phase 3 closure-recap-flagged sub-gaps either shipped (with
   tests) or re-deferred to Phase 5+ with explicit reason.

Combined runtime budget: ~5 minutes (heavier than Phase 3's 3-min
budget because of the perf-baseline test and multi-turn integration).

---

## 5. Open clarifications

These are the genuinely controversial decisions worth surfacing
before any code lands.

### 5.1 J.B Option A (envelope) vs Option B (handler reference)

**Option A — `PromptPayload` carries `editable_metadata`.**
- Pros: Smaller surface; runner-side handler stays focused on relay;
  no new daemon-side coupling.
- Cons: PromptPayload grows; serialization surface expands.

**Option B — `PromptOperatorHandler` gets `permission_plugin` reference.**
- Pros: Daemon-side handler can look up any plugin state on demand;
  more flexible for future plugin additions.
- Cons: Daemon-side coupling; PromptOperatorHandler's surface grows.

**Lean: Option A.** Matches the Phase 3 pattern of "carry state in
envelope, not via daemon-side references" (e.g., §3.3a's
SessionInitEnvelope, Path C's runtime project/location fields,
Path D's plugin list).

### 5.2 Idle-shutdown: restart-on-demand vs session-failure

**Option A — restart-on-demand**: next `session.send_message` after
shutdown transparently re-spawns the runner; user sees a small
latency spike.

**Option B — session-failure**: shutdown marks the session as
`failed`; next call returns an error; operator explicitly resumes.

**Lean: Option A.**  Better UX for deployments with sporadic activity;
the spawn latency is acceptable; the dual-path complexity in option B
isn't worth the marginal explicitness.

### 5.3 Performance baseline test fixtures

**Option A — real-provider end-to-end** (against zhipuai/glm-5-turbo
or similar).
- Pros: Realistic measurements.
- Cons: Network-dependent; rate-limit sensitive; CI flakiness.

**Option B — offline mock provider**.
- Pros: Deterministic; CI-stable.
- Cons: Mock provider may not reproduce real timing characteristics.

**Option C — hybrid** (offline for CI, real-provider for manual
ratchet review).
- Pros: Best of both.
- Cons: Two test suites to maintain.

**Lean: Option C.** Phase 3's integration-test cycles taught us that
real-provider tests catch class of bugs offline tests can't; but
those tests aren't CI-stable.  Hybrid keeps CI green while
preserving real-provider verification at decision points.

---

## 6. Risk register

**6.1 Multi-turn UX regression class (J.A/J.B).** Cycle-13's 16-prompt
loop may be a deeper bug than J.A+J.B suggest.  Mitigation: §4.7
investigation runs FIRST; if J.A+J.B don't close it, the actual
root cause becomes §4.8.

**6.2 Performance regression from Phase 3 baseline.** Phase 3 didn't
measure session-create latency; we may discover that the seat-flip
+ 6 fix paths added noticeable overhead.  Mitigation: §4.5 baseline
is the first measurement; thresholds are set per the measurement,
not pre-committed.

**6.3 §3.11 isolated-runner spawn-cost amplification.** Each
isolated subagent costs spawn-latency + RSS.  For deep cascades
this could be material.  Mitigation: profile during §4.5; if
amortization is poor, surface as a Phase 5 perf concern.

**6.4 Backlog drift.** Phase 3 closure recap lists 4 explicit sub-gaps
+ 1 investigation.  Phase 4 may surface additional layers via the
acceptance-gate multi-turn test.  Mitigation: same audit-discipline
that ran 20-for-20 in Phase 3; backlog entries filed for any
findings not in scope.

**6.5 Phase 4 vs Phase 5 boundary creep.** Performance-related work
sits between hardening (Phase 5) and basic baseline (Phase 4).
Mitigation: §4.5 establishes baseline only; any optimization work
that's not a simple ratchet-fix gets deferred to Phase 5.

---

## 7. Out-of-scope (Phase 5, Phase 6)

This list is informational; not a Phase 4 task list.

- **Phase 5 (production hardening)**: multi-tenant correctness
  end-to-end acceptance gate (per parent §8); per-operator policy
  store (cross-session permission rules); capability-based fragment
  loading; daemon-side AppArmor profile prep (without landing it
  yet — Phase 6 lands it).
- **Phase 6 (cleanup + cross-platform)**: daemon AppArmor profile;
  remove `apparmor_confine` thread-context machinery; remove
  `SafeThreadPoolExecutor` apparmor-recovery hook helper; cross-
  platform runner (macOS/Windows compatibility).

---

## 8. Audit-discipline inheritance from Phase 3

Phase 4 commits should explicitly inherit the 8 disciplines codified
in Phase 3 closure recap §"What the next phase should preserve":

1. Pre-implementation grep before architectural commits.
2. Audit-doc commits before implementation commits at architectural
   decision points.
3. In-flight scope narrowing when implementation cross-grep surfaces
   audit gaps.
4. Backlog discipline for adjacent findings — capture, don't fold.
5. Inverse-virtue activations — cancel proposed work when audit
   reveals no consumer.
6. Worker-correction over reviewer-deference — bias toward upstream
   localization.
7. Probe-driven localization for silent failures (when no errors fire);
   strip probes once localization closes.
8. Per-callback-TYPE + per-callback-SCOPE enumeration for any
   in-process-to-RPC migration.

For Phase 4 specifically, the new addition worth flagging:

9. **Real-provider integration tests at every architectural change**
   (Phase 3's cycle-1-13 lesson).  Unit-test mocks structurally
   couldn't reach the 10 architectural layers Phase 3's integration
   cycles exposed.  Multi-turn integration testing is part of
   §4.7's acceptance gate.

---

End of plan.  Estimated calendar: 2-3 weeks of focused work; carryover
sub-tracks (§4.1–§4.4) are mostly mechanical with audit-discipline
gates; new tasks (§4.5–§4.7) need their own audit cycles.

PR-size estimate: deferred until §4.5 baseline lands and we know how
much perf-test infrastructure adds.  Prior plan-size estimates have
been understated by ~20-40% (per Phase 2/3 retrospective); calibrate
upward.
