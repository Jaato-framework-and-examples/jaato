# Phase 5 §5.1 — Default `RuntimeLimits` for isolated subagents

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.1.
**Ledger item:** `phase4_implementation_audits.md` §4.3.9 item 1.
**Theme:** A — Resource-isolation defaults (security).

Companion items in Theme A (§5.2 nested cgroups, §5.3 cgroup-leak audit) get
their own audit docs.  Per Phase 5 plan §3, Theme A items can ship
independently — they share machinery but not implementations.

---

## 1. Problem

`SessionManager._spawn_isolated_runner` provisions the sub-cgroup only
when both:

1. `CgroupsManager.is_available()` returns True, and
2. `profile.runtime_limits is not None and runtime_limits.has_kernel_limits()`.

The second predicate makes the entire resource isolation **conditional on
the supervisor having written kernel-enforceable fields into the
profile**.  Profiles that omit `runtime_limits` (or supply only the
application-enforced subset) silently skip cgroup creation — the
sub-runner inherits the daemon's default cgroup.

This contradicts the security mental model the `agent_params.isolated=true`
opt-in establishes: "isolation implies bounds".  A profile author who
forgets to write `runtime_limits` doesn't see the security regression
(no error envelope, no warning) — the sub-runner happens to launch
successfully under no caps.  The existing `INFO`-level log line ("sub-cgroup
skipped … sub-runner will inherit default cgroup") records the gap but
doesn't surface it operationally.

`session_manager.py:1150-1154` flags this gap inline as a known Phase 5
hardening target.  §5.1 closes it.

## 2. Scope of this commit

In-scope:

- Add a module-level `ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS` constant in
  `shared/runtime_limits.py` that carries the conservative default
  (2 GiB / 128 pids / cpu.weight=100 / 120s tool timeout / 1 MiB output).
- Add an `apply_isolated_defaults(supplied)` helper in the same module
  that returns a per-field merge: supplied wins when non-None, default
  fills the rest.
- Patch `SessionManager._spawn_isolated_runner` to call
  `apply_isolated_defaults` immediately after profile reconstruction and
  use the effective limits for both the cgroup provision check and the
  downstream spawn calls.
- Patch `SessionManager._do_spawn_isolated_runner` to forward the
  effective limits' app-layer fields (`max_output_bytes`,
  `tool_timeout_seconds`) into the `RunnerSpawner.spawn(...)` call so the
  runner subprocess's CLI plugin actually sees them via the existing
  `JAATO_RUNNER_*` env passthrough.

Out of scope (Phase 5 follow-ups, recorded so reviewers don't expect
them here):

- **Mainline (non-isolated) runner spawn doesn't pass `max_output_chars`
  / `tool_timeout_seconds` to `RunnerSpawner.spawn` either** — the
  `runner_spawn.spawn_session_runner` call site omits them today.  Fixing
  the mainline path is a separate task (it broadens beyond the
  isolation-implies-bounds invariant §5.1 is scoped to).  Tracked
  internally — file a follow-up audit when picked up.
- **Wiring `runtime_limits` into `SessionInitEnvelope`** — today the
  schema doesn't carry it, so app-layer fields flow only via the
  spawn-time env-var path.  Adding it to the envelope would let
  GC/permission plugins read effective limits at handler time too, but
  isn't required for §5.1's scope.
- **Theme A §5.2 (nested cgroups), §5.3 (cgroup-leak audit at session
  shutdown)** — own audit docs when picked up.
- **Phase 6 daemon AppArmor profile / cross-platform** — parent design
  §6.3.

## 3. Architectural decisions

### 3.1 Default values — kernel layer

| Field | Default | Rationale |
|---|---|---|
| `memory_max_mb` | 2048 (2 GiB) | Plan-stated.  Comfortable for tool-running subagents; tight enough to OOM-kill runaway processes. |
| `pids_max` | 128 | Plan-stated.  Generous for shell/cli workloads; rejects fork-bomb classes outright. |
| `cpu_weight` | 100 | Plan-stated.  Equal weighting with peer cgroups (cgroup v2 default).  Not a hard cap — just a fair-share lever under contention. |

### 3.2 Default values — application layer

| Field | Default | Rationale |
|---|---|---|
| `tool_timeout_seconds` | 120.0 | Conservative wall-clock cap on individual `subprocess.run(...)` invocations.  Long enough for legitimate test runs / package installs; short enough to cap runaway. |
| `max_output_bytes` | 1_048_576 (1 MiB) | Conservative truncation of captured stdout/stderr.  Prevents pathologically chatty tools from blowing the runner's heap or saturating the wire. |

The application-layer defaults are forward-compatible with how the wire
path is set up today: `RunnerSpawner.spawn` accepts
`max_output_chars` + `tool_timeout_seconds` kwargs and forwards them as
`JAATO_RUNNER_*` env vars; the runner-side cli plugin already reads
those env vars at startup.  §5.1 wires the isolated spawn call to pass
the effective values; the mainline non-isolated path can adopt the same
pattern in a follow-up.

### 3.3 Merge semantics — per-field, supplied wins

A profile that sets `RuntimeLimits(memory_max_mb=4096)` opts into a
heavier memory cap; we honour it but still fill `pids_max=128 /
cpu_weight=100 / tool_timeout_seconds=120 / max_output_bytes=1MiB` from
the default.  This matches the "isolation implies bounds" invariant
without overriding deliberate per-field tightenings (or loosenings).

Edge cases:

- **`profile.runtime_limits is None`** — default applies to all fields.
- **`profile.runtime_limits = RuntimeLimits()` (all None)** — default
  applies to all fields.  Functionally identical to None per the merge
  helper, but semantic intent differs (explicit-empty vs absent).
- **Profile sets only `tool_timeout_seconds`** — kernel fields fill from
  default; supplied app-layer field wins.
- **Profile sets `cpu_weight=1`** (intentionally starved) — default
  doesn't override; explicit minimum is honoured.
- **`runtime_limits.extra` (forward-compat dict for unknown keys)** —
  preserved verbatim from the supplied value; defaults don't touch it.

### 3.4 Audibility

The existing `_spawn_isolated_runner` `INFO`-level log at line 1209-1213
(sub-cgroup provisioned) and 1215-1225 (sub-cgroup skipped) stays in
place.  After §5.1 the "skipped" branch only triggers when
`CgroupsManager` is unavailable on the host — the
`has_kernel_limits()` predicate always returns True for the default,
so a kernel-with-cgroups host always provisions.

A new `INFO`-level log at apply time records when the default was
applied + which fields filled from default (audit-trail visibility for
operators reviewing isolated-subagent activity).

### 3.5 Backwards compatibility

A profile that previously launched with `agent_params.isolated=true` and
no `runtime_limits` would have skipped cgroup creation entirely.  After
§5.1 it gets 2 GiB / 128 pids / cpu.weight=100.  This is a behaviour
change.  Mitigations:

- Default values are generous — typical subagent workloads (LLM-driven
  tool calls) stay well under all caps.
- Operators who really want unbounded isolated subagents can use
  `agent_params.isolated=false` (the default-share path) to get the
  parent's cgroup membership without the per-subagent caps.
- Per the user's plan-Q3 decision: there's no per-field opt-out for the
  isolation-implies-bounds invariant.  Setting `runtime_limits: {}`
  doesn't bypass the default — only declining isolation does.

## 4. Test plan

Regression pins in `server/tests/test_spawn_isolated_runner_helper.py`
(or a new sibling if signal volume warrants).  Each test names the
property it pins, never the implementation:

1. `test_default_applies_when_profile_omits_runtime_limits` —
   profile with `runtime_limits=None` → cgroup provisioned with default
   memory/pids/cpu_weight.
2. `test_default_applies_when_profile_supplies_empty_runtime_limits` —
   profile with `runtime_limits={}` (all-None) → same defaults applied.
3. `test_supplied_kernel_fields_win_per_field` — profile with
   `memory_max_mb=4096` → cgroup uses 4096 for memory + default for
   pids/cpu_weight.
4. `test_supplied_app_fields_win_per_field` — profile with
   `tool_timeout_seconds=30` → spawner receives 30 for timeout +
   default for output cap.
5. `test_app_layer_defaults_wired_to_spawner` — `RunnerSpawner.spawn`
   call site receives non-None `max_output_chars` +
   `tool_timeout_seconds` derived from the effective limits.
6. `test_extra_passthrough_preserved` — `RuntimeLimits(extra={"foo":1})`
   survives the merge (forward-compat).
7. `test_default_does_not_mutate_supplied` — calling the helper twice
   with the same input returns equal-but-distinct instances; no aliasing
   of the module-level default.

Plus one merge-helper unit test in `shared/tests/test_runtime_limits.py`
(or sibling) — `test_apply_isolated_defaults_merges_per_field`.

## 5. Real-host verification recommendation

Per Phase 5 audit-discipline #3 (manual real-host verification for
security-sensitive primitives), operators rolling §5.1 into production
should run a cgroup-v2 + AppArmor-enabled real-host check:

1. Spawn a top-level session under apparmor opt-in.
2. From the supervisor, invoke `spawn_subagent(...,
   agent_params={"isolated": true})` with a profile that omits
   `runtime_limits`.
3. Verify `cat /sys/fs/cgroup/jaato/jaato-ws-{parent}__sub_{subagent}/memory.max`
   shows `2147483648` (2 GiB in bytes).
4. Verify `cat .../pids.max` shows `128`.
5. Verify `cat .../cpu.weight` shows `100`.
6. Fork-bomb test: have the subagent run `:(){ :|:& };:` under cli; observe
   pids cap halts it.

These steps belong in the AppArmor setup guide alongside the §4.3.9
playbook; §5.1 references but does not extend that guide.

## 6. Open questions

None.  Plan-Q1 (default values), plan-Q2 (explicit-empty semantics),
and plan-Q3 (audit scope) decided 2026-05-11 via AskUserQuestion.

## 7. Phase 6 carryover

Nothing from §5.1 belongs in Phase 6 explicitly.  The mainline-runner
app-field passthrough gap noted in §2 is a Phase 5 follow-up, not a
Phase 6 one.

---

End of audit.
