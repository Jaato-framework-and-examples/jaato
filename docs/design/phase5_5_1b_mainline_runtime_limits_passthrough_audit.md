# Phase 5 §5.1b — Mainline `RuntimeLimits` app-layer passthrough to runner

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.1b
(Theme A).
**Ledger item:** `phase4_implementation_audits.md` §4.3.9 item 11.
**Predecessor:** §5.1 (commit `f486fd0`) closed the same gap for
isolated subagents.

---

## 1. Problem

`shared/runtime_limits.py:RuntimeLimits` carries two enforcement
layers:

- **Kernel-enforced** (cgroup v2): `memory_max_mb`, `pids_max`,
  `cpu_weight` — applied via `CgroupsManager.provision_cgroup`.
- **Application-enforced**: `tool_timeout_seconds`,
  `max_output_bytes` — read by the runner-side cli plugin at
  tool-call time.

The application-enforced layer flows to the runner subprocess via
two env vars that `RunnerSpawner.spawn` sets when its
`max_output_chars` + `tool_timeout_seconds` kwargs are non-None:

```
JAATO_RUNNER_MAX_OUTPUT_CHARS=<int>
JAATO_RUNNER_TOOL_TIMEOUT_SECONDS=<float>
```

But the mainline `runner_spawn.spawn_session_runner` (the IPC + WS
entry point) **never calls `spawn` with those kwargs** — the
function signature didn't accept them, and the call site at line
118 passes only `profile_name` / `session_id` / `workspace_path` /
`log_path` / `disable_confine` / `cgroup_attach`.

Result: a main-session profile that sets
`runtime_limits.tool_timeout_seconds=30` or
`runtime_limits.max_output_bytes=8192` silently has no effect on
the runner subprocess.  The cli plugin uses its compile-time
defaults instead.

§5.1 wired the **isolated-subagent** spawn site
(`SessionManager._do_spawn_isolated_runner`) to forward these
kwargs.  The mainline path remains.

## 2. Scope

In-scope:

- Extend `runner_spawn.spawn_session_runner` to read
  `server._profile.runtime_limits` (the WS path already reads the
  same field for cgroup provision at `websocket.py:622`) and
  forward the app-layer fields to `RunnerSpawner.spawn`.
- When no profile is set, or the profile has no `runtime_limits`,
  pass `None` for both kwargs — `RunnerSpawner.spawn` already
  short-circuits the env-var write in that case.

Out of scope:

- **Defaulting** for mainline sessions.  §5.1's
  `apply_isolated_defaults` is specific to the
  isolation-implies-bounds invariant — mainline sessions don't
  inherit it.  Profiles that omit `runtime_limits` keep the
  runner's compile-time defaults; this commit only closes the
  wiring gap.
- **Disk-restore** envelope construction.  Restored sessions
  rebuild their profile from persisted state and call
  `spawn_session_runner` through the same path, so the fix
  benefits them automatically.
- **Ephemeral subagent fan-out** (default-share).  Default-share
  subagents reuse the parent's runner — no new spawn — so they
  inherit the parent session's caps (or lack thereof) by
  construction.

## 3. Architectural decisions

### 3.1 Source of truth: `server._profile.runtime_limits`

`spawn_session_runner` already takes a `server` arg.  The WS pre-init
hook at `websocket.py:620-624` reads `getattr(cgroup_profile,
"runtime_limits", None)` for cgroup provision.  Reading the same field
inside `spawn_session_runner` mirrors that pattern and keeps the
spawn-time wiring in one place.

Alternative considered + rejected: add explicit
`max_output_chars` / `tool_timeout_seconds` kwargs to
`spawn_session_runner` and have callers compute + pass them.
Adds two parameters that both callers (IPC + WS) would compute
identically from the same `server._profile` — caller-side
duplication for no surface benefit.

### 3.2 No defaulting on the mainline path

§5.1's default values target `agent_params.isolated=true`, where
the opt-in establishes "isolation implies bounds".  Mainline
sessions have no equivalent opt-in — the supervisor manages the
host directly.  A profile that omits `runtime_limits` keeps the
runner's compile-time defaults (e.g., the cli plugin's
`DEFAULT_MAX_OUTPUT_CHARS`), not an injected framework default.

### 3.3 None-safe read

`server._profile` is `None` for inline-spec / no-profile sessions
(parent design §3.3a).  `getattr(server, "_profile", None)`
returns `None`; `getattr(profile, "runtime_limits", None)`
returns `None`; both kwargs forward as `None`; `RunnerSpawner.spawn`
short-circuits the env-var write.  No new code paths added for the
no-profile case.

## 4. Test plan

Regression pins extend
`server/tests/test_runner_spawn.py` (reuses the existing
`_FakeSpawner` + `_FakeJaatoServer` scaffolding so the new tests
exercise the same kwarg-capture path the existing
`test_spawn_calls_spawner_with_session_fields` already pins).
Each test names the property it pins:

1. `test_app_layer_fields_forwarded_when_profile_sets_them` —
   profile with `runtime_limits(tool_timeout_seconds=30,
   max_output_bytes=8192)` → `RunnerSpawner.spawn` receives those
   values verbatim.
2. `test_no_profile_passes_none_for_both_kwargs` — `server._profile
   = None` → both kwargs forwarded as `None`.
3. `test_profile_without_runtime_limits_passes_none` —
   `server._profile.runtime_limits = None` → both kwargs `None`.
4. `test_kernel_fields_do_not_leak_into_app_layer_kwargs` — profile
   that sets only `memory_max_mb=4096` → both app-layer kwargs
   stay `None` (kernel fields don't affect the env-passthrough).
5. `test_partial_app_layer_supplied_forwards_only_set_fields` —
   profile sets `tool_timeout_seconds=30` only → timeout kwarg is
   30.0, output kwarg is `None` (per-field independence).

## 5. Phase 6 carryover

None.  §5.1b is a pure wiring fix; nothing in its surface drifts
toward Phase 6.

---

End of audit.
