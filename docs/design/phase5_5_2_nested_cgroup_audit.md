# Phase 5 §5.2 — Nested cgroup structure: Phase 6 starting reference

**Status:** **Phase 6 deferred.**  Phase 5 §5.2 shipped as visibility-
only instrumentation (PR #62, commit `2a26e6f`).  Companion audit
recording the deferral rationale + scope of the visibility ship:
`phase5_5_2_nested_cgroup_deferral_audit.md`.

This document captures the design that **will** land when Phase 6
picks the work up.  It exists so the Phase 6 worker doesn't have to
re-derive the approach, limit policy, or decomposition.

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.2.
**Ledger item:** `phase4_implementation_audits.md` §4.3.9 item 2.
**Predecessor design:** Audit 7 (`phase4_implementation_audits.md`
§"Sub-cgroup structure decision: SIBLING, not NESTED").
**Trigger condition for revisiting** (from the standdown):
1. PR #62's nesting-visibility INFO log fires repeatedly on a real
   deployment, OR
2. A multi-tenant operator files a concrete request for parent-
   bounded composition.

---

## 1. Problem (what §4.3.5 deferred)

Today's sub-cgroup layout:

```
/sys/fs/cgroup/jaato/
├── jaato-ws-{parent}/                  (parent runner PID lives here)
└── jaato-ws-{parent}__sub_{subagent}/  (sub-runner PID lives here)
```

Sibling structure → sub-runner bounds are **independent** of the
parent's bounds.  A sub-runner with `memory.max=4GiB` gets 4 GiB
even if the parent has `memory.max=2GiB`.  Audit 7's design intent
("untrusted code-execution subagent inside an otherwise-trusted
workspace") matched this: the supervisor's bounds are about its
own work, not about caging the subagent.

Audit 7 flagged the Phase 5+ follow-up:

> if a future use case needs composable bounds (sub-cgroup bounded
> by parent's bounds), the restructuring (`provision_cgroup` migrates
> the runner to `/main/`, enables `cgroup.subtree_control` on parent,
> creates child) lands as a single coordinated change.  Out of scope
> for §4.3.5.

When Phase 6 picks this up, §6 below is the agreed decomposition.

## 2. cgroup v2 structural constraint

A cgroup v2 directory can be either:

- **A leaf** — contains PIDs in `cgroup.procs`, no child cgroups
  with `subtree_control` enabled.
- **A container** — has child cgroups with `subtree_control`
  enabled, contains no PIDs directly.

Not both.  The "no internal processes" rule means that to nest
under `jaato-ws-{parent}/`, the parent runner's PID must first move
to a `main/` leaf:

```
/sys/fs/cgroup/jaato/
└── jaato-ws-{parent}/                  (container — no PIDs)
    ├── cgroup.subtree_control: "memory pids cpu"
    ├── main/                           (parent runner PID here)
    └── sub_{subagent}/                 (sub-runner PID here)
```

The umbrella limits live on `jaato-ws-{parent}/` itself; per-leaf
limits live on `main/` and `sub_*/`.  Kernel enforces `min(parent,
leaf)` for `memory.max` and `pids.max`; `cpu.weight` is fair-share
across siblings under the umbrella.

## 3. Approach options

### Approach A — Universal nesting (eager restructure)

All session cgroups use the container + main layout, regardless of
whether they ever spawn sub-runners.  Simplest design; fixed
provisioning cost per session.

**Pros:**
- One code path; no flag plumbing.
- Future sub-runner spawn just creates a sibling under the parent's
  container — no late-stage migration.
- Single mental model for operators inspecting the cgroup tree.

**Cons:**
- Touches every session-spawn path (IPC, WS, disk-restore).
- Bigger blast radius — every existing cgroup test needs updating.
- Adds the `main/` indirection for sessions that never use it.
- Backward-compat: daemon restart with running sessions may find
  cgroups left over from the old (sibling) layout; cleanup logic
  must tolerate both shapes for one release.

### Approach B — Per-session opt-in via profile (recommended for Phase 6 v1)

A profile field (`enable_nested_subagents: bool = false`) signals
intent.  When true, that session's cgroup is provisioned in container
mode (with `main/` leaf for the parent's own PID); sub-runners spawn
as nested leaves.  When false, today's sibling structure applies
unchanged.

**Pros:**
- Backward-compatible by construction — sessions without the flag
  see no behavior change.
- Honest about the trade-off: nesting costs the `main/` indirection
  in exchange for composable bounds, and the supervisor picks per
  session whether it's worth it.
- Audit 7's "if a future use case needs composable bounds" framing
  reads naturally as an opt-in, not a wholesale shift.

**Cons:**
- Two layouts to maintain in `CgroupsManager`.
- Sub-runner spawn must check the parent's mode before deciding
  where to place itself.

### Approach C — Lazy promotion on first sub-runner spawn

Parent provisions as a leaf; first sub-runner spawn promotes it:
migrate parent runner PID to a new `main/` cgroup, enable
`subtree_control`, then create the sub-leaf.

**Pros:**
- Zero cost for sessions that never spawn sub-runners.

**Cons (load-bearing rejection):**
- Live runtime migration of an active runner PID.  Migration
  between cgroups under load can race against subprocesses the
  runner is concurrently spawning (the runner's `make_attach_callback`
  has captured the OLD path) — silent confinement-loss for any
  subprocess forking during the migration window.
- Recovery semantics from a half-completed promotion are murky.
- The lazy-promotion code path is exercised only when supervisors
  actually opt into isolation, which is rare today — low signal,
  high stakes.

**Reject Approach C** — runtime migration of a confined process is
exactly the class of operation that should not happen at all.

## 4. Recommended design — Approach B

### 4.1 Profile field

Add to `SubagentProfile` (`shared/plugins/subagent/config.py`):

```python
# Whether this session's cgroup is provisioned in container mode
# so isolated subagents can be nested underneath (parent-bounded
# bounds).  False (default) keeps today's sibling layout — the
# session's cgroup is a leaf with the parent runner's PID and
# `cgroup.subtree_control` empty.  See Phase 5 §5.2 audit doc.
enable_nested_subagents: bool = False
```

Resolved through `build_inline_profile` + inheritance rules as a
scalar-override field (matches `runtime_limits` cadence).

### 4.2 CgroupsManager API surface

Two new paths derived from a single `session_id`:

```python
def get_cgroup_path(self, session_id: str) -> Path:
    """The session's container path: `{root}/jaato-ws-{session_id}/`."""

def get_cgroup_main_leaf_path(self, session_id: str) -> Path:
    """The session's main leaf: `{root}/jaato-ws-{session_id}/main/`.
    Only meaningful when the session opted into nesting."""

def get_sub_cgroup_path(
    self, parent_session_id: str, subagent_id: str,
) -> Path:
    """A nested sub-runner leaf:
    `{root}/jaato-ws-{parent}/sub_{subagent}/`.  Only used when
    the parent opted into nesting."""
```

`provision_cgroup` gains a `nested: bool` kwarg.  When `False`
(today's default) it behaves identically.  When `True`:

1. Create `{root}/jaato-ws-{session_id}/` (the container).
2. Write the umbrella limits onto it (full RuntimeLimits).
3. Enable `memory pids cpu` in
   `{root}/jaato-ws-{session_id}/cgroup.subtree_control`.
4. Create `{root}/jaato-ws-{session_id}/main/` (the leaf).
5. Write the same RuntimeLimits onto `main/` (Policy 4.4.a — see §4.4).

`make_attach_callback(session_id)` consults the nested flag at
provision time — when the session is nested, the callback targets
`main/cgroup.procs` instead of the container's `cgroup.procs`.
Source-of-truth: the manager records the per-session layout on
provision and reads it back on attach.

### 4.3 Sub-runner spawn path

`SessionManager._spawn_isolated_runner` gets a new pre-check:

- If the parent session is in nested mode, the sub-runner's
  cgroup is created at `{root}/jaato-ws-{parent}/sub_{subagent}/`,
  inheriting the parent's umbrella limits.
- If the parent session is NOT in nested mode, today's sibling
  layout applies (no change from §5.1).

A new manager method `provision_sub_cgroup_nested(parent_session_id,
subagent_id, config)` creates the nested sub leaf.  Caller chooses
between this and the existing sibling provision based on the
parent's mode.

The sub-runner's own `runtime_limits` (post-§5.1 defaulting via
`apply_isolated_defaults`) writes to the nested
`sub_{subagent}/`'s controller files, exactly as it would on the
sibling layout.  cgroup v2 enforces `min(umbrella, sub)` at runtime
— that's the cap-stacking contract operators reason against.

### 4.4 Umbrella limit policy — Policy 4.4.a (pinned)

**Decision: Policy 4.4.a.**  Single declared RuntimeLimits value
written to both the umbrella AND the `main/` leaf.

Reasoning:

- cgroup v2's `min(parent, child)` semantics give parent-bounded
  composition for free, which is exactly what §4.3.9 item 2 was
  filed to enable.  Umbrella has `memory_max_mb=2048` → `main`
  writes its 2048 → any sub-leaf that declares more (e.g., 4096) is
  automatically capped at the umbrella's remaining budget.  The
  supervisor's whole isolated-subagent tree can't exceed the
  parent's declared envelope.
- Simple mental model wins for the v1 ship.  One set of values per
  profile, no new fields, no umbrella-vs-leaf reasoning for profile
  authors.
- Forward-compatible: Policy 4.4.b adds an optional
  `profile.umbrella_runtime_limits` field; when absent, the
  framework falls back to 4.4.a semantics (umbrella value = main
  value).  The data model evolution is contiguous.

**Operator caveat to document in rollout notes:** cgroup v2 charges
memory to the leaf where it was allocated but aggregates
`memory.current` across all children of the umbrella.  A sub-leaf's
"remaining budget" is **dynamic** — depends on `main`'s instantaneous
usage.  Heavy main + heavy sub will OOM the sub first (main was
charged earlier).  Right semantic but counterintuitive on first
encounter; needs a callout in the operator-facing rollout doc.

**Phase 6.1 follow-on — Policy 4.4.b.**  An optional explicit
umbrella budget separate from per-leaf caps.  Use case: "main runner
2 GiB, sub can use up to 4 GiB" asymmetric budgets.  Uncommon in the
multi-tenant deployments §4.3.9 item 2 targets (those want bounded
subs, not subs with extra headroom).  Defer until an operator files
an asymmetric-budget request; ship 4.4.a's forward-compat hooks now
so the migration is purely additive.

### 4.5 Teardown semantics

Container teardown order (cgroup v2 requires leaves removed
before the container):

1. Kill + rmdir every `sub_*/` leaf (already invoked per sub-runner
   teardown via existing `teardown_cgroup`; just needs the path
   to be aware of the nested layout).
2. Kill + rmdir `main/`.
3. Rmdir the container itself.

Cleanup is best-effort with INFO-level logging if any step fails;
orphaned cgroups are listed by the daemon's startup scan (Phase 5
§5.3 — that work is independent and lands first under the current
plan).

### 4.6 Backward compatibility

The flag defaults to `False`.  Existing profiles + sessions remain
on the sibling layout — no migration needed.  A profile setting
`enable_nested_subagents: true` AFTER a session has already been
provisioned is honored on next session creation; in-flight sessions
keep their current layout (which is fine: the flag is per-session,
not global).

Operator setup (see `cgroups.py` module docstring) is unchanged —
the existing `/sys/fs/cgroup/jaato/` root still hosts both layouts.

PR #62's visibility instrumentation continues working — when
nesting is enabled for a session, the visibility log goes quiet for
that session's sub-spawns (sub bounds DO compose under parent's;
the "exceeds" condition can no longer trigger silent
over-allocation).  Sessions without the flag still emit the
visibility log as before.

## 5. Decisions pinned for the Phase 6 picker-upper

The standdown response confirmed these as the v1 design:

| Question | Decision | Rationale source |
|---|---|---|
| Approach | **B — per-session opt-in** | §3 above; backward-compat + honest trade-off |
| Limit policy | **4.4.a — single value on umbrella + main** | §4.4 above; cap-stacking via min(parent, child) |
| Decomposition | **Staged sub-commits §5.2a–d** | §6 below; per-stage real-host verification |
| Field name | `enable_nested_subagents: bool = False` | §4.1; verbose but self-documenting |
| Sub-runner own limits | Write to `sub_{subagent}/`'s controllers | §4.3; cap-stacking layering |

If you disagree with any pin when picking this up, file a scope
re-decision audit per Phase 4 §4.3.0 / audit-discipline rule 2
before code lands.

## 6. Decomposition into sub-commits (pinned)

The §4.3 sub-track precedent is the right ladder for §5.2.

- **§5.2a — audit + scaffolding (this doc + skeleton API).**  Ship
  immediately on any host.  Adds `get_cgroup_main_leaf_path` +
  `get_sub_cgroup_path` to `CgroupsManager` as additive surface
  with no behavior change.  Skeleton tests pin the path templates.
  ~50 LoC + tests.  **No real-host verification required** — pure
  additive seam.  Mirrors the §4.3.1 tracer-bullet stub pattern.
- **§5.2b — provision-side nesting.**  `provision_cgroup(nested=True)`
  builds the container + main layout; `make_attach_callback`
  consults per-session mode.  Profile field + parser update.
  Sub-runner spawn path unchanged — still siblings.  Behavior-
  preserving for any profile that doesn't set the flag.
  **Real-host verification required** per Phase 5 audit-discipline
  rule 3 — provision tree + subtree_control + main leaf
  observable via `/sys/fs/cgroup/...`.
- **§5.2c — sub-runner spawn-side nesting.**
  `_spawn_isolated_runner` consults parent's mode and provisions
  nested sub leaf when applicable.  This is where nested behavior
  becomes visible to supervisors.  **Real-host verification
  required** — OOM test (sub over-allocates past umbrella's
  memory.max; kernel kills sub).
- **§5.2d — teardown + cleanup.**  Nested teardown order; daemon
  startup scan tolerates both layouts.  Composes with §5.3's
  leak-audit pass once §5.3 lands.  **Real-host verification
  required** — parent teardown removes every nested leaf atomically;
  daemon restart reaps orphaned nested leaves.

Staging rationale (from the standdown response):

- §5.2a always ships unconditionally as the API seam.
- §5.2b/c/d each need individual real-host signoff.  Staging lets
  each ship as soon as its verification clears; single-PR gates the
  whole landing on the slowest verification step.  Phase 4 §4.3.5
  / §4.3.6 / §4.3.8 each carried their own manual-verification
  gate — same cadence here.
- Revert safety.  Smaller blast radius per commit makes a real-host
  regression easier to revert without losing the other sub-commits'
  work.

## 7. Real-host verification (Phase 6 worker's playbook)

Per Phase 5 audit-discipline #3, operators rolling §5.2b/c/d
should verify on a cgroup-v2 + AppArmor-enabled real host:

1. Provision a session with `enable_nested_subagents: true`.
2. Verify `/sys/fs/cgroup/jaato/jaato-ws-{sess}/` exists with
   `cgroup.subtree_control` listing `memory pids cpu` and
   `cgroup.procs` empty.
3. Verify `.../main/cgroup.procs` contains the parent runner PID.
4. From the supervisor, spawn an isolated subagent.
5. Verify `.../sub_{subagent}/cgroup.procs` contains the sub-runner
   PID.
6. OOM test: have the sub-runner allocate past the umbrella's
   `memory.max`; observe kernel kills it even when the sub's own
   `memory.max` would allow more.
7. Tear down the parent session; verify all nested leaves are
   removed atomically.

These steps belong in the AppArmor setup guide alongside the
§4.3.9 playbook.

## 8. Phase 6+ carryover from this audit

- **Policy 4.4.b (asymmetric umbrella budget).**  Documented in §4.4
  as a follow-on with forward-compat migration via an optional
  `profile.umbrella_runtime_limits` field.  Ship when an operator
  files an asymmetric-budget request.
- **Approach A (universal nesting as default).**  Documented in §3
  as the "cleaner long-term" alternative.  Candidate for Phase 7+
  only after §5.2b–d (Approach B) has had real-host soak time.

---

End of audit (Phase 6 reference draft).  When the trigger condition
fires, this doc + the deferral audit (`phase5_5_2_nested_cgroup_
deferral_audit.md`) together give the Phase 6 worker the full
context to ship without re-derivation.
