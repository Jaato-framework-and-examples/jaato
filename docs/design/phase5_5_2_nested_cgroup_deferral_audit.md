# Phase 5 §5.2 — Nested cgroup layout: defer to Phase 6, ship visibility

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.2
(Theme A).
**Ledger item:** `phase4_implementation_audits.md` §4.3.9 item 2.
**Predecessor:** Phase 4 §4.3.5 audit (the original SIBLING-not-
NESTED structure decision).

---

## 1. Problem

Sub-cgroup structure today is **sibling** to the parent — both live
at `/sys/fs/cgroup/jaato/`:

- Parent: `/sys/fs/cgroup/jaato/jaato-ws-{parent}/`
- Sub:    `/sys/fs/cgroup/jaato/jaato-ws-{parent}__sub_{subagent}/`

Their bounds are **independent**.  A sub declaring `memory.max=4G`
gets 4G regardless of the parent's `memory.max=2G`.  §4.3.5
explicitly chose this because cgroup v2's "no internal processes"
rule forbids a cgroup with both procs AND children-with-
subtree_control — the parent cgroup hosts the parent runner's PID,
so enabling subtree_control would require migrating the parent
runner into a `/main/` sub-cgroup first.  §4.3.5 called that
restructure "coordinated change, not incremental" and deferred it
to §4.3.9 item 2.

The use case for nesting: multi-tenant deployments where the
operator wants a parent's `memory.max` to **cap** every isolated
subagent the parent spawns.  Today an isolated sub can exceed the
parent's bound because cgroups don't compose under sibling layout.

## 2. Decision: defer to Phase 6

**No use case has been filed for parent-bounded composition.**  The
multi-tenant deployments that would need it haven't surfaced.  The
work — touching every `provision_cgroup` callsite + parent-runner
PID migration + dual code paths through `CgroupsManager` — is
non-trivial.  Per audit-discipline rule 2 ("scope re-decision on
uncertainty"), shipping a 150–200 LoC restructure speculatively
when no concrete request drives it is the wrong call.

**Phase 5 §5.2 scope shrinks to instrumentation only.**  When an
isolated subagent is provisioned with a kernel-enforced limit that
exceeds the parent's, the daemon logs an `INFO`-level message
naming both values.  Operators get visibility into where a Phase 6
nested-layout migration WOULD have changed behavior; no production
code path changes.

## 3. Scope

### In-scope

- `SessionManager._do_spawn_isolated_runner` — after the sub-cgroup
  provision step, look up the parent session's `runtime_limits`
  and compare against the sub's effective limits.  Log `INFO` with
  the field-by-field comparison when the sub exceeds the parent
  on `memory_max_mb` or `pids_max` (kernel-enforced caps).  Skip
  `cpu_weight` (share, not cap — different semantics).
- Log the no-parent-limits case explicitly: "parent has no kernel
  runtime_limits; sub bounds are independent."  Useful signal
  that nesting wouldn't have constrained anything anyway.
- Ledger update: `phase4_implementation_audits.md` §4.3.9 item 2 →
  status `Phase 6 deferred`.
- Phase 5 plan §5.2 → status `deferred`; Phase 6 preview (§7) →
  gains the nested-layout item.
- 3 regression pins covering: sub-exceeds-parent (log fires),
  parent-no-limits (log fires with explanation), sub-within-parent
  (log silent).

### Out of scope

- **Any cgroup layout change.**  No `subtree_control` write, no
  parent-runner migration, no `/main/` cgroup creation.  Sibling
  structure preserved bit-exact.
- **Behavior change on the spawn path.**  Sub-cgroup limits are
  applied as the supervisor declared them.  Instrumentation is
  observability-only.
- **CPU weight comparison.**  `cpu_weight` is a share within a
  cgroup hierarchy, not an absolute cap; comparing parent vs sub
  doesn't capture composition semantics.  When Phase 6 ships the
  restructure, cpu_weight composition becomes meaningful; until
  then, omitting it from the log avoids misleading operators.
- **Phase 6 ledger item drafting.**  The Phase 5 plan §7 update
  is a stub pointing back at §4.3.9 item 2; the full Phase 6 plan
  is a separate work item when Phase 6 starts.

## 4. Architectural decisions

### 4.1 Source of truth for parent's limits

`SessionManager._sessions[parent_session_id].server._profile.runtime_limits`.
Same read path the WS pre-init hook + `runner_spawn.spawn_session_runner`
already use (§5.1b precedent).  The handler-bound parent session id
is authoritative — the wire-format `parent_session_id` is a sanity
echo per Audit 5's confused-deputy protection.

### 4.2 What "exceeds" means

For a kernel-enforced cap field, "sub exceeds parent" is well-
defined when both values are set:

```
sub.memory_max_mb is not None
  AND parent.memory_max_mb is not None
  AND sub.memory_max_mb > parent.memory_max_mb
```

Three legitimate "no comparison" cases collapse into one
visibility message ("parent has no kernel limits"):

- Parent has no profile (inline-spec session).
- Parent's profile has no `runtime_limits`.
- Parent's `runtime_limits` field is `None`.

The `cpu_weight` field is omitted (audit §3 out-of-scope above).

### 4.3 Log level: INFO

`logger.info` matches the §5.1 default-application log + §4.3.5's
provision-success log.  Visibility messages aren't errors and
aren't warnings — they're operator-facing signal.  An operator
who wants to silence them filters by message prefix in their log
config.

### 4.4 No event emission

This is a daemon-internal observability message, not a supervisor-
facing event.  The supervisor opted into isolation; the sub's
effective limits were honored.  Surfacing the sibling-layout
quirk to the supervisor at every spawn would be noise — they
didn't ask about layout.  Phase 6's nested layout, if shipped,
*may* warrant a supervisor-visible event when their declared
sub-bound is capped; that decision belongs to the Phase 6 audit.

## 5. Test plan

Regression pins extend the existing
`server/tests/test_spawn_isolated_runner_helper.py` (the §4.3.5
test file with the cgroup-provision scaffolding the §5.1 tests
already extend).  Each test names the property it pins:

1. `test_nesting_visibility_logs_when_sub_exceeds_parent_memory` —
   parent has `runtime_limits.memory_max_mb=2048`, sub is
   provisioned with effective `memory_max_mb=4096` → INFO log
   fires naming both values + the field name.
2. `test_nesting_visibility_logs_when_parent_has_no_limits` —
   parent's profile has no `runtime_limits` → INFO log fires
   with "parent has no kernel runtime_limits" message + sub's
   declared bound.
3. `test_nesting_visibility_silent_when_sub_within_parent` —
   parent has `memory_max_mb=4096`, sub has `memory_max_mb=2048`
   → no nesting-visibility log line (lookup happens, comparison
   passes, function returns silent).

Uses `caplog` (pytest standard) to capture the INFO log line and
assert on its prefix + payload.  No behavior changes are pinned
beyond the log line (no production code path moves).

## 6. Phase 6 carryover

The full nested-cgroup restructure — `subtree_control` enablement,
parent-runner PID migration into `/main/`, dual code paths in
`CgroupsManager.provision_cgroup` / `attach_pid` / `teardown_cgroup`,
spawn-time layout selection — remains in Phase 6.  Phase 6 plan
should pick this up when a multi-tenant deployment files a request
for parent-bounded composition.

The instrumentation §5.2 ships is the visibility-bridge: operators
see in their logs when nesting WOULD have mattered, which becomes
the evidence base for prioritising the Phase 6 work.

---

End of audit.
