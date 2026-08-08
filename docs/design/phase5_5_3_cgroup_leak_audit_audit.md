# Phase 5 §5.3 — Cgroup-leak audit at session shutdown

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.3
(Theme A).
**Ledger item:** `phase4_implementation_audits.md` §4.3.9 item 4.
**Sibling shipped:** §5.1 (default RuntimeLimits) + §5.1b (mainline
passthrough) + §5.2 (Phase 6 deferred + visibility instrumentation).

---

## 1. Problem

`SessionManager._cascade_teardown_isolated_subagents` walks the
in-memory `_isolated_sub_runners` dict and tears down every sub-
runner the parent **knows about**.  When the dict's known-set
matches the kernel's cgroup state perfectly, everything is reaped.

Two paths produce kernel-side state that the dict doesn't track:

1. **Rollback failure.**  `_rollback_isolated_resources` runs when
   sub-runner spawn fails between cgroup provision and handle
   registration.  It calls `cgroups_manager.teardown_cgroup(...)`
   best-effort; if that call itself errors (kernel transient, EBUSY
   while processes are still dying), the rollback continues — log
   WARNING, return the error envelope to caller, and the orphaned
   cgroup dir stays on `/sys/fs/cgroup/jaato/`.
2. **Mid-spawn crash.**  A sub-runner that crashes (or its handle
   registration races with shutdown) can leave a provisioned cgroup
   dir without a corresponding entry in `_isolated_sub_runners`.
   Today there's no detection — the cgroup persists with whatever
   processes are still inside it (potentially zombie sub-runner
   subprocesses the kernel hasn't reaped).

§4.3.9 item 4:

> If a sub-runner crashes before parent shutdown, the sub-cgroup
> may be left with zombie processes.  Phase 5: audit pass at parent
> teardown that finds + reaps orphaned sub-cgroups by name pattern.

§5.3 ships that audit pass.

## 2. Scope

### In-scope

- Add `CgroupsManager.list_orphan_sub_cgroups(parent_session_id,
  known_isolated_session_ids)` — pure-FS scan returning isolated-
  session-ids whose cgroup dirs exist on disk but aren't in the
  caller's known-set.  Name-pattern based:
  `jaato-ws-{parent_session_id}__sub_*`.
- Extend `SessionManager._cascade_teardown_isolated_subagents`:
  after the known-handles loop completes, call the orphan scanner
  with the set of isolated-session-ids the cascade just torn down,
  then call `teardown_cgroup` on each orphan.  Each orphan reap is
  best-effort; failures log WARNING but don't block the cascade
  return.
- WARNING-level log per orphan found (`Reaped orphaned sub-cgroup
  jaato-ws-X__sub_Y at parent teardown — likely cause: rollback or
  mid-spawn crash`).  Visibility into when the leak audit had work
  to do, so operators can correlate with prior spawn failures.

### Out of scope

- **Daemon-startup orphan scan.**  Cleaning up cgroups left by a
  crashed daemon process is a related concern but distinct call
  site.  Ledger entry says "audit pass at parent teardown" — that's
  the §5.3 contract; daemon-startup cleanup is a separate task
  (file as `project_backlog_daemon_startup_cgroup_orphan_reap` if
  picked up).
- **AppArmor sub-profile leak audit.**  The ledger entry is
  cgroup-specific; sub-profile leaks would need a parallel scan
  against `aa-status` output.  Out of scope here.
- **Sub-cgroup leak detection across daemon restarts.**  If the
  daemon crashes mid-cascade, the next-restart parent-teardown
  wouldn't see those orphans because the parent-session-id may
  also be gone.  Detection-via-name-pattern across all parents is
  closer to a daemon-startup scan and lives there.
- **Nested cgroup layout (§5.2 / Phase 6).**  When Phase 6 picks
  up nested cgroups, the orphan-scan's name pattern needs to learn
  the nested layout (`jaato-ws-X/sub_Y/` vs `jaato-ws-X__sub_Y/`).
  The §5.2 reference audit (§4.5) already flags this composition.
  Phase 5 §5.3 only handles the sibling layout that exists today.

## 3. Architectural decisions

### 3.1 Source of truth for "known"

The cascade-teardown caller passes the set of isolated-session-ids
it just processed.  That's authoritative: those are the handles
that existed in `_isolated_sub_runners` at cascade time, so
anything ELSE matching the name pattern is by definition orphaned.

A pre-existing design choice supports this: every isolated session
id follows the `{parent}__sub_{subagent}` template (Audit 7).  The
scan is name-pattern-based on `jaato-ws-{parent}__sub_*` and never
needs to reach across to other parents' sub-runners.

### 3.2 Scanner returns ids, not paths

`list_orphan_sub_cgroups` returns isolated-session-ids (strings),
not `Path` objects.  Caller passes those back through the existing
`teardown_cgroup(session_id)` API — single code path for reap-by-id
whether the source was a known handle or an orphan scan.  Avoids
introducing a parallel `teardown_cgroup_by_path` surface.

### 3.3 Reap path = existing `teardown_cgroup`

`teardown_cgroup` already handles cgroup.kill + rmdir atomically
via the existing implementation (cgroups.py:235).  Orphans get the
same treatment as known sub-runners.  This means:

- Linux ≥ 5.14: `cgroup.kill` atomically terminates the entire
  process tree inside the orphan, even if zombies are sleeping
  there.
- Older kernels: SIGTERM-then-rmdir fallback (best-effort; operator
  may need to clean up manually if processes refuse to die — same
  contract as `teardown_cgroup`'s existing fallback).

### 3.4 WARNING vs INFO log level

Orphan-found is operationally **unusual** — it means something
crashed or rolled back without cleanup.  WARNING level is correct:
it's not an error (the orphan is reaped successfully) but it
shouldn't be silent.  Operators investigating reliability incidents
can grep WARNING for "Reaped orphaned sub-cgroup" to count
how often the leak audit had work.

When the leak audit finds zero orphans, no log line is emitted
(the cascade-teardown INFO log already records the parent's
teardown event).

### 3.5 Best-effort, never blocks cascade return

Each orphan reap wraps in try/except.  A failed `teardown_cgroup`
on an orphan logs WARNING but doesn't propagate — the cascade
teardown return value is the count of **handles** torn down (not
orphans reaped).  Orphan handling is observability + cleanup, not a
correctness contract.

### 3.6 Graceful degradation when cgroups unavailable

`CgroupsManager.is_available()` returning False short-circuits the
scan: no scan, no orphans, no reap.  Matches the existing pattern
in `teardown_cgroup` / `provision_cgroup` / `attach_pid`.

## 4. Test plan

Two test surfaces:

### 4.1 `CgroupsManager.list_orphan_sub_cgroups` (pure FS scan)

Lives in a new module-local fixture or extends
`shared/tests/test_cgroups.py`.  Each test pins one property:

1. `test_list_orphan_returns_empty_when_no_matching_dirs` — empty
   fake cgroup root → empty list.
2. `test_list_orphan_returns_dirs_matching_pattern` — fake root has
   `jaato-ws-A__sub_X` + `jaato-ws-A__sub_Y` + unrelated
   `jaato-ws-B`; known-set = `{"A__sub_X"}` → returns
   `["A__sub_Y"]` (only the matching, non-known one).
3. `test_list_orphan_ignores_other_parents` — fake root has
   `jaato-ws-A__sub_X` + `jaato-ws-OTHER__sub_Y`; scan for parent
   `A` returns only `A__sub_X` matches (ignores OTHER).
4. `test_list_orphan_silent_when_cgroups_unavailable` — manager
   with `is_available()` False → returns empty list without
   scanning.
5. `test_list_orphan_tolerates_non_directory_entries` — fake root
   has a regular file matching the pattern → scanner skips it
   (only directories count).

### 4.2 `SessionManager._cascade_teardown_isolated_subagents` orphan reap

Extends
`server/tests/test_cascade_teardown_isolated_subagents.py` (or
mirror the existing test cadence in
`test_spawn_isolated_runner_helper.py`).  Each test pins one
property:

1. `test_cascade_reaps_orphan_sub_cgroups_after_known_handles` —
   cascade torn down 1 known handle; scanner reports 1 orphan;
   `teardown_cgroup` invoked for both.
2. `test_cascade_returns_known_handle_count_not_orphan_count` —
   contract pin: return value is the count of **handles** torn
   down, not orphans reaped.  Orphans handled silently in terms of
   the return value.
3. `test_cascade_logs_warning_per_orphan` — `caplog` captures a
   WARNING per orphan found.
4. `test_orphan_teardown_failure_does_not_block_cascade_return` —
   `teardown_cgroup` raises on orphan; cascade still returns the
   known-handle count.

## 5. Real-host verification

The orphan reap path uses the existing `teardown_cgroup` machinery,
which has had real-host verification through the §4.3.5 / §4.3.6
sub-track.  §5.3 adds only the scanner.  Recommended verification:

1. Spawn an isolated subagent on a cgroup-v2 host.
2. SIGKILL the sub-runner process directly (simulates a mid-life
   crash).
3. Tear down the parent session.
4. Verify `/sys/fs/cgroup/jaato/jaato-ws-{parent}__sub_{subagent}`
   is gone after teardown.
5. Confirm a WARNING log line records the orphan reap.

These steps belong in the AppArmor setup guide as a §5.3-specific
playbook entry; out of scope for this audit doc.

## 6. Phase 6 carryover

When Phase 6 picks up the §5.2 nested cgroup layout, the orphan
scanner needs to learn the nested name template
(`jaato-ws-X/sub_Y/` directory hierarchy vs today's
`jaato-ws-X__sub_Y/` flat name).  The Phase 6 §5.2 reference audit
(§4.5) flags this composition.

Additionally, a daemon-startup orphan scan is the natural extension
of §5.3: scan all `jaato-ws-*__sub_*` cgroups at boot and reap those
without a known live session.  Out of scope for §5.3 but a clean
follow-on once §5.3 ships.

---

End of audit.
