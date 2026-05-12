# Phase 5 §5.10 — AppArmor child-subprofile escape-vector closure

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.10
(Theme C).
**Memory backlog:** `project_backlog_apparmor_child_subprofile`.
**Inline reference:** `jaato-server/server/apparmor.py:413-449`
(template comment block `!!! KNOWN ESCAPE VECTOR !!!`).
**Status:** Draft.  Awaiting design confirmation before §5.10c/d
(subprocess wire-up) land.

---

## 1. Problem — the verified escape vector

The per-session AppArmor profile `jaato-ws-{session_id}` grants
three rules to support `apparmor_confine.__exit__`'s restore-on-exit
path:

```
change_profile -> unconfined,
/proc/self/attr/current      w,
/proc/self/task/*/attr/current w,
```

Phase 4 documented this as a known escape vector (template comment
at `apparmor.py:413-449`).  Verified empirically against template
v10:

```
sudo aa-exec -p jaato-ws-... -- python3 -c '
    open("/proc/self/attr/current","w").write("changeprofile unconfined")
    os.listdir("/etc/apt/apt.conf.d")  # 19 entries — UNCONFINED'
```

The kernel gates the `/proc/self/attr/current` write by the
**writer's profile capabilities**, not by who-in-userspace called
write().  Two attack surfaces exist:

1. **Subprocesses inherit the profile** via `fork()+exec()`.  A
   `cli`-tool invocation like
   `python3 -c 'open("/proc/self/attr/current","w").write("changeprofile unconfined")'`
   escapes the profile.
2. **In-process tools** (`file_edit`, `readFile`,
   `glob_files`, …) run on the daemon worker thread under the same
   profile.  An LLM-driven path that targets
   `/proc/self/attr/current` from inside Python would also escape.

**Phase 4 closed surface 2 (in-process).**
`sandbox_utils.check_path_with_jaato_containment` denies
`/proc/**/attr/**` explicitly, so in-process tools reject the
attack before AppArmor even sees the write.

**Surface 1 (subprocesses) remains open.**  This is the §5.10
target.

## 2. Why this is the highest-impact remaining Phase 5 item

- **Verified escape, not theoretical.**  The aa-exec command above
  produces a real unconfined process.
- **LLM-controlled attack vector.**  Every cli-tool / shell-tool
  invocation is one Python `open(…).write(…)` from breaking out of
  the profile.  Model-driven path injection becomes a
  confinement-bypass vector.
- **No in-process defense reaches it.**  The Phase 4 denylist sits
  in the in-process tools' path validators; a subprocess executes
  arbitrary code without consulting those validators.
- **Closes a whole class.**  Once the subprocess is in `//child`
  (which lacks the dangerous rules), the kernel rejects the write —
  no need for path-pattern matching, no defense-in-depth gaps.

## 3. Solution shape — `//child` sub-profile + transition-on-spawn

### 3.1 Architectural concept

AppArmor sub-profiles do not inherit base rules (verified
empirically against AppArmor 4.0.1; see `apparmor.py:494`).
Each sub-profile re-declares every allow + deny.

The fix:

1. **Add a `jaato-ws-{session_id}//child` sub-profile** in the
   template that mirrors the body the runner subprocess needs
   (workspace I/O, venv reads, source-root reads, tool_hat-style
   information-isolation read-denies) but **DROPS** the three
   escape rules:

   ```
   change_profile -> unconfined,    ← removed
   /proc/self/attr/current      w,  ← removed
   /proc/self/task/*/attr/current w, ← removed
   ```

   Result: a process in `//child` cannot write to
   `/proc/self/attr/current` (kernel rejects with EACCES per
   `proc_pid_attr_write`'s capability check) and cannot
   `change_profile` anywhere.  The subprocess is locked into
   `//child` for its lifetime.

2. **No explicit `change_profile -> //child` rule needed.**  AppArmor
   implicitly authorizes transitions to inline-declared sub-profiles
   from their parent (same mechanism the existing `//tool_hat`
   transition relies on — the base profile has no
   `change_profile -> jaato-ws-X//tool_hat` rule, only
   `change_profile -> unconfined`, yet `apparmor_confine` enters
   `//tool_hat` successfully).  Declaring `profile child { ... }`
   inline inside the parent suffices.

3. **Wire subprocess-spawning plugins to transition into `//child`
   between fork() and exec()** via a `preexec_fn` that writes
   `changeprofile jaato-ws-{session_id}//child` to
   `/proc/self/attr/current`.  Same mechanism as the existing
   `CgroupsManager.make_attach_callback` — runs in the forked
   child, before exec, so the new program comes up already in
   `//child`.

The parent profile's `/proc/self/attr/current w` rule stays
in place — `apparmor_confine.__exit__` needs it, and only the
framework's parent-tier code (running on daemon worker threads
in the base profile) can use it.  Model-controlled SUBPROCESSES
land in `//child` and the rule isn't there.

### 3.2 Why not just drop the parent profile's rules entirely?

Per the inline comment at `apparmor.py:446-448`:

> Helper-thread "do unconfine from outside the profile" doesn't work
> because proc_pid_attr_write enforces `current != task → -EACCES` —
> only the task itself can write its own attr/current.

The framework's `apparmor_confine.__exit__` runs on the same thread
that entered the profile; that thread MUST write its own
attr/current.  Removing the rule from the base profile traps every
thread that uses `apparmor_confine` (which is how prefetch,
reactor dispatch, all the in-process tool-hat transitions work).

So: keep the rules in base + tool_hat, MOVE subprocesses to a
restricted `//child` that lacks them.

### 3.3 Why not AppArmor `cx -> child` exec transitions?

`cx -> child` declares "on exec of this path, auto-transition to
`//child`".  Requires enumerating every executable path that
subprocesses might exec (`/usr/bin/python3`, `/usr/bin/sh`,
`/usr/bin/git`, `/usr/local/bin/npm`, …).  Brittle: missing an
entry leaves a hole, surplus entries can break legitimate flows.

`preexec_fn` writing `changeprofile` is path-agnostic — the
transition fires for every subprocess the runner spawns,
regardless of what the model asks to exec.

## 4. Scope

### In-scope

- New `//child` sub-profile in
  `apparmor.py:AppArmorManager._build_profile` template (mirrors
  `tool_hat`'s body minus the three escape rules).  Inline
  declaration inside the parent profile is sufficient — no
  explicit `change_profile -> //child` rule needed (same mechanism
  as the existing `//tool_hat` transition).
- New `AppArmorManager.make_child_transition_callback(profile_name)`
  returning a zero-arg `preexec_fn`-style callable that writes
  `changeprofile {profile_name}//child` to
  `/proc/self/attr/current`.  Mirrors the
  `CgroupsManager.make_attach_callback` shape.
- Plugin wire-up: `cli` and `interactive_shell` plugins gain an
  AppArmor child-transition callback alongside their existing
  cgroup-attach callback.  `preexec_fn` is a composition of both
  callables — `apparmor_transition` runs FIRST, then `cgroup_attach`,
  then exec.  (Transition before cgroup attach so the new profile's
  rules apply during the cgroup write; the cgroup write is to
  `/sys/fs/cgroup/...`, which `//child` allows.)
- Template snapshot test gains a `//child` block; unit tests for
  the transition callback; plugin tests for preexec_fn
  composition.
- Operator playbook entry in the AppArmor setup guide.

### Out of scope

- **Sub-runner isolated-subagent profiles.**  The §4.3.4
  sub-profile (`jaato-ws-{parent}//{subagent}`) is a separate
  surface; its `//child` analog can land as a Phase 5+ follow-up
  once §5.10 stabilizes.  Today's sub-profile already drops
  `add_reference_fragment` and applies tool_hat read-denies; it
  has the same parent-tier `/proc/self/attr/current w` rule that
  needs the same `//child` treatment.  Tracked as part of §5.10
  but staged as the last sub-commit.
- **In-process denylist removal.**  The Phase 4
  `check_path_with_jaato_containment` denylist stays in place —
  defense in depth.  Removing it would couple §5.10's correctness
  to the kernel-only enforcement, removing a useful belt-and-
  braces layer.  Phase 6+ once §5.10 has soaked.
- **`apparmor_confine` API change.**  The base-profile
  transitions used by prefetch / reactor dispatch / in-process
  tool execution stay unchanged.  Only NEW transitions
  (subprocess preexec_fn) are added.

## 5. Decomposition into sub-commits

Following the §4.3.5/§4.3.6 cadence — each sub-commit ships an
isolated slice with its own verification gate.

- **§5.10a — `//child` sub-profile in template.**  Add the
  template body + `change_profile -> jaato-ws-S//child` rules to
  base/tool_hat.  Template snapshot test pins the new structure.
  **No real-host verification gate required** — the sub-profile
  exists on disk after `provision_profile` but nothing transitions
  into it yet, so behavior is unchanged.  ~150 LoC template + tests.
- **§5.10b — `make_child_transition_callback` helper.**  Add the
  callback factory to `AppArmorManager`.  Unit tests with a fake
  `/proc/self/attr/current` path verify the write-string semantics.
  Still no integration with subprocess spawn — the callback exists
  but isn't called.  **No real-host verification gate required.**
  ~30 LoC + tests.
- **§5.10c — cli plugin wire-up.**  Plumb the callback through
  `set_apparmor_context` (or a sibling method) on `ToolExecutor`;
  compose with `_cgroup_attach` in `preexec_fn`.  Unit tests pin
  composition order.  **REAL-HOST VERIFICATION GATE.**  Operator
  must verify the escape exploit at `apparmor.py:418-420` is
  CLOSED on a cgroup-v2 + AppArmor-enabled host after this
  commit lands.  ~50 LoC + tests + playbook entry.
- **§5.10d — interactive_shell plugin wire-up.**  Mirror §5.10c
  for the PTY-based plugin's `pexpect.spawn`.  **REAL-HOST
  VERIFICATION GATE.**  Same exploit check, this time targeting a
  shell-spawned process.  ~50 LoC + tests.
- **§5.10e — sub-runner sub-profile `//child` analog.**  Apply
  the same pattern to the §4.3.4 isolated-subagent sub-profile.
  **REAL-HOST VERIFICATION GATE.**  ~80 LoC + tests.

Sub-commits §5.10a + §5.10b are ship-able without real-host
verification.  §5.10c/d/e need an operator-driven real-host
signoff before merge.

## 6. Architectural decisions

### 6.1 Transition order in preexec_fn

When both cgroup-attach and apparmor-transition are wired, the
forked child runs:

```
fork()
├── preexec_fn():
│     apparmor_transition_callback()  # write changeprofile to /proc/self/attr/current
│     cgroup_attach_callback()        # write pid to cgroup.procs
└── exec(program)                     # runs in //child + per-session cgroup
```

Order: **AppArmor first, cgroup second.**  The cgroup write
targets `/sys/fs/cgroup/...`, which both base and `//child`
profiles allow.  Once in `//child`, the cgroup write still
succeeds.

Reverse order (cgroup first) also works today since both writes
land in profiles that allow them, but ApparMor-first is
defensive: any future tightening of cgroup write paths in `//child`
wouldn't break because the cgroup write happens before the
transition only in the wrong-ordered version.

### 6.2 Failure modes in preexec_fn

`preexec_fn` is signal-handler-like territory (between fork and
exec).  Any exception in `preexec_fn` propagates to the parent as a
`SubprocessError`-like failure — the subprocess never starts.  This
is the correct behaviour: a failed AppArmor transition means the
child would inherit the dangerous parent profile, which is the
exact escape we're trying to prevent.  Fail-closed.

The cgroup-attach callable already follows this contract
(`apparmor.py:21-22` documents the parallel argument for
preexec_fn).

### 6.3 `//child` body — what it allows

The `//child` body is `tool_hat`'s body minus the escape rules.
That means:

- Workspace `rw` (same as tool_hat).
- Integrity write-denies on `.jaato/` subpaths (same).
- tool_hat-style read-denies on `.jaato/agents/**`, `.jaato/profiles/**`,
  etc. (same).
- venv + source-root reads (same).
- `@{HOME}/.jaato/...` reads (same).
- Denied capabilities (same).
- DROP: `change_profile -> unconfined`,
  `/proc/self/attr/current w`,
  `/proc/self/task/*/attr/current w`.

A subprocess in `//child` has the same useful capability surface as
in tool_hat (it can read its inputs, write outputs to the
workspace, run tools) but cannot escape the confinement.

### 6.4 Sub-runner sub-profile case (§5.10e preview)

The §4.3.4 isolated-subagent sub-profile
(`jaato-ws-{parent}//{subagent}`) has the same escape vector — it
also grants `change_profile -> unconfined` + `/proc/self/attr/current w`
to support its own `apparmor_confine.__exit__` path.  The fix is
parallel: add `jaato-ws-{parent}//{subagent}//child` and wire the
sub-runner's plugin spawn paths to transition.

The naming gets verbose (`//{subagent}//child` is a three-level
nest).  AppArmor supports this — sub-sub-profiles work the same
way.  But the template generation needs care to keep the rendered
profile valid.  Staged last (§5.10e) so §5.10c/d's pattern is
established first.

## 7. Test plan

### 7.1 Template snapshot (§5.10a)

`shared/tests/test_apparmor.py` extends with assertions on the
rendered profile:

1. `test_child_subprofile_block_present` — the rendered profile
   contains a `profile child {` block.
2. `test_child_subprofile_drops_escape_rules` — the `child` block
   does NOT contain `change_profile -> unconfined` or
   `/proc/self/attr/current` rules.
3. `test_base_profile_authorizes_child_transition` — base profile
   contains `change_profile -> jaato-ws-{session_id}//child`.
4. `test_tool_hat_authorizes_child_transition` — same for the
   tool_hat sub-profile.

### 7.2 Transition callback (§5.10b)

`shared/tests/test_apparmor_child_transition.py` (new):

1. `test_callback_writes_changeprofile_string` — fake
   `/proc/self/attr/current` path; callback writes
   `changeprofile jaato-ws-X//child`.
2. `test_callback_uses_session_profile_name` — the rendered
   string includes the caller-supplied profile name verbatim.
3. `test_callback_no_op_when_apparmor_unavailable` — manager
   reports apparmor unavailable; callback returns a no-op.
4. `test_callback_raises_on_write_failure` — fake path that's
   not writable; callback raises (fail-closed semantics).

### 7.3 Plugin preexec_fn composition (§5.10c, §5.10d)

`server/runner/tests/test_cli_runner_apparmor_child.py` (new) +
similar for `interactive_shell`:

1. `test_preexec_invokes_apparmor_then_cgroup` — record call order
   of mocked callables.
2. `test_preexec_apparmor_failure_blocks_spawn` — apparmor mock
   raises; Popen wrapper surfaces the failure (subprocess never
   starts).
3. `test_preexec_no_op_when_apparmor_unwired` — manager not
   passed in; preexec_fn falls back to cgroup-only (today's
   behavior).

### 7.4 Real-host verification (§5.10c, §5.10d, §5.10e)

Operator playbook:

1. Provision a session on a cgroup-v2 + AppArmor-enabled host.
2. From the supervisor, invoke a cli tool that runs:
   ```python
   python3 -c '
       open("/proc/self/attr/current","w").write("changeprofile unconfined")
       os.listdir("/etc/apt/apt.conf.d")
   '
   ```
3. **Before §5.10c**: the listdir succeeds (escape worked).
4. **After §5.10c**: the `open(...).write(...)` raises EACCES (kernel
   rejected the write because `//child` lacks the rule).
5. Repeat with an `interactive_shell` PTY session after §5.10d.
6. Repeat from an isolated subagent after §5.10e.

These steps belong in the AppArmor setup guide as a §5.10-specific
playbook entry.

## 8. Phase 6 carryover

- **Path-validation denylist removal** (`check_path_with_jaato_containment`
  rule for `/proc/**/attr/**`).  Once §5.10 is fully verified and
  has had soak time, the in-process denylist is technically
  redundant.  Removing it tightens the surface; keeping it adds
  defense-in-depth.  Decision belongs to the Phase 6 worker.
- **AppArmor `cx -> child` exec rules.**  Alternative transition
  mechanism considered + rejected in §3.3.  Could be revisited if
  the preexec_fn approach proves fragile.

---

End of audit (draft).  Sub-commits §5.10a + §5.10b can land
without real-host verification; §5.10c onwards needs operator
signoff per Phase 5 audit-discipline #3.
