# Implement Phase 2: Runner skeleton + RPC protocol

## Goal

Land the runner-subprocess machinery + RPC protocol described in
`docs/design/per_session_confined_runner.md` (Phase 1 deliverable, on
main as of commit 852339a2).  Phase 2 ships **the bone, not the
flesh**: enough to spawn a runner, prove the cross-workspace
isolation works end-to-end against the 7:3 regression case, and
validate the §4.1.1 streaming/cancellation contract.  Plugin
migration in bulk is Phase 3 — DO NOT do it here.

## Read first

`docs/design/per_session_confined_runner.md` is the spec.  Read it
end-to-end before coding.  The sections that constrain Phase 2
specifically:

- **§3 architectural target** — the diagram + the runner-per-top-level
  session framing.  One runner hosts one parent JaatoSession + N
  subagent JaatoSessions by default.
- **§4.1 + §4.1.1** — RPC protocol (Unix socketpair, length-prefixed
  JSON frames, multiplexed) + streaming/cancellation contract.  These
  are the stable wire format Phase 3 will build on.
- **§4.6 runner lifecycle** — spawn / bootstrap / death paths.  Read
  the "Daemon apparmor-state constraint" callout especially — it
  explains why task 2.1 below must land before 2.3.
- **§4.7 apparmor profile generation** — the existing per-session
  profile template stays.  Only the load-site relocates.
- **§4.8 tool result schema** — the typed envelope.  All three RPCs
  Phase 2 implements (echo, cli.execute, generic tool.execute) MUST
  use it.
- **§7 Phase 2 critical-path tasks** — the THREE load-bearing
  changes that must land together.  Order matters.  Don't skip 2.1.
- **§8 success criteria** — Phase 2's acceptance gate is a
  *subset* of §8 (see "Phase 2 acceptance gate" below).

## Phase 2 deliverables

### 2.1 Daemon-side: remove daemon-thread per-session confinement

[**Critical-path #1.**  Land this BEFORE 2.3.  Otherwise the runner
inherits a confined profile and the self-confine in bootstrap step 3
fails — the multitenancy promise breaks silently.]

- Delete `apparmor_confine` enter/exit calls in
  `server/__main__.py:_register_ipc_apparmor_hook` (the per-session
  thread-confinement pattern).
- Delete the `SafeThreadPoolExecutor` pre-task apparmor hook
  (`shared/safe_pool.py` or wherever it lives).
- The daemon process becomes unconfined for Phase 2.  Phase 6 may
  later add a daemon-side narrow profile per §4.7; out of scope for
  Phase 2.
- Existing `apparmor_confine` machinery in `server/apparmor.py`
  STAYS in the file for now (other code may reference it during the
  transition).  Phase 6 cleanup deletes it.  DO NOT delete it as
  part of Phase 2 — it's a Phase 6 task.

### 2.2 New package: `jaato-server/server/runner/`

A new top-level package under `server/`.  Layout:

```
jaato-server/server/runner/
├── __init__.py
├── __main__.py             # entry point (python -m server.runner)
├── bootstrap.py            # AppArmor self-confinement (§4.6 steps 1-3)
├── rpc_server.py           # runner-side RPC dispatcher
├── tool_executor.py        # runner-local ToolExecutor (echoes the daemon-side shape)
├── echo_tool.py            # single test tool (no plugin discovery)
├── cli_runner.py           # the ONE plugin migrated in Phase 2 (§2.5 below)
└── tests/
    ├── test_bootstrap.py        # AppArmor self-confine, env handling
    ├── test_rpc_protocol.py     # framing, multiplex, streaming
    └── test_cli_runner.py       # cli streaming, cancellation, subprocess inheritance
```

`__main__.py` does the bootstrap-step-1-2-3 dance + starts the RPC
server on stdin/stdout (no — on socketpair fd 3 per §4.1; daemon
sets up the socketpair + passes fd via inherit on fork+exec).
Refuses to proceed if `aa_change_profile` doesn't take effect
(`os._exit(2)` per §4.6 bootstrap step 3 — explicit no-fallback-to-
unconfined contract).

### 2.3 Daemon-side: `RunnerSpawner`

[**Critical-path #2.**  After 2.1.]

- New module `jaato-server/server/runner_spawner.py`.
- `RunnerSpawner.spawn(session_id, workspace_path, profile_name, env)`:
  - Creates `socketpair(AF_UNIX, SOCK_STREAM)`.
  - Forks.
  - Child: dup socketpair[1] to fd 3, exec `python -m server.runner`.
  - Parent: returns `(child_pid, socket)` pair.
- Wire into `SessionManager.create_session` AFTER the AppArmor
  profile load (`_run_pre_initialize_hooks` already loads the
  profile; spawn happens after the load returns).
- Daemon-side RPC client (`server/runner_rpc.py`): wraps the socket,
  exposes `call(method, args) → response`, manages request_ids,
  receives stream frames + forwards to `on_output`, handles cancel
  frame.

### 2.4 RPC framing: shared module

- Factor `_read_frame` / `_write_frame` out of `server/ipc.py` into
  `server/framing.py` (or `shared/framing.py` — pick one; document
  the choice in the implementation plan).
- Both `server/ipc.py` (daemon→client) and the runner-RPC code
  (daemon↔runner) import from there.
- The factoring should be a pure code move — same byte-for-byte
  framing, same error-handling shape, no behavioral changes.

### 2.5 ONE plugin migrated: `cli`

[The validation-vehicle plugin.  cli exercises subprocess +
streaming + cancellation, which is the §4.1.1 contract end-to-end.
Pick cli for Phase 2 because:

- It's the simplest of the runner-tier plugins (one subprocess
  spawn per call, no PTY, no long-lived state).
- Its result enrichment (line buffering for `on_output`) is the
  canonical streaming case.
- Its cancellation path (CancelToken → kill subprocess) is the
  canonical cancellation case.

DO NOT migrate any other plugin in Phase 2.  file_edit /
interactive_shell / todo / lsp / template / service_connector / mcp
/ webhook are all Phase 3 work.]

- Move cli plugin's `_execute_impl` into the runner.
- Daemon-side cli plugin becomes a thin RPC stub (registers tool
  schema with the model, forwards execute calls to the runner).
- Streaming chunks flow runner→daemon via `kind: "stream"` frames
  (§4.1.1).
- Cancellation flows daemon→runner via `kind: "cancel"` frames.
- Tests cover all three paths.

### 2.6 Integration test (Phase 2 acceptance gate)

`tests/integration/test_phase2_multitenant_apparmor.py`:

1. Start daemon (subprocess, IPC socket at /tmp/jaato-test.sock).
2. Connect client A from workspace A (handoff_test fixture).
3. `session.new` with profile that has cli plugin loaded.
4. Run `cli` tool that touches workspace-A files (echo > sandbox/file).
5. Concurrently: connect client B from workspace B (kb-enablement-2.0
   fixture).
6. `session.new` from B — **MUST succeed** without daemon bounce
   (the 7:3 regression case).
7. Run cli from B touching workspace-B files.
8. Verify cli from B CANNOT touch workspace-A files (apparmor
   denies; check AppArmor audit log via `dmesg | grep apparmor`
   matching the workspace-B profile name).
9. Both sessions complete cleanly.

This test IS the §8.2 7:3 regression test.  Phase 2 is not done
until it's green.

## Phase 2 acceptance gate (subset of §8)

For Phase 2 to be "done":

| Criterion | What it means for Phase 2 |
|---|---|
| §8.2 cross-workspace isolation | GREEN — including the 7:3 regression test (§2.6 above) |
| §8.4 operational soundness | GREEN — runner crash → daemon emits SessionFailedEvent within 100ms; daemon shutdown SIGTERM ladder works for cli + an open subprocess |
| §8.3 performance budget | RPC overhead ≤ 5ms p50 measured against the echo tool + cli echo-line case.  Full-suite tail latency is Phase 5. |

Phase 2 does NOT need to satisfy:
- §8.1 (full functional cascade) — that's Phase 3 + Phase 5
- §8.5 (full backwards compat) — that's Phase 3 (plugin migration)

## Out of scope for Phase 2

- **DO NOT migrate plugins beyond cli.**  file_edit, interactive_shell,
  todo, lsp, template, service_connector, mcp, webhook, references,
  permission, memory — all Phase 3.
- **DO NOT touch the reactor framework.**  It's daemon-tier and
  unchanged.  ctx.create_session continues to work because
  RunnerSpawner is wired into SessionManager.create_session
  transparently.
- **DO NOT touch model_provider plugins.**  They stay daemon-tier.
- **DO NOT add the daemon-side narrow profile (§4.7).**  Phase 6.
- **DO NOT add cross-platform compat (Windows / macOS in-process
  runner).**  Phase 6.
- **DO NOT optimize runner-spawn latency.**  Cold-start work is
  Phase 5.  The 200ms p99 is a Phase 5 budget; Phase 2 just needs
  spawn-works-correctly.
- **DO NOT migrate JaatoSession out of the daemon yet.**  Phase 2
  keeps JaatoSession daemon-side; the runner only hosts a thin
  ToolExecutor + cli plugin.  The full session-state move is
  Phase 3.

## What NOT to do

- **DO NOT add a "legacy mode" flag.**  §5 explicitly says drop
  in-process mode.  `JAATO_RUNNER_DISABLE` exists ONLY for the
  developer pdb-attach loop and ships with a clear "this is not
  a supported deployment" docstring.
- **DO NOT add fallbacks for "AppArmor not available".**  If
  apparmor isn't loadable the runner refuses to start; daemon
  emits a clear error.  No silent degradation to unconfined.  On
  macOS / Windows, Phase 6 adds the in-process compat runner;
  for Phase 2 the runner is Linux-only.
- **DO NOT add try/except around `aa_change_profile`** beyond what
  surfaces the kernel error verbatim.  If apparmor refuses, fail
  loud with the exact errno.
- **DO NOT add `// TODO` comments**, debug `print()` calls, or
  intermediate scratch files.  Work from the design doc + this
  prompt.
- **DO NOT add new plugin entry-point groups.**  Same plugin
  loader; the runner just runs a SUBSET (cli only in Phase 2).
- **DO NOT add the Straddle tier back to §4.2.**  It has zero
  current entries.  If you find yourself wanting to classify
  something as Straddle, the framing in §4.2's "No straddle tier"
  paragraph applies — generic daemon RPC primitives are not
  plugin-specific straddles.
- **DO NOT add backwards-compat for the old apparmor_confine
  thread pattern.**  The two patterns can't coexist (per the
  daemon-apparmor-state constraint in §4.6); deletion of the
  daemon-thread confinement IS the Phase 2 task.

## Phase 2 critical-path constraint (re-emphasized)

Phase 2 deliverables MUST land in this order:

1. **2.1 BEFORE 2.3.**  Remove daemon-thread confinement before
   adding the RunnerSpawner.  Otherwise fork-inherits-confined-profile
   → runner self-confine fails → multitenancy silently broken.
2. **2.4 BEFORE 2.5.**  Factor framing module before migrating cli.
   cli's RPC stub depends on the shared framing.
3. **2.6 IS THE GATE.**  No "this passes locally" → "ship Phase 2".
   The integration test must pass in CI before Phase 2 is marked
   done.

## Code anchors (copied from §7)

- `jaato-server/server/__main__.py:_register_ipc_apparmor_hook`
  (lines 656–835) — task 2.1
- `jaato-server/server/session_manager.py:create_session`
  (lines 948–1382) — task 2.3 wiring
- `jaato-server/shared/ai_tool_runner.py:execute / _execute_impl`
  (lines 756–950) — daemon-side stub becomes RPC; the actual
  `_execute_impl` body moves to `runner/tool_executor.py`
- `jaato-server/server/ipc.py` — task 2.4 (factoring `_read_frame`
  / `_write_frame`)
- `jaato-server/shared/jaato_runtime.py:create_session` and
  `jaato-server/shared/jaato_session.py` — TOUCH MINIMALLY in
  Phase 2; full session-state move is Phase 3
- `jaato-server/shared/plugins/cli/plugin.py` — task 2.5 migration
- `jaato-server/server/apparmor.py:PROFILE_TEMPLATE` — UNCHANGED;
  only the load site moves earlier (before runner spawn)

## What I want first

**A 200-300 line implementation plan**, NOT code yet.  Specifically:

1. **File layout proposal** for `jaato-server/server/runner/` —
   confirm or adjust the §2.2 sketch.  Justify any deviations.
2. **Task-by-task breakdown** — for each of 2.1 / 2.2 / 2.3 / 2.4 /
   2.5 / 2.6, list every file you'll touch + every function you'll
   add or modify.  Identify the order within each task.
3. **Test scenarios for 2.5 (cli)** — list the test cases that
   validate the §4.1.1 streaming + cancellation contract.  At
   minimum: streaming chunks arrive in order, cancellation mid-stream
   works, subprocess SIGTERM on parent shutdown works.
4. **Integration test sketch for 2.6** — pseudo-code structure of
   `test_phase2_multitenant_apparmor.py`.  List the fixtures, the
   ordering, the assertions.  Identify any infrastructure
   prerequisites (does the test need root for apparmor_parser?
   if so how do we run it in CI?).
5. **Open clarifications** — anything in the design that's unclear
   or that you'd want clarified before writing code.  Specifically:
   - Where does the runner's stdout/stderr go?  (The design says
     fd 3 is the socketpair; what about fd 1, fd 2?)
   - What's the daemon's expected behavior if the runner takes
     >5s to send `RunnerReadyEvent`?  (Spawn timeout policy.)
   - How does cli plugin discovery work in the runner?  (Does the
     runner discover ALL plugins via entry points and only
     instantiate the ones in §4.2's runner-tier list?)
6. **Risk register** — list anything in Phase 2 that could fail
   silently or surprise the implementer.  Examples: the apparmor
   inherit-vs-reset behavior of `exec`, the SIGCHLD handling on
   runner crash, fd inheritance via fork+exec.

Don't write code yet.  Land the implementation plan, surface for
review, then start.

## Branch + commit hygiene

- Work on branch `claude/implement-confined-runner-phase2-<random>`.
- Commit early, commit often.  Each task (2.1, 2.2, 2.3, 2.4, 2.5,
  2.6) gets at least one commit.  Larger tasks split into
  reviewable chunks.
- Commit messages reference §X of the design doc.
- After the implementation plan lands and is reviewed, code commits
  follow.  Don't squash before review.

## Done criteria

Phase 2 is done when:

- All §8.2 + §8.3 + §8.4 (Phase-2 subset) acceptance criteria pass.
- The 7:3 regression test (§2.6 / §8.2) is green in CI.
- Existing test suites still pass (jaato-server pytest, jaato-sdk
  pytest, jaato-tui smoke).
- The implementation plan + every commit is reviewed and approved.

When all conditions hold, mark Phase 2 done and surface for Phase 3
scoping.  Phase 3 (bulk plugin migration) is a separate prompt.
