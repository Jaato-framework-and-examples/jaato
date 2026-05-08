# Phase 2 Implementation Plan — Confined Runner Skeleton + RPC

**Status:** Plan only. Awaiting review. No code yet.

**Branch:** `claude/confined-runner-phase2-jvtVs`

This plan covers the six Phase 2 deliverables in
`docs/design/per_session_confined_runner_phase2_prompt.md`. Section
references (§3 / §4.1 / §4.6 / §4.7 / §4.8 / §8.x) point into
`docs/design/per_session_confined_runner.md`.

## 1. File-layout proposal for `jaato-server/server/runner/`

```
jaato-server/server/runner/
├── __init__.py
├── __main__.py             # `python -m server.runner` entry point
├── bootstrap.py            # libapparmor self-confine + /proc verify (§4.6 steps 1-3)
├── rpc.py                  # runner-side bidirectional dispatcher (frame loop,
│                           # request_id table, stream/cancel emission).
├── envelope.py             # typed-envelope schema (§4.8 ok/result/error/warnings/telemetry)
│                           # — used by both daemon and runner.
├── tool_executor.py        # runner-local executor; Phase 2: echo + cli only.
├── echo_tool.py            # test executor for §8.3 RPC overhead.
├── cli_runner.py           # cli plugin's _execute_streaming body, runner-side.
└── tests/
    ├── test_bootstrap.py        # libapparmor lookup, env handling, /proc verify, exit-2.
    ├── test_envelope.py         # round-trip + error encoding.
    ├── test_rpc_protocol.py     # framing reuse, multiplex, stream, cancel.
    └── test_cli_runner.py       # cli streaming, cancellation, subprocess inheritance.
```

**Deviations from the §2.2 sketch:**
- `rpc_server.py` → `rpc.py`. Bidirectional dispatcher; "server" misleads —
  Phase 3 has runner→daemon RPCs (permission ASK, memory) so the structure
  must already be symmetric.
- New `envelope.py`: typed envelope (§4.8) defined exactly once, importable
  from both daemon and runner.
- `cli_runner.py` (not `cli.py`): explicitly the runner-side migration of
  `shared/plugins/cli/plugin.py:_execute_streaming`. Phase 3 may rename
  if it absorbs the full plugin.

**Framing module (§2.4 question):** put the shared framing in
`jaato-server/shared/framing.py`. Rationale: `server/runner/` runs inside
the runner process, which must NOT import the rest of `server.*`
(`session_manager`, `apparmor.AppArmorManager`, IPC server are
daemon-only). `shared/` is the right layer.

## 2. Task-by-task breakdown

### 2.1 — Remove daemon-thread per-session confinement

Files touched:
- `server/__main__.py:_register_ipc_apparmor_hook` (lines 656–835): the
  apparmor profile-provisioning logic stays (still loaded via
  `AppArmorManager`); the `server.set_apparmor_confinement(...)` call at
  line 827 is removed. Also remove the
  `loop.set_default_executor(SafeThreadPoolExecutor(...))` block (lines
  378–386) — its sole purpose was per-task apparmor reset.
- `server/websocket.py` and `server/workspace_command.py`: grep for
  other `set_apparmor_confinement` callers (WS hook had its own); same
  treatment.
- `server/apparmor.py` lines 1688–1696: delete the module-load
  registration of `_thread_unconfine_safe` as a SafeThreadPool pre-task
  hook. The helper itself stays for any third-party reuse.
- `shared/safe_pool.py`: keep the file (verified by grep — **four**
  production callers consume `SafeThreadPoolExecutor` independent of
  the apparmor hook):
  1. `shared/ai_tool_runner.py:_auto_background_pool` (this repo).
  2. `server/__main__.py:loop.set_default_executor` (this repo).
  3. `shared/plugins/subagent/plugin.py` (this repo).
  4. `jaato-premium/jaato_premium/reactors/engine.py:52` (premium
     0.1.184, added 2026-05-07 — reactor dispatch pool).

  The class is general-purpose `ThreadPoolExecutor`; the apparmor
  hook used to REGISTER a pre-task callback on it via
  `apparmor.py:1688-1696`. After 2.1 deletes that registration,
  the class becomes a plain `ThreadPoolExecutor` subclass with an
  empty pre-task hook list — fine for the four call sites above,
  none of which depend on the apparmor-recovery semantics.

  **Premium hint:** the comment in
  `jaato-premium/jaato_premium/reactors/engine.py:52-65` references
  the now-removed apparmor pre-task hook. Post-Phase-2 the comment
  becomes misleading and should be updated to "we use
  SafeThreadPoolExecutor for the conventional class lineage; no
  apparmor-recovery semantics after Phase 2." This is a premium-
  side follow-up, not a blocker for Phase 2.

  Phase 6 cleanup may revisit if the pre-task-callback machinery
  itself becomes vestigial.

What stays for Phase 6 cleanup:
- `apparmor.apparmor_confine`, `make_confine_context`,
  `make_tool_hat_context` — all unchanged.
- `ToolExecutor.set_apparmor_context` and `_apparmor_context` —
  unused after 2.1, deleted in Phase 6 per the spec.

One commit. Verify daemon still starts and serves an unconfined IPC
client. The `shared/tests/test_apparmor.py` suite covers the unchanged
primitives; no test edits needed.

### 2.4 — Factor framing into `shared/framing.py`

Done before 2.5 (§"Critical-path constraint" rule 2).

Files touched:
- New `shared/framing.py`: hosts `HEADER_SIZE`, `MAX_MESSAGE_SIZE`,
  async `read_frame(reader)` / `write_frame(writer, payload)` (pure
  code-move from `server/ipc.py:489-528`), AND synchronous
  `read_frame_sync(fileobj)` / `write_frame_sync(fileobj, payload)`
  for the runner side, which uses blocking sockets in a worker thread
  (asyncio is daemon-side only).
- `server/ipc.py`: replace the inline `_read_message` / `_write_message`
  bodies with calls to `shared.framing` — same byte-for-byte framing,
  no behavioral change.

Tests: `shared/tests/test_framing.py` — round-trip identical bytes,
oversize-frame rejection, EOF-mid-header.

### 2.2 — `server/runner/` package

Implementation order within the task:

1. `envelope.py` — pure dataclasses + `to_dict` / `from_dict`. No I/O.
2. `bootstrap.py` — `confine_to_profile(profile_name)` raises
   `RuntimeError` on libapparmor lookup failure or post-write `/proc`
   verify mismatch. `__main__.py` calls it before any plugin import.
   Contains the `os._exit(2)` policy.
3. `rpc.py` — `class RunnerRPC` with `serve(socket_fd)` blocking loop.
   Decodes one frame at a time; `kind: "request"` → registered method;
   `kind: "cancel"` → trip per-call `CancelToken`. Streaming chunks
   from tools flow via a thread-local `on_output` shim that writes
   `kind: "stream"` frames.
4. `tool_executor.py` — minimal `ToolExecutor`: name → callable. Phase
   2: `echo`, `cli_based_tool`. NO permission/reliability/auto-background
   (all Phase 3).
5. `echo_tool.py` — `def execute_echo(args)`. Pure function for §8.3
   RPC-overhead measurement.
6. `cli_runner.py` — paste-and-trim of the existing
   `_execute_streaming` body; receives args + on_output; returns the
   same dict shape as today.
7. `__main__.py` — reads `JAATO_RUNNER_PROFILE`,
   `JAATO_RUNNER_WORKSPACE`, `JAATO_RUNNER_SESSION_ID` from env;
   `bootstrap.confine_to_profile(...)`; instantiates
   `tool_executor.ToolExecutor`; connects fd 3; runs `RunnerRPC.serve`.

### 2.3 — Daemon-side `RunnerSpawner`

Files added/touched:
- New `server/runner_spawner.py`:
  - `RunnerSpawner.spawn(session_id, workspace_path, profile_name, env)
    → SpawnedRunner(pid, socket)`.
  - `socket.socketpair(AF_UNIX, SOCK_STREAM)` → `os.fork()`. Child:
    `os.dup2(child_end.fileno(), 3)`, close other fds, `os.execvpe(
    sys.executable, ["-m", "server.runner"], env_with_profile)`.
  - Parent: returns the parent socket adapted to asyncio
    (`loop.connect_accepted_socket`).
- New `server/runner_rpc.py`:
  - `RunnerRPCClient` wraps the parent socket. `call(method, args,
    on_output, cancel_token) → result`. Maintains `id → Future`;
    background asyncio task reads frames; `kind: "stream"` → on_output;
    `kind: "response"` → resolves the Future. Cancel-token tripping
    writes a `kind: "cancel"` frame.
- `server/session_manager.py:create_session`: AFTER
  `_run_pre_initialize_hooks` returns (apparmor profile now provisioned)
  and BEFORE `server.initialize()`, call `RunnerSpawner.spawn(...)`.
  Stash the resulting `RunnerRPCClient` on the `JaatoServer` (new attr
  `runner_rpc`).
- `server/core.py:JaatoServer`: add `runner_rpc` attribute; plumb into
  the per-session `ToolExecutor` so the cli stub can reach it.

Order: write `RunnerSpawner.spawn` first; manually verify it forks
+ execs `server.runner` and reads `RunnerReadyEvent` over fd 3; THEN
wire into `SessionManager.create_session`.

### 2.5 — cli plugin migration (the validation vehicle)

Files touched:
- `shared/plugins/cli/plugin.py`: turn `_execute` and `_execute_streaming`
  into thin daemon-side stubs. The stub:
  - Reads `runner_rpc` from a session-injected handle (set by
    `JaatoServer` at plugin configure time — same lifecycle slot
    `set_apparmor_context` used to occupy).
  - Calls `runner_rpc.call("tool.execute", {"name": "cli_based_tool",
    "args": args}, on_output=self._get_effective_output_callback(),
    cancel_token=get_current_cancel_token())`.
  - Returns the envelope's `result` (or raises on `error`).
- `server/runner/cli_runner.py`: receives the actual subprocess.Popen +
  thread-pair-reader logic. The on_output callback writes
  `kind: "stream"` frames into `RunnerRPC`.
- `shared/plugins/cli/__init__.py`: unchanged (`PLUGIN_KIND = "tool"`).

**Cancellation flow:**
1. `JaatoSession.request_stop()` → trips per-call `CancelToken`.
2. `RunnerRPCClient` sees the trip; sends `{"kind": "cancel", "id": <id>}`.
3. Runner's `RunnerRPC` trips its per-call token; cli_runner's
   timeout-loop checks the token between `proc.poll()` calls;
   `proc.terminate()`, 2s grace, `proc.kill()`.
4. cli_runner returns envelope `ok=false, error.type="CancelledException"`.
5. Daemon → `FinishReason.CANCELLED` (existing path).

### 2.6 — Integration test (acceptance gate). See §4 below.

### Non-IPC bootstrap path deferral

Phase 2 wires runner spawn for ONE of the four session-bootstrap
entry points. The other three are explicitly deferred to Phase 3:

| Bootstrap path | Phase 2? | Reason |
|---|---|---|
| `SessionManager._create_session_impl` (IPC `session.new`) | YES | Plan §2.3 — the load-bearing IPC client opt-in path. |
| `SessionManager._load_session_impl` (disk-restore) | NO — Phase 3 | No client_id at restore time; restoring a session that was previously confined needs design work for "do we re-spawn a runner for a session loaded from disk?" The honest answer is "yes once the runner-tier list expands beyond cli", which makes it Phase 3 alongside the bulk plugin migration. |
| `SessionManager.run_ephemeral_session` (remote subagent) | NO — Phase 3 | Ephemeral sessions are subagent fan-out; per §4.3 default they share the parent's runner. Phase 2 doesn't have a parent-runner reference to pass into the ephemeral path. |
| `JaatoWSServer` standalone bootstrap | NO — Phase 3 | The WS server has its own pre-init apparmor hook (`websocket.py:_apparmor_pre_init_hook`); Phase 2 only removes the daemon-thread confinement there (§2.1). The WS-side runner spawn lands when WS migrates off in-process tool execution, expected alongside the bulk plugin migration. |

The IPC apparmor pre-init hook (post-rebase: a 4-arg hook installed
via `add_pre_initialize_hook`) returns early when `client_id is None`
(the disk-restore path passes None) — so loaded-from-disk and
ephemeral sessions transparently fall through to the in-process
tool execution path, same behavior as Phase 1 / pre-rebase main.
This keeps Phase 2 from regressing those paths while their full
runner integration is being designed.

For Phase 3 the wiring will move from "IPC apparmor hook" to a
generic SessionManager-level "spawn runner for this session"
helper that all four bootstrap paths call — the §2.3 plumbing
(JaatoServer slots, RunnerSpawner, RunnerRPCClient, registry-
attribute injection) is already in place to support that.

### Post-rebase §2.3 wiring shift

The original §2.3 implementation lived inside the IPC apparmor
SESSION hook (post-init), not in `_create_session_impl` between
the pre-init hooks and `server.initialize()` as the plan said.
Reviewer caught this; post-rebase fix moves the IPC hook from
`add_session_hook` → `add_pre_initialize_hook` so:

- The runner is spawned BEFORE `initialize()` runs.
- `registry.runner_rpc` is populated by the time plugins'
  `set_plugin_registry` hooks fire during configure.
- Phase 3 plugins that need configure-time access to the runner
  RPC handle work without the lazy-execute-time dance the cli
  stub currently uses.

Two new fields on `JaatoServer` plumb the pre-init hand-off:
- `_runner_rpc` / `_spawned_runner` (already added in §2.3).
- `_planned_sandbox_mode` (new): the pre-init hook can't write
  `Session.sandbox_mode` directly because `Session` is built
  AFTER `initialize()` returns. Hook stashes the planned value;
  `_create_session_impl` reads it back when constructing the
  Session record.

`_run_pre_initialize_hooks` gains an optional `client_id` parameter
with backwards-compat for the WS pre-init hook's legacy 3-arg
signature (introspected via `inspect.signature`). Phase 3 converts
the WS hook to 4-arg alongside its runner-spawn migration.

## 3. Test scenarios for 2.5 (cli) — §4.1.1 contract

`server/runner/tests/test_cli_runner.py` — runner-internal unit tests
(no fork; instantiate `RunnerRPC` over `socket.socketpair()` in a worker
thread):

| Scenario | Assertion |
|----------|-----------|
| Single-shot success | `cli_based_tool(command="echo hi")` → envelope `ok=true, result.stdout="hi\n"`. |
| Streaming chunks in order | `for i in 1 2 3; do echo $i; sleep 0.05; done`: assert `on_output` called ≥3 times with `"1"`, `"2"`, `"3"` in order. |
| stderr/stdout interleaving | command emits to both; assert separate `on_output` invocations with distinct `source` AND per-source ordering (within each source, chunks arrive in command-order). **Do NOT assert cross-source ordering** — line-buffered reads on stdout/stderr can interleave at line boundaries; in-line interleaving is filesystem- and OS-dependent. The contract is "source distinguishes the streams; chunks within a stream stay ordered." |
| Mid-stream cancellation | `sleep 30 && echo done`; trip `CancelToken` 100ms after spawn. Envelope `ok=false, error.type="CancelledException"`; subprocess exited within 3s; no `done` chunk. |
| Subprocess SIGTERM on parent shutdown | spawn `sleep 30`, send SIGTERM to the runner. Subprocess receives SIGTERM (verify via `proc.poll()` within 7s — §8.4 budget). Requires runner-side signal handler to propagate. |
| Output truncation | command exceeds `max_output_chars`; envelope `warnings` carries `output-truncated`, `result.truncated=true`. |
| Timeout | `tool_timeout_seconds=1` with `sleep 5`; `result.timed_out=true`, `warnings: timeout-near-cap`. |

## 4. Integration test sketch for 2.6

Location: `jaato-server/tests/integration/test_phase2_multitenant_apparmor.py`
(new directory; existing tests live next to source, but integration tests
need a daemon subprocess + apparmor and benefit from segregation).

**Infrastructure prerequisite:** `apparmor_parser` requires CAP_MAC_ADMIN
and the dmesg audit assertion requires CAP_SYSLOG. Per operator
direction, this regression test runs on a **user-hosted server** with
the necessary capabilities (not in standard CI). The test still ships
with `@pytest.mark.apparmor` + `_apparmor_available()` skipif gate
(reused from `shared/tests/test_apparmor.py`) so it skips cleanly
elsewhere rather than failing confusingly. The Phase 2 done-criterion
"green in CI" in the prompt is read as "green on the user-hosted
runner that exercises the apparmor mark."

**Profile-fixture spec.** `_provision_workspace(ws, profile="cli_test")`
writes a minimum YAML profile that exercises the runner-side `cli`
plugin. Shape:

```yaml
# .jaato/profiles/cli_test.yaml
name: cli_test
provider: anthropic                    # any provider with a deterministic test fixture
model: claude-sonnet-4-6
plugins:
  - signal_completion(preload)         # standard completion shape
  - cli                                # the runner-tier plugin under test
gc:
  type: budget
  threshold_percent: 80.0
plugin_configs:
  permission:
    policy:
      defaultPolicy: allow             # autonomous-only test run
      blacklist:
        tools: []
```

No agent persona file is needed for the integration test — the
session sends a literal trigger message and `cli` execution is what
exercises the runner.

**Pseudo-code:**

```python
@pytest.mark.apparmor
@pytest.mark.skipif(not _apparmor_available(), reason="AppArmor not available")
def test_two_workspaces_one_daemon_no_bounce(tmp_path):
    ws_a, ws_b = tmp_path / "ws_a", tmp_path / "ws_b"
    _provision_workspace(ws_a, profile="cli_test")  # writes .jaato/profiles/cli_test.yaml
    _provision_workspace(ws_b, profile="cli_test")  # YAML, not JSON — modern jaato profile format

    sock = tmp_path / "jaato.sock"
    daemon = _start_daemon(sock)
    try:
        client_a = IPCClient(sock); client_a.connect(workspace=ws_a, apparmor=True)
        sid_a = client_a.session_new(profile="cli_test")
        client_a.send_message(f"create file: echo hi > {ws_a}/sandbox/x")
        _wait_for_idle(client_a)
        assert (ws_a / "sandbox/x").exists()

        # The 7:3 regression case — second workspace, NO daemon bounce.
        client_b = IPCClient(sock); client_b.connect(workspace=ws_b, apparmor=True)
        sid_b = client_b.session_new(profile="cli_test")  # MUST succeed
        client_b.send_message(f"create file: echo hi > {ws_b}/sandbox/y")
        _wait_for_idle(client_b)
        assert (ws_b / "sandbox/y").exists()

        # Cross-workspace deny — kernel-enforced.
        denial = client_b.send_message(f"read file: cat {ws_a}/sandbox/x")
        assert "Permission denied" in denial.last_tool_result["stderr"]
        dmesg = subprocess.check_output(["dmesg", "-T"]).decode()
        assert f'profile="jaato-ws-{sid_b}' in dmesg and "DENIED" in dmesg
    finally:
        _stop_daemon(daemon)
```

Fixtures: `_apparmor_available()` (cap + binary check),
`_start_daemon(sock)` (subprocess `python -m server --ipc-socket
<sock> --daemon`, poll for socket up to 5s), existing `IPCClient`,
`_wait_for_idle(client, timeout=30)` (drains until
`AgentCompletedEvent`).

Sister tests in same dir:
- `test_phase2_runner_crash.py` — kill the runner pid; assert
  `SessionFailedEvent` arrives within 100ms (§8.4).
- `test_phase2_rpc_overhead.py` — echo tool 1000x; assert p50 ≤ 5ms (§8.3).

## 5. Open clarifications

**5.1 Runner stdout/stderr.** §4.1 says fd 3 is the socketpair; fd 1
and fd 2 are unspecified. Redirect to a per-session log file under
`<workspace>/.jaato/logs/runner-<session_id>.log` via `os.dup2`
before `execvpe`. **Per-session, NOT under `~/.jaato/logs/`** (that's
daemon-scoped); the runner is per-session so its log follows the
per-session convention used elsewhere (`JAATO_SESSION_LOG_DIR`). The
AppArmor profile already grants `rwkl` on the workspace subtree so
the redirect lands correctly post-confinement. 10MB cap, rotation
deferred to Phase 5+.

**5.2 RunnerReadyEvent timeout.** §4.6 doesn't specify. Hard-code
10s for Phase 2 (covers cold-start apparmor-cache miss); on timeout
SIGKILL the runner and emit `SessionFailedEvent(reason="runner did
not become ready in 10s")`. Phase 5 measures actual cold-start.

**5.3 Plugin discovery in the runner.** Phase 2: hardcoded allowlist
in `tool_executor.py` init (cli + echo only). The full entry-points
discovery contract becomes a real question in Phase 3 when the
runner-tier list expands; Phase 2 doesn't need it.

**5.4 Daemon-side cli stub: where does `runner_rpc` come from?**
Phase 2 keeps the daemon-side cli plugin instantiated (so the model
sees `cli_based_tool` in tool schemas), with `_execute` rewritten as
the RPC forwarder. **Injection mechanism: registry-attribute pattern**
(matching how plugins discover `BackupManager` from `file_edit` today).
`JaatoServer` sets `registry.runner_rpc = <RunnerRPCClient>` after
`RunnerSpawner.spawn` returns; the cli plugin's
`set_plugin_registry(registry)` hook (already in the plugin protocol)
captures the reference. No new plugin-protocol method to add; no new
config key in `plugin_configs.cli` to plumb. Phase 3 may invert
(runner instantiates the plugin, daemon holds only the schema), at
which point the daemon-side stub disappears entirely.

## 6. Risk register

**6.1 fork-inherits-apparmor (the 2.1↔2.3 ordering trap).** Even with
2.1 done, any `setattr` to `/proc/self/attr/current` in a daemon thread
before `RunnerSpawner.spawn` re-confines the daemon. Mitigation:
spawner asserts `/proc/self/attr/current` starts with `unconfined`
immediately before `os.fork()`. Loud failure on mismatch.

**6.2 fd inheritance via fork+exec.** `os.fork()` inherits all open fds;
`execvpe` honors `O_CLOEXEC`. Python stdlib defaults `O_CLOEXEC=True`
for sockets/files opened via standard APIs, but we MUST verify (and
add a `os.closerange(4, max_fd)` after dup2 in the child as
belt-and-braces) — otherwise the runner inherits sensitive fds (IPC
client sockets, OAuth token files).

**6.3 SIGCHLD / zombie reaping.** Daemon must reap the runner on exit.
Use `asyncio.PidfdChildWatcher` (Linux 5.4+) — fires on runner exit,
emits `SessionFailedEvent` within the §8.4 100ms budget. Without it,
zombies accumulate AND the failed-session event misses budget.

**6.4 libapparmor.so.1 lookup.** §4.6 step 2 uses `ctypes` against
`libapparmor.so.1`. On distros without `libapparmor1`, lookup fails
and runner exits — acceptable per spec, but daemon error must
distinguish "apparmor not installed" from "kernel refused profile
transition" so operators know what to fix.

**6.5 Silent confine-failure via `change_profile -> unconfined` only.**
Today's per-session profile lacks `change_profile -> jaato-ws-*`. If
2.1 is incomplete (any path still confines a daemon thread), the
runner's self-confine silently fails. Mitigation: bootstrap step 3
verifies `/proc/self/attr/current` matches the *expected* profile
name, NOT just "non-unconfined". Catches the silent failure.

**6.6 Streaming back-pressure.** §6.4 of the design surfaced this.
Pathological cli (`yes | head -c 100M`) fills the socket buffer,
sender blocks, cancel frame can't get through. Phase 2 mitigation:
enforce `max_output_chars` IN the runner's streaming reader before
emitting chunks (the truncation already exists in the synchronous
path; mirror it in the streaming path).

**6.7 EOF mid-frame (bidirectional).** Two symmetric cases:
(a) daemon-side reader gets EOF mid-frame because the runner crashed
or got SIGKILL'd mid-write; (b) runner-side reader gets EOF mid-frame
because the daemon went down. Both are benign — the OTHER end is gone,
no protocol-error to report, just close cleanly. Mitigation: BOTH
readers treat partial-frame EOF as a single info-level log + clean
close (NOT protocol-error). Runner-side: triggers normal shutdown
path (clean teardown of any open `interactive_shell` PTYs etc.).
Daemon-side: triggers `SessionFailedEvent` per §4.6 if confidence
that the runner died (vs. graceful shutdown sequence).

**6.8 CI cap — RESOLVED.** §2.6 needs CAP_MAC_ADMIN + CAP_SYSLOG.
Per operator direction, the regression test runs on a user-hosted
server with those capabilities, not standard GitHub-Actions CI. The
test still ships with the `apparmor` mark + skipif gate so it skips
cleanly on capability-less runners.

## 7. Reviewer-flagged clarifications (added during review)

**7.1 AppArmorManager interaction with thread-confinement removal.**
2.1 removes per-thread confinement but keeps profile loading via
`AppArmorManager`. Pre-implementation check: confirm the
`AppArmorManager` state machine doesn't internally track
"thread X is confined to profile P" — if it does, that bookkeeping
needs to go too (or the check needs documentation explaining why
it's still correct without thread-confinement). Likely no-op
(profile loading is a kernel-level operation independent of which
thread will eventually use the profile), but worth a 5-minute
audit before code commits.

**7.2 Estimated test/prod LOC split.** Of the ~1500 lines added,
estimate is roughly:
- ~600 production code (runner package + RunnerSpawner + framing
  module + cli stub edits + session_manager wiring).
- ~900 tests (unit tests in `server/runner/tests/`, framing tests
  in `shared/tests/`, integration tests in
  `tests/integration/`).
This skew is intentional — the §4.1.1 streaming/cancellation
contract has many surfaces worth pinning, and the integration test
covers the §8.2 acceptance gate (the load-bearing 7:3 reproducer).

**7.3 Lazy vs eager runner-side plugin import.** Phase 2's
`tool_executor.py` init imports `cli_runner` eagerly (~100ms cold
cost). For Phase 2's single-tool surface (cli + echo) this is fine
— amortized against the apparmor profile load and fork+exec, the
eager import is invisible. Phase 3 will need to revisit when the
runner-tier plugin set expands to ~10 plugins (~1s eager-import
risk). Defer the lazy-import-via-entry-points machinery until then.

## 8. Post-rebase review fixes (added 2026-05-07)

A second review against rebased main surfaced four items, all
addressed before merge:

**8.1 Rebase against main.** The branch had diverged 12 commits
behind main (server 0.6.59 → 0.6.71, sdk 0.12.0 → 0.13.0).
Rebased cleanly — Phase 2's surfaces don't overlap with
0.6.60-0.6.71's edits (script_loader, signal_completion gating,
secret URIs, ContextVars, sys.modules snapshot, helper-name index,
fresh-context wrapper).

**8.2 §2.3 wiring incompleteness.** The original §2.3 commit added
`runner_spawner.py` / `runner_rpc.py` / the `JaatoServer` slots
but DID NOT edit `session_manager.py` — the spawn lived in the
IPC apparmor SESSION hook (post-init).  Post-rebase fix moves the
hook to PRE-INIT (see "Post-rebase §2.3 wiring shift" above).
Critically: the call site is `_create_session_impl`, not the
public `create_session` wrapper, because 0.6.71 split the two
(public wrapper runs the impl inside a fresh `contextvars.Context`).

**8.3 §2.1 audit list.** Original audit listed three SafeThreadPoolExecutor
callers; jaato-premium 0.1.184 added a fourth (reactor engine).
Plan §2.1 above now lists all four.  The premium-side comment
referencing the now-removed apparmor pre-task hook is flagged
for a follow-up update post-merge.

**8.4 Non-IPC bootstrap path scope.** Phase 2 now explicitly defers
`_load_session_impl`, `run_ephemeral_session`, and the standalone
WS bootstrap to Phase 3 (see "Non-IPC bootstrap path deferral"
above). Loaded-from-disk and ephemeral sessions transparently
fall through to the in-process tool execution path (the IPC hook
returns early when `client_id is None`); WS-provisioned sessions
keep their existing apparmor pre-init hook (legacy 3-arg
signature, runner spawn lands in Phase 3 alongside WS-side
plugin migration).

**8.5 Memory follow-up.** `project_backlog_apparmor_restore_unconfined_eperm.md`
becomes moot once Phase 2 lands — the EPERM error class is
unreachable when no daemon thread is confined. Post-merge update.

---

End of plan. Code commits + post-rebase review fixes landed.
Estimated PR size after all six tasks + rebase fixes: ~1700
lines added, ~210 removed.
