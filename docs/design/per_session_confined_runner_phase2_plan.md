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
- `shared/safe_pool.py`: keep the file. The class is harmless without
  registered hooks; deleting it would touch unrelated callers.

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

## 3. Test scenarios for 2.5 (cli) — §4.1.1 contract

`server/runner/tests/test_cli_runner.py` — runner-internal unit tests
(no fork; instantiate `RunnerRPC` over `socket.socketpair()` in a worker
thread):

| Scenario | Assertion |
|----------|-----------|
| Single-shot success | `cli_based_tool(command="echo hi")` → envelope `ok=true, result.stdout="hi\n"`. |
| Streaming chunks in order | `for i in 1 2 3; do echo $i; sleep 0.05; done`: assert `on_output` called ≥3 times with `"1"`, `"2"`, `"3"` in order. |
| stderr/stdout interleaving | command emits to both; assert separate `on_output` invocations with distinct `source` (mirrors today's `(source, text, mode)` callback). |
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

**Pseudo-code:**

```python
@pytest.mark.apparmor
@pytest.mark.skipif(not _apparmor_available(), reason="AppArmor not available")
def test_two_workspaces_one_daemon_no_bounce(tmp_path):
    ws_a, ws_b = tmp_path / "ws_a", tmp_path / "ws_b"
    _provision_workspace(ws_a, profile="cli_test")  # writes .jaato/profiles/cli_test.json
    _provision_workspace(ws_b, profile="cli_test")

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
and fd 2 are unspecified. Default I'd take: redirect to per-runner log
file under `~/.jaato/logs/runner-<session_id>.log` via `os.dup2`
before `execvpe`. Matches `JAATO_SESSION_LOG_DIR`; gives operators a
`tail -f` target. 10MB cap, rotation deferred.

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
the RPC forwarder. `JaatoServer` injects `runner_rpc` into the plugin
at configure time — same lifecycle slot today's
`set_apparmor_context` occupies. Phase 3 may invert (runner instantiates
the plugin, daemon holds only the schema). Worth confirming.

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

**6.7 SIGTERM mid-frame-write.** Daemon shutdown SIGTERM arrives
mid-frame; daemon-side reader sees a partial frame. Mitigation: treat
EOF-mid-frame as benign (single warning, not protocol-error) on the
daemon side.

**6.8 CI cap — RESOLVED.** §2.6 needs CAP_MAC_ADMIN + CAP_SYSLOG.
Per operator direction, the regression test runs on a user-hosted
server with those capabilities, not standard GitHub-Actions CI. The
test still ships with the `apparmor` mark + skipif gate so it skips
cleanly on capability-less runners.

---

End of plan. Awaiting review before code commits begin. Estimated PR
size after all six tasks: ~1500 lines added, ~200 removed (most
removals are 2.1's daemon-thread confinement deletions).
