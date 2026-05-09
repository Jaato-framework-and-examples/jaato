# Phase 3 §3.3c — Session RPC Dispatch Surface

Status: **dispatch surface complete; daemon-shell rewrite (the actual seat-flip) is the remaining work.**

This doc inventories the daemon→runner RPC handlers built up across
the §3.3c precursor commits, plus the daemon-side wrapper API +
migration call sites the seat-flip will progressively migrate.

## 1. Why the surface exists

The §3.3c seat-flip removes the daemon-side `JaatoSession` instance
from `JaatoServer` (`self._jaato`) and dispatches every session-
level operation to the runner-side `JaatoSession` via runner-RPC
instead.  That migration touches ~95 daemon-side `self._jaato.X`
call sites in `core.py` alone.  The §3.3c plan calls this **one
commit** but it's realistically multi-PR work.

The precursor surface lets each daemon-side migration be a local
change against a frozen RPC contract — the wire shape, error
envelopes, and edge cases are pinned by ~85+ tests before the
first migration touches `self._jaato`.

## 2. Runner-side handlers

All handlers live in `jaato-server/server/runner/rpc.py`,
dispatched from `RunnerRPC._dispatch_method`.  Each follows the
same shape:

- Args decoded from `env.args` (a JSON-friendly dict).
- Common precondition via `_require_ready_session()` returning
  clean `stage="no_host"` / `stage="no_session"` errors when the
  runner isn't bootstrapped or the session is None.
- Method-specific arg validation surfacing as `stage="decode"`
  errors.
- Underlying call wrapped in try/except returning
  `stage="<read|set|write>"` errors on failure (defensive: the
  probe must never crash the runner process).
- Success returns `(True, <result_dict>)`.

| Method | Direction | Body | Wrapper |
|---|---|---|---|
| `session.health_check` | read | `{has_host, ready, session_id, tool_count}` | `session_health_check()` |
| `session.get_session_state` | read | `{value: Any}` | `session_get_state(key, default)` |
| `session.set_session_state` | write | `{ok: True}` | `session_set_state(key, value)` |
| `session.get_all_session_state` | read | `{state: dict}` | `session_get_all_state()` |
| `session.is_running` | read | `{running: bool}` | `session_is_running()` |
| `session.request_stop` | write | `{cancelled: bool}` | `session_request_stop(reason)` |
| `session.get_history` | read | `{history: [dict, ...]}` | `session_get_history(raw=False)` |
| `session.get_context_usage` | read | `{usage: dict}` | `session_get_context_usage()` |
| `session.get_turn_accounting` | read | `{turns: [dict, ...]}` | `session_get_turn_accounting()` |
| `session.set_terminal_width` | write | `{ok: True}` | `session_set_terminal_width(width)` |
| `session.set_streaming_enabled` | write | `{ok: True}` | `session_set_streaming_enabled(enabled)` |
| `session.set_presentation_context` | write | `{ok: True}` | `session_set_presentation_context(ctx)` |
| `session.reset` | write | `{ok: True}` | `session_reset()` |
| `session.shutdown` | write | `{shutdown_session_id: str}` | `session_shutdown()` |

Each wrapper has a `_threadsafe` variant for synchronous worker-
thread callers (mirroring the `bootstrap_session_threadsafe`
template).

## 3. Daemon-side wrapper contract

All wrappers live in
`jaato-server/server/runner_rpc_client.py:RunnerRPCClient`.

- Async + threadsafe variants per method.
- Uses `_call_named(method, args, timeout)` internal helper
  (mirrors the `bootstrap_session` template).
- Raises `RunnerCallError` on transport failure or
  protocol-level error (`response.ok=False`).
- Returns the unwrapped value (bool / dict / list / str / None)
  rather than the raw result envelope — daemon-side callers get
  a typed Python API.

## 4. Defensive contracts (cross-handler)

| Concern | Behavior |
|---|---|
| Runner not bootstrapped (no host) | `stage="no_host"` error envelope — daemon distinguishes from "session present" |
| Host bootstrapped but session=None (test-stub mode / configure failed) | `stage="no_session"` error |
| Underlying setter / reader raises | `stage="set"` / `stage="read"` error envelope; runner process never crashes |
| Non-dict / non-list / non-bool from custom session subclass | `stage="read"` error rather than letting JSON encoder choke |
| Idempotent re-call (e.g. shutdown twice) | success no-op, not error |
| Per-message serialization failure in `get_history` | placeholder substituted; count preserved; the buggy message doesn't drop the whole history |

## 5. Test coverage

~85+ tests across:

- **Per-handler unit tests** — `test_session_<area>_rpc.py` files;
  exercise direct `_handle_session_X` calls + dispatch routing.
- **Daemon-side wrapper e2e tests** —
  `test_session_method_wrappers_e2e.py`; drive each wrapper over
  a real `socketpair()` connecting `RunnerRPC` to
  `RunnerRPCClient`.
- **Lifecycle composition test** —
  `test_session_dispatch_lifecycle_e2e.py`; 13-step session arc
  proves handlers compose correctly across a realistic
  bootstrap → state ops → config → shutdown sequence.

## 6. Daemon-side migrations done

| Daemon site | Old behavior | New behavior |
|---|---|---|
| `JaatoServer.shutdown` | closes RPC transport directly (SIGTERM races plugin teardown) | calls `runner_rpc.session_shutdown_threadsafe(timeout=5)` first; then transport close.  Best-effort: failures log + proceed |
| `JaatoServer.terminal_width` setter | propagates only to in-process `_jaato` + formatter pipelines | also forwards via `runner_rpc.session_set_terminal_width_threadsafe(width, timeout=2)` |
| `JaatoServer.set_presentation_context` | propagates only to in-process `_jaato` | also forwards via `runner_rpc.session_set_presentation_context_threadsafe(ctx, timeout=2)` |

All three follow the same pattern:

1. Update daemon-side state (unchanged).
2. If runner attached, forward via the matching wrapper with a
   short timeout.
3. Wrap the forward in try/except logging at DEBUG — failures
   never block the daemon-side flow.
4. Use `getattr` + `callable` to skip gracefully when the
   wrapper method is absent (forward-compat with rolling
   upgrades).

## 7. Remaining work (the seat-flip itself)

The dispatch surface is comprehensive enough that the seat-flip
can proceed handler-by-handler:

### 7a. Always-spawn the runner

Currently the runner spawns only when the IPC client opts into
apparmor (`_provision_ipc_apparmor_and_spawn_runner`).  For the
seat-flip to work, every IPC + WS-standalone session needs a
runner so daemon-side `_jaato.X` calls have something to dispatch
to.  Decoupling runner spawn from apparmor opt-in is a single
focused change.

### 7b. Migrate remaining `self._jaato.X` call sites

`core.py` has ~95 `self._jaato.X` references.  Most fall into
patterns the dispatch surface already covers:

- `_jaato.set_*` → `runner_rpc.session_set_*_threadsafe`
- `_jaato.get_session().get_session_state(...)` → `runner_rpc.session_get_state`
- `_jaato.is_processing` → `runner_rpc.session_is_running`
- `_jaato.stop()` → `runner_rpc.session_request_stop`

The HARD ones aren't in the surface yet:

- `_jaato.send_message(prompt, ...)` — streams, multi-turn,
  permission interaction, plugin enrichment.  The biggest
  single handler.  Existing `tool.execute` already shows the
  streaming pattern (`_make_on_output` callback through the
  stream-frame channel); `session.send_message` follows it but
  with a much larger plugin/permission/enrichment surface.
- `_jaato.respond_to_permission`, `respond_to_clarification`,
  etc. — interaction responses that flow through channels.
- Provider auth + credential flow (`_jaato.get_runtime`).

### 7c. Remove the in-process `JaatoSession`

After all `self._jaato.X` callers migrate, the `_jaato` field can
be removed from `JaatoServer.__init__`.  The runner-side host
becomes the single source of truth.  `JAATO_RUNNER_HOSTS_SESSION`
flag goes away.

### 7d. Dependent migrations

- **§3.11 default-share + isolation knob**: ephemeral subagents
  share the parent's runner via `BootstrapEnvelope.parent_runner_handle`.
- **§3.12 ASK queue + drain**: runner-side permission plugin
  buffers ASK prompts when no client is attached
  (`Session.restored_pending_attach`); flushes on attach.
- **Cgroup attach migration**: cgroup attach moves runner-side
  alongside the seat-flip.

## 8. Test invariant for the next contributor

If a daemon-side migration commits to the runner-RPC surface,
the matching handler MUST have:

- A unit test in `jaato-server/server/runner/tests/test_session_<area>_rpc.py`.
- A daemon-side wrapper test in
  `jaato-server/server/runner/tests/test_session_method_wrappers_e2e.py`
  (or a dedicated e2e file).
- The lifecycle composition test
  (`test_session_dispatch_lifecycle_e2e.py`) extended with the
  new step if the migration introduces a new ordering invariant.

This ensures the wire contract stays frozen while the daemon-
side code churns.
