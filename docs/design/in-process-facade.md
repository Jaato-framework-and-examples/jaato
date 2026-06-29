# Design: transport-agnostic convenience facade (facade in-process)

**Goal:** let the existing `ask`/`complete`/`stream` facade run against an
**embedded** `jaato.JaatoClient` (no daemon, no runner, no socket), so the same
facade code runs in two modes — IPC daemon and embedded — and the python-sdk
examples become "one code, two modes."

## What already exists (the reusable chain)

The IPCClient's events are *already* a callback→event conversion of an internal
`JaatoSession`'s callbacks. The chain today:

```
JaatoSession.send_message(prompt, on_output=…)  +  AgentUIHooks   (the real runtime)
        │  runner: _make_on_output (rpc.py:362) + _AgentUIHooksNotificationShim (rpc.py:4435)
        ▼  callbacks → NotificationFrames
   runner → daemon RPC
        ▼
JaatoServer  (notifications → typed events: AgentOutputEvent, PermissionRequestedEvent, …)
        │  emit() → self._on_event  (core.py:1383)   ← transport-agnostic tap
        ▼
   IPC / WS sink  →  IPCClient.subscribe(EventType, …)
```

Two facts this nails down:
- **The callback→event mapping is already written** — `AgentUIHooks` (15 hooks,
  `shared/plugins/subagent/ui_hooks.py`) + `on_output` → typed events, via the
  runner shim and JaatoServer.
- **`_on_event` is a transport-agnostic tap** (core.py:1383; settable via
  `set_event_callback`). IPC/WS are just consumers; an in-process consumer is
  equally valid.

**The one catch (core.py:387 "daemon no longer constructs JaatoClient"):** the
runner-tier work intentionally removed JaatoServer's in-process execution path —
the daemon now *always* drives a runner subprocess. So reuse is not free; the
in-process facade must run a `JaatoSession` locally and re-target the existing
mapping to emit in-process events instead of RPC notifications.

## The facade's client contract (what an in-process client must implement)

From `jaato_sdk/client/convenience.py`, the facade rides on exactly this surface:

| Member | Used at | Notes |
|---|---|---|
| `subscribe(EventType, cb) -> unsub` | 72, 156, 238 | sync pub/sub, returns unsubscribe |
| `subscribe_once(EventType, cb) -> unsub` | 157-158, 191-193, 239-240 | one-shot |
| `async send_message(prompt, sources, parallel_tools, attachments)` | 160, 195, 242 | the turn |
| `async respond_to_permission(request_id, resp)` | 103, 108 | permission reply |
| `async connect(timeout) -> bool` | 272 | lifecycle |
| `async create_session(**kwargs) -> sid` | 285 | lifecycle |
| `async register_client_tools(specs)` | 284 | host tools (optional) |
| `async disconnect()` | 287, 294 | lifecycle |

Events it must emit: `AGENT_OUTPUT`, `TURN_COMPLETED`, `SESSION_TERMINATED`,
`AGENT_COMPLETED`, `PERMISSION_REQUESTED`.

`open_session(client_cls, …)` is already generic on the client class (it backs
both `IPCClient.session` and `IPCRecoveryClient.session`) — but it constructs
`client_cls(socket_path, **ipc_ctor_kwargs)`, so the in-process client needs a
ctor that accepts/ignores those, or a small carve-out in `open_session`.

## Two design shapes

### Shape 1 — `InProcessClient` adapter over the embedded `jaato.JaatoClient` (RECOMMENDED)

A new `InProcessClient` that implements the facade contract by wrapping
`jaato.JaatoClient`:
- an **in-process event emitter** (`subscribe`/`subscribe_once`/`emit`) — plain
  pub/sub, no socket;
- an **`AgentUIHooks` impl** installed via the embedded client's
  `set_ui_hooks(...)` that maps each hook → `emit(<typed event>)` — the *same*
  mapping the runner shim does, emitting in-process instead of NotificationFrames;
- `on_output(source,text,mode)` → `emit(AGENT_OUTPUT)`;
- a **permission bridge**: the embedded permission callback → `emit(PERMISSION_REQUESTED)`
  + hold the request; `respond_to_permission` resolves it;
- `async send_message` → `asyncio.to_thread(embedded.send_message, …)`, emit
  `TURN_COMPLETED` on return;
- `connect`/`create_session`/`disconnect` → the embedded `connect()` +
  `configure_tools()` setup/teardown;
- `InProcessClient.session(...)` classmethod + a `_SessionContext` analog (or
  teach `open_session` the in-process ctor shape).

The facade (`convenience.py`) runs **unchanged** on top.

**Pros:** true embedded — no daemon, no runner, no socket, no subprocess (the
actual value of in-process mode); lightweight; `set_ui_hooks` is the natural
seam (the embedded client already takes it); facade untouched.
**Cons:** must replicate-or-share the `AgentUIHooks→event` mapping that lives in
the runner shim + daemon; permission path needs wiring.

### Shape 2 — run JaatoServer in-process with a re-introduced embedded execution path

Re-add the in-process (non-runner) session path JaatoServer lost at core.py:387,
point `_on_event` at an in-process client, ride the facade on that.

**Pros:** reuses JaatoServer's *complete* event machinery (notification→event,
emit, replay, state) — maximal fidelity by construction.
**Cons:** re-opens a path the runner-tier work deliberately closed (fights the
architecture's direction); JaatoServer is the heavy daemon core (plugins, GC,
init) — not "lightweight embedded"; high regression risk; you've essentially
rebuilt the daemon in-process minus the socket.

**Recommendation: Shape 1.** It delivers the real value (embed jaato with no
subprocess), stays light, leaves the facade untouched, and uses a clean seam.
Shape 2 reintroduces a deliberately-removed path and drags daemon weight.

## Fidelity strategy (the load-bearing risk)

The in-process events MUST match the daemon's event shapes exactly, or the
result-equality test measures the adapter, not jaato's in-process mode.

- **v1:** write the `AgentUIHooks→event` mapping fresh in the adapter, mirroring
  `_AgentUIHooksNotificationShim`; add a **contract test** asserting field-parity
  against the shim's frames per event type. Accept slight duplication.
- **v2 (optional):** factor the `AgentUIHooks→typed-event` mapping into a shared
  translator used by BOTH the runner shim (emits NotificationFrames) and the
  in-process adapter (emits events). One mapping, two sinks → fidelity by
  construction, no drift. Higher blast radius (touches the load-bearing runner
  shim), so do it second, behind the v1 tests.

## Scope ceiling (be explicit)

**The governing principle (learned the hard way across this review):** the
daemon is a *host* for the runtime+session. Capabilities that live on
`JaatoRuntime` / `JaatoSession` run wherever the session runs — in-process OR in
a runner. The in-process `JaatoClient` exposes the runtime via `get_runtime()`,
so a capability is "daemon-only" ONLY if it is genuinely about the
daemon's transport. Almost nothing is:

| Capability | Where it lives | In-process? |
|---|---|---|
| Agent loop (model→tools→model) | `JaatoSession.send_message` | ✅ session-level |
| Plugins ("server plugins") | loaded into the runtime registry | ✅ runtime-level |
| Tools / permissions | runtime + session | ✅ |
| Subagents (default-share) | `runtime.create_session` + thread pool (plugin.py:552/1852) | ✅ runtime-level |
| **Event bus** | **`JaatoRuntime._event_bus = EventBus()` (jaato_runtime.py:317)** | ✅ **runtime-level** |
| Reactor engine | premium **bus subscriber** (core.py:147); not in jaato-server | ✅ if wired to the in-proc bus |
| AppArmor sandbox | daemon confines the runner subprocess | ⚠️ isolation *property*, not a functional result |
| IPC/WS transport + **recovery/reconnect** | the daemon socket | ❌ no socket in-process |

So the mapping is:

- **ex01-04 + ex07** (ask/complete/stream, session-memory, typed completion,
  permissions) — core facade, in-process. ✅
- **ex05** (host/client tools) — *trivial* in-process: a "host tool" is the
  client↔daemon round-trip for a tool whose handler runs in your process;
  in-process every tool already does (`configure_tools`). Outcome maps. ✅
- **ex06** (server plugins + the loop) — in-process. The "daemon runs the loop"
  framing is **misleading**: the loop is `JaatoSession.send_message`, which runs
  wherever the session runs. Plugins load into the in-process runtime. The only
  daemon-specific bit is the AppArmor sandbox — an *isolation property*, not a
  functional-result difference (so it doesn't break result-equality). ✅
- **ex08** (subagent, default-share) — runtime: `create_session` + thread pool +
  `AgentUIHooks` lifecycle. Wire `set_runtime(...)` + load the `subagent` plugin.
  Only `agent_params={"isolated": True}` needs a runner (and errors today). ✅
- **ex09** (reactor cascade) — in-process-capable. The EventBus is RUNTIME-level
  (jaato_runtime.py:317), `ctx.create_session` is runtime-level, and the reactor
  engine is a premium **bus subscriber**. The only missing piece is wiring the
  premium engine onto the in-process runtime's bus — a wiring + premium-dependency
  question, NOT a daemon-fundamental one. ✅ (with premium)
- **ex10** (IPCRecoveryClient recovery) — the **only genuinely daemon-only**
  example: it is definitionally about reconnecting the IPC socket to the daemon.
  No socket in-process → nothing to recover. ❌

**Net:** the embedded facade can cover ex01-09; the lone hard exclusion is ex10
(socket recovery), plus the sandbox *property* of ex06 (orthogonal to results).
For a v1 build I'd still START with ex01-04 + ex07 + ex08 (no premium dependency,
no reactor-engine wiring) and treat ex09 as a fast-follow once the premium engine
is wired to the in-process bus. Terminal signal: `TURN_COMPLETED` for plain
turns, `AGENT_COMPLETED` for delegation.

## Components + sizing (Shape 1)

| # | Component | Rough size | Notes |
|---|---|---|---|
| 1 | `InProcessEventEmitter` (subscribe/once/emit) | ~50 LOC | plain pub/sub |
| 2 | `_InProcessUIHooks(AgentUIHooks)` → emit events | ~150-250 LOC | the meaty part; mirror the shim; match event shapes |
| 3 | Permission bridge (callback ↔ respond_to_permission) | ~50 LOC | embedded permission hook |
| 4 | `InProcessClient` (facade contract over JaatoClient) | ~150 LOC | to_thread send_message; lifecycle |
| 5 | `.session()` + `_SessionContext` analog / `open_session` carve-out | ~50 LOC | |
| 6 | Fidelity contract tests + result-equality suite (ex01-04+07) | meaningful | the proof; per-event field-parity vs shim |
| 7 | (v2) shared `AgentUIHooks→event` translator | refactor | optional; kills duplication |

**Effort:** a focused framework feature — ~1 PR for the v1 adapter + tests, an
optional 2nd PR for the v2 shared-mapping refactor. Medium. The mapping (2) and
the fidelity tests (6) are the bulk; everything else is small glue.

## Risks

- **Fidelity** (above) — the #1 risk; mitigated by the contract test / v2 share.
- **Permission semantics** — the embedded permission flow differs from the
  daemon's request/respond cycle; map carefully.
- **async-over-sync timing** — `asyncio.to_thread` around the blocking
  `send_message`; validate that `on_output` emits interleave correctly and that
  cancellation works while the worker thread runs.
- **Scope creep** — keep this to the facade-in-process feature; do NOT let it
  pull in subagent/reactor/recovery (daemon-only).

## Prerequisites — in-process credential resolution

The daemon resolves credential config values — including
`plugin_configs[<provider>].api_key = pass://...` — UPSTREAM, in the
config-value resolver at `subagent/config.py:555-574` (`${VAR}` expansion +
`_resolve_secret_uri`) during profile/spec resolution. `create_provider`
(`jaato_runtime.py:~1201-1235`) only PROMOTES the knob to `config.api_key` and
then **fail-loud-validates** it (`looks_like_unresolved_secret_uri`, PR #415) —
it does NOT resolve.

The embedded path has no upstream resolution step, so the adapter MUST resolve
credential secret URIs itself before `configure_tools` / `create_provider`:

```python
from shared.config_resolver import resolve_secret_uri
api_key = resolve_secret_uri(plugin_configs[provider]["api_key"])
```

`resolve_secret_uri` is the public entry point (added so neither the adapter nor
examples import the private `shared.plugins.subagent.config._resolve_secret_uri`).
It is a no-op for plain keys / env-var configs; `pass://`-style URIs additionally
require **jaato-premium installed in the embedding venv** (the `jaato.premium` →
`secret_resolvers` entry point; resolvers are discovered lazily on first call).

## Decision asked

Build it (Shape 1, v1 first)? If yes, it's framework work with its own PRs +
the fidelity bar — separate from the dual-mode examples (those ship now on the
direct `send_message` API regardless). If no, document in-process as the
embedded sync API and the examples frame it as a different approach.
