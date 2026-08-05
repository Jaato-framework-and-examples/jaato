# Reactor Bus Session-Scope — Why Unloaded Sessions Don't Reach Reactors

**Status:** current-state reference (2026-06-25). Documents a deliberate
implementation detail that repeatedly surprises people (two engineers + the
advisor in one week): the reactor *engine* is daemon-wide, but the event *bus*
is **per-session**, so a reactor's session-independence is bounded to **loaded**
sessions. This is the gap the [unified-event-bus](unified-event-bus.md)
migration closes; until then, daemon-level event sources (e.g. gate releases for
*unloaded* sessions) need an explicit catch-all.

## The contract people assume (and why it's only half-true)

The reactor mental model: a reactor is not bound to a session; it subscribes to
event *types* and reacts regardless of which session emitted them. People then
reason — correctly as a *design* — "therefore the relay is a single daemon-wide
event bus."

The **effect** matches that model, but only for **loaded** sessions. The
**mechanism** today is *not* a single daemon-wide bus.

## What the code actually does (each row verified 2026-06-25)

| Fact | Evidence |
|------|----------|
| Each runtime creates its **own** `EventBus` instance | `jaato-server/shared/jaato_runtime.py:317` — `self._event_bus = EventBus()` |
| …**deliberately**, for session isolation | `jaato_runtime.py:511` docstring — *"Each runtime has its own EventBus instance, ensuring session isolation"* |
| Each `JaatoServer` builds its own runtime (in `connect`) | `jaato-server/server/core.py:1933` — `self._runtime = JaatoRuntime(...)` |
| ⇒ **N sessions = N EventBus instances** | (the two above) |
| The reactor **engine** is a single daemon-lifetime object | premium `reactors/engine.py` — one instance, `engine.start()` called once |
| `engine.start()` does **not** subscribe to any bus | `engine.py:77` — loads home rules + starts the file watcher only |
| The engine subscribes **per loaded session** | `engine.py` `on_session_ready` (≈:162) — `bus = server.event_bus; bus.subscribe(...)`, keyed `self._subscriptions[session_id]` |
| The subscription is torn down on unload | the per-session keying above |

So the daemon-wide component is the **engine**, which achieves
session-independence by subscribing to **each session's own bus as that session
loads** — bridging N per-session buses, not reading one daemon-wide bus.

## The consequence

A reactor reacts to an event regardless of source session **only while that
session is loaded** (the engine holds a live subscription to its bus). The
moment a session unloads:

- its `EventBus()` instance is gone, and
- the engine's subscription to it is removed.

Any event emitted *for* that session after unload has **no bus to land on and no
subscriber to receive it**. There is no daemon-lifetime bus for it.

## Where this bites: `gate.released` for an unloaded session (reliability T2)

`registry._emit_released` (premium) delivers `gate.released` via the parked
session's `server.emit()` — which resolves to that session's per-session bus.
For the reliability "free-the-runner" tier (T2), the parked session is
**deliberately unloaded** during the human-approval wait. When the gate is later
released, `server` is `None` → the release is skipped → the resume reactor never
fires. This is not a bug in the gate code; it is the per-session-bus scope.

## End state: the unified-event-bus migration

The [unified-event-bus](unified-event-bus.md) direction is a single daemon-wide
`EventBus` as the internal backbone, with **per-session filtering** (not
per-session *instances*) for isolation. Under that model a daemon-level engine
subscription survives unloads and the gap closes structurally. That migration is
designed + substantially landed but not finished; the per-session `EventBus()` +
"session isolation" docstring above is exactly the legacy it replaces.

## Closing the gap before the full migration: three options

| Option | Shape | Cost | Notes |
|--------|-------|------|-------|
| **A — full daemon-wide bus** | Replace per-session `EventBus()` instances with one daemon bus + per-session filtering; engine subscribes once | **High** | The unified-event-bus migration step itself; touches the isolation model; needs the migration owner |
| **B — premium gate-registry callback** | `GateRegistry` (daemon-lifetime) directly invokes a registered resume callback when a released gate's session is unloaded; no bus | **Low** | Bespoke — not the generic reactor-subscribe pattern; couples gate-release to resume |
| **C — daemon-lifetime catch-all bus** | Add one daemon-lifetime `EventBus` the engine subscribes to at `start()`; daemon-level / unloaded-session events publish there | **Low–Modest** | Additive (keeps per-session isolation); preserves the reactor-subscribe pattern; a scoped stepping-stone toward A |

### Option C ("catch-all") in detail — the recommended stepping-stone

The smallest change that keeps the *reactor-subscribe* pattern:

- **jaato-server:** construct one daemon-lifetime `EventBus` at daemon init and
  expose it on `_ExtensionContext` (`server/__main__.py:73` — it already carries
  `session_manager` + `broadcast_event`; `event_bus` is the natural sibling).
- **premium engine:** `engine.start()` subscribes to `ctx.event_bus` (the
  catch-all), **in addition to** the per-session subscriptions. This one is
  daemon-lifetime, so it survives all unloads.
- **premium gate registry:** in `_emit_released`, when the parked session is
  unloaded (`server is None`), publish `gate.released` to the catch-all bus
  instead of skipping.
- **dispatch:** catch-all events have no live session/server, so the engine's
  `_dispatch` must tolerate `server=None`; the reactor routes off the event's
  `session_id` (already present on `gate.released`) and uses `ctx.session_manager`
  (daemon-side) — no live session context needed.

**Cost:** ~50–80 LOC across both repos + tests. **No isolation regression** — the
catch-all carries only daemon-level events (not per-session payload), so the
per-session buses keep isolating per-session data.

**Risks / subtleties:**
- Two engine subscription paths (per-session + catch-all). Manageable.
- No double-delivery: the unloaded branch publishes **only** to the catch-all;
  the loaded branch keeps using the per-session bus. `_emit_released` already
  branches on session-loaded, so it is a clean either/or.
- `_dispatch` must not assume a live `server` for catch-all events. Reactors
  subscribing to catch-all events must be written for the no-server case —
  `reliability_revive` already is: it forks from persisted history via
  `session_manager.get_persisted_history` (jaato #391) + `create_headless_session`.
- Migration alignment: C is a **scoped** daemon-wide bus. The full migration (A)
  can later absorb/generalize it — a stepping-stone, not a throwaway.

## Implemented (jaato #393): the "sink" design — Daniel's call, 2026-06-25

Daniel chose a design that keeps per-session isolation **and** gives reactors true
daemon-wide delivery — a hybrid of A and C rather than either alone:

- Per-session `EventBus` instances stay (per-session subscribers keep their
  isolation, unchanged).
- **One daemon-wide reactor `EventBus`**, owned by `SessionManager`, is added.
  Every per-session bus **sinks** into it via a `reactor_bus_sink` forwarding
  subscription (`callback=reactor_bus.publish`, `replay_history=False`) wired at
  session build, right after `server.initialize()`.  Only the forward crosses
  into the daemon-wide bus — per-session payload stays isolated on the
  per-session bus.
- The daemon-wide bus is exposed on `_ExtensionContext.event_bus`.
- The reactor engine subscribes **once** to `ctx.event_bus` (premium
  `engine.start()`), *replacing* the per-session `on_session_ready` subscriptions,
  so it receives events from all sessions and survives unloads.  (It keeps the
  rule-MERGE half of `on_session_ready` — just not the per-session subscribe.)
- For an **unloaded** session there is no per-session bus to sink from, so the
  daemon-level source publishes straight to the daemon-wide bus: premium
  `registry._emit_released` publishes `gate.released` to `ctx.event_bus` when the
  parked session's `server is None` (instead of skipping), and `_dispatch`
  tolerates `server=None`, routing off the event's `session_id`.

This subsumes option **C** (the sink covers loaded sessions; daemon-level sources
cover unloaded ones) without touching the isolation model, and is the incremental
down-payment toward the full **A** (unified-event-bus) end-state.

**jaato-server foundation: PR #393** — the daemon-wide bus + the sink +
`_ExtensionContext.event_bus` (4 tests).  **Premium wiring** (engine subscribe,
gate-registry publish, `_dispatch` `server=None`) layers on top.  Workspace for the
unloaded fork rides the release **outcome** (`event['outcome']['workspace']`), not
a framework lookup — the intent is nulled at `_release` before `_emit_released`,
but the outcome is deepcopied onto the event.

## TL;DR

- The reactor **engine** is daemon-wide; the event **bus** is **per-session**
  (one `EventBus()` per runtime, for isolation).
- Reactor session-independence therefore holds **only for loaded sessions**.
- Unloaded-session events (e.g. `gate.released` during reliability T2) reach no
  reactor — no daemon-lifetime bus exists for them.
- Fixes: **A** (full daemon-wide bus = the migration), **B** (premium callback),
  or **C** (daemon-lifetime catch-all bus — low-modest, additive, the
  recommended stepping-stone).
