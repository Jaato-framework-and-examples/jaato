# Cascade-as-Client — Design Document

**Status**: Phase 0 (design) — decisions locked, awaiting Phase 1
**Author**: Advisor + Daniel (decisions locked 2026-05-21)
**Date**: 2026-05-21
**Origin**: kb-orchestrator v152-retry-7 + v152-retry-10 surfaced two
related gaps in headless-session lifecycle (Finding A: observability
gap; Finding B: cascade stall on terminal-error).  Daniel's reframing
on 2026-05-21: **"all the cascade should be the client."**  Both
findings resolve to a single architectural shape where the cascade
is a first-class client identity, not a session attribute.

---

## 1. Executive Summary

Today, `cascade_driver_id` is an *attribute* on sessions — stamped at
session.new, persisted on `PoolSlot`, used for slot-affinity routing
(Phase 2 of the cascade-sharing arc, PRs #163 / #167 / #168 / #173
/ #174).

This design promotes `cascade_driver_id` from session attribute to
**first-class client identity**.  A cascade-client:

- Subscribes to events from every session stamped with its cid
- Owns lifecycle decisions for those sessions (e.g., on terminal-
  error: cleanup / escalate / abort whole cascade)
- Replaces the `_HEADLESS_CLIENT_ID` synthetic placeholder for
  reactor-spawned sessions
- Is the SDK primitive the peer asked for ("subscribe_cascade")

The shape unifies three previously-separate problems:

| Symptom | Resolution via cascade-as-client |
|---|---|
| Finding A: kb-orchestrator can't observe `SessionTerminatedEvent` for reactor-spawned sessions (peer's v152-retry-7) | Driver registers cascade-client for its cid → receives events for all sessions stamped with that cid |
| Finding B: cascade stalls indefinitely on terminal-error (peer's v152-retry-10) | Cascade-client handler receives `SessionTerminatedEvent` → calls `delete_session` / aborts cascade |
| Reactor `_HEADLESS_CLIENT_ID` synthetic placeholder hack | Reactor extension IS a cascade-client (registered in-process); the placeholder goes away |
| SDK primitive: subscribe to cascade events | `IPCClient.cascade_events(cid)` async iterator backed by the cascade-client registry |
| Future: TUI cascade workflows | TUI generates cid, registers as cascade-client for human-driven cascades |

---

## 2. Background

### 2.1 Today's model

- A client = an external IPC/WS connection (with a real socket + real
  event sink)
- A session = belongs to its creating client (`attached_clients` is
  the membership set)
- A cascade = a string attribute stamped on multiple sessions
- `_HEADLESS_CLIENT_ID` = synthetic placeholder for sessions with no
  real client (reactor-spawned via `ctx.create_session`)

### 2.2 Why this model breaks down

**Observability gap (Finding A)**: events fire to `attached_clients`
only.  Reactor-spawned sessions have only the synthetic
`_HEADLESS_CLIENT_ID` attached.  No external observer (including the
cascade-driver that generated the cid) can see events for those
sessions.

**Lifecycle gap (Finding B)**: model_thread emits
`SessionTerminatedEvent(reason="error")` for headless sessions, but
the unload-trigger chain
(`_handle_turn_tracking_event` → `_maybe_unload_session`) doesn't fire
because:

1. Terminal-error path doesn't emit `AgentStatusChangedEvent(status="done")`
2. Even if it did, `_maybe_unload_session`'s gate `if session.attached_clients` is True (synthetic client still attached) → unload refused

Result: session lives forever, runner runs forever, slot stays
acquired.

**Reactor coupling**: reactor extension's `ActionContext.create_session`
must pass through the synthetic `_HEADLESS_CLIENT_ID` placeholder.
There's no clean place for reactor-level lifecycle policy.

---

## 3. Locked Decisions (Daniel call, 2026-05-21)

| # | Decision | Locked answer |
|---|---|---|
| 1 | Cascade-client identity | **Namespaced `"_cascade:{cid}"` client_id**.  Prefix distinguishes from real-client UUIDs and the `_headless` legacy placeholder.  Literal `cascade_driver_id` stays as session attribute for PoolSlot routing (existing Phase 2 mechanism). |
| 2 | Registration shape | **Both**.  In-process API (`SessionManager.register_in_process_client(client_id, callback, ...)`) for extensions (reactor).  IPC RPC verb (`cascade.register`) for external clients (kb-orchestrator smoke driver, future TUI cascade).  Both routes go through the same registry. |
| 3 | Event observability scope | **Subscriber-defined filter**.  `register_cascade_client(cid, event_types=[...])` accepts the subscriber's event-type list.  Owner typically subscribes to all; observers can subscribe to a narrow set (e.g., only lifecycle events for fail-fast). |
| 4 | Lifecycle policy | **Framework default + handler override**.  Default policy: on `SessionTerminatedEvent(reason="error")` for a cascade-attached session, framework auto-triggers `_maybe_unload_session` (closes Finding B for free, even with a buggy / incomplete cascade-client).  Cascade-client's handler runs alongside the default — can add escalate / abort-whole-cascade / restart-with-downgrade logic. |
| 5 | Multi-client per cascade | **Multi-observer + owner separated**.  One **owner** per cid (the entity that created the cascade — reactor or smoke driver).  Multiple **observers** can register read-only subscriptions for the same cid.  Owner has lifecycle authority; observers receive events. |
| 6 | Cleanup | **Both explicit + auto-GC backstop**.  Explicit `unregister_cascade_client(cid, client_id=...)` for clean shutdowns.  Auto-GC after `cascade_client_idle_timeout_seconds` (default 300s, matching cascade-idle slot teardown from Phase 2) of no active sessions in the cid.  Backstop catches crash recovery (SDK client disconnect mid-cascade). |
| 7 | `_HEADLESS_CLIENT_ID` disposition | **Keep for non-cascade one-offs**.  Reactor migrates to cascade-client (no more synthetic placeholder for reactor-spawned sessions).  But preserve `_HEADLESS_CLIENT_ID` for genuinely-one-off headless sessions that have no cascade context (future `jaato-server --run-one-prompt 'hello'` mode, or direct extension usage outside the cascade model). |

All 7 decisions locked.  Phase 1 implementation can begin.

---

## 4. Concepts

### 4.1 Cascade-client identity

- **Format**: `_cascade:{cascade_driver_id}` (e.g.,
  `_cascade:0a61ab7a503349cb9f3d696f41fb4c4a`).
- **Source**: derived deterministically from the
  `cascade_driver_id` supplied by the cascade-driver.
- **Owner vs observers**: same cid can have ONE owner + N observers.
  Owner's client_id is the canonical `_cascade:{cid}`; observers
  register with a secondary client_id (e.g., the IPC connection's
  own UUID, or an extension-supplied label).  Internally the
  registry keys by `(cid, role, client_id)` triple.
- **Collision policy**: cascade_driver_id is operator-/client-
  supplied (UUID convention).  Cross-client collisions theoretically
  possible; framework logs a warning when an `_cascade:{cid}` is
  claimed as owner by a different client_id than the first.

### 4.2 Cascade-client lifecycle

```
1. Cascade-driver creates a cascade by generating a cid.
2. Driver registers as cascade-client (owner):
     - In-process (reactor): session_manager.register_in_process_client(
           client_id=f"_cascade:{cid}",
           role="owner",
           callback=self._on_cascade_event,
           event_types=[...],
       )
     - IPC RPC (SDK): await client.register_as_cascade_client(
           cid, event_types=[...], role="owner",
       )
3. Driver creates sessions stamped with cascade_driver_id=cid.
4. _emit_to_session fans out to:
     - session.attached_clients (existing behavior; UI clients)
     - cascade-client registry entries matching session.cascade_driver_id
       (new path; reactor + smoke driver)
5. On terminal events, framework default policy fires
   (_maybe_unload_session for headless sessions).
6. Cascade-client handler also fires (cascade-level policy).
7. Cleanup:
     - Owner calls unregister_cascade_client (explicit)
     - OR auto-GC after 300s of no active sessions in cid
     - OR IPC client disconnects (server-side cleanup detects)
```

**Registration timing**: registration can happen BEFORE or AFTER
first session of cid is created.  If a session arrives before its
cascade-client registers, the daemon-side cascade-client registry
is checked at every event-emit; entries that don't exist yet are
skipped silently.  This avoids buffering events for unregistered
cids (which would leak memory).  Drivers that need to catch the
FIRST event reliably must register BEFORE the first
`session.new` (the typical reactor pattern: register at
extension-load; cid generated at cascade-start; sessions follow).

**Daemon restart**: cascade-client registrations are in-memory
only.  Daemon restart loses all registrations; in-flight cascades
become orphans (their sessions persist via disk save, but the
cascade-driver client must re-register if it survives the
restart).  Out-of-scope for Phase 0; Phase 4+ can revisit.

### 4.3 Event dispatch

`SessionManager._emit_to_session` extends as follows:

```python
def _emit_to_session(self, session_id: str, event: Event) -> None:
    with self._lock:
        session = self._sessions.get(session_id)
        if session is None:
            return

        # Existing: turn-tracking handler.
        self._handle_turn_tracking_event(session, event)

        # Existing: per-client fan-out.
        for client_id in session.attached_clients:
            self._emit_to_client(client_id, event)

        # NEW: cascade-client fan-out (filtered by event-type per
        # registration).
        cid = getattr(session, "cascade_driver_id", None)
        if cid is not None:
            for entry in self._cascade_clients.get(cid, []):
                if entry.event_type_match(event):
                    entry.dispatch(event)

        # NEW: default lifecycle policy.
        self._apply_default_cascade_policy(session, event)
```

`entry.dispatch(event)` routes via:
- In-process: invoke callback synchronously
- IPC client: enqueue on the per-client event queue (existing
  IPC event channel — drain task already handles it)

**Ordering**: cascade-client receives events in the same order
they're emitted by the session.  No re-ordering, no batching.
Each `_emit_to_session` call dispatches synchronously to all
matching cascade-clients.

### 4.4 Lifecycle policy

**Default (framework-applied)** for `SessionTerminatedEvent(reason="error")`:

```python
def _apply_default_cascade_policy(self, session, event):
    if isinstance(event, SessionTerminatedEvent):
        if event.reason == "error":
            # Headless sessions — unload immediately (closes Finding B).
            # Sessions with real clients attached stay alive (real
            # client may reconnect to see history).
            is_headless = (
                session.attached_clients == {self._HEADLESS_CLIENT_ID}
            )
            cid = getattr(session, "cascade_driver_id", None)
            has_cascade_owner = bool(
                self._cascade_clients.get(cid, []) if cid else False
            )
            if is_headless or has_cascade_owner:
                # Defer to background unload (existing _do_session_unload).
                self._maybe_unload_session_forced(session.session_id)
```

**Override surface**: cascade-client owner can register a handler
that runs alongside (NOT instead of) the default.  Handler
signature:

```python
def on_cascade_event(self, event: Event) -> None:
    # Owner-defined policy:
    if isinstance(event, SessionTerminatedEvent) and event.reason == "error":
        # E.g., abort the rest of the cascade:
        for sid in self._sessions_in_cascade:
            session_manager.delete_session(sid)
        self._cascade_aborted = True
```

Handler runs BEFORE the framework default so owner can preempt
(e.g., owner's handler may itself delete the session, making the
framework default a no-op).

### 4.5 SDK surface

**In-process (extension) API**:

```python
session_manager.register_in_process_client(
    client_id=f"_cascade:{cid}",
    role="owner",  # or "observer"
    callback=self._on_cascade_event,
    event_types=[SessionTerminatedEvent, AgentCompletedEvent],
)
# ... later ...
session_manager.unregister_cascade_client(cid, client_id=...)
```

**SDK API** (IPCClient):

```python
# Subscribe — async iterator pattern, mirrors client.events():
async for event in client.cascade_events(
    cid="abc123",
    event_types=[SessionTerminatedEvent, AgentCompletedEvent],
    role="observer",
):
    if isinstance(event, SessionTerminatedEvent) and event.reason == "error":
        # Driver decides: log + abort, or continue
        break

# Underlying mechanism: SDK sends `cascade.register` RPC at
# subscribe time, receives events on the existing event channel
# filtered by cid + types.  AsyncContextManager pattern handles
# auto-unregister on `break` or exception.
```

**RPC protocol** (new verbs):
- `cascade.register(cid, event_types, role)` → returns subscription id
- `cascade.unregister(cid, subscription_id)` → returns ack

### 4.6 Migration from `_HEADLESS_CLIENT_ID`

Per §3.7: `_HEADLESS_CLIENT_ID` kept for genuinely-one-off headless
sessions.  Reactor extension migrates:

```python
# OLD (premium reactor today):
class ActionContext:
    def create_session(self, ..., cascade_driver_id: Optional[str] = None):
        sid = self.session_manager.create_headless_session(
            cascade_driver_id=cascade_driver_id, ...
        )
        # Session attached to _HEADLESS_CLIENT_ID synthetic placeholder.
        # No event sink.  Lifecycle gaps (Finding A + B).

# NEW (after Phase 3):
class ReactorExtension:
    def start(self):
        # Register the reactor as cascade-client owner for ALL cids
        # spawned during this extension's lifetime.  Dispatch uses a
        # per-cid lookup table to route events to the right cascade
        # handler.
        # ... per-cid registration happens at cascade-start ...

class ActionContext:
    def create_session(self, ..., cascade_driver_id: Optional[str] = None):
        # Ensure cascade-client is registered for this cid.
        if cascade_driver_id and not self.session_manager.has_cascade_client(
            cascade_driver_id, role="owner"
        ):
            self.session_manager.register_in_process_client(
                client_id=f"_cascade:{cascade_driver_id}",
                role="owner",
                callback=self._reactor_event_handler,
                event_types=ALL_LIFECYCLE_EVENTS,
            )
        sid = self.session_manager.create_headless_session(
            cascade_driver_id=cascade_driver_id, ...
        )
        # Session attached to cascade-client (NOT _HEADLESS_CLIENT_ID).
        # Lifecycle gaps closed: cascade-client handler + framework
        # default policy fire on terminal events.
```

`_HEADLESS_CLIENT_ID` callers without a `cascade_driver_id`
continue working unchanged (the placeholder path remains for
non-cascade headless use cases).

---

## 5. Edge cases (placeholder)

- 5.1 Cascade-client registers AFTER first session of cid is created
- 5.2 Multiple cascade-clients race-register for same cid (per §3.5)
- 5.3 Cascade-client unregisters mid-cascade (sessions still running)
- 5.4 Session terminates while cascade-client is offline / disconnected
- 5.5 `_HEADLESS_CLIENT_ID` legacy callers (per §3.7)
- 5.6 Daemon restart with in-flight cascades (cascade-clients re-register?)
- 5.7 Cascade-client handler raises (lifecycle policy failure)

---

## 6. Implementation Phases (placeholder)

| Phase | Scope | Approx LoC |
|---|---|---|
| Phase 0 | This document | — |
| Phase 1 | Daemon-side cascade-client registry + event dispatch + default lifecycle policy | ~150 |
| Phase 2 | SDK-side `IPCClient` cascade-client API (in-process + optional RPC per §3.2) | ~80 |
| Phase 3 | Premium reactor migration (replace `_HEADLESS_CLIENT_ID` with cascade-client; lifecycle handler) | ~50 |
| Phase 4 | Documentation + memory entries; mark `_HEADLESS_CLIENT_ID` per §3.7 | ~20 |
| Phase 5 | Validation cascade — peer's kb-orchestrator migrates from sentinel-poll to `client.cascade_events(cid)` | ~50 kb-side |

**Total**: ~250 LoC framework + ~50 LoC kb-side.

---

## 7. Out of Scope (deferred)

- Cross-cascade event subscription (subscribing to events from
  ALL cascades, not just one cid)
- Cascade-level resource limits (memory / cpu / token budget tracked
  per cid)
- Hierarchical cascades (sub-cascade as a child of parent cascade)
- Persisting cascade-client registrations across daemon restarts

---

## 8. Risks

- **Backward-compat**: existing `_HEADLESS_CLIENT_ID` callers
  (reactor today, possibly other extensions).  Per §3.7 decision.
- **Race**: session.new with a cid arrives BEFORE the cascade-client
  registers.  Daemon must buffer or drop events; either choice has
  consequences.
- **Multi-client lifecycle conflict** (per §3.5 candidate B): two
  cascade-clients for same cid disagree on policy (e.g., one wants
  to terminate, another wants to observe).  Owner semantics needed.
- **SDK RPC complexity** (per §3.2 candidate B or C): subscribe RPC
  + ongoing event stream + clean disconnect.  Risk of leaking
  registrations on client crash.

---

## 9. Backlog Cross-References

- [`runner-cascade-sharing.md`](runner-cascade-sharing.md) — the prior
  arc that introduced `cascade_driver_id` as a session attribute
- Peer's v152-retry-7 evidence (cascade-sharing validation)
- Peer's v152-retry-10 findings (this design's surfacing case)
- [`feedback_protocol_method_addition_audits_all_plugins`](../../.claude/projects/-home-apanoia-Sources-Jaato-framework-and-examples-jaato/memory/feedback_protocol_method_addition_audits_all_plugins.md) — anti-pattern memory
- [`project_cascade_sharing_validated_2026-05-21`](../../.claude/projects/-home-apanoia-Sources-Jaato-framework-and-examples-jaato/memory/project_cascade_sharing_validated_2026-05-21.md) — cascade-sharing arc completion

---

End of Phase 0 skeleton.  Decisions in §3 to lock; concept sections
in §4-§5 to flesh out after lock.
