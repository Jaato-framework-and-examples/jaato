# HandoffGate API

A coordination + discovery primitive for daemon-side reactors and the
SDK applications that observe them.

## 1. Motivation

Reactors are daemon-singletons whose handlers are invoked once per
matching event. When a handler dispatches asynchronous work — most often
by spawning a headless session — three concurrency concerns arise:

1. **Producer dedup.** Two concurrent invocations of the same producer
   reactor both decide "no work in flight, spawn one" and double-fire.
2. **Cross-reactor handoff.** The producer and the completer (the reactor
   that observes the spawned session's `agent.completed`) are different
   handler invocations, often different reactors entirely. They need
   shared state that outlives any single invocation.
3. **External observability.** SDK applications (orchestrators,
   dashboards, audit tools) want to know when reactors spawn sessions so
   they can attach, render, or audit. Today reactor-spawned headless
   sessions are invisible to external clients unless someone tells them
   the session ID.

A simple lock solves (1). A persistent flag solves (2). Neither solves
(3). `HandoffGate` solves all three by combining a CAS-based mutex with
intent metadata published as typed events on the daemon event bus.

## 2. Concepts

| Term | Definition |
|---|---|
| **Gate** | A named, daemon-singleton state cell with two values: GREEN (free) and RED (busy). |
| **Producer** | The reactor that calls `try_acquire()`. On success, it has the right to dispatch work. |
| **Completer** | The reactor that calls `release()`. May be the same reactor as the producer (synchronous work) or a different reactor that fires when the spawned work completes. |
| **Intent** | Structured metadata describing what the producer is doing while the gate is RED. Published as part of the announce event. |
| **Outcome** | Structured metadata describing how the work ended. Published as part of the release event. |

A gate's lifecycle:

```
GREEN ──try_acquire()──▶ RED (silent)
                             │
                             │ producer spawns work, learns session_id
                             ▼
                         RED (announced) ──── GateAnnouncedEvent ──▶ subscribers
                             │
                             │ work runs asynchronously
                             ▼
                         RED (announced) ──release(outcome)──▶ GREEN
                                                  │
                                                  └─────── GateReleasedEvent ──▶ subscribers
```

Two phases. `try_acquire()` is silent (CAS, no event); `announce(intent)`
publishes the gate's purpose once the producer has the metadata to share
(typically the spawned `session_id`). This avoids the race where the
event fires before the session exists.

If the producer crashes between acquire and announce, a finally-clause
calls `release()` with `outcome={"error": "...", "announced": False}`.
Subscribers see a release without a prior announce and treat it as a
no-op recovery.

## 3. Server-Side API

### 3.1 Registry

```python
class GateRegistry:
    """Daemon-singleton holding all named gates."""
    
    def get_or_create(self, name: str) -> HandoffGate: ...
    def list(self) -> List[GateState]: ...
```

Registered as a service on the `_ExtensionContext` so reactor extensions
can fetch gates by name without re-implementing the registry.

### 3.2 Gate

```python
class HandoffGate:
    name: str
    state: Literal["green", "red"]
    intent: Optional[Dict[str, Any]]      # populated on announce
    acquired_at: Optional[datetime]
    acquired_by: Optional[str]            # reactor identifier
    
    def try_acquire(self, owner: str) -> bool:
        """CAS GREEN → RED. Owner is the reactor identifier (e.g.
        'memory-advisor.producer'). Returns True iff acquired."""
    
    def announce(self, intent: Dict[str, Any]) -> None:
        """Publish the gate's intent. Must be called after a successful
        try_acquire by the same owner. Emits GateAnnouncedEvent on the
        daemon event bus."""
    
    def release(
        self,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set state to GREEN. Idempotent: releasing a green gate is a
        no-op. Emits GateReleasedEvent if the gate was red."""
    
    def is_red(self) -> bool:
        """Non-mutating peek."""
    
    def get_state(self) -> GateState:
        """Snapshot for inspection / event reconstruction."""
```

### 3.3 Intent shape

`intent` is an open dict, but the convention is:

```python
{
    "kind": "memory-advisor",            # required, describes the work
    "session_id": "sess_abc123",         # optional, set when work is a session
    "description": "consolidate 12 raw memory entries from session X",
    "started_by_session": "sess_xyz",    # optional, the triggering session
    "tags": ["memory", "background"],    # optional, free-form
    "expected_duration_seconds": 30,     # optional, hint for clients
}
```

`session_id` is the key field for SDK consumers — its presence signals
"there is a session you can attach to."

### 3.4 Outcome shape

```python
{
    "status": "success" | "failure" | "cancelled",
    "duration_seconds": 12.4,
    "error": "...",                      # optional, if status != success
    "result_summary": "...",             # optional, free-form
    "session_id": "sess_abc123",         # mirror of intent for correlation
}
```

## 4. Events

Two new typed events on the daemon event bus, registered in
`jaato-sdk/jaato_sdk/events.py` and `EVENT_TYPE_MAP`.

### 4.1 `GateAnnouncedEvent`

```python
class GateAnnouncedEvent(Event):
    type: Literal[EventType.GATE_ANNOUNCED]
    gate_name: str
    owner: str
    intent: Dict[str, Any]
    announced_at: datetime
```

Fired when a producer calls `gate.announce(intent)`. SDK consumers
subscribed to this event learn what the daemon is doing and can take
action — typically, calling `attach_session(intent["session_id"])` if
that field is present and the consumer cares about that gate's `kind`.

### 4.2 `GateReleasedEvent`

```python
class GateReleasedEvent(Event):
    type: Literal[EventType.GATE_RELEASED]
    gate_name: str
    owner: str
    outcome: Optional[Dict[str, Any]]
    released_at: datetime
    was_announced: bool                  # False if released without announce
```

Fired when `gate.release()` flips the state from RED to GREEN.
`was_announced=False` indicates a crash recovery path (acquired but
never announced). Subscribers that auto-attached on the announce event
should detach when they see the release.

## 5. Python SDK Proposal

### 5.1 Subscription helper

```python
class IPCClient:
    # ... existing methods ...
    
    def subscribe_gates(
        self,
        gate_name: Optional[str] = None,
        kind_filter: Optional[Set[str]] = None,
        on_announced: Optional[Callable[[GateAnnouncedEvent], None]] = None,
        on_released: Optional[Callable[[GateReleasedEvent], None]] = None,
    ) -> "GateSubscription":
        """Register handlers for gate events.
        
        Args:
            gate_name: If set, only events for this specific gate.
            kind_filter: If set, only events whose intent['kind'] is in
                this set. Applies to both announced and released events
                (released event correlates by gate_name + owner).
            on_announced: Called with each matching GateAnnouncedEvent.
            on_released: Called with each matching GateReleasedEvent.
        
        Returns a subscription handle with an unsubscribe() method.
        """
```

### 5.2 Convenience: auto-attach orchestrator

```python
class GateAttacher:
    """Auto-attaches to sessions announced via gate intents.
    
    Common case: an orchestrator wants to observe every memory-advisor
    session the daemon spawns, so it can fold the cost into a budget
    or display progress in a dashboard.
    """
    
    def __init__(self, client: IPCClient, kind_filter: Set[str]):
        self._client = client
        self._sub = client.subscribe_gates(
            kind_filter=kind_filter,
            on_announced=self._on_announced,
            on_released=self._on_released,
        )
        self._attached: Dict[str, str] = {}   # gate_name → session_id
    
    async def _on_announced(self, event):
        sid = event.intent.get("session_id")
        if sid:
            await self._client.attach_session(sid)
            self._attached[event.gate_name] = sid
    
    async def _on_released(self, event):
        sid = self._attached.pop(event.gate_name, None)
        # Optional: detach. Most consumers will just stop caring;
        # the session is already terminal at this point.
    
    def stop(self):
        self._sub.unsubscribe()
```

Usage:

```python
async with IPCClient.connect("/tmp/jaato.sock") as client:
    attacher = GateAttacher(client, kind_filter={"memory-advisor", "reviewer"})
    # ... orchestrator main loop ...
    attacher.stop()
```

## 6. TypeScript SDK Proposal

Mirror the Python surface in `jaato-sdk-ts`, idiomatic to TS.

### 6.1 Event types

```typescript
interface GateAnnouncedEvent {
  type: "gate.announced";
  gateName: string;
  owner: string;
  intent: GateIntent;
  announcedAt: string;       // ISO 8601
}

interface GateReleasedEvent {
  type: "gate.released";
  gateName: string;
  owner: string;
  outcome: GateOutcome | null;
  releasedAt: string;
  wasAnnounced: boolean;
}

interface GateIntent {
  kind: string;
  sessionId?: string;
  description?: string;
  startedBySession?: string;
  tags?: string[];
  expectedDurationSeconds?: number;
  [k: string]: unknown;       // extensible
}

interface GateOutcome {
  status: "success" | "failure" | "cancelled";
  durationSeconds?: number;
  error?: string;
  resultSummary?: string;
  sessionId?: string;
  [k: string]: unknown;
}
```

### 6.2 Subscription

```typescript
interface GateSubscriptionOptions {
  gateName?: string;
  kindFilter?: Set<string>;
  onAnnounced?: (e: GateAnnouncedEvent) => void | Promise<void>;
  onReleased?: (e: GateReleasedEvent) => void | Promise<void>;
}

class IpcClient {
  subscribeGates(opts: GateSubscriptionOptions): GateSubscription;
}

interface GateSubscription {
  unsubscribe(): void;
}
```

### 6.3 Auto-attach helper

```typescript
class GateAttacher {
  constructor(
    private client: IpcClient,
    private kindFilter: Set<string>,
  ) {
    this.sub = client.subscribeGates({
      kindFilter,
      onAnnounced: this.onAnnounced.bind(this),
      onReleased: this.onReleased.bind(this),
    });
  }
  
  private attached = new Map<string, string>();
  private sub: GateSubscription;
  
  private async onAnnounced(e: GateAnnouncedEvent) {
    const sid = e.intent.sessionId;
    if (sid) {
      await this.client.attachSession(sid);
      this.attached.set(e.gateName, sid);
    }
  }
  
  private async onReleased(e: GateReleasedEvent) {
    this.attached.delete(e.gateName);
  }
  
  stop() {
    this.sub.unsubscribe();
  }
}
```

## 7. Use Cases

### 7.1 memory-advisor (intra-daemon coordination)

Two reactors, one gate:

- **Producer reactor** (on `agent.completed` for non-advisor sessions):
  ```python
  if not gate.try_acquire("memory-advisor.producer"):
      return
  try:
      sid = mgr.create_headless_session(agent_name="memory-advisor", ...)
      gate.announce({
          "kind": "memory-advisor",
          "session_id": sid,
          "description": f"consolidate {n} raw entries from {triggering_sid}",
          "started_by_session": triggering_sid,
      })
  except Exception:
      gate.release({"status": "failure", "error": str(e)})
      raise
  ```

- **Completer reactor** (on `agent.completed` for advisor sessions):
  ```python
  if self._role(agent_id) != "memory-advisor":
      return
  gate.release({
      "status": "success",
      "session_id": agent_id,
      "duration_seconds": elapsed,
  })
  ```

### 7.2 Orchestrator observability (cross-process discovery)

A dashboard application wants to render every reactor-spawned session.
It runs as an SDK app:

```python
attacher = GateAttacher(client, kind_filter={"memory-advisor"})
# Now every memory-advisor session is auto-attached and the dashboard
# receives all its events as if it had spawned the session itself.
```

No daemon changes, no reactor changes, no coupling between the
orchestrator and the reactors that are spawning these sessions.

### 7.3 Budget guardrail (cross-reactor read)

A budget-guard reactor reads `gate.is_red()` for `"expensive-task"` to
decide whether to permit a new costly operation. It doesn't need to
acquire — just inspect.

### 7.4 DAG orchestrator coordination (cross-process write)

The DAG orchestrator (an SDK app) wants to acquire a daemon-side gate to
prevent a daemon-side reactor from interfering with a critical section.
This requires a server-side command (`gate.acquire` / `gate.release`)
exposed to clients. **Out of scope for v1** — see §10.

## 8. Concurrency & Crash Semantics

- All gate state mutations go through a single `threading.RLock` on the
  registry. CAS in `try_acquire` is the lock's only contended operation.
- Events are emitted *outside* the lock to avoid holding it across
  potentially-blocking event-bus dispatch.
- On daemon crash all gates default GREEN at startup — the in-memory
  registry has no persistence. This is correct: any in-flight work is
  either still loaded (and will eventually complete and release a gate
  that's already green — idempotent no-op) or lost (gate is correctly
  GREEN because no work is actually in flight).
- Released-without-announce is normal in error paths; subscribers must
  handle it.
- Released-without-acquire (i.e., releasing a green gate) is a no-op.

## 9. Routing & Visibility

`GateAnnouncedEvent` and `GateReleasedEvent` are emitted on the
daemon-wide event bus. By default they fan out to every attached client.

This is fine for single-tenant deployments. For multi-tenant or
premium-SSO setups, intent metadata may leak workload information across
users. Two mitigations are possible (see open questions):

1. **Owner-scoped routing.** Only deliver gate events to clients whose
   identity matches the reactor that owns the gate. Requires extending
   `_emit_to_client` with a predicate.
2. **Visibility hints.** Add `intent["visibility"]: "public" | "owner-only"`
   and let consumers (or the daemon) filter.

v1 picks (a): single-tenant assumption, all gate events go to all
attached clients. Premium can layer routing on top via the existing
extension surface.

## 10. Open Questions

1. **Should clients be able to acquire/release gates?** Useful for
   cross-process coordination (DAG orchestrator example in §7.4).
   Requires a server command and an authorisation model. Defer to v2.
2. **Persistent intent for late subscribers.** A client that connects
   after a gate is announced misses the event. Should the daemon
   replay current gate states on connect (analogous to
   `SessionInfoEvent`)? Probably yes — adds a `GatesSnapshotEvent`
   bundling all currently-RED gates' announce payloads.
3. **TTL on RED state.** A producer that crashes between acquire and
   announce leaves the gate RED indefinitely if no completer fires.
   Should the registry auto-release after T seconds with a synthesised
   `outcome={"status": "timeout"}`? Probably yes — defaults to 1 hour,
   per-gate override.
4. **Hierarchical gates.** A "memory-advisor" parent gate with
   per-session child gates. Useful if the daemon ever needs to track
   multiple concurrent advisors of the same kind. Out of scope for v1
   (mutex semantics enforce one).
5. **Naming convention.** Recommended: `<kind>.<scope>` —
   `memory-advisor`, `reviewer.project-42`, `budget-guard`. Should the
   registry enforce a regex? Probably soft convention only.
6. **Telemetry.** Emit OpenTelemetry spans bracketing acquire→release
   so users can see gate dwell time. Cheap to add; align with existing
   span hierarchy in `docs/opentelemetry-design.md`.

## 11. Out of Scope

- Cross-daemon gate coordination (multi-daemon clusters). Would require
  a distributed lock service — separate design.
- Client-side gate mutation (§10.1). Server-side reactors only for v1.
- Gate composition (e.g., "acquire A and B atomically"). Add only if a
  use case emerges.

## 12. Migration / Adoption

The first reactor to migrate is `memory-advisor`, which currently has a
backlog item ("singleton coordination") that this API resolves directly.
After that, any new reactor that spawns headless sessions should adopt
the gate pattern by default.

No existing surface is broken: gates are additive. Reactors that don't
use them continue to work as today; clients that don't subscribe to
gate events ignore them.

## 13. References

- `docs/design/daemon-extensions.md` — extension/reactor surface
- `docs/design/task-graph-reactor.md` — DAG orchestrator that consumes
  these events
- `docs/jaato_memory_assessment.md:381` — original "reactor singleton"
  backlog item
- `jaato-sdk/jaato_sdk/client/ipc.py:937` — `attach_session` (used by
  `GateAttacher`)
- `jaato-server/server/session_manager.py:446` — `attached_clients`
  fan-out (the mechanism that makes auto-attach work)
