# HandoffGate API

A coordination + discovery primitive for daemon-side reactors and the
SDK applications that observe them.

## Where this lives

The HandoffGate is part of the **jaato-premium reactor framework**, not
the public daemon. The split:

| Component | Package | Reason |
|---|---|---|
| `GateRegistry`, `HandoffGate`, lease tokens, persistence, watchdog | `jaato-premium` | Coordination primitive for the reactor runtime that lives in premium |
| `GateAnnouncedEvent`, `GateReleasedEvent`, `GatesSnapshotEvent` types | `jaato-sdk` (public) | Wire-level event types must be deserializable by any client; pre-registering them in the public SDK avoids forcing every subscriber to install premium SDK |
| `subscribe_gates()`, `list_gates()`, `GateAttacher` helper | `jaato-sdk` (public) | Read-only subscriber surface; orchestrators that observe gate events shouldn't depend on premium |
| Reactor framework hooks consuming the registry | `jaato-premium` | The framework owns the lifecycle; premium reactors (memory-advisor, handoff) are the consumers |
| Client-side gate mutation (acquire/release from SDK) | `jaato-premium` command handler | Out of scope for v1 anyway; if added, the command handler is premium |

The public daemon doesn't need to know gates exist. Premium publishes
gate events onto the daemon's existing EventBus; the public SDK
subscribes using the existing event-routing mechanism. Public types are
inert when premium isn't installed (no events ever fire, but the types
deserialize cleanly if they ever do).

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
| **Gate** | A named, daemon-singleton state cell with two values: GREEN (free) and RED (busy). Always tenant-scoped (`_daemon` for daemon-wide gates — see multi-tenancy doc §7.5). |
| **Producer** | The reactor that calls `try_acquire()`. On success, holds a `GateLease` token granting the right to dispatch work and call `announce`/`release`. |
| **Completer** | The reactor that calls `release()`. May be the same reactor as the producer (synchronous work) or a different reactor that fires when the spawned work completes. Completers possess a lease either inherited from the producer or read back from the persisted gate state. |
| **Lease** | An opaque, single-use token returned by `try_acquire`. Required by `announce` and `release` to authenticate the caller. Prevents accidental release by code that didn't acquire. |
| **Intent** | Structured metadata describing what the producer is doing while the gate is RED. Split into public (operational health) and private (workload content) per `public_intent_fields`. Published as part of the announce event. |
| **Outcome** | Structured metadata describing how the work ended. Published as part of the release event. |

A gate's lifecycle:

```
GREEN ──try_acquire(owner) → lease──▶ RED (silent)
                                          │
                                          │ producer spawns work, learns session_id
                                          ▼
                                      RED (announced) ──── GateAnnouncedEvent ──▶ subscribers
                                          │
                                          │ work runs asynchronously
                                          ▼
                                      RED (announced) ──release(lease, outcome)──▶ GREEN
                                                            │
                                                            └─── GateReleasedEvent ──▶ subscribers
                                          │
                                          │ (or watchdog fires if TTL expires)
                                          ▼
                                      GREEN (synthesised release with outcome={"status":"timeout"})
```

Two phases. `try_acquire()` is silent (CAS, no event) and returns a
lease. `announce(lease, intent)` publishes the gate's purpose once the
producer has the metadata to share (typically the spawned
`session_id`). This avoids the race where the event fires before the
session exists.

If the producer crashes between acquire and announce, the watchdog
(§9.3) auto-releases after the gate's TTL elapses. Subscribers see a
release without a prior announce (`was_announced=False`) and treat it
as a recovery signal.

## 3. Premium Reactor-Framework API

The runtime lives in jaato-premium and is consumed by reactors.
Public-SDK consumers don't see this surface — they observe the events
in §4 and use the SDK helpers in §5–6.

### 3.1 Registry

```python
class GateRegistry:
    """Premium-singleton holding all named gates.
    
    Owned by the reactor framework; not exposed on the public
    _ExtensionContext.  Reactors get a reference via the framework
    context provided to their factory.
    """
    
    def get_or_create(
        self,
        name: str,
        *,
        tenant_id: str,                              # required; "_daemon" for daemon-wide
        ttl_seconds: int = 3600,                     # auto-release after RED for this long
        public_intent_fields: Set[str] = frozenset(),
    ) -> HandoffGate: ...
    
    def list(
        self,
        tenant_filter: Optional[str] = None,
    ) -> List[GateState]: ...
```

### 3.2 Gate

```python
class GateLease:
    """Opaque, single-use token returned by try_acquire.
    
    Carries enough info for the registry to validate that the holder
    is the legitimate caller of announce/release.  Internally a
    (gate_name, secret_uuid, acquired_at) tuple; the secret never
    leaves the daemon process.
    """
    gate_name: str
    # Other fields are private to the registry.

class HandoffGate:
    name: str
    tenant_id: str
    state: Literal["green", "red"]
    intent: Optional[Dict[str, Any]]                   # populated on announce
    acquired_at: Optional[datetime]
    acquired_by: Optional[str]                         # service identity ID
    expires_at: Optional[datetime]                     # acquired_at + ttl_seconds
    
    def try_acquire(self, owner: str) -> Optional[GateLease]:
        """CAS GREEN → RED.  Returns a fresh lease iff acquired,
        None if the gate was already RED.
        
        ``owner`` must be the caller's service-identity ID
        (validated against the multi-tenancy doc's identity registry).
        Unknown owners raise immediately rather than silently
        acquiring with a bogus attribution.
        """
    
    def announce(self, lease: GateLease, intent: Dict[str, Any]) -> None:
        """Publish the gate's intent.  Validates the lease before
        emitting; an invalid or stale lease raises GateLeaseInvalid.
        Emits GateAnnouncedEvent on the daemon event bus."""
    
    def release(
        self,
        lease: GateLease,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set state to GREEN.  Validates the lease.  Emits
        GateReleasedEvent if the gate was red.  Releasing a green gate
        whose last lease matches yours is a no-op (idempotent
        recovery).  Releasing a green gate held by someone else
        raises GateLeaseInvalid."""
    
    def is_red(self) -> bool:
        """Non-mutating peek."""
    
    def get_state(self) -> GateState:
        """Snapshot for inspection / event reconstruction."""
```

### 3.3 Concurrency contract

- All gate operations are **synchronous**.  The registry holds a
  `threading.RLock` for the duration of each operation.  The lock is
  released before any event is emitted (events go on the EventBus,
  which dispatches on a separate thread).
- Reactor authors **must not** hold a lease across an `await`.  The
  pattern is: synchronous block to acquire + spawn (which uses public
  daemon APIs that may be async-friendly internally but the producer's
  call is sync); release the lease as part of crash recovery; let the
  completer reactor (different invocation, different thread) call
  release later.
- Async reactors (premium handler runs in an event loop): wrap the
  gate operation in `asyncio.to_thread()` or use the shipped
  `AsyncGateAdapter` that does this.

### 3.4 Intent shape

`intent` is an open dict.  Public/private split is enforced at event
publication: cross-tenant subscribers see only the keys listed in the
gate's `public_intent_fields` (declared at `get_or_create`); same-tenant
subscribers see everything.  Convention:

```python
{
    # --- public fields (operational health) ---
    "kind": "memory-advisor",            # required, describes the work
    "session_id": "sess_abc123",         # optional, set when work is a session
    "expected_duration_seconds": 30,     # optional, hint for clients
    
    # --- private fields (workload content) ---
    "description": "consolidate 12 raw memory entries from session X",
    "started_by_session": "sess_xyz",    # the triggering session
    "tags": ["memory", "background"],
}
```

`session_id` is the key field for SDK consumers — its presence signals
"there is a session you can attach to."  It is conventionally placed
in the public set so cross-tenant operators can attach for monitoring
without seeing user content.

### 3.5 Outcome shape

```python
{
    "status": "success" | "failure" | "cancelled" | "timeout",
    "duration_seconds": 12.4,
    "error": "...",                      # optional, if status != success
    "result_summary": "...",             # optional, free-form
    "session_id": "sess_abc123",         # mirror of intent for correlation
}
```

`status: "timeout"` is reserved for outcomes synthesised by the
watchdog when a gate's TTL expires (§9.3).

## 4. Events

Three typed events on the daemon event bus, registered in the
**public** `jaato-sdk/jaato_sdk/events.py` and `EVENT_TYPE_MAP` so any
client can deserialize them.  Production of these events is gated on
the premium reactor framework being installed; the public daemon never
emits them on its own.

### 4.1 `GateAnnouncedEvent`

```python
class GateAnnouncedEvent(Event):
    type: Literal[EventType.GATE_ANNOUNCED]
    gate_name: str
    tenant_id: str                       # for multi-tenant filtering
    owner: str                           # service-identity ID
    intent: Dict[str, Any]               # public fields only for cross-tenant subscribers
    announced_at: datetime
```

Fired when a producer calls `gate.announce(lease, intent)`.  Delivery
is filtered by `(read, gate)` against `tenant_id` per the
multi-tenancy doc §7.5 — same-tenant subscribers receive the full
intent, cross-tenant subscribers receive only the keys listed in the
gate's `public_intent_fields`.

### 4.2 `GateReleasedEvent`

```python
class GateReleasedEvent(Event):
    type: Literal[EventType.GATE_RELEASED]
    gate_name: str
    tenant_id: str
    owner: str
    outcome: Optional[Dict[str, Any]]
    released_at: datetime
    was_announced: bool                  # False if released without prior announce
```

Fired when `gate.release(lease)` flips state RED → GREEN, **or** when
the watchdog auto-releases on TTL expiry (`outcome.status="timeout"`,
`was_announced` reflects whether announce had fired before timeout).
`was_announced=False` indicates the producer crashed before announce;
subscribers that auto-attached on the announce event simply have
nothing to detach.

### 4.3 `GatesSnapshotEvent`

```python
class GatesSnapshotEvent(Event):
    type: Literal[EventType.GATES_SNAPSHOT]
    gates: List[GateState]               # all currently RED + recently RELEASED
    snapshot_at: datetime
```

Sent to a client immediately after it subscribes to gate events
(analogous to `SessionInfoEvent` on session attach).  Lets late
subscribers reconstruct the current state of in-flight work without
having seen the original announces.  Each `GateState` carries the same
fields as `GateAnnouncedEvent`'s payload plus the gate's current
state.  Filtered by tenant authz like the per-event delivery.

## 5. Python SDK Proposal

### 5.1 Subscription helper

```python
class IPCClient:
    # ... existing methods ...
    
    def subscribe_gates(
        self,
        gate_name: Optional[str] = None,
        kind_filter: Optional[Set[str]] = None,
        tenant_filter: Optional[Set[str]] = None,
        on_snapshot: Optional[Callable[[GatesSnapshotEvent], None]] = None,
        on_announced: Optional[Callable[[GateAnnouncedEvent], None]] = None,
        on_released: Optional[Callable[[GateReleasedEvent], None]] = None,
    ) -> "GateSubscription":
        """Register handlers for gate events.
        
        Args:
            gate_name: If set, only events for this specific gate.
            kind_filter: If set, only events whose intent['kind'] is in
                this set.
            tenant_filter: If set, only events for these tenants.
                Authz still applies — the daemon never delivers a gate
                event the caller can't read.
            on_snapshot: Called once with the current state of all RED
                gates immediately after subscribe.  Use this instead of
                tracking external state across reconnects.
            on_announced: Called with each matching GateAnnouncedEvent.
            on_released: Called with each matching GateReleasedEvent.
        
        Returns a subscription handle with an unsubscribe() method.
        """

    async def list_gates(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[GateState]:
        """Read the current gate registry, filtered by authz.
        
        Use this for one-shot status queries; subscribe_gates() with
        on_snapshot is the right call for orchestrators that want to
        track state continuously.
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
            on_snapshot=self._on_snapshot,
            on_announced=self._on_announced,
            on_released=self._on_released,
        )
        self._attached: Dict[str, str] = {}   # gate_name → session_id
    
    async def _on_snapshot(self, event):
        # Catch up on gates already RED at subscribe time.
        for state in event.gates:
            sid = state.intent.get("session_id") if state.intent else None
            if sid and state.kind in self._kind_filter:
                await self._client.attach_session(sid)
                self._attached[state.gate_name] = sid
    
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
  tenantId: string;
  owner: string;
  intent: GateIntent;
  announcedAt: string;       // ISO 8601
}

interface GateReleasedEvent {
  type: "gate.released";
  gateName: string;
  tenantId: string;
  owner: string;
  outcome: GateOutcome | null;
  releasedAt: string;
  wasAnnounced: boolean;
}

interface GatesSnapshotEvent {
  type: "gates.snapshot";
  gates: GateState[];
  snapshotAt: string;
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
  status: "success" | "failure" | "cancelled" | "timeout";
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

Two reactors, one gate.  Each session that the reactor framework
spawns carries a session-attached-state field `purpose` that filters
distinguish on (this is the "is this the kind of session I care
about?" primitive — see multi-tenancy doc's session-attached-state
hook).

- **Producer reactor** (on `agent.completed` for sessions where
  `state.get("purpose") != "memory-advisor"`):
  ```python
  gate = registry.get_or_create(
      "memory-advisor",
      tenant_id="_daemon",
      ttl_seconds=600,
      public_intent_fields={"kind", "session_id", "expected_duration_seconds"},
  )
  lease = gate.try_acquire(self.service_identity.id)
  if lease is None:
      return                           # already in flight
  try:
      sid = mgr.create_headless_session(
          agent_name="memory-advisor",
          tenant_id="_daemon",
          initial_session_state={"purpose": "memory-advisor"},
      )
      gate.announce(lease, {
          "kind": "memory-advisor",
          "session_id": sid,
          "expected_duration_seconds": 30,
          "description": f"consolidate {n} raw entries from {triggering_sid}",
          "started_by_session": triggering_sid,
      })
      # Lease is now held by the framework; completer reads it back.
  except Exception as e:
      gate.release(lease, {"status": "failure", "error": str(e)})
      raise
  ```

- **Completer reactor** (on `agent.completed` for sessions where
  `state.get("purpose") == "memory-advisor"`):
  ```python
  lease = gate.read_lease()             # framework persists the lease
  if lease is None:
      return                            # gate already released (e.g. timeout)
  gate.release(lease, {
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
# The on_snapshot handler catches up on gates that were already RED
# at subscribe time; on_announced handles new ones.  No external
# state needed across reconnects.
```

### 7.3 Budget guardrail (cross-reactor read)

A budget-guard reactor reads `gate.is_red()` for `"expensive-task"` to
decide whether to permit a new costly operation. It doesn't need to
acquire — just inspect.

### 7.4 DAG orchestrator coordination (cross-process write — out of scope)

The DAG orchestrator (an SDK app) wants to acquire a daemon-side gate
to prevent a reactor from interfering with a critical section.  This
requires a premium-side command handler (`gate.acquire` / `gate.release`
exposed as IPC/WS commands) and a public SDK method.  **Out of scope
for v1** — see §10.

## 8. Persistence

The gate registry persists state to a single JSON file at
`<premium_state>/handoff_gates.json`, written atomically (tmp + rename)
on every state transition.  The file holds the full registry: every
gate's name, tenant, current state, lease (with secret), intent (if
announced), `acquired_at`, `expires_at`, and TTL.

**On premium startup:**

1. Load `handoff_gates.json` if present.
2. For each gate:
   - If `state == "green"`: keep as-is.
   - If `state == "red"`: check `expires_at`.  If past, transition to
     GREEN and emit a synthesised `GateReleasedEvent` with
     `outcome={"status": "timeout"}` — operators see the recovery on
     reconnect.
   - If `state == "red"` and not expired: rebuild the in-memory entry
     keeping the original lease.  The framework can still match the
     completer's `release(lease, ...)` because the lease is the
     persisted secret.

This eliminates the "post-crash duplicate spawn" race the v1 sketch
had: a session that survived restart still maps to a held gate; a new
producer firing post-restart sees RED and no-ops, exactly as it would
have without the crash.

The persistence file is an implementation detail of premium and not
part of the SDK contract.

## 9. Concurrency, Crash, & TTL Semantics

### 9.1 Concurrency

- All gate state mutations go through a single `threading.RLock` on
  the registry.  CAS in `try_acquire` is the lock's only contended
  operation.
- The lock is **always released before** any event emission, before
  any disk I/O, and before any callback invocation.  Persistence is
  done by serialising the in-memory state under the lock, copying the
  bytes, releasing the lock, then writing the file.
- Async reactor handlers must use `asyncio.to_thread()` or the shipped
  `AsyncGateAdapter` — never hold a lease across an `await`.

### 9.2 Crash semantics

- **Producer crashes between try_acquire and announce.**  Lease holds
  the gate RED.  TTL fires (§9.3); watchdog releases with
  `outcome.status="timeout"`.  No subscriber confusion because no
  announce ever fired.
- **Producer crashes after announce, before completer fires.**  Same
  as above; subscribers saw the announce, then see a timeout release.
  Subscribers that auto-attached to the announced session see the
  session terminate normally (or get cleaned up by the workspace
  reaper) — gate timeout and session lifecycle are independent.
- **Daemon crash with gates RED.**  Persisted state is loaded on
  restart (§8).  Gates whose TTL hasn't expired stay RED with their
  original leases; gates whose TTL expired during downtime emit
  synthesised release events on first event-bus subscriber.

### 9.3 TTL & watchdog

Every gate carries `ttl_seconds` (default 3600, per-gate override at
`get_or_create`).  A background watchdog thread (interval = 30s) scans
the registry; for any gate where `now() > expires_at`, it:

1. Acquires the registry lock.
2. Synthesises a release: `outcome = {"status": "timeout",
   "duration_seconds": elapsed}`.
3. Marks the gate GREEN.  Persists.
4. Releases the lock.
5. Emits `GateReleasedEvent` with `was_announced` matching whether
   announce had fired.

TTL exists specifically to break the "producer crashed before
announce" stuck-forever case.  It is **not** a general timeout for
the spawned work — that's the spawned session's responsibility (model
turn timeout, watchdog inside the reactor framework, etc.).

### 9.4 Idempotency rules

- `release(lease)` on a green gate whose last lease matches the
  passed lease: no-op.  This handles the "completer fires after
  watchdog timeout" race (the watchdog already released; the
  completer's lease is now stale; we silently accept).
- `release(lease)` on a green gate held by a *different* lease: raise
  `GateLeaseInvalid`.  Indicates a real bug.
- `announce(lease)` on a green gate, or on a red gate held by a
  different lease: raise `GateLeaseInvalid`.

## 10. Routing & Visibility

`GateAnnouncedEvent`, `GateReleasedEvent`, and `GatesSnapshotEvent` are
delivered through the daemon's existing event-routing machinery,
which the multi-tenancy doc §7.5 extends with `(read, gate)` checks
per-tenant.

Same-tenant subscribers receive the full intent.  Cross-tenant
subscribers (operators, support engineers) receive only the gate's
declared `public_intent_fields` — the registry strips private fields
at emission time, not the subscriber.  This keeps the public/private
split a server-enforced contract, not a client honour-system.

## 11. Open Questions

1. **Should clients be able to acquire/release gates?** Useful for
   cross-process coordination (DAG orchestrator example in §7.4).
   Requires a premium command handler (`gate.acquire` / `gate.release`),
   per-action authz checks, and a public SDK method.  Defer to v2 unless
   a concrete need emerges.

2. **Hierarchical gates.** A "memory-advisor" parent gate with
   per-session child gates.  Useful if the daemon ever needs to track
   multiple concurrent advisors of the same kind.  Out of scope for v1
   (mutex semantics enforce one).

3. **Naming convention.** Recommended: `<kind>` for daemon-wide gates,
   `<kind>:<scope>` for tenant-scoped ones — `memory-advisor`,
   `reviewer:project-42`, `budget-guard:acme`.  Soft convention only;
   no registry enforcement.

4. **Telemetry.** Emit OpenTelemetry spans bracketing acquire → release
   so operators can see gate dwell time.  Span attributes: gate name,
   tenant, kind, owner identity.  Cheap addition; align with the span
   hierarchy in `docs/opentelemetry-design.md`.

5. **Service-token rotation impact.** A reactor's service identity
   changes (token rotation per multi-tenancy doc §11).  Outstanding
   leases held under the old identity must remain valid until release
   or timeout — keying leases on (gate, secret) rather than (gate,
   owner) already gives this; just make sure the implementation
   doesn't accidentally tie lease validity to current identity.

6. **Multi-daemon coordination.**  Cross-daemon gate state would need
   a distributed lock service.  Out of scope for v1.

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

- `docs/design/daemon-extensions.md` — public extension surface
- `docs/design/multi-tenancy.md` — tenant scoping, RBAC, service
  identities consumed by §3, §4, §7, §10 of this doc
- `docs/design/task-graph-orchestrator.md` — DAG orchestrator that consumes
  these events
- `docs/jaato_memory_assessment.md:381` — original "reactor singleton"
  backlog item
- `jaato-sdk/jaato_sdk/client/ipc.py:937` — `attach_session` (used by
  `GateAttacher`)
- `jaato-server/server/session_manager.py:446` — `attached_clients`
  fan-out (the mechanism that makes auto-attach work)
