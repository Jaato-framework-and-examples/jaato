# HandoffGate — SDK wire contract

Reference for SDK consumers (Python, TypeScript, any other language
binding) that subscribe to gate events emitted by the **jaato-premium**
reactor framework. Covers only the public wire surface: event payloads
and subscribe APIs.

The full design — registry, lease semantics, persistence, watchdog,
crash recovery, multi-tenancy authz model — lives in the premium repo
at `jaato-premium/docs/design/handoff-gate-api.md`. Read that if you
need to understand *why* the events look this way; this doc covers
*what* arrives on the wire.

The public daemon does not emit gate events on its own. When premium
isn't installed they simply never fire — the wire types still
deserialize cleanly, so subscriber code is portable across deployments.

## Events

Three typed events. Registered in `jaato-sdk/jaato_sdk/events.py` and
in the `EVENT_TYPE_MAP`, so they deserialize on every connected client.

### `GateAnnouncedEvent`

A producer reactor acquired a gate and announced its intent (typically
after spawning a session whose ID is now known).

```python
class GateAnnouncedEvent(Event):
    type: EventType = EventType.GATE_ANNOUNCED
    gate_name: str
    tenant_id: str
    owner: str                           # service-identity ID
    intent: Dict[str, Any]               # see "Intent shape" below
    announced_at: str                    # ISO 8601
```

Delivery is filtered by tenant authz: same-tenant subscribers receive
the full intent; cross-tenant subscribers receive only the keys the
gate declared as `public_intent_fields`.

### `GateReleasedEvent`

The gate transitioned RED → GREEN, either because the owner called
`release()` or because the watchdog auto-released on TTL expiry.

```python
class GateReleasedEvent(Event):
    type: EventType = EventType.GATE_RELEASED
    gate_name: str
    tenant_id: str
    owner: str
    outcome: Optional[Dict[str, Any]]
    released_at: str
    was_announced: bool                  # False if released without prior announce
```

`was_announced=False` means the producer crashed between `try_acquire`
and `announce` — subscribers that auto-attached on the announce event
simply have nothing to detach.

`outcome.status="timeout"` indicates the watchdog fired.

### `GatesSnapshotEvent`

Sent once to a client immediately after it subscribes to gate events.
Lets late subscribers reconstruct the current state of in-flight work
without having seen the original announces (analogous to
`SessionInfoEvent` for sessions).

```python
class GatesSnapshotEvent(Event):
    type: EventType = EventType.GATES_SNAPSHOT
    gates: List[GateState]
    snapshot_at: str

class GateState(BaseModel):
    gate_name: str
    tenant_id: str
    state: str                           # "green" | "red"
    owner: Optional[str]                 # service-identity ID (when RED)
    intent: Optional[Dict[str, Any]]     # populated when RED + announced
    acquired_at: Optional[str]
    expires_at: Optional[str]            # acquired_at + ttl
```

Filtered by tenant authz like the per-event delivery.

### Intent shape

`intent` is a `Dict[str, Any]` with these conventional keys; producers
may add others.

| Key | Type | Meaning |
|---|---|---|
| `kind` | str | Gate purpose tag (e.g. `"memory-advisor"`, `"reviewer"`). Consumers filter on this. |
| `session_id` | str | Spawned session, when the producer's work runs in a session. Subscribers use this to `attach_session()`. |
| `description` | str | Human-readable label. |
| `started_by_session` | str | Originating session, for attribution. |
| `tags` | list[str] | Free-form classifiers. |

### Outcome shape

`outcome.status` is one of `"success"`, `"failure"`, `"cancelled"`,
`"timeout"`. Other keys (`duration_seconds`, `error`, `result_summary`,
`session_id`) are conventional but not required.

## Python SDK

```python
class IPCClient:
    def subscribe_gates(
        self,
        gate_name: Optional[str] = None,
        kind_filter: Optional[Set[str]] = None,
        tenant_filter: Optional[Set[str]] = None,
        on_snapshot: Optional[Callable[[GatesSnapshotEvent], Awaitable[None]]] = None,
        on_announced: Optional[Callable[[GateAnnouncedEvent], Awaitable[None]]] = None,
        on_released: Optional[Callable[[GateReleasedEvent], Awaitable[None]]] = None,
    ) -> "GateSubscription": ...

    async def list_gates(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[GateState]: ...
```

Authz is enforced server-side: the daemon never delivers a gate event
the caller can't read.

### Auto-attach helper

```python
class GateAttacher:
    """Attaches to every session announced via gate intents matching
    kind_filter.  Common case: an orchestrator wants to observe every
    memory-advisor or reviewer session the daemon spawns."""

    def __init__(self, client: IPCClient, kind_filter: Set[str]): ...
    def stop(self) -> None: ...
```

Usage:

```python
async with IPCClient.connect("/tmp/jaato.sock") as client:
    attacher = GateAttacher(client, kind_filter={"memory-advisor", "reviewer"})
    # ... orchestrator main loop ...
    attacher.stop()
```

## TypeScript SDK

Mirror surface in `jaato-sdk-ts`:

```typescript
interface GateAnnouncedEvent {
  type: "gate.announced";
  gateName: string;
  tenantId: string;
  owner: string;
  intent: GateIntent;
  announcedAt: string;
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
  [k: string]: unknown;
}

interface GateOutcome {
  status: "success" | "failure" | "cancelled" | "timeout";
  durationSeconds?: number;
  error?: string;
  resultSummary?: string;
  sessionId?: string;
  [k: string]: unknown;
}

class IpcClient {
  subscribeGates(opts: {
    gateName?: string;
    kindFilter?: Set<string>;
    onAnnounced?: (e: GateAnnouncedEvent) => void | Promise<void>;
    onReleased?: (e: GateReleasedEvent) => void | Promise<void>;
  }): GateSubscription;
}
```

## See also

- `jaato-premium/docs/design/handoff-gate-api.md` — full design,
  registry/lease/persistence/watchdog semantics, multi-tenancy authz.
- `jaato_premium/reactors/README.md` — reactor-side `ctx.gate(...)`
  API that producers and completers use.
