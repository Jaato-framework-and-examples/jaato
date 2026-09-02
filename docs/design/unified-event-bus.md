# Unified Event Bus Design

## Goal

Make the EventBus the single internal event backbone. All events — plan/step lifecycle, tool calls, text responses, agent lifecycle — flow through the EventBus first. `server.emit()` becomes a subscriber that forwards events to IPC/WebSocket clients.

This enables plugins to subscribe to **any** event type through one mechanism.

## Current State

Two separate event systems:

```
todo plugin → EventBus.publish() → subscribers (cross-agent coordination)
AgentUIHooks → server.emit() → IPC/WebSocket clients
```

The EventBus `Event` has plan-centric top-level fields (`source_plan_id`, `source_step_id`, etc.) that don't generalize to tool calls or text responses.

## Target State

```
AgentUIHooks ──→ EventBus.publish() ──→ subscriber: activity detector plugin
todo plugin  ──→                    ──→ subscriber: cross-agent coordination
webhook      ──→                    ──→ subscriber: client forwarder → server.emit() → clients
```

---

## 1. Generalized Event Schema

**File:** `jaato-sdk/jaato_sdk/event_bus.py`

Replace the plan-centric `Event` with a minimal universal envelope. All domain-specific data goes in `payload`.

```python
@dataclass
class Event:
    """Universal event on the bus.

    Minimal envelope carrying event identity and source. All domain-specific
    data (plan context, tool args, text content, token counts) lives in
    ``payload``. This keeps the envelope stable as new event types are added.
    """
    event_id: str
    event_type: EventType
    timestamp: str
    source_agent: str          # agent_id, plugin name, or "server"
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source_agent: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> 'Event':
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            source_agent=source_agent,
            payload=payload or {},
        )
```

### Migration of existing plan events

Current callers like `todo/plugin.py` that set `source_plan_id`, `source_step_description`, etc. move those into `payload`:

```python
# Before
Event.create(
    event_type=EventType.STEP_COMPLETED,
    source_agent="main",
    source_plan_id=plan.plan_id,
    source_step_id=step.step_id,
    source_step_description=step.description,
    payload={"output": output, "result": result},
)

# After
Event.create(
    event_type=EventType.STEP_COMPLETED,
    source_agent="main",
    payload={
        "plan_id": plan.plan_id,
        "step_id": step.step_id,
        "step_description": step.description,
        "output": output,
        "result": result,
    },
)
```

---

## 2. Extended EventType Enum

**File:** `jaato-sdk/jaato_sdk/event_bus.py`

New entries for server-originated events. Dot-separated naming for consistency.

```python
class EventType(Enum):
    # ── Plan lifecycle (existing) ──
    PLAN_CREATED = "plan.created"
    PLAN_STARTED = "plan.started"
    PLAN_COMPLETED = "plan.completed"
    PLAN_FAILED = "plan.failed"
    PLAN_CANCELLED = "plan.cancelled"
    PLAN_CLEARED = "plan.cleared"          # NEW (was only a server event)

    # ── Step lifecycle (existing) ──
    STEP_ADDED = "step.added"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"
    STEP_BLOCKED = "step.blocked"
    STEP_UNBLOCKED = "step.unblocked"

    # ── Agent lifecycle (new) ──
    AGENT_CREATED = "agent.created"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_COMPLETED = "agent.completed"
    AGENT_OUTPUT = "agent.output"          # Model text, system text, plugin text

    # ── Tool execution (new) ──
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_COMPLETED = "tool.call_completed"
    TOOL_OUTPUT = "tool.output"

    # ── Context & turns (new) ──
    TURN_COMPLETED = "turn.completed"
    TURN_PROGRESS = "turn.progress"
    CONTEXT_UPDATED = "context.updated"

    # ── Permission (new) ──
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"

    # ── External (existing) ──
    EXTERNAL_EVENT = "external.event"
```

Not all 41 server event types need EventBus entries on day one. Start with the ones that plugins actually need to observe. Others can be added incrementally.

---

## 3. EventFilter — payload-aware matching

**File:** `jaato-sdk/jaato_sdk/event_bus.py`

`EventFilter` keeps its existing interface but its named fields (`plan_id`, `step_id`) now match against `payload` keys. No change needed for existing subscribers.

```python
@dataclass
class EventFilter:
    """Filter for subscribing to specific events on the bus.

    Named fields (agent_id, plan_id, step_id) are convenience filters
    that match against the event envelope and payload respectively.
    ``event_types`` filters on event type. All fields are optional —
    None means "match any".
    """
    agent_id: Optional[str] = None
    plan_id: Optional[str] = None
    step_id: Optional[str] = None
    event_types: List[EventType] = field(default_factory=list)

    def matches(self, event: Event) -> bool:
        # Agent filter — matches against envelope
        if self.agent_id and self.agent_id != "*":
            if event.source_agent != self.agent_id:
                return False

        # Plan filter — matches against payload
        if self.plan_id and event.payload.get("plan_id") != self.plan_id:
            return False

        # Step filter — matches against payload
        if self.step_id and event.payload.get("step_id") != self.step_id:
            return False

        # Event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False

        return True
```

---

## 4. Server emit → EventBus publish bridge

**File:** `jaato-server/server/core.py`

`JaatoServer.emit()` publishes to the EventBus. A built-in subscriber forwards to IPC/WebSocket clients (the current behavior).

### 4a. Emit publishes to EventBus

```python
class JaatoServer:

    def emit(self, event: ServerEvent) -> None:
        """Publish event to the EventBus. Client forwarding is a subscriber."""
        bus = self._get_event_bus()
        if bus:
            bus_event = self._to_bus_event(event)
            bus.publish(bus_event)
        else:
            # Fallback during early init before runtime exists
            self._forward_to_clients(event)
```

### 4b. Server event → bus event mapping

```python
# Mapping from server EventType to bus EventType
_SERVER_TO_BUS: Dict[ServerEventType, BusEventType] = {
    ServerEventType.AGENT_CREATED: BusEventType.AGENT_CREATED,
    ServerEventType.AGENT_OUTPUT: BusEventType.AGENT_OUTPUT,
    ServerEventType.AGENT_STATUS_CHANGED: BusEventType.AGENT_STATUS_CHANGED,
    ServerEventType.AGENT_COMPLETED: BusEventType.AGENT_COMPLETED,
    ServerEventType.TOOL_CALL_START: BusEventType.TOOL_CALL_STARTED,
    ServerEventType.TOOL_CALL_END: BusEventType.TOOL_CALL_COMPLETED,
    ServerEventType.TOOL_OUTPUT: BusEventType.TOOL_OUTPUT,
    ServerEventType.PLAN_UPDATED: BusEventType.PLAN_UPDATED,
    ServerEventType.PLAN_CLEARED: BusEventType.PLAN_CLEARED,
    ServerEventType.TURN_COMPLETED: BusEventType.TURN_COMPLETED,
    ServerEventType.TURN_PROGRESS: BusEventType.TURN_PROGRESS,
    ServerEventType.CONTEXT_UPDATED: BusEventType.CONTEXT_UPDATED,
    ServerEventType.PERMISSION_INPUT_MODE: BusEventType.PERMISSION_REQUESTED,
    ServerEventType.PERMISSION_RESOLVED: BusEventType.PERMISSION_RESOLVED,
    # Events not mapped are forwarded to clients directly (init, error, etc.)
}


def _to_bus_event(self, server_event: ServerEvent) -> Optional[BusEvent]:
    """Convert a server event to a bus event.

    Extracts the server event's dataclass fields into the bus event payload.
    Unmapped event types return None (forwarded to clients only).
    """
    bus_type = _SERVER_TO_BUS.get(server_event.type)
    if bus_type is None:
        return None

    # Flatten dataclass fields to payload dict, excluding base Event fields
    payload = {
        k: v for k, v in server_event.to_dict().items()
        if k not in ("type", "timestamp")
    }

    return BusEvent.create(
        event_type=bus_type,
        source_agent=payload.get("agent_id", "server"),
        payload=payload,
    )
```

### 4c. Client forwarder subscriber

Registered during server initialization, after the runtime and EventBus exist.

```python
def _setup_client_forwarder(self) -> None:
    """Subscribe to EventBus to forward events to IPC/WebSocket clients.

    This replaces the direct server.emit() → client path. The EventBus
    is now the single dispatch point; this subscriber is the bridge
    to external clients.
    """
    bus = self._get_event_bus()

    def forward_to_clients(bus_event: BusEvent) -> None:
        # Reconstruct the server event from payload and forward
        server_event = self._to_server_event(bus_event)
        if server_event:
            self._forward_to_clients(server_event)

    bus.subscribe(
        subscriber_name="server.client_forwarder",
        filter=EventFilter(),  # All events
        callback=forward_to_clients,
        replay_history=False,
    )
```

### 4d. Unmapped events bypass the bus

Events that don't need internal observation (init progress, errors, help text, session list, etc.) go directly to clients without touching the EventBus:

```python
def emit(self, event: ServerEvent) -> None:
    bus = self._get_event_bus()
    if bus:
        bus_event = self._to_bus_event(event)
        if bus_event:
            bus.publish(bus_event)  # Goes through bus → subscribers → client forwarder
            return
    # No bus or unmapped event type — forward directly
    self._forward_to_clients(event)
```

---

## 5. Plan events — dual publish eliminated

Currently, the todo plugin publishes to EventBus (`STEP_STARTED`, `STEP_COMPLETED`) AND the `LivePlanReporter` emits `PlanUpdatedEvent` via `server.emit()`. These are parallel paths for the same state change.

After unification, the todo plugin publishes to EventBus only. The client forwarder subscriber handles delivery to clients. The `LivePlanReporter` callbacks become EventBus subscribers instead of direct `server.emit()` callers.

This eliminates the dual-publish for plan events.

---

## 6. Activity detector plugin — subscribing

**File:** `jaato-server/shared/plugins/activity_detector/__init__.py`

```python
PLUGIN_KIND = "enrichment"
```

**File:** `jaato-server/shared/plugins/activity_detector/plugin.py`

The plugin subscribes to the EventBus during `set_session()` auto-wiring. It filters for the five event types it cares about.

```python
from jaato_sdk.event_bus import Event, EventFilter, EventType

# The events this plugin observes
OBSERVED_EVENTS = [
    EventType.PLAN_CREATED,
    EventType.STEP_STARTED,
    EventType.STEP_COMPLETED,
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.AGENT_OUTPUT,
]


class ActivityDetectorPlugin:
    """Detects and identifies agent activity transitions.

    Subscribes to the EventBus for plan lifecycle, tool execution,
    and text response events. Provides activity state that other
    plugins or the server can query.

    Lifecycle:
        Created by plugin discovery. Receives EventBus access via
        set_session() auto-wiring (session → runtime → event_bus).
        Subscribes during wiring, unsubscribes on cleanup.
    """

    PLUGIN_KIND = "enrichment"

    def __init__(self):
        self._subscription_id: Optional[str] = None
        self._session = None

    # ── Auto-wiring (called by PluginRegistry) ──

    def set_session(self, session) -> None:
        """Receive session reference and subscribe to EventBus.

        Called automatically by PluginRegistry during configure().
        Uses session → runtime → event_bus to access the bus.
        """
        self._session = session
        bus = session.get_runtime().event_bus
        self._subscription_id = bus.subscribe(
            subscriber_name="activity_detector",
            filter=EventFilter(event_types=OBSERVED_EVENTS),
            callback=self._on_event,
            replay_history=False,
        )

    # ── Event handler ──

    def _on_event(self, event: Event) -> None:
        """Dispatch incoming bus event to the appropriate handler."""
        handler = self._handlers.get(event.event_type)
        if handler:
            handler(self, event)

    def _on_plan_created(self, event: Event) -> None:
        """Plan was created."""
        # event.payload: {plan_id, plan_title, steps: [...]}
        ...

    def _on_step_started(self, event: Event) -> None:
        """Plan step transitioned to in_progress."""
        # event.payload: {plan_id, step_id, step_description, ...}
        ...

    def _on_step_completed(self, event: Event) -> None:
        """Plan step transitioned to completed."""
        # event.payload: {plan_id, step_id, step_description, output, result, ...}
        ...

    def _on_tool_call_started(self, event: Event) -> None:
        """Tool execution began."""
        # event.payload: {agent_id, tool_name, tool_args, call_id}
        ...

    def _on_tool_call_completed(self, event: Event) -> None:
        """Tool execution finished."""
        # event.payload: {agent_id, tool_name, call_id, success, duration_seconds, ...}
        ...

    def _on_agent_output(self, event: Event) -> None:
        """Agent produced text output."""
        # event.payload: {agent_id, source, text, mode}
        # Filter: source == "model" for model text responses
        ...

    _handlers = {
        EventType.PLAN_CREATED: _on_plan_created,
        EventType.STEP_STARTED: _on_step_started,
        EventType.STEP_COMPLETED: _on_step_completed,
        EventType.TOOL_CALL_STARTED: _on_tool_call_started,
        EventType.TOOL_CALL_COMPLETED: _on_tool_call_completed,
        EventType.AGENT_OUTPUT: _on_agent_output,
    }

    # ── EnrichmentPlugin protocol (minimal) ──

    def subscribes_to_prompt_enrichment(self) -> bool:
        return False

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return False

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        return False

    # ── Cleanup ──

    def cleanup(self) -> None:
        """Unsubscribe from EventBus."""
        if self._subscription_id and self._session:
            bus = self._session.get_runtime().event_bus
            bus.unsubscribe(self._subscription_id)
            self._subscription_id = None
```

---

## 7. Migration strategy

The migration is incremental — both paths work during transition.

### Phase 1: Generalize Event schema
- Simplify `Event` to envelope + payload
- Update `EventFilter.matches()` to look into payload
- Update todo plugin callers to put plan fields in payload
- Update existing EventBus tests

### Phase 2: Bridge server.emit() to EventBus
- Add new `EventType` entries for tool/agent/text events
- Add `_to_bus_event()` mapping in `JaatoServer`
- Add client forwarder subscriber
- Modify `emit()` to publish through bus for mapped events
- Direct forwarding for unmapped events (init, error, etc.)

### Phase 3: Eliminate dual-publish for plan events
- `LivePlanReporter` callbacks become EventBus subscribers
- Remove direct `server.emit(PlanUpdatedEvent)` from plan hooks
- Plan events flow: todo plugin → EventBus → client forwarder → clients

### Phase 4: Activity detector plugin
- Create plugin skeleton subscribing to EventBus
- Register in plugin discovery

---

## 8. Performance considerations

**Streaming text chunks:** Every `AgentOutputEvent` (model streaming) goes through EventBus dispatch. The bus iterates subscribers under a lock, calls matched callbacks synchronously. With <10 subscribers this is negligible (~microseconds) compared to the model generation latency.

**Event history:** The bus stores up to 1000 events. Streaming text chunks could fill this quickly. Options:
- Exclude `AGENT_OUTPUT` from history (set a `transient` flag)
- Increase history cap
- Let it rotate naturally (current behavior)

**Thread safety:** Already handled — the EventBus uses `threading.Lock` for subscriptions and `threading.Condition` for long-poll. ServerAgentHooks are already called from background threads.
