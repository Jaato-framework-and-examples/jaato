# SDK Event Catalog

Wire-protocol event catalog for the Jaato Python and TypeScript SDKs.
Used by the typed `subscribe` / `subscribeOnce` / `subscribeMany` API to
let clients react to specific events without a switch.

## Source of truth

`jaato-sdk/jaato_sdk/events.py` is the single source of truth for the
wire protocol:

- `EventType` enum — string identifier for each event (e.g. `"agent.output"`)
- `Event` (pydantic base) and one subclass per event type
- `_EVENT_CLASSES: Dict[str, type]` — maps `EventType.value` → subclass

The TypeScript side is **generated** from these via
`scripts/codegen_ts_events.py`:

- `EventTypeValue` const → mirrors `EventType`
- `EventType` string-literal union → mirrors enum values
- `JaatoEvents` discriminated union → mirrors all subclasses
- One TS interface per Python event class

CI fails if `jaato-sdk-ts/src/events.ts` is stale relative to
`jaato-sdk/jaato_sdk/events.py`. **Never hand-edit `events.ts`.**

## Naming clarification

There are two unrelated `EventType` enums in the codebase:

| Enum | Module | Scope |
|---|---|---|
| `EventType` | `jaato_sdk.events` | **Wire protocol** — what flows over IPC/WS between server and clients |
| `EventType` | `jaato_sdk.event_bus` | **Internal bus** — plugin↔plugin↔session coordination inside the daemon (PLAN_*, STEP_*, DRIFT_MEASURED, EXTERNAL_EVENT) |

The subscribe API uses **`jaato_sdk.events.EventType`** (the wire one).
Some internal-bus events bridge into wire events (e.g. `PLAN_UPDATED`),
but the two enums are intentionally separate — internal coordination
need not appear on the wire and vice versa.

Aliasing for clients (`from jaato_sdk import EventType`) re-exports the
wire enum.

## Event catalog (wire)

The wire protocol currently has **108 event types** across the
categories below. See `events.py` for the canonical list and field
schemas.

| Category | Direction | Examples |
|---|---|---|
| **Connection lifecycle** | S→C | `CONNECTED`, `DISCONNECTED` |
| **Agent lifecycle** | S→C | `AGENT_CREATED`, `AGENT_OUTPUT`, `AGENT_STATUS_CHANGED`, `AGENT_COMPLETED` |
| **Tool execution** | S→C | `TOOL_CALL_START`, `TOOL_CALL_END`, `TOOL_OUTPUT`, `TOOL_STATUS`, `TOOL_ID_REGISTRY` |
| **Permission flow** | S↔C | `PERMISSION_REQUESTED`, `PERMISSION_INPUT_MODE`, `PERMISSION_RESOLVED`, `PERMISSION_RESPONSE`, `PERMISSION_STATUS` |
| **Clarification flow** | S↔C | `CLARIFICATION_REQUESTED`, `CLARIFICATION_QUESTION`, `CLARIFICATION_RESOLVED`, `CLARIFICATION_BATCH`, `CLARIFICATION_BATCH_RESPONSE` |
| **Reference selection** | S↔C | `REFERENCE_SELECTION_REQUESTED`, `REFERENCE_SELECTION_RESOLVED`, `REFERENCE_SELECTION_RESPONSE` |
| **Workspace mismatch** | S↔C | `WORKSPACE_MISMATCH_REQUESTED`, `WORKSPACE_MISMATCH_RESOLVED`, `WORKSPACE_MISMATCH_RESPONSE` |
| **Plan updates** | S→C | `PLAN_UPDATED`, `PLAN_STEP_UPDATED`, `PLAN_CLEARED` |
| **Context / token** | S→C | `CONTEXT_UPDATED`, `TURN_COMPLETED`, `TURN_PROGRESS`, `INSTRUCTION_BUDGET_UPDATED` |
| **System messages** | S→C | `SYSTEM_MESSAGE`, `HELP_TEXT`, `ERROR`, `INIT_PROGRESS`, `RETRY` |
| **Session management** | S↔C | `SESSION_LIST`, `SESSION_INFO`, `SESSION_DESCRIPTION_UPDATED`, `SESSION_PROFILES`, `STOP` |
| **Memory / sandbox / service** | S→C | `MEMORY_LIST`, `SANDBOX_PATHS`, `SERVICE_LIST` |
| **Client requests** | C→S | `SEND_MESSAGE`, `COMMAND`, `COMMAND_LIST_REQUEST`, `INSTRUCTION_BUDGET_REQUEST` |
| **Command list** | S→C | `COMMAND_LIST`, `COMMAND_LIST_REFRESH` |
| **Tool management** | C→S, C↔S | `TOOL_DISABLE_REQUEST`, `TOOLS_REGISTER_CLIENT`, `TOOL_EXECUTE_REQUEST`, `TOOL_EXECUTE_RESULT` |
| **History** | C↔S | `HISTORY_REQUEST`, `HISTORY` |
| **Client config** | C→S | `CLIENT_CONFIG` |
| **Mid-turn prompts** | S→C | `MID_TURN_PROMPT_QUEUED`, `MID_TURN_PROMPT_INJECTED`, `MID_TURN_INTERRUPT` |
| **Session recovery** | S→C | `INTERRUPTED_TURN_RECOVERED` |
| **Post-auth setup** | S↔C | `POST_AUTH_SETUP`, `POST_AUTH_SETUP_RESPONSE` |
| **Workspace management** | C↔S | `WORKSPACE_LIST_REQUEST`, `WORKSPACE_LIST`, `WORKSPACE_CREATE_REQUEST`, `WORKSPACE_CREATED`, `WORKSPACE_SELECT_REQUEST`, `CONFIG_STATUS`, `CONFIG_UPDATE_REQUEST`, `CONFIG_UPDATED` |
| **File staging (WS)** | C↔S | `WORKSPACE_FILES_STAGE_REQUEST`, `WORKSPACE_FILES_STAGED` |
| **Workspace file monitoring** | S→C | `WORKSPACE_FILES_CHANGED`, `WORKSPACE_FILES_SNAPSHOT` |
| **External events** | C→S | `EVENT_EXTERNAL` |
| **Subscription** | S→C | `EVENTS_SUBSCRIBED` |
| **SDK feature parity (session primitives)** | C↔S | `INJECT_PROMPT_REQUEST`, `REPLAY_MESSAGES_REQUEST`, `REPLAY_MESSAGES_RESULT`, `RESOLVE_FORK_POINT_REQUEST`, `RESOLVE_FORK_POINT_RESULT` |
| **SDK feature parity (permission verbs)** | C↔S | `PERMISSION_ADD_WHITELIST_REQUEST`, `PERMISSION_ADD_BLACKLIST_REQUEST`, `PERMISSION_REMOVE_REQUEST`, `PERMISSION_CLEAR_REQUEST`, `PERMISSION_SET_DEFAULT_REQUEST`, `PERMISSION_POLICY_SNAPSHOT_REQUEST`, `PERMISSION_POLICY_SNAPSHOT` |
| **Peer channel (server↔server gossip)** | S↔S | `PEER_HEARTBEAT`, `PEER_SPAWN_REQUEST`, `PEER_SPAWN_ACCEPTED`, `PEER_SPAWN_REJECTED`, `PEER_AGENT_OUTPUT`, `PEER_AGENT_COMPLETED`, `PEER_STOP_REQUEST`, `PEER_STOP_ACKNOWLEDGED` |

## Subscribe API (Python)

```python
from jaato_sdk import IPCClient, EventType

client = IPCClient(...)

# Single typed handler
unsub = client.subscribe(
    EventType.PERMISSION_REQUESTED,
    lambda e: print(f"perm: {e.tool_name}"),
)

# Fire once then auto-unsubscribe
unsub = client.subscribe_once(EventType.AGENT_COMPLETED, on_done)

# Catchall (no type filter)
unsub = client.subscribe_all(lambda e: print(e))

# Many handlers in one call (atomic unsub_all)
unsub_all = client.subscribe_many({
    EventType.PERMISSION_REQUESTED: on_perm,
    EventType.TOOL_CALL_START: on_tool_start,
    EventType.AGENT_COMPLETED: on_done,
})
```

Async handlers (`async def handler(event): ...`) are dispatched
fire-and-forget via `asyncio.create_task`. Order of *delivery* is
FIFO; order of *completion* of async handlers is not guaranteed.

## Subscribe API (TypeScript)

```ts
import { JaatoClient, EventTypeValue } from "jaato-sdk-ts";

const client = new JaatoClient(...);

// Single typed handler — `event` is correctly inferred
const unsub = client.subscribe("PermissionRequested", (event) => {
  console.log(event.requestId);
});

// Fire once
const unsub = client.subscribeOnce("AgentCompleted", onDone);

// Catchall
const unsub = client.subscribeAll((event) => console.log(event));

// Many handlers (atomic unsub)
const unsubAll = client.subscribeMany({
  permissionRequested: onPerm,
  toolCallStart: onToolStart,
  agentCompleted: onDone,
});
```

Async handlers (`async (e) => {...}`) are fire-and-forget. Errors
thrown / promises rejected are logged and swallowed; never break the
event loop.

## Error & threading semantics

| Concern | Behavior |
|---|---|
| **Sync handler throws** | Logged with `event.event_id` + handler index, swallowed. Other handlers for the same event still execute. |
| **Async handler rejects** | Same as above (via `.add_done_callback` in Python, `.catch` in TS). |
| **Unsub during dispatch** | Snapshot of handler list is taken at dispatch start; `unsub()` takes effect on the *next* event. |
| **Pre-connect subscription** | Handlers registered before `connect()` capture the very first events (e.g. `ConnectedEvent`). |
| **Late subscription** | No replay. Handler only sees events from registration onwards. |
| **Threading (Python)** | All dispatch runs on the asyncio event loop where `_recv_loop` lives. Sync handlers execute inline; async via `create_task`. |
| **Threading (TS)** | All dispatch runs on the JS event loop. Sync inline; async via promise. |
