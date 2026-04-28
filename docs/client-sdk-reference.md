# JAATO Client SDK — Complete Reference

> Scope: Comprehensive reference for building clients against jaato-server using either the Python SDK (`jaato-sdk`) or the TypeScript SDK (`@jaato/sdk`), covering architecture, transport, event protocol, configuration, and feature parity.

## Table of Contents

1. [What Are the JAATO Client SDKs?](#1-what-are-the-jaato-client-sdks)
2. [Architecture Overview](#2-architecture-overview)
3. [Transport Layers](#3-transport-layers)
4. [Event Protocol](#4-event-protocol)
5. [Client API — Full Method Reference](#5-client-api--full-method-reference)
6. [Connection Recovery](#6-connection-recovery)
7. [Configuration](#7-configuration)
8. [Plugin Protocol Types](#8-plugin-protocol-types)
9. [Event Bus (Server-Side)](#9-event-bus-server-side)
10. [Shared Helpers](#10-shared-helpers)
11. [Codegen Pipeline](#11-codegen-pipeline)
12. [Runtime Internals](#12-runtime-internals)
13. [Source Code Map](#13-source-code-map)

---

## 1. What Are the JAATO Client SDKs?

JAATO ships two official client SDKs that provide programmatic access to `jaato-server`:

| SDK | Package | Version | Language | Transport |
|-----|---------|---------|----------|-----------|
| Python | `jaato-sdk` | 0.3.4 | Python 3.10+ | IPC (Unix socket / Windows named pipe) + WebSocket |
| TypeScript | `@jaato/sdk` | 0.1.0 | TypeScript 5.4+ / JavaScript | WebSocket only |

Both SDKs expose the same wire protocol and the same set of typed methods. The Python SDK additionally supports local IPC transport (for TUI/desktop clients), while the TS SDK is WebSocket-only (for browser and Node.js consumers).

**Design principles:**

- **Single source of truth for the event protocol.** The Python SDK's `events.py` (pydantic models) is the canonical definition. The TS SDK's `events.ts` is code-generated from it via `scripts/codegen_ts_events.py`.
- **Method-for-method parity.** Every typed method on `IPCClient` / `IPCRecoveryClient` (Python) has a matching method on `JaatoClient` (TypeScript) with the same semantics and the same noun naming (camelCase per JS convention).
- **Wire-format lockstep.** CI fails if `events.ts` is stale relative to `events.py`. The Python test suite has 90+ baseline JSON snapshots (`tests/baselines/events_wire_format/`) that gate any drift.

---

## 2. Architecture Overview

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  TUI Client  │────▶│  jaato-server       │◀────│  Web Client  │
│  (Python)    │     │  (daemon)           │     │  (TS SDK)    │
│  IPC / WS    │     │                     │     │  WebSocket   │
└─────────────┘     └──────────┬──────────┘     └──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  jaato-sdk (Python)  │
                    │  @jaato/sdk (TS)     │
                    └─────────────────────┘
```

**Key architectural decisions:**

- **Daemon model:** The server runs as a persistent process. Multiple clients can connect to the same session concurrently (TUI + dashboard + reactor). Agent state outlives any single client connection.
- **Event-driven I/O:** Both SDKs expose a typed `subscribe(type, handler)` family plus a catchall `subscribe_all` / `subscribeAll`. Python additionally exposes the underlying `events()` async iterator. See `docs/sdk-event-catalog.md` for the full event catalog and `docs/sdk-perf-baseline.md` for dispatch perf reference numbers.
- **Pydantic → JSON Schema → TypeScript codegen.** The Python SDK's pydantic models are the single source of truth for event shapes. `pydantic.TypeAdapter` emits JSON Schema; `json-schema-to-typescript` generates TS interfaces.

---

## 3. Transport Layers

### 3.1 Python SDK — IPC Transport

The Python SDK connects via platform-native IPC:

| Platform | Default endpoint | Implementation |
|----------|-----------------|----------------|
| Unix / Linux / macOS | `/tmp/jaato.sock` | `asyncio.open_unix_connection` |
| Windows | `\\.\pipe\jaato` | `loop.create_pipe_connection` |

**Message framing:** 4-byte big-endian length prefix + JSON payload. Maximum message size: 10 MB.

```python
HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB
```

**Auto-start:** When `auto_start=True` (default), the client launches `python -m server --daemon` if the socket/pipe is not available. On Windows, the resolved pipe path is passed to avoid shell mangling of backslashes.

**Windows named pipe resolution:** The client accepts multiple input formats (`\.\pipe\jaato`, `pipe\jaato`, `jaato`) and normalizes to `\\.\pipe\<name>`. Uses `WaitNamedPipeW` (1ms timeout) for existence probing and `WaitNamedPipeW` (full timeout) for availability checks — both consume no pipe instances, avoiding ghost-client issues.

### 3.2 TypeScript SDK — WebSocket Transport

The TS SDK uses the standard WebSocket API exclusively (no npm `ws` dependency):

```typescript
// Browser and Node 21+
const ws = new WebSocket("ws://localhost:8080?token=<bearer>");
ws.binaryType = "arraybuffer";
```

**Authentication:** Bearer token via `?token=<token>` query parameter (browser-compatible). Node.js also supports custom headers via the constructor's third argument (when the runtime supports it).

**Binary frames:** The TS transport supports sending raw binary frames via `transport.sendBinary(data)` for multi-frame protocols like file staging.

### 3.3 Handshake Protocol

Both transports follow the same post-connect handshake:

1. **Server → Client:** `ConnectedEvent` with `protocol_version`, `server_info` (includes `client_id`, `server_version`)
2. **Client → Server:** `ClientConfigRequest` with trace paths, working directory, env file path, and `PresentationContext`

**Version gating:** Both SDKs enforce a minimum **wire-protocol** version (`MIN_PROTOCOL_VERSION = "1.0"`) checked against `ConnectedEvent.protocol_version` during handshake. Compat is semver-flavoured (major must match; server minor must be ≥ client's required minor). The daemon's package version (`server_version`) is surfaced for diagnostics but no longer used for compat. Mismatch raises `IncompatibleServerError`. See [`docs/sdk-protocol-versioning.md`](sdk-protocol-versioning.md) for bump policy and CHANGELOG.

---

## 4. Event Protocol

### 4.1 Event Types

The protocol defines ~100 event types in `EventType` (Python) / `EventTypeValue` (TS). Major categories:

| Category | Direction | Examples |
|----------|-----------|---------|
| Connection lifecycle | Server → Client | `connected`, `session.interrupted_turn_recovered` |
| Agent lifecycle | Server → Client | `agent.created`, `agent.output`, `agent.completed` |
| Tool execution | Server → Client | `tool.call_start`, `tool.call_end`, `tool.output` |
| Permission flow | Bidirectional | `permission.requested`, `permission.resolved`, `permission.response` |
| Clarification flow | Bidirectional | `clarification.requested`, `clarification.resolved`, `clarification.batch` |
| Reference selection | Bidirectional | `reference_selection.requested`, `reference_selection.resolved` |
| Plan updates | Server → Client | `plan.updated`, `plan.step_updated`, `plan.cleared` |
| Context / tokens | Server → Client | `context.updated`, `turn.completed`, `turn.progress` (each carries a typed `usage: UsageBreakdown` with token counts, cache hits, reasoning/thinking tokens, and `cost_usd` when known); `gc.config` carries GC strategy/threshold separately |
| Session management | Bidirectional | `session.list`, `session.info`, `session.profiles` |
| Workspace management | Bidirectional | `workspace.list`, `workspace.create`, `config.update` |
| File staging | Bidirectional | `workspace.files.stage_request`, `workspace.files.staged` (multi-frame) |
| Peer channel | Server ↔ Server | `peer.spawn_request`, `peer.agent_output`, `peer.heartbeat` |
| SDK parity verbs | Client → Server | `inject_prompt.request`, `replay_messages.request`, `resolve_fork_point.request` |
| Permission policy | Client → Server | `permission.add_whitelist`, `permission.set_default`, `permission.policy_snapshot.request` |

### 4.2 Event Base Model (Python)

All events inherit from `Event(BaseModel)` with `model_config = ConfigDict(extra='ignore')` for forward compatibility:

```python
class Event(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

Serialization uses `model_dump_json()` (Pydantic v2) with `mode='json'` to handle str-enum subclasses correctly.

### 4.3 Event Union (TypeScript)

The TS SDK exports a discriminated union type `JaatoEvent` that can be narrowed by `event.type`:

```typescript
function handle(event: JaatoEvent): void {
  switch (event.type) {
    case "agent.output":
      // event narrowed to AgentOutputEvent
      console.log(event.text);
      break;
    case "permission.requested":
      // event narrowed to PermissionRequestedEvent
      break;
  }
}
```

### 4.4 PresentationContext

Clients describe their display capabilities via `PresentationContext`, sent in the `ClientConfigRequest` handshake:

```python
class PresentationContext(BaseModel):
    content_width: int = 80
    content_height: Optional[int] = None
    supports_markdown: bool = True
    supports_tables: bool = True
    supports_code_blocks: bool = True
    supports_images: bool = False
    supports_rich_text: bool = True
    supports_unicode: bool = True
    supports_mermaid: bool = False
    supports_expandable_content: bool = False
    client_type: ClientType = ClientType.TERMINAL
    communication_style: Optional[CommunicationStyle] = None
```

`ClientType` values: `TERMINAL`, `WEB`, `CHAT`, `API`. `CommunicationStyle` values: `CONVERSATIONAL` (for chat platforms), `NARRATIVE` (for terminals/web).

The `to_system_instruction()` method generates a compact block injected into the model's system prompt, adapting output formatting based on client capabilities.

---

## 5. Client API — Full Method Reference

### 5.1 Method Parity Table

Every method below exists in both SDKs with identical semantics. Python uses `snake_case`, TypeScript uses `camelCase`.

#### Lifecycle

| Python (`IPCClient` / `IPCRecoveryClient`) | TypeScript (`JaatoClient`) | Purpose |
|---|---|---|
| `connect(timeout=5.0)` | `connect()` | Open transport, handshake, version gate |
| `disconnect()` | `close()` | Close transport, cancel reconnect |
| `events() → AsyncIterator[Event]` | — | Python async iterator (drives the receive loop). TS uses the auto-started internal pump. |
| `drain_events()` | — | Convenience: drive events() and return when disconnected |
| `subscribe(event_type, handler) → unsub` | `subscribe(type, handler) → () => void` | Typed handler — fires only for that event type, narrowed |
| `subscribe_once(event_type, handler) → unsub` | `subscribeOnce(type, handler) → () => void` | One-shot typed handler; auto-unsubscribes after first match |
| `subscribe_all(handler) → unsub` | `subscribeAll(handler) → () => void` | Catchall — every event regardless of type |
| `subscribe_many({EventType: handler, ...}) → unsub_all` | `subscribeMany({type: handler, ...}) → () => void` | Bulk-register typed handlers; single unsub removes all atomically |
| — | `onStatus(handler) → () => void` | Connection state transitions |

#### Conversation

| Python | TypeScript | WS verb |
|---|---|---|
| `send_message(text, attachments?, parallel_tools?)` | `sendMessage(text, attachments?, parallelTools?)` | `message.send` |
| `inject_prompt(text, source_type?, source_id?)` | `injectPrompt(text, sourceType?, sourceId?)` | `inject_prompt.request` |
| `replay_messages(request_id, messages?, timeout_seconds?)` | `replayMessages(requestId, messages?, timeoutSeconds?)` | `replay_messages.request` |
| `resolve_fork_point(request_id, after_message?, after_tool_call?, after_timestamp?)` | `resolveForkPoint(requestId, opts)` | `resolve_fork_point.request` |
| `stop()` | `stop(agentId?)` | `session.stop` |
| `request_history(agent_id?)` | `requestHistory(agentId?)` | `history.request` |
| `respond_to_permission(request_id, response, edited_arguments?)` | `respondToPermission(requestId, response, editedArguments?)` | `permission.response` |
| `respond_to_clarification(request_id, response)` | `respondToClarification(requestId, response, questionIndex?)` | `clarification.response` |
| `respond_to_reference_selection(request_id, response)` | `respondToReferenceSelection(requestId, response)` | `reference_selection.response` |

#### Session Management

| Python | TypeScript | WS verb |
|---|---|---|
| `create_session(name?, profile?, agent?, agent_params?)` | `createSession({name?, profile?, agent?, agentParams?})` | `command.execute` `session.new` — `profile` is polymorphic: `str`/`string` references `.jaato/profiles/<name>.json` on the server; `dict`/`object` is an inline spec (same shape) shipped via `CommandRequest.payload['spec']`. Server rejects inline specs missing `model` with an `ErrorEvent` (no silent fallback). Mutually exclusive with `--profile` argv. `agent` is orthogonal and composes with either form. |
| `attach_session(session_id)` | `attachSession(sessionId)` | `command.execute` `session.attach` |
| `get_default_session()` | `getDefaultSession()` | `command.execute` `session.default` |
| `list_sessions()` | `listSessions()` | `command.execute` `session.list` |
| `list_profiles()` | `listProfiles()` | `command.execute` `session.profiles` — response is a versioned `SessionProfilesEvent` (`schema_version: "1.0"`) carrying a typed `profiles: List[ProfileSummary]` and a separate `parse_errors: List[ProfileParseError]`. `ProfileSummary` exposes only safe-to-display fields (env values are filtered to names; `system_instructions` / `icon_name` / `inherits` are deliberately omitted as deprecated or already-resolved). |
| `end_session()` | `endSession()` | `command.execute` `session.end` |
| `delete_session(session_id)` | `deleteSession(sessionId)` | `command.execute` `session.delete` |

#### Tools

| Python | TypeScript | WS verb |
|---|---|---|
| `register_client_tools(tools, categories?)` | `registerClientTools(tools, categories?)` | `tools.register_client` |
| `respond_to_tool_execution(call_id, result?, error?)` | `respondToToolExecution(callId, result?, error?)` | `tool.execute_result` |
| `disable_tool(tool_name)` | `disableTool(toolName)` | `tools.disable` |
| `request_command_list()` | `requestCommandList()` | `command.list_request` |
| `execute_command(command, args?)` | `executeCommand(command, args?)` | `command.execute` |
| — | `sendRawEvent(event)` | arbitrary type (daemon extensions) |

#### File Staging (TS only — multi-frame protocol)

| TypeScript | WS verb |
|---|---|
| `stageFiles(workspaceId, files)` → Promise<StageFilesEvent> | `workspace.files.stage_request` (TEXT) + N binary frames |

#### Permission Policy Verbs

| Python | TypeScript | WS verb |
|---|---|---|
| `add_whitelist_tools(tools?, patterns?)` | `addWhitelistTools(tools?, patterns?)` | `permission.add_whitelist` |
| `add_blacklist_tools(tools?, patterns?)` | `addBlacklistTools(tools?, patterns?)` | `permission.add_blacklist` |
| `remove_permission_rules(target, tools?, patterns?)` | `removePermissionRules(target, tools?, patterns?)` | `permission.remove` |
| `clear_permission_rules(target?)` | `clearPermissionRules(target?)` | `permission.clear` |
| `set_default_policy(policy)` | `setDefaultPolicy(policy)` | `permission.set_default` |
| `request_policy_snapshot(request_id?)` | `requestPolicySnapshot(requestId?)` | `permission.policy_snapshot.request` |

### 5.2 Key Method Details

**`send_message` / `sendMessage`**

Sends user text to the model. The `parallel_tools` parameter (added in SDK parity Phase 1) provides per-call override of the `JAATO_PARALLEL_TOOLS` env default. `None` keeps the env-configured behavior; `True`/`False` forces the mode for that turn only.

**`inject_prompt` / `injectPrompt`**

Injects a prompt into the session's message queue. The `source_type` dimension selects priority:
- `"user"` — USER priority (mid-turn "steer", interrupts the model at the next safe point)
- `"child"` — CHILD priority (queued behind in-flight work; the "follow-up" pattern)
- `"system"` / `"event"` / `"parent"` — reactor / hook callers

This single verb covers pi-agent's `steer` and `followUp` patterns.

**`replay_messages` / `replayMessages`**

Re-runs the model loop against an explicit message list. When `messages` is omitted/null, replays the session's current `get_history()` — semantically equivalent to "continue from the current state with no new user input". The response arrives via `ReplayMessagesResultEvent`.

**`resolve_fork_point` / `resolveForkPoint`**

Resolves a fork point in session history to a message index. Exactly one specifier should be supplied. The response arrives via `ResolveForkPointResultEvent`. Composes with `replay_messages` for fork/interrogate workflows.

**`stageFiles` (TypeScript only)**

Multi-frame WS protocol for uploading files to a workspace:
1. Client sends `StageFilesRequest` (TEXT frame) declaring file names and sizes
2. Client sends N raw BINARY frames in declared order
3. Server responds with `StageFilesEvent` listing successes/failures per file

Server caps: per-file 10 MB, total 50 MB (configurable per deployment).

**`sendRawEvent` (TypeScript only)**

Escape hatch for daemon-extension verbs that register their own top-level message type (e.g., premium's `reconnect.list`, `auth.token`). The caller owns shape correctness — no validation is performed on the client side.

### 5.3 Python-Only Methods

The Python SDK exposes several methods not in the TS SDK:

| Method | Purpose |
|--------|---------|
| `_send_client_config()` | Sends `.env` path, trace paths, workspace, `PresentationContext` |
| `_start_server()` | Auto-starts daemon subprocess if not running |
| `_wait_for_socket(timeout)` | Waits for IPC endpoint availability |
| `_check_server_running()` | Checks PID file for running daemon |

These are internal to the IPC transport layer. The TS SDK sends equivalent configuration via `JaatoClientOptions.clientConfig` during the `connect()` handshake.

---

## 6. Connection Recovery

### 6.1 Python: `IPCRecoveryClient`

Wraps `IPCClient` with automatic reconnection. State machine:

```
DISCONNECTED → CONNECTING → CONNECTED → RECONNECTING → CONNECTED
                                          └────────────────→ CLOSED
```

**RecoveryConfig:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable auto-reconnection |
| `max_attempts` | `int` | 10 | Max reconnect attempts |
| `base_delay` | `float` | 1.0 | Initial backoff delay (seconds) |
| `max_delay` | `float` | 60.0 | Cap on backoff delay |
| `jitter_factor` | `float` | 0.3 | Random jitter range (±30%) |
| `connection_timeout` | `float` | 5.0 | Per-attempt timeout (seconds) |
| `reattach_session` | `bool` | `True` | Auto-reattach after reconnect |

**Validation:** `RecoveryConfig.__post_init__` enforces: `max_attempts >= 1`, `base_delay >= 0.1`, `max_delay >= base_delay`, `jitter_factor` in `[0.0, 1.0]`, `connection_timeout >= 1.0`.

**Error classification:** `_classify_error()` distinguishes permanent errors (`IncompatibleServerError`, `FileNotFoundError`, permission denied, authentication) from transient errors (connection refused, timeout). Permanent errors stop the reconnect loop immediately.

### 6.2 TypeScript: Built-in Recovery

The `JaatoClient` has recovery built in (no separate recovery class). Configured via `JaatoClientOptions.recovery`:

```typescript
interface RecoveryConfig {
  autoReconnect: boolean;           // default: true
  maxReconnectAttempts: number | null; // default: null (infinite)
  initialBackoffSeconds: number;    // default: 1.0
  maxBackoffSeconds: number;        // default: 30.0
  jitterFactor: number;             // default: 0.1
  autoReattachSessionId: boolean;   // default: false (opt-in)
}
```

**Key difference:** The Python SDK defaults to `reattach_session=True`, while the TS SDK defaults to `autoReattachSessionId=false` (opt-in). When enabled in TS, the client automatically calls `attachSession(sessionId)` on every RECONNECTING → CONNECTED transition.

### 6.3 Cross-SDK Error Types

| Python | TypeScript | Purpose |
|--------|-----------|---------|
| `IncompatibleServerError` | `IncompatibleServerError` | Server version too old; non-retryable |
| `ConnectionError` | `ConnectionError` | Connection failed |
| `ReconnectingError` | `ReconnectingError` | Send attempted during reconnect |
| `ConnectionClosedError` | `ConnectionClosedError` | Operation after permanent close |

---

## 7. Configuration

### 7.1 Python Configuration

Loaded with layered precedence (highest wins):

1. **Environment variables** (`JAATO_IPC_*`)
2. **Project config** (`.jaato/client.json`)
3. **User config** (`~/.jaato/client.json`)
4. **Built-in defaults** (dataclass defaults)

**Environment variable mapping:**

| Env var | Config path |
|--------|------------|
| `JAATO_IPC_AUTO_RECONNECT` | `recovery.enabled` |
| `JAATO_IPC_RETRY_MAX_ATTEMPTS` | `recovery.max_attempts` |
| `JAATO_IPC_RETRY_BASE_DELAY` | `recovery.base_delay` |
| `JAATO_IPC_RETRY_MAX_DELAY` | `recovery.max_delay` |
| `JAATO_IPC_RETRY_JITTER` | `recovery.jitter_factor` |
| `JAATO_IPC_CONNECTION_TIMEOUT` | `recovery.connection_timeout` |
| `JAATO_IPC_REATTACH_SESSION` | `recovery.reattach_session` |

### 7.2 TypeScript Configuration

Passed as constructor options:

```typescript
const client = new JaatoClient({
  url: "ws://localhost:8080",
  token: "<bearer-token>",
  recovery: { autoReconnect: true, autoReattachSessionId: true },
  clientConfig: {
    working_dir: "/home/app",
    permission_timeout: 0,
    presentation: { content_width: 120, client_type: "web" },
  },
  openTimeoutMs: 5000,
  minServerVersion: "0.5.27",  // override only for dev
});
```

---

## 8. Plugin Protocol Types

### 8.1 ToolPlugin Protocol

The `ToolPlugin` (Python `Protocol`) defines the interface for all plugins:

**Required methods:** `name`, `get_tool_schemas()`, `get_executors()`, `initialize()`, `shutdown()`, `get_system_instructions()`, `get_auto_approved_tools()`, `get_user_commands()`

**Optional extensions:** `get_config_schema()`, `get_model_requirements()`, enrichment methods (`enrich_prompt`, `enrich_system_instructions`, `enrich_tool_result`), `format_permission_request()`, `get_command_completions()`, `supports_interactivity()`, `get_supported_channels()`, `set_channel()`, auto-wiring (`set_plugin_registry`, `set_workspace_path`), session persistence (`get_persistence_state`, `restore_persistence_state`).

### 8.2 EnrichmentPlugin Protocol

Lightweight alternative for plugins that don't provide model tools — only participate in the enrichment pipeline (prompt, system instructions, tool results).

### 8.3 ToolSchema

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    category: Optional[str]  # filesystem, code, search, etc.
    discoverability: str  # "core" or "discoverable"
    editable: Optional[EditableContent]
    traits: FrozenSet[str]  # TRAIT_FILE_WRITER, TRAIT_FRAMEWORK_LEVEL, etc.
```

**Tool traits:** Semantic tags driving cross-cutting behavior:
- `TRAIT_FILE_WRITER` — participates in file-enrichment pipeline
- `TRAIT_FRAMEWORK_LEVEL` — opts out of AppArmor sandboxing
- `TRAIT_REPLAY_SAFE` — safe to include in replay/fork sessions

### 8.4 Model Provider Types

Provider-agnostic abstractions shared across all providers:

| Type | Purpose |
|------|---------|
| `Message` | Conversation message (role, parts, message_id, model, provider) |
| `Part` | Content part (text, function_call, function_response, thought, etc.) |
| `FunctionCall` | Tool invocation requested by the model |
| `ToolResult` | Result of executing a tool |
| `ProviderResponse` | Unified response wrapping any provider's output |
| `TokenUsage` | Token statistics (prompt, output, cache, reasoning, thinking) |
| `TurnResult` | Discriminated turn outcome (RESPONSE, ERROR, CANCELLED, etc.) |
| `TurnOutcome` | Discriminator: RESPONSE, TOOL_USE, CANCELLED, ERROR, SAFETY, MAX_TOKENS |
| `CancelToken` | Thread-safe cancellation across threads |

---

## 9. Event Bus (Server-Side)

The `EventBus` (defined in `event_bus.py`) provides cross-agent and cross-plugin event coordination:

**Event types:** Plan lifecycle (created, started, completed, failed, cancelled), step lifecycle (added, started, completed, failed, skipped, blocked, unblocked), agent lifecycle (bridged from server events), tool execution, context/turn, permission, drift measurement, external events.

**Key types:**
- `Event` — universal envelope with `event_id`, `event_type`, `timestamp`, `source_agent`, `payload`
- `EventFilter` — filter by `agent_id`, `plan_id`, `step_id`, `event_types`
- `Subscription` — filter + action (callback, unblock_step, inject_message) with optional expiry

### 9.1 Event Payloads (TypedDict schemas)

`event_payloads.py` defines canonical payload schemas for every bus event type, organized in three groups:

1. **Plan/step lifecycle** — from the todo plugin
2. **Bridged from server events** — mirrors `AgentOutputEvent`, `TurnCompletedEvent`, etc.
3. **Plugin-originated** — webhook, drift monitor

---

## 10. Shared Helpers

### 10.1 Cache Hit Percentage

Both SDKs provide `compute_cache_hit_percent(event)`:

```python
# Python
from jaato_sdk.helpers import compute_cache_hit_percent
percent = compute_cache_hit_percent(turn_completed_event)  # float or None
```

```typescript
// TypeScript
import { computeCacheHitPercent } from "@jaato/sdk";
const percent = computeCacheHitPercent(turnCompletedEvent); // number | null
```

**Contract:**
- Returns `None`/`null` when the provider doesn't report cache stats
- Returns `0.0` when the provider supports caching but had no hits
- Otherwise returns percentage in `[0.0, 100.0]`
- Denominator: `cache_read_tokens + prompt_tokens` (excludes `cache_creation_tokens`)

### 10.2 Trace Logging

Python SDK provides shared trace utilities in `trace.py`:
- `trace(component, msg)` — writes to `JAATO_TRACE_LOG` path
- `provider_trace(component, msg)` — writes to `JAATO_PROVIDER_TRACE` path
- Per-agent provider trace routing via `set_trace_agent_context(agent_id)`

---

## 11. Codegen Pipeline

```
jaato-sdk/jaato_sdk/events.py  (source of truth, pydantic BaseModel)
            │
            ▼
scripts/codegen_ts_events.py    (pydantic.TypeAdapter → JSON Schema →
                                json-schema-to-typescript → TypeScript interfaces)
            │
            ▼
jaato-sdk-ts/src/events.ts     (generated, committed)
```

**CI gate:** `npm run codegen:check` (or `scripts/codegen_ts_events.py --check`) exits non-zero with a diff if `events.ts` is stale.

**Regeneration:**
```bash
# From repo root
.venv/bin/python scripts/codegen_ts_events.py

# Or from jaato-sdk-ts/
npm run codegen
```

---

## 12. Runtime Internals

### 12.1 Event Buffering

Both SDKs buffer events received during request-response operations (e.g., `create_session` consuming init-progress events) so the event stream doesn't lose them. The `events()` iterator drains buffered events first.

Python: `_buffered_events` list on `IPCClient`.
TypeScript: `incoming[]` array on the transport, plus `_bufferedEvents[]` on `JaatoClient`.

### 12.2 Concurrent Reader Prevention

On Python, `events()` and `create_session()` cannot read from the same socket simultaneously (asyncio.StreamReader race). When `_events_active` is `True`, `create_session()` falls back to fire-and-forget and lets `events()` pick up the `SessionInfoEvent`.

On TypeScript, the single transport's event iterator is consumed by one consumer at a time.

### 12.3 Ghost Client Prevention

On Windows, if `wait_for()` cancels a coroutine after `create_pipe_connection()` already established a transport, a ghost client appears on the server. The Python SDK mitigates this by:
- Using the full timeout budget for each attempt (not short per-attempt timeouts)
- Only retrying on errors that prove no transport was created (`ConnectionRefused`, `FileNotFoundError`, `OSError`)
- Stopping on `TimeoutError` (transport may have been created)

### 12.4 Forward Compatibility

`Event.model_config['extra'] = 'ignore'` (Python) silently drops unknown fields from newer servers. The TS type system uses `JaatoEvent` discriminated union — events from newer servers that add new `type` values won't narrow against the existing union, but can still be accessed via `sendRawEvent`.

### 12.5 Client-Side Tool Execution Protocol

1. Client registers tools via `register_client_tools(tools)` → server creates proxy tools
2. Model calls a client-registered tool → server emits `ToolExecuteRequestEvent`
3. Client executes the tool and returns result via `respond_to_tool_execution(call_id, result, error)`
4. Server resumes the model loop with the tool's result

---

## 13. Source Code Map

### Python SDK (`jaato-sdk/`)

| File | Contents |
|------|----------|
| `jaato_sdk/__init__.py` | Public re-exports |
| `jaato_sdk/client/__init__.py` | Client package exports |
| `jaato_sdk/client/config.py` | `RecoveryConfig`, `ClientConfig`, layered config loading, env var mapping |
| `jaato_sdk/client/ipc.py` | `IPCClient` — IPC transport, message I/O, session management, typed methods |
| `jaato_sdk/client/recovery.py` | `IPCRecoveryClient` — auto-reconnection wrapper, state machine |
| `jaato_sdk/events.py` | All pydantic event models (single source of truth), serialization |
| `jaato_sdk/event_payloads.py` | TypedDict payload schemas for the EventBus |
| `jaato_sdk/event_bus.py` | EventBus types (`Event`, `EventFilter`, `Subscription`) |
| `jaato_sdk/constants.py` | SDK-wide constants (`PRERENDERED_LINE_PREFIX`) |
| `jaato_sdk/helpers.py` | `compute_cache_hit_percent` |
| `jaato_sdk/trace.py` | Shared trace logging utilities |
| `jaato_sdk/plugins/__init__.py` | Plugin protocol types package |
| `jaato_sdk/plugins/base.py` | `ToolPlugin` and `EnrichmentPlugin` protocols, `PluginSetting`, `UserCommand`, etc. |
| `jaato_sdk/plugins/model_provider/__init__.py` | Re-exports from types.py |
| `jaato_sdk/plugins/model_provider/types.py` | Provider-agnostic types (`ToolSchema`, `Message`, `Part`, `ProviderResponse`, `TokenUsage`, etc.) |
| `jaato_sdk/plugins/todo/__init__.py` | TODO plugin types |
| `jaato_sdk/plugins/todo/channels.py` | TODO channel types |
| `jaato_sdk/plugins/todo/models.py` | TODO model types |
| `jaato_sdk/tests/test_events_wire_format.py` | Wire format baseline tests |
| `jaato_sdk/tests/test_helpers.py` | Helper function tests |
| `jaato_sdk/tests/test_sdk_parity_methods.py` | Cross-SDK method parity tests |
| `pyproject.toml` | Package metadata (v0.3.4, Python 3.10+, pydantic, python-dotenv) |

### TypeScript SDK (`jaato-sdk-ts/`)

| File | Contents |
|------|----------|
| `src/index.ts` | Public surface re-exports |
| `src/client.ts` | `JaatoClient` class — WS client with typed methods, handshake, reconnect |
| `src/transport.ts` | `openTransport()` — low-level WS transport, frame parse/serialise, auth |
| `src/state.ts` | `ConnectionState`, `ConnectionStatus`, `RecoveryConfig`, `DEFAULT_RECOVERY_CONFIG` |
| `src/helpers.ts` | `computeCacheHitPercent` (mirror of Python) |
| `src/errors.ts` | `ConnectionError`, `ReconnectingError`, `ConnectionClosedError`, `IncompatibleServerError` |
| `src/events.ts` | **Generated** — all event/request TypeScript interfaces (12905 lines) |
| `src/client.test.ts` | 40 unit tests with MockWebSocket |
| `package.json` | Package metadata (v0.1.0, ESM, zero runtime deps) |
| `README.md` | Comprehensive README with API reference, consuming options, publishing workflow |

### Shared (in `jaato-server/shared/`)

| File | Contents |
|------|----------|
| `jaato_client.py` | Server-side session management (consumes the SDK event protocol) |
| `event_bus.py` | EventBus runtime (publisher/subscriber dispatch) |
| `ai_tool_runner.py` | Tool execution framework (uses `ToolPlugin` protocol) |
