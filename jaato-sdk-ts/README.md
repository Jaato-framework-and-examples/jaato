# @jaato/sdk

TypeScript / JavaScript SDK for [jaato-server](https://github.com/Jaato-framework-and-examples/jaato).

Mirrors the Python [`jaato-sdk`](../jaato-sdk/) method-for-method, with
identical noun naming (camelCase per JS convention) so cross-language
parity is enforced by construction.

**Status: pre-alpha (Phase 2).** This package currently ships only the
codegen-generated event/request types. The `JaatoClient` class that
wraps the WS protocol arrives in Phase 3 — see
[`project_backlog_sdk_feature_parity.md`](../docs/) for the full plan.

## Background: why this exists

The motivating audit was a comparison against
[`@mariozechner/pi-agent-core`](https://github.com/badlogic/pi-mono/blob/main/packages/agent/README.md),
a TypeScript library that exposes a stateful agent with `prompt`,
`steer`, `followUp`, `continue`, `abort`, and `subscribe` methods —
plus `beforeToolCall` / `afterToolCall` hooks and BYO browser-resident
tools.

Walking through pi-agent's surface against jaato-server uncovered three
realisations:

1. **Most of pi-agent's capabilities already exist on the jaato side**
   — under different names. Client-side tools, mid-turn injection,
   per-session caching, streaming events, abort, parallel tool
   execution, fork-and-replay primitives — all shipped.
2. **The gap was at the SDK layer, not the daemon.** Many capabilities
   were reachable only as model-callable tools (premium `session_ops`)
   or via stringly-typed `CommandRequest("permissions", [...])` calls.
   Programmatic clients (TS web components, third-party SDKs) couldn't
   drive them ergonomically.
3. **A TS SDK without parity to the Python SDK would just shift drift
   one level up.** Both languages have to expose the same surface, with
   jaato-native naming (no `prompt`/`steer`/`continue` borrowed from
   pi-agent), or every protocol change splits into two divergent
   implementations.

The result was **the SDK feature parity workstream** —
[`project_backlog_sdk_feature_parity.md`](../docs/) — with five phases:

* **Phase 0** — migrate `events.py` from `@dataclass` to
  `pydantic.BaseModel` so the codegen pipeline has a real schema
  source. ✅ shipped (jaato-server 0.5.26 / jaato-sdk 0.3.0).
* **Phase 1** — typed WS verbs over `JaatoSession.inject_prompt`,
  `replay_messages`, `resolve_fork_point`; typed permission-policy
  mutators; per-call `parallel_tools` override on `SendMessageRequest`.
  Matching async methods on Python `IPCClient` /
  `IPCRecoveryClient`. ✅ shipped (jaato-server 0.5.27 /
  jaato-sdk 0.3.1).
* **Phase 2** — codegen pipeline (this package). ✅ shipped.
* **Phase 3** — `JaatoClient` class wrapping the WS protocol
  method-for-method with the Python SDK. *In progress.*

What pi-agent calls `agent.prompt()` is `JaatoClient.send_message()`
here. `agent.steer(msg)` is `inject_prompt(text, source_type="user")`.
`agent.followUp(msg)` is `inject_prompt(text, source_type="child")`.
`agent.continue()` is `replay_messages()` with no message argument
(replays current history). The naming is jaato's; the capability is
pi-agent-equivalent.

What jaato has that pi-agent doesn't:

* **Daemon model**: the same session can be driven from multiple
  clients concurrently (TUI + dashboard + reactor); the agent state
  outlives any single client connection.
* **Plugin system**: tools, GC strategies, model providers, telemetry,
  permission policies are all pluggable.
* **Fork / interrogate primitives** (`replay_messages` +
  `resolve_fork_point`): an external client can fork a session at any
  point in history, ask a question on the fork, and never disturb the
  source. Premium's `session_ops` plugin builds these into
  model-callable tools (`interrogate_session`, `setup_replay_workspace`,
  etc.); the SDK exposes the underlying primitives so JS / TS clients
  can compose their own flows.
* **Subagents**: a session can spawn child sessions that share the
  parent's runtime but maintain isolated state, with a priority-based
  message queue between them.

What pi-agent has that jaato doesn't (today):

* **In-process embedding** — pi-agent runs as a TS library inside the
  caller's process. Jaato runs as a daemon; clients connect over IPC
  or WebSocket. Different deployment model; not on the parity backlog.

## Wire-protocol types

The full event/request type surface is generated from the Python
side's pydantic models:

```
jaato-sdk/jaato_sdk/events.py  (source of truth, pydantic)
            │
            ▼
scripts/codegen_ts_events.py    (uses pydantic.TypeAdapter to emit JSON Schema,
                                 then pipes through json-schema-to-typescript)
            │
            ▼
jaato-sdk-ts/src/events.ts     (generated, committed)
```

CI fails any PR that touches `events.py` without re-running codegen and
committing the regenerated `events.ts`.

## Regenerating

From the repo root:

```bash
.venv/bin/python scripts/codegen_ts_events.py
```

Or from `jaato-sdk-ts/`:

```bash
npm run codegen
```

## Verifying

The CI staleness gate uses:

```bash
.venv/bin/python scripts/codegen_ts_events.py --check
```

which exits non-zero (with a unified diff) if the committed
`events.ts` is stale relative to a fresh regeneration.

## Importing

```typescript
import {
  EventType,
  JaatoEvent,
  SendMessageRequest,
  AgentOutputEvent,
  // ...
} from "@jaato/sdk";

function handle(event: JaatoEvent): void {
  switch (event.type) {
    case EventType.AGENT_OUTPUT:
      // event narrowed to AgentOutputEvent
      console.log(event.text);
      break;
    case EventType.PERMISSION_REQUESTED:
      // ...
      break;
  }
}
```

## License

BUSL-1.1 (matches jaato-server / jaato-sdk).
