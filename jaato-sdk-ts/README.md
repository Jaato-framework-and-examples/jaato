# @jaato/sdk

TypeScript / JavaScript SDK for [jaato-server](https://github.com/Jaato-framework-and-examples/jaato).

Mirrors the Python [`jaato-sdk`](../jaato-sdk/) method-for-method, with
identical noun naming (camelCase per JS convention) so cross-language
parity is enforced by construction.

**Status: pre-release (Phase 3 code + full Python parity shipped,
not yet on npm).** The `JaatoClient` class is implemented and
tested (40 unit tests pass against a mock WebSocket).  Method
surface is feature-equivalent to the Python `IPCClient` —
every wire verb the TUI / dashboard / external SDK consumers
need is exposed as a typed method (see [API reference](#api-reference)
below), including the multi-frame `stageFiles` and opt-in
`autoReattachSessionId` recovery.  The codegen-generated event /
request types stay in lockstep with the Python SDK via the CI
staleness gate.

What's still pending: the npm publish workflow and the first
`npm publish @jaato/sdk@0.1.0`.  Consume locally per the
[Consuming this SDK](#consuming-this-sdk-before-its-published-to-npm)
section below until the first publish lands.

Plan history: [`project_backlog_sdk_feature_parity.md`](../docs/).

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
  method-for-method with the Python SDK. ✅ code + tests
  shipped; first npm publish still pending (see
  [Publishing](#publishing) below).
* **Phase 3.1** — closed the remaining 6 gaps premium flagged
  before starting the jaato-task migration: `attachSession`,
  `createSession`, `getDefaultSession`, `listSessions`,
  `listProfiles`, `respondToToolExecution`.  The last is also
  new on the Python side (jaato-sdk 0.3.3).  ✅ shipped.
* **Phase 3.2** — landed the two items I'd initially deferred
  to v0.2: `stageFiles` (multi-frame TEXT + N binary frame
  protocol; `transport.sendBinary` exposed for any other future
  multi-frame verbs) and opt-in `recovery.autoReattachSessionId`
  (consumer no longer needs to wire the re-attach status
  handler manually).  ✅ shipped.

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

Wire-protocol types only:

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

Full client (connect, send, subscribe):

```typescript
import { JaatoClient, EventType } from "@jaato/sdk";

const client = new JaatoClient({
  url: "ws://localhost:8080",
  token: "<bearer-token>",  // omit when behind a proxy that injects it
  recovery: {
    autoReconnect: true,
    autoReattachSessionId: true,  // re-attach session automatically after a reconnect
  },
});

// Subscribe to all events.
const unsub = client.subscribe((event) => {
  if (event.type === EventType.AGENT_OUTPUT) {
    document.getElementById("chat")!.append(event.text);
  }
});

await client.connect();
await client.createSession({ profile: "researcher" });
await client.sendMessage("Summarise the latest commits.");
```

If you want to react to connection-state transitions yourself
(e.g. show a "Reconnecting…" banner) instead of relying on the
opt-in re-attach, drop `autoReattachSessionId` and wire the
handler explicitly:

```typescript
import { ConnectionState } from "@jaato/sdk";

client.onStatus((status) => {
  if (status.state === ConnectionState.RECONNECTING) {
    showBanner(`Reconnecting (attempt ${status.reconnectAttempt})…`);
  }
  if (status.state === ConnectionState.CONNECTED && client.sessionId) {
    void client.attachSession(client.sessionId);
    hideBanner();
  }
});
```

File staging (multi-frame protocol — TEXT request + N binary
frames + typed response):

```typescript
const fileBlob = await fetch("/some-asset.png").then((r) => r.arrayBuffer());

const result = await client.stageFiles("workspace_abc", [
  { name: "logo.png", data: fileBlob, contentType: "image/png" },
  { name: "config.json", data: new TextEncoder().encode(JSON.stringify(cfg)) },
]);

if (result.failed.length > 0) {
  console.error("Some files failed:", result.failed);
}
console.log("Staged:", result.staged.map((f) => f.name));
```

The call resolves with the typed `StageFilesEvent` once the
server reports back; per-file failures are surfaced in
`result.failed` so partial successes are recoverable.
Concurrent `stageFiles` calls on the same client must be
serialised (the response correlation is by ordering, not by ID).

## API reference

Method-for-method mirror of the Python `IPCClient` /
`IPCRecoveryClient`.  All methods are `async` and return
`Promise<void>` unless otherwise noted; results arrive on the
event stream and are correlated by `request_id` where applicable.

**Lifecycle**

| Method | WS verb | Purpose |
|---|---|---|
| `connect()` | (handshake) | Open WS, await `ConnectedEvent`, enforce `MIN_SERVER_VERSION` |
| `close()` | — | Close WS and cancel any pending reconnect |
| `subscribe(handler)` | — | Receive every incoming `JaatoEvent` |
| `events()` | — | Async iterator alternative to `subscribe` |
| `onStatus(handler)` | — | Connection-state transitions |

**Conversation**

| Method | WS verb |
|---|---|
| `sendMessage(text, attachments?, parallelTools?)` | `message.send` |
| `injectPrompt(text, sourceType?, sourceId?)` | `inject_prompt.request` (steer / follow-up) |
| `replayMessages(requestId, messages?, timeoutSeconds?)` | `replay_messages.request` (continue from current) |
| `resolveForkPoint(requestId, opts)` | `resolve_fork_point.request` |
| `stop(agentId?)` | `stop` |
| `requestHistory(agentId?)` | `history.request` |

**Session management**

| Method | WS verb |
|---|---|
| `createSession({ name?, profile?, agent?, agentParams? })` | `command.execute` `session.new` |
| `attachSession(sessionId)` | `command.execute` `session.attach` |
| `getDefaultSession()` | `command.execute` `session.default` |
| `listSessions()` | `command.execute` `session.list` |
| `listProfiles()` | `command.execute` `session.profiles` |
| `endSession()` | `command.execute` `session.end` (terminate current attached session) |
| `deleteSession(sessionId)` | `command.execute` `session.delete` (purge from disk + memory) |

**Tools (model-callable + client-registered)**

| Method | WS verb |
|---|---|
| `registerClientTools(tools, categories?)` | `tools.register_client` |
| `respondToToolExecution(callId, result?, error?)` | `tool.execute_result` (return result for client-registered tool) |
| `disableTool(toolName)` | `tool.disable.request` |
| `requestCommandList()` | `command_list.request` |
| `executeCommand(command, args?)` | `command.execute` (escape hatch for any verb without a typed method) |

**File staging**

| Method | WS verb |
|---|---|
| `stageFiles(workspaceId, files)` | `workspace.files.stage_request` (TEXT) + N binary frames; resolves with `StageFilesEvent` |

**Permissions**

| Method | WS verb |
|---|---|
| `addWhitelistTools(tools?, patterns?)` | `permission.add_whitelist` |
| `addBlacklistTools(tools?, patterns?)` | `permission.add_blacklist` |
| `removePermissionRules(target, tools?, patterns?)` | `permission.remove` |
| `clearPermissionRules(target?)` | `permission.clear` |
| `setDefaultPolicy(policy)` | `permission.set_default` |
| `requestPolicySnapshot(requestId?)` | `permission.policy_snapshot.request` |
| `respondToPermission(requestId, response, editedArgs?)` | `permission.response` |

**Prompts (mid-flow)**

| Method | WS verb |
|---|---|
| `respondToClarification(requestId, response, questionIndex?)` | `clarification.response` |
| `respondToReferenceSelection(requestId, response)` | `reference_selection.response` |

## Consuming this SDK before it's published to npm

Premium webcomponents (and any other early consumer) can wire the
SDK in locally without waiting for the first `npm publish`. Pick
the option that matches your setup.

### Option A — `npm link` (developer-mode symlink)

Best when you're actively iterating on both the SDK and the
consumer. A symlink in the consumer's `node_modules` points at
your local `jaato-sdk-ts/dist/`, so a rebuild here is picked up
immediately on the consumer side.

```bash
# In jaato repo:
cd jaato-sdk-ts
npm install
npm run build
npm link

# In premium repo (any consuming package.json):
npm link @jaato/sdk
```

Re-run `npm run build` in `jaato-sdk-ts/` whenever you edit a
source file. The consumer doesn't need to reinstall.

### Option B — `file:` dependency

Better for CI or repeatable test environments. The consumer's
`package.json` declares a relative path; `npm install` copies the
built package into `node_modules`.

```json
{
  "dependencies": {
    "@jaato/sdk": "file:../jaato/jaato-sdk-ts"
  }
}
```

Run `npm install` again in the consumer to pick up SDK changes.
The SDK must be built (`npm run build`) before the consumer
installs.

### Option C — `npm pack` (most production-like)

Closest to what `npm publish` would deliver — produces a tarball
containing exactly what would be uploaded to the registry. Use
this right before publishing to catch missing files in the
package's `files` field, broken `exports`, etc.

```bash
cd jaato-sdk-ts
npm run build
npm pack
# produces jaato-sdk-0.1.0.tgz

cd ../../jaato-premium/<consumer-package>
npm install /path/to/jaato-sdk-0.1.0.tgz
```

### Option D — direct ESM import (vanilla JS, no build step)

If the consumer is a vanilla-JS webcomponent served as a static
file (no `package.json`, no bundler), it can import the built ESM
output directly:

```html
<script type="module">
  import { JaatoClient } from "/path/to/jaato-sdk-ts/dist/index.js";

  const client = new JaatoClient({
    url: "ws://localhost:8080",
    token: "<bearer-token>",
  });
  await client.connect();
  client.subscribe((event) => console.log(event));
  await client.sendMessage("hello");
</script>
```

Or copy `jaato-sdk-ts/dist/` into the consumer's static asset
path. Cost: no autocomplete or type-checking on the consumer
side — for that, introduce a build step (Vite, esbuild) and use
Option A / B / C instead.

### Recommended workflow for premium webcomponent migration

The SDK surface is feature-complete as of Phase 3.2 (commit
`1181fb7e`) — every wire verb the dashboard's webcomponents use
today (including `stageFiles` and the auto re-attach pattern)
is exposed as a typed method.  No "premium keeps it inline"
caveats remain.

1. Use **Option A** during active development — fastest iteration.
2. Switch to **Option C** right before any publish to verify the
   tarball contents are correct.
3. If a webcomponent is currently vanilla JS with no build
   pipeline, this is a good moment to introduce one (Vite or
   esbuild + single-file bundle output). That unlocks
   tree-shaking, type-checking, and pulls the SDK into the same
   module graph instead of relying on a script tag.
4. Pin **`jaato-server >= 0.5.28`** in production so the
   migrated webcomponent gets the `AgentCompletedEvent.token_usage`
   regression fix (the SDK floor itself is `0.5.27`, but the
   server pin is operationally stricter).

## Publishing

**Status: not yet on npm.** `npm install @jaato/sdk` will fail
until the first publish lands. Use one of the local-consumption
options above in the meantime.

What's still needed before the first publish:

* **A publish workflow** in `.github/workflows/publish-npm-sdk-ts.yml`
  that runs `npm run build` + `npm test` + `npm publish` on
  `workflow_dispatch`. Mirror of the existing
  `publish-testpypi-{server,sdk,tui}.yml` workflows but for npm.
* **An npm registry decision** — public npmjs.com under the
  `@jaato` scope (requires registering the org or claiming the
  scope), or GitHub Packages for now (no extra setup; consumers
  add a `.npmrc` pointing at `npm.pkg.github.com`). The two are
  switchable later.
* **An access token** in the repo's `Settings → Secrets`
  (`NPM_TOKEN` for npmjs.com, or the existing `GITHUB_TOKEN`
  for GitHub Packages) that the workflow exposes via
  `NODE_AUTH_TOKEN`.

Once those land, the publish flow is the same as
`jaato-server` / `jaato-sdk`: bump the version in
`package.json`, commit, trigger the workflow, the new version
appears on the registry. The CI staleness gate already prevents
publishes that would carry a stale `events.ts`.

Versioning policy mirrors the Python SDK: minor bumps for
additive surface (new methods, new event types), patch bumps for
fixes, major bumps for protocol-breaking changes. The `MIN_SERVER_VERSION`
constant in `src/client.ts` documents the minimum compatible
jaato-server version (currently `0.5.27`).

**Server pinning recommendation.** Although `MIN_SERVER_VERSION`
is `0.5.27` (the Phase-1 cut where the SDK's typed verbs landed),
consumers should pin **`jaato-server >= 0.5.28`** in production —
0.5.28 fixed a Phase-0 regression where `AgentCompletedEvent.token_usage`
would crash any agent calling `signal_completion` with a typed
completion payload.  The SDK floor stays at 0.5.27 because the
client itself works against either; the recommendation is
operational.

## License

BUSL-1.1 (matches jaato-server / jaato-sdk).
