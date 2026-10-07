# Exposing a jaato Session — or a Cascade — as MCP Tools

**Status**: design sketch (no implementation).  Nothing here is built.
**Companion to**: [Exposing a session or a reactor workflow as one A2A
agent](a2a-plugin.md).  That document's §2.4 splits the work into a
protocol-agnostic **task engine** and a **transport**.  This one is the
MCP transport, plus one structural choice the A2A sketch did not weigh:
running the server *beside* the daemon as an SDK client instead of
*inside* it as an extension.
**Scope**: an out-of-tree package, `jaato-mcp`, that lets an MCP client
(Claude Code, Claude Desktop, the Claude API's MCP connector, Cursor, any
agent framework speaking MCP) call a jaato session (one task) or a cascade
(an ordered set of stages run as one task) as an ordinary tool.

---

## 1. What the installed MCP SDK actually supports

The A2A sketch left this open (its §13 Q2).  It is answered here, against
`mcp` **2.2.0** as installed in this tree's dev environment, by reading
the package rather than the specification.

| Fact | Where |
|---|---|
| The newest wire is **2026-07-28** | `mcp.types.LATEST_PROTOCOL_VERSION` |
| `Tool` carries `output_schema` and `execution.task_support` (`forbidden` / `optional` / `required`) | `mcp.types.Tool`, `ToolExecution` |
| `tools/call` params carry `task`, `input_responses` and `request_state` | `CallToolRequestParams` |
| The **task methods** (`tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel`, `notifications/tasks/status`) exist in the **2025-11-25** wire types and **not** in the 2026-07-28 ones | `mcp_types/_v2025_11_25` vs `mcp_types/_v2026_07_28` |
| 2026-07-28's core mechanism for a mid-call question is **Multi Round-Trip Requests**: the server answers `tools/call` with an `InputRequiredResult` (`inputRequests` + an opaque `requestState`), and the client retries the call with `inputResponses` | `InputRequiredResult`, `InputResponses` |
| There is **no `tasks/update` method** in any wire the SDK knows | grep over `mcp_types` |
| The server SDK serves **no** task method itself.  An extension may add one: `mcp.server.extension.MethodBinding` ("a new request method an extension serves, e.g. `tasks/get`") under `Extension.identifier` | `mcp/server/extension.py` |
| A `MethodBinding` may not name a core method, and `tasks/*` is not one at 2026-07-28 | `SPEC_CLIENT_METHODS` |

Three consequences, and the first two correct [a2a-plugin.md](a2a-plugin.md) §2.2:

1. **Tasks are an extension at 2026-07-28, not a core feature.**  A
   2025-11-25 client may use core tasks.  A 2026-07-28 client gets them
   only if both sides negotiate `io.modelcontextprotocol/tasks`.  Nothing
   in the installed SDK implements that extension; the plugin would.
2. **Clarification does not need tasks at all.**  `input_required` +
   `tasks/update` is not the mechanism.  On 2026-07-28 the mechanism is
   MRTR, and it works on a plain blocking `tools/call` (§4.2).  That makes
   it the first thing to ship rather than the third.
3. **Push exists on the older wire** (`notifications/tasks/status`).
   "MCP is poll-only" is true of the 2026-07-28 extension as published and
   false of 2025-11-25.  Which one a given caller speaks is a fact about
   the caller.

What the SDK does **not** tell us is what the clients do.  Whether Claude
Code or Claude Desktop request task augmentation, honour `InputRequiredResult`,
or time out a long blocking call is not answerable from this tree (§10).

---

## 2. Where the server runs: beside the daemon, not inside it

[a2a-plugin.md](a2a-plugin.md) §5 puts the server half in the daemon as a
`jaato.extensions` entry point, subscribing through
`register_in_process_client`.  For MCP the other placement is better, at
least first: **`jaato-mcp` is a separate process that is an MCP server on
one side and an ordinary `IPCClient` (or `WSClient`) on the other.**

```
 Claude Code ─stdio─▶ jaato-mcp ─IPC socket─▶ jaato daemon ─▶ runners
                      (MCP server +            (unchanged)
                       SDK client)
```

| | Separate process | Daemon extension (A2A sketch's shape) |
|---|---|---|
| How MCP hosts reach it | **stdio**: the host launches `jaato-mcp --connect /tmp/jaato.sock --expose .jaato/mcp-expose.yaml` | needs an HTTP listener and its auth |
| Caller identity | free: the daemon reads `SO_PEERCRED` and stamps `created_by` (see CLAUDE.md, *Two Principals on One Socket*) | must be built before the first commit (A2A §11.1) |
| What it depends on | the versioned client protocol, whose verbs already refuse an older daemon by name | `SessionManager` internals and the `_lock` re-entrancy rule (A2A §5.2) |
| A port to secure | none in stdio mode | yes |
| Cascades | the process **is** the cascade driver, so the fork-cid gap (A2A §7.1) does not arise | depends on reactors and that gap |
| Crash isolation | a crash ends one MCP connection | a crash is in the daemon |

The engine is still written behind one interface (A2A §2.4), so an
in-daemon backend can be added later.  It is the right home for a
multi-tenant HTTP deployment, and the wrong one for the first version.

### 2.1 HTTP without moving into the daemon

A streamable-HTTP `jaato-mcp` can still stay outside the daemon.  It holds
one entry in `--ws-app-credentials` (#1074), authenticates each MCP
principal itself (the MCP authorization spec, or a bearer token for a
private deployment), and calls `ticket.bind` per principal.  Each MCP
caller then reaches the daemon as its own `app:user`, with attribution,
workspace ownership (#1113) and the per-application workspace root (#1496)
working unchanged.  The daemon learns nothing about MCP.

---

## 3. What the caller sees

### 3.1 One tool per exposed skill

An explicit manifest decides what is callable.  It is also the export
allowlist the A2A sketch requires (§11.2): a caller can never name a
profile the manifest does not list.

```yaml
# .jaato/mcp-expose.yaml
server:
  name: acme-agents
  workspace: provision            # provision | <absolute path>
  max_concurrent_tasks: 4
tools:
  review_pr:
    kind: session
    profile: reviewer
    description: "Reviews a diff and returns findings."  # never the persona
    input: spawn_schema           # spawn_payload_schema, or {prompt}
    budget_control:
      limits: {usd: 2.0}
      degrade: [{at: 100, action: abort}]
    task_support: optional
  ship_feature:
    kind: cascade
    stages:
      - {name: plan,      profile: planner}
      - {name: implement, profile: implementer, input_from: plan}
      - {name: review,    profile: reviewer,    input_from: implement}
    result: last                  # last | all
    task_support: required
```

| MCP field | Comes from |
|---|---|
| `name` | the manifest key, checked against MCP's tool-name grammar |
| `description` | the manifest, **never** the profile's `description` or persona text, which are internal authoring artifacts |
| `inputSchema` | the profile's `spawn_payload_schema` (every property a string, #883), or `{prompt: string}` |
| `outputSchema` | the profile's `completion_payload_schema` (for `result: all`, an object keyed by stage name) |
| `execution.taskSupport` | the manifest |

The result is returned as `structuredContent`, with a text rendering
beside it for clients that ignore structured output.  The completion gate
validates the payload before `signal_completion` succeeds, so a tool with
an `outputSchema` is held to it by the callee, not only checked by the
caller.

**Rejected: a generic tool set** (`jaato_start`, `jaato_send`,
`jaato_status`) in place of per-skill tools.  It is more flexible and
untyped, and it puts the profile name in the caller's hands, which is the
#944 / #1052 failure one layer out.

### 3.2 A cascade is one tool and one task

A `kind: cascade` tool runs its stages in order, each a headless session
under one `cascade_driver_id` minted per call, with the completion payload
of stage *n* handed to stage *n+1* (`input_from`) through that stage's
spawn schema.  The caller sees one call and one result.  Stage progress is
reported as progress notifications and, where a task is in play, as the
task's `statusMessage` (`stage 2/3: implement`).

The manifest's stage list is deliberately **linear with named inputs**:
data, not a driver program.  Branching and fan-in are reactor territory
(A2A §6.2, §7.3).  A later `kind: reactor` tool type can point at a rule
set once A2A §7.1's fork-cid propagation and §7.2's terminal rule exist.

### 3.3 Continuing a finished task

An optional generic tool, `jaato_continue(task_id, message)`, drives a
completed session again through `session.wake`.  #913 / #915 made a
completed session drivable.  It is off by default: a caller holding a task
id can steer a session that already spent its budget, and the manifest
should say so explicitly (`allow_continue: true` per tool).

---

## 4. Mapping the lifecycle

### 4.1 The happy path

| Step | jaato client call | MCP |
|---|---|---|
| tool call arrives | `create_session(profile=..., cascade_driver_id=...)`, then `send_message` with the wrapped input | `tools/call` |
| work proceeds | `AgentStatusChangedEvent`, `TurnProgressEvent`, `ToolOutputEvent` | progress notifications |
| completion | `AgentCompletedEvent.payload` | `CallToolResult.structuredContent` |
| budget stop | `SessionTerminatedEvent(reason="budget_exhausted")` | `isError: true`, naming the ceiling |
| caller cancels | `stop_session` / `cascade.cancel <cid>` | `notifications/cancelled`, or `tasks/cancel` |
| caller disconnects | nothing: `runtime_limits.max_orphan_seconds` (#812) reaps it | — |

`SessionTerminatedEvent.reason` maps as in the A2A sketch §4:
`natural` → completed, `budget_exhausted` / `error` → failed,
`stopped` / `client_request` / `cascade_cancelled` → cancelled.  The
`Session.terminus` the SDK facade exposes (#1007) is exactly this
information.

### 4.2 A clarification is a round trip, not a task state

`request_clarification` maps onto MRTR on a plain blocking call:

1. The session emits `ClarificationBatchEvent`.
2. `jaato-mcp` answers the pending `tools/call` with an
   `InputRequiredResult`: one elicitation `inputRequest` per question
   (choices become a JSON-Schema `enum`, free text a `string`) and a
   `requestState` that names the session and the batch's `request_id`.
   The session stays blocked on the clarification, as it does for any
   client.
3. The client retries `tools/call` with `inputResponses` and the same
   `requestState`.
4. `jaato-mcp` calls `respond_to_clarification_batch`, and the call blocks
   again until the next event that ends or interrupts it.

`requestState` must be **opaque and authenticated** (an HMAC over
session id, request id and caller principal, keyed per process).  It is a
handle to a live session, and a forged one would let a caller answer
another caller's question.  On a 2025-11-25 client the same questions go
out as elicitation requests carrying related-task metadata instead.

Two limits, stated rather than discovered:

- **No file answers.**  Elicitation forms carry no file type, so a
  clarification that wants an attachment (#989) cannot be answered this
  way.  The tool description says so.
- **A permission ASK is not a clarification.**  A remote model answering
  "may I run this?" on the user's behalf is a rubber stamp.  `jaato-mcp`
  refuses at startup to expose a profile whose effective permission policy
  can ASK, unless the manifest opts in per tool
  (`permission_prompts: elicit`) and the client advertises elicitation.
  The effective policy is known daemon-side (#1474's layers);
  `scaffold.validate` is where that check would run.

### 4.3 Long calls and tasks

A blocking call is the v1 shape.  Whether it is good enough depends on the
host's tool timeout, which this tree cannot measure (§10).  The tasks
layer comes second, through the SDK's extension seam:

| Method | Backed by |
|---|---|
| `tools/call` with `task` | create the session, return the task handle at once |
| `tasks/get` | the task store: id → session or cid, state, last status message |
| `tasks/result` | blocks until terminal, then the same result as a blocking call |
| `tasks/cancel` | `stop_session` / `cascade.cancel` |
| `tasks/list` | the caller's tasks only, filtered by principal |
| `notifications/tasks/status` | on the 2025-11-25 wire, pushed from the same event stream |

The task store survives a `jaato-mcp` restart because the daemon's records
already carry what it needs: a session's `ended_at` / `end_reason`
(protocol 1.29) is the terminal state, and the store only has to persist
the task id → session id (or cid) map.

### 4.4 Files the agent produced

A file the agent wrote is returned as a `resource_link` and served by
`resources/read`, never inlined.  Over WS the bytes come from
`workspace.file.fetch` (protocol 1.20), which already refuses `.env`,
stored credentials and anything that leaves the workspace.  Over IPC there
is no such verb: the Python SDK has none, because an IPC client sits on the
daemon's host.  `jaato-mcp` would then read the file itself, as its own
uid, and must apply the same three refusals.  Copying them is the risk;
whether the Python SDK should gain the verb is a question for §10.

---

## 5. Security posture

Most of A2A §11 carries over.  What changes or is added:

1. **Stdio is the safe default.**  The process speaks for the user who
   launched it and nobody else.  HTTP mode is opt-in and refuses to start
   without auth configured (§2.1).
2. **The workspace comes from the manifest, never the caller.**  The
   default, `provision`, gives each task a workspace of its own, so one
   caller's task cannot read another's.  A fixed path is for a single-user
   stdio deployment.
3. **Caller input is untrusted.**  It is wrapped the way a wake payload is
   (`_wrap_wake_content`, #845), so text arriving from a remote model reads
   as data to the model that receives it.
4. **Concurrency is capped** per tool and per server, on top of each
   task's `budget_control`.  An MCP host retrying a tool in a loop is a
   fan-out nobody planned.
5. **A tool's own description is the only text published.**  Profile
   names, descriptions and personas stay internal (A2A §8.3).
6. **The manifest is validated at startup** through `scaffold.validate`
   (protocol 1.34): a profile with errors is not exposed, and the
   manifest's own checks (tool-name collisions, a tool with no completion
   schema, an ASK-able policy without opt-in) run beside it.

---

## 6. What the package owns, and what it does not

**Owns**: the MCP server (stdio first, streamable HTTP second), the
manifest and its validation, the task store and `requestState` signing,
the cascade runner for `kind: cascade`, result and resource rendering.

**Does not own**: anything in the daemon.  No framework change is required
for phases 0–3 below.  Phase 5 needs A2A §9's premium change (fork cid
propagation).

**Packaging**: a separate distribution, `jaato-mcp`, depending on
`jaato-sdk` and `mcp`.  It declares no `jaato.plugins` entry point, so the
`PLUGIN_TIER` / `PLUGIN_KIND` trap (#917) does not apply.  If it ever
contributes a plugin, that rule does.

---

## 7. Phasing

| Phase | Deliverable | Needs |
|---|---|---|
| 0 | stdio, `kind: session` tools, blocking `tools/call`, `structuredContent` | nothing |
| 1 | MRTR clarification ↔ `ClarificationBatchEvent`; startup refusal of ASK-able profiles | nothing |
| 2 | tasks through the extension seam; restart-safe task store; `notifications/tasks/status` where the wire has it | nothing |
| 3 | `kind: cascade` (linear stages, `input_from`); `resource_link` outputs; `jaato_continue` | nothing |
| 4 | streamable HTTP with app credential + `ticket.bind` per principal | nothing (#1074 exists) |
| 5 | `kind: reactor` tools | A2A §7.1 and §7.2 (premium) |

---

## 8. Relationship to the A2A sketch

| A2A sketch | Here |
|---|---|
| §5 server inside the daemon | §2: beside it, as an SDK client, first |
| §7.1 fork cid gap blocks workflows | does not arise for `kind: cascade`; still blocks `kind: reactor` |
| §2.2 `input_required` + `tasks/update` | §4.2: MRTR on a blocking call; `tasks/update` does not exist in the SDK |
| §12 "MCP is poll-only" | §1: push exists on 2025-11-25; the 2026-07-28 extension is poll-only as published |
| §13 Q2 which MCP revision the SDK speaks | answered in §1 |

---

## 9. Decisions

1. **Placement**: beside the daemon first (recommended) or inside it.
   Decides whether A2A §7.1 blocks the cascade case.
2. **Cascade definition**: manifest stages (§3.2) first, reactor rule sets
   later, or reactors only.
3. **Permission ASK**: refuse at startup (recommended) or bridge to
   elicitation by default.
4. **Engine sharing with A2A**: one package with two transports, or two
   packages sharing an engine library.

---

## 10. What was verified, and what was not

**Read and verified in the installed `mcp` 2.2.0**: everything in §1's
table, including the absence of `tasks/update` and of any built-in task
handler.

**Read and verified in this tree**: the client verbs §4 relies on exist on
`IPCClient` (`create_session` with `cascade_driver_id`, `send_message`,
`stop_session`, `wake_session`, `respond_to_clarification_batch`,
`respond_to_permission`, `list_profiles`); `cascade.cancel` is a daemon
command, not an SDK method; the Python SDK has no `workspace.file.fetch`.

**Not verified**:

- what any MCP **host** does: whether Claude Code / Desktop send task
  augmentation, honour `InputRequiredResult`, or time out a long blocking
  `tools/call`.  This decides how much of phase 2 is needed;
- that the SDK's `MCPServer` lets a tool handler return an
  `InputRequiredResult` directly, or whether that needs the lowlevel
  server;
- the `io.modelcontextprotocol/tasks` extension's own method set at
  2026-07-28 (not shipped in the SDK, so not readable here);
- how a cascade-driving SDK client observes stage events across N sessions
  under one cid.  The daemon side is `register_in_process_client`; the
  client-side subscription path was not traced.

**Not measured at all**: performance.  Every claim here is structural.
