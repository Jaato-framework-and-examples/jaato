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
| How MCP hosts reach it | **stdio**: the host launches `jaato-mcp --connect /tmp/jaato.sock --config mcp.yaml` | needs an HTTP listener and its auth |
| Caller identity | free: the daemon reads `SO_PEERCRED` and stamps `created_by` (see CLAUDE.md, *Two Principals on One Socket*) | must be built before the first commit (A2A §11.1) |
| What it depends on | the versioned client protocol, whose verbs already refuse an older daemon by name | `SessionManager` internals and the `_lock` re-entrancy rule (A2A §5.2) |
| A port to secure | none in stdio mode | yes |
| Cascades | the process hosts the cascade driver script (§3.3), and the script orchestrates in code or through reactor rules | depends on reactors and the fork-cid gap (A2A §7.1) |
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

### 3.1 Two kinds of tool

A tool starts one of two things, and nothing else:

| Kind | `jaato-mcp` starts | Done when |
|---|---|---|
| `session` | one session from a profile | the session completes (`AgentCompletedEvent`, or a terminal `SessionTerminatedEvent`) |
| `driver` | a **cascade driver script**, the workspace author's own code | the script returns, raises, or hits the tool's deadline |

What a driver script does inside is its own business.  It may orchestrate
**programmatically** (start stage A, wait for its payload, start B), or
**semantically** (start an entry session and let premium reactor rules fork
the next stages), or mix the two.  `jaato-mcp` cannot tell these apart and
does not need to.  An earlier draft of this sketch had two further kinds, a
manifest-declared list of stages (`kind: cascade`) and a reactor entry point
(`kind: reactor`); both were the MCP layer knowing something about topology
that belongs to the script, and both are dropped.

### 3.2 Declaring what is exposed

Something has to name what an MCP caller may invoke.  That is also the
export allowlist the A2A sketch requires (§11.2): a caller can never name a
profile or script that is not listed.  The declaration is
**`jaato-mcp`'s own format**, owned and validated by the
`jaato-mcp-transport` repository.  It adds no key to jaato's profile schema
and nothing to the `.jaato/` tree's fixed layout.

It has **one form**, a YAML file passed as `--config PATH`.  There are no
per-tool flags, so one tool and ten are declared the same way.  A
session tool's budget, plugins and limits come from its profile, so the
file does not repeat them:

```yaml
server:
  name: acme-agents
  workspace: /srv/ws              # or per tool; or `provision`
  max_concurrent_tasks: 4
defaults:
  deadline_seconds: 1800
  permission_prompts: refuse      # refuse | elicit
tools:
  review_pr:
    kind: session
    profile: reviewer               # budget_control etc. live in the profile
    description: "Reviews a diff and returns findings."  # never the persona
    task_support: optional
  ship_feature:
    kind: driver
    script: scripts/ship_feature.py
    description: "Plans, implements and reviews a feature."
    deadline_seconds: 3600
    max_concurrent: 1
    task_support: required
```

Rules for a server exposing several tools:

- **Tool names are unique**; startup refuses a duplicate.  The name is the
  MCP contract and is independent of any profile or script name.
- **Each tool is validated on its own.**  A broken entry is dropped with a
  logged reason and the rest are served, so a missing tool is never silent.
- **Every call gets its own `cascade_driver_id`**, including two concurrent
  calls to the same tool.
- **Concurrency is capped per tool and in total**, because one driver call
  may fan out many sessions and must not starve the other tools.
- **One process per workspace** is the default shape; a per-tool
  `workspace` lets one process serve several, if it can reach all of them.

Where each MCP field comes from:

| MCP field | `kind: session` | `kind: driver` |
|---|---|---|
| `name` | the config key, checked against MCP's tool-name grammar | same |
| `description` | the config, **never** the profile's `description` or persona | the config |
| `inputSchema` | the profile's `spawn_payload_schema` (every property a string, #883), or `{prompt: string}` | the script's `INPUT_SCHEMA` |
| `outputSchema` | the profile's `completion_payload_schema` | the script's `OUTPUT_SCHEMA` |
| `execution.taskSupport` | the config | the config |

The result is returned as `structuredContent`, with a text rendering beside
it for clients that ignore structured output.  For a session tool the
completion gate validates the payload before `signal_completion` succeeds,
so the `outputSchema` is held by the callee.  For a driver tool `jaato-mcp`
validates the script's return value against `OUTPUT_SCHEMA` and reports a
mismatch as `isError: true` rather than passing it on.

**Rejected: a generic tool set** (`jaato_start`, `jaato_send`,
`jaato_status`) in place of per-skill tools.  It is more flexible and
untyped, and it puts the profile name in the caller's hands, which is the
#944 / #1052 failure one layer out.

### 3.3 The driver contract

A driver script is a Python module exposing two schemas and one coroutine:

```python
INPUT_SCHEMA = {...}     # JSON Schema of the tool's arguments
OUTPUT_SCHEMA = {...}    # JSON Schema of what run() returns

async def run(input: dict, ctx) -> dict:
    # ctx.client     a connected SDK client, workspace already selected
    # ctx.cid        the cascade_driver_id jaato-mcp minted for this call
    # ctx.cancelled  set when the caller cancels or the deadline passes
    # ctx.progress(message)   -> MCP progress / task statusMessage
    ...
```

`jaato-mcp` keeps everything that is about the *call*, so a script never
has to:

| Per call, `jaato-mcp` | Why it, not the script |
|---|---|
| mints the cid and passes it in | it must cancel and observe the cascade whatever the script does |
| registers as the cid's observer (`cascade.register`) **before** `run()` starts | no event from the chain can arrive before the subscription exists |
| forwards clarifications from **any** session under the cid to the MCP caller (§4.2) | the script should not re-implement MRTR |
| refuses permission ASKs by default (§4.2) | same rule for every tool |
| enforces the deadline, and sends `cascade.cancel <cid>` on MCP cancel or deadline | a hung or crashed script must not leave sessions running |
| validates the return value against `OUTPUT_SCHEMA` | the schema is the published contract |

**The one rule a script must follow: every session it creates carries
`ctx.cid`** (`create_session(..., cascade_driver_id=ctx.cid)`).  A session
created without it escapes cancellation and clarification forwarding.  For
semantic orchestration that rule extends to the reactor forks, which is why
jaato-premium#79 (forks inherit the cid) is a prerequisite for driver
scripts that delegate to reactor rules.  Knowing when such a chain has
*finished* is the script's problem, not the MCP layer's; jaato-premium#80
(an explicit workflow terminus) is what a script would wait for.

`jaato-scaffold new cascade` already emits driver scripts; a driver for
`jaato-mcp` is that shape with the fixed signature above, and the archetype
is the obvious place to generate one.

**In-process, not a subprocess, for v1.**  The script is loaded the way
completion processors are (`spec_from_file_location`) and runs on
`jaato-mcp`'s event loop.  A subprocess would stop a crashing script taking
the server down, at the cost of carrying `ctx` across a boundary.  The
deadline plus `cascade.cancel` already bounds what a hung script's sessions
can do, so isolation is deferred.  The script runs as `jaato-mcp`'s uid with
`jaato-mcp`'s access, which is the same trust as any driver script the
workspace author runs by hand.

### 3.4 Continuing a finished task

An optional generic tool, `jaato_continue(task_id, message)`, drives a
completed session again through `session.wake`.  #913 / #915 made a
completed session drivable.  It is off by default: a caller holding a task
id can steer a session that already spent its budget, and the config
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
  can ASK, unless the config opts in per tool
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
2. **The workspace comes from the config, never the caller.**  The
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
6. **The config is validated at startup** through `scaffold.validate`
   (protocol 1.34): a profile with errors is not exposed, a driver script
   that does not load or declares no schemas is not exposed, and the
   config's own checks (tool-name collisions, a tool with no completion
   schema, an ASK-able policy without opt-in) run beside it.

---

## 6. What the package owns, and what it does not

**Owns**: the MCP server (stdio first, streamable HTTP second), the
tool configuration file and its validation, the driver-script
loader and the `ctx` it hands over, the task store and `requestState`
signing, result and resource rendering.

**Does not own**: anything in the daemon.  No framework change is required
for any phase below.  A driver script that delegates to reactor rules
needs jaato-premium#79 (forks inherit the cid) and, to know when the chain
ended, jaato-premium#80; that is a property of the script, not of
`jaato-mcp`.

**Packaging**: a separate repository,
[jaato-mcp-transport](https://github.com/Jaato-framework-and-examples/jaato-mcp-transport),
publishing a distribution `jaato-mcp` that depends on
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
| 3 | `kind: driver` tools (§3.3): cid minting, `cascade.register` before `run()`, clarification forwarding across the cid, deadline + `cascade.cancel`; `resource_link` outputs; `jaato_continue` | nothing for programmatic drivers; jaato-premium#79 / #80 for drivers that delegate to reactor rules |
| 4 | streamable HTTP with app credential + `ticket.bind` per principal | nothing (#1074 exists) |

---

## 8. Relationship to the A2A sketch

| A2A sketch | Here |
|---|---|
| §5 server inside the daemon | §2: beside it, as an SDK client, first |
| §7.1 fork cid gap blocks workflows | blocks only driver scripts that delegate to reactor rules (jaato-premium#79) |
| §2.2 `input_required` + `tasks/update` | §4.2: MRTR on a blocking call; `tasks/update` does not exist in the SDK |
| §12 "MCP is poll-only" | §1: push exists on 2025-11-25; the 2026-07-28 extension is poll-only as published |
| §13 Q2 which MCP revision the SDK speaks | answered in §1 |

---

## 9. Decisions

Recorded 2026-10-07:

1. **Placement**: a separate process beside the daemon, an SDK client
   (§2).  An in-daemon backend stays possible behind the engine interface.
2. **Cascades**: one `driver` kind (§3.3).  How the script orchestrates,
   in code or through premium reactor rules, is not the MCP layer's
   concern.  Manifest-declared stages and a `kind: reactor` entry point are
   dropped.
3. **Permission ASK**: refused at startup by default; a tool opts in with
   `permission_prompts: elicit` (§4.2).
4. **Repository**: implementation lives in
   [jaato-mcp-transport](https://github.com/Jaato-framework-and-examples/jaato-mcp-transport),
   and ships separately from any A2A work.
5. **Configuration**: owned by `jaato-mcp`, as a YAML `--config` file
   (§3.2); jaato's profile schema and `.jaato/` layout are unchanged.

Still open:

- **Engine sharing with A2A**: one package with two transports, or two
  packages sharing an engine library.
- **Driver isolation**: in-process for v1 (§3.3); whether a subprocess
  mode is worth its cost.

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
- that the SDK's cascade-event subscription (`cascade.register`) delivers
  clarification batches from every session under the cid in a form
  `jaato-mcp` can answer, without being attached to each session.

**Not measured at all**: performance.  Every claim here is structural.
