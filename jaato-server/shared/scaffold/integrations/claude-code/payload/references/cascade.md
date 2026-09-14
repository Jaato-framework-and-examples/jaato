# Cascade drivers

A cascade is a multi-stage pipeline where each stage is its own jaato session.
Start from the generator — `jaato-scaffold explain archetype cascade` states
what it writes and why, and `new cascade --dry-run` shows the exact tree.

## Driver-as-graph, not supervisor-agent

Two shapes are possible and they are not equivalent:

- **A driver expresses the graph in code** — Python control flow opens each
  stage, awaits its typed payload, and decides what runs next.
- **A supervisor agent spawns children** via the `subagent` plugin.

Prefer the driver when the pipeline is a fixed DAG. A model-driven parent
receives a child's result as **injected prose, not a typed payload**, so every
downstream stage re-parses what a schema already guaranteed. Use a supervisor
when the shape itself must be decided at runtime.

## Transport: daemon, not in-process

`InProcessClient` resolves a named profile to `model`, `provider`, `plugins`,
`plugin_configs`, `system_instructions`, `completion_payload_schema` and
`suppress_base_instructions` **only**. `completion_processors`, `max_turns`,
`spawn_payload_schema` and `budget_control` are not applied — so gates a
pipeline depends on silently do not run. Over IPC the whole contract applies.

Confirm against your installed version rather than trusting this line:
`jaato-scaffold explain clients` and `explain transports`.

## One cascade id, many sessions

Pass the same `cascade_driver_id` to every stage of one run. It makes them share
a warm runner slot, and it is the handle an observer attaches to. Mint it once
per run.

**Two tenants, one daemon:** slots are affine to a cascade and cross-cascade
reuse is forbidden by design, so a second concurrent cascade competes for pool
capacity. `JAATO_RUNNER_POOL_SIZE` is a floor on unreserved idle slots and
`JAATO_RUNNER_POOL_MAX_SIZE` a ceiling on the total; `explain env` carries both.
A run that must not contend belongs on its own `--ipc-socket`.

## Observing a run

`jaato-scaffold new observer` emits a read-only client that attaches by cascade
id and live-traces events. Between `open` and the payload returning, a
driver-sequenced cascade is blocked and has nothing to say — the daemon is
emitting the whole time.

Two things worth knowing before you build one:

- **One subscription per `(cascade_id, client_id)`.** Registration is
  one-per-pair by intent, so several `cascade_events` calls overwrite each
  other's `event_types` and only the last survives. Open ONE iterator carrying
  every type and dispatch client-side on `type(event).__name__`.
- **`event_types` are class names, not wire values.** `"session.terminated"`
  registers successfully and then yields nothing, silently, forever. Use
  `"SessionTerminatedEvent"`.
- **Attribute by `session_id`.** `agent_id` is `"main"` for every top-level
  session, so concurrent siblings are indistinguishable by it.
  `AgentCreatedEvent` carries `session_id` and `profile_name` together, so an
  observer can build its own map and needs nothing from the driver.
- The per-client event queue is bounded and lossy for high-volume traffic
  (`JAATO_IPC_EVENT_QUEUE_MAX`), so subscribe narrowly.

An observer must never be the thing that starts a daemon (`auto_start=False`),
and must never fail the run — but it should ANNOUNCE when it is off, because a
silent display is indistinguishable from a hung cascade.

## Long-lived sessions for a debate

When two or more agents argue across turns, give each its own long-lived session
and have the driver relay the opponent's last turn. Each side then keeps its own
history natively. `send_to_sibling` is fire-and-forget and is not a control-flow
primitive.

## Journaling and resume

The framework persists sessions, not pipeline position — resume is the driver's
job and is worth about fifty lines: write each stage's typed payload to a
per-run journal as soon as it exists, skip on restart whatever the journal
already holds, clear on success. Granularity is your choice; a debate journals
per turn.

Keep the journal **outside** `<workspace>/.jaato/` — that tree is the
framework's config_root.

## Host tools

Tools your driver executes, passed as `client_tools=` so the facade registers
them after connect and before session creation — a tool registered mid-session
is invisible to the model. `jaato-scaffold new host-tools` emits the recipe.

They exist only while the driver is attached: a cold session woken with no
client sees them as deferred. Wrap handlers in a deadline; a hung handler is a
hung stage. `auto_approve` whitelists them with the permission plugin so a
headless run never blocks on a prompt.

Host-tool calls DO reach a cascade observer as `ToolCallStartEvent` with
`tool_name` set, even though the handler runs in your process.

## Completion gates

`explain completion` is the contract. In short: the schema IS the
`signal_completion` tool's parameters, so payload fields are flat at the top
level; a processor's `validate` returns errors that block the completion and are
handed back to the model as retry instructions; `render` writes files. Bound the
retry loop (`max_refusals` + `on_exhausted`) or a rejecting gate can spin.

Give every schema an `errors[]` and `warnings[]` array so an agent can say "I
could not answer" instead of inventing one, and treat a non-empty `errors[]` as
a failed stage.
