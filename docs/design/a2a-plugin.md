# Exposing a jaato Session — or a Whole Reactor Workflow — as One A2A Agent

**Status**: design sketch (no implementation).  Nothing here is built.
**Scope**: whether A2A support can live entirely OUT of tree, what it
would own, and the three gaps that stand between "a chain of sessions
ran" and "one A2A task completed".

---

## 1. The question

[A2A](https://a2a-protocol.org/) (Agent2Agent, v1.0) is the wire an
agent uses to hand work to *another* agent it does not host.  The ask:
can an out-of-tree package expose **any jaato session — and a whole
cascade, as a single entity** — as an A2A agent that other frameworks
call as a tool?

**Yes, for a single session, with no framework change at all.**  For a
multi-stage workflow the answer is yes *and* it needs three small
things that do not exist today, two of them premium-side.  Sections 6
and 7 are those things.

The finding that shapes everything below: the primitive an A2A server
needs most — **a client identity that is not a socket** — already
exists and is already wired.  Nothing else in this document is as
load-bearing as that.

---

## 2. What A2A actually requires

Pinned to spec **v1.0.0**.  The canonical spec names its RPCs in
gRPC style (`SendMessage`, `SendStreamingMessage`, `GetTask`,
`ListTasks`, `CancelTask`, `SubscribeToTask`, the four
`*TaskPushNotificationConfig` verbs, `GetExtendedAgentCard`); the
JSON-RPC binding spells the same calls as method strings
(`message/send`, `tasks/get`, …).

> **Do not hardcode the method strings from this document.**  They were
> not confirmed against the v1.0 binding table while it was written, and
> v1.0 renamed parts of the object model (a file Part now carries `raw`
> / `url`, not `bytes` / `uri`).  Take both from the `a2a-sdk` version
> you build against, and pin it.

A server owes: an **Agent Card** (identity + `skills[]`), a **Task**
with a lifecycle, **Artifacts**, **Parts** (`text` / `raw` / `url` /
`data`), SSE streaming, and optional push notifications.  Task states:
`SUBMITTED`, `WORKING`, `INPUT_REQUIRED`, `AUTH_REQUIRED`,
`COMPLETED`, `FAILED`, `CANCELED`, `REJECTED`.

---

## 3. The mapping

The reason this is worth doing: almost every A2A concept already has a
jaato primitive with the same shape, and several of jaato's are
*stronger* than what a typical A2A server can offer.

| A2A | jaato | Note |
|---|---|---|
| Agent Card `skills[]` | profiles, or a reactor rule-set | §5 — the two engines |
| skill input schema | `spawn_payload_schema` | already string-typed for a wire (#883) |
| `Task` | a session, or a `cascade_driver_id` | §5 |
| `contextId` | `cascade_driver_id` | already a first-class tenant id |
| `WORKING` | `AgentStatusChangedEvent`, `TurnProgressEvent` | |
| `INPUT_REQUIRED` | `ClarificationBatchEvent` | `batch_only=True` on runner sessions; answered with one `ClarificationBatchResponseEvent` |
| input-required carrying a file | `answer_attachments` (#989) | media round-trips in both directions |
| `COMPLETED` + `Artifact` | `AgentCompletedEvent.payload` | **schema-validated** via `completion_payload_schema` — stronger than most A2A servers offer |
| `FAILED` / `CANCELED` | `SessionTerminatedEvent.reason` (#1007) | `budget_exhausted` → FAILED; `stopped` → CANCELED |
| `CancelTask` | `cancel_cascade(cid)` / `stop_session(id)` | `session_manager.py:4350` / `:4713` |
| `TaskArtifactUpdateEvent` | `ToolOutputEvent` incl. `mime_type` / `data_b64` | #824 / #830 |
| follow-up turn on a live task | `session.wake` / `inject_prompt` **with attachments** (#845) | #913/#915 made a completed session drivable again |
| push notification | `ctx.post_webhook` (premium) or the plugin's own | |
| per-task spend ceiling | `budget_control` | an A2A task with a dollar cap, free |
| abandoned-task reaping | `runtime_limits.max_orphan_seconds` (#812) | written for exactly "the client went away" |
| caller identity | `created_by` / `set_client_user()` (#859) | an A2A principal lands on the audit trail |

Two rows deserve emphasis.  `completion_payload_schema` means a jaato
A2A agent can advertise a **typed** artifact and be held to it by its
own completion gate — the artifact is validated before it is claimed,
not after.  And `budget_control` means an A2A task can carry a ceiling
its caller never has to trust the callee to respect.

---

## 4. The extension surface that already exists

`docs/design/daemon-extensions.md` names five hooks.  Three matter
here, one is unusable, and one is a gift.

| # | Hook | Verdict for A2A |
|---|---|---|
| 1 | `jaato.extensions` entry point → `create_extension(ctx)` with `session_manager`, `ws_server`, `plugin_registry` | **the home for the server half** |
| 2 | `ws_server.set_connection_interceptor(check, handler)` | **unusable** — fires post-upgrade on a `websockets` connection (`websocket.py:1252`, served by `websockets.serve` at `:1132`). A2A is HTTP JSON-RPC + SSE, which needs `process_request`. Run an own HTTP server instead |
| 3 | `session_manager.add_session_hook(hook)` | per-session wiring, if the plugin needs any |
| 4 | `env_plugin.register_aspect(name, handler)` | lets an agent introspect its own A2A exposure |
| 5 | `subagent_plugin.register_remote_handler(handler)` | **the gift** — see §9 |

### 4.1 The primitive that makes this cheap

```python
session_manager.register_in_process_client(
    client_id="_a2a:<taskId>",
    callback=on_event,              # sync, inside _emit_to_session
    cascade_driver_id=<taskId>,
    role="owner",                   # or "observer"
    event_types={"AgentCompletedEvent", "SessionTerminatedEvent", ...},
)
```

`_emit_to_session` (`session_manager.py:5502`) is a single fan-out
chokepoint: it stamps `session_id` on the event and dispatches to every
cascade-client registered for that session's cid.  So an in-process
consumer receives **every session-scoped event for every session
stamped with a cid** — no socket, no synthetic transport, no polling.

Together with `create_session(...)` (`:6458`, taking `profile_name`,
`agent_name`, `agent_params`, `cascade_driver_id`, `budget_control`,
`workspace_path`, `config_root`), `create_headless_session(...)`
(`:7888`) and `handle_request(client_id, session_id, event)` (`:11919`),
that is a complete task engine reachable from an entry point.

### 4.2 The one re-entrancy rule

The callback runs **synchronously inside `_emit_to_session` while
`SessionManager._lock` is held**.  It must not call back into any
`SessionManager` method that takes `_lock`.  Premium's own handler
(`jaato_premium/reactors/extension.py`) carries this contract in a
docstring headed *"load-bearing — read before extending"*.

Consequence for the SSE bridge: `queue.put_nowait` and return.  And the
queue must be **bounded with a per-class policy** — lossy for
artifact/output chunks, essential for lifecycle events — or a slow SSE
consumer grows daemon memory without limit.  That is
`JAATO_IPC_EVENT_QUEUE_MAX` exactly; the policy is already written down
and should be copied rather than re-derived.

---

## 5. Two engines, and the free/premium line falls out of them

An A2A **skill** can be backed by either of two things jaato already
has, and they are not the same product.

### 5.1 Session-scoped — public tree only

One skill = one **profile**.  One task = one **session**.  The card's
skill list is the profile list (`SessionProfilesEvent`; #1052 already
computes `_available_profile_names` as a wire enum, so the valid set is
a solved problem).  The artifact is that session's
`AgentCompletedEvent.payload`.

Everything this needs is public.  This is the whole Phase 0.

### 5.2 Workflow-scoped — premium reactors

One skill = a **reactor rule-set**.  One task = a `cascade_driver_id`
spanning N sessions.

This is the "cascade as a single entity" case, and reactors are what
make it tractable: a jaato cascade written as an SDK driver is *code*
(`jaato-scaffold new cascade` emits a WORKLIST of `(profile, agent,
prompt)` stages), which cannot be exposed declaratively.  A reactor
chain is **data**:

```json
{"version": 1, "rules": [
  {"id": "implementer-to-reviewer",
   "match": {"event_type": "agent.completed",
             "where": "agent_id == 'implementer' && success == `true`"},
   "action": {"script": "reactors/handoff.py",
              "params": {"target_agent": "reviewer"}}}
]}
```

`match{event_type, where: <JMESPath>}` → `action{script, params}` with
`${event.*}` / `${env.*}` substitution, resolved from `~/.jaato/`
(hot-reloaded every 2s) and `<workspace>/.jaato/` (workspace overrides
by rule id).  The `ActionContext` a script receives
(`jaato_premium/reactors/action_context.py`) already offers:

| `ctx` | Line | A2A relevance |
|---|---|---|
| `fork_from_originating(target_agent, budget="clone")` | 531 | the handoff, inheriting the budget ceiling |
| `create_session(..., cascade_driver_id=...)` | 326 | **takes a cid, and registers the reactor as cascade owner** |
| `gate(name, ttl_seconds, public_intent_fields, tenant_id)` | 114 | lease / announce / release with a TTL watchdog |
| `post_webhook(url, body, headers)` | 982 | A2A push notifications, nearly free |
| `emit_event(type, payload)` | 1029 | rule chaining |
| `run_shell(cmd, cwd, timeout)` | 1005 | — and see §10 |
| `external_event` as a matchable event type | — | an inbound trigger already exists |

So the split is not a licensing invention; it is where the capability
actually sits:

- **free** — A2A over a profile.  One skill, one session, one artifact.
- **premium** — A2A over a reactor workflow.  One skill, one cid, N
  sessions, one artifact.

---

## 6. What reactors do *not* give, and it is the interesting part

Reactors make the **topology** declarative.  They do not give the chain
an **identity** or a **terminus** — and A2A needs both.

### 6.1 The cid does not survive a handoff

`fork_from_originating` → `_spawn_with_history` →
`create_headless_session(profile_name, agent_name, workspace_path,
initial_history, initial_session_state, session_name, budget_control)`
— `action_context.py:872`.  **No `cascade_driver_id`.**

`create_headless_session` accepts one (`session_manager.py:7900`).  The
fork simply never passes it, and nothing propagates the originating
session's.  So a declarative chain fragments into N sessions with no
shared tenant id, and there is nothing to subscribe to.

This is the highest-leverage fix in this document and it is roughly one
argument.  It pays off well past A2A: the observability gap that
`cascade-as-client.md` was filed for (Finding A) is only half-closed
while forks drop the cid.

**Policy decision required.**  Does a fork inherit unconditionally, or
opt in — `fork_from_originating(..., cascade="inherit"|"new"|None)`?
Inheriting by default matches how `budget="clone"` already behaves and
matches the plain reading that a handoff belongs to the same run.
Opt-in is safer against a tenant whose existing rules fan out into
genuinely independent work and who would not expect them tenanted
together.  **Recommendation: inherit by default**, with an explicit
`"new"` for deliberate detachment — a chain that shares a budget
ceiling and a warm slot already behaves as one run in every respect but
this one.

### 6.2 Nothing says the workflow is over

A reactor chain ends because the last `agent.completed` matched no
rule.  From outside, that is **indistinguishable from a rule that
failed to fire** — the same ambiguity class as #955's *"a ladder that
never logs is indistinguishable from one that is not wired"*.  A2A
needs a definite `COMPLETED` carrying an artifact, and a definite
`FAILED`.

The fix should not invent a workflow DSL.  It should be **one more
reactor rule**:

```json
{"id": "a2a-respond",
 "match": {"event_type": "agent.completed",
           "where": "agent_id == 'deployer' && success == `true`"},
 "action": {"script": "reactors/a2a_complete.py",
            "params": {"artifact_name": "deployment-report"}}}
```

The A2A package ships `a2a_complete.py` as a normal action script; the
tenant wires it like any other rule.  The terminal condition is then
expressed in the same JMESPath vocabulary as every other edge in the
graph, a failure branch is a second rule with ``success == `false` ``
→ FAILED, and the package adds **no new configuration surface at all**.

### 6.3 Gates are per-child completion, not a barrier

Worth stating because it is tempting to reach for gates as the join.
`HandoffGate` is a **binary RED/GREEN latch with a single lease**
(`gates/gate.py`), auto-released when the spawned session's agent
completes (`GateAutoCompleter._on_agent_completed` →
`registry.release_for_session`, `gates/auto_completer.py:33`), emitting
a client-visible `GateReleasedEvent`.

That is "this one child finished", not "all three finished".  A genuine
fan-in needs N gates plus a counting action script, and JMESPath in a
`where` clause cannot aggregate over external state.  **Fan-out/fan-in
workflows are out of scope for a first version**; linear chains and
conditional branches are in.

---

## 7. Proposed architecture

```
  message/send ──▶ mint taskId  (== cascade_driver_id)
                   create_headless_session(cid, entry_profile)      ENTRY
                   inject the A2A message parts
                        │
                        ▼
               reactors.json drives the chain          ← TENANT DATA
               (every fork inherits the cid — §6.1)
                        │
                        ▼
               terminal rule → reactors/a2a_complete.py            EXIT
                   artifact + COMPLETED / FAILED

  register_in_process_client(cid, "owner") ──▶ bounded queue ──▶ SSE
                                                             └─▶ tasks/get
```

### 7.1 What the plugin owns

- HTTP/JSON-RPC + SSE server, on its own port.  The in-tree `webhook`
  plugin (`shared/plugins/webhook/http_server.py`) is the precedent and
  already carries TLS/mTLS, CIDR allowlists, rate limiting and replay
  refusal (#713) — read it before writing a second one.
- Agent Card generation from an explicit manifest (§7.3).
- Task store: id minting, state, artifacts, history.  Durable enough to
  answer `GetTask` after a daemon restart — session records persist, so
  `ListTasks` can lean on `session.list` / `list_orphan_sessions`.
- The entry and exit seams, and the bounded event queue.
- `a2a_complete.py`, shipped as an action script.

### 7.2 What the plugin does NOT own

The workflow.  Everything between entry and exit is `reactors.json` —
tenant data, hot-reloadable, with no plugin release in the loop.

### 7.3 The card needs its own manifest

With reactors in play a skill is a rule-set, not a profile, so the card
cannot be generated by iterating profiles — and it should not be
anyway.  Profile names, descriptions and personas are internal
authoring artifacts; publishing them to the open internet is a leak in
the same family as an over-broad Agent Card.

`.jaato/a2a.json`: which skills are exposed, each one's entry profile
or rule-set, input/output modes, examples, and the card's identity
block.  It doubles as the **export allowlist**, which §10 requires
regardless.

---

## 8. Changes outside the plugin

The point of this document is how little there is.

| # | Change | Where | Size |
|---|---|---|---|
| 1 | propagate the cid through `_spawn_with_history` | **premium** | ~1 argument + a policy decision (§6.1) |
| 2 | `add_event_observer()` beside `set_event_callback` | public | small, and optional — see below |

**On #2.**  `_emit_to_client` (`session_manager.py:4068`) dispatches
through a single `self._event_callback` owned by the transport, so
client-scoped replies (a `session.new` answer, `ErrorEvent(SessionError)`,
a profile listing) do not reach an extension.  Session-scoped events do,
via the cascade-client route, and that is the overwhelming majority of
what A2A needs.  `create_session()` also returns the id synchronously,
so the entry path does not depend on an event.  **So this is a
nice-to-have, not a blocker** — worth filing, not worth waiting for.

Everything else — creating sessions, driving turns, subscribing to a
cid, cancelling, reaping — is already public API.

---

## 9. The outbound direction is nearly free

The mirror — a jaato agent *calling* a remote A2A agent — may need no
new tool at all.  Extension point 5,
`subagent_plugin.register_remote_handler(handler)`, exists to make
`spawn_subagent(server=...)` delegate to a peer; premium uses it for
gossip clustering.  An A2A handler registered there means **every
jaato agent can already delegate to a remote A2A agent using a tool it
already has**: `server="a2a://peer/agent"`, task → `SendMessage`,
subscribe or poll, return the artifact as the subagent result.

One documented caveat carries over verbatim: `spawn_subagent`'s
`profile` enum is built from *local* profiles, so under
`api_params.strict_tools: true` a spawn naming a peer-only profile is
schema-invalid.  The existing workaround — declare a local stub profile
of that name — applies unchanged.

If a first-class tool is wanted instead, it is an ordinary out-of-tree
`ToolPlugin` via `jaato.plugins`.  **It must declare `PLUGIN_TIER` and
`PLUGIN_KIND`** or it is discovered, listed by `jaato-scaffold
plugins`, and silently absent from every session (#917).  Note that the
in-tree reference, `out-of-tree-plugins/moon-phase`, declares neither —
the example currently sets the trap.

---

## 10. Security posture

This listens on a port and hands arbitrary remote callers a turn in an
agent that may hold `cli`, `file_edit` and — through a reactor action
script — `ctx.run_shell`.  Four things are not optional.

1. **Bearer auth from the first commit.**  Mirror `--ws-token-file`,
   never `--ws-token`: a token in argv is served by
   `/proc/<pid>/cmdline` (#712).  Store the digest, compare with
   `hmac.compare_digest`.  A2A `securitySchemes` advertises it; the
   authenticated principal should land on `created_by` (#859) so
   approvals and the ledger name who asked.
2. **A skill allowlist, not a profile lookup.**  A caller must never be
   able to name an arbitrary profile.  This is #944 and #1052's lesson
   one layer out: the valid set must be the thing the model — or here,
   the peer — reads *before* it chooses, and the runtime refusal stays
   as the only layer that actually refuses.
3. **The inbound message is untrusted.**  `_wrap_wake_content` (#845)
   is the exact precedent: a wake payload arriving from a webhook or a
   cron is already named rather than inlined, and an attachment is
   described (mime, display name, ingest id) rather than trusted.  An
   A2A message is the same animal and should reuse it rather than grow
   a second boundary.
4. **Size caps are real.**  10 MB runner frame (#920), ~6 MiB per
   clarification submission (#989).  A `FilePart` above some threshold
   must be carried as `url`, not `raw`.

---

## 11. Phasing

| Phase | Deliverable | Needs |
|---|---|---|
| 0 | Card + blocking `SendMessage` + `GetTask`, one allowlisted profile | nothing — public tree only |
| 1 | SSE streaming off the same callback, bounded queue | nothing |
| 2 | `INPUT_REQUIRED` ↔ `ClarificationBatchEvent`, attachments both ways | nothing |
| 3 | Workflow-scoped tasks: cid propagation + `a2a_complete.py` | §8 #1 (premium) |
| 4 | Push notifications; outbound via `register_remote_handler` | nothing |

Phase 0 is small: create a session, drive one turn, await
`AgentCompletedEvent` on the cid, return `payload` as the artifact.

---

## 12. Open questions

1. **Fork cid inheritance** — default-on or opt-in (§6.1).  Blocks Phase 3.
2. **`emit_event` scope.**  It publishes to `self.server.event_bus`, the
   *originating session's* bus.  Whether a rule on session B can match
   an event a script emitted from session A decides whether cross-session
   rule chaining works at all, or whether every hop must be a fork.
   **Not verified** — one method was read, not the bus.
3. **Task granularity when both engines are present.**  If a tenant
   exposes a profile-backed skill and a workflow-backed one, is
   `contextId` ever distinct from `taskId`?  A2A's model wants
   `contextId` to group a *conversation* across tasks, which suggests
   cid = `contextId` and a separate `taskId` per invocation — at odds
   with §7's cid = `taskId`.  Worth settling before the task store is
   written, because it is the schema.
4. **Fan-in.**  Declared out of scope in §6.3.  If it comes back,
   the question is whether a counting barrier belongs in the gate
   registry or in an action script.
5. **Where the package lives.**  A free core with a premium reactor
   adapter is two distributions; the alternative is one distribution
   that degrades when premium is absent.  `codebase-split-licensing.md`
   presumably already answers this.

---

## 13. What was verified, and what was not

Because a design doc that does not say this invites its own drift.

**Read and verified in-tree**: `register_in_process_client` and the
`_emit_to_session` dispatch; `create_session` / `_create_session_impl` /
`create_headless_session` signatures; `handle_request`;
`_emit_to_client`'s single-callback dispatch; the WS interceptor's
post-upgrade placement; the five extension points; `reactors.json`'s
schema and the `ActionContext` methods cited; the **absence** of
`cascade_driver_id` in `_spawn_with_history`'s call; gate release and
auto-completion semantics; premium's cascade handler being
observability-only.

**Not verified**: the A2A v1.0 JSON-RPC method-name binding and the
well-known Agent Card path (§2 — take both from `a2a-sdk`); the event
bus's cross-session reach (§12.2); anything about how the `a2a-sdk`
Python package structures a server, which may change the shape of §7.1
considerably.

**Not measured at all**: performance.  Every claim here is structural.
