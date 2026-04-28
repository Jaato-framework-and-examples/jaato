# Reactor Engine — Tenant Setup and Client Integration Guide

> Scope: What a client application (tenant) using the jaato SDKs needs to know about the reactor engine — how it works, how to set it up, and whether the tenant needs to worry about anything.

## Table of Contents

1. [What Is the Reactor? (Tenant Perspective)](#1-what-is-the-reactor-tenant-perspective)
2. [Do Tenants Need to Do Anything?](#2-do-tenants-need-to-do-anything)
3. [How the Reactor Works — Event Flow](#3-how-the-reactor-works--event-flow)
4. [What the Reactor Can Do](#4-what-the-reactor-can-do)
5. [Setting Up Reactor Rules — `reactors.json`](#5-setting-up-reactor-rules--reactorsjson)
6. [Writing Action Scripts](#6-writing-action-scripts)
7. [Completion Payload Schemas — Typed Handoffs](#7-completion-payload-schemas--typed-handoffs)
8. [Common Handoff Patterns](#8-common-handoff-patterns)
9. [The SDK's Relationship to the Reactor](#9-the-sdks-relationship-to-the-reactor)
10. [What Tenants Don't Need to Worry About](#10-what-tenants-dont-need-to-worry-about)
11. [Hot-Reloading and Lifecycle](#11-hot-reloading-and-lifecycle)
12. [Troubleshooting](#12-troubleshooting)
13. [Source Code Map](#13-source-code-map)

---

## 1. What Is the Reactor? (Tenant Perspective)

The **Reactor Engine** is a server-side (daemon) component that watches events happening inside a jaato session and automatically reacts to them by running Python scripts. It is a **jaato-premium** feature.

From the tenant's perspective, the reactor is an **invisible automation layer**:

- It runs inside the daemon — not in your client application
- It does not require any SDK code or client-side integration
- It is configured entirely through files in the workspace (`.jaato/`) or user home (`~/.jaato/`)
- It reacts to events that the agent and SDK emit naturally during their work

The reactor's primary use case is **agent handoff**: when one agent finishes its task, the reactor automatically starts another agent — seeding it with context from the first agent's session. This enables multi-agent workflows (implementer → validator → deployer, for example) without any client-side orchestration.

> **Companion reference:** This guide covers setup and tenant concerns. For the full engine internals (data models, threading, event bus bridge), see `reactor-implementation`.

---

## 2. Do Tenants Need to Do Anything?

**Short answer: No, unless you want agent handoffs or event-driven automation.**

The reactor is **inactive by default**. A jaato session works perfectly without any reactor rules. The reactor only activates when it finds a `reactors.json` file in one of its configuration locations.

| Scenario | Reactor Needed? |
|----------|-----------------|
| Single-agent session (chat, coding) | No |
| Single agent with subagents (delegation via SDK) | No — the subagent plugin handles lifecycle |
| Multi-agent handoffs (agent A finishes → agent B starts) | Yes |
| Event-driven webhooks on agent completion | Yes |
| Automated workflows triggered by plan/step events | Yes |
| Background tasks on session lifecycle events | Yes |

If you don't create a `reactors.json` file, the reactor loads an empty rule list and does nothing — zero overhead.

### What "tenant" means here

In jaato's architecture, a **tenant** is a client application that connects to the jaato daemon via the Python SDK (`jaato-sdk`) or TypeScript SDK (`@jaato/sdk`) over IPC or WebSocket. The reactor is **not a client** — it is a daemon-internal extension. However, the tenant controls the reactor's behavior by placing configuration files in the workspace.

---

## 3. How the Reactor Works — Event Flow

### The Event Bus

Every jaato session has an internal **event bus**. As the agent works, it emits typed events:

```
agent.created → step_started → turn.completed → step_completed → agent.completed
```

Both SDKs expose a `subscribe(type, handler)` API to listen to these events from the client side. The reactor is an **additional server-side subscriber** — it listens to the same events your client does.

### Reactor Processing Pipeline

```
1. Session emits event on event bus (e.g. agent.completed)
2. ReactorEngine._dispatch() receives it
3. For each enabled rule:
   a. Does the rule's event_type match?  →  No → skip
   b. Does the rule's JMESPath where clause match?  →  No → skip
   c. Submit action script to thread pool
4. Action script runs on a worker thread with:
   - params (from rule, after template substitution)
   - event (flat dict of event + payload fields)
   - ctx (ActionContext — fork, inject, webhook, shell, emit)
5. Action script does something (fork a session, post webhook, etc.)
```

### Infinite Loop Prevention

Events emitted by the reactor itself carry `source_agent="reactor"` and are **silently skipped** by `_dispatch()`. This prevents rule A from triggering rule B from triggering rule A infinitely. Your action script can safely call `ctx.emit_event()` to chain rules without creating loops.

### Where Reactor Actions Run

Action scripts execute **on the daemon server**, not on the client. They have access to:

- The server's session manager (for cross-session operations)
- The originating session's workspace
- The operating system environment
- The session's event bus

They do **not** have access to client-side resources (browser DOM, client filesystem, etc.).

---

## 4. What the Reactor Can Do

The `ActionContext` object passed to every action script provides these capabilities:

| Method | What it does |
|--------|-------------|
| `ctx.fork_from_originating(target_agent)` | **The canonical handoff.** Creates a new session under a different agent, seeded with the current session's conversation history. Returns `{"session_id": str, "agent_id": str}`. |
| `ctx.fork_from_session(session_id, target_agent)` | Fork from an arbitrary loaded session (cross-session orchestration). |
| `ctx.fork_from_waypoint(waypoint_id, target_agent)` | Fork from a past point in time (requires jaato-server >= 0.5.19). |
| `ctx.inject_prompt(text, target_session_id)` | Inject a prompt into a session (like a synthetic user message). |
| `ctx.create_session(agent, prompt, ...)` | Create a brand-new independent session. |
| `ctx.spawn_subagent(profile, task)` | Spawn a subagent as a child of the originating session. |
| `ctx.post_webhook(url, body)` | Send an HTTP POST (webhook notification). |
| `ctx.run_shell(cmd)` | Run a shell command. |
| `ctx.emit_event(event_type, payload)` | Publish a new event on the event bus (for rule chaining). |

All methods are **fire-and-forget from the SDK's perspective**. The client application doesn't initiate reactor actions — the daemon does. The client only observes the results as events on the event bus.

---

## 5. Setting Up Reactor Rules — `reactors.json`

### Where to Put the File

| Location | Scope | Hot-reloaded? |
|----------|-------|---------------|
| `~/.jaato/reactors.json` | All sessions on this daemon | Yes (every 2 seconds) |
| `<workspace>/.jaato/reactors.json` | Only sessions in this workspace | No (read at session start) |

Workspace rules **override** home rules by rule `id`. Rules with different IDs from both tiers are combined.

### Minimal Example

Create `<workspace>/.jaato/reactors.json`:

```json
{
  "version": 1,
  "rules": [
    {
      "id": "handoff-on-completion",
      "enabled": true,
      "match": {
        "event_type": "agent.completed",
        "where": "agent_id == 'implementer' && success == `true`"
      },
      "action": {
        "script": "reactors/handoff.py",
        "params": {
          "target_agent": "reviewer",
          "message": "The implementer finished. Review the changes."
        }
      }
    }
  ]
}
```

Then create `<workspace>/.jaato/reactors/handoff.py`:

```python
def execute(params, event, ctx):
    result = ctx.fork_from_originating(target_agent=params["target_agent"])
    if result:
        ctx.inject_prompt(params["message"], target_session_id=result["session_id"])
```

That's it. The reactor is now active for sessions in this workspace.

### Rule Schema

```json
{
  "id": "string (required) — unique rule identifier",
  "enabled": "boolean (default: true)",
  "match": {
    "event_type": "string (required) — see EventType catalog below",
    "where": "string (optional) — JMESPath expression"
  },
  "action": {
    "script": "string (required) — path to Python action script",
    "params": "object (default: {}) — passed to execute() after substitution"
  }
}
```

### Validation Rules

The daemon enforces these at parse time:

- `version` must be `1`
- Every rule must have `id` (string), `match.event_type` (string), and `action.script` (string)
- `enabled` defaults to `true` when absent
- `match.where` is optional (no filtering when absent)
- `action.params` defaults to `{}`
- Invalid JSON or unparseable rules → file is silently skipped with a warning log

### Available Event Types

The reactor can match on any event from the session event bus. The most commonly used types for tenant workflows:

| Event Type | When it fires | Key fields in merged view |
|------------|--------------|---------------------------|
| `agent.completed` | Agent signals it's done (or crashes) | `agent_id`, `success`, `payload`, `error` |
| `agent.created` | New agent session starts | `agent_id`, `agent_name`, `profile_name` |
| `agent.status_changed` | Agent status changes | `agent_id`, `status`, `error` |
| `plan_completed` | A plan finishes | `plan_id`, `plan_title`, `summary` |
| `plan_failed` | A plan fails | `plan_id`, `plan_title` |
| `step_completed` | A plan step finishes | `plan_id`, `step_id`, `result`, `output` |
| `step_failed` | A plan step fails | `plan_id`, `step_id`, `error` |
| `step_unblocked` | A blocked step's dependencies resolve | `plan_id`, `step_id`, `received_outputs` |
| `turn.completed` | A model turn finishes | `agent_id`, `total_tokens`, `turn_number` |
| `tool.call_completed` | A tool call finishes | `tool_name`, `success`, `duration_seconds` |
| `drift.measured` | Drift monitor reports | `step_id`, `drift_score`, `drift_flagged` |
| `external_event` | External trigger (webhook, etc.) | `source`, `payload` |

### JMESPath `where` Clauses

The optional `where` field filters events using JMESPath expressions evaluated against a **merged view** (event envelope + payload fields flattened into a single dict).

```json
// String equality
"where": "agent_id == 'reviewer'"

// Boolean literal (backtick-quoted)
"where": "success == `true`"

// Nested payload field (from completion_payload_schema)
"where": "payload.passed == `true` && payload.errors.length == `0`"

// Numeric comparison
"where": "total_tokens > `10000`"

// Compound
"where": "success == `false` || error != ''"
```

Invalid JMESPath → rule is skipped with a warning. Missing fields → `null` (falsy).

### Template Substitution in `params`

Action params support `${...}` placeholders:

| Placeholder | Resolves to | Example |
|-------------|------------|---------|
| `${event.agent_id}` | Event payload field via JMESPath | `"reviewer"` |
| `${event.payload.passed}` | Nested payload field | `true` |
| `${env.WEBHOOK_URL}` | Environment variable | `"https://..."` |

Type is preserved when the entire value is a single placeholder:
`{"n": "${event.total_tokens}"}` → `{"n": 5000}` (int, not string).

Mixed with text → everything becomes a string:
`{"m": "Agent ${event.agent_id}"}` → `{"m": "Agent reviewer"}`.

---

## 6. Writing Action Scripts

An action script is a Python file with a module-level `execute()` function:

```python
# .jaato/reactors/my_action.py
def execute(params, event, ctx):
    """
    params: dict  — rule.action.params after template substitution
    event:  dict  — flat merged view of the event + payload
    ctx:    ActionContext — server-side API for session manipulation
    """
    # Your logic here
    pass
```

### Script Resolution

The engine looks for scripts in this order:

1. **Absolute path** (if the script path starts with `/`)
2. **Workspace**: `<workspace>/.jaato/<script>`
3. **Home**: `~/.jaato/<script>`

### Error Handling

All exceptions in action scripts are caught and logged — they never propagate to the event bus or affect the session. A failing script does not block other rules.

### Script Reloading

Scripts are reloaded on every invocation (using `load_script_symbol` with a `_jaato_reactor` module prefix). You can edit a script and the changes take effect on the next matching event — no restart needed.

---

## 7. Completion Payload Schemas — Typed Handoffs

### What They Are

A **completion payload schema** lets you define a structured shape for the data an agent passes when it signals completion. Without a schema, `signal_completion` accepts only a free-text `summary: str`. With a schema, the agent emits a typed `payload` that the reactor can match on with precision.

### How to Declare

In the agent's profile JSON:

```json
{
  "name": "validator-tier1",
  "completion_payload_schema": {
    "type": "object",
    "properties": {
      "passed": {"type": "boolean"},
      "errors": {"type": "array", "items": {"type": "string"}},
      "summary": {"type": "string"}
    },
    "required": ["passed", "summary"]
  }
}
```

Or reference an external file:

```json
{
  "completion_payload_schema": ".jaato/completion_schemas/validator.json"
}
```

Schema resolution follows the same 3-tier pattern as scripts and rules: absolute → workspace `.jaato/completion_schemas/` → home `~/.jaato/completion_schemas/`.

### How It Works End-to-End

```
1. Profile declares completion_payload_schema
2. signal_completion tool's parameters are rebuilt: payload replaces summary
3. Model generates structured payload (enforced by provider at sampling time)
4. Server validates with jsonschema (defense-in-depth)
5. AgentCompletedEvent emitted with typed payload field
6. Reactor receives event — payload fields are in the merged view
7. Reactor rule's where clause can match on payload.passed, payload.errors, etc.
8. Action script receives typed data in event dict
```

### Why This Matters for Tenants

Typed payloads let the reactor make **precise routing decisions**:

- Without schema: `"where": "success == `true`"` — you only know it succeeded, not *how*
- With schema: `"where": "payload.passed == `true` && payload.errors.length == `0`"` — you know validation passed with zero errors, and you can route the `errors` list to a fix-agent on failure

---

## 8. Common Handoff Patterns

### Pattern 1: Simple Agent Handoff

When agent A finishes, start agent B with the same conversation history:

```json
{
  "id": "implementer-to-reviewer",
  "match": {
    "event_type": "agent.completed",
    "where": "agent_id == 'implementer' && success == `true`"
  },
  "action": {
    "script": "reactors/handoff.py",
    "params": {
      "target_agent": "reviewer",
      "message": "The implementer finished. Review the changes above."
    }
  }
}
```

### Pattern 2: Conditional Handoff Based on Validation Result

Two rules for the same event — one for pass, one for fail:

```json
{
  "id": "validation-pass",
  "match": {
    "event_type": "agent.completed",
    "where": "agent_id == 'validator' && payload.passed == `true`"
  },
  "action": {
    "script": "reactors/handoff.py",
    "params": {
      "target_agent": "deployer",
      "message": "All validations passed. Deploy."
    }
  }
}
```

```json
{
  "id": "validation-fail",
  "match": {
    "event_type": "agent.completed",
    "where": "agent_id == 'validator' && payload.passed == `false`"
  },
  "action": {
    "script": "reactors/handoff.py",
    "params": {
      "target_agent": "implementer",
      "message": "Validation failed. Fix these errors: ${event.payload.errors}"
    }
  }
}
```

### Pattern 3: Webhook Notification

```json
{
  "id": "notify-on-completion",
  "match": {
    "event_type": "agent.completed",
    "where": "success == `true`"
  },
  "action": {
    "script": "reactors/notify.py",
    "params": {
      "url": "${env.WEBHOOK_URL}",
      "agent": "${event.agent_id}"
    }
  }
}
```

```python
# .jaato/reactors/notify.py
def execute(params, event, ctx):
    ctx.post_webhook(
        url=params["url"],
        body={"agent": params["agent"], "success": event.get("success")},
    )
```

### Pattern 4: Rule Chaining

One action emits an event that triggers another rule:

```python
# .jaato/reactors/chain.py
def execute(params, event, ctx):
    ctx.emit_event("external_event", {
        "handoff_result": "success",
        "target_agent": params["next_agent"],
    })
```

A second rule matches on `"event_type": "external_event"`.

---

## 9. The SDK's Relationship to the Reactor

### The Reactor Is Server-Side

The reactor runs entirely inside the jaato daemon. Neither the Python SDK nor the TypeScript SDK includes reactor-specific APIs. A tenant application does **not**:

- Start or stop the reactor
- Call reactor methods directly
- Receive reactor-specific events different from normal session events

### What the SDK Does Provide

Both SDKs give the tenant full visibility into reactor effects through the **event subscription API**:

```python
# Python SDK
client.subscribe("agent.created", handler=my_handler)
client.subscribe("agent.completed", handler=my_completion_handler)
client.subscribe("step.unblocked", handler=my_step_handler)
```

```typescript
// TypeScript SDK
client.subscribe("agent.completed", (event) => { /* ... */ });
client.subscribe("step.unblocked", (event) => { /* ... */ });
```

When the reactor forks a new session or injects a prompt, the resulting events (`agent.created`, `agent.status_changed`, etc.) flow through the same event bus and arrive at your client just like any other event.

### `inject_prompt` Priority

The SDK's `inject_prompt` / `injectPrompt` method supports a `source_type` parameter:

| `source_type` | Use case |
|---------------|----------|
| `"user"` | Human user sends a message (mid-turn steer) |
| `"child"` | Subagent follow-up pattern |
| `"system"` / `"event"` / `"parent"` | **Reactor and hook callers** |

When the reactor injects a prompt into a session (via `ctx.inject_prompt()`), it uses these higher-priority source types internally. This means reactor-injected prompts are processed before queued user messages — the reactor's "you've taken over" prompt arrives before any human input.

### Client-Provided Tools

The reactor runs server-side and cannot call client-provided (browser-side) tools. If a reactor action script needs to interact with a browser, it must go through the session's agent (e.g., fork a session with an agent that has browser tools in its profile). See `host-provided-tools` for details on client-provided tool registration.

### Presentation Context Inheritance

All fork actions (`fork_from_originating`, `fork_from_session`, `create_session`) **inherit the originating session's display surface** by default. If the parent session is rendering in a TUI, the forked session renders in the same TUI. If it's in a web dashboard, the fork renders there. Pass an explicit `presentation` argument to override.

---

## 10. What Tenants Don't Need to Worry About

### Threading and Concurrency

The reactor uses a `ThreadPoolExecutor` (default 4 workers) to run action scripts. Scripts can block on I/O without affecting event delivery. The tenant does not manage threads — the daemon handles this.

### Event Bus Subscription Management

The reactor automatically subscribes to each session's event bus when the session starts (via a session hook). It cleans up subscriptions when sessions end. No tenant action required.

### Headless Sessions

When the reactor forks a session, it creates a **headless session** — a fully functional session with no attached client. Its client-facing events are silently dropped. The tenant's client doesn't need to handle or even know about headless sessions. If a tenant client connects later, it can attach to any loaded session including headless ones.

### Infinite Loop Prevention

The reactor skips events with `source_agent="reactor"`. This is automatic — the tenant doesn't need to configure or worry about it.

### Script Isolation

Scripts are loaded with a `_jaato_reactor` module prefix to prevent naming collisions. Each invocation reloads the module. No tenant action needed.

---

## 11. Hot-Reloading and Lifecycle

### Home Rules: Hot-Reloaded

`~/.jaato/reactors.json` is polled every 2 seconds for mtime changes. When the file changes, rules are re-parsed and replace the in-memory list.

> **Caveat:** Active sessions are not immediately updated — merged rules are computed at session start. New or restarted sessions get the updated rules.

### Workspace Rules: Read Once

`<workspace>/.jaato/reactors.json` is read when the session starts. To update workspace rules, **restart the session**.

### Script Changes: Immediate

Action scripts (`.py` files) are reloaded on every invocation. Edit a script and the next matching event uses the new version.

### Daemon Startup Sequence

```
jaato daemon starts
  └─ ReactorExtension.start()
       ├─ Load ~/.jaato/reactors.json (or empty if missing)
       ├─ Start FileWatcher for home rules
       └─ Register session hook
            └─ On each new session:
                 ├─ Merge home + workspace rules
                 ├─ Filter to enabled rules
                 └─ Subscribe to session event bus
```

---

## 12. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Reactor not firing | No `reactors.json` found | Create at `~/.jaato/reactors.json` or `<workspace>/.jaato/reactors.json` |
| Rules not matching | `where` clause syntax error | Check JMESPath syntax — backtick-quoted booleans: `` `true` `` not `true` |
| Action script not found | Wrong path | Scripts resolve: workspace `.jaato/` first, then home `~/.jaato/` |
| Session not forking | Agent name not recognized | Ensure `target_agent` matches a configured agent in `.jaato/agents/` |
| Infinite loop | Rule emitting events that match itself | Events from `ctx.emit_event()` carry `source_agent="reactor"` and are skipped — if you're seeing loops, check that your rule isn't matching events from non-reactor sources |
| Workspace rules ignored after edit | Workspace rules aren't hot-reloaded | Restart the session |
| `signal_completion` missing `payload` field | No `completion_payload_schema` on profile | Add the schema to the agent's profile JSON |

---

## 13. Source Code Map

### Reactor Engine (jaato-premium)

| File | Contents |
|------|----------|
| `reactors/engine.py` | Core loop: loads rules, subscribes to event bus, dispatches actions |
| `reactors/rules.py` | `Rule`, `MatchSpec`, `ActionSpec` — data models, parsing, validation, merge |
| `reactors/matcher.py` | `build_merged_view()`, `matches_where()` — JMESPath evaluation |
| `reactors/templating.py` | `substitute_params()` — `${event.*}` / `${env.*}` expansion |
| `reactors/action_context.py` | `ActionContext` — fork, inject, spawn, webhook, shell, emit |
| `reactors/extension.py` | `ReactorExtension` — daemon entry point, lifecycle |
| `reactors/watcher.py` | `FileWatcher` — mtime-poll hot-reload for home rules |
| `reactors/tests/` | Unit and integration tests for all components |

### Event Infrastructure (jaato-sdk)

| File | Contents |
|------|----------|
| `jaato_sdk/event_bus.py` | `EventType` enum, `Event`, `EventFilter`, `Subscription` |
| `jaato_sdk/event_payloads.py` | TypedDict payload schemas for all event types |
| `jaato_sdk/events.py` | `AgentCompletedEvent` and all server event dataclasses |

### Server Integration (jaato-server)

| File | Contents |
|------|----------|
| `server/session_manager.py` | `create_headless_session()`, `inject_prompt_to_session()` |
| `server/core.py` | Agent identity propagation (`_main_agent_id`) |
| `shared/lifecycle_tools.py` | `signal_completion` with typed payload validation |
| `shared/completion_schema_loader.py` | 3-tier schema path resolution |
| `shared/script_loader.py` | Script resolution and loading with module prefix |

### Related References

| Reference | Covers |
|-----------|--------|
| `reactor-implementation` | Full engine internals, threading model, dispatch pipeline |
| `host-provided-tools` | Client-side tool registration and execution protocol |
| `client-sdk-reference` | SDK API, event subscription, `inject_prompt` priorities |
| `jaato-subagent-profiles` | Profile system, `completion_payload_schema`, agent configuration |
