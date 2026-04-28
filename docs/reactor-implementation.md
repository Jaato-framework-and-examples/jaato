# Reactor Engine — Complete Reference

> Scope: The rules-driven event-action system in jaato-premium, including how to author reactor rules for distinct event types, write action scripts, and combine completion payload schemas with reactor handoffs for inter-agent delegation.

## Table of Contents

1. [What Is the Reactor Engine?](#1-what-is-the-reactor-engine)
2. [Architecture Overview](#2-architecture-overview)
3. [Rule Authoring — `reactors.json`](#3-rule-authoring--reactorsjson)
4. [Event Types and Payload Schemas](#4-event-types-and-payload-schemas)
5. [JMESPath Matching with `where` Clauses](#5-jmespath-matching-with-where-clauses)
6. [Template Substitution in Action Params](#6-template-substitution-in-action-params)
7. [Writing Action Scripts](#7-writing-action-scripts)
8. [ActionContext API Reference](#8-actioncontext-api-reference)
9. [Completion Payload Schemas — Typed Handoffs](#9-completion-payload-schemas--typed-handoffs)
10. [Agent Handoff Patterns](#10-agent-handoff-patterns)
11. [Hot-Reloading and the File Watcher](#11-hot-reloading-and-the-file-watcher)
12. [Configuration Reference](#12-configuration-reference)
13. [Runtime Internals](#13-runtime-internals)
14. [Source Code Map](#14-source-code-map)

---

## 1. What Is the Reactor Engine?

The Reactor Engine is a **rules-driven event-action system** that runs inside the jaato daemon. It observes events on each session's runtime event bus and, when a rule's match clause fires, executes the rule's action script on a worker thread.

Think of it as an **event loop outside the model**: the model does its work, emits events (completion, tool calls, plan changes, drift measurements), and the reactor reacts to those events autonomously — spawning new sessions, injecting prompts, posting webhooks, or chaining further events.

The reactor is the mechanism that enables **agent handoffs**: an agent completes its task (optionally with a typed completion payload), the reactor observes the `agent.completed` event, and forks a new session under a different agent — seeding it with the completed session's history and priming it with a synthetic "you've taken over" prompt.

### Where It Fits

```
jaato daemon startup
  └─ ReactorExtension.start()
       ├─ Load ~/.jaato/reactors.json (daemon-global rules)
       ├─ Start FileWatcher for hot-reload
       └─ Register session hook: on_session_ready()
            └─ Per session:
                 ├─ Merge home + workspace rules
                 ├─ Filter to enabled rules
                 └─ Subscribe to session's EventBus
                      └─ _dispatch() on every event
                           └─ Match → _run_action() on ThreadPoolExecutor
```

---

## 2. Architecture Overview

### Core Components

| Component | File | Role |
|-----------|------|------|
| `ReactorExtension` | `extension.py` | Daemon entry point. Creates engine, registers session hook. |
| `ReactorEngine` | `engine.py` | Core loop: loads rules, subscribes to event bus, dispatches actions. |
| `Rule` / `MatchSpec` / `ActionSpec` | `rules.py` | Data models for reactor rules. JSON parsing, validation, merge. |
| `build_merged_view` / `matches_where` | `matcher.py` | Builds flat event view and evaluates JMESPath `where` clauses. |
| `substitute_params` | `templating.py` | `${event.*}` / `${env.*}` template substitution in action params. |
| `ActionContext` | `action_context.py` | API exposed to action scripts: fork, inject, spawn, webhook, shell. |
| `FileWatcher` | `watcher.py` | mtime-poll watcher for hot-reloading home rules. |

### Data Flow

```
Session EventBus publishes Event
  → ReactorEngine._dispatch()
    → build_merged_view(event)  → flat dict
    → For each enabled rule:
        → Match event_type
        → Evaluate where clause (JMESPath)
        → ThreadPoolExecutor.submit(_run_action)
          → resolve_script_path()
          → load_script_symbol("execute")
          → substitute_params(rule.action.params, view, env)
          → ActionContext(server, session_id, ...)
          → execute(params, view, ctx)
```

### Infinite Loop Prevention

Events published with `source_agent == "reactor"` are **silently skipped** by `_dispatch()`. This prevents a reactor action that calls `ctx.emit_event()` from triggering its own rule again. All events emitted via `ActionContext.emit_event()` automatically carry `source_agent="reactor"`.

---

## 3. Rule Authoring — `reactors.json`

Rules live in two tiers (merged at session start):

| Tier | Path | Scope |
|------|------|-------|
| **Home** (daemon-global) | `~/.jaato/reactors.json` | All sessions on this daemon |
| **Workspace** (per-session) | `<workspace>/.jaato/reactors.json` | Only sessions in this workspace |

Workspace rules **override** home rules by `id`. Non-overlapping IDs from both tiers are combined.

### Schema

```json
{
  "version": 1,
  "rules": [
    {
      "id": "handoff-on-completion",
      "enabled": true,
      "match": {
        "event_type": "agent.completed",
        "where": "success == `true` && payload.passed == `true`"
      },
      "action": {
        "script": "reactors/handoff.py",
        "params": {
          "target_agent": "memory-advisor",
          "message": "The ${event.agent_id} agent completed successfully. Review its work."
        }
      }
    }
  ]
}
```

### Validation Rules (from `parse_rules()`)

- `version` must be `1` — any other value raises `ValueError("Unsupported reactors.json version: ...")`
- Every rule must have `id` (string) — missing raises `ValueError("Rule at index N is missing required 'id' field")`
- Every rule must have `match.event_type` (string) — missing raises `ValueError("Rule 'X' is missing match.event_type")`
- Every rule must have `action.script` (string) — missing raises `ValueError("Rule 'X' is missing action.script")`
- `enabled` defaults to `true` when absent
- `match.where` is optional (defaults to `None`, meaning no additional filtering)
- `action.params` defaults to `{}` when absent
- Invalid JSON or unparseable rules → file is silently skipped with a warning log (returns `[]`)

### Merge Semantics (`merge_rules()`)

```python
# Workspace overrides home by id:
home = [Rule(id="r1", action=ActionSpec(script="old.py"))]
ws   = [Rule(id="r1", action=ActionSpec(script="new.py", enabled=False))]
merged = merge_rules(home, ws)
# → [Rule(id="r1", script="new.py", enabled=False)]
```

---

## 4. Event Types and Payload Schemas

The reactor subscribes to **all events** (`EventFilter()` with empty event_types), then filters by `rule.match.event_type` in `_dispatch()`. The event types available on the bus are defined in `jaato_sdk.event_bus.EventType`.

### Complete EventType Catalog (Bus Events)

#### Plan Lifecycle

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `plan_created` | `PlanCreatedPayload` | `plan_id`, `plan_title`, `steps` |
| `plan_started` | `PlanStartedPayload` | `plan_id`, `plan_title` |
| `plan_completed` | `PlanCompletedPayload` | `plan_id`, `plan_title`, `summary?`, `progress?` |
| `plan_failed` | `PlanFailedPayload` | `plan_id`, `plan_title`, `summary?`, `progress?` |
| `plan_cancelled` | `PlanCancelledPayload` | `plan_id`, `plan_title`, `summary?`, `progress?` |

#### Step Lifecycle

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `step_added` | `StepAddedPayload` | `plan_id`, `step_id`, `step_description`, `step_sequence` |
| `step_started` | `StepStartedPayload` | `plan_id`, `step_id`, `step_description`, `step_sequence` |
| `step_completed` | `StepCompletedPayload` | `plan_id`, `step_id`, `step_description`, `step_sequence`, `result?`, `output?`, `provides?` |
| `step_failed` | `StepFailedPayload` | `plan_id`, `step_id`, `step_description`, `step_sequence`, `error?` |
| `step_skipped` | `StepSkippedPayload` | `plan_id`, `step_id`, `step_description`, `step_sequence` |
| `step_blocked` | `StepBlockedPayload` | `plan_id`, `step_id`, `blocked_by` |
| `step_unblocked` | `StepUnblockedPayload` | `plan_id`, `step_id`, `received_outputs`, `unblocked_by` |

#### Agent Lifecycle

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `agent.created` | `AgentCreatedPayload` | `agent_id`, `agent_name`, `agent_type`, `profile_name?`, `parent_agent_id?` |
| `agent.status_changed` | `AgentStatusChangedPayload` | `agent_id`, `status`, `error?` |
| `agent.completed` | `AgentCompletedPayload` | `agent_id`, `completed_at`, `success`, `token_usage?`, `turns_used?`, `error?`, **`payload?`** |
| `agent.output` | `AgentOutputPayload` | `agent_id`, `source`, `text`, `mode` |

#### Tool Execution

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `tool.call_started` | `ToolCallStartedPayload` | `agent_id`, `tool_name`, `tool_args`, `call_id?` |
| `tool.call_completed` | `ToolCallCompletedPayload` | `agent_id`, `tool_name`, `success`, `duration_seconds`, `error_message?` |
| `tool.output` | `ToolOutputPayload` | `agent_id`, `call_id`, `chunk` |

#### Context & Turn

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `turn.completed` | `TurnCompletedPayload` | `agent_id`, `turn_number`, `prompt_tokens`, `output_tokens`, `total_tokens`, `duration_seconds`, `function_calls` |
| `turn.progress` | `TurnProgressPayload` | `agent_id`, `total_tokens`, `context_limit`, `percent_used`, `pending_tool_calls` |
| `context.updated` | `ContextUpdatedPayload` | `agent_id`, `total_tokens`, `percent_used`, `turns`, `gc_threshold?`, `gc_strategy?` |

#### Permission

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `permission.requested` | `PermissionRequestedPayload` | `agent_id`, `request_id`, `tool_name`, `response_options` |
| `permission.resolved` | `PermissionResolvedPayload` | `agent_id`, `request_id`, `tool_name`, `granted`, `method` |

#### Drift Monitor & External

| EventType value | Payload TypedDict | Key fields |
|-----------------|-------------------|------------|
| `drift.measured` | `DriftMeasuredPayload` | `step_id`, `step_description`, `drift_score`, `drift_flagged`, `strategic_drift?` |
| `external_event` | `ExternalEventPayload` | `source`, `event_type`, `headers`, `payload`, `plan_id`, `step_id` |

### The Merged View

The matcher builds a **flat dict** from the bus `Event` object:

```python
def build_merged_view(event) -> dict:
    # Envelope fields
    view = {
        "event_id": event.event_id,
        "event_type": event.event_type.value,   # e.g. "agent.completed"
        "timestamp": event.timestamp,
        "source_agent": event.source_agent,
    }
    # Payload fields hoisted — payload wins on collision
    view.update(event.payload or {})
    return view
```

This means in a `where` clause or template param, you reference payload fields directly:

- `${event.agent_id}` — from `AgentCompletedPayload.agent_id`
- `${event.success}` — from `AgentCompletedPayload.success`
- `${event.payload.passed}` — when the payload contains nested data
- `${event.token_usage.total_tokens}` — nested dict access via JMESPath

> **Payload collision note:** If a payload field has the same name as an envelope field (e.g. `source_agent`), the payload value wins. This is by design — payload data is more specific to the event.

---

## 5. JMESPath Matching with `where` Clauses

The optional `where` field on a rule's `match` is a **JMESPath expression** evaluated against the merged view. The rule fires only when the expression returns a truthy value.

### Examples

```json
// String equality
"where": "agent_id == 'reviewer'"

// Boolean literal (backtick-quoted)
"where": "success == `true`"

// Numeric comparison
"where": "duration_seconds > `60`"

// Compound expression
"where": "success == `false` || duration_seconds > `60`"

// Nested field access
"where": "token_usage.total_tokens > `10000`"

// Typed payload field (from completion_payload_schema)
"where": "payload.passed == `true` && payload.errors.length == `0`"
```

### Error Handling

- Invalid JMESPath expressions → `matches_where()` returns `False` with a warning log
- Missing fields → JMESPath returns `null`, which is falsy
- The rule is simply skipped — no exception propagates to the bus thread

---

## 6. Template Substitution in Action Params

Action `params` support `${...}` placeholders that are resolved before the script receives them.

### Namespaces

| Namespace | Resolution | Example |
|-----------|------------|---------|
| `${event.<path>}` | JMESPath on the merged event view | `${event.agent_id}`, `${event.token_usage.total_tokens}` |
| `${env.<NAME>}` | `os.environ` lookup | `${env.HOME}`, `${env.MY_API_KEY}` |
| Unknown | Warning + empty string | `${unknown.foo}` → `""` |

### Type Preservation

When the **entire string** is a single placeholder, the resolved type is preserved:

```json
{"n": "${event.token_usage.total_tokens}"}     → {"n": 5000}        // int preserved
{"b": "${event.success}"}                       → {"b": true}         // bool preserved
```

When placeholders are **mixed with other text**, all values are stringified:

```json
{"m": "Agent ${event.agent_id} used ${event.turns_used} turns"}  → {"m": "Agent reviewer used 5 turns"}
```

### Recursion

Substitution recurses into nested dicts and lists:

```json
{
  "config": {
    "agent": "${event.agent_id}",
    "threshold": "${env.THRESHOLD}"
  },
  "files": ["${event.output_file}", "static.txt"]
}
```

Non-string values (int, bool, None, nested dicts/lists) pass through unchanged.

---

## 7. Writing Action Scripts

An action script is a Python file with an `execute(params, event, ctx)` function:

```python
# reactors/my_handoff.py
"""Handoff action: fork a new session when a validator completes."""

def execute(params, event, ctx):
    """
    Args:
        params: dict — rule.action.params after template substitution
        event: dict  — the merged view (flat event + payload)
        ctx: ActionContext — helpers for session manipulation
    """
    target = params["target_agent"]

    # Fork from the originating session's history
    result = ctx.fork_from_originating(target_agent=target)

    if not result:
        ctx.logger.warning("Handoff fork failed for agent '%s'", target)
        return

    new_session_id = result["session_id"]

    # Prime the new session with a handoff prompt
    ctx.inject_prompt(
        f"You are taking over from {event.get('agent_id', 'unknown')}. "
        f"Review the conversation history and continue the work.",
        target_session_id=new_session_id,
    )
```

### Script Resolution

Scripts are resolved via `shared.script_loader.resolve_script_path()`:

1. If the path is **absolute**, use it directly
2. Try `<workspace>/.jaato/<script>`
3. Fall back to `~/.jaato/<script>`

The script must expose an `execute` function at module level. The engine imports it with a `_jaato_reactor` module prefix to avoid collisions.

### Error Handling

All exceptions in action scripts are caught by `_run_action()` and logged — they never propagate to the event bus thread. A failing script does not affect other rules or the session.

---

## 8. ActionContext API Reference

The `ctx` parameter passed to every action script is an `ActionContext` dataclass.

### Constructor Fields

| Field | Type | Description |
|-------|------|-------------|
| `server` | `Any` (JaatoServer) | The originating session's server instance |
| `session_id` | `str` | The session whose event fired the rule |
| `workspace_path` | `str \| None` | The session's workspace directory |
| `env` | `Mapping[str, str]` | Snapshot of `os.environ` at dispatch time |
| `session_manager` | `Any \| None` | The daemon's SessionManager (for cross-session ops) |
| `logger` | `Logger` | Logger for script output (channel: `jaato_premium.reactors.action`) |

### Methods

#### `inject_prompt(text, target_session_id=None)`

Inject a prompt as a user message. Without `target_session_id`, routes to the originating session. With it, routes to the specified session via `SessionManager.inject_prompt_to_session()`.

**Key behavior:** When `target_session_id` is set and routing fails (session not loaded, no session_manager), the method logs a warning and returns — it does **not** fall back to the originating session.

#### `spawn_subagent(profile, task)`

Spawn a subagent as a **child** of the originating session. The subagent shares the parent's lifecycle. Uses the subagent plugin's internal executor directly.

#### `create_session(profile=None, agent=None, initial_prompt=None, session_name=None, presentation=None) → str`

Create a new **top-level session** with an independent lifecycle. Returns the session ID, or empty string on failure. Inherits the originating session's `PresentationContext` by default.

#### `fork_from_originating(target_agent, target_profile=None, workspace_path=None, presentation=None, session_name=None) → Dict[str, str]`

**The canonical handoff action.** Forks a new session seeded with the originating session's history, running under a different agent.

Returns `{"session_id": str, "agent_id": str}` on success, `{}` on failure.

#### `fork_from_session(source_session_id, target_agent, ...) → Dict[str, str]`

Generalizes `fork_from_originating` to arbitrary loaded sessions. Useful for cross-session orchestration.

#### `fork_from_waypoint(waypoint_id, target_agent, ...) → Dict[str, str]`

Forks from a **past point in time** captured by a waypoint. Requires jaato-server >= 0.5.19. Returns `{}` if the waypoint has no history snapshot.

#### `post_webhook(url, body=None, headers=None, method="POST", timeout=10.0)`

HTTP request via stdlib `urllib`. Logs outcome, never raises.

#### `run_shell(cmd, cwd=None, timeout=60.0)`

Subprocess call. Defaults `cwd` to `workspace_path`. Logs outcome, never raises.

#### `emit_event(event_type, payload)`

Publish a new event on the session's event bus. Automatically sets `source_agent="reactor"` to prevent infinite loops. Enables rule chaining.

---

## 9. Completion Payload Schemas — Typed Handoffs

### What They Are

A **completion payload schema** is an optional JSON Schema declared on a subagent profile's `completion_payload_schema` field. When set, it replaces the legacy `summary: str` parameter of `signal_completion` with a typed `payload: <schema>` parameter that the provider enforces at sampling time.

### Declaration (in profile JSON)

```json
{
  "name": "validator-tier1",
  "completion_payload_schema": {
    "type": "object",
    "properties": {
      "passed": {"type": "boolean"},
      "errors": {"type": "array", "items": {"type": "string"}},
      "files_checked": {"type": "array", "items": {"type": "string"}},
      "summary": {"type": "string"}
    },
    "required": ["passed", "summary"]
  }
}
```

Alternatively, reference an external schema file:

```json
{
  "name": "validator-tier1",
  "completion_payload_schema": ".jaato/completion_schemas/validator.json"
}
```

### Schema Resolution (3-Tier)

1. **Absolute path** — used directly
2. **Workspace** — `<workspace>/.jaato/completion_schemas/<path>`
3. **Home** — `~/.jaato/completion_schemas/<path>`

Implemented in `shared/completion_schema_loader.resolve_completion_schema()`.

### How It Works at Runtime

1. **Profile loading**: `SubagentProfile.completion_payload_schema` is resolved (inline dict or path → loaded JSON)
2. **Session creation**: The resolved schema is stored on `JaatoSession._completion_payload_schema`
3. **LifecycleTools construction**: `resolve_completion_schema()` is called to get the final schema dict
4. **Tool schema generation**: `get_tool_schemas()` returns either:
   - Legacy: `signal_completion(summary: str)`
   - Typed: `signal_completion(payload: <schema>)` — the schema is embedded in the tool parameters
5. **Provider enforcement**: Providers that constrain tool calls at sampling (Anthropic, OpenAI, Google, Ollama) enforce the shape automatically
6. **Server-side validation**: `jsonschema.validate()` runs as defense-in-depth. On failure, returns a structured error to the model (no event emitted) so it can self-correct
7. **Event emission**: On success, the validated payload is forwarded to `hooks.on_agent_completed(payload=...)`, which emits `AgentCompletedEvent` with the payload in its `payload` field

### Inheritance

`completion_payload_schema` follows **scalar-override** semantics: parent profiles must agree on the value, or the child must explicitly override. Cycles are detected and reported.

### The `AgentCompletedEvent` Bridge

When a profile with a `completion_payload_schema` calls `signal_completion(payload={...})`:

```
signal_completion(payload={"passed": true, "errors": []})
  → LifecycleTools validates against schema ✓
  → hooks.on_agent_completed(payload={"passed": true, "errors": []})
    → AgentCompletedEvent emitted on EventBus
      → AgentCompletedPayload: {
           agent_id: "validator-tier1",
           success: true,
           payload: {"passed": true, "errors": []}  ← typed!
         }
        → Reactor receives event
          → Merged view: {"agent_id": "validator-tier1", "success": true, "passed": true, "errors": []}
```

Without a schema, the event has `payload: None` and only `summary` (a free-text string) is available.

---

## 10. Agent Handoff Patterns

### Pattern 1: Simple Completion Handoff

React to `agent.completed` and fork a new agent session:

```json
{
  "version": 1,
  "rules": [{
    "id": "handoff-to-advisor",
    "match": {
      "event_type": "agent.completed",
      "where": "agent_id == 'implementer' && success == `true`"
    },
    "action": {
      "script": "reactors/handoff_to_advisor.py",
      "params": {
        "target_agent": "advisor",
        "message": "The implementer finished. Review and advise."
      }
    }
  }]
}
```

```python
# reactors/handoff_to_advisor.py
def execute(params, event, ctx):
    result = ctx.fork_from_originating(target_agent=params["target_agent"])
    if result:
        ctx.inject_prompt(params["message"], target_session_id=result["session_id"])
```

### Pattern 2: Typed Completion + Conditional Handoff

Combine `completion_payload_schema` with a `where` clause for precise control:

```json
// Profile for validator:
{
  "name": "validator",
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

// Reactor rule — only hand off on validation pass:
{
  "id": "pass-handoff",
  "match": {
    "event_type": "agent.completed",
    "where": "agent_id == 'validator' && payload.passed == `true`"
  },
  "action": {
    "script": "reactors/pass_handoff.py",
    "params": {
      "target_agent": "deployer",
      "message": "All validations passed. Proceed with deployment."
    }
  }
}

// Reactor rule — handle validation failure:
{
  "id": "fail-handoff",
  "match": {
    "event_type": "agent.completed",
    "where": "agent_id == 'validator' && payload.passed == `false`"
  },
  "action": {
    "script": "reactors/fail_handoff.py",
    "params": {
      "target_agent": "implementer",
      "message": "Validation failed with errors: ${event.payload.errors}"
    }
  }
}
```

### Pattern 3: Multi-Validator Fan-In

React to `step_unblocked` (all validators passed) before handing off:

```json
{
  "id": "all-validators-passed",
  "match": {
    "event_type": "step_unblocked",
    "where": "step_id == 'validation-gate'"
  },
  "action": {
    "script": "reactors/fan_in_handoff.py",
    "params": {
      "target_agent": "integrator"
    }
  }
}
```

### Pattern 4: Webhook Notification on Completion

```json
{
  "id": "notify-external",
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
# reactors/notify.py
def execute(params, event, ctx):
    ctx.post_webhook(
        url=params["url"],
        body={"agent": params["agent"], "success": event.get("success")},
    )
```

### Pattern 5: Rule Chaining

Use `ctx.emit_event()` to chain rules:

```python
# reactors/chain.py
def execute(params, event, ctx):
    # Do some work, then emit a custom event for another rule
    ctx.emit_event("external_event", {
        "source": "reactor",
        "handoff_result": "success",
        "target_agent": params["next_agent"],
    })
```

A second rule with `"event_type": "external_event"` will fire on this emitted event.

---

## 11. Hot-Reloading and the File Watcher

The `FileWatcher` polls `~/.jaato/reactors.json` for mtime changes every 2 seconds (configurable via `poll_interval`). When a change is detected:

1. `load_rules_file()` re-reads and re-parses the file
2. The new rules replace `_home_rules` in the engine
3. **Active sessions are not immediately updated** — the merged rules are computed at `on_session_ready()` time, which fires once per session lifecycle

> **Important:** Workspace-local rules (`<workspace>/.jaato/reactors.json`) are **not** hot-reloaded. They are read once when the session starts. To update workspace rules, restart the session.

The watcher runs on a daemon thread (`daemon=True`) and stops gracefully via a `threading.Event`.

---

## 12. Configuration Reference

### `reactors.json` Schema

```json
{
  "version": 1,
  "rules": [
    {
      "id": "string (required)",
      "enabled": "boolean (default: true)",
      "match": {
        "event_type": "string (required) — EventType value from jaato_sdk.event_bus",
        "where": "string (optional) — JMESPath expression"
      },
      "action": {
        "script": "string (required) — path to Python script with execute()",
        "params": "object (default: {}) — template-expanded params for execute()"
      }
    }
  ]
}
```

### Rule Locations

| Location | Scope | Hot-reload |
|----------|-------|------------|
| `~/.jaato/reactors.json` | All sessions | Yes (2s poll) |
| `<workspace>/.jaato/reactors.json` | Sessions in this workspace | No (read at session start) |

### Script Locations

Resolved by `shared.script_loader.resolve_script_path()`:

| Priority | Path pattern |
|----------|-------------|
| 1 | Absolute path (if script starts with `/`) |
| 2 | `<workspace>/.jaato/<script>` |
| 3 | `~/.jaato/<script>` |

### Completion Schema Locations

Resolved by `shared.completion_schema_loader.resolve_completion_schema()`:

| Priority | Path pattern |
|----------|-------------|
| 1 | Absolute path (if reference starts with `/`) |
| 2 | `<workspace>/.jaato/completion_schemas/<reference>` |
| 3 | `~/.jaato/completion_schemas/<reference>` |

### ThreadPool Configuration

`ReactorEngine.__init__(max_workers=4)` — default 4 worker threads. Configured in `ReactorExtension.__init__()`.

---

## 13. Runtime Internals

### Session Hook Registration

The `ReactorExtension` registers `engine.on_session_ready` as a session hook via `session_manager.add_session_hook()`. This hook fires for every new session (including headless sessions created by the reactor itself), enabling recursive handoff chains.

### Event Bus Subscription

The engine subscribes with an empty `EventFilter()` (match everything) and a named subscriber `"reactor:<session_id>"`. The subscription ID is stored in `_subscriptions[session_id]` for lifecycle management.

### Dispatch Threading Model

`_dispatch()` runs **on the event bus thread** — it must return fast. It performs matching (in-process) and submits matching rules to the `ThreadPoolExecutor`. `_run_action()` runs on worker threads — it can block (I/O, subprocess) without affecting event delivery.

### Script Loading

Scripts are loaded via `shared.script_loader.load_script_symbol(resolved_path, symbol="execute", module_prefix="_jaato_reactor")`. The module prefix prevents naming collisions between reactor scripts and other loaded modules. Each invocation reloads the module, so script changes take effect on the next event (no restart needed for script edits).

### The `completeStepWithOutput` Bridge

When a subagent calls `completeStepWithOutput(step_id="final", output={passed: true, ...})`, the subagent plugin emits `AgentCompletedEvent` with `payload=output`. This flows through the same event bus path:

```
Subagent.completeStepWithOutput(output={...})
  → subagent plugin emits AgentCompletedEvent(payload={...})
    → EventBus publishes event
      → Reactor._dispatch() matches on "agent.completed"
        → Action script receives the typed payload in the merged view
```

This is how reactor-based handoffs integrate with plan-coordinated delegation: the parent's dependency-tracking step unblocks when the subagent completes, and the reactor independently forks a new session for the next phase.

### Presentation Context Inheritance

All fork actions (`fork_from_originating`, `fork_from_session`, `fork_from_waypoint`) and `create_session` **inherit the originating session's `PresentationContext`** by default. This means reactor-spawned sessions render in the same display surface (TUI, web dashboard, etc.) as the parent. Pass an explicit `presentation` argument to override.

---

## 14. Source Code Map

### jaato-premium Reactor Package

| File | Lines | Contents |
|------|-------|----------|
| `reactors/__init__.py` | 7 | Package docstring |
| `reactors/engine.py` | 160 | `ReactorEngine` — rule loading, event dispatch, worker pool |
| `reactors/rules.py` | 97 | `Rule`, `MatchSpec`, `ActionSpec` — data models, parsing, merge |
| `reactors/matcher.py` | 42 | `build_merged_view()`, `matches_where()` — JMESPath matching |
| `reactors/templating.py` | 81 | `substitute_params()` — `${event.*}` / `${env.*}` expansion |
| `reactors/action_context.py` | 524 | `ActionContext` — fork, inject, spawn, webhook, shell, emit |
| `reactors/extension.py` | 44 | `ReactorExtension` — daemon entry point, lifecycle |
| `reactors/watcher.py` | 79 | `FileWatcher` — mtime-poll hot-reload |
| `reactors/tests/test_engine.py` | 113 | End-to-end dispatch tests |
| `reactors/tests/test_action_context.py` | 475 | Tests for all ActionContext methods |
| `reactors/tests/test_matcher.py` | 65 | Tests for merged view and JMESPath |
| `reactors/tests/test_rules.py` | 108 | Tests for parsing, validation, merge |
| `reactors/tests/test_templating.py` | 74 | Tests for template substitution |
| `reactors/tests/fixtures/sample_script.py` | 6 | Sample action script for tests |

### jaato-sdk Event Infrastructure

| File | Lines | Contents |
|------|-------|----------|
| `jaato_sdk/event_bus.py` | 249 | `EventType` enum, `Event`, `EventFilter`, `Subscription` |
| `jaato_sdk/event_payloads.py` | 423 | TypedDict payload schemas for all event types |
| `jaato_sdk/events.py` | 1869 | Server event dataclasses (`AgentCompletedEvent`, etc.) |

### jaato-server Completion Schema Infrastructure

| File | Lines | Contents |
|------|-------|----------|
| `shared/completion_schema_loader.py` | 128 | `resolve_completion_schema()` — 3-tier path resolution |
| `shared/lifecycle_tools.py` | 198 | `LifecycleTools` — `signal_completion` with typed payloads |
| `shared/plugins/subagent/config.py` | 1530 | `SubagentProfile` dataclass with `completion_payload_schema` field |
| `shared/plugins/subagent/tests/test_profile_inheritance.py` | — | Tests for `completion_payload_schema` inheritance |

### External Dependencies

| Package | Purpose |
|---------|---------|
| `jmespath` | JMESPath expressions for `where` clauses and `${event.*}` params |
| `jsonschema` | Server-side validation of typed completion payloads |
