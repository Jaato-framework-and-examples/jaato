# Webhook Plugin — Webhook-Driven Daemon Sessions

## Overview

The Webhook plugin enables **long-running daemon agent sessions** that react to
external events delivered via HTTP webhooks. A daemon session is just a regular
`JaatoSession` created with a profile whose system prompt instructs it to
subscribe to events and process them eternally.

There is no new "daemon mode" concept — the plugin provides two capabilities:

1. **Inbound HTTP listener** — receives webhooks and publishes them to the
   `TaskEventBus`.
2. **Event subscription tools** — the model subscribes to bus events and
   long-polls for new ones.

The session never "completes." It stays idle between events and wakes up when a
webhook arrives via the event bus.

```
External Service                  Webhook Plugin               TaskEventBus
(GitHub, Slack, etc.)             (HTTP listener)              (shared singleton)
        │                                │                            │
        │  POST /webhook/github          │                            │
        ├───────────────────────────────►│                            │
        │                                │  publish(WebhookEvent)     │
        │                                ├───────────────────────────►│
        │                                │                            │
        │                   200 OK       │                            │
        │◄───────────────────────────────┤                            │
        │                                │                            │
                                                                      │
                                              Daemon Session          │
                                              (webhook_subscribe)     │
                                                      │               │
                                                      │  wait_for_events()
                                                      │◄──────────────┤
                                                      │               │
                                                      │  [WebhookEvent]
                                                      ├──processes────┤
                                                      │               │
                                                      │  wait_for_events()
                                                      │◄──────────────┤
                                                      │   (blocks)    │
```

## Motivation

Jaato has **internal pub/sub** (`TaskEventBus` for cross-agent task
coordination) and **daemon extensions** (external packages hook into the server
lifecycle), but no way to receive events from the outside world. The TODO
plugin's `WebhookReporter` sends progress updates to an HTTP endpoint, but
that's scoped to plan/step lifecycle reporting — not a general-purpose webhook
system.

The Webhook plugin is the first real webhook infrastructure in the codebase. It
adds an HTTP ingress that receives external events and feeds them into the
existing `TaskEventBus`, enabling event-driven agent sessions.

## Distribution: Public Plugin, Premium Profiles

The **plugin itself** lives in the public codebase (`shared/plugins/webhook/`).
It's a straightforward HTTP→EventBus bridge built on stdlib — no proprietary
logic, and all its dependencies (`TaskEventBus`, config precedence, plugin base)
are already public. Gating it behind premium would feel wrong given that more
complex plugins (`interactive_shell`, `service_connector`) are open.

**Premium ships curated profiles** — production-ready daemon session profiles
with battle-tested system prompts, route configurations, and HMAC setups for
specific integrations:

- `github-watcher` — PR review, commit summarization, issue triage
- `slack-responder` — channel monitoring, thread responses, slash commands
- `jira-triager` — issue classification, priority assignment, label management

These profiles encode operational knowledge (what events to subscribe to, how to
process them, what tools to use) that enterprise users pay for. The bare plugin
gives you the plumbing; premium gives you the recipes.

## Design Principles

1. **No shared ports.** Each plugin instance owns its own HTTP listener port,
   configured per-instance — never shared across the server or other plugins.
2. **Standard config precedence.** Configuration is loaded in order:
   profile `plugin_configs` → workspace `.jaato/webhook.json` →
   user `~/.jaato/webhook.json` → built-in defaults.
3. **Bus-mediated delivery.** Webhooks publish to `TaskEventBus`; sessions
   subscribe via tools. This decouples ingress from session lifecycle and
   enables fan-out to multiple sessions.
4. **The model drives the loop.** The session's system prompt instructs it to
   call `webhook_subscribe` then loop on `webhook_poll`. The plugin doesn't
   inject messages or force turns — the model is in control.

## Configuration

### Config File: `webhook.json`

```json
{
  "port": 9100,
  "host": "127.0.0.1",
  "secret": "${WEBHOOK_SECRET}",
  "routes": {
    "github": {
      "path": "/webhook/github",
      "secret_header": "X-Hub-Signature-256",
      "secret_algo": "hmac-sha256",
      "event_type_header": "X-GitHub-Event",
      "metadata": { "source": "github" }
    },
    "slack": {
      "path": "/webhook/slack",
      "secret_header": "X-Slack-Signature",
      "secret_algo": "hmac-sha256",
      "event_type_header": null,
      "metadata": { "source": "slack" }
    },
    "generic": {
      "path": "/webhook",
      "allow_unauthenticated": true,
      "metadata": { "source": "generic" }
    }
  },
  "max_body_size": 1048576,
  "response_timeout": 5.0
}
```

### Config Precedence

```
1. Profile plugin_configs.webhook   (highest — per-session override)
2. <workspace>/.jaato/webhook.json  (project-level)
3. ~/.jaato/webhook.json            (user-level)
4. Built-in defaults                (lowest)
```

Each layer is merged, not replaced — a profile can override just `port` while
inheriting routes from the workspace config.

### Built-in Defaults

| Key | Default | Description |
|-----|---------|-------------|
| `port` | `9100` | HTTP listener port |
| `host` | `127.0.0.1` | Bind address (localhost only by default) |
| `secret` | `null` | Global shared secret for HMAC verification |
| `routes` | `{}` | No default route (fail-closed — a zero-config open endpoint was removed; empty ⇒ all paths 404). Each unsigned route needs `allow_unauthenticated: true` or mTLS / an IP allowlist. |
| `max_body_size` | `1048576` (1 MB) | Maximum request body size |
| `response_timeout` | `5.0` | Seconds before responding to webhook sender |
| `tls` | `{"enabled": false}` | TLS/SSL config (`enabled`, `certfile`, `keyfile`, `ca_certfile`) |
| `allowed_ips` | `[]` | IP/CIDR allowlist. Empty = all allowed. |
| `rate_limit_per_second` | `0` | Per-IP rate limit (token bucket). 0 = unlimited. |

### Environment Variable Support

Config values support `${VAR}` expansion (via existing `expand_variables()`):

- `${WEBHOOK_SECRET}` — webhook verification secret
- `${WEBHOOK_PORT}` — listener port
- `${workspaceRoot}` — workspace path (for file-based config paths)

## Plugin Architecture

### Module Structure

```
shared/plugins/webhook/
├── __init__.py          # PLUGIN_KIND = "tool", create_plugin()
├── plugin.py            # WebhookPlugin — tool plugin with HTTP server
├── http_server.py       # Lightweight async HTTP server (http.server in thread)
├── config.py            # WebhookConfig dataclass, config loading/merging
├── routes.py            # Route matching, secret verification, body parsing
└── tests/
    ├── test_plugin.py
    ├── test_http_server.py
    ├── test_config.py
    └── test_routes.py
```

### Plugin Class

```python
class WebhookPlugin:
    """Webhook ingress plugin for daemon agent sessions.

    Starts a per-instance HTTP server that receives webhooks and publishes
    them to the TaskEventBus. Provides tools for the model to subscribe to
    and poll for events.

    Lifecycle:
        1. initialize(config) — load and merge config, but don't start server yet
        2. set_workspace_path(path) — resolve workspace-relative config paths
        3. First tool call or explicit start — spin up HTTP listener
        4. shutdown() — stop HTTP listener, clean up subscriptions

    The HTTP server runs in a background thread to avoid blocking the
    session's event loop. It uses the standard library's http.server for
    minimal dependencies.
    """

    PLUGIN_KIND = "tool"

    @property
    def name(self) -> str:
        return "webhook"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load config with standard precedence: explicit > workspace > user > defaults."""
        ...

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for event subscription and polling."""
        ...

    def get_executors(self) -> Dict[str, Any]:
        """Return tool executors."""
        ...

    def shutdown(self) -> None:
        """Stop HTTP server and clean up subscriptions."""
        ...
```

### Tool Schemas

The plugin exposes three tools, all `discoverability="discoverable"`:

#### `webhook_subscribe`

Subscribe to webhook events on the bus. Returns a subscription ID.

```json
{
  "name": "webhook_subscribe",
  "description": "Subscribe to incoming webhook events. Returns a subscription ID for polling. Call this once at session start.",
  "parameters": {
    "type": "object",
    "properties": {
      "sources": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter by source names (route keys from config). Empty = all sources."
      },
      "event_types": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter by event type (e.g., 'push', 'pull_request'). Empty = all types."
      }
    },
    "required": []
  }
}
```

**Returns:**
```json
{
  "subscription_id": "abc123",
  "message": "Subscribed to webhook events from sources: [github]. Listening on http://127.0.0.1:9100/webhook/github",
  "endpoints": [
    { "source": "github", "url": "http://127.0.0.1:9100/webhook/github" }
  ]
}
```

#### `webhook_poll`

Long-poll for events on an existing subscription. Blocks up to `timeout`
seconds. The model calls this in a loop.

```json
{
  "name": "webhook_poll",
  "description": "Poll for new webhook events. Blocks up to timeout seconds waiting for events. Call this in a loop after subscribing.",
  "parameters": {
    "type": "object",
    "properties": {
      "subscription_id": {
        "type": "string",
        "description": "Subscription ID from webhook_subscribe"
      },
      "timeout": {
        "type": "number",
        "description": "Max seconds to wait (1-30, default 15)"
      },
      "after_event_id": {
        "type": "string",
        "description": "Cursor — only return events after this ID (from last poll)"
      }
    },
    "required": ["subscription_id"]
  }
}
```

**Returns:**
```json
{
  "events": [
    {
      "event_id": "evt_abc123",
      "source": "github",
      "event_type": "push",
      "timestamp": "2026-03-05T10:30:00Z",
      "headers": { "X-GitHub-Event": "push", "X-GitHub-Delivery": "..." },
      "payload": { "ref": "refs/heads/main", "commits": [...] }
    }
  ],
  "cursor": "evt_abc123"
}
```

When no events arrive within the timeout, returns `{"events": [], "cursor": "..."}`.

#### `webhook_status`

Check the HTTP listener status, active subscriptions, and event stats.

```json
{
  "name": "webhook_status",
  "description": "Show webhook listener status, active routes, and event statistics.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**Returns:**
```json
{
  "listening": true,
  "host": "127.0.0.1",
  "port": 9100,
  "routes": [
    { "name": "github", "path": "/webhook/github", "events_received": 42 },
    { "name": "generic", "path": "/webhook", "events_received": 3 }
  ],
  "active_subscriptions": 1,
  "total_events_published": 45
}
```

## Event Bus Integration

### Event Type Extension

The plugin defines a new event type constant for the `TaskEventBus`:

```python
# In webhook/plugin.py or as extension to TaskEventType
WEBHOOK_EVENT_TYPE = "webhook_received"
```

Since `TaskEventBus` currently uses `TaskEventType` enum (from
`jaato_sdk/plugins/todo/models.py`), the Webhook plugin publishes events using
a **wrapper pattern** — it creates `TaskEvent` objects with a custom event type
that carries the webhook payload in the event's data fields.

### Publishing Flow

```python
def _on_webhook_received(self, route_name, event_type, headers, payload):
    """Called by HTTP server when a webhook is received."""
    event = TaskEvent.create(
        event_type=TaskEventType.CUSTOM,  # New enum value for plugin events
        agent_id=f"webhook:{route_name}",
        data={
            "source": route_name,
            "event_type": event_type,
            "headers": dict(headers),
            "payload": payload,
        }
    )
    bus = TaskEventBus.get_instance()
    bus.publish(event)
```

### Subscription Flow

The `webhook_subscribe` tool creates a `TaskEventBus` subscription filtered to
`CUSTOM` events from `webhook:*` agents:

```python
def _execute_subscribe(self, args):
    bus = TaskEventBus.get_instance()
    sources = args.get("sources", [])

    sub_id = bus.subscribe(
        subscriber_name=self._session_agent_id,
        filter=EventFilter(
            event_types=[TaskEventType.CUSTOM],
            agent_id_prefix="webhook:",
            # Further source filtering done in callback
        ),
        callback=lambda event: self._buffer_event(sub_id, event, sources),
        replay_history=False,  # Don't replay old webhooks
    )
    return {"subscription_id": sub_id, ...}
```

### Polling Flow

The `webhook_poll` tool uses the bus's `wait_for_events()` for efficient
long-polling:

```python
def _execute_poll(self, args):
    sub_id = args["subscription_id"]
    timeout = min(max(args.get("timeout", 15), 1), 30)
    cursor = args.get("after_event_id")

    events = self._poll_buffered_events(sub_id, timeout, cursor)
    return {
        "events": [self._format_event(e) for e in events],
        "cursor": events[-1].event_id if events else cursor,
    }
```

## HTTP Server

### Implementation: `http.server` in a Thread

The HTTP server uses Python's `http.server.HTTPServer` running in a dedicated
daemon thread. This avoids adding `aiohttp` as a dependency and keeps the
implementation minimal.

```python
class WebhookHTTPServer:
    """Lightweight HTTP server for receiving webhooks.

    Runs in a background daemon thread. Routes incoming POSTs to
    registered handlers after body-size checks and optional HMAC
    verification.

    Thread-safe: the handler callback is called from the server thread;
    it must be safe to call from any thread (TaskEventBus.publish() is).
    """

    def __init__(self, host, port, routes, on_webhook, max_body_size):
        ...

    def start(self) -> None:
        """Start the server in a background thread."""
        ...

    def stop(self) -> None:
        """Stop the server and join the thread."""
        ...
```

### Request Handling

1. **Method check** — only POST accepted (405 otherwise).
2. **Path matching** — match request path against configured routes.
3. **Body size check** — reject bodies exceeding `max_body_size` (413).
4. **Content-Type** — must be `application/json` (415).
5. **Secret verification** — if route has `secret_header` + `secret_algo`,
   verify HMAC signature. Reject with 403 on mismatch.
6. **Parse body** — JSON-decode the body.
7. **Extract event type** — from the header specified in `event_type_header`,
   or `"unknown"` if not configured.
8. **Publish** — call `on_webhook(route_name, event_type, headers, payload)`.
9. **Respond** — 200 OK with `{"status": "accepted"}`.

### Security

The HTTP server implements a layered security pipeline. All checks use
Python stdlib only — no external dependencies.

**Request pipeline** (checks applied in order):
1. IP allowlist → 403 Forbidden
2. Per-IP rate limit → 429 Too Many Requests
3. Route matching → 404 Not Found
4. Body size limit → 413 Payload Too Large
5. Content-Type check → 415 Unsupported Media Type
6. HMAC signature verification → 403 Forbidden
7. JSON body parsing → 400 Bad Request

**Network security:**
- **Bind to localhost by default** (`127.0.0.1`). Explicitly set `host` to
  `0.0.0.0` to accept external connections.
- **TLS/SSL** — optional HTTPS via `ssl.SSLContext`. Supports server-only TLS
  and mutual TLS (client certificate verification via `ca_certfile`).
- **IP allowlisting** — `allowed_ips` config with CIDR support via
  `ipaddress` module. IPv4-mapped IPv6 addresses (`::ffff:1.2.3.4`) are
  normalized. Empty list (default) allows all IPs.

**Application security:**
- **HMAC verification** per-route using HMAC-SHA256. Supports GitHub's
  `sha256=` prefix convention.
- **Body size limits** to prevent memory exhaustion (default 1 MB).
- **Per-IP rate limiting** — token-bucket algorithm. Configurable via
  `rate_limit_per_second` (default 0 = unlimited). Each source IP gets its
  own bucket with 1-second burst capacity.
- **No path traversal** — routes are matched as exact strings.
- **Secret via env var** — `${WEBHOOK_SECRET}` avoids hardcoding
  secrets in config files.

**Corporate deployment example:**
```json
{
  "host": "0.0.0.0",
  "secret": "${WEBHOOK_SECRET}",
  "tls": {
    "enabled": true,
    "certfile": "/etc/ssl/webhook.pem",
    "keyfile": "/etc/ssl/webhook-key.pem",
    "ca_certfile": "/etc/ssl/corporate-ca.pem"
  },
  "allowed_ips": ["10.0.0.0/8", "172.16.0.0/12"],
  "rate_limit_per_second": 50
}
```

## Daemon Session Profile

A daemon session is created with a profile that loads the Webhook plugin and
instructs the model to subscribe and loop:

### Example: `.jaato/profiles/github-watcher.json`

```json
{
  "name": "github-watcher",
  "description": "Daemon session that reacts to GitHub webhook events",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "plugins": ["webhook(preload)", "cli", "file_edit", "todo"],
  "plugin_configs": {
    "webhook": {
      "port": 9100,
      "routes": {
        "github": {
          "path": "/webhook/github",
          "secret_header": "X-Hub-Signature-256",
          "secret_algo": "hmac-sha256",
          "event_type_header": "X-GitHub-Event"
        }
      }
    }
  },
  "system_instructions": "You are a GitHub automation daemon. On startup, call webhook_subscribe with sources=['github']. Then loop forever calling webhook_poll. For each event, analyze it and take appropriate action:\n- push events: review the commits and summarize changes\n- pull_request events: review the PR diff and post feedback\n- issue events: triage and label the issue\nNever stop polling. After processing each batch of events, immediately call webhook_poll again.",
  "max_turns": 0,
  "gc": {
    "type": "budget",
    "threshold_percent": 75.0,
    "preserve_recent_turns": 3
  }
}
```

### Key Profile Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| `max_turns` | `0` | Unlimited — daemon runs forever |
| `webhook(preload)` | — | Load tools immediately, no discovery step |
| `gc.type` | `"budget"` | Proactive GC keeps context window healthy |
| `gc.preserve_recent_turns` | `3` | Keep recent event processing, discard old |

### Launching

```bash
# Start the jaato server daemon
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Create a daemon session from the profile
# (via TUI or programmatically via SDK)
```

Via SDK:
```python
client = await IPCClient.connect("/tmp/jaato.sock")
await client.create_session(profile="github-watcher")
await client.send_message("Start watching for GitHub events.")
```

Via TUI:
```
session.new github-daemon --profile github-watcher
> Start watching for GitHub events.
```

## Max Turns: `0` = Unlimited

The `SubagentProfile.max_turns` field currently defaults to `10`. For daemon
sessions, we need unlimited turns. Convention:

- `max_turns: 0` → no turn limit (daemon mode)
- `max_turns: N` (N > 0) → stop after N turns (current behavior)

This requires a small change in the subagent plugin's turn-counting logic to
skip the limit check when `max_turns == 0`. For main sessions created from
profiles (via `session.new --profile`), the server already doesn't enforce
`max_turns` — it's only enforced in the subagent execution loop.

## TaskEventType Extension

The `TaskEventType` enum needs a `CUSTOM` value for plugin-generated events.
Alternatively, the Webhook plugin can define its own event type string and use a
custom `EventFilter.matches()` predicate.

**Option A: Add `CUSTOM` to TaskEventType** (preferred — minimal, reusable):
```python
class TaskEventType(str, Enum):
    ...
    CUSTOM = "custom"  # Plugin-defined events
```

**Option B: Webhook-specific filter** (no SDK changes needed):
The plugin subscribes with a callback filter that checks `event.source_agent`
prefix instead of `event_type`. This works with the existing `EventFilter`
`agent_id` field.

Recommendation: **Option B** initially (zero SDK changes), migrate to Option A
if other plugins also need custom event types.

## Buffered Event Delivery

The plugin maintains per-subscription event buffers to decouple webhook arrival
from model polling:

```
Webhook arrives → TaskEventBus.publish() → subscription callback
    → appends to per-subscription buffer (thread-safe deque)

Model calls webhook_poll → drains buffer (up to limit)
    → returns events + cursor
    → if buffer empty, blocks on threading.Event for up to timeout
```

This ensures:
- Webhooks are acknowledged immediately (200 OK) regardless of model state
- Events queue up if the model is busy processing a previous batch
- Long-poll efficiently blocks without busy-waiting
- Buffer has a max size (default 1000) to prevent memory growth

## Error Handling

| Scenario | Behavior |
|----------|----------|
| HTTP server fails to bind | `webhook_subscribe` returns error with diagnostic |
| Webhook body too large | 413 response, event dropped |
| Invalid JSON body | 400 response, event dropped |
| HMAC verification fails | 403 response, event dropped |
| Subscriber not found on poll | Error result, suggest re-subscribing |
| Event buffer overflow | Oldest events dropped (FIFO eviction) |
| Model stops polling | Events buffer up to max, then evict oldest |
| HTTP server thread crashes | Logged, status tool reports `listening: false` |

## Lifecycle & Cleanup

1. **Plugin initialize** — config loaded and merged, server not yet started.
2. **First `webhook_subscribe` call** — HTTP server starts lazily (avoids
   binding ports for sessions that don't use webhooks).
3. **Session active** — server running, events flowing.
4. **Session GC** — old turns with processed events are garbage-collected;
   the subscription and server persist.
5. **Session stop / shutdown** — `shutdown()` stops HTTP server, unsubscribes
   all subscriptions from the bus, drains buffers.

## Testing Strategy

### Unit Tests

- **Config loading** — precedence merging, variable expansion, validation.
- **Route matching** — path matching, HMAC verification, content-type checks.
- **Event buffering** — buffer append, drain, overflow eviction, cursor logic.
- **Tool executors** — subscribe/poll/status return correct structures.

### Integration Tests

- **HTTP server** — start server, POST webhook, verify event on bus.
- **End-to-end** — subscribe → POST webhook → poll → receive event.
- **Concurrent webhooks** — multiple rapid POSTs, all events captured.
- **Secret verification** — valid/invalid HMAC signatures.

### No External Dependencies

Tests use `http.client` or `urllib.request` to POST to the test server.
No `requests` or `aiohttp` needed in tests.

## Future Extensions

- **Outbound actions** — `webhook_respond` tool for sending HTTP responses
  back to webhook sources (e.g., GitHub status checks).
- **Event filtering DSL** — JSONPath or jq-like expressions for payload filtering.
- **Webhook registration** — tools to dynamically add/remove routes at runtime.
- **Retry/dead-letter** — persistent queue for events that fail processing.
- **Multi-session fan-out** — multiple daemon sessions subscribing to the same
  webhook source with different event type filters.
- **WebSocket ingress** — accept events via WebSocket in addition to HTTP POST.
