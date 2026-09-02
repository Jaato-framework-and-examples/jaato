# Webhook Plugin

The webhook plugin provides an inbound HTTP listener that receives webhooks from external services (GitHub, Slack, Jira, etc.) and makes them available to agent sessions via subscribe/poll tools. It enables long-running daemon sessions that react to external events.

## Overview

The plugin provides three capabilities:

1. **HTTP listener** — receives webhook POSTs with optional HMAC verification
2. **Event subscription** — model subscribes to events from specific sources
3. **Long-poll delivery** — model polls for events, blocking until one arrives

The HTTP server starts lazily on first subscribe call — no ports are bound for sessions that don't use webhooks.

## Tools

All tools have `discoverability="discoverable"` (loaded on demand).

| Tool | Description | Auto-approved |
|------|-------------|---------------|
| `webhook_subscribe` | Subscribe to webhook events, starts HTTP listener | No |
| `webhook_poll` | Long-poll for new events on a subscription | No |
| `webhook_status` | Check listener status and event stats | Yes |

### `webhook_subscribe`

Call once at session start. Starts the HTTP server and returns a subscription ID.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sources` | string[] | No | Filter by route names. Empty = all sources. |

**Response:**
```json
{
  "subscription_id": "abc123",
  "message": "Subscribed to webhook events from sources: all",
  "endpoints": [
    { "source": "github", "url": "http://127.0.0.1:9100/webhook/github" }
  ]
}
```

### `webhook_poll`

Call in a loop after subscribing. Blocks up to `timeout` seconds waiting for events.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subscription_id` | string | Yes | From `webhook_subscribe` |
| `timeout` | number | No | Max seconds to wait (1-30, default 15) |

**Response:**
```json
{
  "events": [
    {
      "event_id": "evt_abc123def456",
      "source": "github",
      "event_type": "push",
      "timestamp": "2026-03-05T10:30:00Z",
      "headers": { "X-GitHub-Event": "push" },
      "payload": { "ref": "refs/heads/main", "commits": [] }
    }
  ],
  "cursor": "evt_abc123def456"
}
```

### `webhook_status`

**Response:**
```json
{
  "listening": true,
  "host": "127.0.0.1",
  "port": 9100,
  "tls_enabled": true,
  "routes": [
    { "name": "github", "path": "/webhook/github", "events_received": 42 }
  ],
  "total_events_received": 42,
  "active_subscriptions": 1,
  "total_events_published": 42,
  "ip_allowlist_size": 3,
  "requests_blocked_ip": 0,
  "rate_limit_per_second": 50,
  "requests_blocked_rate": 2
}
```

## Configuration

### Config File: `.jaato/webhook.json`

```json
{
  "port": 9100,
  "host": "127.0.0.1",
  "secret": "${WEBHOOK_SECRET}",
  "tls": {
    "enabled": true,
    "certfile": "/etc/ssl/webhook.pem",
    "keyfile": "/etc/ssl/webhook-key.pem",
    "ca_certfile": "/etc/ssl/corporate-ca.pem"
  },
  "allowed_ips": ["10.0.0.0/8", "172.16.0.0/12"],
  "rate_limit_per_second": 50,
  "routes": {
    "github": {
      "path": "/webhook/github",
      "secret_header": "X-Hub-Signature-256",
      "secret_algo": "hmac-sha256",
      "event_type_header": "X-GitHub-Event",
      "metadata": { "source": "github" }
    },
    "generic": {
      "path": "/webhook",
      "allow_unauthenticated": true
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

Each layer is deep-merged, not replaced — a profile can override just `port` while inheriting routes from the workspace config.

### Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | int | `9100` | HTTP listener port |
| `host` | str | `127.0.0.1` | Bind address (localhost only by default) |
| `secret` | str | `null` | Global HMAC secret. Use `${ENV_VAR}` syntax. |
| `routes` | object | `{}` | Named routes. **No default route** — empty means the listener 404s every path (fail-closed; a zero-config open endpoint was removed). |
| `max_body_size` | int | `1048576` | Maximum request body in bytes (1 MB) |
| `response_timeout` | float | `5.0` | Seconds before responding to sender |
| `tls` | object | `{"enabled": false}` | TLS/SSL configuration (see below) |
| `allowed_ips` | string[] | `[]` | IP/CIDR allowlist. Empty = all allowed. |
| `rate_limit_per_second` | float | `0` | Per-IP rate limit. 0 = unlimited. |

### Route Configuration

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `path` | str | Yes | URL path (must start with `/`) |
| `secret_header` | str | No | Header containing HMAC signature |
| `secret_algo` | str | No | Algorithm — only `hmac-sha256` supported |
| `event_type_header` | str | No | Header to extract event type from |
| `metadata` | object | No | Static metadata merged into every event |
| `allow_unauthenticated` | bool | No | Accept **unsigned** requests on this route (default `false`, fail-closed). A route with no `secret_header` is refused unless mutual TLS or an IP allowlist is configured, or this is set. |

> **Authentication (fail-closed).** A route is accepted only when it is
> authenticated by one of: an HMAC secret (`secret_header` + `secret_algo`),
> mutual TLS (`tls.ca_certfile` set), a non-empty `allowed_ips` allowlist, or an
> explicit `allow_unauthenticated: true`. A matched route with none of these
> returns **401** — an untrusted caller can never drive agent sessions through an
> unsigned endpoint left open by omission.

### TLS Configuration

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `enabled` | bool | No | Enable HTTPS (default: false) |
| `certfile` | str | When enabled | Path to PEM certificate (or chain) |
| `keyfile` | str | When enabled | Path to PEM private key |
| `ca_certfile` | str | No | CA cert for mutual TLS (client certs) |

### Environment Variable Support

All string config values support `${VAR}` expansion:

```json
{
  "secret": "${WEBHOOK_SECRET}",
  "tls": {
    "certfile": "${WEBHOOK_CERT_PATH}",
    "keyfile": "${WEBHOOK_KEY_PATH}"
  }
}
```

## Corporate / Enterprise Hardening

The plugin includes three security features for corporate deployment, all using Python stdlib (no external dependencies):

### TLS/SSL

Enable HTTPS by setting `tls.enabled: true` with cert/key paths. For mutual TLS (client certificate verification), also set `ca_certfile`:

```json
{
  "tls": {
    "enabled": true,
    "certfile": "/etc/ssl/webhook.pem",
    "keyfile": "/etc/ssl/webhook-key.pem",
    "ca_certfile": "/etc/ssl/corporate-ca.pem"
  }
}
```

When `ca_certfile` is set, clients must present a valid certificate signed by that CA. This is useful for service-to-service authentication in zero-trust networks.

### IP Allowlisting

Restrict which IPs can send webhooks. Supports individual IPs, CIDR ranges, and IPv6:

```json
{
  "allowed_ips": [
    "192.168.1.0/24",
    "10.0.0.5",
    "::1",
    "2001:db8::/32"
  ]
}
```

- Empty list (default) allows all IPs
- IPv4-mapped IPv6 addresses (`::ffff:1.2.3.4`) are normalized for matching
- Blocked requests receive 403

### Rate Limiting

Per-IP token-bucket rate limiting prevents abuse:

```json
{
  "rate_limit_per_second": 50
}
```

- `0` (default) = unlimited
- Each source IP gets its own bucket
- Excess requests receive 429

### Request Security Pipeline

Security checks are applied in order for every POST:

1. IP allowlist (if configured) → 403
2. Rate limit (if configured) → 429
3. Route matching → 404
4. Body size limit → 413
5. Content-Type check → 415
6. HMAC signature verification (per-route) → 403
7. JSON body parsing → 400

## Usage

### With Agent Profile

Create a daemon session profile in `.jaato/profiles/github-watcher.json`:

```json
{
  "name": "github-watcher",
  "description": "Daemon session that reacts to GitHub webhook events",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "plugins": ["webhook(preload)", "cli", "file_edit"],
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
  "system_instructions": "You are a GitHub automation daemon. Call webhook_subscribe with sources=['github'], then subscribeToTasks(event_types=['external_event']) to receive events automatically. Process each event as it arrives. No polling needed.",
  "gc": { "type": "budget", "threshold_percent": 75.0 }
}
```

Launch via TUI:
```
session.new github-daemon --profile github-watcher
> Start watching for GitHub events.
```

### Programmatic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_tool("webhook", config={
    "port": 9100,
    "routes": {
        "github": {
            "path": "/webhook/github",
            "secret_header": "X-Hub-Signature-256",
            "secret_algo": "hmac-sha256",
            "event_type_header": "X-GitHub-Event",
        }
    }
})
```

## Architecture

```
shared/plugins/webhook/
├── __init__.py          # PLUGIN_KIND = "tool", create_plugin()
├── plugin.py            # WebhookPlugin — tool plugin with event buffers
├── http_server.py       # HTTPServer in daemon thread, TLS, IP/rate checks
├── config.py            # WebhookConfig, TLSConfig, RouteConfig, loading/merging
├── routes.py            # Route matching, HMAC-SHA256 verification, body parsing
└── tests/
    ├── test_plugin.py       # Plugin protocol, subscribe/poll/status integration
    ├── test_http_server.py  # HTTP server, IP allowlist, rate limiting
    ├── test_config.py       # Config loading, merging, validation, TLS config
    └── test_routes.py       # Route matching, HMAC verification
```

### Key Design Decisions

- **Lazy server start** — HTTP server starts on first `webhook_subscribe`, not on plugin init
- **Per-subscription buffers** — each subscription gets its own `deque(maxlen=1000)` for FIFO eviction
- **Threading.Event for long-poll** — efficient blocking without busy-wait
- **Source filtering** — subscriptions can filter by route name to receive only relevant events
- **Stdlib only** — `http.server`, `ssl`, `ipaddress`, `hmac` — no external deps

## Dependencies

None. The plugin uses only Python standard library modules.

## Tests

104 tests covering:
- Config loading, merging, precedence, validation (including TLS and security fields)
- Route matching and HMAC-SHA256 verification
- HTTP server start/stop, POST handling, body size limits
- IP allowlisting (single IP, CIDR, IPv6, IPv4-mapped IPv6, integration)
- Rate limiting (token bucket, per-IP, refill, integration)
- Plugin protocol compliance, subscribe/poll/status tools
- End-to-end: subscribe → POST webhook → poll → receive event
- Source filtering across multiple routes

```bash
pytest jaato-server/shared/plugins/webhook/tests/ -v
```
