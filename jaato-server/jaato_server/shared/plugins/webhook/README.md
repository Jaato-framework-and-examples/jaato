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
    "gitlab": {
      "path": "/webhook/gitlab",
      "secret_header": "X-Gitlab-Token",
      "secret_algo": "token",
      "event_type_header": "X-Gitlab-Event",
      "metadata": { "source": "gitlab" }
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
| `secret` | str | `null` | Global shared secret — the HMAC key under `hmac-sha256`, the expected header value under `token`. Overridden per route by `routes.<name>.metadata.secret`. Use `${ENV_VAR}` syntax. |
| `routes` | object | `{}` | Named routes. **No default route** — empty means the listener 404s every path (fail-closed; a zero-config open endpoint was removed). |
| `max_body_size` | int | `1048576` | Maximum request body in bytes (1 MB) |
| `response_timeout` | float | `5.0` | Seconds before responding to sender |
| `tls` | object | `{"enabled": false}` | TLS/SSL configuration (see below) |
| `allowed_ips` | string[] | `[]` | IP/CIDR allowlist. Empty = all allowed. |
| `rate_limit_per_second` | float | `0` | Per-IP rate limit. 0 = unlimited. |
| `replay_cache_size` | int | `10000` | Delivery keys the listener remembers for replay refusal. One cache serves every route. |

### Route Configuration

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `path` | str | Yes | URL path (must start with `/`) |
| `secret_header` | str | No | Header carrying the route's credential — an HMAC digest, or the shared secret itself under `secret_algo: "token"` |
| `secret_algo` | str | No | How that header is verified: `hmac-sha256` (preferred) or `token` (weaker — see below) |
| `signature_scheme` | str | No | How the signed payload is **constructed**: `body` (default), `slack-v0`, `stripe-v1` — see [Replay protection](#replay-protection) |
| `timestamp_header` | str | No | Header carrying the signed timestamp (`slack-v0` only). Refused on `stripe-v1` and `body`. |
| `max_age_seconds` | int | No | Freshness window, default `300`. `0` disables it (announced at WARNING). |
| `replay_key_header` | str | No | Header carrying a unique delivery id (`X-GitHub-Delivery`, `X-Gitlab-Event-UUID`) used to dedupe repeats |
| `event_type_header` | str | No | Header to extract event type from |
| `metadata` | object | No | Static metadata merged into every event. `metadata.secret` overrides the global `secret` for this route. |
| `allow_unauthenticated` | bool | No | Accept **unsigned** requests on this route (default `false`, fail-closed). A route with no `secret_header` is refused unless mutual TLS or an IP allowlist is configured, or this is set. |

`secret_header` and `secret_algo` are a pair: declaring one without the other is
a **500**, never a silent downgrade to unsigned, and an unrecognised
`secret_algo` is a hard config-validation error. A typo can't skip verification.

#### `secret_algo: "hmac-sha256"` — the default choice

The header carries an HMAC-SHA256 digest keyed by the shared secret. Under the
default `signature_scheme: "body"` the digest covers the request body,
optionally prefixed `sha256=` (GitHub convention). The secret never travels, and
a captured request cannot be replayed against a *different* body. Use this
whenever the producer signs bodies — GitHub, Stripe, Slack and most large
senders do.

> It **can** be replayed against the *same* body, indefinitely, unless the route
> also configures replay protection. `secret_algo` says how the credential is
> checked; it cannot say *what was signed*, and a digest over a body alone binds
> no time. See [Replay protection](#replay-protection).

#### `secret_algo: "token"` — for producers that don't sign

Some producers ship the shared secret **verbatim** in a header and expect a
constant-time comparison. GitLab is the canonical case: it sends the configured
secret in `X-Gitlab-Token` and signs nothing. Without this mode the only way to
ingest such a webhook was `allow_unauthenticated: true` — throwing away a secret
that was right there in the request.

```json
"gitlab": {
  "path": "/webhook/gitlab",
  "secret_header": "X-Gitlab-Token",
  "secret_algo": "token",
  "event_type_header": "X-Gitlab-Event"
}
```

> **`token` is weaker than `hmac-sha256`, not a peer of it.** The secret is
> present in **every** request, so it is readable by anything that terminates
> TLS (a load balancer, a reverse proxy, an ingress controller, a logging
> sidecar that records headers), and a captured request replays forever against
> any payload — the body is not covered. **Pair it with TLS** (`tls.enabled`),
> keep it off shared ingress paths where you can, and rotate the secret on any
> suspicion of exposure. Reach for it only when the producer gives you no
> signature to check; if it signs bodies, use `hmac-sha256`.
>
> Every `token` route logs a **WARNING at listener startup** naming the route
> and header, and a louder one when TLS is off — the same posture as
> `--ws-unsafe-no-auth` and `scrub_secret_env: none`. The mode is deliberately
> not the quiet path of least resistance.
>
> **Replay protection in `token` mode** is `replay_key_header` and nothing else.
> There is no signed payload, so no timestamp can be bound into anything; and
> the credential is byte-identical on every request, so a cache keyed on it
> would refuse the second *legitimate* delivery rather than a replay. A delivery
> id (`X-Gitlab-Event-UUID`) refuses a verbatim replay within the TTL. It does
> not stop an attacker who *holds* the token from minting fresh requests with
> new ids — that is credential compromise, which this mode concedes by design.

> **Authentication (fail-closed).** A route is accepted only when it is
> authenticated by one of: the route's shared secret (`secret_header` +
> `secret_algo`, in either mode), mutual TLS (`tls.ca_certfile` set), a non-empty
> `allowed_ips` allowlist, or an explicit `allow_unauthenticated: true`. A
> matched route with none of these returns **401** — an untrusted caller can
> never drive agent sessions through an unsigned endpoint left open by omission.

### Replay protection

A signature over the request body alone authenticates the same bytes **forever**.
Anyone who observes one delivery — a proxy log, a mirrored port, a misrouted
retry — can replay it verbatim, indefinitely, and it authenticates every time.
On a listener whose purpose is to drive agent sessions that is not a duplicate
row: it is a re-triggered turn, with tool calls, ledger spend and whatever side
effects the persona authorises.

Two mechanisms close it, and they are complements rather than alternatives.

**1. A freshness window** — `signature_scheme` + `max_age_seconds`. Only a
scheme that binds the timestamp **into the signature** can carry one; a
timestamp the signature does not cover is rewritten by whoever is replaying the
request, so checking it would be theatre. `timestamp_header` on
`signature_scheme: "body"` is therefore a config error and a 500, not a warning.

| `signature_scheme` | Signed payload | Timestamp from | Senders |
|---|---|---|---|
| `body` (default) | the request body | — (binds none) | GitHub, GitLab, most internal senders |
| `slack-v0` | `v0:{ts}:{body}` | `timestamp_header` | Slack |
| `stripe-v1` | `{ts}.{body}` | `t=` inside the signature header | Stripe |

**2. A replay cache** — bounded, TTL'd to the window, keyed per delivery. The
timestamp alone cannot catch a replay *inside* the window, because every copy
carries the same signed timestamp and is equally fresh. This is the part that
does. Only an authenticated, fresh delivery is ever recorded, so nobody can
poison the cache with a guessed id to have the genuine delivery refused. A
repeat answers **409**, distinct from 403 so a repeat and a forgery are
distinguishable in the access log.

```json
"slack": {
  "path": "/webhook/slack",
  "secret_header": "X-Slack-Signature",
  "secret_algo": "hmac-sha256",
  "signature_scheme": "slack-v0",
  "timestamp_header": "X-Slack-Request-Timestamp",
  "max_age_seconds": 300
},
"stripe": {
  "path": "/webhook/stripe",
  "secret_header": "Stripe-Signature",
  "secret_algo": "hmac-sha256",
  "signature_scheme": "stripe-v1"
},
"github": {
  "path": "/webhook/github",
  "secret_header": "X-Hub-Signature-256",
  "secret_algo": "hmac-sha256",
  "replay_key_header": "X-GitHub-Delivery",
  "event_type_header": "X-GitHub-Event"
}
```

#### What each route shape actually gets

| Route | Freshness window | Replay cache |
|---|---|---|
| `slack-v0` / `stripe-v1` | yes, on by default (300s) | yes, keyed on the signature — automatic, no extra config |
| `body` + `replay_key_header` | none available | yes, keyed on the delivery id |
| `body`, no delivery id | none available | **none** — a replay is accepted |
| `token` + `replay_key_header` | none available | yes, keyed on the delivery id |
| `token`, no delivery id | none available | **none** |

**The last-but-one row is where every route configured before this feature
sits**, and it is a deliberate choice rather than an oversight: the protection
could not be switched on for them, because the sender signs no timestamp and
there is nothing to check. What is switched on instead is *saying so* — each
such route logs a **WARNING at listener startup** naming itself and both
remedies. The cost is that the weakness persists for a deployment that reads no
logs and changes nothing; the alternative was to break every existing route.

Where the material *does* exist, the safe posture is the default: configuring
`slack-v0` or `stripe-v1` gets a 300-second window and a replay cache with no
further keys, and turning the window off (`max_age_seconds: 0`) is the explicit,
WARNING-announced act.

#### Fail-closed rules

Widening the vocabulary widens what a route may **say**, never what it may omit.
Each of these is a `validate_config` **error** and a **500** at request time —
never a fall-through to a weaker check:

- a `signature_scheme` outside the vocabulary (a typo does not fall back to `body`)
- `slack-v0` with no `timestamp_header`
- `stripe-v1` **with** a `timestamp_header` (its timestamp is in the signature header; a second source could disagree with the signed one)
- `timestamp_header` on `body` (the value is not signed)
- a timestamp-bound scheme with `secret_algo` other than `hmac-sha256`

A timestamp that is present but unparseable is a **refusal**, not a skipped
check. And no scheme accepts another scheme's construction: `slack-v0` rejects a
bare hex digest and an HMAC over the body alone, and `stripe-v1` ignores Stripe's
legacy `v0` test-mode signature entirely, per Stripe's own downgrade guidance.

Every credential comparison — both `secret_algo` modes and both schemes — goes
through `hmac.compare_digest`, via one helper (`constant_time_equals`) that
encodes to UTF-8 first so a crafted non-ASCII header cannot turn a verification
failure into a 500.

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
from jaato_server.shared.plugins.registry import PluginRegistry

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
pytest jaato-server/jaato_server/shared/plugins/webhook/tests/ -v
```
