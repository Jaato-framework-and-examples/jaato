# SDK protocol versioning

The wire protocol between SDK clients and the jaato daemon is versioned
independently of any package version. This document describes the
versioning scheme, when to bump, and how clients use it.

## Two versions, one is for compat

| Field | Where | What it means | Used by compat check? |
|---|---|---|---|
| `protocol_version` | `ConnectedEvent.protocol_version` | Wire-protocol semver — bumped intentionally when shapes change | **Yes** |
| `server_version` | `ConnectedEvent.server_info.server_version` | Daemon's package version (`importlib.metadata.version("jaato-server")`) | **No** — diagnostics only |

The package version (`server_version`) tells the operator *which build
of the daemon is running*. It bumps for reasons unrelated to wire
compatibility (bug fixes, internal refactors, dependency updates), so
pinning a client against it produced false alarms — bump the package
to fix a memory leak, every client suddenly reports "incompatible" until
they re-pin.

The wire-protocol version (`protocol_version`) tells the operator
*whether the wire shapes the client understands match what the daemon
speaks*. It only bumps when an SDK consumer would actually need to
care.

## Bump policy

`protocol_version` follows simplified semver: `"MAJOR.MINOR"`.

### Major bump (`1.0 → 2.0`)

A field on the wire was **removed, renamed, or retyped**. An old
client cannot read or send the new shape. Examples:
- A flat field `event.prompt_tokens` becomes nested
  `event.usage.prompt_tokens` (gap 3 would have triggered this had we
  not declared the post-gap-3 state as the new 1.0 baseline).
- A field changes from `str` to `Dict[str, Any]`.
- An event type is removed or renamed.

### Minor bump (`1.0 → 1.1`)

A new **optional** field was added. Old clients ignore it (pydantic
`extra='ignore'` on `Event` already drops unknown fields silently); new
clients can rely on the field being present *if* the server's minor
is at least where the field was introduced. Examples:
- `CommandRequest.payload` was added in gap 4 — that would warrant a
  minor bump on a future change.
- `UsageBreakdown.cost_usd` and `reasoning_tokens` and `thinking_tokens`
  were all introduced in gap 3 — same story.

### Compat algorithm

The client carries a `MIN_PROTOCOL_VERSION` constant. On connect, it
compares the daemon's `protocol_version` against its minimum:

```
if server.major != client.major:           REFUSE   (wire shapes incompatible)
if server.minor <  client.required_minor:  REFUSE   (daemon missing fields client needs)
otherwise:                                 OK
```

The "server minor higher than client" case is fine — the client just
won't see any field that was added after its minimum.

## Worked examples

| Client requires | Server is | Outcome | Why |
|---|---|---|---|
| `1.0` | `1.0` | ✅ connect | Exact match |
| `1.0` | `1.5` | ✅ connect | Server has more fields, client ignores them |
| `1.2` | `1.0` | ❌ refuse | Server lacks fields client expects (e.g. `payload`) |
| `1.0` | `2.0` | ❌ refuse | Server speaks 2.x wire shapes; client only knows 1.x |
| `2.0` | `1.5` | ❌ refuse | Different major in either direction |

## CHANGELOG

### 1.0 — current baseline (post gap 1-4)

Initial wire-protocol contract following the daruma-operate integration
gaps. Includes:

- Subscribe API typed handlers (gap 1) — no wire shape changes, but
  consumer-facing pattern documented
- `SessionProfilesEvent` typed `ProfileSummary` shape with separate
  `parse_errors` field (gap 2)
- `CommandRequest.payload` for SDK-only structured payload (gap 4)
- `UsageBreakdown` carrying token counts + cost on
  `TurnCompletedEvent`, `TurnProgressEvent`, `ContextUpdatedEvent` (gap 3)
- `GCConfigEvent` extracted from `ContextUpdatedEvent` (gap 3)

The historical `protocol_version="1.0"` field that existed pre-gap-1
was a placeholder that never bumped through three breaking shape
changes. We declared the post-gap-3 state as the new 1.0 rather than
backdating bumps — older daemons against this baseline are simply
incompatible (see the gap 1-4 deployment plan).

## How to bump

When you change a wire shape:

1. Decide: major or minor (see policy above).
2. Bump `PROTOCOL_VERSION` in `jaato-sdk/jaato_sdk/events.py`.
3. The TS const mirror (`MIN_PROTOCOL_VERSION` exported in
   `jaato-sdk-ts/src/client.ts`) is hand-maintained — update it in
   the same PR, with the *minimum required* (typically the same value
   you set the server to, unless the change is server-only).
4. Update `MIN_PROTOCOL_VERSION` in any client that needs the new
   field/shape (the TUI's class constant, the TS SDK's export, etc.).
5. Add an entry to the CHANGELOG above.
6. Regenerate event baselines (`REGENERATE_EVENT_BASELINES=1 pytest`).

## Why hand-maintain TS, not codegen it?

The Python `PROTOCOL_VERSION` is the source of truth on the server
side. The TS `MIN_PROTOCOL_VERSION` is the *client's* requirement —
it's deliberately set by humans after deciding "the new field has
landed in TS, we should refuse to talk to daemons missing it". Bumping
on both sides is intentional, not mechanical, so codegen would obscure
the decision rather than enforce it.
