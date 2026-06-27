# SDK convenience layer

**Status:** Phase 1 (this PR). **Audience:** jaato-sdk users + maintainers.

## Problem

The jaato SDK exposes only the low-level event-loop primitives. The common
path — "open a session, ask, get the answer" — costs ~10 lines of
`asyncio.Event` + `subscribe`/`subscribe_once` + `done.wait()` +
`getattr(e, "text", "")` plumbing, and that recipe is subtle enough that the
canonical scaffold template shipped an infinite-hang (PR #399: waiting on
`SESSION_TERMINATED` only, which a plain turn never emits). When the front-door
pattern can hang, it is too low-level to be the front door.

This is **accidental** complexity, not inherent to being a daemon client. The
daemon architecture makes ~1 line ("connect to a server") unavoidable; it does
not justify the event-loop boilerplate. The fix is a thin, **additive** facade
over `IPCClient` that owns the send-and-wait recipe so user code can't
reproduce it (or its hangs).

Non-goal: replacing or hiding the event API. Streaming, permissions, cascades,
and observers keep using `subscribe`/`events`/`cascade_events` exactly as today.

## Design principle: simple things simple, complex things possible

The facade is pure sugar over existing methods. Every existing method is
untouched. Both config styles the framework already supports are preserved
verbatim, because the facade **forwards `create_session`'s parameters
unchanged**:

- **declarative** — `profile="researcher"` (named profile in `.jaato/profiles/`),
  `agent="pirate"` (named persona in `.jaato/agents/`)
- **programmatic** — `profile={"model": ..., "provider": ...}` (inline spec dict)

`agent` composes with either `profile` form (it already does in
`create_session`), so all four combinations (programmatic|declarative ×
text|typed) work with no new config concepts.

## Public surface (Phase 1)

```python
from jaato_sdk import IPCClient, ask, AgentError, PermissionUnhandled

# context manager — multi-turn capable
async with IPCClient.session(profile="researcher", agent="pirate") as s:
    text    = await s.ask("Research tide pools.")        # -> str
    text2   = await s.ask("And summarize.")              # same session = memory

# programmatic inline spec
async with IPCClient.session(profile={"model": "gpt-4o", "provider": "openai"}) as s:
    print(await s.ask("Who are you?"))

# typed completion (completion-gated profile)
async with IPCClient.session(profile="person-extractor") as s:
    payload = await s.complete("Alice is 30.")           # -> dict | None

# one-shot module function (sugar over the context manager)
text = await ask("Who are you?", profile={"model": "gpt-4o", "provider": "openai"})
```

### `IPCClient.session(...)` → async context manager

Constructs the client, `connect()`s, `create_session()`s on `__aenter__`,
yields a `Session`, and `disconnect()`s on `__aexit__`.

Forwarded `create_session` params: `profile: str | dict`, `agent: str`,
`agent_params: dict`, `cascade_driver_id: str`.

Connection knobs (defaults chosen so the common case needs none):

| kwarg | default | rationale |
|---|---|---|
| `socket_path` | ctor default (`/tmp/jaato.sock`) | |
| `client_type` | `ClientType.API` | facade is completion-oriented (keeps `signal_completion`) |
| `auto_start` | `True` | |
| `env_file` | `".env"` | `None` crashes the handshake — keep a real default |
| `workspace_path` | `None` → cwd | |
| `connect_timeout` | `120.0` | cold autostart ~30–60s |
| `on_permission` | `None` | see Permissions |

Raises `ConnectionError` if `connect()` fails and `RuntimeError` if
`create_session` returns no id (fail loud, never yield a dead session).

### `Session.ask(prompt, *, sources=("model",)) -> str`

Sends the prompt and waits on **first-of `{TURN_COMPLETED,
SESSION_TERMINATED}`** — the PR #399 invariant, so it never hangs:

- a **plain** turn emits only `TURN_COMPLETED` (the session goes IDLE) →
  returns the collected text;
- a **completion-gated** turn emits `SESSION_TERMINATED(reason="natural")`;
- an **error** turn emits `SESSION_TERMINATED(reason="error")` →
  raises `AgentError(error_type, error_summary)` (D1).

`sources` selects which `AGENT_OUTPUT` chunks to collect by `.source`
(`"model"`, `"tool"`, `"system"`, `"thinking"`, plugin names). Default
`("model",)` = a clean answer string; `sources=None` = no filter (collect
everything). Deterministic — no hidden filtering. (D3)

Multi-turn safe (the session persists across `ask`s). Single-flight: one
turn at a time per session.

### `Session.complete(prompt) -> dict | None`

For completion-gated profiles. Captures the `AGENT_COMPLETED.payload`
(emitted before the terminal), waits on first-of `{SESSION_TERMINATED,
TURN_COMPLETED}`, returns the typed payload (`None` if the profile declared
no `completion_payload_schema` or the model didn't complete). Raises
`AgentError` on an error terminal.

## Permissions (D2 — the second hang trap)

If a session has gated tools and nobody answers `PERMISSION_REQUESTED`, the
turn blocks forever — the same hang class as #399. The facade keeps the
"never hangs" guarantee:

- **`on_permission=callback`** on `session(...)`: invoked with the
  `PermissionRequestedEvent`; its return (`"y"`/`"n"`/`"a"`/… , sync or
  async) is sent via `respond_to_permission`.
- **no callback**: the facade cannot make a policy decision, so it **fails
  loud** — auto-denies (to unstick the daemon) and records the tool name;
  the in-flight `ask`/`complete` then raises `PermissionUnhandled(tool_name)`.
  It never silently degrades and never hangs. The message points the caller
  at `on_permission=` or the low-level API.

## Errors (D1)

| condition | result |
|---|---|
| `connect()` fails | `ConnectionError` from `__aenter__` |
| `create_session` returns no id | `RuntimeError` from `__aenter__` |
| turn ends `reason="error"` | `AgentError(error_type, error_summary)` from `ask`/`complete` |
| gated tool, no `on_permission` | `PermissionUnhandled(tool_name)` from `ask`/`complete` |

Raising (vs returning a status object) makes failures impossible to ignore
and mirrors LangChain's `.invoke` contract.

## What's explicitly out of scope (Phase 2)

- `Session.stream(prompt)` — async-iterator of chunks for live streaming.
- `IPCRecoveryClient.session(...)` parity (auto-reconnect facade).
- A synchronous wrapper for non-async callers.

These are additive over the same `Session` shape; deferred to keep the first
PR small and reviewable.

## Implementation notes

- Lives in `jaato_sdk/client/convenience.py`; `IPCClient.session` is a thin
  classmethod that delegates there (lazy import — no circular dependency).
- `ask`/`complete` subscribe with the registry's idempotent `Unsubscribe`
  handles and always unsubscribe in a `finally`, so repeated turns don't leak
  handlers.
- No new daemon/server code — the facade is 100% client-side over the
  existing event protocol.
