---
name: jaato-sdk-client
description: Build, validate, and debug a jaato SDK client or profile WITHOUT reading the framework source. Use when writing an IPCClient/orchestrator, authoring or validating a .jaato profile / profile-set, or diagnosing daemon-connect / pass:// / event-tracing failures. Prefer the two executable tools below over reading code — they introspect the INSTALLED framework, so they never go stale.
---

# jaato SDK client + profile authoring

Two executable tools carry the truth (they read the *installed* framework, so
they can't drift). Reach for them before reading source.

## 1. Preflight — `jaato-doctor`

Run this FIRST, from the same Python env as the daemon you target. It answers
the questions that otherwise cost an hour:

```
jaato-doctor --workspace . --env-file .env --secret pass://jaato/<provider>/api-key
```

(`jaato-doctor` is the console-script shortcut installed with `jaato-sdk`;
`python -m jaato_sdk.doctor` remains a working equivalent if the SDK isn't on
your `PATH`.)

It checks: `server` importable (autostart needs it), socket listening / **stale**
(the dead-pidfile state that blocks autostart, with the exact fix), the
**daemon's HOME vs yours** (read from `/proc/<pid>/environ` — a mismatch is why
`pass://` secrets resolve from the wrong store), `pass://` resolvability, and
where profiles/logs land. Non-zero exit on any FAIL → usable as a gate.

**Debug a *running* session** (not preflight):

```
jaato-doctor --session <id|latest> --workspace DIR
```

reads that session's logs under `<workspace>/.jaato/logs/` and reports whether
its **runner-tier path plugins resolved the workspace** (`PASS=<ws>`) or got
`workspace=none` (`FAIL` — path tools `readFile`/`file_edit`/`cli` get
Permission-denied; the #344 class, fixed by the client sending `working_dir`).
The map it applies — the session/runner entities, the workspace flow, and where
each log lands — is `jaato-scaffold explain runtime`.

## 2. Interrogate / validate / scaffold — `jaato-scaffold`

```
jaato-scaffold explain                      # plugins · providers · gc · archetypes
jaato-scaffold explain plugin <name>        # its tools (core/discoverable), config
jaato-scaffold explain provider <name>      # capabilities · knobs (by layer, typed) · quirks
jaato-scaffold explain sets --workspace DIR # profile-sets + the provider/model each pins
jaato-scaffold explain archetypes           # what `new` WRITES, per archetype
jaato-scaffold explain archetype <name>     # its file tree, file by file
jaato-scaffold validate <profile.yaml|workspace> [--set S]   # lint vs the live registry
jaato-scaffold new profile-set --workspace DIR --set <provider_model> \
    --provider P --model M --agents a,b,c   # emit base+set, then re-validate
jaato-scaffold new client|fire|cascade|observer|sweep|host-tools \
    --workspace DIR --provider P --model M [--recoverable]
jaato-scaffold new <archetype> ... --dry-run  # the exact tree, written nowhere
jaato-scaffold explain clients              # IPCClient (simple) vs IPCRecoveryClient
```

**Do not reverse-engineer the generator.** `explain archetype <name>` states
what `new` writes, what is in each file, and which parts are placeholders you
must edit versus the recipe you must not touch; `new ... --dry-run` shows the
exact tree your flags would produce (create vs append) without writing it. Both
answer the question that otherwise sends a reader into `build.py` — reading the
templates instead of running the generator costs far more than one `--dry-run`.

`validate` catches the silent-ignore failures the runtime drops without a word:
a mistyped `api_params.temprature`, an unknown plugin, a quirk the provider
doesn't honor. `new` runs its own output back through `validate` (profiles) or a
compile-check (clients), so scaffolded output is valid by construction.
Add `--json` to any verb for machine consumption.

## Transports — IPC (this SDK) vs WebSocket (TS SDK)

**This skill and the Python SDK (`jaato_sdk.IPCClient`) are IPC-only** — a local
Unix socket, unauthenticated (`--socket-mode`, default 660). **WebSocket clients
are authored with the TypeScript SDK (`jaato-sdk-ts`) / the browser client (`jaato-web/`),
NOT the Python SDK** — so `jaato-scaffold new client` scaffolds the Python IPC
client; for a WS client start from `jaato-web/` (React) or `@jaato/sdk` directly.

What you still own from here, even for a TS client, is the **daemon's WS side**:

```
jaato-scaffold explain transports        # IPC-vs-WS matrix, daemon flags, auth contract
jaato-doctor --web-socket [host:]port    # preflight: WS port + bearer-token file + auth mode
```

Daemon WS flags: `--web-socket [HOST:]PORT`, plus one of `--ws-token TOKEN` /
`--ws-token-file PATH` / `--ws-unsafe-no-auth` (no flag → the daemon
auto-generates `~/.jaato/ws.token`, mode 0600). A TS/browser client presents the
bearer either as `Authorization: Bearer <token>` (header) or `?token=<token>`
(query param — browsers can't set headers on `new WebSocket()`); a bad token is
closed with WS code 1008.

## Mental model (the one thing to hold in your head)

A jaato client **attaches to a stateful daemon singleton**; the daemon's
identity (HOME, config-root, socket) — invisible from the client API — decides
`pass://` resolution, where logs land, and which profiles load. So:

- Run `jaato-scaffold` / the doctor in the **same env** as that daemon.
- A wrong-HOME daemon (sandboxed harness, sudo) makes your shell's `pass`
  secrets invisible to it — the doctor's HOME-match check catches this.

## Known-good client recipe

Don't hand-write it — `jaato-scaffold new client` emits it. It bakes in:
`IPCClient(client_type=ClientType.API)` (keeps `signal_completion`),
`connect(timeout=120)` (cold autostart ~30-60s), `env_file` never None (None
crashes the handshake), and completion via
`subscribe_once(EventType.SESSION_TERMINATED)` then wait (NOT
`set_event_callback` — that method does not exist).

**Simple vs recoverable client.** That recipe uses the plain `IPCClient`; pass
`--recoverable` to `new client` to emit `IPCRecoveryClient` instead —
auto-reconnect state machine + `on_status_change` callback, with
`IncompatibleServerError` treated as permanent. Reach for it for anything
long-lived (a TUI, an observer, a cascade driver, anything that must survive a
daemon restart — a per-run `jaato-server --stop` + autostart). `jaato-scaffold
explain clients` lays out the choice.

## Three ways a harness hangs with nothing logged

All three end the same way — the daemon is content, the work is done, and your
driver sits there — so none of them looks like the bug it is. `jaato-scaffold
validate` now reports the first two before you run anything.

**1. `echo` with no `usage` → no terminal event, ever.** A turn is recorded
only when the provider reported tokens (`jaato_session.py`: `if
turn_data['total'] > 0`), and the post-turn hook gated on that record is the
ONE site that emits both `TurnCompletedEvent` and the quiescence flush
(`SessionTerminatedEvent`). `Session.complete()` settles on those, not on
`AGENT_COMPLETED` — so a zero-usage turn delivers its payload and then nothing.
Every echo-backed profile needs a spend:

```yaml
plugin_configs:
  echo:
    usage: {prompt_tokens: 1000, output_tokens: 200}
```

The framework's own conformance profiles all pass one. Validator code:
`echo_reports_no_usage`. (Also: `echo` is a legitimate, installed provider —
it is only hidden from `explain providers`.)

**2. A `spawn_payload_schema` property typed anything but `string`.**
`agent_params` cross the IPC wire as `key=value` **argv tokens**
(`ipc.py`: `args.append(f"{key}={value}")`), so the daemon validates strings.
`{"iteration": {"type": "integer"}}` is refused on EVERY spawn, the refusal is
logged daemon-side and never answered, and the caller gets a 60s
`SessionNotConfirmed` saying the session *may* have been created — for this
cause it never is. Type them `string`, add a `pattern` for shape, parse in the
prefetch. Validator code: `spawn_schema_type_unreachable`.

**3. The workspace `.env` IS the session env.** A profile's `${VAR}` and every
`pass://` / `vault://` URI is resolved daemon-side against that file, not
against your driver's process env. A driver that *writes* the workspace `.env`
(to set `JAATO_PROFILE_SET`, say) must COMPOSE it from the operator's env-file
rather than replace it, or the provider credential has no way to reach the
session.

When you do hit a hang, the discriminating probe is cheap: subscribe to every
`EventType`, run the stage, and print what arrived. `AGENT_COMPLETED` present
with `TURN_COMPLETED`/`SESSION_TERMINATED` absent is trap 1; no session at all
is trap 2.

## Profile sets

Tier-1 `_base_<agent>.yaml` (provider-agnostic, `inherits:`) + tier-2
`<provider>_<model>/<agent>.yaml` (binds provider+model), selected at runtime by
`JAATO_PROFILE_SET`. Keep the base inherit-able — binding a provider/model in it
breaks set-selection. `jaato-scaffold new profile-set` generates this layering;
`explain sets` enumerates existing ones.
