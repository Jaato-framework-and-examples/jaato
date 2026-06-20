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

## 2. Interrogate / validate / scaffold — `jaato-scaffold`

```
jaato-scaffold explain                      # plugins · providers · gc · archetypes
jaato-scaffold explain plugin <name>        # its tools (core/discoverable), config
jaato-scaffold explain provider <name>      # capabilities · knobs (by layer, typed) · quirks
jaato-scaffold explain sets --workspace DIR # profile-sets + the provider/model each pins
jaato-scaffold validate <profile.yaml|workspace> [--set S]   # lint vs the live registry
jaato-scaffold new profile-set --workspace DIR --set <provider_model> \
    --provider P --model M --agents a,b,c   # emit base+set, then re-validate
jaato-scaffold new client|fire|cascade|observer --workspace DIR --provider P --model M
```

`validate` catches the silent-ignore failures the runtime drops without a word:
a mistyped `api_params.temprature`, an unknown plugin, a quirk the provider
doesn't honor. `new` runs its own output back through `validate` (profiles) or a
compile-check (clients), so scaffolded output is valid by construction.
Add `--json` to any verb for machine consumption.

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

## Profile sets

Tier-1 `_base_<agent>.yaml` (provider-agnostic, `inherits:`) + tier-2
`<provider>_<model>/<agent>.yaml` (binds provider+model), selected at runtime by
`JAATO_PROFILE_SET`. Keep the base inherit-able — binding a provider/model in it
breaks set-selection. `jaato-scaffold new profile-set` generates this layering;
`explain sets` enumerates existing ones.
