---
name: jaato-sdk
description: Build, run, validate and debug anything on the jaato SDK — a client, a host-tools client, a multi-stage cascade driver, an observer, a profile set — WITHOUT reading framework source. Use when writing an IPCClient/IPCRecoveryClient or a cascade orchestrator, authoring or validating .jaato profiles and personas, wiring completion gates or prefetch, choosing a provider or transport, or diagnosing daemon-connect, session-hang, permission or pass:// failures. Two executable tools introspect the INSTALLED framework, so they never go stale — reach for them before reading code.
---

# Building on the jaato SDK

## The rule

**Do not read jaato source to learn what the framework offers. Ask it.**

Two executables carry the truth. They read the *installed* framework, so unlike
any document — this one included — they cannot drift:

| tool | answers |
|---|---|
| `jaato-doctor` | is my environment able to run this at all, and why not |
| `jaato-scaffold` | what does the framework offer, is my config valid, write me the known-good recipe |

Run both from the **same Python environment as the daemon you target**.

### When `explain` has no answer

It happens. When it does, that is a **finding, not a detour**:

1. Say out loud that introspection was silent — don't skip to grep.
2. Read the source, get your answer, keep moving.
3. **File the gap.** A missing `explain` topic costs every future reader the
   same hour. This has already happened repeatedly: the `new` archetypes were
   undocumented until someone noticed they had backed off to the source three
   times in two minutes (→ `explain archetypes`, `explain archetype <name>`,
   `--dry-run`); the completion-processor contract was folklore (→ `explain
   completion`).

Reading source to **diagnose a suspected framework bug** is different and
legitimate — that is evidence-gathering, and a good issue needs exact code
sites. The rule is about learning the API surface, not about debugging.

## 1. Preflight — `jaato-doctor`

```
jaato-doctor --workspace . --env-file .env --secret pass://jaato/<provider>/api-key
```

Checks `server` importable (autostart needs it), the socket listening vs
**stale** (the dead-pidfile state that blocks autostart, with the fix), the
**daemon's HOME vs yours** (read from `/proc/<pid>/environ` — a mismatch is why
`pass://` secrets resolve from the wrong store), `pass://` resolvability, and
where profiles and logs land. Non-zero exit on any FAIL, so it works as a gate.

Debug a **running** session rather than the environment:

```
jaato-doctor --session <id|latest> --workspace DIR
```

reads that session's logs under `<workspace>/.jaato/logs/` and reports whether
its runner-tier path plugins resolved the workspace (`PASS=<ws>`) or got
`workspace=none` (`FAIL` — `readFile`/`file_edit`/`cli` then get
permission-denied). The map it applies is `jaato-scaffold explain runtime`.

## 2. Interrogate, validate, generate — `jaato-scaffold`

Every topic below is real; run one before assuming what it holds.

```
jaato-scaffold explain                       # overview: counts + every drill-down
jaato-scaffold explain plugins               # the registry's tool plugins
jaato-scaffold explain plugin <name>         # its tools (eager vs discovery-gated), config knobs
jaato-scaffold explain providers             # what is installed
jaato-scaffold explain provider <name>       # capabilities, knobs BY LAYER, quirks
jaato-scaffold explain profile               # every profile key + its INHERITANCE rule
jaato-scaffold explain completion            # the completion-gate contract (output side)
jaato-scaffold explain prefetch              # the {{!py:...}} contract (input side)
jaato-scaffold explain sets --workspace DIR  # profile sets present + what each pins
jaato-scaffold explain clients               # IPCClient vs IPCRecoveryClient
jaato-scaffold explain transports            # IPC vs WS, daemon flags, auth contract
jaato-scaffold explain runtime               # session/runner entities, workspace flow, logs
jaato-scaffold explain paths                 # ~/.jaato vs <workspace>/.jaato; config_root
jaato-scaffold explain tiers                 # model tiers + roles
jaato-scaffold explain gc                    # strategies + GCConfig fields
jaato-scaffold explain archetypes            # what `new` WRITES, per archetype
jaato-scaffold explain archetype <name>      # its tree, file by file, and its self-check
```

**`dependencies` is a word you append to any of them**, not a topic of its own —
what a provider imports, what a plugin shells out to, which extras a transport
needs, and whether this environment agrees with itself:

```
jaato-scaffold explain dependencies             # distributions, version skew, extras
jaato-scaffold explain provider openrouter deps # what its code actually imports
jaato-scaffold explain plugin cli deps
```

Everything there is derived — requirements from installed metadata, imports by
parsing the implementation, health by trying it. There is no table mapping a
provider to a package, because such a table is wrong the moment someone adds an
import, and a wrong table is worse than none: the reader stops checking.

Two things it catches that nothing else does. **Version skew** — `pip` records a
version at install time while an editable install keeps pointing at a tree that
moves, so every version-derived answer can name a build that is not running.
**Shadowing** — `PYTHONPATH` pointing at a checkout makes that checkout answer
instead of the installed copy, which is why the doctor insists on the daemon's
environment.

Generate rather than hand-write. Archetypes: **`profile-set`, `cascade`,
`client`, `fire`, `host-tools`, `observer`, `processor`, `sweep`.**

```
jaato-scaffold new <archetype> --workspace DIR [--provider P --model M] [--recoverable]
jaato-scaffold new <archetype> ... --dry-run   # the exact tree, written nowhere
jaato-scaffold validate <profile.yaml|workspace> [--set S]
```

`validate` catches the silent-ignore failures the runtime drops without a word:
a mistyped `api_params.temprature`, an unknown plugin, a quirk the provider does
not honour. `new` runs its own output back through `validate` (profiles) or a
compile check (clients), so generated output is valid by construction. `--json`
on any verb for machine consumption.

**Never reverse-engineer the generator.** `explain archetype <name>` states what
`new` writes, what is placeholder versus recipe, and `--dry-run` shows the exact
tree your flags produce. Reading the templates costs far more than one
`--dry-run`.

## Mental model — the one thing to hold

A jaato client **attaches to a stateful daemon singleton**. The daemon's
identity — HOME, config_root, socket — is invisible from the client API and
decides `pass://` resolution, where logs land, and which profiles load. So run
your tooling in the daemon's environment, and remember:

- **The workspace `.env` IS the session env.** A profile's `${VAR}` and every
  `pass://` / `vault://` URI resolves daemon-side against that file, not against
  your driver's process env. A driver that *writes* that `.env` must COMPOSE it
  from the operator's, never replace it.
- **`<workspace>/.jaato/` is the framework's** — profiles, agents, instructions,
  and the framework's own `logs/` and `sessions/`. Your application's runtime
  state does not go there; writes to tenant-invented subpaths are denied under
  confinement. Put driver state beside it, e.g. `<workspace>/.<yourapp>/`.

## Known-good client recipe

Don't hand-write it — `jaato-scaffold new client` emits it, and bakes in the
parts that are load-bearing and non-obvious: `client_type=ClientType.API` (keeps
`signal_completion` on the wire — the daemon strips it for terminal/web/chat
clients), `connect(timeout=120)` (a cold autostart takes 30–60 s against an SDK
default of 5), and `env_file` never `None` (it crashes the handshake with an
opaque `os.PathLike` TypeError).

Pass `--recoverable` for anything long-lived — a TUI, an observer, a cascade
driver, anything that must survive a daemon restart. `explain clients` lays out
the choice.

**`complete()` vs `ask()`** decides whether your turn ever ends. A
completion-gated session (its profile declares `completion_payload_schema`)
terminates at `signal_completion`, so use `await session.complete(prompt)` — it
returns the typed payload. A plain session's turn IS its terminus, so use
`ask()`/`stream()`. Mixing them up hangs or returns half-finished work: an agent
that stops in prose without signalling is re-prompted and keeps going, so the
turn event fires mid-flight.

## Three ways a harness hangs with nothing logged

All three end the same way — the daemon is content, the work is done, and your
driver sits there — so none looks like the bug it is. `validate` reports the
first two before you run anything.

**1. `echo` with no `usage` → no terminal event, ever.** A turn is recorded only
when the provider reported tokens, and the post-turn hook gated on that record
is the one site emitting both `TurnCompletedEvent` and the quiescence flush
(`SessionTerminatedEvent`). So a zero-usage turn delivers its payload and then
nothing. Every echo-backed profile needs a spend:

```yaml
plugin_configs:
  echo:
    usage: {prompt_tokens: 1000, output_tokens: 200}
```

Validator code: `echo_reports_no_usage`. (`echo` is a real installed provider —
it is only hidden from `explain providers`.)

**2. A `spawn_payload_schema` property typed anything but `string`.**
`agent_params` cross the wire as `key=value` argv tokens, so the daemon
validates strings. `{"iteration": {"type": "integer"}}` is refused on EVERY
spawn, the refusal is logged daemon-side and never answered, and the caller gets
a 60 s `SessionNotConfirmed` saying the session *may* have been created — for
this cause it never is. Type them `string`, add a `pattern`, parse in the
prefetch. Validator code: `spawn_schema_type_unreachable`.

#883 ratified this as the contract rather than a daemon quirk, so the
in-process `spawn_subagent` boundary now refuses a typed value the same way —
it used to accept one the wire could never deliver — and the refusal names the
PROFILE as the cause instead of the value you passed.

**3. A prefetch that touches the network.** `{{!py:...}}` runs inside the runner
at session-prep, inside `session.bootstrap`'s RPC budget. A slow fetch is a
FAILED session, not a slow one — the driver sees `create_session: no answer
within 60.0s`, the same symptom as a refused spawn. Bound everything that runs
at prep, and prefer `{{!py?:...}}` so a failure drops the placeholder instead of
aborting session-prep.

When you do hit a hang, the discriminating probe is cheap: subscribe to every
`EventType`, run the stage, print what arrived. `AGENT_COMPLETED` with
`TURN_COMPLETED`/`SESSION_TERMINATED` absent is trap 1; no session at all is
trap 2 or 3.

## Keeping this file honest

This skill is the payload of the `claude-code` integration, shipped as package
data of the framework it describes — so the copy you are reading came from a
specific build and says which:

```
cat ~/.claude/skills/jaato-sdk/.jaato-integration    # the build this copy came from
jaato-scaffold integration claude-code --force       # replace it with the current one
jaato-doctor                                         # reports absent / stale / edited
```

If something here contradicts `explain`, **`explain` is right** — it reads the
installed framework, this file was written against one. Fix the file, and if
`explain` was silent on the point, file that too.

## Deeper references

Load only what the task needs:

- **`references/cascade.md`** — driver-as-graph vs supervisor-agent, topology,
  journaling and resume, debates and long-lived sessions, attaching an observer,
  host tools.
- **`references/profiles.md`** — the two-tier set layout, the inheritance rules
  that bite (including the one-level merge that silently drops nested keys),
  personas and `agent_params`.
- **`references/apparmor.md`** — confinement invariants, and how to find the
  code without a stale line number.
