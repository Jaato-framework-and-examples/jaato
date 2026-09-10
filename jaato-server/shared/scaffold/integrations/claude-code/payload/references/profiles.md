# Profiles, sets and personas

`jaato-scaffold explain profile` is the authority on every key and its
inheritance rule. This file carries only what bites in practice.

## Two tiers

Tier 1 `_base_<agent>.yaml` — provider-agnostic, holds stage determinism
(plugins, schemas, gates, ceilings). Tier 2 `<set>/<agent>.yaml` — binds
provider and model, `inherits: [_base_<agent>]`. `JAATO_PROFILE_SET` in the
workspace `.env` selects the set daemon-side.

Keep the base inheritable: binding a provider or model in it breaks
set-selection. `new profile-set` generates the layering; `explain sets`
enumerates what a workspace already has.

## The inheritance rules that bite

- `plugins` is a **union** — a child cannot remove what a parent added, and
  `plugins: []` in a child **adds nothing; it does not clear the parent's**.
  You do NOT need to repeat a parent's list to extend it. To reduce the tool
  surface: `tool_scopes`, the permission whitelist, or do not inherit.
- `completion_processors` **concatenate**; the only removal is naming an entry
  in `suppress_inherited_processors`, and an entry matching nothing is a load
  error.
- `max_turns` is **most-restrictive-wins**.
- Scalars are child-replaces.
- `plugin_configs`, `env`, `tool_scopes`, `quirks` are **per-key dict-merge —
  ONE level deep.**

That last one is the trap. Per-key merge applies at the first level under the
plugin name; a **nested dict is replaced wholesale**:

```yaml
# parent
plugin_configs: {openrouter: {api_params: {temperature: 0.0}}}
# child
plugin_configs: {openrouter: {api_params: {enable_thinking: true}}}
# resolved — temperature is GONE
plugin_configs: {openrouter: {api_params: {enable_thinking: true}}}
```

Nothing fails. The sessions run, the payloads validate, and a determinism knob
you thought was inherited quietly is not. So: put in a shared parent only the
keys **no child overrides**, and keep any dict a child extends in the child.

Verify a resolution rather than reasoning about it — resolve the profile and
print what came back.

Both of these were stated BACKWARDS in a widely-copied skill: "lists are
REPLACED, not concatenated… you must repeat the base's plugin list", and
"dicts merge recursively until they hit a leaf scalar/list". Two agents ran
into it on the same day; one nearly rewrote a working profile because of it.
The rules above are what the installed framework does — measured, not read.
When this file and `explain profile` disagree, believe `explain`, and when
neither settles it, resolve a profile and look.

## Personas and `agent_params`

`.jaato/agents/<name>.md` — YAML frontmatter (`description`, `params`) plus a
body with `{{param}}` / `{{param:default}}`.

- **`agent_params` must all be strings.** They cross the wire as `key=value`
  argv tokens; a `spawn_payload_schema` property typed anything else is refused
  on every spawn and the caller only ever sees a 60 s `SessionNotConfirmed`.
- Long content — reports, transcripts — goes in the trigger prompt, not in a
  param.
- **Never put a credential in a param.** The rendered persona is persisted.

## Prefetch

`{{!py:script.py}}` runs **inside the runner at session-prep**, so whatever it
imports must be importable there, and it runs inside `session.bootstrap`'s RPC
budget. Bound it. `{{!py?:...}}` makes a failure drop the placeholder instead of
aborting session-prep. `explain prefetch` is the contract.

## Where things live

`explain paths` is the authority. The short version: `~/.jaato/` is
daemon-global (credentials, installed reactors) and shared by every session —
never override `$HOME` to "isolate" a run, since that hides the credentials from
the daemon. `<workspace>/.jaato/` is the per-session config_root and is
**framework-owned**: profiles, agents, instructions, plus the framework's own
`logs/` and `sessions/`. Isolate a run with a fresh workspace, and keep your own
application state out of that tree.
