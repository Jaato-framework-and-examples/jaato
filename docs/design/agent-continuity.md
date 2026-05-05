# Agent Continuity via `{{continuity_scope}}` + Existing Primitives

## Summary

Agent persona continuity across discrete sessions — the "this agent
remembers prior runs in the same scope" property — is **fully expressible
today** using existing jaato primitives, without new framework code.

The pattern composes three things you already have:

1. The **memory plugin's `enrich_prompt`** auto-surfaces memories whose
   tags appear (paragraph-coherent) in the assembled prompt.
2. The **memory plugin's raw-then-curated lifecycle** lets an advisor
   agent post-process freshly stored memories into validated/refined
   ones.
3. **Agent rendering** substitutes `{{param}}` placeholders from
   `agent_params` (or prompt-template params via `%prompt-name`).

Author the agent's `.md` (or a `%prompt`) with a `{{continuity_scope}}`
placeholder and a small postamble nudge; have the caller fill the param
with a stable scope-id (project, A2A `contextId`, ticket number, etc.);
the rest happens for free.

This document is the canonical reference. It is intentionally a
**design pattern**, not a feature spec — there is no code to ship in
either jaato-server or jaato-premium beyond optional documentation
notes and (optional) curated example agents.

## Status & verification disclaimer

Pattern was converged in a 2026-04-16 design thread. This document
draws the technical claims from that thread and verifies them against
current repo state at write time:

| Claim                                                | Verified at      |
|------------------------------------------------------|------------------|
| Memory plugin `enrich_prompt`                        | `shared/plugins/memory/plugin.py:585` |
| Tag-coherence matching (paragraph-coherent)          | `shared/plugins/memory/plugin.py:730` (`_tag_coherent_in_paragraphs`) |
| `agent_params` plumbed end-to-end                    | `jaato_runtime.py:879`, `session_manager.py:890+` |
| `agent_params` validated against profile schema      | `session_manager.py:1110+` |
| `%prompt-name` expansion + param substitution        | `session_manager.py:2463` (`_expand_prompt_references`) |
| Raw → curated memory lifecycle (advisor pattern)     | `shared/plugins/memory/plugin.py` docstrings (lines 4, 53, 184, 240, 328) |
| Memory advisor as auto-firing reactor on `agent.completed` | **Pattern shipped in `jaato-knowledge-manager/.jaato.example/`** as the canonical example. Reactor rule in `reactors.json` matches `agent.completed where source_agent == 'main'`, fires `reactors/on_session_complete.py`, which spawns a headless session with the `memory-advisor` profile (`profiles/memory-advisor.json`) — see "Reference implementation" section below. Continuity works without auto-curation too — the agent's own postamble nudges `store_memory`, and the next session's `enrich_prompt` surfaces the raw memory regardless. Curation refines what's surfaced; absence delays refinement, doesn't block continuity. |

Future readers: re-verify before relying on file:line citations.
Re-verify the advisor-reactor question if you're building a system
where automatic curation timing matters.

## The pattern

### Three composed primitives

#### 1. Memory plugin enrichment

The memory plugin subscribes to the `enrich_prompt` surface. It
receives the assembled prompt (system instructions + user message +
plugin contributions), tokenises into tags, and surfaces an
`💡 Available Memories` hint listing memories whose tags match.

The match is **paragraph-coherent**: a tag like `ctx-acme-customer-api`
matches a prompt that mentions both `acme` and `customer-api` in the
same paragraph, but doesn't false-positive when the components appear
in unrelated contexts.

**Continuity uses this**: a literal scope-id string in the prompt is
itself a tag. Memories stored under that tag in prior sessions auto-
surface in the new session's prompt, no explicit lookup required.

#### 2. Memory raw-then-curated lifecycle

When an agent calls `store_memory(tags=[...], content=...)`, the entry
lands as **raw**. The memory plugin's "advisor agent" pattern (see
`shared/plugins/memory/plugin.py:4-7, 53-57`) provides for a separate
agent to consume `get_pending_curation`, refine raw entries into
validated/curated ones (deduplicating, merging, tagging consistency),
and write back via `update_memory`.

**Continuity uses this**: agents storing continuity summaries don't
need to think about consistency — the advisor (when invoked) handles
that. Until curated, raw memories still surface; they're just less
refined.

When (or if) automatic curation lands as a reactor on
`agent.completed`, the cycle becomes hands-off. Today, callers can
invoke the advisor on a schedule or after batches.

#### 3. `agent_params` substitution

`agent_params` is a dict the caller passes into session creation.
Values substitute into `{{param}}` placeholders during agent
resolution (`jaato_runtime.py:879`, threaded into the system
instruction render).

When a profile declares an `agent_params` schema, the framework
validates the supplied dict against it before creating the session
(`session_manager.py:1110+`).

**Continuity uses this**: pass `continuity_scope` as a param; agent's
`.md` references it as `{{continuity_scope}}`; the literal scope value
lands in the rendered prompt where the memory plugin can see it.

## Author's contract

Continuity-aware agents include a small block in their `.md` (or in a
prompt template, see Caller's contract below). Two concerns: surface
prior context, and store new context.

### Recommended block

```markdown
Your continuity scope is `{{continuity_scope}}`.  If memory hints
surface under that tag, retrieve them — they carry accumulated
context from prior runs in this scope.

Before signalling completion, store a memory tagged
`{{continuity_scope}}` with a consolidated summary so future runs
in the same scope benefit.
```

Place it wherever the persona/task description makes sense (commonly
right after the role declaration, before the per-turn instructions).

### Where to put it

Two entry points, one pattern:

**Prompt templates** (`.jaato/prompts/<name>.md`) — primary path for
**task-level** continuity. A `%gen-references` invocation accumulating
KB-generation knowledge across runs under a given scope; a
`%refactor-pass` accumulating the project's refactor history. The
prompt's body is rendered server-side via `_expand_prompt_references`
which substitutes params before dispatching.

**Agent markdown** (`.jaato/agents/<name>.md`) — secondary path for
**persona-level** continuity. Use when `--agent <name>` sets
session-wide instructions and the persona itself wants memory across
invocations (`code-reviewer`, `research-assistant`, etc.).

## Caller's contract

Supply `continuity_scope` via the existing param-substitution path
appropriate to the invocation mode.

### TUI / IPC with a prompt invocation (the dominant workflow)

```
%gen-references <positional_args> continuity_scope=acme-customer-api
```

The server-side `_expand_prompt_references` at
`session_manager.py:2463` substitutes the param into the rendered
prompt body before dispatching. This is a server-side `%` dispatch,
not a client-side feature — works identically across TUI, IPC, and
WebSocket clients.

### TUI / IPC with `--agent`

```
session.new --agent code-reviewer continuity_scope=acme-customer-api
```

The `key=value` tail is parsed by the command router into
`agent_params` and substituted during agent resolution.

### A2A adapter (premium)

Pass the A2A `contextId` as the `continuity_scope` param — either via
`agent_params` when mapping to an agent, or via the prompt-expansion
path when mapping to a `%prompt`. The A2A adapter already needs to
pass `contextId` through somewhere; this is the natural target.

(See `jaato-premium/docs/design/handoff-via-fork-replay.md` for the
related but distinct fork-and-replay handoff pattern; that's about
**transferring** session state mid-cascade, not about persisting
continuity across discrete sessions.)

### Reactor-spawned sessions

Same mechanisms; the reactor's spawn action chooses which mapping
fits the agent it's spawning. Reactor scripts have access to
`ctx.create_session(profile=..., agent=..., agent_params={...})` (via
the premium reactor framework) — populate `continuity_scope` in the
agent_params dict at spawn time.

## Observable flow

A typical session-N → session-N+1 cycle, when both invocations supply
the same `continuity_scope`:

1. **Session N starts.** Caller provides
   `agent_params["continuity_scope"] = "acme-customer-api"`. Agent's
   `.md` renders with the scope filled in. The literal string
   `acme-customer-api` lands in the system instruction.
2. **Memory enrichment fires.** Memory plugin's `enrich_prompt` scans
   the assembled prompt, finds the scope string as a known tag (a
   prior session N-1 stored a memory tagged `acme-customer-api`),
   surfaces an `💡 Available Memories` hint with the prior memory ID.
3. **Agent retrieves.** Agent sees the hint, calls
   `retrieve_memories(ids=[...])`, absorbs the prior context as part
   of its working set.
4. **Session N runs.** Domain work happens.
5. **Postamble fires.** The continuity block in the agent's `.md`
   nudges the agent to call
   `store_memory(tags=["acme-customer-api"], content=<summary>)`
   before signalling completion.
6. **`AgentCompletedEvent` fires.** Memory is now in raw state,
   surfaceable to future sessions immediately. (When the advisor
   pattern runs, the raw entry gets curated/validated/merged.)
7. **Session N+1 starts** with the same scope. Step 2 now finds
   session N's memory + any earlier curated ones. Cycle repeats.

## Watch items

### Tag-coherence false-positives

A scope value like `ctx-acme-123` tokenises on `-` into components,
one of which is the numeric `123`. The paragraph-coherence rule
(`_tag_coherent_in_paragraphs`) requires multiple components to
co-occur, which mitigates most false-positive risk — but the
verbatim-scope-string-as-tag path is the safest contract.

**Mitigation**: when authoring a continuity-aware agent, add a
test that a memory tagged `ctx-acme-123` doesn't leak into a prompt
mentioning unrelated `123` (e.g., line numbers, status codes). The
existing tag-coherence regression tests are the right home.

### Missing-param handling

If a caller invokes a continuity-aware agent without supplying
`continuity_scope`, the placeholder may render literally — the
prompt would contain the string `{{continuity_scope}}` as plain text.

Four graceful-degradation options, in preference order:

1. **Caller discipline**: pick non-continuity agents when continuity
   isn't desired. The simplest contract; documented expectation.
2. **Profile-level default**: declare a default in the agent's
   profile (`agent_params_defaults: {continuity_scope: ""}`), then
   the agent's `.md` checks for empty before referencing memory.
3. **Template-engine fallback syntax**: `{{continuity_scope|default:""}}`
   if the template engine supports it. **Verify the engine before
   relying on this.**
4. **Hard schema enforcement** via the profile's `spawn_payload_schema`:
   list `continuity_scope` as a required property, and the framework
   rejects spawns that omit it before the session is created. Trades
   silent literal-rendering for loud validation failure at the spawn
   boundary. See
   [`payload-schema-conventions.md` §6.1](./payload-schema-conventions.md)
   for the spawn-side contract and the strictness trade-offs.

For the first continuity-aware agent shipped, option 1 is the
zero-risk default; promote to 2 or 3 if usage patterns warrant;
promote to 4 when the profile already declares a
`spawn_payload_schema` and you want continuity-scope omission to
fail loudly at the caller.

### Interaction with strict spawn schemas

If a continuity-aware profile **also** declares a
`spawn_payload_schema` with `additionalProperties: false`, the
schema MUST list `continuity_scope` as an (optional or required)
property, otherwise spawning with the param will reject as an
extra-key violation. Authors choosing strict-shape enforcement on
spawn schemas absorb the framework-level params explicitly. See
[`payload-schema-conventions.md` §6](./payload-schema-conventions.md)
for the full discussion.

### Agent cooperation on storage

The `store_memory` call at session end is **agent-cooperative**: if
the model skips the postamble or finishes without storing, nothing
gets recorded. Acceptable risk for cut 1; the pattern degrades to
"continuity only when the agent remembers to store."

**Belt-and-braces enhancement** if data shows the risk is real:
teach a future memory advisor (or a session-end reactor) to detect
"prompt contained scope string, no memory stored under that tag at
completion" and run a compactor over the session transcript.

### Curation timing

If the memory advisor curation isn't auto-fired (see verification
table at top), curated entries lag raw ones. Continuity still
**works** — raw entries surface — but they may be redundant or
inconsistent until a curation pass runs. For high-frequency
continuity workflows, schedule the advisor explicitly (cron, daemon
extension, manual invocation) until automatic curation lands.

## Reference implementation: `jaato-knowledge-manager/.jaato.example/`

The `jaato-knowledge-manager` sibling repo ships the canonical
auto-curation wiring. Continuity-aware tenants can adopt the same
shape directly.

### Reactor rule (`reactors.json`)

```json
{
  "version": 1,
  "rules": [
    {
      "id": "queue-completed-sessions",
      "enabled": true,
      "match": {
        "event_type": "agent.completed",
        "where": "source_agent == 'main'"
      },
      "action": {
        "script": "reactors/on_session_complete.py",
        "params": {}
      }
    }
  ]
}
```

Match clause `source_agent == 'main'` is the load-bearing guard:
the rule fires only on the originating agent's completion, not on
the advisor's own completion (which would loop). The reactor script
adds a second guard against self-triggering.

### Reactor script (`reactors/on_session_complete.py`)

```python
"""Spawn a memory-advisor session when a main agent completes."""

def execute(params, event, ctx):
    session_id = ctx.session_id
    if not session_id:
        return

    # Self-trigger guard: don't fire when the advisor itself completes.
    profile = getattr(ctx.server, '_profile', None)
    profile_name = getattr(profile, 'name', '') if profile else ''
    if profile_name == 'memory-advisor':
        return

    ctx.create_session(
        agent="memory-advisor",
        profile="memory-advisor",
        initial_prompt=f"sessions={session_id}",
        session_name=f"memory-advisor ({session_id})",
    )
```

Two guards together:
1. The reactor rule's `where` clause excludes advisor completions
   that don't come from `source_agent == 'main'`.
2. The script's profile-name check is belt-and-suspenders for the
   case where the rule's match logic evolves or where multiple
   advisors share a `source_agent` value.

### Advisor profile (`profiles/memory-advisor.json`)

Loads the `memory` plugin (for `get_pending_curation` and
`update_memory` access), `environment` (env access), `service_connector`
(for any external lookups during curation), `todo` (multi-step
curation plans), and `permission` (sandbox the advisor's tool use).

GC config tuned for the advisor's longer-running curation work
(75% threshold, 50% target, 5 preserved recent turns).

### What a tenant copies vs. what they author

To adopt auto-curation in your own tenant:

1. **Copy** `reactors.json` (or merge the `queue-completed-sessions`
   rule into your existing reactor file) and
   `reactors/on_session_complete.py` verbatim.
2. **Copy** `profiles/memory-advisor.json` — adjust the plugin list
   if your tenant has different defaults, but keep `memory` preloaded.
3. **Author** `agents/memory-advisor.md` for your tenant — the
   persona instructing the advisor what to extract from completed
   sessions (lessons, conventions, recurring patterns) and how to
   use `get_pending_curation` / `update_memory`.

The framework provides the rails; the advisor's persona is yours.

## Cross-reference: typed payload schemas

When an agent declares a `completion_payload_schema` in its profile
(see [`payload-schema-conventions.md`](./payload-schema-conventions.md)),
it emits structured data via `signal_completion(payload=...)` instead
of free-form text.

For continuity, this means the **next session in the same scope can
consume the typed fields directly** via memory hint retrieval, rather
than re-parsing prose summaries. The structured payload becomes the
canonical continuity state; the prose summary stays as a human-
readable fallback.

This is **complementary, not a replacement**:

- Agents *without* a typed schema continue using prose-memory
  postamble exactly as documented above.
- Agents *with* a typed schema get a cleaner, validated continuity
  contract on top — `signal_completion`'s payload IS the structured
  continuity state, automatically valid against the profile's
  schema, and the next session reads it back as typed data.

The choice is per-agent, driven by whether the profile has
`completion_payload_schema` declared. Memory advisor (when present)
curates either kind.

## Bitter-lesson note

This design retires *by writing less prose*. If future models hold
enough state natively to make explicit continuity summaries
unnecessary, agent authors simply stop including the continuity
block in their `.md`. There's nothing to deprecate, no code to
remove, no migration. The ceiling is as low as it gets.

The same applies if framework-level continuity primitives ever
land — they would replace the postamble nudge, not the overall
pattern. The flow (caller fills scope → agent reads memory →
agent stores summary → reactor fires) is what matters; the
implementation of each step can evolve independently.

## What's NOT in scope

Explicitly rejected during the design thread:

- **No new enrichment plugin.** Memory plugin's existing
  `enrich_prompt` does the surfacing; adding a parallel
  `continuity` plugin would duplicate logic.
- **No `with_continuity(...)` utility.** The `agent_params`
  substitution path covers it.
- **No `continuity` field in agent frontmatter.** `agent_params`
  schema declaration is the established profile-level contract;
  adding a sibling field would fragment the surface.
- **No new kwarg on `SessionManager.create_session`.**
  `agent_params` already handles it.
- **No `agent_metadata` plumbing to surface frontmatter to plugins.**
  The plugin sees the rendered prompt; the scope is in the prompt
  as a literal string. No metadata channel needed.
- **No separate compaction reactor.** Memory advisor's existing
  raw-then-curated lifecycle covers consolidation.
- **No session pooling or persistent workspace.** Orthogonal
  concerns; continuity is about state, not about process lifecycle.

## Worked example

A `code-reviewer` agent with project-level continuity:

**Agent file** (`.jaato/agents/code-reviewer.md`):

```markdown
You are the code-reviewer agent.

Your continuity scope is `{{continuity_scope}}`.  If memory hints
surface under that tag, retrieve them — they carry accumulated
context from prior reviews in this scope (project conventions,
known anti-patterns the team has decided about, prior verdicts on
recurring issues).

## Process

[review steps...]

## Before signal_completion

Store a memory tagged `{{continuity_scope}}` summarising:
- New conventions / decisions surfaced this review
- Recurring issues you flagged (so future reviews can reference)
- Anything the team should remember across PRs

Then call signal_completion with your verdict.
```

**Profile** (`.jaato/profiles/code-reviewer.yaml`):

```yaml
name: code-reviewer
inherits: [_base_code-reviewer]
agent_params:
  type: object
  required: [continuity_scope]
  properties:
    continuity_scope:
      type: string
      description: >
        Stable scope identifier (project name, repo slug, etc.).
        Memories accumulated under this scope surface in future
        sessions invoked with the same scope.
plugins:
  - memory(preload)        # so retrieve_memories / store_memory available immediately
  - cli
  - file_edit
```

**Caller** (TUI, reviewing a PR for the `acme-customer-api` repo):

```
session.new --agent code-reviewer continuity_scope=acme-customer-api
```

**Session 1**: memory plugin finds nothing tagged `acme-customer-api`,
no hints surface; agent does its review, stores a summary memory
tagged `acme-customer-api` before completing.

**Session 2** (new PR, same project, same scope): memory plugin
surfaces session 1's memory in the `💡 Available Memories` hint;
agent retrieves it, brings prior conventions and known issues into
the new review's context.

The framework hasn't gained any code. The agent author wrote a
continuity-aware persona; the caller passed a scope string; the
existing memory + agent_params + prompt-render machinery does the
rest.
