# Competitor memory systems — what is a gap, and what is only unwritten

## Summary

Nine agent-memory products were surveyed against jaato.  The useful output
is **not** a feature list: for a framework whose product is patterns, almost
everything on that list is already expressible, and several of the items are
already written down in this org's example repos.

What survives the survey is a **sorting rule**.  Anything a competitor does
falls into one of four buckets, and only two of them cost framework work:

| Bucket | Meaning | Framework owes |
|---|---|---|
| **Pattern** | A driver, a profile set, a persona, an evaluator script | nothing |
| **Seam** | A pattern *would* express it, but the hot path is not pluggable | an extension point |
| **Fidelity** | A pattern IS written and breaks, because a primitive misreports | a fix |
| **Not ours** | A vertical product's UX or connector catalogue | nothing, ever |

The mistake this document exists to prevent is reading a competitor's
feature as a bucket-1 gap in jaato.  Most are bucket 0 — already done.

## Status & verification disclaimer

Framework claims verified against `82554bd6` (2026-08-28), re-checked
after `main` moved past the first pass at `bc934d3e` — the §4 seams, the
§5 citations and the absence of any temporal-validity field all still
hold.  `jaato-premium` claims are at `3d4b8ab`.  Competitor claims are
from vendor documentation and repositories on the same date; each is
sourced inline.  Two items were **fixed during the writing of this
document** and are recorded as such rather than silently dropped — see
§5.  Re-verify file:line citations before relying on them: this document
was twice overtaken by the tree it describes while being written, which is
itself the §5 argument.

---

## 1. The nine, sorted by whether they compete

| | | |
|---|---|---|
| **Infrastructure** — competes with jaato | Letta, mem0, Zep/Graphiti, Sylph, DIY (Claude Code + git) | copy the mechanism, or deliberately don't |
| **Applications** — could be *built on* jaato | GBrain, Pletor, Gorgias Cortex, Slite | evidence of demand, not a feature backlog |

Pletor is a marketing-creative tool with brand guardrails; Cortex is
Gorgias's internal agent over their warehouse and codebase, emitting
dashboards, Notion docs and GitHub PRs.  Neither is a framework.  Reading
their features as jaato gaps imports a product roadmap that isn't ours.

## 2. What they offer beyond memory

Memory is the framing of the list, not its content.

**Scheduled routines.**  GBrain runs an overnight "dream cycle" — dedup,
citation repair, salience scoring, contradiction detection — on a durable
job queue with crash recovery.  Sylph's agents are AI employees with
cadences (Chief of Staff daily, CMO weekly).  Slite crawls continuously.

**Async human review as a surface.**  Slite's Triage UI: the agent detects
drift, drafts the fix, and *nothing auto-applies*.  Sylph enforces it
structurally — every output lands in `_drafts/` or `_logs/` and a human
approves before anything leaves the repo.

**Self-rewriting procedures.**  Sylph's real differentiator: after you
approve an output, the skill diffs what it drafted against what you kept
and **rewrites its own rules**.  Correction applied to procedures, not
facts — it compounds harder, because a better skill improves every future
run while a better memory only pays off when retrieval hits.

**Eval harness.**  GBrain benchmarks against LongMemEval with cross-modal
consistency and contradiction checks.  Gorgias built an engineering
culture around agent debugging.

**Context inspection.**  Letta's ADE inspects context windows, memory
blocks and run history visually, and debugs tools in a sandbox.

**Temporal invalidation.**  Zep/Graphiti models facts with validity
windows — when a fact was true, and when it was recorded — so a changed
fact is end-dated rather than overwritten.

Sources: [Letta ADE](https://docs.letta.com/agent-development-environment/ade/),
[Graphiti](https://blog.getzep.com/graphiti-hits-20k-stars-mcp-server-1-0/),
[Sylph](https://github.com/getnao/sylph),
[GBrain](https://github.com/garrytan/gbrain),
[Slite Agent](https://slite.com/blog/slite-announcing-self-maintaining-knowledge-base),
[Gorgias](https://medium.com/gorgias-engineering/creating-a-culture-of-agent-debugging-97ba4a50e956).

## 3. Bucket 1 — patterns.  The framework owes nothing.

Every item in §2 except temporal invalidation and context inspection is
expressible today, and several are already written:

- **Scheduling / self-wake.**  A sibling ends its turn asking to be woken;
  the driver honours it.  `perpetual-monologue-cascade` states the general
  case outright — *"No framework changes are required.  This is a profile
  set, two personas, one permission evaluator, and a driver script."*  Its
  pacer is a timestamp check inside a permission evaluator
  (`.jaato/policies/pace_monologue.py`), not a scheduler.
- **Curator on a trigger.**  The same repo's driver wakes the memory
  curator on every successful `store_memory` via `session.wake`, which
  revives a cold target.  A nightly variant is a different trigger on the
  same shape.
- **Human judgment mid-tool-call.**  A tool-call validator decides whether
  an approval needs a human, puts the session to sleep, and re-wakes it
  with the answer.
- **Eval harness.**  Shipped as `jaato-eval`.
- **Self-rewriting skills, human-correction capture, owner routing,
  markdown knowledge output.**  Driver-and-persona work over existing
  tools.

The framework's job for this bucket is **discoverability**, not code.  See
§7.

## 4. Bucket 2 — seams.  Where a pattern cannot reach.

A pattern needs a place to stand.  Two hot paths in the memory plugin have
none:

- **`MemoryStorage` is a concrete JSONL class.**  A driver cannot
  substitute a store, so Zep-style temporal validity, markdown-per-memory,
  or a forgetting curve cannot be written as patterns.  Nothing in the
  tree carries `valid_from` / `invalidated_at` / `superseded_by`; the only
  contradiction handling is the curator's `dismissed` or an overwrite.
- **Ranking lives inside `indexer.py`.**  Matches sort by tag-overlap then
  raw recency (`indexer.py:227-231`) — `confidence` and `usage_count` are
  stored on every `Memory` and never scored.  A deployment that wants
  decay cannot express it.

**The recommendation is the seam, not the semantics.**  Don't implement
bi-temporality; make the store backend and the ranking function pluggable,
and let temporal validity, decay and markdown storage be patterns someone
writes in a repo like the others.  Shipping Zep's model would put one
opinion in the framework where the framework's whole thesis is that
opinions belong in the harness.

## 5. Bucket 3 — fidelity.  Written patterns that break.

The highest-value category, and the one the survey nearly missed.  A
pattern that is written, correct, and defeated by a primitive that
misreports costs more than a pattern nobody wrote.

**Fixed while this document was being written:**

- **A stalled daemon loop killed a cascade half** (`perpetual-monologue-
  cascade` §7.18).  `RunnerRPCTimeout` now logs
  `MODEL_THREAD_TRANSPORT_ERROR`, emits `recoverable=True` and returns, so
  the turn fails and the session stays loaded (`server/core.py:5202-5232`).
  Worth reading as its own lesson: #628 shipped the *comment* describing
  this control flow without the `return`, and a half still died 3.5 minutes
  in, WARNING and INFO one millisecond apart on the same exception.
- **A driver saw a turn end and nothing else** (#654).
  `TurnCompletedEvent.completion_gap = "not_signalled_after_nudges"` fires
  when the framework asked twice for `signal_completion` and gave up.  A
  downstream consumer had spent an afternoon inspecting a schema that was
  fine.
- **`list_memory_tags` reported an empty store while holding a raw queue.**
  Fixed on this branch: `pending_curation` is now in the model-facing
  result, not only in `_telemetry`.  Two curator sessions had concluded
  there was nothing to curate with twelve raw memories on disk.
- **A tool's outcome vocabulary never reached the event stream.**  Fixed on
  this branch: `ToolCallEndEvent.result_status` carries the result's own
  `status`, so a driver can tell `refused` (backpressure — let the peer
  work) from `sibling_cold` (the loop is over) without matching on prose.

**Still open:**

- **Stuck-volley detection.**  Nothing distinguishes progress from two
  halves circling one thought; every event looks healthy while it happens.
- **No time-based reactor trigger.**  Reactors match on event type plus a
  JMESPath `where` (`jaato_premium/reactors/matcher.py`); the only clock in
  the engine is the rule-file mtime poll.  Bucket 1 covers wake-at-time
  via a driver, so this is a convenience, not a blocker — but it is the
  difference between every deployment writing the same driver and none of
  them doing so.

## 6. What jaato is ahead on

Worth stating, because the survey's shape invites the opposite conclusion:

- **Isolation.**  Letta's tool sandbox is cloud sandboxing.  jaato has
  per-session AppArmor confinement, cgroups, a self-confining pre-warm
  runner pool, and egress control.  Nobody on the list is close.
- **Provider breadth.**  15+ providers with a quirks system, against their
  one or two.
- **Memory provenance.**  `maturity` / `confidence` / `scope` / `evidence`
  / `source_agent` / `source_session` on every `Memory`
  (`memory/models.py`) is richer than mem0's or Letta's.  The raw →
  validated → escalated lifecycle ("The School",
  `docs/design/agent-memory-knowledge-escalation.md`) has no equivalent on
  the list.

## 7. The one competitive lesson worth acting on

If patterns are the product, the pattern corpus needs the guarantees of an
API.

GBrain ships 50+ skills; Sylph ships a folder structure with named agent
roles.  Both are legible in one look.  This org's patterns live across two
dozen repositories with nothing indexing them — and the cost is
measurable: `jaato-eval`'s own premise records that *"14 of the 18 concepts
such a harness needs already existed in jaato, and three org repos had each
privately rebuilt a piece of it."*

The mechanism already exists, in one repo.
`jaato-cascade-coordination-example/certify/` is nine claims written to
fail, with three-valued verdicts where `BLOCKED` (surface absent) is
explicitly not green — *"CI must not read `2` as success."*  That is a
better instrument than anything on the competitor list.  It is not
systematic, and it does not run against jaato `main`.

A split like `3f019999` — raw queue separated from the curated store,
handler never repointed — is exactly what a corpus-wide certify suite
catches at the commit.  Instead it surfaced months later, in a run, as two
curator sessions reasoning soundly from a false premise.

## 8. Recommended order

1. **Hoist `certify/` into a corpus-wide suite run against jaato `main`.**
   Bucket 3 defects are found by patterns, so run the patterns.
2. **Close the remaining fidelity gaps** — stuck-volley visibility first.
3. **Open the memory seams** (store backend, ranking function).  Then
   temporal validity and decay are patterns, and this document's largest
   competitor differentiator costs the framework an extension point rather
   than an opinion.
4. **Index the pattern corpus** so a user — or a model — can find the
   wake-at-time driver without being told it exists.

Not to chase: connector catalogues, vertical UX, model-zoo aggregation.
