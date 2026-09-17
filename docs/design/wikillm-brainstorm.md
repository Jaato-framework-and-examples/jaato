# wikiLLM for jaato agents — a brainstorm

Status: **brainstorm, not a spec.** Nothing here is scheduled, and §5 is
deliberately the only section that says what the framework would owe.

Tree claims verified at `afd067a` (2026-09-17). Re-verify file:line
citations before relying on them — two of the three documents this one
builds on record being overtaken by the tree while they were written.

Prior art in this repo, and the relationship to each:

| Document | Relationship |
|---|---|
| [The School](agent-memory-knowledge-escalation.md) | the closest thing. A wiki is its §8 open questions 1 and 5 answered by the *data model* instead of by a smarter advisor |
| [Agent Continuity](agent-continuity.md) | the read path this would inherit, and the trap it would inherit with it (§5, Fidelity) |
| [References as Knowledge Bundles](references-knowledge-bundles.md) | the on-disk shape (`Proposed`, not shipped). A wiki is a bundle with a write path |
| [Competitor memory systems](competitor-memory-systems.md) | the sorting rule §5 borrows wholesale |

---

## 0. The observation this starts from

**This repository's own `CLAUDE.md` is already a wikiLLM**, and noticing
that is worth more than any feature list.

Look at what it actually is. Roughly forty sections, each titled as a
narrative claim rather than a noun — *"A Cap Nobody Was Wearing (#735)"*,
*"A Slot the Pool Kept Offering After Its Channel Died (#1058)"*, *"A
Request the Daemon Wrote and the Runner Does Not Have (#856)"*. Each
carries measured evidence (`tool ran 60.02 s`, `2 of 5 running session
runners`, `16.13 MB frame`). Each states what was decided, what the cost
of the decision is, and — the part with no Wikipedia analogue — what was
**deliberately not done**. Sections cite each other by anchor. Issue
numbers are the citation format. It is edited in place, not appended to.

It even has enforcement: `test_docs_do_not_contradict_the_tree.py` fails
the build when a doc's claim and the tree disagree, and the complexity
ratchet fails when a baselined entry goes stale. That is a wiki with a
bot patrolling recent changes.

So the question is not *should jaato agents have a wiki*. They have one.
The question is what happens when you take the artifact that is already
working and make it **per-article, multi-agent-writable, and paid for by
the article instead of by the request**.

That last clause is the economic argument, and it is the whole reason to
bother. `CLAUDE.md` sits in the prompt-cache prefix: every session pays
for the OpenRouter audio wire while it edits the notebook sandbox. A
monolith cannot charge per topic. A wiki can.

---

## 1. What a wiki is that memory and references are not

The unit. That is the entire difference, and everything else follows from
it.

| | unit | written by | lifecycle | on a tenth encounter |
|---|---|---|---|---|
| **memory** | an **event** — *"I learned X while debugging Y on 2026-09-01"* | the working agent | `raw → validated → escalated \| dismissed` | a tenth near-duplicate row |
| **references** | a **document** — curated, versioned, embedded | a human, or `gen-references` | static until rewritten | not applicable; nobody is writing |
| **wiki** | a **topic** — *"X"*, impersonal, standing | any agent, mediated | stub → article → featured → disputed → deleted | the **same article, edited** |

A memory accretes. An article gets **rewritten**. That is why
deduplication is not a pre-clustering optimisation (The School's open
question 5) and consensus is not an advisor tie-break (its open question
1): if there is exactly one article per topic, the tenth agent to learn
something has nowhere to put it except the existing text, and reconciling
its claim with what is already written **is** the edit.

The three also differ in what they record about **who produced them**,
and the gradient runs the wrong way. A memory persists `source_agent` and
`source_session` (`memory/models.py:104-105`, written by `asdict` in
`storage.py:117`). A reference persists **nothing** — see §5, Seam 4. And
neither records the human, although the daemon authenticated one. §8 is
where that matters.

The corollary is uncomfortable and should be said plainly: a wiki makes
the write path adversarial. Two agents that disagree cannot both be
right in one paragraph, and the system has to do something about it.
Memory never had to — two contradictory memories coexist happily and the
model reads both. §6 is about that.

---

## 2. Six properties, and which one is load-bearing

1. **One article per topic.** The merge property, above. Load-bearing.
2. **Talk pages.** The article says what is currently believed; the talk
   page holds the disagreement, the rejected edit, the evidence that
   contradicted it. This matters more for an LLM than for a human,
   because contradictory evidence is both expensive in context and
   essential for correctness — so it must be *reachable* and *not
   injected by default*. Article cheap and canonical; talk expensive and
   read only when an agent is about to change the article.
3. **Revisions with attribution.** Every edit names the session, the
   agent, and — see §8 — the model binding. Rollback is one operation.
   This is how you answer *why does the agent believe this*, which no
   memory store in this tree can answer today.
4. **Links as retrieval structure.** `[[wikilink]]`s are a graph a
   previous editor already curated. For an LLM they are cheaper and far
   more precise than embeddings, because the adjacency judgement was made
   once, by something that had the full context, instead of being
   re-guessed per query from a cosine. Embeddings find the *entry point*;
   links do the expansion.

   **Half of this already ships**, which was not obvious until someone
   asked whether references can link to references. `references` walks a
   transitive graph today — see §5, Seam 3, where the finding and the
   remaining gap are written down. The gap is not traversal. It is that
   every edge means the same thing.
5. **Policy as executable rules.** Wikipedia has verifiability, no
   original research, notability, NPOV. The translations are in §3 and
   §4, and one of them is genuinely mechanisable in a way Wikipedia's
   never was.
6. **Deletion.** §8.

If only one survives, it is (1). A wiki without the single-article
invariant is a memory store with better formatting.

---

## 3. A wiki written for an LLM reader is not a wiki written for a human

This is where "wikiLLM" stops being "a wiki, for agents" and becomes its
own thing. The reader is different in six ways, and each has a drafting
consequence:

| The human reader | The LLM reader | Consequence |
|---|---|---|
| skims, follows links over minutes | consumes the whole article at once | **no lead/body redundancy.** Wikipedia repeats its lead in the body deliberately; here that is pure token waste |
| remembers the last article | has no memory between reads | every article is self-contained **at its own level**, and links rather than recaps |
| can ask a follow-up | cannot | ambiguity must be resolved in the text, not left for the reader to resolve |
| reads for free | pays per token, per read, forever | length is a recurring cost, so brevity is a *policy*, not a style preference |
| will not act on it | will act on it immediately | **negative results are first-class** |
| can weigh an author — knows who is senior, who was guessing | reads a name as a credential | **attribution is not authority** (below) |

The sixth row is the one with a hard rule attached. Put a human's name in
the *article text* and the model will weight the claim by who wrote it
rather than by its evidence — which inverts §4's whole argument, and ends
with an agent deferring to a senior engineer's stale claim over a passing
test. Wikipedia keeps author names out of article bodies and in history
for the same reason. So: **attribution lives in `history.jsonl`, never in
the text the model reads.** It is for the curator, the auditor and the
rollback, not for the reader.

The fifth row is the important one and it is the one `CLAUDE.md` already
gets right. An LLM handed an article about a subsystem will, unprompted,
re-derive and re-attempt the approaches the article does not mention.
So a wikiLLM article needs two sections a human wiki has no analogue
for:

- **Deliberately not done**, with the reason. (`CLAUDE.md` does this
  constantly: *"Not done here: ..."*, *"Deliberately NOT ..."*.)
- **Measured negative results.** *"All four GC strategies are already
  pair-safe"* is recorded in `CLAUDE.md` precisely because it is a
  negative result, and without it the next agent re-audits four
  strategies.

And the house style that falls out: **a rule with its incident attached.**
Not *"register the call on the reader thread"* but *"the worker's first
act was to register the call, so a cancel already in the socket buffer
lost the race — 12/12 under load, 0/10 idle"*. The rule alone is
followed; the rule with the measurement is *understood*, and an
understood rule survives a case the rule did not anticipate.

---

## 4. The one thing Wikipedia cannot do and this can

Wikipedia has no compiler. A claim is verified by a human following a
citation, once, at edit time, and it rots silently thereafter.

jaato has a compiler, a test suite, a shell, and a workspace. So:

> **Every article carries an executable freshness check, and an article
> whose check fails is not deleted — it is demoted to `disputed` and
> fenced when injected.**

This is the single strongest idea here, and the machinery for it is
already declared in the tree. `ReferenceContents` (`references/models.py`)
already has a `validation` field described as *"mandatory
post-implementation validation shell scripts"*, alongside `templates`,
`policies` and `scripts`. A reference directory already knows how to
carry executable content. An article is a reference directory whose
`validation/` script asserts the article's own claim rather than the
model's output.

Three grades of check, cheapest first:

| Grade | Example | Cost |
|---|---|---|
| **structural** | the file:line this article cites still exists; the symbol is still named that | milliseconds, run on every wiki load |
| **behavioural** | the reproducer still reproduces — `sleep 60` still runs 60.02 s under a 2 s cap | seconds, run on a schedule |
| **absent** | the claim is about a vendor's wire, a person's decision, a judgement call | never — and the article says so, which is itself information |

The third row is not a gap. An article that *declares* it has no
mechanical check is telling the next reader exactly how much to trust it,
which is more than any memory in the tree does today.

The failure this prevents is the one that makes knowledge systems net
negative: an article that was true at commit X, read with total confidence
at commit Y, sending an agent confidently down a path that stopped
existing. A stale memory wastes a retrieval. A stale *article* — canonical,
single, authoritative by construction — wastes a session.

---

## 5. The sorting rule, applied

[competitor-memory-systems.md](competitor-memory-systems.md) establishes
the rule this repo sorts knowledge-system ideas by, and the mistake it
exists to prevent — reading something as a framework gap when it is
already expressible. Applying it here is what separates "a wiki would be
nice" from "here is the one extension point that does not exist".

### Pattern — the framework owes nothing

Every one of these is a profile, a persona, a script, or a driver:

- **the editor/curator agent** — a profile with a
  `completion_payload_schema` and `completion_processors`; the gate
  already refuses a bad payload and hands the agent its errors, bounded
  by `max_refusals`
- **the notability threshold** — a line in the curator's persona
- **talk-page deliberation** — a markdown file the curator writes
- **stub creation by working agents** — `store_memory` with a topic-tag
  convention
- **scheduled compaction** (GBrain's "dream cycle") — a reactor, or a
  cron-fired headless session
- **independent review** — a second profile on a different `model_tiers`
  binding (§8 explains why the binding matters)
- **federation** — a wiki is a git repository. Pull requests between
  wikis are pull requests.

### Seam — a pattern *would* express it, but the hot path is not pluggable

Four. The first is the largest ask; the last two are the smallest, and
both land on the same dataclass:

1. **There is no topic-keyed store with revisions and a safe write.**
   Memory's storage is append-JSONL under `raw/` plus a `curated.jsonl`,
   keyed by memory id. There is no *"replace the article at topic T,
   given base revision R"*, and no history. A third-party plugin can
   invent one — and then it re-implements the tag matcher, the embedding
   reconcile pass, and the enrichment wiring that `memory` and
   `references` each already have their own copy of. The honest form of
   the ask is a choice, not a new subsystem: either **references gains a
   writable bundle**, or **memory gains a topic-keyed collapse**. Both
   are smaller than a wiki plugin.

2. **The enrichment hot path is not composable.** `memory` and
   `references` each subscribe to `enrich_prompt` / `enrich_tool_result`
   and each implement their own matcher — memory's is
   `_tag_coherent_in_paragraphs` (`memory/plugin.py:1099`), references'
   is its own regex pass with separator normalisation. A third knowledge
   surface writes a third. The seam is a *shared matcher*, not a third
   consumer.

3. **The reference graph has edges and no edge *types*.** Enough of this
   ships that the gap has to be stated precisely, or it reads as a
   feature request for something already built.

   `references` already traverses a graph.
   `_resolve_transitive_references` (`references/plugin.py:749`) runs a
   BFS from the selected and preselected set, bounded by
   `MAX_TRANSITIVE_DEPTH = 10` (`plugin.py:100`), discovering edges two
   ways: **ID mention**, where `_find_referenced_ids` (`plugin.py:639`)
   scans a reference's content for catalog ids as whole words — the regex
   deliberately tolerates `@ref:id`, **`[[id]]`**, backticks and bare
   prose — and **path resolution**, where `_find_referenced_paths`
   (`plugin.py:668`) extracts markdown links and `./` / `../` paths,
   resolves them against the source's own directory, and matches other
   LOCAL sources by `resolved_path`. It keeps edge provenance in a
   `parent_map` (discovered id → the parents that referenced it, held as
   `_transitive_parent_map`, `plugin.py:144`), and that provenance
   reaches the model: the instruction block annotates
   *"(Transitively included — referenced by @parent)"* and
   `listReferences` emits `transitive: true` with `transitive_from`.

   So a reference already pulls in its neighbourhood **and tells the
   model why each neighbour arrived**. What it does not have:

   | Missing | Consequence |
   |---|---|
   | **no declared edges** — `ReferenceSource` has no `links` field | every edge is inferred from body text at read time, so renaming an id silently deletes every inbound edge. No integrity check, no dangling-link finding |
   | **untyped edges** | *mentioned* is the only relation. No `depends-on` vs `supersedes` vs `contradicts` — and most of a knowledge graph's value is in the edge labels |
   | **no reverse index** | *"what points at B?"* costs a walk of the whole catalog |
   | **not queryable** | no tool or subcommand exposes the graph; `transitive_from` surfaces only for refs already in *this* selection |
   | **effectively local-file only** | an edge needs `_get_reference_content` to return a body, so URL sources pay a network read and MCP sources largely do not participate |

   The one that bites in practice is none of those individually. It is
   that **expansion is bounded in depth and not in cost**: depth 10, any
   fan-out, no edge weights, no relevance ranking, no token budget. In a
   densely cross-referencing catalog, *pull in the neighbourhood* is
   *pull in the catalog*. Depth 10 is a runaway guard, not a relevance
   bound — the same distinction §7 draws about the index, and the same
   one `CLAUDE.md` draws about `max_completion_nudges` being a per-turn
   and not a per-session budget.

   **The ask is a `links` field with a small closed `rel` vocabulary**,
   and inference is *kept* beside it: an inferred edge can never go
   stale, because it is recomputed from content, and it is what makes
   drop-in-a-file work at all. Declared edges add what inference cannot
   supply — typing, direction, and referential integrity — for the few
   relations that carry weight.

   The payoff is not the declaration. It is that **the edge type decides
   the expansion policy**, which is what makes traversal simultaneously
   cheaper and better where an untyped graph can only be one or the
   other:

   | `rel` | expansion |
   |---|---|
   | `depends-on` | auto-expand — A is not comprehensible without B |
   | `elaborates` | do **not** expand; surface as a hint, the way unselected references already are |
   | `supersedes` | rewrite the selection — pull the newer one *instead of*, never *as well as* |
   | `contradicts` | never auto-expand. This is what a **curator** reads (§6), not what a working agent is handed |

   Two things fall out for free: a reverse index built at load, and a
   `reference_link_dangling` finding for `jaato-scaffold validate`, which
   is unrepresentable today — an edge to a renamed reference does not
   break, it stops existing.

   Stated cost, because the argument cuts both ways: declared edges are a
   maintenance surface and they *do* go stale, which is the exact charge
   §4 lays against articles. That is why the proposal is a hybrid rather
   than a replacement — declare the load-bearing few, infer the rest —
   and why `supersedes` is the most valuable entry in the vocabulary: it
   is the one relation whose staleness is self-announcing.

4. **A reference records nothing about who produced it.**
   `ReferenceSource` is `id`, `name`, `description`, `type`, `mode`, the
   access fields, `fetch_hint`, `tags`, `contents`, `embedding`,
   `bundle_name` — and `to_dict` / `from_dict` round-trip exactly that
   set, so nothing is being dropped on save. There is no author, no
   session, no user, no created-or-modified timestamp. The two
   near-misses are not attribution: `SelectionRequest.timestamp`
   (`references/models.py:333`) belongs to an in-flight channel request,
   and `EmbeddingMetadata.source_hash` is a staleness fingerprint —
   content identity, not authorship.

   That is **coherent for what references were**: a human-curated catalog
   written by `gen-references` or by hand, where *who wrote this* was
   answered by git. It stops being coherent the moment an agent writes
   one, which is the whole wikiLLM turn.

   Memory is the contrast and the cautionary tale. It persists
   `source_agent` and `source_session` (`memory/models.py:104-105`,
   `storage.py:117` writes the whole dataclass with `asdict`) — and that
   field was **null for every cascade session** after PR-196, because the
   registry-shared plugin read whichever sibling bootstrapped last, so
   one sibling's id leaked into another's memories and the runner-side
   path read `None` outright. The plugin's own docstrings record the
   measurement (`memory/plugin.py:184`, `226-231`, `334`: *"peer 7:1
   retry-49 post-PR-196 still showed source_session=null on 4/4
   memories"*), fixed by stamping per-session in `bootstrap_session` and
   reading through `_get_session_id()`.

   The lesson is the one §8 keeps arriving at from different directions:
   **attribution that silently reports null is worse than none.** A
   revision history you cannot trust is one you cannot roll back or audit
   with, and it still looks like one.

   The consequence for this document is sharper than "unimplemented".
   §8's rule that **provenance gates placement** — locally reproducible
   evidence in the trusted region, web-derived fenced — is on references
   not merely unbuilt but **unrepresentable**: there is no field for the
   fence to read. So the ask is a provenance block beside Seam 3's
   `links`, on the same dataclass, both additive and both optional:
   `source_agent`, `source_session`, `created_by`, `witnessed_by`,
   `binding`, `created_at`. §8 says which of those are free and which are
   not, and what it costs to write a human's name onto an artifact that
   is designed to be shared.

### Fidelity — a pattern IS written and breaks, because a primitive misreports

- **Curated-only enrichment.** The memory index is built from
  `curated.jsonl` alone (`memory/plugin.py:343` — *"Build index from
  CURATED memories only"*). A wiki built on memory enrichment is
  **invisible until a curator has run**, which is exactly the trap
  `agent-continuity.md` had to correct itself about in 2026-06. Inherited
  wholesale, and worth stating before it is rediscovered a third time.
- **Lexical tag matching.** An article titled *"slot reuse key"* does not
  match a prompt saying *"the pool keeps handing me a dead runner"*.
  Wikipedia solves this with redirects — a cheap, per-article,
  author-supplied alias list — which is a better fit here than a second
  embedding pass, because the aliases are written once by something that
  understood the topic.

### Not ours, ever

A hosted wiki UI. A public federation network. A human-facing reader.
These are products; importing them imports a roadmap that is not this
framework's.

---

## 6. The write path is the whole problem

Reads are solved three times over in this tree — references' hint blocks,
memory's enrichment, and deferred tool loading's `list_tools()` →
`get_tool_schemas()`. All three are the same shape: advertise cheaply,
fetch on demand.

**Nothing in jaato lets an agent safely edit a shared document.** And the
concurrency is not hypothetical: two sessions on one daemon, N subagents
sharing one `PluginRegistry`, and cascade stages that hand a warm pool
slot on. `CLAUDE.md`'s own §*"A shared registry is a shared mutable
object (#938)"* records what sharing a mutable structure across a spawn
already cost — a `RuntimeError: Set changed size during iteration` inside
the parent's model loop, reaching the caller as an opaque
`RunnerCallError`.

Three candidate answers:

| | mechanism | cost |
|---|---|---|
| **compare-and-swap** | edit carries a base revision; a stale base is refused | the refusal lands on an agent mid-task, which then spends turns re-reading and re-merging. Expensive in exactly the currency that matters |
| **single writer** | only a curator writes articles | knowledge is never available inside the session that learned it |
| **append at the edge, merge at the centre** | agents append immutable *claims*; a curator collapses claims into the article | one curation lag |

The third is the proposal, and it is attractive because it needs no new
concurrency primitive at all:

> **Working agents never edit an article. They append a claim — a
> distinct file, or a row in an append-only log. The curator is the only
> writer of article text, so it serialises by construction.**

That is what memory's `raw/` directory already is. The wiki's write path
is the memory write path, unchanged; what is new is that the curator's
output is keyed by **topic** instead of accumulating by **event**.

And the edit-conflict problem does not vanish so much as move to where it
can be afforded: the curator, between sessions, with the old revision,
both claims, and the talk page in front of it, under a completion gate
that can refuse its output. Reconciling two contradictory claims is a
*reasoning* task given to something with time to do it, rather than an
interruption thrown at an agent trying to finish something else.

**The stated cost**, because it is real: a lesson learned in turn 3 is not
in the wiki for turn 40, nor for the sibling subagent running right now.
For a long cascade that is a genuine loss. The mitigation is not to
weaken the invariant but to note that within one session the claim is
already in history, and across a cascade the parent can pass it down —
both of which exist.

---

## 7. The read path, and the prefix problem

Four ways an article could reach the model, with what each costs:

| | mechanism | cost |
|---|---|---|
| **index in the system prompt** | titles + one-line abstracts + tags | paid on **every request, forever** — but stable, so the prompt-cache prefix survives |
| **enrichment hint on match** | what memory and references do today | free until matched |
| **an explicit tool** | `wiki_read`, `wiki_search`, `discoverable` | one round trip, and the body arrives as a *tool result* — after the prefix |
| **top-K bodies injected at turn start** | embedding search, full text | expensive, and **invalidates the prefix every turn** |

The fourth is the obvious design and it is wrong here, for a reason this
tree has already paid for twice: `CLAUDE.md` records that the
`spawn_subagent` profile enum is **sorted** rather than in discovery
order specifically because *"the tool schema sits in the prompt-cache
prefix, so a per-host order would re-read the whole prefix for nothing"*.
Anything that varies per turn must live **after** the prefix.

So the shape is forced, and it is the shape the tree already uses twice:

> **the index is stable and in the prefix; bodies are tool results.**

Which gives the economic claim its number. A monolithic `CLAUDE.md` of
*N* tokens costs *N* every request. A wiki costs
`index` + `Σ(articles actually read)` — and the index is the part that
must be ruthlessly short, because it is the part that is paid for
unconditionally. One line per article. Titles that are claims, so the
line is informative without the body.

There is a tension worth naming rather than resolving: a **growing**
index also breaks the prefix, once, whenever an article is added. That is
fine — it is one invalidation per curation cycle, not one per turn — but
it argues for curation running on a cadence rather than continuously.

---

## 8. Four ways this fails

### Correlated consensus is not consensus

This is the sharpest failure mode and the least obvious.

Wikipedia's peer review works because its observers are *independent*.
Three agent sessions on the same model, with the same persona, over the
same codebase, are not independent in any useful sense — they are three
samples from one distribution, and they will agree on the same wrong
thing with high confidence. *"Three sessions confirmed it"* is therefore
much weaker evidence than it looks, and a notability threshold built on
it manufactures false authority at scale.

Two mitigations, and jaato can express both:

- **Require independent evidence, not independent agreement.** A claim
  is corroborated by a *reproducer that runs*, not by a second agent
  saying so. This is §4 again, arriving from a different direction.
- **Cite the binding.** `_observe_binding_usage`
  (`jaato_session.py:10948`) already segregates spend by
  `(provider, model, tier)` — because, as `CLAUDE.md` puts it, a session
  that calls `enter_tier` has *"one history and several bills"*. The same
  triple is exactly what an article's revision should record, which makes
  *"confirmed by a different binding"* a **checkable predicate** rather
  than a hope. A review profile on a different tier is then a real second
  opinion.

### A warrant that nobody gave

The section above is about mistaking correlated agents for independent
ones. This is the same mistake about the human, and it starts by
correcting a framing: agent attribution and human attribution are not
alternatives. An agent runs **inside a session**, and that session very
often has a person on the other end. Four axes, each answering a question
none of the others can:

| Axis | Answers | Today |
|---|---|---|
| `source_agent` | the writer's **competence and bias** — a documentation persona's claim about layout is not a security reviewer's claim about the same code | on a memory; not on a reference |
| `source_session` | lets the context be **reconstructed** — the history, the tools, the turn | on a memory; not on a reference |
| **the human** | **accountability**, and what makes review or escalation mean anything | on neither |
| the binding | the independence predicate above | nowhere durable |

So record the human. The trap is that **"the human" is not one thing**,
and the difference is exactly the information a promotion decision needs:

| The person | Relationship to the claim |
|---|---|
| approved the tool call that produced it | the strongest warrant available |
| in the session, never saw this turn | present, uninvolved |
| five hops up a cascade | `_create_subagent_session` sets `created_by=self._creator_of(parent)`, so identity propagates down. Right for accountability, **false** as "this person saw it" |
| absent | `ClientType.API`, `_HEADLESS_CLIENT_ID` |

Collapsing rows 1 and 3 into one `created_by` manufactures a warrant
nobody gave — the same shape as the correlated-consensus error, arriving
through the front door. So the field splits: **`created_by`** is the
accountable identity, inherited down the cascade; **`witnessed_by`** is
set only when a person actually saw *this* claim's evidence.

**The tree already has the vocabulary for that distinction.** #859
records `user_id` *and* `approver` on `PermissionResolvedEvent` and the
ledger's `permission-check` row, and #951 added `asked=` to the DECISION
line **precisely because `method` could not say whether anybody was
consulted** — `allow_all` is produced both by a silent pre-approval and
by a human typing `a`. That is this distinction one layer down. *A human
approved the tool call that produced this claim* is already a recorded
fact; it simply never reaches the memory that resulted.

**How a plugin reaches the human: `get_session_user()`, beside
`get_session_env()`.** The obvious route is the wrong one, and the module
that owns this seam says so in its own docstring. An in-tree plugin *can*
reach the live session — `get_current_session()`
(`memory/plugin.py:193-194`) is how the PR-196 `source_session` fix works
— but `jaato_sdk/session_env.py` lists exactly that name as
**deliberately not exported**, because it hands back a `JaatoSession` and
*"a plugin reaching into `session._runtime` is not something to make
easier from out of tree"*. An out-of-tree knowledge plugin — the audience
Seam 4 is about — cannot take that route, and the SDK is right to refuse
it.

So the identity belongs where the session-scoped **credential** read
already lives. `jaato_sdk/session_env.py` exports three names
(`get_session_env`, `set_session_env`, `clear_session_env`) over one
module-private `ContextVar`, for a reason that transfers verbatim: a
plugin that reads `os.environ` directly gets **another session's** value
on a daemon serving two, because `JaatoServer._with_session_env()`
overlays the process environment per turn (#918). A plugin that reads a
user identity from anywhere but session scope has the same bug with worse
consequences.

The setter needs no new machinery. `_with_session_env` (`core.py:1568`)
is already the per-turn scope, already sets the ContextVar first and
clears it in `finally`, and `JaatoServer._client_user_id` is already
populated from `SessionInitEnvelope.created_by`
(`session_manager.py:3831`). One line, in a block that exists.

**And exactly one thing must NOT be symmetric with its neighbour.**
`get_session_env` falls back to `os.environ`, correctly — an env var has
a legitimate ambient source. **A user identity has none.** Giving
`get_session_user()` any env fallback would relocate the
`_resolve_telemetry_user_id` hole (`jaato_session.py:1603`, whose
precedence drops to `JAATO_TELEMETRY_USER_ID` from the per-session env)
into the SDK, where it would look sanctioned — a workspace `.env` forging
authorship on a shared knowledge artifact. So: no fallback, `None`
outside session context, and absence means *the transport authenticated
nobody*, never *guess*. Positive evidence only, the posture #1014 and
#1023 take about confinement labels. Telemetry keeps its env fallback,
which is legitimate there and is the whole reason the two accessors must
not be the same function.

**One ContextVar, one definition.** `shared/session_context.py` imports
the trio rather than declaring its own (`:82-86`), because — its
docstring again — *"a second copy in the SDK would read empty, fall
through to `os.environ`, and reintroduce the bug in a form that looks
fixed"*. The fourth name joins that import, and the guard that asserts
**object identity** rather than behaviour extends to cover it.

**It answers `created_by` and cannot answer `witnessed_by`.** Different
lifetimes: the accountable identity is per-SESSION, which is what a
ContextVar is shaped for; the witness is per-CLAIM — *did a person
approve this tool call* — and comes from the permission decision, where
#859's `approver` and #951's `asked=` already live. One accessor
answering both would re-perform exactly the collapse this section exists
to prevent.

The binding is the one that is genuinely new: `_observe_binding_usage`
has `(provider, model, tier)` but only as per-response spend, stamped on
nothing durable.

**What it buys** is a promotion rule the current model cannot express at
all: a **human-witnessed** claim clears a lower notability bar. One
person who approved the tool call and read the output is better evidence
than three correlated sessions agreeing — and with `asked=` already
recorded, that is checkable rather than assumed.

**What it costs**, stated because this is the section for it: an identity
written onto a knowledge artifact **travels with it**. `created_by` on a
session record is local; on a reference bundle or a wiki article it
crosses org boundaries by design — bundles merge, `scope: universal`
lands in `~/.jaato`, wikis federate as git. That needs an export policy
(strip, or pseudonymise at the bundle boundary), and on a long-lived
artifact *"delete this person"* becomes a history rewrite. The #1074
qualified form (`app:user`) namespaces the identity, which helps with
collisions and not at all with this.

### An article is untrusted content the moment a subagent wrote it

The naive wiki puts article text in the system prompt region — the
*trusted* region, where the model reads instructions as legitimate. But
an article can be written by a curator that read a web page, an MCP
server's output, or a subagent's summary of either. jaato already treats
all three as untrusted (`TRAIT_UNTRUSTED_CONTENT`,
`wrap_untrusted_content` at
`jaato-sdk/jaato_sdk/plugins/model_provider/types.py:281`,
`TRAIT_UNTRUSTED_SCHEMA` for MCP's self-authored descriptions).

So provenance has to gate placement:

| an article whose evidence is | placement |
|---|---|
| locally reproducible (a command, a test, a file) | trusted region |
| derived from the open internet or a third party | **fenced**, as tool-result-tier content |
| mixed | fenced — the weaker provenance wins |

This is not paranoia about a hostile wiki. It is that a wiki is a
*persistence* mechanism, and an injected instruction that reaches an
article has been laundered into the trusted region **and made permanent**,
which is strictly worse than the single-turn version the fence was built
for.

### The token budget forces deletionism

Every article costs index space in the prefix forever, so the
inclusionist position is not affordable. Deletion must be policy, and
automatic:

- a failing behavioural check that is not repaired within N cycles
- usage decay — injected M times, acted on zero
- superseded-by, which is a **link**, not a delete: the article is
  replaced by a redirect, so an agent arriving with the old vocabulary is
  still routed correctly

The thing to measure and refuse to look away from: an article that is
injected often and never acted on is not neutral. It is a recurring
charge against every session, and the honest response is to delete it.

---

## 9. Shape sketch

Not a proposal — a sketch, to make the reuse concrete. It is deliberately
a [bundle](references-knowledge-bundles.md), so that *"drop in a
directory"* and *"merge two knowledge sets"* are the operations that
design already gives:

```
.jaato/wiki/                          # project scope
  wiki.json                           # policy: notability, decay, trust tiers
  index.json                          # title + abstract + tags + revision + health
  embedding_config.json               # the bundle's own model/dim stamp + rows
  wiki.embeddings.npy
  articles/
    runner-pool-slot-reuse/
      article.md                      # the claim, current revision
      talk.md                         # disputes, rejected edits, the evidence against
      history.jsonl                   # revision, base_rev, diff + the four axes of §8:
                                      #   source_agent, source_session,
                                      #   created_by (accountable),
                                      #   witnessed_by (saw the evidence),
                                      #   binding (provider, model, tier)
                                      #   -- never rendered into article.md (§3)
      links.json                      # declared edges + redirects, ON TOP of the
                                      #   inferred ones references already walks:
                                      #   [{"to": ..., "rel": "depends-on"
                                      #     | "elaborates" | "supersedes"
                                      #     | "contradicts"}]  -- see §5 Seam 3
      validation/                     # the freshness check (§4)
  claims/                             # append-only, edge-written, curator-drained
~/.jaato/wiki/                        # universal scope — same shape
```

Two tiers mirroring the existing config tiering exactly — `scope:
project` under the workspace, `scope: universal` under `~/.jaato` — which
is the split The School's §3.8 already argues for and the search path
already implements for profiles, agents and references.

---

## 10. How you would know it worked

The metric is not article count, and it is not retrieval count. Both go
up when the system is failing.

The metric is **re-derivation rate**: how often an agent rediscovers
something the wiki already knows. It is measurable at the one place that
sees both — the curator, noticing that an incoming claim duplicates an
existing article. Rising duplicate rate against a growing wiki means the
read path is not working, whatever the retrieval counters say.

Supporting numbers, each attached to a specific way it could be failing:

| Measure | What a bad value means |
|---|---|
| index tokens vs. article-body tokens read | the index is carrying weight it should be delegating, or nobody is opening anything |
| freshness-check pass rate | knowledge is rotting faster than it is curated |
| injected-and-acted-on ratio | articles are charging rent (§8) |
| revisions per article | 1.0 forever means it is a memory store with better formatting (§1) |

And the honest negative, stated in advance so it is not argued away
later: **if re-derivation does not drop, the wiki is a cost with no
benefit** — an index paid on every request, a curator session per cycle,
and a maintenance surface. `jaato-eval` is the place that could answer it,
by running the same task corpus with the wiki injected and withheld.

---

## 11. Open questions

1. **Is the article the right grain, or is the *section*?** `CLAUDE.md`'s
   sections are ~40 topics; some are three paragraphs, some are two
   thousand words. A fixed grain may be wrong in both directions.
2. **Who writes the index line?** If the curator writes it, it drifts
   from the article. If it is generated, it costs a model call per edit.
   If it is the article's first line, the article's drafting is
   constrained by an index it cannot see.
3. **Does the talk page need to be a wiki page at all**, or is it just
   the claims log with the article's decision appended?
4. **What happens to an article when the subsystem it describes is
   deleted?** Superseded-by handles replacement; deletion has no link.
5. **Can an agent be trusted to write the `validation/` script for its
   own claim?** A check the claimant authored is a check that passes.
6. **At which boundary is a human identity stripped?** §8 says it must
   be, and the candidates are all defensible and incompatible: at export
   from a bundle, at the `project` → `universal` promotion, at the git
   remote, or never (an internal team wiki where the name is the point).
   Whichever is chosen, it has to be *mechanical* — a policy that relies
   on a curator remembering is a policy that leaks on the first busy day.
7. **Who declares a `rel` edge (§5, Seam 3) — and does a wrong one cost
   more than no edge at all?** An inferred edge is imprecise and
   self-healing; a declared `supersedes` pointing the wrong way
   *actively suppresses* the article that should have been read. The
   asymmetry says declared edges want a narrower writer than declared
   articles do, which may mean the curator and nobody else.
6. **Does the wiki version with the code?** A git-tracked wiki answers
   *"what did we believe at commit X"* for free, and makes every branch a
   fork of the knowledge base — which may be an excellent property or a
   merge nightmare, and the difference is not obvious from here.

---

## 12. Deliberately not proposed here

- **A new plugin.** §5 argues the ask is a write path on an existing
  store, and a wiki plugin would be a third copy of the matcher, the
  reconcile pass and the enrichment wiring.
- **Replacing memory or references.** The wiki is the *join* between
  them: references' read path and bundles, memory's write path and
  provenance, and a merge policy that makes the unit a topic.
- **Retiring `CLAUDE.md`.** It is the working proof, and the honest
  sequence is to find the article grain by splitting it experimentally,
  not to commit to a mechanism first.
- **Inline editing during a session.** §6 gives the reason; it is the
  single decision most likely to be revisited, and it should be revisited
  with a measurement rather than an argument.
