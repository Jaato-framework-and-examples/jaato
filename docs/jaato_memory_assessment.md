# Jaato Memory Mechanisms — Assessment Against the LLM Memory Design Map

This document assesses jaato's memory mechanisms against the axes laid out in
*"Everything you need to know about LLM memory"* (Rosebud Journal). The article
frames every memory system as a path through nine design axes; this document
walks each axis and locates jaato's choices on it.

> **Last updated:** 2026-04-14 — reflects the raw/curated split storage
> layout, sentence-coherence tag matching, and the curator-only enrichment
> path landed in commits `3f019999`, `d5ad97f3`, `b97dda14`, and `e0afcda0`.

---

## Inventory of jaato's memory-carrying mechanisms

| # | Mechanism | Role |
|---|---|---|
| 1 | `SessionHistory` (`shared/session_history.py`) | Canonical in-session conversation (raw messages) |
| 2 | `InstructionBudget` (`shared/instruction_budget.py`) | Token accounting + per-source GC policy (`LOCKED` / `PRESERVABLE` / `PARTIAL` / `EPHEMERAL`) |
| 3 | `gc_truncate` / `gc_summarize` / `gc_hybrid` / `gc_budget` | Context-window GC strategies |
| 4 | `memory` plugin — workspace `<ws>/.jaato/memories/` + global `~/.jaato/memories/` (split: `raw/{id}.json` + `curated.jsonl`) | Agent-curated long-term knowledge |
| 5 | Memory **curation lifecycle** ("The School": `raw → validated → escalated → dismissed`) — implemented as a queue (raw folder) drained by the `memory-advisor` reactor | Background housekeeping by advisor agent |
| 6 | `references` plugin | Static/curated documentation catalog with optional embeddings (hybrid tag + semantic) |
| 7 | Pinned preselected references (`SYSTEM.SELECTED_REFERENCES`) | GC-surviving reference content |
| 8 | Prompt-enrichment pipeline (`EnrichmentPlugin`) — runs on **user prompts** AND **tool results** | Hook-driven injection of hints into the model's context |
| 9 | Session persistence (`session` plugin) | Disk snapshot of session state across restarts |
| 10 | `InstructionTokenCache` | Content-addressed token-count cache |

---

## Axis 1 — What gets stored (Raw ↔ Derived)

Jaato straddles the spectrum deliberately rather than picking a position.

- **Raw end:** `SessionHistory` keeps verbatim `Message` objects (user turns,
  tool calls, tool results, assistant text). Session persistence snapshots
  this raw list to disk.
- **Derived end:**
  - `gc_summarize` / `gc_hybrid` produce **session summaries** when the context
    fills.
  - The **memory plugin** stores LLM-written `Memory` records with `content`,
    `description`, `tags`, `confidence`, `evidence`, `scope`. These are the
    article's *self-directed prompts* and *cross-session inferences*.
  - `references` is **externally authored derived** material (curated docs,
    specs), optionally accompanied by **embeddings** (sidecar `.npy` matrix).

What's missing versus the article's catalog: no **graph data** (no
entity/relationship triples like Zep), no **topic fan-out summaries**, no
automatic **daily / weekly / monthly rollups**.

---

## Axis 2 — When derivation happens

Jaato uses all three timings the article lists.

- **Synchronous (at conversation time):** `store_memory` tool calls during a
  session write to the **raw queue**; GC fires synchronously when
  `InstructionBudget.utilization_percent` crosses `threshold_percent`
  (default 80%) or continuously after each turn under `pressure_percent` mode.
- **Asynchronous (background):** The **curation lifecycle** is the article's
  "nightly auto-dreams" analogue — the `memory-advisor` reactor agent fires on
  `agent.completed` events, drains the raw queue, and transitions entries to
  `validated`, `escalated`, or `dismissed`. Decoupled from the working session.
- **On-demand (retrieval time):** The `references` plugin resolves and fetches
  content lazily when the model invokes `selectReferences` or when prompt
  enrichment detects `@reference` mentions.

---

## Axis 3 — What triggers a write

Multi-curator, not a single policy.

| Trigger type | Jaato mechanism |
|---|---|
| Write everything | `SessionHistory.append()` captures every turn verbatim |
| Heuristic | GC trigger (`GCTriggerReason.THRESHOLD`, `TURN_LIMIT`, `PRE_MESSAGE`, `CONTEXT_LIMIT`) produces summary artifacts |
| LLM-as-curator | The working agent decides when to call `store_memory` during a turn — the tool docstring explicitly instructs "only store substantial, reusable information" |
| User-triggered | `references select <id>` command, memory curation via `update_memory`, explicit `reset` command |

The LLM-as-curator path carries the article's critique: the working agent is
bad at predicting what will matter later. Jaato mitigates this with the
**raw-queue + advisor pattern** — raw writes are cheap and tentative, never
surfaced as hints, and the advisor later promotes the worthwhile ones into the
curated store.

---

## Axis 4 — Where it gets stored

Jaato is **filesystem-first**, explicitly avoiding vector / graph DBs for the
core path. The memory store uses a **split layout** matched to the producer /
consumer access pattern:

- **Raw queue (folder-of-files):** `<base>/raw/{id}.json`. Write-many (any
  agent storing a memory), read-by-curator. Each producer writes its own file
  atomically via `tempfile + os.rename`, eliminating write contention. Raw
  memories are never surfaced as enrichment hints.
- **Curated store (single JSONL):** `<base>/curated.jsonl`. Read-many (every
  enrichment pass), write-by-curator-only. The curator drains raw entries
  into this file; rewrites use `tempfile + os.rename` so concurrent readers
  see either the old or new file in full, never half-state.
- **Two scopes:** workspace-local under `<workspace>/.jaato/memories/` and
  cross-session global under `~/.jaato/memories/`. Same split layout each.
- **Filesystem + sidecar vectors:** `references` uses a JSON catalog plus an
  optional `.npy` embedding matrix (pgvector-style inline, not a managed
  vector DB).
- **In-memory:** `SessionHistory`, `InstructionBudget`, `MemoryIndexer`
  (canonical-tag → ID map, built from curated only), `InstructionTokenCache`.
- **Disk snapshot:** `session` plugin serializes `SessionState` for restart.
- **No SQL, no NoSQL, no managed vector DB, no graph DB.**

The trade-off the article implies: filesystem makes provenance and human
auditing trivial (you can `cat` the JSONL), but forces filename + grep +
model-navigation as the primary retrieval modality, which is weaker than
semantic search at conceptual queries.

---

## Axis 5 — How it gets retrieved

Conservative on the memory side, hybrid on references.

- **Memory plugin — sentence-coherence tag matching** (shared
  `shared/tag_coherence.py` engine):
  - The retrieval text is segmented on sentence terminators (`.!?\s+`) and
    line breaks, with a 250-char per-segment cap. This stops long structured
    dumps from trivially satisfying multi-component coherence by chance.
  - **Single-component tags** (`java`, `gpg`, `gc`, `spring.boot`) use strict
    word-boundary regex — adjacent alphanumerics, hyphens, underscores, dots,
    or slashes block the match. So `java` matches "the Java SDK" but NOT
    "java.util.concurrent" or "Foo.java".
  - **Multi-component tags** (`circuit-breaker`) match if the full string
    appears verbatim in any segment OR if all of its ≥3-char components
    co-occur in one segment. Hyphen / underscore / colon / space are
    interchangeable separators — `circuit-breaker` matches `circuit_breaker`
    or `circuit breaker`.
  - **Dots are treated as qualifiers**, not separators — `spring.boot` stays
    atomic so it doesn't match inside `org.spring.boot.autoconfigure`.
  - **Curated-only:** the indexer is built solely from `curated.jsonl`. Raw
    memories are queue-only and never surface as hints.
  - Results ranked by overlap count, then recency. ID-based fetch
    (`retrieve_memories(ids=[...])`) bypasses tag matching entirely.
- **References plugin:** Hybrid — same sentence-coherence tag engine
  (extracted from memory) plus optional semantic similarity via
  `EmbeddingProviderProtocol` / `SemanticMatcherProtocol`. The semantic
  veto in hybrid mode filters spurious component matches.
- **Session history:** Direct list access; no retrieval layer — it's all in
  context until GC.
- **Filesystem navigation:** General CLI tools (`grep`, `read`) give the model
  the "filesystem exploration" retrieval path for anything outside the
  structured stores.

Compared to the article's table: jaato's *memory* retrieval is closer to
Claude Code (pointer + explicit navigation) than to Rosebud V1 (Pinecone
semantic similarity), but with sentence-coherence relaxation that catches
compound concepts the user mentions naturally. Its *references* retrieval
resembles QMD (BM25 + embedding hybrid).

---

## Axis 6 — Post-retrieval processing

Minimal but present.

- **Token-budget trimming:** `InstructionBudget` tracks every injected byte
  and `GCPlugin.collect()` enforces the budget.
- **Deduplication:** `MemoryIndexer` deduplicates memory IDs within
  `_tag_index` buckets (an ID is never listed twice under the same tag).
- **Filtering by maturity:** Raw memories are excluded at the storage layer
  (they live in the raw queue, not the curated store the indexer reads from).
  `escalated` and `dismissed` are also excluded by `ACTIVE_MATURITIES = {raw,
  validated}` — though raw is moot now since it's not in the curated store.
- **Triggering tags surfaced honestly in the hint:** the enrichment
  notification shows only the tags that actually matched the input text, not
  the full tag set of every surfaced memory (which historically misled the
  agent into firing N retrieve calls).
- **One-call retrieval form:** the hint includes the exact
  `retrieve_memories(ids=[...])` invocation so the agent fetches all
  surfaced memories in a single call.
- **No re-ranking layer, no LLM-based narrowing of candidates, no explicit
  date-range filtering.**

The `references` plugin has stronger post-processing via its semantic
matcher, but the core memory path is intentionally shallow — the article's
observation that "cheap retrieval plus smart re-ranking often beats expensive
retrieval alone" is mostly unrealized in jaato.

---

## Axis 7 — When retrieval happens

Two-phase hybrid that maps cleanly to the article's three modes.

| Article mode | Jaato realization |
|---|---|
| **Always injected** | `SYSTEM` instructions (base + client + framework + **pinned preselected references**). Pinned content survives GC via `SystemChildType.SELECTED_REFERENCES` (`LOCKED`). |
| **Hook-driven** | The enrichment pipeline runs on **both user prompts and tool results**. `MemoryPlugin.enrich_prompt()` and `enrich_tool_result()` share a single core that segments the text, matches against the curated tag index, and injects **lightweight hints** (memory IDs + descriptions + the one-call retrieve form). The references plugin does the same for `@reference` mentions and tool result contents. |
| **Tool-driven** | Hints tell the model what's available and how to fetch them; it then calls `retrieve_memories(ids=[...])` / `selectReferences` to pull full content. This is Phase 2 of jaato's documented "two-phase retrieval." |

This composition directly addresses the article's critique that tool-driven
retrieval fails because "the model doesn't know what it doesn't know."
Jaato's hook-driven hint injection is precisely the solution — surface
existence cheaply, let the model decide to pay for content, and tell it the
exact one-call invocation to use.

---

## Axis 8 — Who's doing the curating

All five of the article's curator types are present.

| Curator | Jaato role |
|---|---|
| **Harness** | `InstructionBudget` policy enforcement, GC trigger logic, indexer build, enrichment pipeline plumbing, raw → curated routing on update_memory |
| **Cheap model** | GC summarizer callable (can be wired to a smaller / cheaper provider for `gc_summarize` / `gc_hybrid`) |
| **Main model** | Every `store_memory`, `retrieve_memories`, `selectReferences` call — the working agent is the primary write curator (writes go to the raw queue) |
| **Background process** | The **`memory-advisor`** agent — a separate `JaatoSession` spawned by the reactor on `agent.completed`. Reads raw queue entries, transitions maturity states, consolidates into the curated store. Cross-process serialization is tracked as a backlog item (reactor singleton). |
| **User** | `references select` / `memory` commands, `reset`, manual editing of `.jaato/memories/curated.jsonl` or removal of files in `.jaato/memories/raw/` |

Jaato's **multi-curator composition** is its most distinctive design choice on
this axis. The article notes that each curator type has different cost /
quality / accountability profiles; jaato pays the "main model on every turn"
cost only for writes (cheap, into raw), the "background LLM" cost only for
curation (advisor agent), and the "harness" cost for the hot path. The split
storage layout is the physical realization of this separation — producers
and consumers don't share a write path.

---

## Axis 9 — Forgetting

This is where jaato's design is most sophisticated and where it also inherits
the article's hardest problems.

**What gets forgotten:**

- **Decay-by-importance:** The `GCPolicy` enum (`LOCKED` / `PRESERVABLE` /
  `PARTIAL` / `EPHEMERAL`) attaches an importance signal to every byte in
  the context. `ConversationTurnType.WORKING` is `EPHEMERAL` (verbose tool
  output goes first); `ORIGINAL_REQUEST` is `LOCKED` (never GC'd).
- **Overwrite-when-superseded:** `update_memory` mutates fields in place and
  may move a memory across stores (raw → curated, or remove on dismissed).
- **Bulk forgetting:** `reset` command clears the entire session history;
  removing files from `memories/raw/` or rewriting `curated.jsonl` forgets
  them.
- **Soft delete:** `maturity="dismissed"` removes the memory from both stores
  in the new layout. (The previous "kept for audit" semantics is gone — when
  the curator dismisses, the file is unlinked.)

**How forgetting propagates:**

This is where jaato does *not* fully solve the article's hardest question.
Provenance tracking exists (`Memory.source_session`, `Memory.source_agent`,
`Memory.evidence`) but there is **no cascade delete**: if the source session
is purged, the memories it wrote stay behind. Similarly, GC-summarized turns
produce a summary message that is now disconnected from the raw content it
replaced — the article's "confidence without provenance" failure mode is
possible here.

**When forgetting happens:**

- **Continuously via decay:** GC fires on threshold (`JAATO_GC_THRESHOLD`,
  default 80%) or continuously under pressure mode.
- **On user request:** `reset`, `memory delete`, manual file edits.
- **Triggered by contradiction:** Partially — the advisor can dismiss a
  memory as "superseded" during curation, but there is no automatic
  contradiction detector.
- **Never (permanent):** `SYSTEM.BASE` (user-provided `.jaato/instructions/*.md`),
  `SYSTEM.FRAMEWORK`, and pinned references are `LOCKED`.

---

## Failure modes — where jaato lands

Mapping the article's failure-mode taxonomy onto jaato.

| Failure mode | Jaato's exposure |
|---|---|
| Session amnesia | **Low** — memory plugin + global `~/.jaato/memories/curated.jsonl` + session persistence cover this |
| Entity confusion | **Medium** — no entity extraction layer, but tag-based retrieval avoids the "two Lunas" problem by requiring exact tag matches or tight component coherence |
| Over-inference | **Low–medium** — the `evidence` field, the `raw → curated` gate (raw never reaches agents), and the curator's review explicitly mediate this |
| Derivation drift | **Medium–high** — `gc_summarize` / `gc_hybrid` can summarize-over-summaries on long sessions; no provenance back to raw |
| Retrieval misfire | **Low** — sentence-coherence matching with strict-boundary single-component tags + segment cap is intentionally strict, at the cost of recall |
| Stale context dominance | **Medium** — recency tiebreak in the indexer helps; no decay weighting on `usage_count` |
| Selective retrieval bias | **Medium** (was HIGH) — sentence-coherence + sub-token component matching catches compound concepts the user mentions naturally; still no cross-topic / semantic retrieval on memory |
| Compaction information loss | **Medium–high** — GC removes `EPHEMERAL` tool outputs wholesale; no "compacted with pointer to raw" pattern like the article's *Lossless Claw* |
| Confidence without provenance | **Low** — every `Memory` carries `source_session`, `source_agent`, `evidence`, `confidence` |
| Memory-induced bias | **Inherent** — unavoidable for any hint-injection system; the curator gate (raw never surfaces) limits the blast radius significantly compared to the prior `min_overlap=2` design |

---

## Where jaato sits on the article's comparison table

Adding a jaato column following the article's schema:

| Axis | jaato |
|---|---|
| **What** | Raw (`SessionHistory`) + derived (agent-written `Memory` records, GC summaries) + externally curated (`references`) |
| **When derived** | Synchronous (GC, `store_memory` → raw) + async (advisor curation loop, raw → curated) + on-demand (reference resolution) |
| **Write trigger** | LLM-as-curator (memory writes, always raw initially) + heuristic (GC thresholds) + user-triggered (references, reset) |
| **Curator** | Main model (writes to raw) + background agent (`memory-advisor`, raw → curated) + harness (GC, enrichment plumbing) + user (commands) |
| **Where** | Filesystem split (raw folder + curated JSONL, workspace + `~/.jaato`) + sidecar `.npy` for reference embeddings + disk snapshot for session state |
| **When retrieved** | Always-injected (`LOCKED` system blocks + pinned refs) + hook-driven (enrichment hints on prompts AND tool results) + tool-driven (`retrieve_memories(ids=...)`, `selectReferences`) |
| **How retrieved** | Sentence-coherence tag matching + recency tiebreak (memories) / hybrid tag + embedding (references) / filesystem + grep (everything else) |
| **Post-retrieval** | Maturity / store filter (curated only) + segment cap + `InstructionBudget` token trimming + triggering-tag-only hints. No re-ranker, no LLM narrowing |
| **Forgetting** | Decay-by-policy (`LOCKED` / `PRESERVABLE` / `PARTIAL` / `EPHEMERAL`) + GC threshold + hard-delete on dismiss + hard-delete on user command. No provenance cascade |

---

## Summary judgement

Against the article's map, jaato is a **conservative, filesystem-first,
multi-curator** system whose distinctive feature is the
**split storage layout (raw queue + curated store)** that physically
implements the producer / consumer separation between agents and the
background advisor. It buys low retrieval-misfire rates with sentence-
coherence matching and the curator gate, at the cost of some
selective-retrieval-bias. It deliberately avoids managed vector / graph
stores in favor of JSONL + tag matching + optional sidecar embeddings.

**Weakest axes against the article's taxonomy:**

1. No provenance cascade for forgetting. If a session is deleted, its
   memories stay behind orphaned.
2. No post-retrieval re-ranking. Tag-coherence finds candidates by structure;
   nothing scores their actual relevance to the prompt before injection.
3. No cross-topic / semantic retrieval on the core memory store. Sentence
   coherence catches more than verbatim tag matches but still misses queries
   phrased entirely without the tag's vocabulary.
4. Derivation drift is possible under `gc_summarize` because compacted
   summaries don't carry pointers back to raw turns — the article's
   *Lossless Claw* pattern is unrealized.

**Strongest axes:**

1. Explicit GC policy per byte in context (`InstructionBudget` + `GCPolicy`).
2. Layered curators with appropriate cost profiles, made physical by the
   raw-queue / curated-store split.
3. Explicit `evidence` / `confidence` / `source_*` fields on every memory
   (provenance-first design).
4. Two-phase hint-then-fetch retrieval that dodges the tool-driven-retrieval
   failure mode identified in the article. The hint surfaces memory IDs
   and the exact one-call retrieval form, eliminating the "N calls per N
   bullets" anti-pattern.
5. Curator-only enrichment: agents only see vetted content as hints, raw
   writes are tentative and cheap.
6. Race-free concurrent producers via per-memory file writes in the raw
   queue (atomic `tempfile + os.rename`).

---

## Open improvement opportunities

Tracked in the project backlog:

- **Semantic retrieval on the memory store** — port the references plugin's
  `EmbeddingProviderProtocol` / `SemanticMatcherProtocol` hybrid into memory.
  Tag-coherence finds candidates fast; embedding ranks them. Addresses the
  Selective Retrieval Bias gap.
- **Provenance cascade for forgetting** — when a session is deleted,
  cascade-delete or orphan-flag its `source_session`-tagged memories.
- **Compacted summary with pointer to raw** — when `gc_summarize` collapses
  turns, persist a pointer to the verbatim raw range so the agent can reach
  back if needed (the article's *Lossless Claw*).
- **Post-retrieval re-ranking** — cheap-model or embedding pass that scores
  candidate relevance before injection.
- **Decay weighting on `usage_count`** — auto-flag never-retrieved memories
  for advisor review.
- **Reactor singleton coordination** — ensure only one `memory-advisor`
  session runs at a time across the daemon.
- **Memory storage cross-process locking** — the per-file atomic writes
  cover concurrent producers; cross-process curator coordination still
  depends on the reactor singleton work.

---

## Source

Article: *Everything you need to know about LLM memory*, Rosebud Journal
(Notion), URL:
`https://rosebudjournal.notion.site/Everything-you-need-to-know-about-LLM-memory-33b328e8e3f780858d3df3acb06d23b9`
