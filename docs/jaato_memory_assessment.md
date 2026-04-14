# Jaato Memory Mechanisms — Assessment Against the LLM Memory Design Map

This document assesses jaato's memory mechanisms against the axes laid out in
*"Everything you need to know about LLM memory"* (Rosebud Journal). The article
frames every memory system as a path through nine design axes; this document
walks each axis and locates jaato's choices on it.

---

## Inventory of jaato's memory-carrying mechanisms

| # | Mechanism | Role |
|---|---|---|
| 1 | `SessionHistory` (`shared/session_history.py`) | Canonical in-session conversation (raw messages) |
| 2 | `InstructionBudget` (`shared/instruction_budget.py`) | Token accounting + per-source GC policy (`LOCKED` / `PRESERVABLE` / `PARTIAL` / `EPHEMERAL`) |
| 3 | `gc_truncate` / `gc_summarize` / `gc_hybrid` | Context-window GC strategies |
| 4 | `memory` plugin — workspace `.jaato/memories.jsonl` + global `~/.jaato/memories.jsonl` | Agent-curated long-term knowledge |
| 5 | Memory **curation lifecycle** ("The School": `raw → validated → escalated → dismissed`) | Background housekeeping by advisor agent |
| 6 | `references` plugin | Static/curated documentation catalog with optional embeddings |
| 7 | Pinned preselected references (`SYSTEM.SELECTED_REFERENCES`) | GC-surviving reference content |
| 8 | Prompt-enrichment pipeline (`EnrichmentPlugin`) | Hook-driven injection of hints into the prompt |
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
  session write raw memories; GC fires synchronously when
  `InstructionBudget.utilization_percent` crosses `threshold_percent`
  (default 80%) or continuously after each turn under `pressure_percent` mode.
- **Asynchronous (background):** The **curation lifecycle** is the article's
  "nightly auto-dreams" analogue — a separate advisor agent later runs
  `get_pending_curation()` over `raw` memories and transitions them to
  `validated`, `escalated`, or `dismissed`. This is decoupled from the working
  session.
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
**maturity lifecycle** — raw writes are cheap and tentative, and the advisor
later filters them.

---

## Axis 4 — Where it gets stored

Jaato is **filesystem-first**, explicitly avoiding vector / graph DBs for the
core path.

- **Filesystem (JSONL):** `<workspace>/.jaato/memories.jsonl` (project scope)
  and `~/.jaato/memories.jsonl` (universal scope). Append-for-create,
  rewrite-whole-file for update / delete. The article names OpenClaw and
  Claude Code as filesystem exemplars; jaato fits alongside them.
- **Filesystem + sidecar vectors:** `references` uses a JSON catalog plus an
  optional `.npy` embedding matrix (pgvector-style inline, not a managed
  vector DB).
- **In-memory:** `SessionHistory`, `InstructionBudget`, `MemoryIndexer`
  (tag → ID map), `InstructionTokenCache`.
- **Disk snapshot:** `session` plugin serializes `SessionState` for restart.
- **No SQL, no NoSQL, no managed vector DB, no graph DB.**

The trade-off the article implies: filesystem makes provenance and human
auditing trivial (you can `cat` the JSONL), but forces filename + grep +
model-navigation as the primary retrieval modality, which is weaker than
semantic search at conceptual queries.

---

## Axis 5 — How it gets retrieved

Deliberately conservative.

- **Memory plugin:** **Exact tag matching** (case-insensitive) with
  `min_overlap=2` required. No semantic similarity on the memory store
  itself. The indexer explicitly documents the reason: "prevents false
  positives from substring matches against large prompts where short tags
  like 'test' would match thousands of unrelated words." Results ranked by
  overlap count, then recency.
- **References plugin:** Hybrid — tag matching + optional semantic similarity
  via `EmbeddingProviderProtocol` / `SemanticMatcherProtocol`. This is the
  only place in jaato where embeddings participate.
- **Session history:** Direct list access; no retrieval layer — it's all in
  context until GC.
- **Filesystem navigation:** General CLI tools (`grep`, `read`) give the model
  the "filesystem exploration" retrieval path for anything outside the
  structured stores.

Compared to the article's table: jaato's *memory* retrieval is closer to
Claude Code (pointer + explicit navigation) than to Rosebud V1 (Pinecone
semantic similarity). Its *references* retrieval resembles QMD (BM25 +
embedding hybrid).

---

## Axis 6 — Post-retrieval processing

Minimal but present.

- **Token-budget trimming:** `InstructionBudget` tracks every injected byte
  and `GCPlugin.collect()` enforces the budget.
- **Deduplication:** `MemoryIndexer` deduplicates memory IDs within
  `_tag_index` buckets (an ID is never listed twice under the same tag).
- **Filtering by metadata:** Maturity filter
  (`ACTIVE_MATURITIES = {raw, validated}`) excludes `escalated` (represented
  by references instead) and `dismissed` entries from enrichment hints.
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
| **Hook-driven** | The enrichment pipeline runs before the model sees the turn. `MemoryPlugin.enrich_prompt()` extracts keywords from the user message, finds matching active memories via the indexer, and injects **lightweight hints** (not full content) into the prompt. The reference plugin does the same for `@reference` mentions. |
| **Tool-driven** | Hints tell the model what's available; it then calls `retrieve_memories` / `selectReferences` to pull full content. This is Phase 2 of jaato's documented "two-phase retrieval." |

This composition directly addresses the article's critique that tool-driven
retrieval fails because "the model doesn't know what it doesn't know."
Jaato's hook-driven hint injection is precisely the solution — surface
existence cheaply, let the model decide to pay for content.

---

## Axis 8 — Who's doing the curating

All five of the article's curator types are present.

| Curator | Jaato role |
|---|---|
| **Harness** | `InstructionBudget` policy enforcement, GC trigger logic, indexer build, enrichment pipeline plumbing |
| **Cheap model** | GC summarizer callable (can be wired to a smaller / cheaper provider for `gc_summarize` / `gc_hybrid`) |
| **Main model** | Every `store_memory`, `retrieve_memories`, `selectReferences` call — the working agent is the primary write curator |
| **Background process** | The **advisor agent** (the curation lifecycle) — a separate `JaatoSession` that reviews pending-curation memories and transitions maturity states |
| **User** | `references select` / `memory` commands, `reset`, manual editing of `.jaato/memories.jsonl` |

Jaato's **multi-curator composition** is its most distinctive design choice on
this axis. The article notes that each curator type has different cost /
quality / accountability profiles; jaato pays the "main model on every turn"
cost only for writes, the "background LLM" cost only for curation, and the
"harness" cost for the hot path. This layered assignment is closest to
MemPalace in the article's comparison table.

---

## Axis 9 — Forgetting

This is where jaato's design is most sophisticated and where it also inherits
the article's hardest problems.

**What gets forgotten:**

- **Decay-by-importance:** The `GCPolicy` enum (`LOCKED` / `PRESERVABLE` /
  `PARTIAL` / `EPHEMERAL`) attaches an importance signal to every byte in
  the context. `ConversationTurnType.WORKING` is `EPHEMERAL` (verbose tool
  output goes first); `ORIGINAL_REQUEST` is `LOCKED` (never GC'd).
- **Overwrite-when-superseded:** `update_memory` mutates fields in place;
  `maturity="dismissed"` is a soft forget (kept for audit, hidden from
  enrichment).
- **Bulk forgetting:** `reset` command clears the entire session history;
  deleting the JSONL file forgets all memories in that scope.
- **Append-only (effectively):** Dismissed memories are *hidden*, not
  deleted — the article's "kept for audit trail" pattern.

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
| Session amnesia | **Low** — memory plugin + global `~/.jaato/memories.jsonl` + session persistence cover this |
| Entity confusion | **Medium** — no entity extraction layer, but tag-based retrieval avoids the "two Lunas" problem by requiring exact tag matches |
| Over-inference | **Medium** — the `evidence` field and the `raw → validated` gate are explicit guardrails, but nothing prevents over-confident initial writes |
| Derivation drift | **Medium–high** — `gc_summarize` / `gc_hybrid` can summarize-over-summaries on long sessions; no provenance back to raw |
| Retrieval misfire | **Low** — exact tag matching + `min_overlap=2` is intentionally strict, at the cost of recall |
| Stale context dominance | **Medium** — recency tiebreak in the indexer helps; no decay weighting on `usage_count` |
| Selective retrieval bias | **High** — tag-based exact match only surfaces what the author of the write tagged correctly; cross-topic relevance is invisible |
| Compaction information loss | **Medium–high** — GC removes `EPHEMERAL` tool outputs wholesale; no "compacted with pointer to raw" pattern like the article's *Lossless Claw* |
| Confidence without provenance | **Low** — every `Memory` carries `source_session`, `source_agent`, `evidence`, `confidence` |
| Memory-induced bias | **Inherent** — unavoidable for any hint-injection system; `min_overlap=2` limits the blast radius |

---

## Where jaato sits on the article's comparison table

Adding a jaato column following the article's schema:

| Axis | jaato |
|---|---|
| **What** | Raw (`SessionHistory`) + derived (agent-written `Memory` records, GC summaries) + externally curated (`references`) |
| **When derived** | Synchronous (GC, `store_memory`) + async (advisor curation loop) + on-demand (reference resolution) |
| **Write trigger** | LLM-as-curator (memory writes) + heuristic (GC thresholds) + user-triggered (references, reset) |
| **Curator** | Main model (writes) + background model (curation / summaries) + harness (GC, enrichment plumbing) + user (commands) |
| **Where** | Filesystem JSONL (workspace + `~/.jaato`) + sidecar `.npy` for reference embeddings + disk snapshot for session state |
| **When retrieved** | Always-injected (`LOCKED` system blocks + pinned refs) + hook-driven (enrichment hints) + tool-driven (`retrieve_memories`, `selectReferences`) |
| **How retrieved** | Exact tag matching + recency tiebreak (memories) / hybrid tag + embedding (references) / filesystem + grep (everything else) |
| **Post-retrieval** | Maturity filter + `min_overlap` threshold + `InstructionBudget` token trimming. No re-ranker, no LLM narrowing |
| **Forgetting** | Decay-by-policy (`LOCKED` / `PRESERVABLE` / `PARTIAL` / `EPHEMERAL`) + GC threshold + soft-delete via `dismissed` maturity + hard-delete on user command. No provenance cascade |

---

## Summary judgement

Against the article's map, jaato is a **conservative, filesystem-first,
multi-curator** system whose distinctive feature is the
**maturity-lifecycle curation layer** bridging the working agent and a
background advisor. It buys low retrieval-misfire rates at the cost of higher
selective-retrieval-bias, and it deliberately avoids managed vector / graph
stores in favor of JSONL + tag matching + optional sidecar embeddings.

**Weakest axes against the article's taxonomy:**

1. No provenance cascade for forgetting.
2. No post-retrieval re-ranking.
3. No cross-topic / semantic retrieval on the core memory store.
4. Derivation drift is possible under `gc_summarize` because compacted
   summaries don't carry pointers back to raw turns.

**Strongest axes:**

1. Explicit GC policy per byte in context (`InstructionBudget` + `GCPolicy`).
2. Layered curators with appropriate cost profiles.
3. Explicit `evidence` / `confidence` / `source_*` fields on every memory
   (provenance-first design).
4. Two-phase hint-then-fetch retrieval that dodges the tool-driven-retrieval
   failure mode identified in the article.

---

## Source

Article: *Everything you need to know about LLM memory*, Rosebud Journal
(Notion), URL:
`https://rosebudjournal.notion.site/Everything-you-need-to-know-about-LLM-memory-33b328e8e3f780858d3df3acb06d23b9`
