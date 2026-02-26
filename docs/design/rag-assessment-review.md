# RAG + Jaato References Assessment: Review & Corrections

This document reviews the prior brainstorming assessment on RAG integration
with the jaato references plugin. It identifies inaccuracies, corrects the
comparison table, and refines the architectural recommendations in light of
what the references plugin already implements.

---

## 1. Key Factual Correction: Auto-Detection Is Already Implemented

The assessment states:

> "Right now the model or user must manually pick references."

**This is incorrect.** The references plugin already implements automatic,
passive reference detection on *all* text flowing through the model — both
user prompts and tool call results. Here is how it works:

### Tag-Based Auto-Detection (Pass 2 of `_enrich_content()`)

Every user prompt and every tool result passes through the enrichment pipeline.
The plugin subscribes to both `enrich_prompt()` and `enrich_tool_result()` hooks.
For each piece of content, it:

1. **Collects unselected SELECTABLE sources** that have tags defined.
2. **Builds regex patterns** from each tag with:
   - Case-insensitive matching (`content.lower()`)
   - Separator normalization: hyphens, spaces, and underscores are treated as
     interchangeable (`re.sub(r'\\-|\\ |_', '[ _-]', escaped)`)
   - Word boundary guards using `(?<![a-zA-Z0-9_./-])` and `(?![a-zA-Z0-9_./-])`
     to prevent false matches inside dotted names (`java.util.List`), file paths
     (`/usr/lib/java/`), or compound identifiers
3. **Scans the content** for matches against these patterns.
4. **Injects lightweight hint blocks** into the content so the model is made
   aware that relevant references exist and can call `selectReferences`.

Example of an automatically injected hint:
```
---
**Reference sources available** — use `selectReferences` with IDs or tags to select:

- @mod-code-001-circuit-breaker: MOD-001: Circuit Breaker (matched: circuit-breaker, resilience4j)
- @adr-004-resilience-patterns: ADR-004: Resilience Patterns (matched: resilience)
---
```

### @reference-id Expansion (Pass 1 of `_enrich_content()`)

If any text (user prompt or tool output) contains `@reference-id` matching a
known catalog source, the plugin immediately authorizes the path and injects
full reference instructions — no tool call needed.

### Preselected Reference Read Detection (Tool Result Enrichment)

When the model reads a file using CLI or readFile, the plugin checks whether
that file path belongs to a preselected reference (via an index built at
initialization: `_preselected_paths`). If detected, it:
- Sets `pinned_reference` metadata for GC protection
- Annotates with reference-context (templates, validation, policies, scripts)
  when the model is reading a root-level markdown in the reference directory

### What This Means for the Assessment

The current system is **not** a "manual/declarative RAG." It is a
**hybrid auto-detection + declarative** system:

- **Auto sources** are injected into system instructions at startup
- **Preselected sources** are configured per-agent and transitively resolved
- **Tag-based detection** passively monitors all flowing text and surfaces
  relevant references as hints
- **@reference-id expansion** provides direct injection from any text surface
- **`selectReferences` tool** exists for the model to explicitly pull in
  references it decided it needs (the model's equivalent of "manual pick")

The `selectReferences` tool is the *fallback*, not the primary mechanism. In
practice, between auto sources, preselected sources with transitive discovery,
tag-based hints, and @id expansion, most references are surfaced without the
model ever needing to call `selectReferences`.

---

## 2. Corrected Comparison Table

The assessment's comparison table needs updating:

| RAG Component | Classic RAG | Current References Plugin |
|---|---|---|
| Knowledge catalog | Vector store with embeddings | JSON catalog with metadata + tags |
| Retrieval trigger | Every query, automatically | **Every prompt AND every tool result, automatically** (via enrichment pipeline) |
| Matching | Cosine similarity on embeddings | **Tag-based regex with separator normalization + @id pattern matching + path containment detection** |
| Chunk selection | Top-K nearest neighbors | Whole documents (with transitive discovery of related docs) |
| Injection | Stuffed into prompt | **Multi-layered**: system instructions (auto), lightweight hints (tag matches), full expansion (@id), tool result (selectReferences) |
| Grounding | Implicit | Explicit — model fetches and reads the source files |

The "Retrieval trigger" row was the most significantly wrong. The current system
triggers on every prompt *and* every tool result — it's arguably *more*
aggressive than classic RAG, which typically only retrieves on the user query.

---

## 3. What RAG Would Actually Add (Revised)

Given that auto-detection already exists, the value proposition of
embedding-based RAG changes. Here's the revised analysis:

### 3.1 Semantic Matching Beyond Keywords (Genuine Value)

The current tag-based matching is **syntactic** — it matches literal words
(with separator normalization). It cannot match:

- **Synonyms**: "retry policy" won't match a reference tagged with
  "resilience" unless "retry" is also a tag
- **Conceptual similarity**: "How should we handle transient failures?" won't
  match "circuit-breaker" unless those exact words appear
- **Paraphrases**: "prevent cascading service outages" is semantically close
  to "circuit-breaker" but won't match syntactically

Embedding-based retrieval would catch these. This is the **strongest argument**
for RAG — it fills the gap between what tags can express and what the user
actually means.

However, the gap is smaller than it first appears. The tag system benefits
from the fact that:
- Tags are human-curated and tend to cover the common search terms
- The model can (and does) call `listReferences` to discover available
  sources and their tags, then call `selectReferences` — this is the model
  doing its own "semantic matching" based on its understanding of the catalog

### 3.2 Sub-Document Retrieval (Moderate Value)

The assessment correctly identifies this as valuable for large reference
directories. The current system loads whole documents. For reference sources
that are directories with many files, this means the model must read through
everything. RAG could surface only the relevant sections.

However, the current architecture partially mitigates this: the model reads
files individually using CLI/readFile (it doesn't get all content stuffed into
the prompt), and the `contents` annotation system guides the model to the
right subdirectories (templates/, validation/, policies/, scripts/).

### 3.3 Cross-Reference Discovery (Low Incremental Value)

The assessment suggests embeddings could find semantically related references
even when they don't mention each other. But the current transitive discovery
system (`_resolve_transitive_references()`) already does this structurally:

- ID matching across content via BFS
- Path matching via markdown link extraction and relative path resolution
- Multi-depth traversal (up to 10 levels)
- Multi-parent tracking

This structural approach is actually **more reliable** than embedding
similarity for reference graphs because it captures *intentional* relationships
(author explicitly referenced another document) rather than *incidental*
similarity (two documents happen to use similar words).

Embeddings would add value only for discovering relationships that authors
*should have* documented but didn't — which is a real but narrow use case.

### 3.4 Conversation-Aware Re-Retrieval (Already Partially Implemented)

The assessment suggests RAG could re-retrieve on each turn. But the current
system already does this:

- `enrich_tool_result()` runs on every tool output, so as the conversation
  progresses and the model reads files or gets CLI output, new tag matches
  can surface new references
- `enrich_prompt()` runs on every user message

What's *not* done is re-evaluating the full conversation history for new
semantic matches. This is where embedding-based re-retrieval on the
accumulated context would genuinely help — but at significant computational
cost per turn.

---

## 4. Revised Architecture Recommendation

### Option A (Assessment's Recommendation) Is Still Correct

Embedding RAG as a layer inside the references plugin remains the right call.
The enrichment pipeline hooks (`enrich_prompt`, `enrich_tool_result`) are the
natural integration points. The change would be:

**Current Pass 2** (tag-based matching):
```
tags → regex patterns → word boundary match → hint injection
```

**Enhanced Pass 2** (hybrid matching):
```
tags → regex patterns → word boundary match → syntactic matches
                                              ↓
query → embed → cosine similarity search → semantic matches
                                              ↓
                              merge(syntactic, semantic) → hint injection
```

The hybrid approach is important: tag matching should **not** be replaced by
embeddings, it should be **supplemented**. Tag matching is:
- Deterministic and debuggable
- Zero-latency (no embedding API call)
- Reliable for exact-term matches

Embeddings add the semantic dimension but introduce latency and
non-determinism.

### Embedding Strategy Consideration

The assessment's suggestion of embedding "metadata + content" is right, but
the hierarchy matters:

1. **Embed reference descriptions** (short, high-signal) — cheapest, fastest
2. **Embed tag lists as natural language** (e.g., "circuit breaker, java,
   resilience4j, module") — combines structured tags with embedding space
3. **Embed content summaries** (auto-generated from first N chars of each
   reference) — middle ground
4. **Embed chunked content** (full RAG) — most expensive, needed only when
   catalog grows large

For a catalog of dozens to low-hundreds of references, option 1+2 is
sufficient and avoids the complexity of content chunking entirely.

### Revised Scoring Matrix

The assessment's scoring table is the most practical part of the
assessment, which I substantially agree with. Let me revise and add nuance to it:

| Approach | Effort | Value | When |
|---|---|---|---|
| Tag-based auto-detection (current) | Done | **Excellent** for structured catalogs with good tagging | Now |
| Embedding auto-select on metadata | Small (~200 LOC) | **Moderate** — fills synonym/paraphrase gap | When users report "it didn't find the right reference" |
| Hybrid tag + embedding scoring | Small-Medium (~300 LOC) | **Good** — best of both worlds | Natural evolution of metadata embedding |
| Full RAG with content chunking | Medium (~500 LOC + deps) | Handles large docs, sub-document retrieval | When individual references exceed ~50 pages |
| Conversation-aware re-retrieval | Large | Best UX, but high per-turn cost | Only if semantic drift is a real problem in practice |

---

## 5. Additional Observations

### What the Assessment Got Right

- The **advantages over naive RAG** section is accurate and insightful:
  whole-document context, transitive discovery, structured content types,
  human-in-the-loop channels — these are real differentiators
- The **"when NOT to RAG"** section is exactly right: subagent profiles with
  deterministic knowledge sets should stay declarative
- The **minimal first step** is pragmatic and well-scoped
- The **embedding provider** analysis leveraging existing `model_provider`
  plugins is architecturally sound
- The **storage** analysis is right — for this catalog size, numpy cosine
  similarity on cached embeddings is sufficient; no vector DB needed

### What the Assessment Understated

- **The enrichment pipeline is the killer feature.** The ability to inspect
  *all* text flowing through the model (not just user queries) means the
  system catches references in tool outputs, file contents, and error
  messages. Classic RAG only retrieves on the user's query. This is a
  significant architectural advantage that the assessment's comparison table
  didn't adequately capture.

- **The channel system provides governance** that RAG systems lack. The
  webhook and file channels allow external approval workflows before
  references are injected. This matters in enterprise contexts where you
  can't just auto-inject arbitrary knowledge into model context.

- **The `contents` annotation system** (templates, validation, policies,
  scripts) is not just "structured content types" — it's an instruction
  framework that tells the model *what to do* with the reference, not just
  *what to read*. RAG systems retrieve text; the references plugin retrieves
  text + behavioral directives.

### What the Assessment Overstated

- **"This would replace the selectReferences tool call entirely for common
  cases."** Even with RAG, `selectReferences` would still be needed. The
  model uses it after seeing hints (whether from tag matching or RAG) to
  formally select references and get their paths authorized. RAG would
  improve the *hint generation* but not eliminate the selection step.

- **The comparison to "manual/declarative RAG."** This framing undersells the
  current system. It has automatic detection, automatic transitive discovery,
  and automatic content annotation. Calling it "manual" is misleading.

---

## 6. Recommended Next Step

If the goal is to add semantic matching capability, the pragmatic path is:

1. **Add an `embedding` field to `ReferenceSource`** (cached vector)
2. **At `initialize()`, embed descriptions + tags** using the session's model
   provider (with a local cache in `.jaato/embeddings.json`)
3. **In `_enrich_content()` Pass 2, add a parallel semantic branch**: embed
   the content, cosine-match against the catalog, merge with tag matches
4. **Use a score threshold** to avoid injecting noise — tag matches should
   have higher base confidence than embedding matches since they're exact
5. **Log semantic matches in trace** so the behavior is debuggable

This preserves everything the current system does well while adding the
semantic dimension. The tag-based matching remains the fast, deterministic
primary path; embeddings provide a semantic safety net for cases where tags
don't cover the user's vocabulary.
