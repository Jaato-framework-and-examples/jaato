# Agent Memory → Knowledge Escalation System ("The School")

## 1. Problem Statement

Jaato agents learn valuable lessons during sessions — debugging insights, project
conventions, failed approaches, successful patterns. Today these lessons are
stored as **memories** (flat JSONL, tag-indexed, no quality gate). Memories
accumulate indefinitely with no review, no deduplication, no contradiction
detection, and no pathway to become structural knowledge.

Meanwhile, the **references** system provides curated, versioned, semantically
indexed knowledge that all agents can consume. But references are static —
created by the `gen-references` prompt or manually. There is no automated
bridge from "something an agent learned" to "a reference all future agents know."

**The gap:** Experiential learning (memories) and structural knowledge
(references) are disconnected systems. Valuable lessons remain locked in the
memory store where they degrade in relevance, while the knowledge base doesn't
grow from operational experience.

## 2. Concept: "The School"

The School is a two-tier knowledge lifecycle that transforms raw agent memories
into curated knowledge references through an advisor-mediated escalation process.

### Metaphor Mapping

| School Concept | System Analogue |
|---|---|
| **Student** | Working agent (any profile) |
| **Field notes** | Raw memories (`maturity: "raw"`) |
| **Teacher/Advisor** | Advisor agent (curator profile) |
| **Peer review** | Consensus from multiple agents encountering the same lesson |
| **Textbook** | Reference entry (graduated knowledge) |
| **Curriculum** | Knowledge hierarchy (ADR → ERI → Module → Skill) |
| **Exam** | Empirical validation by advisor (testing the claim) |
| **Graduation** | Memory → Reference promotion |
| **Alumni knowledge** | `type: "learned"` references available to all future agents |
| **Tenure** | `mode: "auto"` — knowledge so validated it's always injected |

### Lifecycle Flow

```
Agent Session                    Between Sessions                 Future Sessions
─────────────────               ─────────────────                ─────────────────

Agent works on task             Advisor agent wakes up           Agents get knowledge
    │                               │                            injected via references
    ├─ Learns something             ├─ Reads raw memories             │
    │  non-obvious                  │                                 │
    ├─ Stores as MEMORY             ├─ Assesses value                 ├─ No longer need to
    │  (raw, subjective,            │  (Was it correct?                │  "remember" — they
    │   first-person)               │   Is it generalizable?          │  simply "know"
    │                               │   Does it contradict            │
    ├─ Continues working            │   existing knowledge?)          └─ Knowledge is
    │                               │                                    structural, not
    └─ Session ends                 ├─ Promotes to KNOWLEDGE             episodic
                                    │  (reference entry)
                                    │
                                    └─ Removes from memories
                                       (escalated)
```

## 3. Architecture

### 3.1 Component Overview

Three new components, layered on existing infrastructure:

```
┌──────────────────────────────────────────────────────────────────┐
│                     EXISTING INFRASTRUCTURE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  MemoryPlugin          ReferencesPlugin        SessionPlugin      │
│  ┌──────────┐         ┌──────────────┐        ┌────────────┐     │
│  │ store    │         │ catalog      │        │ on_session_ │     │
│  │ retrieve │         │ selectRefs   │        │ _end()     │     │
│  │ list_tags│         │ enrichment   │        │ hooks      │     │
│  └──────────┘         │ embeddings   │        └────────────┘     │
│       │               └──────────────┘              │             │
│       │                      ▲                      │             │
├───────┼──────────────────────┼──────────────────────┼─────────────┤
│       │        NEW COMPONENTS│                      │             │
│       ▼                      │                      ▼             │
│  ┌──────────┐         ┌──────────────┐     ┌────────────────┐    │
│  │ Enhanced │         │ Escalation   │     │ Curation       │    │
│  │ Memory   │────────►│ Mechanism    │     │ Trigger        │    │
│  │ Model    │         │              │     │ (hook +        │    │
│  └──────────┘         └──────────────┘     │  user-init)    │    │
│    maturity              ▲                  └───────┬────────┘    │
│    confidence            │                          │             │
│    evidence              │                          │             │
│    scope                 │                          ▼             │
│    source_agent    ┌─────┴──────────┐      ┌────────────────┐    │
│                    │ Advisor Agent  │◄─────│ Headless Mode  │    │
│                    │ (curator       │      │ (autonomous)   │    │
│                    │  profile)      │      └────────────────┘    │
│                    └────────────────┘                             │
│                      TRIAGE                                      │
│                      VALIDATE                                    │
│                      ASSESS                                      │
│                      PROMOTE / MERGE / RETAIN / DISCARD          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Enhanced Memory Model

Extend the existing `Memory` dataclass with maturity lifecycle fields:

```python
@dataclass
class Memory:
    # --- existing fields (unchanged) ---
    id: str
    content: str
    description: str
    tags: List[str]
    timestamp: str
    usage_count: int = 0
    last_accessed: Optional[str] = None

    # --- new lifecycle fields ---
    maturity: str = "raw"               # raw | validated | escalated
    confidence: float = 0.0             # 0.0–1.0, self-assessed by storing agent
    evidence: Optional[str] = None      # What triggered this learning (error, tool result)
    scope: str = "project"              # project | universal
    source_agent: Optional[str] = None  # Profile name that stored this
    source_session: Optional[str] = None  # Session ID for traceability
    escalated_to: Optional[str] = None  # Reference ID if promoted (lineage)
```

**Backward compatibility:** All new fields have defaults. Existing JSONL files
load without modification — missing fields get default values.

**Maturity states:**

```
     ┌──── store_memory() ────┐
     ▼                         │
   [raw] ──advisor──► [validated] ──advisor──► [escalated]
     │                    │                        │
     │                    │                        └─► Reference created
     │                    │                            Memory frozen
     └──advisor──► [discarded]                         (audit trail)
                   (deleted or
                    archived)
```

- **raw**: Just stored by an agent. Unreviewed. Surfaced in enrichment hints.
- **validated**: Advisor confirmed correctness but not yet mature enough to
  graduate. Still surfaced in enrichment. May need more usage evidence.
- **escalated**: Promoted to a reference. No longer surfaced in memory
  enrichment (the reference takes over). Kept as audit trail with `escalated_to`
  pointing to the reference ID.
- **discarded**: Advisor determined the memory is incorrect, trivial, or
  superseded. Removed from storage.

### 3.3 Memory Plugin Enhancements

#### Enrichment filtering by maturity

The `enrich_prompt()` method currently surfaces all matching memories as hints.
With maturity tracking, it should **exclude escalated memories** — their
knowledge now lives in the references system and will be surfaced by the
references plugin at enrichment priority 20 (before memory at priority 80).

```python
def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
    # ... existing keyword extraction ...
    matches = self._indexer.find_matches(keywords, limit=5)

    # NEW: Filter out escalated memories (knowledge lives in references now)
    matches = [m for m in matches if m.maturity != "escalated"]

    # ... rest of enrichment unchanged ...
```

This requires the indexer to carry the `maturity` field in `MemoryMetadata`.

#### New fields in MemoryMetadata

```python
@dataclass
class MemoryMetadata:
    id: str
    description: str
    tags: List[str]
    timestamp: str
    maturity: str = "raw"    # NEW: for enrichment filtering
    scope: str = "project"   # NEW: for cross-project awareness
```

#### Enhanced `store_memory` tool

The existing tool gains optional parameters for the new fields:

```python
ToolSchema(
    name="store_memory",
    parameters={
        "content": {"type": "string", "required": True},
        "description": {"type": "string", "required": True},
        "tags": {"type": "array", "items": {"type": "string"}, "required": True},
        # NEW optional fields:
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
        "scope": {"type": "string", "enum": ["project", "universal"]},
    }
)
```

The `source_agent` and `source_session` fields are set automatically by the
plugin from the session context (not provided by the model).

#### New curation tools (advisor-only)

These tools are intended for the advisor agent profile. They are `discoverable`
(not core) and only relevant when the memory plugin is loaded in a curation
context.

| Tool | Purpose | Discoverability |
|------|---------|-----------------|
| `list_memories_for_curation` | List raw/validated memories with usage stats | discoverable |
| `validate_memory` | Mark a memory as validated (advisor confirmed) | discoverable |
| `escalate_memory` | Promote a memory to a reference entry | discoverable |
| `discard_memory` | Remove an incorrect/trivial memory | discoverable |
| `merge_memories` | Combine duplicate/related memories into one | discoverable |

**`escalate_memory` is the key tool** — it bridges the two systems:

```python
def _execute_escalate_memory(self, args: dict) -> dict:
    """Promote a validated memory to a learned reference.

    Creates a new reference entry in .jaato/references/ with type 'learned',
    marks the memory as escalated with a back-reference to the new reference ID,
    and optionally computes embeddings for the new reference.

    Args:
        memory_id: ID of the memory to escalate
        reference_id: Desired reference ID (e.g., 'learned-postgres-upsert')
        name: Display name for the reference
        description: Reference description (may differ from memory description)
        content: Curated knowledge content (advisor may rewrite for clarity)
        tags: Reference tags (may differ from memory tags)
        mode: Injection mode - 'selectable' (default) or 'auto'
        scope: 'project' or 'universal'

    Returns:
        dict with status, reference_id, memory_id, reference_path
    """
```

### 3.4 The Advisor Agent Profile

A new profile: `.jaato/profiles/advisor-knowledge-curator.json`

```json
{
  "name": "advisor-knowledge-curator",
  "description": "Reviews raw agent memories, assesses their value through empirical validation, and promotes worthy lessons to structural knowledge references. Operates autonomously in headless mode or with human approval when user-initiated.",
  "plugins": [
    "memory",
    "references",
    "cli",
    "filesystem_query"
  ],
  "plugin_configs": {
    "memory": {
      "curation_mode": true
    }
  },
  "system_instructions": "see section 3.5",
  "max_turns": 20,
  "gc": {
    "type": "budget",
    "threshold_percent": 80.0
  }
}
```

**Plugin selection rationale:**
- `memory`: Read raw memories, validate, escalate, discard, merge
- `references`: Read existing catalog to detect contradictions/duplicates
- `cli`: Run commands to empirically verify claims (e.g., run a test, check a config)
- `filesystem_query`: Explore codebase to verify memory assertions about code structure

### 3.5 Advisor System Instructions

The advisor's system instructions define its assessment methodology:

```markdown
# Knowledge Curator Instructions

You are the Knowledge Curator — an advisor agent responsible for reviewing
raw agent memories and deciding which ones deserve promotion to structural
knowledge (references).

## Assessment Process

For each batch of raw memories:

### Phase 1: TRIAGE
- Group related memories (same topic, similar tags)
- Detect duplicates (same lesson stored by different agents/sessions)
- Flag contradictions (memory A says X, memory B says not-X)

### Phase 2: VALIDATE
- Cross-reference against existing knowledge references
- For codebase assertions: verify by reading the relevant files
- For behavioral claims: attempt to reproduce (run commands, check configs)
- For convention claims: search for counter-examples

### Phase 3: ASSESS
Score each memory on four dimensions:
- **Frequency**: Was this lesson encountered by multiple agents? (check usage_count)
- **Generalizability**: Project-specific or universally applicable?
- **Accuracy**: Does it contradict known references or codebase reality?
- **Uniqueness**: Does existing knowledge already cover this?

### Phase 4: DECIDE
For each assessed memory, choose one action:
- **PROMOTE**: Create a new reference entry (the memory is correct, valuable, unique)
- **MERGE**: Combine with an existing memory or reference (consolidation)
- **RETAIN**: Keep as raw memory (not yet mature enough — needs more evidence)
- **DISCARD**: Remove (incorrect, trivial, outdated, or superseded)

## Promotion Guidelines

When promoting a memory to a reference:
1. **Rewrite for clarity**: The memory was written in first-person by an agent
   during a task. The reference should be third-person, objective, actionable.
2. **Distill**: Remove situational context. Keep the generalizable lesson.
3. **Tag properly**: References need broader, more systematic tags than memories.
4. **Choose mode**: Default to 'selectable'. Only use 'auto' for truly universal
   knowledge that every agent should always see.
5. **Set scope**: 'project' for project-specific conventions, 'universal' for
   cross-project knowledge.

## Output

After processing, produce a curation report summarizing:
- Total memories reviewed
- Promoted (with reference IDs created)
- Merged (with target IDs)
- Retained (with reasoning)
- Discarded (with reasoning)
```

### 3.6 Curation Triggers

Two trigger mechanisms, both converging on the same advisor agent:

#### A. User-Initiated (interactive, human approval)

The user runs a command (e.g., `curate-knowledge` or invokes the advisor
manually). The advisor presents its assessments and the user approves or
rejects each promotion before it takes effect.

This uses the standard TUI client interaction model — the advisor is a
subagent that the main agent spawns, or the user starts a session with
the curator profile directly.

#### B. Session-Finalization Hook (autonomous, headless)

On session end, if raw memory count exceeds a configurable threshold, a
headless curation session is triggered automatically.

**Integration point:** `SessionPlugin.on_session_end()`

The hook:
1. Checks raw memory count against threshold (e.g., ≥ 5 new raw memories)
2. Spawns a headless session with the `advisor-knowledge-curator` profile
3. The advisor runs autonomously with auto-approved permissions
4. Produces a curation report written to `.jaato/logs/curation/`

```python
# In SessionPlugin.on_session_end() or a new CurationHookPlugin

def on_session_end(self, state: SessionState, config: SessionConfig) -> None:
    # ... existing auto-save logic ...

    # NEW: Check if curation is warranted
    raw_count = self._memory_storage.count_by_maturity("raw")
    if raw_count >= self._curation_threshold:
        self._trigger_curation(autonomous=True)

def _trigger_curation(self, autonomous: bool = True) -> None:
    """Spawn a headless advisor session for knowledge curation."""
    if autonomous:
        # Launch headless curation session
        # Uses existing headless mode infrastructure
        # Auto-approve permissions, file-based output
        pass
    else:
        # Queue for next interactive session
        # User will be notified that curation is pending
        pass
```

**Threshold configuration** (environment variable or `.jaato/config.json`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `JAATO_CURATION_THRESHOLD` | 5 | Min raw memories to trigger auto-curation |
| `JAATO_CURATION_AUTO` | true | Enable automatic curation on session end |
| `JAATO_CURATION_LOG_DIR` | `.jaato/logs/curation` | Where curation reports go |

### 3.7 The Escalation Mechanism (Graduation)

When the advisor promotes a memory, `escalate_memory` performs these steps:

```
escalate_memory(memory_id, reference_id, name, description, content, tags, mode, scope)
    │
    ├─ 1. Create reference JSON in .jaato/references/
    │     {
    │       "id": "learned-postgres-upsert-pattern",
    │       "name": "PostgreSQL Upsert Pattern",
    │       "description": "Use INSERT...ON CONFLICT for migrations with unique constraints",
    │       "type": "learned",           ◄── NEW source type
    │       "mode": "selectable",
    │       "tags": ["postgresql", "migrations", "upsert", "database"],
    │       "content": "...(curated content)...",     ◄── INLINE type content
    │       "scope": "universal",
    │       "lineage": {                  ◄── Provenance tracking
    │         "source_memory_ids": ["mem_20240128_143022"],
    │         "curated_by": "advisor-knowledge-curator",
    │         "curated_at": "2024-01-30T10:00:00Z",
    │         "original_agents": ["analyst-codebase-documentation"],
    │         "original_sessions": ["session_abc123"]
    │       }
    │     }
    │
    ├─ 2. Compute embedding for the new reference (if embedding provider configured)
    │     - Reuses existing LocalEmbeddingProvider
    │     - Adds to sidecar .npy matrix
    │     - Source hash from content enables staleness detection / versioning
    │
    ├─ 3. Mark memory as escalated
    │     memory.maturity = "escalated"
    │     memory.escalated_to = reference_id
    │     storage.update(memory)
    │
    └─ 4. Return { status: "promoted", reference_id, reference_path }
```

**New reference type: `learned`**

The `SourceType` enum in `shared/plugins/references/models.py` gains a new
value:

```python
class SourceType(str, Enum):
    LOCAL = "local"
    URL = "url"
    MCP = "mcp"
    INLINE = "inline"
    LEARNED = "learned"    # NEW: graduated from memory system
```

Learned references behave like `INLINE` references (content embedded in the
JSON) but carry additional `lineage` metadata for provenance tracking.

### 3.8 Knowledge Scope and Cross-Project Flow

Memories and graduated knowledge have a `scope` field:

- **`project`**: Knowledge is specific to this project's codebase, conventions,
  or tooling. The reference is stored in `.jaato/references/` and only injected
  in this project's sessions.

- **`universal`**: Knowledge applies across projects (e.g., "PostgreSQL JSONB
  indexes don't support partial matching"). The reference is stored in
  `~/.jaato/references/` (user-level, not project-level) and injected in all
  sessions regardless of project.

The embedding system already handles cross-project discovery — embeddings are
computed per content hash, so the same knowledge appearing in different
projects will deduplicate naturally when the advisor encounters it.

### 3.9 Knowledge Staleness and Drift

Graduated knowledge can become outdated as code evolves. Three mechanisms
address this:

#### A. Source hash staleness (existing)

Embedding metadata includes a `source_hash`. When the content of a learned
reference changes (advisor rewrites it), the hash changes, triggering
re-embedding. The old embedding row is invalidated.

#### B. Artifact tracking integration (future)

If the files/patterns a learned reference describes change (detected via the
artifact tracker), the reference could be flagged for re-validation. The
advisor would re-run its validation phase on flagged references.

#### C. Usage decay (future)

If a learned reference stops being retrieved (no `selectReferences` calls,
no enrichment matches for N sessions), it may be stale. A periodic advisor
run could review low-usage learned references for relevance.

## 4. Integration with Existing Systems

### 4.1 Enrichment Pipeline

No changes to the pipeline architecture. The new flow is:

| Priority | Plugin | What changes |
|----------|--------|--------------|
| 20 | references | Now also surfaces `learned` type references (no code change needed — learned references are just reference entries) |
| 40 | template | Unchanged |
| 60 | multimodal | Unchanged |
| 80 | memory | **Filters out escalated memories** from hints |
| 90 | session | Unchanged |

The net effect: graduated knowledge moves from priority 80 (memory hints,
lightweight, model must retrieve) to priority 20 (reference hints with
semantic matching, auto-detection, richer context). This is a quality upgrade.

### 4.2 Instruction Budget / GC

Learned references are treated like any other reference by the GC system.
Their budget policy follows the injection mode:

- `mode: "auto"` → SYSTEM budget (LOCKED, never GC'd)
- `mode: "selectable"` → ENRICHMENT budget (EPHEMERAL, regenerated each turn)

Since most graduated knowledge will be `selectable`, it won't permanently
consume context — it's surfaced as hints when relevant and loaded on demand.

### 4.3 Knowledge Hierarchy

Learned references sit **outside** the formal ADR → ERI → Module → Skill
hierarchy. They are operational wisdom, not architectural decisions. However,
the advisor may recognize that a learned reference *should* become an ERI
or inform an ADR. In that case, the advisor notes this in the curation report
for human review — it doesn't autonomously modify the knowledge hierarchy.

```
ADR ──► ERI ──► Module ──► Skill     (architectural, human-authored)
                  ▲
                  │ (advisor may recommend)
                  │
Learned References                    (operational, agent-authored)
       ▲
       │ (escalation)
       │
Raw Memories                          (experiential, ephemeral)
```

### 4.4 System Instructions Enhancement

Principle 12 ("Continuous Learning Through Memory") is extended with guidance
on the new maturity fields:

```markdown
## Principle 12: Continuous Learning Through Memory (Enhanced)

... (existing content) ...

### Structured Learning Metadata

When storing a lesson, include confidence and evidence:

store_memory(
  content="...",
  description="...",
  tags=["lesson-learned", ...],
  confidence=0.8,        # How sure are you? (0.0 = guess, 1.0 = proven)
  evidence="Build failed with error X, fixed by approach Y, verified by re-running",
  scope="project"        # "project" for local conventions, "universal" for general knowledge
)

**Confidence Guidelines:**
- 0.9–1.0: Verified empirically (ran the test, saw the result)
- 0.7–0.8: Strong evidence but not independently verified
- 0.5–0.6: Reasonable inference, needs more evidence
- Below 0.5: Speculation — consider whether it's worth storing

**Scope Guidelines:**
- "project": References specific files, configs, or conventions in this codebase
- "universal": Would apply to any project using the same technology

Your memories will be reviewed by the Knowledge Curator, who will promote
valuable lessons to structural knowledge and discard inaccurate ones. Higher
confidence and stronger evidence increase the chance of promotion.
```

## 5. Implementation Phases

### Phase 1: Enhanced Memory Model (backward-compatible)

**Changes:**
- `models.py`: Add `maturity`, `confidence`, `evidence`, `scope`, `source_agent`,
  `source_session`, `escalated_to` fields to `Memory`. Add `maturity`, `scope`
  to `MemoryMetadata`.
- `storage.py`: Handle new fields in serialization/deserialization (backward-compatible).
  Add `count_by_maturity(maturity: str)` method. Add `search_by_maturity(maturity: str)`.
- `indexer.py`: Carry `maturity` in `MemoryMetadata`. Support filtering by maturity.
- `plugin.py`: Update `store_memory` schema with optional `confidence`, `evidence`,
  `scope` parameters. Auto-populate `source_agent` and `source_session` from context.
  Filter escalated memories in `enrich_prompt()`.

**Tests:** Extend existing tests for new fields, backward-compat loading.

### Phase 2: Curation Tools

**Changes:**
- `plugin.py`: Add `list_memories_for_curation`, `validate_memory`,
  `escalate_memory`, `discard_memory`, `merge_memories` tools. All `discoverable`.
  Gate on `curation_mode` plugin config.
- `references/models.py`: Add `LEARNED` to `SourceType` enum. Add `lineage`
  field to `ReferenceSource`.
- `references/plugin.py`: Handle `learned` type sources (load inline content,
  support lineage metadata display).

**Tests:** Unit tests for each curation tool. Integration test for full
escalation flow (memory → reference).

### Phase 3: Advisor Profile + System Instructions

**Changes:**
- `.jaato/profiles/advisor-knowledge-curator.json`: New profile with assessment
  methodology in system instructions.
- `.jaato/instructions/00-system-instructions.md`: Enhance Principle 12 with
  confidence/evidence/scope guidance.

**Validation:** Manual testing — spawn advisor with test memories, verify
assessment and promotion flow.

### Phase 4: Curation Triggers

**Changes:**
- Session finalization hook: After `on_session_end`, check raw memory count
  against threshold, trigger headless curation if warranted.
- Environment variables: `JAATO_CURATION_THRESHOLD`, `JAATO_CURATION_AUTO`,
  `JAATO_CURATION_LOG_DIR`.
- Curation report format: Structured output to log directory.

**Tests:** Hook trigger logic, threshold behavior, headless session spawning.

### Phase 5: Cross-Project + Staleness (future)

**Changes:**
- Universal scope: Store learned references in `~/.jaato/references/` for
  cross-project injection.
- Artifact tracking integration: Flag learned references when source files change.
- Usage decay: Periodic review of low-usage learned references.

## 6. Token Economics

**Cost of curation:** One advisor session ≈ 15-20 turns × model cost. Amortized
across all future sessions that benefit from the graduated knowledge.

**Savings from graduation:** A memory that was retrieved 10 times across 10
sessions consumed enrichment tokens each time (metadata hint + full retrieval).
As a reference, it benefits from the references plugin's more efficient
injection (semantic matching, auto-detection, richer context) and is never
"forgotten" by GC.

**Break-even estimate:** If a lesson is retrieved in ≥ 3 future sessions, the
one-time curation cost is recovered through more efficient knowledge delivery.

## 7. Risk Analysis

| Risk | Mitigation |
|------|------------|
| Advisor promotes incorrect knowledge | Empirical validation phase (run commands, read files). User-initiated mode includes human approval. |
| Advisor is too conservative (nothing gets promoted) | Tune assessment criteria. Track promotion rate. Lower confidence threshold for well-used memories. |
| Accumulated memories overwhelm advisor context | Batch processing with pagination. GC policy on advisor session. Process newest-first. |
| Cross-project knowledge contamination | Scope field + per-project vs user-level reference directories. Advisor explicitly evaluates scope. |
| Curation hook causes delays on session exit | Hook spawns headless session asynchronously. Main session exit is not blocked. |
| Backward-incompatible memory format | All new fields have defaults. Existing JSONL loads without modification. |

## 8. Open Design Questions (for future iteration)

1. **Multi-advisor consensus**: Should multiple advisor runs converge on the
   same promotions? If two advisors disagree, what wins?

2. **User override**: Can a user manually promote or demote a memory without
   going through the advisor? (Likely yes — direct `memory` commands.)

3. **Learned reference versioning UI**: How should the user see the evolution
   of a learned reference across advisor revisions?

4. **Advisor self-learning**: Should the advisor itself store memories about
   its curation decisions? (Meta-learning — the advisor learns what kind of
   memories are worth promoting.)

5. **Embedding-based deduplication**: Before the advisor runs, should the
   system pre-cluster memories by semantic similarity to reduce advisor
   workload? (Optimization for large memory stores.)
