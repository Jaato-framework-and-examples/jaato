# JAATO Knowledge Management

## Executive Summary

JAATO manages **external knowledge** -- documentation, specifications, coding standards, templates -- through two cooperating plugins: **references** and **template**. The references plugin maintains a catalog of knowledge sources and handles their selection, injection, and transitive discovery. The template plugin detects reusable code patterns within that injected content and makes them available for structured generation. Together they form a knowledge pipeline where references flow in, templates are extracted, and the model can both learn from documentation and generate consistent code from it -- all without the user manually pointing to files.

---

## Part 1: The Knowledge Catalog

JAATO's knowledge management starts with a **catalog** of reference sources. Each source is a pointer to documentation the model might need -- not the content itself, but metadata describing what it is, where it lives, and how to fetch it.

### Source Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REFERENCE SOURCE TYPES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │  LOCAL    │   │   URL    │   │   MCP    │   │  INLINE  │         │
│  │          │   │          │   │          │   │          │         │
│  │ Files &  │   │ HTTP(S)  │   │ MCP tool │   │ Content  │         │
│  │ folders  │   │ endpoints│   │ calls    │   │ embedded │         │
│  │ on disk  │   │          │   │          │   │ in config│         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│                                                                      │
│  Key principle: The plugin stores METADATA only.                     │
│  The model fetches content using existing tools:                     │
│  • LOCAL → readFile / CLI                                            │
│  • URL → fetch_url                                                   │
│  • MCP → call the specified MCP server tool                          │
│  • INLINE → content provided directly (no fetch needed)              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Injection Modes

Each source has a **mode** that determines when it enters the model's context:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **AUTO** | Included in system instructions at startup; model fetches immediately | Always-needed docs (README, coding standards, project rules) |
| **SELECTABLE** | Available on-demand; user or model triggers selection | Specialized docs (API specs, module guides, design patterns) |

### Catalog Configuration

Sources are configured in `references.json` or auto-discovered from `.jaato/references/`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CATALOG CONFIGURATION SOURCES                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Config File (references.json)                                    │
│     ├─ REFERENCES_CONFIG_PATH env var                                │
│     ├─ ./references.json                                             │
│     ├─ ./.references.json                                            │
│     └─ ~/.config/jaato/references.json                               │
│                                                                      │
│  2. Auto-Discovery (.jaato/references/)                              │
│     └─ Each .json file defines one source                            │
│        ├─ id, name, description, type, path/url/content              │
│        ├─ mode (auto/selectable)                                     │
│        └─ tags (for topic-based discovery)                           │
│                                                                      │
│  Merge Rule: Explicit config sources take priority.                  │
│  Auto-discovered sources are added only if their ID is new.          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tags

Every source can carry **tags** -- topic keywords that enable two features:

1. **Filtered selection**: The model calls `selectReferences(filter_tags=["auth"])` to show only relevant sources
2. **Prompt enrichment hints**: Tags are matched against user prompts to surface unselected-but-relevant references (see Part 5)

---

## Part 2: Selection Channels

When the model or user requests reference selection, the interaction flows through a **channel** -- a pluggable communication protocol for presenting options and collecting choices.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELECTION CHANNELS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CONSOLE CHANNEL (default)                                    │    │
│  │  ├─ Interactive terminal prompts                              │    │
│  │  ├─ Numbered list with comma-separated selection              │    │
│  │  └─ Supports: 'all', 'none', or index list (e.g., '1,3,4')  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  QUEUE CHANNEL (TUI integration)                              │    │
│  │  ├─ Output via callback (renders in TUI output panel)         │    │
│  │  ├─ Input via shared queue from main input handler            │    │
│  │  └─ No direct stdin -- works with full-screen terminals       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  WEBHOOK CHANNEL                                              │    │
│  │  ├─ HTTP POST to external endpoint                            │    │
│  │  ├─ Response: {"selected_ids": ["api-spec", "auth-guide"]}   │    │
│  │  └─ For: Slack bots, dashboards, approval workflows           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FILE CHANNEL                                                 │    │
│  │  ├─ Writes request to {base}/requests/{id}.json               │    │
│  │  ├─ Polls for response at {base}/responses/{id}.json          │    │
│  │  └─ For: Background services, automation, CI pipelines        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Multiple Channels?

| Concern | Console only | Multi-channel |
|---------|-------------|---------------|
| **TUI clients** | Blocks stdin (conflicts with full-screen UI) | Queue channel works with any UI |
| **Automation** | Requires human at terminal | File/Webhook channels are scriptable |
| **Remote approval** | Not possible | Webhook routes to Slack, dashboards, etc. |
| **Testing** | Requires mocking stdin | File channel is deterministic |

---

## Part 3: Reference Tools

The references plugin exposes three interaction surfaces: two model tools and one user command.

### Model Tools

| Tool | Purpose | Auto-Approved |
|------|---------|---------------|
| `selectReferences` | Model selects references by ID or tags; receives resolved paths | Yes |
| `listReferences` | Model queries the catalog (filter by tags or mode) | Yes |

### User Command

| Command | Purpose |
|---------|---------|
| `references` | User lists, selects, or unselects references directly |

### Selection Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REFERENCE SELECTION FLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model calls selectReferences(ids=["api-spec"])                      │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. RESOLVE SOURCE                                            │    │
│  │     Look up source in catalog by ID                           │    │
│  │     Resolve path against project root                         │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. SANDBOX AUTHORIZATION                                     │    │
│  │     LOCAL source? → Authorize path as readonly via sandbox    │    │
│  │     ├─ Try sandbox_manager plugin API first                   │    │
│  │     └─ Fallback: direct registry authorization                │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. TRANSITIVE RESOLUTION (if enabled)                        │    │
│  │     Scan selected source content for mentions of other        │    │
│  │     catalog IDs → automatically select them too               │    │
│  │     (see Part 4 for details)                                  │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  4. RETURN TO MODEL                                           │    │
│  │     {                                                         │    │
│  │       "status": "success",                                    │    │
│  │       "selected_count": 3,                                    │    │
│  │       "transitive_count": 2,                                  │    │
│  │       "sources": [                                            │    │
│  │         {"id": "api-spec", "resolved_path": "/docs/api.md"},  │    │
│  │         {"id": "auth-guide", "transitive": true,              │    │
│  │          "transitive_from": ["api-spec"]},                    │    │
│  │         ...                                                   │    │
│  │       ]                                                       │    │
│  │     }                                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Model then uses readFile/CLI to fetch content from resolved paths   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Sandbox Integration

When a LOCAL reference is selected, its path is authorized for readonly access:

```
┌──────────────────────────────────────────────────────────────────┐
│  SELECT reference "api-spec" (path: /docs/api-spec.md)            │
│       │                                                           │
│       ▼                                                           │
│  sandbox_manager.add_path_programmatic("/docs/api-spec.md",       │
│                                        access="readonly")         │
│       │                                                           │
│  UNSELECT reference "api-spec"                                    │
│       │                                                           │
│       ▼                                                           │
│  sandbox_manager.remove_path_programmatic("/docs/api-spec.md")    │
└──────────────────────────────────────────────────────────────────┘
```

This means references outside the workspace are automatically made accessible to the model's file-reading tools, and automatically cleaned up when unselected.

---

## Part 4: Transitive Reference Discovery

The most opinionated feature of the knowledge system is **transitive injection**: when a selected document mentions other catalog references, those references are automatically selected too.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSITIVE REFERENCE RESOLUTION                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User pre-selects: "circuit-breaker-pattern"                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  circuit-breaker-pattern.md contains:                         │    │
│  │  "See retry-pattern for retry configuration."                 │    │
│  │  "Refer to [timeout](./timeout-pattern.md) for timeouts."     │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│           ┌────────────────┼────────────────┐                        │
│           ▼                                 ▼                        │
│  ┌─────────────────┐              ┌──────────────────┐              │
│  │  retry-pattern   │              │  timeout-pattern  │              │
│  │  (ID match)      │              │  (path match)     │              │
│  │                  │              │                   │              │
│  │  Content:        │              │  Content:         │              │
│  │  "See error-     │              │  "No further      │              │
│  │   handling for   │              │   references."    │              │
│  │   error codes."  │              │                   │              │
│  └────────┬─────────┘              └───────────────────┘              │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  error-handling  │                                                │
│  │  (depth 2)       │                                                │
│  └──────────────────┘                                                │
│                                                                      │
│  Result: 4 sources selected (1 explicit + 3 transitive)              │
│  Parent map:                                                         │
│    retry-pattern    ← circuit-breaker-pattern                        │
│    timeout-pattern  ← circuit-breaker-pattern                        │
│    error-handling   ← retry-pattern                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Two Detection Strategies

Transitive references are discovered through two complementary mechanisms:

**1. ID-based matching** (`_find_referenced_ids`):

Scans content for catalog IDs appearing as whole words. Handles common syntaxes:
- Direct mentions: `retry-pattern`
- Reference syntax: `@ref:retry-pattern`, `[[retry-pattern]]`
- Backtick-wrapped: `` `retry-pattern` ``

**2. Path-based matching** (`_find_referenced_paths`):

Resolves relative file paths in markdown links against the source document's directory:
- `[retry](./retry-pattern.md)` → resolves relative to source's directory
- `[guide](../guides/retry.md)` → resolves `../` paths
- Strips anchor fragments: `retry.md#config` → matches `retry.md`

### Safety Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Max depth** (`MAX_TRANSITIVE_DEPTH = 10`) | Prevents runaway chains |
| **Cycle detection** | Visited set prevents infinite loops |
| **Whole-word boundaries** | `skill-001` does not match `skill-001-extended` |
| **Path-only for LOCAL** | URL and MCP sources cannot be scanned (content requires external fetch) |

### Parent Map

The resolver tracks **which parent caused each transitive discovery**. This metadata flows into:
- System instructions (transitive sources are annotated with `Transitively included via @parent-id`)
- Enrichment notifications (one-time hint on first prompt)
- Tool responses (`transitive_from` field in selectReferences results)

---

## Part 5: Enrichment Pipeline

The references and template plugins participate in JAATO's enrichment pipeline -- a chain of plugins that can modify prompts and system instructions before they reach the model.

### References Enrichment (Priority 20)

The references plugin enriches content through two passes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REFERENCES ENRICHMENT PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User prompt: "Implement the circuit breaker for the payment service" │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PASS 1: Transitive Notification (one-time)                   │    │
│  │                                                               │    │
│  │  If transitive references were discovered during init or      │    │
│  │  recent selection, append a hint to the prompt:               │    │
│  │                                                               │    │
│  │  "Transitively selected references:                           │    │
│  │   • @appendix-a (from @main-doc)                              │    │
│  │   • @error-handling (from @retry-pattern)"                    │    │
│  │                                                               │    │
│  │  Fires ONLY on the first user prompt (not tool results).      │    │
│  │  Never fires again until a new transitive selection occurs.   │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PASS 2: Tag-Based Hints                                      │    │
│  │                                                               │    │
│  │  Scan prompt for tags from UNSELECTED sources:                │    │
│  │  • "circuit" matches source "Circuit Breaker Guide"           │    │
│  │  • Tag matching uses word boundaries (not dotted names,       │    │
│  │    file extensions, or path segments)                         │    │
│  │                                                               │    │
│  │  Appends hint:                                                │    │
│  │  "Available references (matched: circuit):                    │    │
│  │   • Circuit Breaker Guide [circuit-breaker-ref]"              │    │
│  │                                                               │    │
│  │  Already-selected sources are excluded from hints.            │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  Enriched prompt with hint annotations                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tag Matching Rules

Tag matching is deliberately strict to avoid false positives:

| Pattern | Matches? | Why |
|---------|----------|-----|
| `java` in "We use java here" | Yes | Standalone word |
| `java` in "JAVA is popular" | Yes | Case-insensitive |
| `java` in "languages (java, python)" | Yes | Punctuation boundary |
| `java` in "java.util.concurrent" | No | Dot boundary (package name) |
| `java` in "CircuitBreaker.java" | No | File extension |
| `java` in "/usr/lib/java/bin" | No | Path segment |
| `circuit breaker` (multi-word) | Yes | Multi-word tags match as phrases |
| `spring.boot` (dotted tag) | Yes, standalone only | Dotted tag matches when standalone |

### Template Enrichment (Priority 40)

The template plugin runs after references, processing the content that references injected:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEMPLATE ENRICHMENT PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: System instructions with injected MODULE.md content          │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. EMBEDDED TEMPLATE EXTRACTION                              │    │
│  │                                                               │    │
│  │  Scan code blocks for template syntax:                        │    │
│  │  • Jinja2: {{ variable }}, {% if %}, {% for %}                │    │
│  │  • Mustache: {{#section}}...{{/section}}, {{^inverted}}       │    │
│  │                                                               │    │
│  │  Extract to .jaato/templates/ with auto-generated names:      │    │
│  │  • Frontmatter ID → prefix (e.g., mod-code-001)              │    │
│  │  • Heading text → template name                               │    │
│  │  • Content hash → fallback                                    │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. STANDALONE TEMPLATE DISCOVERY                             │    │
│  │                                                               │    │
│  │  Query references plugin for selected LOCAL directory sources │    │
│  │  Scan those directories for .tpl / .tmpl files               │    │
│  │  Index them WITHOUT copying (stay in original location)       │    │
│  │                                                               │    │
│  │  Auto-detect syntax (Jinja2 vs Mustache)                      │    │
│  │  Extract variable names for the index                         │    │
│  │  Handle name collisions via parent-folder disambiguation      │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. ANNOTATE & PERSIST                                        │    │
│  │                                                               │    │
│  │  Append annotations to system instructions:                   │    │
│  │  "TEMPLATE AVAILABLE: Entity.java.tpl                         │    │
│  │   Variables: Entity, basePackage, entityFields                │    │
│  │   Use: renderTemplate(template_name="Entity.java.tpl", ...)" │    │
│  │                                                               │    │
│  │  Persist index to .jaato/templates/index.json                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Enrichment Chain

The two plugins form a deliberate pipeline where ordering matters:

```
User Prompt
     │
     ▼
┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐
│ refs    │ → │ template │ → │ multimodal │ → │ memory  │ → ...
│ (20)    │   │ (40)     │   │ (60)       │   │ (80)    │
│         │   │          │   │            │   │         │
│ Injects │   │ Extracts │   │            │   │         │
│ MODULE  │   │ templates│   │            │   │         │
│ content │   │ from it  │   │            │   │         │
└─────────┘   └──────────┘   └────────────┘   └─────────┘
     │
     ▼
Enriched Prompt
```

References (priority 20) injects documentation content. Template (priority 40) then scans that injected content for embedded templates. This ordering is essential -- reversing it would mean templates are never discovered.

---

## Part 6: Notification System

When the system makes implicit decisions (transitive selections, tag matches), it communicates them through a layered notification system.

### Notification Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NOTIFICATION TYPES                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SYSTEM INSTRUCTION ANNOTATIONS (persistent)                      │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  Transitively included sources are annotated inline:      │    │
│     │                                                           │    │
│     │  ### Appendix A                                           │    │
│     │  *Supplementary material*                                 │    │
│     │  *(Transitively included via @main-doc)*                  │    │
│     │  **Location**: docs/appendix-a.md                         │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  2. PROMPT ENRICHMENT HINTS (one-time)                               │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  Appended to the first user prompt after transitive       │    │
│     │  selection occurs:                                        │    │
│     │                                                           │    │
│     │  "Transitively selected references:                       │    │
│     │   • @retry-ref (from @circuit-breaker-ref)"               │    │
│     │                                                           │    │
│     │  Fires once, then resets. Skips tool results.             │    │
│     │  Re-arms when a new runtime selection triggers transitive.│    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  3. TAG MATCH HINTS (per-prompt)                                     │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  Appended when unselected sources match prompt tags:      │    │
│     │                                                           │    │
│     │  "Available references (matched: circuit):                │    │
│     │   • Circuit Breaker Guide [circuit-breaker-ref]"          │    │
│     │                                                           │    │
│     │  Fires on every prompt. Excludes already-selected.        │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  4. REGISTRY FALLBACK NOTIFICATIONS (TUI)                            │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  When enrichment metadata includes transitive info,       │    │
│     │  the registry generates a user-visible notification:      │    │
│     │                                                           │    │
│     │  "References transitively included:                       │    │
│     │   @retry-ref (from @circuit-breaker-ref)                  │    │
│     │   @timeout-ref (from @circuit-breaker-ref)"               │    │
│     │                                                           │    │
│     │  Truncates at 3 items with "+N more" for large sets.      │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  5. TEMPLATE EXTRACTION NOTIFICATIONS                                │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  System instruction annotations for discovered templates: │    │
│     │                                                           │    │
│     │  "TEMPLATE AVAILABLE: Entity.java.tpl                     │    │
│     │   Syntax: mustache                                        │    │
│     │   Variables: Entity, basePackage, entityFields             │    │
│     │   Use: renderTemplate(template_name=...)"                 │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why One-Time Hints?

The transitive notification fires only once per selection event because:
- It communicates a **state change**, not a persistent condition
- Repeating it on every prompt would waste context tokens
- The system instructions already contain the persistent annotations
- A new runtime selection re-arms the notification

---

## Part 7: The Template Index

All templates -- whether extracted from documentation or discovered as standalone files -- are registered in a **unified index** that decouples the model from filesystem details.

### Index Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED TEMPLATE INDEX                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model refers to templates BY NAME only:                             │
│  renderTemplate(template_name="Entity.java.tpl", ...)               │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Template Index (in-memory + persisted to index.json)         │    │
│  │                                                               │    │
│  │  "Entity.java.tpl" → {                                       │    │
│  │      source_path: "/knowledge/mod-015/templates/domain/       │    │
│  │                     Entity.java.tpl",                         │    │
│  │      syntax: "mustache",                                      │    │
│  │      variables: ["Entity", "basePackage", "entityFields"],    │    │
│  │      origin: "standalone"                                     │    │
│  │  }                                                            │    │
│  │                                                               │    │
│  │  "mod-code-001-basic.java.tmpl" → {                           │    │
│  │      source_path: ".jaato/templates/mod-code-001-basic.       │    │
│  │                    java.tmpl",                                │    │
│  │      syntax: "jinja2",                                        │    │
│  │      variables: ["circuitBreakerName", "fallbackMethod"],     │    │
│  │      origin: "embedded"                                       │    │
│  │  }                                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Two origins:                                                        │
│                                                                      │
│  STANDALONE                         EMBEDDED                         │
│  • .tpl/.tmpl files in referenced   • Code blocks with template      │
│    directories                        syntax in documentation        │
│  • Left in original location        • Extracted to .jaato/templates/ │
│  • Indexed, not copied              • Named from frontmatter + heading│
│  • Name collisions disambiguated    • Content-hashed for dedup       │
│    by parent folder                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Template Syntax Detection

The plugin auto-detects which template engine to use:

| Pattern | Detection | Engine |
|---------|-----------|--------|
| `{{#section}}...{{/section}}` | Mustache section/loop | Mustache |
| `{{^inverted}}` | Mustache inverted section | Mustache |
| `{{.}}` | Mustache current item | Mustache |
| `{% if %}`, `{% for %}` | Jinja2 control | Jinja2 |
| `{{ var \| filter }}` | Jinja2 pipe filter | Jinja2 |
| `{{ variable }}` only | Ambiguous -- defaults to Jinja2 | Jinja2 |

### Cross-Plugin Discovery

The template plugin queries the references plugin to find directories containing standalone templates:

```
┌──────────────────────────────────────────────────────────────────┐
│  Template Plugin                  References Plugin               │
│       │                                │                          │
│       │  _get_reference_directories()  │                          │
│       │ ──────────────────────────────>│                          │
│       │                                │ get_selected_ids()       │
│       │                                │ get_sources()            │
│       │                                │ Filter: LOCAL + selected │
│       │<────────────────────────────── │                          │
│       │  [Path("/knowledge/mod-015")]  │                          │
│       │                                │                          │
│       ▼                                                           │
│  _discover_standalone_templates(Path)                             │
│  → Scan for .tpl/.tmpl files                                     │
│  → Auto-detect syntax                                            │
│  → Extract variables                                             │
│  → Add to index                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Only LOCAL sources that are currently selected contribute template directories. URL and MCP sources are skipped (their content is remote).

---

## Part 8: Template Tools

The template plugin exposes four tools for code generation:

| Tool | Purpose | Discoverability |
|------|---------|----------------|
| `renderTemplate` | Render template with full Jinja2 or Mustache support | Discoverable |
| `renderTemplateToFile` | Render and write to file with overwrite control | Discoverable |
| `listExtractedTemplates` | List all templates in the unified index | Discoverable |
| `listTemplateVariables` | List variables required by a specific template | Discoverable |

### Rendering Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEMPLATE RENDERING FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  renderTemplateToFile(                                               │
│    template_name="Repository.java.tpl",                              │
│    variables={"Entity": "Customer", "basePackage": "com.bank"},      │
│    output_path="src/.../CustomerRepository.java"                     │
│  )                                                                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. RESOLVE TEMPLATE                                          │    │
│  │     Index lookup: "Repository.java.tpl"                       │    │
│  │     → source_path: /knowledge/mod-015/templates/domain/       │    │
│  │                     Repository.java.tpl                       │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. DETECT SYNTAX                                             │    │
│  │     {{#items}} found → Mustache engine                        │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. RENDER                                                    │    │
│  │     Apply variables to template                               │    │
│  │     Jinja2: SandboxedEnvironment + StrictUndefined            │    │
│  │     Mustache: chevron.render()                                │    │
│  └────────────────────────┬──────────────────────────────────┘    │
│                            │                                          │
│                            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  4. WRITE OUTPUT                                              │    │
│  │     Create parent directories                                 │    │
│  │     Write rendered content to output_path                     │    │
│  │     Return: {success, output_path, bytes_written, ...}        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Security

| Engine | Protection |
|--------|-----------|
| **Jinja2** | `SandboxedEnvironment` -- no arbitrary code execution; `include`/`import` disabled; `StrictUndefined` catches typos |
| **Mustache** | Logic-less by design -- no code execution possible; only variable substitution and sections |
| **Both** | File writes require permission approval; parent directories created safely |

---

## Part 9: Pre-selected References and Subagent Profiles

References can be **pre-selected** in subagent profiles, enabling knowledge-specialized agents that start with relevant documentation already loaded.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBAGENT KNOWLEDGE PROFILES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Profile: "circuit-breaker-specialist"                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  {                                                            │    │
│  │    "plugins": ["cli", "file_edit", "references", "template"], │    │
│  │    "plugin_configs": {                                        │    │
│  │      "references": {                                          │    │
│  │        "preselected": ["circuit-breaker-pattern"],             │    │
│  │        "exclude_tools": ["selectReferences"]                  │    │
│  │      }                                                        │    │
│  │    }                                                          │    │
│  │  }                                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  At session startup:                                                 │
│  1. "circuit-breaker-pattern" looked up from master catalog          │
│  2. Content scanned → transitive refs discovered:                    │
│     • retry-pattern (mentioned in content)                           │
│     • timeout-pattern (linked via relative path)                     │
│     • error-handling (transitively from retry-pattern)               │
│  3. All 4 sources included in system instructions                    │
│  4. Template plugin discovers .tpl files in those directories        │
│  5. selectReferences tool excluded (all knowledge pre-loaded)        │
│                                                                      │
│  Result: Agent starts with full knowledge graph loaded,              │
│  templates indexed, and no interactive selection needed.              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `preselected` | `List[str]` | `[]` | Source IDs to select at startup |
| `transitive_injection` | `bool` | `true` | Auto-discover referenced sources |
| `exclude_tools` | `List[str]` | `[]` | Tools to hide (e.g., `["selectReferences"]`) |
| `sources` | `List` | `[]` | Source IDs (strings) or full source objects |

---

## Part 10: Complete Knowledge Flow

End-to-end flow from configuration to code generation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    END-TO-END KNOWLEDGE FLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  CONFIGURE       │  references.json + .jaato/references/*.json   │
│  │  (catalog)       │  Define sources, tags, modes, paths            │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  INITIALIZE      │  Auto-discover + merge sources                │
│  │  (startup)       │  Pre-select configured sources                │
│  │                  │  Resolve transitive references                 │
│  │                  │  Authorize sandbox paths                       │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  INJECT          │  AUTO sources → system instructions           │
│  │  (instructions)  │  Transitive sources → annotated instructions  │
│  │                  │  Template plugin → extract embedded templates  │
│  │                  │  Template plugin → discover standalone .tpl    │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  ENRICH          │  Per-prompt: tag hints for unselected sources  │
│  │  (runtime)       │  One-time: transitive selection notifications  │
│  │                  │  Per-tool-result: template extraction from CLI │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  SELECT          │  Model calls selectReferences (by ID or tags) │
│  │  (on-demand)     │  User calls 'references select <id>'          │
│  │                  │  → Transitive resolution on new selections     │
│  │                  │  → Sandbox path authorization                  │
│  │                  │  → Template re-discovery for new directories   │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  GENERATE        │  Model uses renderTemplate with indexed        │
│  │  (code)          │  templates and resolved variables               │
│  │                  │  → Consistent, pattern-compliant output        │
│  └─────────────────┘                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Key Files

| Component | Path | Purpose |
|-----------|------|---------|
| References plugin | `shared/plugins/references/plugin.py` | Core plugin with selection, transitive resolution, enrichment |
| Reference models | `shared/plugins/references/models.py` | `ReferenceSource`, `SourceType`, `InjectionMode` |
| Selection channels | `shared/plugins/references/channels.py` | Console, Webhook, File, Queue channel implementations |
| Config loader | `shared/plugins/references/config_loader.py` | Config file parsing, validation, auto-discovery |
| Template plugin | `shared/plugins/template/plugin.py` | Template extraction, index, rendering, standalone discovery |
| Reference tests | `shared/plugins/references/tests/` | Registry integration, transitive resolution, enrichment |
| Template tests | `shared/plugins/template/tests/` | Template index, discovery, rendering, cross-plugin integration |

---

## Part 12: Color Coding Suggestion for Infographic

- **Blue:** Reference sources and catalog (the knowledge being managed)
- **Green:** Selection flow and channels (how knowledge is chosen)
- **Yellow:** Transitive discovery (automatic knowledge graph expansion)
- **Orange:** Enrichment pipeline (notifications and hints flowing to the model)
- **Purple:** Template index and rendering (knowledge materialized as code)
- **Gray:** Infrastructure (sandbox authorization, config loading, persistence)
