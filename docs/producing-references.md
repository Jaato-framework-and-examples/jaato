# Producing References — Complete Reference

> **Scope**: How to produce a comprehensive, self-contained reference document and its companion reference JSON catalog entry in JAATO's knowledge management system, including embedding, validation, and the full lifecycle from research to publication.

---

## Table of Contents

1. [What Is a Reference?](#1-what-is-a-reference)
2. [The Two Artifacts](#2-the-two-artifacts)
3. [Research Phase — Read Before You Write](#3-research-phase--read-before-you-write)
4. [Document Structure](#4-document-structure)
5. [Writing Rules](#5-writing-rules)
6. [Reference JSON Catalog Entry](#6-reference-json-catalog-entry)
7. [Tags — Semantic Matching](#7-tags--semantic-matching)
8. [Embedding and the Sidecar Matrix](#8-embedding-and-the-sidecar-matrix)
9. [Validation](#9-validation)
10. [The Enrichment Pipeline — How References Reach the Model](#10-the-enrichment-pipeline--how-references-reach-the-model)
11. [Transitive Discovery](#11-transitive-discovery)
12. [Runtime Internals — What the Source Code Reveals](#12-runtime-internals--what-the-source-code-reveals)
13. [Source Code Map](#13-source-code-map)

---

## 1. What Is a Reference?

A **reference** is a pointer from JAATO's knowledge catalog to a document (or collection of documents) that an agent may need. The reference itself is lightweight metadata — an ID, a name, a description, tags, and instructions on how to fetch the actual content. The heavy lifting (reading files, fetching URLs, calling MCP tools) is done by the model using its standard tools at query time.

References serve two audiences:

| Audience | What they need |
|---|---|
| **The model** (at runtime) | Enough metadata to decide whether to select a reference and how to fetch its content |
| **Future agents** (at indexing time) | A self-contained document that fully explains a feature without requiring access to source code |

A well-produced reference satisfies both: the JSON catalog entry gives the runtime what it needs, and the markdown document gives a future agent everything it needs to understand, configure, and reason about the feature.

### Types of References

References can point to different kinds of content:

| Type | Description | Model access method |
|---|---|---|
| `local` | Files or directories on disk | `readFile` or CLI tools |
| `url` | HTTP(S) endpoints | `fetch_url` |
| `mcp` | MCP tool calls | Call the specified MCP server + tool |
| `inline` | Content embedded directly in the JSON | No fetch needed — content is in the catalog |

### Injection Modes

| Mode | Behavior | Use case |
|---|---|---|
| `auto` | Injected into system instructions at startup | Always-needed docs (coding standards, project rules) |
| `selectable` | Available on-demand; model or user explicitly selects | Specialized docs (API specs, design patterns) |

---

## 2. The Two Artifacts

Producing a reference means creating **two files**:

### 2.1 The Reference Document (Markdown)

A comprehensive, self-contained markdown file that a future agent — with no prior knowledge of the codebase — can read and use to fully understand the feature. This is the **primary artifact**. The reference JSON is just a pointer to it.

**Location**: `docs/<feature-kebab>.md` (workspace-relative)

### 2.2 The Reference JSON (Catalog Entry)

A JSON file in `.jaato/references/` that registers the document in JAATO's knowledge catalog. Contains metadata (ID, name, description, type, path, tags, embedding info).

**Location**: `.jaato/references/<feature-kebab>.json`

### The Relationship

```
.jaato/references/my-feature.json    →    docs/my-feature.md
  (catalog entry — metadata only)          (the actual knowledge)
  { "id": "my-feature",                    # My Feature — Complete Reference
    "path": "docs/my-feature.md",          ## 1. What Is My Feature?
    "tags": ["feature", "config"], ... }   ## 2. Configuration Schema
                                           ## 3. Runtime Internals
                                           ## ...
```

The JSON tells the system *where* to find the document and *what* it's about. The document itself contains *everything* the agent needs to know.

---

## 3. Research Phase — Read Before You Write

The most important step. A reference that only summarizes surface-level documentation is nearly useless — the value comes from capturing what the docs don't say.

### What to Read (in priority order)

1. **Entry-point files** — the files the user identified as most relevant. Read every line.

2. **Data models and config files** — dataclasses, Pydantic models, JSON schemas, type definitions. These define the *shape* of the feature. Search for `class |dataclass|BaseModel|TypedDict`.

3. **Validation functions** — `validate_|_validate_|is_valid|check_`. These reveal exact constraints and edge cases that informal docs omit.

4. **Test files** — Tests are the most precise specification of expected behavior. They show valid inputs, error outputs, and edge cases.

5. **Plugin registration and tool schemas** — How the feature exposes itself to the rest of the system. Look for `register_tool|tool_name|get_tool_schemas|execute_`.

6. **Cross-references** — When source code references other files (imports, function calls), follow those references. A complete reference traces the full call chain.

### Reading Strategy

- **Read whole files, not snippets.** Partial reads miss context.
- **Never fabricate.** If you didn't read it, don't write about it. If an implementation isn't visible, say "implementation details not inspected."
- **Track what you've read.** Keep a list of files and line ranges inspected. Confidence in each section should correlate with source coverage.
- **Note gaps.** If you couldn't find something (e.g., it's in a premium package), explicitly say so.

---

## 4. Document Structure

Organize so a reader can find what they need quickly. Use this template:

```
# <Feature Name> — Complete Reference

> Scope: one-sentence summary

## Table of Contents

## 1. What Is <Feature>?
   (Definition, purpose, where it fits in the system)

## 2–N. <Topic Sections>
   (One section per major aspect. Each self-contained.)

## N+1. Configuration / Schema / API Reference
   (Complete field-by-field reference with types, defaults, constraints.
    Include validation rules from actual validation functions.)

## N+2. Runtime Internals — What the Source Code Reveals
   (Implementation details only discoverable by reading source code.)

## N+3. Source Code Map
   (Table mapping each source file to what it contains.)
```

### Section Writing Rules — The Stranger Test

Every section must pass the "stranger test": a developer who has never seen this codebase reads only that section and can use the topic correctly.

To pass:
- **Define all terms before using them.** Don't assume prior knowledge.
- **Include concrete examples** (JSON, code snippets) for every abstract concept.
- **Show the shape of data** (field names, types, constraints), not just narrative.
- **Explain why, not just what.** Design decisions are as important as mechanics.
- **Include validation rules.** What happens with wrong input?

### Style Rules

- Use **tables** for field references and comparisons
- Use **code blocks** with language tags for all examples
- Use **bold** for field names in prose
- Use **blockquotes** for design rationale
- **Number sections** and provide a table of contents
- Do NOT use ANSI escape codes

---

## 5. Writing Rules

### The Self-Containedness Principle

The reference document is the **sole artifact** a future agent will read. Therefore:

- **The document must contain everything.** Do not rely on the agent reading other files.
- **Do not point to external files for essential information.** The `fetchHint` can suggest supplementary reading, but all required knowledge must be in the document.
- **Capture implementation details.** If something is only discoverable by reading source code, include it in a "Runtime Internals" section.
- **Be precise.** "The system validates X" is weak. "The `validate_reference_file()` function in `config_loader.py` rejects X if Y is missing, returning `('id' is required)`" is strong.

### Anti-Patterns to Avoid

| Anti-pattern | Why it's bad | What to do instead |
|---|---|---|
| Pointer references: "See `config.py` for the full schema" | The agent reading the reference can't necessarily read `config.py` | Include the schema in the document |
| Vague descriptions: "The system handles errors gracefully" | Doesn't say what errors or how | Specify the errors, handling behavior, and user-visible output |
| Missing edge cases: only documenting the happy path | Validation rules and error paths are often more useful | Document what happens with invalid input |
| Stale references: hardcoding line numbers | Line numbers change; file paths are more stable | Reference file paths and function names |
| Truncated coverage: documenting 80% and leaving the rest | The agent doesn't have a "reader" | Cover everything or explicitly note gaps |
| Copying existing docs without expansion | Surface docs miss implementation details | Incorporate existing docs AND add what source code reveals |

---

## 6. Reference JSON Catalog Entry

The JSON file registers the document in JAATO's knowledge catalog. Here's the full schema with all fields:

### Minimal Valid Entry

```json
{
  "id": "my-feature",
  "name": "My Feature",
  "description": "Brief description of what this reference covers"
}
```

### Full Entry (All Fields)

```json
{
  "id": "my-feature",
  "name": "My Feature — Complete Reference",
  "description": "Comprehensive description covering all major aspects of the feature, suitable for semantic matching",
  "type": "local",
  "path": "docs/my-feature.md",
  "mode": "selectable",
  "tags": ["feature", "config", "runtime", "validation"],
  "fetchHint": "Read the full document at the path above. Start with the Table of Contents.",
  "contents": {
    "templates": null,
    "validation": null,
    "policies": null,
    "scripts": null
  },
  "embedding": {
    "index": 0,
    "source_hash": "sha256:abcdef1234567890..."
  },
  "source": {
    "type": "git",
    "url": "https://github.com/org/repo",
    "ref": "main",
    "fetched_at": "2026-04-20T18:00:00Z"
  }
}
```

### Field Reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | `string` | **Yes** | — | Unique identifier. Used for selection and transitive references. Must be unique across all references. |
| `name` | `string` | **Yes** | — | Human-readable title. Shown in `listReferences` output. |
| `description` | `string` | No | `""` | Human-readable description. Used for display and semantic matching. |
| `type` | `string` | No | `"local"` | Source type: `"local"`, `"url"`, `"mcp"`, `"inline"`. |
| `mode` | `string` | No | `"selectable"` | Injection mode: `"auto"` or `"selectable"`. |
| `path` | `string` | No | `null` | For `local` type: path to the content file or directory. Can be absolute or relative. |
| `resolved_path` | `string` | No | `null` | Populated at load time: absolute or project-relative resolved path. Not set manually. |
| `url` | `string` | No | `null` | For `url` type: the HTTP(S) URL to fetch. |
| `server` | `string` | No | `null` | For `mcp` type: the MCP server name. |
| `tool` | `string` | No | `null` | For `mcp` type: the MCP tool name to call. |
| `args` | `object` | No | `null` | For `mcp` type: arguments to pass to the tool. |
| `content` | `string` | No | `null` | For `inline` type: the content itself, embedded in the JSON. |
| `fetchHint` | `string` | No | `null` | Hint for the model on how to access the content. Shown after location info. |
| `tags` | `string[]` | No | `[]` | Topic keywords for filtered selection and semantic matching. |
| `contents` | `object` | No | `null` | Typed subfolder declarations. See [Contents Subfolder](#contents-subfolder-declarations) below. |
| `embedding` | `object` | No | `null` | Embedding metadata linking to the sidecar `.npy` matrix. See [Section 8](#8-embedding-and-the-sidecar-matrix). |
| `source` | `object` | No | `null` | Provenance metadata for remote sources. |

### Contents Subfolder Declarations

The `contents` field declares which typed subfolders exist within a reference directory. Only meaningful for `local` directory references.

```json
{
  "contents": {
    "templates": "templates/",
    "validation": "validation/",
    "policies": "policies/",
    "scripts": "scripts/"
  }
}
```

| Key | Meaning |
|---|---|
| `templates` | Subfolder with authoritative `.tpl`/`.tmpl` template files |
| `validation` | Subfolder with mandatory post-implementation validation scripts |
| `policies` | Subfolder with markdown constraint documents |
| `scripts` | Subfolder with deterministic helper scripts |

Valid keys are: `"templates"`, `"validation"`, `"policies"`, `"scripts"`. All are optional; `null` means the subfolder is not present.

### Validation Rules

The `validate_reference_file()` function in `config_loader.py` enforces these rules:

| Rule | Details |
|---|---|
| `id` required | Must be non-empty string |
| `name` required | Must be non-empty string |
| `type` must be valid | One of: `"local"`, `"url"`, `"mcp"`, `"inline"` |
| `mode` must be valid | One of: `"auto"`, `"selectable"` |
| `type`-specific fields | `local` requires `path`; `url` requires `url`; `mcp` requires `server` + `tool`; `inline` requires `content` |
| `tags` must be array of strings | Must be a list, all elements must be strings |
| `contents` must be object | All values must be `string` or `null`; unknown keys produce warnings |
| `embedding.index` must be non-negative integer | Must be `int`, not float or string |
| `embedding.source_hash` must be string | Warning if not prefixed with `"sha256:"` |

---

## 7. Tags — Semantic Matching

Tags serve two purposes:

1. **Filtered selection**: The model calls `selectReferences(filter_tags=["auth"])` to show only matching sources.
2. **Semantic matching**: Tags are compared against user prompts to surface unselected-but-relevant references (via the enrichment pipeline).

### Tag Selection Rules

| Pattern | Matches? | Why |
|---|---|---|
| `java` in "We use java here" | Yes | Standalone word |
| `java` in "JAVA is popular" | Yes | Case-insensitive |
| `java` in "languages (java, python)" | Yes | Punctuation boundary |
| `java` in "java.util.concurrent" | No | Dot boundary (package name) |
| `java` in "CircuitBreaker.java" | No | File extension |
| `java` in "/usr/lib/java/bin" | No | Path segment |
| `circuit breaker` (multi-word) | Yes | Multi-word tags match as phrases |
| `spring.boot` (dotted tag) | Yes | Dotted tags match when standalone |

### Choosing Good Tags

Choose 8–20 tags covering:

| Category | Examples |
|---|---|
| Feature name and synonyms | `"circuit-breaker"`, `"retry"`, `"timeout"` |
| Related subsystems | `"resilience"`, `"fault-tolerance"`, `"patterns"` |
| Technology domain | `"java"`, `"spring"`, `"microservice"` |
| Configuration-related | `"config"`, `"schema"`, `"validation"` |
| Domain-specific terms | `"hystrix"`, `"bulkhead"`, `"rate-limiter"` |

A future agent searching for "how do I configure garbage collection for a subagent" should match via tags like `"gc"`, `"subagent"`, `"profile"`.

---

## 8. Embedding and the Sidecar Matrix

References can be embedded with vector representations for semantic matching at runtime. This is optional — references without embeddings still work via tag-based matching.

### How It Works

1. The `compute_embedding` tool computes a vector for the reference document's content.
2. Vectors are collected and stored in a **sidecar `.npy` file** — a NumPy binary matrix of shape `(N, D)` where N is the number of embedded references and D is the dimension count (384 for `all-MiniLM-L6-v2`).
3. Each reference's `embedding.index` is its row position in the matrix.
4. The `embedding.source_hash` is a SHA-256 hash of the content that was embedded, used for staleness detection.

### Embedding Metadata in the JSON

```json
{
  "embedding": {
    "index": 0,
    "source_hash": "sha256:a1b2c3d4..."
  }
}
```

### Embedding Config (Separate Mode)

When using separate mode (the default), embedding configuration lives in `embedding_config.json` alongside the reference files:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding_sidecar": "references.embeddings.npy"
}
```

This file is auto-discovered by the references plugin at startup.

### Computing Embeddings

Call the `compute_embedding` tool:

```json
// Embed a file
compute_embedding(file="docs/my-feature.md")
// Returns: { "embedding": [0.0123, -0.0456, ...], "model": "all-MiniLM-L6-v2", "dimensions": 384 }
```

After computing all embeddings:

1. Compute SHA-256 of each document's content
2. Set `embedding.index` to the row position (0-based)
3. Set `embedding.source_hash` to `"sha256:<hex>"`
4. Write the `.npy` sidecar matrix in index order
5. Write the `embedding_config.json` with model, dimensions, and sidecar filename

### Incremental Re-Indexing

On re-index, if `source_hash` matches the current file content, skip re-embedding — reuse the existing row. Only re-embed changed files, then rewrite the sidecar.

### Available Embedding Models

| Provider | Model | Dimensions | Latency | Notes |
|---|---|---|---|---|
| `local` | `all-MiniLM-L6-v2` | 384 | ~5ms CPU | Default. Zero network dependency |
| `local` | `nomic-embed-text-v1.5` | 768 | ~15ms CPU | Higher quality |
| `vertexai` | `text-embedding-004` | 768 | ~20–40ms + net | Native to Gemini stack |

---

## 9. Validation

### Reference JSON Validation

Use the `validateReference` tool to validate a reference JSON file:

```
validateReference(path=".jaato/references/my-feature.json")
```

This calls `validate_reference_file()` which checks all the rules listed in [Section 6](#reference-json-catalog-entry).

### Common Validation Errors

| Error | Cause | Fix |
|---|---|---|
| `'id' is required` | Missing or empty `id` field | Add a unique ID |
| `'name' is required` | Missing or empty `name` field | Add a human-readable name |
| `'path' is required for local type` | `type` is `"local"` but no `path` | Add the `path` field |
| `Invalid type 'foo'` | Unknown source type | Use one of: `local`, `url`, `mcp`, `inline` |
| `'contents' has unknown keys` | `contents` has keys other than `templates`, `validation`, `policies`, `scripts` | Remove unknown keys |
| `'embedding.index' must be an integer` | Index is a string or float | Use an integer |
| `'embedding.source_hash' should start with 'sha256:'` | Hash missing prefix | Add `"sha256:"` prefix |

### Validation Warnings

Warnings don't block but indicate potential issues:

| Warning | Meaning |
|---|---|
| `path does not exist on disk` | The referenced file wasn't found at load time |
| `'contents' has unknown keys` | Non-standard contents keys (not validated, just noted) |
| `'embedding.source_hash' should start with 'sha256:'` | Hash format is non-standard but still functional |

---

## 10. The Enrichment Pipeline — How References Reach the Model

References don't just sit in a catalog — they actively participate in prompt enrichment.

### The Pipeline Chain

```
User Prompt
     │
     ▼
┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐
│ refs    │ → │ template │ → │ multimodal │ → │ memory  │ → ...
│ (pri 20)│   │ (pri 40) │   │ (pri 60)   │   │ (pri 80)│
└─────────┘   └──────────┘   └────────────┘   └─────────┘
```

References enrichment (priority 20) runs in two passes:

**Pass 1 — Transitive Notification (one-time):**
If transitive references were discovered during initialization or a recent selection, append a one-time hint to the first user prompt listing the transitively selected sources. This fires once per selection event, then re-arms when a new transitive selection occurs.

**Pass 2 — Tag-Based Hints (per-prompt):**
Scan the prompt for tags from unselected sources. If matches are found, append a hint listing the matching references. Already-selected sources are excluded.

### Auto-Injected References

References with `mode: "auto"` are included in system instructions at startup. The model reads their content immediately — no selection step needed. These are for always-needed documentation like coding standards or project rules.

### Pre-selected References in Subagent Profiles

References can be pre-selected via subagent profile `plugin_configs`:

```json
{
  "plugins": ["references", "template"],
  "plugin_configs": {
    "references": {
      "preselected": ["circuit-breaker-pattern"],
      "exclude_tools": ["selectReferences"]
    }
  }
}
```

This loads the reference (and its transitive dependencies) at session startup, so the subagent starts with full knowledge and doesn't need interactive selection.

---

## 11. Transitive Discovery

When a selected document mentions other catalog references, those references are automatically selected too.

### Detection Strategies

**ID-based matching:** Scans content for catalog IDs as whole words. Handles: `retry-pattern`, `@ref:retry-pattern`, `[[retry-pattern]]`, `` `retry-pattern` ``.

**Path-based matching:** Resolves relative file paths in markdown links against the source directory. `[retry](./retry.md)` → resolves relative to source. Strips anchor fragments.

### Safety Mechanisms

| Mechanism | Value |
|---|---|
| Max depth | `MAX_TRANSITIVE_DEPTH = 10` |
| Cycle detection | Visited set prevents infinite loops |
| Whole-word boundaries | `skill-001` does not match `skill-001-extended` |
| Path-only for LOCAL | URL and MCP sources cannot be scanned |

### What Gets Selected

When reference A mentions reference B (by ID or path), B is selected and annotated as "transitively included via @A". If B mentions C, C is also selected (depth 2). The parent chain is tracked so the model knows why each reference was selected.

---

## 12. Runtime Internals — What the Source Code Reveals

### 12.1 Config Loading: `load_config()` in `config_loader.py`

The `load_config()` function merges sources from three locations:

1. **Explicit config file** (`references.json`) — searched in order: `REFERENCES_CONFIG_PATH` env var, `./references.json`, `./.references.json`, `~/.config/jaato/references.json`
2. **Auto-discovered files** from `.jaato/references/*.json` — each `.json` file defines one source
3. **Merge rule**: Explicit sources take priority. Auto-discovered sources are added only if their ID is new.

After loading, `_load_embedding_config()` discovers `embedding_config.json` in the references directory and populates `embedding_model`, `embedding_dimensions`, and `embedding_sidecar`.

### 12.2 Path Resolution

For `local` sources, paths are resolved as follows:

- **Absolute paths**: resolved as-is via `Path.resolve()`
- **`.jaato/` relative paths**: resolved against the project root (parent of `.jaato/`), NOT against the reference file's directory
- **Other relative paths**: resolved against the reference file's directory
- **Windows MSYS2**: paths are normalized to forward slashes

The resolved path is stored in `resolved_path` and made relative to the project root for the model to use.

### 12.3 Sandbox Authorization

When a `local` reference is selected via `selectReferences`, its path is authorized for readonly access through the sandbox manager. When unselected, the authorization is removed. This means references outside the workspace are automatically made accessible and automatically cleaned up.

### 12.4 The `compute_embedding` Tool

The tool supports two mutually exclusive inputs:
- `input`: a text string to embed
- `file`: a path to a file whose contents should be embedded

It returns the embedding vector, model name, dimensions, and input token count. Both the tool handler and the references plugin's internal API delegate to the same underlying function, ensuring vector space consistency.

### 12.5 Separate vs Merged Embedding Mode

In **separate mode** (the default), embedding metadata is written to `embedding_config.json` alongside individual reference JSON files. The references plugin discovers this file at startup.

In **merged mode**, embedding metadata lives in the top-level `references.json` under `embedding_model`, `embedding_dimensions`, and `embedding_sidecar` fields. Merged mode takes precedence if both exist.

### 12.6 The `embedding_config.json` Format

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding_sidecar": "references.embeddings.npy"
}
```

All three fields are required. If any is missing, the config is ignored (logged as warning).

---

## 13. Source Code Map

| File | Contents |
|---|---|
| `shared/plugins/references/models.py` | `ReferenceSource`, `SourceType`, `InjectionMode`, `EmbeddingMetadata`, `ReferenceContents`, `SelectionRequest`, `SelectionResponse` dataclasses |
| `shared/plugins/references/config_loader.py` | `load_config()`, `discover_references()`, `validate_config()`, `validate_reference_file()`, `validate_source()`, `resolve_source_paths()`, `_load_embedding_config()`, `ReferencesConfig`, `ConfigValidationError` |
| `shared/plugins/references/plugin.py` | Core plugin: `selectReferences`, `listReferences`, `validateReference`, `compute_embedding` tool handlers; selection, transitive resolution, enrichment logic |
| `shared/plugins/references/channels.py` | Console, Webhook, File, Queue channel implementations for selection interaction |
| `shared/plugins/references/tests/` | Tests for registry integration, transitive resolution, enrichment |
| `shared/plugins/template/plugin.py` | Template extraction, index, rendering, standalone discovery |
| `shared/plugins/template/tests/` | Template index, discovery, rendering, cross-plugin integration tests |
| `docs/jaato_knowledge_management.md` | Architecture overview of the full knowledge pipeline |
| `docs/compute-embedding-tool-spec.md` | Specification for the `compute_embedding` tool and embedding infrastructure |
| `docs/jaato_subagent_profiles_reference.md` | Related: how subagent profiles can pre-select references |
