---
description: Produce a comprehensive self-contained reference document and reference JSON for a codebase feature, suitable for semantic matching and future agent retrieval
tags: ['references', 'documentation', 'knowledge-generation', 'codebase-analysis', 'embedding', 'self-contained-reference']
---

You are given a codebase feature to document. Your job is to produce a **single comprehensive markdown document** that a different agent — with no prior knowledge of this codebase — can read and use to fully understand, configure, and reason about the feature.

## Input

- **Feature**: `{{feature}}` — the name or description of the codebase feature to reference
- **Entry-point files**: `{{entry_point_files}}` — comma-separated list of files the user believes are most relevant (starting points for exploration)

If either parameter is unresolved (still contains `{{...}}` placeholders), ask the user to provide it before starting.

## The two outputs you must produce

1. **Reference document** (`docs/<feature-kebab>.md`) — the comprehensive markdown document
2. **Reference JSON** (`.jaato/references/<feature-kebab>.json`) — the catalog entry pointing to it

Both must be validated before you finish.

## Research phase — read before you write

Before writing a single word of the document, you must **thoroughly read the source code**. This is the most important step. A reference that only summarizes surface-level documentation is nearly useless — the value comes from capturing what the docs don't say.

### What to read (in order)

1. **Entry-point files** — read every line of every file the user specified. These are the primary sources.

2. **Data models and config files** — search for dataclasses, Pydantic models, config schemas, JSON schemas, type definitions. These define the *shape* of the feature. Use `grep_content` with patterns like `class |@dataclass|@dataclass_json|Schema|BaseModel|TypedDict` in the relevant directories.

3. **Validation functions** — search for `validate_|_validate_|is_valid|check_` in the feature's source files. Validation functions reveal the exact constraints and edge cases.

4. **Test files** — search for `test_` in the feature's test directories. Tests are often the most precise specification of expected behavior — they show what inputs are valid, what errors are produced, and what edge cases exist.

5. **Plugin registration and tool schemas** — search for how the feature exposes itself to the rest of the system. Look for `register_tool|tool_name|get_tool_schemas|execute_` in the feature's plugin file.

6. **Cross-references** — when the source code references other files (imports, function calls to other modules), follow those references. A complete reference traces the full call chain, not just the local function.

### Reading strategy

- **Read whole files, not snippets.** Partial reads miss context. Use `readFile(path, offset=N, limit=M)` for large files.
- **Never fabricate.** If you didn't read it, don't write about it. If a function's implementation isn't visible in what you read, say "implementation details not inspected" rather than guessing.
- **Track what you've read.** Keep a list of files and line ranges you've actually inspected. Your confidence in each section of the document should correlate with how much source code you read for it.
- **Note gaps.** If you couldn't find the implementation of something (e.g., it's in a premium package you can't access), explicitly say so in the document.

## Document structure

Organize the document so a reader can find what they need quickly. Use this structure:

```
# <Feature Name> — Complete Reference

> Scope: one-sentence summary of what the document covers

## Table of Contents

## 1. What Is <Feature>?
   (Definition, purpose, where it fits in the system)

## 2. <Core Concept> — Overview
   (High-level architecture or mental model)

## 3–N. <Topic Sections>
   (One section per major aspect. Each section should be self-contained
    enough that reading it alone gives you full understanding of that topic.)

## N+1. Configuration / Schema / API Reference
   (Complete field-by-field reference with types, defaults, constraints.
    This is the section a developer uses as a lookup table.
    Include validation rules from the actual validation functions.)

## N+2. Runtime Internals — What the Source Code Reveals
   (Implementation details and design decisions that can only be learned
    by reading source code. This is what makes the document self-contained
    for an agent that can't run `grep` itself.)

## N+3. Source Code Map
   (Table mapping each source file to what it contains.
    Enables a human or agent to know where to look for more detail.)
```

### Section writing rules

**Every section must pass the "stranger test":** if a developer who has never seen this codebase reads only that section, can they understand the topic well enough to use it correctly?

To pass this test:
- Define all terms before using them
- Include concrete examples (JSON, code snippets) for every abstract concept
- Show the *shape* of data (field names, types, constraints) not just a narrative description
- Explain *why*, not just *what* — design decisions are as important as mechanics
- Include validation rules — what happens when you pass wrong input?

### Style rules

- Use tables for field references and comparisons
- Use code blocks (with language tags) for all examples
- Use bold for field names when referenced in prose
- Use blockquotes for design rationale
- Number sections and provide a table of contents
- Do NOT use ANSI escape codes for syntax highlighting

## Reference JSON

After writing the document, create the reference catalog entry:

```json
{
  "id": "<feature-kebab>",
  "name": "<Human Readable Name>",
  "description": "<Comprehensive description covering all major aspects>",
  "type": "local",
  "path": "/absolute/path/to/docs/<feature-kebab>.md",
  "mode": "selectable",
  "tags": ["<tag1>", "<tag2>", ...],
  "fetchHint": "Read the full document at the path above",
  "contents": {
    "templates": null,
    "validation": null,
    "policies": null,
    "scripts": null
  }
}
```

### Tags

Choose 8–20 tags that cover:
- The feature name and synonyms
- Related features or subsystems it interacts with
- The technology domain (e.g., "gc", "permissions", "serialization")
- Configuration-related tags
- Any domain-specific terms

Tags are used for semantic matching. A future agent searching for "how do I configure garbage collection for a subagent" should match this reference via tags like `"gc"`, `"subagent"`, `"profile"`.

## Embedding and validation

1. Compute the SHA-256 hash of the document's content: `sha256sum docs/<feature-kebab>.md`
2. Compute the embedding of the document using `compute_embedding(file=<absolute-path>)`
3. Add the embedding metadata to the reference JSON:
   ```json
   "embedding": {
     "index": 0,
     "source_hash": "sha256:<hex-digest>"
   }
   ```
4. Persist the embedding vector to `.jaato/references/.embeddings_main.json`
5. Save the sidecar matrix to `.jaato/references/references.embeddings.npy`
6. Validate the reference JSON using `validateReference(path=".jaato/references/<feature-kebab>.json")`
7. Fix any validation errors before finishing

## The self-containedness principle

The reference document is the **sole artifact** a future agent will read. The reference JSON is just a pointer. Therefore:

- **The document must contain everything.** Do not rely on the agent reading other files — the document IS the reference.
- **Do not point to external files for essential information.** The `fetchHint` can suggest supplementary reading, but all required knowledge must be in the document itself.
- **Capture implementation details.** If something is only discoverable by reading source code (design decisions, implicit constraints, runtime wiring), include it in a "Runtime Internals" section. This is what separates a useful reference from a useless one.
- **Be precise.** "The system validates X" is weak. "The `validate_reference_file()` function in `config_loader.py:315` rejects X if Y is missing, returning `('id' is required)`" is strong.

## Anti-patterns to avoid

- ❌ **Pointer references**: "See `config.py` for the full schema" — the agent reading the reference can't necessarily read `config.py`. Include the schema in the document.
- ❌ **Vague descriptions**: "The system handles errors gracefully" — say what errors, how they're handled, and what the user sees.
- ❌ **Missing edge cases**: Only documenting the happy path. Validation rules and error paths are often more useful than the success path.
- ❌ **Stale references**: Pointing to files or line numbers that may change. Reference file paths and function names, but don't hardcode line numbers.
- ❌ **Truncated coverage**: Documenting 80% of a feature and leaving the rest as "exercise for the reader." The agent doesn't have a reader.
- ❌ **Copying existing docs**: If an architecture doc already exists, don't just point to it — incorporate its content and expand on it with what the source code reveals.
- ✅ **Complete, precise, self-contained** — the gold standard.

## Quality checklist (verify before finishing)

- [ ] Every field in every schema is documented with type, default, constraints
- [ ] Validation rules are listed explicitly (from actual validation functions)
- [ ] At least one concrete example per major concept
- [ ] Design decisions have rationale (why, not just what)
- [ ] Runtime wiring is explained (how the pieces connect at runtime)
- [ ] The "Runtime Internals" section contains information not available from docs alone
- [ ] Tags cover all reasonable search queries a future agent might use
- [ ] The reference JSON validates with 0 errors and 0 warnings
- [ ] The embedding is computed from the document, not from a different file
- [ ] A stranger could read the document and correctly configure/use the feature
