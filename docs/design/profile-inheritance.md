# Profile Inheritance

**Status:** Approved for implementation
**Date:** 2026-03-17 (updated 2026-03-29)

## Problem

Profiles already function as role definitions for agents — they control which
plugins load, permission policy overrides, system instructions, model/provider,
GC strategy, and environment variables. This is RBAC in all but name.

The missing piece is **composability**. When multiple profiles share a common
baseline (e.g. read-only access, or a standard set of dev tools), the plugin
list, permission config, and other fields must be duplicated across each profile.
This makes maintenance fragile and violates DRY.

## Proposal: `inherits` Field

Add an `inherits` field to `SubagentProfile` that references one or more parent
profile names. The child profile merges parent fields with its own overrides.

```json
{
  "name": "researcher",
  "inherits": ["readonly", "web_capable"],
  "model": "claude-sonnet-4-20250514",
  "system_instructions": "You are a research analyst."
}
```

### Multiple Inheritance

Multiple parents are supported. Merge conflicts are **hard errors** — if two
parents define conflicting values for the same scalar field and the child does
not explicitly override it, session creation fails with a clear diagnostic.

This avoids MRO complexity entirely. There is no implicit conflict resolution;
the child must be explicit.

## Merge Semantics

Fields fall into two categories based on their type:

### Collection Fields (union, no conflicts possible)

| Field | Merge Rule |
|-------|------------|
| `plugins` | Union of all parent lists + child list. Duplicates removed (preserving child order preference). |
| `preloaded_plugins` | Union of all parent sets + child set. |
| `env` | Dict merge: parents merged left-to-right, child overrides last. **Conflict rule**: if two parents define the same env key with different values and child doesn't override → error. |
| `plugin_configs` | Deep merge by plugin name. Within a plugin's config dict, same conflict rule as `env`. |

### Scalar Fields (agreement-or-override)

| Field | Merge Rule |
|-------|------------|
| `model` | If multiple parents define it, values must agree or child must override. |
| `provider` | Same as `model`. |
| `system_instructions` | Concatenation in inheritance order (grandparent → parent → child), separated by `\n\n`. No conflict possible — all layers contribute. |
| `max_turns` | Most restrictive (minimum) across parents. Child can override. |
| `gc` | If multiple parents define it, values must agree (field-by-field) or child must override the entire `gc` block. |
| `description` | Child must define its own. Not inherited. |

### The Golden Rule

> **A child can only narrow collections inherited from parents by explicitly
> removing entries (not yet supported — omission means "inherit all"). A child
> can override any scalar. Conflicts between parents are always errors.**

## Error Reporting

When a conflict is detected, session creation fails with a message like:

```
Profile 'research_writer' inherits from ['readonly', 'writer'].
Conflict on field 'model':
  - readonly: "claude-haiku-4-5-20251001"
  - writer:   "claude-sonnet-4-20250514"
Override 'model' explicitly in 'research_writer' to resolve.
```

Errors are collected exhaustively — all conflicts reported at once, not
fail-on-first.

## Cycle Detection

Inheritance chains are resolved with cycle detection. If `A inherits B` and
`B inherits A`, profile discovery reports an error and neither profile is
available.

Implementation: standard visited-set DFS during profile resolution.

## Resolution Order

1. **Profile discovery** scans all sources (workspace → user → premium) as today.
   Raw profiles are stored with their `inherits` field unresolved.

2. **Profile resolution** is a new phase that runs after discovery. For each
   profile with `inherits`:
   a. Recursively resolve parent profiles (with cycle detection).
   b. Merge parent fields bottom-up (deepest ancestor first).
   c. Apply child overrides on top.
   d. Validate: detect conflicts, report errors.
   e. Store the **resolved** profile (no `inherits` field remains).

3. **Session creation** uses resolved profiles only. No inheritance logic at
   session creation time — it's all front-loaded during discovery.

This keeps the hot path (session creation) simple and fast.

## Schema Change

```python
@dataclass
class SubagentProfile:
    name: str
    description: str
    inherits: Optional[List[str]] = None  # NEW — parent profile names
    plugins: List[str] = field(default_factory=list)
    preloaded_plugins: set = field(default_factory=set)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_instructions: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 10
    gc: Optional[GCProfileConfig] = None
    env: Dict[str, str] = field(default_factory=dict)
```

JSON schema adds:

```json
{
  "inherits": {
    "type": "array",
    "items": { "type": "string" },
    "description": "Parent profile names to inherit from. Multiple parents supported; conflicts are errors."
  }
}
```

A single string is also accepted and normalized to a one-element list:

```json
{ "inherits": "readonly" }
```

## Examples

### Base profiles (no inheritance)

```json
// .jaato/profiles/readonly.json
{
  "name": "readonly",
  "description": "Read-only codebase access",
  "plugins": ["filesystem_query", "memory"],
  "plugin_configs": {
    "permission": {
      "defaultPolicy": "deny",
      "whitelist": { "tools": ["readFile", "listDirectory", "grep_search"] }
    }
  }
}
```

```json
// .jaato/profiles/web_capable.json
{
  "name": "web_capable",
  "description": "Web search and fetch capabilities",
  "plugins": ["web_search"],
  "plugin_configs": {
    "permission": {
      "whitelist": { "tools": ["web_search", "web_fetch"] }
    }
  }
}
```

### Child profile (single inheritance)

```json
// .jaato/profiles/researcher.json
{
  "name": "researcher",
  "inherits": "readonly",
  "description": "Research analyst with read-only codebase access",
  "plugins": ["web_search"],
  "model": "claude-sonnet-4-20250514",
  "system_instructions": "You are a research analyst. Cite sources."
}
```

Resolved `plugins`: `["filesystem_query", "memory", "web_search"]`

### Child profile (multiple inheritance)

```json
// .jaato/profiles/research_reviewer.json
{
  "name": "research_reviewer",
  "inherits": ["readonly", "web_capable"],
  "description": "Reviews code with web research capability",
  "system_instructions": "You review code and can research best practices online."
}
```

Resolved `plugins`: `["filesystem_query", "memory", "web_search"]` (union, deduplicated)

Resolved `plugin_configs.permission.whitelist.tools`:
`["readFile", "listDirectory", "grep_search", "web_search", "web_fetch"]` (union)

### Conflict example (error)

```json
// .jaato/profiles/fast_reader.json
{
  "name": "fast_reader",
  "inherits": "readonly",
  "description": "Fast read-only agent",
  "model": "claude-haiku-4-5-20251001"
}
```

```json
// .jaato/profiles/slow_writer.json
{
  "name": "slow_writer",
  "inherits": "readonly",
  "description": "Careful file writer",
  "model": "claude-sonnet-4-20250514",
  "plugins": ["file_edit"]
}
```

```json
// .jaato/profiles/broken.json — THIS WILL FAIL
{
  "name": "broken",
  "inherits": ["fast_reader", "slow_writer"],
  "description": "Conflicting parents"
}
```

Error:
```
Profile 'broken' inherits from ['fast_reader', 'slow_writer'].
Conflict on field 'model':
  - fast_reader: "claude-haiku-4-5-20251001"
  - slow_writer: "claude-sonnet-4-20250514"
Override 'model' explicitly in 'broken' to resolve.
```

Fix: add `"model": "claude-sonnet-4-20250514"` to `broken.json`.

## Implementation Scope

### Phase 1: Core Inheritance

1. Add `inherits` field to `SubagentProfile` dataclass.
2. Add `resolve_profiles()` function in `config.py` that runs after
   `discover_profiles()` — handles recursive resolution, cycle detection,
   merge logic, and conflict detection.
3. Update `discover_profiles()` to return unresolved profiles, then call
   `resolve_profiles()` as a second pass.
4. Update `validate_profile()` to accept `inherits` field.
5. Update profile parsing in `_scan_profiles_dir()` to read `inherits`.

### Phase 2: Wiring

6. Update `SubagentConfig` to call `resolve_profiles()` during initialization.
7. Update `session.new --profile` flow to use resolved profiles.
8. Update `session.profiles` listing to show inheritance chain.

### Phase 3: UX

9. Add `--show-resolved` flag to `session.profiles` to display the fully
   merged profile (useful for debugging inheritance).
10. Profile validation command: `profiles validate` that checks all profiles
    for parse errors, missing parents, cycles, and conflicts.

## Non-Goals

- **Plugin removal syntax** (e.g. `"-cli"` in plugins list). Not needed yet.
  If a parent grants a plugin you don't want, create a different base profile.
- **Trait-based filtering**. Profiles work at plugin-name granularity, which is
  concrete and sufficient. Trait-based rules add abstraction without clear
  immediate value.
- **Runtime re-resolution**. Profiles are resolved at discovery time. Changing a
  parent profile requires re-discovery (server restart or explicit reload).
