# JAATO Subagent Profiles — Complete Reference

> **Scope**: Everything a profile JSON file can configure, how profiles are discovered and resolved at runtime, how per-turn model-tier switching works, and how the `gen-references` pipeline auto-generates profiles from knowledge bases.

---

## Table of Contents

1. [What Are Subagent Profiles?](#1-what-are-subagent-profiles)
2. [Where Profiles Live — 3-Tier Discovery](#2-where-profiles-live--3-tier-discovery)
3. [Profile JSON Schema — Complete Field Reference](#3-profile-json-schema--complete-field-reference)
4. [GC Configuration (`gc` sub-object)](#4-gc-configuration-gc-sub-object)
4.4. [Runtime Limits (`runtime_limits` sub-object)](#44-runtime-limits-runtime_limits-sub-object)
4.5. [Typed Completion Payloads — `completion_payload_schema`](#45-typed-completion-payloads--completion_payload_schema)
5. [Multi-Tier Model Config (`model_tiers`)](#5-multi-tier-model-config-model_tiers)
6. [Variable Expansion and Secret Resolution](#6-variable-expansion-and-secret-resolution)
7. [Plugin Preloading](#7-plugin-preloading)
8. [Embedding Plugin Configs into Profiles](#8-embedding-plugin-configs-into-profiles)
9. [Profile Inheritance](#9-profile-inheritance)
10. [Inline vs Profile-Based Delegation](#10-inline-vs-profile-based-delegation)
11. [SubagentConfig — Top-Level Plugin Configuration](#11-subagentconfig--top-level-plugin-configuration)
12. [The Delegation Lifecycle](#12-the-delegation-lifecycle)
13. [Cross-Provider Subagents](#13-cross-provider-subagents)
14. [Permission Bridging](#14-permission-bridging)
15. [Resource Sharing vs Isolation](#15-resource-sharing-vs-isolation)
16. [SubagentResult](#16-subagentresult)
17. [Profile Validation](#17-profile-validation)
18. [gen-references Pipeline — Auto-Generating Profiles](#18-gen-references-pipeline--auto-generating-profiles)
19. [Reference JSON Format](#19-reference-json-format)
20. [Source Code Map](#20-source-code-map)
21. [Runtime Internals — What the Source Code Reveals](#21-runtime-internals--what-the-source-code-reveals)

## 1. What Are Subagent Profiles?

A **subagent profile** is a named JSON or YAML configuration file that defines the capabilities and constraints of a child agent spawned by the parent agent during delegation. Profiles live in `.jaato/profiles/` (workspace) or `~/.jaato/profiles/` (user-global) and are discovered automatically at startup.

Each profile specifies:
- **Which tools** the subagent can use (`plugins`)
- **Which model/provider** it runs on (can differ from the parent)
- **How many turns** it may take before returning
- **What GC strategy** manages its context window
- **What environment variables** it has access to
- **What other profiles** it inherits from
- **Per-plugin configuration** via `plugin_configs` (e.g., embedding permission policies)
- **What structured output schema** it must produce on completion
- **Per-turn model-tier config** via `model_tiers` (multi-model switching within a session)

Profiles are the mechanism by which JAATO implements **specialized delegation**: instead of one monolithic agent doing everything, the parent delegates focused tasks (code review, testing, documentation generation, etc.) to child agents whose capabilities are precisely scoped through profiles.

### Relationship to Agents

Profiles are **runtime configuration** — they define tools, model overrides, GC strategy, and environment variables. They are distinct from **agents** (`.jaato/agents/*.md`), which provide the *instructions* (system prompt) that tell the subagent *how* to behave.

When both an agent and a profile are specified, the agent's rendered markdown becomes the `system_instructions`, and the profile provides the runtime scaffolding. The `system_instructions` field on profiles is **deprecated** in favor of agents.

---

## 2. Where Profiles Live — 3-Tier Discovery

Profiles are discovered from three sources in decreasing order of precedence:

```
┌─────────────────────────────────────────────┐
│  Tier 1: Workspace (highest precedence)     │
│  {base_path}/.jaato/profiles/*.json         │
│  {base_path}/.jaato/profiles/*.yaml         │
│  {base_path}/.jaato/profiles/*.yml          │
├─────────────────────────────────────────────┤
│  Tier 2: User-global                        │
│  ~/.jaato/profiles/*.json                   │
│  ~/.jaato/profiles/*.yaml                   │
│  ~/.jaato/profiles/*.yml                    │
├─────────────────────────────────────────────┤
│  Tier 3: Premium (entry points)             │
│  jaato.premium → profiles entry point       │
└─────────────────────────────────────────────┘
```

**Conflict resolution**: When the same profile name appears in multiple tiers, the higher-precedence source wins. Workspace profiles always override user-global and premium profiles.

**File naming**: The profile's `name` field takes precedence over the filename stem. If `name` is absent, the filename (without extension) is used as the profile name.

**Discovery is triggered by**: `discover_profiles(profiles_dir, base_path)` which scans all three tiers, parses each file into a `SubagentProfile` dataclass, then resolves inheritance.

---

## 3. Profile JSON Schema — Complete Field Reference

### Minimal Valid Profile

```json
{
  "name": "my_agent",
  "description": "Does something useful"
}
```

### Full Profile (All Fields)

```json
{
  "name": "code_reviewer",
  "description": "Reviews code changes for quality, security, and best practices",
  "plugins": ["file_edit", "cli", "grep_content", "template(preload)"],
  "plugin_configs": {
    "lsp": { "config_path": "${projectPath}/.lsp.json" },
    "mcp": { "config_path": "${projectPath}/.mcp.json" }
  },
  "system_instructions": "You are a code reviewer. Focus on bugs, security issues, and readability.",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "max_turns": 5,
  "gc": {
    "type": "truncate",
    "threshold_percent": 80.0,
    "target_percent": 60.0,
    "pressure_percent": 0,
    "preserve_recent_turns": 3,
    "notify_on_gc": true
  },
  "env": {
    "PROJECT_ROOT": "${workspaceRoot}",
    "API_KEY": "vault://secret/myapp#db_password"
  },
  "inherits": ["readonly", "web_capable"],
  "completion_payload_schema": {
    "type": "object",
    "properties": {
      "passed": { "type": "boolean" },
      "errors": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["passed"]
  },
  "model_tiers": {
    "planner": "claude-opus-4-7",
    "dispatcher": "claude-sonnet-4-6",
    "executor": "claude-haiku-4-5",
    "initial": "dispatcher",
    "fallback": "dispatcher"
  }
}
```

### Field Reference Table

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `string` | **Yes** | — | Unique identifier. Used to select the profile during delegation. |
| `description` | `string` | **Yes** | — | Human-readable description. Shown in `list_subagent_profiles` output and UI. |
| `plugins` | `string[]` | No | `[]` | Plugin names to enable. Core plugins (introspection, file_edit) are auto-added. Supports `(preload)` suffix. |
| `plugin_configs` | `object<string, object>` | No | `{}` | Per-plugin configuration overrides. Keys are plugin names, values are config dicts. Supports `${VAR}` expansion. |
| `system_instructions` | `string \| null` | No | `null` | **Deprecated.** Use `.jaato/agents/` instead. |
| `model` | `string \| null` | No | `null` | Model override. `null` = inherit from parent. **Ignored** when `model_tiers` is non-empty (with warning). |
| `provider` | `string \| null` | No | `null` | Provider override (e.g., `"anthropic"`, `"google_genai"`). `null` = inherit from parent. |
| `max_turns` | `integer` | No | `10` | Maximum conversation turns before the subagent returns to the parent. Must be positive. |
| `gc` | `object \| null` | No | `null` | GC configuration. See [Section 4](#4-gc-configuration-gc-sub-object). |
| `env` | `object<string, string>` | No | `{}` | Session-scoped environment variables. Values support `${VAR}` expansion and secret URI resolution. |
| `inherits` | `string \| string[]` | No | `null` | Parent profile names for inheritance. Resolved during `discover_profiles()`. |
| `completion_payload_schema` | `object \| string \| null` | No | `null` | JSON Schema constraining `signal_completion` payload. See [§4.5](#45-typed-completion-payloads--completion_payload_schema). |
| `runtime_limits` | `object` | No | `null` | Per-session resource consumption caps. See [§4.4](#44-runtime-limits-runtime_limits-sub-object). |
| `model_tiers` | `object` | No | `{}` | Per-turn model-tier configuration. See [Section 5](#5-multi-tier-model-config-model_tiers). Empty dict = single-model mode. |

### Validation Rules

| Field | Rule |
|---|---|
| `name` | Must be non-empty string |
| `description` | Must be non-empty string |
| `plugins` | Must be array of strings (if present) |
| `plugin_configs` | Must be object of objects (if present) |
| `max_turns` | Must be positive integer (not bool) |
| `model` | Must be string or null |
| `provider` | Must be string or null |
| `env` | Must be object with string keys and string values (if present) |
| `inherits` | Must be string, array of strings, or null |
| `gc.type` | Must be one of: `truncate`, `summarize`, `hybrid`, `budget` |
| `gc.threshold_percent` | Must be 0–100 |
| `gc.target_percent` | Must be 0–100 |
| `gc.pressure_percent` | Must be 0–100 |
| `gc.preserve_recent_turns` | Must be non-negative integer |
| `gc.max_turns` | Must be positive integer (if present) |
| `model_tiers` | Must be object (if present). Tier keys must be in `{"planner", "dispatcher", "executor", "vision"}`. Reserved keys `"initial"` and `"fallback"` must be strings. At least one tier→model mapping required. A tier entry's `provider` / `description` must be non-empty strings when set; tiers may name different providers. |

---

## 4. GC Configuration (`gc` sub-object)

Each subagent can have its own garbage collection strategy. This controls how the agent's context window is managed to prevent token overflow during long-running tasks.

```json
{
  "gc": {
    "type": "truncate",
    "threshold_percent": 80.0,
    "target_percent": 60.0,
    "pressure_percent": 0,
    "preserve_recent_turns": 3,
    "notify_on_gc": true,
    "summarize_middle_turns": null,
    "max_turns": null,
    "plugin_config": {}
  }
}
```

### GC Strategy Types

| Type | Behavior |
|---|---|
| `truncate` | Discards oldest turns until context falls below `target_percent`. Fastest, lossy. |
| `summarize` | Summarizes older turns into a compact representation. Slower, preserves semantics. |
| `hybrid` | Summarizes middle turns (controlled by `summarize_middle_turns`), truncates if needed. |
| `budget` | Budget-based GC with continuous pressure tracking. Used for long-running research agents. |

### GC Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `"truncate"` | Strategy type. |
| `threshold_percent` | `float` | `80.0` | Trigger GC when context usage exceeds this percentage. |
| `target_percent` | `float` | `60.0` | Target usage after GC completes. |
| `pressure_percent` | `float \| null` | `90.0` | When PRESERVABLE turns can be touched. `0` or `null` enables **continuous mode**. |
| `preserve_recent_turns` | `int` | `5` | Number of recent turns to always preserve during GC. |
| `notify_on_gc` | `bool` | `true` | Inject a notification into history after GC runs. |
| `summarize_middle_turns` | `int \| null` | `null` | For `hybrid` strategy: number of middle turns to summarize. |
| `max_turns` | `int \| null` | `null` | Trigger GC when turn count exceeds this limit. |
| `plugin_config` | `object` | `{}` | Additional plugin-specific configuration passed to the GC plugin. |

### Continuous Mode

Setting `pressure_percent` to `0` or `null` enables continuous mode — GC runs aggressively to keep context as small as possible. Useful for long-running subagents that accumulate lots of tool output.

---

## 4.4 Runtime Limits (`runtime_limits` sub-object)

Per-session resource consumption caps, orthogonal to GC (which manages context window size). Answers "how much can this session *consume*?" vs sandboxing/AppArmor which answers "what can it *touch*?".

The field is an object with five optional keys. All fields default to `null` (no limit / inherit host default):

```json
{
  "runtime_limits": {
    "memory_max_mb": 4096,
    "pids_max": 1024,
    "cpu_weight": 200,
    "tool_timeout_seconds": 600,
    "max_output_bytes": 1048576
  }
}
```

### Field Reference

| Field | Type | Layer | What it does |
|---|---|---|---|
| `memory_max_mb` | int (positive) | Kernel (cgroup v2) | Written to `memory.max`. Process tree gets OOM-killed if it exceeds this. |
| `pids_max` | int (positive) | Kernel (cgroup v2) | Written to `pids.max`. `fork()` returns EAGAIN beyond this count. |
| `cpu_weight` | int 1–10000 | Kernel (cgroup v2) | Written to `cpu.weight` (default 100). Relative scheduling weight against sibling cgroups. |
| `tool_timeout_seconds` | float (positive) | App (Python) | Wall-clock cap on each subprocess tool call. SIGTERM with 2s grace, then SIGKILL. |
| `max_output_bytes` | int (positive) | App (Python) | Override of the default stdout/stderr capture cap in CLI tool results. |

### Two Enforcement Layers

The fields split into two enforcement layers, but a profile author treats them as one knob set:

- **Kernel-enforced** (`memory_max_mb`, `pids_max`, `cpu_weight`): Written once into cgroup v2 controller files when the session starts. When `has_kernel_limits()` returns `False`, no cgroup directory is created at all.
- **Application-enforced** (`tool_timeout_seconds`, `max_output_bytes`): Read by the CLI / interactive_shell plugins and applied per-tool-call at the Python layer.

### Validation

All validation runs at profile load time via `RuntimeLimits.from_dict()` (frozen dataclass with `__post_init__` checks). Invalid profiles fail at parse time, not mid-session.

### Graceful Degradation

If cgroup v2 is unavailable (cgroup v1 host, missing controllers, non-writable root, non-Linux), kernel-enforced limits are skipped silently. App-layer limits still apply.

### Inheritance

`runtime_limits` follows the **scalar-override** rule (§9): parent profiles must agree, or the child must override.

---

## 4.5 Typed Completion Payloads — `completion_payload_schema`

When a profile declares a `completion_payload_schema`, the agent's `signal_completion` tool is **dynamically rewritten** to accept a typed `payload` parameter instead of the legacy untyped `summary: str`.

### How It Works

1. **Sampling-time enforcement (provider-level):** `LifecycleTools.get_tool_schemas()` rebuilds the `signal_completion` tool parameters. When a schema is declared, `summary: str` is replaced with `payload: <resolved schema>`.
2. **Server-side validation (defense-in-depth):** `LifecycleTools._execute_signal_completion()` runs `jsonschema.validate(instance=payload, schema=resolved_schema)` on every call.

### Schema Resolution

The `completion_payload_schema` field accepts three forms:

| Form | Behavior | Example |
|---|---|---|
| `null` (absent) | Legacy mode: `signal_completion` has `summary: str` | — |
| Inline dict | Used as-is | `{"type": "object", "properties": {"passed": {"type": "boolean"}}, "required": ["passed"]}` |
| String path | Resolved through 3-tier lookup, loaded as JSON | `"validator-result.json"` |

String paths are resolved by `resolve_completion_schema()` through: absolute path → `<workspace>/.jaato/completion_schemas/<path>` → `~/.jaato/completion_schemas/<path>`.

### Validation Error Response

When `jsonschema.validate` fails, the model receives:

```json
{
  "error": "validation_failed",
  "message": "The 'payload' argument did not match the profile's completion_payload_schema.",
  "validation_error": "'passed' is a required property",
  "schema_path": ["properties", "passed"]
}
```

### Inheritance

`completion_payload_schema` follows the **scalar-override** rule (§9).

---

## 5. Multi-Tier Model Config (`model_tiers`)

The `model_tiers` field enables **per-turn model switching** within a single session. Instead of running the entire conversation on one model, the agent (or subagent) can dynamically switch between a small set of named model tiers as the complexity of the current task changes — using an expensive model for deep planning, a mid-tier model for coordination, and a cheap model for mechanical tool execution.

> **Design motivation**: In a typical agent session, most turns are cheap tool calls that don't need the strongest model. Multi-tier switching lets you reserve the expensive model for the turns where it matters, reducing cost without sacrificing capability.

### 5.1 The Four Named Tiers

Three **cognitive** tiers, plus one **modality** role:

| Tier | Role | Typical Use | Cost |
|---|---|---|---|
| `planner` | Deep thought, multi-step reasoning, complex decomposition | Architecture decisions, complex debugging, cross-file analysis | Highest |
| `dispatcher` | Coordination, light reasoning, deciding which tools to call | Delegating tasks, reading results, deciding next steps | Medium |
| `executor` | Mechanical tool calls and result interpretation | Running commands, editing files, formatting output | Lowest |
| `vision` | Viewing image content (screenshots, diagrams) | Switched into before reading an image, and back out after | — |

`vision` is a modality role rather than a cognitive one, but it shares the
same single-active-tier machinery: the session is in exactly one tier at a
time. Content of a modality the active model can't accept is **withheld**,
with a tool-result note telling the model which tier to enter first. See
[Multimodal Model Support](design/multimodal-model-support.md).

Since the `modalities` key (see [§5.2](#tier-entry-forms)), that role is
carried by the **key, not the name** — any tier can declare it, and the gate
finds the tier by role. `vision` remains the only name with a built-in
meaning: it implies `modalities: [image]` so profiles written before the key
behave unchanged.

The names are **semantic conventions**, not capability levels. You can assign
any model to any tier — the framework doesn't enforce that `planner` gets the
strongest model. What the model is told each tier is *for* comes from the
`enter_tier` tool description, which you can override per tier (see
[§5.2 Tier Entry Forms](#tier-entry-forms)).

### 5.2 Schema — Unified Dict

`model_tiers` is a single-level dict that mixes **tier→model mappings** with **reserved control keys**:

```json
{
  "model_tiers": {
    "planner": "claude-opus-4-7",
    "dispatcher": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
    "executor": {
      "model": "claude-haiku-4-5",
      "description": "apply the agreed migration plan file by file; do not re-plan"
    },
    "vision": {"model": "claude-sonnet-4-6", "modalities": ["image"]},
    "initial": "dispatcher",
    "fallback": "dispatcher"
  }
}
```

#### Tier Entry Forms

Each tier value can be either:

| Form | Example | When to Use |
|---|---|---|
| **Shorthand** (string) | `"claude-opus-4-7"` | When the session's main provider handles this model |
| **Rich** (dict) | `{"model": "claude-sonnet-4-6", "provider": "anthropic"}` | When you need a different provider, or your own tier prose |

Rich-form keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `model` | `string` | yes | The model this tier selects. |
| `provider` | `string` | no | Provider plugin for this tier. Tiers may name **different** providers (see §5.5). Unset = the session's main provider. |
| `description` | `string` | no | What this tier is *for*, in the words the model reads. Becomes that tier's bullet in the `enter_tier` tool description, replacing the framework's default wording for the name. |
| `modalities` | `string[]` or `map` | no | Non-text roles this tier fills, and in which direction. Map form `{image: inbound}`; list form `["image"]` is sugar for inbound. Kinds: `image`, `audio`, `video`, `file`. Directions: `inbound`, `outbound`, `bidirectional`. A tier named `vision` implies `image: inbound`. |

##### `modalities` — declaring a role, not asserting a capability

Two places used to branch on the literal tier name `vision`: the content
gate (which withholds an attachment the active model can't view, and tells
the agent which tier to enter) and the startup capability check (which fails
loud when the image tier maps to a model that can't see). Both now resolve
the tier **by role**:

```yaml
model_tiers:
  executor: openai/gpt-5-mini
  planner:
    model: google/gemini-3-pro
    modalities: {image: inbound}   # the planner doubles as the image tier
  initial: executor
  fallback: executor
```

An image read from the `executor` tier is now withheld with
`Call enter_tier("planner") first` — before the key, the note said no vision
tier existed, because the profile happened not to use that name.

> **This DECLARES a role and is VERIFIED — the opposite direction from
> `plugin_configs.<provider>.modalities`, which ASSERTS what a model supports
> in order to correct catalog detection.** Declaring a role the tier's model
> can't fill is a config error, and catching it is the point: session
> creation fails loud at connect rather than at the first image. Use the
> provider knob when the catalog is wrong about a model; use this key when
> you're saying which tier plays which part.

`text` is rejected: every model accepts text, so declaring it would assert
nothing. The gate can withhold `image` / `audio` / `video` / `file`, and a
tier can declare any of them — though only image *conversion* is implemented
across providers today, so the others are declarable ahead of the converters
that will fill them.

##### Direction

Each role carries a direction. The list form is sugar for the common case:

```yaml
modalities: [image]                 # ≡ {image: inbound}
modalities: {image: inbound}        # the tier can ACCEPT images
modalities: {audio: outbound}       # the tier can EMIT audio
modalities: {audio: bidirectional}  # both
```

`bidirectional`, not `both` — "both" says nothing about what it is both *of*,
and doesn't parallel `inbound`/`outbound`. Not `duplex` either: that connotes
simultaneity, and a tier declares capability, not concurrency. Writing any of
`both` / `duplex` / `inout` produces an error that names `bidirectional`.

Internally a role is stored as **two sets** (`inbound_modalities` /
`outbound_modalities`) rather than a map, because every consumer asks a
directional question; `bidirectional` simply lands the role in both.

> **Outbound roles parse but are INERT today.** No adapter parses
> model-generated media and the streaming callback is text-only, so nothing
> can deliver what an outbound role promises. `jaato-scaffold validate` emits
> a **warning** (not an error) saying so — the declaration is accepted
> deliberately, so a profile can be written ahead of that work landing. The
> startup capability check verifies an outbound role only against a provider
> implementing `supports_output_modality`; none do yet, so it is skipped
> rather than failing falsely. See
> [Binary Media Chunks](design/binary-media-chunks.md).

##### Why override `description`

The framework knows what a tier is *called*, not what your deployment means
by it. Its default prose for `executor` says "mechanical tool calls … when
the work doesn't need reasoning" — accurate as a generic gloss, useless if
your `executor` tier is specifically "apply the migration plan verbatim,
never re-plan". The `description` key is how a profile says so.

It is read **once**, when the tool schema is built at session configure time.
The tool block sits in the prompt-cache prefix, so this must be stable for
the life of a session — which is why a `budget_control` degrade rung may
**not** set one (a rung rebinds a tier's model, never its role; declaring a
description in an overlay is a config error).

#### Reserved Control Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `initial` | `string` | `"dispatcher"` | Which tier to start in when the session begins. Must be a declared tier name. |
| `fallback` | `string` | `"dispatcher"` | Which tier to route to when `enter_tier` references a tier that isn't declared. Must be a declared tier name. |

These keys are **unambiguous** because they are never valid tier names — the parser splits on `VALID_TIER_NAMES` membership (`planner`, `dispatcher`, `executor`, `vision`).

### 5.3 Resolution Order

The framework resolves tier config in this priority order:

```
1. Profile `model_tiers` (non-empty) → profile wins, env vars ignored
2. Profile lacks `model_tiers` (or no profile) → check env vars
3. Neither set → None → single-model mode (no enter_tier tool)
```

#### Environment Variable Fallback

When no profile-level tiers are declared, the framework checks these env vars:

| Env Var | Maps To |
|---|---|
| `JAATO_TIER_PLANNER` | `planner` tier model |
| `JAATO_TIER_DISPATCHER` | `dispatcher` tier model |
| `JAATO_TIER_EXECUTOR` | `executor` tier model |
| `JAATO_TIER_INITIAL` | `initial` tier name |
| `JAATO_TIER_FALLBACK` | `fallback` tier name |

Env vars only support the **shorthand** form (model name string). If at least one `JAATO_TIER_*` model var is set, a `ModelTierConfig` is built from them. Blank-only values are silently skipped.

```bash
# Quick experimentation without editing profiles:
export JAATO_TIER_PLANNER=claude-opus-4-7
export JAATO_TIER_DISPATCHER=claude-sonnet-4-6
export JAATO_TIER_EXECUTOR=claude-haiku-4-5
```

### 5.4 Interaction with `model` Field

When a profile declares **both** `model` and a non-empty `model_tiers`, the `model` field is **silently ignored** and a warning is logged:

```
Profile 'my_agent' declares both 'model' and 'model_tiers';
'model' will be ignored — the active model is selected
per turn from 'model_tiers[<active_tier>]'.
```

**Rationale**: `model` selects one model for the entire session; `model_tiers` selects per-turn. They're mutually exclusive concepts. The profile author should remove `model` when `model_tiers` is set.

When `model_tiers` is empty (or absent), the session falls back to the standard model resolution chain: `model` → `SubagentConfig.default_model` → parent session's model.

### 5.5 Cross-Provider Tiers

Tiers may declare **different** providers. The historical V1 same-provider
gate (`_validate_same_provider_v1`) is gone:

```json
{
  "model_tiers": {
    "executor": {"model": "glm-4.6", "provider": "zhipuai"},
    "vision":   {"model": "google/gemini-2.5-flash-lite", "provider": "openrouter"},
    "initial": "executor",
    "fallback": "executor"
  }
}
```

**How the swap works**: when the entered tier names a provider other than the
active one, `JaatoSession.switch_tier` swaps `self._provider` to a per-tier
instance cached by `_provider_for_tier`. Conversation history is
provider-neutral, so it flows across the swap untouched, and switching back is
a cache hit (O(1), no re-create). Each tier's provider reads its **own**
`plugin_configs` section, so the OpenRouter tier above picks up
`plugin_configs.openrouter.api_key`.

A tier that leaves `provider` unset uses the session's main provider, which
switches model in place via `provider.connect(model, skip_model_test=True)` —
no swap path is taken, so single-provider ladders behave exactly as before.

**Allowed patterns**:
- All tiers as shorthand (no provider): ✅ — session's provider handles all
- Mix of shorthand and rich with the same provider: ✅
- Tiers with `provider: null` mixed with unset providers: ✅ (treated as consistent)
- Tiers naming **different** providers: ✅ (per-tier provider instances)

**Caveat — cost of a cross-provider vision tier**: the startup capability
check (`_validate_vision_tier_capability`) only fail-fast validates a `vision`
tier that lives on the *active* provider. Validating one on another provider
would eagerly create that provider on turn 1 even if vision is never entered,
so such tiers are validated lazily on first entry, plus by the content gate.

### 5.6 The `enter_tier` Lifecycle Tool

When tier mode is active (a non-null `ModelTierConfig` is resolved), the framework registers an `enter_tier` tool that the model can call to switch tiers mid-conversation.

#### Tool Schema

```json
{
  "name": "enter_tier",
  "description": "Switch the session's active model tier.  Pick the one that matches what you're about to do:\n\n* `planner` — ...\n* `executor` — ...\n\n  This session starts in `dispatcher`.\n...",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "enum": ["planner", "executor"]
      }
    },
    "required": ["name"]
  }
}
```

Both the `enum` and the description bullets are built from the tiers **this
profile declares** — not from `VALID_TIER_NAMES`. A profile declaring only
`planner` and `executor` produces exactly the schema above: `dispatcher` and
`vision` are never advertised, so the model can't ask for a tier that would
silently route to `fallback`. Each bullet's prose is that tier's
`description` when set, else the framework's default wording for the name.
Tier order is canonical (`planner`, `dispatcher`, `executor`, `vision`), not
set-iteration order, so the schema is byte-stable across processes — it lives
in the prompt-cache prefix.

The `enum` constraint means providers that enforce tool params at sampling time (Anthropic, Google, OpenAI) reject invalid tier names before they reach the executor. The executor still validates against the full `VALID_TIER_NAMES` as defence-in-depth for providers that don't enforce enums; a valid-but-undeclared name routes to `fallback` and reports `status: "fallback_used"`.

#### Tool Properties

- **Auto-approved**: `enter_tier` is in `LifecycleTools.get_auto_approved_tools()`. No permission prompt needed — the model switches tiers freely.
- **Conditionally registered**: Only present when `_tier_config is not None`. Single-model sessions never see this tool (no protocol noise, full backwards compat).
- **Cheap to call**: Switching calls `provider.connect(new_model, skip_model_test=True)` — no network round-trip, just sets the model name on the provider.

#### Tool Result Shapes

| Status | When | Result |
|---|---|---|
| `"switched"` | Tier changed successfully | `{status, active_tier, model}` |
| `"already_at_tier"` | Requested tier == current tier (idempotent no-op) | `{status, active_tier, requested_tier, model}` |
| `"fallback_used"` | Requested tier not declared, routed to fallback | `{status, active_tier, requested_tier, model}` |
| `"invalid_argument"` | Missing or empty `name` | `{error, message}` |
| `"invalid_tier"` | `name` not in `{"planner", "dispatcher", "executor"}` | `{error, message}` |
| `"tier_mode_inactive"` | Session not in tier mode (defence in depth) | `{error, message}` |
| `"switch_failed"` | `provider.connect()` raised an exception | `{error, message}` |

#### Fallback Routing

When `enter_tier("planner")` is called but the profile only declared `dispatcher` and `executor`, the request routes to `tier_fallback` (default: `dispatcher`). The result tells the model what happened:

```json
{
  "status": "fallback_used",
  "active_tier": "dispatcher",
  "requested_tier": "planner",
  "model": "claude-sonnet-4-6"
}
```

If the fallback happens to be the same as the current tier, the status is `"already_at_tier"` (not `"fallback_used"`) — the model still sees the `requested_tier` field so it knows its original ask was rerouted.

### 5.7 Session Initialization in Tier Mode

When `configure(tier_config=...)` is called on a `JaatoSession`:

1. `_tier_config` is set to the resolved config
2. `_active_tier` is set to `tier_config.initial_tier` (default: `"dispatcher"`)
3. `_model_name` is **overridden** to the initial tier's model — the provider connects to this model from turn 0
4. The `enter_tier` tool is registered (via `LifecycleTools`)
5. The system prompt is dynamically augmented with a tier identity line

```python
# Session init in core.py (line ~1356):
if tier_config is not None:
    self._tier_config = tier_config
    self._active_tier = tier_config.initial_tier
    initial_model = tier_config.tiers[tier_config.initial_tier].model
    self._model_name = initial_model  # Overrides whatever was passed
```

### 5.8 System Prompt Augmentation

When tier mode is active, `_get_effective_system_instruction()` appends a dynamic line to the assembled system instruction:

```
You are currently operating in the `dispatcher` tier.
```

This line is **recomputed dynamically** on each access — not stored on `_system_instruction`. This means:

- Tier switches take effect immediately (no prompt re-assembly needed)
- The assembled `_system_instruction` stays stable (important for providers that key prompt cache on it)
- When tier mode is not active, no augmentation is added

### 5.9 Tier Switch Mechanics

`JaatoSession.switch_tier(requested_tier)` is the internal method called by the `enter_tier` tool:

```
switch_tier("planner")
    ├── _tier_config.model_for("planner") → (actual_tier, TierEntry)
    │   └── If "planner" not in tiers → routes to tier_fallback
    ├── If actual_tier == _active_tier → return "already_at_tier"
    ├── provider.connect(entry.model, skip_model_test=True)
    │   └── No network round-trip; sets _model_name on provider
    ├── _active_tier = actual_tier
    ├── _model_name = entry.model
    └── Return {status, active_tier, requested_tier, model}
```

### 5.10 Current Limitations

| Limitation | Details |
|---|---|
| **Subagent tier wiring incomplete** | As of this writing, the subagent plugin's `create_session()` call in `plugin.py` does not pass `tier_config`. Tiers currently work for main sessions created via `core.py._build_profile_session_kwargs()`, but not for spawned subagents. |
| **Inherited as a unit** | `model_tiers` IS handled in `_merge_profiles()` (scalar-override): a child declaring any tiers replaces the parent's whole ladder rather than merging entry by entry. A child declaring none inherits the parent's. |
| **Env vars support shorthand only** | The `JAATO_TIER_*` vars take a model name only — no per-tier `provider` or `description`, and there is no `JAATO_TIER_VISION`. Anything past a single-provider three-tier ladder needs a profile. |

### 5.11 Full Examples

#### Minimal Two-Tier Config (cost optimization)

```json
{
  "name": "cost_optimizer",
  "description": "Uses dispatcher for reasoning, executor for tool calls",
  "model_tiers": {
    "dispatcher": "claude-sonnet-4-6",
    "executor": "claude-haiku-4-5",
    "initial": "dispatcher",
    "fallback": "dispatcher"
  }
}
```

#### Three-Tier with Explicit Provider

```json
{
  "name": "full_tier_agent",
  "description": "Full three-tier setup with Anthropic",
  "provider": "anthropic",
  "model_tiers": {
    "planner": "claude-opus-4-7",
    "dispatcher": "claude-sonnet-4-6",
    "executor": "claude-haiku-4-5",
    "initial": "dispatcher",
    "fallback": "dispatcher"
  }
}
```

#### Start in Planner Tier

```json
{
  "name": "planner_first",
  "description": "Starts in planner for initial analysis",
  "model_tiers": {
    "planner": "claude-opus-4-7",
    "dispatcher": "claude-sonnet-4-6",
    "initial": "planner",
    "fallback": "dispatcher"
  }
}
```

#### Partial Tier Config (only dispatcher + executor)

```json
{
  "name": "two_tier",
  "model_tiers": {
    "dispatcher": "claude-sonnet-4-6",
    "executor": "claude-haiku-4-5",
    "fallback": "dispatcher"
  }
}
```

If the model calls `enter_tier("planner")`, it gets routed to `dispatcher` (the fallback) with status `"fallback_used"`.

#### Env-Var-Only Configuration (no profile edit)

```bash
export JAATO_TIER_PLANNER=claude-opus-4-7
export JAATO_TIER_DISPATCHER=claude-sonnet-4-6
export JAATO_TIER_EXECUTOR=claude-haiku-4-5
export JAATO_TIER_INITIAL=dispatcher
```

This activates tier mode for the main session even without a profile declaring `model_tiers`. Profile-level tiers always win over env vars.

### 5.12 ModelTierConfig Data Model

The resolved tier config is represented by `ModelTierConfig` (frozen dataclass in `shared/model_tiers.py`):

```python
@dataclass(frozen=True)
class ModelTierConfig:
    tiers: Dict[str, TierEntry]           # tier name → entry
    initial_tier: str = "dispatcher"      # must be in tiers
    tier_fallback: str = "dispatcher"     # must be in tiers

@dataclass(frozen=True)
class TierEntry:
    model: str                  # e.g. "claude-opus-4-7"
    provider: Optional[str]     # None = the session's main provider
    description: Optional[str]  # this tier's bullet in the enter_tier tool
    inbound_modalities: FrozenSet[str]   # roles the tier can ACCEPT
    outbound_modalities: FrozenSet[str]  # roles the tier can EMIT (inert today)
```

**Key methods**:

| Method | Description |
|---|---|
| `ModelTierConfig.from_unified_dict(raw)` | Parse from the profile JSON dict shape |
| `ModelTierConfig.from_env(env=None)` | Build from env vars; returns `None` if no tier vars set |
| `ModelTierConfig.resolve(profile_model_tiers, env=None)` | Priority: profile → env → None |
| `config.model_for(tier_name)` | Resolve to `(actual_tier, TierEntry)` with fallback routing |

### 5.13 Validation Reference

All validation runs at construction time (`__post_init__`). Invalid configs raise `ModelTierConfigError` (subclass of `ValueError`):

| Rule | Error Message Pattern |
|---|---|
| At least one tier mapping required | `"at least one tier mapping"` |
| Reserved keys alone aren't enough | `"at least one tier mapping"` |
| Unknown tier names | `"unknown tier names: [...]"` |
| `initial` must be in declared tiers | `"initial_tier '...' not in declared"` |
| `fallback` must be in declared tiers | `"tier_fallback '...' not in declared"` |
| `initial` must be a string | `"model_tiers.'initial' must be a string"` |
| `fallback` must be a string | `"model_tiers.'fallback' must be a string"` |
| Tier model must be non-empty string | `"tier '...': 'model' must be a non-empty string"` |
| Tier provider must be non-empty when set | `"tier '...': 'provider' must be a non-empty string"` |
| Tier description must be non-empty when set | `"tier '...': 'description' must be a non-empty string"` |
| Modality must be a known kind | `"tier '...': '...' is not a modality"` |
| `text` may not be declared as a role | `"'modalities' may not list 'text'"` |
| Direction must be known | `"tier '...': '...' is not a modality direction"` |
| `modalities` must be a list or a map | `"'modalities' must be a list ... or a map"` |
| Invalid tier value type | `"tier '...': expected str or dict, got ..."` |

---

## 6. Variable Expansion and Secret Resolution

Profile values support two-phase expansion:

### Phase 1: Variable Substitution

`${VAR_NAME}` patterns are replaced from context variables, then environment variables.

| Variable | Source | Example |
|---|---|---|
| `${workspaceRoot}` | Auto-detected from `.git` or `.jaato` directory | `/home/user/project` |
| `${cwd}` | Current working directory | `/home/user/project` |
| `${HOME}` | Environment variable | `/home/user` |
| `${USER}` | Environment variable | `user` |
| `${projectPath}` | Context variable (passed by caller) | `/app/my-project` |
| `${ANY_ENV_VAR}` | `os.environ` lookup | (any env var) |

**Undefined variables** are kept as-is (literal `${UNKNOWN}` stays in the string).

Expansion works recursively in dicts and lists:
```json
{
  "plugin_configs": {
    "lsp": { "config_path": "${projectPath}/.lsp.json" }
  },
  "env": {
    "PROJECT_ROOT": "${workspaceRoot}",
    "OUTPUT": "${PROJECT_ROOT}/generated"
  }
}
```

### Phase 2: Secret URI Resolution

If the fully-expanded string matches `scheme://path[#key]` and a `SecretResolver` is registered for that scheme, the value is resolved to its plaintext secret.

```json
{
  "env": {
    "DB_PASSWORD": "vault://secret/myapp#db_password",
    "API_KEY": "awssm:///myapp/api-key"
  }
}
```

**Rules:**
- Only applies when the **entire** string is a URI (not embedded in a larger string)
- Resolvers are discovered via the `jaato.premium` → `secret_resolvers` entry point
- If no resolver is registered for the scheme, a warning is logged and the literal URI is used
- Failed resolution raises `SecretResolutionError`

---

## 7. Plugin Preloading

Plugins in the `plugins` list can include a `(preload)` annotation:

```json
{
  "plugins": ["file_edit", "cli", "template(preload)", "grep_content"]
}
```

- **Without `(preload)`**: The plugin is active, but its discoverable tools are loaded lazily.
- **With `(preload)`**: ALL of the plugin's tools are loaded into the initial context immediately.

**Syntax**: `"plugin_name(preload)"` or `"plugin_name (preload)"` (optional space before parenthesis).

---

## 8. Embedding Plugin Configs into Profiles

The `plugin_configs` field is a generic mechanism for passing configuration to **any** plugin at subagent session creation time.

### How It Works

```
Profile JSON:  plugin_configs["permission"] = { "policy": { ... } }
    ↓
create_session() iterates plugin_configs
    ↓
registry.expose_tool("permission", config)
    ↓
PermissionPlugin.initialize(config)
```

### Permission Policy Embedding

```json
{
  "plugin_configs": {
    "permission": {
      "policy": {
        "defaultPolicy": "allow",
        "whitelist": { "tools": ["*"], "patterns": [".*"] },
        "blacklist": { "tools": [], "patterns": [] }
      }
    }
  }
}
```

### Provider-Specific Knobs via `plugin_configs`

The model provider is itself a plugin, so provider-specific knobs go under `plugin_configs[provider_name]`:

#### Zhipu AI (zhipuai) Knobs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_thinking` | `bool` | `false` | Enable extended chain-of-thought reasoning |
| `thinking_budget` | `int` | `10000` | Max thinking tokens per request |
| `context_length` | `int` | (model default) | Override context window size |
| `base_url` | `string` | `https://api.z.ai/api/anthropic` | Override API base URL |

#### Anthropic (anthropic) Knobs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_thinking` | `bool` | `false` | Enable Claude extended thinking |
| `thinking_budget` | `int` | `10000` | Max thinking tokens |

#### Google GenAI (google_genai) Knobs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `location` | `string` | `""` | Vertex AI region (e.g. `us-central1`) |

---

## 9. Profile Inheritance

Profiles can inherit from other profiles using the `inherits` field:

```json
{
  "name": "code_reviewer_security",
  "inherits": ["code_reviewer", "security_scanner"],
  "plugins": ["web_search"],
  "max_turns": 15
}
```

### Merge Semantics

| Merge Type | Fields | Behavior |
|---|---|---|
| **Collection (union)** | `plugins`, `preloaded_plugins`, `env`, `plugin_configs` | Parents first (in order), then child. Deduplicated. |
| **Scalar (agreement-or-override)** | `model`, `provider`, `max_turns`, `gc`, `runtime_limits`, `completion_payload_schema` | Parents must agree. If they conflict, child MUST override. |
| **Concatenation** | `system_instructions` | Grandparent → parent → child, joined with double newlines. |
| **Never inherited** | `name`, `description`, `model_tiers` | Always from the child profile. |

> **Note**: `model_tiers` is **not** inherited across profiles. Each profile that needs per-turn switching must declare its own `model_tiers` dict. This is because tier configs are tightly coupled to the specific models available to a given provider setup, and merging tier dicts from different parent profiles would produce ambiguous or invalid configurations.

### max_turns Special Case

`max_turns` uses the **most restrictive** (minimum) value across parents, unless the child explicitly overrides.

---

## 10. Inline vs Profile-Based Delegation

### Profile-Based (Pre-defined)

```python
delegate(profile="code_reviewer", task="Review the login module")
```

### Inline (Ad-Hoc)

```python
delegate(
    task="Search for security vulnerabilities",
    plugins=["grep_content", "file_edit", "web_search"],
    system_instructions="Focus on OWASP top 10 vulnerabilities",
    max_turns=5
)
```

### Comparison

| Aspect | Profile-Based | Inline |
|---|---|---|
| Configuration | Pre-defined, version-controlled | Ad-hoc per request |
| Tools available | Profile's `plugins` list | `inline_allowed_plugins` whitelist |
| Model override | Per-profile `model`/`provider` | Inherits parent's model |
| Tier switching | Per-profile `model_tiers` | Not available (no profile) |
| GC strategy | Per-profile `gc` config | Parent's default |
| Inheritance | Supported | Not applicable |

---

## 11. SubagentConfig — Top-Level Plugin Configuration

```json
{
  "project": "my-gcp-project",
  "location": "us-central1",
  "default_model": "gemini-2.5-flash",
  "default_provider": "google_genai",
  "allow_inline": true,
  "inline_allowed_plugins": ["grep_content", "file_edit", "web_search"],
  "auto_discover_profiles": true,
  "profiles_dir": ".jaato/profiles",
  "profiles": {}
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `project` | `string` | `""` | GCP project ID for Vertex AI. |
| `location` | `string` | `""` | Vertex AI region. |
| `default_model` | `string \| null` | `null` | Default model for subagents. `null` = inherit from parent. |
| `default_provider` | `string \| null` | `null` | Default provider. Must match `default_model`'s provider if set. |
| `allow_inline` | `bool` | `true` | Whether inline subagent creation is permitted. |
| `inline_allowed_plugins` | `string[]` | `[]` | Plugins available for inline delegation. |
| `auto_discover_profiles` | `bool` | `true` | Whether to scan `profiles_dir` for profile files at startup. |
| `profiles_dir` | `string` | `".jaato/profiles"` | Directory to scan for profile files. |
| `profiles` | `object` | `{}` | Inline profile definitions (alternative to file-based). |

---

## 12. The Delegation Lifecycle

When the parent agent calls `delegate(profile="code_reviewer", task="Review PR #42")`:

```
1. RESOLVE PROFILE
2. DETERMINE MODEL + PROVIDER
   profile.model → profile.provider → SubagentConfig defaults → parent defaults
   (If model_tiers is non-empty, model is ignored — tiers are resolved separately)
3. ENSURE CORE PLUGINS
4. EXPAND VARIABLES
5. RESOLVE INHERITANCE
6. CREATE SESSION (tier_config not yet passed for subagents — see §5.10)
7. CONFIGURE GC
8. SET ENVIRONMENT
9. WIRE PERMISSIONS
10. EMIT UI EVENTS
11. RUN CHAT LOOP
12. EMIT COMPLETION
13. RETURN RESULT
```

---

## 13. Cross-Provider Subagents

A subagent can use a different AI provider than its parent via the `provider` field. When cross-provider, the runtime creates a new provider instance with its own SDK client.

---

## 14. Permission Bridging

Subagents share the parent's `PermissionPlugin` but use a `ParentBridgedChannel` for user interaction. Whitelist/blacklist/suspension decisions apply uniformly across parent and subagents.

---

## 15. Resource Sharing vs Isolation

### Shared (Runtime Level)

| Resource | Why Shared |
|---|---|
| PluginRegistry | Single discovery, consistent tool schemas |
| PermissionPlugin | Unified approval state |
| TokenLedger | Aggregated billing |
| TelemetryPlugin | Correlated traces |
| Provider configs | No redundant auth/connection setup |

### Isolated (Per Session)

| Resource | Why Isolated |
|---|---|
| Conversation history | Each agent has its own context |
| Provider instance | Independent SDK client per session |
| Model selection | Subagent can use different model |
| Tool subset | Profile defines available plugins |
| GC plugin + config | Different collection strategy per agent |
| Tier config (`_tier_config`, `_active_tier`) | Each session manages its own tier state |
| CancelToken | Independent cancellation per session |
| Environment variables | Profile-scoped, never leaks |

---

## 16. SubagentResult

When a subagent completes, it returns a structured result:

```json
{
  "success": true,
  "response": "I found 3 issues...",
  "turns_used": 4,
  "token_usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 380,
    "total_tokens": 1630
  },
  "agent_id": "session_abc123",
  "output_streamed": false
}
```

---

## 17. Profile Validation

Profiles are validated by `validate_profile(data)` which checks required fields, type constraints, GC sub-validation, plugin format, env format, inherits format, and runtime_limits (via `RuntimeLimits.from_dict()`).

Returns `(is_valid: bool, errors: list[str], warnings: list[str])`.

---

## 18. gen-references Pipeline — Auto-Generating Profiles

The `gen-references` agent pipeline automatically generates profiles from a knowledge base. The pipeline generates profile + agent pairs, validates each profile with `validateProfile` before writing.

---

## 19. Reference JSON Format

References are catalog entries validated by `validateReference(path="...")`. See the existing reference JSON at `.jaato/references/jaato-subagent-profiles.json` for the canonical example.

---

## 20. Source Code Map

| File | Contents |
|---|---|
| `jaato-server/shared/plugins/subagent/config.py` | `SubagentProfile`, `GCProfileConfig`, `SubagentConfig`, `SubagentResult` dataclasses; `validate_profile()`, `discover_profiles()`, `resolve_profiles()`, `_merge_profiles()`, `expand_variables()`, `gc_profile_to_plugin_config()` |
| `jaato-server/shared/model_tiers.py` | `ModelTierConfig`, `TierEntry`, `ModelTierConfigError`; `from_unified_dict()`, `from_env()`, `resolve()`, `model_for()`, `ordered_tier_names()`, `describe_tier()`; `TIER_ORDER` / `DEFAULT_TIER_DESCRIPTIONS` |
| `jaato-server/shared/plugins/subagent/plugin.py` | `_execute_spawn_subagent()`, `_execute_validate_profile()`, tool registration, UI hooks |
| `jaato-server/shared/lifecycle_tools.py` | `LifecycleTools` — `enter_tier` tool schema/executor, `signal_completion` rewrite, `get_tool_schemas()`, `get_auto_approved_tools()` |
| `jaato-server/shared/jaato_session.py` | `JaatoSession` — `configure(tier_config=...)`, `switch_tier()`, `_get_effective_system_instruction()` (dynamic tier line) |
| `jaato-server/shared/jaato_runtime.py` | `JaatoRuntime.create_session(tier_config=...)` |
| `jaato-server/server/core.py` | `_build_profile_session_kwargs()` — resolves `ModelTierConfig.resolve()` for main sessions |
| `jaato-server/shared/runtime_limits.py` | `RuntimeLimits` frozen dataclass |
| `jaato-server/shared/completion_schema_loader.py` | `resolve_completion_schema()` |
| `jaato-server/shared/plugins/subagent/tests/test_profile_inheritance.py` | Inheritance merge semantics tests |
| `jaato-server/shared/plugins/subagent/tests/test_validate_profile.py` | Validation rule tests |
| `jaato-server/shared/tests/test_model_tiers.py` | `ModelTierConfig` validation, resolution, `JaatoSession` tier mode, `LifecycleTools.enter_tier` tests |

---

## 21. Runtime Internals — What the Source Code Reveals

### 21.1 The Actual Spawn Function

The delegation entry point is `_execute_spawn_subagent(args)` in `plugin.py`. It returns **immediately** with `{success: true, subagent_id: "..."}`. The subagent runs asynchronously in a thread pool.

### 21.2 Model/Provider Resolution — The Actual Chain

```python
# Model resolution:
model = profile.model                          # 1. Profile override
    or self._config.default_model               # 2. SubagentConfig default
    or self._parent_session._model_name         # 3. Parent session's model

# Provider resolution:
provider = profile.provider
    or self._config.default_provider
    or self._parent_session._provider_name_override
```

When `model_tiers` is non-empty, `profile.model` is silently ignored (warning logged at load time). The tier config is resolved separately and overrides the model at session init time.

### 21.3 Self-Spawn Loop Prevention

The runtime rejects spawning the same profile the current agent was created from.

### 21.4 Inline Profile Creation

When `profile` is not specified, the runtime creates an inline profile from the parent's plugin list. Inline subagents do **not** support `model_tiers` (no profile to declare them from).

### 21.5 GC Inheritance — Parent Fallback

When a profile has no `gc` config, the subagent **inherits from the parent** — a fresh plugin instance of the same type.

### 21.6 Environment Variable Lifecycle

Profile `env` variables are applied with proper save/restore semantics in the subagent thread.

### 21.7 Cancel Token Propagation

Subagents inherit the parent's cancel token. When the user cancels the parent, cancellation propagates to all children.

### 21.8 Tier Config Wiring — Main Session vs Subagent

**Main session** (via `core.py._build_profile_session_kwargs()`):
```python
tier_config = ModelTierConfig.resolve(
    profile_model_tiers=self._profile.model_tiers if self._profile else None,
)
# Passed to runtime.create_session(tier_config=tier_config)
```

**Subagent session** (via `plugin.py._run_subagent_async()`):
```python
session = self._runtime.create_session(
    model=model,
    tools=profile.plugins,
    ...
    # tier_config is NOT passed — see §5.10 limitation
)
```

### 21.9 Remote Subagent Delegation

The `server` parameter enables cross-machine delegation via jaato-premium.

### 21.10 Output Streaming Architecture

Subagent output flows through a callback chain: `session.send_message()` → `subagent_output_callback` → `ui_hooks.on_agent_output()` → Server EventBus → Client(s).
