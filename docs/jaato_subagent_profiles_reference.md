# JAATO Subagent Profiles — Complete Reference

> **Scope**: Everything a profile JSON file can configure, how profiles are discovered and resolved at runtime, and how the `gen-references` pipeline auto-generates profiles from knowledge bases.

---

## Table of Contents

1. [What Are Subagent Profiles?](#1-what-are-subagent-profiles)
2. [Where Profiles Live — 3-Tier Discovery](#2-where-profiles-live--3-tier-discovery)
3. [Profile JSON Schema — Complete Field Reference](#3-profile-json-schema--complete-field-reference)
4. [GC Configuration (`gc` sub-object)](#4-gc-configuration-gc-sub-object)
5. [Variable Expansion and Secret Resolution](#5-variable-expansion-and-secret-resolution)
6. [Plugin Preloading](#6-plugin-preloading)
7. [Embedding Plugin Configs into Profiles](#7-embedding-plugin-configs-into-profiles)
8. [Profile Inheritance](#8-profile-inheritance)
9. [Inline vs Profile-Based Delegation](#9-inline-vs-profile-based-delegation)
10. [SubagentConfig — Top-Level Plugin Configuration](#10-subagentconfig--top-level-plugin-configuration)
11. [The Delegation Lifecycle](#11-the-delegation-lifecycle)
12. [Cross-Provider Subagents](#12-cross-provider-subagents)
13. [Permission Bridging](#13-permission-bridging)
14. [Resource Sharing vs Isolation](#14-resource-sharing-vs-isolation)
15. [SubagentResult](#15-subagentresult)
16. [Profile Validation](#16-profile-validation)
17. [gen-references Pipeline — Auto-Generating Profiles](#17-gen-references-pipeline--auto-generating-profiles)
18. [Reference JSON Format](#18-reference-json-format)
19. [Source Code Map](#19-source-code-map)

---

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
  }
}
```

### Field Reference Table

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `string` | **Yes** | — | Unique identifier. Used to select the profile during delegation. |
| `description` | `string` | **Yes** | — | Human-readable description. Shown in `list_subagent_profiles` output and UI. |
| `plugins` | `string[]` | No | `[]` | Plugin names to enable. Core plugins (introspection, file_edit) are auto-added. Supports `(preload)` suffix. |
| `plugin_configs` | `object<string, object>` | No | `{}` | Per-plugin configuration overrides. Keys are plugin names, values are config dicts. Supports `${VAR}` expansion. Configs are passed to each plugin's `initialize()` method when the subagent session is created via `registry.expose_tool()`. This allows embedding any plugin's configuration directly into the profile — see [Embedding Plugin Configs](#embedding-plugin-configs-into-profiles). |
| `system_instructions` | `string \| null` | No | `null` | **Deprecated.** Use `.jaato/agents/` instead. When an agent is specified via `--agent`, its rendered markdown replaces this field. |
| `model` | `string \| null` | No | `null` | Model override. `null` = inherit from parent. |
| `provider` | `string \| null` | No | `null` | Provider override (e.g., `"anthropic"`, `"google_genai"`). `null` = inherit from parent. |
| `max_turns` | `integer` | No | `10` | Maximum conversation turns before the subagent returns to the parent. Must be positive. |
| `gc` | `object \| null` | No | `null` | GC configuration. See [Section 4](#4-gc-configuration-gc-sub-object). |
| `env` | `object<string, string>` | No | `{}` | Session-scoped environment variables. Values support `${VAR}` expansion and secret URI resolution. Never leaks to other sessions. |
| `inherits` | `string \| string[]` | No | `null` | Parent profile names for inheritance. Resolved during `discover_profiles()`. After resolution, this field is cleared. |
| `completion_payload_schema` | `object \| string \| null` | No | `null` | JSON Schema constraining `signal_completion` payload. Inline dict or path to `.jaato/completion_schemas/` file. |

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
| `pressure_percent` | `float \| null` | `90.0` | When PRESERVABLE turns can be touched. `0` or `null` enables **continuous mode** (always-on GC). |
| `preserve_recent_turns` | `int` | `5` | Number of recent turns to always preserve during GC. |
| `notify_on_gc` | `bool` | `true` | Inject a notification into history after GC runs. |
| `summarize_middle_turns` | `int \| null` | `null` | For `hybrid` strategy: number of middle turns to summarize. |
| `max_turns` | `int \| null` | `null` | Trigger GC when turn count exceeds this limit. |
| `plugin_config` | `object` | `{}` | Additional plugin-specific configuration passed to the GC plugin. |

### Continuous Mode

Setting `pressure_percent` to `0` or `null` enables continuous mode — GC runs aggressively to keep context as small as possible. Useful for long-running subagents that accumulate lots of tool output.

---

## 5. Variable Expansion and Secret Resolution

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

### Built-in Secret Resolver Schemes

| Scheme | Resolver | Example |
|---|---|---|
| `vault://` | HashiCorp Vault | `vault://secret/myapp#db_password` |
| `awssm://` | AWS Secrets Manager | `awssm:///myapp/api-key` |
| `gcpsm://` | GCP Secret Manager | `gcpsm://projects/myapp/secrets/key` |
| `sops://` | SOPS (encrypted files) | `sops://secrets/db.yaml#password` |
| `keyring://` | OS keyring | `keyring://service/username` |

(Actual available schemes depend on which premium resolvers are installed.)

---

## 6. Plugin Preloading

Plugins in the `plugins` list can include a `(preload)` annotation:

```json
{
  "plugins": ["file_edit", "cli", "template(preload)", "grep_content"]
}
```

- **Without `(preload)`**: The plugin is active, but its discoverable tools are loaded lazily (deferred tool loading) to save initial context space.
- **With `(preload)`**: ALL of the plugin's tools — including discoverable ones — are loaded into the initial context immediately. This is necessary when the subagent needs to discover and use tools that wouldn't otherwise appear in its tool list.

**Syntax**: `"plugin_name(preload)"` or `"plugin_name (preload)"` (optional space before parenthesis).

During parsing, the `(preload)` suffix is stripped and the plugin name is stored in both `plugins` (clean name) and `preloaded_plugins` (set of preloaded names).
---

## 7. Embedding Plugin Configs into Profiles

The `plugin_configs` field is a generic mechanism for passing configuration to **any** plugin at subagent session creation time. When `create_session()` is called, it iterates over `plugin_configs` and calls `registry.expose_tool(plugin_name, config)` for each entry, which invokes the plugin's `initialize()` method with the provided config dict.

This means any plugin that accepts config via `initialize()` can have its configuration embedded directly into a profile — no separate config file needed.

### How It Works

```
Profile JSON:  plugin_configs["permission"] = { "policy": { ... } }
    ↓
create_session() iterates plugin_configs
    ↓
registry.expose_tool("permission", config)
    ↓
PermissionPlugin.initialize(config)
    ↓  reads config["policy"] → creates PermissionPolicy.from_config(policy_dict)
    ↓
Subagent session has its own permission policy
```

### Permission Policy Embedding

The most common use case is embedding a permission policy to control which tools the subagent can execute without prompting:

```json
{
  "name": "permissive_agent",
  "description": "Agent with all tools auto-approved",
  "plugins": ["file_edit", "cli", "grep_content"],
  "plugin_configs": {
    "permission": {
      "policy": {
        "defaultPolicy": "allow",
        "whitelist": {
          "tools": ["*"],
          "patterns": [".*"]
        },
        "blacklist": {
          "tools": [],
          "patterns": []
        }
      }
    }
  }
}
```

The `policy` dict follows the same shape as `.jaato/permissions.json` and is consumed by `PermissionPolicy.from_config()`. Valid values for `defaultPolicy` are `"allow"`, `"deny"`, and `"ask"`.

For a restrictive agent:

```json
{
  "name": "readonly_agent",
  "description": "Agent that can only read files and search",
  "plugins": ["grep_content"],
  "plugin_configs": {
    "permission": {
      "policy": {
        "defaultPolicy": "deny",
        "whitelist": {
          "tools": ["grep_content", "readFile"],
          "patterns": []
        }
      }
    }
  }
}
```

### Other Plugin Configs

The same mechanism works for any plugin. Examples:

```json
{
  "plugin_configs": {
    "lsp": { "config_path": "${projectPath}/.lsp.json" },
    "mcp": { "config_path": "${projectPath}/.mcp.json" },
    "sandbox_manager": { "allowed_paths": ["/tmp"], "block_network": true }
  }
}
```

Each plugin's `initialize(config)` method defines what keys it accepts from the config dict. Consult the individual plugin's source code or documentation for available options.

### Runtime-Injected Config Keys

The runtime automatically injects additional keys into `plugin_configs` that are **not** from the profile JSON:

| Key | Injected Into | Value |
|---|---|---|
| `agent_name` | Every plugin | The profile's display name |
| `base_path` | `template` | Parent's workspace path |
| `_injected_reporter` | `todo` | Plan reporter for coordination |

These cannot be overridden by the profile author.

---

## 8. Profile Inheritance

Profiles can inherit from other profiles using the `inherits` field:

```json
{
  "name": "code_reviewer_security",
  "inherits": ["code_reviewer", "security_scanner"],
  "plugins": ["web_search"],
  "max_turns": 15
}
```

### Inheritance Resolution Process

1. `discover_profiles()` scans all three tiers and collects all profiles
2. `resolve_profiles()` performs topological traversal of the inheritance graph
3. For each profile with `inherits`, parents are resolved first (recursive)
4. Parents' fields are merged with the child's overrides
5. After resolution, `inherits` is cleared (profile is fully flattened)
6. Cycles are detected and reported as errors

### Merge Semantics

| Merge Type | Fields | Behavior |
|---|---|---|
| **Collection (union)** | `plugins`, `preloaded_plugins`, `env`, `plugin_configs` | Parents first (in order), then child. Deduplicated. |
| **Scalar (agreement-or-override)** | `model`, `provider`, `max_turns`, `gc`, `completion_payload_schema` | Parents must agree. If they conflict, child MUST override. |
| **Concatenation** | `system_instructions` | Grandparent → parent → child, joined with double newlines. |
| **Never inherited** | `name`, `description` | Always from the child profile. |

### max_turns Special Case

`max_turns` uses the **most restrictive** (minimum) value across parents, unless the child explicitly overrides:

```json
// parent A: max_turns = 5
// parent B: max_turns = 8
// child (no override) → max_turns = 5 (minimum)
// child (override: 20) → max_turns = 20
```

### Conflict Errors

When parents disagree on a scalar field and the child doesn't override, resolution fails with a detailed conflict message:
```
Profile 'child' inherits from ['parent_a', 'parent_b'].
Conflicts between parents (child must override):
  model: 'parent_a': 'claude-sonnet-4-5', 'parent_b': 'gemini-2.5-flash'
```

### plugin_configs Deep Merge

`plugin_configs` uses per-key, per-field merge:
- If parent A sets `plugin_configs.lsp.config_path = "/a"` and parent B sets `plugin_configs.lsp.config_path = "/b"` → conflict unless child overrides
- If parent A sets `plugin_configs.lsp.config_path` and parent B sets `plugin_configs.mcp.config_path` → no conflict (different plugins)
- Child always wins on conflicts

---

## 9. Inline vs Profile-Based Delegation

### Profile-Based (Pre-defined)

```python
delegate(profile="code_reviewer", task="Review the login module")
```

Uses a pre-defined `SubagentProfile` from `.jaato/profiles/`. The profile defines everything: tools, model, instructions, constraints.

### Inline (Ad-Hoc)

```python
delegate(
    task="Search for security vulnerabilities",
    plugins=["grep_content", "file_edit", "web_search"],
    system_instructions="Focus on OWASP top 10 vulnerabilities",
    max_turns=5
)
```

Creates a one-off subagent. Tools are restricted to the `inline_allowed_plugins` whitelist from the `SubagentConfig`. Requires `allow_inline: true`.

### Comparison

| Aspect | Profile-Based | Inline |
|---|---|---|
| Configuration | Pre-defined, version-controlled | Ad-hoc per request |
| Tools available | Profile's `plugins` list | `inline_allowed_plugins` whitelist |
| Auto-approval | Via `plugin_configs.permission.policy` (see [Embedding Plugin Configs](#embedding-plugin-configs-into-profiles)) | Follows default policy |
| Model override | Per-profile `model`/`provider` | Inherits parent's model |
| GC strategy | Per-profile `gc` config | Parent's default |
| Inheritance | Supported | Not applicable |
| Use case | Repeatable specialized tasks | One-off explorations |

---

## 10. SubagentConfig — Top-Level Plugin Configuration

The `SubagentConfig` is the top-level configuration for the subagent plugin itself (not individual profiles). It's typically in `.jaato/subagent.json` or embedded in the server config.

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
  "profiles": {
    "code_reviewer": {
      "description": "Reviews code for quality and security",
      "plugins": ["file_edit", "cli", "grep_content"],
      "max_turns": 5
    }
  }
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

## 11. The Delegation Lifecycle

When the parent agent calls `delegate(profile="code_reviewer", task="Review PR #42")`:

```
1. RESOLVE PROFILE
   Look up SubagentProfile by name
   Or create inline profile from delegate() arguments

2. DETERMINE MODEL + PROVIDER
   profile.model → profile.provider → SubagentConfig defaults → parent defaults

3. ENSURE CORE PLUGINS
   runtime._get_essential_plugins(profile.plugins)
   Adds introspection, file_edit, etc. automatically

4. EXPAND VARIABLES
   expand_variables() on all profile fields
   expand_plugin_configs() on plugin_configs
   Resolve secret URIs (vault://, awssm://, etc.)

5. RESOLVE INHERITANCE
   If profile has inherits, resolve and merge parent profiles
   After resolution, profile is fully flattened

6. CREATE SESSION
   runtime.create_session(
     model=resolved_model,
     provider_name=resolved_provider,
     tools=profile.plugins,
     system_instructions=profile.system_instructions  (or agent prompt)
   )

7. CONFIGURE GC
   gc_profile_to_plugin_config(profile.gc)
   session.set_gc_plugin(gc_plugin, gc_config)

8. SET ENVIRONMENT
   Profile env vars applied to os.environ for the subagent thread
   Restored on exit (never leaks to other sessions)

9. WIRE PERMISSIONS
   ParentBridgedChannel connects subagent → parent
   Permission requests flow through parent's channel

10. EMIT UI EVENTS
    on_agent_created(agent_id, profile, icon_name)

11. RUN CHAT LOOP
    session.send_message(task, on_output=..., max_turns=profile.max_turns)
    Iterates until model stops or max_turns reached

12. EMIT COMPLETION
    on_agent_completed(agent_id, result, token_usage)

13. RETURN RESULT
    SubagentResult(success, response, turns_used, token_usage, agent_id)
```

---

## 12. Cross-Provider Subagents

A subagent can use a different AI provider than its parent:

```json
{
  "name": "fast_search",
  "model": "gemini-2.5-flash",
  "provider": "google_genai"
}
```

When the parent uses Anthropic (Claude) and the subagent profile specifies `google_genai`:
1. The runtime looks up the `google_genai` ProviderConfig
2. Creates a new provider instance with its own SDK client
3. The subagent runs on Google's model while the parent continues on Claude
4. No redundant connections — provider configs are shared, only instances differ

**Use cases:**
- Fast/cheap models for routine subtasks (search, formatting)
- Specialized models for specific capabilities
- Cost optimization across the agent hierarchy

---

## 13. Permission Bridging

Subagents share the parent's `PermissionPlugin` but use a `ParentBridgedChannel` for user interaction:

```
Subagent executes tool → needs permission
    → PermissionPlugin checks (shared whitelist/blacklist/suspension)
    → If must prompt user → ParentBridgedChannel
    → Parent's channel shows: "[subagent:code_reviewer] Allow cli.run?"
    → User decision flows back through the bridge
```

**Key behaviors:**
- Whitelist/blacklist/suspension decisions apply uniformly across parent and subagents
- If user types "all" on parent, subagents also get blanket approval
- Thread-isolated: uses parent's I/O while maintaining subagent's execution context
- Nested subagents bridge through each level

---

## 14. Resource Sharing vs Isolation

### Shared (Runtime Level)

| Resource | Why Shared |
|---|---|
| PluginRegistry | Single discovery, consistent tool schemas |
| PermissionPlugin | Unified approval state |
| TokenLedger | Aggregated billing across all agents |
| TelemetryPlugin | Correlated traces across hierarchy |
| Provider configs | No redundant auth/connection setup |
| Base system instructions | Consistent behavioral rules |

### Isolated (Per Session)

| Resource | Why Isolated |
|---|---|
| Conversation history | Each agent has its own context |
| Provider instance | Independent SDK client per session |
| Model selection | Subagent can use different model |
| Tool subset | Profile defines available plugins |
| System instructions | Profile/agent-specific instructions |
| GC plugin + config | Different collection strategy per agent |
| CancelToken | Independent cancellation per session |
| Environment variables | Profile-scoped, never leaks |

---

## 15. SubagentResult

When a subagent completes, it returns a structured result:

```json
{
  "success": true,
  "response": "I found 3 issues in the login module...",
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

When `output_streamed` is `true` (output was shown to user via UI hooks in real-time), the `response` field is omitted and replaced with a note telling the parent model not to repeat it.

---

## 16. Profile Validation

Profiles are validated by `validate_profile(data)` which checks:

1. **Required fields**: `name` and `description` must be non-empty
2. **Type constraints**: each field must match its expected type
3. **GC sub-validation**: `gc.type` must be valid, numeric ranges (0–100), `preserve_recent_turns` non-negative
4. **Plugin format**: `plugins` must be array of strings
5. **Env format**: `env` must be object with string keys and values
6. **Inherits format**: `inherits` must be string, array of strings, or null

Returns `(is_valid: bool, errors: list[str], warnings: list[str])`.

Validation is available as a tool: `validateProfile(path="...")` reads a file, parses JSON, and runs `validate_profile()`.

---

## 17. gen-references Pipeline — Auto-Generating Profiles

The `gen-references` agent pipeline automatically generates profiles from a knowledge base:

### Phase 4: Profile Generation

For each MODULE or SKILL folder discovered in the knowledge base:

1. **Skill profiles**: One per module/skill folder, with:
   - `plugins` derived from the module's tool requirements
   - `inherits` from any base profiles the module declares
   - `model`/`provider` overrides if specified in frontmatter
   - `gc` config appropriate for the task complexity
   - `completion_payload_schema` if the module defines output contracts

2. **Validator profiles**: Generated for validation tiers (type checking, testing, linting)
   - Typically `inherits` from a base validator profile
   - Scoped `plugins` for the validation tools needed
   - High `max_turns` to accommodate fix-and-retry cycles

3. **Analyst/investigator profiles**: For exploration and analysis tasks
   - Read-only tool sets
   - Budget or hybrid GC for long-running analysis

4. **Each profile is validated** with `validateProfile` before writing
5. **Profiles are written** to `{{profiles_dir}}/` (default: `.jaato/profiles/`)

### Profile ↔ Agent Pairs

The pipeline generates pairs:
- **Agent** (`.jaato/agents/<name>.md`): The instructions/system prompt
- **Profile** (`.jaato/profiles/<name>.json`): The runtime configuration

The agent's YAML frontmatter can declare profile requirements:
```yaml
---
profile: skill-code-001-circuit-breaker
inherits: [base-coder]
model: null  # inherit from parent
plugins: [file_edit, cli, lsp, grep_content]
gc:
  type: budget
  threshold_percent: 85
---
```

---

## 18. Reference JSON Format

References are the catalog entries that the runtime uses to discover knowledge. They are validated by `validateReference(path="...")`.

### Schema

```json
{
  "id": "mod-code-001-circuit-breaker",
  "name": "MOD-001: Circuit Breaker",
  "description": "Implements the circuit breaker resilience pattern",
  "type": "local",
  "path": "/absolute/path/to/folder",
  "mode": "selectable",
  "tags": ["resilience", "circuit-breaker", "java"],
  "fetchHint": "Read MODULE.md first, then patterns/ directory",
  "contents": {
    "templates": "templates/",
    "validation": "validation/",
    "policies": null,
    "scripts": "scripts/"
  },
  "embedding": {
    "index": 0,
    "source_hash": "sha256:a1b2c3d4..."
  },
  "source": {
    "type": "git",
    "url": "https://github.com/org/repo",
    "ref": "main",
    "fetched_at": "2026-04-20T18:00:00Z"
  }
}
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier (folder name) |
| `name` | `string` | Human-readable title |

### Type-Specific Fields

| Type | Required Fields | Description |
|---|---|---|
| `local` | `path` | Content on local disk |
| `url` | `url` | HTTP URL to fetch |
| `mcp` | `server`, `tool` | MCP tool to call |
| `inline` | `content` | Content embedded directly |

### Optional Fields

| Field | Type | Description |
|---|---|---|
| `mode` | `"auto"` \| `"selectable"` | When to offer to model. Default: `"selectable"`. |
| `tags` | `string[]` | Topic-based discovery tags |
| `fetchHint` | `string` | Hint for which file to read first |
| `contents` | `object` | Typed subfolder declarations (templates, validation, policies, scripts) |
| `embedding` | `object` | Links to sidecar `.npy` matrix (index + source_hash) |
| `source` | `object` | Provenance for remote sources (type, url, ref, fetched_at) |

### Embedding Sidecar

Embeddings are stored as a NumPy `.npy` matrix at `references.embeddings.npy`:
- Shape: `(N, D)` where N = number of references, D = embedding dimensions (typically 384 for all-MiniLM-L6-v2)
- Row `i` contains the embedding vector for the reference with `embedding.index == i`
- `embedding.source_hash` is `sha256:` + hex digest of the entry-point file content

---

## 19. Source Code Map

| File | Contents |
|---|---|
| `jaato-server/shared/plugins/subagent/config.py` | `SubagentProfile`, `GCProfileConfig`, `SubagentConfig`, `SubagentResult` dataclasses; `validate_profile()`, `discover_profiles()`, `resolve_profiles()`, `_merge_profiles()`, `expand_variables()`, `expand_plugin_configs()`, `gc_profile_to_plugin_config()`, `parse_plugin_entry()`, `_normalize_inherits()`, secret resolver protocol |
| `jaato-server/shared/plugins/subagent/plugin.py` | `_execute_delegate()`, `_execute_validate_profile()`, `_execute_create_profile()`, `_execute_list_profiles()`, tool registration, UI hooks |
| `jaato-server/shared/plugins/subagent/serializer.py` | Profile serialization/deserialization |
| `jaato-server/shared/plugins/subagent/ui_hooks.py` | `AgentUIHooks` protocol, event definitions |
| `jaato-server/shared/plugins/subagent/tests/test_profile_inheritance.py` | Inheritance merge semantics tests |
| `jaato-server/shared/plugins/subagent/tests/test_validate_profile.py` | Validation rule tests |
| `jaato-server/shared/plugins/subagent/tests/test_profile_discovery.py` | 3-tier discovery tests |
| `jaato-server/shared/plugins/subagent/tests/test_config.py` | Config loading tests |
| `jaato-server/shared/plugins/references/models.py` | `ReferenceSource`, `EmbeddingMetadata`, `ReferenceContents`, `SourceType`, `InjectionMode` |
| `jaato-server/shared/plugins/references/config_loader.py` | `validate_reference_file()`, `validate_config()`, reference loading |
| `docs/jaato_subagent_architecture.md` | Architecture overview (design-level) |
| `~/.jaato/prompts/gen-references.md` | Main gen-references prompt (Phases 0–1.5) |
| `~/.jaato/prompts/gen-references-processing.md` | Processing prompt (Phase 2, reference schemas) |
| `~/.jaato/prompts/gen-references-indexing.md` | Indexing prompt (Phases 3–3b, 4.5 flow generation) |


---

## 19. Runtime Internals — What the Source Code Reveals

> This section covers implementation details and design decisions that are only discoverable by reading `plugin.py` and `config.py` source code. A different agent reading only the schema and architecture docs would miss these.

### 19.1 The Actual Spawn Function: `_execute_spawn_subagent`

The delegation entry point is `_execute_spawn_subagent(args)` in `plugin.py` (line 1971). It accepts these arguments from the model's tool call:

| Arg | Type | Description |
|---|---|---|
| `task` | `string` | **Required.** The task description sent to the subagent. |
| `profile` | `string` | Profile name to look up from `SubagentConfig.profiles`. |
| `agent` | `string` | Agent name (`.jaato/agents/<name>.md`) — overrides `system_instructions`. |
| `agent_params` | `dict` | Variables for the agent's YAML frontmatter `{{params}}`. |
| `context` | `string \| dict` | Additional context. If dict, supports `files`, `findings`, `notes` keys. |
| `inline_config` | `dict` | Inline profile overrides (when no `profile` specified). |
| `name` | `string` | Custom display name for the subagent. |
| `server` | `string` | Remote server name for cross-machine delegation (requires jaato-premium). |

**Key detail**: The function returns **immediately** with `{success: true, subagent_id: "..."}`. The subagent runs asynchronously in a thread pool. The model is instructed to "END YOUR TURN NOW" — real events arrive via the injection queue.

### 19.2 Self-Spawn Loop Prevention

The runtime rejects spawning the same profile that the current agent was created from:

```python
if profile_name and profile_name == self._self_profile_name:
    return SubagentResult(
        success=False,
        error=f"Cannot spawn profile '{profile_name}' — this is your own profile."
    )
```

This prevents infinite recursion where agent A (profile X) spawns agent B (also profile X) which spawns agent C (also profile X)...

### 19.3 Inline Profile Creation (No `profile` Argument)

When `profile` is not specified, the runtime creates an inline profile from:

1. **`self._parent_plugins`** — the parent's plugin list (inherited)
2. **`inline_config`** overrides — `plugins`, `system_instructions`, `max_turns`, `gc`
3. **Plugin validation** — inline plugins are checked against `inline_allowed_plugins` whitelist
4. **Tech stack detection** — `detect_workspace_tech_stack()` scans for `pom.xml`, `pyproject.toml`, etc. and injects a preamble constraining output to the detected stack

```python
# Auto-injected preamble for inline subagents:
"WORKSPACE TECHNOLOGY CONTEXT:
Java project (Maven - pom.xml detected)

You MUST constrain your output to the detected technology stack."
```

This only happens for inline subagents, not profile-based ones.

### 19.4 Agent Override Replaces system_instructions

When both `profile` and `agent` are specified:

```python
if agent_name_arg:
    agent_result = SessionManager._resolve_agent(agent_name_arg, agent_params_arg, parent_cwd)
    profile.system_instructions = agent_result["system_instructions"]
```

The agent's rendered markdown **replaces** the profile's `system_instructions` entirely — they don't concatenate. This is why `system_instructions` on profiles is deprecated.

### 19.5 Model/Provider Resolution — The Actual Chain

The source code reveals a 4-level fallback chain:

```python
# Model resolution (plugin.py lines ~2390-2396):
model = profile.model                          # 1. Profile override
    or self._config.default_model               # 2. SubagentConfig default
    or self._parent_session._model_name         # 3. Parent session's model
                                               # 4. Server default (implicit)

# Provider resolution (same pattern):
provider = profile.provider
    or self._config.default_provider
    or self._parent_session._provider_name_override
```

**Implication**: If you set `default_model` on `SubagentConfig` but the parent uses a different model, profile-based subagents will use the config default, NOT the parent's model. Only subagents with `model: null` (explicit or absent) will inherit the parent's model.

### 19.6 Plugin Config Injection — Invisible Additions

The runtime injects extra keys into `plugin_configs` that the profile author never specified:

```python
for plugin_name in profile.plugins:
    effective_plugin_configs[plugin_name]["agent_name"] = agent_display_name
    if plugin_name == "todo":
        effective_plugin_configs[plugin_name]["_injected_reporter"] = self._plan_reporter
    if plugin_name == "template":
        effective_plugin_configs[plugin_name]["base_path"] = parent_cwd
```

Every plugin gets `agent_name` injected. The `todo` plugin gets the plan reporter. The `template` plugin gets the parent's workspace as `base_path`. These are not documented in the profile schema but affect runtime behavior.

### 19.7 GC Inheritance — Parent Fallback

When a profile has no `gc` config, the runtime **inherits from the parent** rather than going without GC:

```python
if profile.gc:
    gc_plugin, gc_config = gc_profile_to_plugin_config(profile.gc, profile.name)
    session.set_gc_plugin(gc_plugin, gc_config)
elif parent_session._gc_plugin and parent_session._gc_config:
    # Create a FRESH plugin instance of the same type
    inherited_plugin = load_gc_plugin(parent_gc_name, inherited_init_config)
    session.set_gc_plugin(inherited_plugin, parent_gc_config)
```

**Design rationale**: Without GC, a subagent accumulating large tool results hits `ContextLimitError` with no recovery. The parent fallback ensures every subagent always has context management, even if the profile author forgot to configure it.

### 19.8 Environment Variable Lifecycle

Profile `env` variables are applied with proper save/restore semantics:

```python
# Before subagent runs:
_saved_profile_env[key] = os.environ.get(key)  # Save (None if absent)
os.environ[key] = expanded_value                # Set

# After subagent completes (both success and error paths):
if previous is None:
    os.environ.pop(key, None)                   # Remove if wasn't set before
else:
    os.environ[key] = previous                  # Restore original value
```

This runs in a **thread pool**, so the `os.environ` mutations are visible to the subagent thread. The save/restore ensures no leaks to the parent or sibling agents — but there's a race window if two subagents set the same env var key simultaneously.

### 19.9 Workspace Path Resolution — Three Sources

The runtime resolves `parent_cwd` (the workspace path) from three sources in priority order:

```python
workspace_path = self._workspace_path                          # 1. Plugin's stored path
    or self._runtime.registry.get_workspace_path()            # 2. Runtime registry
    or os.environ.get("JAATO_WORKSPACE_ROOT")                # 3. Environment variable
    or os.getcwd()                                           # 4. Fallback
```

This is used for:
- `os.chdir(workspace_path)` — subagent thread changes to parent's workspace
- `${workspaceRoot}` expansion in profile fields
- `base_path` injection into template plugin config
- Tech stack detection

### 19.10 The `send_to_subagent` Follow-Up Pattern

After spawning, the parent can send additional messages via `send_to_subagent(subagent_id, message)`. These are processed by `_execute_send_to_subagent`:

- If the subagent session is **idle** (between turns): the message is sent immediately
- If the subagent session is **busy** (mid-turn): the message is queued for processing after the current turn completes

This enables interactive multi-turn subagent conversations, not just fire-and-forget delegation.

### 19.11 Clarification and Permission Reconfiguration

After session creation, both clarification and permission plugins are reconfigured for subagent mode:

```python
clarification_plugin.configure_for_subagent(session)     # Routes clarification through parent
permission_plugin.configure_for_subagent(session)        # Routes permission requests through parent
```

This is what makes the `ParentBridgedChannel` work — it's not a separate channel object, it's the existing plugins reconfigured to forward through the parent session instead of blocking locally.

### 19.12 Cancel Token Propagation

Subagents inherit the parent's cancel token:

```python
if parent_session._cancel_token:
    session.set_parent_cancel_token(parent_token)
```

When the user cancels the parent (e.g., Ctrl+C), the cancellation propagates to all child subagents automatically.

### 19.13 UI Hooks — What Gets Forwarded

The runtime wires these callbacks to the session:

| Callback | What it forwards |
|---|---|
| `subagent_output_callback` | All model/tool output text |
| `subagent_usage_callback` | Real-time token usage during streaming |
| `_on_running_state_changed` | active ↔ idle status transitions |
| Turn accounting | Per-turn token counts, durations, function calls |
| Context updates | Total tokens, percent used, turns |
| History updates | Full conversation history snapshots |
| GC config notification | Strategy, threshold, continuous mode |

The UI hooks are shared across all subagents — a single `AgentUIHooks` instance receives events from every child agent, tagged with their `agent_id`.

### 19.14 Remote Spawn (Cross-Machine Delegation)

The `server` parameter enables cross-machine delegation:

```python
if server:
    if self._remote_spawn_handler is None:
        return error("Remote delegation requires jaato-premium")
    return self._remote_spawn_handler(server=server, task=task, ...)
```

This requires the `jaato-premium` package and a registered remote spawn handler. The subagent runs on a different machine entirely, with results streamed back. Profile resolution still happens locally before the remote spawn.

### 19.15 Profile `env` Values Are Expanded at Thread Time

Variable expansion in `env` values uses `expand_variables(profile.env, workspace_root_override=workspace_path)`:

```python
expanded_env = expand_variables(profile.env, workspace_root_override=workspace_path)
```

The `workspace_root_override` parameter is critical — it ensures `${workspaceRoot}` resolves to the **parent's** workspace, not whatever `os.getcwd()` happens to be when the subagent thread starts (since `os.chdir` is process-wide and racy).

### 19.16 max_turns Enforcement

The `max_turns` value is stored in the session registry entry:

```python
self._active_sessions[agent_id] = {
    'max_turns': profile.max_turns,
    'turn_count': 0,
    ...
}
```

Turn counting happens after `send_message()` returns, based on the session's `get_turn_accounting()` method. The subagent is NOT forcibly killed mid-turn — it completes its current turn, then the session returns.

### 19.17 Error Handling and Cleanup

On error, the runtime:
1. Forwards the error to the parent via `inject_prompt` (CHILD source type)
2. Removes the session from `_active_sessions`
3. Emits `on_agent_status_changed(status="error")` to UI hooks
4. Clears the per-agent trace context
5. **Restores environment variables** (same save/restore as the success path)

The parent receives the error as an injected prompt, not as a structured result — this means the parent model sees it as context, not as a tool return value.


---

## 19. Runtime Internals — What the Source Code Reveals

> This section covers implementation details and design decisions that can only be learned by reading `plugin.py` and `config.py`. A profile JSON reference alone doesn't capture these.

### 19.1 Actual Delegation Implementation (`_execute_spawn_subagent`)

The tool exposed to the model is called `spawn_subagent` (not `delegate`). The implementation lives in `_execute_spawn_subagent()` in `plugin.py` (~700 lines).

**The call flow is NOT sequential** — spawning is asynchronous:

```
Parent calls spawn_subagent(profile, task, context, ...)
    │
    ├── _execute_spawn_subagent() validates args
    │   ├── Self-spawn loop prevention (reject if profile == self._self_profile_name)
    │   ├── Remote spawn path (if "server" param → requires jaato-premium)
    │   └── Profile resolution or inline creation
    │
    ├── Submits _run_subagent_async() to thread pool executor
    │   └── Returns IMMEDIATELY with { subagent_id, status: "spawned" }
    │       → Parent is told: "END YOUR TURN NOW"
    │
    └── _run_subagent_async() runs in background thread:
        ├── os.chdir(workspace_path) + set JAATO_WORKSPACE_ROOT env var
        ├── Apply profile.env (save previous, set new, restore on exit)
        ├── Resolve model: profile.model → config.default_model → parent session model
        ├── Resolve provider: profile.provider → config.default_provider → parent session provider
        ├── Inject agent_name into every plugin_config (auto-set, not from JSON)
        ├── Inject base_path into template plugin config (auto-set)
        ├── runtime.create_session(model, tools, system_instructions, plugin_configs, provider, preloaded_plugins)
        ├── Restore self._parent_session (create_session overwrites it!)
        ├── Configure clarification plugin for subagent mode (routes to parent)
        ├── Configure permission plugin for subagent mode (ParentBridgedChannel)
        ├── Set parent cancel token (propagates cancellation)
        ├── Wire UI hooks for output forwarding
        ├── Wire running-state callback (auto active/idle transitions)
        ├── Configure GC (profile.gc → inherit from parent → none)
        ├── session.send_message(prompt, callbacks...)
        └── Restore profile.env on exit (both success and error paths)
```

### 19.2 Key Design Decisions (and why they matter)

#### Why API keys are NOT in profiles

`ProviderConfig` objects live in the `JaatoRuntime` registry, shared across all agents. Profiles reference providers by name (`"provider": "google_genai"`) but never contain API keys. This is a **security boundary**: profile JSON files can be committed to git without exposing credentials.

#### Why `system_instructions` is deprecated

The field still works, but agents (`.jaato/agents/*.md`) replace it entirely when specified via the `--agent` or `agent` parameter. The reason: profiles are *runtime config* (tools, model, GC), while agents are *instructions* (behavioral rules). Mixing both in one file conflates concerns and makes it harder to reuse the same profile with different instructions.

#### Why spawning is async (thread pool)

`spawn_subagent` returns immediately with a `subagent_id`. The parent's turn ends. The subagent runs in a background thread. Output is forwarded to the parent via `session._parent_session.inject_prompt()` with `source_type=SourceType.CHILD`. This means:
- The parent doesn't wait — it gets its turn back immediately
- Subagent output arrives as `[SUBAGENT agent_id=xxx event=...]` injected prompts
- The parent must subscribe to events or listen for child source injections to get results
- `send_to_subagent(agent_id, message)` allows the parent to send follow-up messages to idle subagents

#### Self-spawn loop prevention

The plugin stores `_self_profile_name` and rejects `spawn_subagent(profile=self._self_profile_name)` with a clear error. This prevents a profile from spawning itself, which would create an infinite recursion.

#### Why `max_turns` uses minimum for inheritance

Most scalar fields use "agreement-or-override" (parents must agree, or child overrides). But `max_turns` uses **minimum across parents** because it's a **safety constraint** — the most restrictive limit should always apply. If parent A says 5 turns and parent B says 8, a child inheriting both should get 5 (the tighter bound).

#### GC inheritance fallback

If a profile doesn't specify `gc`, the subagent **inherits from the parent session** — a fresh plugin instance of the same type is created so the subagent gets its own GC state while using the same strategy. Without this fallback, a subagent accumulating large tool results would hit `ContextLimitError` with no recovery path.

#### Environment variable save/restore

Profile `env` variables are applied by saving the previous values in `_saved_profile_env`, setting the new values, and restoring on exit. This happens in a `try/finally` pattern covering both success and error paths. The save/restore is per-thread, so concurrent subagents don't interfere.

#### Auto-injected plugin configs

The runtime injects certain configs into every plugin automatically — these are NOT from the profile JSON:
- `agent_name` → set to the display name (for trace logging)
- `base_path` → set to parent's workspace (for template plugin)
- `_injected_reporter` → set for the `todo` plugin (for plan coordination)

### 19.3 Profile Resolution Priority (actual code)

From `_run_subagent_async()`:

```python
# Model resolution
model = profile.model or self._config.default_model
if model is None and self._parent_session:
    model = getattr(self._parent_session, '_model_name', None)

# Provider resolution
provider = profile.provider or self._config.default_provider
if provider is None and self._parent_session:
    provider = getattr(self._parent_session, '_provider_name_override', None)
```

The chain is: **profile field → SubagentConfig default → parent session's active model/provider**. If all three are None, the runtime's default provider is used.

### 19.4 Inline Subagent Creation (no profile)

When `spawn_subagent` is called without a `profile` parameter, the subagent inherits the parent's plugin list. The `inline_config` parameter can override specific properties:

```python
# inline_config can override:
- plugins          → validated against inline_allowed_plugins
- system_instructions
- max_turns
- gc               → parsed into GCProfileConfig
```

For inline subagents, the runtime auto-detects the workspace tech stack (Java, Python, Go, etc.) and prepends a `WORKSPACE TECHNOLOGY CONTEXT` block to the system instructions, constraining the subagent to the detected stack.

### 19.5 Remote Subagent Delegation

The `server` parameter on `spawn_subagent` delegates to a remote peer server instead of running locally:

```python
spawn_subagent(
    server="remote-worker-1",
    task="Analyze this codebase",
    profile="analyst"
)
```

This requires `jaato-premium` and a registered remote spawn handler. The local plugin serializes the request and forwards it; the remote server handles execution and streams results back.

### 19.6 The `context` Parameter Shape

The `context` parameter on `spawn_subagent` has a specific structure that's validated:

```python
# String context (simple)
context = "Here is some background information..."

# Structured context (preferred for file sharing)
context = {
    "files": {
        "src/auth.py": "<file content>",  # MUST be a dict, NOT a list
        "config.yaml": "<file content>"
    },
    "findings": ["Bug found in auth module", "Missing null check"],
    "notes": "Focus on the authentication flow"
}
```

**Common mistake**: passing `files` as a list instead of a dict. The code explicitly validates and rejects this: `"context.files must be a dict mapping file paths to content, not a list."`

### 19.7 What `SubagentConfig` Controls vs What Profiles Control

| Concern | SubagentConfig (server-level) | Profile (per-subagent) |
|---|---|---|
| API keys | ✅ (via ProviderConfig) | ❌ |
| Default model/provider | ✅ (`default_model`, `default_provider`) | ✅ (overrides) |
| Inline delegation | ✅ (`allow_inline`) | ❌ |
| Inline allowed plugins | ✅ (`inline_allowed_plugins`) | ❌ |
| Profile discovery | ✅ (`auto_discover_profiles`, `profiles_dir`) | ❌ |
| Tool selection | ❌ | ✅ (`plugins`) |
| Turn limit | ❌ | ✅ (`max_turns`) |
| GC strategy | ❌ | ✅ (`gc`) |
| Environment variables | ❌ | ✅ (`env`) |
| Inheritance | ❌ | ✅ (`inherits`) |
| System instructions | ❌ | ✅ (deprecated) |
| Completion schema | ❌ | ✅ (`completion_payload_schema`) |

### 19.8 Cancellation Propagation

Subagents inherit the parent's cancel token via `session.set_parent_cancel_token(parent_token)`. When the user cancels the parent session:
1. Parent's cancel token is set
2. Subagent checks parent token before each turn
3. Subagent raises `CancelledError` and exits cleanly
4. Profile env variables are restored even on cancellation

### 19.9 Output Streaming Architecture

Subagent output flows through a callback chain:

```
session.send_message(prompt)
    → model generates response
    → subagent_output_callback(source, text, mode)
        → ui_hooks.on_agent_output(agent_id, source, text, mode)
            → Server EventBus
                → Client(s) receive streamed output
```

When `output_streamed=True` in the result, the parent model is told not to repeat the response since the user already saw it. The `on_output` callback also fires for tool calls (`source="tool"`) and system messages (`source="system"`).

### 19.10 Trace Log Isolation

Each subagent gets isolated trace output via `set_trace_agent_context(agent_id)`. Provider trace writes are routed to per-agent files (e.g., `provider_trace_subagent_1.log`) using Python `ContextVar`, so concurrent subagents don't interfere with each other's traces.
