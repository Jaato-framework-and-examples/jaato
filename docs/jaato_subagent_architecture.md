# JAATO Subagent Architecture

## Executive Summary

JAATO's **subagent architecture** enables an AI agent to spawn specialized child agents that share the parent's runtime infrastructure while maintaining isolated conversation state. The design separates **shared resources** (provider config, plugin registry, permissions, token ledger, telemetry) from **per-agent state** (conversation history, model selection, tool subset), enabling fast, lightweight spawning without redundant connections. Subagents can use different models, different providers, and different tool profiles than the parent, and their permissions flow through a **parent-bridged channel** for unified approval.

---

## Part 1: The Runtime/Session Split

JAATO's architecture is founded on a clean separation between two layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RUNTIME vs SESSION                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  JaatoRuntime (Shared Environment)                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   Shared across ALL agents:                                  │    │
│  │   • ProviderConfigs (multi-provider registry)                │    │
│  │   • PluginRegistry (tool schemas + executors)                │    │
│  │   • PermissionPlugin (whitelist/blacklist/approval)           │    │
│  │   • TokenLedger (aggregated accounting)                      │    │
│  │   • TelemetryPlugin (traces)                                 │    │
│  │   • Base system instructions (.jaato/system_instructions.md) │    │
│  │                                                              │    │
│  └────────────────────────┬─────────────────────────────────────┘    │
│                            │                                          │
│              create_session()                                         │
│                            │                                          │
│           ┌────────────────┼────────────────┐                        │
│           ▼                ▼                ▼                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Session A   │  │  Session B   │  │  Session C   │                 │
│  │  (Main)      │  │  (Subagent)  │  │  (Subagent)  │                 │
│  │             │  │             │  │             │                 │
│  │  History    │  │  History    │  │  History    │                 │
│  │  Provider   │  │  Provider   │  │  Provider   │                 │
│  │  Model      │  │  Model      │  │  Model      │                 │
│  │  Tools      │  │  Tools      │  │  Tools      │                 │
│  │  GC Plugin  │  │  GC Plugin  │  │  GC Plugin  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
│  Each session has ISOLATED:                                          │
│  • Conversation history (messages)                                   │
│  • Provider instance (own SDK client)                                │
│  • Model selection (can differ per session)                          │
│  • Tool subset (profile-defined plugin list)                         │
│  • GC plugin and config (per-session collection strategy)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Design?

| Concern | Single-object design | Runtime/Session split |
|---------|---------------------|----------------------|
| **Spawning** | Full re-initialization | Lightweight `create_session()` |
| **Connections** | Redundant SDK clients | Shared provider configs |
| **Permissions** | Separate approval systems | Unified whitelist/blacklist |
| **Accounting** | Per-agent ledgers | Aggregated across agents |
| **Memory** | Duplicated registry | Single registry, shared schemas |

---

## Part 2: Subagent Profiles

Subagents are defined through **profiles** -- named configurations that specify what tools, model, system instructions, and constraints a subagent has. Profiles are discoverable from JSON/YAML files in `.jaato/profiles/`.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBAGENT PROFILE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SubagentProfile                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  name: "code_reviewer"                                       │    │
│  │  description: "Reviews code changes for quality"             │    │
│  │  plugins: ["file_edit", "cli", "grep"]                       │    │
│  │  model: "gemini-2.5-flash"        ◄── Optional override     │    │
│  │  provider: "google_genai"          ◄── Optional override     │    │
│  │  max_turns: 10                                               │    │
│  │  auto_approved: false                                        │    │
│  │  system_instructions: "You are..."                           │    │
│  │  plugin_configs: {...}             ◄── Per-plugin overrides  │    │
│  │  gc:                                                         │    │
│  │    type: "budget"                  ◄── GC strategy per agent │    │
│  │    threshold_percent: 80.0                                   │    │
│  │    pressure_percent: 0             ◄── Continuous mode       │    │
│  │  icon_name: "code_assistant"                                 │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Profile Discovery

```
discover_profiles(".jaato/profiles")
         │
         ├──► Scan directory for .json / .yaml / .yml files
         ├──► Parse each file into SubagentProfile
         ├──► Name from 'name' field or filename stem
         └──► Return { name → SubagentProfile }
```

### Profile File Example

```json
{
  "name": "code_reviewer",
  "description": "Reviews code for quality, security, and best practices",
  "plugins": ["file_edit", "cli", "grep_content"],
  "model": "gemini-2.5-flash",
  "max_turns": 5,
  "auto_approved": true,
  "system_instructions": "You are a code reviewer. Focus on bugs, security issues, and readability.",
  "gc": {
    "type": "budget",
    "threshold_percent": 80.0,
    "pressure_percent": 0,
    "preserve_recent_turns": 3
  }
}
```

### Variable Expansion

Profile configurations support `${variable}` expansion:

| Variable | Source | Example |
|----------|--------|---------|
| `${workspaceRoot}` | Auto-detected from `.git` | `/home/user/project` |
| `${projectPath}` | Context variable | `/app/my-project` |
| `${cwd}` | Current working directory | `/home/user` |
| `${HOME}` | Environment variable | `/home/user` |
| `${USER}` | Environment variable | `user` |

---

## Part 3: The Delegation Flow

When the parent model calls the `delegate` tool, the subagent plugin orchestrates the full lifecycle:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DELEGATION LIFECYCLE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Parent Model                                                        │
│       │                                                              │
│       │  delegate(profile="code_reviewer", task="Review PR #42")    │
│       │                                                              │
│       ▼                                                              │
│  SubagentPlugin._execute_delegate()                                  │
│       │                                                              │
│       ├──► 1. RESOLVE PROFILE                                        │
│       │    Look up SubagentProfile by name                           │
│       │    Or create inline profile from args                        │
│       │                                                              │
│       ├──► 2. DETERMINE MODEL + PROVIDER                             │
│       │    profile.model → profile.provider → parent defaults        │
│       │                                                              │
│       ├──► 3. ENSURE CORE PLUGINS                                    │
│       │    runtime._get_essential_plugins(profile.plugins)           │
│       │    Adds introspection, file_edit, etc. automatically         │
│       │                                                              │
│       ├──► 4. CREATE SESSION                                         │
│       │    runtime.create_session(                                   │
│       │        model=resolved_model,                                 │
│       │        provider_name=resolved_provider,                      │
│       │        tools=profile.plugins,                                │
│       │        system_instructions=profile.system_instructions       │
│       │    )                                                         │
│       │                                                              │
│       ├──► 5. CONFIGURE GC (if profile has gc config)                │
│       │    gc_plugin, gc_config = gc_profile_to_plugin_config(...)   │
│       │    session.set_gc_plugin(gc_plugin, gc_config)               │
│       │                                                              │
│       ├──► 6. WIRE PERMISSIONS                                       │
│       │    ParentBridgedChannel connects subagent → parent          │
│       │                                                              │
│       ├──► 7. EMIT UI EVENTS                                        │
│       │    on_agent_created(agent_id, profile, icon)                 │
│       │                                                              │
│       ├──► 8. RUN CHAT LOOP                                         │
│       │    session.send_message(task, on_output=..., max_turns=...)  │
│       │    Iterates until model stops or max_turns reached           │
│       │                                                              │
│       ├──► 9. EMIT COMPLETION                                        │
│       │    on_agent_completed(agent_id, result, token_usage)         │
│       │                                                              │
│       └──► 10. RETURN RESULT                                         │
│            SubagentResult(success, response, turns_used, ...)        │
│                                                                      │
│  Parent Model receives result, continues its turn                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Resource Sharing vs Isolation

### What Is Shared

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SHARED RESOURCES                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Resource              Why Shared                                    │
│  ────────              ──────────                                    │
│  PluginRegistry        Single discovery, consistent tool schemas     │
│  PermissionPlugin      Unified approval state (whitelist/blacklist)  │
│  TokenLedger           Aggregated billing across all agents          │
│  TelemetryPlugin       Correlated traces across agent hierarchy      │
│  Provider configs      No redundant auth/connection setup            │
│  Base system instr.    Consistent behavioral rules for all agents    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### What Is Isolated

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ISOLATED PER SESSION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Resource              Why Isolated                                  │
│  ────────              ────────────                                  │
│  Conversation history  Each agent has its own context                │
│  Provider instance     Independent SDK client per session            │
│  Model selection       Subagent can use different model              │
│  Tool subset           Profile defines available plugins             │
│  System instructions   Profile-specific instructions prepended       │
│  GC plugin + config    Different collection strategy per agent       │
│  CancelToken           Independent cancellation per session          │
│  Turn counter          Each agent tracks its own turns               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Permission Bridging

When a subagent executes a tool that requires permission, the request must reach the user through the parent. The `ParentBridgedChannel` handles this:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMISSION BRIDGING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Subagent Session                                                    │
│       │                                                              │
│       │  Tool execution requires permission                          │
│       ▼                                                              │
│  PermissionPlugin.check_permission(tool, args)                       │
│       │                                                              │
│       ├──► Suspension check (turn/idle/all active?)  → ALLOW         │
│       ├──► Blacklist check → DENY                                    │
│       ├──► Whitelist check → ALLOW                                   │
│       └──► Must prompt user                                          │
│            │                                                         │
│            ▼                                                         │
│  ParentBridgedChannel                                                │
│       │                                                              │
│       │  Forwards permission request to parent's channel             │
│       │  (Thread-isolated: uses parent's I/O while maintaining       │
│       │   subagent's execution context)                              │
│       │                                                              │
│       ▼                                                              │
│  Parent's Permission Channel                                         │
│  (ConsoleChannel / QueueChannel / WebhookChannel)                    │
│       │                                                              │
│       │  User sees: "[subagent:code_reviewer] Allow cli.run?"        │
│       │                                                              │
│       ▼                                                              │
│  User decision flows back:                                           │
│  ParentBridgedChannel → PermissionPlugin → SubagentSession           │
│                                                                      │
│  KEY: The shared PermissionPlugin means whitelist/blacklist/          │
│  suspension decisions apply uniformly across parent and subagents.   │
│  If user types "all" on parent, subagents also get blanket approval. │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Cross-Provider Subagents

A subagent can use a different AI provider than its parent. The profile specifies the `provider` field, and the runtime resolves the correct `ProviderConfig`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-PROVIDER SUBAGENTS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Parent Agent: Anthropic (claude-sonnet-4-5)                        │
│       │                                                              │
│       │  delegate(profile="fast_search",                             │
│       │           task="Find all TODO comments")                     │
│       │                                                              │
│       ▼                                                              │
│  SubagentPlugin                                                      │
│       │                                                              │
│       │  Profile: provider="google_genai", model="gemini-2.5-flash" │
│       │                                                              │
│       ├──► runtime.create_session(                                   │
│       │        model="gemini-2.5-flash",                             │
│       │        provider_name="google_genai"                          │
│       │    )                                                         │
│       │                                                              │
│       ├──► runtime.create_provider("gemini-2.5-flash", "google_genai│")
│       │    ├──► Looks up _provider_configs["google_genai"]           │
│       │    ├──► load_provider("google_genai", config)               │
│       │    └──► provider.connect("gemini-2.5-flash")                │
│       │                                                              │
│       └──► Subagent runs with Google GenAI while parent uses Claude  │
│                                                                      │
│  Benefits:                                                           │
│  • Use fast/cheap models for routine subtasks                        │
│  • Use specialized models for specific capabilities                  │
│  • Balance cost vs quality across the agent hierarchy                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Output Streaming and UI Integration

Subagent output flows through the UI hooks system, which emits events for each lifecycle stage:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBAGENT UI EVENTS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AgentUIHooks Protocol:                                              │
│                                                                      │
│  on_agent_created(agent_id, agent_type, profile, icon)               │
│       │  → Client displays agent card/header                        │
│       │                                                              │
│  on_output(source, text, mode)                                       │
│       │  → Client streams text from subagent                        │
│       │     source: "model" | "tool" | "system"                     │
│       │     mode: "write" (new block) | "append" (continuation)     │
│       │                                                              │
│  on_tool_call_start(tool_name, args, call_id)                        │
│       │  → Client shows tool execution indicator                    │
│       │                                                              │
│  on_tool_call_end(tool_name, call_id, success, duration)             │
│       │  → Client updates tool status                               │
│       │                                                              │
│  on_agent_status_changed(agent_id, status)                           │
│       │  → Client updates agent status (active/idle/done/error)     │
│       │                                                              │
│  on_agent_completed(agent_id, result, token_usage)                   │
│       │  → Client shows completion summary                          │
│       │                                                              │
│  Event Transport:                                                    │
│  Subagent → SubagentPlugin → Server EventBus → Client(s)            │
│                                                                      │
│  output_streamed Flag:                                               │
│  When UI hooks are active, SubagentResult.output_streamed = True     │
│  This tells the parent model NOT to echo the response, since         │
│  the user already saw it streaming in real-time.                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Inline vs Profile-Based Delegation

The subagent plugin supports two delegation modes:

### Profile-Based

```python
delegate(profile="code_reviewer", task="Review the login module")
```

Uses a pre-defined `SubagentProfile` from `.jaato/profiles/` or the `SubagentConfig`. The profile defines tools, model, instructions, and constraints.

### Inline (Ad-Hoc)

```python
delegate(
    task="Search for security vulnerabilities",
    plugins=["grep_content", "file_edit", "web_search"],
    system_instructions="Focus on OWASP top 10 vulnerabilities",
    max_turns=5
)
```

Creates a one-off subagent with tools from the `inline_allowed_plugins` list. Requires `allow_inline: true` in the subagent config.

### Comparison

| Aspect | Profile-Based | Inline |
|--------|--------------|--------|
| Configuration | Pre-defined, version-controlled | Ad-hoc per request |
| Tools available | Profile's plugin list | `inline_allowed_plugins` whitelist |
| Auto-approval | Configurable per profile | Follows default policy |
| Model override | Per-profile setting | Inherits parent's model |
| GC strategy | Per-profile configuration | Parent's default |
| Use case | Repeatable specialized tasks | One-off explorations |

---

## Part 9: Subagent Execution Constraints

### Turn Limits

Each subagent has a `max_turns` limit (default: 10). When reached, the subagent returns its accumulated response to the parent. This prevents runaway agents.

### Tool Subset Enforcement

Subagents only have access to the plugins listed in their profile. The runtime's `get_tool_schemas(plugin_names)` and `get_executors(plugin_names)` methods filter to the specified subset. Core plugins (introspection, file_edit, etc.) are automatically added by `_get_essential_plugins()`.

### GC Per Agent

Each subagent can have its own GC strategy and configuration. A long-running research subagent might use `gc_budget` with continuous collection, while a quick code reviewer might use `gc_truncate` with a high threshold.

### No Nesting Limits

Subagents can spawn their own subagents (nested delegation). Each level shares the same runtime resources. Permissions propagate through bridged channels at each level.

---

## Part 10: SubagentResult

When a subagent completes, it returns a structured result to the parent:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SubagentResult                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  success: bool          Whether the subagent completed normally      │
│  response: str          Final accumulated response text              │
│  turns_used: int        Number of turns consumed                     │
│  error: str | None      Error message if failed                      │
│  token_usage: dict      Token accounting for the subagent's run      │
│  agent_id: str          Session ID (for potential continuation)      │
│  output_streamed: bool  True if output was already shown to user     │
│                                                                      │
│  When output_streamed is True, to_dict() omits the response text    │
│  and includes a note telling the model not to repeat it.             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_model_harness.md](jaato_model_harness.md) | How instructions, tools, and permissions form the harness |
| [jaato_multi_provider.md](jaato_multi_provider.md) | Provider abstraction enabling cross-provider subagents |
| [jaato_tool_system.md](jaato_tool_system.md) | Tool schemas, discoverability, execution pipeline |
| [jaato_permission_system.md](jaato_permission_system.md) | Permission evaluation, bridged channels |
| [jaato_gc_system.md](jaato_gc_system.md) | GC strategies including per-agent budget GC |

---

## Part 12: Color Coding Suggestion for Infographic

- **Blue:** JaatoRuntime (shared environment, provider configs)
- **Green:** JaatoSession instances (isolated conversation state)
- **Orange:** SubagentPlugin and delegation flow
- **Purple:** SubagentProfile (configuration, tool lists, constraints)
- **Red:** Permission bridging (ParentBridgedChannel, approval flow)
- **Yellow:** Data flow arrows (task delegation, result return)
- **Gray:** UI hooks and event streaming
- **Cyan:** Cross-provider connections (different SDK clients per session)
