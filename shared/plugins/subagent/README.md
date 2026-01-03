# Subagent Plugin

The Subagent plugin enables the parent model to delegate tasks to specialized subagents with their own tool configurations, system instructions, and model selection. Supports multi-turn conversations, parallel execution, cancellation propagation, and shared state for inter-agent communication.

## Demo

The demo below shows spawning a code-review subagent to analyze the CLI plugin source file for potential improvements. The subagent runs autonomously and returns its analysis to the parent agent.

![Subagent Plugin Demo](demo.svg)

## Architecture Overview

The subagent plugin uses the shared `JaatoRuntime` to create lightweight sessions, avoiding redundant provider connections:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Parent Agent                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      JaatoRuntime (Shared)                       │    │
│  │  • Provider config    • PluginRegistry    • Permissions          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                    │                               │                     │
│                    ▼                               ▼                     │
│  ┌──────────────────────────┐      ┌──────────────────────────┐        │
│  │   JaatoSession (Main)    │      │  SubagentPlugin          │        │
│  │   • History              │      │  • spawn_subagent        │        │
│  │   • Tools                │      │  • list_profiles         │        │
│  │   • Model                │      │                          │        │
│  └──────────────────────────┘      └────────────┬─────────────┘        │
│                                                  │                      │
│                                    runtime.create_session()             │
│                                                  │                      │
│                                                  ▼                      │
│                                    ┌──────────────────────────┐        │
│                                    │  JaatoSession (Subagent) │        │
│                                    │  • Own history           │        │
│                                    │  • Own model selection   │        │
│                                    │  • Tool subset           │        │
│                                    │  • Shares runtime        │        │
│                                    └──────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits of Runtime Sharing:**
- No redundant provider connections for each subagent
- Fast subagent spawning (lightweight session creation)
- Shared permissions and token accounting

## Features

- **Runtime Sharing**: Subagents use `JaatoRuntime.create_session()` for efficient spawning (no redundant connections)
- **Cross-Provider Support**: Subagents can use different AI providers than the parent (e.g., Anthropic parent → Google GenAI subagent)
- **Plugin Inheritance**: Subagents automatically inherit the parent's plugin configuration by default
- **Model/Provider Inheritance**: If not specified in profile, subagents inherit the parent's model and provider
- **Optional Overrides**: Use `inline_config` to override specific properties (plugins, max_turns, system_instructions)
- **Predefined Profiles**: Configure named profiles for common subagent configurations
- **Profile Auto-Discovery**: Automatically discover profiles from `.jaato/profiles/` directory (JSON/YAML files)
- **Connection Inheritance**: Subagents automatically inherit parent's GCP project, location, and model
- **Multi-Turn Conversations**: Send follow-up messages to active subagent sessions
- **Parallel Execution**: Spawn subagents in background for concurrent task execution
- **Cancellation Propagation**: Parent cancellation automatically propagates to child subagents
- **Shared State**: Thread-safe shared state for inter-agent communication

## Tools Exposed

| Tool | Description | Auto-Approved |
|------|-------------|---------------|
| `spawn_subagent` | Spawn a subagent to handle a task (supports `background=true` for parallel) | ✗ |
| `continue_subagent` | Send follow-up message to an active subagent session | ✗ |
| `close_subagent` | Close an active subagent session | ✗ |
| `cancel_subagent` | Cancel a running subagent operation | ✗ |
| `get_subagent_result` | Get result of a background subagent | ✗ |
| `list_active_subagents` | List active sessions and background agents | ✓ |
| `list_subagent_profiles` | List available predefined profiles | ✓ |
| `set_shared_state` | Store a value in shared state | ✗ |
| `get_shared_state` | Retrieve a value from shared state | ✗ |
| `list_shared_state` | List all keys in shared state | ✓ |

## User Commands

| Command | Description | Share with Model |
|---------|-------------|------------------|
| `profiles` | List available subagent profiles | ✓ |

## Usage

### Basic Usage (Inherited Plugins)

The simplest way to spawn a subagent - it inherits all parent plugins:

```python
# Model just provides the task
spawn_subagent(task="Analyze the codebase structure and summarize it")
```

### With Optional Overrides

Override specific properties while inheriting others:

```python
# Override max_turns only (inherits parent's plugins)
spawn_subagent(
    task="Quick file check",
    inline_config={"max_turns": 5}
)

# Override system_instructions only (inherits parent's plugins)
spawn_subagent(
    task="Research this topic",
    inline_config={"system_instructions": "Be concise and factual"}
)

# Override plugins (replaces inherited plugins)
spawn_subagent(
    task="Run shell commands only",
    inline_config={"plugins": ["cli"]}
)
```

### With Predefined Profiles

Use named profiles for common configurations:

```python
spawn_subagent(task="Analyze the code", profile="code_assistant")
```

### With Context

Provide additional context from the current conversation:

```python
spawn_subagent(
    task="Fix the bug we discussed",
    context="The user reported a NullPointerException in UserService.java line 42"
)
```

## Configuration

### Profile Auto-Discovery

The subagent plugin automatically discovers profile definitions from `.jaato/profiles/` directory. Each `.json` or `.yaml` file in this directory is parsed as a profile definition.

**Directory structure:**
```
.jaato/
└── profiles/
    ├── code_assistant.json
    ├── research_agent.yaml
    └── custom_agent.json
```

**Example profile file (`.jaato/profiles/code_assistant.json`):**
```json
{
  "name": "code_assistant",
  "description": "Subagent for code analysis and review",
  "plugins": ["cli", "file_edit"],
  "system_instructions": "You are a code review specialist.",
  "max_turns": 10,
  "auto_approved": false
}
```

**Profile with explicit model and provider (cross-provider):**
```json
{
  "name": "gemini_research_agent",
  "description": "Research agent using Google Gemini for cost efficiency",
  "plugins": ["web_search", "references"],
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "system_instructions": "You are a research specialist.",
  "max_turns": 15
}
```

> **Model/Provider Inheritance:** If `model` or `provider` is not specified in the profile, the subagent inherits from the parent session. This allows a Claude-based main agent to spawn subagents that also use Claude without redundant configuration.

**Profile with plugin-specific configuration:**
```json
{
  "name": "skill-add-retry",
  "description": "Add retry pattern to Java services",
  "plugins": ["cli", "file_edit", "references"],
  "plugin_configs": {
    "references": {
      "preselected": ["adr-001-resilience-patterns", "eri-002-retry"],
      "exclude_tools": ["selectReferences"]
    }
  },
  "system_instructions": "Implement retry pattern following the pre-selected references.",
  "max_turns": 15
}
```

**Profile with garbage collection configuration:**
```json
{
  "name": "long-running-agent",
  "description": "Agent for extended tasks requiring context management",
  "plugins": ["cli", "file_edit"],
  "gc": {
    "type": "hybrid",
    "threshold_percent": 80.0,
    "preserve_recent_turns": 5,
    "notify_on_gc": true,
    "summarize_middle_turns": 10
  },
  "max_turns": 50
}
```

The `gc` field configures context garbage collection for the subagent:

| GC Option | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | `"truncate"` | GC strategy: `"truncate"`, `"summarize"`, or `"hybrid"` |
| `threshold_percent` | float | 80.0 | Trigger GC when context usage exceeds this % |
| `preserve_recent_turns` | int | 5 | Number of recent turns to always preserve |
| `notify_on_gc` | bool | true | Inject notification into history after GC |
| `summarize_middle_turns` | int | null | For hybrid: turns to summarize (not truncate) |
| `max_turns` | int | null | Trigger GC when turn count exceeds this |
| `plugin_config` | object | {} | Additional plugin-specific configuration |

The `plugin_configs` field allows per-plugin configuration overrides:

| Plugin | Config Option | Description |
|--------|---------------|-------------|
| `references` | `preselected` | List of source IDs to pre-select at startup |
| `references` | `exclude_tools` | List of tools to hide (e.g., `["selectReferences"]`) |
| `references` | `sources` | Override available sources (IDs or full objects) |

**Configuration options:**
- `auto_discover_profiles`: Enable/disable auto-discovery (default: `true`)
- `profiles_dir`: Directory to scan for profiles (default: `.jaato/profiles`)

```python
plugin.initialize({
    'auto_discover_profiles': True,      # Enable auto-discovery
    'profiles_dir': '.jaato/profiles',   # Custom profiles directory
})
```

**Merge behavior:** Discovered profiles are merged with explicitly configured profiles. Explicit profiles take precedence on name conflicts.

### Plugin Initialization

```python
from shared.plugins.subagent import SubagentPlugin

plugin = SubagentPlugin()
plugin.initialize({
    'project': 'my-gcp-project',         # Optional: inherited from parent
    'location': 'us-central1',            # Optional: inherited from parent
    'default_model': None,                # None = inherit from parent (recommended)
    'default_provider': None,             # None = inherit from parent (recommended)
    'profiles': {
        'code_assistant': {
            'description': 'Subagent for code analysis',
            'plugins': ['cli'],
            'max_turns': 10,
            # model/provider not set = inherits from parent
        },
        'gemini_research': {
            'description': 'Research using Gemini',
            'plugins': ['web_search'],
            'model': 'gemini-2.5-flash',     # Explicit model
            'provider': 'google_genai',      # Explicit provider (must match model)
            'max_turns': 15,
        }
    },
    'allow_inline': True,                 # Allow inline_config (default: True)
    'inline_allowed_plugins': [],         # Restrict inline plugins (empty = all allowed)
    'auto_discover_profiles': True,       # Auto-discover from profiles_dir (default: True)
    'profiles_dir': '.jaato/profiles',    # Directory to scan for profiles
})
```

> **Important:** When setting `model` in a profile, you should also set `provider` to ensure they match. If only `model` is set without `provider`, the provider is inherited from parent, which may not support that model.

### Connection Inheritance

When using with `JaatoClient`, connection settings are automatically passed to the subagent plugin:

```python
client = JaatoClient()
client.connect(project_id, location, model)
client.configure_tools(registry)  # Automatically configures subagent plugin
```

The subagent plugin receives:
- Project ID, location, and model from parent
- List of exposed plugins from parent (for inheritance)

### Profile Configuration

Profiles can be added programmatically:

```python
from shared.plugins.subagent import SubagentProfile

# Profile that inherits model/provider from parent
plugin.add_profile(SubagentProfile(
    name='simple_agent',
    description='Inherits parent model and provider',
    plugins=['cli', 'todo'],
    system_instructions='You are a specialized assistant.',
    max_turns=20,
    auto_approved=False,
))

# Profile with explicit cross-provider configuration
plugin.add_profile(SubagentProfile(
    name='gemini_agent',
    description='Uses Google Gemini regardless of parent provider',
    plugins=['cli', 'file_edit'],
    model='gemini-2.5-flash',
    provider='google_genai',
    system_instructions='You are a code generation specialist.',
    max_turns=15,
))
```

## Behavior Summary

| Scenario | Plugins Used | Other Settings |
|----------|--------------|----------------|
| `spawn_subagent(task="...")` | Inherited from parent | Defaults |
| `spawn_subagent(task="...", inline_config={max_turns: 5})` | Inherited from parent | max_turns=5 |
| `spawn_subagent(task="...", inline_config={plugins: ['cli']})` | ['cli'] | Defaults |
| `spawn_subagent(task="...", profile="x")` | From profile | From profile |

## Parallel Execution

Spawn subagents in the background for concurrent task execution:

```python
# Spawn a background agent (returns immediately)
spawn_subagent(
    task="Analyze the API module",
    background=True
)
# Returns: {'success': True, 'background': True, 'agent_id': 'subagent_abc123', ...}

# Spawn multiple agents in parallel
spawn_subagent(task="Review authentication code", background=True)  # agent_1
spawn_subagent(task="Check database queries", background=True)       # agent_2
spawn_subagent(task="Analyze error handling", background=True)       # agent_3

# Check status of all agents
list_active_subagents()
# Returns list of active sessions and background agents with their status

# Get result when ready
get_subagent_result(agent_id="subagent_abc123")
# Returns: {'success': True, 'response': '...', 'status': 'completed'}
# Or: {'success': True, 'status': 'running'} if still in progress
```

Background agents run in daemon threads and store their results for later retrieval. Use `list_active_subagents` to monitor progress and `get_subagent_result` to collect responses.

## Cross-Provider Support

Subagents can use different AI providers than the parent agent. This enables scenarios like:
- **Cost optimization**: Main agent uses Claude for complex reasoning, subagents use Gemini for simpler tasks
- **Capability matching**: Use the best provider for each task type
- **Fallback strategies**: Switch providers if one is rate-limited

### Configuration

Specify `model` and `provider` in the profile:

```json
{
  "name": "gemini_coder",
  "description": "Code generation using Gemini",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "plugins": ["cli", "file_edit"]
}
```

### Inheritance Rules

The inheritance chain for `model` and `provider` is:

1. **Profile level**: `profile.model` / `profile.provider` (if explicitly set)
2. **Config level**: `default_model` / `default_provider` from SubagentConfig (if set)
3. **Parent level**: Inherited from parent session (if both above are None)

| Profile Setting | Config Default | Result |
|-----------------|----------------|--------|
| `model: "gemini-2.5-flash"`, `provider: "google_genai"` | (any) | Uses Gemini on Google GenAI |
| `model: null`, `provider: null` | `null`, `null` | Inherits parent's model and provider |
| `model: null`, `provider: null` | `"gemini-2.5-flash"`, `"google_genai"` | Uses config defaults |

### Provider Registration

The runtime automatically registers providers when they're first used. For providers requiring specific configuration, register them explicitly:

```python
from shared import JaatoRuntime, ProviderConfig

runtime = JaatoRuntime(provider_name="anthropic")
runtime.connect(project_id, location)

# Register additional provider for subagents
runtime.register_provider("google_genai", ProviderConfig(
    project="my-gcp-project",
    location="us-central1"
))
```

## Cancellation Propagation

Parent cancellation automatically propagates to child subagents:

```python
# When the parent agent is cancelled (e.g., user presses Ctrl+C),
# all running subagents are automatically cancelled too.

# Manual cancellation of a specific subagent:
cancel_subagent(agent_id="subagent_abc123")
# Returns: {'success': True, 'message': 'Cancellation requested for subagent_abc123'}
```

The cancellation mechanism works through shared `CancelToken` objects:
- Each session has its own cancel token
- Subagents receive a reference to their parent's cancel token
- Subagents check both their own and parent's token before each operation
- When parent is cancelled, all children see it and stop gracefully

## Callback Propagation

Subagents automatically inherit callbacks from the parent to ensure consistent behavior across all agents:

### Retry Callback

Rate limit retry messages are routed through the same channel as the parent:

```python
# Rich client sets up retry callback on subagent plugin
subagent_plugin.set_retry_callback(on_retry)

# When subagents are spawned, they inherit this callback
# Retry messages appear in the output panel, not console
```

The retry callback ensures that API rate limit retries from subagents are displayed consistently with the parent's output (e.g., in the rich client's output panel rather than raw console output).

### UI Hooks

UI hooks are similarly propagated to subagent sessions for consistent agent lifecycle tracking:

```python
subagent_plugin.set_ui_hooks(hooks)
# All spawned subagents emit lifecycle events through these hooks
```

## Shared State

Thread-safe shared state for inter-agent communication:

```python
# Store a value (accessible by all agents)
set_shared_state(key="analysis_results", value={"files_checked": 42, "issues": []})

# Retrieve a value
get_shared_state(key="analysis_results")
# Returns: {'success': True, 'value': {"files_checked": 42, "issues": []}}

# List all keys
list_shared_state()
# Returns: {'success': True, 'keys': ['analysis_results', 'other_key']}

# Missing key returns null
get_shared_state(key="nonexistent")
# Returns: {'success': True, 'value': None}
```

Use cases for shared state:
- **Coordination**: One agent sets a flag when ready, others wait for it
- **Result aggregation**: Multiple agents contribute to a shared collection
- **Configuration sharing**: Store computed values for other agents to use
- **Progress tracking**: Update shared counters or status indicators

All shared state operations are thread-safe and can be used safely with parallel execution.

## Integration with JaatoClient

```python
from shared import JaatoClient, PluginRegistry

# Setup
registry = PluginRegistry()
registry.discover()
registry.expose_tool('cli')
registry.expose_tool('mcp')
registry.expose_tool('subagent')

client = JaatoClient()
client.connect(project_id, location, model)
client.configure_tools(registry)  # Subagent inherits ['cli', 'mcp']

# Now subagents spawned will have access to cli and mcp by default
response = client.send_message("Spawn a subagent to analyze the code")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JAATO_TRACE_LOG` | Path to trace log file for debug output. Useful when running with rich terminal UIs that occupy the console. Set to empty string to disable. | `/tmp/rich_client_trace.log` |
| `PROJECT_ID` | GCP project ID (fallback if not provided in config) | - |
| `LOCATION` | Vertex AI region (fallback if not provided in config) | - |
| `MODEL_NAME` | Default model name (fallback if not provided in config) | Inherited from parent |

## Profile Schema Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique identifier for the profile |
| `description` | string | `""` | Human-readable description |
| `plugins` | string[] | `[]` | List of plugin names to enable |
| `plugin_configs` | object | `{}` | Per-plugin configuration overrides |
| `system_instructions` | string | `null` | Additional system instructions |
| `model` | string | `null` | Model name (e.g., `"gemini-2.5-flash"`, `"claude-sonnet-4-20250514"`). `null` = inherit from parent |
| `provider` | string | `null` | Provider name (e.g., `"google_genai"`, `"anthropic"`). `null` = inherit from parent |
| `max_turns` | int | `10` | Maximum conversation turns |
| `auto_approved` | bool | `false` | Spawn without permission prompt |
| `icon` | string[] | `null` | Custom 3-line ASCII art icon |
| `icon_name` | string | `null` | Name of predefined icon |
| `gc` | object | `null` | Garbage collection configuration |
