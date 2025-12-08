# Plugin Framework

This document explains how to use the jaato plugin framework from a client perspective and how to implement new plugins.

## Overview

The plugin framework provides a way to dynamically discover, load, and manage tool implementations that can be used by the AI model. Plugins are discovered via two mechanisms:

1. **Entry Points** (recommended): Plugins register via Python entry points in `pyproject.toml`. This allows external packages to provide plugins.
2. **Directory Scanning** (fallback): Plugins in `shared/plugins/` are auto-discovered for development.

Discovered plugins can be exposed/unexposed at runtime.

## Client Usage

### Basic Usage

```python
from shared.plugins import PluginRegistry

# Create a registry and discover all available plugins
registry = PluginRegistry()
registry.discover()

# List available plugins
print(registry.list_available())  # ['cli', 'mcp', ...]
```

### Exposing Plugin Tools

By default, all discovered plugins should be exposed using `expose_all()`. This makes all tools available to the AI model.

```python
# Expose all discovered plugins (recommended)
registry.expose_all()

# Expose all with configuration for specific plugins
registry.expose_all({
    'cli': {'extra_paths': ['/usr/local/bin']},
    'todo': {'reporter_type': 'console'},
})

# Check what's exposed
print(registry.list_exposed())  # ['cli', 'mcp', 'todo', ...]

# Unexpose all plugins' tools (cleanup)
registry.unexpose_all()
```

For selective exposure (opt-in instead of opt-out), use `expose_tool()`:

```python
# Expose only specific plugins
registry.expose_tool('cli')
registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})

# Unexpose a specific plugin
registry.unexpose_tool('cli')
```

### Getting Tool Declarations, Executors, and User Commands

Once plugins are exposed, you can retrieve their tool declarations (for the AI model), executors (for running the tools), and user commands (for direct user interaction).

```python
# Get FunctionDeclarations for Vertex AI (model tools)
declarations = registry.get_exposed_tool_schemas()

# Get executor callables
executors = registry.get_exposed_executors()
# Returns: {'tool_name': callable, ...}

# Get user-facing commands for autocompletion
user_commands = registry.get_exposed_user_commands()
# Returns: [('command_name', 'description'), ...]
```

**Note**: User commands are distinct from model tools. They are commands that users (human or agent) can invoke directly without going through the model's function calling.

### Model-Aware Plugin Registry

Some plugins require specific model capabilities. For example, the `multimodal` plugin requires Gemini 3+ for multimodal function responses. Pass the `model_name` to the registry to enable automatic compatibility checking:

```python
from shared.plugins import PluginRegistry

# Create registry with model name for compatibility checking
model_name = 'gemini-2.5-flash'
registry = PluginRegistry(model_name=model_name)
registry.discover()

# expose_tool() will skip plugins that don't support this model
registry.expose_all()
# Output: [PluginRegistry] Plugin 'multimodal' skipped: model 'gemini-2.5-flash' not in ['gemini-3-pro*', ...]

# Check which plugins were skipped and why
skipped = registry.list_skipped_plugins()
# Returns: {'multimodal': ['gemini-3-pro*', 'gemini-3.5-*', 'gemini-4*']}
```

This prevents runtime errors when a plugin's features aren't supported by the current model.

### Integration with JaatoClient

The recommended way to use plugins is with `JaatoClient`:

```python
from shared import JaatoClient, PluginRegistry, TokenLedger

model_name = 'gemini-2.5-flash'

# Setup with model-aware registry
registry = PluginRegistry(model_name=model_name)
registry.discover()
registry.expose_all()  # Incompatible plugins automatically skipped

# Create and configure client
jaato = JaatoClient()
jaato.connect('my-project', 'us-central1', model_name)
jaato.configure_tools(registry, ledger=TokenLedger())

# Run prompts (SDK manages history internally)
# The callback receives (source, text, mode) for real-time output
def on_output(source: str, text: str, mode: str):
    print(f"[{source}]: {text}")

response = jaato.send_message('List files in current directory', on_output=on_output)

# Multi-turn conversations work automatically
response2 = jaato.send_message('Now show hidden files too', on_output=on_output)

# Access history when needed
history = jaato.get_history()

# Reset session to start fresh
jaato.reset_session()

# Cleanup
registry.unexpose_all()
```

### Dynamic Plugin Switching

You can expose/unexpose plugins between prompts to change what tools are available.
Note: Calling `configure_tools()` creates a new chat session, so conversation history is reset.

```python
from shared import JaatoClient, PluginRegistry

model_name = 'gemini-2.5-flash'
registry = PluginRegistry(model_name=model_name)
registry.discover()

jaato = JaatoClient()
jaato.connect('my-project', 'us-central1', model_name)

# First session: All plugins
registry.expose_all()
jaato.configure_tools(registry)
response1 = jaato.send_message('List files', on_output=lambda s, t, m: None)

# Second session: Only specific plugins (new chat session)
registry.unexpose_all()
registry.expose_tool('mcp')
jaato.configure_tools(registry)
response2 = jaato.send_message('Search GitHub issues', on_output=lambda s, t, m: None)

# Cleanup
registry.unexpose_all()
```

---

## Implementing a New Plugin

### Option 1: Entry Points (Recommended)

Register your plugin via entry points in `pyproject.toml`. This works for both built-in and external plugins.

```toml
# In pyproject.toml
[project.entry-points."jaato.plugins"]
my_plugin = "my_package.plugins:create_plugin"
```

The entry point must reference a `create_plugin()` factory function that returns a `ToolPlugin` instance.

### Option 2: Directory Placement (Development)

For quick development, add a Python file to `shared/plugins/` that implements the `ToolPlugin` protocol.

### Plugin Protocol

Plugins provide two types of capabilities:

1. **Model tools**: Functions the AI model can invoke via function calling
2. **User commands**: Commands the user can invoke directly (without model mediation)

> **Note on "user"**: In this context, "user" refers to the entity directly interfacing with the client. This could be a human operator OR another AI agent in agent-to-agent communication scenarios. User commands bypass the model's function calling and execute directly.

Every plugin must implement these methods:

```python
from typing import Protocol, List, Dict, Any, Callable, Optional, NamedTuple
from google.genai import types

class UserCommand(NamedTuple):
    """Declaration of a user-facing command.

    Attributes:
        name: Command name for invocation and autocompletion.
        description: Brief description shown in autocompletion/help.
        share_with_model: If True, command output is added to conversation
            history so the model can see/use it. If False (default),
            output is only shown to the user.
    """
    name: str
    description: str
    share_with_model: bool = False

class ToolPlugin(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        ...

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return provider-agnostic ToolSchema objects for model tools.

        These are functions the AI model can invoke via function calling.
        ToolSchema is converted to the appropriate SDK type by the provider.
        """
        ...

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return a mapping of tool names to their executor callables."""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called when the plugin is exposed. Setup resources here."""
        ...

    def shutdown(self) -> None:
        """Called when the plugin is unexposed. Cleanup resources here."""
        ...

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the model about this plugin's tools."""
        ...

    def get_auto_approved_tools(self) -> List[str]:
        """Return tool/command names that should be auto-approved without permission prompts.

        Tools listed here will be automatically whitelisted when used with
        the permission plugin. Use this for:
        - Read-only tools with no security implications
        - User commands (since they are invoked directly by the user, not the model)

        IMPORTANT: User commands from get_user_commands() should typically be
        listed here. Otherwise they will trigger unexpected permission prompts.

        Return empty list if all tools/commands require permission.
        """
        ...

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands this plugin provides.

        User commands are different from model tools:
        - Model tools: Invoked by the AI via function calling
        - User commands: Invoked directly by the user without model mediation

        The "user" can be a human operator OR another AI agent in
        agent-to-agent communication scenarios.

        Each command declares share_with_model to control whether its output
        is added to conversation history (visible to the model) or only
        displayed to the user.

        Most plugins only provide model tools and should return an empty list.
        Use this for plugins that also provide direct interaction commands.

        Returns:
            List of UserCommand objects for autocompletion and execution.
        """
        ...

    # ==================== Optional Methods ====================
    # The following methods are optional extensions to the protocol.
    # Implement them to enable model-specific or prompt enrichment features.

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time plugin output.

        When set, the plugin can emit output via the callback during tool
        execution. This enables real-time feedback during long-running
        operations.

        The callback signature is: (source: str, text: str, mode: str) -> None
        - source: Plugin name (e.g., "permission", "cli")
        - text: The output text
        - mode: "write" for new output block, "append" to continue

        Most plugins don't need this - implement only if your plugin has
        user-facing output during tool execution.

        Args:
            callback: OutputCallback function, or None to disable.
        """
        ...

    def get_model_requirements(self) -> Optional[List[str]]:
        """Return glob patterns for models this plugin requires.

        If the PluginRegistry has a model_name set and it doesn't match
        any of the patterns, the plugin will be skipped during expose_tool()
        with a warning message.

        Examples:
            ["gemini-3-pro*", "gemini-3.5-*"]  # Requires Gemini 3+
            ["gemini-2.5-*", "gemini-3-*"]     # Requires 2.5 or 3.x
            None                               # Works with any model (default)

        Returns:
            List of glob patterns, or None if plugin works with any model.
        """
        ...

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Return True if this plugin wants to process prompts before sending.

        Plugins that subscribe will have their enrich_prompt() method called
        with the user's prompt before it is sent to the model. This enables:
        - Detecting and processing @file references
        - Adding context or instructions based on prompt content
        - Tracking referenced resources for later tool calls

        Returns:
            True to subscribe, False otherwise (default).
        """
        ...

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Enrich a user prompt before sending to the model.

        Called only if subscribes_to_prompt_enrichment() returns True.
        The plugin can inspect and modify the prompt.

        IMPORTANT: Plugins should NOT remove @references from the prompt.
        The framework handles @reference cleanup after all plugins have
        processed the prompt.

        Args:
            prompt: The user's original prompt text.

        Returns:
            PromptEnrichmentResult with the enriched prompt and metadata.
        """
        ...
```

### Minimal Plugin Example

Here's a minimal plugin that provides a single tool:

```python
# shared/plugins/example.py

from typing import Dict, List, Any, Callable, Optional
from google.genai import types
from shared.plugins.base import UserCommand


class ExamplePlugin:
    """A simple example plugin."""

    def __init__(self):
        self._initialized = False

    @property
    def name(self) -> str:
        return "example"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Perform any setup here
        self._initialized = True

    def shutdown(self) -> None:
        # Cleanup resources here
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(
            name='example_tool',
            description='An example tool that echoes input',
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                "required": ["message"]
            }
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {'example_tool': self._execute}

    def get_system_instructions(self) -> Optional[str]:
        return "You have access to example_tool which echoes messages."

    def get_auto_approved_tools(self) -> List[str]:
        # Return empty list - this tool requires permission
        return []

    def get_user_commands(self) -> List[UserCommand]:
        # Return empty list - this plugin only provides model tools
        return []

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        message = args.get('message', '')
        return {'echo': message, 'length': len(message)}


def create_plugin() -> ExamplePlugin:
    """Factory function required for plugin discovery."""
    return ExamplePlugin()
```

### Key Requirements

1. **Factory Function**: Your plugin module must export a `create_plugin()` function that returns an instance of your plugin class.

2. **Unique Name**: The `name` property must return a unique identifier for your plugin.

3. **ToolSchema Format**: Tool declarations use provider-agnostic ToolSchema:
   ```python
   ToolSchema(
       name='tool_name',
       description='What the tool does',
       parameters={
           "type": "object",
           "properties": {
               "param_name": {
                   "type": "string",  # or "number", "boolean", "array", "object"
                   "description": "What this parameter is for"
               }
           },
           "required": ["param_name"]  # List of required parameters
       }
   )
   ```

4. **Executor Signature**: Each executor must accept a `Dict[str, Any]` and return a JSON-serializable result:
   ```python
   def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
       # args contains the parameters from the AI model
       return {'result': 'some value'}
   ```

### Plugin with Configuration

If your plugin needs configuration, handle it in `initialize()`:

```python
class ConfigurablePlugin:
    def __init__(self):
        self._api_key: Optional[str] = None
        self._timeout: int = 30

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self._api_key = config.get('api_key')
            self._timeout = config.get('timeout', 30)

    # ... rest of implementation
```

Client usage:
```python
# Via expose_all with config
registry.expose_all({
    'configurable': {'api_key': 'secret123', 'timeout': 60}
})

# Or via expose_tool for selective exposure
registry.expose_tool('configurable', config={
    'api_key': 'secret123',
    'timeout': 60
})
```

### Plugin with Multiple Tools

A single plugin can provide multiple tools:

```python
class MultiToolPlugin:
    @property
    def name(self) -> str:
        return "multi"

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(name='tool_a', description='...', parameters={...}),
            ToolSchema(name='tool_b', description='...', parameters={...}),
            ToolSchema(name='tool_c', description='...', parameters={...}),
        ]

    def get_executors(self) -> Dict[str, Callable]:
        return {
            'tool_a': self._execute_a,
            'tool_b': self._execute_b,
            'tool_c': self._execute_c,
        }
```

### Plugin with User Commands

If your plugin provides commands that users can invoke directly (bypassing the model):

```python
from shared.plugins.base import UserCommand, CommandCompletion

class SearchPlugin:
    @property
    def name(self) -> str:
        return "search"

    def get_tool_schemas(self) -> List[ToolSchema]:
        # Model tools - invoked by the AI
        return [
            ToolSchema(
                name='search_index',
                description='Search the index',
                parameters={...}
            )
        ]

    def get_executors(self) -> Dict[str, Callable]:
        return {'search_index': self._execute_search}

    def get_auto_approved_tools(self) -> List[str]:
        # IMPORTANT: Include user commands here so they don't trigger
        # permission prompts. Users invoke these directly, not the model.
        return ["search", "reindex", "stats"]

    def get_user_commands(self) -> List[UserCommand]:
        # User commands - invoked directly by user (human or agent)
        # share_with_model controls whether output is added to conversation history
        return [
            UserCommand("search", "Search the index directly", share_with_model=True),
            UserCommand("reindex", "Rebuild the search index", share_with_model=False),
            UserCommand("stats", "Show index statistics", share_with_model=False),
        ]

    def get_command_completions(self, command: str, args: List[str]) -> List[CommandCompletion]:
        # Optional: provide autocompletion for command arguments
        # Return empty list if no completions available
        return []

    # ... other methods ...
```

User commands appear in client autocompletion and can be typed directly without going through the model's function calling. The `share_with_model` flag controls whether the command's output is added to conversation history:

- `share_with_model=True`: Output is added to history, model can see and use the results
- `share_with_model=False` (default): Output is only displayed to the user

This is useful for:

- **Shared commands** (`share_with_model=True`): Search results, listing data, status that informs the model
- **User-only commands** (`share_with_model=False`): Administrative tasks, health checks, cache operations

### Command Argument Completions

Plugins can optionally provide autocompletion for their user command arguments by implementing `get_command_completions()`:

```python
from shared.plugins.base import UserCommand, CommandCompletion

class ConfigPlugin:
    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand("config", "Manage configuration: get <key>, set <key> <value>"),
        ]

    def get_command_completions(self, command: str, args: List[str]) -> List[CommandCompletion]:
        """Provide completions for config command arguments."""
        if command != "config":
            return []

        subcommands = [
            CommandCompletion("get", "Get a config value"),
            CommandCompletion("set", "Set a config value"),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            # Partial subcommand - filter matches
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        return []
```

The client will call `get_command_completions()` when the user types a command and presses Tab:
- `command`: The command name being typed (e.g., "config")
- `args`: List of arguments typed so far (may include partial input)

Return a list of `CommandCompletion(value, description)` tuples matching the current input.

### Plugin with Background Resources

For plugins that need persistent connections or background threads (like the MCP plugin):

```python
import threading

class BackgroundPlugin:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._background_work, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def _background_work(self):
        while self._running:
            # Do background work
            pass
```

---

## Built-in Plugins

### CLI Plugin (`cli`)

Executes local shell commands.

**Configuration:**
- `extra_paths`: List of additional directories to add to PATH

**Tools:**
- `cli_based_tool`: Execute a shell command

**Example with configuration:**
```python
registry.expose_all({'cli': {'extra_paths': ['/opt/custom/bin']}})
```

### MCP Plugin (`mcp`)

Connects to MCP (Model Context Protocol) servers defined in `.mcp.json` and exposes their tools.

**Configuration:** None (reads from `.mcp.json`)

**Tools:** Dynamic - depends on connected MCP servers

**Example:**
```python
registry.expose_all()  # MCP tools are available automatically
```

### Web Search Plugin (`web_search`)

Searches the web for current information using DuckDuckGo.

**Configuration:**
- `max_results`: Maximum number of results to return (default: 10)
- `safesearch`: Safe search level - "off", "moderate", "strict" (default: "moderate")
- `region`: Region for search results (default: "wt-wt" for no region)

**Tools:**
- `web_search`: Search the web for information on any topic

**Auto-approved:** Yes (read-only operation)

**Example:**
```python
registry.expose_all({'web_search': {'max_results': 5, 'safesearch': 'strict'}})
```

### File Edit Plugin (`file_edit`)

Provides tools for reading, modifying, and managing files with integrated permission approval (showing diffs) and automatic backups.

**Configuration:**
- `backup_dir`: Directory for storing backups (default: `.jaato/backups`)

**Tools:**
- `readFile`: Read file contents (auto-approved)
- `updateFile`: Update existing file (shows diff for approval, creates backup)
- `writeNewFile`: Create new file (shows content for approval)
- `removeFile`: Delete file (creates backup)
- `undoFileChange`: Restore from most recent backup (auto-approved)

**Auto-approved:** `readFile`, `undoFileChange`

**Environment Variables:**
- `JAATO_FILE_BACKUP_COUNT`: Maximum backups per file (default: 5)

**Example:**
```python
registry.expose_all({'file_edit': {'backup_dir': '/custom/backup/path'}})
```

### Clarification Plugin (`clarification`)

Allows the model to request clarification from users by asking questions with multiple choice options or free-text responses.

**Configuration:**
- `actor_type`: How to collect responses - `"console"` (default) or `"auto"` (for testing)
- `actor_config`: Actor-specific configuration

**Tools:**
- `request_clarification`: Ask the user one or more questions

**Auto-approved:** Yes (user interaction is inherently approved)

**Features:**
- Multiple questions per request
- Question types: `single_choice`, `multiple_choice`, `free_text`
- Optional questions with defaults
- Questions and choices use ordinal indices (1, 2, 3...) for simplicity
- Console UI shows required/optional status

**Example:**
```python
registry.expose_all({'clarification': {'actor_type': 'console'}})
```

**Tool Usage Example:**
```json
{
  "context": "I need to configure the deployment.",
  "questions": [
    {
      "text": "Which environment?",
      "question_type": "single_choice",
      "choices": ["Development", "Staging", "Production"],
      "default_choice": 1
    },
    {
      "text": "Enable optional features?",
      "question_type": "multiple_choice",
      "choices": ["Logging", "Metrics", "Tracing"],
      "required": false
    }
  ]
}
```

---

## File Structure

```
shared/plugins/
├── __init__.py      # Exports PluginRegistry, ToolPlugin
├── base.py          # ToolPlugin Protocol definition
├── registry.py      # PluginRegistry class
├── README.md        # This documentation
├── cli/             # CLI tool plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── README.md
│   └── tests/
├── mcp/             # MCP tool plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── README.md
│   └── tests/
├── permission/      # Permission control plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── README.md
│   └── tests/
├── todo/            # Plan tracking plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── README.md
│   └── tests/
├── web_search/      # Web search plugin
│   ├── __init__.py
│   ├── plugin.py
│   └── README.md
├── file_edit/       # File editing plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── backup.py
│   ├── diff_utils.py
│   ├── README.md
│   └── tests/
├── clarification/   # User clarification plugin
│   ├── __init__.py
│   ├── plugin.py
│   ├── models.py
│   ├── actors.py
│   └── tests/
└── references/      # Documentation injection plugin
    ├── __init__.py
    ├── plugin.py
    ├── README.md
    └── tests/
```

---

## External Plugin Packages

External packages can provide jaato plugins by registering entry points.

### Creating an External Plugin Package

1. Create your plugin module with a `create_plugin()` factory:

```python
# my_jaato_plugin/plugin.py
from shared.plugins.base import ToolPlugin

class MyPlugin:
    # ... implement ToolPlugin protocol ...

def create_plugin():
    return MyPlugin()
```

2. Register the entry point in `pyproject.toml`:

```toml
[project]
name = "my-jaato-plugin"
version = "0.1.0"
dependencies = ["jaato"]

[project.entry-points."jaato.plugins"]
my_plugin = "my_jaato_plugin.plugin:create_plugin"
```

3. Install your package:

```bash
pip install -e .  # or pip install my-jaato-plugin
```

4. Your plugin will be automatically discovered:

```python
registry = PluginRegistry()
discovered = registry.discover()
# ['cli', 'mcp', 'permission', 'todo', 'references', 'my_plugin']
```

### Discovery Modes

```python
# Discover via both entry points and directory scanning (default)
registry.discover()

# Discover via entry points only (installed packages)
registry.discover(include_directory=False)
```

---

## Manual Plugin Registration

Some plugins are not discovered via entry points or directory scanning. These "special" plugins (like the session persistence plugin) are configured separately but may still need to participate in registry-managed features like prompt enrichment.

### register_plugin()

Use `register_plugin()` to manually add a plugin to the registry:

```python
from shared.plugins.session import create_plugin as create_session_plugin

# Create and initialize the plugin
session_plugin = create_session_plugin()
session_plugin.initialize({'storage_path': '.jaato/sessions'})

# Register with the registry
registry.register_plugin(session_plugin, expose=True)
```

Parameters:
- `plugin`: The plugin instance to register
- `expose`: If `True`, also expose the plugin's tools (equivalent to calling `expose_tool()`)
- `enrichment_only`: If `True`, only participate in prompt enrichment (see below)
- `config`: Optional configuration dict if exposing

### Enrichment-Only Mode

Some plugins only need to participate in prompt enrichment without exposing their tools to the model. This is useful when:

1. The plugin's tools are already registered elsewhere (e.g., via `JaatoClient.set_session_plugin()`)
2. You want to avoid duplicate tool declarations
3. The plugin only provides prompt enrichment, not model tools

```python
# Register for prompt enrichment only - no tools exposed
registry.register_plugin(session_plugin, enrichment_only=True)
```

When `enrichment_only=True`:
- The plugin IS included in `get_prompt_enrichment_subscribers()`
- The plugin is NOT included in `get_exposed_tool_schemas()`
- The plugin is NOT included in `get_exposed_executors()`
- The plugin is NOT included in `get_exposed_user_commands()`

This is how the session plugin integrates with the registry:

```python
# Session plugin is set on JaatoClient for tools/commands
jaato.set_session_plugin(session_plugin, config)

# And registered with registry for prompt enrichment only
registry.register_plugin(session_plugin, enrichment_only=True)
```

The session plugin's `enrich_prompt()` is called by the registry's prompt enrichment pipeline, but its tools (like `session_describe`) are managed separately by JaatoClient.

### Prompt Enrichment Pipeline

Plugins can subscribe to enrich user prompts before they are sent to the model. This enables features like:

- **@file references**: Detect and process `@filename.png` patterns
- **Lazy loading**: Model decides when to load heavy content
- **Context injection**: Add relevant information based on prompt content
- **Session descriptions**: Request model-generated descriptions after N turns

To participate in prompt enrichment, implement these methods:

```python
from shared.plugins.base import PromptEnrichmentResult

class MyPlugin:
    def subscribes_to_prompt_enrichment(self) -> bool:
        """Return True to receive prompts for enrichment."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Inspect and optionally modify the prompt.

        Args:
            prompt: The user's prompt text.

        Returns:
            PromptEnrichmentResult with enriched prompt and optional metadata.
        """
        enriched = prompt + "\n[Injected context from MyPlugin]"
        return PromptEnrichmentResult(
            prompt=enriched,
            metadata={'injected': True}
        )
```

The registry calls `enrich_prompt()` on all subscribed plugins in order, passing the result of each to the next.
