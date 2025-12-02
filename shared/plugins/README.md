# Plugin Framework

This document explains how to use the jaato plugin framework from a client perspective and how to implement new plugins.

## Overview

The plugin framework provides a way to dynamically discover, load, and manage tool implementations that can be used by the AI model. Plugins are auto-discovered from the `shared/plugins/` directory and can be exposed/unexposed at runtime.

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

### Exposing and Unexposing Plugin Tools

Plugins must be explicitly exposed before their tools can be used by the AI model. This allows fine-grained control over which tools are available.

```python
# Expose a plugin's tools to the model
registry.expose_tool('cli')

# Expose with configuration
registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})

# Check what's exposed
print(registry.list_exposed())  # ['cli']

# Unexpose a plugin's tools
registry.unexpose_tool('cli')

# Expose all discovered plugins' tools
registry.expose_all()

# Unexpose all plugins' tools (cleanup)
registry.unexpose_all()
```

### Getting Tool Declarations and Executors

Once plugins are exposed, you can retrieve their tool declarations (for the AI model) and executors (for running the tools).

```python
# Get FunctionDeclarations for Vertex AI
declarations = registry.get_exposed_declarations()

# Get executor callables
executors = registry.get_exposed_executors()
# Returns: {'tool_name': callable, ...}
```

### Integration with JaatoClient

The recommended way to use plugins is with `JaatoClient`:

```python
from shared import JaatoClient, PluginRegistry, TokenLedger

# Setup
registry = PluginRegistry()
registry.discover()
registry.expose_tool('cli')

# Create and configure client
jaato = JaatoClient()
jaato.connect('my-project', 'us-central1', 'gemini-2.5-flash')
jaato.configure_tools(registry, ledger=TokenLedger())

# Run prompts (SDK manages history internally)
response = jaato.send_message('List files in current directory')

# Multi-turn conversations work automatically
response2 = jaato.send_message('Now show hidden files too')

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

registry = PluginRegistry()
registry.discover()

jaato = JaatoClient()
jaato.connect('my-project', 'us-central1', 'gemini-2.5-flash')

# First session: CLI tools only
registry.expose_tool('cli')
jaato.configure_tools(registry)
response1 = jaato.send_message('List files')
registry.unexpose_tool('cli')

# Second session: MCP tools only (new chat session)
registry.expose_tool('mcp')
jaato.configure_tools(registry)
response2 = jaato.send_message('Search GitHub issues')
registry.unexpose_tool('mcp')

# Third session: Both tools (new chat session)
registry.expose_tool('cli')
registry.expose_tool('mcp')
jaato.configure_tools(registry)
response3 = jaato.send_message('List files and search GitHub')
registry.unexpose_all()
```

---

## Implementing a New Plugin

To create a new plugin, add a Python file to `shared/plugins/` that implements the `ToolPlugin` protocol.

### Plugin Protocol

Every plugin must implement these methods:

```python
from typing import Protocol, List, Dict, Any, Callable, Optional
from google.genai import types

class ToolPlugin(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        ...

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return google-genai FunctionDeclaration objects for this plugin's tools."""
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
```

### Minimal Plugin Example

Here's a minimal plugin that provides a single tool:

```python
# shared/plugins/example.py

from typing import Dict, List, Any, Callable, Optional
from google.genai import types


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

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        return [types.FunctionDeclaration(
            name='example_tool',
            description='An example tool that echoes input',
            parameters_json_schema={
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

3. **FunctionDeclaration Format**: Tool declarations must follow the google-genai schema format:
   ```python
   types.FunctionDeclaration(
       name='tool_name',
       description='What the tool does',
       parameters_json_schema={
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

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(name='tool_a', description='...', parameters_json_schema={...}),
            types.FunctionDeclaration(name='tool_b', description='...', parameters_json_schema={...}),
            types.FunctionDeclaration(name='tool_c', description='...', parameters_json_schema={...}),
        ]

    def get_executors(self) -> Dict[str, Callable]:
        return {
            'tool_a': self._execute_a,
            'tool_b': self._execute_b,
            'tool_c': self._execute_c,
        }
```

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

**Example:**
```python
registry.expose_tool('cli', config={'extra_paths': ['/opt/custom/bin']})
```

### MCP Plugin (`mcp`)

Connects to MCP (Model Context Protocol) servers defined in `.mcp.json` and exposes their tools.

**Configuration:** None (reads from `.mcp.json`)

**Tools:** Dynamic - depends on connected MCP servers

**Example:**
```python
registry.expose_tool('mcp')
# Tools from GitHub MCP server, Atlassian MCP server, etc.
```

---

## File Structure

```
shared/plugins/
├── __init__.py      # Exports PluginRegistry, ToolPlugin
├── base.py          # ToolPlugin Protocol definition
├── registry.py      # PluginRegistry class
├── cli.py           # CLI tool plugin
├── mcp.py           # MCP tool plugin
├── README.md        # This documentation
└── your_plugin.py   # Your custom plugin
```
