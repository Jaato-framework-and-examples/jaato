# LSP Plugin for Jaato

The LSP (Language Server Protocol) plugin provides semantic code intelligence tools
for AI agents, enabling accurate code navigation, symbol lookup, and diagnostics.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Model / Agent                        │
│   Uses tools: lsp_goto_definition, lsp_find_references...  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LSPToolPlugin                          │
│  • Tool schemas for model consumption                       │
│  • User commands (lsp status, lsp connect, etc.)           │
│  • Executor methods bridging sync→async                     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │      Background Thread        │
              │   (asyncio event loop)        │
              └───────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   LSPClient     │  │   LSPClient     │  │   LSPClient     │
│   (Python)      │  │   (TypeScript)  │  │   (Go)          │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  pyright-       │  │  typescript-    │  │    gopls        │
│  langserver     │  │  language-      │  │                 │
│  (subprocess)   │  │  server         │  │  (subprocess)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Architecture

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `LSPToolPlugin` | `plugin.py` | Main plugin class implementing `ToolPlugin` protocol |
| `LSPClient` | `lsp_client.py` | Async JSON-RPC client for LSP communication |
| Data Types | `lsp_client.py` | `Position`, `Range`, `Location`, `Diagnostic`, etc. |

### Threading Model

The plugin uses a background thread with an asyncio event loop to handle
the inherently async LSP protocol while exposing synchronous tool executors:

```
Main Thread (sync)          Background Thread (async)
─────────────────          ──────────────────────────

executor() called
    │
    ├──→ request_queue ──→ await client.method()
    │                           │
    │                           ▼
    │                      LSP Server (subprocess)
    │                           │
    │                           ▼
    ◄── response_queue ◄── result
    │
    ▼
return result
```

### LSP Client

The `LSPClient` class implements the Language Server Protocol:

```python
class LSPClient:
    """Async LSP client using JSON-RPC over stdio."""

    async def start() -> None:
        """Start server process and initialize."""

    async def stop() -> None:
        """Shutdown and terminate server."""

    # Document synchronization
    async def open_document(path, text=None) -> None
    async def close_document(path) -> None

    # Core LSP methods
    async def goto_definition(path, line, char) -> List[Location]
    async def find_references(path, line, char) -> List[Location]
    async def hover(path, line, char) -> Optional[Hover]
    async def get_completions(path, line, char) -> List[CompletionItem]
    async def get_document_symbols(path) -> List[SymbolInformation]
    async def workspace_symbols(query) -> List[SymbolInformation]
    async def rename(path, line, char, new_name) -> Dict[str, Any]

    # Diagnostics (received via notifications)
    def get_diagnostics(path) -> List[Diagnostic]
```

### JSON-RPC Protocol

LSP uses JSON-RPC 2.0 over stdio with Content-Length headers:

```
→ Content-Length: 123\r\n
→ \r\n
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}

← Content-Length: 456\r\n
← \r\n
← {"jsonrpc":"2.0","id":1,"result":{"capabilities":{...}}}
```

## Configuration

Create `.lsp.json` in your project root or home directory:

```json
{
  "languageServers": {
    "python": {
      "command": "pyright-langserver",
      "args": ["--stdio"],
      "languageId": "python",
      "autoConnect": true
    },
    "typescript": {
      "command": "typescript-language-server",
      "args": ["--stdio"],
      "languageId": "typescript",
      "env": {
        "NODE_OPTIONS": "--max-old-space-size=4096"
      }
    },
    "rust": {
      "command": "rust-analyzer",
      "args": [],
      "languageId": "rust",
      "rootUri": "file:///path/to/project"
    }
  }
}
```

### Configuration Options

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Executable path or command name |
| `args` | string[] | Command-line arguments |
| `languageId` | string | Language identifier (python, typescript, etc.) |
| `env` | object | Additional environment variables |
| `rootUri` | string | Workspace root URI (defaults to cwd) |
| `autoConnect` | boolean | Connect on plugin initialization (default: true) |

### Configuration Search Order

1. Custom path from `plugin_configs` (see below)
2. `.lsp.json` in current working directory
3. `~/.lsp.json` in home directory

### Using with Subagent Profiles

You can specify a custom `.lsp.json` path per subagent using `plugin_configs`:

```json
// .jaato/profiles/cobol_agent.json
{
  "name": "cobol_agent",
  "description": "COBOL development agent with z/OS tooling",
  "plugins": ["lsp", "mcp", "cli"],
  "plugin_configs": {
    "lsp": {
      "config_path": "/projects/mainframe/.lsp.json"
    },
    "mcp": {
      "config_path": "/projects/mainframe/.mcp.json"
    }
  },
  "system_instructions": "You are a COBOL expert working with IBM z/OS systems."
}
```

This allows different subagents to use different LSP server configurations
for the same codebase or different projects.

## Tools

### lsp_goto_definition

Navigate to where a symbol is defined.

```json
{
  "file_path": "/path/to/file.py",
  "line": 10,
  "character": 5
}
```

Returns:
```json
[{"file": "/path/to/module.py", "line": 42, "character": 0}]
```

### lsp_find_references

Find all usages of a symbol across the workspace.

```json
{
  "file_path": "/path/to/file.py",
  "line": 10,
  "character": 5,
  "include_declaration": true
}
```

Returns:
```json
[
  {"file": "/path/to/file.py", "line": 10, "character": 5},
  {"file": "/path/to/other.py", "line": 25, "character": 12}
]
```

### lsp_hover

Get type information and documentation for a symbol.

```json
{
  "file_path": "/path/to/file.py",
  "line": 10,
  "character": 5
}
```

Returns:
```json
{"contents": "def my_function(x: int) -> str\n\nDocstring here..."}
```

### lsp_get_diagnostics

Get errors and warnings for a file.

```json
{
  "file_path": "/path/to/file.py"
}
```

Returns:
```json
[
  {
    "severity": "Error",
    "message": "Cannot find name 'undefined_var'",
    "line": 15,
    "character": 8,
    "source": "pyright"
  }
]
```

### lsp_document_symbols

List all symbols defined in a file.

```json
{
  "file_path": "/path/to/file.py"
}
```

Returns:
```json
[
  {"name": "MyClass", "kind": "Class", "location": "/path/to/file.py:10"},
  {"name": "my_function", "kind": "Function", "location": "/path/to/file.py:50"}
]
```

### lsp_workspace_symbols

Search for symbols across the entire workspace.

```json
{
  "query": "MyClass"
}
```

Returns:
```json
[
  {"name": "MyClass", "kind": "Class", "location": "/path/to/file.py:10"},
  {"name": "MyClassHelper", "kind": "Class", "location": "/path/to/utils.py:5"}
]
```

### lsp_rename_symbol

Get workspace edits for renaming a symbol.

```json
{
  "file_path": "/path/to/file.py",
  "line": 10,
  "character": 5,
  "new_name": "better_name"
}
```

Returns a workspace edit object describing changes needed.

## User Commands

| Command | Description |
|---------|-------------|
| `lsp list` | List configured language servers |
| `lsp status` | Show connection status and capabilities |
| `lsp connect <name>` | Connect to a configured server |
| `lsp disconnect <name>` | Disconnect from a running server |
| `lsp reload` | Reload configuration and reconnect |
| `lsp logs [server\|clear]` | Show or clear interaction logs |

## Popular Language Servers

| Language | Server | Install |
|----------|--------|---------|
| Python | pyright | `npm install -g pyright` |
| TypeScript/JS | typescript-language-server | `npm install -g typescript-language-server typescript` |
| Rust | rust-analyzer | [rust-analyzer.github.io](https://rust-analyzer.github.io/) |
| Go | gopls | `go install golang.org/x/tools/gopls@latest` |
| C/C++ | clangd | [clangd.llvm.org](https://clangd.llvm.org/) |
| Java | jdtls | [Eclipse JDT LS](https://github.com/eclipse/eclipse.jdt.ls) |

## Comparison with MCP

| Aspect | MCP Plugin | LSP Plugin |
|--------|------------|------------|
| Protocol | Model Context Protocol | Language Server Protocol |
| Purpose | General tool execution | Code intelligence |
| Servers | Custom MCP servers | Existing LSP ecosystem (100+) |
| Config file | `.mcp.json` | `.lsp.json` |
| Use case | Domain-specific tools | Semantic code navigation |

## Position Indexing

**Important:** LSP uses 0-indexed positions:

- Line 1 in your editor = `line: 0` in LSP
- Column 1 in your editor = `character: 0` in LSP

The tools return 1-indexed values for human readability (matching editor display).

## Error Handling

If no LSP servers are connected, tools return:
```json
{"error": "No LSP servers connected. Use 'lsp connect <server>' first."}
```

Connection failures are logged and accessible via `lsp logs`.

## Extension Points

The plugin follows jaato's plugin architecture:

```python
from shared.plugins.lsp import LSPToolPlugin, create_plugin

# Create via factory
plugin = create_plugin()

# Or instantiate directly
plugin = LSPToolPlugin()
plugin.initialize(config={"autoConnect": False})

# Access tool schemas
schemas = plugin.get_tool_schemas()

# Get executors for integration
executors = plugin.get_executors()
```

## Limitations

1. **Workspace scope**: Most LSP features work within a single workspace/project
2. **Document sync**: Files must be "opened" before some features work
3. **Server-specific**: Capabilities vary by language server implementation
4. **Diagnostics**: Received asynchronously via notifications (may have slight delay)
