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
| `extraPathsKey` | string | Settings key for extra module paths (for dependency tracking) |

### Extra Paths Configuration

For cross-file dependency tracking to work, the LSP server needs to know about
additional module paths. Configure this using `extraPathsKey` with the server-specific
settings key:

```json
{
  "languageServers": {
    "python": {
      "command": "pylsp",
      "languageId": "python",
      "extraPathsKey": "pylsp.plugins.jedi.extra_paths"
    },
    "python-pyright": {
      "command": "pyright-langserver",
      "args": ["--stdio"],
      "languageId": "python",
      "extraPathsKey": "python.analysis.extraPaths"
    }
  }
}
```

When analyzing file dependencies, the plugin sends a `workspace/didChangeConfiguration`
notification with the configured key to tell the server about the workspace paths.

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

The LSP tools use a **symbol-based API** - just provide the symbol name instead of
line/character positions. This is more natural for AI agents who understand code
semantically, not positionally.

### Symbol-Based Tools

#### lsp_goto_definition

Find where a symbol is defined.

```json
{
  "symbol": "UserService",
  "file_path": "/path/to/file.py"  // optional, helps with disambiguation
}
```

Returns:
```json
[{"file": "/path/to/module.py", "line": 42, "character": 0}]
```

#### lsp_find_references

Find all usages of a symbol across the workspace.

```json
{
  "symbol": "processOrder",
  "include_declaration": true
}
```

Returns:
```json
[
  {"file": "/path/to/service.py", "line": 10, "character": 5},
  {"file": "/path/to/handler.py", "line": 25, "character": 12}
]
```

#### lsp_hover

Get type information and documentation for a symbol.

```json
{
  "symbol": "calculate_total"
}
```

Returns:
```json
{"contents": "def calculate_total(items: List[Item]) -> Decimal\n\nCalculate the total price..."}
```

### Refactoring Tools

These tools modify files. They require explicit approval (not auto-approved).

#### lsp_rename_symbol

Rename a symbol across all files in the workspace.

```json
{
  "symbol": "old_name",
  "new_name": "better_name",
  "apply": false  // default: preview only
}
```

**Preview mode (default):**
```json
{
  "mode": "preview",
  "symbol": "old_name",
  "new_name": "better_name",
  "files_affected": 5,
  "changes": [
    {"file": "/path/to/service.py", "edits": 3},
    {"file": "/path/to/handler.py", "edits": 2}
  ],
  "message": "Would rename 'old_name' to 'better_name' in 5 file(s). Set apply=true to apply."
}
```

**Apply mode (`apply: true`):**
```json
{
  "mode": "applied",
  "symbol": "old_name",
  "new_name": "better_name",
  "success": true,
  "files_modified": ["/path/to/service.py", "/path/to/handler.py"],
  "changes": [
    {"file": "/path/to/service.py", "edits_applied": 3, "lines_before": 100, "lines_after": 100}
  ]
}
```

#### lsp_get_code_actions

Discover available refactoring operations for a code region.

```json
{
  "file_path": "/path/to/file.py",
  "start_line": 10,
  "start_column": 1,
  "end_line": 20,
  "end_column": 1,
  "only_refactorings": true  // optional: filter to refactoring actions only
}
```

Returns:
```json
{
  "actions": [
    {"title": "Extract method", "kind": "refactor.extract", "has_edit": true, "affected_files": 1},
    {"title": "Extract to constant", "kind": "refactor.extract.constant"},
    {"title": "Inline variable", "kind": "refactor.inline"}
  ],
  "count": 3
}
```

Common code action kinds:
- `refactor.extract` - Extract method/function/variable
- `refactor.inline` - Inline variable/function
- `refactor.rewrite` - Rewrite/restructure code
- `quickfix` - Quick fixes for diagnostics
- `source.organizeImports` - Organize imports

#### lsp_apply_code_action

Apply a discovered code action by its title.

```json
{
  "file_path": "/path/to/file.py",
  "start_line": 10,
  "start_column": 1,
  "end_line": 20,
  "end_column": 1,
  "action_title": "Extract method"
}
```

Returns:
```json
{
  "action": "Extract method",
  "success": true,
  "files_modified": ["/path/to/file.py"],
  "changes": [
    {"file": "/path/to/file.py", "edits_applied": 2, "lines_before": 100, "lines_after": 108}
  ]
}
```

**Workflow example:**
1. Call `lsp_get_code_actions` to see available refactorings
2. Choose an action from the list
3. Call `lsp_apply_code_action` with the exact title

### File-Based Tools

#### lsp_get_diagnostics

Get errors and warnings for a file. **RECOMMENDED: Use BEFORE builds for instant feedback.**

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

#### lsp_document_symbols

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

### Query-Based Tools

#### lsp_workspace_symbols

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

## Programmatic API for Plugin Integration

The LSP plugin exposes methods that other plugins can use for cross-plugin integration.

### get_file_dependents(file_path)

Find all files that depend on (reference) a given file. This is useful for:
- Understanding impact of changes before modifying code
- Automatically tracking related artifacts
- Building dependency graphs

```python
# Get the LSP plugin from registry
lsp_plugin = registry.get_plugin("lsp")

# Find files that import/reference api.py
dependents = lsp_plugin.get_file_dependents("/path/to/api.py")
# Returns: ["/path/to/handler.py", "/path/to/tests/test_api.py", ...]
```

**How it works:**
1. Gets all document symbols from the file (via `textDocument/documentSymbol`)
2. Filters to "exportable" symbol kinds: Class, Function, Method, Enum, Interface, Constant, Struct, Module
3. For each symbol, finds all references across the workspace (via `textDocument/references`)
4. Returns deduplicated list of files containing those references

**Integration with Artifact Tracker:**

The artifact tracker plugin uses this method to automatically discover dependencies when files are modified:

```
File A.py modified via updateFile
         │
         ▼
┌─────────────────────────────────────────────┐
│  LSP Plugin (priority 30)                    │
│  • Runs diagnostics                          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Artifact Tracker (priority 50)              │
│  • Calls lsp.get_file_dependents("A.py")     │
│  • Flags dependent files for review          │
│  • Shows notification to user                │
└─────────────────────────────────────────────┘
```

User sees:
```
  ╭ result ← lsp: checked A.py, no issues found
  ╰ result ← artifact_tracker: flagged for review: B.py, C.py
```

## Language Server Capabilities

Different language servers support different refactoring operations:

| Server | Rename | Extract Method | Inline | Organize Imports |
|--------|--------|----------------|--------|------------------|
| pyright (Python) | Yes | Limited | No | Yes |
| pylsp (Python) | Yes | Yes (via rope) | Yes | Yes |
| jdtls (Java) | Yes | Yes | Yes | Yes |
| typescript-language-server | Yes | Yes | Yes | Yes |
| gopls (Go) | Yes | Yes | Limited | Yes |
| rust-analyzer | Yes | Yes | Yes | Yes |
| clangd (C/C++) | Yes | Limited | Limited | Yes |

Use `lsp status` to see the capabilities of connected servers.

## Limitations

1. **Workspace scope**: Most LSP features work within a single workspace/project
2. **Document sync**: Files must be "opened" before some features work
3. **Server-specific**: Capabilities vary by language server implementation
4. **Diagnostics**: Received asynchronously via notifications (may have slight delay)
5. **Refactoring scope**: Extract method and similar refactorings depend on server support
