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
notification to tell the server about the workspace paths. The dotted key format
(e.g., `pylsp.plugins.jedi.extra_paths`) is automatically converted to a nested
structure that LSP servers expect:

```json
{
  "settings": {
    "pylsp": {
      "plugins": {
        "jedi": {
          "extra_paths": ["/path/to/workspace"]
        }
      }
    }
  }
}
```

### Cross-File Reference Support

For `find_references` to work across multiple files (essential for dependency tracking),
the LSP client implements several mechanisms:

1. **Workspace Folders**: During initialization, the client sends `workspaceFolders` to
   the server, enabling it to index files across the workspace.

2. **Dynamic Folder Addition**: When documents are opened from directories outside the
   initial workspace root, the client sends `workspace/didChangeWorkspaceFolders` to
   add those directories.

3. **Extra Paths Configuration**: For Python servers, `extra_paths` tells Jedi/Pyright
   where to look for module imports.

4. **Document Tracking**: The client tracks open documents and sends appropriate
   `textDocument/didOpen`, `textDocument/didChange`, and `workspace/didChangeWatchedFiles`
   notifications.

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

### Plugin Configuration Reference

Settings live under `plugin_configs.lsp` in a profile:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `config_path` | string | _(unset — falls back to `<workspace>/.lsp.json` then `~/.lsp.json`)_ | Override `.lsp.json` discovery path. |
| `connect_timeout_seconds` | float | `30.0` | Per-server LSP `initialize` handshake timeout. Raise for heavy-init servers — Eclipse JDT LS (jdtls) on Maven / Gradle workspaces typically needs 30-60s; default `15.0` (pre server-version-bump) starves it. Clamped to `[1.0, 300.0]`; out-of-range values are clamped and logged in the trace, not rejected. |
| `diagnostics_max_wait_seconds` | float | `5.0` | Upper bound on the post-`didOpen` / post-`didChange` wait for the server's first `textDocument/publishDiagnostics` batch. The framework awaits a per-URI `asyncio.Event` signalled by the JSON-RPC reader, so calls return AS SOON AS the batch arrives — raising the max costs nothing in the fast-server case. Pre-0.6.134 this was a hard-coded `0.8s` sleep that starved jdtls (Maven cold cache: 3-8s first batch). Clamped to `[0.0, 60.0]`; `0` disables the await entirely (legacy "read cache as-is" behavior). |
| `diagnostics_min_wait_seconds` | float | `0.5` | Floor on the same wait. Even when an early `publishDiagnostics` arrives, we wait at least this long so multi-stage analysis pipelines (parser → compiler → linter) have a chance to deliver their later batches before the cache read. Clamped to `[0.0, diagnostics_max_wait_seconds]`. |
| `diagnostics_convergence_window_seconds` | float | `3.0` | After the first `publishDiagnostics` lands, keep listening for follow-up batches that overwrite the cache. Each follow-up resets the window timer (jdtls's multi-stage cascade may span several re-publishes). Closes the per-URI convergence race where the first publish carries transient errors (e.g. jdtls's intra-project imports still resolving) and a follow-up publish 1–3 s later carries the settled state. Pre-server-0.6.193 behaviour (`0.0`) returns on first publish. Empirical default `3.0` is grounded in the 2026-06-05 instrumented-cascade analysis: 91 adjacent-publish races across 30 distinct `.java` URIs; p50 = 1.46 s, p90 = 17.9 s. The p90 tail is dominated by edit-cycle re-races (the agent re-modified the file, triggering a new `didChange` cycle) — those can't be fixed by a longer convergence window, only by re-calling enrichment, which happens automatically. `3.0` captures the fresh-render cluster without paying the edit-cycle cost. `max_wait` is still a hard ceiling on the total wait. Clamped to `[0.0, 30.0]`. |
| `debug_log_path` | string | `".jaato/logs/lsp_debug.log"` | Path to the plugin's cross-session diagnostic log (append-only). Relative paths resolve against the session's `workspace_path`; absolute paths pass through. Default is workspace-relative so the per-session AppArmor profile composed by `get_apparmor_rules` covers the write. Empty string disables the diagnostic log entirely. Pre-0.6.136 this was hardcoded to `tempfile.gettempdir()/lsp_debug.log` (e.g. `/tmp/lsp_debug.log`) which apparmor-confined runners couldn't write — the failure was misclassified as a config-load error and silently broke the entire LSP enrichment chain. |

Example codegen profile snippet:

```yaml
plugin_configs:
  lsp:
    config_path: "${workspaceRoot}/.lsp.json"
    connect_timeout_seconds: 60.0
    diagnostics_max_wait_seconds: 10.0            # raise for cold Maven workspaces
    diagnostics_min_wait_seconds: 1.0             # let jdtls multi-stage settle
    diagnostics_convergence_window_seconds: 3.0   # wait for jdtls to settle
                                                  # after the first publish; closes
                                                  # the 12→0-errors-in-2.2s race
```

### AppArmor Exec Grants for Configured Servers

Since server 0.6.137, the lsp plugin's `get_apparmor_rules`
classmethod also emits `ix` (inherit-exec) grants for each LSP
server configured in `.lsp.json`. Under PR-148 apparmor
confinement, the runner-side `connect_server` call uses
`asyncio.create_subprocess_exec` to launch the configured
server — without an `ix` grant on the canonical binary path,
the spawn fails with EACCES and `connected_servers` stays
empty.

For each server entry, the composer emits:

1. `<canonical_path> ix,` — resolved via `shutil.which` +
   `os.path.realpath`. Symlinks followed.
2. `<install-dir>/** r,` — derived as the binary's grandparent
   (standard `<install-dir>/bin/<command>` layout). Read-only
   access to bundled plugins, jars, config files.
3. **For Python-wrapper scripts** (`#!/usr/bin/env python3`
   shebang): also emits `<python-interpreter> ix,` because the
   wrapper itself execs the interpreter. Canonical case: jdtls
   ships as a Python wrapper script.

### Data-directory grants (server 0.6.138+)

Some LSP servers persist runtime state to a data directory
specified via a CLI flag. Eclipse JDT LS (jdtls) is the canonical
case — its Python wrapper crashes at `tempfile.gettempdir()` under
apparmor confinement if no writable temp dir is reachable, so
operators pass an explicit `-data <path>` to redirect it.

Since server 0.6.138, the lsp plugin's apparmor composer
auto-detects these flags in each server's `args` and emits matching
rw grants. Recognised flags (covering the common LSP server
conventions):

| Flag | Server |
|---|---|
| `-data <path>` | jdtls |
| `--data-dir <path>` | pyright, several others |
| `--data <path>` | alternate jdtls syntax |

Both space-separated (`-data <path>`) and equals-separated
(`-data=<path>`) forms are recognised. Variable expansion
(`${workspaceRoot}`, `${HOME}`, etc.) at composer time mirrors
the runtime exactly so the granted path always lines up with what
the binary writes to.

**Example `.lsp.json` snippet** to unblock jdtls under apparmor
confinement:

```json
{
  "languageServers": {
    "java": {
      "command": "jdtls",
      "args": ["-data", "${workspaceRoot}/.jaato/jdtls-data"],
      "languageId": "java"
    }
  }
}
```

This causes the composer to emit:

```
<workspace>/.jaato/jdtls-data/    rw,
<workspace>/.jaato/jdtls-data/**  rw,
```

…in the per-session apparmor profile.

**Runtime side (server 0.6.139+, PR-156):** at subprocess spawn
time, the lsp plugin's `connect_server` also auto-injects
`TMPDIR=<resolved_data_path>` into the LSP server's environment.
This sidesteps an upstream **jdtls Python wrapper bug** (line 74 of
`bin/jdtls.py` pre-commit `d871e83`) where the wrapper computes
`tempfile.gettempdir()` *eagerly* as the default value for the
`-data` argparse argument — BEFORE argparse parses CLI input.
Under apparmor confinement that gettempdir() call crashes because
`/tmp` / `/var/tmp` / `/usr/tmp` aren't reachable, even if `-data`
is passed on the command line.

Python's `tempfile.gettempdir()` honors `TMPDIR` first per its
documented precedence chain, so injecting it makes the wrapper's
line 74 succeed and the wrapper continues into normal startup.

If the operator already sets `"env": {"TMPDIR": "..."}` in
`.lsp.json`, the plugin **does NOT override it** — operator-explicit
TMPDIR always wins.

Forward-compatibility: upstream jdtls fixed this in commit
`d871e83` (Oct 2025) by replacing `gettempdir()` with
`$HOME/.cache` on Linux. After that fix lands in packaged releases,
our TMPDIR injection becomes a no-op for jdtls on Linux (the
wrapper no longer reads TMPDIR), so the framework injection is
harmless for any operator who upgrades.

**Limitations:**

- **AppArmor profiles compose at session bootstrap.** Operator
  changes to `.lsp.json` mid-session do NOT update the
  profile — session restart required to pick up new servers.
- **Chained execs beyond one level need operator-supplied
  grants** — addressed by `plugin_configs.lsp.apparmor_extra_rules`
  in the profile YAML (server 0.6.141+, see below).

### Operator-supplied AppArmor rules (`apparmor_extra_rules`)

For chained execs the framework cannot auto-detect (e.g. jdtls's
Python wrapper execs `java` from the system JDK), operators can
supply raw apparmor rules via the **profile YAML**:

```yaml
# .jaato/profiles/_base_codegen.yaml — NOT .lsp.json
plugin_configs:
  lsp:
    apparmor_extra_rules:
      - "/usr/bin/java ix,"
      - "/usr/lib/jvm/** r,"
      - "/etc/java-*/** r,"
```

Each rule is variable-expanded (`${workspaceRoot}`, `${HOME}`,
etc.) and emitted verbatim into the per-session apparmor profile.

#### Critical: trust boundary

This knob lives in **`.jaato/profiles/<name>.yaml`** — not in
the workspace's `.lsp.json`. The framework's apparmor template
denies writes to `.jaato/profiles/**` from BOTH the runner main
thread AND the //child sub-profile (where LLM-driven tool
subprocesses run):

```
audit deny <workspace>/.jaato/profiles/** wlk,
```

So profile YAMLs are operator-only territory: the framework
won't honor runtime tampering. By contrast, `.lsp.json` lives
at the workspace root and IS writable from inside confinement
(it's tenant territory). **Putting `apparmor_extra_rules` in
`.lsp.json` would create a cross-session privilege-escalation
vector** — an LLM-driven tool could inject `["/** rwklix,"]` and
the framework would honor it on the next session. The knob is
deliberately scoped to profile YAML to close this class.

**Verification:** after restart, run `lsp logs` in a session.
A successful exec emits:

```
[java] Connecting to server
[java] Connected successfully
```

A denied exec still surfaces the EACCES message clearly (PR-153
fix kept the error surface honest), which is the operator's
signal to extend the apparmor policy or add a missing server
binary.

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
