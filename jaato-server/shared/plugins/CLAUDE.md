# Plugin Implementation Guide

## Critical: `PLUGIN_KIND` in `__init__.py`

Every plugin **must** declare `PLUGIN_KIND` in its `__init__.py`. Without it, the
plugin is **silently skipped** during directory discovery and never loaded.

```python
# shared/plugins/my_plugin/__init__.py

PLUGIN_KIND = "tool"  # REQUIRED - "tool", "gc", "session", or "model_provider"

from .plugin import MyPlugin, create_plugin

__all__ = ["MyPlugin", "create_plugin", "PLUGIN_KIND"]
```

## Critical: `SESSION_INDEPENDENT` for Auth Plugins

Auth plugins must also declare `SESSION_INDEPENDENT = True` in `__init__.py`.
Without it, the plugin's commands only appear after a session is loaded — but
auth plugins exist to establish credentials *before* connecting to a provider.

```python
# shared/plugins/my_auth/__init__.py

PLUGIN_KIND = "tool"
SESSION_INDEPENDENT = True  # REQUIRED for auth plugins

from .plugin import MyAuthPlugin, create_plugin

__all__ = ["MyAuthPlugin", "create_plugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
```

**Why this matters:** The daemon (`server/__main__.py`) has two command sources:

1. **Session-bound plugins** — discovered per-session, gated behind `session.is_loaded`
2. **Daemon-level plugins** — discovered at daemon startup via `SESSION_INDEPENDENT`

Without `SESSION_INDEPENDENT`, auth commands are invisible until a session exists,
creating a chicken-and-egg problem: users can't authenticate because the auth
command requires a session, but connecting a session may require authentication.

**Why this matters:** `PluginRegistry._discover_via_directory()` (in `registry.py`)
checks every module with:

```python
module_kind = getattr(module, 'PLUGIN_KIND', None)
if module_kind != plugin_kind:
    continue  # silently skipped — no error, no warning
```

Missing `PLUGIN_KIND` means `None != "tool"` → plugin never loads → its user
commands never register → clients send the input to the model as prompt text
instead of executing it as a command. No autocompletion either.

## Two Plugin Patterns

### Pattern 1: Model Tools (e.g., `cli/`, `todo/`, `file_edit/`)

Plugins that provide tools the AI model invokes via function calling.

```python
# plugin.py
class MyToolPlugin:
    @property
    def name(self) -> str:
        return "my_tool"

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(name="my_function", description="...", parameters={...})]

    def get_executors(self) -> Dict[str, Any]:
        return {"my_function": self._execute_my_function}

    def get_user_commands(self) -> List[UserCommand]:
        return []  # No user commands

    def get_auto_approved_tools(self) -> List[str]:
        return []  # Model tools typically require permission
```

### Pattern 2: User Commands Only (e.g., `anthropic_auth/`, `github_auth/`, `zhipuai_auth/`)

Plugins that provide commands users invoke directly (not through the model).

```python
# plugin.py
class MyAuthPlugin:
    @property
    def name(self) -> str:
        return "my_auth"

    def get_tool_schemas(self) -> List[ToolSchema]:
        return []  # No model tools

    def get_executors(self) -> Dict[str, Any]:
        return {"my-auth": lambda args: self.execute_user_command("my-auth", args)}

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name="my-auth",
                description="Manage My Service authentication",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, logout, status, or help",
                        required=True,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_auto_approved_tools(self) -> List[str]:
        return ["my-auth"]  # User commands should be auto-approved

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        if command != "my-auth":
            return []
        actions = [
            CommandCompletion("login", "Authenticate with My Service"),
            CommandCompletion("logout", "Clear stored credentials"),
            CommandCompletion("status", "Show authentication status"),
            CommandCompletion("help", "Show detailed help"),
        ]
        if not args:
            return actions
        if len(args) == 1:
            partial = args[0].lower()
            return [a for a in actions if a.value.startswith(partial)]
        return []

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> str:
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()
        if action_lower == "login":
            return self._cmd_login()
        elif action_lower == "help":
            return self._cmd_help()
        # ...

    def _cmd_help(self) -> HelpLines:
        """Return HelpLines (not str) for pager display."""
        return HelpLines(lines=[
            ("My Auth Command", "bold"),
            ("", ""),
            ("USAGE", "bold"),
            ("    my-auth <action>", ""),
            # ...
        ])
```

## Tool Traits

Tools can declare semantic **traits** on their `ToolSchema` via the `traits` field
(`FrozenSet[str]`). Traits drive cross-cutting behavior (enrichment routing,
permission defaults, etc.) without hardcoding tool names in session or plugin code.

### Currently Defined Traits

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_FILE_WRITER` | `"file_writer"` | Tool writes/modifies files. Result must include `path` (str), `files_modified` (list), or `changes[].file`. Triggers full-JSON enrichment (LSP diagnostics, artifact tracking). |

### How to Declare Traits

```python
from ..model_provider.types import ToolSchema, TRAIT_FILE_WRITER

ToolSchema(
    name="myWriteTool",
    description="...",
    parameters={...},
    traits=frozenset({TRAIT_FILE_WRITER}),
)
```

### How Traits Are Consumed

- **Session** (`jaato_session.py`): Calls `registry.get_tool_traits(tool_name)` to
  decide enrichment strategy. Tools with `TRAIT_FILE_WRITER` get full-JSON
  enrichment (LSP diagnostics, artifact tracking).
- **Enrichment plugins** (LSP, artifact_tracker): Receive all tool results that the
  session routes to them. They extract file paths generically from the result dict
  using the standard keys (`path`, `files_modified`, `changes[].file`).

### Adding a New Trait

1. Add a `TRAIT_*` constant in `shared/plugins/model_provider/types.py` with a
   docstring documenting the contract.
2. Update consumers (session, plugins) to query `registry.get_tool_traits()` for
   the new trait.

## Checklist for New Plugins

1. `__init__.py` has `PLUGIN_KIND = "tool"` (or appropriate kind)
2. `__init__.py` exports `PLUGIN_KIND` in `__all__`
3. `plugin.py` has `create_plugin()` factory function
4. Plugin class implements `ToolPlugin` protocol (see `base.py`)
5. User commands listed in `get_auto_approved_tools()` (prevents permission prompts)
6. `get_command_completions()` implemented for subcommand autocompletion
7. Help command returns `HelpLines` (not `str`) for pager display
8. **Auth plugins:** `__init__.py` has `SESSION_INDEPENDENT = True`
9. **Model providers:** `verify_auth()` works before `initialize()` (no `self._client` access)
10. **File-writing tools:** Declare `traits=frozenset({TRAIT_FILE_WRITER})` and include `path`/`files_modified` in result

## Critical: `verify_auth()` in Model Provider Plugins

Model provider plugins (`shared/plugins/model_provider/<name>/provider.py`) must
implement `verify_auth()` so it works **before `initialize()` is called**.

The runtime calls `verify_auth()` on a **fresh, uninitialized** provider instance
to check if credentials exist before creating a session (see
`jaato_runtime.py:verify_auth()` — "Create a temporary provider instance just for
auth verification. We don't call initialize() yet.").

This means `verify_auth()` must **never** use `self._client` or any state set by
`initialize()`. It should only check whether credentials are available — not
whether they are valid (that happens later during `initialize()` + first request).

```python
# CORRECT — checks credential availability without needing initialized state
def verify_auth(self, allow_interactive=False, on_message=None) -> bool:
    api_key = resolve_api_key() or get_stored_api_key()
    if api_key:
        if on_message:
            on_message("Found API key")
        return True
    if on_message:
        on_message("No credentials found")
    return False
```

```python
# WRONG — crashes with 'NoneType' has no attribute 'messages' because
# self._client is None on an uninitialized provider instance
def verify_auth(self, allow_interactive=False, on_message=None) -> bool:
    self._client.messages.create(...)  # self._client is None!
    return True
```

**Reference:** See `AnthropicProvider.verify_auth()` for the canonical pattern —
it checks PKCE tokens, OAuth tokens, and API keys without touching `self._client`.

## IPC Completion Flow

In IPC (daemon) mode, completions come from two sources:

**Session-independent plugins** (always available):
1. `_discover_daemon_plugins()` at daemon startup scans for `SESSION_INDEPENDENT = True`
2. `_get_command_list()` iterates `self._daemon_plugins` unconditionally
3. Subcommands pre-expanded into `CommandListEvent`

**Session-bound plugins** (only when session loaded):
1. `_get_command_list()` iterates loaded sessions → `registry.list_exposed()`
2. Calls `plugin.get_user_commands()` → `plugin.get_command_completions()`
3. Pre-expanded into same `CommandListEvent`

Deduplication (by command name) ensures no duplicates when both sources provide
the same command.

**Command execution** follows matching priority:
1. Static commands (session.*, tools.*) → daemon handles directly
2. Daemon-level plugins → `_execute_daemon_command()` (no session required)
3. Session plugins → `session_manager.handle_request()` → `server.execute_command()`

If `PLUGIN_KIND` is missing, the plugin is never discovered at either level.
If `SESSION_INDEPENDENT` is missing from an auth plugin, it only works at level 3.
