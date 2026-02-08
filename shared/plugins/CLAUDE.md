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

## Checklist for New Plugins

1. `__init__.py` has `PLUGIN_KIND = "tool"` (or appropriate kind)
2. `__init__.py` exports `PLUGIN_KIND` in `__all__`
3. `plugin.py` has `create_plugin()` factory function
4. Plugin class implements `ToolPlugin` protocol (see `base.py`)
5. User commands listed in `get_auto_approved_tools()` (prevents permission prompts)
6. `get_command_completions()` implemented for subcommand autocompletion
7. Help command returns `HelpLines` (not `str`) for pager display

## IPC Completion Flow

In IPC (daemon) mode, completions flow through:

1. Server calls `_get_command_list()` in `server/__main__.py`
2. Iterates `registry.list_exposed()` → finds plugin
3. Calls `plugin.get_user_commands()` → gets command declarations
4. Calls `plugin.get_command_completions(cmd.name, [])` → gets subcommands
5. Pre-expands into `CommandListEvent`: `"my-auth login"`, `"my-auth logout"`, etc.
6. Client receives event → registers commands for autocompletion

If any step in 1-2 fails (e.g., missing `PLUGIN_KIND`), the entire chain breaks
silently and the command is treated as prompt text sent to the model.
