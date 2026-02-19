# Sandbox Manager Plugin

This plugin provides runtime management of filesystem sandbox permissions through user commands.

## Overview

The `sandbox` command allows users to dynamically grant or revoke filesystem access during a session. It operates on a three-tiered configuration model where session-level settings have the highest precedence.

## Configuration Levels

| Level | Path | Precedence | Persistence |
|-------|------|------------|-------------|
| **Global** | `~/.jaato/sandbox_paths.json` | Lowest | User-managed |
| **Workspace** | `<workspace>/.jaato/sandbox.json` | Medium | Project-managed |
| **Session** | `<workspace>/.jaato/sessions/<id>/sandbox.json` | Highest | Runtime-managed |

The `sandbox list` command shows the merged, effective view from all three levels.

## User Commands

### `sandbox list`

Displays all currently active sandbox paths from all configuration levels.

```
Effective Sandbox Paths:

Path                              Action   Source
/opt/company_tools                ALLOW    global
/var/www/project/assets           ALLOW    workspace
/tmp/temp_data                    ALLOW    session
/home/user/sensitive              DENY     session
```

### `sandbox add <path>`

Grants access to a path for the current session.

- Adds `<path>` to `allowed_paths` in session config
- Removes from `denied_paths` if present
- Takes effect immediately

### `sandbox remove <path>`

Blocks access to a path for the current session, even if allowed at global/workspace level.

- Adds `<path>` to `denied_paths` in session config
- Removes from `allowed_paths` if present
- Takes effect immediately

## Configuration File Format

All three configuration files use the same JSON format:

```json
{
  "allowed_paths": [
    "/path/to/allow",
    {"path": "/another/path", "added_at": "2024-01-15T10:30:00Z"}
  ],
  "denied_paths": [
    "/path/to/deny"
  ]
}
```

Both simple strings and objects with metadata are supported.

## Registry Integration

The plugin integrates with `PluginRegistry` for path validation:

1. **On initialization**: Loads all config files and syncs to registry
2. **Allowed paths**: Registered via `registry.authorize_external_path()`
3. **Denied paths**: Registered via `registry.deny_external_path()`
4. **Precedence**: Session denials override workspace/global allows

Path validation in `sandbox_utils.check_path_with_jaato_containment()` checks denied paths first, so denial always takes precedence.

## Initialization

The plugin requires both workspace path and session ID to be set:

```python
from shared.plugins.sandbox_manager import create_plugin

plugin = create_plugin()
plugin.initialize({"session_id": "my-session-123"})
plugin.set_workspace_path("/path/to/workspace")
```

When used through the registry, these are set automatically via auto-wiring.

## Plugin Protocol

| Method | Returns |
|--------|---------|
| `get_tool_schemas()` | `[]` (no model tools) |
| `get_user_commands()` | `[UserCommand("sandbox", ...)]` |
| `get_auto_approved_tools()` | `["sandbox"]` |
| `get_system_instructions()` | `None` |

This is a **user-command-only plugin** - it provides commands for users to invoke directly, not tools for the model to call.
