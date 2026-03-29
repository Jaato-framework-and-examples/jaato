# Adding Plugin Configuration to Profiles

Profiles can pass per-plugin configuration via the `plugin_configs` field. This is how plugins receive settings that vary by agent role — a researcher profile might configure `web_search` with a specific region, while a developer profile configures `cli` with extra PATH entries.

## Structure

```json
{
  "name": "my-profile",
  "description": "Example profile",
  "plugins": ["cli", "web_search", "references", "permission"],
  "plugin_configs": {
    "cli": {
      "max_output_chars": 20000,
      "extra_paths": ["/usr/local/go/bin"]
    },
    "web_search": {
      "max_results": 5,
      "region": "us-en"
    }
  }
}
```

Each key in `plugin_configs` is a plugin name, and its value is a dict passed to that plugin's `initialize(config)` method.

## How it works

The flow from profile to plugin:

```
Profile JSON
  → plugin_configs: {"cli": {"max_output_chars": 20000}}

Session creation (session_manager.py / subagent plugin)
  → Expands ${VAR} references in config values
  → Passes to PluginRegistry.expose_tool("cli", config={"max_output_chars": 20000})

PluginRegistry
  → Calls plugin.initialize(config)

Plugin.initialize()
  → self._max_output_chars = config.get("max_output_chars", 50000)
```

## Discovering available settings

Every plugin that accepts configuration declares its settings via `get_config_schema()`. You can query them programmatically:

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()

for setting in registry.get_plugin_config_schema("cli"):
    print(f"{setting.name}: {setting.type} = {setting.default}")
    print(f"  {setting.description}")
```

Output:
```
extra_paths: list[str] = []
  Additional PATH entries to prepend
max_output_chars: int = 50000
  Maximum characters to return from command output
auto_background_threshold: float = 10.0
  Seconds before auto-backgrounding
background_max_workers: int = 4
  Maximum concurrent background workers
```

Settings with `choices` restrict valid values:

```python
for setting in registry.get_plugin_config_schema("web_search"):
    if setting.choices:
        print(f"{setting.name}: one of {setting.choices}")
```

Settings with `env_var` can be overridden by environment variables.

## Examples by plugin

### cli

```json
"plugin_configs": {
  "cli": {
    "extra_paths": ["/opt/tools/bin"],
    "max_output_chars": 100000,
    "auto_background_threshold": 30.0,
    "background_max_workers": 2
  }
}
```

### web_search

```json
"plugin_configs": {
  "web_search": {
    "max_results": 20,
    "timeout": 15,
    "region": "uk-en",
    "safesearch": "strict"
  }
}
```

### web_fetch

```json
"plugin_configs": {
  "web_fetch": {
    "timeout": 60,
    "max_length": 200000,
    "cache_ttl": 600
  }
}
```

### references

```json
"plugin_configs": {
  "references": {
    "lookup_strategy": "tags_only",
    "preselected": ["adr-001", "mod-code-015"],
    "transitive_injection": false,
    "exclude_tools": ["selectReferences"]
  }
}
```

### permission

```json
"plugin_configs": {
  "permission": {
    "channel_type": "webhook",
    "channel_config": {
      "endpoint": "https://approvals.internal/api",
      "auth_token": "${APPROVAL_TOKEN}"
    },
    "evaluators": {
      "default": "policies/global_guard.py",
      "cli_based_tool": "policies/cli_guard.py"
    }
  }
}
```

### interactive_shell

```json
"plugin_configs": {
  "interactive_shell": {
    "max_sessions": 4,
    "max_lifetime": 300,
    "max_idle": 120,
    "idle_timeout": 1.0
  }
}
```

### subagent

```json
"plugin_configs": {
  "subagent": {
    "allow_inline": true,
    "inline_allowed_plugins": ["cli", "web_search"],
    "auto_discover_profiles": true,
    "profiles_dir": ".jaato/profiles"
  }
}
```

### notebook

```json
"plugin_configs": {
  "notebook": {
    "default_backend": "kaggle",
    "max_output_length": 50000,
    "sandbox_mode": "strict"
  }
}
```

### GC plugins (via profile `gc` field)

GC plugins are configured via the top-level `gc` field, not `plugin_configs`:

```json
{
  "name": "long-running",
  "gc": {
    "type": "budget",
    "threshold_percent": 80.0,
    "target_percent": 60.0,
    "preserve_recent_turns": 5,
    "notify_on_gc": true
  }
}
```

## Variable expansion

Values in `plugin_configs` support `${VAR}` expansion from the profile's `env` field and from environment variables:

```json
{
  "name": "secured",
  "env": {
    "APPROVAL_TOKEN": "vault://secret/approvals#token"
  },
  "plugin_configs": {
    "permission": {
      "channel_config": {
        "auth_token": "${APPROVAL_TOKEN}"
      }
    }
  }
}
```

Secret URIs (`vault://`, `aws-sm://`, etc.) are resolved at session creation time.

## Adding configuration to your own plugin

If you're building a plugin and want it to accept profile-level configuration:

### 1. Read config in `initialize()`

```python
def initialize(self, config=None):
    config = config or {}
    self._threshold = config.get("threshold", 0.8)
    self._mode = config.get("mode", "passive")
```

### 2. Declare the schema

```python
from jaato_sdk.plugins.base import PluginSetting

def get_config_schema(self):
    return [
        PluginSetting(
            name="threshold",
            type="float",
            default=0.8,
            description="Detection threshold (0.0 to 1.0)",
        ),
        PluginSetting(
            name="mode",
            type="str",
            default="passive",
            description="Operating mode",
            choices=["passive", "active", "strict"],
        ),
    ]
```

### 3. Reference in a profile

```json
{
  "name": "monitored",
  "plugins": ["cli", "my_plugin"],
  "plugin_configs": {
    "my_plugin": {
      "threshold": 0.6,
      "mode": "active"
    }
  }
}
```

The plugin's `initialize()` receives `{"threshold": 0.6, "mode": "active"}` as its config dict.

## Inheritance

When using profile inheritance, `plugin_configs` are deep-merged by plugin name:

```json
// base.json
{
  "name": "base",
  "plugin_configs": {
    "cli": {"max_output_chars": 20000},
    "permission": {"channel_type": "console"}
  }
}

// child.json
{
  "name": "child",
  "inherits": "base",
  "plugin_configs": {
    "cli": {"extra_paths": ["/opt/bin"]}
  }
}
```

Resolved `plugin_configs` for `child`:
```json
{
  "cli": {"max_output_chars": 20000, "extra_paths": ["/opt/bin"]},
  "permission": {"channel_type": "console"}
}
```

Parent keys are inherited. Child keys override on a per-key basis within each plugin's config dict. Conflicts between multiple parents follow the same rules as other profile fields — see `docs/design/profile-inheritance.md`.
