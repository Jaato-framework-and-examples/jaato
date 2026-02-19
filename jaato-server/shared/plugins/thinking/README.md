# Thinking Plugin

The Thinking plugin provides a user-facing command (`/thinking`) to control extended thinking/reasoning modes in AI providers. This allows users to dynamically adjust how much "thinking" the model does before responding.

## Features

- **User-Only Control**: The `/thinking` command is explicitly NOT shared with the model - thinking mode is 100% under user control
- **Config-Driven Presets**: Presets are loaded from config files, not hardcoded
- **Dynamic Completions**: Tab completion shows available presets from config
- **Multi-Provider Support**: Works with any provider that supports thinking mode:
  - **Anthropic**: Extended thinking with configurable budget
  - **Google Gemini**: Thinking mode (Gemini 2.0+)

## Quick Start

### Basic Usage

```python
from shared import JaatoClient, PluginRegistry
from shared.plugins.thinking import create_plugin

# Create client and connect
jaato = JaatoClient()
jaato.connect(project="my-project", location="us-central1", model="claude-sonnet-4")

# Set up thinking plugin
thinking_plugin = create_plugin()
thinking_plugin.initialize()
jaato.set_thinking_plugin(thinking_plugin)

# User can now use /thinking command
# The plugin will apply thinking config to the provider
```

### User Commands

The plugin provides a single user command:

| Command | Description |
|---------|-------------|
| `/thinking` | Show current thinking mode status |
| `/thinking off` | Disable thinking |
| `/thinking on` | Enable thinking (default: 10,000 tokens) |
| `/thinking deep` | Enable deep thinking (25,000 tokens) |
| `/thinking ultra` | Enable ultra thinking (100,000 tokens) |
| `/thinking 50000` | Enable with custom token budget |

## Configuration

### Configuration File

Create `.jaato/thinking.json` in your project or `~/.jaato/thinking.json` for user-level defaults:

```json
{
  "default": "off",
  "presets": {
    "off": { "enabled": false, "budget": 0 },
    "on": { "enabled": true, "budget": 10000 },
    "deep": { "enabled": true, "budget": 25000 },
    "ultra": { "enabled": true, "budget": 100000 },
    "custom_preset": { "enabled": true, "budget": 50000 }
  }
}
```

### Config Search Order

1. Explicit path passed to `initialize({"config_path": "..."})`
2. `.jaato/thinking.json` (project-level)
3. `~/.jaato/thinking.json` (user-level)
4. Built-in defaults

### Built-in Default Presets

| Preset | Enabled | Budget |
|--------|---------|--------|
| `off` | false | 0 |
| `on` | true | 10,000 |
| `deep` | true | 25,000 |
| `ultra` | true | 100,000 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                            │
│                                                                     │
│  User types: /thinking deep                                        │
│                                                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ThinkingPlugin                               │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  UserCommand    │    │  Config Loader  │    │  Completions   │  │
│  │ share_with_     │    │                 │    │  (dynamic)     │  │
│  │ model=False     │    │ .jaato/         │    │                │  │
│  │                 │    │ thinking.json   │    │  off, on,      │  │
│  └─────────────────┘    └─────────────────┘    │  deep, ultra   │  │
│                                                └────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         JaatoSession                                │
│                                                                     │
│  set_thinking_config() ──► Provider.set_thinking_config()          │
│                                                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Provider                                 │
│                                                                     │
│  Anthropic: enable_thinking=True, thinking_budget=25000            │
│  Gemini: thinking_mode=True (future)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Why User-Only Control?

The thinking command uses `share_with_model=False` for important reasons:

1. **Prevent Self-Modification**: The model cannot change its own thinking configuration
2. **User Authority**: Users have complete control over resource allocation
3. **Cost Control**: Thinking uses tokens and costs money - users decide the budget
4. **Transparency**: Users always know exactly what thinking mode is active

## Programmatic API

The plugin also provides a programmatic API for non-UI usage:

```python
from shared.plugins.thinking import create_plugin, ThinkingConfig

# Create and initialize
plugin = create_plugin()
plugin.initialize({"config_path": ".jaato/thinking.json"})
plugin.set_session(session)

# Set thinking mode programmatically
plugin.set_thinking_mode("deep")       # Use preset
plugin.set_thinking_mode(50000)        # Custom budget
plugin.set_thinking_mode(0)            # Disable

# Get current config
config = plugin.get_current_config()
print(f"Enabled: {config.enabled}, Budget: {config.budget}")
```

## Integration with Providers

### Anthropic

When thinking is enabled, the Anthropic provider sends requests with:

```python
{
    "model": "claude-sonnet-4-20250514",
    "thinking": {
        "type": "enabled",
        "budget_tokens": 25000
    },
    ...
}
```

The response includes thinking content in `response.thinking`.

### Google Gemini

*Coming soon* - Gemini 2.0+ supports thinking mode but integration is not yet complete.

## Data Models

### ThinkingConfig

```python
@dataclass
class ThinkingConfig:
    enabled: bool = False
    budget: int = 10000
```

### ThinkingPreset

```python
@dataclass
class ThinkingPreset:
    enabled: bool
    budget: int
```

### ThinkingPluginConfig

```python
@dataclass
class ThinkingPluginConfig:
    default: str = "off"
    presets: Dict[str, ThinkingPreset]
```

## File Structure

```
shared/plugins/thinking/
├── __init__.py      # Module exports
├── config.py        # Configuration loading and data models
├── plugin.py        # ThinkingPlugin implementation
└── README.md        # This documentation
```

## Best Practices

1. **Start with defaults**: Begin with `off` or `on` preset, increase as needed
2. **Use presets for consistency**: Define project-specific presets in config
3. **Monitor token usage**: Higher budgets cost more - use `deep`/`ultra` for complex tasks only
4. **Check provider support**: Not all providers support thinking mode

## Troubleshooting

### "Current provider does not support thinking mode"

The provider you're using doesn't have thinking support. Currently only Anthropic fully supports extended thinking.

### Thinking not appearing in responses

1. Check that thinking is enabled: `/thinking` should show `enabled: true`
2. Verify the model supports thinking (Claude 3.5+, Claude 4)
3. Ensure the budget is sufficient for the task

### Config file not loading

1. Check file path: `.jaato/thinking.json` in project root
2. Validate JSON syntax
3. Check file permissions
