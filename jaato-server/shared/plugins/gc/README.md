# Context Garbage Collection (GC) Plugins

Context GC plugins manage conversation history to prevent context window overflow. They implement different strategies for freeing context space while preserving important information.

## Architecture

The GC system follows a plugin architecture similar to Java's garbage collector:

- **GCPlugin Protocol**: Defines the interface all GC strategies must implement
- **GCConfig**: Configuration for thresholds and preservation settings
- **GCResult**: Results of a GC operation including tokens freed
- **GCTriggerReason**: Why GC was triggered (threshold, turn limit, manual, etc.)

## Available Strategies

### gc_truncate (Simple Truncation)

The simplest and fastest strategy. Removes oldest turns entirely.

```python
from shared.plugins.gc_truncate import create_plugin
from shared.plugins.gc import GCConfig

plugin = create_plugin()
plugin.initialize({
    "preserve_recent_turns": 10,  # Keep last 10 turns
    "notify_on_gc": True          # Inject notification into history
})

client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
```

**Best for**: Fast execution, when old context is not valuable.

### gc_summarize (Summarization)

Compresses old turns into a summary. Requires a summarizer function.

```python
from shared.plugins.gc_summarize import create_plugin
from shared.plugins.gc import GCConfig

def my_summarizer(conversation: str) -> str:
    # Use your model to generate summary
    return model.generate(f"Summarize this conversation:\n{conversation}")

plugin = create_plugin()
plugin.initialize({
    "preserve_recent_turns": 10,
    "summarizer": my_summarizer,  # Required
    "notify_on_gc": True
})

client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
```

**Best for**: Preserving context information while freeing space.

### gc_hybrid (Generational)

Combines truncation and summarization like Java's generational GC:
- **Recent turns**: Always preserved intact
- **Middle turns**: Summarized (if summarizer provided)
- **Ancient turns**: Truncated

```python
from shared.plugins.gc_hybrid import create_plugin
from shared.plugins.gc import GCConfig

plugin = create_plugin()
plugin.initialize({
    "preserve_recent_turns": 5,    # Keep last 5 intact
    "summarize_middle_turns": 15,  # Summarize next 15
    "summarizer": my_summarizer    # Optional
})

client.set_gc_plugin(plugin, GCConfig(threshold_percent=75.0))
```

**Best for**: Balance between speed and context preservation.

## Configuration

### GCConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold_percent` | float | 80.0 | Trigger GC when context usage exceeds this %. Can be overridden via `JAATO_GC_THRESHOLD` env var. |
| `max_turns` | int | None | Trigger GC when turn count exceeds this |
| `auto_trigger` | bool | True | Enable automatic GC triggering |
| `check_before_send` | bool | True | Check GC before each send_message() |
| `preserve_recent_turns` | int | 5 | Default recent turns to preserve |
| `pinned_turn_indices` | List[int] | [] | Turn indices to never remove |

### Plugin Config Options

Each plugin accepts additional options via `initialize()`:

| Option | Plugins | Description |
|--------|---------|-------------|
| `preserve_recent_turns` | all | Override GCConfig's preserve count |
| `notify_on_gc` | all | Inject notification into history |
| `notification_template` | all | Custom notification message |
| `summarizer` | summarize, hybrid | Function to generate summaries |
| `summarize_middle_turns` | hybrid | Turns to summarize (not truncate) |

## Configuration Methods

GC can be configured in three ways:

### 1. File-Based Configuration (Main Agent)

Create `.jaato/gc.json` in your project root:

```json
{
  "type": "hybrid",
  "threshold_percent": 80.0,
  "preserve_recent_turns": 5,
  "notify_on_gc": true,
  "summarize_middle_turns": 10,
  "max_turns": null,
  "plugin_config": {}
}
```

The rich client automatically loads this file on startup. For custom clients:

```python
from shared.plugins.gc import load_gc_from_file

result = load_gc_from_file()  # Defaults to .jaato/gc.json
if result:
    gc_plugin, gc_config = result
    client.set_gc_plugin(gc_plugin, gc_config)
```

### 2. Profile-Based Configuration (Subagents)

Add a `gc` field to subagent profile JSON files in `.jaato/profiles/`:

```json
{
  "name": "my-subagent",
  "description": "A specialized subagent",
  "plugins": ["cli", "file_edit"],
  "gc": {
    "type": "hybrid",
    "threshold_percent": 80.0,
    "preserve_recent_turns": 5,
    "notify_on_gc": true,
    "summarize_middle_turns": 10
  }
}
```

The subagent plugin automatically initializes GC when spawning agents with this configuration.

### 3. Inline Configuration (Ad-Hoc Subagents)

For quick testing or one-off subagents, use `inline_config.gc` when spawning:

```python
spawn_subagent(
    task="Stress test GC behavior",
    inline_config={
        "gc": {
            "type": "truncate",
            "threshold_percent": 5.0,  # Trigger early for testing
            "preserve_recent_turns": 3
        }
    }
)
```

This is useful for:
- **Testing GC strategies**: Use a low threshold (e.g., 5%) to trigger GC quickly
- **Per-task tuning**: Different tasks may need different context management
- **Debugging**: Observe GC behavior without modifying profile files

Available `inline_config.gc` options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | string | `"truncate"` | GC strategy: `"truncate"`, `"summarize"`, or `"hybrid"` |
| `threshold_percent` | float | 80.0 | Trigger GC at this context usage % |
| `preserve_recent_turns` | int | 5 | Turns to always preserve |
| `notify_on_gc` | bool | true | Inject notification after GC |
| `summarize_middle_turns` | int | null | For hybrid: turns to summarize |
| `max_turns` | int | null | Trigger GC at turn count |
| `plugin_config` | object | {} | Plugin-specific options |

### 4. Programmatic Configuration

For direct control in custom clients:

```python
from shared.jaato_client import JaatoClient
from shared.plugins.gc import GCConfig, load_gc_plugin

# Create client
client = JaatoClient()
client.connect(project, location, model)

# Load and set GC plugin
gc_plugin = load_gc_plugin('gc_truncate', {
    'preserve_recent_turns': 10
})
client.set_gc_plugin(gc_plugin, GCConfig(
    threshold_percent=75.0,
    check_before_send=True
))

# Use client normally - GC triggers automatically
response = client.send_message("Hello")

# Manual GC
result = client.manual_gc()
print(f"Freed {result.tokens_freed} tokens")

# View GC history
for result in client.get_gc_history():
    print(f"{result.plugin_name}: {result.items_collected} items collected")
```

## Plugin Discovery

GC plugins are registered via entry points and can be discovered dynamically:

```python
from shared.plugins.gc import discover_gc_plugins, load_gc_plugin

# List available plugins
plugins = discover_gc_plugins()
print(plugins.keys())  # ['gc_truncate', 'gc_summarize', 'gc_hybrid']

# Load by name
plugin = load_gc_plugin('gc_truncate', {'preserve_recent_turns': 10})
```

## Creating Custom GC Plugins

Implement the `GCPlugin` protocol:

```python
from shared.plugins.gc import GCPlugin, GCConfig, GCResult, GCTriggerReason

class MyCustomGCPlugin:
    @property
    def name(self) -> str:
        return "gc_custom"

    def initialize(self, config=None):
        self._config = config or {}

    def shutdown(self):
        pass

    def should_collect(self, context_usage, config):
        percent = context_usage.get('percent_used', 0)
        if percent >= config.threshold_percent:
            return True, GCTriggerReason.THRESHOLD
        return False, None

    def collect(self, history, context_usage, config, reason):
        # Implement your GC logic
        new_history = ...
        result = GCResult(
            success=True,
            items_collected=...,
            tokens_before=...,
            tokens_after=...,
            plugin_name=self.name,
            trigger_reason=reason
        )
        return new_history, result

def create_plugin():
    return MyCustomGCPlugin()
```

Register in `pyproject.toml`:

```toml
[project.entry-points."jaato.gc_plugins"]
gc_custom = "my_package:create_plugin"
```
