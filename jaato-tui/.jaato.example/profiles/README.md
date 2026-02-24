# Subagent Profiles

Profiles define reusable configurations for specialized subagents. Each profile
specifies which plugins are loaded, which references are preselected, and custom
system instructions.

## How Profiles Work

When jaato spawns a subagent, it can use a profile to configure that agent's
capabilities. The subagent shares the parent's runtime (provider connections,
permissions, token ledger) but gets its own session with the profile's settings.

## Profile Schema

```json
{
  "name": "unique-identifier",
  "description": "When to use this profile and what it does.",
  "plugins": ["cli", "filesystem_query", "references", "todo"],
  "plugin_configs": {
    "references": {
      "preselected": ["reference-id-1", "reference-id-2"],
      "exclude_tools": ["selectReferences"]
    }
  },
  "system_instructions": "Custom instructions for the subagent.",
  "max_turns": 15,
  "auto_approved": false,
  "icon_name": "document",
  "gc": {
    "type": "budget",
    "threshold_percent": 80.0,
    "target_percent": 60.0,
    "pressure_percent": 0,
    "preserve_recent_turns": 5,
    "notify_on_gc": false
  }
}
```

## Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | -- | Unique identifier for the profile |
| `description` | Yes | -- | When/why to use this profile |
| `plugins` | No | `[]` (inherit parent) | List of plugin names to load |
| `plugin_configs` | No | `{}` | Per-plugin configuration overrides |
| `system_instructions` | No | `null` (inherit) | Custom system prompt |
| `model` | No | `null` (inherit) | Override model name |
| `provider` | No | `null` (inherit) | Override provider name |
| `max_turns` | No | `10` | Maximum agentic turns |
| `auto_approved` | No | `false` | Skip permission prompts |
| `icon_name` | No | `null` | Predefined icon identifier |
| `gc` | No | `null` | GC config (recommended for max_turns > 15) |

## Provider and Model Overrides

Profiles can override the provider and model used by the subagent. This lets you
route different tasks to different providers — for example, use a cheap/fast model
for the main session but a stronger model for research subagents.

```json
{
  "name": "my-profile",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  ...
}
```

The resolution priority is:

1. **Profile-level** — `provider` / `model` fields in the profile JSON
2. **Config-level** — `default_provider` / `default_model` in subagent config
3. **Parent session** — inherits from the session that spawned the subagent

If both fields are omitted (or `null`), the subagent inherits whatever provider
and model the parent session uses.

## Common Plugin Sets

- **Research**: `["cli", "filesystem_query", "memory", "references", "todo", "web_search"]`
- **Code modification**: `["artifact_tracker", "cli", "filesystem_query", "lsp", "memory", "references", "template", "todo"]`
- **Validation**: `["cli", "filesystem_query", "lsp", "references", "todo"]`
- **Web research**: `["memory", "todo", "web_search", "web_fetch"]`

## GC Configuration Tips

- For profiles with many tool calls: use `pressure_percent: 0` (continuous mode)
- For short validator runs: use defaults (`pressure_percent: 90`)
- Set `notify_on_gc: false` for `auto_approved` profiles to reduce noise

## Examples

See the included example profiles:

- **analyst-research.json** -- Research and documentation agent
- **researcher-claude.json** -- Research agent with provider/model override
- **validator-tier1-universal.json** -- Basic quality gate validator
