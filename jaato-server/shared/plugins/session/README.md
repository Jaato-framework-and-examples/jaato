# Session Persistence Plugin

The session plugin provides conversation persistence, allowing users to save and resume sessions across client restarts.

## Features

- **Save/Resume sessions** - Persist conversation history to JSON files
- **Timestamp-based naming** - Sessions identified by creation timestamp (e.g., `20251207_143022`)
- **Model-generated descriptions** - Automatic prompt enrichment requests brief descriptions after N turns
- **Auto-save on exit** - Configurable automatic saving when client exits cleanly
- **Checkpoint saves** - Optional saves every N turns
- **Session cleanup** - Automatic removal of oldest sessions when limit exceeded

## User Commands

The plugin contributes five user commands to the interactive client:

| Command | Description |
|---------|-------------|
| `save` | Save the current session for later resumption |
| `resume` | Resume a previously saved session (lists available if no ID given) |
| `sessions` | List all available saved sessions with metadata |
| `delete-session <id>` | Delete a saved session by ID |
| `backtoturn <id>` | Revert conversation to a specific turn (use `history` to see turn IDs) |

### Reverting to a Previous Turn

The `backtoturn` command allows you to undo recent conversation turns and go back to a specific point:

```
> history
============================================================
  Conversation History: 8 message(s), 3 turn(s)
  Tip: Use 'backtoturn <turn_id>' to revert to a specific turn
============================================================

────────────────────────────────────────────────────────────
  ▶ TURN 1
────────────────────────────────────────────────────────────
  [USER]
  Help me debug the auth issue

  [MODEL]
  I'll help you investigate...

────────────────────────────────────────────────────────────
  ▶ TURN 2
────────────────────────────────────────────────────────────
  [USER]
  Check the token refresh
  ...

> backtoturn 2
Reverted to turn 2 (removed 1 turn(s)).
```

This is useful when:
- The model went in a wrong direction and you want to try a different approach
- You want to undo accidental tool executions
- You need to replay from a specific point with different input

## Model Tool

The plugin provides one model tool:

| Tool | Description |
|------|-------------|
| `session_describe` | Set a brief description for the current session (called via prompt enrichment) |

## Configuration

Configuration is loaded from `.jaato/.sessions.json`:

```json
{
  "storage_path": ".jaato/sessions",
  "auto_save_on_exit": true,
  "auto_save_interval": null,
  "checkpoint_after_turns": 10,
  "auto_resume_last": false,
  "request_description_after_turns": 3,
  "max_sessions": 20
}
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `storage_path` | string | `.jaato/sessions` | Directory for session files |
| `auto_save_on_exit` | bool | `true` | Save session on clean shutdown |
| `auto_save_interval` | int/null | `null` | Auto-save interval in seconds (disabled if null) |
| `checkpoint_after_turns` | int/null | `null` | Save checkpoint every N turns |
| `auto_resume_last` | bool | `false` | Automatically resume last session on connect |
| `request_description_after_turns` | int | `3` | Request model description after N turns |
| `max_sessions` | int | `20` | Maximum sessions to keep (oldest deleted) |

## Session File Format

Sessions are stored as JSON files named by timestamp:

```
.jaato/sessions/
├── 20251207_143022.json
├── 20251207_150815.json
└── 20251208_091230.json
```

Each file contains:

```json
{
  "version": "1.0",
  "session_id": "20251207_143022",
  "description": "Debugging auth refresh issue",
  "created_at": "2025-12-07T14:30:22",
  "updated_at": "2025-12-07T15:45:00",
  "turn_count": 12,
  "turn_accounting": [
    {"prompt": 150, "output": 300, "total": 450},
    ...
  ],
  "connection": {
    "project": "my-gcp-project",
    "location": "us-central1",
    "model": "gemini-2.5-flash"
  },
  "history": [
    {
      "role": "user",
      "parts": [{"type": "text", "text": "Help me debug..."}]
    },
    {
      "role": "model",
      "parts": [{"type": "text", "text": "I'll help..."}]
    },
    ...
  ]
}
```

## Architecture

The session plugin follows the same pattern as the GC plugin - it's not managed by `PluginRegistry` but connects directly to `JaatoClient`:

```
┌──────────────────┐
│  JaatoClient     │
│  ┌────────────┐  │
│  │ Session    │◄─┼─── set_session_plugin(plugin, config)
│  │ Plugin     │  │
│  └────────────┘  │
│  save_session()  │
│  resume_session()│
│  list_sessions() │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ FileSessionPlugin│
│  ├─ save()       │
│  ├─ load()       │
│  ├─ list()       │
│  └─ hooks        │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ .jaato/sessions/ │
│  └─ *.json       │
└──────────────────┘
```

## Usage Example

### Basic Setup

```python
from shared import JaatoClient
from shared.plugins.session import create_plugin, load_session_config

# Create client
client = JaatoClient()
client.connect(project_id, location, model)
client.configure_tools(registry)

# Set up session plugin
config = load_session_config()
session_plugin = create_plugin()
session_plugin.initialize({'storage_path': config.storage_path})
client.set_session_plugin(session_plugin, config)
```

### Manual Save/Resume

```python
# Save current session
session_id = client.save_session()
print(f"Saved: {session_id}")

# Later, resume that session
state = client.resume_session(session_id)
print(f"Resumed: {state.turn_count} turns")

# Or list available sessions
sessions = client.list_sessions()
for s in sessions:
    print(f"{s.session_id} - {s.description}")
```

### Auto-Save on Exit

With `auto_save_on_exit: true`, sessions are automatically saved when `client.close_session()` is called (typically on clean shutdown).

## Prompt Enrichment

After `request_description_after_turns` turns, the plugin enriches prompts with a request for the model to provide a session description:

```
[User's message...]

[System: This conversation has been ongoing for a while. Please provide
a brief 3-5 word description summarizing its main topic by calling the
session_describe tool. This is for session management only and won't
interrupt the conversation flow.]
```

The model then calls `session_describe(description="Debugging auth issue")` to set the description.

## Lifecycle Hooks

The plugin provides lifecycle hooks called by JaatoClient:

| Hook | Called When | Purpose |
|------|-------------|---------|
| `on_session_start(config)` | Client connects | Auto-resume if configured |
| `on_turn_complete(state, config)` | After each turn | Checkpoint saves |
| `on_session_end(state, config)` | Client closes | Auto-save on exit |

## History Serialization

The plugin serializes `google.genai.types.Content` objects to JSON:

- **Text parts** → `{"type": "text", "text": "..."}`
- **Function calls** → `{"type": "function_call", "name": "...", "args": {...}}`
- **Function responses** → `{"type": "function_response", "name": "...", "response": {...}}`

This allows full history restoration including tool calls and responses.
