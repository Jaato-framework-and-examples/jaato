# Waypoint Plugin

**Mark your journey, return when needed.**

## The Journey Metaphor

Every coding session is a journey - you and the model exploring solutions together,
making discoveries, sometimes taking wrong turns. Waypoints let you mark significant
moments along this journey, creating safe points you can return to if the path ahead
becomes treacherous.

Unlike version control which captures code state, waypoints capture the full context
of your collaboration - both the code changes *and* the conversation that led to them.

## How It Works

When you create a waypoint, jaato captures:
- **Code state**: All files that have been modified are tracked via backups
- **Conversation state**: The full conversation history at that moment

When you restore to a waypoint, you can choose to restore:
- **Code only**: Revert files while keeping the current conversation
- **Conversation only**: Rewind the conversation while keeping file changes
- **Both**: Return completely to the waypoint state

## Usage

```
waypoint                         # List all waypoints
waypoint create                  # Create with auto-generated description
waypoint create "before refactor" # Create with custom description
waypoint restore w1              # Restore both code and conversation
waypoint restore w1 code         # Restore code only
waypoint restore w1 conversation # Restore conversation only
waypoint delete w1               # Delete a waypoint
waypoint delete all              # Delete all user waypoints
waypoint info w1                 # Show detailed waypoint info
```

## Waypoint IDs

- `w0` - The implicit initial waypoint (session start), cannot be deleted
- `w1`, `w2`, `w3`, ... - User-created waypoints, numbered sequentially

## How Code Restoration Works

The waypoint system uses a clever approach that avoids duplicating backups:

1. When you create a waypoint, the BackupManager is notified to tag all future
   backups with that waypoint ID (the "diverged_from" field)

2. When you edit a file, a backup is created *before* the edit. This backup
   contains the file's state AT the current waypoint (before diverging)

3. When you restore to a waypoint, jaato finds all backups tagged as having
   diverged from that waypoint. These backups contain the file states that
   existed at the waypoint.

This means:
- No extra disk usage at waypoint creation time
- Backups are only created when files are actually edited
- Each backup knows which waypoint it diverged from

## Configuration

The plugin requires access to the `BackupManager` from the `file_edit` plugin:

```python
from shared.plugins.waypoint import create_plugin

plugin = create_plugin()
plugin.initialize({
    "backup_manager": backup_manager,  # Required
    "storage_path": Path(".jaato/waypoints.json"),  # Optional
})
```

For conversation restoration, session callbacks must be set:

```python
plugin.set_session_callbacks(
    get_history=session.get_history,
    set_history=session.reset_session,
    serialize_history=serialize_history,
    deserialize_history=deserialize_history,
)
```

## API Reference

### Commands

| Command | Description |
|---------|-------------|
| `waypoint` | List all waypoints with their IDs and descriptions |
| `waypoint create` | Create a new waypoint (auto-generates description) |
| `waypoint create "desc"` | Create with custom description |
| `waypoint restore <id>` | Restore to waypoint (both code and conversation) |
| `waypoint restore <id> code` | Restore code only |
| `waypoint restore <id> conversation` | Restore conversation only |
| `waypoint delete <id>` | Delete a specific waypoint |
| `waypoint delete all` | Delete all user-created waypoints |
| `waypoint info <id>` | Show detailed information about a waypoint |

### Restore Modes

| Mode | Behavior |
|------|----------|
| Both (default) | Restores files and rewinds conversation |
| Code | Reverts files, keeps current conversation |
| Conversation | Rewinds conversation, keeps current files |

## Best Practices

1. **Create waypoints before risky changes**: Before a major refactor or
   experimental approach, drop a waypoint

2. **Use descriptive names**: "before auth refactor" is better than "checkpoint 1"

3. **Combine with git**: Waypoints are for session-level undo, git is for
   permanent history. Create git commits for stable states, waypoints for
   exploration

4. **Code-only restore for iteration**: If the model's approach is right but
   execution is wrong, use `restore code` to keep the good conversation context

## Limitations

- Waypoints are session-scoped - they don't persist across session restarts
- Only file edits via the `file_edit` plugin are tracked (not bash commands)
- Waypoint w0 (initial state) cannot be deleted

## Technical Details

### Storage

Waypoints are stored in `.jaato/waypoints.json`:

```json
{
  "next_id": 3,
  "waypoints": [
    {
      "id": "w0",
      "description": "session start",
      "created_at": "2025-01-15T10:00:00",
      "turn_index": 0,
      "is_implicit": true,
      "history_snapshot": null,
      "message_count": 0
    },
    {
      "id": "w1",
      "description": "before auth refactor",
      "created_at": "2025-01-15T10:30:00",
      "turn_index": 5,
      "is_implicit": false,
      "history_snapshot": "[...]",
      "message_count": 12
    }
  ]
}
```

### Backup Metadata

The BackupManager stores waypoint tags in `.jaato/backups/_backup_metadata.json`:

```json
{
  "/path/to/backup_2025-01-15T10-35-00.bak": {
    "backup_path": "/path/to/backup_2025-01-15T10-35-00.bak",
    "original_path": "/project/src/auth.py",
    "timestamp": "2025-01-15T10:35:00",
    "size": 1234,
    "diverged_from": "w1"
  }
}
```
