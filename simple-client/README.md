# Simple Interactive Client

A console-based interactive client that demonstrates the `askPermission` plugin behavior and supports multi-turn conversation.

## Overview

This client allows you to:
1. Enter task descriptions as prompts
2. Send prompts to a Gemini model via Vertex AI
3. See interactive permission prompts when tools are called
4. Accept or reject tool executions in real-time
5. Have multi-turn conversations where the model remembers context

## Setup

1. Ensure you have a `.env` file in the project root with:
   ```
   PROJECT_ID=your-gcp-project
   LOCATION=us-central1
   MODEL_NAME=gemini-2.5-flash
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

2. Install dependencies:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

## Usage

### Interactive Mode

```bash
.venv/bin/python simple-client/interactive_client.py --env-file .env
```

### Single Prompt Mode

```bash
.venv/bin/python simple-client/interactive_client.py --env-file .env --prompt "List files in current directory"
```

## Commands

In interactive mode, the following commands are available:

| Command | Description |
|---------|-------------|
| `help` | Show help message |
| `tools` | List available tools |
| `reset` | Clear conversation history (start fresh) |
| `history` | Show full conversation history |
| `quit` / `exit` / `q` | Exit the client |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate through prompt history |
| `←` / `→` | Move cursor within line |
| `Ctrl+A` / `Ctrl+E` | Jump to start/end of line |

## Multi-Turn Conversation

The client maintains conversation history across prompts. The model remembers previous exchanges within a session.

Example:
```
You> What is 2 + 2?
Model> 2 + 2 equals 4.

You> And what is that multiplied by 3?
Model> 4 multiplied by 3 equals 12.

You> reset
[History cleared - starting fresh conversation]

You> What were we talking about?
Model> This is the start of our conversation. Is there something you'd like to discuss?
```

### Viewing History

Use the `history` command to see the full conversation with tool calls:

```
You> history

==================================================
  Conversation History: 4 message(s)
==================================================

[1] USER
----------------------------------------
  List files in the current directory

[2] MODEL
----------------------------------------
  📤 CALL: cli_based_tool({'command': 'ls -la'})

[3] TOOL
----------------------------------------
  📥 RESULT: cli_based_tool → {'stdout': 'file1.txt\nfile2.py', ...}

[4] MODEL
----------------------------------------
  The current directory contains: file1.txt and file2.py
```

## Permission Prompts

When the model attempts to execute a tool, you'll see a prompt like:

```
============================================================
[askPermission] Tool execution request:
  Tool: cli_based_tool
  Arguments: {
      "command": "ls -la"
  }
============================================================

Options: [y]es, [n]o, [a]lways, [never], [once]
```

Response options:
- **y/yes** - Allow this execution
- **n/no** - Deny this execution
- **a/always** - Allow and remember for this session (won't ask again for this tool)
- **never** - Deny and block for this session
- **once** - Allow just this once (same as yes, but semantically explicit)

## Configuration

The default policy asks for permission on all tool calls. You can customize this by modifying the policy in `interactive_client.py` or using a `permissions.json` file:

```json
{
  "defaultPolicy": "ask",
  "whitelist": {"tools": ["safe-command"], "patterns": []},
  "blacklist": {"tools": ["rm", "sudo"], "patterns": []}
}
```

## Architecture

The client uses `JaatoClient` from the shared module:

```python
from shared import JaatoClient, PluginRegistry, PermissionPlugin

# Create and connect
jaato = JaatoClient()
jaato.connect(project_id, location, model_name)

# Configure tools (creates SDK chat session)
jaato.configure_tools(registry, permission_plugin, ledger)

# Send messages (SDK manages history internally)
response = jaato.send_message("Hello!")
response = jaato.send_message("Tell me more")

# Access history when needed
history = jaato.get_history()

# Reset session (clear history or set custom history)
jaato.reset_session()
jaato.reset_session(modified_history)
```

The SDK chat API manages conversation history internally. Use `get_history()` to inspect it and `reset_session()` to clear or modify it.
