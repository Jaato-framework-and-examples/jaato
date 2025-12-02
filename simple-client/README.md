# Simple Interactive Client

A minimal console-based interactive client that demonstrates the `askPermission` plugin behavior.

## Overview

This client allows you to:
1. Enter task descriptions as prompts
2. Send prompts to a Gemini model via Vertex AI
3. See interactive permission prompts when tools are called
4. Accept or reject tool executions in real-time

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
  "default_policy": "ask",
  "whitelist": ["safe-command *"],
  "blacklist": ["rm *", "sudo *"]
}
```
