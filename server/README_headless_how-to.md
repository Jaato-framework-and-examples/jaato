# Headless Mode How-To

This guide explains how to run the jaato rich client in **headless mode** (no TTY, file-based output) and how to **send commands to a running headless session** from another terminal.

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 1. Start the Server Daemon

The server must be running before you launch headless clients. Start it as a background daemon with an IPC socket:

```bash
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon
```

Useful daemon management commands:

```bash
# Check if the server is running
.venv/bin/python -m server --status

# Stop the server
.venv/bin/python -m server --stop

# Restart with the same parameters
.venv/bin/python -m server --restart

# View server logs
tail -f /tmp/jaato.log
```

> **Note:** If you omit `--daemon`, the server runs in the foreground (useful for debugging). You can also add `--verbose` for DEBUG-level logging and `--web-socket :8080` to expose a WebSocket endpoint for remote clients.

## 2. Run the Rich Client in Headless Mode

Headless mode requires three flags: `--connect`, `--headless`, and `--prompt`.

```bash
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --headless \
    --prompt "Analyze the code in src/ for security issues"
```

### What Happens

1. The client connects to the daemon over the IPC socket.
2. Permissions are set to **auto-approve** (`permissions default allow`), so the agent can work without human confirmation.
3. The **clarification tool** is disabled (there is no user to answer questions).
4. The prompt is sent to the model.
5. All output is written to files in `{workspace}/jaato-headless-client-agents/`:
   - `main.log` -- main agent output (tool calls, plans, text).
   - `{agent_id}_{name}.log` -- one file per subagent.
6. The process exits once the model finishes and all subagents have completed.

### Available Flags

| Flag | Description |
|------|-------------|
| `--connect SOCKET_PATH` | **(required)** Path to the server's IPC socket. |
| `--headless` | **(required)** Enable headless mode with file output. |
| `--prompt TEXT` | **(required)** The prompt to send to the model. |
| `--workspace DIR` | Workspace directory for output files. Defaults to the current directory. |
| `--new-session` | Create an isolated session instead of resuming the default. Recommended for parallel headless runs. |
| `--no-auto-start` | Don't auto-start the server if it isn't running. Without this flag, the client will attempt to launch the daemon automatically. |
| `--env-file PATH` | Path to `.env` file for environment variables (default: `.env`). |

### Session Isolation

By default, a headless client attaches to the default session, which means sequential headless runs share conversation history and permission state. Use `--new-session` to start a fresh, isolated session:

```bash
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --headless \
    --new-session \
    --prompt "Write unit tests for the auth module"
```

This is especially important when running multiple headless clients in parallel.

## 3. Monitor Headless Output

Output files contain ANSI escape codes for colored rendering. Use `less -R` to view them:

```bash
# Watch the main agent output in real-time
tail -f jaato-headless-client-agents/main.log

# View with colors
less -R jaato-headless-client-agents/main.log
```

Progress messages are also printed to **stderr** during execution:

```
[headless] Connecting to server at /tmp/jaato.sock...
[headless] Connected!
[headless] Setting permission policy to auto-approve...
[headless] Disabling clarification tool...
[headless] Waiting for session acquisition...
[headless] Session ID: a1b2c3d4
[headless] Sending prompt...
[headless] Main agent working on input prompt...
[headless] Output written to: /home/user/project/jaato-headless-client-agents
```

You can redirect stderr to capture these status lines separately:

```bash
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --headless \
    --prompt "Refactor the database layer" \
    2>headless-status.log
```

## 4. Send Commands to a Headless Session

While a headless session is running, you can interact with it from another terminal using `--cmd` mode. This requires the `--session` flag to specify which session to target.

### Get the Session ID

The headless client prints the session ID to stderr on startup (`Session ID: ...`). You can also list sessions:

```bash
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session any \
    --cmd "session list"
```

### Send a Command

```bash
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session <SESSION_ID> \
    --cmd "<COMMAND>"
```

### Available Commands

**Agent Control:**

| Command | Effect |
|---------|--------|
| `stop` | Stop the model mid-generation. |
| `exit` / `quit` | End the session and signal all attached clients to exit. |
| `reset` | Reset conversation history. |

**Send a Follow-Up Prompt:**

Any text that isn't a recognized command is sent as a new user message:

```bash
# Send a follow-up instruction to the running session
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session <SESSION_ID> \
    --cmd "Now also add integration tests"
```

**Permission Management:**

```bash
# Revoke auto-approve (switch to deny-by-default)
--cmd "permissions default deny"

# Show current permission state
--cmd "permissions status"

# Whitelist a specific tool
--cmd "permissions whitelist cli"
```

**Tool Management:**

```bash
# List all tools and their enabled/disabled status
--cmd "tools list"

# Disable a specific tool
--cmd "tools disable web_search"

# Re-enable all tools
--cmd "tools enable all"
```

**Session Management:**

```bash
# List all sessions
--cmd "session list"

# Get conversation history
--cmd "history"
```

**Model Configuration:**

```bash
# Switch model mid-session
--cmd "model gemini-2.5-pro"
```

**Workspace Monitoring:**

```bash
# Show file changes as a tree (default)
--cmd "workspace tree"

# Flat list, JSON, or CSV format
--cmd "workspace list"
--cmd "workspace json"
--cmd "workspace csv"
```

## 5. Workspace Monitoring

The `workspace` command lets you inspect which files the agent has created, modified, or deleted during a session. This is especially useful for headless runs where you can't see the workspace panel.

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `workspace tree` | Indented directory tree grouped by directory (default) |
| `workspace list` | Flat file list, one path per line with status indicator |
| `workspace json` | Machine-readable JSON output |
| `workspace csv`  | CSV output (`path,status`) |

### Usage

```bash
# Tree view (default if no subcommand given)
--cmd "workspace tree"

# Example output:
# Workspace changes (4 files), 1 deleted
# ├── docs/
# │   └── guide.md  [~]
# ├── src/
# │   ├── main.py  [~]
# │   └── utils/
# │       ├── constants.py  [+]
# │       └── helper.py  [+]
# └── README.md  [-]
```

Status symbols: `[+]` created, `[~]` modified, `[-]` deleted.

```bash
# Flat list
--cmd "workspace list"

# Example output:
# [~] docs/guide.md
# [+] src/utils/constants.py
# [+] src/utils/helper.py
# [~] src/main.py
# [-] README.md
#
# 5 files (4 active, 1 deleted)
```

```bash
# JSON for scripting / CI pipelines
--cmd "workspace json"

# Example output:
# {
#   "files": [
#     {"path": "src/main.py", "status": "modified"},
#     {"path": "src/utils/helper.py", "status": "created"},
#     ...
#   ],
#   "summary": {"total": 5, "created": 2, "modified": 2, "deleted": 1}
# }
```

```bash
# CSV for spreadsheet / data processing
--cmd "workspace csv"

# Example output:
# path,status
# docs/guide.md,modified
# src/main.py,modified
# src/utils/constants.py,created
# ...
```

### CI Example: Capture Changed Files

```bash
# Get list of files the agent touched, as JSON
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session "$SESSION_ID" \
    --cmd "workspace json" > workspace-changes.json
```

## 6. Complete Example: Headless with Remote Control

**Terminal 1** -- Start server and headless session:

```bash
# Start the daemon
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Run a headless session with a long task
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --headless \
    --new-session \
    --prompt "Perform a full code review of the repository" \
    2>review-status.log
```

**Terminal 2** -- Monitor and interact:

```bash
# Watch the session ID from the status log
grep "Session ID" review-status.log
# Output: [headless] Session ID: a1b2c3d4

# Follow output in real-time
tail -f jaato-headless-client-agents/main.log

# Check tool activity
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session a1b2c3d4 \
    --cmd "tools list"

# Send a follow-up instruction
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session a1b2c3d4 \
    --cmd "Focus on the authentication module next"

# Stop the agent if needed
.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --session a1b2c3d4 \
    --cmd "stop"
```

## 7. CI/CD Integration

Headless mode works in non-TTY environments like CI pipelines:

```yaml
# .github/workflows/review.yml
jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up jaato
        run: |
          python3 -m venv .venv
          .venv/bin/pip install -r requirements.txt

      - name: Run AI code review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          .venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon
          .venv/bin/python jaato-tui/rich_client.py \
              --connect /tmp/jaato.sock \
              --headless \
              --new-session \
              --prompt "Review the changes in this PR for bugs and security issues"

      - name: Upload review output
        uses: actions/upload-artifact@v4
        with:
          name: code-review
          path: jaato-headless-client-agents/
```

## 8. Environment Variables

These variables are relevant to headless operation:

| Variable | Purpose |
|----------|---------|
| `JAATO_TRACE_LOG` | Path to a file for detailed debug logs from the headless client. |
| `JAATO_SESSION_LOG_DIR` | Directory for per-session server logs (relative to workspace, default: `.jaato/logs`). |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution (default: `true`). |
| `JAATO_PROVIDER` | Model provider (`anthropic`, `google_genai`, `github_models`, etc.). |
| `MODEL_NAME` | Model to use (e.g., `gemini-2.5-flash`, `claude-sonnet-4-5`). |

Set these in the `.env` file or export them in the shell before running headless mode.

## Troubleshooting

**Connection refused / server not running:**
The headless client auto-starts the server by default. If you used `--no-auto-start`, make sure the daemon is running (`python -m server --status`).

**Output directory empty:**
Verify the `--workspace` path is writable. Output goes to `{workspace}/jaato-headless-client-agents/`.

**Permission prompts blocking execution:**
Headless mode sets `permissions default allow` automatically. If a tool is blacklisted and still triggers a prompt, the client responds with a one-time `y`. Check `permissions status` if behavior is unexpected.

**Session not found with `--cmd`:**
Use `--cmd "session list"` to see available sessions. Session IDs are printed to stderr when the headless client starts.
