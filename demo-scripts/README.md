# Plugin Demo Recording Scripts

Scripts for recording authentic terminal demos of each jaato plugin using the real simple client.

## Prerequisites

```bash
# Install required packages
pip install pexpect termtosvg

# Ensure you have valid API credentials in .env
```

## Quick Start

```bash
# Record all demos
./record_all.sh

# Record specific demo
./record_all.sh cli

# Record multiple demos
./record_all.sh cli file_edit web_search
```

## Manual Recording

You can also record demos manually with custom settings:

```bash
# Record CLI demo with custom geometry
termtosvg -c "python run_demo.py cli" -g 100x40 cli_demo.svg

# Record with different template
termtosvg -c "python run_demo.py file_edit" -t window_frame file_edit_demo.svg
```

## Available Demos

| Demo | Description | Recommended Size |
|------|-------------|------------------|
| `cli` | Shell command execution with permission prompts | 100x40 |
| `file_edit` | File modification with diff preview | 100x45 |
| `web_search` | Web search queries and results | 100x40 |
| `todo` | Plan creation and progress tracking | 100x45 |
| `references` | Documentation source selection | 100x42 |
| `subagent` | Spawning specialized subagents | 100x45 |

## How It Works

The `run_demo.py` script uses `pexpect` to:

1. Spawn the real `simple-client/interactive_client.py`
2. Wait for prompts and feed user inputs via PTY
3. Respond to permission prompts automatically
4. Type inputs with realistic delays for visual effect

This produces authentic recordings of the actual client behavior.

## Customizing Demos

Edit `run_demo.py` to modify:

- **Prompts**: Change what questions are asked
- **Timing**: Adjust `time.sleep()` calls for pacing
- **Typing speed**: Modify `delay` parameter in `type_slowly()`
- **Permission responses**: Change 'y', 'a', 'n' responses

## Output

Recordings are saved to `demo-scripts/recordings/` by default.

After recording, you can copy them to plugin directories:

```bash
# Manually copy
cp recordings/cli_demo.svg ../shared/plugins/cli/demo/demo.svg

# Or use the script's prompt after recording
./record_all.sh  # Will ask to copy after recording
```

## Troubleshooting

### "pexpect not found"
```bash
pip install pexpect
```

### "termtosvg not found"
```bash
pip install termtosvg
```

### Recordings are blank/too short
- Increase timeout values in `run_demo.py`
- Check that `.env` has valid credentials
- Run demo directly first: `python run_demo.py cli`

### Permission prompts not detected
- Adjust the regex patterns in `wait_for_permission()`
- Some tools are auto-approved and won't show prompts
