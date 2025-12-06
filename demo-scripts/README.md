# Plugin Demo Recording Scripts

Scripts for recording authentic terminal demos of jaato plugins using the real simple client.

## Prerequisites

```bash
# Install required packages
pip install pexpect pyyaml termtosvg

# Ensure you have valid API credentials in .env
```

## Quick Start

```bash
# Record a plugin demo
cd demo-scripts
termtosvg -c "python run_demo.py ../shared/plugins/cli/demo.yaml" -g 100x40 ../shared/plugins/cli/demo.svg

# Or run without recording to test
python run_demo.py ../shared/plugins/cli/demo.yaml
```

## Demo Script Format

Demos are defined in YAML files located in each plugin's directory:

```yaml
name: CLI Plugin Demo
timeout: 120

# Optional setup commands (run before starting client)
setup:
  - mkdir -p /tmp/demo
  - echo "test" > /tmp/demo/file.txt

steps:
  # Full form with explicit permission response
  - type: "List the Python files in the current directory"
    permission: "y"

  # Use 'a' for "always" permission
  - type: "Show me the git status"
    permission: "a"

  # Simple string (uses default 'y' permission)
  - "What files are in /tmp?"

  # Exit command (no permission wait)
  - type: "quit"
    delay: 0.08
```

### Script Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | filename | Display name for the demo |
| `timeout` | int | 120 | Timeout in seconds for each operation |
| `setup` | list | none | Shell commands to run before starting |
| `steps` | list | required | List of interaction steps |

### Step Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | required | Text to type at the prompt |
| `permission` | string | "y" | Permission response: y, n, a, never, once, all |
| `delay` | float | 0.05 | Delay between keystrokes (seconds) |
| `local` | bool | false | Local command (like "plan") that doesn't go through the model |

## Available Demos

| Plugin | Script Location |
|--------|-----------------|
| CLI | `shared/plugins/cli/demo.yaml` |
| File Edit | `shared/plugins/file_edit/demo.yaml` |
| Web Search | `shared/plugins/web_search/demo.yaml` |
| TODO | `shared/plugins/todo/demo.yaml` |
| References | `shared/plugins/references/demo.yaml` |
| Subagent | `shared/plugins/subagent/demo.yaml` |

## How It Works

The `run_demo.py` script uses `pexpect` to:

1. Run optional setup commands from the script
2. Spawn the real `simple-client/interactive_client.py`
3. Wait for prompts and feed user inputs via PTY
4. Respond to permission prompts automatically
5. Type inputs with realistic delays for visual effect

This produces authentic recordings of the actual client behavior.

## Creating New Demos

1. Create a `demo.yaml` in your plugin directory
2. Define the steps (prompts, permission responses)
3. Add optional setup commands if needed
4. Test with: `python run_demo.py path/to/demo.yaml`
5. Record with: `termtosvg -c "python run_demo.py path/to/demo.yaml" output.svg`

## Animation Behavior

By default, `record_all.sh` post-processes SVG files to play only once instead of looping indefinitely. To fix existing SVGs:

```bash
# Make a single SVG play once
sed -i 's/repeatCount="indefinite"/repeatCount="1"/g' demo.svg

# Fix all SVGs in plugin directories
find shared/plugins -name "demo.svg" -exec sed -i 's/repeatCount="indefinite"/repeatCount="1"/g' {} \;
```

## Troubleshooting

### "pexpect not found"
```bash
pip install pexpect
```

### "pyyaml not found"
```bash
pip install pyyaml
```

### "termtosvg not found"
```bash
pip install termtosvg
```

### Recordings are blank/too short
- Increase timeout values in the YAML script
- Check that `.env` has valid credentials
- Run demo directly first: `python run_demo.py demo.yaml`

### Permission prompts not detected
- Some tools are auto-approved and won't show prompts
- The script handles both cases (permission asked or already granted)
