# Rich Client

Interactive TUI client for jaato with multi-agent support, streaming output, and rich formatting.

## Overview

The rich client provides:
- Multi-agent conversation view with agent cycling
- Streaming output with real-time rendering
- Permission prompts for tool execution
- Plan panel for tracking agent plans
- Customizable keybindings and themes

## Running

```bash
# Direct execution
.venv/bin/python rich-client/rich_client.py

# Connect to running server
.venv/bin/python rich-client/rich_client.py --connect /tmp/jaato.sock
```

## Commands

### General

| Command | Description |
|---------|-------------|
| `help` | Show help message and available commands |
| `reset` | Clear conversation history |
| `history` | Show full conversation history |
| `context` | Show context window usage |
| `export [file]` | Export session to YAML |
| `clear` | Clear output panel |
| `quit` / `exit` | Exit the client |

### Model

| Command | Description |
|---------|-------------|
| `model <name>` | Switch to a different model |

### Tools

| Command | Description |
|---------|-------------|
| `tools` | Manage tools available to the model |
| `tools list` | List all tools with enabled/disabled status |
| `tools enable <name>` | Enable a tool (or `all`) |
| `tools disable <name>` | Disable a tool (or `all`) |

### Plugins

| Command | Description |
|---------|-------------|
| `plugins` | List available plugins with status |

### Sessions

| Command | Description |
|---------|-------------|
| `save [name]` | Save current session |
| `resume [name]` | Resume a saved session |
| `sessions` | List saved sessions |

### Plan

| Command | Description |
|---------|-------------|
| `plan` | Show current plan status |

### Screenshot

Capture TUI state as images. The command is intercepted client-side. By default, a system hint is sent to the model with the capture path.

| Command | Description |
|---------|-------------|
| `screenshot` | Capture and send hint to model |
| `screenshot nosend` | Capture only, no hint to model |
| `screenshot format F` | Set output format (svg, png, html) |
| `screenshot auto` | Toggle auto-capture on turn end |
| `screenshot interval N` | Capture every N ms during streaming (0=off) |
| `screenshot help` | Show help |

Captures are saved to `/tmp/jaato_vision/` by default. SVG is the default format. PNG requires `cairosvg` package.

### Keybindings

| Command | Description |
|---------|-------------|
| `keybindings` | Manage keyboard shortcuts |
| `keybindings list` | Show current keybinding configuration |
| `keybindings set <action> <key>` | Set a keybinding |
| `keybindings profile` | Show/switch terminal-specific profiles |
| `keybindings reload` | Reload keybindings from config files |

### Authentication

**Anthropic Claude:**

| Command | Description |
|---------|-------------|
| `anthropic-auth login` | Open browser for OAuth authentication |
| `anthropic-auth code <code>` | Complete login with authorization code |
| `anthropic-auth logout` | Clear stored OAuth tokens |
| `anthropic-auth status` | Show current authentication status |

**Antigravity (Google IDE Backend):**

| Command | Description |
|---------|-------------|
| `antigravity-auth login` | Open browser for Google OAuth |
| `antigravity-auth code <code>` | Complete login with authorization code |
| `antigravity-auth logout` | Clear stored accounts |
| `antigravity-auth status` | Show current authentication status |
| `antigravity-auth accounts` | List all authenticated accounts |

### Theme

| Command | Description |
|---------|-------------|
| `/theme` | Show current theme info |
| `/theme <name>` | Switch theme (dark, light, high-contrast) |
| `/theme reload` | Reload from config files |

## Default Keybindings

| Action | Key | Description |
|--------|-----|-------------|
| Submit | `Enter` | Send message |
| Newline | `Escape` `Enter` | Insert newline |
| Clear input | `Escape` `Escape` | Clear input buffer |
| Cancel | `Ctrl-C` | Cancel current operation |
| Exit | `Ctrl-D` | Exit the client |
| Scroll up | `PageUp` | Scroll output up |
| Scroll down | `PageDown` | Scroll output down |
| Scroll top | `Home` | Scroll to top |
| Scroll bottom | `End` | Scroll to bottom |
| Toggle plan | `Ctrl-P` | Show/hide plan panel |
| Toggle tools | `Ctrl-T` | Show/hide tools panel |
| Cycle agents | `Ctrl-A` | Cycle through agents |
| Yank/copy | `Ctrl-Y` | Copy to clipboard |
| View full | `V` | View full content in pager |

## Configuration

### Keybindings

Keybindings can be configured via:
1. Project config: `.jaato/keybindings.json`
2. User config: `~/.jaato/keybindings.json`
3. Environment variables: `JAATO_KEY_<ACTION>=<key>`

Priority: Environment variables > Project config > User config > Defaults

### Themes

The rich client supports customizable themes with automatic persistence.

**Built-in Themes:**

| Theme | Description |
|-------|-------------|
| `dark` | Dark background with cyan/green accents (default) |
| `light` | Light background for bright terminals |
| `high-contrast` | High contrast for accessibility |

Custom themes can be defined in:
- Project: `.jaato/theme.json`
- User: `~/.jaato/theme.json`

Theme selection is persisted to `~/.jaato/preferences.json`.

**Custom Theme Example:**

```json
{
  "name": "my-theme",
  "version": "1.0",
  "colors": {
    "primary": "#5fd7ff",
    "secondary": "#87d787",
    "success": "#5fd75f",
    "warning": "#ffff5f",
    "error": "#ff5f5f",
    "muted": "#808080",
    "background": "#1a1a1a",
    "surface": "#333333",
    "text": "#ffffff",
    "text_muted": "#aaaaaa"
  }
}
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `JAATO_THEME` | Override theme (temporary) |
| `JAATO_VISION_DIR` | Screenshot output directory (default: `/tmp/jaato_vision`) |
| `JAATO_VISION_FORMAT` | Screenshot format: `svg` (default), `png`, `html` |
| `JAATO_COPY_MECHANISM` | Clipboard provider: `osc52` (default) |
| `JAATO_COPY_SOURCES` | Sources to copy: `model` (default), or `model&user&tool` |

## File References

Reference files in prompts with `@` prefix:

```
What's in @screenshot.png?
Review @src/main.py
```

## Related

- [Vision Capture Plugin](../shared/plugins/vision_capture/README.md) - Screenshot implementation
- [Architecture](../docs/architecture.md) - Framework architecture
