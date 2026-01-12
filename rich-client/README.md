# Rich Client

A terminal UI client for jaato using prompt_toolkit with a sticky plan display and scrolling output.

## Running

```bash
# Standalone mode
.venv/bin/python rich-client/rich_client.py

# Connect to running server
.venv/bin/python rich-client/rich_client.py --connect /tmp/jaato.sock
```

## Theming

The rich client supports customizable themes with automatic persistence.

### Built-in Themes

| Theme | Description |
|-------|-------------|
| `dark` | Dark background with cyan/green accents (default) |
| `light` | Light background for bright terminals |
| `high-contrast` | High contrast for accessibility |

### Commands

```
/theme                  Show current theme info
/theme dark             Switch to dark theme
/theme light            Switch to light theme
/theme high-contrast    Switch to high-contrast theme
/theme reload           Reload from config files
```

Your theme selection is automatically saved and restored on next startup.

### Custom Themes

Create `.jaato/theme.json` (project-level) or `~/.jaato/theme.json` (user-level):

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

### Environment Override

Set `JAATO_THEME` to temporarily override the saved preference:

```bash
JAATO_THEME=light .venv/bin/python rich-client/rich_client.py
```

## Keybindings

See the main [CLAUDE.md](../CLAUDE.md#rich-client-keybindings) for keybinding configuration.
