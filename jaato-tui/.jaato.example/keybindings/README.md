# Keybinding Profiles

Terminal-specific keybinding profiles. Jaato auto-detects your terminal and
loads the matching profile. You can also force a profile with:

```bash
export JAATO_KEYBINDING_PROFILE=vscode
```

Or reload at runtime:

```
keybindings reload
```

## How Profiles Work

1. Jaato detects your terminal from `$TERM_PROGRAM`, `$TMUX`, etc.
2. It looks for `.jaato/keybindings/<terminal>.json`
3. Profile settings override the base `keybindings.json` (only the keys you
   specify are replaced; the rest use defaults)

## Key Syntax

Keys use [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) syntax:

| Syntax | Meaning |
|--------|---------|
| `"enter"` | Enter key |
| `"c-c"` | Ctrl+C |
| `"c-d"` | Ctrl+D |
| `"s-tab"` | Shift+Tab |
| `"f1"` | F1 function key |
| `"pageup"` | Page Up |
| `["escape", "enter"]` | Escape followed by Enter |
| `"<scroll-up>"` | Mouse scroll up |

## Available Profiles

- **vscode.json** - VS Code integrated terminal (remaps keys that conflict
  with VS Code shortcuts)
- **tmux.json** - tmux sessions (avoids conflicts with the tmux prefix key)

## Creating Your Own Profile

Copy one of the existing profiles and modify it. Only include the keys you
want to override -- unspecified keys fall back to the base `keybindings.json`.

```json
{
  "_comment": "My custom terminal profile",
  "submit": "enter",
  "toggle_plan": "f3"
}
```

## Environment Variable Overrides

Individual keys can be overridden via environment variables:

```bash
export JAATO_KEY_SUBMIT=enter
export JAATO_KEY_CANCEL=c-c
export JAATO_KEY_NEWLINE="escape enter"
```

Environment variables take highest precedence (override both base config and
profile).
