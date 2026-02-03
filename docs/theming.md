# Rich Client Theming System

The rich client uses a **two-tier color system** that balances simplicity with fine-grained control.

## Tier 1: Base Palette (11 colors)

The foundation is a palette of 11 named colors that define the visual identity:

| Color | Purpose | Default (Dark) |
|-------|---------|----------------|
| `primary` | Main accent, interactive elements | `#5fd7ff` (cyan) |
| `secondary` | Secondary accent | `#87d787` (light green) |
| `accent` | Tertiary accent | `#d7af87` (tan/orange) |
| `success` | Success states, confirmations | `#5fd75f` (green) |
| `warning` | Warnings, cautions | `#ffff5f` (yellow) |
| `error` | Errors, failures | `#ff5f5f` (red) |
| `muted` | Deemphasized elements | `#808080` (gray) |
| `background` | Main background | `#1a1a1a` |
| `surface` | Raised surfaces (bars, panels) | `#333333` |
| `text` | Primary text | `#ffffff` |
| `text_muted` | Secondary/dimmed text | `#aaaaaa` |

Changing a palette color affects all UI elements that reference it.

## Tier 2: Semantic Styles (~90 tokens)

Semantic styles map UI elements to palette colors plus text modifiers. Each style can specify:

- `fg`: Foreground color (palette name or hex)
- `bg`: Background color (palette name or hex)
- `bold`: Bold text
- `italic`: Italic text
- `dim`: Dimmed/faded appearance
- `underline`: Underlined text

**Example definitions:**

```json
{
  "agent_tab_selected": {"fg": "primary", "bold": true, "underline": true},
  "tool_output": {"fg": "#87D7D7", "italic": true},
  "plan_completed": {"fg": "success", "bg": "surface"},
  "thinking_content": {"dim": true, "italic": true}
}
```

## Semantic Style Categories

| Category | Tokens | Examples |
|----------|--------|----------|
| **Agent tabs** | 10 | `agent_tab_selected`, `agent_processing`, `agent_error` |
| **Session bar** | 6 | `session_bar_label`, `session_bar_id`, `session_bar_workspace` |
| **Status bar** | 5 | `status_bar_label`, `status_bar_value`, `status_bar_warning` |
| **Plan panel** | 18 | `plan_pending`, `plan_in_progress`, `plan_completed`, `plan_blocked` |
| **Output headers** | 4 | `user_header`, `model_header`, `user_header_separator` |
| **Thinking blocks** | 6 | `thinking_header`, `thinking_border`, `thinking_content` |
| **Tool display** | 12 | `tool_output`, `tool_name`, `tool_success`, `tool_error`, `tool_duration` |
| **Permissions** | 7 | `permission_prompt`, `permission_text`, `permission_bar_focused` |
| **Clarification** | 6 | `clarification_label`, `clarification_answer`, `clarification_resolved` |
| **System messages** | 10 | `system_info`, `system_error`, `system_warning`, `system_success` |
| **UI elements** | 12 | `spinner`, `separator`, `hint`, `scroll_indicator`, `panel_border` |
| **Budget panel** | 6 | `budget_header`, `budget_gc_locked`, `budget_gc_preservable` |

## Built-in Themes

Available via `JAATO_THEME` environment variable or `/theme` command:

- **dark** (default) - Dark background with cyan/green accents
- **light** - Light background with adjusted contrast
- **high-contrast** - Maximum contrast for accessibility
- **dracula** - Popular dark theme
- **latte** - Catppuccin Latte (light)
- **mocha** - Catppuccin Mocha (dark)

## Theme Customization

### Option 1: Select a preset

```bash
export JAATO_THEME=high-contrast
```

Or interactively:

```
/theme high-contrast
```

### Option 2: Override a built-in theme

Create `~/.jaato/themes/dark.json` to override the built-in dark theme:

```json
{
  "name": "dark",
  "colors": {
    "primary": "#ff6600"
  }
}
```

### Option 3: Create a custom theme

Create `.jaato/theme.json` (project) or `~/.jaato/theme.json` (user):

```json
{
  "name": "my-theme",
  "description": "Custom corporate theme",
  "version": "1.0",
  "colors": {
    "primary": "#0066cc",
    "secondary": "#00aa44",
    "background": "#0a0a0a"
  },
  "semantic": {
    "tool_output": {"fg": "#00aacc", "italic": true},
    "model_header": {"fg": "primary", "bold": true}
  }
}
```

## Theme Discovery Order

1. `JAATO_THEME` environment variable (temporary override)
2. Saved preference (`~/.jaato/preferences.json`)
3. Project theme (`.jaato/theme.json`)
4. User theme (`~/.jaato/theme.json`)
5. Built-in theme overrides (`~/.jaato/themes/<name>.json`)
6. Built-in themes from package
7. Hardcoded fallback

## Hot Reload

Themes reload automatically when files change. Use `/theme reload` to force a refresh.

## Dual Output Support

The theme system generates styles for both:

- **prompt_toolkit** - TUI chrome (bars, input, panels)
- **Rich** - Output formatting (markdown, syntax highlighting, tables)

Both derive from the same `ThemeConfig`, ensuring visual consistency across the entire interface.

## Complete Semantic Style Reference

### Agent Tab Bar

| Token | Description |
|-------|-------------|
| `agent_tab_selected` | Currently selected agent tab |
| `agent_tab_dim` | Inactive agent tabs |
| `agent_tab_separator` | Separator between tabs |
| `agent_tab_hint` | Keyboard hint text |
| `agent_tab_scroll` | Scroll indicators |
| `agent_processing` | Agent actively processing |
| `agent_awaiting` | Agent waiting for input |
| `agent_finished` | Agent completed |
| `agent_error` | Agent encountered error |
| `agent_permission` | Agent waiting for permission |

### Plan Panel

| Token | Description |
|-------|-------------|
| `plan_pending` | Step not yet started |
| `plan_in_progress` | Step currently executing |
| `plan_completed` | Step finished successfully |
| `plan_failed` | Step failed |
| `plan_skipped` | Step skipped |
| `plan_blocked` | Step waiting on dependencies |
| `plan_active` | Overall plan is active |
| `plan_cancelled` | Plan was cancelled |
| `plan_popup_border` | Popup panel border |
| `plan_popup_background` | Popup background |
| `plan_result` | Step result text |
| `plan_error_text` | Step error message |
| `plan_dependency` | Cross-agent dependency indicator |
| `plan_received_output` | Output received from another agent |

### Tool Display

| Token | Description |
|-------|-------------|
| `tool_output` | Tool output text |
| `tool_source_label` | Source label (e.g., plugin name) |
| `tool_name` | Tool name in output |
| `tool_border` | Tool output box borders |
| `tool_success` | Successful tool execution |
| `tool_error` | Failed tool execution |
| `tool_pending` | Tool currently running |
| `tool_duration` | Execution time display |
| `tool_selected` | Selected tool in list |
| `tool_unselected` | Unselected tool in list |
| `tool_indicator` | Spinner/progress indicator |

### System Messages

| Token | Description |
|-------|-------------|
| `system_info` | Default system message |
| `system_highlight` | Informational highlights |
| `system_error` | Error messages |
| `system_error_bold` | Emphasized errors |
| `system_warning` | Warnings/interrupts |
| `system_success` | Success messages |
| `system_emphasis` | Emphasized text |
| `system_version` | Version/release names |
| `system_progress` | Progress indicators |
| `system_init_error` | Initialization failures |
