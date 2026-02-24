# Themes

Jaato ships with six built-in themes. Place any of these files in your project's
`.jaato/themes/` directory to override the built-in version, or create your own.

## Built-in Themes

| Theme | Description | Background |
|-------|-------------|------------|
| **dark** | Default dark theme with cyan accents | Dark (#1a1a1a) |
| **light** | Light theme for bright terminals | Light gray (#f5f5f5) |
| **high-contrast** | High contrast for accessibility | Pure black (#000000) |
| **dracula** | Dracula-inspired with pink/purple accents | Dark purple (#282a36) |
| **latte** | Warm cream theme with orange accents | Warm cream (#f8f4f0) |
| **mocha** | Warm brownish theme with teal accents | Dark brown (#3d2b2b) |

## Switching Themes

At runtime:

```
/theme dark
/theme dracula
/theme light
```

Via environment variable:

```bash
export JAATO_THEME=dracula
```

Your choice is persisted in `~/.jaato/preferences.json`.

## Creating a Custom Theme

1. Copy `custom-example.json` (or any built-in theme) and rename it
2. Edit the `colors` section -- these 11 palette colors are referenced by
   the `semantic` styles throughout the UI
3. Optionally customize `semantic` styles for fine-grained control over
   individual UI elements

### Color Palette (required)

Every theme must define these 11 colors:

| Key | Purpose |
|-----|---------|
| `primary` | Main accent color (headers, borders, highlights) |
| `secondary` | Secondary accent (user messages, success-adjacent) |
| `accent` | Tertiary accent (workspace labels, warm highlights) |
| `success` | Success states (completed, approved) |
| `warning` | Warning states (pending, caution) |
| `error` | Error states (failed, denied) |
| `muted` | De-emphasized text and borders |
| `background` | Main background |
| `surface` | Raised panels, status bars |
| `text` | Primary text color |
| `text_muted` | Secondary/diminished text |

### Semantic Styles (optional)

The `semantic` section maps UI element names to style definitions. Each style
can include:

```json
{
  "fg": "#hexcolor or palette_name",
  "bg": "#hexcolor or palette_name",
  "bold": true,
  "italic": true,
  "dim": true,
  "underline": true
}
```

Using palette names (like `"primary"`, `"error"`) instead of hex codes means
your semantic styles automatically adapt when you change the palette.

Unspecified semantic styles fall back to built-in defaults.

## Theme Search Order

1. `JAATO_THEME` environment variable
2. Saved preference (`~/.jaato/preferences.json`)
3. Project theme (`.jaato/theme.json` or `.jaato/themes/<name>.json`)
4. User theme (`~/.jaato/theme.json` or `~/.jaato/themes/<name>.json`)
5. Built-in themes (bundled with jaato-tui)
6. Hardcoded fallback
