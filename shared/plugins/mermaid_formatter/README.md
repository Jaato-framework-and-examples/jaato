# Mermaid Formatter Plugin

Streaming formatter that renders Mermaid diagrams inline in the terminal using the best available graphics protocol.

## Overview

The mermaid formatter plugin:
- Detects ` ```mermaid ` code blocks in model output
- Renders diagrams to PNG via mermaid-cli or mermaid-py
- Displays inline using the best terminal graphics protocol
- Falls back to Unicode half-block art when no native protocol is available
- Saves PNG artifacts for model feedback via vision capture

This is a **formatter pipeline plugin** (not a tool plugin) — it processes the model's output stream, not tool invocations.

## Pipeline Position

```
hidden_content_filter (priority 10)
    ↓
diff_formatter (priority 20)
    ↓
notebook_output_formatter (priority 22)
    ↓
table_formatter (priority 25)
    ↓
mermaid_formatter (priority 28)  ← intercepts ```mermaid blocks
    ↓
code_block_formatter (priority 40)
    ↓
vision_capture_formatter (priority 95)
```

The mermaid formatter runs **before** the code block formatter (priority 40) so it intercepts mermaid blocks before they get generic syntax highlighting. When no renderer is available, the source is passed through as a ` ```mermaid ` block for the code block formatter to handle.

## Terminal Graphics Backends

The plugin auto-detects the best rendering backend from terminal capabilities:

| Backend | Protocol | Terminals | Detection |
|---------|----------|-----------|-----------|
| `kitty` | Kitty graphics protocol | Kitty, Ghostty | `TERM_PROGRAM` |
| `iterm` | iTerm2 inline images | iTerm2, WezTerm, Mintty | `TERM_PROGRAM` |
| `sixel` | Sixel bitmap | foot, mlterm | `TERM` / `TERM_PROGRAM` |
| `rich_pixels` | Unicode half-blocks (▀) | Everything else | Universal fallback |

Detection is handled by `shared/terminal_caps.py`, which caches results process-wide. Multiplexers (tmux, screen) automatically fall back to `rich_pixels` since they strip graphics escape sequences.

### Backend Selection Priority

1. `JAATO_MERMAID_BACKEND` env var (explicit override)
2. `JAATO_GRAPHICS_PROTOCOL` env var (global graphics override)
3. Auto-detection from `TERM_PROGRAM` / `TERM`
4. `rich_pixels` fallback (always works)

## Mermaid Rendering

Diagrams are rendered to PNG using the first available strategy:

1. **mmdc** (mermaid-cli) — Gold standard. Requires Node.js:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```

2. **mermaid-py** — Python package fallback. Produces SVG, converted to PNG via cairosvg:
   ```bash
   pip install mermaid cairosvg
   ```

3. **Passthrough** — When nothing is available, the source is shown as a syntax-highlighted code block with an install hint.

## Configuration

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `JAATO_MERMAID_BACKEND` | `kitty`, `iterm`, `sixel`, `ascii`, `off` | auto-detect | Force a specific graphics backend |
| `JAATO_MERMAID_THEME` | `default`, `dark`, `forest`, `neutral` | `default` | Mermaid diagram theme |
| `JAATO_MERMAID_SCALE` | Integer | `2` | Rasterization scale factor |
| `JAATO_GRAPHICS_PROTOCOL` | `kitty`, `iterm`, `sixel`, `none` | auto-detect | Global terminal graphics override |
| `JAATO_VISION_DIR` | Path | `/tmp/jaato_vision` | Directory for PNG artifacts |

### Pipeline Configuration

In `.jaato/formatters.json`:

```json
{
  "formatters": [
    {"name": "mermaid_formatter", "enabled": true, "config": {
      "theme": "dark",
      "scale": 3,
      "background": "transparent"
    }}
  ]
}
```

### Programmatic Configuration

```python
from shared.plugins.mermaid_formatter import create_plugin

formatter = create_plugin()
formatter.initialize({
    "theme": "dark",
    "scale": 2,
    "background": "white",
    "priority": 28,
})
```

## Architecture

```
shared/plugins/mermaid_formatter/
├── __init__.py           # Package exports
├── plugin.py             # MermaidFormatterPlugin (streaming block detection)
├── renderer.py           # Mermaid source → PNG (mmdc / mermaid-py)
├── backends/
│   ├── __init__.py       # Backend protocol + auto-selection
│   ├── kitty.py          # Kitty graphics protocol
│   ├── iterm.py          # iTerm2 inline image protocol
│   ├── sixel.py          # Sixel bitmap encoding
│   └── rich_pixels.py    # Unicode half-block fallback
└── tests/
    ├── test_plugin.py    # Block detection + rendering tests
    ├── test_renderer.py  # mmdc / mermaid-py rendering tests
    └── test_backends.py  # Backend selection + protocol tests
```

### Data Flow

```
Model streams: "Here's the architecture:\n```mermaid\ngraph TD\n    A-->B\n```\n"
    │
    ▼
MermaidFormatterPlugin.process_chunk()
    ├─ Yields "Here's the architecture:\n" immediately
    ├─ Buffers "graph TD\n    A-->B" (inside mermaid block)
    └─ On closing ```: triggers render pipeline
           │
           ▼
    renderer.render(source) → PNG bytes
           │
           ▼
    backends.select_backend() → KittyBackend / ITermBackend / SixelBackend / RichPixelsBackend
           │
           ▼
    backend.render(png_data) → terminal escape sequences or Unicode art
           │
           ▼
    Yields rendered output + saves PNG artifact
```

## Dependencies

### Required

- `Pillow>=10.0.0` — Image processing for all graphics backends

### Optional

- `@mermaid-js/mermaid-cli` (npm) — High-fidelity Mermaid rendering
- `cairosvg>=2.7.0` — SVG→PNG conversion for mermaid-py output

## Related

- [Terminal Capabilities](../../../shared/terminal_caps.py) — Shared detection module
- [Vision Capture Plugin](../vision_capture/README.md) — Artifact capture integration
- [Code Block Formatter](../code_block_formatter/) — Downstream fallback for unrendered blocks
- [Formatter Pipeline](../formatter_pipeline/) — Pipeline architecture
