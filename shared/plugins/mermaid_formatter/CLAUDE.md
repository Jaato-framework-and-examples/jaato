# CLAUDE.md — mermaid_formatter plugin

## Rendering Strategy Chain

The renderer tries strategies in strict order and returns the first success:

1. **mmdc** (local) — `shutil.which("mmdc")`. Gold standard, requires `npm install -g @mermaid-js/mermaid-cli` + Node.js + headless browser. Supports all options: theme, scale, background.
2. **kroki.io POST** (remote) — `POST https://kroki.io/mermaid/png` with diagram source as plain text body. No local dependencies. Configurable URL via `JAATO_KROKI_URL` for self-hosted instances (useful when public kroki.io is blocked by enterprise firewalls).
3. **Passthrough** — When both fail, the raw mermaid source is re-emitted as a ` ```mermaid ` code block for `code_block_formatter` to syntax-highlight. A dim hint is shown.

## Why not mermaid-py?

The `mermaid` PyPI package (`mermaid-py`) was removed because it is **not** a native Python renderer. It is a thin HTTP client that calls `mermaid.ink` via **GET** with the entire diagram base64-encoded in the URL path. This fails with HTTP 400 for any non-trivial diagram due to URL length limits. The kroki.io POST API solves the same problem (remote rendering, no Node.js) without the URL size constraint.

There is no native Python mermaid renderer. Mermaid is a JavaScript library that requires a browser DOM to produce SVG. Every Python package either shells out to Node.js, calls an external API, or generates HTML that needs a browser.

## `is_renderer_available()` Contract

This function determines whether system instructions should tell the model to use ` ```mermaid ` blocks. It must reflect actual rendering capability:

- Returns `True` only if mmdc is installed **or** kroki.io is reachable (verified by a real test render, cached process-wide).
- If it returns `True` but rendering later fails at call time (e.g., transient network error), the plugin falls back to the passthrough code block — not a crash.
- If it returns `False`, the mermaid system instruction is suppressed, so the model won't produce mermaid blocks it can't render.

## Theme Injection

For kroki rendering, theme is injected via mermaid's init directive:
```
%%{init: {'theme': 'dark'}}%%
graph TD
    A-->B
```
The directive is only injected when theme is not `"default"` and when the source doesn't already contain a `%%{` directive.

For mmdc, theme is passed via the `-t` CLI flag.

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `JAATO_KROKI_URL` | Custom kroki endpoint (default: `https://kroki.io`) |
| `JAATO_MERMAID_BACKEND` | `kitty`, `iterm`, `sixel`, `ascii`, `off` |
| `JAATO_MERMAID_THEME` | `default`, `dark`, `forest`, `neutral` |
| `JAATO_MERMAID_SCALE` | Integer scale factor (default: `2`) |

## Testing Notes

- All renderer strategies are tested via mocks (no real network calls or mmdc binary needed).
- The `reset_caches` autouse fixture in `test_renderer.py` resets `_mmdc_path`, `_mmdc_checked`, and `_kroki_available` between tests — always reset all three when adding tests.
- Plugin tests mock `renderer` at the module level (`@patch("shared.plugins.mermaid_formatter.plugin.renderer")`).

## File Roles

| File | Responsibility |
|------|---------------|
| `renderer.py` | Source text → PNG bytes (strategy chain) |
| `plugin.py` | Streaming block detection + fallback formatting |
| `backends/` | PNG bytes → terminal output (kitty/iterm/sixel/unicode) |
