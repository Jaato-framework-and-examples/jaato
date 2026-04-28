# Jaato Semantic Markup — Complete Reference

> Scope: all `<j-*>` and `<nb-*>` semantic markup tags emitted by the
> server's formatter pipeline, and how different clients render them.

## Table of Contents

1. Architecture Overview
2. `<j-code>` — Code Blocks
3. `<j-table>` — Tables
4. `<nb-row>` — Notebook Cells
5. `<j-collapse>` / `<j-expand>` — Collapsible Sections
6. `<security-warning>` — Security Warnings
7. Pipeline Configuration
8. Source Code Map

## 1. Architecture Overview

Jaato's model output passes through a **formatter pipeline** on the server
before being streamed to clients. Plugins in this pipeline detect structured
content (tables, code blocks, notebook cells) and emit **client-agnostic
semantic markup** using custom HTML-like tags. Each attached client then
renders these tags into its own native format.

```
Model output
    │
    ▼
Formatter Pipeline (server-side)
    ├─ hidden_content_filter (priority 10)
    ├─ diff_formatter (priority 20)
    ├─ notebook_output_formatter (priority 22) → <nb-row>
    ├─ table_formatter (priority 25)       → <j-table> <j-thead> <j-tr> <j-td>
    ├─ mermaid_formatter (priority 28)    → rendered image
    ├─ code_validation_formatter (priority 35)
    └─ code_block_formatter (priority 40) → <j-code> <j-line> <j-tok>
    │
    ▼
Wire format (neutral semantic tags)
    │
    ├── TUI client  → ANSI via rich (box-drawing tables, syntax-highlighted code)
    ├── Web client → HTML (<table>, Pygments CSS classes)
    ├── Chat client → card/list layout, or stripped plain text
    └── API client → structured JSON
```

The `j-` prefix identifies jaato pipeline semantic tags. Clients that
don't understand them simply strip the tags and render the inner text.

## 2. `<j-code>` — Code Blocks

### Server-side emission

The `code_block_formatter` plugin (priority 40) detects markdown fenced code
blocks and converts them into semantic markup.

**Input (model output):**
````python
def hello():
    return 42
````
````

**Output (wire format):**
```xml
<j-code language="python">
<j-line><j-tok t="k">def</j-tok> hello():</j-line>
<j-line>    <j-tok t="k">return</j-tok> <j-tok t="mi">42</j-tok></j-line>
</j-code>
```

### Tag reference

| Tag | Attributes | Description |
|-----|-----------|-------------|
| `<j-code>` | `language="..."` | Wraps the entire code block. The `language` attribute is the raw language string from the fenced block (not normalised — clients decide how to alias it). |
| `<j-line>` | `n="..."` | One per source line. Optional `n` attribute for line numbers (emitted when `line_numbers: true` in config). |
| `<j-tok>` | `t="..."` | Wraps a Pygments-classified token run. The `t` attribute carries the short token name from `pygments.token.STANDARD_TYPES` (e.g. `k`, `mi`, `s2`). Plain text runs have no `<j-tok>` wrapper. |

### Token names (`t` attribute)

The `t` value comes from Pygments' own `STANDARD_TYPES` mapping. The
server walks up the token hierarchy to find a match:

| Pygments category | `t` value | Example tokens |
|-------------------|-----------|-----------------|
| Keyword | `k` | `def`, `class`, `for`, `if`, `return` |
| Name | `n` | identifiers, function names, class names |
| Name.Entity | `nc` | class names, type names |
| Name.Function | `nf` | function definitions |
| Name.Decorator | `nd` | `@staticmethod`, `@property` |
| String | `s` | string literals |
| Number.Integer | `mi` | integer literals |
| Number.Float | `mf` | float literals |
| Operator | `o` | `+`, `-`, `*`, `/` |
| Comment | `c` | `# comment` |
| Generic.Subheading | `gh` | markdown headings inside code |
| Text | (none) | plain text, no `<j-tok>` wrapper |

### HTML escaping

The server escapes three characters that collide with its tag namespace:
- `<` → `&lt;`
- `>` → `&gt;`
- `&` → `&amp;`

Clients must reverse this when rendering.

### Configuration

```python
{
    "name": "code_block_formatter",
    "enabled": true,
    "config": {
        "line_numbers": true  # emit n="..." on <j-line>
    }
}
```

### Client-side rendering (TUI)

The TUI client (`j_markup_renderer.py`) extracts raw code from `<j-line>` /
`<j-tok>` wrappers, then renders through `rich.syntax.Syntax` using the
active Pygments theme. Language aliases are resolved client-side:
`js` → `javascript`, `ts` → `typescript`, `py` → `python`, `yml` → `yaml`,
`sh`/`shell` → `bash`, `md` → `markdown`, etc.

Theme mapping: `dark` → `monokai`, `light` → `solarized-light`,
`high-contrast` → `native`.

## 3. `<j-table>` — Tables

### Server-side emission

The `table_formatter` plugin (priority 25) detects markdown tables and converts
them to semantic markup.

**Input (model output):**
```
| Name | Age | Role |
|:-----|:---:|:-----|
| Alice | 30 | Engineer |
| Bob | 25 | Designer |
```

**Output (wire format):**
```xml
<j-table>
<j-thead>
<j-th>Name</j-th><j-th>Age</j-th><j-th>Role</j-th>
</j-thead>
<j-tr><j-td>Alice</j-td><j-td>30</j-td><j-td>Engineer</j-td></j-tr>
<j-tr><j-td>Bob</j-td><j-td>25</j-td><j-td>Designer</j-td></j-tr>
</j-table>
```

### Tag reference

| Tag | Description |
|-----|-------------|
| `<j-table>` | Wraps the entire table. |
| `<j-thead>` | Wraps the header row. |
| `<j-th>` | A single header cell. |
| `<j-tr>` | A data row. Contains one or more `<j-td>` cells. |
| `<j-td>` | A single data cell. |

### Alignment

Alignment is parsed from the separator row (`|:---:|` = center,
`|---:|` = right, `|---|` = left). Currently alignment is parsed
server-side but **not propagated to the wire format** — clients may
choose to honor it or not.

### ASCII grid tables

Tables using `+---+---+` borders are detected but passed through
unchanged — they already carry their own visual borders.

### Character width handling

The server uses `unicodedata.east_asian_width()` to calculate display width,
with configurable ambiguous width via `JAATO_AMBIGUOUS_WIDTH` env var (default 1,
set to 2 for CJK terminals). This ensures cell content is measured
correctly for wide characters, though the wire format itself carries raw
text — width is a rendering concern.

### Client-side rendering (TUI)

The TUI renders `<j-table>` through `rich.table.Table` with `box=SQUARE`.
Headers are bold. `max_table_width` must be passed explicitly to avoid Rich
defaulting to 80 columns (which overflows in narrow terminals).

## 4. `<nb-row>` — Notebook Cells

### Server-side emission

The `notebook_output_formatter` plugin (priority 22) detects
`<notebook-cell>` markers produced by the notebook plugin and converts them
into semantic markers.

**Input (model output):**
```xml
<notebook-cell type="input" exec="3">
```python
x = 1 + 1
```
</notebook-cell>
```

**Output (wire format):**
```xml
<nb-row type="input" label="In [3]:">
```python
x = 1 + 1
```
</nb-row>
```

### Tag reference

| Attribute | Description |
|----------|-------------|
| `type` | Cell type: `input`, `stdout`, `stderr`, `result`, `display`, `error` |
| `exec` | Execution count (integer) |
| `label` | Display label, e.g. `In [3]:`, `Out [3]:`, `Err [3]:` |

### Cell type to label mapping

| type | Label |
|------|-------|
| `input` | `In [n]:` |
| `result` / `display` | `Out [n]:` |
| `stderr` / `error` | `Err [n]:` |
| `stdout` | _(no label, content shown directly)_ |

### Pipeline ordering

Priority 22, **before** both `table_formatter` (25) and
`code_block_formatter` (40). This means:

1. `<nb-row>` markers pass through the table formatter without being
   interpreted as tables.
2. Code fences inside notebook cells are still highlighted by the
   `code_block_formatter` because it runs after notebook output.

### Client-side rendering (TUI)

The TUI's `OutputBuffer` detects `<nb-row>` markers and renders them
as a 2-column layout:
- Column 1: cell label (right-aligned, e.g. `In [3]:`)
- Column 2: cell content (may include syntax-highlighted code)

## 5. `<j-collapse>` / `<j-expand>` — Collapsible Sections

These tags mark collapsible content regions in the output. The TUI client
detects them and wraps the content in a collapsible panel.

The Telegram client's `ResponseRenderer` supports `supports_expandable_content`
in its presentation context, indicating it renders these as expandable/collapsible
blocks.

The server emits these tags based on content analysis or explicit model
instructions. See `jaato-tui/output_buffer.py` for the client-side handling.

## 6. `<security-warning>` — Security Warnings

Security-sensitive content (e.g., code execution plans, permission details)
is wrapped in `<security-warning>` blocks. Clients with a security focus
may render these with special styling (warning colours, collapsed by default).

## 7. Pipeline Configuration

### Default pipeline (priority order)

| Priority | Plugin | Output |
|----------|--------|--------|
| 10 | `hidden_content_filter` | Strips content marked as hidden |
| 20 | `diff_formatter` | Unified/side-by-side diffs with semantic tags |
| 22 | `notebook_output_formatter` | `<nb-row>` markers |
| 25 | `table_formatter` | `<j-table>` markup |
| 28 | `mermaid_formatter` | Rendered diagram images |
| 35 | `code_validation_formatter` | LSP-based diagnostics |
| 40 | `code_block_formatter` | `<j-code>` markup |

### Configuration file (`.jaato/formatters.json`)

```json
{
  "formatters": [
    {"name": "hidden_content_filter", "enabled": true},
    {"name": "diff_formatter", "enabled": true},
    {"name": "notebook_output_formatter", "enabled": true},
    {"name": "table_formatter", "enabled": true},
    {"name": "mermaid_formatter", "enabled": true},
    {"name": "code_validation_formatter", "enabled": true},
    {"name": "code_block_formatter", "enabled": true, "config": {"line_numbers": true}}
  ]
}
```

### Configuration via environment variables

- `JAATO_AMBIGUOUS_WIDTH` — East Asian Ambiguous character width
  (`"1"` or `"2"`, default `"1"`). Affects table cell width
  calculation server-side.

### Wire format neutrality

The wire format is deliberately **not** pre-rendered. This allows:

1. **Multi-client sessions** — TUI, web dashboard, and chat bridge can
   co-attach to the same session and each render natively.
2. **Theme independence** — the server doesn't need to know or care about
   each client's colour scheme or display capabilities.
3. **Post-processing** — clients can re-flow content to their display width
   without the server having to know the terminal size.

### FormatterPlugin protocol

All formatters implement:

```python
class FormatterPlugin:
    name: str            # Unique identifier
    priority: int         # Pipeline ordering
    process_chunk(chunk: str) -> Iterator[str]  # Main entry point
    flush() -> Iterator[str]                # End-of-turn cleanup
    reset() -> None                      # Reset state
    initialize(config: dict) -> None         # Load config
    shutdown() -> None                      # Cleanup
    set_console_width(width: int) -> None  # Terminal width hint
```

## 8. Source Code Map

| File | What it contains |
|------|-------------------|
| `server/shared/plugins/table_formatter/plugin.py` | `<j-table>` detection and emission, markdown/ASCII grid table parsing, alignment, character width |
| `server/shared/plugins/code_block_formatter/plugin.py` | `<j-code>` detection, Pygments tokenisation, `<j-line>`/`<j-tok>` assembly, HTML escaping, line numbers |
| `server/shared/plugins/notebook_output_formatter/plugin.py` | `<notebook-cell>` detection, `<nb-row>` emission, cell type labels |
| `server/shared/plugins/mermaid_formatter/plugin.py` | Mermaid block detection, image rendering via multiple backends (iTerm2, Kitty, Sixel, Rich pixels) |
| `server/shared/plugins/diff_formatter/` | Unified diff rendering with multiple modes (unified, side-by-side, compact), syntax highlighting in diffs |
| `server/shared/plugins/hidden_content_filter/plugin.py` | Strips content marked as hidden from the output stream |
| `server/shared/plugins/formatter_pipeline/registry.py` | Plugin discovery, configuration loading, pipeline assembly |
| `server/shared/plugins/formatter_pipeline/pipeline.py` | `FormatterPipeline` — chains plugins in priority order |
| `server/shared/plugins/formatter_pipeline/protocol.py` | `FormatterPlugin` base protocol / `ConfigurableFormatter` mixin |
| `tui/j_markup_renderer.py` | TUI client-side renderer for `<j-code>` and `<j-table>`, ANSI rendering via rich |
| `tui/output_buffer.py` | TUI output buffer with `<nb-row>` 2-column table rendering |
| `tui/tests/test_j_markup_renderer.py` | Tests for the TUI renderer |
| `tui/pt_display.py` | Terminal presentation layer that invokes `rewrite_j_markup` |
