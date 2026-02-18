# Agent Presentation Awareness

**Status:** Brainstorm / RFC
**Date:** 2025-02-18
**Problem:** Agents generate output (tables, code blocks, diagrams) without
knowing the display constraints of the client, leading to broken rendering on
narrow or capability-limited surfaces (mobile Telegram, small terminals, plain-text APIs).

---

## Problem Statement

The model produces markdown tables, wide code blocks, and other formatted output
with no knowledge of:

1. **Display width** — a Docker `ps` table that looks fine at 120 columns wraps
   disastrously at 45 columns on a Telegram mobile client.
2. **Format capabilities** — some clients don't render markdown tables at all
   (plain-text APIs), some can render images inline (web), some have limited
   markdown (Telegram), some support Mermaid diagrams (TUI with sixel).
3. **Content strategy** — even if the client *can* render a table, a narrow
   viewport might be better served by a vertical key-value list or a summary.

### Current State

The plumbing for `terminal_width` already exists end-to-end:

```
Client (IPC/WS)
  → ClientConfigRequest.terminal_width
  → SessionManager._apply_client_config() stores in _client_config
  → JaatoServer.terminal_width setter
  → JaatoClient.set_terminal_width()
  → JaatoSession._terminal_width
  → FormatterPipeline.set_console_width()
```

**Gap:** `_terminal_width` is used exclusively for **client-side formatting**
(enrichment notifications, formatter pipeline). It is **never injected into the
model's system instructions**, so the model has zero awareness of the display.

There is also no concept of **capabilities** beyond width.

---

## Design: `PresentationContext`

### Data Model

```python
@dataclass
class PresentationContext:
    """Display capabilities and constraints of the connected client.

    Assembled by the client at connection time and sent to the server.
    Used in two places:
      1. System instructions — so the model adapts its output format.
      2. Formatter pipeline — as a safety net for client-side reformatting.
    """

    # ── Dimensions ──────────────────────────────────────────────
    content_width: int = 80
    content_height: Optional[int] = None   # None = unlimited scroll

    # ── Format capabilities ─────────────────────────────────────
    supports_markdown: bool = True
    supports_tables: bool = True           # markdown pipe-tables
    supports_code_blocks: bool = True      # fenced ``` blocks
    supports_images: bool = False          # inline image rendering
    supports_rich_text: bool = True        # bold, italic, links
    supports_unicode: bool = True          # wide chars, emoji
    supports_mermaid: bool = False         # diagram rendering

    # ── Client hint ─────────────────────────────────────────────
    client_type: str = "terminal"          # terminal | web | telegram | slack | api
```

### System Instruction Generation

`PresentationContext` generates a compact instruction block:

```python
def to_system_instruction(self) -> str:
    """Generate display-context system instructions for the model."""
    lines = [f"## Display Context",
             f"Output width: {self.content_width} characters."]

    if self.content_width < 60:
        lines.append(
            "This is a NARROW display. Avoid markdown tables — "
            "use vertical key: value lists instead. "
            "Keep lines under {self.content_width} characters."
        )
    elif self.content_width < 100:
        lines.append(
            "Prefer compact tables (≤3-4 columns). "
            "For wider data, use key: value format."
        )

    if not self.supports_tables:
        lines.append("Markdown tables are NOT supported. Use lists or indented text.")

    if not self.supports_code_blocks:
        lines.append("Fenced code blocks are NOT supported. Indent code with 4 spaces.")

    if self.supports_images:
        lines.append("Inline images are supported (the client can render them).")

    if not self.supports_markdown:
        lines.append("Markdown is NOT supported. Use plain text only.")

    return "\n".join(lines)
```

### Integration Points

#### 1. Wire into `ClientConfigRequest`

Extend the existing event with optional presentation fields:

```python
@dataclass
class ClientConfigRequest(Event):
    # ... existing fields ...
    terminal_width: Optional[int] = None

    # New: presentation capabilities
    presentation: Optional[Dict[str, Any]] = None
    # Keys: content_width, supports_tables, supports_code_blocks,
    #        supports_images, client_type, ...
```

Backwards-compatible: old clients send `terminal_width` only, which still works.
New clients send `presentation` dict with richer info. Server falls back to
constructing a default `PresentationContext(content_width=terminal_width)` when
only `terminal_width` is present.

#### 2. Store on `JaatoSession`

```python
class JaatoSession:
    def __init__(self, ...):
        self._presentation_context: PresentationContext = PresentationContext()

    def set_presentation_context(self, ctx: PresentationContext) -> None:
        self._presentation_context = ctx
        self._terminal_width = ctx.content_width  # keep backwards compat
```

#### 3. Inject into System Instructions

In `JaatoRuntime.get_system_instructions()`, add a new assembly step between
step 4 (formatter pipeline) and step 5 (task completion):

```python
# 4.5 Presentation context (display capabilities)
if self._presentation_context:
    ctx_instruction = self._presentation_context.to_system_instruction()
    if ctx_instruction:
        result_parts.append(ctx_instruction)
```

Since `get_system_instructions()` lives on the runtime (shared), the context
should be passed as a parameter rather than stored on the runtime:

```python
def get_system_instructions(
    self,
    plugin_names=None,
    additional=None,
    presentation_context=None,   # NEW
) -> Optional[str]:
```

The session calls it with its own context:

```python
instructions = self._runtime.get_system_instructions(
    plugin_names=...,
    additional=...,
    presentation_context=self._presentation_context,
)
```

#### 4. Formatter Pipeline Safety Net

Add a `TableReformatter` stage to the formatter pipeline that:
- Detects markdown tables in model output
- Measures their rendered width
- If wider than `content_width`, applies a reformatting strategy:
  - **Truncate cells** — `some long val...`
  - **Transpose** — flip rows/columns for narrow displays
  - **Vertical list** — convert each row to a key: value block

This is a safety net for when the model ignores the system instruction hint.

---

## Client Declarations

Each client type creates its context at connection time:

### TUI Client (terminal)
```python
PresentationContext(
    content_width=shutil.get_terminal_size().columns - 6,
    supports_markdown=True,
    supports_tables=True,
    supports_code_blocks=True,
    supports_mermaid=has_sixel,
    supports_unicode=True,
    client_type="terminal",
)
```

### Telegram Bot
```python
PresentationContext(
    content_width=45,         # typical mobile viewport
    supports_markdown=True,   # limited: bold, italic, code, links
    supports_tables=False,    # monospace pipe-tables break on mobile
    supports_code_blocks=True,
    supports_images=True,     # can send images as separate messages
    client_type="telegram",
)
```

### Web Client
```python
PresentationContext(
    content_width=100,        # responsive, but reasonable default
    supports_markdown=True,
    supports_tables=True,
    supports_code_blocks=True,
    supports_images=True,
    supports_mermaid=True,
    client_type="web",
)
```

### Plain API / Headless
```python
PresentationContext(
    content_width=120,
    supports_markdown=False,
    supports_tables=False,
    supports_code_blocks=False,
    supports_images=False,
    client_type="api",
)
```

---

## Dynamic Updates

Terminal resize is already handled (`SIGWINCH` → width update). Extend this to
update the full `PresentationContext`:

```python
# In TUI client, on terminal resize:
new_ctx = PresentationContext(content_width=new_width, ...)
await client.send_event(ClientConfigRequest(presentation=asdict(new_ctx)))
```

The server updates the session's context and, on the next turn, the model sees
the updated display constraints in its system instructions.

**Important:** Mid-turn updates only affect the *next* turn's system
instructions, not the current one. This is acceptable — the model is already
generating output for the current turn.

---

## Token Cost Analysis

The presentation context instruction block is compact:

| Scenario | Instruction Size |
|----------|-----------------|
| Wide terminal (default) | ~30 tokens (just width mention) |
| Narrow mobile | ~80 tokens (width + table/list guidance) |
| No-markdown API | ~60 tokens (width + format restrictions) |

This is negligible compared to tool schemas (~2000+ tokens) and base system
instructions (~500+ tokens).

---

## Alternative Considered: Client-Only Reformatting

**Why not just reformat on the client side?**

Client-side reformatting can fix **layout** (wrap cells, truncate columns) but
cannot fix **content strategy**. Only the model can decide:

- "This 10-column table would be better as a 3-column summary on a narrow screen"
- "Instead of a table, let me describe the top 3 results in prose"
- "Let me group by status and show counts instead of listing all 50 rows"

The hybrid approach (model hints + client safety net) gives the best results.

---

## Implementation Phases

### Phase 1: System Instruction Injection (Minimum Viable)
- Add `PresentationContext` dataclass
- Wire `to_system_instruction()` into `get_system_instructions()`
- Pass `content_width` (already available) to construct default context
- **Result:** Model starts adapting tables to terminal width

### Phase 2: Extended Client Capabilities
- Extend `ClientConfigRequest` with `presentation` dict
- TUI client sends full capabilities
- Add client_type-specific defaults

### Phase 3: Formatter Pipeline Safety Net
- Add `TableReformatter` to formatter pipeline
- Detect too-wide tables in output stream
- Apply truncation/transposition as fallback

### Phase 4: Dynamic Context Updates
- Handle resize events updating presentation context
- Support mid-session client capability changes (e.g., window → mobile view)

---

## Open Questions

1. **Should presentation context live on Runtime or Session?**
   Session (recommended) — different subagents could have different output targets
   (e.g., main agent writes to TUI, subagent writes to a file).

2. **Should we include a `max_table_columns` hint?**
   Probably yes for narrow displays. `content_width=45` + `max_table_columns=3`
   gives the model a clearer constraint.

3. **How verbose should the instruction be?**
   Minimal. One line for width, one line per disabled capability. The model
   doesn't need a lecture — a concise constraint is enough.

4. **Should the formatter pipeline rewrite tables in streaming mode?**
   Tricky — table rows arrive incrementally. Likely need to buffer until the
   table is complete (detect closing `|` row), then reformat. This is Phase 3
   complexity.
