# Agent Presentation Awareness

**Status:** Implemented (Phase 1 + 2)
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

### Current State (Before This Change)

The plumbing for `terminal_width` already existed end-to-end:

```
Client (IPC/WS)
  → ClientConfigRequest.terminal_width
  → SessionManager._apply_client_config() stores in _client_config
  → JaatoServer.terminal_width setter
  → JaatoClient.set_terminal_width()
  → JaatoSession._terminal_width
  → FormatterPipeline.set_console_width()
```

**Gap:** `_terminal_width` was used exclusively for **client-side formatting**
(enrichment notifications, formatter pipeline). It was **never injected into the
model's system instructions**, so the model had zero awareness of the display.

There was also no concept of **capabilities** beyond width.

---

## Design: `PresentationContext`

### Data Model

Defined in `shared/plugins/model_provider/types.py`:

```python
@dataclass
class PresentationContext:
    # ── Dimensions ──────────────────────────────────────────────
    content_width: int = 80
    content_height: Optional[int] = None

    # ── Format capabilities ─────────────────────────────────────
    supports_markdown: bool = True
    supports_tables: bool = True
    supports_code_blocks: bool = True
    supports_images: bool = False
    supports_rich_text: bool = True
    supports_unicode: bool = True
    supports_mermaid: bool = False
    supports_expandable_content: bool = False

    # ── Client hint ─────────────────────────────────────────────
    client_type: str = "terminal"
```

### `supports_expandable_content`

Some presentation layers (Telegram inline keyboards, web `<details>` blocks,
TUI scrollable panels) can collapse overflow behind an expand/click affordance.
When `supports_expandable_content = True`:

- The model is told it may use full-width tables and detailed output freely.
- The **client** is responsible for detecting overflow and wrapping it in its
  native expandable widget.
- This keeps the overflow UX decision in the client where it belongs — the model
  produces content, the client decides how to present overflow.

When `False` (the default), the model is given width constraints and asked to
adapt its formatting (vertical lists, compact tables, etc.).

### System Instruction Generation

`PresentationContext.to_system_instruction()` generates a compact instruction
block injected as step 5 in `get_system_instructions()`:

**Narrow display (< 60 chars):**
```
## Display Context
Output width: 45 characters.
This is a NARROW display. Avoid markdown tables — use vertical key: value lists instead. Keep lines under 45 characters.
Markdown tables are NOT supported. Use bullet lists or indented key: value pairs.
```

**Medium display (60-99 chars):**
```
## Display Context
Output width: 80 characters.
Prefer compact tables (3-4 columns max). For wider data, use vertical key: value format.
```

**Wide display (100+ chars):**
```
## Display Context
Output width: 180 characters.
```

**Expandable-content client:**
```
## Display Context
Output width: 45 characters.
The client can collapse wide or long content behind an expandable control. You may use full-width tables and detailed output freely.
```

### Integration Points

#### 1. `ClientConfigRequest` — Event transport

```python
@dataclass
class ClientConfigRequest(Event):
    # ... existing fields ...
    terminal_width: Optional[int] = None     # backwards compat
    presentation: Optional[Dict[str, Any]] = None  # NEW
```

Backwards-compatible: old clients send `terminal_width` only. New clients send
the `presentation` dict. Server falls back to `terminal_width`-only when no
presentation dict is present.

#### 2. `JaatoSession` — Storage and propagation

```python
class JaatoSession:
    self._presentation_context: Optional[PresentationContext] = None

    def set_presentation_context(self, ctx: PresentationContext) -> None:
        self._presentation_context = ctx
        self._terminal_width = ctx.content_width  # backwards compat
```

Propagation chain: `JaatoServer.set_presentation_context()` →
`JaatoClient.set_presentation_context()` → `JaatoSession.set_presentation_context()`

#### 3. `JaatoRuntime.get_system_instructions()` — Injection

New parameter `presentation_context` accepted. Injected as step 5 in the
assembly pipeline (between formatter pipeline and task completion instruction):

```python
# 5. Presentation context (client display constraints and capabilities)
if presentation_context is not None:
    ctx_instruction = presentation_context.to_system_instruction()
    if ctx_instruction:
        result_parts.append(ctx_instruction)
```

#### 4. `SessionManager` — Server-side construction

`_apply_client_config()` and `_apply_client_config_to_server()` construct
`PresentationContext` from the event's `presentation` dict and call
`server.set_presentation_context()`.

---

## Client Declarations

Each client type creates its context at connection time:

### TUI Client (terminal)
```python
presentation = {
    "content_width": terminal_width - 4,
    "supports_markdown": True,
    "supports_tables": True,
    "supports_code_blocks": True,
    "supports_images": False,
    "supports_rich_text": True,
    "supports_unicode": True,
    "supports_mermaid": False,
    "supports_expandable_content": False,
    "client_type": "terminal",
}
```

### Telegram Bot (example)
```python
PresentationContext(
    content_width=45,
    supports_markdown=True,
    supports_tables=False,
    supports_code_blocks=True,
    supports_images=True,
    supports_expandable_content=True,  # inline keyboard buttons
    client_type="telegram",
)
```

### Web Client (example)
```python
PresentationContext(
    content_width=100,
    supports_markdown=True,
    supports_tables=True,
    supports_code_blocks=True,
    supports_images=True,
    supports_mermaid=True,
    supports_expandable_content=True,  # <details> blocks
    client_type="web",
)
```

### Plain API / Headless (example)
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

Terminal resize is already handled (`SIGWINCH` → width update). The same
`ClientConfigRequest` mechanism can send updated presentation context:

```python
# In a client, on viewport resize:
await client.send_event(ClientConfigRequest(
    presentation=new_context.to_dict()
))
```

**Important:** Mid-turn updates only affect the *next* turn's system
instructions, not the current one. This is acceptable — the model is already
generating output for the current turn.

---

## Token Cost Analysis

The presentation context instruction block is compact:

| Scenario | Instruction Size |
|----------|-----------------|
| Wide terminal (default) | ~15 tokens (just width mention) |
| Narrow mobile | ~60 tokens (width + table/list guidance) |
| Expandable-content client | ~40 tokens (width + expandable note) |
| No-markdown API | ~50 tokens (width + format restrictions) |

This is negligible compared to tool schemas (~2000+ tokens) and base system
instructions (~500+ tokens).

---

## Overflow Handling Philosophy

**The model decides *what* to output. The client decides *how* to present overflow.**

- On narrow displays without expandable content: the model is asked to use
  compact formats (vertical lists, fewer columns).
- On displays with expandable content: the model outputs freely, and the client
  wraps overflow in its native expand/collapse widget (Telegram inline buttons,
  `<details>`, scrollable TUI panel, etc.).
- Server-side table reformatting (TableReformatter) was explicitly rejected in
  favour of this approach — each client knows its own UX idioms best.

---

## Resolved Questions

1. **Presentation context lives on Session** (not Runtime) — different subagents
   could have different output targets.

2. **Instruction verbosity:** Minimal. One line for width, one line per disabled
   capability. No lectures.

3. **Overflow UX:** Client-side, not pipeline-side. The `supports_expandable_content`
   flag tells the model it can be generous with content.
