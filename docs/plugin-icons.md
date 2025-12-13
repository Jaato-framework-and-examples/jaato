# Plugin Icon Prompts for Nano Banana Pro

This document contains image generation prompts for creating consistent icons for each jaato plugin.

## Style Guidelines

All icons should follow these consistent style parameters:

**Base Style Prompt (append to each icon prompt):**
```
3D rendered icon, modern minimalist design, soft shadows, indigo and cyan color palette (#635bff primary, #4f46e5 darker accent, cyan highlights), clean geometric shapes, subtle gradients, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish, rounded corners
```

**Color Palette:**
| Color | Hex | Usage |
|-------|-----|-------|
| Primary Indigo | `#635bff` | Main icon elements |
| Dark Indigo | `#4f46e5` | Shadows, depth |
| Cyan Accent | `#00d4ff` | Highlights, energy |
| Success Green | `#10b981` | Positive actions |
| Warning Amber | `#f59e0b` | Caution elements |
| Background | `#1e1e2e` | Icon background |
| White | `#ffffff` | Bright accents |

---

## Core Infrastructure Plugins

### 1. Model Provider (`model_provider/`)
**Visual Concept:** Neural network hub connecting to multiple clouds

```
A 3D rendered brain-shaped hub with multiple glowing connection lines extending outward to floating cloud symbols, representing multi-provider AI integration. The brain pulses with indigo energy (#635bff), connection lines glow cyan (#00d4ff). Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 2. Plugin Registry (`registry.py`)
**Visual Concept:** Organized grid of plugin slots with a central coordinator

```
A 3D rendered circular registry dial or index wheel with multiple small plugin modules arranged in slots around its circumference. Center shows a glowing indigo core (#635bff) that coordinates all modules. Some slots glow active (cyan #00d4ff), others dim. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## Tool Execution Plugins

### 3. CLI Plugin (`cli/`)
**Visual Concept:** Terminal window with command prompt

```
A 3D rendered terminal window icon with a blinking cursor and command prompt symbol (>_). The terminal frame is dark with indigo (#635bff) glowing edges. Inside shows minimal text lines representing commands. A small lightning bolt accent indicates execution speed. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 4. MCP Plugin (`mcp/`)
**Visual Concept:** Protocol bridge connecting multiple servers

```
A 3D rendered bridge or connector hub with the letters "MCP" subtly integrated. Multiple server towers on different sides connected by glowing cyan (#00d4ff) protocol lines flowing through a central indigo (#635bff) gateway. Represents Model Context Protocol server orchestration. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 5. Background Plugin (`background/`)
**Visual Concept:** Running tasks in parallel lanes

```
A 3D rendered icon showing multiple horizontal lanes or tracks with small glowing orbs moving along them at different speeds. A circular progress indicator overlays the center. Represents parallel background task execution. Indigo (#635bff) tracks, cyan (#00d4ff) moving orbs. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## File & Content Management Plugins

### 6. File Edit Plugin (`file_edit/`)
**Visual Concept:** Document with diff markers and edit pencil

```
A 3D rendered document icon with visible diff markers (green + and red - lines on the side). A glowing indigo (#635bff) pencil or stylus hovers over it ready to edit. Small backup/history icon in corner. Represents file editing with version control. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 7. References Plugin (`references/`)
**Visual Concept:** Open book with floating citation markers

```
A 3D rendered open book or documentation icon with small floating citation bubbles or reference markers emerging from the pages. The bubbles glow cyan (#00d4ff), book has indigo (#635bff) accents. Represents documentation source injection. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 8. Multimodal Plugin (`multimodal/`)
**Visual Concept:** Eye viewing an image frame

```
A 3D rendered stylized eye icon with a small image/picture frame reflected in the pupil. The iris glows with indigo (#635bff) and cyan (#00d4ff) gradients. Represents image viewing and multimodal understanding. Small @ symbol accent. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 9. Slash Command Plugin (`slash_command/`)
**Visual Concept:** Forward slash with template variables

```
A 3D rendered large forward slash (/) symbol in bold indigo (#635bff) with small template placeholder brackets {{ }} floating nearby in cyan (#00d4ff). Represents slash command templating. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## Memory & State Management Plugins

### 10. Memory Plugin (`memory/`)
**Visual Concept:** Brain with memory crystals

```
A 3D rendered stylized brain icon with small glowing crystal or gem shapes embedded within it, representing stored memories. The brain outline is indigo (#635bff), memory crystals glow cyan (#00d4ff). Some crystals appear brighter (recently accessed). Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 11. Session Plugin (`session/`)
**Visual Concept:** Conversation bubble with save/load arrows

```
A 3D rendered chat bubble or conversation icon with a circular arrow (refresh/restore) symbol integrated. Small disk/save icon in corner. Represents session persistence and resumption. Indigo (#635bff) bubble, cyan (#00d4ff) arrows. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 12. Todo Plugin (`todo/`)
**Visual Concept:** Checklist with progress bar

```
A 3D rendered checklist icon with three items - one checked (green #10b981), one in progress (indigo #635bff with subtle animation glow), one pending (dim). A progress bar runs along the bottom. Represents plan tracking and task management. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## User Interaction & Permissions Plugins

### 13. Permission Plugin (`permission/`)
**Visual Concept:** Shield with approve/deny indicators

```
A 3D rendered shield icon with a keyhole or lock symbol in the center. One side glows green (#10b981) for approve, other side has a subtle red tint for deny. A question mark hovers above for interactive approval. Represents access control and permission management. Main shield is indigo (#635bff). Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 14. Clarification Plugin (`clarification/`)
**Visual Concept:** Question bubbles with response options

```
A 3D rendered icon showing a large question mark (?) with multiple small response option bubbles (radio buttons, checkboxes) floating around it. The question mark glows indigo (#635bff), option bubbles glow cyan (#00d4ff). Represents requesting user clarification with multiple choice support. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## Context Management (GC) Plugins

### 15. GC Truncate Plugin (`gc_truncate/`)
**Visual Concept:** Scissors cutting a document stack

```
A 3D rendered icon showing a stack of document layers with glowing scissors cutting away the bottom/oldest layers. The kept layers glow bright, cut layers fade. Represents simple truncation-based garbage collection. Indigo (#635bff) scissors, cyan (#00d4ff) cut line. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 16. GC Summarize Plugin (`gc_summarize/`)
**Visual Concept:** Document stack compressing into summary

```
A 3D rendered icon showing multiple documents being compressed/squeezed into a single compact summary document. Compression lines or arrows point inward. The original documents are semi-transparent, the summary glows bright indigo (#635bff). Represents compression via summarization. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 17. GC Hybrid Plugin (`gc_hybrid/`)
**Visual Concept:** Generational layers with different treatments

```
A 3D rendered icon showing three horizontal tiers or generations. Top tier (recent) is bright and intact, middle tier is compressed/summarized, bottom tier is being truncated/faded away. Represents Java-style generational garbage collection. Indigo (#635bff) tiers, cyan (#00d4ff) transition effects. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## Specialized Capability Plugins

### 18. Web Search Plugin (`web_search/`)
**Visual Concept:** Magnifying glass over globe

```
A 3D rendered magnifying glass hovering over a stylized globe or world icon. Search results appear as small floating cards behind. The magnifying glass frame is indigo (#635bff), lens has cyan (#00d4ff) glow. Small DuckDuckGo-style duck silhouette accent optional. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 19. Subagent Plugin (`subagent/`)
**Visual Concept:** Parent node spawning child agents

```
A 3D rendered icon showing a large central agent node (circle with inner glow) with smaller child agent nodes connected by delegation lines branching outward. The parent glows indigo (#635bff), children glow cyan (#00d4ff). Connection lines show task flow. Represents subagent spawning and delegation. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

### 20. Calculator Plugin (`calculator/`)
**Visual Concept:** Mathematical symbols in geometric arrangement

```
A 3D rendered icon showing mathematical operation symbols (+, ×, =) arranged in a clean geometric pattern or floating around a result display. The symbols glow indigo (#635bff) and cyan (#00d4ff). Represents mathematical calculation capabilities. Modern minimalist design, soft shadows, dark slate background (#1e1e2e), 512x512 pixels, professional software icon style, slight glossy finish.
```

---

## Usage Instructions

### Generating Icons with Nano Banana Pro

1. **Copy the base style prompt** from the Style Guidelines section
2. **Select a plugin prompt** from the relevant category
3. **Combine them** - paste the plugin-specific prompt first, then append the base style elements if not already included
4. **Generate at 512x512** for best quality
5. **Export both PNG and SVG** versions if possible

### Storage Location

Store generated icons in:
```
docs/api/assets/images/plugins/
```

### Recommended File Naming Convention

```
docs/api/assets/images/plugins/
├── plugin-cli.png
├── plugin-cli.svg
├── plugin-mcp.png
├── plugin-mcp.svg
├── plugin-file-edit.png
├── plugin-file-edit.svg
├── plugin-todo.png
├── plugin-todo.svg
├── plugin-web-search.png
├── plugin-web-search.svg
├── plugin-permission.png
├── plugin-permission.svg
├── plugin-session.png
├── plugin-session.svg
├── plugin-memory.png
├── plugin-memory.svg
├── plugin-subagent.png
├── plugin-subagent.svg
├── plugin-background.png
├── plugin-background.svg
├── plugin-clarification.png
├── plugin-clarification.svg
├── plugin-references.png
├── plugin-references.svg
├── plugin-multimodal.png
├── plugin-multimodal.svg
├── plugin-slash-command.png
├── plugin-slash-command.svg
├── plugin-calculator.png
├── plugin-calculator.svg
├── plugin-gc-truncate.png
├── plugin-gc-truncate.svg
├── plugin-gc-summarize.png
├── plugin-gc-summarize.svg
├── plugin-gc-hybrid.png
├── plugin-gc-hybrid.svg
├── plugin-model-provider.png
├── plugin-model-provider.svg
├── plugin-registry.png
└── plugin-registry.svg
```

### Integration Points

**README.md:** Add icons inline with plugin descriptions in the Available Plugins section
```markdown
| ![cli](docs/api/assets/images/plugins/plugin-cli.png) | **cli** | Execute shell commands... |
```

**HTML Docs:** Reference from plugin pages:
```html
<img src="../../assets/images/plugins/plugin-cli.png" alt="CLI Plugin" width="48">
```
- `docs/api/api-reference/plugins/*.html`
- `docs/api/guides/tool-plugins.html`
- Plugin cards on the main index page

**Individual Plugin Dirs (optional):** Symlink or copy `icon.png` to each plugin's directory for local reference

---

## Alternative Simplified Prompts

If the detailed prompts produce inconsistent results, try these simplified versions:

| Plugin | Simplified Prompt |
|--------|-------------------|
| cli | "3D terminal icon with cursor, indigo glow, dark background" |
| mcp | "3D server bridge connector icon, protocol lines, indigo cyan" |
| file_edit | "3D document with pencil and diff markers, indigo style" |
| permission | "3D shield with lock, green approve red deny, indigo" |
| memory | "3D brain with glowing memory crystals, indigo cyan" |
| session | "3D chat bubble with save icon, indigo style" |
| todo | "3D checklist with progress bar, indigo green checkmarks" |
| web_search | "3D magnifying glass over globe, indigo cyan glow" |
| subagent | "3D agent nodes hierarchy, parent spawning children, indigo" |
| background | "3D parallel running tracks with orbs, indigo cyan" |
| gc_truncate | "3D scissors cutting document stack, indigo style" |
| gc_summarize | "3D documents compressing into one, indigo style" |
| gc_hybrid | "3D three-tier generational layers, indigo cyan" |
| clarification | "3D question mark with choice bubbles, indigo cyan" |
| references | "3D open book with citation markers, indigo style" |
| multimodal | "3D eye with image in pupil, indigo cyan glow" |
| slash_command | "3D forward slash with template brackets, indigo" |
| calculator | "3D math symbols arrangement, indigo cyan" |
| model_provider | "3D brain hub with cloud connections, indigo cyan" |
| registry | "3D circular registry dial with plugin slots, indigo" |
