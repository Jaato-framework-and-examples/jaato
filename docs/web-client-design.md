# Web Client Design Document

## Overview

This document outlines the design for a web-based client for jaato that connects via WebSockets, providing feature parity with the existing rich-client TUI.

## Goals

1. **Feature parity** with rich-client for core functionality
2. **Responsive design** for desktop and tablet use
3. **Accessibility** compliant (WCAG 2.1 AA)
4. **Low latency** real-time streaming experience
5. **Offline-resilient** with reconnection handling

## Non-Goals (Initial Release)

- Mobile-first design (desktop priority)
- PWA/offline mode
- Multi-session tabs (single session per browser tab)

---

## Technology Stack

### Frontend Framework: **React 18+**

**Rationale:**
- Mature ecosystem with excellent TypeScript support
- Efficient virtual DOM for high-frequency updates (streaming)
- Rich component libraries available
- Team familiarity (assumed)

**Alternatives considered:**
- Vue 3: Good, but smaller ecosystem for our needs
- Svelte: Less mature for complex state management
- Vanilla JS: Too much boilerplate for this complexity

### State Management: **Zustand**

**Rationale:**
- Lightweight (1.5kb), minimal boilerplate
- Built-in subscriptions for real-time updates
- Works well with React concurrent features
- Easy to split into domain slices

**Alternatives considered:**
- Redux Toolkit: Overkill for our use case
- Jotai/Recoil: Atomic model less suited to our event-driven architecture
- React Context: Performance issues with frequent updates

### Styling: **Tailwind CSS + CSS Variables**

**Rationale:**
- Utility-first for rapid iteration
- CSS variables for theming (dark/light/custom)
- No runtime cost
- Excellent responsive utilities

### Code Highlighting: **Shiki**

**Rationale:**
- VSCode-quality syntax highlighting
- WASM-based, runs in browser
- Supports all languages we need
- Theme-aware

### Markdown Rendering: **react-markdown + remark-gfm**

**Rationale:**
- GFM support (tables, task lists)
- Plugin ecosystem for custom rendering
- Integrates with Shiki for code blocks

### WebSocket: **Native WebSocket API + reconnecting-websocket**

**Rationale:**
- Native API is sufficient
- `reconnecting-websocket` adds automatic reconnection with backoff
- No need for Socket.io complexity (our protocol is simple JSON)

### Build Tool: **Vite**

**Rationale:**
- Fast HMR for development
- Efficient production builds
- First-class TypeScript support

---

## Architecture

### Component Hierarchy

```
App
├── ConnectionProvider          # WebSocket connection context
│   ├── Header                  # Logo, connection status, model selector
│   ├── MainLayout
│   │   ├── Sidebar (collapsible)
│   │   │   ├── AgentList       # Multi-agent switcher
│   │   │   ├── SessionList     # Session management
│   │   │   └── ToolTree        # Active/completed tools
│   │   ├── ContentArea
│   │   │   ├── PlanPanel       # Sticky plan display (top)
│   │   │   ├── OutputPane      # Scrollable message stream
│   │   │   │   ├── MessageBlock
│   │   │   │   ├── ToolCallBlock
│   │   │   │   └── SystemMessage
│   │   │   └── InputArea       # User prompt input
│   │   │       ├── PromptInput
│   │   │       ├── AttachmentBar
│   │   │       └── SendButton
│   │   └── StatusBar           # Token usage, GC status
│   └── ModalLayer
│       ├── PermissionModal     # Tool approval prompts
│       ├── ClarificationModal  # Multi-question dialogs
│       └── SettingsModal       # Configuration
└── ToastProvider               # Notifications
```

### State Architecture (Zustand Slices)

```typescript
// stores/connection.ts
interface ConnectionStore {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
  clientId: string | null;
  serverInfo: ServerInfo | null;
  connect: (url: string) => void;
  disconnect: () => void;
  send: (event: ClientEvent) => void;
}

// stores/agents.ts
interface AgentStore {
  agents: Map<string, Agent>;
  selectedAgentId: string;
  createAgent: (id: string, name: string, type: string) => void;
  updateStatus: (id: string, status: AgentStatus) => void;
  appendOutput: (id: string, source: string, text: string, mode: OutputMode) => void;
  selectAgent: (id: string) => void;
}

// stores/tools.ts
interface ToolStore {
  activeTools: Map<string, ToolCall>;
  completedTools: ToolCall[];
  startTool: (call: ToolCallStart) => void;
  endTool: (id: string, result: ToolCallEnd) => void;
  updateOutput: (id: string, output: string) => void;
}

// stores/permissions.ts
interface PermissionStore {
  pendingRequest: PermissionRequest | null;
  history: PermissionDecision[];
  setRequest: (request: PermissionRequest) => void;
  respond: (requestId: string, response: string) => void;
  clearRequest: () => void;
}

// stores/plan.ts
interface PlanStore {
  steps: PlanStep[];
  expanded: boolean;
  updatePlan: (steps: PlanStep[]) => void;
  toggleExpanded: () => void;
  clear: () => void;
}

// stores/context.ts
interface ContextStore {
  totalTokens: number;
  promptTokens: number;
  outputTokens: number;
  contextLimit: number;
  percentUsed: number;
  gcThreshold: number;
  gcStrategy: string;
  update: (usage: ContextUpdate) => void;
}

// stores/ui.ts
interface UIStore {
  theme: 'dark' | 'light' | 'system';
  sidebarOpen: boolean;
  planPanelOpen: boolean;
  setTheme: (theme: string) => void;
  toggleSidebar: () => void;
  togglePlanPanel: () => void;
}
```

### Event Flow

```
WebSocket Message
      ↓
ConnectionProvider.onMessage()
      ↓
parseEvent(json) → Event
      ↓
eventRouter.dispatch(event)
      ↓
┌─────────────────────────────────────────────┐
│ Route by event.type:                        │
│                                             │
│ agent.output    → agentStore.appendOutput() │
│ agent.status    → agentStore.updateStatus() │
│ tool.start      → toolStore.startTool()     │
│ tool.end        → toolStore.endTool()       │
│ permission.req  → permissionStore.set()     │
│ context.updated → contextStore.update()     │
│ plan.updated    → planStore.updatePlan()    │
│ ...                                         │
└─────────────────────────────────────────────┘
      ↓
Zustand notifies subscribed components
      ↓
React re-renders affected components
```

---

## Event Protocol Mapping

### Server → Client Events

| Event Type | Store Action | UI Component |
|------------|--------------|--------------|
| `connected` | connectionStore.setConnected() | Header (status indicator) |
| `agent.created` | agentStore.createAgent() | AgentList |
| `agent.output` | agentStore.appendOutput() | OutputPane |
| `agent.status_changed` | agentStore.updateStatus() | AgentList, Spinner |
| `agent.completed` | agentStore.markComplete() | OutputPane (summary) |
| `tool.call_start` | toolStore.startTool() | ToolTree |
| `tool.call_end` | toolStore.endTool() | ToolTree |
| `tool.output` | toolStore.updateOutput() | ToolTree (expandable) |
| `permission.requested` | permissionStore.setRequest() | PermissionModal |
| `permission.resolved` | permissionStore.clearRequest() | ToolTree (status) |
| `clarification.question` | clarificationStore.addQuestion() | ClarificationModal |
| `context.updated` | contextStore.update() | StatusBar |
| `plan.updated` | planStore.updatePlan() | PlanPanel |
| `plan.cleared` | planStore.clear() | PlanPanel |
| `error` | toastStore.error() | Toast notification |
| `system_message` | toastStore.info() | Toast notification |
| `retry` | toastStore.warning() | Toast notification |
| `session.info` | sessionStore.setInfo() | SessionList |

### Client → Server Events

| User Action | Event Type | Payload |
|-------------|------------|---------|
| Submit prompt | `message.send` | `{ text, attachments }` |
| Permission response | `permission.response` | `{ request_id, response }` |
| Clarification answer | `clarification.response` | `{ request_id, answer }` |
| Stop generation | `stop` | `{ agent_id? }` |
| Execute command | `command` | `{ name, args }` |
| Request history | `history` | `{}` |
| Client config | `client.config` | `{ cwd, env, terminal_width }` |

---

## UI Components Design

### 1. OutputPane (Central Component)

**Responsibilities:**
- Render streaming message content
- Handle append vs write modes
- Virtualized scrolling for performance
- Auto-scroll with smart pause on user scroll

**Implementation:**
```typescript
interface OutputPaneProps {
  agentId: string;
}

// Uses react-window for virtualization
// Each message block is a variable-height item
// Markdown rendered with react-markdown
// Code blocks highlighted with Shiki
```

**Message Block Types:**
- `UserMessage` - User input with avatar
- `ModelMessage` - Streaming AI response
- `ToolCallMessage` - Tool execution with expand/collapse
- `SystemMessage` - Info/warning/error notices
- `PlanMessage` - Inline plan display

### 2. PermissionModal

**Requirements:**
- Modal overlay (blocks interaction)
- Display formatted prompt lines
- Syntax-highlighted diffs when `format_hint === "diff"`
- Response buttons with keyboard shortcuts
- Warnings displayed prominently

**Response Options Display:**
```
┌─────────────────────────────────────────────┐
│ Permission Required                          │
├─────────────────────────────────────────────┤
│ Tool: file_edit                             │
│                                              │
│ ┌─────────────────────────────────────────┐ │
│ │ - old line                              │ │
│ │ + new line                              │ │
│ └─────────────────────────────────────────┘ │
│                                              │
│ ⚠️ Warning: Modifies production file         │
│                                              │
│ [y] Yes  [n] No  [a] Always  [t] This turn  │
└─────────────────────────────────────────────┘
```

### 3. PlanPanel (Sticky Header)

**Requirements:**
- Sticky position at top of content area
- Collapsible (show/hide details)
- Step status indicators: ○ pending, ◐ in_progress, ● completed, ✗ failed
- Current step highlighted
- Smooth transitions

**Compact View:**
```
┌────────────────────────────────────────────┐
│ Plan: 3/5 ● ● ● ◐ ○                    [▼] │
└────────────────────────────────────────────┘
```

**Expanded View:**
```
┌────────────────────────────────────────────┐
│ Plan Progress                          [▲] │
├────────────────────────────────────────────┤
│ ● Research existing code                   │
│ ● Design component structure               │
│ ● Implement core functionality             │
│ ◐ Write tests                              │
│ ○ Update documentation                     │
└────────────────────────────────────────────┘
```

### 4. ToolTree (Sidebar)

**Requirements:**
- Hierarchical display (nested tool calls)
- Real-time status updates
- Expandable output preview
- Permission status inline
- Duration display on completion

**Structure:**
```
Tools
├─ ● read_file (0.2s)
│  └─ src/main.ts
├─ ◐ bash [awaiting permission]
│  └─ npm test
└─ ○ file_edit (pending)
   └─ src/utils.ts
```

### 5. StatusBar

**Requirements:**
- Token usage bar (visual + numeric)
- GC threshold indicator
- Model name display
- Connection status

**Layout:**
```
┌────────────────────────────────────────────────────────────┐
│ gemini-2.5-pro │ ████████░░░░░░░ 52% (104k/200k) │ GC: 80% │
└────────────────────────────────────────────────────────────┘
```

### 6. InputArea

**Requirements:**
- Multi-line text input (auto-expand)
- File attachment support (drag & drop, button)
- Keyboard shortcuts (Enter to send, Shift+Enter for newline)
- Disable during permission prompts
- Command completion (@file, /command, model names)

---

## Theming System

### CSS Variable Structure

```css
:root {
  /* Base palette */
  --color-primary: #6366f1;
  --color-secondary: #8b5cf6;
  --color-success: #22c55e;
  --color-warning: #f59e0b;
  --color-error: #ef4444;

  /* Semantic */
  --color-bg: #0f172a;
  --color-surface: #1e293b;
  --color-text: #f1f5f9;
  --color-text-muted: #94a3b8;
  --color-border: #334155;

  /* Plan status */
  --color-plan-pending: var(--color-text-muted);
  --color-plan-progress: var(--color-warning);
  --color-plan-complete: var(--color-success);
  --color-plan-failed: var(--color-error);

  /* Diff colors */
  --color-diff-add: #22c55e;
  --color-diff-remove: #ef4444;
  --color-diff-context: var(--color-text-muted);
}

[data-theme="light"] {
  --color-bg: #ffffff;
  --color-surface: #f8fafc;
  --color-text: #0f172a;
  --color-text-muted: #64748b;
  --color-border: #e2e8f0;
}
```

### Theme Switching

```typescript
// Persist to localStorage
// Respect system preference via matchMedia
// Allow override via settings
```

---

## Keyboard Shortcuts

| Action | Shortcut | Context |
|--------|----------|---------|
| Send message | `Enter` | Input focused |
| New line | `Shift+Enter` | Input focused |
| Stop generation | `Escape` | Any |
| Toggle plan | `Ctrl+P` | Any |
| Toggle sidebar | `Ctrl+B` | Any |
| Focus input | `/` | Not in input |
| Permission: Yes | `y` | Permission modal |
| Permission: No | `n` | Permission modal |
| Permission: Always | `a` | Permission modal |
| Permission: Turn | `t` | Permission modal |
| Scroll up | `PageUp` | Output focused |
| Scroll down | `PageDown` | Output focused |

---

## Error Handling & Resilience

### Connection Loss

```typescript
const reconnectionStrategy = {
  maxRetries: Infinity,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
};

// On disconnect:
// 1. Show "Reconnecting..." banner
// 2. Attempt reconnection with exponential backoff
// 3. On reconnect, request SessionInfoEvent to restore state
// 4. Show "Reconnected" toast
```

### Event Ordering

```typescript
// Handle out-of-order events:
// - Agent events before AgentCreatedEvent → queue in pendingEvents
// - Tool end before tool start → log warning, create synthetic start
// - Permission resolved before requested → ignore gracefully
```

### Partial Message Recovery

```typescript
// If connection drops mid-stream:
// 1. Mark current message as "interrupted"
// 2. On reconnect, request history to fill gaps
// 3. Show visual indicator for interrupted messages
```

---

## Performance Considerations

### Virtual Scrolling

Use `react-window` or `@tanstack/virtual` for output pane:
- Only render visible messages
- Variable height support for different message types
- Smooth scrolling with overscan

### Debounced Updates

```typescript
// Batch rapid AgentOutputEvent updates
// Use requestAnimationFrame for smooth rendering
// Debounce status bar updates (token counts)
```

### Code Highlighting

```typescript
// Lazy-load Shiki WASM
// Cache highlighted blocks
// Use web workers for large files
// Fallback to plain text if highlighting slow
```

### Memory Management

```typescript
// Limit stored messages (configurable, default 1000)
// Offload old messages to IndexedDB if needed
// Clear tool output after completion (keep summary)
```

---

## Implementation Phases

### Phase 1: MVP (Core Functionality)

**Goal:** Basic chat with streaming, permissions, and tool visibility

**Features:**
- [ ] WebSocket connection with reconnection
- [ ] Message input and streaming output
- [ ] Basic markdown rendering
- [ ] Permission modal (all response types)
- [ ] Tool call display (flat list)
- [ ] Token usage display
- [ ] Stop button
- [ ] Dark theme only

**Timeline:** 2-3 weeks

### Phase 2: Feature Parity

**Goal:** Match rich-client capabilities

**Features:**
- [ ] Plan panel (sticky + expanded)
- [ ] Multi-agent support (agent switcher)
- [ ] Clarification dialogs
- [ ] Code syntax highlighting
- [ ] Diff rendering in permissions
- [ ] Light/dark theme toggle
- [ ] Keyboard shortcuts
- [ ] File attachments
- [ ] Session management (reset)

**Timeline:** 2-3 weeks

### Phase 3: Enhanced UX

**Goal:** Polish and advanced features

**Features:**
- [ ] Virtual scrolling for performance
- [ ] Command completion
- [ ] Session persistence (save/load)
- [ ] Custom themes
- [ ] Settings panel
- [ ] Accessibility audit
- [ ] Mobile-responsive layout
- [ ] Search in conversation
- [ ] Export conversation

**Timeline:** 2-3 weeks

### Phase 4: Production Readiness

**Goal:** Deploy-ready with monitoring

**Features:**
- [ ] Error tracking (Sentry integration)
- [ ] Analytics (optional)
- [ ] Performance monitoring
- [ ] Documentation
- [ ] E2E tests
- [ ] Security audit
- [ ] Docker deployment

**Timeline:** 1-2 weeks

---

## File Structure

```
web-client/
├── public/
│   ├── index.html
│   └── favicon.svg
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root component
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── MainLayout.tsx
│   │   │   └── StatusBar.tsx
│   │   ├── output/
│   │   │   ├── OutputPane.tsx
│   │   │   ├── MessageBlock.tsx
│   │   │   ├── ToolCallBlock.tsx
│   │   │   ├── CodeBlock.tsx
│   │   │   └── DiffBlock.tsx
│   │   ├── input/
│   │   │   ├── InputArea.tsx
│   │   │   ├── PromptInput.tsx
│   │   │   └── AttachmentBar.tsx
│   │   ├── plan/
│   │   │   ├── PlanPanel.tsx
│   │   │   └── PlanStep.tsx
│   │   ├── tools/
│   │   │   ├── ToolTree.tsx
│   │   │   └── ToolNode.tsx
│   │   ├── agents/
│   │   │   ├── AgentList.tsx
│   │   │   └── AgentCard.tsx
│   │   ├── modals/
│   │   │   ├── PermissionModal.tsx
│   │   │   ├── ClarificationModal.tsx
│   │   │   └── SettingsModal.tsx
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Spinner.tsx
│   │       ├── Badge.tsx
│   │       └── Toast.tsx
│   ├── stores/
│   │   ├── connection.ts
│   │   ├── agents.ts
│   │   ├── tools.ts
│   │   ├── permissions.ts
│   │   ├── clarification.ts
│   │   ├── plan.ts
│   │   ├── context.ts
│   │   └── ui.ts
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useKeyboardShortcuts.ts
│   │   ├── useTheme.ts
│   │   └── useAutoScroll.ts
│   ├── lib/
│   │   ├── events.ts               # Event types & parsing
│   │   ├── protocol.ts             # Client→Server event builders
│   │   ├── highlighter.ts          # Shiki setup
│   │   └── markdown.ts             # Markdown config
│   ├── styles/
│   │   ├── globals.css
│   │   ├── themes.css
│   │   └── components/
│   └── types/
│       ├── events.ts               # Server event types
│       ├── stores.ts               # Store types
│       └── components.ts           # Component prop types
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── README.md
```

---

## API Reference

### WebSocket Connection

```typescript
// Connect
const ws = new WebSocket('ws://localhost:8080');

// On connect, receive:
{
  "type": "connected",
  "server_info": {
    "version": "1.0.0",
    "client_id": "client_1",
    "protocol_version": "1"
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}

// Send client config immediately after connect:
{
  "type": "client.config",
  "cwd": "/home/user/project",
  "env": {},
  "terminal_width": 120,
  "timestamp": "2024-01-15T10:30:00.001Z"
}
```

### Message Format

All messages are JSON with required fields:
- `type`: Event type string
- `timestamp`: ISO 8601 UTC timestamp

---

## Security Considerations

1. **Input Sanitization**: Sanitize all user inputs before sending
2. **XSS Prevention**: Use React's built-in escaping, sanitize markdown HTML
3. **WebSocket Security**: Use WSS in production, validate origin
4. **Content Security Policy**: Strict CSP headers
5. **Sensitive Data**: Never log or store API keys, tokens in browser

---

## Testing Strategy

### Unit Tests (Vitest)
- Store logic
- Event parsing
- Utility functions

### Component Tests (React Testing Library)
- Component rendering
- User interactions
- State updates

### Integration Tests (Playwright)
- WebSocket communication
- Full user flows
- Permission dialogs
- Error scenarios

### Visual Regression (Chromatic/Percy)
- Component snapshots
- Theme consistency
- Responsive layouts

---

## Open Questions

1. **Authentication**: Should the web client support authentication? Currently, WebSocket server has no auth.
2. **Multiple Tabs**: Should we prevent/warn about multiple tabs connecting to same server?
3. **Mobile Priority**: How important is mobile support for initial release?
4. **Offline Mode**: Is PWA/offline capability needed?
5. **File Uploads**: Maximum file size? Supported formats?

---

## References

- [Server Events Protocol](../server/events.py)
- [WebSocket Server](../server/websocket.py)
- [Rich Client Implementation](../rich-client/)
- [Architecture Overview](./architecture.md)
