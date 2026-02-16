# Jaato Web Client

A web-based client for jaato that connects via WebSockets, providing an alternative to the TUI client (jaato-tui).

## Features

- Real-time streaming output with markdown rendering
- Multi-agent support with agent switcher
- Interactive permission prompts with keyboard shortcuts
- Plan/todo tracking with progress indicators
- Token usage monitoring
- Dark/light theme support
- Responsive design

## Development

### Prerequisites

- Node.js 18+
- npm or pnpm
- Jaato server running with WebSocket enabled

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Environment Variables

Create a `.env` file or set environment variables:

```bash
VITE_WS_URL=ws://localhost:8080  # WebSocket server URL
```

### Running with Server

1. Start the jaato server with WebSocket support:

```bash
.venv/bin/python -m server --web-socket :8080
```

2. Start the web client:

```bash
npm run dev
```

3. Open http://localhost:3000

## Architecture

See [docs/web-client-design.md](../docs/web-client-design.md) for detailed design documentation.

### Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Zustand** - State management
- **Tailwind CSS** - Styling
- **Vite** - Build tool

### Server-Side Rendering Pipeline

Syntax highlighting, diff rendering, and other output formatting are handled by the **server's output pipeline**, not the client. This ensures a consistent experience across all clients (web, TUI, future clients).

The client receives pre-styled content via `AgentOutputEvent` and renders it with appropriate CSS. This separation of concerns means:
- **Pipeline** (server): Produces structured, styled data
- **Client** (web): Chooses visual presentation (layout, colors, animations)

### Why This Stack?

#### React 18

- **Streaming fits React's model** - `AgentOutputEvent` with `mode: "append"` maps naturally to state updates and re-renders
- **Concurrent features** - `useTransition` and automatic batching help with high-frequency updates from streaming
- **Ecosystem maturity** - Well-tested libraries for markdown (`react-markdown`), virtual scrolling (`react-window`)

Vue 3 or Svelte would also work, but React has the deepest ecosystem for real-time updates and complex UI state.

#### Zustand over Redux/Context

- **Minimal boilerplate** - No actions, reducers, or providers. Just functions that update state
- **Subscriptions** - Components only re-render when their specific slice changes (critical for streaming)
- **Works outside React** - WebSocket handler can update stores directly without hooks
- **Tiny footprint** - 1.5kb bundle impact

Redux Toolkit would add ceremony we don't need. React Context causes cascading re-renders with frequent updates.

#### Tailwind CSS

- **CSS variables for theming** - `var(--color-primary)` makes dark/light themes trivial
- **No runtime cost** - Styles compiled at build time
- **Responsive utilities** - `md:hidden lg:flex` for sidebar behavior

CSS-in-JS (styled-components, Emotion) adds runtime overhead we don't need for a real-time app.

#### Vite over webpack/CRA

- **Fast HMR** - Sub-50ms hot reload during development
- **ESM-native** - No bundling during dev, instant server start
- **Simple config** - TypeScript, React, Tailwind work out of box

Create React App is deprecated. Webpack is slower and more complex to configure.

#### Native WebSocket + reconnecting-websocket

- **Our protocol is simple** - JSON messages, no rooms, no namespaces needed
- **Server compatibility** - Server already uses standard WebSocket (`websockets` library)
- **Less overhead** - Socket.io adds protocol complexity we don't use
- **Reconnection** - `reconnecting-websocket` (2kb) adds automatic reconnection with exponential backoff

#### Trade-offs Accepted

| Choice | Trade-off |
|--------|-----------|
| React | Larger bundle than Svelte/Preact |
| Zustand | Less structured than Redux (fine for our scale) |
| Tailwind | Class-heavy HTML (acceptable for productivity) |
| Server-side styling | Client depends on server for syntax/diff formatting |

The stack prioritizes **developer velocity** and **real-time performance** over minimal bundle size, which matches our use case of a developer tool running locally.

### Project Structure

```
web-client/
├── src/
│   ├── components/      # React components
│   │   ├── layout/      # Header, Sidebar, StatusBar
│   │   ├── output/      # Message rendering
│   │   ├── input/       # User input
│   │   ├── plan/        # Plan panel
│   │   └── modals/      # Permission, settings dialogs
│   ├── stores/          # Zustand state stores
│   ├── hooks/           # Custom React hooks
│   ├── lib/             # Utilities, event handling
│   ├── types/           # TypeScript types
│   └── styles/          # Global CSS
├── public/              # Static assets
└── package.json
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Send message | `Enter` |
| New line | `Shift+Enter` |
| Stop generation | `Escape` |
| Toggle plan | `Ctrl+P` |
| Toggle sidebar | `Ctrl+B` |
| Focus input | `/` |
| Permission: Yes/No/etc | `y`/`n`/`a`/`t` |

## Implementation Status

### Phase 1: MVP (Current)
- [x] WebSocket connection with reconnection
- [x] Message input and streaming output
- [x] Basic markdown rendering
- [x] Permission modal
- [x] Token usage display
- [ ] Render server-provided styled output (syntax highlighting, diffs)

### Phase 2: Feature Parity
- [ ] Multi-agent support (UI)
- [ ] Clarification dialogs
- [ ] File attachments
- [ ] Session management

### Phase 3: Enhanced UX
- [ ] Virtual scrolling
- [ ] Command completion
- [ ] Custom themes
- [ ] Search in conversation

## Contributing

1. Follow the existing code style
2. Add types for all new code
3. Test with the jaato server
4. Update documentation as needed
