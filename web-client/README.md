# Jaato Web Client

A web-based client for jaato that connects via WebSockets, providing an alternative to the TUI rich-client.

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
- **Shiki** - Syntax highlighting (planned)

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
- [ ] Code syntax highlighting

### Phase 2: Feature Parity
- [ ] Multi-agent support (UI)
- [ ] Clarification dialogs
- [ ] Diff rendering in permissions
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
