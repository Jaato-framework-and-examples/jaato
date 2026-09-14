# jaato-web

Browser client for a jaato daemon — the web counterpart of `jaato-tui`.
It connects to `python -m server --web-socket …` over the daemon's
WebSocket transport through the TypeScript SDK (`@jaato/sdk`, in
`../jaato-sdk-ts`), and renders the same session a TUI can be attached to.

Design notes, stack rationale and the input model live in
[`docs/web-client-design.md`](../docs/web-client-design.md).

## Stack

| Layer | Choice | Why |
|---|---|---|
| UI | React 19 + Vite 7 | largest widget ecosystem for the pieces a client like this needs (virtualised lists, docking, terminals) |
| State | Zustand, one event-sourced reducer | the protocol is a stream of typed events; the store folds them, components subscribe to slices |
| Protocol | `@jaato/sdk` (workspace `file:` dep, source-aliased) | codegen'd event types stay in lockstep with `events.py`; no hand-written event mirror |
| Styling | Tailwind v4 + CSS variables | the eleven base colours of the TUI's `themes/*.json` become CSS custom properties, so both clients share one palette definition |
| Rendering | own `<j-*>` + markdown parsers, no `innerHTML` | the server emits neutral markup (`<j-code>`, `<j-table>`, Pygments token classes); the client only maps classes to theme colours |
| Tests | Vitest (protocol, store) + Playwright (UI against a scripted mock daemon) | |

## Commands are words, not `/verbs`

jaato commands are typed as bare words: `model gpt-4o`, `tools enable cli`,
`permissions status`. The composer proposes matching commands while the
first word is typed; **Esc on the proposal means "I mean this word
verbatim"** and the line is sent to the model as text even though it starts
with a command word. A hint under the box always states what Enter will do.
`@path`, `@@path`, `%prompt` and `/name` (workspace commands under
`.jaato/commands/`) pass through untouched and disable command proposals,
as in the TUI. See `src/protocol/commands.ts` for the routing rules, which
are a port of `jaato-tui/client_commands.py`.

## Develop

```bash
# once: build the SDK declarations the type-check uses
npm --prefix ../jaato-sdk-ts install && npm --prefix ../jaato-sdk-ts run build

npm install
npm run dev              # http://localhost:5173, proxies /ws → ws://127.0.0.1:8080
```

Against a real daemon:

```bash
.venv/bin/python -m server --web-socket :8080 --workspace-root /srv/workspaces   # or without --workspace-root
cat ~/.jaato/ws.token                                                            # paste into the connect form
```

Against the scripted mock (no provider needed):

```bash
npm run mock-daemon      # ws://127.0.0.1:8090 — try: code, tool, permit, ask, fail, subagent
```

Connect the form to `ws://127.0.0.1:8090`. `MOCK_WORKSPACES=1` turns on the
workspace-first flow, `MOCK_TOKEN=x` enforces bearer auth.

## Verify

```bash
npm run typecheck        # builds the SDK, then tsc
npm test                 # vitest: protocol + store
npm run build            # production bundle in dist/
npm run e2e              # Playwright, starts the mock daemon and Vite itself
```

Set `PLAYWRIGHT_CHROMIUM=/path/to/chrome` to use a pre-installed browser
instead of Playwright's download.

## Layout

```
src/
  protocol/    commands.ts (routing + completion), jmarkup.ts, markdown.ts, pygments.ts — pure, unit-tested
  store/       types.ts, store.ts (reduce(state, event)) — event-sourced client state
  sdk/         connection.ts — JaatoClient lifecycle, frame-batched dispatch, workspace verbs
  app/         actions.ts — what a submitted line does (prompt answers, client/server commands, messages)
  components/  output/ (JMarkup, ToolBlockView, OutputPane, ToolOutputPopup, MediaView)
               input/Composer.tsx   prompts/ (Permission, Clarification, ReferenceSelection)
               panels/ (Plan, Budget, Workspace, AgentTabs)   layout/StatusBar.tsx
  screens/     ConnectScreen, WorkspaceScreen, SessionScreen
  theme/       themes.ts (imports ../jaato-tui/themes/*.json), theme.css
mock/daemon.ts scripted daemon speaking the wire protocol, for dev + e2e
e2e/           Playwright smoke suite
```

## Keys

`Enter` send · `Shift+Enter` newline · `Tab` complete / re-arm proposals ·
`Esc` dismiss proposal (send verbatim) · `Ctrl+P` plan · `Ctrl+B` budget ·
`Alt+W` files · `Ctrl+T` expand/collapse tools · `Ctrl+A` next agent ·
`Ctrl+O` next running tool in the popup · `Ctrl+C` (nothing selected) stop.
