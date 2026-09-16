# jaato-web-coder-ui

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

## Run it

The client is published on npm as `@jaato/web-coder-ui`; the package is the built
bundle plus a dependency-free launcher, so `npx` fetches it in one go:

```bash
.venv/bin/python -m server --web-socket :8080 --daemon   # the daemon, on this machine
npx @jaato/web-coder-ui                                            # serves the client, opens the browser
```

The launcher serves `dist/` on `http://127.0.0.1:5180/`, hands the page the
daemon URL and — for a local daemon — the token from `~/.jaato/ws.token`, and
the page connects on its own. Against a daemon elsewhere:

```bash
npx @jaato/web-coder-ui --daemon ws://build-box:8080 --token-file ./ws.token
npx @jaato/web-coder-ui --daemon wss://jaato.example.org --no-token     # type the token in the form
```

| Flag | Default | Meaning |
|---|---|---|
| `--daemon URL` | `ws://127.0.0.1:8080` | the daemon's `--web-socket` address |
| `--token TOKEN` / `--token-file PATH` | `~/.jaato/ws.token` when the daemon is loopback and the file exists | bearer token handed to the page |
| `--no-token` | | hand out no token; the form asks for it (or leave it empty for `--ws-unsafe-no-auth`) |
| `--host HOST` / `--port PORT` | `127.0.0.1` / `5180` (`0` = any free port) | where the client is served |
| `--no-open` | | do not open a browser |
| `--root DIR` | the package's `dist/` | serve another build |

The token is only ever handed to the page on a loopback bind: with
`--host 0.0.0.0` the launcher serves the client to the network and the
person at the browser types the token. On loopback the `Host` header must
name the bound address, so a DNS-rebinding page cannot fetch it either.

### Hosting the bundle yourself

`dist/` is static (assets are referenced relatively, so it can sit under any
path) and connects to whatever daemon URL the form is given. Put it behind
nginx, a CDN or a file server, and optionally publish a `config.json` next to
`index.html` to pre-fill the form:

```json
{"daemon": "wss://jaato.example.org", "autoConnect": true}
```

`token` is also accepted there, but a token in a file every visitor can read
is only right when the file server is as private as the daemon. Every
publish run of the npm workflow attaches `jaato-web-coder-ui-dist-<version>.tar.gz`
(the contents of `dist/`) to the GitHub Release `web-coder-ui-v<version>` for
this use; `npm run build` produces the same directory from a checkout.

### Knowing what you deployed

The SDK is compiled into the bundle, so the build stamps what it speaks:
`dist/build-info.json` carries the UI version, the `@jaato/sdk` revision,
its protocol floor and the commit. The connect screen and the status bar
show the same line, and the launcher prints it:

```bash
npx @jaato/web-coder-ui --version
# 0.1.0
# @jaato/sdk 0.6.0 · protocol ≥ 1.0 · cb65e5e · built 2026-09-15T20:40:00.000Z
```

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
npm test                 # vitest (protocol, store) + node --test (launcher)
npm run build            # production bundle in dist/
npm start -- --no-open   # the launcher, serving that dist/
npm run e2e              # Playwright, starts the mock daemon and Vite itself
npm pack                 # the npm tarball (refuses without a dist/)
```

Publishing is manual: the *Publish @jaato/web-coder-ui to npm* workflow
(`.github/workflows/publish-npm-web.yml`) runs the same gates as CI, refuses a
version already on the registry, builds, and publishes with the `@jaato` org
token. Bump `version` in `package.json` first.

Set `PLAYWRIGHT_CHROMIUM=/path/to/chrome` to use a pre-installed browser
instead of Playwright's download.

## Layout

```
bin/         jaato-web-coder-ui.js — the launcher shipped as the package's `bin` (static server + config.json + browser open), tested with node --test
src/
  app/         actions.ts (what a submitted line does), launcherConfig.ts (the page's side of config.json)
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
