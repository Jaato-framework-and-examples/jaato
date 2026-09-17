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
| Styling | Tailwind v4 + CSS variables | the eleven base colours of the TUI's `themes/*.json` become CSS custom properties, so both clients share one palette definition; the redesign's plates, chrome type and interface accent are one layer on top of them (see [Look](#look)) |
| Rendering | own `<j-*>` + markdown parsers, no `innerHTML` | the server emits neutral markup (`<j-code>`, `<j-table>`, Pygments token classes); the client only maps classes to theme colours |
| Tests | Vitest (protocol, store) + Playwright (UI against a scripted mock daemon) | |

## Look

The client is drawn on a blueprint (Claude Design, *Jaato Web UI
Redesign*, proposal 01c): square, hairline-bordered **plates** with
registration marks at their corners, **Barlow Condensed** for what the
interface says (kickers, buttons, tab names) and monospace reserved for
what the daemon said (ids, paths, models, arguments), and one **steel**
interface accent carrying the chrome while the theme's own colours shrink
to state glyphs.  The light face (`theme light`, the default) is the
design's paper ground; `theme dark` and the other four TUI themes draw the
same structure on their own ground, because the structure is written in
terms of the theme variables plus a derived `--c-steel`
(`src/theme/themes.ts`).  Fonts are self-hosted from `@fontsource`, so a
deployment behind a corporate proxy needs no font CDN.

The five screens the design draws: the connect plate (what jaato is on the
left, the one thing to do on the right, daemon settings behind a
disclosure, the build stamp at the foot); workspaces as a table;
new-session with resume and start side by side; the session with its
identity in a 46px header (brand, agent tabs, workspace / model / context),
tool calls as rows, user turns numbered in the gutter, one persistent rail
whose Plan / Budget / Files sections open and close (same toggles as
before: `Ctrl+P` / `Ctrl+B` / `Alt+W` and the status bar), and a 26px
status bar; and the permission request as a full-width warning plate with
the diff at full measure and one solid action.

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

## Files go into the workspace, not the prompt

Drop files on the composer, paste them, or press **Attach**, and they are
**staged into the session's workspace** on the daemon — the same
`StageFilesRequest` verb the premium `<jaato-task>` component uses (one
JSON frame naming the files, one binary frame each, one answer per file;
`docs/sdk-file-staging.md`). A strip above the box shows each file's state
(queued / staging / ✓ staged / ✗ with the daemon's reason), the Files
panel lists them as the daemon's monitor sees them, and the next message
ends with one line naming the paths so the model knows where they are:
`@path` then reads them. The strip's **into** field picks a folder for the
files that follow; a dropped directory keeps its structure.

The session picker has the same strip, for a session that should start
with files in place: with a workspace selected they are staged before
`session.new`; on a daemon that provisions the workspace as part of
`session.new` they wait for its `session.info` and are staged then, still
ahead of the first turn. The daemon caps a file at 10 MB and a request at
50 MB; the client applies the same caps before sending.

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

### Signing in from the prompt

Nothing has to be configured before the first session, exactly as in the
TUI. Connect, and the "New session" card offers three ways in: an agent
profile, the workspace defaults (`JAATO_PROVIDER` / `MODEL_NAME` from its
`.env`), or **sign in to a provider first** -- every daemon-level auth
command the daemon advertises (`anthropic-auth login`, `openrouter-auth
key …`, …) is a button, and any of them also runs from the prompt with no
session open. After a successful login the daemon sends its `auth.setup`
offer, which the page renders as one card: which model, whether to save
the provider and model to the workspace `.env`, open the session or not.
Accepting makes the daemon create the session itself.

A session that is already open resolved its credentials when its runner
booted. Storing a key afterwards (`zhipuai-auth key …`) refreshes that
session automatically when it runs the same provider; after editing a
workspace `.env` by hand, type `session reload_env` to do the same.

On a daemon with `--workspace-root`, the workspace list comes first; picking
a workspace goes straight to that card whether or not the workspace already
names a provider. The daemon's manual provider / model / API-key form is one
click away behind `configure` on each row, never a gate.

Served by `jaato-web-coder-server` with its key store enabled (`config.json`
names `credentialsUrl`), that form's API-key field is a combobox of the keys
you stored before for the selected provider — label and a masked hint, never
the secret — plus "New key…". The newest one is preselected, so a second
workspace on the same provider is one click; the key is revealed once when
you save and forwarded to the daemon exactly as a typed one is. A
`<provider>-auth key …` typed at the prompt is remembered the same way. The
daemon knows nothing of the store: it is the backend's, keyed by who signed
in (`src/app/credentials.ts`, `src/components/workspace/CredentialPicker.tsx`).

Two ways out, as buttons. The workspace list's header says who is signed in
and offers the backend's **Sign out** (which also revokes that user's
daemon tickets); without a backend it offers **Disconnect** instead. The
status bar's **exit** is the `exit` command: detach from the daemon and
return to the connect screen, leaving the session on the daemon for
`session attach` later. A page served with `autoConnect` does not connect
straight back after an exit; the next click does (`src/app/exitIntent.ts`).

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
(`.github/workflows/publish-npm-web-coder-ui.yml`) runs the same gates as CI,
refuses a version already on the registry, builds, and **stages** the version
with the `@jaato` org's stage-only token; a maintainer with 2FA promotes it
(`npm stage list @jaato/web-coder-ui`, then `npm stage approve <stage-id>`;
the 2FA step opens the browser for a passkey, or takes `--otp <code>`). It needs `@jaato/sdk` published first only at build time, from
the sibling checkout. Bump `version` in `package.json` first.

npm cannot stage a package it has never seen (`404` on the stage endpoint,
measured on this package's first run), so the **first** version was published
directly by a maintainer with 2FA from a checkout at the release commit:

```bash
cd jaato-web-coder-ui
npm --prefix ../jaato-sdk-ts ci && npm ci
npm run build                                # builds the SDK, type-checks, vite build
npm publish --access public     # after `npm login`; the 2FA step opens
                                #   the browser (passkey) or asks for a code
```

The workflow refuses by name while the package is unknown to the registry;
every later version stages.

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
  theme/       themes.ts (imports ../jaato-tui/themes/*.json, derives the steel accent), theme.css (tokens, plates, chrome type)
  components/layout/Plate.tsx  the redesign's unit of surface: a square hairline plate with registration marks
mock/daemon.ts scripted daemon speaking the wire protocol, for dev + e2e
e2e/           Playwright smoke suite
```

## Keys

`Enter` send · `Shift+Enter` newline · `Tab` complete / re-arm proposals ·
`Esc` dismiss proposal (send verbatim) · `Ctrl+P` plan · `Ctrl+B` budget ·
`Alt+W` files · `Ctrl+T` expand/collapse tools · `Ctrl+A` next agent ·
`Ctrl+O` next running tool in the popup · `Ctrl+C` (nothing selected) stop ·
drop / paste a file on the composer to stage it into the workspace.
