# Web Client Design (`jaato-web/`)

## Overview

`jaato-web/` is the browser client for a jaato daemon — the web counterpart
of `jaato-tui`.  It attaches to a daemon started with `--web-socket` and
presents the same session a TUI can be attached to at the same time.  This
document records the stack decision, the architecture, and the one input
convention that a generic chat box gets wrong for jaato: **commands are
words, not `/verbs`**.

It replaces the earlier `web-client/` proof of concept, which hand-wrote a
partial mirror of the event protocol and never tracked it.  The PoC's screen
flow (connect → workspace → session) is kept; nothing else was carried over.

## What already existed, and what that decided

Three facts about the tree shaped every choice below.

1. **The TS SDK is browser-ready.**  `jaato-sdk-ts/` speaks the wire with the
   standard `WebSocket` (no `ws` dependency), authenticates with the
   browser-compatible `?token=` form, and its `events.ts` is **generated**
   from the pydantic models in `jaato-sdk/jaato_sdk/events.py` with a CI
   staleness gate.  A client that hand-writes event types — as the PoC did,
   covering roughly half the protocol — drifts by construction.  `jaato-web`
   consumes `@jaato/sdk` and has no event type definitions of its own.
2. **The server's output is already client-neutral.**  The formatter pipeline
   emits `<j-code language="…">` / `<j-line>` / `<j-tok t="<pygments class>">`
   and `<j-table>` markup plus markdown; every client renders it in its own
   idiom (`jaato-tui/j_markup_renderer.py` → ANSI, `jaato-web/src/protocol/
   jmarkup.ts` → DOM).  The web client therefore needs **no syntax
   highlighter**: it maps Pygments token classes to theme colours, exactly as
   the TUI feeds them through Rich's Pygments theme.  Code colouring follows
   the UI theme in both clients.
3. **The model can be told it is in a browser.**  `PresentationContext`
   already has `ClientType.WEB` and `supports_expandable_content`; the client
   declares `client_type: "web"`, expandable content, images and
   `renderable_media: ["image/*", "audio/*"]`, so the model need not shorten
   output for a narrow terminal and media the model cannot consume can still
   be delivered to the person.

## Stack

| Layer | Choice | Rationale | Alternatives weighed |
|---|---|---|---|
| Delivery | Static SPA talking WebSocket directly to the daemon | Simplest; matches `--web-socket` + bearer token. A thin BFF (token custody, SSO via `set_client_user()`) or a Tauri shell (local IPC socket, desktop file dialogs) can wrap the same bundle later — the SPA has no Node-only assumptions. | BFF-first; VS Code webview |
| UI framework | **React 19 + Vite 7** | Largest ecosystem for exactly this client's widgets: virtualised lists (TanStack Virtual, used), VS Code-style docking (dockview, for split panes later), xterm.js (for `interactive_shell` output later). React Compiler removes most memo work. | **Svelte 5** (runes) or **Solid** — fine-grained signals suit token-rate streaming better and ship smaller, but no docking binding and thinner kits. The framework-neutral layers (SDK, parsers, theme) make a later switch cheap. Lit was considered for embeddability; too manual for a whole app. |
| State | Zustand; one `reduce(state, event)` | The protocol is an event stream, so the store is an event-sourced reducer — testable by replaying recorded events with no DOM. Streaming chunks are batched per animation frame by the SDK adapter: one React commit per frame, not per token. | Jotai/atoms, XState for the connection machine (the SDK already exposes one) |
| Styling | Tailwind v4 + CSS custom properties | The TUI's six `themes/*.json` are **imported at build time**; their eleven base colours become CSS variables and every rule — Pygments roles included — is written in terms of them. One palette definition for both clients. | shadcn/Radix for later dialog/menu primitives |
| Rendering | Own `<j-*>` and markdown parsers → React elements, never `innerHTML` | Model output cannot inject markup. The markdown reader is deliberately small (paragraphs, headings, lists, quotes, rules, inline code/emphasis/links) because fences and tables are already lifted out by the server. | react-markdown (pulls a full pipeline for a subset the server already handled) |
| Media | Web Audio for headerless pcm16 (what a speaking model streams), `<audio>`/`<img>` for container formats | `ToolOutputEvent` chunks with `mime_type`/`data_b64`; model speech arrives under the reserved `call_id` `"model-output"` and its `final` chunk closes the block. | |
| Tests | Vitest for `protocol/` and `store/`; Playwright against a **scripted mock daemon** (`mock/daemon.ts`) that speaks the real wire protocol | The UI is exercised end to end without a model provider or a Python install, so the CI job is cheap and deterministic. | |

## Commands are words, not `/verbs` — the input model

jaato's built-in commands are typed as bare words: `model gpt-4o`,
`permissions status`, `tools enable cli`.  The TUI decides lexically at
submit time (`client_commands.parse_user_input`): a whole line equal to a
client command (`help`, `clear`, …), or a first word that is a known server
prefix or the base word of a daemon-advertised command, routes as a command;
everything else is a prompt.  While the first word is typed, the completer
proposes matching commands.  The only prefixed form is `/name`, and that is a
**user-authored** workspace command under `.jaato/commands/`, expanded
server-side, so it travels as message text.

`jaato-web` ports that rule verbatim (`src/protocol/commands.ts`, tested
against the TUI's cases) and adds one explicit affordance the terminal
expresses implicitly:

* **Proposal.**  `Tab`/`↑↓` pick, `Tab` accepts, `Enter` submits the line as
  typed.  Only the first two words participate (`tools enable ` stops
  proposing — the argument is the user's).  `@`, `%` and a leading `/`
  disable proposals, as in the TUI.
* **Verbatim.**  `Esc` on the proposal dismisses it **and** marks the line
  verbatim: it is sent as a message even though its first word names a
  command (`model this is broken` reaches the model as text).  Verbatim
  lasts until the first word changes or the box is cleared; `Tab` re-arms
  the proposal.  A pasted line that would route as a command gets the same
  `Esc` escape hatch even though no proposal was shown.
* **Hint.**  A line under the box always states what `Enter` will do —
  "runs command `model` · Esc to send as text" or "sending as text · Tab to
  make it a command" — so the routing is never a surprise.
* **Prompt capture.**  While a permission, clarification or reference
  selection is pending for the selected agent, `Enter` sends the typed text
  as that prompt's answer (`y`, `a`, an option number, a free-text reply),
  exactly like typing into the TUI while a prompt is up.

## Architecture

```
┌──────────────┐  JSON frames   ┌──────────────────┐  JaatoEvent[]  ┌──────────────┐
│ jaato daemon │ ─────────────▶ │ @jaato/sdk       │ ─────────────▶ │ sdk/connection│ rAF-batched
│ --web-socket │ ◀───────────── │ JaatoClient      │                │  enqueue/flush│ ─────┐
└──────────────┘  requests      └──────────────────┘                └──────────────┘      ▼
                                                                                 ┌──────────────────┐
   components subscribe to slices ◀──────────────────────────────────────────── │ store.reduce()   │
   (OutputPane, ToolBlockView, PermissionPrompt, PlanPanel, …)                  │ zustand          │
                                                                                 └──────────────────┘
   Composer ──▶ app/actions.submitInput ──▶ parseUserInput ──▶ client.sendMessage / executeCommand /
                                                              respondToPermission / respondToClarification…
```

* `src/protocol/` — pure, framework-free: `commands.ts` (routing +
  completion), `jmarkup.ts`, `markdown.ts`, `pygments.ts`.  Unit-tested.
* `src/store/` — `types.ts` mirrors what `output_buffer.py` keeps
  (`OutputLine` / `ToolBlock` / `ActiveToolCall`); `store.ts` folds every
  wire event.  `AgentOutputEvent(mode="append")` extends the last text block
  of the same source; tool events route by `call_id`; permission
  `requested` + `input_mode` upsert one record; the two clarification wire
  shapes (`batch_only` batch vs per-question) converge on one record with a
  `batchOnly` flag that selects the reply shape.
* `src/sdk/connection.ts` — one live `JaatoClient`; status → store; every
  event queued and flushed once per animation frame; the workspace verbs the
  SDK exposes only as raw events.  `probeWorkspaceMode()` sends
  `workspace.list` and settles on the first of `WorkspaceListEvent`
  (daemon started with `--workspace-root`) or the "Workspace mode not
  enabled" error (single-workspace daemon).
* `src/app/actions.ts` — what a submitted line does, in the TUI's order of
  precedence (pending prompt → client command → server command → message).
* `src/screens/` — `Connect` (URL + bearer token; the Vite dev server proxies
  `/ws` to a local daemon), `Workspace` (list/select/create + provider
  configuration when the daemon reports it incomplete), `Session` (agent
  tabs, output, prompts, composer, side panels, status bar).
* `src/theme/` — `themes.ts` imports `../jaato-tui/themes/*.json`;
  `theme.css` defines everything in terms of the eleven variables.

## Feature coverage against the TUI

| TUI | jaato-web | Notes |
|---|---|---|
| Streaming output, j-markup, markdown | ✅ | virtualised; auto-follow with a "follow output" pill when scrolled up |
| Tool blocks: collapse/expand, status, duration, error | ✅ | failed calls open by default; `show_output` honoured |
| Live tool-output popup (tail -f, tabs, continuation groups) | ✅ | `Ctrl+O` cycles running tools |
| Permission prompt with modes (`y n a t i once never all`), diff `prompt_lines`, warnings | ✅ | typed key, click, or Tab-focus + Enter |
| Clarification (batch and per-question), reference selection | ✅ | choice questions as buttons; `cancel` supported |
| Plan panel, budget panel, workspace changed-files tree | ✅ | side panels; `Ctrl+P` / `Ctrl+B` / `Alt+W` |
| Multi-agent tabs | ✅ | `Ctrl+A` cycles; split panes are a follow-up (dockview) |
| Bare-word commands with completion; `@`/`@@`/`%`/`/` pass-through | ✅ | see the input model above; `@path` completion needs a server-side listing and is a follow-up |
| Themes (6 JSON files) | ✅ | `theme <name>`; persisted in `localStorage` |
| Binary media (images, audio, pcm16 speech) | ✅ | Web Audio queue for pcm16 |
| Session save/resume/profiles | partial | profile picker on session creation; `save`/`resume`/`sessions` route to the daemon as commands |
| Client-side edit of tool args (`e`) | ✗ | follow-up: editor dialog fed by `editable_metadata` |
| `interactive_shell` terminal emulation (pyte) | ✗ | follow-up: xterm.js for `continuation_id` groups |
| Keybinding configuration, external editor, screenshots, headless mode | ✗ | terminal-specific; not planned |

## Development and verification

```bash
cd jaato-web
npm run typecheck   # builds ../jaato-sdk-ts declarations, then tsc (strict, noUncheckedIndexedAccess)
npm test            # vitest: protocol + store
npm run build       # production bundle
npm run e2e         # Playwright: starts mock/daemon.ts + Vite, drives the UI
npm run dev         # against python -m server --web-socket :8080 (proxied at /ws)
npm run mock-daemon # ws://127.0.0.1:8090; prompts: code, tool, permit, ask, fail, subagent
```

The CI job `web-client` in `.github/workflows/ci-tests.yml` runs all of the
above on Node 22.

## Follow-ups

1. Split panes / docking across agents (dockview) — the TUI's `split_pane` /
   `move_agent`.
2. xterm.js for interactive-shell continuation groups.
3. `@path` completion: needs a `workspace.files.list`-style verb or reuse of
   `WorkspaceFilesSnapshotEvent`.
4. Client-side tool-argument editing on permission prompts.
5. A thin BFF package for token custody and SSO, and a Tauri shell for the
   local (IPC) use case — both wrap this bundle unchanged.
6. History replay on attach (`HistoryEvent` → blocks) for reattaching to a
   running session.
