# A Terminal in the Web Client — design

**Status**: proposed design (no code in this change)
**Date**: 2026-09-27
**Scope**: a person using the web coder opens an interactive shell in the
rail, started in the workspace of the session they are attached to, and
types into it. The shell is for the **person**, not the model: its output
never enters history and the agent does not see it.

The browser half is small: a terminal widget in a lazily loaded rail
section. Almost all of the design is about the daemon. It gives a person
a process on the daemon's host, and the questions that matter are which
user that process runs as, what bounds it, and who may open one.

---

## 1. What exists, and what does not

| Piece | State |
|---|---|
| a PTY the **model** drives | `interactive_shell` (`pexpect`, runner-side). It strips ANSI, reads until idle, and hands text back as a tool result. That is the right shape for a model and the wrong one for a person, who needs raw bytes both ways and a real terminal emulator |
| confinement for a child process | the session's `//child` sub-profile transition, handed to `cli`, `interactive_shell` and `notebook` as a `preexec_fn` (`ToolExecutor.set_apparmor_child_transition_callback`, #1323) |
| resource limits | the runner PROCESS is migrated into the session's cgroup at spawn, so its children inherit it (#735) |
| a subprocess environment | secret scrubbing (#863), the workspace HOME and managed venv (#1225, #1274), the jaato tool shims on `PATH` (#1273) |
| redaction of printed secrets | `RunnerRPC._write` redacts every event and response frame the runner sends (#1215) |
| an ownership rule | `memory_verbs.may_curate`: the workspace owner may, anyone may on an unowned workspace, an identity-less connection on an owned one may not |
| a human shell verb | **none**. No daemon code opens a PTY for a client |
| a terminal widget in the client | **none** |

Every row above the last two is something the new verb should reuse
rather than re-derive. Each one exists because a second copy of it drifted
somewhere else in this tree.

---

## 2. Decisions

### D1. The PTY lives in the session's runner, not in the daemon

The daemon is unconfined, sits in no session cgroup, and holds the
environment of every session it serves. A shell forked from it would need
the confinement, the cgroup, the scrub, the HOME/venv environment and the
redaction built a second time. Every one of those already applies to a
child of the runner, so the runner spawns the shell with the same
`preexec_fn` `cli` receives, and the daemon only relays bytes.

Consequences:

- **A terminal needs a loaded session.** The rail shows the section only
  while the client is attached to one. That matches the Files panel, which
  is also session-scoped.
- **A terminal dies with its runner.** Session unload, `session.stop`, the
  #812 watchdog and daemon shutdown all end it; nothing outlives the
  process that confined it.
- **A pool slot must not keep one.** `session.end` returns a warm slot to
  the pool, and a live shell there would be handed to the next session of
  the cascade (#890's lesson about state carried across the boundary). The
  `session.end` handler kills every terminal before it answers, and a
  failure to kill one fails the handler, which stops the daemon pooling
  that slot (the posture #1100 takes for a telemetry shutdown that raised).

### D2. It runs as the daemon's user, so a root daemon refuses

Nothing in the tree drops privileges (#1168). On a root daemon the shell
would be a root shell in a browser tab, and the `//child` profile does not
change who the process is. The first version therefore refuses to open a
terminal when the daemon's effective uid is 0, with a message naming the
cause and the remedy (run the daemon as a service user). There is no
override flag: an override would be the easiest thing in this design to
turn on and the hardest to justify. When privilege dropping exists, this
decision is revisited, not bypassed.

### D3. It is confined like `cli`, and requires an enforced boundary

The shell gets exactly what the agent's `cli` subprocess gets: the
`//child` transition, the cgroup, the scrubbed environment, and the
workspace as its working directory, verified at spawn the way
`ShellSession` verifies its `cwd` (#503).

It differs from `cli` in one place. `cli` can run on an unconfined host
because it has an application-layer check: every path in a command goes
through `check_path_with_jaato_containment`. A person typing into a PTY
has no such check. `interactive_shell` shows that analysing typed input
is best effort at most. On an unconfined host, a human shell would
therefore reach strictly more than the agent can. So the terminal
**requires an enforced AppArmor label** on the spawning thread, using the
predicate `interactive_shell.require_confinement` already uses
(`AppArmorLabel.enforced`, #1014). A complain-mode profile does not
count.

`--ws-terminal-unconfined` is the explicit opt-out for a host where the
person could log in anyway (a single-user dev box). It is logged at
WARNING once per daemon, like `--ws-unsafe-no-auth`.

### D4. Only the workspace owner may open one

The rule is `may_curate` applied to the session's workspace:

| Connection | Owned workspace | Unowned workspace |
|---|---|---|
| its owner (`app:user` from a bound ticket) | allowed | allowed |
| another identity | refused | allowed |
| no identity (shared bearer token) | refused | allowed |

"Allowed on an unowned workspace" is what a shared-bearer deployment has
always meant: everyone holding the token is the operator. The identity is
read from the transport, never from the request (#859). One predicate
serves both the check and the `may_open_terminal` flag the client reads
to decide whether to show the section, so the two cannot disagree.

### D5. Off by default, and WS only

`--ws-terminal` enables the verbs. Without it, every terminal request is
answered `disabled`, naming the flag, and the client hides the section.
IPC is not served: an IPC client is on the daemon's host and already has
a shell, so a verb there would add a path without adding a capability.

### D6. Output is redacted

The runner's frame writer redacts every value in the session's secret set
(#1215), so `cat .env` in the terminal prints `‹redacted:ANTHROPIC_API_KEY›`.
This is deliberate, not a side effect to be engineered around. The screen
is exactly where a key was leaked the first time, a browser tab is
screen-shared and recorded far more often than a local terminal, and a
second writer that skipped redaction would be the one path #1215 does not
cover. The cost: a person who genuinely needs to see a stored key does so
from a host shell, not from the browser.

### D7. Nothing typed is recorded; opening and closing are

The ledger gains a `terminal` event written at open and at close: `user`,
`session_id`, `workspace`, the boundary the kernel reported (`enforce`,
or `unconfined` under the opt-out), `duration_seconds`, `exit_status`,
and the close reason. Keystrokes and output are **not** recorded, and
`explain audit` says so in those words. A record of what a person typed
would be a keylogger with retention rules. The Article 12 obligation is
about what the AI system does, and this shell is not the AI system. The
event is added to `jaato_sdk.audit.AUDIT_SCHEMA`, so the guard that walks
every writer the schema names covers it.

---

## 3. Protocol (1.29)

Client → daemon:

| Request | Fields |
|---|---|
| `TerminalOpenRequest` | `request_id`, `cols`, `rows` |
| `TerminalInputRequest` | `terminal_id`, `data_b64` |
| `TerminalResizeRequest` | `terminal_id`, `cols`, `rows` |
| `TerminalAckRequest` | `terminal_id`, `bytes` (credit returned, see §4) |
| `TerminalCloseRequest` | `terminal_id` |

Daemon → client:

| Event | Fields |
|---|---|
| `TerminalOpenedEvent` | `request_id`, `ok`, `terminal_id`, `category`, `error`, `shell`, `boundary` |
| `TerminalOutputEvent` | `terminal_id`, `seq`, `data_b64` |
| `TerminalClosedEvent` | `terminal_id`, `reason`, `exit_status` |

`category` on a refused open is one of `disabled`, `not_owner`,
`root_daemon`, `unconfined`, `no_session`, `too_many`, `spawn_failed`.
`SessionInfoEvent` gains `may_open_terminal`, derived from the same
predicate as the refusal.

**Base64 in JSON, not binary frames.** The file download pairs a header
with one binary frame (1.20), which works because there is only one frame
per request. A terminal produces thousands of chunks, and a header-plus-
frame pair per chunk doubles the frame count for no saving that matters
at terminal rates. The runner coalesces output (up to 16 ms or 64 KiB per
event), so the overhead is one JSON envelope per batch. If a measured
workload shows otherwise, binary frames can be adopted inside this same
protocol version without changing the verbs.

**The runner side.** `terminal.open`, `terminal.input`, `terminal.resize`,
`terminal.ack` and `terminal.close` are **control-lane** RPCs, because
typing must not queue behind a running turn (the reason the permission
status read is control-lane). Output travels as a `terminal_output`
notification frame, relayed by the daemon's notification table to the one
client that opened the terminal, never broadcast to the session's other
attached clients.

**SDKs.** The TS SDK gets `openTerminal()`, returning a handle with
`write`, `resize`, `close` and an output callback, and refuses below 1.29
(an older daemon answers "Unknown message type" and never the open, the
1.7 missing-verb rule). The Python SDK does not get it, for the reason IPC
is not served.

---

## 4. Lifecycle and bounds

| Event | Effect |
|---|---|
| `TerminalCloseRequest` | SIGHUP to the shell's process group, SIGKILL after 2 s, `TerminalClosedEvent(reason="closed")` |
| the shell exits | `reason="exited"` with its status |
| the WS connection drops | the terminal is closed. A reconnect grace (like #1106's) is a later refinement; v1 does not keep a PTY for a client that may never come back |
| the client detaches or attaches to another session | closed. A terminal belongs to one (connection, session) pair |
| session unload / stop / `session.end` / runner shutdown | closed (D1) |
| no input and no output for `terminal_idle_seconds` (default 1800) | `reason="idle"` |
| a third open on one connection | refused `too_many` (two per connection) |

**Backpressure is credit-based.** The runner stops reading the PTY once
the bytes it has sent and the client has not yet acknowledged pass a
window (1 MiB). It resumes on `TerminalAckRequest`. When it stops reading,
the kernel's PTY buffer fills and the shell blocks on write. That is the
right behaviour for `yes` or `cat bigfile`: the terminal slows down
instead of the daemon queueing unboundedly. The alternative of dropping
output was rejected: a terminal stream with a hole in the middle
corrupts every escape sequence that follows it. Output is therefore not
subject to the IPC queue's lossy class either, and is sent as essential.

---

## 5. The client

- A **Terminal** section in the rail, shown when `may_open_terminal` is
  true, lazily loaded with its emulator like the log, image and code
  views. Opening it does not start a shell; a **Start shell** button does,
  so collapsing and expanding the rail never spawns one by accident.
- **The emulator**: ghostty-web or xterm.js. Both speak the same API
  shape, so the choice is made in the implementation PR by measuring the
  gzipped chunk size, input handling (IME, paste, mobile keyboards) and
  rendering in both themes, and is recorded there. Neither is chosen here
  without measurements.
- **Placement**: a shell wants about 80 columns (~600 px). The rail goes
  to 720 px, which is enough, but a **pop-out** to a bottom dock the width
  of the transcript is offered for real work. The terminal is fitted to
  its container and sends `TerminalResizeRequest` on every change.
- **Theme**: colours come from the theme tokens, so `theme dark` and the
  rest apply.
- The section header shows the boundary from `TerminalOpenedEvent`. An
  unconfined shell under the opt-out is marked in the warning tone, so a
  person can see what the daemon was told.

---

## 6. What this does not do

- **Give the agent the terminal.** The model has `interactive_shell`.
  Terminal output never enters history and no tool reads it.
- **Survive a reconnect.** See §4.
- **Serve a root daemon.** D2.
- **Record what was typed.** D7.
- **Hide a secret from a determined person.** Redaction matches exact
  values. `base64 < .env` prints an encoded key, which #1215 already
  states as a limit. The person here is the workspace owner, so this is
  about screens and recordings, not about the owner.
- **Share a terminal between two viewers.** Output goes only to the
  connection that opened it.

---

## 7. Rollout and tests

1. **Runner and daemon**: the RPCs, the `session.end` kill, the gates, the
   ledger event, protocol 1.29, and the TS SDK handle. Guards:
   - off by default (`disabled` without the flag);
   - refused for a non-owner, and for an identity-less connection on an
     owned workspace;
   - refused on euid 0 (the uid patched in both directions, as #1168's
     guard does);
   - refused without an enforced label, and allowed under the opt-out;
   - the child transition callback is passed to the spawn (an AST check
     of the one spawn site, since a behavioural test cannot see a
     `preexec_fn` in effect without a kernel);
   - output passes the redactor;
   - the credit window stops and resumes reading;
   - `session.end` leaves no process behind;
   - the input and output RPCs are classified control-lane.
2. **Client**: the section, the emulator chosen by measurement, the
   pop-out, and e2e against a mock daemon that speaks 1.29 in the daemon's
   shape: open, type, see output, resize, close, and a refused open
   rendering its category.

## 8. Open questions

- Should a **reconnect grace** keep the PTY for the #1106 window? It
  makes a tablet blink cheaper and a forgotten shell last longer.
- Should the terminal be offered on a **session-less** workspace (the
  workspace list screen)? That needs a confined process with no runner,
  which D1 does not provide.
- Should `terminal_idle_seconds` be a `runtime_limits` key, per session,
  or a daemon flag? It bounds a process in the session's cgroup, which
  argues for `runtime_limits`.
