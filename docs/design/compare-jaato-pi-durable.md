# Jaato vs. Pi Durable — what a durable agent harness has that jaato does not

## Summary

[Pi Durable](https://earendil.com/posts/pi-durable/) is an experimental
harness Earendil released alongside Pi 1.0 (October 2026): about 15,000
lines of TypeScript that run long-lived agents on a durable-execution model
borrowed from Temporal. *"Every step of a run is a task that stores a
checkpoint before it moves on"*, and *"if the process dies, a new process
opens the same storage, finds the unfinished tasks, and continues each one
from its last checkpoint."*

The question this document answers is whether that model exposes something
jaato lacks. The answer is **no**. Every property Pi Durable advertises is
either already in jaato, reachable by composing primitives jaato already
has, or deliberately not worth adopting given jaato's process model:

| Pi Durable property | jaato | Verdict |
|---|---|---|
| Crash recovery from a per-step checkpoint | interrupted-turn record + recovery on load (§3) | **even** |
| The run continues by itself after a crash | a cascade driver wakes the session (§4) | **even, by composition** |
| Per-tool replay semantics (`replay: "safe"`) | every interrupted call reported to the model | **not adopted** (§5) |
| Fork a conversation, including a live one | headless session seeded with history + live session state (§6) | **even** |
| Many clients on one conversation | attached-client set, state replay, paged history, any client may answer | **even** |
| Agents outliving the session that started them (pi-fabric residency) | the daemon's normal model | **jaato ahead** (§7) |
| Many concurrent conversations | one confined runner per session | **jaato ahead** (§7) |
| Background compaction | four GC strategies, between turns | **different trade-off** (§8) |
| Hot extension updates | daemon restart + revive | **not adopted** (§9) |
| Small, portable (Bun, Cloudflare Durable Objects) | a Linux daemon with an LSM-based security model | **different target** |

The two "not adopted" rows were considered and rejected; the reasoning is
kept below so the question does not have to be re-asked.

## Status & sources

Pi Durable is described from its announcement and two secondary write-ups
as of 2026-10-05; it is marked experimental and its API may change.
jaato claims are checked against `b4e6adc` (2026-10-05); file references
are to that tree.

- [Pi Durable announcement](https://earendil.com/posts/pi-durable/)
- [Hacker News discussion](https://zeli.app/story/49925969)
- [80aj.com write-up](https://www.80aj.com/2026/10/02/earendil-pi-durable-ai-agent/)
- [pi-fabric: Durable residency through Pi](https://cdn.jsdelivr.net/npm/pi-fabric@0.92.4/docs/residency-runtime.md)

## 1. What Pi Durable is

| | |
|---|---|
| Unit of durability | a **task**: model request, tool call, compaction, or an extension-defined task, each checkpointed before the run moves on |
| Storage | pluggable backends: memory, SQLite, JSONL. On SQLite only the working set (active transcripts, live tasks, pending submissions) is kept in memory |
| Process model | *"One process owns a storage at a time, and other clients attach to that process."* Conversations run concurrently in that process, each with its own task queue |
| Tool calls after a crash | a tool declared `replay: "safe"` reruns; any other tool's call is reported to the model as interrupted |
| Model requests after a crash | resent; the partial answer stays in the transcript, marked aborted |
| Idempotent submissions | a `requestId` returns the original submission to a client that retries after a crash |
| App state | **documents**: typed JSON committed in the same atomic write as the transcript, rewindable, with a per-document fork policy (`asOf`, `current`, `fresh`) |
| Forking | *"A conversation starts fresh or forks another one at any point in its transcript, and sees the parent's history up to that point without copying it."* |
| Compaction | background summarisation of older messages; a turn waits only when the next request would not fit otherwise |
| Extensions | named bundles of prompt sections, tools, hooks and tasks; installing under an existing name replaces the extension in place |
| Clients | any number per conversation; a snapshot, then deltas; any client may steer or queue a follow-up |
| Runtime targets | Node, Bun, Cloudflare Durable Objects (backends need no Node APIs) |

pi-fabric adds **durable residency**: a background host per Fabric root
keeps durable actors and one-shot durable agents running after the Pi
session that started them closes, with mesh state and a residency directory
for reconnecting and routing control.

## 2. Different durability units, same outcome

Pi Durable checkpoints **steps**; jaato checkpoints **sessions**. A jaato
session record (`<ws>/.jaato/sessions/<id>.json`) holds the history, the
resolved profile, the rendered prompt, budget state and every
`TRAIT_SESSION_PERSISTENT` plugin's state, sealed by the daemon (#1529).
It is written on unload, at turn end, at daemon shutdown, and, which is
what matters for a crash, in the background at the start of every tool call.

The second difference is ownership. In Pi Durable the harness process is
both the scheduler and the executor. In jaato the daemon owns the session
and a separate runner process executes it, so "the process died" splits
into two cases with different answers: a runner dying (#851, surfaced as a
terminal error on that session) and the daemon dying (§3).

## 3. Crash recovery

Pi Durable's recovery is *find the unfinished task, continue from its
checkpoint*. jaato's is *find the unfinished turn, tell the model what was
cut off*:

1. **During the turn.** `SessionManager._handle_turn_tracking_event`
   (`server/session_manager.py`) keeps `session.interrupted_turn` current:
   the agent, the user prompt, when the turn started, and the tool calls
   still pending. Each `ToolCallStartEvent` adds a call and triggers
   `_save_session_async`, so the record on disk holds the history up to
   that point plus the list of calls in flight. A call's end removes it;
   the turn's end clears the record.
2. **On the next load** (`_load_session_impl` → `_recover_interrupted_turn`),
   each pending call is answered with a synthetic error result:
   `tool_interrupted` / `server_restart`, *"was interrupted by server
   restart. You may retry this operation if appropriate."* This works for
   the main agent (through the runner's `session.append_history_message`)
   and for restored subagents.
3. **Clients are told**: an `InterruptedTurnRecoveredEvent`, a
   warning-style `SystemMessageEvent` naming the tools, and a `done` status
   so a waiting spinner stops.

This is Pi Durable's handling of a tool that is not replay-safe, applied to
every tool. What it does not do by itself is continue the run; that is §4.

Two smaller differences, neither a defect:

- **A model request in flight is not resent.** The history survives up to
  the last save; the partial answer of the interrupted request does not.
  The woken turn (§4) sends a fresh request over that history, which is
  what a resend would have done.
- **The tool-start save runs in the background**, so a crash in the
  moments after a call starts can leave the record one step behind. The
  model then sees no record of that call and does not get the interrupted
  error for it. For a side-effecting call this is the same uncertainty a
  `may_have_run` verdict expresses on the runner channel (#856), and the
  answer is the same: the model checks before redoing.

## 4. Continuing after a crash is the cascade driver's job

Pi Durable's harness continues a recovered run on its own. jaato closes the
recovered turn (`done`) and lets whoever drives the session decide. For a
cascade, that is the driver, and every piece it needs is there:

| The driver needs | jaato provides |
|---|---|
| to survive the daemon dying | it is a separate client process; `IPCRecoveryClient` reconnects once the daemon is back |
| to trigger recovery | `session.wake` revives a cold session, and the revive runs `_recover_interrupted_turn` before the woken turn starts, so the model's first view of the history includes the `tool_interrupted` results |
| not to know whether a crash happened | waking unconditionally is correct: an uninterrupted session simply gets the next turn |
| to retry safely | `session.wake` and `session.message` drop a duplicate `event_id`, and the durable inbox keeps that check across a daemon restart |

Two details a driver author should know:

- A wake's text is wrapped as untrusted content (`wrap_untrusted_content`),
  so "continue" reads to the model as a request to weigh, not an
  instruction. For a resume that is the right weight.
- A wake carrying a `cascade_driver_id` on a session revived with no client
  attached is **deferred** (`SessionWokenEvent` to the cascade's
  observers) until a client attaches. A driver that is itself the observer
  attaches and the turn runs.

Putting the policy in the driver rather than the daemon is deliberate. The
interrupted-turn record knows the turn was cut off, not whether continuing
it is still wanted; the cascade knows. This is the same choice jaato makes
for agent continuity ([Agent Continuity Pattern](agent-continuity.md)):
compose primitives, do not grow a daemon policy.

## 5. Per-tool replay: considered, not adopted

The one Pi Durable mechanism with no jaato counterpart is the per-tool
`replay` declaration. Its jaato form would be a trait on `ToolSchema`
(`TRAIT_REPLAY_SAFE`, beside `TRAIT_FILE_WRITER`), and a branch in
`_recover_interrupted_turn` that re-runs pending calls to such tools through
the runner's executor and writes their real results, keeping the
`tool_interrupted` error for the rest.

Its value is one model round trip, and only on the crash path: when the
daemon dies while read-only calls (`readFile`, `glob_files`) are still
running, the model would get their results instead of re-asking for them.
Against that:

- **Most tools cannot honestly claim it.** `cli`, `interactive_shell`,
  `notebook_execute`, the file writers, `call_service`: the framework
  cannot know whether a command changes anything. The tools that qualify
  are the plain reads, which are exactly the ones the model re-asks for
  cheaply.
- **It is a flag every tool author has to decide and keep accurate**, for a
  path that should be rare. A wrong `safe` is worse than no flag: it
  silently re-executes a side effect.
- **Today's behaviour is already correct.** Every interrupted call is
  reported as interrupted and the model decides what to redo, which is
  what it must do for the calls that matter anyway.

Not adopted. Revisit only if crash recovery becomes a frequent path, which
would itself be the defect to fix.

## 6. Forking, including a running session

Pi Durable forks at any transcript point without copying the parent's
history. jaato's primitive is
`SessionManager.create_headless_session(initial_history=...,
initial_session_state=...)`:

- **History** is seeded into the new runner with
  `JaatoSession.set_initial_history` (over the runner RPC) after
  `server.initialize()` and before any prompt. Only the *new* session must
  be idle and empty; the source may be mid-turn.
- **Live state** comes from `source.get_all_session_state()`, which calls
  each `register_session_state_provider` callback, so state a plugin grows
  incrementally (premium's pseudonymization table) is carried as it is at
  fork time, not as it was last pushed. The same providers feed the
  journal save and waypoint snapshots.
- **Which point**: any prefix of the source's history the caller passes.
  Waypoints already capture `session_state_snapshot` at creation for this
  use.

The in-tree consumers are premium's handoff (`fork_session_from_history`)
and test harnesses; a fork-from-waypoint verb is on the backlog
(`project_backlog_waypoint_fork_to_session.md`, named in
`plugins/waypoint/models.py`). Waypoint *restore* is a different operation:
it restores files within one session and leaves the conversation alone.

The one implementation difference, copy versus reference, matters only for
storage size with very long transcripts. jaato's records are one JSON file
per session, so a copy is the natural representation, and a forked session
is then independent of its parent's lifetime, GC and retention.

## 7. Where jaato is ahead

**Residency.** pi-fabric's durable residency (actors and agents outliving
the Pi session that started them) is the jaato daemon's normal model, with
more machinery around it:

| | |
|---|---|
| sessions outlive their clients | always; a client is an audience, not an owner |
| bounding sessions nobody watches | `max_orphan_seconds` (default 900 s), `max_session_seconds`, enforced daemon-side (#812) |
| a reconnect costs nothing | `unload_grace_seconds` (default 60 s, #1106) |
| reaching a session that unloaded | `session.wake`, cold revive |
| messages that must not be lost | the durable per-session inbox, drained at turn end, on attach and by the watchdog |
| sessions talking to each other | the `courier` plugin: any-to-any within a group, waking a cold peer, with files |

**Isolation.** Pi Durable runs every conversation in one process. jaato runs
each session in its own runner process (pre-warmed pool), confined by
AppArmor or SELinux, a seccomp deny-list, dropped capabilities, cgroups and,
on a root daemon, a uid drop. A model-driven command in one session cannot
reach another session's memory or credentials.

**Multiple clients** are at parity: jaato keeps an attached-client set per
session, replays state on attach (`emit_current_state`), pages history
(protocol 1.28), re-sends pending permission and clarification prompts to a
client that attaches later, and lets any attached client answer. It sends
typed events rather than the exact operations of each commit, which covers
the same ground for its clients.

## 8. Compaction: a different trade-off

Pi Durable compacts in the background and makes a turn wait only when the
next request would not fit. jaato's GC runs on turn boundaries and before a
send, with richer policy: four strategies (truncate, summarize, hybrid,
budget), a second denominator for media bytes (#850), calibration of its
estimate against the provider's reported prompt size (#1440), space
reserved for output (#1444), and eviction of consumed audio.

Running collection alongside a turn would buy latency on the turn that
crosses the threshold, at the cost of a second writer to history while a
turn is reading it; the history invariant (#674) is enforced at one seam
precisely because several subsystems already edit history independently.
Not pursued.

## 9. Hot extension updates: considered, not adopted

In Pi Durable, installing an extension under an existing name replaces it
in the running process; calls already running finish on the old code, the
next ones use the new code, and conversations store extension names, so a
restart also picks up the latest code.

In jaato plugin code is fixed per runner process, at more than one level:

- a session's runner discovers and imports its plugins at bootstrap;
- pool slots are forked from a template process that imported every runner
  plugin when the daemon started, so even a new session can start on old
  code (refreshing the template without a restart is listed as not done in
  the pool documentation);
- a revived session restores the prompt it was rendered with (#787) unless
  the daemon runs with `JAATO_REVIVE_PERSONA=disk`.

So the update mechanism is a **daemon restart**. Session records store
plugin and profile names, not code, so sessions revive on the new code, and
a turn the restart interrupted comes back through §3 and §4. Narrower live
reloads already exist where they matter: `session.reload_env` for
credentials, the references catalog refresh (#1145), the `.lsp.json`
re-scan (#1345).

What Pi Durable's version adds is a restart without the interruption. For
jaato that is not worth its cost: code updates are a deployment act, not a
mid-conversation one; and hot-swappable plugins would work against the
pre-warmed template's imports, slot reuse keyed on the plugin set (#890,
#1033), and AppArmor profiles rendered from the plugins in use.

## 10. Different targets

Pi Durable is small enough for an agent to read and runs wherever
JavaScript runs, including Cloudflare Durable Objects. jaato is a Linux
daemon whose security model rests on kernel features (LSM confinement,
seccomp, cgroups, uid drop) that do not exist in those environments. Neither
property transfers, and neither is a gap for the other's use case.
