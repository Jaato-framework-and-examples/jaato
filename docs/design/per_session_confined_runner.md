# Per-Session Confined Runner

**Status:** Phase 1 design. Not implemented. Awaiting review.

**Branch:** `claude/design-confined-runner-tnkc3`

**Closes:** `project_backlog_apparmor_child_subprofile`. The runner-subprocess
boundary IS the child sub-profile boundary — the original "child sub-profile
for cli / interactive_shell subprocesses" backlog is subsumed because every
tool subprocess now inherits the runner's per-session profile rather than the
daemon's broad profile.

## 1. Problem statement

The daemon's AppArmor profile is pinned to a single workspace at startup
(today: whichever session attaches first via the IPC AppArmor hook in
`server/__main__.py:_register_ipc_apparmor_hook`). Once the profile is loaded,
the daemon's kernel-side allow list is fixed. A second client connecting from
a different workspace can't even complete the IPC handshake — the daemon's
discovery sites (`.jaato/profiles/`, `.jaato/agents/`, `.jaato/prompts/`,
`.jaato/services/<name>/`, `.jaato/instructions/`, `.jaato/references/`,
`.jaato/reactors.json`, etc.) are all denied by the first workspace's profile.

The 7:3 reproducer (`Permission denied: kb-enablement-2.0/.jaato/profiles`)
is this failure: the daemon was confined under workspace-A's profile and tried
to read workspace-B's `.jaato/profiles/` during the second client's
`session.new` request. Kernel-level deny → request fails → second workspace
locked out.

The deny is structurally correct: workspace-B's `.jaato/` is not in
workspace-A's profile, and we *want* it that way for cross-workspace
isolation. The bug is that the daemon — which serves N concurrent workspaces
— is the thing being confined to one workspace's profile.

## 2. Why not the read-everywhere alternative

The discarded alternative: broaden the daemon's profile to allow read of
every workspace's `.jaato/`, then enforce per-session isolation in-process via
the existing `sandbox_utils.check_path_with_jaato_containment` denylist plus a
session-scoped write filter. Ships in days.

It leaves a kernel-level read-side hole. A buggy plugin in workspace-X's
session — or model-driven `cli` / `interactive_shell` subprocesses, which
inherit the daemon's profile via fork+exec — can read workspace-Y's config
because the kernel allows it. The in-process denylist only protects in-process
tools; subprocess tools are out of its reach. We have ample existing evidence
that kernel-level confinement matters in this codebase: the entire
`tool_hat` sub-profile (template v13) was added precisely because the
in-process denylist is not sufficient for tools.

The confined-runner shape closes both holes: tool subprocesses inherit the
runner's per-session profile, kernel-enforced; and the daemon — relieved of
running tools — needs only a much narrower profile (or no profile at all).
We do not reintroduce the read-everywhere fallback as "phase 1" — it is a
different design that defers the actual problem.

## 3. Architectural target

```
┌─────────────────────────────────────────────────────────┐
│  jaato daemon (one process, unconfined or narrow rules) │
│                                                         │
│  • IPC + WS transports                                  │
│  • SessionManager, JaatoServer per session              │
│  • Daemon-tier plugins: model_provider/*, references,   │
│    memory, telemetry, cache_*, anthropic_auth, etc.     │
│  • Permission rules + UI relay                          │
│  • EventBus + reactor framework hooks                   │
│  • RunnerSpawner: forks one runner per session          │
└──────────────┬──────────────────────────────────────────┘
               │ Unix socketpair, length-prefixed JSON
               │ (frame format mirrors server/ipc.py)
               ▼
┌─────────────────────────────────────────────────────────┐
│  jaato runner subprocess (one per session)              │
│                                                         │
│  • aa_change_profile("jaato-ws-{session_id}") at start  │
│  • Refuses to proceed if confinement fails              │
│  • Runner-tier plugins: cli, file_edit,                 │
│    interactive_shell, filesystem_query, todo, lsp,      │
│    template, service_connector, mcp, webhook            │
│  • Local ToolExecutor; daemon-tier calls go back over   │
│    RPC (permission, memory writes, …)                   │
│  • Tool subprocesses inherit per-session profile        │
└─────────────────────────────────────────────────────────┘
```

One process per session. One AppArmor profile per session. Cross-workspace
isolation is enforced by the kernel because the runner that touches workspace
files is itself confined to that workspace's profile.

## 4. Load-bearing decisions

### 4.1 Tool dispatch protocol

**Decision:** dedicated Unix `socketpair(AF_UNIX, SOCK_STREAM)` per runner,
length-prefixed JSON frames, single multiplexed channel per socket carrying
both directions' calls and call-progress events.

- Length prefix: 4-byte big-endian, max frame 10 MB. Same shape as
  `server/ipc.py:HEADER_SIZE / MAX_MESSAGE_SIZE`. We deliberately reuse the
  existing framing so the helpers (`_read_frame`, `_write_frame`) can be
  shared between IPC and runner-RPC; we don't end up with two slightly
  different framing dialects.
- Multiplexed: every frame carries `{"id": <int>, "kind": "request" | "response"
  | "stream", "payload": {...}}`. Bidirectional — daemon→runner is mostly
  `tool.execute`, but runner→daemon is `permission.check`, `event.emit`,
  `memory.write`, `references.add_fragment`, etc. Single socket avoids
  fan-out logic and the connect/teardown overhead of N sockets per session.
- Streaming: tools yield progress chunks via `on_output` callbacks today
  (e.g. `cli`'s line-buffered stdout, `web_fetch`'s download progress). The
  runner emits `kind: "stream"` frames carrying `{request_id, chunk, source,
  mode}` for the in-flight `tool.execute` call; the daemon's RPC client
  forwards them to the session's `on_output` callback. The terminating
  `kind: "response"` frame carries the final `Tuple[bool, result]`.
- No HTTP, no gRPC, no length-prefixed protobuf. JSON is sufficient (we
  already serialize tool args + results as JSON for IPC), debuggable from
  `strace`, and works without compiling a schema. The cost is encode/decode
  throughput; runner-RPC is local-loopback, dominated by tool wall time, not
  framing overhead.
- Socket pair, not abstract socket or named pipe: a `socketpair()` is created
  by the daemon before fork, the runner inherits it as fd 3 by convention.
  No filesystem path → no race with another local user creating a same-named
  socket, no need to clean up on crash.

**Rejected alternative — separate sockets per concern (tool, event, permission):**
debugging "which channel deadlocked the session" gets harder as the channel
count grows; multiplexing onto one channel with a request-id keeps the wire
trace linear.

#### 4.1.1 Streaming and cancellation

The protocol distinguishes three result-delivery shapes:

- **Single-shot (most tools).** `tool.execute` returns one terminating
  `kind: "response"` envelope with the final `(ok, result)`. No interim
  frames.
- **Streamed display, single-shot data (cli, web_fetch, MCP progress).**
  The tool emits `on_output` chunks while running. Each chunk lands as
  `{kind: "stream", id: <call_id>, channel: "display", source, text, mode}`.
  These are forwarded to the session's `on_output` callback for user
  display **only** — they are not part of the structured result.
  The terminating `kind: "response"` envelope carries the final
  `(ok, result)`. Plugin authors must not smuggle structured data through
  the chunk channel; if the model needs it, it goes in `result`.

  MCP `notifications/progress` from a long-running `tools/call` map to
  the same `kind: "stream"` shape — the `mcp` plugin's progress handler
  forwards them through `on_output`, which the runner already turns
  into stream frames.
- **Deferred result (auto-backgrounded tools).** When a tool exceeds the
  `BackgroundCapable` plugin's threshold, the runner sends a non-final
  `kind: "response"` envelope carrying `{ok: true, result: {kind:
  "background_handle", task_id, ...}}` and the model gets the handle on
  this turn. The runner continues executing in its background thread
  pool. When the task completes, the runner emits `{kind: "event", event:
  TaskCompletedEvent(task_id, ok, result, …)}` on the daemon's event
  channel. The daemon's `background` plugin (daemon-tier in §4.2)
  consumes the event, updates task status, and re-injects the completion
  into the session's history via the existing `_task_done_callback`
  pipeline. The original `tool.execute` RPC is fully closed once the
  handle is returned — there is no half-open call sitting on the socket
  for hours.

  Symmetric for `interactive_shell`: each `shell_*` tool call is its own
  bounded RPC. The session-id-keyed PTY state lives in the runner; long
  reads use the chunk channel, never a half-open response.

**Cancellation across the boundary.** Today's `CancelToken` is set on
thread-local in the runner's `ToolExecutor.execute`. With RPC, the
daemon's stop path (`session.request_stop` → `client.stop()`) sends a
`{kind: "cancel", id: <call_id>}` frame for every in-flight call. The
runner's RPC dispatcher trips the corresponding `CancelToken`; the tool
detects it via `get_current_cancel_token()` (already the supported
plugin contract) and either returns early or raises `CancelledException`.
Whichever it does, the runner emits a terminating envelope with `ok:
false, error.type: "CancelledException"`. The daemon translates that to
`FinishReason.CANCELLED` exactly as today.

Cancellation while a tool is in the **chunk-streaming** shape: the cancel
frame is processed in-band on the same socket; chunks already in flight
land at the daemon and are forwarded to `on_output` (the user sees them);
the terminating envelope arrives with the cancelled error. Cancellation
of an already-**deferred** task: daemon sends `{kind: "request",
method: "background.cancel", task_id}`; the runner cancels the task in
its pool and emits `TaskCompletedEvent(ok: false, error: cancelled)`.

### 4.2 Plugin tier classification

**Criteria, in priority order:**

1. **Hard daemon constraints — the plugin or one of its operations
   *cannot* run in a confined runner.** Two categories qualify:
   - **Privileged operations.** The current per-session profile uses
     `/usr/bin/** ix,` (inherit-exec). Under `ix`, AppArmor strips
     setuid on the exec'd child, so the runner cannot effectively
     elevate via `sudo` even though the daemon's sudoers rule covers
     the same UID. Anything that needs root (`sudo apparmor_parser
     -r`, ulimit-style tunables, etc.) has to run in the daemon.
   - **Independent trust boundary.** Even when `Ux` could be added to
     allow runner-side sudo, the existing sudoers rule covers
     `apparmor_parser` with arbitrary args — a model-driven runner
     could load arbitrary profiles or unload sibling sessions'
     profiles. Validation belongs in a process the model doesn't
     drive directly.
   Plus the obvious daemon-bound state: provider OAuth tokens that
   span sessions, the EventBus the reactor framework subscribes to,
   the daemon's IPC/WS transports.
2. **Cross-session in-memory sharing as an *optimization*.** Embedding
   models, prompt caches, semantic indexes — loading per-runner is
   wasteful but not incorrect. Daemon placement here is a soft call
   driven by RAM cost; "move to runner" is always available if the
   topology cost outweighs the optimization.
3. **Everything else → runner.** Per-session FS state in the workspace,
   subprocess spawn that should inherit AppArmor, anything per-session
   that doesn't trip criterion 1 or 2.

**Tiers:**
- **Daemon-tier** — criterion 1 or 2 applies; never instantiated in
  the runner.
- **Runner-tier** — criterion 3; never instantiated in the daemon.
- **Straddling** — the plugin's *invocation* is runner-tier but one
  specific operation is hard-daemon (criterion 1). The split is
  bounded to that single RPC, not the whole plugin.

| Plugin                | Tier      | Rationale |
|-----------------------|-----------|-----------|
| `model_provider/*`    | Daemon    | Provider client / OAuth tokens / API keys live in daemon. Tool calling is not FS work; the runner asks the daemon "send this turn to the model" via RPC, and the daemon streams responses back. Avoids re-loading tokens into N runners and re-doing OAuth on each spawn. |
| `cli`                 | Runner    | `subprocess.Popen` for shell commands. Kernel-confined inheritance is the whole point. |
| `interactive_shell`   | Runner    | `pexpect` PTY sessions. Same reason. Sessions live as long as the runner; runner death tears them down via the existing reaper. |
| `file_edit`           | Runner    | Reads/writes workspace files. |
| `filesystem_query`    | Runner    | Reads workspace files. |
| `todo`                | Runner    | Per-session JSON file under `.jaato/todos/`. |
| `lsp`                 | Runner    | Spawns LSP server subprocess (rust-analyzer, pyright, etc.). |
| `template`            | Runner    | Reads workspace template files. |
| `service_connector`   | Runner    | Reads `.jaato/services/<name>/` config + makes outbound HTTP. The HTTP doesn't need confinement, but the config read does. |
| `mcp`                 | Runner    | Each MCP server is a long-lived stdio subprocess; same lifecycle as `interactive_shell`. |
| `webhook`             | Runner    | HTTP listener for inbound webhooks; per-session subscription buffers. |
| `notebook`            | Runner    | Spawns Python interpreter subprocess. |
| `ast_search`          | Runner    | Reads workspace source files. |
| `web_fetch`           | Runner    | Outbound HTTP — could go either way, but today its result enrichment (artifact_tracker) is runner-local and it benefits from session-scoped runtime limits. |
| `web_search`          | Runner    | Outbound HTTP, but stateless; same logic as `web_fetch`. |
| `bundle`              | Runner    | Bundles workspace files. |
| `subagent`            | Runner    | Spawns subagent sessions (see §4.3). |
| `multimodal`          | Runner    | Reads workspace images. |
| `vision_capture`      | Runner    | Writes screenshots to workspace temp dir. |
| `waypoint`            | Runner    | Reads/writes `.jaato/waypoints.json`. |
| `sandbox_manager`     | Runner    | Per-session allowlist; needs to be where the tools run. |
| `clarification`       | Runner    | Inline UX, but tool-local. |
| `environment`         | Runner    | Reads env, writes scratch files. |
| `prompt_library`      | Runner    | Reads `~/.claude/skills/` and `.jaato/prompts/`. |
| `references`          | **Straddle** | Whole plugin runs in the runner (catalog read from `~/.jaato/references/` and `.jaato/references/`, `selectReferences` tool, embedding/semantic match). Single daemon-only RPC: `apparmor.add_fragment` writes to `/etc/apparmor.d/jaato/<sid>.refs.d/` and runs `sudo apparmor_parser -r`. Daemon-side because criterion 1 applies — `ix` strips setuid so the confined runner can't effectively call sudo, and validation of fragment paths (`_validate_path_for_fragment`) belongs outside model-driven code. |
| `memory`              | Runner    | `~/.jaato/memories/` is rw under every session's profile (template line 334), so the runner writes `memories/raw/<id>.json` and `curated.jsonl` directly via tempfile-rename — same concurrency story as today. Embedding-cache sharing (criterion 2) is a soft argument for daemon, but per-runner load cost is acceptable for the simpler topology. Revisit if measured RAM cost is a problem. |
| `permission`          | **Straddle** | Rules/state in daemon (whitelist, blacklist, evaluators, suspension, channels) — daemon owns the UI relay path and the rules outlive any single runner. The runner's tool executor delegates `check_permission` to the daemon via RPC (see §4.5). |
| `reliability`         | Runner    | Per-session failure tracking. Daemon placement was a soft call for cross-session adaptive trust; in practice reliability state is most useful within a single agent session. Move to daemon later if cross-session trust becomes a real feature. |
| `artifact_tracker`    | Runner    | Enriches file-writer tool results in-process. Runs alongside the tools it observes; result enrichment happens before the result crosses the RPC boundary. Daemon placement would force a full-result round-trip just to annotate. |
| `cache`, `cache_*`    | Daemon    | Cache state lives in the daemon (provider-side prompt caching). |
| `gc`, `gc_*`          | Daemon    | GC operates on the session's history, which is a daemon concept (history lives in `JaatoSession`). |
| `streaming`           | Daemon    | Per-session streaming infrastructure tied to provider responses. |
| `telemetry`           | Daemon    | OTel spans cross the boundary; the runner emits sub-spans via RPC, the daemon publishes them on the bus. |
| `session`             | Daemon    | Session lifecycle plugin — by definition daemon-tier. |
| `*_auth` (anthropic, antigravity, github, nim, openrouter, zhipuai) | Daemon | `SESSION_INDEPENDENT = True`; these run before sessions exist. They never had any business in a runner. |
| `introspection`       | Runner    | Lists tools currently exposed to the model — the model lives "in" the runner from the tool dispatch POV, so the runner is the right place for the introspection commands. |
| `thinking`            | Runner    | Provider-backed thinking blocks; the tool surfaces them through the runner so chained tool calls are visible together. |
| `code_block_formatter`, `code_validation_formatter`, `diff_formatter`, `inline_markdown_formatter`, `mermaid_formatter`, `notebook_output_formatter`, `table_formatter`, `enrichment_formatter`, `formatter_pipeline`, `hidden_content_filter` | Daemon | Pure enrichment over tool results / model output. They post-process strings; no FS work; running them daemon-side avoids serializing big tool results across the boundary just to be reformatted and serialized again. |
| `reactor` (jaato-premium) | Daemon | Lives outside this repo. Subscribes to the daemon's EventBus. Out of scope for this design (it's already daemon-tier). |
| `background`          | Daemon    | Auto-background pool for long-running tools. The execution still runs in the runner, but the supervision (timeout escalation, status events) lives daemon-side. The runner posts `tool.background_promoted` and the daemon takes over status. |

**Don't-fit-cleanly flags:**

- `references` is the only true straddle: a single daemon-only RPC
  (the kernel-grant write) wrapped around an otherwise runner-tier
  plugin. The straddle is unavoidable because criterion 1 applies to
  `sudo apparmor_parser` (see §4.7).
- `permission` is the second straddle by design: UI relay + rule
  storage are daemon-side because they outlive runners and depend on
  channels that are daemon-owned. The *call site* is runner-side. §4.5
  has the full plan.
- `subagent` is runner-tier in the *parent*, but spawning a subagent
  creates a new session, which shares the parent's runner by default
  (see §4.3).
- `web_fetch` and `web_search` have no FS state. They sit in the runner
  because (a) the per-session cgroup applies to outbound network
  bandwidth via cgroup controllers, (b) `artifact_tracker` (also
  runner-tier) expects results from the same dispatch site as `cli`'s,
  (c) it keeps the daemon's dependency surface narrower.
- Soft daemon picks (criterion 2) that may move later: `cache_*` (provider
  prompt-cache state — sharing across sessions is the optimization),
  `gc_*` (history lives daemon-side today; if `JaatoSession` history
  fully moves to the runner, GC follows). None block Phase 5.

### 4.3 Subagent semantics

**Decision:** by default, subagents share the parent session's runner — so
they share the parent's AppArmor profile. Opt-in for an isolated runner via a
`subagent_isolation` knob in the spawn payload (`isolated: true`).

Rationale:

- Today subagents share the parent's `JaatoRuntime` in-process. The performance
  expectation is "spawn is cheap, ~10 ms." A new runner subprocess is ~50–200 ms
  (fork + AppArmor change_profile + plugin discovery). For a deep cascade
  (5 specialists × 3 tiers = 15 runners), the wall-time hit is real and the
  RSS hit is significant.
- The isolation that a separate runner buys you between *sibling subagents in
  the same workspace* is small. They're already in the same `tool_hat`
  sub-profile; they already have the same workspace allow list. The isolation
  that matters — between *workspaces* — is preserved because subagents always
  inherit their parent's workspace.
- A single runner per session is also the natural place for the EventBus.
  Sibling subagents need to share a bus to coordinate plan/handoff state.
  Splitting them across runners would require yet another bus-replication
  mechanism.

Opt-in for isolated runner: `agent_params.isolated: true` (or a profile-level
`isolation: "runner"`). When set, the subagent gets its own runner with a
fresh AppArmor profile name (`jaato-ws-{session_id}//{subagent_id}` —
sub-profile of the parent, but loaded as a separate runner process). Use case:
running an untrusted code-execution subagent inside an otherwise-trusted
workspace.

This is the contested decision (see §6).

### 4.4 Reactor / EventBus

**Decision:** EventBus lives in the daemon. The runner emits events to the
daemon via RPC; subscriptions live daemon-side. Loss is bounded by an
acknowledgement window; runner crash within the window means events are
dropped, and the daemon emits `RunnerCrashedEvent(session_id,
last_acked_seq)` so reactor consumers know to re-evaluate state.

Details:

- The runner has a small per-session outbox: in-memory deque, max 256 events.
  Each event carries a per-session monotonic `seq`.
- The runner sends `{kind: "event", seq, event: {...}}`. The daemon ACKs by
  carrying `last_acked_seq` on its next reply (any reply piggybacks the ack
  — no separate ack frame). Runner trims its outbox up to the ack.
- On daemon → runner reconnect, the runner replays from `last_acked_seq + 1`.
  But: there's no reconnect path in v1. If the daemon dies, the runner dies
  (it can't continue without the daemon's RPC stubs for permission/memory/
  provider calls). The outbox is purely a back-pressure cushion.
- Lossy on runner crash. The daemon detects via socket EOF and emits
  `RunnerCrashedEvent(session_id, last_seq, exit_code)`. The reactor framework
  must handle "I might have missed events" — this matches today's contract,
  where reactor consumers already cope with daemon restart.
- The reactor framework (in `jaato-premium`) does not move. It still
  subscribes to the daemon's EventBus. Its observation of session lifecycle
  is unchanged: the daemon fires the same `AgentCompletedEvent`,
  `HandoffGate*Event`, etc., the same way.

**Rejected alternative — EventBus replicated in the runner:** would let
runner-side plugins subscribe locally and avoid the RPC for plan-state events.
Cost: bus state has to merge across processes (every session's plan state
becomes a distributed log), which is far more complexity than the use case
demands. Today every consumer that matters lives daemon-side (telemetry,
reactor, broadcast).

### 4.5 Permission flow

**Decision:** check stays in the runner's `ToolExecutor.execute`, but
`check_permission` becomes an RPC into the daemon. The daemon runs the policy
+ channel + UI relay. The runner caches the result for the session's lifetime
when the decision is `whitelist`, `session_whitelist`, `evaluator_session_whitelist`,
`allow_all`, `turn_suspension`, or `idle_suspension` — i.e. anything that
doesn't depend on per-call args (or where the policy explicitly carries a
"this is sticky" hint).

Detailed flow:

```
runner.ToolExecutor.execute(name, args)
  └─ runner.permission_cache.lookup(name)
       └─ hit (e.g. whitelist) → allow, no RPC, proceed
       └─ miss → RPC permission.check(name, args, context) → daemon
              └─ daemon.PermissionPlugin.check_permission()
                  └─ static rule hit: returns immediately
                  └─ ASK_CHANNEL: emits PermissionRequestedEvent on the bus,
                     waits on channel for response (the existing flow), then
                     replies on the RPC
              ← runner receives (allowed, perm_info)
       └─ if perm_info.method ∈ STICKY_METHODS:
              cache it (see "cache key" below)
       └─ proceed or return Permission denied
```

**Cache key semantics.** Tool name alone is NOT sufficient. Per-tool
argument-pattern whitelists (the `arguments.<tool>.<arg>: [glob, ...]`
shape verified at `permission/plugin.py:375-385`) are arg-dependent:
`call_service` with `service: maven_central` may be allowed while
`service: malicious_service` is denied. Caching by tool name alone would
allow the second call after the first.

The runner's permission cache MUST be keyed by `(tool_name,
arg-fingerprint)` whenever the policy that resolved the call has any
per-arg rule that could apply. Two valid implementations:

1. **Fingerprint always.** Compute a stable hash of (tool_name, sorted
   args), use as cache key. ~5–10 µs overhead per cache lookup; safe.
2. **Daemon flags arg-dependent decisions.** The daemon's reply carries
   `arg_dependent: true` when any per-arg rule was evaluated; runner caches
   only when `arg_dependent: false`. Avoids hashing on every lookup at
   the cost of the daemon knowing more about its own policy.

Default to (1) for the simpler invariant: cache lookups always
fingerprint, no daemon-side classification needed. For evaluator /
allow_with_comment / dynamic-context decisions, skip caching entirely
(both methods imply the answer may change).


- The latency hit per uncached interactive permission is a single round-trip
  on a Unix socket plus the existing channel wait. Round-trip is ≤ 1 ms;
  channel wait is human-bound, so the additional ~1 ms is invisible.
- Cached permission (whitelist, allow_all, turn_suspension): zero RPC,
  zero round-trip.
- Cache invalidation: the runner subscribes to permission-change events
  (rule added/removed, suspension cleared). When the daemon clears
  `_turn_suspended` at turn end, it sends `permission.invalidate()` to the
  runner; the runner clears its cache.
- `trusted_bridge_context` (notebook plugin's nested tool calls) stays
  thread-local in the runner — no RPC needed; the runner already knows
  it's in a trusted context.
- Edited args (the `was_edited` / `modified_args` path): the daemon's reply
  carries `modified_args` over the wire when present. Runner uses them.

Subagent permission inheritance (`ParentBridgedChannel`) needs no special
handling: it's a daemon-side construct, and the runner's RPC just sees the
final allow/deny.

### 4.6 Runner lifecycle

**Decision:**

- **Spawn:** `RunnerSpawner.spawn(session_id, workspace_path, profile_name,
  env)` is called from `SessionManager.create_session` AFTER the AppArmor
  profile is loaded (today's `_run_pre_initialize_hooks` already provisions
  the profile before `server.initialize()`; we hook in here). Spawn creates
  the socketpair, forks, exec's `python -m server.runner` with workspace
  context in env. The fork inherits no Python state — clean cold start.
- **Bootstrap:** the runner entry point does, in order:
    1. Read profile name from `JAATO_RUNNER_PROFILE` env.
    2. Call `aa_change_profile(profile_name)` via `ctypes` against
       `libapparmor.so.1` (we don't add `pyspnego`-style native bindings —
       the one libapparmor symbol we need is `aa_change_profile`).
    3. Verify by reading `/proc/self/attr/current`. The expected value is
       the profile name + ` (enforce)`. If it doesn't match, log the actual
       value and `os._exit(2)`. No fallback to unconfined — that is the
       failure class we're fixing.
    4. *Now* import plugin code. Discover via the normal entry-point
       mechanism. Initialize runner-tier plugins. Build daemon-tier RPC
       stubs (auto-generated from a manifest — see §4.8).
    5. Send `RunnerReadyEvent` to daemon. Begin serving.
- **Death — graceful:**
    - Daemon sends `runner.shutdown` (gives the runner a chance to flush
      tool output, close interactive_shell sessions cleanly, terminate MCP
      servers).
    - Wait up to 5 s for runner to exit. SIGTERM. Wait 2 s. SIGKILL.
    - **Atomic-write requirement for runner-tier persistence.** SIGTERM
      arriving mid-write to a runner-tier state file (waypoints.json,
      todos.md, .artifact_tracker.json, etc.) is a real failure mode.
      Every runner-tier plugin that persists state MUST use the
      `tempfile + os.replace` atomic-write pattern (waypoint already does;
      memory's curated layer does; **artifact_tracker's `_save_state`
      currently does NOT** — flag for fix as a Phase 3 audit task). The
      audit also covers todo, file_session (tested via session_manager
      pre-resolution), and any runner-tier plugin that lands during
      Phase 3 migration. A signal-handler that sets a "shutting down"
      flag the plugins check before starting a write is NOT sufficient —
      the SIGTERM-after-write-began window is what atomic-write closes.
- **Death — idle:** runner has no idle timer. Idle handling stays
  daemon-side (where session idle is already tracked); on long idle the
  daemon issues `runner.shutdown`. Reasoning: keeping the runner alive
  costs ~30 MB RSS, which is acceptable for an interactive workflow but
  not for a daemon hosting hundreds of stale sessions. The current default
  is to keep runners alive for the lifetime of the session; operators
  with high session counts can flip a daemon-level idle-runner-shutdown
  knob (out of scope for v1).
- **Death — daemon shutdown:** all runners get `runner.shutdown` → SIGTERM
  → SIGKILL. Same as today's WS workspace teardown.
- **Death — runner crash:** daemon detects via socket EOF. Marks session
  as `failed` (new status). Emits `SessionFailedEvent(session_id, reason,
  exit_code)`. Future `SendMessageRequest` against this session returns
  an error. The session record is preserved on disk; restart-by-rerun is
  the operator's choice. We do **not** auto-respawn; that masks the
  underlying failure.
- **Apparmor profile load:** stays in the daemon (it requires
  `sudo apparmor_parser -r`, root-side). The daemon provisions the
  profile, then spawns the runner. `_run_pre_initialize_hooks` already
  fires before `initialize()`; the runner spawn slots in there and
  `initialize()` itself happens in the runner.
- **Post-spawn config:** the daemon's `JaatoServer` still exists per
  session; it now holds runner state (profile name, RPC client,
  daemon-tier plugin instances) instead of the in-process `JaatoRuntime` +
  `JaatoSession`. The runner holds the actual `JaatoSession`. This split
  is the biggest refactor in Phase 3.

### 4.7 AppArmor profile generation

**Decision:** keep the existing per-workspace profile name pattern
`jaato-ws-{session_id}` (one profile per session, because
`{session_id}` is unique). The runner is exec'd by the daemon with the
profile name in env; the runner self-confines via `aa_change_profile`. No
`aa-exec` wrapper.

- Same template (`PROFILE_TEMPLATE` in `server/apparmor.py`). The runner
  inherits exactly what the daemon's confined-thread pattern inherits today.
  Including the `change_profile -> unconfined,` rule and the
  `tool_hat` sub-profile. Subprocess inheritance still works.
- Sub-profile transition for tool execution stays the same:
  `change_profile -> jaato-ws-{session_id}//tool_hat,` is in the base profile;
  the runner's `ToolExecutor.execute` enters the sub-profile per call (same
  context manager pattern as today's `apparmor_confine`). The difference is
  that the **whole runner process** is in `jaato-ws-{session_id}` from the
  moment it starts, not just specific worker threads in a shared daemon.
- Reference fragments (per-`selectReferences` grants under
  `jaato-ws-{session_id}.refs.d/`) are loaded by the daemon. Runner
  emits `apparmor.add_fragment` RPC; daemon writes the file and runs
  `sudo apparmor_parser -r`. The runner picks up the new rule on its
  next file open via the existing `include if exists` directive — no
  runner restart needed. **Why not the runner directly:** the runner
  is AppArmor-confined and the per-session profile uses `/usr/bin/**
  ix,`. Under `ix` the kernel strips setuid on exec, so even though
  the daemon's sudoers rule covers the runner's UID, the runner's
  `sudo` invocation cannot elevate to root. Switching to `Ux` for
  `/usr/bin/sudo` would let sudo run **unconfined** (with anything it
  invokes also unconfined) — exactly the escape vector the per-session
  profile exists to close. Independently, the existing sudoers entry
  permits `apparmor_parser` with arbitrary args; an LLM-driven runner
  with sudo access could load attacker-controlled profiles or unload
  sibling sessions' profiles, and `_validate_path_for_fragment` would
  no longer be a meaningful gate. Keeping the kernel-mutation step
  daemon-side preserves both the `ix` confinement and the validation
  trust boundary.
- Why not parametric profile + change_profile at runtime: the daemon
  already supports per-workspace profiles via the same mechanism we'd
  reuse parametrically. The cost of the per-session profile load is
  amortized — `apparmor_parser` caches against `~/.jaato/apparmor-cache/`
  and re-loads in milliseconds for an unchanged template. The per-session
  profile gives finer kernel-level audit attribution
  (`dmesg | grep apparmor` shows `jaato-ws-20260506_001234`, not a
  generic profile name).
- The daemon itself: we need a new (much narrower) daemon profile that
  allows reading every workspace's `.jaato/profiles/`, `.jaato/agents/`,
  etc. This is **opt-in** via the same `ClientConfigRequest.apparmor`
  flag that exists today, but it's now a daemon-side profile separate
  from any session profile. Phase 6 lands the daemon profile.

### 4.8 Tool result schema across the boundary

**Decision:** typed envelope. Every RPC reply uses

```json
{
  "id": <int>,
  "kind": "response",
  "ok": <bool>,
  "result": <any>,
  "error": null | {"type": str, "message": str, "traceback": str?},
  "warnings": [<str>],
  "telemetry": {<str>: <any>}
}
```

- `result` is the existing `Tuple[bool, Any]` collapsed into a JSON-friendly
  payload. The `Any` side is whatever the executor returns today (typically
  a dict, sometimes a string). We do **not** define a stricter schema for
  the result body — every plugin already builds its own dict shape, and
  enforcing a single schema would force a rewrite that's out of scope.
- `ok` is the boolean half of today's `Tuple[bool, Any]`. Redundant with
  the absence of `error` for normal cases; explicit because some tools
  legitimately return `(False, {...})` without raising.
- `error.traceback` is the full Python traceback when the executor
  threw. Currently the in-process `_execute_impl` returns
  `{"error": ..., "traceback": ...}`; we preserve that information in
  the typed envelope so the model still sees the same diagnostic on
  permission errors / executor exceptions.

  **Cross-tenant info-leak caveat.** Python tracebacks contain absolute
  filesystem paths (workspace_root in module imports, source-line file
  references). Under multitenant deployment a stale traceback in
  daemon-side logs OR in an event forwarded to a different operator
  could expose another tenant's workspace path. The runner SHOULD
  path-sanitize tracebacks before crossing the RPC boundary —
  redact absolute paths matching `/<workspace_root>/...` to
  `<WORKSPACE>/...`, and absolute paths matching `~/.jaato/...` to
  `<HOME>/.jaato/...`. Phase 3 task; the sanitization is a single
  regex pass over the joined traceback text.
- `warnings`: aligned with the codebase-wide payload-schema convention
  (`docs/design/payload-schema-conventions.md`) of "every contract has
  `warnings[]`". The runner injects warnings for: timeout-near-cap,
  output-truncated, deprecated-tool-name. Most calls have empty `warnings`.
- `telemetry`: replaces the `_telemetry` key the existing tool runner
  injects into result dicts (cgroup deltas etc.). Pulling it out of the
  payload is a small cleanup — the daemon's OTel forwarder picks it up
  from the envelope instead of from `result["_telemetry"]`. Backwards-
  compat caveat: agent-visible tool results should never contain
  `_telemetry` after this lands; verify this isn't being read by any
  downstream consumer (Phase 3 subtask).

The same envelope shape applies to runner→daemon RPCs (permission,
memory, references, telemetry). Symmetric, single decoder.

JSON-non-serializable values (e.g. `bytes`, `datetime`, custom objects)
must be handled at the runner side before send. The `cli` plugin already
emits text; `interactive_shell` already emits text; `multimodal` emits
base64 in JSON. We don't add a generic pickle path — it would silently
move trust over the boundary.

## 5. Single-tenant in-process mode after Phase 6

**Decision: drop it.** No `legacy_mode=True` flag.

The daemon runs as `runner-spawner + daemon-tier plugins`, full stop. The
runner runs runner-tier plugins. There is no path that runs a tool in the
daemon's process. The only configuration knob is `JAATO_RUNNER_DISABLE=true`,
which is not a supported deployment — it's an escape hatch for the developer
loop ("run jaato in one process under pdb, accept that AppArmor and
multitenancy don't work"). We keep this knob for ~one release for plugin
authors who can't easily attach to a runner subprocess; we delete it after
that. Mention this prominently in the v6 release notes.

Reasoning: the failure mode this design fixes is exactly "two clients can't
share a daemon." Keeping a code path that re-introduces that failure mode as
"legacy" creates a permanent test matrix burden and a permanent foot-gun
("oh, you're hitting the cross-workspace bug because you set
`legacy_mode=True`"). The scope of changes is also too deep for a feature
flag — `JaatoSession` lives in the runner now; threading "or maybe in the
daemon" through every site that touches `runtime.create_session()` means
the legacy path doesn't actually share code with the new path.

If we discover after Phase 5 that there's a deployment that genuinely
needs in-process mode — e.g., embedded use-cases where forking is
expensive or impossible (Windows? — we don't currently support Windows
runners; the AppArmor primitive doesn't exist there) — we'll redesign.
A reasonable shape is "run the runner as a thread group inside the daemon
on platforms where AppArmor is unavailable, log a clear warning that
multitenancy is unsupported on this platform." Out of scope for v1.

**Windows handling explicitly:** the daemon today supports Windows via
named pipes for IPC. The runner design as written is Linux-only (no
AppArmor → no per-session confinement). For Windows, Phase 6 adds an
in-process compatibility runner: same RPC interface, but it runs in the
daemon process. This is **not** the "legacy_mode" — it's a platform
compatibility layer for a platform that has no equivalent of AppArmor.
Multitenancy on Windows remains unsupported. A startup WARN log
makes this visible to operators.

**macOS handling explicitly:** macOS has no AppArmor; the equivalent
primitive is `sandbox-exec` (deprecated but still functional) or the
SIP / endpoint-security framework (more complex, requires entitlements).
For v1, macOS gets the same in-process compatibility runner as Windows:
same RPC interface, runs in the daemon process, multitenancy
unsupported, startup WARN log. A future v2 design could add
sandbox-exec-backed per-session confinement; out of scope for this
document. The Phase 6 cross-platform compatibility runner is shared
between Windows and macOS — same code path, same multitenancy-disabled
posture.

## 6. Open / contested items

These are surfaced for review rather than decided unilaterally.

**6.1 Subagent runner sharing.** The default chosen above (subagents share
the parent's runner) trades isolation for spawn cost. The alternative
(every subagent gets its own runner) is more correct from a defense-in-depth
perspective but ~20× slower to spawn and ~5–15× more memory for deep
cascades. I picked "share by default, opt-in for isolation" because the
existing subagent flow already gives parent-and-children the same workspace
profile in the daemon; the runner-level sharing matches that. **Contested:**
some readers may prefer "isolate by default, opt out for cost." The right
default depends on the kind of workloads we expect to be the common case.
For the cascade workloads in `handoff_test`, share-by-default is much faster.

**6.2 Permission cache invalidation breadth.** The cache plan above is
"cache anything the policy says is sticky." A more conservative plan is
"cache nothing; pay one round-trip per tool call." For the common case
(model emits 5–10 tool calls per turn, most of them whitelist hits via
permission), the savings are ~5–10 ms per turn — not nothing, but not huge.
**Contested:** is the cache worth the invalidation complexity? A safer
"cache nothing" v1 lets us defer the invalidation contract to v2.

**6.3 Daemon profile in Phase 6.** Whether the daemon also gets confined to
its own profile (limiting what it can read across workspaces) is an extra
hardening pass that's not strictly required by the multitenancy goal. The
runner-per-session pattern alone already prevents cross-workspace tool
exfiltration. **Contested:** do we land the daemon profile in Phase 6, or
defer to a follow-up? Argument for landing: the daemon can today read
every workspace's `.jaato/`, which is information leakage across tenants
even with kernel-confined runners. Argument for deferring: the threat
model for the daemon (compromised by what?) is much narrower than the
threat model for tools (driven by an LLM that may be jailbroken).

**6.4 RPC call backpressure.** The dispatch protocol is multiplexed onto
one socket. Long-running tool RPCs (`shell_input` waiting on a slow REPL,
a 30-second `cli` invocation) are multiplexed with high-frequency event
emissions (telemetry spans, plan state changes). The 10 MB frame cap is
enough; the buffering on a Unix socket is enough. But pathological cases
(e.g., a tool that emits 100 KB/s of stdout for an hour) could in
principle starve event delivery. **Contested:** do we add a separate
event channel up front to side-step the bandwidth issue, or measure first
in Phase 4? I picked "single channel, measure first" — but readers with
high-throughput tooling experience may have an opinion.

**6.5 Runner crash session-recovery.** The decision above is "no
auto-respawn, mark session failed." This matches today's behavior on a
daemon crash but is a regression compared to "the daemon process is
robust" today. **Contested:** for cascading workloads where the runner
might OOM under a runaway tool, do we want auto-respawn-once with a
fresh empty history? The trade-off is silent failure-recovery vs.
forcing the operator to notice. Lean toward the latter for v1.

**6.6 MCP server lifetime across runner restart.** MCP servers today are
spawned by the `mcp` plugin during session initialize. If a runner dies
and the daemon-tier policy chose to auto-respawn (per 6.5), the MCP
servers are also gone — every connected MCP server goes through teardown
and the new runner re-spawns them. This is correct behavior (the MCP
protocol expects a fresh client session) but is observable as latency.
Worth flagging in the docs but not a blocker.

**6.7 Per-client `apparmor` flag vs. process-level daemon profile.** §4.7
notes that "the daemon itself: we need a new (much narrower) daemon
profile" and is "opt-in via the same `ClientConfigRequest.apparmor` flag
that exists today." This punts a real contradiction to Phase 6: today's
`apparmor` flag is **per-client** (each `ClientConfigRequest` carries
its own value); a daemon profile is **process-level** (one daemon, one
loaded profile, applies to every connected client). What does it mean
for client-A to set `apparmor: True` and client-B to set `apparmor: False`
on the same daemon? Three resolutions, none decided in this design:

- **Per-client wins for runners; daemon profile is daemon-level only.**
  The flag scopes only what gets enforced on the runner's profile; the
  daemon-side profile is set once at daemon startup via a daemon-level
  config knob, independent of any client.
- **Strictest-client wins for daemon.** Daemon loads its own profile
  whenever ANY connected client requests apparmor; profile stays loaded
  until ALL clients disconnect. Adds connection-tracking state and
  reload-on-last-disconnect logic.
- **Daemon profile is operator-configured, not client-driven.** The
  per-client `apparmor` flag controls only the runner; daemon profile
  is `--daemon-apparmor` at startup. Cleanest separation; matches the
  way other daemon-level flags work today.

Lean toward the third resolution — operator-configured daemon profile
is a deployment decision, not a per-session ask. Decide explicitly in
Phase 6.

## 7. Phased plan recap

The brief already lays out Phases 1–6. This document is Phase 1's
deliverable. Phase 2 (runner skeleton + RPC) is the next checkpoint;
no implementation should start until this design is reviewed and
approved.

Existing code anchors that Phase 2 will touch:

- `jaato-server/server/__main__.py:_register_ipc_apparmor_hook` (lines
  656–835) — the daemon-side AppArmor hook gets relocated to spawn a
  runner instead of confining a daemon thread.
- `jaato-server/server/session_manager.py:create_session` (lines
  948–1382) — `RunnerSpawner` integration lands after profile
  provisioning, before `server.initialize()`.
- `jaato-server/shared/ai_tool_runner.py:execute / _execute_impl`
  (lines 756–950) — the daemon-side `ToolExecutor` becomes an RPC stub;
  the actual `_execute_impl` body moves to the runner.
- `jaato-server/server/ipc.py` — the framing helpers (`_read_frame`,
  `_write_frame`) get factored into a shared module that both the IPC
  server and the runner-RPC client/server import.
- `jaato-server/shared/jaato_runtime.py:create_session` and
  `jaato-server/shared/jaato_session.py` — `JaatoSession` moves to the
  runner; the daemon retains `JaatoServer` (a thin shell) and a
  runner-RPC handle.
- `jaato-server/shared/plugins/permission/plugin.py:check_permission`
  (lines 1058+) — split into a thin runner-side cache + a daemon-side
  full implementation, glued by RPC.
- `jaato-server/server/apparmor.py:PROFILE_TEMPLATE` — unchanged
  template, but the load site moves earlier (before runner spawn) and
  the unload site moves later (after runner exit).

Phase 6 cleanup will remove: the `apparmor_confine` thread-context
machinery from `server/apparmor.py` (the per-thread pattern is no longer
needed because the whole runner process is confined), the
`SafeThreadPoolExecutor`'s pre-task AppArmor hook (no thread is ever
confined daemon-side), and the
`set_apparmor_context` / `_apparmor_context` plumbing in
`shared/ai_tool_runner.py`.

## 8. Success criteria (Phase 5 acceptance gate)

Phase 5 validates multi-tenant correctness end-to-end. The design ships
when ALL of the following hold; until then Phases 2–4 are not "done."

**8.1 Functional correctness.**

- Two cascades from two different workspaces, started concurrently
  against a single daemon, both run end-to-end with no permission-deny
  errors at IPC handshake or tool execution.
- Daemon restart is NOT required to switch workspaces. A client
  connecting from workspace-B after workspace-A's session is in flight
  succeeds without daemon bounce.
- handoff_test cascade `--case stp_approval` produces byte-identical
  output to the pre-runner baseline (modulo legitimate timestamps).
  Same for kb-enablement-2.0's smoke cascade.

**8.2 Cross-workspace isolation.**

- Integration test (`tests/integration/test_multitenant_apparmor.py`)
  confirms a tool call in session-A workspace cannot write to OR read
  sensitive paths in workspace-B's tree. Verified via:
    - attempted-write-fails (positive assertion: write returns EACCES)
    - AppArmor audit log entries (`dmesg | grep apparmor` shows the
      kernel-level deny for the cross-workspace path).
- 7:3's original failure (`Permission denied:
  kb-enablement-2.0/.jaato/profiles` during cross-workspace IPC
  handshake) reproduces as a regression test that's now green.

**8.3 Performance budget.**

- **Tool-call RPC overhead ≤ 5 ms p50** for in-memory tools (todo
  list, file_edit metadata operations, introspection). Measured via
  the existing `_telemetry` field in the result envelope.
- **End-to-end cascade wall-time regression ≤ 10%** vs. the pre-runner
  baseline on handoff_test stp_approval and kb-enablement-2.0 smoke.
- **Runner spawn latency ≤ 200 ms p99**. Measured on a cold runner
  (no apparmor-cache hit) and a warm runner (cache hit) separately.

**8.4 Operational soundness.**

- Runner crash → daemon emits `SessionFailedEvent` within 100 ms of
  socket EOF.
- Daemon restart → all runners receive `runner.shutdown`; SIGTERM
  ladder completes within 7 s for a session with one open
  `interactive_shell` PTY.
- Apparmor-unavailable platform (macOS, Windows, Linux without
  apparmor module) → daemon starts with WARN log, multi-tenancy
  flagged-unsupported, single-tenant cascade still works in-process.

**8.5 Backwards compatibility.**

- Existing test suite passes (no functional regressions visible to
  clients). Specifically: jaato-server's full pytest suite, jaato-sdk's
  full pytest suite, handoff_test orchestrator integration tests,
  kb-enablement-2.0 cascade smoke.
- `_telemetry` removal from agent-visible tool results: zero downstream
  consumers reading `tool_result["_telemetry"]` — verified via grep
  across all consumer repos before merge.

If 8.1–8.5 hold, the design ships. If any criterion fails, surface as
a Phase-N regression and address before promoting to "done."
