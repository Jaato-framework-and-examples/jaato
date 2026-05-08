# Phase 3 Implementation Plan — Bulk Plugin Migration + JaatoSession Move

**Status:** Draft, not implementation-ready. Awaiting review.

**Parent design:** `per_session_confined_runner.md` (Phase 1 design).
**Predecessor:** `per_session_confined_runner_phase2_plan.md` (merged
2026-05-08, PR #53, merge commit `f9971a50`).

Phase 2 shipped the runner skeleton, RPC framing, the `cli` plugin
migration as the validation vehicle, and the IPC-path runner spawn
(via the 4-arg pre-init hook).  Phase 3 is the scope expansion:
move every other runner-tier plugin into the runner, move
`JaatoSession` itself, wire the two daemon-tier RPC primitives that
runner-tier plugins consume (`client.prompt_operator` for
permission, `apparmor.add_fragment` for references), close the
three deferred bootstrap paths, and address the audit tasks the
parent design flagged for Phase 3.

Section references (§3 / §4.1 / §4.6 / §4.7 / §4.8 / §8.x) point
into the parent design doc.  Phase 2 plan section references are
prefixed `phase2.§X.Y`.

---

## 1. Goal + critical-path constraint

**Goal.** After Phase 3 lands, the daemon process holds: provider
clients, OAuth state, GC, streaming, formatter pipelines, telemetry
forwarding, IPC + WS transports, the EventBus, the reactor framework,
and the two RPC-server primitives (`client.prompt_operator`,
`apparmor.add_fragment`).  Every other plugin runs runner-side.
`JaatoSession` lives in the runner; the daemon retains a thin
`JaatoServer` shell holding the runner-RPC handle and the daemon-tier
plugin instances.  All four session-bootstrap entry points spawn a
runner when AppArmor opt-in is set.

**Critical-path constraint.** The migration order must respect three
dependency edges:

1. **`client.prompt_operator` RPC primitive (§3.2) must land BEFORE
   the permission-plugin migration (§3.7).** Without the primitive,
   permission's ASK path has nowhere to relay through.
2. **`apparmor.add_fragment` RPC primitive (§3.2) must land BEFORE
   the references-plugin migration (§3.8).** Same reason —
   `selectReferences` widening the runner profile needs the daemon
   to actually load the fragment.
3. **JaatoSession runner-side scaffolding (§3.3) must land BEFORE
   ANY runner-tier plugin migration (§3.4–§3.10).** Plugins are
   instantiated against a `JaatoSession`; without the session
   runner-side, there's nowhere to host them.

Tasks within those constraints can ship in any order; intra-task
ordering is documented per task.

The migration order this plan recommends is conservative: ship the
two RPC primitives + the JaatoSession scaffolding first; then bulk
plugin migrations grouped by complexity; close bootstrap paths
last.  See §5.1 for the migration-order alternatives.

---

## 2. File-layout updates

### 2.1 New `jaato-server/server/runner/`

```
jaato-server/server/runner/
├── (Phase 2 files unchanged: __main__.py, bootstrap.py, rpc.py,
│   envelope.py, tool_executor.py, echo_tool.py, cli_runner.py)
├── session.py              # NEW — runner-side JaatoSession host
│                           # (instantiates JaatoSession against the
│                           # session profile, wires plugins, runs
│                           # configure(), holds the live session)
├── plugin_loader.py        # NEW — runner-side plugin discovery
│                           # (entry-points scan + tier filter +
│                           # instantiation; mirrors daemon's
│                           # PluginRegistry.discover but filtered
│                           # to runner-tier).
├── rpc_client.py           # NEW — runner→daemon RPC client
│                           # (calls client.prompt_operator,
│                           # apparmor.add_fragment, telemetry
│                           # publish; bidirectional dispatcher
│                           # was already in rpc.py from Phase 2).
└── sanitize.py             # NEW — path-redaction regex pass for
                            # tool-result envelopes (§3.1).
```

### 2.2 Daemon-side primitive surfaces

```
jaato-server/server/
├── runner_rpc_handlers/    # NEW — daemon's handlers for runner
│   ├── __init__.py         #   →daemon RPCs.  Each handler is a
│   ├── prompt_operator.py  #   small async fn registered into
│   ├── apparmor_fragment.py#   the RunnerRPCServer's dispatch
│   └── telemetry.py        #   table at session-init time.
└── runner_rpc_server.py    # Existing from Phase 2 — gains the
                            # registration plumbing.
```

`runner_rpc_handlers/` keeps the dispatch table tight without
ballooning `runner_rpc_server.py`.  Each handler imports the
daemon-tier API it relays (e.g.
`prompt_operator.py` calls into the existing
`PermissionRequestChannel` / `PermissionResponseChannel`).

### 2.3 Plugin migrations

Most plugins keep their `shared/plugins/<name>/plugin.py` location;
the body splits into:

- **Daemon-side stub** — minimal class that forwards `_execute_*`
  through `registry.runner_rpc`.  Pattern from Phase 2 cli.
- **Runner-side body** — the full `_execute_streaming` / `_execute`
  bodies, plus any helpers the daemon doesn't need.

The runner side typically lands as `shared/plugins/<name>/runner.py`
or stays in `plugin.py` if the file is small (the daemon-side stub
just imports the same `_execute` for the in-process fallback path
during the transition).

Phase 6 cleanup will collapse the stub-pair pattern once nothing
calls plugins in-process from the daemon side.

---

## 3. Task-by-task breakdown

### 3.1 — Tool-result sanitization (path-redaction regex)

Lands first.  It's small, it's defensive, and it prevents
cross-tenant info-leak going forward.  Per parent §4.8 line 709.

Files touched:
- New `jaato-server/server/runner/sanitize.py`: one function
  `sanitize_traceback(text: str, workspace_root: str) -> str`.  Two
  regex passes:
  1. `re.sub(r'(?<!\w)' + re.escape(workspace_root) + r'/', '<WORKSPACE>/', text)` — tenant-specific path.
  2. `re.sub(r'(?<!\w)' + re.escape(str(Path.home())) + r'/\.jaato/', '<HOME>/.jaato/', text)` — operator-side path.
  Order matters: workspace path may live under `~/.jaato/` for some
  configs; tenant pass runs first so the more-specific path wins.
- `runner/envelope.py`: when constructing an error envelope, call
  `sanitize_traceback` on `error.traceback` before serializing.
- `cli_runner.py` and any other runner-side error path that
  serializes `traceback.format_exc()` — pipe through `sanitize`.

Tests: `runner/tests/test_sanitize.py` — workspace path replaced,
home-jaato path replaced, no false positives on ordinary words
that contain "/home" as a substring (the `(?<!\w)` lookbehind is
load-bearing).

One commit.  No interaction with other work.

### 3.2 — Daemon-tier RPC primitives

Two thin handlers registered into the runner-RPC server.  Both are
generic capabilities the runner consumes; neither is plugin-specific.
Per parent §4.5 (permission) and §4.7 (references).

#### 3.2.1 — `client.prompt_operator(prompt: PromptPayload) → PromptResponse`

Files touched:
- `runner_rpc_handlers/prompt_operator.py` — async handler, takes a
  `PromptPayload`, calls into the existing
  `SessionManager.permission_request_channel` (the same channel
  today's daemon-side permission plugin uses), awaits the response,
  returns it.
- `runner/rpc_client.py` — runner-side wrapper.  Single async method
  `prompt_operator(payload) → response`.
- `shared/plugins/permission/types.py` — `PromptPayload` /
  `PromptResponse` dataclasses (re-exports of the existing channel
  types).
- `runner_rpc_server.py` — registers the handler at session-init.

The primitive intentionally takes a PromptPayload (string + options
+ session-id + tool-name) rather than permission-specific args.
Future plugins (interactive consent, UI confirmation, etc.) reuse
the same primitive without adding a sibling RPC.

Tests: `tests/test_runner_rpc_handlers.py:test_prompt_operator_*` —
round-trip with a stub permission channel; cancellation propagation;
multiple concurrent prompts (per session).

#### 3.2.2 — `apparmor.add_fragment(fragment: AppArmorFragment) → FragmentLoadResult`

Files touched:
- `runner_rpc_handlers/apparmor_fragment.py` — async handler, takes
  an `AppArmorFragment` (a typed envelope wrapping the existing
  `AppArmorManager.add_fragment` args), validates the path against
  the per-session allow-list (`_validate_path_for_fragment`), calls
  `AppArmorManager.add_fragment`, returns the result.
- `runner/rpc_client.py` — wrapper.
- Validation lives daemon-side (the LLM-driven runner cannot be
  trusted to validate paths it asks to load).
- `runner_rpc_server.py` — handler registration.

Tests: `tests/test_runner_rpc_handlers.py:test_apparmor_fragment_*` —
valid fragment loads, invalid path rejects, fragment-already-loaded
no-op, concurrent fragment loads serialize at the daemon.

One commit per primitive.  Order doesn't matter between the two.

### 3.3 — JaatoSession runner-side scaffolding

Files touched:
- New `runner/session.py` — hosts the live `JaatoSession` instance
  on the runner side.  Receives a `session_init_envelope` over RPC
  at runner startup containing: session_id, workspace_path, profile
  name, plugin list (resolved by daemon's profile loader), plugin
  configs, system instructions.  Constructs the `JaatoSession`,
  runs `configure()`, signals readiness via `RunnerReadyEvent`.
- `shared/jaato_session.py` — gain `to_runner_envelope()` /
  `from_runner_envelope()` for the daemon→runner handshake.  No
  semantic change; this is serialization.
- `shared/jaato_runtime.py` — the daemon retains `JaatoRuntime`
  (provider config, OAuth state, ledger).  `create_session` no
  longer instantiates `JaatoSession` in-process; it builds the
  envelope, dispatches to the runner, and stores the runner-RPC
  handle.
- `server/core.py` — `JaatoServer` shell.  Holds: provider config
  (from runtime), runner-RPC handle, daemon-tier plugin instances
  (cache, gc, streaming, telemetry, formatter pipeline).  Loses:
  in-process `JaatoSession` reference, the in-process
  `ToolExecutor` (already gone in Phase 2 for IPC-path).
- `runner/__main__.py` — receives the session-init envelope as the
  first frame after RunnerReadyEvent, hands it to
  `runner.session.bootstrap_session(envelope)`.

This is the load-bearing refactor.  Once it lands, the runner
hosts JaatoSession; daemon retains only the shell.  Plugin
migrations §3.4–§3.10 can then proceed in parallel.

Tests: `tests/test_jaato_session_runner_handoff.py` — envelope
round-trip; session lifecycle (init / configure / send_message /
shutdown) entirely runner-side; cancellation; runner crash
during configure surfaces as `SessionFailedEvent`.

### 3.4 — Plugin migration wave 1: pure-FS plugins

Move all of these to runner-tier in one commit (or one per plugin
if review prefers smaller diffs — see §5.1).  These plugins have no
cross-process state, no RPC needs, just FS reads/writes within the
workspace.  Pattern: each plugin's daemon-side stub becomes a thin
RPC forwarder, body moves runner-side.

| Plugin | Notes |
|---|---|
| `file_edit` | Already partially scaffolded in Phase 2 (BackupManager registry pattern) |
| `filesystem_query` | Read-only; pure migration |
| `todo` | `.jaato/todos/` per-session JSON; uses tempfile+rename already |
| `template` | Reads workspace template files |
| `ast_search` | Reads workspace source files |
| `bundle` | Bundles workspace files |
| `vision_capture` | Writes screenshots to workspace temp dir |
| `prompt_library` | Reads `~/.claude/skills/` + `.jaato/prompts/` |
| `environment` | Reads env, writes scratch files |
| `multimodal` | Reads workspace images |

Tests per plugin: existing `tests/` directory under each plugin
keeps its coverage; add one integration test per plugin that
exercises the runner-side path end-to-end (spawn runner, invoke
tool, verify result envelope).

### 3.5 — Plugin migration wave 2: subprocess-spawning plugins

The plugins that spawn subprocesses NOW INHERIT the runner's
AppArmor profile via fork — the multitenancy promise becomes real
for these.  Pre-Phase-3 they ran in the daemon's unconfined
context; post-Phase-3 they're confined.

| Plugin | Subprocess type |
|---|---|
| `interactive_shell` | `pexpect` PTY |
| `lsp` | `rust-analyzer` / `pyright` / etc. |
| `mcp` | Each MCP server (long-lived stdio) |
| `notebook` | Python interpreter |

Per-plugin acceptance: existing test suite + a multi-tenant
integration test verifying the spawned subprocess takes the
session's profile (read `/proc/<child>/attr/current` from the
runner; assert it matches the session profile name).

### 3.6 — Plugin migration wave 3: outbound HTTP plugins

These don't need confinement for the HTTP traffic itself, but
their config reads + result enrichment are workspace-local.

| Plugin | Notes |
|---|---|
| `web_fetch` | Outbound HTTP; result enrichment via artifact_tracker (also runner-tier, §3.10) |
| `web_search` | Outbound HTTP; stateless |
| `service_connector` | Reads `.jaato/services/<name>/` config |
| `webhook` | Inbound HTTP listener; per-session subscription buffers |

### 3.7 — Permission plugin migration (depends on §3.2.1)

The big one.  Per parent §4.5: the entire plugin lives runner-side;
state keyed by session-id within the runner (parent + subagents
sharing the runner per §4.3 default).  Static-rule decisions resolve
locally, zero RPC.  ASK decisions cross via `client.prompt_operator`
from §3.2.1.

Files touched:
- `shared/plugins/permission/plugin.py` — body moves runner-side.
  `check_permission` runs entirely runner-local for static-rule /
  evaluator decisions.  ASK path calls
  `runner.rpc_client.prompt_operator(payload)`.
- Daemon-side stub: minimal — just enough for the model to see
  permission-related tools (`permission_status`, suspend/resume) in
  the schema.  Their bodies forward via RPC.
- Cross-session policy mutation (operator's
  `/permissions whitelist <tool>` command): daemon receives the
  command, RPCs `permission.add_rule(rule)` to the target session's
  runner.  Per §4.5 v1: per-session only, no daemon-side rule store.

Tests: existing permission test suite lifts and shifts to the
runner tests directory; add cross-runner mutation tests
(`/permissions whitelist X` from operator → session A's runner sees
the new rule; session B's runner unchanged).

### 3.8 — References plugin migration (depends on §3.2.2)

Per-tenant catalog reads, per-tenant embeddings, per-tenant index,
`selectReferences`, semantic match — all runner-local.  The
fragment-loading capability (used when `selectReferences` admits a
new path outside the workspace) crosses via
`apparmor.add_fragment` from §3.2.2.

Files touched:
- `shared/plugins/references/plugin.py` — body moves runner-side.
- Daemon-side stub: forwards the discoverable tool list (the
  `selectReferences` tool schema) so introspection sees it.
- Per-session references state (loaded references, embedding cache
  for that session) is runner-local.  Cross-session embedding cache
  reuse is criterion-2 optimization deferred — daemon placement
  available later if RAM cost bites.

Tests: existing references test suite lifts and shifts; add an
integration test that exercises `selectReferences` admitting a new
external path (verifies the fragment-load round-trip lands).

### 3.9 — Memory plugin migration

`~/.jaato/memories/` is rw under every session's profile (template
line 334).  The runner writes `memories/raw/<id>.json` and
`curated.jsonl` directly via tempfile-rename — same concurrency
story as today.

Files touched:
- `shared/plugins/memory/plugin.py` — body moves runner-side.
- Daemon-side stub: thin forwarder for the memory tool schema.

Embedding-cache sharing (criterion 2) — defer per parent §4.2.

Tests: existing memory test suite lifts and shifts; add a
multi-tenant test exercising `memories/raw/` concurrent writes
from two runners (verifies the existing tempfile+rename path
holds under cross-runner concurrency).

### 3.10 — Plugin migration wave 4: small runner-tier plugins

| Plugin | Notes |
|---|---|
| `artifact_tracker` | **CRITICAL: fix `_save_state` to atomic-write (§4.6 audit task).**  Use the same `tempfile.NamedTemporaryFile` + `os.replace` pattern as `waypoint`. |
| `reliability` | Per-session failure tracking |
| `waypoint` | Atomic-write already in place |
| `sandbox_manager` | Per-session allowlist |
| `clarification` | Inline UX, tool-local |
| `thinking` | Provider-backed thinking blocks; tool-local surface |
| `introspection` | Tool listing — model-perspective tool, runner is the right place |
| `subagent` | Migrates with §3.11 (subagent runner-sharing wires here) |

`artifact_tracker._save_state` is the audit task the parent design
flagged at §4.6 line 587.  Fix lands as part of the migration commit
for that plugin; no new helper, just adopt the pattern from
waypoint inline.

### 3.11 — Subagent runner-sharing semantics (depends on §3.10's `subagent`)

Per parent §4.3.  Default: subagents share the parent's runner.
Opt-in: `agent_params.isolated: true` (or profile-level
`isolation: "runner"`) spawns a new runner with a sub-profile of
the parent's profile (`jaato-ws-{session_id}//{subagent_id}`).

Files touched:
- `shared/plugins/subagent/plugin.py` — runner-side body.
  `spawn_subagent` checks the isolation knob; default path creates
  a new `JaatoSession` in the SAME runner (the runner's session
  table now keys by session-id).  Isolated path RPCs the daemon to
  spawn a new runner with the sub-profile.
- `runner/session.py` — supports multiple JaatoSessions per runner
  (the parent + N subagents).  Permission state already keyed by
  session-id from §3.7.
- `server/session_manager.py` — `spawn_isolated_runner` helper for
  the opt-in path.  Reuses §3.2's runner-spawn machinery with a
  sub-profile name.

Tests: parallel subagents in shared runner (default); isolated
subagent in fresh runner (opt-in); permission state isolation
between parent and subagent in the shared-runner case.

### 3.12 — Three deferred bootstrap paths

Closes Phase 2 plan §8.4's deferral list.  All three need the
runner-spawn wiring that Phase 2 only added to
`_create_session_impl`.

| Path | Phase 3 wiring |
|---|---|
| `SessionManager._load_session_impl` (disk-restore) | Re-spawn a runner for sessions loaded from disk that had AppArmor opt-in.  Profile name is preserved on the Session record (`Session.sandbox_mode`, set in Phase 2).  No client_id at restore time — accept that the IPC `client.prompt_operator` round-trip fails for restored sessions until the next interactive client attaches.  Document in the Session record that ASK decisions are deferred for orphan-restored sessions. |
| `SessionManager.run_ephemeral_session` | Ephemeral sessions are subagent fan-out; per §4.3 default they share the parent's runner.  Phase 3 adds the parent-runner reference to the ephemeral path so they actually share rather than spawn fresh.  Isolated-runner ephemeral subagents follow §3.11. |
| `JaatoWSServer` standalone bootstrap | The WS server has its own pre-init apparmor hook (`websocket.py:_apparmor_pre_init_hook`).  Phase 3 converts it from the legacy 3-arg signature to the 4-arg form (matching the IPC hook from Phase 2's §2.3 fix).  Runner spawn wires alongside. |

Each path gets one commit + integration test.

### 3.13 — IPC apparmor hook relocation (pre-init → inline session.new)

Per Phase 2 plan line 225.  The 4-arg pre-init hook from Phase 2
was a transitional step; the design endpoint is for the runner spawn
to live inline in the IPC `session.new` handler, which is already
keyed by client_id at that point.  This collapses one indirection
layer.

Files touched:
- `server/__main__.py` — remove
  `_register_ipc_apparmor_hook` / its `add_pre_initialize_hook`
  registration.  Logic moves into the IPC `session.new` handler
  in `server/ipc.py:_handle_session_new`.
- `server/ipc.py:_handle_session_new` — invoke the apparmor
  provisioning + spawn directly, before forwarding to
  `SessionManager.create_session`.
- `server/session_manager.py:_run_pre_initialize_hooks` — the
  4-arg signature stays (for the WS-side hook + any third-party
  pre-init hooks); the IPC hook just no longer registers there.

Tests: existing Phase 2 tests still pass; one new test verifies
the IPC `session.new` path spawns a runner without going through
`_run_pre_initialize_hooks`.

This task is independent of the bulk plugin migration; can land
early or late.  Listed late in the plan because the pre-init
hook is the safer state to ship migrations against (any plugin
issue surfaces in the existing pre-init path, not in a new code
path).

### 3.14 — JaatoSession shutdown signal-handler audit

Per parent §4.6 line 587.  The handler that sets a "shutting down"
flag plugins check before starting a write is **NOT** sufficient —
the SIGTERM-after-write-began window is what atomic-write closes.
The audit covers all runner-tier plugins that persist state.

Already in scope from earlier tasks: artifact_tracker (§3.10),
todo (§3.4), waypoint (§3.10 — already atomic), memory (§3.9 —
already atomic), file_session (§3.9 — already atomic via
session_manager pre-resolution).

§3.14's contribution: a single `tests/integration/test_runner_sigterm_durability.py`
that drives a runner mid-write on each persisting plugin, sends
SIGTERM, verifies the on-disk state is either the pre-write
version OR the post-write version (never partial / corrupted /
truncated).

### 3.15 — `_telemetry` key cleanup

Per parent §4.8.  Today the in-process tool runner injects
`_telemetry` into result dicts (cgroup deltas etc.).  Phase 3
pulls it out: telemetry rides the envelope's `telemetry` field
(populated by the runner side); the daemon's OTel forwarder reads
the envelope, not `result["_telemetry"]`.

Verification: grep for downstream consumers reading
`result["_telemetry"]` — expected zero.  Any hits get migrated to
the envelope side at the same time.

Files touched:
- `shared/ai_tool_runner.py` (or whatever survives of it post-§3.3)
  — stop injecting `_telemetry` into result dicts.
- `runner/envelope.py` — `telemetry` field already in the envelope
  schema (Phase 2); just ensure it's populated runner-side.
- Daemon-side OTel forwarder — reads from envelope.

One commit, small.  Could land any time after §3.3.

---

## 4. Test plan + acceptance gate

**Acceptance gate.** Phase 3 is done when:

1. `tests/integration/test_phase3_full_plugin_migration.py` —
   spawn a session with each runner-tier plugin loaded; verify
   every plugin's tool calls round-trip end-to-end (i.e. the
   daemon-side stub forwards correctly + runner-side body
   produces the right result envelope).
2. `tests/integration/test_phase3_subagent_runner_share.py` —
   parent + N subagents in the same runner; permission state
   isolated by session-id; opt-in isolated subagent gets its own
   runner with a sub-profile.
3. `tests/integration/test_phase3_disk_restore_runner_respawn.py`
   — session created with apparmor, daemon restarts, session
   loaded from disk, runner respawns, tools work.
4. `tests/integration/test_phase3_ws_bootstrap_runner.py` —
   WS-provisioned session takes a runner.
5. `tests/integration/test_runner_sigterm_durability.py` (§3.14)
   — runner-tier plugins survive SIGTERM mid-write.
6. The Phase 2 acceptance gate
   (`test_phase2_multitenant_apparmor.py`) is still green.

Combined runtime budget: ~3 minutes on the test host (heavier
than Phase 2's gate since it spans every runner-tier plugin).

**Per-task tests** are listed in each §3.x task above.

---

## 5. Open clarifications

These are the genuinely-controversial decisions worth surfacing
before any code lands.  For each, two recommended paths with
pros/cons; reviewer to pick.

### 5.1 Migration order — bulk vs incremental

**Option A: Bulk migration per-wave** (this plan's recommendation).
Each §3.4 / §3.5 / §3.6 / §3.10 wave lands as ONE commit migrating
all plugins in that wave.

- Pros: tighter PR review (one shape, repeated); one set of tests
  exercising the migration pattern; Phase 6 cleanup is simpler
  because the in-process fallback path is removed at one boundary.
- Cons: a plugin-specific bug surfaces inside a wave commit and is
  harder to bisect.

**Option B: One commit per plugin** (each §3.4–§3.10 entry → its
own commit).

- Pros: trivially bisectable; small reviewable diffs; bugs in
  plugin X don't block plugin Y.
- Cons: ~20 commits in Phase 3 just for the migration; reviewers
  see the same shape repeated 20× and patience erodes.

**Recommend A** for waves 1, 2, 3 (the pure / subprocess / HTTP
groups — same shape per plugin).  **Recommend B** for permission
(§3.7), references (§3.8), memory (§3.9) — these have plugin-specific
state shape and bisectability is more valuable.

### 5.2 JaatoSession migration — full move vs facade-then-move

**Option A: Full move in §3.3** (this plan's recommendation).
JaatoSession's class file moves to the runner's import path; daemon
no longer instantiates it.  One large commit.

- Pros: clean break, no period of "session lives in both places";
  the in-process fallback path collapses immediately.
- Cons: large commit; harder to incrementally debug; a regression
  surfaces against the whole shape change rather than a single
  method.

**Option B: Daemon-side facade + per-method migration**.
JaatoSession stays daemon-side initially as a facade that RPCs every
method through.  Per-method migrations move logic runner-side over
N commits.

- Pros: bisectable per-method; the facade pattern is familiar from
  the Phase 2 cli stub.
- Cons: longer in-flight period of two implementations; the facade
  itself is throwaway code; ~15 commits adds to PR review fatigue.

**Recommend A.**  The Phase 2 cli stub already proved the pattern
generally works; the JaatoSession move is contained to one logical
unit (session lifecycle) so a single commit is reasonable.  Reviewer
push-back on diff size pivots to Option B.

### 5.3 `client.prompt_operator` shape — generic vs permission-specific

**Option A: Generic primitive** (this plan's recommendation).
`client.prompt_operator(payload: PromptPayload) → PromptResponse`,
where `PromptPayload` is a discriminated-union of prompt kinds
(permission-ask, generic-confirm, free-text-input).

- Pros: one RPC for all current + future operator-interaction needs;
  parent design §4.5 explicitly frames it as a generic capability.
- Cons: the discriminated-union grows; payload types proliferate.

**Option B: Permission-specific** `permission.ask(args) → decision`.

- Pros: simpler API; type-tight today.
- Cons: violates parent §4.5's framing; future plugins (interactive
  consent, confirmation dialogs) duplicate the relay machinery.

**Recommend A.**  Parent design is explicit about this.

### 5.4 Subagent isolation default — share vs new

**Option A: Default to share parent's runner** (this plan's
recommendation, matches parent §4.3).  Opt-in `isolated: true` per
spawn for untrusted code-execution subagents.

- Pros: cheap spawn (~10 ms vs ~50–200 ms); EventBus shared
  naturally; matches today's in-process subagent semantics.
- Cons: subagent gets parent's profile permissions (broader than
  necessary for some workloads).

**Option B: Default to new runner; opt-in `shared: true`**.

- Pros: stricter isolation; sub-profile generation per subagent.
- Cons: cascade-of-runners RAM + spawn cost; EventBus replication
  needed.

**Recommend A.**  Parent §4.3 chose this; Phase 3 implements the
choice.

### 5.5 Disk-restore runner respawn — eager vs lazy

**Option A: Eager respawn at restore time** (this plan's
recommendation).  When `_load_session_impl` restores a session that
had AppArmor opt-in, immediately spawn a runner.

- Pros: session is fully ready when control returns to the caller;
  tool calls work without an additional async step.
- Cons: daemon startup with N sessions on disk → N runner spawns
  at boot; meaningful latency for crash-recovery.

**Option B: Lazy respawn on first tool call**.

- Pros: zero startup cost; runners only exist for actually-used
  sessions.
- Cons: first tool call adds ~50–200 ms latency; the runner-spawn
  failure mode now surfaces at tool-call time instead of restore
  time, harder to diagnose.

**Recommend A** with a daemon-level `eager_runner_respawn: true`
config (default true) so operators with high session counts can
flip to lazy if startup latency bites.

### 5.6 Bootstrap unification — one helper or four code paths

The four bootstrap paths (`_create_session_impl`,
`_load_session_impl`, `run_ephemeral_session`, WS standalone) all
need substantially similar runner-spawn wiring.

**Option A: Keep four code paths** (this plan's recommendation).
Each path calls a shared `_maybe_spawn_runner(server, session_id,
workspace_path, sandbox_mode, parent_runner_handle)` helper but
the call site stays inline.

- Pros: each bootstrap path's special-case handling stays visible
  at the call site; minimal refactor.
- Cons: the helper signature grows over time as new bootstrap
  paths add new constraints.

**Option B: Unify through one bootstrap helper**.  All four paths
funnel through `SessionManager._bootstrap_session(envelope)` where
envelope captures the path-specific bits.

- Pros: one place to audit; future paths add via envelope flags.
- Cons: significant refactor across call sites that today have
  distinct shapes (ephemeral session vs disk-restore have very
  different lifecycle expectations).

**Recommend A.**  Save the unification refactor for Phase 6 if
the helper signature actually does grow unmanageably.

---

## 6. Risk register

**6.1 Plugin discovery contract drift.** The runner's plugin
discovery (§3.3 plugin_loader.py) must match the daemon's tier
classification table (parent §4.2).  If the runner loads a plugin
the daemon also instantiates, there are two instances and the
session has a split state.  Mitigation: a single source-of-truth
annotation on each plugin (`PLUGIN_TIER = "daemon" | "runner"`)
that the loader checks; daemon's `PluginRegistry.discover` filters
to daemon-tier, runner's `plugin_loader` filters to runner-tier;
unit test that asserts the two filters partition the set with no
overlap.

**6.2 RPC fan-out overhead.** With N runner-tier plugins each
potentially making runner→daemon RPCs (telemetry, permission ASK,
fragment loads), the per-session RPC volume could grow.  Mitigation:
the dominant case is static-rule permission decisions (zero RPC per
parent §4.5).  Telemetry batches at the envelope level (one RPC
per tool call carrying all sub-spans).  Fragment loads are rare
(only on `selectReferences` admitting a new path).  Measure on
the §4 acceptance gate; fail the gate if RPC overhead exceeds 5%
of tool-call wall time on the smoke profile.

**6.3 Subagent permission state leak.** Per §3.7 + §3.11, multiple
JaatoSessions in one runner share the runner process but must have
isolated permission state keyed by session-id.  A bug in the keying
could leak ASK responses across siblings.  Mitigation: the
permission state class is constructed-per-session at subagent spawn,
NOT lazily inserted into a runner-global dict on first access (the
former crashes loudly on missing key; the latter silently inserts
and leaks).  Add an integration test exercising A→ASK-response→ALLOW
followed by B→same-tool→verify B still gets ASK.

**6.4 Disk-restore + ASK without client.** Per §3.12: a session
restored from disk has no IPC client attached at restore time.  If
the model invokes a tool that triggers ASK before the next
interactive client connects, the runner's
`client.prompt_operator` RPC has no client to relay to.  Mitigation:
the daemon-side handler returns `decision="defer"` with a queued
prompt; the runner's permission plugin treats `defer` as
"hold the tool call" (existing pattern from session-suspended
logic).  When a client attaches, the daemon flushes queued prompts.
Document this in the Session record.

**6.5 WS-side parallel migration.** The WS bootstrap path migration
(§3.12) intersects with WS auth + WS interceptor surfaces.  Need to
verify that none of those rely on a daemon-side `JaatoSession` that
no longer exists.  Mitigation: read all `_ws_*` hook registrations
in the codebase before §3.12 lands; flag any that touch
JaatoSession directly.

**6.6 Phase 2 cli stub becomes redundant.** Phase 2 kept the
daemon-side cli plugin instantiated as a forwarder (per phase2
plan §5.4).  Once §3.4 lands and every other plugin uses the same
forwarder pattern, the daemon-side cli stub IS the same shape as
those — at which point Phase 6 can collapse the stub-pair into
a single forwarding-only daemon-side surface.  Not a Phase 3
concern; flagging here so Phase 6 knows.

**6.7 Premium reactor cross-runner subscription.** The reactor
framework (jaato-premium, daemon-tier) subscribes to per-runner
events via the EventBus.  The bus crosses the RPC boundary already
(events emitted from runner are forwarded to the daemon's bus).
Phase 3 doesn't change this contract.  Verify on the §4 acceptance
gate.

---

## 7. Out-of-scope (Phases 4-6)

This list is informational; not a Phase 3 task list.

- **Phase 4** (lightweight): Performance pass.  Measure runner
  spawn latency, RPC overhead per plugin, RAM overhead per
  runner.  Tune.
- **Phase 5**: Production hardening.  Idle-runner-shutdown knob.
  Per-operator policy store (cross-session permission rules).
  Capability-based fragment loading (runner declares the *kind*
  of path it needs, daemon decides which fragment to load).
- **Phase 6**: Cleanup.  Daemon AppArmor profile (re-confine the
  daemon to a daemon-only profile per §4.7).  Remove
  `apparmor_confine` thread-context machinery (no thread is ever
  confined daemon-side).  Remove `SafeThreadPoolExecutor`'s
  pre-task AppArmor hook (already deleted in Phase 2 §2.1).
  Remove `set_apparmor_context` / `_apparmor_context` plumbing
  in `shared/ai_tool_runner.py`.  Collapse the daemon-side stub
  pair into a single forwarding surface (per §6.6).

---

End of plan. Estimated PR size: ~3500–4500 lines added (ignoring
moves), ~2500 lines removed (daemon-side bodies that move to
runner).  Larger than Phase 2's ~5700-line PR (which counted moves
as adds + deletes) measured by *new logic*, smaller than Phase 2
when measured by line-count delta.  Estimated calendar: 4–6 weeks
of focused work; the bulk plugin migrations are repetitive and
amenable to parallelization across reviewers.
