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

Phase 2 shipped two RPC modules with confusing names:

- `server/runner_rpc.py` (daemon-side) hosts `RunnerRPCClient` — the
  daemon's *outgoing-request* client to the runner.
- `server/runner/rpc.py` (runner-side) hosts `RunnerRPC` — the
  runner's bidirectional dispatcher (already does both
  request-handling and outgoing requests; the name was hedged for
  Phase 3).

Phase 3 needs a third surface: a **daemon-side request *handler***
for runner→daemon RPCs.  To avoid further name confusion, Phase 3
renames the existing daemon-side module up-front:

```
jaato-server/server/
├── runner_rpc_client.py    # RENAMED from runner_rpc.py — hosts
│                           # RunnerRPCClient (daemon → runner outgoing).
├── runner_rpc_server.py    # NEW — hosts RunnerRPCServer, the
│                           # daemon-side dispatch table for
│                           # runner → daemon incoming RPCs.
│                           # Wires the runner-side socket fd into
│                           # an asyncio reader; dispatches frames to
│                           # registered handlers.
└── runner_rpc_handlers/    # NEW — handlers registered into
    ├── __init__.py         #   RunnerRPCServer at session-init.
    ├── prompt_operator.py  #   Each handler is a small async fn
    ├── apparmor_fragment.py#   that imports the daemon-tier API it
    └── telemetry.py        #   relays.
```

The rename of `runner_rpc.py` → `runner_rpc_client.py` is a
mechanical move + import-path update across `server/__main__.py`,
`server/core.py`, `server/session_manager.py`, and the
`shared/plugins/cli/plugin.py` stub — all sites that imported
`server.runner_rpc.RunnerRPCClient` switch to
`server.runner_rpc_client.RunnerRPCClient`.  Lands as the first
commit in §3.2 (no behavior change).

`runner_rpc_handlers/` keeps the dispatch table tight without
ballooning `runner_rpc_server.py`.  Each handler imports the
daemon-tier API it relays (§3.2.1's `prompt_operator` re-uses the
existing `PermissionRequestedEvent` / `PermissionResponseRequest`
event pair via `JaatoServer.emit()`; §3.2.2's `apparmor_fragment`
calls `AppArmorManager.add_reference_fragment` directly).

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

Tests: `runner/tests/test_sanitize.py` —
- workspace path replaced;
- home-jaato path replaced;
- no false positives on ordinary words containing "/home" as a
  substring (the `(?<!\w)` lookbehind is load-bearing);
- **nested-path ordering** (per peer-review Mn1): construct a
  workspace_root that is itself a child of `~/.jaato/` (e.g.
  `~/.jaato/sandbox/workspace-A/`); assert the workspace-pass
  replacement runs first so the result is `<WORKSPACE>/...`,
  not `<HOME>/.jaato/sandbox/workspace-A/...`.  This pins the
  `workspace-first, home-jaato-second` ordering against
  refactor regressions.

One commit.  No interaction with other work.

### 3.2 — Daemon-tier RPC primitives

Two thin handlers registered into the runner-RPC server.  Both are
generic capabilities the runner consumes; neither is plugin-specific.
Per parent §4.5 (permission) and §4.7 (references).

#### 3.2.1 — `client.prompt_operator(prompt: PromptPayload) → PromptResponse`

Today's permission plugin owns its own `Channel` (the abstract base
in `shared/plugins/permission/channels.py:316`); the plugin instance
exposes `_get_channel()` at line 90 returning the configured channel
(`ConsoleChannel` / `QueueChannel` / `WebhookChannel` / `FileChannel`
/ `ParentBridgedChannel`).  When the plugin migrates runner-side
(§3.7), the channel moves with it — the runner is where ASK
decisions wait.

The daemon-side `prompt_operator` handler is therefore NOT a
channel; it's an event-relay surface.  It re-uses the existing
`PermissionRequestedEvent` / `PermissionResponseRequest` pair via
`JaatoServer.emit()` (the same event surface the IPC + WS clients
already render today):

1. Runner-side permission plugin's ASK path constructs a
   `PromptPayload` (tool-name, args summary, session-id, prompt
   text).
2. Runner-side `rpc_client.prompt_operator(payload)` sends the RPC.
3. Daemon-side `prompt_operator` handler emits a
   `PermissionRequestedEvent` (carrying a fresh request-id) on
   `JaatoServer` and awaits the matching `PermissionResponseRequest`
   on the daemon's existing `PermissionResponseChannel`-style
   future-keyed-by-request-id pattern.
4. Connected client (TUI / WS / IPC) receives the event, surfaces
   the prompt, sends back `PermissionResponseRequest`.
5. Daemon-side handler resolves the future, returns the
   `PromptResponse` over the RPC.
6. Runner-side plugin records the response in its local permission
   state (session_whitelist, turn_suspended, etc.) per §3.7.

Files touched:
- `runner_rpc_handlers/prompt_operator.py` — async handler doing
  steps 3+5 above.  Stores in-flight requests in a request-id-keyed
  futures dict; resolves on the matching `PermissionResponseRequest`
  message arrival.
- `runner/rpc_client.py` — runner-side wrapper.  Single async method
  `prompt_operator(payload) → response`.
- `shared/plugins/permission/types.py` — `PromptPayload` /
  `PromptResponse` dataclasses (re-exports of the existing event
  schema).
- `runner_rpc_server.py` — registers the handler at session-init.

The primitive intentionally takes a `PromptPayload` (string +
options + session-id + tool-name) rather than permission-specific
args.  Future plugins (interactive consent, UI confirmation, etc.)
reuse the same primitive without adding a sibling RPC.

Tests: `tests/test_runner_rpc_handlers.py:test_prompt_operator_*` —
round-trip with a stub event subscriber; cancellation propagation
(client disconnect mid-prompt); multiple concurrent prompts (per
session); request-id correlation under concurrent ASKs.

#### 3.2.2 — `apparmor.add_reference_fragment(fragment: AppArmorFragment) → FragmentLoadResult`

The daemon-side method is `AppArmorManager.add_reference_fragment`
(at `server/apparmor.py:1038`).  The RPC method-name matches —
the runner asks for "add a reference fragment", the daemon delivers
it.  No wrapper-rename in this round; the name is already correct.

Files touched:
- `runner_rpc_handlers/apparmor_fragment.py` — async handler, takes
  an `AppArmorFragment` (a typed envelope wrapping the existing
  `AppArmorManager.add_reference_fragment` args), validates the path
  against the per-session allow-list (the existing
  `_validate_path_for_fragment` at `apparmor.py:937`), calls
  `add_reference_fragment`, returns the result.
- `runner/rpc_client.py` — wrapper.
- Validation lives daemon-side (the LLM-driven runner cannot be
  trusted to validate paths it asks to load).
- `runner_rpc_server.py` — handler registration.

Tests: `tests/test_runner_rpc_handlers.py:test_apparmor_fragment_*` —
valid fragment loads, invalid path rejects, fragment-already-loaded
no-op, concurrent fragment loads serialize at the daemon.

One commit per primitive.  Order doesn't matter between the two.

### 3.3 — JaatoSession runner-side scaffolding (multi-commit)

This is the load-bearing refactor.  Per peer-review M1, what was
originally listed as a single commit is realistically three
reviewable commits with intra-task ordering.  Each sub-task ships
independently with its own tests; sub-tasks must land in
declared order.

#### 3.3a — Envelope schema + roundtrip tests

Files touched:
- `shared/jaato_session.py` — gain `to_runner_envelope()` /
  `from_runner_envelope()` for the daemon→runner handshake.  No
  semantic change; this is pure serialization.
- New `shared/session_envelope.py` — `SessionInitEnvelope` typed
  schema: session_id, workspace_path, profile name, plugin list
  (already resolved by daemon's profile loader), plugin configs,
  system instructions.  Defined once, imported on both sides.
- `runner/envelope.py` — re-exports `SessionInitEnvelope` so the
  runner imports from one path.

Tests: `tests/test_session_envelope.py` — round-trip identity
(envelope serializes + deserializes byte-identically), versioning
field present (for Phase 4+ field additions), oversize handling
(reuses §2.4 framing constraint).

One commit.  Lands first; nothing depends on it yet.

#### 3.3b — Runner-side session host

Files touched:
- New `runner/session.py` — hosts the live `JaatoSession` instance
  on the runner side.  Receives a `SessionInitEnvelope` over RPC
  at runner startup, constructs the `JaatoSession`, runs
  `configure()`, signals readiness via `RunnerReadyEvent`.
- `runner/__main__.py` — receives the session-init envelope as the
  first frame after RunnerReadyEvent, hands it to
  `runner.session.bootstrap_session(envelope)`.

The daemon side still instantiates `JaatoSession` in-process at
this point; the runner-side host coexists with it under a feature
flag (`JAATO_RUNNER_HOSTS_SESSION=true`) for the duration of the
3.3b → 3.3c window.

Tests: `runner/tests/test_runner_session_host.py` — bootstrap
from envelope succeeds; `JaatoSession.configure()` runs with the
expected plugin set; runner crash during configure surfaces as
`SessionFailedEvent` to the daemon.

One commit.  Depends on 3.3a's envelope landing.

#### 3.3c — Daemon shell rewrite

Files touched:
- `shared/jaato_runtime.py` — daemon retains `JaatoRuntime`
  (provider config, OAuth state, ledger).  `create_session` no
  longer instantiates `JaatoSession` in-process; it builds the
  envelope, dispatches to the runner, stores the runner-RPC handle.
- `server/core.py` — `JaatoServer` shell.  Holds: provider config
  (from runtime), runner-RPC handle, daemon-tier plugin instances
  (cache, gc, streaming, telemetry, formatter pipeline).  Loses:
  in-process `JaatoSession` reference, the in-process
  `ToolExecutor` (already gone in Phase 2 for IPC-path).
- Removes `JAATO_RUNNER_HOSTS_SESSION` feature flag.

Tests: `tests/test_jaato_session_runner_handoff.py` — envelope
round-trip + full session lifecycle (init / configure /
send_message / shutdown) entirely runner-side; cancellation;
runner crash during send_message surfaces as `SessionFailedEvent`.

Once §3.3c lands, the runner hosts JaatoSession; daemon retains
only the shell.  Plugin migrations §3.4–§3.10 proceed in parallel
from this point.

One commit.  Depends on 3.3b.

### 3.3.5 — `PLUGIN_TIER` annotation + tier-filtered discovery

Per peer-review C5.  Every plugin's `__init__.py` already declares
`PLUGIN_KIND = "tool" | "enrichment"` (verified at
`shared/plugins/cli/__init__.py:10`,
`shared/plugins/file_edit/__init__.py:18`, etc.).  Phase 3 adds a
sibling annotation `PLUGIN_TIER = "daemon" | "runner"` so the
loader filters can partition the plugin set.

Files touched:
- `shared/plugins/<every-plugin>/__init__.py` — add
  `PLUGIN_TIER = "daemon"` or `PLUGIN_TIER = "runner"` per the
  parent design §4.2 table.
- `shared/plugins/registry.py:PluginRegistry.discover` — gains a
  `tier_filter: Optional[Literal["daemon", "runner"]] = None`
  parameter.  When set, the discovery walk skips plugins whose
  `PLUGIN_TIER` doesn't match.  Default `None` preserves Phase 2
  behavior (daemon discovers everything).
- `runner/plugin_loader.py` (new — see §2.1) — calls
  `PluginRegistry.discover(tier_filter="runner")`.
- `server/__main__.py` (or wherever `PluginRegistry.discover` is
  called daemon-side) — Phase 3 starts passing `tier_filter="daemon"`
  once §3.4 lands and the runner-tier plugins are no longer needed
  in the daemon-side registry for in-process fallback.  Until §3.4,
  daemon-side discovery stays unfiltered (the in-process fallback
  path uses runner-tier plugins).

Tests: `shared/tests/test_plugin_tier_partition.py` — exhaustive
test asserting the union of (daemon-discovered, runner-discovered)
is the full plugin set AND the intersection is empty.  Fails the
build if a new plugin lands without a `PLUGIN_TIER` annotation.

One commit, ordered between §3.3c (daemon shell) and §3.4 (first
plugin-migration wave).  Without the annotation, §3.4 has no
deterministic answer to "which plugins should the runner load?".

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

**Cgroup attach + runtime limits plumbing** (per peer-review M2):
Phase 2's cli stub left the `preexec_fn=None` path in
`cli_runner.py` flagged as Phase 3 work.  The cgroup attach
machinery currently lives in `ToolExecutor.set_runtime_limits` and
is invoked daemon-side before the subprocess is spawned.  Wave 2's
migration must move the cgroup attach to the runner side too —
otherwise the kernel-enforced runtime limits (memory_max_mb,
tool_timeout_seconds, max_output_bytes) silently disappear for the
migrated plugins, and §4 acceptance gate's runtime-limits
verification fails.  The cli plugin gains its cgroup attach in
this wave alongside the four subprocess-spawning plugins (closes
the Phase 2 carryover).  Pair this with §3.15's `_telemetry`
cleanup — both touch the cgroup-deltas measurement path, so they
ship together as a single internal coordination.

Per-plugin acceptance: existing test suite + a multi-tenant
integration test verifying the spawned subprocess takes the
session's profile (read `/proc/<child>/attr/current` from the
runner; assert it matches the session profile name).  The runtime-
limits test extends to verify the limits are actually enforced
(spawn a subprocess that exceeds memory_max_mb; assert SIGKILL).

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

The plugin's `Channel` (the abstract base in
`permission/channels.py:316` with concrete `ConsoleChannel` /
`QueueChannel` / `WebhookChannel` / `FileChannel` /
`ParentBridgedChannel` subclasses) moves with the plugin —
`_get_channel()` runs runner-side; the channel waits on the future
the runner-side request opens; the daemon-side `prompt_operator`
handler resolves that future via the event-relay flow described in
§3.2.1.

Files touched:
- `shared/plugins/permission/plugin.py` — body moves runner-side.
  `check_permission` runs entirely runner-local for static-rule /
  evaluator decisions.  ASK path calls
  `runner.rpc_client.prompt_operator(payload)` which the runner-side
  channel awaits.
- `shared/plugins/permission/channels.py` — the channels also move
  runner-side.  No semantic change; they wait on futures keyed by
  request-id, the daemon resolves the future via §3.2.1's relay.
- Daemon-side stub: minimal — just enough for the model to see
  permission-related tools (`permission_status`, suspend/resume) in
  the schema.  Their bodies forward via RPC.
- **Cross-session policy mutation race** (peer-review M3).  The
  operator's `/permissions whitelist <tool>` command path: daemon
  receives the command, RPCs `permission.add_rule(rule)` to the
  target session's runner.  The runner-side `add_rule` handler MUST
  acquire the same per-session lock the channel ops use today
  (today's plugin holds a per-session lock around channel.send /
  channel.wait_for_response sequences); otherwise a whitelist arrival
  mid-ASK lets the next call for the same tool bypass the prompt
  nondeterministically.  Concretely: every mutation of the per-session
  policy state holds `session._policy_lock` for its critical section;
  ASK-resolution holds the same lock between "rule miss" check and
  the channel wait; the lock is added to the runner-side plugin's
  per-session state record at session-init.
- Per §4.5 v1: per-session only, no daemon-side rule store.

Tests: existing permission test suite lifts and shifts to the
runner tests directory; add cross-runner mutation tests
(`/permissions whitelist X` from operator → session A's runner sees
the new rule; session B's runner unchanged).  Add a race test:
operator-issues-add_rule(X) at the same instant a check_permission(X)
is mid-ASK; assert the in-flight ASK's decision is honored (operator
doesn't pre-empt) AND the next call for X uses the new rule (no
prompt).

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

`subagent` is intentionally NOT in this table — it migrates with
§3.11 (the runner-sharing wiring is the actual content of that
plugin's migration; listing it here would suggest it's a wave-4
shape, which it isn't).

`artifact_tracker._save_state` is the audit task the parent design
flagged at §4.6 line 587.  Fix lands as part of the migration commit
for that plugin; no new helper, just adopt the pattern from
waypoint inline.

### 3.11 — Subagent runner-sharing semantics + `subagent` plugin migration

Per parent §4.3.  Default: subagents share the parent's runner.
Opt-in: `agent_params.isolated: true` (or profile-level
`isolation: "runner"`) spawns a new runner with a sub-profile of
the parent's profile (`jaato-ws-{session_id}//{subagent_id}`).
This task delivers BOTH the `subagent` plugin migration and the
runner-sharing wiring (separating the two would just split a
single shape across two commits).

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

**Subagent state teardown** (per peer-review M4).  When a subagent
finishes — whether normal `signal_completion`, error termination,
or operator cancel — its session-id-keyed state in the shared
runner MUST be removed.  Affected registries (every store that
keys by session-id post-§3.7 / §3.9 / §3.10):

| Registry | Owner | Teardown trigger |
|---|---|---|
| Permission per-session policy + state | runner-side `permission/plugin.py` | `_on_subagent_terminated(session_id)` hook |
| Memory cache (in-process embedding cache) | runner-side `memory/plugin.py` | same hook |
| Telemetry sub-span buffers | runner-side telemetry adapter | same hook |
| Reliability counters | runner-side `reliability/plugin.py` | same hook |
| `_trusted_bridge_context` thread-locals | already thread-local; no extra teardown | n/a |

The hook fires via the runner's `JaatoSession.on_terminated()`
lifecycle event (already exists daemon-side; moves runner-side with
JaatoSession in §3.3).  Each plugin registers its teardown
callback against the hook at plugin configure-time.

Without this, a long-lived parent session accumulates unbounded
permission state from completed subagents — the leak is silent
and only bites under sustained subagent-spawn workloads.

Tests: parallel subagents in shared runner (default); isolated
subagent in fresh runner (opt-in); permission state isolation
between parent and subagent in the shared-runner case;
**state-teardown test**: spawn N subagents serially, complete each,
assert the runner's permission/memory/telemetry registries return
to baseline (no growth in entry count).

### 3.12 — Three deferred bootstrap paths

Closes Phase 2 plan §8.4's deferral list.  All three need the
runner-spawn wiring that Phase 2 only added to
`_create_session_impl`.

| Path | Phase 3 wiring |
|---|---|
| `SessionManager._load_session_impl` (disk-restore) | Re-spawn a runner for sessions loaded from disk that had AppArmor opt-in.  Profile name is preserved on the Session record (`Session.sandbox_mode`, set in Phase 2).  Per peer-review M5: the restored session is **explicitly auto-suspended** until a client reattaches (default behavior, not implicit).  The Session record gains a `restored_pending_attach: bool` flag set true on disk-restore; while set, `check_permission` short-circuits to "deny — session pending reattach" for any tool that would trigger an ASK; static-rule permits still pass through.  When a client attaches, the daemon emits `SessionRestoredEvent(session_id, pending_tool_call_count: int)` and the client surfaces a "this session was restored — review pending tool calls?" prompt; the operator's reattach response clears the flag.  This avoids the silent-stall failure mode an operator-restart-and-walk-away would otherwise create. |
| `SessionManager.run_ephemeral_session` | Ephemeral sessions are subagent fan-out; per §4.3 default they share the parent's runner.  Phase 3 adds the parent-runner reference to the ephemeral path so they actually share rather than spawn fresh.  Isolated-runner ephemeral subagents follow §3.11. |
| `JaatoWSServer` standalone bootstrap | The WS server has its own pre-init apparmor hook (`websocket.py:_apparmor_pre_init_hook`).  Phase 3 converts it from the legacy 3-arg signature to the 4-arg form (matching the IPC hook from Phase 2's §2.3 fix).  Runner spawn wires alongside. |

Each path gets one commit + integration test.

### 3.13 — IPC apparmor hook relocation (pre-init → inline session.new)

Per Phase 2 plan line 225.  The 4-arg pre-init hook from Phase 2
was a transitional step; the design endpoint is for the runner spawn
to live inline in the IPC `session.new` handler, which is already
keyed by client_id at that point.  This collapses one indirection
layer.

The IPC `session.new` handler lives in
`server/command_router.py:_handle_session_new` (line 247), called
from the IPC dispatcher at `command_router.py:161`.  Phase 2's
command-router refactor moved the per-command handlers out of
`server/ipc.py` into `command_router.py`; `server/ipc.py` retains
only the framing + connection management.

Files touched:
- `server/__main__.py` — remove
  `_register_ipc_apparmor_hook` / its `add_pre_initialize_hook`
  registration.  Logic moves into the IPC `session.new` handler
  in `server/command_router.py:_handle_session_new`.
- `server/command_router.py:_handle_session_new` — invoke the
  apparmor provisioning + runner spawn directly, before forwarding
  to `SessionManager.create_session`.
- `server/session_manager.py:_run_pre_initialize_hooks` — the
  4-arg signature stays (the WS-side pre-init hook still registers
  there with its legacy 3-arg form per Phase 2 §8.4 deferral; any
  third-party pre-init hooks also continue using this surface).
  Only the IPC hook stops registering.

Tests: existing Phase 2 tests still pass; one new test verifies
the IPC `session.new` path spawns a runner without going through
`_run_pre_initialize_hooks`.

This task is independent of the bulk plugin migration; can land
early or late.  Listed late in the plan because the pre-init
hook is the safer state to ship migrations against (any plugin
issue surfaces in the existing pre-init path, not in a new code
path).

### 3.14 — Atomic-write contract enforcement (per-plugin, ongoing)

Per parent §4.6 line 587 + peer-review M6.  The handler that sets
a "shutting down" flag plugins check before starting a write is
**NOT** sufficient — the SIGTERM-after-write-began window is what
atomic-write closes.  The original framing of "one-shot audit pass"
underspecifies: any new persistent state file landing during Phase 3
(e.g. `lsp` per-session snapshots, `webhook` subscription state
moving runner-side, future plugin migrations) gets the same
requirement.

Reframe: §3.14 establishes the atomic-write contract, **enforced
per runner-tier plugin shipped in Phase 3**:

1. **Contract.** Any runner-tier plugin that persists state to disk
   under `<workspace>/.jaato/` or `~/.jaato/` MUST use the
   `tempfile.NamedTemporaryFile` + `os.replace` pattern (or a
   `shared/atomic_write.py` helper if one lands as part of this
   task).
2. **Per-plugin enforcement.** Each plugin migration commit (§3.4 –
   §3.10) includes:
   - A grep / lint for `open(..., 'w')` on persistent paths in the
     migrated plugin's source.
   - A SIGTERM-mid-write integration test for that plugin (drives
     a runner mid-write, sends SIGTERM, asserts on-disk state is
     either pre-write or post-write — never partial / corrupted).
3. **Already-known fixes** that land in their respective tasks:
   `artifact_tracker._save_state` (§3.10 — currently
   `open(path, 'w'); json.dump`), `todo` (§3.4 — verify), `lsp`
   (§3.5 — verify if it persists state).
4. **Known-good (no fix needed)**: `waypoint`, `memory`,
   `file_session` (atomic via session_manager pre-resolution).

§3.14's standalone deliverable: a single
`tests/integration/test_runner_sigterm_durability.py` that
parametrizes over every runner-tier plugin that persists state and
runs the SIGTERM-mid-write assertion on each.  The fixture is
shared with the per-plugin migration tests (the parametrized
runner fixture in §4 acceptance gate).  CI gate: any new
runner-tier plugin landing without atomic-write coverage fails this
test.

This converts the audit from a one-shot pass into an enforced
contract.

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
   parametrized over every runner-tier plugin (per-plugin
   fixture, NOT a single mega-test).  Per peer-review Mn3: each
   plugin gets its own assertion case, sharing a runner fixture;
   CI runs them in parallel.  Each case verifies the plugin's
   tool calls round-trip end-to-end (daemon-side stub forwards
   correctly + runner-side body produces the right result
   envelope).  ~20 plugins × ~5s each, parallelized to ≤3 min
   wall-clock.  Bisect-friendly: a regression in plugin X fails
   only the `test_<plugin_x>_*` parametrize cases, not the whole
   gate.
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

**Why this knob earns the feature-flag exception** (per
peer-review Mn4): the Phase 2 plan was deliberate about avoiding
feature flags as feature-gating shims.  This knob is different —
it's an **operator-side cost-regression escape hatch**, not a
feature gate: every behaviorally-correct outcome works under both
settings, the only difference is daemon startup latency vs first-
tool-call latency.  Operators with N sessions on disk where
N · spawn_latency > acceptable boot time get the flip; the default
preserves the simpler topology.  No model-visible behavior
changes; no test matrix bifurcation (the lazy path is exercised
by §3.12's existing test suite for ephemeral sessions, which were
always lazy).

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

**Recommend B** (REVISED per peer-review M7).  At Phase 3 scope
all four bootstrap paths acquire 4–5 special-case-per-path lines
each (apparmor opt-in lookup, pre-init runner spawn, sandbox_mode
plumbing, runner reaping in shutdown, ephemeral parent-runner
attach).  The path-specific bits are smaller than the shared bits
once §3.12 + §3.13 land.  Inverting the recommendation to a
unified `_bootstrap_session(envelope)` helper avoids 4× duplicated
wiring change in the Phase 3 PR.  Phase 6 unification is too late
— the wiring IS the code that lands now.

The original Recommend-A framing is preserved here as a fallback:
if §3.12's three deferred bootstrap paths land cleanly with
distinct call-site shapes (i.e. the path-specific bits turn out
to be larger than expected and the shared bits smaller), pivot
back to keeping four call sites + a shared helper.  The
diff-shape is the deciding evidence.

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
  confined daemon-side).  `SafeThreadPoolExecutor`'s pre-task
  AppArmor hook registration was deleted in Phase 2 §2.1, but the
  helper `_thread_unconfine_safe` itself stayed in
  `server/apparmor.py` ("kept for any third-party reuse" — Phase 2
  caveat).  Phase 6 deletion target includes the helper if no
  third party showed up by then.  Remove `set_apparmor_context` /
  `_apparmor_context` plumbing in `shared/ai_tool_runner.py`.
  Collapse the daemon-side stub pair into a single forwarding
  surface (per §6.6).

---

End of plan.  Estimated calendar: 4–6 weeks of focused work; the
bulk plugin migrations are repetitive and amenable to
parallelization across reviewers.

**PR-size estimate is deferred** until §3.3a/b/c land.  Per
peer-review Mn6, my prior "3500-4500 added / 2500 removed" guess
was anchored on too-loose assumptions: `JaatoSession` alone is
~5000 lines today, and the move-with-facade shape inflates the
delta significantly.  The Phase 2 line-count estimate also drifted
during implementation.  Re-estimate after the JaatoSession move
shape settles; until then, treat PR sizing as "large, multiple
reviewable commits, bulk migrations parametrize over plugins so
they read repetitively."

---

## 8. Revisions

**v2 (2026-05-08)** — addresses peer-review of v1 (commit
94a3a807).  Critical fixes:

- **C1**: §3.13 file path corrected from
  `server/ipc.py:_handle_session_new` to
  `server/command_router.py:_handle_session_new` (verified at
  line 247).  `server/ipc.py` retains framing + connection
  management only; per-command handlers moved out in Phase 2's
  command-router refactor.
- **C2**: §2.2 documents the `runner_rpc.py` →
  `runner_rpc_client.py` rename + the new
  `runner_rpc_server.py` sibling.  Phase 2's existing classes
  (`RunnerRPCClient` daemon-side, `RunnerRPC` runner-side) keep
  their names; the rename is module-level only to make the
  three-surface relationship legible.
- **C3**: §3.2.1 + §3.7 corrected — `Channel` lives on the
  permission plugin (`_get_channel()` at
  `permission/plugin.py:90`), not on `SessionManager`.  Channel
  moves with the plugin into the runner.  Daemon-side
  `prompt_operator` handler is an event-relay surface using
  `PermissionRequestedEvent` / `PermissionResponseRequest`, NOT
  a channel.
- **C4**: §3.2.2 method name corrected to
  `add_reference_fragment` (at `apparmor.py:1038`); the RPC
  method-name aligns; no rename needed.
- **C5**: New §3.3.5 task adds `PLUGIN_TIER` annotation alongside
  the existing `PLUGIN_KIND` constant (verified at
  `cli/__init__.py:10`, `file_edit/__init__.py:18`); updates
  `PluginRegistry.discover` to filter by tier; partition-no-overlap
  unit test fails the build if a new plugin lands without the
  annotation.

Moderate fixes:

- **M1**: §3.3 split into 3.3a (envelope + tests), 3.3b (runner-side
  host under feature flag), 3.3c (daemon shell rewrite + flag
  removal) — three reviewable commits with intra-task ordering.
- **M2**: §3.5 explicitly notes cgroup attach + runtime-limits
  plumbing migration (closing the Phase 2 cli stub carryover);
  paired with §3.15's `_telemetry` cleanup.
- **M3**: §3.7 documents the per-session `_policy_lock` race
  fix for cross-session policy mutation; race test added to
  the suite.
- **M4**: §3.11 adds subagent-state-teardown sub-task with the
  registry table (permission, memory, telemetry, reliability)
  and the `_on_subagent_terminated` hook; teardown integration
  test added.
- **M5**: §3.12 disk-restore path explicit auto-suspend behavior
  with `restored_pending_attach: bool` flag + `SessionRestoredEvent`
  for client-side reattach UX; replaces the implicit-stall
  failure mode.
- **M6**: §3.14 reframed as ongoing per-plugin atomic-write
  contract enforcement (lint + integration test per migration),
  not a one-shot audit pass.
- **M7**: §5.6 flipped to recommend Option B (unified bootstrap
  helper); preserves Option A as fallback if path-specific bits
  turn out larger than shared bits during §3.12 implementation.

Minor fixes:

- **Mn1**: §3.1 sanitize tests will pin the workspace-pass-runs-
  first ordering with a nested-path case.
- **Mn2**: `subagent` plugin moved out of §3.10 wave-4 table
  into §3.11's intro (its migration IS the runner-sharing
  wiring).
- **Mn3**: §4 acceptance test #1 is parametrized per plugin (not
  a single mega-test); CI parallelizes; bisect-friendly.
- **Mn4**: §5.5 documents why the `eager_runner_respawn` knob
  earns the feature-flag exception (operator-side cost-regression
  escape hatch, no model-visible behavior change).
- **Mn5**: §7 Phase 6 cleanup adds `_thread_unconfine_safe`
  helper deletion as a Phase 6 target (kept for now under Phase 2
  caveat).
- **Mn6**: PR-size estimate deferred until §3.3a/b/c lands; prior
  "3500-4500 / 2500" numbers withdrawn pending the JaatoSession
  move shape.

**v1 (2026-05-08)** — initial draft, commit 94a3a807.
