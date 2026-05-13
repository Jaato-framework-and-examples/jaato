# Shape 3 — Workspace-State Relocation (Daemon → Runner)

**Status:** Design — multi-PR project; PR 1 specified, PRs 2-4 outlined.
**Origin:** 2026-05-13 design conversation on `jaato --new-session` regression — user articulated the principle that *per-workspace provided attributes should be resolved by the per-workspace process*. Initial draft (commit `98b2f0f7`) framed this as "client reads workspace files and sends over wire" — that framing was wrong about the topology (see §0). This revision frames it correctly: **the runner subprocess is the per-workspace process**; workspace file reading moves from daemon to runner. No wire-format changes are needed.
**Forcing function:** profile-less `jaato --new-session` regression introduced by `6406fe35` (Phase 3 §7c step 1, 2026-05-09). Stage 3.1 (5-LoC runner-side `os.environ` fallback) was rejected as a halfway house that preserves the architectural smell — daemon-side `dotenv_values(workspace/.env)` still runs.

## 0. Correction from the initial draft (2026-05-13)

The initial doc (PR #83, commit `98b2f0f7`) framed Shape 3 as "client sends workspace `.env` over the wire; daemon stops reading workspace files." That framing was based on a wrong mental model of WS-mode topology — that WS clients are co-located with workspaces and would need to ship workspace contents over the wire.

The actual topology:

| Mode | Client | Daemon | Workspace | Runner |
|---|---|---|---|---|
| IPC | Local TUI, same machine as daemon | Local | Filesystem path the client provided | Forked by daemon, same host |
| WS | Remote UI (browser, etc.) | Server host | Provisioned by daemon's `WorkspaceProvisioner` on daemon's filesystem | Forked by daemon, same host |

In BOTH modes, the **runner is on the daemon's host and has FS access to the workspace**. The client is purely a UI layer that selects which workspace via `working_dir` / `workspace_path` — a path string, never workspace contents. Memory `feedback_daemon_is_workspace_agnostic` is consistent: "client provides workspace_path".

So the runner can ALWAYS read workspace files directly. No wire-format work is needed for Shape 3 today. A future "daemon-remote-from-workspace" deployment would require client-to-runner wire-format for workspace contents, but that architecture doesn't exist in jaato today and isn't on the roadmap; defer it to its own design conversation when it becomes a real requirement.

## 1. Principle

> *Per-workspace state belongs to the per-workspace process — the runner.*

The daemon is a single long-running process that spawns N runner subprocesses, one per session. Workspace-tied state — `.env` values, profile YAML contents, auth tokens, secrets via `pass://` — belongs to the runner where the workspace work happens, not to the daemon orchestrator.

## 2. What's wrong today

`JaatoServer._resolve_session_env()` at `jaato-server/server/core.py:893` calls `dotenv_values(self.env_file)` against a path the client passed in. The daemon literally reads the client-supplied workspace `.env` from its own filesystem. Similar daemon-side workspace-file reading happens for:

- `<workspace>/.jaato/profiles/*.yaml` — daemon parses profile via `shared/plugins/subagent/config.py:build_inline_profile` to construct the `SubagentProfile`
- `<workspace>/.jaato/auth/*_auth.json` — daemon reads OAuth tokens for provider auth verification
- `<workspace>/.jaato/agents/*.md` — daemon reads agent markdown for system-instructions assembly
- Profile.env `pass://` URI resolution — daemon's `_resolve_secret_uri` substitutes literal values

All this happens on the daemon side, with results threaded to the runner via the bootstrap envelope. That's the architecturally backwards arrangement: the daemon is doing per-workspace work that belongs to the per-workspace process.

**Symptoms this design closes (beyond the architectural smell):**

- Profile-less `jaato --new-session` regression: daemon resolves MODEL_NAME from workspace `.env` into `server._model_name`, but `build_session_envelope` populates `envelope.model_name` only from `profile.model` (with no env-var fallback). Fixing this daemon-side adds a workspace-coupled fallback to a function that shouldn't be workspace-aware; fixing it runner-side puts the resolution where it naturally belongs.
- Phase 4 backlog `project_backlog_env_propagation_seat_flip_gap`: "env propagation across seat-flip is broken — daemon-side resolution works but resolved values don't reach runner". Closed by moving the resolution to the runner.
- Daemon-side `pass://` resolver requires the daemon to have access to the user's GPG agent. Works today because daemon and user share a machine; structurally fragile.

## 3. Target state

| Today | Shape 3 target |
|---|---|
| Daemon calls `dotenv_values(workspace/.env)` | Runner calls `dotenv_values(workspace/.env)` in its own subprocess |
| Daemon parses workspace profile YAML | Runner parses workspace profile YAML |
| Daemon reads workspace auth files | Runner reads workspace auth files |
| Daemon resolves `pass://` URIs via local `pass` binary | Runner resolves `pass://` URIs via local `pass` binary (runner has GPG access via same machine) |
| `JaatoServer._session_env` is authoritative | `JaatoServer._session_env` is removed; runner-side `JaatoSession` reads its own env |
| Envelope carries `model_name`, `provider_name`, `system_instructions`, etc. (profile-derived) | Envelope carries `profile_name`; runner reads profile and derives the rest |
| Daemon-side `_resolve_secret_uri`, `_with_session_env`, `dotenv_values` | All removed |

The daemon's role narrows to: routing, session lifecycle bookkeeping, AppArmor/cgroup provisioning, runner spawn. Workspace files are read exclusively by the per-workspace process.

## 4. Stages (PR plan)

Four PRs, each independently reviewable and rollback-safe. Each preserves back-compat during the migration via daemon-side fallback paths that get removed in PR 4.

### PR 1 — Runner reads workspace `.env`

**Scope:**

- Runner-side `bootstrap_session` (in `server/runner/session.py`) reads `<workspace_path>/.env` via `dotenv_values()` BEFORE constructing the session
- Resolves `pass://` URIs runner-side
- Populates the runner subprocess's `os.environ` with the resolved values
- Populates a new `JaatoSession._session_env` (mirrors the daemon's current attribute) so plugin code can call `session.get_session_env(...)`
- Daemon-side `JaatoServer._resolve_session_env()` no longer reads `dotenv_values(workspace/.env)`; that file path is removed from the call (daemon STILL resolves `env_overrides` from the envelope, which currently carries `profile.env` values daemon-side)
- Closes the profile-less `jaato --new-session` regression by having the runner read `MODEL_NAME` / `JAATO_PROVIDER` from its own resolved env

**Touchpoints:**
- `jaato-server/server/runner/session.py:bootstrap_session` — add env-file reading before `_build_session`
- `jaato-server/shared/jaato_session.py` — add `_session_env` instance dict + `get_session_env(key)` accessor
- `jaato-server/server/core.py:_resolve_session_env` — remove `dotenv_values(workspace/.env)` call; daemon keeps reading only what it still needs (provider config for auth-pending paths, etc.)
- `jaato-server/server/runner_spawn.py:build_session_envelope` — read fallback model_name from envelope's `env_overrides` when `profile.model` is empty

**Effort:** ~150 LoC + tests. ~half day.

**Closes:** profile-less `jaato --new-session` regression. Cascade and other flows continue working (back-compat — daemon's profile-derived path still populates envelope fields).

**No wire-format change.** SDK and TUI untouched.

### PR 2 — Runner reads workspace profile YAML

**Scope:**

- Runner-side `bootstrap_session` parses `<workspace_path>/.jaato/profiles/<profile_name>.yaml` itself via `shared/plugins/subagent/config.py:_load_profile` (already exists; just relocate the call site)
- Runner-side derives `model_name`, `provider_name`, `plugins`, `plugin_configs`, `system_instructions`, `gc`, `completion_payload_schema`, `runtime_limits`, `env` from the parsed profile
- Daemon-side stops parsing profiles for runner-bound sessions; envelope sheds the profile-derived fields (or marks them deprecated for transition)
- Envelope carries `profile_name` (string) instead of `profile` (dict) for runner-bound sessions

**Touchpoints:**
- `jaato-server/server/runner/session.py:bootstrap_session` — add profile-parsing call
- `jaato-server/shared/session_envelope.py` — `SessionInitEnvelope` adds `profile_name` field; existing profile-derived fields marked deprecated
- `jaato-server/server/session_manager.py` — stop parsing profile for envelope construction; just pass `profile_name`
- `jaato-server/server/runner_spawn.py:build_session_envelope` — read profile from runner only

**Effort:** ~250 LoC + tests. ~1-2 days. Larger than PR 1 because profile parsing is non-trivial (inheritance, validation, plugin_configs merging).

**Risk:** profile resolution has subtle behaviors (inheritance chains, env-var expansion, plugin_configs precedence) that daemon-side has gotten right; runner-side needs the same. Mostly a matter of relocating function calls, not rewriting logic.

**No wire-format change for client-daemon.** Envelope schema evolves (daemon-internal).

### PR 3 — Runner reads workspace auth files

**Scope:**

- Auth plugins are already `PLUGIN_TIER="runner"` (per memory + audit) — they instantiate runner-side
- This PR makes the credential FILE READING also runner-side
- `<workspace>/.jaato/auth/*_auth.json` reads move from daemon-side `verify_auth` / provider construction to the runner-side auth-plugin instances
- `pass://` resolution for auth credentials (today daemon-side) moves to runner — runner's `pass` access still works because runner is on the user's machine

**Touchpoints:**
- `jaato-server/server/core.py:verify_auth` — RPC to runner instead of daemon-side credential read
- `jaato-server/shared/jaato_runtime.py:verify_auth` — receives RPC, executes runner-side
- Per-provider auth plugins — already runner-tier; audit each for any remaining daemon-side credential FILE READING

**Effort:** ~200 LoC + tests. ~1 day. Most of the auth lifecycle is already runner-side; this just closes the file-reading gap.

### PR 4 — Daemon-side cleanup

**Scope:**

After PRs 1-3, daemon-side `_session_env`, `_resolve_session_env`, `_with_session_env`, `dotenv_values(env_file)` are dead code paths. Remove them.

- `JaatoServer._session_env` attribute removed
- `JaatoServer._resolve_session_env()` removed
- `JaatoServer._with_session_env()` removed
- Daemon-side `_resolve_secret_uri` (`shared/plugins/subagent/config.py`) is no longer called from daemon; keep for runner-side profile parsing (per PR 2)
- Update `feedback_daemon_is_workspace_agnostic` memory — daemon now truly is workspace-agnostic for file reads

**Touchpoints:**
- `jaato-server/server/core.py` — strip workspace-file-reading methods
- `jaato-server/server/session_manager.py` — strip callers
- Tests — remove tests pinning daemon-side workspace reading

**Effort:** ~150 LoC removed + tests. ~half day. Net negative LoC.

## 5. Back-compat + rollback story

Each PR ships back-compat shims so the system stays bisectable:

- **PR 1**: daemon-side `_resolve_session_env` keeps reading profile.env values (just not workspace .env); runner reads workspace .env and merges with daemon-derived env_overrides
- **PR 2**: envelope's profile-derived fields stay populated daemon-side for one release after runner-side profile reading lands; deprecation log on the daemon-side parse
- **PR 3**: daemon-side auth-file reading kept as fallback for one release
- **PR 4**: removal — once telemetry confirms no consumer hits the daemon-side fallbacks for ≥ one week

The cold-path daemon-side reading remains functional throughout the migration. Each PR independently rollbackable via `git revert` + daemon restart.

## 6. Non-goals

- **Daemon-remote-from-workspace deployment** (where workspace lives on a different host from the daemon-runner pair) — out of scope. That architecture would require wire-format work for shipping workspace contents over the network; defer to its own design conversation when it becomes a real requirement.
- **Profile-inheritance + schema-validation logic itself** — stays as today; just the call site moves from daemon to runner
- **The Phase 4 `env_propagation_seat_flip_gap` backlog item** — closed by PR 1 (runner reads env directly; the "propagation" path becomes a no-op)
- **WS-client-protocol changes** — none needed

## 7. Memory references

- `feedback_daemon_is_workspace_agnostic` — the principle stated; this design implements it operationally
- `project_backlog_env_propagation_seat_flip_gap` — Phase 4 backlog item closed by PR 1
- `feedback_session_env_from_workspace_dotenv_only` — confirms per-session env is workspace-tied; PR 1 cleans up the location of the resolution
- `project_env_secret_uri_resolution` — server 0.6.64+ daemon-side `pass://` resolution; PR 3 moves the workspace auth path runner-side
- `feedback_never_substitute_pass_uri_with_literal_in_env` — applies to files on disk; pass:// resolution happening runner-side instead of daemon-side doesn't change this principle
- `feedback_no_jaato_changes_without_authorization` — each PR's framework code edit + daemon restart requires explicit per-PR user authorization
- `feedback_cascade_aware_daemon_restart_coordination` — daemon restarts during this work coordinate with peer cascade runs
- `project_backlog_daemon_as_pure_factory` — Shape 3 is one step toward that end state
- `docs/design/runner_prewarm_pool_plan.md` — pool work is complementary; pool slots read their own env via the PR 1 mechanism

## 8. Acceptance gates

Each PR lands behind:
- Differential test sweep: zero new failures vs main (same pattern used for Phase 5 §5.x review cycles)
- New regression tests pinning the new wire/state at the migration point
- Manual verification: profile-less `jaato --new-session` works (PR 1+), and existing profile-driven flows unchanged (every PR)
- Real-host gate: cascade smoke runs unchanged before merge (coordinate with peer 7:2)

## 9. Open questions

1. **Envelope schema versioning** — PRs 2 + 3 evolve the envelope (sheds profile-derived fields). Use a `schema_version` bump to fail-fast on runner-daemon version mismatch?
2. **Pool composition** — pre-warm pool slots (per `runner_prewarm_pool_plan.md`) inherit template state. When pool work lands, slots read their own workspace `.env` via the PR 1 path. Confirm: PR 1's `bootstrap_session` reads env at envelope-handling time (after slot is assigned to a session), not at template-fork time. This is the natural code position; just calling it out explicitly.
3. **Profile inheritance** — daemon-side resolution walks inheritance chains and merges. Runner-side needs the same. Verify `build_inline_profile` is import-clean from the runner (it should be — it's `shared/plugins/subagent/config.py`).
4. **`workspace_path` is None edge case** — some headless / sub-spawn paths have `workspace_path=None`. Runner skips workspace file reading in that case; falls through to envelope-carried values (PR 1 keeps the fallback for this path).

## 10. Estimated effort

| PR | LoC | Tests | Risk |
|---|---|---|---|
| PR 1 (runner reads .env) | ~150 | ~30 | LOW — additive runner-side; daemon-side change is a single deletion |
| PR 2 (runner reads profile YAML) | ~250 | ~50 | MEDIUM — touches profile resolution semantics |
| PR 3 (runner reads auth files) | ~200 | ~40 | MEDIUM — auth flow has post-auth-wizard interactions |
| PR 4 (daemon-side cleanup) | -150 (net negative) | -20 (tests removed) | LOW |
| **Total** | **~600 net** | **~120** | MEDIUM |

PR 1 alone closes the profile-less `jaato --new-session` regression and the env-propagation Phase 4 backlog item. PRs 2-4 finish the architectural cleanup.

## 11. Decision log

- **2026-05-13 (initial draft, PR #83 commit `98b2f0f7`)** — framed Shape 3 as "client reads workspace files and sends over wire". Premise was based on wrong WS-mode topology assumption.
- **2026-05-13 (this revision)** — user corrected: in both IPC and WS modes the runner is co-located with the workspace; no wire-format work is needed. Rewrote with runner-as-per-workspace-process framing. Original doc preserved in git history at commit `98b2f0f7` for the wire-format work that would only matter in a hypothetical daemon-remote-from-workspace deployment.
- **2026-05-13** — Stage 3.1 (5-LoC runner-side `os.environ` fallback) considered and rejected. Reason: preserves the daemon-reads-workspace-files smell; doesn't address the architectural shape.
