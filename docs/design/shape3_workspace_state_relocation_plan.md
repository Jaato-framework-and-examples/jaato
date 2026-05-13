# Shape 3 — Workspace-State Relocation (Daemon → Runner)

**Status:** PR 1 shipped + reverted; PRs 2-4 design pending revisit after the AppArmor + secret-resolution constraint surfaced (see §0.1).  Specifically:
- PR 1 (runner reads workspace .env) — shipped in PR #91 (server 0.6.77), REVERTED by PR #92 (server 0.6.78).  The runner-side workspace .env reading was load-bearing-broken under AppArmor confinement.  See §0.1 for the constraint + the wire-transit shape that replaces it.
- PR 2 (runner reads profile YAML) — scope needs revisit.  Profile YAML structure parsing can still move; profile.env value resolution stays daemon-side.
- PR 3 (runner reads auth files) — scope needs revisit.  Auth credentials are by definition secrets; reading them runner-side hits the same AppArmor constraint that broke PR 1.
- PR 4 (daemon-side cleanup) — premise (eliminate daemon-side workspace state) is invalidated.  Daemon retains secret-resolution authority indefinitely.

**Origin:** 2026-05-13 design conversation on `jaato --new-session` regression — user articulated the principle that *per-workspace provided attributes should be resolved by the per-workspace process*. Initial draft (commit `98b2f0f7`) framed this as "client reads workspace files and sends over wire" — that framing was wrong about the topology (see §0). This revision frames it correctly: **the runner subprocess is the per-workspace process**; workspace file reading moves from daemon to runner. No wire-format changes are needed.

**Forcing function:** profile-less `jaato --new-session` regression introduced by `6406fe35` (Phase 3 §7c step 1, 2026-05-09). Stage 3.1 (5-LoC runner-side `os.environ` fallback) was rejected as a halfway house that preserves the architectural smell — daemon-side `dotenv_values(workspace/.env)` still runs.

## 0.1 AppArmor + secret-resolution constraint (PR #91 → #92 retrospective)

PR 1 shipped per this plan's design in PR #91 (server 0.6.77) — runner-side `bootstrap_session` read `<workspace>/.env`, ran `expand_variables` (which calls `_resolve_secret_uri`), and applied resolved values to `os.environ`.

Under the AppArmor-confined-runner model (Phase 3 §4.6 et al.), this broke.  `jaato_premium.secret_resolvers.PassResolver.__init__` shells to `pass version` as a sanity check.  The confined runner's AppArmor profile correctly blocks the exec — `pass version` returns exit 126, `_discover_secret_resolvers()` registers zero schemes, `_resolve_secret_uri()` returns the literal URI with a warning, the provider plugin uses the 28-byte literal as a bearer token, and Z.AI rejects with 401.  Surfaced 2026-05-13 via v64 cascade smoke.

**The principle of this plan is invalidated for secret resolution.**  "Per-workspace state belongs to the per-workspace process" works for non-secret workspace state (profile YAML structure, agent markdown, GC config) but collides with confinement for any operation that needs to exec resolver subprocesses or otherwise access the user's password store.  The daemon (unconfined) is the only process with the capability; the runner (confined) must consume pre-resolved values.

**Resolution (PR #92, server 0.6.78):**
- Daemon-side `JaatoServer._resolve_session_env` resumes reading `<workspace>/.env`.
- New `SessionInitEnvelope.session_env: Dict[str, str]` field carries the fully-resolved env (workspace .env + profile.env + env_overrides, all `${VAR}` and secret URIs decoded daemon-side).
- Runner-side `bootstrap_session` applies `envelope.session_env` to `os.environ` verbatim — never runs `_resolve_secret_uri`, never execs `pass`.
- Wire channel: existing `runner_rpc` socketpair (FD-pass, private to daemon-runner pair).
- Trust posture: identical to pre-PR-91 fork-inherit semantics.  Runner sees specific session secrets needed for provider HTTP calls; cannot enumerate the password store.

**Audit performed pre-merge of PR #92** (four exposure surfaces verified clean before adding the new field carrying plaintext secrets):
- **Logging**: no `envelope.to_dict()` log site; step 1b logs key COUNT, never values.
- **Persistence**: envelope is in-memory only; `SessionRecord` / `WaypointRecord` / subagent serializers exclude envelope fields.
- **Client events**: no event class declares `envelope` / `session_env` / `env_overrides` fields; `ProfileSummary.env_var_names` exposes KEYS only with docstring contract "values never leave the daemon".
- **Fork-replay**: every fork path (waypoint, headless reactor, disk-restore) constructs a fresh `BootstrapEnvelope` at entry — fresh daemon-side resolution per fork.

The audit framework is reusable for any future field carrying resolved secrets across the wire.

**General rule for Shape 3 (revised):** per-workspace relocation needs an explicit security-side review for any operation requiring exec or kernel access outside the confined profile's allow-list.  Daemon stays the unconfined authority for those operations.  Workspace state that DOES NOT require such access (file parsing, YAML/JSON load) can still relocate per the original principle.

See `feedback_secret_resolution_stays_daemon_side` memory + PR #92 description for context.

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

### PR 1 — Runner reads workspace `.env` — **SHIPPED PR #91, REVERTED PR #92**

**As-shipped (PR #91, server 0.6.77, merged 2026-05-13):**

- Runner-side `bootstrap_session` reads `<workspace_path>/.env` via `dotenv_values()` BEFORE constructing the session.
- Resolves `pass://` URIs runner-side via `expand_variables` → `_resolve_secret_uri`.
- Populates runner's `os.environ` with resolved values.
- New `JaatoSession._session_env` + `get_session_env(key, default)` accessor.
- Daemon-side `JaatoServer._resolve_session_env` no longer reads `dotenv_values(workspace/.env)`.

**Reverted (PR #92, server 0.6.78, merged 2026-05-13):**

The runner-side secret-URI resolution failed under AppArmor confinement — `pass` exec blocked, resolvers don't register, URIs survive as literals.  See §0.1 above for the full retrospective.  PR #92 reshapes the wiring:

- Daemon resumes workspace .env reading (revert).
- New `SessionInitEnvelope.session_env` field ships fully-resolved env daemon → runner.
- Runner applies the dict to `os.environ` verbatim; no resolver discovery, no `pass` exec.
- `JaatoSession._session_env` + `get_session_env` accessor stay (now populated by the envelope-applied dict).

**Net architectural change after PR 1 + PR 92:**

- ✅ Workspace .env READING stays daemon-side (file lives on daemon-runner host; daemon is unconfined; can exec resolvers).
- ✅ Resolved env REACHES runner via a dedicated wire field (`envelope.session_env`).  Cold-spawn path also relies on fork-inherit `os.environ` overlay (`_with_session_env`) — pool-served path relies solely on the envelope wire transit.  Both produce the same runner-side `os.environ` state.
- ✅ `JaatoSession.get_session_env(key)` accessor available for plugin code that needs per-session env lookup separate from `os.environ`.

**No wire-format break.**  `envelope.session_env` is additive; old runners ignore unknown fields.

### PR 2 — Runner reads workspace profile YAML — **SCOPE PENDING REVISIT (§0.1 constraint)**

Original scope (pre-§0.1): runner parses profile YAML + derives all profile fields runner-side.  Profile.env values resolved via `expand_variables` runner-side.

§0.1 invalidates the resolution part: `profile.env` values containing `pass://` (a common pattern) cannot resolve runner-side under AppArmor confinement.  Same failure mode that took down PR 1.

**Revised scope (split into two layers):**

- **Profile YAML structure parsing**: CAN move runner-side.  Reads the file via `_load_profile`, derives `model_name`, `provider_name`, `plugins`, `plugin_configs`, `system_instructions`, `gc`, `completion_payload_schema`, `runtime_limits`.  No exec required; no secrets in this layer.
- **Profile.env value resolution**: STAYS daemon-side.  Daemon reads `profile.env`, runs `expand_variables`, contributes to `server._session_env`, ships via `envelope.session_env`.  Same path as workspace .env per PR #92.

**Open design questions before specifying this PR:**

1. Does the daemon need profile YAML to be parsed in order to populate the envelope's other fields, OR can profile.env be a separate read (`yaml.safe_load(<profile>.yaml)["env"]`) while the rest of the profile parses runner-side?  Probably yes — daemon needs `model_name` etc. for its own bookkeeping today (transports, telemetry).
2. If daemon STILL parses profile YAML for its own needs, what does "runner reads profile YAML" actually deliver beyond a redundant re-parse?  Two answers: (a) test the runner-side parser pin against daemon-side equivalence; (b) future "daemon as pure factory" target needs runner-side parsing as a prerequisite.
3. Is the `_load_profile` import surface clean from the runner (the audit in §0 / open question 3 of v1 doc).  Probably yes; `shared/plugins/subagent/config.py` is import-clean.

PR 2 specification deferred until these are answered.  Workspace state relocation is no longer a clean reorg — secret-touching layers stay daemon-side.

**Effort estimate (revised):** ~150 LoC + tests for the structural parse relocation; profile.env resolution is unchanged (no work).  ~1 day.

### PR 3 — Runner reads workspace auth files — **SCOPE INVALIDATED (§0.1 constraint)**

Original scope: relocate `<workspace>/.jaato/auth/*_auth.json` reads from daemon to runner, including `pass://` resolution of auth credentials.

§0.1 invalidates this entirely.  Auth credentials are by definition secrets.  Reading them runner-side means resolving `pass://` runner-side means hitting the AppArmor exec block.  Same failure mode as PR 1.

**Revised position:** auth file reading + secret resolution STAY daemon-side.  Auth plugins remain `PLUGIN_TIER="runner"` (their per-session lifecycle is runner-scoped), but the credential MATERIAL is daemon-provided.  Today's mechanism (daemon resolves env, ships via envelope; auth plugins read os.environ in the runner) already implements this correctly — no work needed.

The fundamental misframing of the original PR 3 was assuming "auth lifecycle runner-side ⇒ credential file reading runner-side."  Those are independent.  Plugin lifecycle can be runner-tier while credential resolution stays daemon-side.

**This PR is closed-as-unneeded.**  No work.

### PR 4 — Daemon-side cleanup — **PREMISE INVALIDATED (§0.1 constraint)**

Original premise: after PRs 1-3, daemon-side `_session_env` / `_resolve_session_env` / `_with_session_env` / `dotenv_values(env_file)` would be dead code; PR 4 removes them.

§0.1 + the PR #92 resolution invalidate this entirely.  Daemon retains:

- `_session_env` attribute — holds the fully-resolved env, shipped via `envelope.session_env`.
- `_resolve_session_env` method — reads workspace .env + profile.env + env_overrides, runs `expand_variables` (including secret URIs).
- `_with_session_env` context manager — applies the resolved env to `os.environ` daemon-side for the brief runtime-construct window where daemon-side code (`build_session_envelope` reading PROJECT_ID etc.) needs it.
- `dotenv_values(env_file)` call — load-bearing for workspace .env reading.

**This PR is closed-as-unneeded.**  Daemon-side workspace-file-reading is structural under the confinement model.

Conceivable scope NARROWING (not removal): if `build_session_envelope` stops reading `PROJECT_ID`/`LOCATION` from `os.environ` (separate backlog: replace with provider-specific config objects), the `_with_session_env` os.environ-overlay could narrow to only the actual daemon-side consumers.  Track via that backlog, not here.

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
- **2026-05-13 (revision 2)** — user corrected: in both IPC and WS modes the runner is co-located with the workspace; no wire-format work is needed. Rewrote with runner-as-per-workspace-process framing. Original doc preserved in git history at commit `98b2f0f7` for the wire-format work that would only matter in a hypothetical daemon-remote-from-workspace deployment.
- **2026-05-13** — Stage 3.1 (5-LoC runner-side `os.environ` fallback) considered and rejected. Reason: preserves the daemon-reads-workspace-files smell; doesn't address the architectural shape.
- **2026-05-13 (revision 3, post-PR-#92)** — AppArmor + secret-resolution constraint surfaced via v64 cascade smoke after PR #91 shipped.  The "per-workspace state to the per-workspace process" principle holds for non-secret state but collides with the confined-runner model for any operation requiring exec or kernel access (resolver subprocesses, GPG agent, password store).  PR 1 reverted via PR #92's wire-transit reshape.  PR 2 scope split (YAML parse OK; profile.env resolution stays daemon-side).  PR 3 closed (auth file reading + secret resolution both stay daemon-side).  PR 4 closed (daemon retains workspace-file-reading authority for secret-touching operations).  General rule documented in §0.1.  Memory: `feedback_secret_resolution_stays_daemon_side`.
