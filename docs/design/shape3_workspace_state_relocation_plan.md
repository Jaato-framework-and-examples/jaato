# Shape 3 — Workspace-State Relocation (Daemon → Runner)

**Status:** Design — multi-PR project; PR 1 specified, PRs 2-4 outlined.
**Origin:** 2026-05-13 design conversation on `jaato --new-session` regression — user articulated the principle that *per-workspace provided attributes should be resolved by the (runner-side) session, not the (process-wide) daemon*, then asked whether the daemon "perhaps might not have access to the runner's workspace" — surfacing the deeper architectural concern that all "daemon reads workspace file" code paths break when daemon and workspace are not co-located.
**Forcing function:** profile-less `jaato --new-session` regression introduced by `6406fe35` (Phase 3 §7c step 1, 2026-05-09). Stage 3.1 (5-LoC runner-side `os.environ` fallback) was rejected as a halfway house that preserves the architectural smell.

## 1. Principle

> *Per-workspace state belongs to the per-workspace process.*

The daemon is a single long-running process that orchestrates many sessions, potentially across many workspaces, potentially across many host machines. The runner subprocess is per-session and lives at (or near) the workspace it serves. Workspace-tied state — `.env` values, secrets, model selection, provider config — belongs to the runner-side process, not the daemon.

## 2. What's wrong today

`JaatoServer._resolve_session_env()` at `jaato-server/server/core.py:893` calls `dotenv_values(self.env_file)` against a path the client passed in. The daemon literally reads the client's workspace `.env` from the daemon's own filesystem. This works **only** because the IPC mode happens to co-locate daemon and client today; it breaks the moment daemon and workspace are on different machines (WS-mode-with-remote-workspaces, multi-tenant SaaS, etc.). Memory `feedback_daemon_is_workspace_agnostic` already states the principle ("client provides workspace_path") — the implementation violates it for `.env` reads.

Downstream effects of "daemon reads workspace" state ownership:

- `JaatoServer._session_env` — authoritative dict of resolved workspace env. Used for `MODEL_NAME`, `JAATO_PROVIDER`, provider API keys, OAuth tokens, `pass://` URIs.
- `JaatoServer._model_name` / `_model_provider` — derived from `_session_env`.
- `JaatoServer.initialize()` — uses `_session_env` to construct the provider plugin instance daemon-side.
- `JaatoServer.verify_auth()` — uses `_session_env` to find credentials.
- `JaatoServer._with_session_env()` — context manager that promotes `_session_env` to `os.environ` so the runner subprocess inherits via `os.environ.copy()` at fork.

The runner subprocess therefore inherits workspace env vars "for free" today — but only because the daemon's process can read the file. Move daemon and workspace apart and the chain breaks.

A secondary smell: `pass://` URI resolution. `pass` is GPG-based and inherently user-machine-local (needs `gpg-agent` + the user's private key). The daemon can resolve `pass://` URIs today **only** because daemon and user share a machine. In any future split, `pass://` MUST be resolved client-side.

## 3. Target state (Shape 3 proper)

| Today (broken under daemon/workspace split) | Shape 3 proper |
|---|---|
| Daemon calls `dotenv_values(workspace/.env)` | Client reads `.env`, sends contents over wire |
| Client sends `env_file` path; daemon reads it | Client sends `env_dict` (resolved values); daemon never opens workspace files |
| Daemon resolves `pass://` via local `pass` binary | Client resolves `pass://` (only the client has the GPG agent + key); sends literals over wire |
| `JaatoServer._session_env` is authoritative | `JaatoServer._session_env` is empty / removed; ownership transfers to runner-side `JaatoSession` |
| Daemon constructs provider plugin instance | Runner subprocess constructs provider plugin instance |
| `JaatoServer.verify_auth()` runs daemon-side | Auth verification runs runner-side |
| Daemon spawns subprocess with `os.environ.copy()` of resolved env | Daemon spawns subprocess with envelope-carried env_dict; subprocess sets its own env from envelope, then connects provider |

The daemon's role narrows to: routing, session lifecycle bookkeeping, AppArmor/cgroup provisioning, RPC plumbing. Workspace data flows through it but is not interpreted by it.

## 4. Stages (PR plan)

Four PRs, each independently reviewable and rollback-safe. Each PR keeps back-compat so the system stays bisectable through the migration.

### PR 1 — Wire-format: client sends `env_dict` over the wire

**Scope:**

- `ClientConfigRequest` (SDK + IPC) gains `env_dict: Optional[Dict[str, str]]` field. When present, daemon uses it as the session env source. When absent, daemon falls back to reading `env_file` (existing path).
- IPC client + TUI populate `env_dict` by reading the workspace `.env` (client-side), resolving `pass://` URIs (client-side, where `gpg-agent` lives), and sending the resolved dict.
- Daemon-side `JaatoServer._resolve_session_env()` accepts the dict as input parameter; uses it verbatim when provided; falls back to `dotenv_values(env_file)` otherwise.
- Daemon's `pass://` resolver remains as fallback for the `env_file`-only path (back-compat with non-upgraded clients) but emits a deprecation log.

**Touchpoints:**
- `jaato-sdk/jaato_sdk/events.py` — add `env_dict` to `ClientConfigRequest`
- `jaato-sdk/jaato_sdk/client/ipc.py` — IPC client reads `.env` and `pass://`-resolves before send
- `jaato-tui/rich_client.py` — wire the new client-side resolution into the bootstrap
- `jaato-server/server/session_manager.py` — accept `env_dict` from client config, pass to JaatoServer ctor
- `jaato-server/server/core.py:_resolve_session_env` — prefer `env_dict` over file read

**Closes user-facing regression:** profile-less `jaato --new-session` works again because `MODEL_NAME` flows correctly to the runner.

**Out of scope for PR 1:** removing daemon's `dotenv_values` path entirely (kept as back-compat fallback until PR 4 confirms migration complete).

### PR 2 — Envelope carries the full resolved env to the runner

**Scope:**

- `SessionInitEnvelope.env_overrides` is extended to carry the FULL resolved session env (not just `profile.env` as today).
- `runner_spawn.build_session_envelope` reads from a clean per-session env source (new `Session.session_env` dict on the `Session` dataclass at session_manager level) instead of from `JaatoServer._session_env`.
- Runner-side `bootstrap_session` applies `envelope.env_overrides` to its `os.environ` BEFORE validate runs.
- Runner-side validate reads `os.environ.get("MODEL_NAME")` / `JAATO_PROVIDER` as authoritative.

**Touchpoints:**
- `jaato-server/shared/session_envelope.py` — extend `env_overrides` documentation (no schema change; field already exists, just gets more data)
- `jaato-server/server/runner_spawn.py:build_session_envelope` — merge session env into env_overrides
- `jaato-server/server/runner/session.py:bootstrap_session` — apply env_overrides to os.environ
- `jaato-server/server/runner/session.py:_validate_envelope` — read from os.environ

**Decouples runner-side validate from daemon's `_with_session_env()` context** — runner no longer relies on the daemon being in a specific context-manager state when fork() happens. The envelope carries the data explicitly.

### PR 3 — Provider construction moves to the runner

**Scope:**

- `JaatoRuntime.create_provider()` and `JaatoRuntime.connect()` execute in the runner subprocess.
- Daemon-side `JaatoServer.initialize()` stops constructing the provider; it sends a `runner.initialize_provider` RPC and awaits the runner-side completion.
- Provider plugin instances live in the runner subprocess; daemon never holds API keys.
- Daemon-side reads of `runtime._provider` are replaced with RPC into the runner (or read provider metadata from the runner-published agent state).

**Touchpoints (heaviest PR):**
- `jaato-server/shared/jaato_runtime.py` — provider construction path
- `jaato-server/server/core.py:JaatoServer.initialize()` — provider construction relocated, replaced by RPC
- `jaato-server/server/runner/session.py:bootstrap_session` — calls `runtime.connect()` + `runtime.create_provider()` on the runner side
- All daemon-side callers of `runtime._provider` — audit and reroute

**Risk:** highest of the four PRs. Touches the central provider-orchestration path. Needs careful coverage of: model-switch (`/model` command), reactor-spawn (subagent with different provider), auth-pending flow, tier-switching.

### PR 4 — Auth verification moves to the runner

**Scope:**

- `JaatoServer.verify_auth()` becomes an RPC into the runner.
- Auth-pending flow re-routes: client sends auth-action commands, daemon forwards to runner, runner executes against its provider.
- Daemon-side `_session_env` becomes empty by design — nothing populates it.
- After PR 4 lands, `_session_env`, `_session_env_resolved`, `_with_session_env`, `dotenv_values(env_file)` can be removed.

**Touchpoints:**
- `jaato-server/server/core.py:verify_auth` — RPC plumbing
- `jaato-server/shared/jaato_runtime.py:verify_auth` — receives the RPC call runner-side
- Auth plugins — already runner-tier per memory (most are); audit those still daemon-tier

## 5. Back-compat + rollback story

Each PR ships back-compat shims:

- **PR 1**: `env_dict` is optional; missing → daemon reads `env_file` as today. Back-compat preserved for non-upgraded clients.
- **PR 2**: envelope's `env_overrides` field already exists; bigger payload is forward-compat. Older runner versions ignore unknown keys.
- **PR 3**: introduces `runner.initialize_provider` RPC behind a `JAATO_RUNNER_OWNS_PROVIDER` env-var flag (defaults `false`). Flipping the flag on stages the migration without forcing it.
- **PR 4**: same flag-gating pattern.

The flag flip from `false → true` is the cutover; the back-compat dead-code removal is a fifth PR that lands after the flag has been `true` in main for ≥ one week.

## 6. Non-goals

- **Multi-host deployment** is the architectural target this enables, but actually deploying daemon and runner on different machines is out of scope for this design.
- **Provider-instance lifecycle** (when to construct, when to destroy) stays as today — one provider instance per session in PR 3, just runner-side instead of daemon-side.
- **Reactor-spawn cross-provider sessions** continue to work via the existing subagent-spawn path; this design doesn't change subagent semantics.
- **`pass://` resolver consolidation** — PR 1 introduces the client-side resolution path but does not remove the daemon-side `pass://` resolver (back-compat). A future PR removes the daemon-side resolver once telemetry confirms zero non-upgraded clients.

## 7. Memory references

- `feedback_daemon_is_workspace_agnostic` — the principle stated; this design implements it for `.env` reads
- `project_backlog_env_propagation_seat_flip_gap` — Phase 4 backlog item closed by PR 2
- `feedback_session_env_from_workspace_dotenv_only` — confirms per-session env is workspace-tied (not daemon `os.environ`)
- `project_env_secret_uri_resolution` — server 0.6.64+ already does daemon-side `pass://` resolution; PR 1 moves that to client-side for the new wire format
- `feedback_never_substitute_pass_uri_with_literal_in_env` — applies to files on disk; sending resolved literals over the wire (in-memory) does not violate it
- `feedback_no_jaato_changes_without_authorization` — each PR's framework code edit + daemon restart requires explicit per-PR user authorization
- `feedback_cascade_aware_daemon_restart_coordination` — daemon restarts during this work must coordinate with peer cascade runs

## 8. Acceptance gates

Each PR lands behind:
- Differential test sweep: zero new failures vs main (the same pattern used for Phase 5 §5.8 / §5.9 / §5.10 review cycles)
- New regression tests pinning the wire/state at the migration point
- Manual verification: profile-less `jaato --new-session` works (PR 1+), and existing profile-driven flows unchanged (every PR)
- Real-host gate: cascade smoke runs unchanged before merge (coordinate with peer 7:3)

## 9. Open questions

1. **Wire-format compactness** — should `env_dict` be a flat dict or carry typed metadata (`{key, value, source: "env_file" | "profile_env" | "client_override"}`) for debugging? Default: flat dict; metadata adds complexity for marginal benefit.
2. **Secret redaction in logs** — currently the daemon never logs `_session_env` contents (good). PR 1 keeps the same posture client-side. Confirm the client doesn't log the resolved dict either.
3. **TUI's env-file flag (`--env-file`)** — keep as-is (client picks the file to read); just the file-read moves from daemon to client.
4. **Headless mode** (`jaato-server/server/README_headless_how-to.md`) — explicitly documents `MODEL_NAME` env-var driven flows. PR 1 doesn't break this (the headless harness can use either the env-file path or its own `env_dict`); but the docs may want an update describing the new preferred path.

## 10. Estimated effort

| PR | LoC est. | Tests | Daemon restart? | Risk |
|---|---|---|---|---|
| PR 1 | ~150 | ~30 | yes (wire format + back-compat fallback) | LOW — additive field |
| PR 2 | ~200 | ~40 | yes | MEDIUM — envelope contract change |
| PR 3 | ~600 | ~60 | yes | HIGH — provider orchestration relocation |
| PR 4 | ~250 | ~30 | yes | MEDIUM — auth flow |
| Total | ~1200 | ~160 | 4 restarts | MEDIUM (per-PR LOW-HIGH) |

Each restart requires peer coordination (`feedback_cascade_aware_daemon_restart_coordination` ping). Cumulative downtime across the four PRs: minutes, spread over ≥ one week real time.

## 11. Decision log

- **2026-05-13** — Stage 3.1 (5-LoC runner-side `os.environ` fallback) considered and rejected. Reason: preserves the "daemon reads workspace files" architectural smell; doesn't address the daemon/workspace-split scenario; would silently regress under future WS-remote-workspace deployments. Chose Shape 3 proper as a multi-PR project instead.
