# Multi-Tenancy & RBAC for jaato — Design Analysis

> **Status:** Analysis / proposal. No code changes.
> **Audience:** Maintainers deciding whether and how to make jaato safe for
> multiple distrustful users sharing a single deployment.

This document is an honest audit of where jaato stands today on
multi-tenancy and role-based access control, followed by a layered
proposal for getting to a real multi-tenant posture without breaking
the existing single-tenant deployments.

It is grounded in the code as of commit `de86c76` (server 0.6.43). Every
claim in §1 has a file:line citation. The proposal in §3-§9 is opinionated
but flagged where it is.

---

## 1. The Honest Audit — What Exists Today

### 1.1 Terminology trap

In jaato's docs, "tenant" already means **a client application connecting
to the daemon over IPC/WS** (see `docs/reactor-tenant-guide.md`, §1).
That is *not* what this document means by tenant. Throughout this
document:

- **Principal** — an authenticated identity (a person, a service account)
- **Tenant** — an isolation boundary owning a set of principals,
  credentials, workspaces, sessions, and quotas; principals from
  different tenants must not be able to read or influence each other's
  state, even by accident
- **Role** — a named bundle of capabilities granted to principals within a
  tenant

This terminology change is itself a deliverable: the existing "tenant"
language in `reactor-tenant-guide.md` should be retitled "client" once
the new tenant concept lands, to avoid two meanings.

### 1.2 Identity is a stub

Bearer auth on the WebSocket transport
(`jaato-server/server/__main__.py:62-67`,
`jaato-server/server/websocket.py:412-416`) compares one daemon-global
SHA-256 digest with `hmac.compare_digest`. It establishes
"this connection presented the right token," not "this connection
belongs to user X."

The plumbing for real identity is *almost* in place but no production
code path uses it:

- `ClientConnection.user_id` field exists
  (`jaato-server/server/websocket.py:324`)
- `set_client_user(client_id, user_id)` is implemented at the WS
  transport (`websocket.py:627-642`), at the IPC transport
  (`ipc.py:755`), and at the event sink fan-out
  (`event_sink.py:73,136-139`)
- `command_router.py:313` and `:1046` already resolve `created_by =
  self._event_sink.get_client_user(client_id)` and pass it into
  session creation
- `Session.created_by` and `RuntimeSessionInfo.created_by` are
  declared (`session_manager.py:87,106`) and propagated through
  `create_session()` (`:888,911,1237,2868`)

**But:** no production code ever calls `set_client_user()`. It is
documented in `CLAUDE.md` as "for jaato-premium SSO." The hook is
load-bearing for the entire identity model and there is currently no
public auth provider that drives it. `created_by` is therefore
populated with `None` everywhere in OSS.

The IPC transport never reads peer credentials
(`SO_PEERCRED` / `getsockopt` is absent from the codebase). Local
clients that share filesystem access to the unix socket are
indistinguishable.

### 1.3 The permission system is tool-scoped, not principal-scoped

`EvalContext` (`shared/plugins/permission/evaluator.py:92-121`) carries
`tool_name`, `args`, `agent_type`, `agent_name`, `session_id`,
`workspace_path`, `turn_index`, `model_preamble`, and an open-ended
`extra: Dict[str, Any]`. It does **not** carry `user_id`, `tenant_id`,
or `roles`.

The permission README aspirationally references `user_id` in
`permission/README.md:805,844,859,871`, but the dataclass does not have
the field and no evaluator can therefore base a decision on it.

Permission state — whitelists, blacklists, "always-allow this turn,"
"suspend until idle" — is held in memory on the `PermissionPlugin`
instance attached to a session
(`shared/plugins/permission/plugin.py:71-76`). It is session-scoped,
not principal-scoped, so two principals attaching to the same session
inherit each other's prior approvals.

`permissions.example.json` models tools and patterns; there are no
roles.

### 1.4 The runtime is per-session, not daemon-shared

The first version of this document got this wrong. Reading the code
carefully:

- `SessionManager._sessions: Dict[str, Session]`
  (`session_manager.py:159`)
- Each `Session` holds its own `server: JaatoServer`
  (`session_manager.py:91-95`); `SessionManager.create_session()`
  constructs a fresh `JaatoServer` per session
  (`session_manager.py:1179`)
- Each `JaatoServer` constructs its own `JaatoClient`
  (`core.py:1283`), which constructs its own `JaatoRuntime`
  (`jaato_client.py:385`)
- `JaatoRuntime.__init__` initialises its own `_provider_configs`,
  `_ledger`, `_registry`, `_permission_plugin`, etc.
  (`jaato_runtime.py:255-269`)

The `JaatoRuntime` docstring "Sessions share the runtime's resources
(registry, permissions, ledger)" (`jaato_runtime.py:884-886`) refers
to **subagents within a single user session sharing the parent
session's runtime** — which is the only context where one
`JaatoRuntime` backs multiple `JaatoSession`s. A daemon serving N
user sessions has N independent runtimes.

So the multi-tenant story for in-memory state is much better than my
audit suggested:

| Resource | Actual scope today |
|----------|---------------------|
| `ProviderConfig` (creds) | per-session (each session has its own) |
| `TokenLedger` | per-session |
| `PluginRegistry` *instance* | per-session |
| `PermissionPlugin` *instance* | per-session (whitelists / suspensions don't bleed across sessions) |
| MCP server fleet | per-session |
| Subagents | share their parent session's runtime (privilege inheritance is intentional) |

What **is** daemon-shared:

- `SessionManager` itself: the `_sessions` index,
  `_client_to_session` mapping, `_client_config` per-client config,
  `_workspace_monitors` index (`session_manager.py:159-177`)
- `_instruction_token_cache: InstructionTokenCache`
  (`session_manager.py:181`) — content-addressed and so leak-safe
- `_session_hooks` registered by daemon extensions
  (`session_manager.py:185`)
- The `_broadcast_callback` for daemon-wide events (HandoffGate from
  jaato-premium, `session_manager.py:174`)
- The OS process: UID/GID, environment, network namespace, filesystem
  view
- On-disk state: `~/.jaato/` user-tier (OAuth tokens, ws.token,
  service credentials, profiles), workspace `.jaato/`,
  `LEDGER_PATH` if shared between sessions

The corrected framing: the *application-level* objects that hold
secrets and policy are already per-session. The remaining sharing is
at the **process and filesystem** layer, which is exactly what
sandboxing has to address — and partially does (next subsection).

### 1.5 Workspace isolation is kernel-enforced, per-session

The first version of this document also got this wrong. There is a
full `AppArmorManager` (`server/apparmor.py:47-1262`) that already
implements per-session kernel-level workspace confinement:

- **Per-session profile.** `provision_profile(session_id,
  workspace_path, config_root, env_file)` renders a profile from a
  versioned template (`apparmor.py:604-703`) and loads it via
  `apparmor_parser -r`. Profile name: `jaato-ws-{session_id}`
  (`apparmor.py:13,1055`). Each session gets its own kernel-enforced
  view of the filesystem keyed by its workspace path.
- **Thread-level confinement on every tool call.** `apparmor_confine
  (profile_name)` is a context manager that writes
  `changeprofile <name>` to `/proc/self/task/<tid>/attr/current`
  (`apparmor.py:1164-1245`). The `ToolExecutor` wraps every tool
  invocation in this context (`ai_tool_runner.py:185-196,1025-1051`).
  This means **in-process** file I/O — `readFile`, `glob_files`,
  `file_edit` — is confined under the same profile that subprocess
  CLI commands get; not just shelled-out work.
- **Subprocess inheritance.** Subprocesses (`cli`,
  `interactive_shell`, MCP servers) inherit the parent thread's
  AppArmor profile via fork+exec (`subprocess_runner.py:14,193`).
- **Cross-thread cleanup.** `apparmor_confine` defensively writes
  `changeprofile unconfined` on entry to recover from a prior
  session's stuck-confinement state — necessary because asyncio
  thread-pool workers are reused across sessions
  (`apparmor.py:1173-1190`).
- **Dynamic per-session reference grants.**
  `add_reference_fragment(session_id, ref_id, path)` writes a fragment
  into a per-session `.refs.d/` directory included by the profile via
  `include if exists`, so the references plugin can grant readonly
  access to specific paths without rewriting the base profile
  (`apparmor.py:887-1031`). The handle a session uses for this is the
  per-session `ReferenceAuthorizer` (`apparmor.py:1271-1298`,
  `jaato_session.py:315-1273`).
- **Activation.** WS deployments confine automatically when AppArmor
  is available; IPC opt-in via `IPCClient(apparmor=True)` (default
  `False`); see `docs/apparmor-setup.md`.
- **Companion resource caps.** `RuntimeLimits`
  (`shared/runtime_limits.py:55-94`) is a per-session profile field
  that maps to cgroup v2 (`memory.max`, `pids.max`, `cpu.weight`) plus
  application-enforced `tool_timeout_seconds` and
  `max_output_bytes`. Provisioned by `server/cgroups.py`. This handles
  the "noisy neighbour / DoS" axis that AppArmor doesn't cover.

So workspace boundaries are **not advisory** under WS deployments
(and IPC with the apparmor flag set): they are kernel-enforced, per
session, and apply both to subprocesses and to the daemon's own
threads while executing tools. Sibling-session paths are denied by
default-deny in the profile (`tests/test_apparmor.py:92` — sibling
workspaces denied test).

The remaining gaps are not "no jail" but "the jail is bound to a
session, not to a principal" — which becomes important once
identity is plumbed through (see §3-§4). Concrete leftover surface:

- `~/.jaato/` user-tier files are readable from every confined
  session (template versions 4, 6, 7 explicitly grant `~/.jaato/`
  reads — `apparmor.py:84-100`). Cross-principal credential isolation
  must happen *above* AppArmor, in the credential store.
- AppArmor is Linux-only and requires `apparmor_parser` plus
  privileges; non-Linux deployments fall back to no kernel jail
  (`apparmor.py:528-602`).
- Standalone `JaatoClient` usage (no daemon) does not provision
  profiles.
- IPC's `apparmor=True` is opt-in, so a default IPC client gets no
  kernel confinement; `cli`, `file_edit`, etc. then revert to
  workspace-cwd defaults that are advisory in the sense the original
  audit claimed — but only for that deployment shape.

### 1.6 Credentials live next to the code

- `~/.jaato/ws.token` (mode 0600) — single shared bearer for the WS
  transport
- Provider API keys / OAuth tokens — env vars, plugin-managed stores
  in `~/.jaato/`, or profile `plugin_configs`
- These are owned by the daemon's OS user. There is no per-principal
  credential store and no way to keep tenant A's API key invisible to
  tenant B.

### 1.7 Audit and quota: anonymous

`TokenLedger` (`shared/token_accounting.py:36-141`) records timestamps,
token counts, and API errors. No `user_id`, no `tenant_id`, no
`session_id` in the schema. Quotas, if any, are global to the daemon.

OTel spans (`docs/jaato_opentelemetry.md`) carry `session_id`,
`agent_type`, `turn_index`. No principal/tenant attributes.

### 1.8 Reactor (premium) is workspace-scoped

`reactor-tenant-guide.md` describes the reactor as workspace-scoped
automation. Rules in `reactors.json` have no principal field; an
agent handoff triggered by user A's completion event will run with
the daemon's privileges, not user A's, and could spawn an agent that
other users see.

### 1.9 What's already designed but not built

| Doc | Built? | Reusable for tenancy? |
|-----|--------|------------------------|
| `jaato_permission_system.md` | Yes (rule engine, evaluators) | Yes — needs principal-aware EvalContext |
| `permission-evaluators.md` | Yes (callable evaluators) | Yes — natural place for capability checks |
| `apparmor-setup.md` | Yes (one daemon profile) | Partially — needs per-session profiles |
| `websocket-workspace-isolation.md` | Yes | Adjacent — it isolates *workspaces*, not principals |
| `compare-rbac-profiles-frameworks.md` | N/A (comparison) | Acknowledges the gap |
| `reactor-tenant-guide.md` | Yes (workspace tenancy) | No — terminology will need migration |

**Summary:** the data plumbing for identity (`set_client_user`,
`created_by`, `user_id` field on connections) is in place but
unused. The decision plane (permission, ledger, plugin registry) is
unaware of identity. OS-level isolation exists at daemon granularity,
not session granularity. There is no role concept anywhere.

---

## 2. What "Multi-Tenant" Should Mean Here

Three deployment shapes are coherent. Pick one, or layer them:

**Tier A — Soft multi-tenancy (single daemon, single OS user, logical
isolation in app code).**
Cheap. Adequate for a single organization where roles matter but
operators trust their tools. Bug-prevention, not malice-prevention.
Roughly: the missing identity / authz / audit work in §3-§7.

**Tier B — Hardened multi-tenancy (single daemon, OS-level sandboxing
per session).**
Per-session AppArmor profile + per-session cgroup. **Mostly already
present** — see §1.5. What's missing for full Tier B: tying the
profile and cgroup to a *principal*, not just a session id; scrubbing
subprocess environments so daemon env vars don't reach MCP servers
of a different tenant; namespacing for non-filesystem resources
AppArmor doesn't cover (PIDs, network, IPC); per-tenant credential
store keeping `~/.jaato/` reads from leaking across principals.

**Tier C — Tenant-per-process.**
The daemon becomes a thin control plane that spawns a worker process
per tenant, with its own OS user (setuid, `systemd-run --uid`), its
own runtime, its own MCP fleet. Strongest isolation, highest
operational cost. The reason to want this anyway despite Tier B's
existence: AppArmor confines filesystem access but not the daemon's
in-process memory, so a tenant-aware bug in jaato itself (or a
malicious in-process plugin) bypasses Tier B.

The proposal below is layered: Tier A first because it forces the
data model that both B and C need; B as the recommended production
target (most of the OS-level pieces exist; remaining work is
principal-binding); C as an optional deployment.

---

## 3. Identity & Authentication

Replace the daemon-global bearer with a pluggable `AuthProvider`
protocol:

```python
class AuthProvider(Protocol):
    def verify_credentials(
        self, transport: str, headers: Mapping[str, str], query: Mapping[str, str]
    ) -> Optional[Principal]: ...
```

Concrete providers:

- **SharedTokenAuthProvider** — current behaviour; resolves to an
  anonymous `Principal(user_id="anon", tenant_id="default", roles=("admin",))`.
  This is the backwards-compat default.
- **TokenListAuthProvider** — file of `(token, user_id, tenant_id, roles)`
  rows.  Useful for small deployments and tests.
- **JWTAuthProvider** — verifies a JWT against a JWKS endpoint;
  maps claims (`sub`, `tenant`, `roles`) to a `Principal`.
- **OIDCAuthProvider** — full OIDC, including refresh.
- **PeerCredAuthProvider** (IPC only) — reads `SO_PEERCRED`, maps OS
  uid → principal via a config file. Free local single-machine RBAC.

The provider runs **after** the connection interceptor and **before**
any session work, exactly where bearer auth runs today (so the
existing 1008-close path is reused). Both transports
(`ipc.py:755`, `websocket.py:627-642`) call `set_client_user()` with the
resolved `Principal.user_id`. The `Principal` object itself is stashed
on the `ClientConnection` (a new field — `user_id` alone is too
narrow now).

Premium SSO becomes "ship one more `AuthProvider` implementation,"
which is the original intent of the hook.

---

## 4. Authorization — RBAC Layered on Top of the Existing Permission Engine

Capabilities are first-class, namespaced strings. Examples:

- `session.create`
- `session.attach:owner` / `session.attach:any`
- `tool.invoke:<tool_name>` (or `tool.invoke:cli` with arg-pattern
  refinement done by the existing rule engine)
- `profile.use:<profile_name>`
- `mcp.connect:<server_name>`
- `provider.use:<provider_name>`
- `workspace.read:<glob>` / `workspace.write:<glob>`
- `reactor.handoff:cross_tenant`

A `Role` is a set of capabilities. A `Tenant` has roles, members
(`user_id → set[role]`), and quotas. Tenant config lives at
`.jaato/tenants/<tenant_id>.json`, hot-reloadable.

The authorization decision is a **two-stage** function evaluated on
every tool call (and on a few session-management actions):

1. **Capability check** — does the principal hold `tool.invoke:<tool>`
   under their tenant's role assignments? Hard deny if not. No prompt,
   no whitelist override.
2. **Existing rule engine** — if capability passes, fall through to
   today's `EvalContext` flow (whitelist / blacklist / patterns /
   evaluator scripts / interactive prompt). This is the layer where
   per-arg refinement and runtime user choices live.

`EvalContext` grows three fields, with default `None` for backward
compatibility:

```python
@dataclass
class EvalContext:
    ...
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: tuple[str, ...] = ()
```

This is a strictly additive change — existing evaluators ignore the
new fields and keep working.

**Why RBAC above the rule engine, not inside it?** Because capability
denial is a security property (must always hold) and the rule engine
is a UX layer (prompts, suspensions, user choices). Conflating them
makes capability bypass too easy. Defense in depth: RBAC says "you
may ask," the rule engine says "you may ask *this way*."

---

## 5. Resource Scoping — What's Already Per-Session vs. What Needs Principal/Tenant Scope

The first version of this section assumed the runtime was
daemon-shared. It isn't (§1.4). Here's the corrected current state
plus what changes for tenancy:

| Resource | Actual scope today | Tenancy change |
|----------|---------------------|-----------------|
| `ProviderConfig` (creds) | per-session | tag with `tenant_id`; resolve `${VAR}` against tenant credential namespace, not daemon env |
| `TokenLedger` | per-session | per-session (unchanged) but each event tagged with `(tenant_id, user_id)`; aggregate quota check at tenant scope |
| `PluginRegistry` instance | per-session | per-session (unchanged) |
| `PermissionPlugin` instance | per-session | per-session (unchanged); capability layer above (§4) is tenant-scoped |
| MCP server fleet | per-session subprocesses | per-session (unchanged) but env scrubbed before exec; only the calling principal's secrets reach the subprocess |
| AppArmor profile | per-session (`jaato-ws-{session_id}`) | per-session; profile *generator* takes the principal's `workspace.read|write` capabilities as input |
| cgroup `RuntimeLimits` | per-session | per-session; profile carries tenant default limits; per-user override allowed |
| Workspace path | per-session | per-user-jailed via principal's capabilities; sharing requires `workspace.read` |
| Sessions list | daemon-wide via `SessionManager` | filtered to caller's tenant unless `session.list:any` capability held |
| `~/.jaato/ws.token` | daemon-global | replaced — `AuthProvider` owns credentials |
| `~/.jaato/services/` (read by every confined session) | daemon-global filesystem | per-tenant subdir; AppArmor template parameterised by tenant id |
| Provider OAuth tokens | daemon-global keyrings/files | per-principal credential store |
| Reactor `reactors.json` | workspace | per-tenant rule set; rules tagged with the principal whose event fires them |
| `_instruction_token_cache` | daemon-shared | unchanged (content-addressed, leak-safe) |
| `_session_hooks` | daemon-shared | unchanged (extension code; runs daemon-side) |

The tagline is: most of the in-memory isolation already holds at the
session boundary. Tenancy work is mostly (a) attaching a principal
identity to every per-session resource, (b) gating cross-session
operations (attach/list/handoff) by capability, and (c) replacing the
daemon-global slices of `~/.jaato/` with per-tenant subdirs that the
AppArmor template renders into the profile.

---

## 6. Workspace Boundary Enforcement — Hardening What's Already There

Most of this is already built (§1.5). The remaining work to take it
from "per-session jail" to "per-tenant jail":

1. **Principal-aware profile generation.**
   `AppArmorManager.provision_profile(session_id, workspace_path,
   config_root, env_file)` (`apparmor.py:604`) takes a session id and
   workspace; it should also take the principal's
   `workspace.read|write` capability set so the rendered template
   reflects per-principal grants instead of "everything under the
   workspace path." The template's `{config_root_rules}` and
   per-session `.refs.d/` mechanisms already prove parameterisation
   works; this is more of the same.

2. **Scrub `~/.jaato/` user-tier reads.**
   Template versions 4, 6, 7 grant reads to `~/.jaato/services/`,
   `~/.jaato/`, and similar so confined sessions can find profiles
   and credentials (`apparmor.py:84-100`). For multi-tenant, the
   user-tier path needs a per-tenant subdir
   (`~/.jaato/tenants/<tenant_id>/`) and the template should render
   only that subdir into the profile. Cross-tenant reads of
   `services/`, `ws.token`, OAuth caches must be denied at the
   kernel level, not the application level.

3. **Scrub subprocess environments.**
   Today, MCP server subprocesses, `cli`, and `interactive_shell`
   inherit the daemon's full env (the AppArmor profile then restricts
   their filesystem reach, but env is unaffected). Build an env
   allowlist computed from the principal's capabilities and the
   resolved provider config; export only that to the subprocess.
   This closes the "API key for tenant A reachable in
   `os.environ` of tenant B's MCP subprocess" leak that AppArmor
   doesn't cover.

4. **Namespace isolation for what AppArmor doesn't cover.**
   AppArmor is filesystem-only. Per-session PID, network, and IPC
   namespaces (Linux: `unshare`/`clone`) plus seccomp filters would
   close the residual surface. This is genuinely new work; cgroups
   already exist (`server/cgroups.py`) so adding namespacing on top
   of the same cgroup is incremental.

5. **Non-Linux fallback.**
   `apparmor.is_available()` returns False on non-Linux
   (`apparmor.py:528-602`). For macOS, `sandbox-exec` is the closest
   fit; for Windows, AppContainer / job objects. Without these, Tier
   B is Linux-only and the documentation should say so.

6. **Standalone-client and IPC-default deployments.**
   Standalone `JaatoClient` (no daemon) and IPC clients with the
   default `apparmor=False` get no kernel jail. The original
   "advisory" warning in §1.5 *does* apply to these shapes; ship a
   prominent warning when running multi-tenant without confinement
   active.

`docs/apparmor-setup.md` and
`docs/websocket-workspace-isolation.md` are the existing references
this work extends.

---

## 7. Audit & Quota

Append-only audit log keyed by `(tenant_id, user_id, session_id,
request_id)`. Events: auth pass/fail, capability check (allow/deny),
tool invocation start/result, permission prompt outcome, profile use,
session attach, credential read/rotation, MCP server start, reactor
rule fire.

OTel spans get principal attributes: `jaato.tenant_id`,
`jaato.user_id`, `jaato.principal.roles`. The OpenInference mapping
in `docs/design/openinference-telemetry-mapping.md` should be reviewed
against `agent.user.id` semantics so the same fields work with both
collectors.

`TokenLedger.charge(...)` returns `Allowed | Throttled | Exceeded`
with the decision driven by both per-user and per-tenant quotas. This
is also where billing attribution falls out for free.

---

## 8. Migration Plan — Non-Breaking, Ordered, Shippable in Phases

**Phase 1 — Foundation (zero behaviour change for existing users).**
- Define `Principal`, `Tenant`, `Role`, `Capability` in
  `shared/security/`
- Extend `EvalContext` with `user_id`, `tenant_id`, `roles`
- Wire `set_client_user()` from a `SharedTokenAuthProvider` that
  resolves the existing `--ws-token` flow into
  `Principal(user_id="anon", tenant_id="default", roles=("admin",))`
- Populate `created_by` end-to-end (the plumbing already exists)
- Add principal attributes to OTel spans
- Audit-log scaffolding, recording the anonymous principal

This phase is high-value on its own: it unlocks audit, billing
attribution, and the data shape every subsequent phase needs. Shipping
just Phase 1 is a defensible release.

**Phase 2 — Auth + RBAC.**
- Real `AuthProvider` implementations (TokenList, JWT, OIDC, PeerCred)
- Tenant config in `.jaato/tenants/`
- Capability-check layer above the existing rule engine
- Tenant-scoped permission policy (a tenant's roles bake out into the
  capability set; per-session whitelists still work as today within
  each role's allowed tools)

**Phase 3 — Resource scoping.**
- Per-tenant `JaatoRuntime`, `ProviderConfig`, `TokenLedger`,
  MCP fleet
- Per-tenant credential store (file-based, with a clean interface so
  HSM/KMS backends can drop in later)
- Per-user workspace roots, with cross-user sharing requiring an
  explicit `workspace.read:<path>` grant
- Audit log fully populated; quotas enforced

**Phase 4 — Hardening (lifting what exists to Tier B).**
- Principal-aware AppArmor profile generation (extend
  `provision_profile` to take a principal's capability set, §6.1)
- Per-tenant slicing of `~/.jaato/` so cross-tenant reads are denied
  at the kernel level, not the app level (§6.2)
- Subprocess env scrubbing for MCP / `cli` / `interactive_shell`
  (§6.3)
- Optional namespacing on top of existing cgroups (§6.4)
- Reactor handoffs become tenant-scoped; cross-tenant requires the
  capability
- Optional per-tenant worker process model (Tier C deployment)

Each phase ships independently. Phases 1-3 cover Tier A; Phase 4
takes the existing per-session AppArmor + cgroup work the rest of
the way to per-tenant Tier B (and optionally C).

---

## 9. Open Questions, Risks, and Things That Need a Decision

**Open questions, in rough order of "you'll have to answer this before
Phase 2":**

1. **Profile ownership.** Profiles live in `.jaato/profiles/` today.
   Tenant-owned with a daemon-global fallback? Or strictly tenant?
   Affects how `profile.use:<name>` is checked.
2. **Subagents inherit the parent's principal — confirmed?** Yes,
   otherwise a tool spawning a subagent is a privilege-escalation
   primitive. This needs to be explicit in `JaatoRuntime.create_session`.
3. **MCP server process model.** Per-tenant or per-(tenant, session)?
   Per-tenant is cheaper but couples blast radius across that
   tenant's sessions. Probably the right default, with a config knob.
4. **Hot-reload of tenant policy.** Changing a role's capability set
   should affect *new* tool calls; in-flight tool calls finish under
   the old policy. Ledger entries record the policy version.
5. **Reactor (premium) cross-tenant handoffs.** Forbidden by default;
   gated by `reactor.handoff:cross_tenant`. The terminology overlap
   in `reactor-tenant-guide.md` will need a rename pass.
6. **Backward compat for `--ws-token` and `~/.jaato/ws.token`.** Keep
   both; treat them as the `SharedTokenAuthProvider` configuration.
   Document them as "anonymous-principal mode," not "auth."
7. **Anthropic / OpenRouter / NIM provider configs in `plugin_configs`.**
   Today they accept `${VAR}` expansion against the daemon's env. Per
   tenant, this becomes "expand against the tenant's credential
   namespace, never the daemon's env." Important — a sloppy default
   here leaks credentials across tenants.
8. **Cross-tenant data leaks via in-process state.** Plan store,
   presentation context, GC summaries, the unified event bus — every
   pipeline component holding session state must be audited for "could
   tenant B's session see tenant A's data via a shared singleton?"
   Most are session-scoped already; the runtime-shared ones (ledger,
   MCP, registry caches) are where to look.

**Recommended default posture once shipped.** Anonymous-principal mode
is the only way today's deployments don't break; it should remain
available behind `--unsafe-anonymous-principal` and emit a startup
WARNING log just like `--ws-unsafe-no-auth` does.

---

## 10. Recommendation

Don't claim multi-tenancy until at least Phase 3 lands. Phase 4 is
required for *hostile* multi-tenancy (mutually distrustful tenants
sharing a daemon).

The cheapest valuable thing is **Phase 1 alone**: it unblocks audit,
billing, and the data shape; it's strictly additive; it has no
behaviour change for existing users. Even if Phases 2-4 slip, Phase 1
on its own makes jaato measurably more operable.

For commercial framing: Phase 1-3 in OSS gives jaato a credible
"single-organization, multi-role" story (which is what most adopters
actually need). Phase 4 plus the Reactor and an OIDC `AuthProvider`
implementation is the natural shape of a commercial multi-tenant
hosting product, reusing the same primitives end-to-end.
