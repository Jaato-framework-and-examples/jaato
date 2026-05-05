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

### 1.4 The runtime is shared across all sessions

`JaatoRuntime` (`shared/jaato_runtime.py:199-485`) is the shared
environment:

- One `ProviderConfig` (one set of credentials)
- One `TokenLedger`
- One `PluginRegistry`
- One `PermissionPlugin`
- One `MCPClientManager` with its set of subprocess MCP servers

Every session — main agent and subagents — references the same
runtime. The runtime is a **convenience** boundary, not a security
boundary.

Concretely this means: if user A's session and user B's session run on
the same daemon, they share Anthropic API credits, share MCP server
processes (which inherit the daemon's full env), share the permission
policy, and share the audit ledger.

### 1.5 Workspace boundaries are advisory

`set_workspace_path()` exists on `cli`, `file_edit`,
`filesystem_query`, `interactive_shell`, etc. It is set globally on
the plugin instance and read by tool implementations to default cwd.

It is **not** a jail. The `cli` plugin runs commands via
`subprocess`; nothing prevents a model-issued `cd /` or a path
traversal in `file_edit`. The MCP plugin spawns servers as
subprocesses inheriting the daemon's full environment and UID. The
existing AppArmor support (`docs/apparmor-setup.md`) provides
kernel-enforced confinement at the daemon level — but it is one
profile, not per-session, so it cannot distinguish user A from user B.

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

**Tier B — Hardened multi-tenancy (single daemon, OS-level sandboxing
per session).**
Per-session AppArmor profile, namespace-isolated FS, dropped
capabilities, scrubbed subprocess env. The workspace path becomes a
real jail. Acceptable for shared SaaS where tenants are mutually
distrustful but blast radius can be capped at "one buggy agent."

**Tier C — Tenant-per-process.**
The daemon becomes a thin control plane that spawns a worker process
per tenant, with its own OS user (setuid, `systemd-run --uid`), its
own runtime, its own MCP fleet. Strongest isolation, highest
operational cost.

The proposal below is layered: Tier A first because it forces the
data model that both B and C need; B as the recommended production
target; C as an optional deployment.

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

## 5. Resource Scoping — What Moves Out of the Runtime

Today's `JaatoRuntime` is daemon-global. The proposal is one
`JaatoRuntime` **per tenant**, lazily created on first use, with the
control-plane `JaatoServer` holding the dict.

| Resource | Today | Proposed scope |
|----------|-------|----------------|
| `ProviderConfig` | daemon | tenant (each tenant brings their own credentials) |
| `TokenLedger` | daemon | tenant (and tagged with `user_id` per event) |
| `PluginRegistry` | daemon | daemon for *code*, per-session for *instances* |
| `PermissionPlugin` instance | session | session (unchanged), but capability layer above is tenant-scoped |
| MCP server fleet | daemon | tenant (started lazily per tenant) |
| Workspace path | session | user-jailed; sharing requires a `workspace.read` capability |
| Sessions | daemon list | tenant-scoped list; cross-tenant `session.attach` requires `:any` |
| `~/.jaato/ws.token` | daemon | replaced — the `AuthProvider` owns credentials |
| Provider OAuth tokens | daemon-global | per-tenant credential store |
| Reactor `reactors.json` | workspace | tenant; rules tagged with the principal whose event fires them |

Sessions still keyed by UUID, but now `(tenant_id, session_id)` is
the global identity. Listing sessions returns only the caller's
tenant's sessions unless they hold `session.list:any`.

---

## 6. Workspace Boundary Enforcement (Tier B)

Tier A treats `JAATO_WORKSPACE_ROOT` as an enforced invariant inside
file-touching plugins (path validation in `file_edit`, `cli`,
`filesystem_query`, `interactive_shell`). This catches bugs but not
malice — `cli` can still execute `cat /etc/passwd`.

Tier B requires kernel-enforced jails. The cleanest path here is to
extend the existing AppArmor work:

- Generate a **per-session** AppArmor profile from the principal's
  capabilities and their `workspace.read|write` globs
- `cli` and `interactive_shell` execute under that profile
  (`aa-exec -p`)
- MCP server subprocesses inherit a **scrubbed** environment:
  only the tenant's credentials, only the tenant's
  `JAATO_WORKSPACE_ROOT`, only the resolved tool config — never the
  daemon's full env

For non-Linux deployments, `bubblewrap` and `firejail` are the usual
fallbacks; on macOS, `sandbox-exec` profiles play the same role.

`docs/websocket-workspace-isolation.md` covers an adjacent piece (per-WS
workspace isolation); the per-session sandboxing builds on it but is
strictly stronger.

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

**Phase 4 — Hardened (Tier B/C).**
- Per-session AppArmor profile generation
- CLI/interactive-shell sandbox wrappers
- Optional per-tenant worker process model (Tier C deployment)
- Reactor handoffs become tenant-scoped; cross-tenant requires the
  capability

Each phase ships independently. Phases 1-3 cover Tier A; Phase 4
moves to Tier B (and optionally C).

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
