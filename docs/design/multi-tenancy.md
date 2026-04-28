# Multi-Tenancy & RBAC

A design for scoping jaato resources by tenant and authorising actions
across tenant boundaries, while keeping single-tenant deployments
unchanged.

## 1. Motivation

Today jaato has partial identity plumbing — `ClientConnection.user_id`,
`Session.created_by`, WS bearer auth — but the data plane is largely
unscoped. The following surfaces leak across users when more than one
authenticated client shares a daemon:

| Surface | Today | Leak |
|---|---|---|
| `Session.attached_clients` | `Set[client_id]` | Any client can `session.attach(<id>)` and receive events |
| `session.list` command | returns all sessions | Enumerates every tenant's work |
| `inject_prompt_to_session` | gated only on existence | Any client can inject into any loaded session |
| EventBus fan-out | broadcasts to all attached clients | No identity check on subscription |
| `TokenLedger` | daemon-wide JSONL | No per-tenant accounting |
| `MCPClientManager` | shared per daemon | All tenants share one MCP namespace |
| Plugin state (memory, todo, references) | per-session, but session attribution is implicit | Cross-tenant aggregation via `session.list` |
| Headless sessions | tagged `_HEADLESS_CLIENT_ID` | No tenant attribution |
| `HandoffGate` events (proposed) | fan out to all attached | Intent metadata exposes workload across tenants |

Three categories of consumer have different needs, and a binary
"tenant or not" model can't serve them:

- **Tenant users** — own their sessions, see their data only.
- **Tenant admins** — manage users within one tenant.
- **Platform operators** — read-only visibility across all tenants for
  monitoring, support, billing reconciliation.

This doc specifies (a) a tenancy model that scopes every resource by
owner, and (b) an RBAC layer that authorises cross-scope access for
roles like operator and superuser.

## 2. Goals & Non-Goals

### Goals

- Single canonical identity flow: auth middleware stamps the connection,
  every downstream record carries it.
- Per-tenant scoping for sessions, events, ledger, plugin state, and
  gate events.
- Role-based access control with default-deny in multi-tenant mode.
- Audit log of cross-tenant actions.
- Single-tenant deployments unchanged: no auth, no policy, everything
  behaves as today.

### Non-goals (v1)

- Network-level tenant isolation (separate ports, daemons-per-tenant).
- Quota enforcement (rate limits per tenant) — adjacent design.
- Federated identity (SAML/OIDC trust chains across daemons).
- Cross-daemon RBAC (peer cluster) — separate design.
- Encryption-at-rest for tenant data — orthogonal.

## 3. Concepts

| Term | Definition |
|---|---|
| **Identity** | The authenticated principal. Stamped on `ClientConnection`. May be a human user or a service. |
| **Tenant** | A scope unit. Owns sessions, ledger entries, plugin state. A user can belong to multiple tenants (data model supports it from v1; the per-connection UX selects one active tenant). The synthetic tenant `_daemon` represents daemon-wide work with no user owner. |
| **Membership** | A `(tenant_id, role_set)` pair. A user has zero or more memberships. The active membership is selected per connection. |
| **Role** | A named set of permissions, defined in `roles.json`. |
| **Permission** | A `(action, resource_type)` tuple, e.g. `(read, session)`, `(write, gate)`. |
| **Scope** | The breadth of a permission: `self`, `tenant`, `global`. |
| **Resource** | A scoped object: a session, a gate, a ledger entry, a plugin record. Always carries an owner `tenant_id`. |
| **Action** | A verb: `read`, `attach`, `write`, `delete`, `list`, `mutate`, `elevate`. |
| **Service identity** | A non-human principal (reactor, peer daemon, admin tool). Authenticated via service token, scoped to a tenant or `_daemon`. See §6.5. |

## 4. Identity Model

### 4.1 Existing surfaces (kept)

- `ClientConnection.user_id: Optional[str]` — set by auth middleware
  via `set_client_user(client_id, user_id)`.
- `Session.created_by: Optional[str]` — stamped at session creation.

### 4.2 Extended `ClientIdentity`

Replace the bare `user_id` with a structured identity object that
supports multi-tenant membership:

```python
@dataclass(frozen=True)
class Membership:
    tenant_id: str
    roles: FrozenSet[str]               # roles within this tenant

@dataclass(frozen=True)
class ClientIdentity:
    user_id: str
    memberships: FrozenSet[Membership]   # zero or more
    active_tenant: str                   # selected per connection
    auth_method: str                     # "bearer", "oidc", "ipc-local", "service-token"
    authenticated_at: datetime
    expires_at: Optional[datetime] = None
    is_service: bool = False             # True for reactors / peer daemons

    @property
    def active_roles(self) -> FrozenSet[str]:
        for m in self.memberships:
            if m.tenant_id == self.active_tenant:
                return m.roles
        return frozenset()

    @property
    def has_global_role(self) -> bool:
        """A role is 'global' if defined as scope=global in roles.json
        and the membership granting it is on the special _global tenant."""
```

The active tenant is set at connect time (default: first membership)
and can be switched via a `client.set_active_tenant(tenant_id)`
command. Switching is itself an authz-checked action.

Storing memberships rather than a flat `(tenant_id, roles)` pair pays
the schema cost once. The day a user joins a second tenant, no
migration is needed.

### 4.3 Single-tenant default

When the daemon starts without `--multi-tenant`:

- All connections receive a synthetic identity with one membership on
  the `default` tenant carrying a synthetic `_local-superuser` role
  granting all actions.
- Auth middleware is bypassed.
- Authorization checks still run (so the code path is identical) but
  always ALLOW.
- Audit logging is **off** by default in single-tenant; turn on with
  `--audit-log` if a hobby user wants it. (No silent disk growth.)

### 4.4 Public-tier identity backends (no premium required)

Multi-tenant mode must work without `jaato-premium`. The public daemon
ships two identity backends:

1. **Local user file (`local-users`).** A JSON file at
   `~/.jaato/users.json` mapping `user_id → {password_hash, memberships,
   service_token?}`. Authenticates against bearer tokens or
   username+password. Suitable for self-hosted single-machine
   multi-user setups.
2. **Service token (`service-token`).** Static tokens for non-human
   callers. Stored at `~/.jaato/service_tokens.json` with the same
   permissions as `local-users`. Used by reactors and peer daemons.

Premium adds:

3. **OIDC (`oidc`).** Validates ID tokens from Keycloak / Auth0 / etc.,
   maps OIDC `groups` claim to memberships via a configurable mapper.

The auth backend is selected by `--auth-backend local-users|oidc` (or
a chain via `--auth-backend local-users,oidc`). Premium-only backends
fail to start without the premium extension installed, with a clear
error message.

### 4.5 Multi-tenant mode

When the daemon starts with `--multi-tenant`:

- Connections without identity are rejected at the transport layer.
- IPC connections require either the `local-users` backend or
  premium SSO (peer-cred lookup is a future enhancement).
- WS connections require one of the configured backends.
- Anonymous connections never receive an identity — they're refused at
  transport before any session work begins.

### 4.6 Identity expiry mid-session

OIDC tokens expire. The flow:

1. On every command, the daemon checks `identity.expires_at`. If past
   or within a `renewal_window` (default 60s), it triggers a
   client-side renewal request via a typed event
   (`IdentityRenewalRequiredEvent`).
2. The client refreshes (transparent in OIDC SDKs) and sends
   `IdentityRefreshRequest` with the new token. The daemon re-authzes
   and updates `ClientConnection.identity`.
3. If renewal fails or times out (default 30s), the connection enters
   **degraded mode**: existing sessions continue running but
   write-actions deny. After `degraded_grace` (default 5 min) the
   connection is closed.

This preserves long-running sessions across token expiry without
requiring full reconnection.

## 5. Tenancy Model

### 5.1 Resource ownership

Every scoped resource gains a `tenant_id` field:

| Resource | Existing field | New field |
|---|---|---|
| `Session` | `created_by` (user) | `tenant_id` (owner tenant) |
| Headless session | `_HEADLESS_CLIENT_ID` | `tenant_id` (inherited from spawning context — see §5.3) |
| `TokenLedger` entry | `session_id` only | `tenant_id`, `user_id` |
| `HandoffGate` | name only | `tenant_id` (`_daemon` for daemon-wide gates) |
| MCP server config | shared | per-tenant namespace |
| Permission grants (whitelist/blacklist) | per-session | per-tenant policy + per-session override |
| **Plugin state** (memory, todo, references) | per-session | **out of scope for v1** — see §7.4 |

### 5.2 Default tenant

When `--multi-tenant` is off, every resource is created with
`tenant_id="default"`. This makes the migration path straightforward:
existing data files get a `"tenant_id": "default"` field added on
first read; new resources stamp it on creation.

### 5.3 Headless session attribution

A reactor that calls `create_headless_session` declares the tenant
context. Three sources, in priority order:

1. **Triggering session** — if the reactor fires in response to an event
   from session X, the new session inherits `X.tenant_id`.
2. **Explicit override** — `create_headless_session(tenant_id=...)` for
   reactors that need to spawn into a specific tenant (e.g. cross-tenant
   admin tooling — requires the reactor's service identity to have
   the relevant `(write, session)` perm in the target tenant).
3. **Daemon-wide** — `tenant_id="_daemon"` for true daemon-wide work
   (memory-advisor consolidating daemon-owned state). The `_daemon`
   tenant is treated identically to any other tenant by the authz
   service — only identities with explicit `_daemon` membership see
   its resources. There is no special-case code path.

A reactor that omits all three in multi-tenant mode raises at spawn
time. No silent default.

### 5.4 Listing & enumeration

`session.list`, `gate.list`, `ledger.summary` etc. become tenant-scoped
by default. The query result is filtered by the caller's accessible
tenants, computed from their roles (§6).

## 6. RBAC

### 6.1 Role model

Roles are config-driven and split across two files with different
ownership and change cadence:

- `.jaato/rbac/roles.json` — **role definitions** (permissions per
  role). Owned by platform engineering; changes rarely.
- `.jaato/rbac/assignments.json` — **user → membership** mapping.
  Owned by HR / SSO sync; changes often. May be sourced from an IdP
  instead of a file (the IdP's group-to-role mapper writes
  memberships at auth time).

Default roles shipped with the daemon:

| Role | Scope | Permissions | Notes |
|---|---|---|---|
| `tenant-user` | self | `read`, `attach`, `write`, `delete` on own sessions; `read` ledger for self; `read` gate for own-tenant gates | Default for any human user |
| `tenant-user-admin` | tenant | Manages **memberships** within a tenant — add/remove users, change role assignments | Carved out from the old `tenant-admin` |
| `tenant-data-admin` | tenant | All `tenant-user` perms across the tenant; `list, session` tenant-wide; `read, ledger` tenant-wide | Sees every user's work in the tenant |
| `operator` | global | Read-only across all tenants: `read, session`, `list, session`, `read, gate`, `read, ledger-summary`, `read, audit` | Monitors, supports, troubleshoots |
| `support-engineer` | global | Operator perms plus `attach, session` (read-only attach — see §7.2) | Can observe live sessions during support |
| `billing-reader` | global | `read, ledger-summary` only (no `read, ledger-detail` — token counts, not prompt content) | For finance / FinOps |
| `service` | per-instance | Configured per service identity in §6.5 | For reactors, peer daemons |

`superuser` is **not** a default role. There is no role that grants
write actions across tenants in the standard catalogue. Cross-tenant
write is achieved via **break-glass elevation** (§6.6), which is
time-bounded, justified, and audited at a higher tier.

A role definition file:

```jsonc
// .jaato/rbac/roles.json
{
  "billing-reader": {
    "scope": "global",
    "permissions": [
      ["read", "ledger-summary"]
    ]
  },
  "support-engineer": {
    "scope": "global",
    "permissions": [
      ["read", "session"],
      ["read", "gate"],
      ["read", "audit"],
      ["attach", "session"]
    ]
  }
}
```

Assignment file:

```jsonc
// .jaato/rbac/assignments.json
{
  "alice@example.com": [
    {"tenant_id": "acme", "roles": ["tenant-data-admin"]}
  ],
  "ops@example.com": [
    {"tenant_id": "_global", "roles": ["operator", "billing-reader"]}
  ]
}
```

Global roles attach to the synthetic `_global` tenant — keeping the
data shape uniform (everything is a membership) and avoiding
"global-or-tenant-id" branches.

### 6.2 Authorization service

A daemon-singleton `AuthorizationService` is the single choke point for
authz decisions:

```python
class AuthorizationService:
    def authorize(
        self,
        identity: ClientIdentity,
        action: str,
        resource_type: str,
        resource_tenant: Optional[str] = None,
        resource_owner: Optional[str] = None,
    ) -> AuthzDecision:
        """Return ALLOW or DENY plus structured reasoning.
        
        resource_tenant=None means "global" resource (e.g. memory-advisor gate).
        resource_owner=None means "no specific user" (a tenant-level resource).
        """
```

`AuthzDecision` carries `(allowed: bool, reason: str, matched_rule: str)`
so denials are debuggable and audit logs are precise.

### 6.3 Enforcement points

Every scoped operation calls the service. The natural enforcement points:

| Point | Action checked | File:line today |
|---|---|---|
| `session.attach` command | `(attach, session)` against the target's `tenant_id` | `command_router.py:147` |
| `session.list` command | filter results by `(list, session)` per tenant | `command_router.py` (list handler) |
| `inject_prompt_to_session` | `(write, session)` against target's `tenant_id` | `session_manager.py:1149` |
| `session.new` | `(write, session)` in caller's tenant | `command_router.py:285` |
| `subagent_spawn` | `(write, session)` in parent's tenant | `subagent/plugin.py` |
| `set_active_tenant` | `(switch, tenant)` — caller must hold a membership in target | new |
| Reactor handler entry | `(execute, reactor)` against the reactor's service identity | new — see §6.5 |
| `create_headless_session` from reactor | `(write, session)` in target tenant against reactor's identity | `session_manager.py:1053` |
| EventBus fan-out | per-event filter when actor.tenant ≠ session.tenant (see §7.2) | `session_manager.py:446` |
| `HandoffGate` event delivery | `(read, gate)` per subscriber per event | `handoff-gate-api.md` |
| Ledger query | `(read, ledger-summary)` or `(read, ledger-detail)` filtered by tenant | `token_accounting.py` |
| MCP tool call | `(invoke, mcp_tool)` filtered by tenant namespace | `mcp_context_manager.py` |

### 6.4 Conflict resolution & rule precedence

A user may hold multiple roles in one membership. Decision rules:

1. **Default deny.** No matching rule → DENY.
2. **Explicit deny wins.** A role with `permissions_deny: [...]` blocks
   even other roles that would allow. This is how tenant-level deny
   policies (e.g. "tenant X bans `Bash(rm)`") cannot be overridden.
3. **Most specific scope wins ties.** `self` > `tenant` > `global`. A
   `tenant-user` rule on the user's own session takes precedence over
   a `global` rule that also matches.
4. **Audit on conflict.** When two rules match with different
   decisions, the audit record carries `matched_rule` *and*
   `dissenting_rules` so operators can debug policy.

### 6.5 Default-deny in multi-tenant mode

In `--multi-tenant`:

- Missing identity → DENY with reason `"unauthenticated"`.
- Identity present but no matching rule → DENY with reason
  `"no rule matched"`.
- Cross-tenant action without an explicit cross-tenant permission →
  DENY with reason `"cross-tenant action requires explicit role"`.

In single-tenant: the synthetic identity's `_local-superuser` role
matches every action; the service is still consulted (so the code path
is uniform) but never denies. Audit logging is off unless explicitly
enabled.

### 6.6 Service identities

Every reactor, peer daemon, and admin tool runs under a **service
identity**. Service identities are the answer to "who is acting?" when
no human is in the loop, and they're authzed identically to user
identities — the only differences are how they authenticate and how
their assignments are managed.

#### 6.6.1 Declaration

A reactor declares its service identity in its extension manifest:

```toml
# pyproject.toml of a reactor extension
[project.entry-points."jaato.extensions"]
memory_advisor = "jaato_premium.memory_advisor:create_extension"

[tool.jaato.service_identity]
name = "memory-advisor"
default_tenant = "_daemon"
required_permissions = [
  ["read", "session"],
  ["write", "session"],     # spawns headless sessions
  ["read", "memory"],
  ["write", "memory"],
]
```

At extension load time the daemon:

1. Reads the manifest's `[tool.jaato.service_identity]` block.
2. Looks up `~/.jaato/service_tokens.json` for a token under the
   declared `name`. If absent and in single-tenant mode, generates one
   and writes it; in multi-tenant mode, refuses to load with a clear
   error pointing the operator to `jaato-admin service-token create`.
3. Constructs a `ClientIdentity(is_service=True, ...)` and binds it to
   the extension's runtime context.

The reactor uses `context.service_identity` instead of any caller's
identity when calling `create_headless_session`,
`inject_prompt_to_session`, or any other authzed surface.

#### 6.6.2 Manifest vs assignment

`required_permissions` in the manifest is a **declaration** of what
the reactor needs. The actual grant lives in `assignments.json` like
any other identity. On load, the daemon checks declaration ⊆ grant; if
the grant is short, the reactor refuses to start with a clear error
listing the missing permissions. This makes capability changes
explicit — bumping a reactor's permissions is a deliberate config
change, not a drive-by code edit.

#### 6.6.3 Reactor execution authz

When a reactor handler fires on an event, the daemon wraps the
invocation in an `(execute, reactor)` authz check. This is mostly a
no-op in normal operation (the service identity has the perm by
declaration) but it's the hook for:

- Disabling a misbehaving reactor without uninstalling it
  (`jaato-admin reactor disable memory-advisor`).
- Per-tenant reactor allow-lists (a tenant can opt out of a daemon's
  shared reactors).
- Audit attribution for reactor actions.

#### 6.6.4 Subagents & nested reactors

A subagent inherits the parent session's tenant — its writes are
attributed to the parent. A reactor that spawns a session that itself
triggers a reactor: each layer's actions are audited under the
*invoking* identity, not the user who started the chain. The audit
record carries a `chain` field tracing back to the original human
caller for debugging.

### 6.7 Break-glass elevation

Cross-tenant write actions (rare, dangerous) are not granted
statically. Instead, an operator with `(elevate, tenant)` permission
on a target tenant can request a **break-glass elevation**:

```
client.request_elevation(
    target_tenant="acme",
    actions=[("write", "session"), ("delete", "session")],
    duration_seconds=900,
    justification="incident #12345 — purging runaway agent",
)
```

The daemon:

1. Records the request in the audit log at tier WARN.
2. Optionally requires a second approver (configurable per tenant).
3. Grants a time-bounded role (`break-glass:tenant=acme:expires=...`)
   for the requested duration.
4. Logs every action taken under the elevation at tier WARN with the
   elevation request ID.
5. On expiry, revokes the role and emits a `BreakGlassExpiredEvent`.

This replaces the `superuser` role for normal operation. A daemon
without any operator with `(elevate, *)` simply has no path to
cross-tenant write — that's the secure default.

## 7. Per-Resource Walk-Through

### 7.1 Sessions

- `Session.tenant_id` added; persisted in session JSON.
- `attach_session(id)` flow:
  1. Resolve target session.
  2. `authz.authorize(identity, "attach", "session", target.tenant_id, target.created_by)`.
  3. If allowed, add caller to `attached_clients`.
- `session.list` returns only sessions where `(list, session)` is
  permitted. Operators see all; tenant users see only their own.
- `session.delete`: requires `(delete, session)` against the target.

### 7.2 Events

EventBus fan-out remains by `attached_clients`, but the model has two
tiers based on whether the actor and the session share a tenant:

**Same-tenant attach (cheap path).** When `actor.active_tenant ==
session.tenant_id`, the attach authz check is the only gate. The full
event stream flows through unfiltered. This is the dominant case
(tenant users attaching to their own sessions, tenant-data-admins
attaching within their tenant) and adding per-event checks would be
pure overhead.

**Cross-tenant attach (filtered path).** When `actor.active_tenant !=
session.tenant_id` — an operator or support-engineer attaching across
tenants — every event is authz-checked individually. Sensitive event
types (`MessageEvent` with raw prompt content, tool arguments,
completion payloads) are either redacted or dropped based on the
actor's `(read, session-content)` perm. Operational events
(`AgentCompletedEvent` summary, `TokenUsageEvent`,
`PermissionRequestedEvent`) flow through unredacted.

The split lets cross-tenant actors monitor session health without
seeing user prompts. A `support-engineer` with explicit
`(read, session-content)` (e.g. granted via break-glass) gets the
unfiltered stream; without it, they see only operational events.

Per-event audit records for cross-tenant attach include the actor and
the event type but not the content — the audit log itself stays out of
the data path.

### 7.3 Ledger

- Each ledger entry carries `tenant_id` and `user_id`.
- `ledger.summary(tenant_id=X)` consults `(read, ledger)` against
  tenant X. Operators get all; users get self.
- Existing ledger code writes to a single JSONL; the migration adds a
  `tenant_id` field to each entry. Single-tenant entries get
  `tenant_id="default"`.

### 7.4 Plugin state (deferred to v2)

Plugin *config* is per-tenant in v1: which MCP servers are available,
which memory namespace the session uses, which references catalogue.
This is enough to give tenants distinct *capabilities* without
auditing every plugin's storage layer.

Plugin *state* (memory entries, todos, references, knowledge bundles)
stays **per-session** in v1. Cross-session aggregation within a tenant
(e.g. memory-advisor consolidating across sessions) is deferred to v2
because:

- Each plugin has its own storage layout — memory has raw + curated,
  todo has list-of-items, references has fragment files. Auditing all
  for cross-tenant leakage is a separate workstream.
- The natural primary key today is `session_id`, not `tenant_id`. A
  schema change that adds `tenant_id` to every record file is a large
  migration with risk that v1 doesn't need.
- A tenant-scoped session doesn't leak per-session state to other
  tenants today, because attach is gated. The remaining concern is
  cross-tenant *aggregation* (a memory consolidator running across
  multiple users in one tenant), which is a feature, not a regression.

When v2 adds tenant-scoped aggregation, the schema change adds
`tenant_id` to plugin records, and aggregating plugins (memory-advisor)
gain a tenant filter. v1 doesn't promise this and doesn't break it.

### 7.5 HandoffGate

Three changes to the gate doc (`handoff-gate-api.md`):

1. **Gates carry `tenant_id`.** Every gate is tenant-scoped. Daemon-wide
   gates use the synthetic `_daemon` tenant — no special-case branch.
2. **Event delivery is filtered.** `GateAnnouncedEvent` /
   `GateReleasedEvent` deliver only to subscribers where
   `(read, gate)` is permitted for that gate's tenant. Same-tenant
   subscribers see the full intent; cross-tenant subscribers see only
   `public_intent_fields`.
3. **Public intent fields.** Each gate definition declares
   `public_intent_fields: Set[str]`. Gate intents are split into
   public (operational health) and private (workload content).
   Operators with `(read, gate)` see public fields across tenants;
   private fields require same-tenant `(read, gate)` or break-glass
   elevation.

This addresses the §9 visibility concern in the gate doc directly,
without introducing a `tenant_id=None` special case.

### 7.6 MCP servers

- Per-tenant MCP namespace: `.jaato/mcp/<tenant_id>/.mcp.json`.
- A global namespace at `.jaato/mcp/.mcp.json` provides shared servers
  (e.g. a corporate Atlassian MCP).
- Tool calls resolve in this order: tenant namespace → global namespace.
- Tenant admins can manage their tenant's namespace; operators can
  inspect global config.

### 7.7 Permissions (whitelist/blacklist)

Tool permission grants stay per-session today. The change:

- Tenant-level policy can pre-populate a session's whitelist/blacklist
  (e.g., a tenant defaults `Bash(rm)` to deny).
- Session-level grants can never *expand* beyond what the tenant
  policy permits — a tenant-level deny is final.
- Composition: `effective = session_grant ∩ tenant_policy`.

### 7.8 Workspace & AppArmor

Already per-WS-client (§see existing `websocket-workspace-isolation.md`).
The change: provisioning consults the caller's tenant so the workspace
root layout becomes:

```
{workspace_root}/sessions/{tenant_id}/{session_id}/
```

AppArmor profiles deny sibling sessions *and* sibling tenants.

## 8. Audit

### 8.1 What gets audited

- Every cross-tenant action by a non-tenant-user identity.
- Every action under a break-glass elevation (tier WARN, with
  elevation request ID).
- Every `mutate` action on tenant policy, RBAC config, or auth state.
- Every authorization denial (rate-limited; not for unauthenticated
  request floods).
- Every reactor `(execute, reactor)` invocation when the reactor's
  service identity has cross-tenant scope.

In single-tenant deployments audit is **off** by default; opt in via
`--audit-log` if needed.

### 8.2 Audit record

```python
@dataclass
class AuditRecord:
    timestamp: datetime
    actor: ClientIdentity
    action: str
    resource_type: str
    resource_id: Optional[str]
    resource_tenant: Optional[str]
    decision: Literal["allow", "deny"]
    reason: str
    matched_rule: Optional[str]
    request_metadata: Dict[str, Any]   # source IP, command name, args summary
```

Persisted to `.jaato/audit/{YYYY-MM}.jsonl`, append-only. Read access
requires `(read, audit)` — operators yes, tenant users no.

### 8.3 Streaming

Audit records emit `AuditRecordEvent` on a separate event channel,
subscribable only by clients with `(read, audit)`. This lets an audit
collector or SIEM ingest in real time.

## 9. SDK Surface

### 9.1 Python

```python
class IPCClient:
    @property
    def identity(self) -> ClientIdentity:
        """The identity the daemon assigned this connection."""

    async def set_active_tenant(self, tenant_id: str) -> None:
        """Switch the active tenant (must hold a membership)."""

    async def list_sessions(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[SessionInfo]:
        """List sessions visible to this identity. Filtered server-side."""

    async def list_tenants(self) -> List[TenantInfo]:
        """List tenants this identity has memberships in.
        Operators with global scope see all."""

    async def request_elevation(
        self,
        target_tenant: str,
        actions: List[Tuple[str, str]],
        duration_seconds: int,
        justification: str,
    ) -> ElevationGrant:
        """Request a break-glass elevation. Returns a grant token
        scoped to the actions and duration; subsequent commands
        within the window run under the elevated permissions."""
```

### 9.2 TypeScript

Mirror surface:

```typescript
interface Membership {
  tenantId: string;
  roles: Set<string>;
}

interface ClientIdentity {
  userId: string;
  memberships: Membership[];
  activeTenant: string;
  authMethod: string;
  isService: boolean;
}

class IpcClient {
  readonly identity: ClientIdentity;
  setActiveTenant(tenantId: string): Promise<void>;
  listSessions(opts?: { tenantId?: string }): Promise<SessionInfo[]>;
  listTenants(): Promise<TenantInfo[]>;
  requestElevation(opts: ElevationRequest): Promise<ElevationGrant>;
}
```

## 10. Migration

### 10.1 Phase 0 — schema additions (single-tenant compatible)

- Add `tenant_id` field to all persisted resources, default `"default"`.
- Add `ClientIdentity` type, deprecate bare `user_id`.
- Existing data: load-time migration adds `"tenant_id": "default"` to
  any record missing it.

No behavior change. Single-tenant deployments work unchanged.

### 10.2 Phase 1 — RBAC service, single-tenant short-circuit

- Add `AuthorizationService` and the policy file format.
- Wire enforcement points but short-circuit to ALLOW in single-tenant.
- Audit log writes records but is functionally read-only via roles
  that don't yet exist in single-tenant.

This phase ships the surface without behavior change; tests verify
authz decisions are correct on synthetic identities.

### 10.3 Phase 2 — multi-tenant mode opt-in

- `--multi-tenant` daemon flag.
- When enabled: identity required, default-deny, audit on by default.
- Public-tier `local-users` and `service-token` backends shipped in
  the public daemon (§4.4). Multi-tenant works without premium.
- Premium SSO extension wires real OIDC identity.
- Per-tenant directory layout for workspaces, MCP namespaces.

### 10.4 Phase 3 — admin tooling

- `jaato-admin` CLI for tenants, users, role assignments, service
  tokens, break-glass approvals.
- Operator dashboard via SDK.
- Tenant import/export.

### 10.5 Phase 4 — plugin state per-tenant (deferred from v1)

- Add `tenant_id` to memory, todo, references record schemas.
- Migrate aggregating plugins (memory-advisor) to filter by tenant.
- Lazy migration on read; write-back on next mutation.

## 11. Open Questions

1. **Tenant hierarchy.** Some orgs want "parent tenant inherits child
   visibility." Ship flat tenants in v1; add hierarchy if demand
   surfaces.

2. **Policy file reload.** Edit `.jaato/rbac/{roles,assignments}.json`
   → daemon re-reads on SIGHUP? File watcher? Static-only? Recommend
   file watcher with atomic reload, similar to the openers.json
   pattern.

3. **Audit retention.** Append-only JSONL grows unbounded. Rotation
   policy: keep N days online, archive older to compressed format,
   support external sink (Splunk, Loki, S3). v2.

4. **Per-tenant rate limiting.** Adjacent to RBAC, often discussed
   together. Probably its own design doc; integrates via the same
   identity surface.

5. **Capability tokens for explicit cross-tenant grants.** A user
   grants temporary read access to their session for a support
   engineer without going through break-glass. Capability tokens
   with TTL? v2 candidate; break-glass covers the urgent case for v1.

6. **Ledger split: summary vs detail.** The roles table already
   distinguishes `read, ledger-summary` from `read, ledger-detail`
   (token counts vs prompt content). Concrete schema split deferred
   to implementation: probably two JSONL files.

7. **Multi-daemon RBAC.** Peer cluster of daemons with shared tenants.
   Federation model? Each daemon trusts identities from a common IdP.
   Out of scope for v1.

8. **Service identity rotation.** Service tokens are static. For
   long-lived deployments, periodic rotation is needed. Add a
   `jaato-admin service-token rotate` command and a grace window
   where the old token still works. v2.

9. **Reactor opt-out per tenant.** A tenant should be able to disable
   a daemon-shared reactor (e.g. memory-advisor) for privacy or
   policy reasons. Implement as a tenant config flag the reactor
   checks at handler entry; defer the UI / admin command to v2.

## 12. Out of Scope

- Quota / rate limiting per tenant (separate doc).
- Encryption at rest for tenant data.
- GDPR data deletion workflow (right to erasure) — addressable on top
  of the schema additions.
- Cross-daemon federation.
- Network isolation (separate ports per tenant).

## 13. References

- `docs/design/daemon-extensions.md` — extension surface that hosts
  premium SSO middleware
- `docs/design/websocket-workspace-isolation.md` — existing workspace
  isolation
- `docs/design/handoff-gate-api.md` — gate visibility consumes this
  design
- `docs/design/task-graph-reactor.md` — orchestrators consume per-tenant
  identity
- `docs/apparmor-setup.md` — kernel-level isolation that complements
  RBAC
- `jaato-server/server/websocket.py:603` — existing `set_client_user`
  hook for premium SSO
- `jaato-server/server/session_manager.py:86` — `Session.created_by`
- `jaato-server/server/event_sink.py:49` — event sink identity surface
