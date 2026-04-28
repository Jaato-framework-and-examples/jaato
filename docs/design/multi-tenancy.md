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
| **Identity** | The authenticated principal: `(user_id, tenant_id, roles)`. Established by auth middleware, stamped on `ClientConnection`. |
| **Tenant** | A scope unit. Owns sessions, ledger entries, plugin state. A user belongs to exactly one tenant in v1; multi-tenant membership is v2. |
| **Role** | A named set of permissions. Roles are tenant-scoped or global. |
| **Permission** | A `(action, resource_type)` tuple, e.g. `(read, session)`, `(write, gate)`. |
| **Scope** | The breadth of a permission: `self`, `tenant`, `global`. |
| **Resource** | A scoped object: a session, a gate, a ledger entry, a plugin record. Always carries an owner tenant. |
| **Action** | A verb: `read`, `attach`, `write`, `delete`, `list`, `mutate`. |

## 4. Identity Model

### 4.1 Existing surfaces (kept)

- `ClientConnection.user_id: Optional[str]` — set by auth middleware
  via `set_client_user(client_id, user_id)`.
- `Session.created_by: Optional[str]` — stamped at session creation.

### 4.2 Extended `ClientIdentity`

Replace the bare `user_id` with a structured identity object:

```python
@dataclass(frozen=True)
class ClientIdentity:
    user_id: str
    tenant_id: str
    roles: FrozenSet[str]                    # role names, e.g. {"tenant-user"}
    auth_method: str                         # "bearer", "oidc", "ipc-local"
    authenticated_at: datetime
    expires_at: Optional[datetime] = None    # for OIDC tokens

    @property
    def is_anonymous(self) -> bool:
        return self.user_id == ANONYMOUS_USER_ID
```

`ClientConnection.identity: Optional[ClientIdentity]` replaces
`user_id`. `set_client_user(client_id, user_id)` becomes
`set_client_identity(client_id, identity)`. The legacy method is kept
as a shim that constructs a `ClientIdentity(user_id=..., tenant_id="default", roles={"tenant-user"})`.

### 4.3 Single-tenant default

When the daemon starts without `--multi-tenant`:

- All connections receive a synthetic identity:
  `ClientIdentity(user_id="local", tenant_id="default", roles={"superuser"}, auth_method="ipc-local")`.
- Auth middleware is bypassed.
- Authorization checks short-circuit to allow.

This keeps every existing deployment working with zero config changes.

### 4.4 Multi-tenant mode

When the daemon starts with `--multi-tenant`:

- Connections without identity are rejected at the transport layer.
- IPC connections require identity from a local-auth source (SSO ticket
  file, local socket peer cred lookup, or premium SSO).
- WS connections require OIDC or bearer-with-identity-binding.
- Anonymous connections receive `ClientIdentity(roles=frozenset())` and
  can do nothing — every authz check denies.

## 5. Tenancy Model

### 5.1 Resource ownership

Every scoped resource gains a `tenant_id` field:

| Resource | Existing field | New field |
|---|---|---|
| `Session` | `created_by` (user) | `tenant_id` (owner tenant) |
| Headless session | `_HEADLESS_CLIENT_ID` | `tenant_id` (inherited from spawning context — see §5.3) |
| `TokenLedger` entry | `session_id` only | `tenant_id`, `user_id` |
| `HandoffGate` | name only | `tenant_id` (or `global=True` for daemon-wide gates like `memory-advisor`) |
| Plugin state record (memory entry, todo, reference) | `session_id` | `tenant_id` |
| MCP server config | shared | per-tenant namespace |
| Permission grants (whitelist/blacklist) | per-session | per-tenant policy + per-session override |

### 5.2 Default tenant

When `--multi-tenant` is off, every resource is created with
`tenant_id="default"`. This makes the migration path straightforward:
existing data files get a `"tenant_id": "default"` field added on
first read; new resources stamp it on creation.

### 5.3 Headless session attribution

A reactor that calls `create_headless_session` must declare the tenant
context. Three sources, in priority order:

1. **Triggering session** — if the reactor fires in response to an event
   from session X, the new session inherits `X.tenant_id`.
2. **Explicit override** — `create_headless_session(tenant_id=...)` for
   reactors that need to spawn into a specific tenant (e.g. cross-tenant
   admin tooling).
3. **Daemon-wide** — `tenant_id=None` for true daemon-wide work
   (memory-advisor consolidating *only* daemon-owned state). These
   sessions are visible only to operators with `global` scope.

A reactor that omits all three in multi-tenant mode raises at spawn
time. No silent default.

### 5.4 Listing & enumeration

`session.list`, `gate.list`, `ledger.summary` etc. become tenant-scoped
by default. The query result is filtered by the caller's accessible
tenants, computed from their roles (§6).

## 6. RBAC

### 6.1 Role model

Roles are config-driven. Default roles ship with the daemon and can be
extended via a policy file at `.jaato/rbac/policy.json`.

| Role | Scope | Permissions |
|---|---|---|
| `tenant-user` | self | `(read, session)`, `(attach, session)`, `(write, session)`, `(delete, session)` on **own sessions**; `(read, ledger)` for self; `(read, gate)` for own-tenant gates |
| `tenant-admin` | tenant | All `tenant-user` perms across the tenant; `(list, session)` tenant-wide; `(read, ledger)` tenant-wide |
| `operator` | global | `(read, session)`, `(list, session)`, `(read, gate)`, `(read, ledger)`, `(read, audit)` across **all tenants**. **No write actions.** |
| `superuser` | global | `*` — every action, every tenant. Use sparingly; every action is audited. |
| `service` | tenant or global | For non-human callers (reactors, peer daemons). Permissions configured per service identity. |

Custom roles compose these primitives. A role definition:

```jsonc
// .jaato/rbac/policy.json
{
  "roles": {
    "billing-reader": {
      "scope": "global",
      "permissions": [
        ["read", "ledger"],
        ["read", "session"]
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
  },
  "user_roles": {
    "alice@example.com": ["tenant-admin"],
    "ops@example.com": ["operator", "billing-reader"]
  }
}
```

User → role mapping comes from either:

- The policy file's `user_roles` map (simple deployments).
- An identity provider claim (premium SSO maps OIDC `groups` → role names).

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
| EventBus fan-out | `(read, session)` per attached client per event | `session_manager.py:446` |
| `HandoffGate` event delivery | `(read, gate)` per subscriber per event | proposed in `handoff-gate-api.md` |
| Ledger query | `(read, ledger)` filtered by tenant | `token_accounting.py` |
| Plugin state read | `(read, <plugin>)` filtered by tenant | per-plugin |

### 6.4 Default-deny in multi-tenant mode

In `--multi-tenant`:

- Missing identity → DENY with reason `"unauthenticated"`.
- Identity present but no matching rule → DENY with reason
  `"no rule matched"`.
- Cross-tenant action without `global` scope role → DENY with reason
  `"cross-tenant action requires operator or superuser"`.

In single-tenant: the synthetic `superuser` identity short-circuits all
checks to ALLOW. The service is still consulted (for audit log
consistency) but never denies.

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

- EventBus fan-out remains by `attached_clients`, but adding a client to
  `attached_clients` is now itself an authz check (§7.1).
- This means once you're attached you see everything from that session
  — no per-event filtering needed for normal events. The model is:
  *attach is the gate; once through, the stream is open*.
- Exception: cross-tenant events (gate events with global scope) need
  per-event filtering. See §7.5.

### 7.3 Ledger

- Each ledger entry carries `tenant_id` and `user_id`.
- `ledger.summary(tenant_id=X)` consults `(read, ledger)` against
  tenant X. Operators get all; users get self.
- Existing ledger code writes to a single JSONL; the migration adds a
  `tenant_id` field to each entry. Single-tenant entries get
  `tenant_id="default"`.

### 7.4 Plugin state

- Memory, todo, references, etc. each store records that today key on
  `session_id`. The data layer adds `tenant_id` to record schemas.
- Memory consolidation (raw → curated) is per-tenant. The
  memory-advisor reactor reads only its tenant's raw queue. For
  daemon-wide memory (e.g. system-prompts the daemon offers all
  tenants), a separate `global` namespace exists.
- Plugin queries from a session are scoped to `session.tenant_id`
  automatically — the plugin asks the runtime for the active tenant
  at query time.

### 7.5 HandoffGate

Three changes to the gate doc (`handoff-gate-api.md`):

1. **Gates carry `tenant_id`.** Daemon-wide gates set `tenant_id=None`.
2. **Event delivery is filtered.** `GateAnnouncedEvent` /
   `GateReleasedEvent` deliver only to subscribers where
   `(read, gate)` is permitted for that gate's tenant.
3. **Anonymous gate intent.** When a gate is announced cross-tenant
   (an operator inspecting), the `intent` payload may omit fields
   marked sensitive in the gate's schema. The gate definition declares
   `public_intent_fields: Set[str]` and operators see only those unless
   they have `superuser`.

This addresses the §9 visibility concern in the gate doc directly.

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
- Every `superuser` action, period.
- Every `mutate` action on tenant policy, RBAC config, or auth state.
- Every authorization denial (rate-limited; not for unauthenticated
  request floods).

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
    
    async def list_sessions(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[SessionInfo]:
        """List sessions visible to this identity. Filtered server-side."""
    
    async def list_tenants(self) -> List[TenantInfo]:
        """List tenants this identity can see. Tenant users see one;
        operators see all."""
```

### 9.2 TypeScript

Mirror surface:

```typescript
interface ClientIdentity {
  userId: string;
  tenantId: string;
  roles: Set<string>;
  authMethod: string;
}

class IpcClient {
  readonly identity: ClientIdentity;
  listSessions(opts?: { tenantId?: string }): Promise<SessionInfo[]>;
  listTenants(): Promise<TenantInfo[]>;
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
- When enabled: identity required, default-deny, full audit.
- Premium SSO extension wires real OIDC identity.
- Per-tenant directory layout for workspaces, MCP, plugin state.

### 10.4 Phase 3 — admin tooling

- `jaato-admin` CLI for managing tenants, users, role assignments.
- Operator dashboard via SDK.
- Tenant import/export.

## 11. Open Questions

1. **Multi-tenant membership.** v1 is one user → one tenant. Real
   orgs need users in multiple tenants (a contractor working across
   two clients). Add a `tenant_memberships: Set[(tenant_id, role_set)]`
   field and let the client pick an active tenant per session?
   Probably yes in v2.

2. **Tenant hierarchy.** Some orgs want "parent tenant inherits child
   visibility." Ship flat tenants in v1; add hierarchy if demand
   surfaces.

3. **Service identities for reactors.** A reactor isn't a human user.
   Should reactors get a `service` role with their own identity?
   Probably yes — gives audit log clarity and tenant-scoped reactors
   for cross-tenant operations.

4. **Policy file reload.** Edit `.jaato/rbac/policy.json` → daemon
   re-reads on SIGHUP? File watcher? Static-only? Recommend file
   watcher with atomic reload, similar to the openers.json pattern.

5. **Audit retention.** Append-only JSONL grows unbounded. Rotation
   policy: keep N days online, archive older to compressed format,
   support external sink (Splunk, Loki, S3). v2.

6. **Per-tenant rate limiting.** Adjacent to RBAC, often discussed
   together. Probably its own design doc; integrates via the same
   identity surface.

7. **Cross-tenant explicit grants.** A user grants temporary read
   access to their session for a support engineer. Capability tokens
   with TTL? Defer to v2.

8. **Ledger redaction.** Operators reading the ledger see token
   counts but should they see the prompts? Probably not by default.
   Add a `ledger.read_redacted` permission for non-superuser global
   read.

9. **Identity expiry mid-session.** OIDC token expires while a
   session is running. Renew on next request? Force reconnect?
   Affects long-running sessions. Suggest: best-effort renew,
   degraded mode if refresh fails (sessions continue but writes
   blocked until re-auth).

10. **Multi-daemon RBAC.** Peer cluster of daemons with shared tenants.
    Federation model? Each daemon trusts identities from a common IdP.
    Out of scope for v1.

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
