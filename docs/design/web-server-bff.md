# jaato-web-coder-server: sign-in and ticket custody for the browser client

**Status:** implemented. The daemon half is #1074 (PR #1075, protocol
1.10, merged); the server is `jaato-web-coder-server/` in this repository
(`direct` mode, OIDC; local users and `proxy` mode are not built), with
its deployment artifacts in `jaato-web-coder-server/deploy/`. This
document remains the rationale; the package README is the operator's
reference.

## 1. What this is, and what it is not

`jaato-web-coder-ui` (the browser client, `@jaato/web-coder-ui`) connects to a daemon started
with `--web-socket` and presents whatever credential it was given. Today
that credential is the daemon's single bearer token: it says *may drive this
daemon* and nothing about who. That is the right credential for the
single-user local case `npx @jaato/web-coder-ui` serves, and the wrong one for a
deployment where several people share a daemon.

`jaato-web-coder-server` is the server-side counterpart of the browser client.
It is a **backend-for-frontend**: the application that authenticated the
user, in the sense #1074 uses the word. It owns three things the browser
must never own:

| It owns | Because |
|---|---|
| **sign-in** (OIDC, or a local user table) | the daemon deliberately learns nothing about realms (#1074 §"Why teach the daemon about realms is the wrong fix") |
| **the app credential** | the long-lived secret that may call `ticket.bind`; it never reaches a page |
| **ticket minting** | one short-lived, single-use ticket per connection, bound to the signed-in user |

It also serves the static bundle, the way the `jaato-web-coder-ui` launcher does.

It is **not**:

- a feature of the daemon. The daemon stays coding-agnostic; the only daemon
  work is #1074, which is application-neutral by design.
- a WebSocket proxy, in its default mode. With #1074 the browser can connect
  to the daemon **directly** with its ticket, so the BFF is out of the data
  path. A proxy mode exists for topologies where the daemon is not reachable
  from browsers (§6), and is the exception.
- an authorization layer. With identity stamped on the connection, "who may
  see which session" is the daemon's question to answer, keyed on the
  qualified identity #1074 gives it. The BFF does not inspect frames.

## 2. The flow

```mermaid
sequenceDiagram
    participant B as Browser (jaato-web-coder-ui)
    participant S as jaato-web-coder-server (BFF)
    participant I as Identity provider
    participant D as jaato daemon

    S->>D: WS Upgrade, Authorization: Bearer <app credential>
    Note over S,D: the bind channel, held open for the BFF's lifetime

    B->>S: GET /
    S-->>B: index.html, config.json {ticketUrl, daemon, autoConnect}
    B->>S: GET /api/session
    S-->>B: 401 (no cookie)
    B->>S: GET /auth/login
    S->>I: OIDC authorization redirect
    I-->>S: /auth/callback?code=…
    S->>I: token exchange
    S-->>B: Set-Cookie: session (HttpOnly, Secure, SameSite=Lax)

    B->>S: POST /api/ticket  (cookie, Sec-Fetch-Site: same-origin)
    S->>D: ticket.bind {request_id, user: "alice", ttl_seconds: 60, single_use: true}
    D-->>S: ticket.bind.result {status: bound, ticket, qualified: "jaato-web-coder:alice", expires_at}
    S-->>B: {ticket, daemon}

    B->>D: WS Upgrade ?token=<ticket>
    D->>D: resolve ticket → BoundIdentity{app_id: "jaato-web-coder", user: "alice"}, consume it
    D-->>B: ConnectedEvent (connection attributed to jaato-web-coder:alice)

    Note over B,D: every session this connection creates carries created_by = "jaato-web-coder:alice"

    B--xD: connection drops
    B->>S: POST /api/ticket  (fresh ticket; the old one was consumed)
    B->>D: WS Upgrade ?token=<new ticket>
```

Two properties the diagram is built around:

- **The app credential never leaves the BFF.** The browser only ever sees
  tickets, and a ticket is worth one connection for one minute.
- **A ticket is consumed at connect.** So a reconnect needs a new one, which
  is why the SDK needs a token *provider* rather than a token (§5.1).

## 3. Components

```
jaato-web-coder-server/                  new package, @jaato/web-coder-server on npm
  bin/jaato-web-coder-server.js          CLI entry: reads config, starts the server
  src/
    config.ts                      typed config (§7), env + file, secrets from files
    bind-channel.ts                one JaatoClient to the daemon, app-credential auth,
                                   ticket.bind / ticket.revoke request-response
    auth/
      oidc.ts                      openid-client: login redirect, callback, token exchange
      local.ts                     users file (argon2 hashes) for deployments without an IdP
      session.ts                   cookie session store (in-memory; pluggable)
    routes.ts                      /auth/*, /api/session, /api/ticket, /api/logout
    static.ts                      serves @jaato/web-coder-ui's dist/ (the launcher's static server, reused)
    proxy.ts                       optional WS relay (§6)
```

Runtime dependencies are deliberately few: `@jaato/web-coder-ui` (the bundle and the
static server), `@jaato/sdk` (the bind channel is an ordinary client), and
`openid-client`. The local-user mode needs a password hash; `argon2` is
optional and only loaded when that mode is configured.

The static server is the one the launcher already ships. `@jaato/web-coder-ui` gains
an `exports` entry so `createStaticServer` is importable, and the BFF passes
it a `config` object whose `ticketUrl` replaces the launcher's `token`.

## 4. The daemon-side contract (#1074, implemented in PR #1075)

Everything in this section is read from PR #1075's branch
(`server/ws_tickets.py`, `server/websocket.py`, `jaato_sdk/events.py`),
not assumed. Protocol **1.10**.

### 4.1 Credentials

| | BFF's use |
|---|---|
| **app credential** | one entry in the daemon's `--ws-app-credentials` JSON file, `{"jaato-web-coder": "<credential>"}` (mode 0600 enforced, at least 16 characters, the key is the `app_id`). The BFF presents it once, on the bind channel's Upgrade, as `Authorization: Bearer` (Node can set the header; no query string). It is **bind-only**: the daemon refuses every frame on that connection other than the two ticket verbs, so it can never open or attach a session |
| **user ticket** | minted per connect through `ticket.bind`, handed to the browser in a JSON response body (never a URL), presented by the browser as `?token=` on its own Upgrade, exactly where the shared token goes today. Consumed at accept when `single_use` (the default) |

Configuring `--ws-app-credentials` turns WS auth **on**, and the flag is
refused alongside `--ws-unsafe-no-auth`. The shared `--ws-token-file` may
stay configured beside it for the TUI and the local launcher; the daemon
tries the shared token, then app credentials, then tickets, hashing the
presented value once.

**Only the app that bound a ticket may revoke it.** The daemon scopes
`ticket.revoke` to the `app_id` of the credential on the channel, and a
ticket belonging to another application answers `not_found`, the same as an
unknown one, so the verb is not an existence oracle across applications.

### 4.2 Wire shapes

Four events, declared in `events.py` and codegen'd into `@jaato/sdk`
(`TicketBindRequest`, `TicketBindResultEvent`, `TicketRevokeRequest`,
`TicketRevokeResultEvent`), all correlated by `request_id` on the protocol
1.3 precedent so one bind channel serves many concurrent logins:

```jsonc
// BFF → daemon
{"type": "ticket.bind", "request_id": "r1",
 "user": "alice", "ttl_seconds": 60, "single_use": true}

// daemon → BFF   (status: bound | denied | invalid | capacity)
{"type": "ticket.bind.result", "request_id": "r1", "status": "bound",
 "ticket": "…", "qualified": "jaato-web-coder:alice", "app_id": "jaato-web-coder",
 "expires_at": "2026-09-15T20:01:00Z"}

// BFF → daemon, on logout: exactly ONE of `ticket` / `user`
{"type": "ticket.revoke", "request_id": "r2", "user": "alice"}

// daemon → BFF   (status: revoked | not_found | denied | invalid)
{"type": "ticket.revoke.result", "request_id": "r2", "status": "revoked", "revoked": 1}
```

- Failure is a `status` on the result event, never a separate error frame,
  and `ticket` is `""` in every non-`bound` result. The BFF branches on
  `status` and reads `ticket` only under `bound`.
- `app_id` is absent from the request: the daemon derives it from the
  credential on the channel and returns both `app_id` and `qualified`, so
  the BFF logs what the daemon will stamp rather than reconstructing it.
  `app_id` may not contain `:`, the join character of `qualified`.
- `user` is the claim the BFF is configured to use (`auth.oidc.subject_claim`,
  §7). Against Keycloak the default is `preferred_username`: unique within
  the realm, qualified by the daemon anyway, and readable in an audit
  (`jaato-web-coder:alice`, not `jaato-web-coder:2f1c9e0a-…`). `sub` is available for a
  deployment that renames users. The daemon refuses an empty, over-long or
  control-character `user` with `invalid`.
- `ttl_seconds` is `1..3600` and a value outside is `invalid`, never
  clamped. The BFF asks for **60**: it mints only in response to a connect
  attempt, so 60 covers the round trip and shortens the window a logged URL
  is useful. The daemon's own default is 300.
- `capacity` means the daemon is at its ceiling of outstanding tickets. The
  BFF answers the browser with 503 and a retry hint; it never retries in a
  loop, since the remedy is tickets expiring.
- **Logout after a completed login revokes nothing** by design: the ticket
  was consumed at connect, so `ticket.revoke {user}` answers `not_found`
  with `revoked: 0`. The BFF treats that as success. What the revoke
  actually protects against is a ticket minted and not yet presented.

### 4.3 The bind channel is a plain `JaatoClient`

The BFF opens it with `@jaato/sdk`'s `JaatoClient`, `headers: {Authorization:
"Bearer <app credential>"}` and **no `clientConfig`**: the client sends
`client.config` after the handshake only when that option is set, and an
app-credential connection answers any non-ticket frame with an error. The
two verbs go through `sendRawEvent` and are matched to results by
`request_id` on `subscribeAll`. The client's reconnect loop keeps the
channel up across daemon restarts; a bind attempted while it is down fails
fast and the browser retries its connect, which re-mints.

### 4.4 What the BFF needs from the identity

`created_by` and the permission-decision `user_id` are stamped from the
connection's identity, and PR #1075 makes `set_client_user` **refuse to
overwrite** a ticket-established identity, so a later SSO hook cannot
silently relabel the connection. The BFF relies on the qualified form
(`jaato-web-coder:alice`) being what those fields carry, so that:

- two BFF deployments against one daemon, each with its own app credential
  (`jaato-web-coder-eu:alice`, `jaato-web-coder-us:alice`), cannot collide;
- daemon-side ownership guards, when they exist in the free package, compare
  the right thing. Today the free daemon *records* identity and never
  compares it (`session.attach` checks id syntax and workspace mismatch only;
  `session.list` returns everything). #1074 makes the identity available at
  the point those checks would go; adding the checks is a follow-up the BFF
  design does not depend on, but a multi-user deployment does.

### 4.5 Where bindings live

In memory, in the daemon. The BFF is compatible with that because it never
holds a ticket across a daemon restart: a restart drops the bind channel,
the BFF's client reconnects it, and the next browser connect mints a fresh
ticket. A ticket bound on one clustered node not resolving on another is
the one consequence the BFF cannot paper over; a deployment fronting a
gossip cluster needs the BFF pinned to the node its browsers reach, or the
registry shared.

## 5. What changes in the existing packages

### 5.1 `@jaato/sdk` (jaato-sdk-ts): a token provider

`JaatoClientOptions.token` is a `string`, and `_openOnce` presents the same
value on every reconnect. A single-use ticket makes every reconnect fail
after the first. The change is small and backwards compatible:

```ts
token?: string | (() => Promise<string>);
```

`_openOnce` resolves the provider before each `openTransport`. A provider
that throws aborts that attempt and lets the existing backoff schedule the
next, so a BFF that is briefly down degrades into the reconnect loop the SDK
already has rather than into a dead connection. The Python SDK has no such
need today (no browser presents tickets through it), but the same shape is
harmless there.

### 5.2 `@jaato/web-coder-ui` (jaato-web-coder-ui): a ticket URL beside the token

`config.json` gains one optional field; the launcher keeps writing `token`,
the BFF writes `ticketUrl`:

```jsonc
{"daemon": "wss://jaato.example.org", "ticketUrl": "./api/ticket", "autoConnect": true}
```

- `launcherConfig.ts` parses it; `connection.ts` turns it into a provider:
  `fetch(ticketUrl, {method: "POST", credentials: "same-origin"})` → `ticket`.
- A 401 from the ticket endpoint means the cookie session is gone, so the
  connect screen becomes a **sign-in** screen: a "Sign in" button that
  navigates to `./auth/login` (a top-level navigation, not a fetch, because
  the OIDC redirect must set the cookie). After the callback the page lands
  back on `/` and auto-connects.
- Logout is a link to `./api/logout`, which clears the cookie, asks the
  daemon to revoke the user's tickets, and, when the IdP supports it,
  redirects through RP-initiated logout.
- `@jaato/web-coder-ui` exports its static server (`exports` in `package.json`), so
  the BFF does not copy it.

None of this changes the launcher's behaviour: with `token` present and no
`ticketUrl`, the client does exactly what it does today.

### 5.3 The daemon

Only #1074. The BFF adds nothing to `server/`.

## 6. Modes

| Mode | Browser connects to | When |
|---|---|---|
| `direct` (default) | the daemon, with its ticket | the daemon's `--web-socket` port is reachable from browsers, typically behind the same TLS terminator as the BFF |
| `proxy` | the BFF at `/ws`, which relays to the daemon | the daemon sits on a private network, or a single public origin is required |

In `proxy` mode the BFF still mints a ticket per browser connection and
opens the upstream connection **with that ticket**, never with the app
credential, so attribution is identical in both modes. The relay is an
opaque pipe, binary frames included; it applies no policy. The one thing the
BFF does in this mode that it cannot in `direct` is cap connections per user.

## 7. Configuration

```yaml
# jaato-web-coder-server.yaml
listen: 0.0.0.0:8443
tls: {cert: /etc/jaato-web-coder/tls.crt, key: /etc/jaato-web-coder/tls.key}   # or terminate in front
public_url: https://jaato.example.org

daemon:
  url: wss://jaato.example.org/daemon       # what the BROWSER connects to (direct mode)
  bind_url: ws://127.0.0.1:8080             # what the BFF connects to; defaults to url
  app_id: jaato-web-coder                       # the key of this BFF's entry in the daemon's --ws-app-credentials file
  app_credential_file: /etc/jaato-web-coder/app.credential   # its value; mode 0600 enforced on both sides
mode: direct                                 # direct | proxy

auth:
  kind: oidc                                 # oidc | local
  oidc:
    issuer: https://jaato.example.org/auth/realms/jaato-web-coder-shell   # the `iss` Keycloak puts in tokens
    backchannel_url: http://127.0.0.1:8180             # optional: discovery, token, JWKS over loopback (§11)
    client_id: jaato-web-coder
    client_secret_file: /etc/jaato-web-coder/oidc.secret
    scopes: [openid, profile]
    subject_claim: preferred_username        # what becomes ticket.bind's `user`
    required_role: jaato-user                # optional: realm or client role that gates sign-in (§11)
  local:
    users_file: /etc/jaato-web-coder/users.yaml    # {name: argon2 hash}

session:
  secret_file: /etc/jaato-web-coder/session.secret
  ttl: 8h
  cookie_name: jaato_web_coder_session

ticket:
  ttl_seconds: 60
```

Secrets are files, never inline values or argv, for the reason
`--ws-token-file` is preferred over `--ws-token` in the daemon: argv is
world-readable through `/proc`.

The daemon side of the same pairing:

```bash
# /etc/jaato/ws-apps.json, mode 0600:  {"jaato-web-coder": "<the same credential>"}
python -m server --web-socket 127.0.0.1:8080 --ws-app-credentials /etc/jaato/ws-apps.json --daemon
```

`--ws-token-file` may stay beside it so the TUI and `npx @jaato/web-coder-ui` keep
working with the shared token on the same daemon.

## 8. Security properties, each with the attack it answers

| Property | Answers |
|---|---|
| the ticket endpoint requires the session cookie **and** `Sec-Fetch-Site: same-origin` (or a matching `Origin`) | a cross-site page cannot mint a ticket with the victim's cookie (CSRF) |
| tickets are returned in a JSON body and used once, within 60 s | a ticket that lands in a proxy log or browser history is dead by the time anyone reads it |
| the app credential is read from a 0600 file and sent only as a header on the bind channel | it never appears in a URL, a page, or `/proc/<pid>/cmdline` |
| session cookie is `HttpOnly; Secure; SameSite=Lax` | page script cannot read it; `Lax` is what lets the OIDC callback's top-level redirect carry it |
| per-user and per-IP rate limits on `/api/ticket` | a script minting tickets in a loop |
| logout revokes the user's outstanding tickets at the daemon | a ticket minted just before logout cannot open a connection after it |
| the BFF holds no daemon token in `direct` mode beyond the bind channel | compromising the BFF's process yields the ability to mint tickets, not a session on every user's behalf — still serious, and why the app credential should be rotatable |

What it does **not** give, stated so nobody reads it in: OS-level separation
between users. Every session still runs as the daemon's uid, as #1074's
"What it does not do" says.

## 9. Phasing and publishing order

The server package depends on `@jaato/sdk` (protocol 1.10 events and
the token provider) and on `@jaato/web-coder-ui` (the bundle and its static
server) **from npm**, so the first publish of each has to happen in this
order: **SDK, then UI, then server**. Until the SDK's first publish,
`npx @jaato/web-coder-ui` cannot work at all (its `npx` has nothing to
fetch), and the server can only be built from a checkout. The server's
publish workflow enforces the order: its checkout links the siblings with
`file:` for development, `scripts/prepare-publish.mjs` rewrites them to
caret ranges at publish time, and the workflow refuses to publish unless
both exact versions are already on the registry.

All three workflows **stage** rather than publish: the `@jaato` token is
a "Read and write (stage only)" granular token, because npm is retiring
direct publish by token in January 2027, and `npm stage publish` uploads
the version non-public until a maintainer with 2FA approves it (`npm stage
approve <stage-id> --otp <code>`, or on npmjs.com). A staged version does
not answer the "is it on npm" probe, so the order above now includes the
approvals: stage the SDK, approve, stage the UI, approve, stage the server,
approve.

One exception, measured on the UI's first run: npm refuses to stage a
package it has never seen (`404 Not Found - POST /-/stage/package/<name>`).
The **first** version of each new package is therefore published directly
by a maintainer with 2FA from a checkout at the release commit (each
package's README has the exact commands), and the workflows refuse by name
while the registry does not know the package. From the second version on,
everything stages.

**The guard checks that a version exists, not what it contains.** A sibling
whose source changed since its last publish must have its version bumped
first, or the range the server publishes with resolves to a build missing
what the server needs. That is the state the registry was in when the
server package landed: `@jaato/sdk` 0.6.0 had been published four months
earlier and predates both the token provider (§5.1) and the transport fix
that puts custom headers where Node's WebSocket reads them, so a server
published against `^0.6.0` would have had its bind channel refused as
anonymous. The SDK's first usable version for this package is **0.7.0**.

1. **#1074 lands** (PR #1075) with the two verbs from §4.2.
2. **SDK token provider** (§5.1). Small, independently testable, and useful
   to any client that rotates credentials.
3. **`jaato-web-coder-ui` ticket URL + sign-in screen** (§5.2), tested with a mock BFF
   in Playwright the way the launcher's `config.json` path is today.
4. **`jaato-web-coder-server` in `direct` mode, OIDC only.** Done: the
   package, its `init` command and `deploy/` (systemd units, Caddy and nginx
   configs). Verified against the real daemon (`npm run test:daemon`): bind
   → ticket → an attributed connection, the ticket refused on replay, a
   wrong app credential refused at the Upgrade, revoke-after-login honestly
   `not_found`. Local users and `proxy` mode are still not built.
5. **Daemon-side ownership checks** keyed on the qualified identity, so a
   multi-user deployment is isolated and not merely attributed. Separate
   issue; not a BFF change.

## 10. Questions put to #1074, and their answers

Asked in the first version of this document; answered in
[the issue's comment](https://github.com/Jaato-framework-and-examples/jaato/issues/1074#issuecomment-5686953669).

| Question | Answer |
|---|---|
| `ticket.bind` / `ticket.revoke` as request/response with a `request_id`? | yes, on the protocol 1.3 `request_id` precedent |
| `app_id` absent from the request, derived from the credential? | yes, and it is the load-bearing half of the design |
| `qualified` returned in `ticket.bound`? | yes |
| events declared in `events.py`, codegen'd into the TS SDK? | yes; protocol 1.10 |
| app credential bind-only? | yes |
| *(raised by the implementer)* who may revoke a ticket? | only the app that bound it, enforced by the daemon from the credential |
| a ticket bound on one clustered node resolving on another? | **still open**; §4.5 states how the BFF behaves either way |

Two details PR #1075 settled differently from this document's first draft,
now corrected in §4: the result events are `ticket.bind.result` /
`ticket.revoke.result` carrying a `status` field (not `ticket.bound` /
`ticket.revoked` plus a separate error frame), and `ticket.revoke` takes
exactly one of `ticket` / `user` rather than `user` alone.

What remains this design's to build, in the order of §9: the SDK token
provider (`jaato-sdk-ts` is outside #1074's scope), the `ticketUrl` path
and sign-in screen in `jaato-web-coder-ui`, and the `jaato-web-coder-server` package.

## 11. Deployment with Keycloak on the same host

The first target deployment runs Keycloak on the host the BFF runs on. That
fixes several choices that a generic-IdP design leaves open.

Keycloak is a **native install** on that host, not a container, and so is
everything beside it: the daemon is a `pip`-installed `jaato-server`, the
BFF an `npm`-installed `@jaato/web-coder-server`, and the front is the
host's reverse proxy. So the deployment artifact is **systemd units and a
proxy config**, not a compose file (§11.7).

### 11.1 One public origin, two internal ports

```
                     https://jaato.example.org
                    ┌──────────────────────────┐
  browser ─────────►│ reverse proxy (TLS)      │
                    │  /            → BFF :8443 (plain HTTP behind the proxy)
                    │  /auth/       → Keycloak :8180
                    │  /daemon      → daemon --web-socket 127.0.0.1:8080  (WebSocket upgrade)
                    └──────────────────────────┘
                                 │ loopback
                    BFF ──ticket.bind──► daemon    (bind channel, app credential)
                    BFF ──token/JWKS──► Keycloak   (back channel, §11.2)
```

- The daemon binds loopback only and is reached by browsers through the
  proxy's `/daemon` route, so the BFF's `daemon.url` is
  `wss://jaato.example.org/daemon` and its `daemon.bind_url` is
  `ws://127.0.0.1:8080`. `direct` mode (§6) is what this topology wants: the
  proxy already gives a single origin, so the BFF stays out of the frame path.
- Keycloak's public URL is under the same origin, so the OIDC redirects are
  same-site and the session cookie's `SameSite=Lax` carries through the
  callback with no exceptions.
- The proxy, not the BFF, terminates TLS. `listen`/`tls` in §7 stay for the
  deployment without a proxy.

### 11.2 Front channel and back channel

The browser reaches Keycloak at its public URL; the BFF can reach it over
loopback. Keycloak stamps `iss` with the **public** URL (`KC_HOSTNAME`), and
OIDC requires the BFF to validate tokens against that same issuer, so the
two must not be confused:

| | URL | Used for |
|---|---|---|
| front channel | `https://jaato.example.org/auth/realms/jaato-web-coder-shell` | authorization redirect, RP-initiated logout; also the `issuer` the BFF validates against |
| back channel | `http://127.0.0.1:8180` (`auth.oidc.backchannel_url`) | discovery document, token exchange, JWKS, userinfo |

Keycloak supports this split when started with `--hostname-backchannel-dynamic=true`
(Keycloak 25+), which lets it answer back-channel requests on the internal
address while keeping the public `iss`. When `backchannel_url` is unset the
BFF uses the issuer URL for everything, which is correct for a Keycloak
reached only through the proxy. Plain HTTP on the back channel is acceptable
only because it is loopback; the BFF refuses an `http://` back channel that
is not a loopback address.

### 11.3 Client registration in Keycloak

The application's realm is **`jaato-web-coder-shell`**: a realm of its own,
so its users, roles and sessions are separate from any other application's
on the same Keycloak, and so `app_id` on the daemon side (§4.1) has a
one-to-one counterpart on the identity side. One confidential client in it:

| Setting | Value |
|---|---|
| Client ID | `jaato-web-coder` |
| Client authentication | on (confidential); the secret goes in `client_secret_file` |
| Standard flow | on; direct access grants and implicit flow **off** |
| PKCE code challenge method | `S256` (the BFF always sends PKCE; this makes Keycloak require it) |
| Valid redirect URIs | `https://jaato.example.org/auth/callback` only |
| Valid post-logout redirect URIs | `https://jaato.example.org/` |
| Back-channel logout URL | `https://jaato.example.org/auth/backchannel-logout` (§11.5) |
| Back-channel logout session required | on |

### 11.4 Who may sign in at all

Keycloak puts realm roles in `realm_access.roles` and client roles in
`resource_access.jaato-web-coder.roles` on the ID token. `auth.oidc.required_role`
names one; a user whose token lacks it is refused at the callback with a
page saying so, and no ticket is ever minted for them. This is the one piece
of authorization the BFF does cheaply and correctly, because it is decided
once at sign-in from a claim the IdP signed. It answers *"may this person use
jaato through this deployment"*, not *"which session may they see"*, which
remains the daemon's question (§4.3).

### 11.5 Logout, both directions

- **User-initiated.** `/api/logout` clears the cookie, sends `ticket.revoke`
  for the user, then redirects to Keycloak's `end_session_endpoint` with
  `id_token_hint` and the post-logout URI, so the Keycloak SSO session ends
  too and the next visit prompts for credentials.
- **Keycloak-initiated (back-channel).** When an administrator logs a user
  out or the SSO session expires, Keycloak POSTs a signed logout token to
  `/auth/backchannel-logout`. The BFF validates it against the JWKS, ends the
  matching cookie session, and revokes the user's tickets. Without this, an
  administrator's "log out everywhere" would not reach a jaato tab. The
  browser's live WebSocket is **not** cut by either path: the ticket was
  consumed at connect and the connection is the daemon's. Cutting it needs a
  daemon-side "close connections for identity X", which #1074 does not
  include; recorded here as the gap it is.

### 11.6 What the design does not assume from Keycloak

Nothing beyond OIDC with PKCE, the standard `end_session_endpoint`, and the
OIDC back-channel logout spec. The two Keycloak-specific claims read
(`realm_access` / `resource_access`) are behind the `required_role` knob and
unused when it is unset, so a second deployment on another IdP loses only
that knob.

### 11.7 Process supervision and the deployment artifacts

Three services under systemd, each in the foreground under its unit rather
than self-daemonising:

| Unit | Runs | Note |
|---|---|---|
| `keycloak.service` | Keycloak's own `kc.sh start` with `--hostname` set to the public URL and `--hostname-backchannel-dynamic=true` | the host's existing install |
| `jaato-server.service` | `python -m server --web-socket 127.0.0.1:8080 --ws-app-credentials /etc/jaato/ws-apps.json` **without `--daemon`** | `--daemon` double-forks, which fights systemd's process tracking; `Type=simple` and let systemd own it |
| `jaato-web-coder-server.service` | `jaato-web-coder-server --config /etc/jaato-web-coder/server.yaml` | `DynamicUser=` or a dedicated user; the secret files are readable by that user only |

Plus the reverse proxy's site config (`/`, `/auth/`, `/daemon` with the
WebSocket upgrade), and one provisioning step that is otherwise the most
error-prone part of the setup: the app credential must be byte-identical in
the daemon's `--ws-app-credentials` file and the BFF's
`app_credential_file`, both mode 0600. `jaato-web-coder-server init`
generates the credential, writes the BFF's side, and prints the daemon-side
JSON entry for the operator to paste, so nobody types a 43-character
secret twice.

These ship in a `deploy/` directory of the server package — the unit
files, a Caddyfile and an nginx equivalent, and the `init` command — as
part of phase 4 (§9). They are the deployment mechanism; nothing here needs
a container, and the repository has no Dockerfile for these components.
