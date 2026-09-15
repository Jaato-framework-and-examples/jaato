# jaato-web-server: sign-in and ticket custody for the browser client

**Status:** design, blocked on #1074, whose implementation is in progress
(scoped to jaato-server and the Python SDK). The wire shapes in §4 were
proposed here and **confirmed** in
[#1074's answer comment](https://github.com/Jaato-framework-and-examples/jaato/issues/1074#issuecomment-5686953669);
the one point still open is §4.4 (clustered daemons). Everything downstream
of §4 is written against the confirmed shape.

## 1. What this is, and what it is not

`jaato-web` (the browser client, `@jaato/web`) connects to a daemon started
with `--web-socket` and presents whatever credential it was given. Today
that credential is the daemon's single bearer token: it says *may drive this
daemon* and nothing about who. That is the right credential for the
single-user local case `npx @jaato/web` serves, and the wrong one for a
deployment where several people share a daemon.

`jaato-web-server` is the server-side counterpart of the browser client.
It is a **backend-for-frontend**: the application that authenticated the
user, in the sense #1074 uses the word. It owns three things the browser
must never own:

| It owns | Because |
|---|---|
| **sign-in** (OIDC, or a local user table) | the daemon deliberately learns nothing about realms (#1074 §"Why teach the daemon about realms is the wrong fix") |
| **the app credential** | the long-lived secret that may call `ticket.bind`; it never reaches a page |
| **ticket minting** | one short-lived, single-use ticket per connection, bound to the signed-in user |

It also serves the static bundle, the way the `jaato-web` launcher does.

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
    participant B as Browser (jaato-web)
    participant S as jaato-web-server (BFF)
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
    S->>D: ticket.bind {user: "alice", ttl_seconds: 60, single_use: true}
    D-->>S: ticket.bound {ticket, qualified: "web:alice", expires_at}
    S-->>B: {ticket, daemon}

    B->>D: WS Upgrade ?token=<ticket>
    D->>D: resolve ticket → BoundIdentity{app_id: "web", user: "alice"}
    D-->>B: ConnectedEvent (connection attributed to web:alice)

    Note over B,D: every session this connection creates carries created_by = "web:alice"

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
jaato-web-server/                  new package, @jaato/web-server on npm
  bin/jaato-web-server.js          CLI entry: reads config, starts the server
  src/
    config.ts                      typed config (§7), env + file, secrets from files
    bind-channel.ts                one JaatoClient to the daemon, app-credential auth,
                                   ticket.bind / ticket.revoke request-response
    auth/
      oidc.ts                      openid-client: login redirect, callback, token exchange
      local.ts                     users file (argon2 hashes) for deployments without an IdP
      session.ts                   cookie session store (in-memory; pluggable)
    routes.ts                      /auth/*, /api/session, /api/ticket, /api/logout
    static.ts                      serves @jaato/web's dist/ (the launcher's static server, reused)
    proxy.ts                       optional WS relay (§6)
```

Runtime dependencies are deliberately few: `@jaato/web` (the bundle and the
static server), `@jaato/sdk` (the bind channel is an ordinary client), and
`openid-client`. The local-user mode needs a password hash; `argon2` is
optional and only loaded when that mode is configured.

The static server is the one the launcher already ships. `@jaato/web` gains
an `exports` entry so `createStaticServer` is importable, and the BFF passes
it a `config` object whose `ticketUrl` replaces the launcher's `token`.

## 4. The daemon-side contract this depends on (#1074)

Everything in this section is what the BFF **assumes** of #1074. Where the
issue leaves a decision open, the BFF's need is stated so the implementer
can weigh it.

### 4.1 Credentials

| | BFF's use |
|---|---|
| **app credential** | presented once, on the bind channel's Upgrade, as `Authorization: Bearer` (the BFF is Node, so the header form is available; no query string). Read from a file with mode 0600, never from argv |
| **user ticket** | minted per connect through `ticket.bind`, handed to the browser in a JSON response body (never a URL), presented by the browser as `?token=` |

**Confirmed: the app credential is bind-only.** The issue asked whether an
app credential alone should open a session; the answer is no. The BFF never
wants that: a session opened on the bind channel would be attributed to the
application, which is exactly the anonymous session the ticket mechanism
exists to make unreachable.

**Confirmed, and it falls out of the same argument: only the app that bound
a ticket may revoke it.** The daemon enforces this from the credential on
the channel, without being told, the same way it qualifies the identity. So
the BFF's `ticket.revoke` can only ever touch its own users' tickets, and a
second application on the daemon cannot log this one's users out.

### 4.2 Wire shapes (confirmed)

The issue sketched `TicketRegistry.bind/resolve/revoke/revoke_user` and said
the bind channel is a WS connection carrying a `ticket.bind` message. The
answer comment confirms the two verbs as request/response pairs with a
correlation id, following the in-tree precedent of protocol 1.3
(`InjectPromptRequest.request_id` + `inject_prompt.result`: one channel,
many concurrent callers, a reply that says what actually happened). They
are declared in `jaato-sdk/jaato_sdk/events.py`, so the codegen puts them
in `@jaato/sdk`'s `events.ts` and the CI staleness gate keeps the two in
step; the additions land as **protocol 1.10**, with a changelog entry
stating how each side degrades against an older peer.

```jsonc
// BFF → daemon
{"type": "ticket.bind", "request_id": "r1",
 "user": "alice", "ttl_seconds": 60, "single_use": true}

// daemon → BFF
{"type": "ticket.bound", "request_id": "r1",
 "ticket": "…", "qualified": "web:alice", "expires_at": "2026-09-15T20:01:00Z"}

// BFF → daemon, on logout
{"type": "ticket.revoke", "request_id": "r2", "user": "alice"}

// daemon → BFF
{"type": "ticket.revoked", "request_id": "r2", "count": 1}

// daemon → BFF, any failure
{"type": "error", "request_id": "r1", "error_type": "TicketBindError", "error": "…"}
```

- `app_id` is **absent from the request** on purpose: the daemon derives it
  from the credential that authenticated the channel, which is the point the
  issue makes about no integrator being able to forget the qualification.
- `user` is the BFF's stable subject: the OIDC `sub` (not `preferred_username`,
  which can change), or the local username. The daemon qualifies it.
- `ttl_seconds: 60` is what the BFF will ask for. The issue's default is 300;
  the BFF mints a ticket only in response to a connect attempt, so 60 covers
  the round trip with margin and shortens the window a logged URL is useful.
- `qualified` crosses the wire in `ticket.bound`, so the BFF logs what the
  daemon will stamp rather than reconstructing it.
- Because the events are typed, the BFF's bind channel is an ordinary
  `JaatoClient` using the generated types, not hand-written dicts. Until
  1.10 is published to npm, `jaato-web-server` builds against the sibling
  `jaato-sdk-ts` checkout the way `jaato-web` does.

### 4.3 What the BFF needs from the identity

`created_by` and the permission-decision `user_id` are stamped from
`get_client_user`, which #1074 sets at connect. The BFF relies on the
qualified form (`web:alice`) being what those fields carry, so that:

- two BFF deployments against one daemon, each with its own app credential
  (`web-eu:alice`, `web-us:alice`), cannot collide;
- daemon-side ownership guards, when they exist in the free package, compare
  the right thing. Today the free daemon *records* identity and never
  compares it (`session.attach` checks id syntax and workspace mismatch only;
  `session.list` returns everything). #1074 makes the identity available at
  the point those checks would go; adding the checks is a follow-up the BFF
  design does not depend on, but a multi-user deployment does.

### 4.4 Where bindings live

The issue leans in-memory. The BFF is compatible with that because it never
holds a ticket across a daemon restart: a restart drops the bind channel, the
BFF reconnects it, and the next browser connect mints a fresh ticket. A
ticket bound on one clustered node not resolving on another is the one
consequence the BFF cannot paper over; a deployment fronting a gossip cluster
needs the BFF pinned to the node its browsers reach, or the registry shared.

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

### 5.2 `@jaato/web` (jaato-web): a ticket URL beside the token

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
- `@jaato/web` exports its static server (`exports` in `package.json`), so
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
# jaato-web-server.yaml
listen: 0.0.0.0:8443
tls: {cert: /etc/jaato-web/tls.crt, key: /etc/jaato-web/tls.key}   # or terminate in front
public_url: https://jaato.example.org

daemon:
  url: wss://jaato.example.org:8080         # what the BROWSER connects to (direct mode)
  bind_url: ws://10.0.0.5:8080              # what the BFF connects to; defaults to url
  app_credential_file: /etc/jaato-web/app.credential   # mode 0600 enforced
mode: direct                                 # direct | proxy

auth:
  kind: oidc                                 # oidc | local
  oidc:
    issuer: https://login.example.org/realms/eng
    client_id: jaato-web
    client_secret_file: /etc/jaato-web/oidc.secret
    scopes: [openid, profile]
    subject_claim: sub                       # what becomes ticket.bind's `user`
  local:
    users_file: /etc/jaato-web/users.yaml    # {name: argon2 hash}

session:
  secret_file: /etc/jaato-web/session.secret
  ttl: 8h
  cookie_name: jaato_web_session

ticket:
  ttl_seconds: 60
```

Secrets are files, never inline values or argv, for the reason
`--ws-token-file` is preferred over `--ws-token` in the daemon: argv is
world-readable through `/proc`.

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

## 9. Phasing

1. **#1074 lands** with the two verbs from §4.2 (or their agreed shape).
2. **SDK token provider** (§5.1). Small, independently testable, and useful
   to any client that rotates credentials.
3. **`jaato-web` ticket URL + sign-in screen** (§5.2), tested with a mock BFF
   in Playwright the way the launcher's `config.json` path is today.
4. **`jaato-web-server` in `direct` mode, OIDC only.** Local users and proxy
   mode follow once the identity plumbing is proven end to end against a real
   daemon.
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
| a ticket bound on one clustered node resolving on another? | **still open**; §4.4 states how the BFF behaves either way |

What remains this design's to build, in the order of §9: the SDK token
provider (`jaato-sdk-ts` is outside #1074's scope), the `ticketUrl` path
and sign-in screen in `jaato-web`, and the `jaato-web-server` package.
