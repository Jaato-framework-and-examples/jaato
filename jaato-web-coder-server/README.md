# jaato-web-coder-server

Sign-in and ticket custody for the jaato browser client
([`@jaato/web-coder-ui`](../jaato-web-coder-ui)). It is the **application**
in the sense of jaato's per-user tickets (#1074, protocol 1.10): people
sign in here through Keycloak (any OpenID Connect issuer works), and for
every connection the browser makes, this server asks the daemon for a
short-lived, single-use ticket bound to that person. The daemon then
attributes the connection — `created_by`, permission decisions, the
ledger — to `jaato-web-coder:<user>`, and the daemon's own credential
never leaves this process.

The design, and the reasons behind each choice, are in
[`docs/design/web-server-bff.md`](../docs/design/web-server-bff.md).

```
browser ──sign in──► this server ──OIDC──► Keycloak
browser ──POST /api/ticket (cookie)──► this server ──ticket.bind (app credential)──► daemon
browser ──WebSocket ?token=<ticket>──► daemon        (direct mode: this server is not in the data path)
```

## Run

```bash
# once: generate the app credential + session secret (0600) and a config template
jaato-web-coder-server init --dir /etc/jaato-web-coder
#   → prints the daemon-side entry for --ws-app-credentials

# edit /etc/jaato-web-coder/server.yaml, paste the Keycloak client secret into oidc.secret, then
jaato-web-coder-server serve --config /etc/jaato-web-coder/server.yaml
```

The daemon must run with `--ws-app-credentials <file>` naming this
application; the full host layout (systemd units, Caddy/nginx, Keycloak
client registration) is in [`deploy/`](deploy/README.md).

Until `@jaato/web-coder-server` is on npm, install from a checkout:
`npm ci && npm run build` here, then run `bin/jaato-web-coder-server.js`.

## Publishing

The checkout's `package.json` is `private` and links `@jaato/sdk` and
`@jaato/web-coder-ui` with `file:` so one `npm install` wires a checkout
with nothing on npm; neither may reach the registry. The *Publish
@jaato/web-coder-server to npm* workflow
(`.github/workflows/publish-npm-web-coder-server.yml`) builds against
those links, then runs `scripts/prepare-publish.mjs`, which rewrites the
two dependencies to the caret range of the version each sibling checkout
declares and drops `private`, and **stages** the version (`npm stage
publish`; the `@jaato` token is stage-only, so a maintainer with 2FA
promotes it with `npm stage approve <stage-id>`). It
**refuses** unless both of those exact versions are already on npm — a
staged-but-unapproved sibling is not — so the order SDK → UI → server is
enforced rather than remembered, approval included.

npm cannot stage a package it has never seen, so the **first** version has
to be published directly by a maintainer with 2FA, from a checkout at the
release commit, with the same manifest rewrite the workflow does:

```bash
cd jaato-web-coder-server
npm --prefix ../jaato-sdk-ts ci && npm --prefix ../jaato-sdk-ts run build
npm --prefix ../jaato-web-coder-ui ci
npm ci && npm run build
node scripts/prepare-publish.mjs             # file: links -> caret ranges, drops private
npm publish --access public     # after `npm login`; the 2FA step opens
                                #   the browser (passkey) or asks for a code
git checkout package.json                    # the rewrite is not for committing
```

The workflow refuses by name while the package is unknown to the registry;
every later version stages. `npm pack` on the development manifest
fails by design (`prepack` runs `scripts/check-publishable.mjs`);
`node scripts/prepare-publish.mjs --dry-run` shows what would ship.

That refusal checks that the sibling **version** is on npm, not that the
published build matches the checkout. If a sibling's source changed since
its last publish, bump its version and publish it first; otherwise the
caret range resolves to the stale build. The bind channel needs
`@jaato/sdk` **0.7.0 or later** (the token provider and the header fix for
Node's built-in WebSocket); the 0.6.0 on npm predates both.

## Configuration

`server.yaml`; every secret is a file beside it, mode 0600 (looser is
refused, not read):

| Key | Meaning |
|---|---|
| `listen` | `host:port` this server binds (loopback, behind the proxy) |
| `public_url` | the origin browsers use; the OIDC redirect URI is `<public_url>/auth/callback` |
| `daemon.url` | what the **browser** connects to (`wss://…/daemon` through the proxy) |
| `daemon.bind_url` | what **this server** connects to for the bind channel (default `daemon.url`) |
| `daemon.app_id` | the key of this application's entry in the daemon's `--ws-app-credentials` file |
| `daemon.app_credential_file` | its value |
| `auth.oidc.issuer` | the issuer as it appears in tokens (`https://…/auth/realms/jaato-web-coder-shell`) |
| `auth.oidc.backchannel_url` | optional loopback URL for discovery / token / JWKS; plain `http` only on loopback |
| `auth.oidc.client_id`, `client_secret_file` | the confidential client |
| `auth.oidc.subject_claim` | the ID-token claim that becomes the daemon-side user (default `preferred_username`) |
| `auth.oidc.required_role` | optional realm or client role that gates sign-in |
| `session.secret_file`, `ttl`, `cookie_name` | the HttpOnly, SameSite=Lax session cookie |
| `ticket.ttl_seconds` | lifetime asked of the daemon per ticket (default 60; the daemon accepts 1..3600) |

## Routes

| Route | Method | |
|---|---|---|
| `/config.json` | GET | `{daemon, ticketUrl, loginUrl, autoConnect}` — what the bundle reads |
| `/auth/login` | GET | 302 to the issuer (PKCE S256, state, nonce) |
| `/auth/callback` | GET | finishes sign-in, sets the cookie, 302 to `/` |
| `/auth/backchannel-logout` | POST | OIDC back-channel logout: ends matching sessions, revokes tickets |
| `/api/session` | GET | `{user, expiresAt}` or 401 |
| `/api/ticket` | POST | one single-use ticket; same-origin only (`Sec-Fetch-Site` / `Origin`) |
| `/api/logout` | GET | ends the session, revokes, redirects through the issuer's logout |
| everything else | GET | the bundle |

## Develop

```bash
npm ci && npm run build       # builds ../jaato-sdk-ts declarations, then tsc → dist/
npm test                      # node:test — config, sessions, bind channel (mock daemon), routes (fake IdP), 26 cases
npm run test:daemon           # the same against the REAL daemon; needs ../.venv/bin/python with jaato-server
                              # (or JAATO_DAEMON_PYTHON=…), skips otherwise
```
