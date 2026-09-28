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
staged-but-unapproved sibling is not — so the server goes after the SDK
and the UI (which have no order between them), enforced rather than
remembered, approval included.

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
Node's built-in WebSocket).

## Configuration

`jaato_server.server.yaml`; every secret is a file beside it, mode 0600 (looser is
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
| `credentials.file`, `key_file` | optional: the per-user store of provider API keys (`src/credentials.ts`), encrypted at rest with a key derived from `key_file` (0600, 32+ chars). Absent = off: the bundle shows its plain key field |
| `github.file`, `key_file` | optional: the per-user store of GitHub grants and workspace bindings (`src/github.ts`), encrypted at rest like `credentials` (0600 `key_file`, 32+ chars). Absent = off |
| `github.client_id` | the **GitHub App**'s client id (non-secret; it rides the authorize URL) |
| `github.client_secret_file` | the GitHub App's client secret, 0600 |
| `github.workspace_root` | optional: the root managed workspaces live under, when THIS server can write there. Set = the bind writes `GH_TOKEN=app://github`, `.home/.gitconfig` and the GitHub guidance itself, contained within it (symlinks resolved). Unset, or a workspace outside it = the **daemon** writes them (`workspace.app_write`, daemon protocol 1.30), which is the only way when this server runs as an account that cannot reach the daemon's workspaces. The `.env` line is not optional: the daemon resolves `app://github` only when it finds it there, so a binding with no line gives sessions no `GH_TOKEN`. Every recorded binding is re-written when the bind channel (re)connects, so a binding made while no write was possible is repaired without picking the account again. The same pass drops a binding that can never work: one recorded by workspace name rather than path, and one whose workspace the daemon answers `not_found` for (deleted, or not this user's). Deleting a workspace is a browser-to-daemon verb this server never sees, so this is what keeps its binding from outliving it. Each drop is logged; a wrongly dropped one costs picking the account again |
| `github.oauth_base_url`, `api_base_url` | optional: point at a GitHub Enterprise host (default `https://github.com` / `https://api.github.com`) |
| `github.noreply_domain` | optional: the commit-email domain seeded into `.gitconfig` (default `users.noreply.github.com`) |
| `environment.tools` | optional block: the operator's allow-list, toolchain → versions, e.g. `node: ["22", "20"]`, `java: {versions: ["21"]}`. `node`, `go`, `bun`, `java`, `maven` and `gradle` can be listed. A major-only entry installs mise's latest of that line at bind time; list an exact version to pin it. Absent block = off: `config.json` names no `environmentUrl` and the page shows no toolchain chips |
| `environment.lsp` | pinned language-server versions: `basedpyright` (also what makes the `python` toolchain bindable), `typescript-language-server` (needs `typescript`), `gopls` (e.g. `v0.20.0`, or `{version, go}`: the Go it is built with, `latest` by default, since a gopls release needs a newer Go than many projects pin), `jdtls` (`java`, `max_heap`, `mirror`). Each entry is a version, or a mapping `{version, ...}` carrying that server's own keys. An unpinned server is never installed |
| `environment.paranoid` | `true` sets `MISE_PARANOID=1` for installs (stricter verification; default `false`, mise's own checksum verification still applies) |
| `environment.install_timeout` | per-step deadline for the plugin's installs (default `15m`, 30s to 2h) |
| `environment.state_file` | where "Not now" answers are remembered per user and workspace (default `environment-state.json`, 0600) |

### Keys a user has used before

Without the store, every new workspace's configure form asks for the
provider's API key again, because the daemon keeps it only in that
workspace's `.env`. With `credentials:` set, the form offers the keys this
user stored before (label and a masked hint, never the secret), plus "New
key…"; a chosen key is revealed to the page once and forwarded to the
daemon as `config.update`'s `api_key`, exactly as a typed one is. A
`<provider>-auth key …` typed at the prompt is filed too, under the
provider the daemon's `auth.setup` offer names. Owner is the OIDC `sub`.
The daemon and the SDK know nothing of this; it is application state,
like the session cookie. What the encryption buys and does not buy is in
`src/credentials.ts`.

### Connect GitHub, per user (#1227)

With a `github:` block a signed-in user can connect one or more GitHub
accounts and bind one to each workspace, so a session's `gh` / `git` acts
as *that* person on GitHub without any token touching the browser or the
workspace `.env` in cleartext. The token travels this server → daemon only,
resolved on demand at every session spawn (`app://github`, #1226).

Register a **GitHub App** (not an OAuth App) and enable **user-to-server
token expiration** (App settings → *Optional features* / *User authorization*
→ "Expire user authorization tokens"), so refresh tokens are issued:

- **Callback URL:** `<public_url>/auth/github/callback`
- **Permissions:** whatever your agents need — the user token's reach is the
  intersection of the user and what the App is *installed on*, so an org can
  scope which repositories agents may touch. `Contents: read/write` and
  `Pull requests: read/write` are typical; `metadata: read` is implied.
- **Request user authorization (OAuth) during installation:** on.
- Copy the **Client ID** into `github.client_id` and a generated **client
  secret** into the file named by `github.client_secret_file` (0600).

The token never reaches the browser: there is **no reveal route** for
GitHub (unlike a provider key), and the `github` provider is refused by the
`credentials` store for the same reason. Refresh tokens rotate; this server
serialises read-refresh-write per grant and re-reads after acquiring, so
concurrent session spawns collapse to one refresh (the #683 pattern).
*Disconnect* deletes the grant, revokes it at GitHub, and asks the daemon to
reload the user's loaded sessions so a live `gh` call then fails. Design:
[`docs/design/per-user-github-credentials.md`](../docs/design/per-user-github-credentials.md) §4–§6.

> The settings UI (a "Connect GitHub" entry and the workspace account
> dropdown) and the pasted fine-grained-PAT fallback are follow-ups; this
> server exposes the endpoints the UI will call.

### Toolchains, language servers and repository guidance (#1344)

With an `environment:` block, a workspace owner can bind toolchains to a
workspace from the page. **This server only holds the policy.** It never
installs, scans or writes anything in a workspace, so it works whatever
account it runs as, including beside a root daemon. The installs are done by
the web coder's toolchains plugin ([`plugin/`](plugin/README.md), shipped in
this package), which runs in each session's runner, as the session's account
and under its confinement. It must be installed into the DAEMON's Python
environment, with its AppArmor fragment: [`plugin/INSTALL.md`](plugin/INSTALL.md).

```yaml
environment:
  tools:
    node: ["22", "20"]
    go: ["1.23"]
    java:                        # long form: {versions: [...]}
      versions: ["21", "temurin-17"]   # any mise java spelling
    maven: ["3.9.9"]
    gradle: ["8.10"]
  lsp:
    basedpyright: "1.31.6"       # short form: just the version
    gopls: "v0.20.0"
    typescript-language-server:  # long form: the version and this server's own keys
      version: "4.4.0"
      typescript: "5.9.3"        # required: the typescript installed beside it
    jdtls:
      version: "1.40.0"          # an Eclipse milestone
      java: "21"                 # the JDK that RUNS jdtls (default 21)
      max_heap: 1G               # -Xmx (default 1G); see #806 below
      # mirror: https://...      # default https://download.eclipse.org/jdtls/milestones
  # install_timeout: 15m         # per install step, 30s..2h
  # paranoid: false              # MISE_PARANOID for every install
```

`workspace_root`, `mise` and `python` are refused by name: they configured
installs this process no longer runs.

**Who does what.**

| Party | Does |
|---|---|
| this server | answers `/api/environment` with the allow-list, this user's "Not now" answers, and the content of the **offer** (`.jaato/toolchain-offer.json`) |
| the page | stages the offer into the session's workspace through the daemon (`StageFilesRequest`); sends the plugin's `toolchain bind|unbind|scan|cancel` user command; reads `.jaato/environment.json` through the daemon (`workspace.file.fetch`) |
| the plugin, in the runner | installs, writes the derived files, proposes from the repositories' markers, contributes the instruction section, and turns a `command not found` into a hint for the model and the page's Bind chip |

A bind, as the plugin runs it:

1. `mise install <tool>@<version>` with HOME, XDG and mise's directories
   under `<ws>/.home`, a clean environment (only proxy and CA variables pass
   through), and `MISE_CEILING_PATHS=<ws>/.home` so a repository's own
   `mise.toml` cannot choose what is downloaded. Every step execs through the
   session's `//child` AppArmor transition, since an installer runs a
   vendor's code;
2. relative symlinks from the install's `bin/` into `<ws>/.home/.local/bin`,
   which is on every command's `PATH` (#1273/#1274). A file there that is not
   one of these links is never replaced;
3. the pinned language server: basedpyright into its own venv under
   `.home/.local/share/jaato-lsp/`, typescript-language-server with the linked
   `npm` (started through the linked `node`), gopls with
   `GOBIN=.home/.local/bin`, jdtls as a milestone checked against its
   published `.sha256` and started with `java -jar` on its own mise JDK
   (`lsp.jdtls.java`), heap capped at `lsp.jdtls.max_heap`, `-data` at the
   framework's `${jdtlsStateRoot}`;
4. the derived files: `.jaato/environment.json` (the record the page reads,
   also read by `get_environment(aspect="runtime")`),
   `.home/.config/mise/config.toml`, and `.lsp.json` (read by the `lsp`
   plugin, whose tools appear once a server is configured, #1345). A mise
   config or `.lsp.json` without the `jaato-managed` marker is the user's and
   is left alone.

The plugin runs one install at a time per workspace (a lock under
`.home/.cache`), and an install stops if its session ends. The page follows
the job by polling the manifest.

**Proposals and guidance** come from the plugin's scan of the workspace root
and its immediate subdirectories (`.nvmrc`, `package.json` engines, `go.mod`,
`.tool-versions`, `.java-version`, `pom.xml`, Gradle files, `pyproject.toml`,
…) at session start, and on `toolchain scan` (the Toolchains section's
*Rescan*). A proposal names only a toolchain the operator allows, with the
allowed version its pin maps to. The repositories' `AGENTS.md`, `CLAUDE.md`,
`CONTRIBUTING.md`, `.github/copilot-instructions.md` and `.cursor/rules` are
named in the plugin's instruction section, never their contents; the root's
own files are the framework's (#1347).

**Binding needs a session.** The plugin runs in a session's runner, so on the
New workspace plate, before any session exists, a choice is remembered in
this browser and bound when the first session in that workspace starts.

**The offer is protected where it can be.** The plugin's AppArmor fragment
denies a confined session writing `.jaato/toolchain-offer.json`, so the agent
cannot widen the allow-list. An unconfined session is not covered; the plugin
validates every field anyway, and an unconfined agent could run mise itself.

**jdtls and #806.** The daemon does not reap a language server at session
end; it lives until its runner slot exits. With `lsp.jdtls` set this server
logs a WARNING at startup saying so, and `lsp.jdtls.max_heap` bounds what each
one left behind can hold (plus the JVM's own overhead). Two sessions in one
workspace share a `${jdtlsStateRoot}`, so the second jdtls finds it locked.

**Ownership.** Every route asks the daemon, over the bind channel, whether
the signed-in user owns the workspace (an empty `workspace.app_write`,
daemon protocol 1.30). A daemon that cannot answer refuses the request.

**Unbind** removes the links, the server's binaries and the tool's entries in
the derived files; the download under `.home/.local/share/mise` is kept, so a
rebind is quick. Not supported: Rust (rustup keeps its homes outside mise and
its proxies need `RUSTUP_HOME` at run time), a shared read-only toolchain
cache (phase 5 of #1344).

## Routes

| Route | Method | |
|---|---|---|
| `/config.json` | GET | `{daemon, ticketUrl, loginUrl, autoConnect[, credentialsUrl]}` — what the bundle reads |
| `/auth/login` | GET | 302 to the issuer (PKCE S256, state, nonce) |
| `/auth/callback` | GET | finishes sign-in, sets the cookie, 302 to `/` |
| `/auth/backchannel-logout` | POST | OIDC back-channel logout: ends matching sessions, revokes tickets |
| `/api/session` | GET | `{user, expiresAt}` or 401 |
| `/api/ticket` | POST | one single-use ticket; same-origin only (`Sec-Fetch-Site` / `Origin`) |
| `/api/logout` | GET | ends the session, revokes, redirects through the issuer's logout |
| `/api/environment?workspace=` | GET | the allow-list, this user's declines, and the offer's content (`environment:` only) |
| `/api/environment/decline`, `undecline` | POST | `{workspace, tool}`; same-origin only |
| `/api/credentials?provider=` | GET | the user's stored keys for a provider: `{entries: [{id, provider, label, hint, createdAt}]}` |
| `/api/credentials` | POST | store `{provider, secret, label?}` → 201 `{entry}`; same-origin only |
| `/api/credentials/<id>/reveal` | POST | `{secret}`; same-origin only |
| `/api/credentials/<id>` | DELETE | forget; same-origin only |
| `/auth/github/login` | GET | 302 to GitHub (App user-to-server authorization) |
| `/auth/github/callback` | GET | stores the grant, 302 to `/?github=connected` |
| `/api/github/accounts` | GET | the user's connected accounts `{login, installations, isDefault}` — never a token |
| `/api/github/bindings` | GET | the user's `{workspace, accountId}` bindings |
| `/api/github/default` | POST | `{id}` — make one account the default; same-origin only |
| `/api/github/disconnect` | POST | `{id}` — delete + revoke a grant, reload the user's sessions; same-origin only |
| `/api/github/bind` | POST | `{workspace, account_id\|null}` — bind/clear an account on a workspace; same-origin only |
| `/api/github/repos?account=` | GET | `{account: {id, login}, repos: [{fullName, private, defaultBranch, pushedAt?}]}` — every repository the account's App installations reach (default account when `account` is omitted), de-duplicated, most recently pushed first, capped at 1000, cached 60s; 404 with no connected account |
| `/api/github/branches?repo=owner/name&account=` | GET | `{repo, defaultBranch?, branches}` (capped at 500); 400 when `repo` is not `owner/name` |
| everything else | GET | the bundle |

The four credential routes exist only with a `credentials:` block; otherwise
they are 404 and `config.json` names no `credentialsUrl`. The GitHub routes
answer to a `github:` block the same way — and there is deliberately **no**
`/api/github/<id>/reveal`: a GitHub token travels this server → daemon only.
The two listing routes call GitHub with the user's token server-side and
return names and flags only; a dead grant answers 409 `{reconnect: true}`,
an unreachable GitHub 502.

## Develop

```bash
npm ci && npm run build       # builds ../jaato-sdk-ts declarations, then tsc → dist/
npm test                      # node:test — config, sessions, bind channel (mock daemon), routes (fake IdP), 26 cases
npm run test:daemon           # the same against the REAL daemon; needs ../.venv/bin/python with jaato-server
                              # (or JAATO_DAEMON_PYTHON=…), skips otherwise
```
