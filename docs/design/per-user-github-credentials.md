# Per-User GitHub Credentials for Web Sessions

**Status:** proposed. Nothing described here is implemented yet; the issues
that implement it (#1225–#1228) are listed at the end.

## 1. The problem

A web deployment (`jaato-web-coder-server` + `jaato-web-coder-ui`, "the WUI")
serves several people from **one daemon running as root**. Each WUI user owns
their workspaces (`WorkspaceInfo.owner`, stamped from the `app:user` identity a
bound ticket carries, #1074). Their agents are told to talk to GitHub through
the `gh` CLI, and each of them has their own GitHub account.

Today nothing in that chain can express "this session acts as *that* person on
GitHub":

| Where a `gh` credential could live today | What goes wrong |
|---|---|
| `gh`'s default, `~/.config/gh/hosts.yml` | the runner inherits the daemon's `HOME=/root`, so every user shares one GitHub identity, and `gh auth setup-git` rewrites `/root/.gitconfig` for everyone |
| a file in the workspace (`hosts.yml`, `.netrc`) | persists on disk in plaintext, root-owned (#1168), readable by the agent's file tools, and outlives the session |
| the workspace `.env`, forwarded by the BFF through `config.update` (the web-server-bff.md §12 path for provider keys) | the same three problems, plus a GitHub user token expires after 8h and the file does not |
| the browser forwards the token when it opens a session | misses every session the browser does not open: cascade stages, `session.wake`, reactor-spawned and revived sessions |

The WUI users are **not host users**. There is no Unix account, no home
directory and no keyring to root a per-user credential in, so any answer has to
be keyed on the application's identity, not the OS's.

## 2. What a credential can and cannot be protected from

Anything `gh` can use, the model can read: `gh auth token`, `echo $GH_TOKEN`,
`cat hosts.yml`. No choice of directory changes that, and this design does not
pretend otherwise. What *can* be controlled:

- **which** user's credential a session receives;
- **how much** a leaked credential is worth: its scope and its lifetime;
- **whether** it is ever written to disk, where backups, file tools and the
  next session can reach it.

The design therefore optimises for: the right user, a short-lived token, and no
plaintext at rest outside the BFF's encrypted store. Making the token
unexfiltratable needs a broker that injects the `Authorization` header so the
runner never holds it (#505's shape); that is noted in §9 as a later step, not
built here.

## 3. The shape, in one picture

```
 once per WUI user          once per workspace            every session spawn, any path
 ─────────────────          ──────────────────            ─────────────────────────────
 WUI  "Connect GitHub"      WUI configure form            daemon _resolve_session_env
  │                          "GitHub account: @alice"      reads  GH_TOKEN=app://github
  ▼                          │                              │
 BFF ⇄ github.com (OAuth)    ▼                              ▼  secret.resolve {user, workspace, name}
  stores refresh token      BFF stores                     BFF (over its bind channel)
  encrypted, keyed by sub    (sub, workspace) → cred id     mints an 8h token from the refresh token
                            daemon .env gets                │
                             GH_TOKEN=app://github           ▼  {value, expires_at}
                             (a reference, not a secret)   envelope env → runner os.environ
                                                           → cli / interactive_shell subprocesses
```

Three roles, each owned by the component that already owns the fact:

| Role | Owner | Why there |
|---|---|---|
| who the user is, and their GitHub grant | the BFF | it holds the OIDC session and already keeps per-user secrets encrypted (`credentials.ts`) |
| which workspaces use GitHub | the workspace `.env`, as a reference | it is where a session's environment is declared, and a reference is harmless on disk |
| turning the reference into a token | the daemon, at spawn | it is already the only process that resolves secret URIs (`_resolve_session_env`), and spawn is the one point every session passes through |

The daemon never learns what GitHub, OIDC or a refresh token is. That is the
same line #1074 drew for identity: the application is the authority on its
users, and the daemon asks it rather than re-deriving the answer.

## 4. Part 1: connecting GitHub (BFF, once per WUI user)

A settings entry in the WUI: **Connect GitHub**.

**Preferred: a GitHub App with user-to-server authorization.** The OAuth web
flow runs entirely server-side: browser → `github.com/login/oauth/authorize` →
callback to the BFF. The BFF exchanges the code, stores the **refresh token**
encrypted under the user's OIDC `sub`, and keeps the GitHub `login` and the
app's installation list beside it (non-secret, for display). User-to-server
access tokens last 8 hours and are minted on demand from the refresh token.

Why an App rather than an OAuth App or a PAT: the token's reach is the
intersection of what the **user** can do and what the **App was installed on**,
so an organisation decides which repositories agents may touch, and a leaked
token expires on its own.

**Fallback: a pasted fine-grained PAT.** Stored in the same store under
provider `github`, with the existing label/hint display. No refresh, no expiry
handling beyond what GitHub enforces; documented as the weaker option.

Rules for this part:

- **The secret never reaches the browser.** `/api/credentials/<id>/reveal`
  exists so the page can forward a provider key through `config.update`; a
  GitHub credential is never revealed. It travels BFF → daemon only (§6).
- **Refresh tokens rotate.** A GitHub App refresh returns a new refresh token
  and voids the old one, so two concurrent spawns refreshing at once would each
  rotate the other's away. The BFF needs the #683 shape: an exclusive lock
  around read–refresh–write, and a **re-read after acquiring** so the waiter
  adopts the token the winner wrote instead of refreshing again.
- **Disconnect** deletes the stored grant *and* revokes it at GitHub
  (`DELETE /applications/{client_id}/grant`), so live sessions fail with 401
  rather than keeping access until expiry.
- A user may connect **more than one** account (personal, work); one is the
  default.

## 5. Part 2: binding an account to a workspace (workspace setup)

The workspace configure form gains one field:

```
GitHub account   [ @alice (default) ▾ ]   (none | @alice | @alice-work)
```

Saving it does two things:

1. **In the BFF:** records `(sub, workspace) → credential id`. This is
   application state; the daemon never stores it.
2. **In the workspace:** writes `GH_TOKEN=app://github` into the workspace
   `.env` (or removes it for *none*). That line is the whole of "this workspace
   uses GitHub". It is a reference, not a credential: harmless in a backup,
   harmless if the agent reads it.

Non-secret material that belongs with the workspace can be seeded at the same
moment into the workspace home (§8):

```ini
# <ws>/.home/.gitconfig
[user]
    name  = Alice Example
    email = 12345+alice@users.noreply.github.com
[credential "https://github.com"]
    helper = !gh auth git-credential
```

The identity comes from GitHub's `/user`; the noreply address avoids
publishing a private email in commits.

## 6. Part 3: resolving at spawn (daemon)

`JaatoServer._resolve_session_env` already reads the workspace `.env` and the
profile's `env:` map, resolves `pass://` / `vault://` / … through the
`SecretResolver` plugins (`shared/plugins/subagent/config.py`), and ships the
resolved dict on the bootstrap envelope. The session record keeps the
**unresolved** form. `app://` is a new scheme at exactly that point:

1. A session is spawned, from any path: WUI, cascade stage, wake, revive.
2. Resolution meets `GH_TOKEN=app://github`.
3. The daemon identifies the **application** that answers for this workspace:
   the owner is `acme:alice`, so the application is `acme`.
4. It sends `secret.resolve {request_id, user: "alice", workspace, name:
   "github"}` to `acme` over the connection `acme` authenticated with its app
   credential (the BFF's bind channel, `bind-channel.ts`).
5. The BFF looks up the binding, mints (or reuses) a token and answers
   `{request_id, status: "ok", value, expires_at}`, or a refusal with a status.
6. The value goes into the envelope's env dict. Nowhere else.

### 6.1 The rules this has to hold

| Rule | Why |
|---|---|
| **Resolve for the workspace OWNER, not the session creator** | a headless cascade stage may carry no `created_by` (still an open question), a revived session's creator is a record rather than a connection, and "that WUI user's workspaces" is literally the owner relation. An **unowned** workspace resolves nothing |
| **Ask only the application the user is qualified under** | `acme:alice` is never asked of `other-app`. One application must not be able to learn it is being asked about another's users, nor answer for them |
| **The request direction is new, and says so** | today the bind channel carries `ticket.bind` / `ticket.revoke`, both application → daemon. `secret.resolve` is daemon → application; it rides the same authenticated connection, correlated by `request_id`, with a deadline |
| **Nothing resolved is persisted** | the session record, the workspace `.env` and the snapshot keep `app://github`. A revived session resolves afresh at its own spawn, which is also what makes revocation take effect |
| **An unresolved `app://` is dropped, never forwarded** | a literal `GH_TOKEN=app://github` in a subprocess is a token that fails with a confusing 401. Dropping the variable makes `gh` report "not logged in", which is the true state |
| **Failure is announced and, by default, not fatal** | if the application is not connected or refuses, the session starts **without** `GH_TOKEN` and a WARNING names the reference and the reason. A session that cannot reach GitHub is still useful; a session refused because the BFF restarted is not. A per-reference strict form (`app://github?required`) turns it into a bootstrap refusal |

### 6.2 What the resolver interface lacks

`SecretResolver.resolve(scheme, path, key)` has no notion of *who is asking*:
every existing scheme resolves the same way for every session. `app://` needs
the workspace (to find the owner) at minimum. The change is an optional
context argument passed by `_resolve_session_env`:

```python
class SecretResolveContext(NamedTuple):
    workspace_path: str
    workspace_owner: Optional[str]   # qualified "app:user"
    session_id: Optional[str]
```

Existing resolvers ignore it. The `app://` resolver is **in-tree** (it is part
of the ticket mechanism, not a premium backend) and refuses when the context is
absent rather than guessing.

### 6.3 Expiry and refresh

A GitHub App user token lasts 8 hours; sessions can outlive that. The BFF
returns `expires_at`, and the daemon schedules a `session.reload_env` for that
session a margin before it (the #683 margin, `JAATO_OAUTH_REFRESH_MARGIN`).
`reload_env` already re-runs the full resolution and replaces the runner's
session env; it is refused mid-turn (`stage="busy"`), so the scheduler retries
at the next idle point, and the token is still valid through the margin.

A session that is **not loaded** needs nothing: it resolves at its next spawn.

### 6.4 Revocation

*Disconnect GitHub* or *workspace → none* in the WUI:

- the BFF deletes the binding (and, for disconnect, revokes the grant at
  GitHub);
- the BFF asks the daemon to `session.reload_env` the owner's loaded sessions
  in the affected workspaces, so the variable disappears rather than lingering
  until the process ends. (The BFF can already see those sessions through the
  owner-scoped `session.list`.)

Revoking at GitHub is what actually ends access; the reload only stops the
session from holding a dead token.

## 7. Part 4: using it in the runner

The session env reaches the runner's `os.environ`; the model-driven subprocess
surfaces then need three things:

- **The scrub exemption.** `GH_TOKEN` is in the default `scrub_secret_env` set
  (#863). A profile that wants `gh` declares, per surface:

  ```yaml
  plugin_configs:
    cli:
      scrub_secret_env: [default, "!GH_TOKEN"]
    interactive_shell:
      scrub_secret_env: [default, "!GH_TOKEN"]
  # mcp keeps `default`: MCP servers get no GitHub token
  ```

- **Non-interactive defaults:** `GH_PROMPT_DISABLED=1`, `GIT_TERMINAL_PROMPT=0`,
  so a missing credential is an error the model reads rather than a prompt
  nobody answers.
- **Persona guidance:** use `gh`; never print or echo the token; on a 401,
  report it instead of attempting `gh auth login` (there is nothing on disk for
  the agent to repair).

## 8. Prerequisite: a workspace home for model-driven subprocesses

Independent of GitHub, the runner's subprocesses currently run with the
daemon's `HOME=/root`, so every tool that writes to `~` shares state across
tenants: `~/.gitconfig`, `~/.config/gh`, `~/.npmrc`, pip and npm caches, shell
history, `~/.ssh/known_hosts`.

**The home is per workspace, inside it: `<ws>/.home/`.** Alternatives
considered:

| Location | Why not |
|---|---|
| the workspace root | dotfiles and caches (possibly gigabytes) mixed into the user's tree; one `git add -A` commits them, credential files included; and `~/.jaato` would become `<ws>/.jaato`, collapsing jaato's own config tiers |
| `<ws>/.jaato/home` | `.jaato` holds framework assets, authored or produced. A tool's home is neither |
| one home per WUI user, outside every workspace | WUI users are not host users; there is no natural per-user place on the host. It also needs new containment and AppArmor grants and opens a channel between a user's workspaces |

What a workspace home gets for free: the workspace already has an owner (so the
home is that user's), `workspace.delete` removes it with nothing to keep in
sync, and it is inside the workspace so neither `cli` path containment nor the
AppArmor profile needs a new rule.

Shape, mirroring the existing `plugin_configs.cli.workspace_venv`
(`shared/plugins/workspace_venv.py`): a relative path resolved against the
workspace root, absolute allowed, relative-without-a-workspace refused.

```yaml
plugin_configs:
  cli:
    workspace_venv: .venv
    workspace_home: .home
```

Details:

1. **Scope:** `HOME` and `XDG_CONFIG_HOME` / `XDG_CACHE_HOME` /
   `XDG_DATA_HOME` / `XDG_STATE_HOME` are set in the environment built for
   `cli`, `interactive_shell` and the notebook kernel. The runner's own `HOME`
   is **unchanged**, so jaato's `~/.jaato` lookups and the `${HOME}` profile
   expansion variable keep their meaning.
2. **Default:** on for workspaces the daemon manages under `workspace_root`,
   opt-in elsewhere, so a session pointed at a user's real checkout over IPC
   does not grow a `.home/`.
3. **Created by the daemon** before the runner spawns, as the session tmpdir is
   (#1171): a confined runner must not be relied on to create it.
4. **Kept out of git** by a `.gitignore` containing `*` inside `.home/` itself,
   so the user's own `.gitignore` is never edited.
5. **Kept out of the Files panel** by adding the home to the workspace
   monitor's ignore patterns. The file in (4) is not enough: `GitignoreParser`
   reads only `<workspace>/.gitignore`, and its only built-in ignore is `.git/`.
6. **Secrets never live here.** It persists and the file tools can read it;
   tokens ride the session env (§6).

## 9. Not in this design

- **A credential broker.** A local proxy that adds `Authorization` for
  `api.github.com` and git smart-HTTP, with `gh` and `git` pointed at it, is
  the only arrangement in which the runner never holds the token. It is the
  right next step if broad-scope tokens are unavoidable; it is out of scope
  here.
- **OS isolation.** Every session still runs as the daemon's uid. This design
  separates identities and credentials inside one uid, not processes;
  privilege dropping is the #1168 follow-up.
- **Other forges.** Nothing here is GitHub-specific in the daemon: `app://<name>`
  resolves whatever the application answers for `<name>`. GitLab or a package
  registry token is a BFF-side addition.

## 10. Open questions

1. Should a headless cascade stage inherit its driver's creator? Not needed by
   this design (it resolves for the owner) but it would change attribution.
2. How the UI chooses between several connected accounts beyond "default", and
   whether one workspace may bind different accounts per host (github.com and
   a GitHub Enterprise instance).
3. Whether the App should be installed per repository, so a workspace's token
   reaches only the repositories that workspace works on.

## 11. Implementation

| Issue | Part |
|---|---|
| #1225 workspace home for model-driven subprocesses | §8 |
| #1226 `app://` secrets resolved by the owning application | §6, §6.1–6.4 |
| #1227 BFF: connect GitHub and bind an account to a workspace | §4, §5, and the BFF side of §6 |
| #1228 profile and persona guidance for `gh` sessions | §7 |
