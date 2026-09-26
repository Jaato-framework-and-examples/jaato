# Forge Plugins for the Web Coder

**Status:** proposed. Nothing described here is built. It generalises the
per-user GitHub integration ([Per-User GitHub Credentials](per-user-github-credentials.md),
#1225–#1228, and [GitHub Workspace Guidance](github-workspace-guidance.md),
#1240) so that GitLab, Gitea/Forgejo and similar forges can be added without
copying it. It is **application work, not daemon work**: everything here lives
in `jaato-web-coder-server` (the BFF) and `jaato-web-coder-ui`. No change
under `jaato-server/` is required.

## 1. The problem

The GitHub integration works in three steps:

1. A web user connects a GitHub account once. The BFF runs the OAuth flow and
   keeps the refresh token, encrypted, keyed by the user's OIDC `sub`.
2. The user binds that account to a workspace. The BFF records the binding,
   writes `GH_TOKEN=app://github` into the workspace `.env`, seeds
   `<ws>/.home/.gitconfig`, and writes `.jaato/instructions/40-github.md`.
3. At every session spawn the daemon sees `app://github`, asks the BFF over
   the bind channel (`secret.resolve`), and gets a short-lived token back.

Step 3 is already forge-neutral. Steps 1 and 2 are written for GitHub only,
and the GitHub assumptions are spread across the BFF and the UI:

| Where | GitHub-only today |
|---|---|
| `server.ts` | one `GitHubService`, attached as the bind channel's only secret resolver |
| `github.ts` `resolveSecret` | refuses any `name` other than `github` |
| `github.ts` `bind` | writes `GH_TOKEN`; one binding per `(user, workspace)` |
| `renderGitConfig` | writes the whole `.gitconfig` for one host, with the `gh auth git-credential` helper |
| `github-api.ts` | GitHub's OAuth endpoints, `/user`, `/user/installations`, grant revocation, the noreply email format |
| `github-store.ts` | account metadata shaped for GitHub (`login`, `githubId`, `installations`) |
| `config.ts` | a single `github:` block |
| `routes.ts` | `/auth/github/*`, `/api/github/*` |
| `launcherConfig.ts` | `githubUrl`, `githubLoginUrl` |
| UI | `GitHubConnect`, `GitHubAccountPicker`, `app/github.ts`, the auto-bind in `WorkspaceScreen` |

Adding GitLab by copying these files would give two stores, two resolvers
competing for the bind channel's single slot, and two writers of the same
`.gitconfig`. This design splits the code into a **host** that holds
everything tied to users, disk and security, and **forge plugins** that only
describe a forge.

## 2. Scope: the BFF, not the daemon

The plugins are TypeScript modules inside the BFF. They are not jaato Python
plugins, for the reason the GitHub design already gives: the daemon knows
users only as `app:user`, while the grant, its rotation and the
`(user, workspace)` binding are the application's state. The daemon side is
already generic:

- `app://<name>` is resolved by asking the owning application, whatever the
  name (`jaato_server/server/app_secret.py`).
- The names resolved from `app://` are granted through the secret scrub on
  `cli` and `interactive_shell` (`granted_env_names`, #1228). The grant is by
  exact name, not by `GH_TOKEN`, so `GITLAB_TOKEN=app://gitlab` reaches `glab`
  with no profile change.

So no daemon change is needed. One small follow-up is optional: the
`validate` / `explain` checks that mention `GH_TOKEN` by name (§9).

## 3. What the host owns, and what a plugin owns

**The host** (generic BFF code) keeps everything that holds a secret, touches
the disk, or decides who may do what:

- the encrypted grant store and its file format;
- the per-grant refresh lock with a re-read after acquiring (#683), so two
  concurrent resolves never rotate each other's refresh token away;
- the OAuth `state` round-trip and CSRF check;
- the bindings, and the ownership checks on them;
- every write into a workspace: `.env` lines, the composed `.gitconfig`,
  managed instruction files, with the existing containment rule
  (`workspace_root`, symlinks resolved);
- dispatching `secret.resolve`, and sending `secret.reload` after a bind,
  unbind or disconnect;
- the HTTP routes and what the launcher config advertises.

**A plugin** describes one kind of forge. It returns data; the host decides
what to do with it. A plugin never receives the store, a file handle, the
bind channel or another forge's tokens.

```ts
/** One kind of forge. Instances are created from config (§4). */
interface ForgePlugin {
  /** Stable kind id: "github", "gitlab", "gitea". Reserved; not configurable. */
  readonly kind: string;

  /** Validate and normalise this kind's config block. Throws ConfigError. */
  parseConfig(raw: unknown, path: string): ForgeInstanceConfig;

  /** Build the forge client for one configured instance. */
  create(cfg: ForgeInstanceConfig, deps: { fetch: typeof fetch }): ForgeClient;
}

interface ForgeClient {
  /** Human label for the UI ("GitHub", "GitLab (corp)"). */
  readonly label: string;
  /** Host used for git credentials and commit identity, e.g. "gitlab.corp.example". */
  readonly gitHost: string;

  // OAuth. The host calls refresh() only inside its per-grant lock.
  authorizeUrl(state: string, redirectUri: string): string;
  exchangeCode(code: string, redirectUri: string): Promise<ForgeTokenSet>;
  refresh(refreshToken: string): Promise<ForgeTokenSet>;
  revoke(tokens: ForgeTokenSet): Promise<"revoked" | "not_supported">;

  /** Who the grant belongs to, plus display metadata. Never contains a token. */
  fetchIdentity(accessToken: string): Promise<ForgeIdentity>;

  /** Env vars the workspace needs. The token var is written as app://<instance id>. */
  readonly env: { token: string; extra?: Record<string, string> };

  /** Username git should send with the token over HTTPS ("x-access-token", "oauth2", ...). */
  readonly gitUsername: string;

  /** Optional working guidance, written through managed-files.ts. */
  guidanceFile?(): ManagedFile;
}

interface ForgeIdentity {
  forgeUserId: string;      // the forge's numeric or opaque id, as a string
  login: string;
  name: string | null;
  commitEmail: string;      // a noreply address where the forge has one
  extra?: Record<string, unknown>;  // e.g. GitHub installations, shown in the UI only
}
```

`ForgeTokenSet` is the existing `GitHubTokenSet` renamed: access token,
optional expiry, optional refresh token and its expiry. Errors keep today's
two classes, generalised: `ForgeGrantRevoked` (the grant is dead; the user
must reconnect) and `ForgeApiError` (anything else, treated as transient). A
plugin must only raise `ForgeGrantRevoked` on an explicit OAuth error such as
`invalid_grant`, for the reason #683 states: calling a transient failure a
dead grant logs the user out.

## 4. Kinds and instances

A **kind** is code (the GitHub plugin). An **instance** is a configured
connection to one forge. One kind can have several instances, for example
gitlab.com and a company's self-hosted GitLab.

```json
{
  "workspace_root": "/srv/workspaces",
  "forges": [
    { "id": "github", "kind": "github",
      "file": "state/github.json", "key_file": "secrets/github.key",
      "client_id": "Iv1.abc", "client_secret_file": "secrets/github.secret" },
    { "id": "gitlab-corp", "kind": "gitlab", "label": "GitLab (corp)",
      "base_url": "https://gitlab.corp.example",
      "file": "state/gitlab-corp.json", "key_file": "secrets/gitlab-corp.key",
      "client_id": "...", "client_secret_file": "secrets/gitlab-corp.secret" }
  ]
}
```

- `id` is the **reference name**: the workspace `.env` gets
  `GITLAB_TOKEN=app://gitlab-corp`, and `secret.resolve` for `gitlab-corp` is
  routed to that instance. Ids must match `[a-z][a-z0-9-]{0,31}` and be
  unique. They are checked at load, like every other config error.
- `workspace_root` moves from the `github:` block to the top level, because
  all forges write into the same workspaces and must agree on the boundary.
- The existing `github:` block keeps working. It loads as
  `{id: "github", kind: "github", ...}`, and a config that has both `github:`
  and a `forges` entry with id `github` is refused.
- **One store file per instance**, as today. The GitHub store file needs no
  migration: instance `github` reads the same file with the same key. Keeping
  instances in separate files also means a corrupt or rotated file affects
  one forge only.

## 5. Storage and bindings

The store generalises the GitHub store's two keyings without changing them:

| Record | Keyed by | Change |
|---|---|---|
| grant | owner `sub`, plus a unique grant id | account metadata becomes `ForgeIdentity`; GitHub-only fields move into `extra` |
| binding | `(user, workspace)` → grant id | unchanged in shape, but now one store per instance, so a workspace can hold one binding per instance |

Because each instance has its own store, "bound to GitHub and to GitLab" is
simply two bindings in two files. No store needs to know other forges exist.

## 6. Resolving a secret

The bind channel keeps one responder, as the SDK expects. The host puts a
dispatcher behind it:

```ts
bind.attachSecretResolver(async (req) => {
  const forge = forges.get(req.name);           // instance id
  if (!forge) return { status: "not_found", detail: `no forge named ${req.name}` };
  return forge.resolveSecret(req);              // today's GitHubService.resolveSecret logic
});
```

The per-instance logic is today's `resolveSecret`, moved into the host and
parameterised by instance: look up the binding for `(user, workspace)`, mint
or reuse a token under the lock, and map the outcome onto `ok` / `not_found`
/ `denied` / `error` with the same `detail` strings (`not_connected`,
`not_bound`, `revoked`). The lock is keyed by `instanceId + grantId`, so two
forges never wait on each other.

`secret.reload` stays per user, not per forge: after any bind, unbind or
disconnect the host asks the daemon to re-resolve that user's loaded
sessions, which re-reads every `app://` name at once.

## 7. What lands in the workspace

Binding and unbinding write three things. Today each belongs to GitHub; with
several forges bound, the host has to **compose** them.

### 7.1 `.env`

Each instance owns exactly one line (plus any `extra` lines its plugin
declares, such as `GITLAB_HOST`). `upsertEnvLine` already edits one key and
leaves the rest alone, so this only generalises the key name. Unbinding
removes only that instance's lines.

### 7.2 `.home/.gitconfig`

Today `renderGitConfig` writes the whole file for one host. With two forges
bound, the second bind would overwrite the first. The host instead rebuilds
the file from **all** bindings of that workspace every time one changes:

```ini
# jaato-managed: forge-gitconfig v1  (rebuilt on every bind; edit .gitconfig.local instead)
[credential "https://github.com"]
	username = x-access-token
	helper = "!f() { test \"$1\" = get && echo \"password=$GH_TOKEN\"; }; f"
[credential "https://gitlab.corp.example"]
	username = oauth2
	helper = "!f() { test \"$1\" = get && echo \"password=$GITLAB_TOKEN\"; }; f"

[user]
	name = Alice Example
	email = 12345+alice@users.noreply.github.com
[includeIf "hasconfig:remote.*.url:https://gitlab.corp.example/**"]
	path = .gitconfig.gitlab-corp

[include]
	path = .gitconfig.local
```

- **Credentials.** One `[credential "https://<host>"]` section per bound
  instance, each reading its own env var. The helper reads the variable
  rather than calling `gh auth git-credential`, so no forge CLI is needed for
  `git push`. The token itself is never written to disk; the file names only
  the variable.
- **Commit identity.** A commit's author should match the forge it is pushed
  to. The first bound instance supplies the default `[user]`; every other
  instance gets a small `.gitconfig.<id>` with its own `[user]`, selected by
  `includeIf "hasconfig:remote.*.url:..."`, which chooses by the repository's
  remote. That condition needs git 2.36 or newer; with an older git, commits
  use the default identity. (Measured here: git 2.43.)
- **User edits.** The composed file is managed and rebuilt, so user settings
  go in `.gitconfig.local`, which the composed file includes. This is the
  same marker-and-skip rule `managed-files.ts` applies: if the user removed
  the marker, the host leaves the file alone and says so in the bind result.

### 7.3 Guidance files

Each plugin may supply one `ManagedFile`, written through the existing
`managed-files.ts` (which already expects a second forge). Numbering by kind
keeps the instruction order stable: `40-github.md`, `41-gitlab.md`,
`42-gitea.md`. Two instances of the same kind share one file, because the
guidance is about the tool, not the host.

## 8. Routes, launcher config and UI

Routes take the instance id:

| Today | Proposed |
|---|---|
| `GET /auth/github/login` | `GET /auth/forge/:id/login` |
| `GET /auth/github/callback` | `GET /auth/forge/:id/callback` |
| `GET /api/github/accounts` | `GET /api/forges/:id/accounts` |
| `GET /api/github/bindings` | `GET /api/forges/:id/bindings` |
| `POST /api/github/{default,disconnect,bind}` | `POST /api/forges/:id/{default,disconnect,bind}` |

The old GitHub paths stay as aliases for instance `github` for at least one
release, because an OAuth App's registered callback URL is configured on the
forge and changing it means editing each forge's app settings.

The launcher config gains a list and keeps the old fields while the aliases
exist:

```json
{ "forges": [
    { "id": "github", "kind": "github", "label": "GitHub",
      "apiUrl": "./api/forges/github", "loginUrl": "./auth/forge/github/login" },
    { "id": "gitlab-corp", "kind": "gitlab", "label": "GitLab (corp)",
      "apiUrl": "./api/forges/gitlab-corp", "loginUrl": "./auth/forge/gitlab-corp/login" } ] }
```

In the UI, `GitHubConnect` and `GitHubAccountPicker` become `ForgeConnect`
and `ForgeAccountPicker`, rendered once per listed forge. The workspace
configure form shows one account picker per forge. The auto-bind on
workspace create (`autoBindDefaultGitHubAccount`) runs for every forge where
the user has a default account. Kind-specific display (GitHub's installation
list) is rendered from `ForgeIdentity.extra` by a small per-kind component,
with a generic fallback.

## 9. Per-kind notes

These are what each first-party plugin has to get right. Items marked
*verify* are from vendor documentation and were not exercised against a live
instance while writing this.

| | GitHub | GitLab | Gitea / Forgejo |
|---|---|---|---|
| OAuth app | GitHub App (user-to-server tokens) | OAuth application, scopes `api`, `read_repository`, `write_repository` | OAuth2 application, per instance |
| Access token lifetime | 8 h | 2 h (*verify*) | 1 h default, instance-configurable (*verify*) |
| Refresh token rotates | yes | yes, on every use | depends on instance setting (*verify*) |
| Scope of a token | App installations ∩ the user | everything the scopes allow for that user | varies by version; older OAuth grants were close to full access (*verify*) |
| Revocation | `DELETE /applications/{client_id}/grant` | `POST /oauth/revoke` (*verify*) | no standard endpoint (*verify*); plugin returns `not_supported` and the host says the grant was only deleted locally |
| Identity | `GET /user` | `GET /api/v4/user` | `GET /api/v1/user` |
| Commit email | `id+login@users.noreply.github.com` | `id-login@users.noreply.<host>` on gitlab.com; self-managed may differ (*verify*) | instance setting (`NO_REPLY_ADDRESS`), so a config key |
| Token env var | `GH_TOKEN` | `GITLAB_TOKEN` (+ `GITLAB_HOST` for self-managed) | `GITEA_TOKEN` (name is ours; the CLIs differ) |
| Git HTTPS username | `x-access-token` | `oauth2` | any non-empty value |
| CLI for the model | `gh` | `glab` | none reliable; guidance points at `git` plus the REST API |

The Gitea/Forgejo plugin should be a generic `oauth-forge` kind with
configurable endpoints, noreply domain and env var name. Most self-hosted
forges can then be supported by configuration, without new code.

Two daemon-side notes, neither required:

- `jaato-scaffold validate` and `explain` name `GH_TOKEN` in their `app://`
  checks. Since the scrub grant (#1228) keeps any resolved `app://` name,
  those checks now matter mainly for a literal token. Generalising them to
  "any env var whose value is `app://…`" is a small, separate change.
- Nothing in `jaato-server/` needs to know the list of forges.

## 10. Packaging and trust

**Start with a static, in-tree registry**: a map from kind to plugin in the
BFF, holding `github`, `gitlab` and `oauth-forge`. Config can only name kinds
in that map.

Loading plugins from npm packages is deliberately left out. A forge plugin
handles refresh tokens and access tokens, so a third-party package would be
code with access to every user's forge credentials. If that is ever needed,
it should follow the rules jaato already applies to Python entry-point
plugins (#684): kind names are reserved, loading is limited to an allowlist
of packages, and the plugin only ever receives the narrow `ForgeClient`
surface in §3, never the store or the filesystem.

## 11. Limitations

- **One instance per kind per workspace.** Two GitLab instances cannot both
  be bound to the same workspace, because `glab` reads one `GITLAB_TOKEN`
  and one `GITLAB_HOST`. Git itself could handle both (one credential section
  per host), but the CLIs cannot, and a workspace where `git push` works and
  `glab` silently talks to the wrong host is worse than a refusal. The bind
  route refuses the second instance of a kind with a clear message.
- **The token is still in the subprocess environment.** As with GitHub, a
  model that wants to print it can. Keeping it out of the runner entirely is
  the credential broker (#505), which would be per host and could reuse the
  same instance list.
- **`includeIf hasconfig:` needs git 2.36+.** On older git every commit uses
  the default identity; pushes still authenticate correctly.
- **Self-hosted forges vary.** Token lifetimes, rotation and revocation are
  instance settings on GitLab and Gitea. The plugin must read what the token
  response says (`expires_in`, whether a new `refresh_token` came back)
  rather than assume the defaults in §9.

## 12. Implementation plan

Each step leaves the product working and is reviewable on its own.

1. **Extract the host, GitHub as the only plugin.** Move the store, lock,
   `resolveSecret`, `bind`, workspace writes and routes into generic host
   code; turn `github-api.ts` and `github-guidance.ts` into the `github`
   plugin. The config still accepts `github:`. Existing tests must pass
   unchanged, and a new test drives the bind channel's dispatcher with an
   unknown name. No visible behaviour change.
2. **Instances and the `forges:` config.** Add the list, per-instance
   stores, id-based routes with the GitHub aliases, and the launcher
   `forges` field. The UI switches to the list.
3. **Composed `.gitconfig`.** Replace `renderGitConfig` with the composer
   in §7.2, including the `.gitconfig.local` include and the
   `hasconfig` identity switch. Test with two fake forges bound to one
   workspace, unbinding each in turn.
4. **The `gitlab` plugin**, with its guidance file. Test the refresh lock
   against a fake that rotates the refresh token on every call.
5. **The `oauth-forge` plugin** for Gitea/Forgejo, configured by endpoints.
6. Optional: generalise the `GH_TOKEN` checks in `validate` / `explain`.

## 13. Open questions

1. Should an unbound workspace get the guidance file of every forge the user
   has connected, or only of the forges bound to it? This design says only
   bound ones, matching GitHub today.
2. Should the per-kind limit in §11 become per host instead, once the CLIs
   are no longer the main way the model talks to a forge?
3. Should instance ids be allowed to change? Today they are baked into every
   bound workspace's `.env`, so renaming one means rewriting those files. The
   simplest answer is "no, add a new instance instead".
