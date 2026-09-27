/**
 * The GitHub credential service: what ties the connect flow, the encrypted
 * store, the rotation lock, the workspace writes and the daemon's
 * ``secret.resolve`` together.
 *
 * The pieces it owns, and why each is here rather than in the daemon:
 *
 * - **Connect / disconnect / set-default / bind** — the operations a
 *   signed-in user drives.  They hold an OIDC session (``sub`` + ``user``),
 *   which the daemon never has: only THIS application knows who its users
 *   are (#1074), so the grant and the ``(user, workspace)`` binding are its
 *   state.
 * - **Minting an 8-hour user token from a rotating refresh token** — the
 *   #683 pattern: an exclusive lock around read-refresh-write, and a
 *   **re-read after acquiring** so a second concurrent resolve for the same
 *   grant adopts the token the first one just wrote instead of refreshing
 *   again (which would rotate the first one's refresh token away).  A single
 *   process owns the store file, so the lock is an in-process keyed async
 *   mutex — the TS equivalent of ``shared/credential_lock.py``'s flock, which
 *   exists because the Python daemon spans processes; here there is one.
 * - **Answering ``secret.resolve``** — the SDK's ``SecretResolveResponder``
 *   (#1226) carries no credential policy; {@link GitHubService.resolveSecret}
 *   is that policy.  It resolves for the workspace OWNER (``user``), only for
 *   a binding this application wrote, and maps its own outcome onto the
 *   wire's closed status set (``ok`` / ``not_found`` / ``denied`` / ``error``)
 *   with a ``detail`` naming the semantic reason (``not_connected`` /
 *   ``not_bound`` / ``revoked``).
 * - **Revocation reload** — on disconnect or bind-to-none, ``secret.reload``
 *   (application -> daemon, #1226 §6.4) asks the daemon to re-resolve the
 *   owner's loaded sessions, so the now-dead reference drops out of a live
 *   session's environment rather than lingering until the process ends.
 *
 * The secret never reaches the browser: there is no reveal here (unlike a
 * provider key), and the only egress of a token is BFF -> daemon over the
 * authenticated bind channel, inside a ``secret.resolve`` answer.
 */
import { randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync, existsSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, sep } from "node:path";
import type { SecretResolveOutcome } from "@jaato/sdk";
import { GitHubApiError, GitHubGrantRevoked, type GitHubApi, type GitHubRepo, type GitHubTokenSet } from "./github-api.js";
import { FileGitHubStore, GitHubStoreError, type GitHubAccount, type GitHubBinding, type GrantSecret } from "./github-store.js";
import { githubGuidanceFile } from "./github-guidance.js";
import { managedContent, removeManagedFile, writeManagedFile } from "./managed-files.js";
import type { WorkspaceWriteAnswer, WorkspaceWriteRequest } from "./bind-channel.js";

/** A user token minted this far before its expiry is refreshed rather than reused (the #683 / JAATO_OAUTH_REFRESH_MARGIN default). */
export const TOKEN_REFRESH_MARGIN_SECONDS = 300;

/** The reference name this application answers; ``GH_TOKEN=app://github``. */
export const GITHUB_REFERENCE_NAME = "github";
/** The line the workspace ``.env`` carries — a reference, never a secret. */
export const GH_TOKEN_ENV = "GH_TOKEN";
export const GH_TOKEN_ENV_VALUE = `app://${GITHUB_REFERENCE_NAME}`;

/** How long a ``(user, account)`` repository listing is served from memory before GitHub is asked again. */
export const REPO_LIST_CACHE_MS = 60_000;
/** The most repositories one listing collects across all installations. */
export const MAX_LISTED_REPOS = 1000;
/** The most branch names one ``branches`` answer collects. */
export const MAX_LISTED_BRANCHES = 500;
/** ``owner/name`` as GitHub spells a repository; anything else is refused before a URL is built from it. */
export const REPO_FULL_NAME_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

/** What ``GET /api/github/repos`` answers: the account used, and its reachable repositories. */
export interface RepoListing {
  account: { id: string; login: string };
  repos: GitHubRepo[];
}

/** What ``GET /api/github/branches`` answers. */
export interface BranchListing {
  repo: string;
  defaultBranch?: string;
  branches: string[];
}

export class GitHubBindError extends Error {
  override name = "GitHubBindError";
  constructor(message: string, readonly status: 400 | 404 = 400) { super(message); }
}

export interface BindResult {
  binding: "set" | "cleared";
  /** Whether ``GH_TOKEN`` was written to / removed from the workspace ``.env``. */
  envWritten: boolean;
  /** Whether ``<ws>/.home/.gitconfig`` was seeded (only on bind-to-account). */
  gitconfigSeeded: boolean;
  /**
   * Whether ``.jaato/instructions/40-github.md`` was written or refreshed on
   * this bind.  ``false`` when it was left untouched (already current, the
   * user replaced it, or no workspace filesystem write happened at all); the
   * reason for a skip / overwrite is on {@link BindResult.note}.
   */
  guidanceWritten: boolean;
  /** How many loaded sessions the daemon re-resolved (0 is the ordinary answer). */
  reloaded: number;
  /** Set when a filesystem write was skipped or a managed file was overwritten, so the caller never believes a write happened silently. */
  note?: string;
}

/** The narrow slice of the bind channel this service needs — reloading a user's sessions. */
export interface SessionReloader {
  reloadUser(user: string): Promise<{ status: string; reloaded: number }>;
}

/**
 * The daemon doing the workspace writes this application cannot do itself
 * (``workspace.app_write``, protocol 1.30).  The bind channel implements it.
 */
export interface WorkspaceWriter {
  canWriteWorkspaces(): boolean;
  writeWorkspace(user: string, workspace: string, request: WorkspaceWriteRequest): Promise<WorkspaceWriteAnswer>;
}

/** What writing one binding's files did; shared by {@link GitHubService.bind} and the resync. */
interface BindingWrite {
  envWritten: boolean;
  /** ``true`` only when the ``.env`` actually changed (the resync reloads on this). */
  envChanged: boolean;
  gitconfigSeeded: boolean;
  guidanceWritten: boolean;
  notes: string[];
}

export interface GitHubServiceOptions {
  store: FileGitHubStore;
  api: GitHubApi;
  reloader: SessionReloader;
  /**
   * The root every managed workspace lives under, when THIS process can write
   * there.  When set, ``.env`` / ``.gitconfig`` / guidance writes happen here,
   * contained within it (symlinks resolved before comparing, the daemon's own
   * rule).  When unset, or for a workspace outside it, the writes go through
   * {@link GitHubServiceOptions.workspaceWriter} (the daemon, which owns the
   * workspaces), and only when neither can write is the write skipped, with a
   * ``note`` saying sessions there will not get ``GH_TOKEN``.
   */
  workspaceRoot?: string;
  /**
   * The daemon's ``workspace.app_write`` (1.30), for a deployment where this
   * application runs as an account that cannot reach the workspaces (a BFF
   * beside a root daemon).  A binding is inert without the ``.env`` line:
   * the daemon resolves only references it finds there.
   */
  workspaceWriter?: WorkspaceWriter;
  marginSeconds?: number;
  log?: (msg: string) => void;
}

/** ``GH_TOKEN=app://github`` upserted into (or removed from) an existing ``.env`` body. */
export function upsertEnvLine(body: string, key: string, value: string | null): string {
  const lines = body.length ? body.split("\n") : [];
  const re = new RegExp(`^\\s*(?:export\\s+)?${key}\\s*=`);
  const kept = lines.filter((l) => !re.test(l));
  // Preserve a trailing empty line iff the original had one, so a rewrite is minimal.
  const trailingBlank = kept.length > 0 && kept[kept.length - 1] === "";
  const core = trailingBlank ? kept.slice(0, -1) : kept;
  if (value !== null) core.push(`${key}=${value}`);
  return core.length ? core.join("\n") + "\n" : "";
}

/** The ``.gitconfig`` seeded into the workspace home: identity plus the ``gh`` credential helper. */
export function renderGitConfig(opts: { name: string; email: string; gitHost: string }): string {
  return [
    "[user]",
    `\tname = ${opts.name}`,
    `\temail = ${opts.email}`,
    `[credential "https://${opts.gitHost}"]`,
    "\thelper = !gh auth git-credential",
    "",
  ].join("\n");
}

/** Serialises token refresh per grant id so two concurrent resolves collapse to one refresh (#683). */
class KeyedMutex {
  private readonly _tail = new Map<string, Promise<unknown>>();
  run<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const prev = this._tail.get(key) ?? Promise.resolve();
    const result = prev.then(() => fn());
    const tail = result.then(() => undefined, () => undefined);
    this._tail.set(key, tail);
    // Drop the chain once it settles and nothing newer replaced it, so the map
    // does not grow without bound; a still-pending chain is kept.
    void tail.then(() => { if (this._tail.get(key) === tail) this._tail.delete(key); });
    return result;
  }
}

export class GitHubService {
  private readonly _store: FileGitHubStore;
  private readonly _api: GitHubApi;
  private readonly _reloader: SessionReloader;
  private readonly _workspaceRoot?: string;
  private readonly _writer?: WorkspaceWriter;
  private readonly _margin: number;
  private readonly _log: (msg: string) => void;
  private readonly _mint = new KeyedMutex();
  /** ``owner \0 accountId`` -> the last listing and when it was taken (see {@link REPO_LIST_CACHE_MS}). */
  private readonly _repoCache = new Map<string, { at: number; listing: RepoListing }>();

  constructor(opts: GitHubServiceOptions) {
    this._store = opts.store;
    this._api = opts.api;
    this._reloader = opts.reloader;
    this._workspaceRoot = (opts.workspaceRoot ? realpathSafe(opts.workspaceRoot) : null) ?? undefined;
    this._writer = opts.workspaceWriter;
    this._margin = opts.marginSeconds ?? TOKEN_REFRESH_MARGIN_SECONDS;
    this._log = opts.log ?? (() => undefined);
  }

  // ---- connect flow -------------------------------------------------------

  authorizeUrl(state: string, redirectUri: string): string {
    return this._api.authorizeUrl(state, redirectUri);
  }

  /** Finish the OAuth exchange and store the grant for ``owner`` (OIDC sub) / ``user`` (daemon identity). */
  async completeConnect(owner: string, user: string, code: string, redirectUri: string): Promise<GitHubAccount> {
    const tok = await this._api.exchangeCode(code, redirectUri);
    const identity = await this._api.fetchIdentity(tok.accessToken);
    const secret = tokenSetToSecret(tok, Date.now());
    return this._store.connect(owner, user, {
      login: identity.login, githubId: identity.id, name: identity.name,
      noreplyEmail: identity.noreplyEmail, installations: identity.installations, secret,
    });
  }

  listAccounts(owner: string): GitHubAccount[] {
    return this._store.listAccounts(owner);
  }

  listBindings(owner: string): GitHubBinding[] {
    return this._store.listBindings(owner);
  }

  setDefault(owner: string, grantId: string): boolean {
    return this._store.setDefault(owner, grantId);
  }

  /** Delete a grant, revoke it at GitHub, and reload the user's loaded sessions so live ones lose the token. */
  async disconnect(owner: string, user: string, grantId: string): Promise<GitHubAccount> {
    // Get a usable token for the revoke BEFORE removing the grant; a dead
    // grant simply cannot be revoked (it already is), which is not an error.
    let accessToken: string | null = null;
    try { accessToken = (await this._mintToken(grantId)).value; }
    catch (e) { this._log(`github disconnect: no usable token to revoke grant ${grantId}: ${(e as Error).message}`); }
    const removed = this._store.disconnect(owner, grantId);
    if (!removed) throw new GitHubBindError("no such connected account", 404);
    if (accessToken) {
      try { await this._api.revokeGrant(accessToken); }
      catch (e) { this._log(`github disconnect: revoke at GitHub failed for ${removed.account.login}: ${(e as Error).message}`); }
    }
    await this._reload(user, `disconnect ${removed.account.login}`);
    return removed.account;
  }

  // ---- repository / branch listing (the New-workspace picker) ------------

  /**
   * The account a listing uses: ``accountId`` when given (it must be one of
   * ``owner``'s), else the default, else the first connected one.  Throws a
   * 404 {@link GitHubBindError} when there is none.
   */
  private _pickAccount(owner: string, accountId?: string | null): GitHubAccount {
    const accounts = this._store.listAccounts(owner);
    if (accountId) {
      const a = accounts.find((x) => x.id === accountId);
      if (!a) throw new GitHubBindError("no such connected account", 404);
      return a;
    }
    const a = accounts.find((x) => x.isDefault) ?? accounts[0];
    if (!a) throw new GitHubBindError("no GitHub account connected", 404);
    return a;
  }

  /**
   * Every repository the account's App installations let ``owner`` reach,
   * de-duplicated by ``fullName`` and sorted most-recently-pushed first (then
   * by name).  The installation set is the stored one UNIONED with a fresh
   * ``/user/installations`` read, so an App installed after the connect shows
   * up without a reconnect, and a removed installation (404) is skipped
   * rather than failing the list.  Served from a per-``(owner, account)``
   * cache for {@link REPO_LIST_CACHE_MS}.  The token is minted through the
   * same per-grant lock as ``secret.resolve`` and never leaves this method.
   *
   * Throws {@link GitHubGrantRevoked} for a dead grant, and
   * {@link GitHubApiError} when every installation failed transiently.
   */
  async listRepos(owner: string, accountId?: string | null): Promise<RepoListing> {
    const account = this._pickAccount(owner, accountId);
    const key = `${owner}\0${account.id}`;
    const now = Date.now();
    const hit = this._repoCache.get(key);
    if (hit && now - hit.at < REPO_LIST_CACHE_MS) return hit.listing;
    for (const [k, v] of this._repoCache) if (now - v.at >= REPO_LIST_CACHE_MS) this._repoCache.delete(k);

    const { value: token } = await this._mintToken(account.id);
    const installationIds = new Set(account.installations.map((i) => i.id));
    try {
      for (const i of (await this._api.fetchIdentity(token)).installations) installationIds.add(i.id);
    } catch (e) {
      if (e instanceof GitHubGrantRevoked) throw e;
      this._log(`github repos: could not refresh installations for ${account.login}: ${(e as Error).message}`);
    }

    const byName = new Map<string, GitHubRepo>();
    let failures = 0;
    let lastError: Error | null = null;
    for (const id of installationIds) {
      if (byName.size >= MAX_LISTED_REPOS) break;
      try {
        for (const r of await this._api.listInstallationRepos(token, id, MAX_LISTED_REPOS - byName.size)) {
          if (!byName.has(r.fullName)) byName.set(r.fullName, r);
        }
      } catch (e) {
        if (e instanceof GitHubGrantRevoked) throw e;
        failures += 1; lastError = e as Error;
        this._log(`github repos: installation ${id} of ${account.login} skipped: ${(e as Error).message}`);
      }
    }
    if (installationIds.size > 0 && failures === installationIds.size) {
      throw new GitHubApiError(`could not list repositories: ${lastError?.message ?? "unknown error"}`);
    }
    const repos = [...byName.values()].slice(0, MAX_LISTED_REPOS).sort(compareRepos);
    const listing: RepoListing = { account: { id: account.id, login: account.login }, repos };
    this._repoCache.set(key, { at: now, listing });
    return listing;
  }

  /**
   * The branch names of ``repo`` (``owner/name``, validated against
   * {@link REPO_FULL_NAME_RE}; a 400 {@link GitHubBindError} otherwise), read
   * with the chosen account's token.  ``defaultBranch`` is filled from the
   * cached repository listing when that repository is in it.
   */
  async listBranches(owner: string, repo: string, accountId?: string | null): Promise<BranchListing> {
    if (typeof repo !== "string" || !REPO_FULL_NAME_RE.test(repo) || repo.split("/").some((p) => p === "." || p === "..")) {
      throw new GitHubBindError("repo must be owner/name", 400);
    }
    const account = this._pickAccount(owner, accountId);
    const { value: token } = await this._mintToken(account.id);
    const [o, n] = repo.split("/") as [string, string];
    const branches = await this._api.listBranches(token, o, n, MAX_LISTED_BRANCHES);
    const cached = this._repoCache.get(`${owner}\0${account.id}`)?.listing.repos.find((r) => r.fullName.toLowerCase() === repo.toLowerCase());
    return cached ? { repo, defaultBranch: cached.defaultBranch, branches } : { repo, branches };
  }

  // ---- bind an account to a workspace ------------------------------------

  /**
   * Bind ``workspace`` to ``accountId`` (or clear it for ``accountId === null``),
   * write / remove ``GH_TOKEN=app://github`` in the workspace ``.env``, seed the
   * workspace ``.gitconfig`` on a bind, and reload the owner's loaded sessions.
   */
  async bind(owner: string, user: string, workspace: string, accountId: string | null): Promise<BindResult> {
    const account = accountId === null ? null : this._store.accountById(accountId);
    if (accountId !== null && (!account || !this._ownsAccount(owner, accountId))) {
      throw new GitHubBindError("no such connected account", 404);
    }
    // Record the binding first: it is what secret.resolve actually reads, and
    // a filesystem write that fails must not lose it.
    try { this._store.bind(owner, user, workspace, accountId); }
    catch (e) { if (e instanceof GitHubStoreError) throw new GitHubBindError(e.message, e.status === 404 ? 404 : 400); throw e; }

    const w = await this._writeBindingFiles(user, workspace, account);

    const { reloaded } = await this._reload(user, accountId === null ? `unbind ${workspace}` : `bind ${account!.login} -> ${workspace}`);
    return {
      binding: accountId === null ? "cleared" : "set",
      envWritten: w.envWritten, gitconfigSeeded: w.gitconfigSeeded, guidanceWritten: w.guidanceWritten, reloaded,
      note: w.notes.length ? w.notes.join("; ") : undefined,
    };
  }

  /**
   * Re-write every recorded binding's files, and reload the sessions of each
   * user whose ``.env`` changed.  Run when the bind channel (re)connects: a
   * binding recorded while no write was possible (no ``workspace_root`` and
   * a daemon below 1.30, or the channel down) gets its ``.env`` line the
   * first time a write is possible, without the user picking the account
   * again.  Every write is idempotent, so resyncing a healthy store changes
   * nothing.
   */
  async resyncWorkspaces(): Promise<{ bindings: number; changed: number; notes: string[] }> {
    const all = this._store.allBindings();
    const changedUsers = new Set<string>();
    const notes: string[] = [];
    let changed = 0;
    for (const b of all) {
      const account = this._store.accountById(b.grantId);
      if (!account) continue;
      const w = await this._writeBindingFiles(b.user, b.workspace, account);
      if (w.envChanged) { changed += 1; changedUsers.add(b.user); }
      for (const n of w.notes) notes.push(`${b.workspace}: ${n}`);
    }
    for (const user of changedUsers) await this._reload(user, "resync");
    this._log(`github resync: ${all.length} binding(s), ${changed} workspace .env updated${notes.length ? `; ${notes.join("; ")}` : ""}`);
    return { bindings: all.length, changed, notes };
  }

  /**
   * Write (``account``) or remove (``null``) one binding's files: the
   * ``GH_TOKEN=app://github`` line, the seeded ``.home/.gitconfig`` and the
   * GitHub guidance.  Here when this process can reach the workspace, else
   * through the daemon, else not at all, and never silently.
   */
  private async _writeBindingFiles(user: string, workspace: string, account: GitHubAccount | null): Promise<BindingWrite> {
    const present = account !== null;
    const resolved = this._resolveWorkspace(workspace);
    if (resolved !== null) {
      const notes: string[] = [];
      const before = readEnvSafe(resolved);
      const envWritten = this._writeEnv(resolved, present);
      const gitconfigSeeded = account ? this._seedGitConfig(resolved, account) : false;
      const g = this._applyGuidance(resolved, present);
      if (g.note) notes.push(g.note);
      return { envWritten, envChanged: readEnvSafe(resolved) !== before, gitconfigSeeded, guidanceWritten: g.written, notes };
    }
    if (this._writer?.canWriteWorkspaces()) return this._writeThroughDaemon(user, workspace, account);
    const why = this._workspaceRoot
      ? "the workspace is outside this server's workspace_root"
      : "this server has no workspace_root";
    return {
      envWritten: false, envChanged: false, gitconfigSeeded: false, guidanceWritten: false,
      notes: [`GH_TOKEN was not written to the workspace .env: ${why}, and the daemon does not accept workspace.app_write (it needs protocol 1.30 and a connected bind channel); sessions in this workspace will not get GH_TOKEN`],
    };
  }

  /** {@link _writeBindingFiles} through the daemon's ``workspace.app_write``. */
  private async _writeThroughDaemon(user: string, workspace: string, account: GitHubAccount | null): Promise<BindingWrite> {
    const present = account !== null;
    const guidance = githubGuidanceFile();
    const files: WorkspaceWriteRequest["files"] = [];
    if (account) {
      files.push({ path: ".home/.gitconfig", content: renderGitConfig({ name: account.name || account.login, email: account.noreplyEmail, gitHost: this._api.gitHost }) });
    }
    files.push({ path: guidance.relativePath, content: present ? managedContent(guidance) : null, managed_by: guidance.markerId });
    const none: BindingWrite = { envWritten: false, envChanged: false, gitconfigSeeded: false, guidanceWritten: false, notes: [] };
    let ans: WorkspaceWriteAnswer;
    try {
      ans = await this._writer!.writeWorkspace(user, workspace, { env: { [GH_TOKEN_ENV]: present ? GH_TOKEN_ENV_VALUE : null }, files });
    } catch (e) {
      return { ...none, notes: [`the daemon could not be asked to write the workspace: ${(e as Error).message}; sessions there will not get GH_TOKEN`] };
    }
    if (ans.status !== "ok") {
      return { ...none, notes: [`the daemon did not write the workspace (${ans.status}${ans.detail ? `: ${ans.detail}` : ""}); sessions there will not get GH_TOKEN`] };
    }
    const notes: string[] = [];
    const envAction = ans.env[GH_TOKEN_ENV] ?? "error";
    if (envAction === "kept-literal") notes.push("left the GH_TOKEN value already in .env in place (it is not an app:// reference)");
    if (envAction === "error") notes.push("the daemon could not update the workspace .env");
    const outcome = (path: string) => ans.files.find((f) => f.path === path);
    const git = outcome(".home/.gitconfig");
    const guide = outcome(guidance.relativePath);
    if (guide?.action === "skipped-user-file") {
      notes.push(present
        ? `kept your own ${guidance.relativePath} (its jaato-managed marker was removed); the shipped GitHub guidance was not written`
        : `left your own ${guidance.relativePath} in place (its jaato-managed marker was removed)`);
    }
    for (const f of ans.files) if (f.action === "error") notes.push(`could not write ${f.path}${f.detail ? `: ${f.detail}` : ""}`);
    return {
      envWritten: present ? envAction === "written" || envAction === "unchanged" : envAction === "removed",
      envChanged: envAction === "written" || envAction === "removed",
      gitconfigSeeded: git?.action === "written" || git?.action === "unchanged",
      guidanceWritten: guide?.action === "written",
      notes,
    };
  }

  // ---- secret.resolve (the SecretResolveResponder handler) ---------------

  /**
   * Answer the daemon's ``secret.resolve`` for one of THIS application's
   * users.  Bound to a {@link SecretResolveResponder}; never called for
   * another application (the daemon routes each workspace only to the
   * application that owns it, and this channel is this application's).
   */
  resolveSecret = async (request: { user: string; workspace: string; name: string }): Promise<SecretResolveOutcome> => {
    if (request.name !== GITHUB_REFERENCE_NAME) {
      return { status: "not_found", detail: `this application resolves only app://${GITHUB_REFERENCE_NAME}` };
    }
    const grantId = this._store.bindingFor(request.user, request.workspace);
    if (!grantId) {
      return this._store.userHasAnyAccount(request.user)
        ? { status: "not_found", detail: "not_bound: no GitHub account is bound to this workspace" }
        : { status: "not_found", detail: "not_connected: this user has not connected GitHub" };
    }
    if (!this._store.accountById(grantId)) {
      return { status: "not_found", detail: "not_bound: the bound GitHub account no longer exists" };
    }
    try {
      const { value, expiresAt } = await this._mintToken(grantId);
      return { status: "ok", value, expiresAt };
    } catch (e) {
      if (e instanceof GitHubGrantRevoked) {
        return { status: "denied", detail: "revoked: the GitHub grant is no longer valid; reconnect GitHub" };
      }
      return { status: "error", detail: `could not mint a GitHub token: ${(e as Error).message}` };
    }
  };

  // ---- token minting, under the per-grant lock (#683) --------------------

  private async _mintToken(grantId: string): Promise<{ value: string; expiresAt: string | null }> {
    return this._mint.run(grantId, async () => {
      // RE-READ after acquiring: a concurrent mint for this grant may have
      // just refreshed and written a fresh access token, in which case we
      // reuse it rather than refreshing (and rotating) again — the half that
      // makes the lock a fix and not merely a queue.
      const secret = this._store.secretOf(grantId);
      if (!secret) throw new GitHubGrantRevoked("the grant no longer exists");
      const now = Date.now();
      if (secret.accessToken && (secret.accessExpiresAt === undefined || secret.accessExpiresAt - now > this._margin * 1000)) {
        return { value: secret.accessToken, expiresAt: secret.accessExpiresAt ? new Date(secret.accessExpiresAt).toISOString() : null };
      }
      if (!secret.refreshToken) throw new GitHubGrantRevoked("access token expired and no refresh token is stored");
      if (secret.refreshExpiresAt !== undefined && secret.refreshExpiresAt <= now) throw new GitHubGrantRevoked("refresh token expired");
      const tok = await this._api.refresh(secret.refreshToken);
      const next = tokenSetToSecret(tok, Date.now(), secret);
      this._store.updateSecret(grantId, next);
      return { value: next.accessToken ?? "", expiresAt: next.accessExpiresAt ? new Date(next.accessExpiresAt).toISOString() : null };
    });
  }

  private _ownsAccount(owner: string, grantId: string): boolean {
    return this._store.listAccounts(owner).some((a) => a.id === grantId);
  }

  private async _reload(user: string, why: string): Promise<{ reloaded: number }> {
    try {
      const r = await this._reloader.reloadUser(user);
      if (r.status !== "ok") this._log(`github ${why}: secret.reload answered ${r.status}`);
      return { reloaded: r.reloaded };
    } catch (e) {
      this._log(`github ${why}: secret.reload failed: ${(e as Error).message}`);
      return { reloaded: 0 };
    }
  }

  // ---- workspace filesystem writes (single-host direct mode) -------------

  /** Resolve + contain a workspace path, or ``null`` when a write must be skipped. */
  private _resolveWorkspace(workspace: string): string | null {
    if (!this._workspaceRoot) return null;
    if (typeof workspace !== "string" || !workspace || !isAbsolute(workspace)) return null;
    const real = realpathSafe(workspace);
    if (real === null) return null;
    // Containment, resolving symlinks on both sides (the daemon's own rule):
    // a path equal to or beneath the root, never the root itself.
    if (real !== this._workspaceRoot && !real.startsWith(this._workspaceRoot + sep)) return null;
    return real;
  }

  private _writeEnv(workspace: string, present: boolean): boolean {
    const path = join(workspace, ".env");
    try {
      const body = existsSync(path) ? readFileSync(path, "utf8") : "";
      const next = upsertEnvLine(body, GH_TOKEN_ENV, present ? GH_TOKEN_ENV_VALUE : null);
      if (next === body) return present; // nothing to change
      atomicWrite(path, next, 0o600);
      return true;
    } catch (e) {
      this._log(`github bind: could not update ${path}: ${(e as Error).message}`);
      return false;
    }
  }

  private _seedGitConfig(workspace: string, account: GitHubAccount): boolean {
    const homeDir = join(workspace, ".home");
    const path = join(homeDir, ".gitconfig");
    try {
      mkdirSync(homeDir, { recursive: true, mode: 0o700 });
      atomicWrite(path, renderGitConfig({ name: account.name || account.login, email: account.noreplyEmail, gitHost: this._api.gitHost }), 0o600);
      return true;
    } catch (e) {
      this._log(`github bind: could not seed ${path}: ${(e as Error).message}`);
      return false;
    }
  }

  /**
   * Write / refresh (on bind-to-account) or remove (on bind-to-none) the
   * GitHub working-guidance file, through the generic managed-file mechanism.
   * Returns whether a write happened and a human note for a skip / overwrite —
   * a clean write, a no-op, or a removal is silent (the model reads the file;
   * the operator does not need a note that it worked).
   */
  private _applyGuidance(workspace: string, present: boolean): { written: boolean; note?: string } {
    const file = githubGuidanceFile();
    if (!present) {
      const outcome = removeManagedFile(workspace, file, this._log);
      if (outcome.action === "skipped-user-file") {
        return { written: false, note: `left your own ${file.relativePath} in place (its jaato-managed marker was removed)` };
      }
      return { written: false };
    }
    const outcome = writeManagedFile(workspace, file, atomicWrite, this._log);
    switch (outcome.action) {
      case "skipped-user-file":
        return { written: false, note: `kept your own ${file.relativePath} (its jaato-managed marker was removed); the shipped GitHub guidance was not written` };
      case "written":
        return outcome.reason === "version-changed"
          ? { written: true, note: `refreshed ${file.relativePath} to the current GitHub guidance` }
          : { written: true };
      default:
        return { written: false };
    }
  }
}

/** Most recently pushed first; a repository with no push time after every one that has; then by name. */
function compareRepos(a: GitHubRepo, b: GitHubRepo): number {
  const ta = a.pushedAt ? Date.parse(a.pushedAt) : NaN;
  const tb = b.pushedAt ? Date.parse(b.pushedAt) : NaN;
  const ha = !Number.isNaN(ta); const hb = !Number.isNaN(tb);
  if (ha && hb && ta !== tb) return tb - ta;
  if (ha !== hb) return ha ? -1 : 1;
  return a.fullName.localeCompare(b.fullName);
}

/** A workspace's ``.env`` body, ``""`` when absent or unreadable. */
function readEnvSafe(workspace: string): string {
  try { return readFileSync(join(workspace, ".env"), "utf8"); } catch { return ""; }
}

/** Real path if the entry exists, else ``null``; symlinks resolved so a planted link is judged by its target. */
function realpathSafe(p: string): string | null {
  try { return realpathSync(p); } catch { return null; }
}

/** Write via temp-file + rename, so a reader never sees a half-written file and the mode is set atomically. */
function atomicWrite(path: string, body: string, mode: number): void {
  const tmp = join(dirname(path), `.${randomBytes(6).toString("hex")}.tmp`);
  writeFileSync(tmp, body, { mode });
  renameSync(tmp, path);
}

/** Fold a GitHub token response into the stored secret, carrying the previous refresh token when the response omits one. */
export function tokenSetToSecret(tok: GitHubTokenSet, now: number, previous?: GrantSecret): GrantSecret {
  return {
    accessToken: tok.accessToken,
    accessExpiresAt: tok.accessExpiresInSeconds !== undefined ? now + tok.accessExpiresInSeconds * 1000 : undefined,
    // GitHub rotates on refresh, so a response WITHOUT a refresh token means
    // "unchanged" (or an App with expiration off); keep the previous one.
    refreshToken: tok.refreshToken ?? previous?.refreshToken,
    refreshExpiresAt: tok.refreshExpiresInSeconds !== undefined ? now + tok.refreshExpiresInSeconds * 1000 : previous?.refreshExpiresAt,
  };
}

// Re-exported so a caller can distinguish a dead grant from a transient failure.
export { GitHubApiError, GitHubGrantRevoked };
