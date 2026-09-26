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
import { removeManagedFile, writeManagedFile } from "./managed-files.js";

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

export interface GitHubServiceOptions {
  store: FileGitHubStore;
  api: GitHubApi;
  reloader: SessionReloader;
  /**
   * The root every managed workspace lives under.  When set, ``.env`` /
   * ``.gitconfig`` writes are contained within it (symlinks resolved before
   * comparing, the daemon's own rule); when unset, those writes are SKIPPED
   * — the binding is still recorded, so a split-host deployment where the
   * daemon owns the workspace filesystem can drive the ``.env`` write through
   * the browser's ``config.update`` instead.  Never a silent skip: the bind
   * result carries a ``note``.
   */
  workspaceRoot?: string;
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

    let envWritten = false;
    let gitconfigSeeded = false;
    let guidanceWritten = false;
    const notes: string[] = [];
    const resolved = this._resolveWorkspace(workspace);
    if (resolved === null) {
      notes.push(this._workspaceRoot
        ? "workspace is outside the configured workspace_root or does not exist; GH_TOKEN not written to .env (nor the GitHub guidance file)"
        : "no workspace_root configured; GH_TOKEN not written to .env (the browser config.update path writes it instead); the GitHub guidance file is not written here either");
    } else {
      envWritten = this._writeEnv(resolved, accountId !== null);
      if (accountId !== null && account) gitconfigSeeded = this._seedGitConfig(resolved, account);
      const g = this._applyGuidance(resolved, accountId !== null);
      guidanceWritten = g.written;
      if (g.note) notes.push(g.note);
    }

    const { reloaded } = await this._reload(user, accountId === null ? `unbind ${workspace}` : `bind ${account!.login} -> ${workspace}`);
    return {
      binding: accountId === null ? "cleared" : "set",
      envWritten, gitconfigSeeded, guidanceWritten, reloaded,
      note: notes.length ? notes.join("; ") : undefined,
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
