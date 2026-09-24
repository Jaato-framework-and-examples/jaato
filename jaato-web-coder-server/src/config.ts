/**
 * Configuration of jaato-web-coder-server: one YAML file, secrets in
 * files beside it.
 *
 * Every secret (the app credential, the OIDC client secret, the session
 * signing secret) is read from a file rather than the YAML, for the reason
 * the daemon prefers ``--ws-token-file`` over ``--ws-token``: a value in
 * argv or a world-readable config is served to every process on the host.
 * Each secret file must be mode 0600 or stricter, the rule
 * ``server/ws_tickets.py`` applies to the daemon's side of the same
 * credential.  The design and the rationale for every field are in
 * ``docs/design/web-server-bff.md`` §7 and §11.
 */
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parse as parseYaml } from "yaml";

export class ConfigError extends Error {
  override name = "ConfigError";
}

export interface OidcConfig {
  issuer: string;
  /** Loopback URL for discovery / token / JWKS when the public issuer is not reachable from the host itself. */
  backchannelUrl?: string;
  clientId: string;
  clientSecret: string;
  scopes: string[];
  /** ID-token claim that becomes ``ticket.bind``'s ``user``. */
  subjectClaim: string;
  /** Realm or client role a user must carry to sign in; unset = anyone the realm authenticates. */
  requiredRole?: string;
}

export interface ServerConfig {
  listen: { host: string; port: number };
  publicUrl: string;
  daemon: {
    /** What the BROWSER connects to. */
    url: string;
    /** What this server connects to for the bind channel; defaults to ``url``. */
    bindUrl: string;
    appId: string;
    appCredential: string;
  };
  mode: "direct";
  auth: { kind: "oidc"; oidc: OidcConfig };
  session: { secret: string; ttlSeconds: number; cookieName: string };
  ticket: { ttlSeconds: number };
  /**
   * The per-user store of provider API keys (``src/credentials.ts``).
   * Absent = the feature is off: ``config.json`` names no
   * ``credentialsUrl`` and the routes answer 404, so the bundle shows the
   * plain key field it always had.
   */
  credentials?: { file: string; key: string };
  /**
   * The per-user store of session notes (``src/notes.ts``).  Absent = the
   * feature is off: ``config.json`` names no ``notesUrl`` and the routes
   * answer 404, so the bundle falls back to ``localStorage`` and says so.
   */
  notes?: { file: string; key: string };
  /**
   * Per-user GitHub connect (``src/github.ts``, #1227).  Absent = the feature
   * is off: ``config.json`` names no ``githubUrl`` and the ``/auth/github`` /
   * ``/api/github`` routes answer 404.  ``client_id`` is the GitHub App's
   * (non-secret, it rides the authorize URL); the client secret is a 0600
   * file like every other secret here.  ``workspace_root`` bounds where the
   * ``.env`` / ``.gitconfig`` writes may land; unset = those writes are
   * skipped (the binding is still recorded).  ``oauth_base_url`` /
   * ``api_base_url`` point at a GitHub Enterprise host.
   */
  github?: {
    file: string;
    key: string;
    clientId: string;
    clientSecret: string;
    oauthBaseUrl?: string;
    apiBaseUrl?: string;
    noreplyDomain?: string;
    workspaceRoot?: string;
  };
}

const DEFAULT_TTL = "8h";

function req<T>(v: T | undefined | null, path: string): T {
  if (v === undefined || v === null || v === "") throw new ConfigError(`missing required key: ${path}`);
  return v;
}

function str(v: unknown, path: string): string {
  if (typeof v !== "string") throw new ConfigError(`${path} must be a string`);
  return v;
}

/** ``8h``, ``30m``, ``3600s`` or a bare number of seconds. */
export function parseDuration(v: unknown, path: string): number {
  if (typeof v === "number" && Number.isFinite(v) && v > 0) return Math.floor(v);
  if (typeof v !== "string") throw new ConfigError(`${path} must be a duration like 8h, 30m or 3600s`);
  const m = /^(\d+)\s*([smhd]?)$/.exec(v.trim());
  if (!m) throw new ConfigError(`${path} must be a duration like 8h, 30m or 3600s, got '${v}'`);
  const n = Number(m[1]);
  const unit = { "": 1, s: 1, m: 60, h: 3600, d: 86400 }[m[2] ?? ""] ?? 1;
  return n * unit;
}

/** ``host:port`` or ``:port`` (host defaults to 127.0.0.1). */
export function parseListen(v: unknown): { host: string; port: number } {
  const s = str(v ?? "127.0.0.1:8443", "listen");
  const i = s.lastIndexOf(":");
  if (i < 0) throw new ConfigError(`listen must be host:port, got ${s}`);
  const host = s.slice(0, i) || "127.0.0.1";
  const port = Number(s.slice(i + 1));
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new ConfigError(`listen port out of range: ${s}`);
  return { host: host.replace(/^\[|\]$/g, ""), port };
}

/** Read a secret file, refusing one that group or others can read (POSIX only). */
export function readSecretFile(path: string, what: string): string {
  let mode: number;
  try { mode = statSync(path).mode; }
  catch (e) { throw new ConfigError(`${what}: cannot read ${path}: ${(e as Error).message}`); }
  if (process.platform !== "win32" && (mode & 0o077) !== 0) {
    throw new ConfigError(`${what}: ${path} is readable by group/others (mode ${(mode & 0o777).toString(8)}); restrict to 0600`);
  }
  const v = readFileSync(path, "utf8").trim();
  if (!v) throw new ConfigError(`${what}: ${path} is empty`);
  return v;
}

export function isLoopbackUrl(u: string): boolean {
  try {
    const h = new URL(u).hostname.replace(/^\[|\]$/g, "").toLowerCase();
    return h === "localhost" || h === "::1" || /^127\.\d+\.\d+\.\d+$/.test(h);
  } catch { return false; }
}

/** Build the typed config from a parsed YAML object; ``baseDir`` resolves relative secret paths. */
export function configFromObject(raw: unknown, baseDir: string): ServerConfig {
  if (!raw || typeof raw !== "object") throw new ConfigError("config must be a YAML mapping");
  const o = raw as Record<string, any>;
  const rel = (p: string) => resolve(baseDir, p);

  const daemon = req(o.daemon, "daemon");
  const url = str(req(daemon.url, "daemon.url"), "daemon.url");
  const bindUrl = daemon.bind_url ? str(daemon.bind_url, "daemon.bind_url") : url;
  for (const [k, v] of [["daemon.url", url], ["daemon.bind_url", bindUrl]] as const) {
    if (!/^wss?:\/\//.test(v)) throw new ConfigError(`${k} must be a ws:// or wss:// URL`);
  }
  const appId = str(req(daemon.app_id, "daemon.app_id"), "daemon.app_id");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(appId)) throw new ConfigError("daemon.app_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ (the daemon refuses ':')");
  const appCredential = readSecretFile(rel(str(req(daemon.app_credential_file, "daemon.app_credential_file"), "daemon.app_credential_file")), "app credential");

  const mode = o.mode ?? "direct";
  if (mode !== "direct") throw new ConfigError(`mode ${String(mode)} is not implemented; only direct`);

  const auth = req(o.auth, "auth");
  if ((auth.kind ?? "oidc") !== "oidc") throw new ConfigError(`auth.kind ${String(auth.kind)} is not implemented; only oidc`);
  const oidcRaw = req(auth.oidc, "auth.oidc");
  const issuer = str(req(oidcRaw.issuer, "auth.oidc.issuer"), "auth.oidc.issuer");
  if (!/^https?:\/\//.test(issuer)) throw new ConfigError("auth.oidc.issuer must be an http(s) URL");
  const backchannelUrl = oidcRaw.backchannel_url ? str(oidcRaw.backchannel_url, "auth.oidc.backchannel_url") : undefined;
  if (backchannelUrl && backchannelUrl.startsWith("http://") && !isLoopbackUrl(backchannelUrl)) {
    throw new ConfigError("auth.oidc.backchannel_url may be plain http only on a loopback address");
  }
  if (issuer.startsWith("http://") && !isLoopbackUrl(issuer)) {
    throw new ConfigError("auth.oidc.issuer may be plain http only on a loopback address (development)");
  }
  const oidc: OidcConfig = {
    issuer,
    backchannelUrl,
    clientId: str(req(oidcRaw.client_id, "auth.oidc.client_id"), "auth.oidc.client_id"),
    clientSecret: readSecretFile(rel(str(req(oidcRaw.client_secret_file, "auth.oidc.client_secret_file"), "auth.oidc.client_secret_file")), "OIDC client secret"),
    scopes: Array.isArray(oidcRaw.scopes) ? oidcRaw.scopes.map((x: unknown) => str(x, "auth.oidc.scopes[]")) : ["openid", "profile"],
    subjectClaim: oidcRaw.subject_claim ? str(oidcRaw.subject_claim, "auth.oidc.subject_claim") : "preferred_username",
    requiredRole: oidcRaw.required_role ? str(oidcRaw.required_role, "auth.oidc.required_role") : undefined,
  };
  if (!oidc.scopes.includes("openid")) oidc.scopes.unshift("openid");

  const sess = o.session ?? {};
  const session = {
    secret: readSecretFile(rel(str(req(sess.secret_file, "session.secret_file"), "session.secret_file")), "session secret"),
    ttlSeconds: parseDuration(sess.ttl ?? DEFAULT_TTL, "session.ttl"),
    cookieName: sess.cookie_name ? str(sess.cookie_name, "session.cookie_name") : "jaato_web_coder_session",
  };
  if (session.secret.length < 32) throw new ConfigError("session secret must be at least 32 characters");

  const ticketTtl = o.ticket?.ttl_seconds ?? 60;
  if (!Number.isInteger(ticketTtl) || ticketTtl < 1 || ticketTtl > 3600) throw new ConfigError("ticket.ttl_seconds must be an integer 1..3600 (the daemon's range)");

  const publicUrl = str(req(o.public_url, "public_url"), "public_url").replace(/\/+$/, "");
  if (!/^https?:\/\//.test(publicUrl)) throw new ConfigError("public_url must be an http(s) URL");

  let credentials: ServerConfig["credentials"];
  if (o.credentials !== undefined && o.credentials !== null) {
    const c = o.credentials;
    if (typeof c !== "object") throw new ConfigError("credentials must be a mapping with file and key_file");
    const file = rel(str(req(c.file, "credentials.file"), "credentials.file"));
    // The same 0600 rule as every other secret here; the store derives its
    // AES key from this value, so a short one is a weak key, refused.
    const key = readSecretFile(rel(str(req(c.key_file, "credentials.key_file"), "credentials.key_file")), "credential key");
    if (key.length < 32) throw new ConfigError("credential key must be at least 32 characters");
    credentials = { file, key };
  }

  let notes: ServerConfig["notes"];
  if (o.notes !== undefined && o.notes !== null) {
    const n = o.notes;
    if (typeof n !== "object") throw new ConfigError("notes must be a mapping with file and key_file");
    const file = rel(str(req(n.file, "notes.file"), "notes.file"));
    // Deliberately its own key_file rather than reusing the credential one
    // by default: an operator may want notes and not the key vault, or the
    // reverse.  Pointing both at ONE file is fine -- the two stores derive
    // different AES keys from it through different HKDF info strings.
    const key = readSecretFile(rel(str(req(n.key_file, "notes.key_file"), "notes.key_file")), "note key");
    if (key.length < 32) throw new ConfigError("note key must be at least 32 characters");
    notes = { file, key };
  }

  let github: ServerConfig["github"];
  if (o.github !== undefined && o.github !== null) {
    const g = o.github;
    if (typeof g !== "object") throw new ConfigError("github must be a mapping");
    const file = rel(str(req(g.file, "github.file"), "github.file"));
    const key = readSecretFile(rel(str(req(g.key_file, "github.key_file"), "github.key_file")), "github key");
    if (key.length < 32) throw new ConfigError("github key must be at least 32 characters");
    const clientId = str(req(g.client_id, "github.client_id"), "github.client_id");
    const clientSecret = readSecretFile(rel(str(req(g.client_secret_file, "github.client_secret_file"), "github.client_secret_file")), "GitHub App client secret");
    const oauthBaseUrl = g.oauth_base_url ? str(g.oauth_base_url, "github.oauth_base_url") : undefined;
    const apiBaseUrl = g.api_base_url ? str(g.api_base_url, "github.api_base_url") : undefined;
    for (const [k, v] of [["github.oauth_base_url", oauthBaseUrl], ["github.api_base_url", apiBaseUrl]] as const) {
      if (v && !/^https:\/\//.test(v)) throw new ConfigError(`${k} must be an https URL`);
    }
    const noreplyDomain = g.noreply_domain ? str(g.noreply_domain, "github.noreply_domain") : undefined;
    // Contained, not taken literally: the write path resolves symlinks and
    // refuses a workspace outside this root (the daemon's own rule).  Resolved
    // to an absolute path so a relative one is anchored at the config dir.
    const workspaceRoot = g.workspace_root ? resolve(baseDir, str(g.workspace_root, "github.workspace_root")) : undefined;
    github = { file, key, clientId, clientSecret, oauthBaseUrl, apiBaseUrl, noreplyDomain, workspaceRoot };
  }

  return {
    listen: parseListen(o.listen),
    publicUrl,
    daemon: { url, bindUrl, appId, appCredential },
    mode: "direct",
    auth: { kind: "oidc", oidc },
    session,
    ticket: { ttlSeconds: ticketTtl },
    credentials,
    notes,
    github,
  };
}

export function loadConfig(path: string): ServerConfig {
  let text: string;
  try { text = readFileSync(path, "utf8"); }
  catch (e) { throw new ConfigError(`cannot read config ${path}: ${(e as Error).message}`); }
  return configFromObject(parseYaml(text), dirname(resolve(path)));
}
