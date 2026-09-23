/**
 * Assemble and run the server from a loaded config: discover the issuer,
 * open the bind channel, then listen.  Both prerequisites are awaited
 * before the port opens, so a misconfigured issuer or a refused app
 * credential fails the start instead of the first user's login.
 */
import { createServer, type Server } from "node:http";
import { BindChannel } from "./bind-channel.js";
import type { ServerConfig } from "./config.js";
import { FileCredentialStore } from "./credentials.js";
import { FileNoteStore } from "./notes.js";
import { FileGitHubStore } from "./github-store.js";
import { HttpGitHubApi } from "./github-api.js";
import { GitHubService } from "./github.js";
import { type IdentityProvider } from "./auth/identity.js";
import { OidcProvider } from "./auth/oidc.js";
import { createRouter } from "./routes.js";
import { SessionStore } from "./session.js";

export interface RunningServer {
  server: Server;
  bind: BindChannel;
  sessions: SessionStore;
  close(): Promise<void>;
}

export interface StartOptions {
  idp?: IdentityProvider;
  distDir?: string;
  log?: (msg: string) => void;
}

export async function startServer(config: ServerConfig, opts: StartOptions = {}): Promise<RunningServer> {
  const log = opts.log ?? ((m: string) => process.stderr.write(`${new Date().toISOString()} ${m}\n`));
  const idp = opts.idp ?? await OidcProvider.discover(config.auth.oidc);
  log(`issuer ${config.auth.oidc.issuer} discovered${config.auth.oidc.backchannelUrl ? ` (back channel ${config.auth.oidc.backchannelUrl})` : ""}`);

  const bind = new BindChannel({ bindUrl: config.daemon.bindUrl, appCredential: config.daemon.appCredential });
  bind.onStatus((s) => log(`bind channel ${s.toLowerCase()}`));
  await bind.connect();
  log(`bind channel open to ${config.daemon.bindUrl} as app ${config.daemon.appId}`);

  const sessions = new SessionStore(config.session.secret, config.session.ttlSeconds);
  const sweeper = setInterval(() => sessions.sweep(), 60_000);
  sweeper.unref();

  // Opt-in: without the block the bundle gets the plain key field it always had.
  const credentials = config.credentials ? new FileCredentialStore(config.credentials.file, config.credentials.key) : undefined;
  log(credentials ? `credential store at ${config.credentials!.file}` : "credential store not configured (no credentials: block)");
  const notes = config.notes ? new FileNoteStore(config.notes.file, config.notes.key) : undefined;
  log(notes ? `session-note store at ${config.notes!.file}` : "session-note store not configured (no notes: block); the page will keep notes in the browser");

  // Opt-in: per-user GitHub connect.  The service answers the daemon's
  // secret.resolve over the SAME bind channel that mints tickets (#1226).
  let github: GitHubService | undefined;
  if (config.github) {
    const gh = config.github;
    const store = new FileGitHubStore(gh.file, gh.key);
    const api = new HttpGitHubApi({
      clientId: gh.clientId, clientSecret: gh.clientSecret,
      oauthBaseUrl: gh.oauthBaseUrl, apiBaseUrl: gh.apiBaseUrl, noreplyDomain: gh.noreplyDomain,
    });
    github = new GitHubService({ store, api, reloader: bind, workspaceRoot: gh.workspaceRoot, log });
    bind.attachSecretResolver(github.resolveSecret);
    log(`github connect enabled (store ${gh.file}; secret.resolve answering on the bind channel${gh.workspaceRoot ? `; workspace writes contained to ${gh.workspaceRoot}` : "; workspace .env writes disabled (no workspace_root)"})`);
  } else {
    log("github connect not configured (no github: block)");
  }

  const server = createServer(createRouter({ config, idp, sessions, bind, credentials, notes, github, distDir: opts.distDir, log }));
  await new Promise<void>((ok, fail) => server.once("error", fail).listen(config.listen.port, config.listen.host, ok));
  const addr = server.address();
  log(`listening on ${typeof addr === "string" ? addr : `${addr?.address}:${addr?.port}`}; public URL ${config.publicUrl}; browsers connect to ${config.daemon.url}`);

  return {
    server, bind, sessions,
    async close() {
      clearInterval(sweeper);
      await new Promise<void>((r) => server.close(() => r()));
      await bind.close();
    },
  };
}
