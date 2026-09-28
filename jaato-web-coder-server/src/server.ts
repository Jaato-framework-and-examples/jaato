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
import { EnvironmentService } from "./environment/service.js";
import { FileEnvironmentStore } from "./environment/store.js";
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
    const svc = new GitHubService({ store, api, reloader: bind, workspaceRoot: gh.workspaceRoot, workspaceWriter: bind, log });
    github = svc;
    bind.attachSecretResolver(svc.resolveSecret);
    const via = gh.workspaceRoot
      ? `written here under ${gh.workspaceRoot}, else by the daemon`
      : bind.canWriteWorkspaces()
        ? "written by the daemon (workspace.app_write)"
        : "NOT written: no workspace_root and the daemon does not accept workspace.app_write (protocol 1.30), so bound workspaces get no GH_TOKEN";
    log(`github connect enabled (store ${gh.file}; secret.resolve answering on the bind channel; workspace .env references ${via})`);
    // A binding recorded while no write was possible gets its .env line the
    // first time one is: now, and on every reconnect of the bind channel.
    const resync = () => { svc.resyncWorkspaces().catch((e) => log(`github resync failed: ${(e as Error).message}`)); };
    resync();
    bind.onConnected(resync);
  } else {
    log("github connect not configured (no github: block)");
  }

  // Opt-in: the environment bootstrap (#1344).  Every operation asks the
  // daemon, over this bind channel, whether the user owns the workspace.
  let environment: EnvironmentService | undefined;
  if (config.environment) {
    const env = config.environment;
    environment = new EnvironmentService({
      workspaceRoot: env.workspaceRoot, tools: env.tools, lsp: env.lsp, typescriptVersion: env.typescriptVersion, jdtls: env.jdtls,
      mise: env.mise, python: env.python, paranoid: env.paranoid, installTimeoutMs: env.installTimeoutSeconds * 1000,
      store: new FileEnvironmentStore(env.stateFile), ownership: bind, log,
    });
    const offered = environment.allowedTools().map((t) => `${t.tool}[${t.versions.join(",")}]${t.server ? `+${t.server.id}` : ""}`);
    log(`environment bootstrap enabled under ${env.workspaceRoot}: ${offered.length ? offered.join(" ") : "no toolchain allowed yet (environment.tools / environment.lsp are empty)"}`);
    if (env.lsp.jdtls && env.tools.java?.length) {
      // #806: the daemon does not reap a language server at session end.  Say what that costs here.
      log(`WARNING: jdtls is enabled. A language server outlives its session until its runner slot exits (jaato #806), so each workspace's jdtls may hold up to -Xmx${env.jdtls.maxHeap} plus JVM overhead after the session that started it. Set environment.lsp.jdtls.max_heap to bound it.`);
    }
  } else {
    log("environment bootstrap not configured (no environment: block)");
  }

  const server = createServer(createRouter({ config, idp, sessions, bind, credentials, notes, github, environment, distDir: opts.distDir, log }));
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
