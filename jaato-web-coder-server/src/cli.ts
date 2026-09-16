/**
 * ``jaato-web-coder-server serve --config PATH`` and
 * ``jaato-web-coder-server init --dir DIR``.
 *
 * ``init`` exists because the app credential must be byte-identical in the
 * daemon's ``--ws-app-credentials`` file and this server's
 * ``app_credential_file``: it generates the credential and the session
 * secret, writes this side's files at mode 0600, writes a config template,
 * and prints the daemon-side JSON entry — so nobody types a 43-character
 * secret twice.
 */
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { ConfigError, loadConfig } from "./config.js";
import { startServer } from "./server.js";

const USAGE = `Usage:
  jaato-web-coder-server serve --config /etc/jaato-web-coder/server.yaml
  jaato-web-coder-server init  --dir /etc/jaato-web-coder [--app-id jaato-web-coder] [--force]

serve   discover the issuer, open the bind channel to the daemon, listen.
init    generate the app credential and the session secret (mode 0600), write
        a server.yaml template beside them, and print the daemon-side
        --ws-app-credentials entry.
`;

function arg(argv: string[], name: string): string | undefined {
  const i = argv.indexOf(name);
  if (i < 0) return undefined;
  const v = argv[i + 1];
  if (v === undefined || v.startsWith("--")) throw new Error(`${name} needs a value`);
  return v;
}

export function configTemplate(appId: string): string {
  return `# jaato-web-coder-server — see docs/design/web-server-bff.md §7 and §11
listen: 127.0.0.1:8443                      # behind the reverse proxy; the proxy terminates TLS
public_url: https://jaato.example.org

daemon:
  url: wss://jaato.example.org/daemon        # what the BROWSER connects to
  bind_url: ws://127.0.0.1:8080              # what this server connects to
  app_id: ${appId}
  app_credential_file: app.credential        # relative to this file

mode: direct

auth:
  kind: oidc
  oidc:
    issuer: https://jaato.example.org/auth/realms/jaato-web-coder-shell
    # backchannel_url: http://127.0.0.1:8180   # discovery/token/JWKS over loopback (Keycloak: --hostname-backchannel-dynamic=true)
    client_id: ${appId}
    client_secret_file: oidc.secret          # paste the client secret from Keycloak here, mode 0600
    scopes: [openid, profile]
    subject_claim: preferred_username
    # required_role: jaato-user

session:
  secret_file: session.secret
  ttl: 8h
  cookie_name: jaato_web_coder_session

ticket:
  ttl_seconds: 60
`;
}

export function runInit(argv: string[]): number {
  const dir = resolve(arg(argv, "--dir") ?? ".");
  const appId = arg(argv, "--app-id") ?? "jaato-web-coder";
  const force = argv.includes("--force");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const credential = randomBytes(32).toString("base64url");
  const files: Array<[string, string, number]> = [
    ["app.credential", credential + "\n", 0o600],
    ["session.secret", randomBytes(48).toString("base64url") + "\n", 0o600],
    ["server.yaml", configTemplate(appId), 0o600],
  ];
  for (const [name] of files) {
    if (!force && existsSync(join(dir, name))) {
      process.stderr.write(`refusing to overwrite ${join(dir, name)} (pass --force)\n`);
      return 2;
    }
  }
  for (const [name, content, mode] of files) writeFileSync(join(dir, name), content, { mode });
  process.stdout.write(`wrote ${dir}/{app.credential,session.secret,server.yaml} (mode 0600)

Daemon side — put this in the file named by --ws-app-credentials (mode 0600):
  {"${appId}": "${credential}"}

Then: edit ${dir}/server.yaml (public_url, daemon.url, issuer), paste the
Keycloak client secret into ${dir}/oidc.secret (mode 0600), and run
  jaato-web-coder-server serve --config ${dir}/server.yaml
`);
  return 0;
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const cmd = argv[0];
  if (!cmd || cmd === "-h" || cmd === "--help") { process.stdout.write(USAGE); return cmd ? 0 : 2; }
  if (cmd === "init") return runInit(argv.slice(1));
  if (cmd === "serve") {
    const path = arg(argv, "--config");
    if (!path) { process.stderr.write("serve needs --config PATH\n"); return 2; }
    const config = loadConfig(path);
    const running = await startServer(config);
    const stop = () => { running.close().then(() => process.exit(0), () => process.exit(1)); };
    process.on("SIGINT", stop);
    process.on("SIGTERM", stop);
    return -1; // keep running
  }
  process.stderr.write(`unknown command ${cmd}\n\n${USAGE}`);
  return 2;
}

export function runCli(): void {
  main().then((code) => { if (code >= 0) process.exit(code); }, (err) => {
    process.stderr.write(`jaato-web-coder-server: ${err instanceof ConfigError ? err.message : (err as Error).stack ?? String(err)}\n`);
    process.exit(1);
  });
}
