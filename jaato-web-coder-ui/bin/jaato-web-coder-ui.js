#!/usr/bin/env node
/**
 * ``jaato-web-coder-ui`` — the launcher that ships in the ``@jaato/web-coder-ui`` package.
 *
 * The browser client is a static bundle (``dist/``, produced by
 * ``vite build``); all this script does is serve that bundle from a
 * local port, tell the page which daemon to talk to, and open a browser
 * on it.  It is deliberately dependency-free (Node's ``http`` and ``fs``
 * only) so ``npx @jaato/web-coder-ui`` installs nothing but the bundle itself.
 *
 *     npx @jaato/web-coder-ui                              # ws://127.0.0.1:8080, token from ~/.jaato/ws.token
 *     npx @jaato/web-coder-ui --daemon ws://build-box:8080 --token-file ./ws.token
 *     npx @jaato/web-coder-ui --port 9000 --no-open
 *
 * How the page learns about the daemon
 * ------------------------------------
 * The bundle knows nothing about this launcher; it fetches ``./config.json``
 * from wherever it was served and, when that answers with
 * ``{"daemon": ..., "token": ..., "autoConnect": true}``, pre-fills the
 * connect form and connects.  Anyone hosting ``dist/`` behind nginx or
 * similar can publish the same file by hand, or none at all, in which
 * case the form starts empty.  See ``src/app/launcherConfig.ts``.
 *
 * Where the token goes
 * --------------------
 * The daemon's bearer token grants full control of the agent, so the
 * launcher only ever hands it to the page on a **loopback** bind: with
 * ``--host 0.0.0.0`` the token is left out of ``config.json`` and the
 * person at the browser types it.  Two more guards apply even on
 * loopback: ``config.json`` is served as JSON with ``nosniff`` (a page on
 * another origin cannot read it — a cross-origin ``fetch`` has no CORS
 * grant and a ``<script src>`` of a JSON document does not execute), and
 * every request must carry a ``Host`` header naming the address we bound,
 * which closes the DNS-rebinding route to the same file.
 */
import { createServer } from "node:http";
import { promises as fs, existsSync, readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = join(HERE, "..", "dist");
const DEFAULT_DAEMON = "ws://127.0.0.1:8080";
const DEFAULT_TOKEN_FILE = join(homedir(), ".jaato", "ws.token");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

const USAGE = `Usage: jaato-web-coder-ui [options]

Serve the jaato browser client and open it against a daemon started with
\`python -m server --web-socket [HOST:]PORT\`.

Options:
  --daemon URL        WebSocket URL of the daemon (default ${DEFAULT_DAEMON})
  --token TOKEN       Bearer token to hand to the page
  --token-file PATH   Read the token from PATH (default ~/.jaato/ws.token when
                      the daemon URL is a loopback address and the file exists)
  --no-token          Do not pass any token; type it in the connect form
  --host HOST         Address to serve on (default 127.0.0.1). The token is
                      only handed out on a loopback bind.
  --port PORT         Port to serve on (default 5180; 0 picks a free port)
  --no-open           Do not open a browser
  --root DIR          Serve this directory instead of the bundled dist/
  -h, --help          Show this help
  -v, --version       Print the package version
`;

/** Parse ``argv`` into launcher options; throws on an unknown flag. */
export function parseArgs(argv) {
  const opts = {
    daemon: DEFAULT_DAEMON,
    token: undefined,
    tokenFile: undefined,
    noToken: false,
    host: "127.0.0.1",
    port: 5180,
    open: true,
    root: DEFAULT_ROOT,
    help: false,
    version: false,
  };
  const takeValue = (i, flag) => {
    const v = argv[i + 1];
    if (v === undefined) throw new Error(`${flag} needs a value`);
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const eq = a.indexOf("=");
    const flag = eq > 0 && a.startsWith("--") ? a.slice(0, eq) : a;
    const inline = eq > 0 && a.startsWith("--") ? a.slice(eq + 1) : undefined;
    const value = () => (inline !== undefined ? inline : (i++, takeValue(i - 1, flag)));
    switch (flag) {
      case "--daemon": opts.daemon = value(); break;
      case "--token": opts.token = value(); break;
      case "--token-file": opts.tokenFile = value(); break;
      case "--no-token": opts.noToken = true; break;
      case "--host": opts.host = value(); break;
      case "--port": {
        const n = Number(value());
        if (!Number.isInteger(n) || n < 0 || n > 65535) throw new Error("--port must be an integer between 0 and 65535");
        opts.port = n;
        break;
      }
      case "--open": opts.open = true; break;
      case "--no-open": opts.open = false; break;
      case "--root": opts.root = resolve(value()); break;
      case "-h": case "--help": opts.help = true; break;
      case "-v": case "--version": opts.version = true; break;
      default: throw new Error(`unknown option ${a}\n\n${USAGE}`);
    }
  }
  return opts;
}

/** ``true`` for the hosts a loopback bind answers to. */
export function isLoopbackHost(host) {
  const h = host.replace(/^\[|\]$/g, "").toLowerCase();
  return h === "localhost" || h === "::1" || /^127\.\d+\.\d+\.\d+$/.test(h);
}

function daemonHost(url) {
  try { return new URL(url).hostname; } catch { return ""; }
}

/**
 * Decide which token (if any) reaches the page.
 *
 * Explicit ``--token`` / ``--token-file`` win; otherwise the default
 * token file is used only for a loopback daemon, because that file
 * belongs to the daemon on *this* machine and would be the wrong secret
 * for any other.
 */
export function resolveToken(opts, { readFile = (p) => readFileSync(p, "utf8"), exists = existsSync } = {}) {
  if (opts.noToken) return { token: undefined, source: "disabled" };
  if (opts.token !== undefined) return { token: opts.token, source: "flag" };
  if (opts.tokenFile !== undefined) return { token: readFile(opts.tokenFile).trim(), source: opts.tokenFile };
  if (isLoopbackHost(daemonHost(opts.daemon)) && exists(DEFAULT_TOKEN_FILE)) {
    return { token: readFile(DEFAULT_TOKEN_FILE).trim(), source: DEFAULT_TOKEN_FILE };
  }
  return { token: undefined, source: "none" };
}

function send(res, status, body, headers = {}) {
  res.writeHead(status, { "X-Content-Type-Options": "nosniff", ...headers });
  res.end(body);
}

/**
 * Build (but do not start) the static server.
 *
 * ``config`` is what ``GET /config.json`` answers; the caller decides
 * whether it carries a token.  Everything under ``root`` is served
 * read-only, with the SPA fallback (any path without a file behind it
 * gets ``index.html``) and immutable caching for Vite's hashed assets.
 * ``allowedHosts`` is the set of ``Host`` header values accepted (read
 * on every request, so it may be filled after ``listen``); ``null``
 * accepts any host.  A request naming another host is refused with 421.
 */
export function createStaticServer({ root, config, allowedHosts }) {
  const rootAbs = resolve(root);
  const configBody = JSON.stringify(config);
  const hostOk = (req) => {
    if (!allowedHosts) return true;
    const h = (req.headers.host ?? "").toLowerCase();
    return allowedHosts.has(h);
  };

  return createServer(async (req, res) => {
    if (!hostOk(req)) return send(res, 421, "Misdirected Request\n", { "Content-Type": "text/plain" });
    if (req.method !== "GET" && req.method !== "HEAD") {
      return send(res, 405, "Method Not Allowed\n", { "Content-Type": "text/plain", Allow: "GET, HEAD" });
    }
    let pathname;
    try { pathname = decodeURIComponent(new URL(req.url ?? "/", "http://x").pathname); }
    catch { return send(res, 400, "Bad Request\n", { "Content-Type": "text/plain" }); }

    if (pathname === "/config.json") {
      return send(res, 200, configBody, { "Content-Type": MIME[".json"], "Cache-Control": "no-store" });
    }

    let file = resolve(rootAbs, "." + pathname);
    if (file !== rootAbs && !file.startsWith(rootAbs + sep)) {
      return send(res, 403, "Forbidden\n", { "Content-Type": "text/plain" });
    }
    let stat = await fs.stat(file).catch(() => null);
    if (stat?.isDirectory()) { file = join(file, "index.html"); stat = await fs.stat(file).catch(() => null); }
    if (!stat?.isFile()) { file = join(rootAbs, "index.html"); stat = await fs.stat(file).catch(() => null); }
    if (!stat?.isFile()) return send(res, 404, "Not Found\n", { "Content-Type": "text/plain" });

    const ext = extname(file).toLowerCase();
    const hashed = file.startsWith(join(rootAbs, "assets") + sep);
    const headers = {
      "Content-Type": MIME[ext] ?? "application/octet-stream",
      "Content-Length": String(stat.size),
      "Cache-Control": hashed ? "public, max-age=31536000, immutable" : "no-cache",
    };
    if (req.method === "HEAD") return send(res, 200, undefined, headers);
    send(res, 200, await fs.readFile(file), headers);
  });
}

function openBrowser(url) {
  const [cmd, args] =
    process.platform === "darwin" ? ["open", [url]]
    : process.platform === "win32" ? ["cmd", ["/c", "start", "", url]]
    : ["xdg-open", [url]];
  try {
    spawn(cmd, args, { stdio: "ignore", detached: true }).on("error", () => {}).unref();
  } catch { /* no opener on this box; the URL is printed anyway */ }
}

function packageVersion() {
  try { return JSON.parse(readFileSync(join(HERE, "..", "package.json"), "utf8")).version; }
  catch { return "unknown"; }
}

/** Start serving per ``opts`` (from ``parseArgs``); resolves to the listening server. */
export async function main(opts) {
  if (opts.help) { process.stdout.write(USAGE); return null; }
  if (opts.version) { process.stdout.write(packageVersion() + "\n"); return null; }
  if (!existsSync(join(opts.root, "index.html"))) {
    throw new Error(`no index.html under ${opts.root} — run \`npm run build\` first, or pass --root`);
  }

  const loopback = isLoopbackHost(opts.host);
  const { token, source } = resolveToken(opts);
  const config = { daemon: opts.daemon, autoConnect: true };
  if (token && loopback) config.token = token;

  // On a loopback bind only the loopback names are accepted in ``Host``,
  // so a request that names any other host (a rebinding attack) never
  // sees the token.  A non-loopback bind serves no token, and the names
  // people reach it by are not ours to know, so it takes any ``Host``.
  const allowed = loopback ? new Set() : null;
  const server = createStaticServer({ root: opts.root, config, allowedHosts: allowed });
  await new Promise((ok, fail) => server.once("error", fail).listen(opts.port, opts.host, ok));
  const { port } = server.address();
  if (allowed) for (const h of ["localhost", "127.0.0.1", "[::1]", opts.host]) allowed.add(`${h}:${port}`.toLowerCase());

  const wildcard = opts.host === "0.0.0.0" || opts.host === "::";
  const hostForUrl = wildcard ? "localhost" : opts.host.includes(":") ? `[${opts.host}]` : opts.host;
  const url = `http://${hostForUrl}:${port}/`;
  const lines = [`jaato-web-coder-ui ${packageVersion()} serving ${opts.root}`, `  open      ${url}${wildcard ? "  (bound to all interfaces)" : ""}`, `  daemon    ${opts.daemon}`];
  if (config.token) lines.push(`  token     from ${source}`);
  else if (token && !loopback) lines.push(`  token     NOT handed to the page: ${opts.host} is not a loopback bind; type it in the form`);
  else lines.push(`  token     none (${source}); type it in the form, or leave empty for --ws-unsafe-no-auth`);
  process.stdout.write(lines.join("\n") + "\n");
  if (opts.open) openBrowser(url);
  return server;
}

// npm installs the bin as a symlink under node_modules/.bin, so argv[1]
// must be resolved through the link before it can equal this module.
function invokedDirectly() {
  if (!process.argv[1]) return false;
  try { return realpathSync(process.argv[1]) === fileURLToPath(import.meta.url); }
  catch { return false; }
}
const isMain = invokedDirectly();
if (isMain) {
  let opts;
  try { opts = parseArgs(process.argv.slice(2)); }
  catch (err) { process.stderr.write(`jaato-web-coder-ui: ${err.message}\n`); process.exit(2); }
  main(opts).then((server) => {
    if (!server) return;
    const stop = () => server.close(() => process.exit(0));
    process.on("SIGINT", stop);
    process.on("SIGTERM", stop);
  }).catch((err) => {
    process.stderr.write(`jaato-web-coder-ui: ${err.message}\n`);
    process.exit(1);
  });
}
