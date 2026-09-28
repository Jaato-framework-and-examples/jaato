/**
 * Installing a toolchain and its language server into a workspace
 * (docs/design/web-coder-environment-bootstrap.md §4.2, §5).
 *
 * Everything runs here, in the BFF, with the workspace owner's consent, and
 * never in the confined runner.  Everything lands under ``<ws>/.home``:
 *
 * | What | Where | Why a confined session may run it |
 * |---|---|---|
 * | mise installs | ``.home/.local/share/mise/installs/<tool>/<ver>/`` | ``.home/.local/share/**\/bin/*`` is granted ``ix`` |
 * | links to their binaries | ``.home/.local/bin/<name>`` | on every command's ``PATH``, granted ``ix`` |
 * | basedpyright | its own venv under ``.home/.local/share/jaato-lsp/`` | as above |
 * | typescript-language-server | ``npm -g --prefix .home/.local/share/jaato-lsp/npm`` | started through the linked ``node``, so its ``.mjs`` entry is read, not exec'd |
 * | gopls | ``GOBIN=.home/.local/bin`` | as above |
 * | jdtls | a checksum-verified Eclipse milestone under ``.home/.local/share/jaato-lsp/jdtls/<v>/``, run by a mise JDK | its ``java`` is under ``.local/share/**\/bin/``; the JDK's ``.so`` files need the ``m`` grant the bootstrap's AppArmor fragment carries |
 *
 * jdtls is started with ``java -jar <launcher>`` rather than its Python
 * wrapper: one exec fewer under confinement, and every flag the wrapper
 * would add is stated here, the heap ceiling above all (#806: nothing reaps
 * a language server at session end, so a bounded heap is what bounds the
 * cost of each one left running).  Its ``-data`` directory is
 * ``${jdtlsStateRoot}``, the framework's sibling-of-the-workspace state
 * directory: Eclipse refuses a ``-data`` nested inside an imported project,
 * and the lsp plugin grants that path ``rw`` because it sees it in
 * ``.lsp.json``.
 *
 * basedpyright deliberately gets its OWN venv rather than the workspace
 * tool-venv (``.jaato/tool-venv``): that venv is created by the runner on
 * first use, may not exist yet when a workspace is bound, and is the model's
 * to ``pip install`` into and out of.
 *
 * Every subprocess gets a CLEAN environment: HOME and the XDG directories
 * under the workspace, mise's directories under them, ``PATH`` with the
 * workspace's linked binaries first, and only the proxy / CA variables from
 * this process.  The BFF's own environment carries its secrets' paths and is
 * not the installer's business.  mise is stopped from reading a project
 * config above ``.home`` (``MISE_CEILING_PATHS``): a repository's
 * ``mise.toml`` is the repository's, and trusting it here would let a cloned
 * repository choose what the BFF downloads.
 */
import { spawn } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, readdirSync, readlinkSync, rmSync, statSync, symlinkSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import {
  JDTLS_MIRROR,
  LOCAL_BIN,
  LSP_DIR,
  MISE_CACHE_DIR,
  MISE_CONFIG_DIR,
  MISE_CONFIG_PATH,
  MISE_DATA_DIR,
  MISE_STATE_DIR,
  TOOLCHAINS,
  type ServerId,
  type ToolId,
} from "./catalog.js";

/** One subprocess the installer runs. */
export interface RunSpec {
  command: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
  timeoutMs: number;
  signal: AbortSignal;
  /** Each line of stdout and stderr, as it arrives. */
  onLine: (line: string, stream: "stdout" | "stderr") => void;
}

/** Runs a subprocess to completion; the test seam. */
export type ProcessRunner = (spec: RunSpec) => Promise<{ code: number | null; signal: string | null }>;

export class InstallError extends Error {
  override name = "InstallError";
}

export class InstallCancelled extends Error {
  override name = "InstallCancelled";
}

/** Environment variables passed through from this process: proxies and CA bundles only. */
export const PASSTHROUGH_ENV = [
  "LANG", "LC_ALL", "TZ",
  "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
  "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "PIP_INDEX_URL", "PIP_CERT",
];

const SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin";

/** The spawn-based {@link ProcessRunner}: no shell, own process group so a cancel kills the whole tree. */
export const spawnRunner: ProcessRunner = (spec) => new Promise((ok, fail) => {
  const child = spawn(spec.command, spec.args, { cwd: spec.cwd, env: spec.env, stdio: ["ignore", "pipe", "pipe"], detached: true });
  const kill = () => { try { if (child.pid) process.kill(-child.pid, "SIGTERM"); } catch { /* gone */ } };
  const timer = setTimeout(kill, spec.timeoutMs);
  spec.signal.addEventListener("abort", kill, { once: true });
  const pipe = (stream: NodeJS.ReadableStream | null, name: "stdout" | "stderr") => {
    let buf = "";
    stream?.setEncoding("utf8");
    stream?.on("data", (chunk: string) => {
      buf += chunk;
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) { spec.onLine(buf.slice(0, i).replace(/\r$/, ""), name); buf = buf.slice(i + 1); }
    });
    stream?.on("end", () => { if (buf) spec.onLine(buf, name); });
  };
  pipe(child.stdout, "stdout");
  pipe(child.stderr, "stderr");
  child.on("error", (e) => { clearTimeout(timer); fail(e); });
  child.on("close", (code, signal) => { clearTimeout(timer); ok({ code, signal }); });
});

export interface InstallerOptions {
  /** The workspace's resolved absolute path. */
  workspace: string;
  mise: string;
  python: string;
  paranoid: boolean;
  timeoutMs: number;
  run: ProcessRunner;
  signal: AbortSignal;
  log: (line: string) => void;
}

/** A toolchain as installed: the version asked for, the directory mise reports, the names linked. */
export interface InstalledToolchain {
  tool: ToolId;
  version: string;
  /** mise's install directory, workspace-relative; ``null`` for python. */
  installDir: string | null;
  /** Names linked into ``.home/.local/bin``. */
  bin: string[];
}

/** What a jdtls install needs besides its own version. */
export interface JdtlsOptions {
  /** The mise ``java`` version that RUNS jdtls (recent jdtls needs 21+), independent of the project's JDK. */
  runtime: string;
  /** ``-Xmx`` for the server, e.g. ``1G``. */
  maxHeap: string;
  /** Base URL of the milestone tree; defaults to {@link JDTLS_MIRROR}. */
  mirror?: string;
}

/** A language server as installed, and how ``.lsp.json`` starts it (absolute paths). */
export interface InstalledServer {
  id: ServerId;
  version: string;
  language: string;
  command: string;
  args: string[];
  /** A runtime the server brings of its own (jdtls's JDK), workspace-relative, so the AppArmor fragment can cover it. */
  runtimeDir?: string;
}

/**
 * Downloads a jdtls milestone, checks it against the ``.sha256`` published
 * beside it, and extracts it (members that would land outside the target
 * refused).  Python rather than Node's ``fetch``: ``urllib`` honours
 * ``HTTPS_PROXY`` and ``SSL_CERT_FILE``, which the clean environment passes
 * through, and it runs as one step the job can cancel.
 */
export const JDTLS_FETCH_SCRIPT = String.raw`
import hashlib, os, re, shutil, sys, tarfile, tempfile, urllib.request
mirror, version, dest = sys.argv[1:4]
base = mirror.rstrip("/") + "/" + version + "/"
def get(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()
try:
    name = get(base + "latest.txt").decode().strip()
except Exception:
    listing = get(base).decode("utf-8", "replace")
    names = sorted(set(re.findall(r"jdt-language-server-" + re.escape(version) + r"-\d+\.tar\.gz(?![.\w])", listing)))
    name = names[-1] if names else ""
if not re.fullmatch(r"jdt-language-server-[\w.-]+\.tar\.gz", name or ""):
    sys.exit("no jdtls " + version + " archive found under " + base)
print("downloading " + base + name, flush=True)
data = get(base + name)
want = get(base + name + ".sha256").decode().split()[0].lower()
got = hashlib.sha256(data).hexdigest()
if got != want:
    sys.exit("checksum mismatch for " + name + ": got " + got + ", published " + want)
os.makedirs(os.path.dirname(dest), exist_ok=True)
tmp = tempfile.mkdtemp(prefix=".jdtls-", dir=os.path.dirname(dest))
try:
    arc = os.path.join(tmp, name)
    with open(arc, "wb") as f:
        f.write(data)
    out = os.path.join(tmp, "x")
    with tarfile.open(arc) as t:
        root = os.path.realpath(out)
        for m in t.getmembers():
            p = os.path.realpath(os.path.join(out, m.name))
            if not (p == root or p.startswith(root + os.sep)) or m.issym() or m.islnk() or m.isdev():
                sys.exit("refusing archive member " + m.name)
        t.extractall(out)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.replace(out, dest)
    print("extracted " + name, flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
`;

/** jdtls's shared configuration directory for this host. */
export function jdtlsConfigDir(platform: string = process.platform, arch: string = process.arch): string {
  const os = platform === "darwin" ? "mac" : platform === "win32" ? "win" : "linux";
  return arch === "arm64" && os !== "win" ? `config_${os}_arm` : `config_${os}`;
}

export class Installer {
  private readonly o: InstallerOptions;
  constructor(opts: InstallerOptions) { this.o = opts; }

  private abs(rel: string): string { return join(this.o.workspace, rel); }

  /** The clean environment every step runs with. */
  env(extra: Record<string, string> = {}): Record<string, string> {
    const home = this.abs(".home");
    const env: Record<string, string> = {};
    for (const k of PASSTHROUGH_ENV) { const v = process.env[k]; if (v !== undefined) env[k] = v; }
    Object.assign(env, {
      HOME: home,
      XDG_CONFIG_HOME: join(home, ".config"),
      XDG_CACHE_HOME: join(home, ".cache"),
      XDG_DATA_HOME: join(home, ".local/share"),
      XDG_STATE_HOME: join(home, ".local/state"),
      PATH: `${this.abs(LOCAL_BIN)}:${SYSTEM_PATH}`,
      MISE_DATA_DIR: this.abs(MISE_DATA_DIR),
      MISE_CONFIG_DIR: this.abs(MISE_CONFIG_DIR),
      MISE_CACHE_DIR: this.abs(MISE_CACHE_DIR),
      MISE_STATE_DIR: this.abs(MISE_STATE_DIR),
      MISE_GLOBAL_CONFIG_FILE: this.abs(MISE_CONFIG_PATH),
      MISE_CEILING_PATHS: home,
      MISE_YES: "1",
    });
    if (this.o.paranoid) env.MISE_PARANOID = "1";
    return { ...env, ...extra };
  }

  /** Run one step; a non-zero exit is an {@link InstallError} naming the step and its last output line. */
  async step(what: string, command: string, args: string[], extraEnv: Record<string, string> = {}): Promise<string[]> {
    if (this.o.signal.aborted) throw new InstallCancelled("cancelled");
    const out: string[] = [];
    let last = "";
    this.o.log(`$ ${[command, ...args].join(" ")}`);
    let result: { code: number | null; signal: string | null };
    try {
      result = await this.o.run({
        command, args, cwd: this.abs(".home"), env: this.env(extraEnv), timeoutMs: this.o.timeoutMs, signal: this.o.signal,
        onLine: (line, stream) => { if (stream === "stdout") out.push(line); if (line.trim()) { last = line.trim(); this.o.log(line); } },
      });
    } catch (e) {
      throw new InstallError(`${what}: could not run ${command}: ${(e as Error).message}`);
    }
    if (this.o.signal.aborted) throw new InstallCancelled("cancelled");
    if (result.code !== 0) {
      throw new InstallError(`${what} failed (${result.signal ? `signal ${result.signal}` : `exit ${result.code}`})${last ? `: ${last}` : ""}`);
    }
    return out;
  }

  /** Install ``tool@version`` with mise and link its ``bin/`` into ``.home/.local/bin``. */
  async installToolchain(tool: ToolId, version: string): Promise<InstalledToolchain> {
    const spec = TOOLCHAINS[tool];
    mkdirSync(this.abs(LOCAL_BIN), { recursive: true });
    if (!spec.mise) return { tool, version, installDir: null, bin: [] };
    const installDir = await this._miseInstall(`${spec.mise}@${version}`);
    const miseRoot = this.abs(MISE_DATA_DIR);
    const bin = linkBinaries(join(installDir, "bin"), this.abs(LOCAL_BIN), miseRoot, this.o.log);
    return { tool, version, installDir: relative(this.o.workspace, installDir), bin };
  }

  /** Install one pinned language server; ``null`` when it has no install route for this toolchain. */
  /** ``mise install`` + ``mise where``; the directory, which must be under the workspace's mise data directory. */
  private async _miseInstall(ref: string): Promise<string> {
    await this.step(`installing ${ref}`, this.o.mise, ["install", ref]);
    const where = (await this.step(`locating ${ref}`, this.o.mise, ["where", ref])).map((l) => l.trim()).filter(Boolean).pop();
    if (!where) throw new InstallError(`locating ${ref}: mise named no directory`);
    const dir = resolve(where);
    if (!dir.startsWith(this.abs(MISE_DATA_DIR) + sep)) throw new InstallError(`locating ${ref}: ${dir} is outside the workspace's mise directory`);
    return dir;
  }

  /** jdtls: its runtime JDK, the verified milestone, and the ``java -jar`` command line. */
  private async _installJdtls(version: string, o: JdtlsOptions): Promise<InstalledServer> {
    const javaHome = await this._miseInstall(`java@${o.runtime}`);
    const dir = join(this.abs(LSP_DIR), "jdtls", version);
    const launcher = () => {
      try { return readdirSync(join(dir, "plugins")).filter((n) => /^org\.eclipse\.equinox\.launcher_.*\.jar$/.test(n)).sort().pop() ?? null; } catch { return null; }
    };
    if (!launcher()) {
      await this.step(`downloading jdtls ${version}`, this.o.python, ["-c", JDTLS_FETCH_SCRIPT, o.mirror ?? JDTLS_MIRROR, version, dir]);
    }
    const jar = launcher();
    if (!jar) throw new InstallError(`jdtls ${version}: the archive has no equinox launcher under plugins/`);
    // The JVM ignores TMPDIR and wants its temp directory to exist at start; /tmp is denied under confinement.
    const tmp = this.abs(".home/.cache/jdtls");
    mkdirSync(tmp, { recursive: true });
    return {
      id: "jdtls", version, language: "java",
      command: join(javaHome, "bin", "java"),
      args: [
        "-Declipse.application=org.eclipse.jdt.ls.core.id1",
        "-Dosgi.bundles.defaultStartLevel=4",
        "-Declipse.product=org.eclipse.jdt.ls.core.product",
        "-Dosgi.checkConfiguration=true",
        `-Dosgi.sharedConfiguration.area=${join(dir, jdtlsConfigDir())}`,
        "-Dosgi.sharedConfiguration.area.readOnly=true",
        "-Dosgi.configuration.cascaded=true",
        `-Djava.io.tmpdir=${tmp}`,
        "-Xms100m", `-Xmx${o.maxHeap}`,
        "-XX:+UseParallelGC", "-XX:GCTimeRatio=4", "-XX:AdaptiveSizePolicyWeight=90", "-XX:-UsePerfData",
        "-Dsun.zip.disableMemoryMapping=true",
        "--add-modules=ALL-SYSTEM",
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "-jar", join(dir, "plugins", jar),
        "-configuration", "${jdtlsStateRoot}/.jaato-config",
        "-data", "${jdtlsStateRoot}",
      ],
      runtimeDir: relative(this.o.workspace, javaHome),
    };
  }

  async installServer(id: ServerId, version: string, extra: { typescriptVersion?: string; jdtls?: JdtlsOptions } = {}): Promise<InstalledServer> {
    const lspDir = this.abs(LSP_DIR);
    mkdirSync(lspDir, { recursive: true });
    if (id === "jdtls") {
      if (!extra.jdtls) throw new InstallError("jdtls: no runtime configured");
      return this._installJdtls(version, extra.jdtls);
    }
    if (id === "basedpyright") {
      const venv = join(lspDir, "basedpyright");
      if (!existsSync(join(venv, "bin", "python"))) await this.step("creating the basedpyright venv", this.o.python, ["-m", "venv", venv]);
      await this.step(`installing basedpyright ${version}`, join(venv, "bin", "python"), ["-m", "pip", "install", "--disable-pip-version-check", "--no-input", `basedpyright==${version}`]);
      return { id, version, language: "python", command: join(venv, "bin", "basedpyright-langserver"), args: ["--stdio"] };
    }
    if (id === "typescript-language-server") {
      const prefix = join(lspDir, "npm");
      const pkgs = [`typescript-language-server@${version}`];
      if (extra.typescriptVersion) pkgs.push(`typescript@${extra.typescriptVersion}`);
      await this.step(`installing typescript-language-server ${version}`, this.abs(`${LOCAL_BIN}/npm`), ["install", "-g", "--no-fund", "--no-audit", "--prefix", prefix, ...pkgs]);
      return {
        id, version, language: "typescript",
        command: this.abs(`${LOCAL_BIN}/node`),
        args: [join(prefix, "lib", "node_modules", "typescript-language-server", "lib", "cli.mjs"), "--stdio"],
      };
    }
    // gopls, with the linked go; GOTOOLCHAIN=local so go does not fetch another toolchain behind the pin.
    const home = this.abs(".home");
    await this.step(`installing gopls ${version}`, this.abs(`${LOCAL_BIN}/go`), ["install", `golang.org/x/tools/gopls@${version}`], {
      GOBIN: this.abs(LOCAL_BIN), GOPATH: join(home, "go"), GOCACHE: join(home, ".cache", "go-build"), GOTOOLCHAIN: "local",
    });
    return { id, version, language: "go", command: this.abs(`${LOCAL_BIN}/gopls`), args: [] };
  }
}

/** Whether ``path`` is a symlink this bootstrap made: one resolving into the workspace's mise directory. */
export function isOurLink(path: string, miseRoot: string): boolean {
  try {
    if (!lstatSync(path).isSymbolicLink()) return false;
    const target = resolve(dirname(path), readlinkSync(path));
    return target.startsWith(miseRoot + sep);
  } catch { return false; }
}

/**
 * Link each executable entry of ``srcBin`` into ``destBin`` with a RELATIVE
 * symlink (the workspace path is the same for the BFF and the runner, and a
 * relative link survives the workspace being moved).  A destination that
 * exists and is not one of our links is left alone and reported.
 */
export function linkBinaries(srcBin: string, destBin: string, miseRoot: string, log: (msg: string) => void): string[] {
  const linked: string[] = [];
  let names: string[] = [];
  try { names = readdirSync(srcBin).sort(); } catch { return linked; }
  for (const name of names) {
    const src = join(srcBin, name);
    let mode = 0;
    try { mode = statSync(src).mode; } catch { continue; }
    if ((mode & 0o111) === 0) continue;
    const dest = join(destBin, name);
    if (existsSync(dest) || isSymlink(dest)) {
      if (!isOurLink(dest, miseRoot)) { log(`kept ${relativeFromHome(dest)}: it is not a link this bootstrap made`); continue; }
      rmSync(dest);
    }
    symlinkSync(relative(destBin, src), dest);
    linked.push(name);
  }
  return linked;
}

/** Remove the links in ``destBin`` named ``names`` that are ours; returns the names removed. */
export function unlinkBinaries(destBin: string, names: string[], miseRoot: string): string[] {
  const removed: string[] = [];
  for (const name of names) {
    const dest = join(destBin, name);
    if (isOurLink(dest, miseRoot)) { rmSync(dest); removed.push(name); }
  }
  return removed;
}

function isSymlink(p: string): boolean {
  try { return lstatSync(p).isSymbolicLink(); } catch { return false; }
}

function relativeFromHome(p: string): string {
  const i = p.lastIndexOf("/.home/");
  return i >= 0 ? p.slice(i + 1) : p;
}
