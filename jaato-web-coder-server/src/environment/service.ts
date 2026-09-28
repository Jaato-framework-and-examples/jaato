/**
 * The environment bootstrap service: detection, consent, installation and
 * the managed files, for one BFF (docs/design/web-coder-environment-bootstrap.md).
 *
 * The page drives it; nothing here acts on its own:
 *
 * | Operation | Does |
 * |---|---|
 * | {@link EnvironmentService.status} | what is bound, what the operator allows, what the workspace's files propose (minus what is bound or declined), and the running install |
 * | {@link EnvironmentService.bind} | starts an install job: mise, the links, the pinned language server, then the five managed files |
 * | {@link EnvironmentService.unbind} | removes the links and the tool from the managed files; the installed data is KEPT, so a rebind is fast |
 * | {@link EnvironmentService.decline} | remembers "not now" for ``(user, workspace, tool)`` |
 * | {@link EnvironmentService.refreshGuidance} | rewrites the repository-guidance pointer; the page calls it after a clone or a pull |
 *
 * **Ownership.**  Every operation first asks the daemon whether the signed-in
 * user owns the workspace ({@link WorkspaceOwnership}), because these routes
 * install software into it.  Containment under ``workspace_root`` alone would
 * let any signed-in user install into any other user's workspace on the same
 * host.  An answer the daemon could not give (the bind channel down, a daemon
 * below protocol 1.30) refuses: the one direction that is safe.
 *
 * **One job per workspace.**  Two installs into one mise directory at once is
 * two writers on one tree; a second bind while one runs is refused (409).
 * Jobs live in memory and are forgotten an hour after they finish; a BFF
 * restart mid-install loses the job, and the manifest, written last, never
 * names a toolchain whose install did not finish.
 */
import { randomBytes, randomUUID } from "node:crypto";
import { mkdirSync, realpathSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, sep } from "node:path";
import { removeManagedFile, writeManagedFile, type ManagedFile, type ManagedWriteOutcome } from "../managed-files.js";
import { LOCAL_BIN, MISE_DATA_DIR, TOOLCHAINS, TOOL_IDS, matchToolVersion, type ServerId, type ToolId } from "./catalog.js";
import { detectWorkspace } from "./detect.js";
import {
  apparmorFragmentFile, apparmorFragmentIdentity, environmentInstructionsFile, environmentInstructionsIdentity, lspConfigFile, lspConfigIdentity,
  manifestFile, miseConfigFile, pinnedServer, readManifest, toolchainOfferFile, toolchainOfferIdentity, type BoundToolchain, type Manifest,
} from "./files.js";
import { findRepoGuidance, repoGuidanceFile, repoGuidanceIdentity } from "./guidance.js";
import { InstallCancelled, Installer, spawnRunner, unlinkBinaries, type JdtlsOptions, type ProcessRunner } from "./installer.js";
import type { FileEnvironmentStore } from "./store.js";

/** Whether ``user`` owns ``workspace`` on the daemon; ``null`` when the daemon could not be asked. */
export interface WorkspaceOwnership {
  owns(user: string, workspace: string): Promise<boolean | null>;
}

export class EnvironmentError extends Error {
  override name = "EnvironmentError";
  constructor(message: string, readonly status: 400 | 403 | 404 | 409 | 503 = 400) { super(message); }
}

export interface EnvironmentOptions {
  workspaceRoot: string;
  /** Allowed versions per toolchain (the operator's pin list). ``python`` needs none. */
  tools: Partial<Record<ToolId, string[]>>;
  /** Pinned language-server versions; an unpinned server is never installed. */
  lsp: Partial<Record<ServerId, string>>;
  /** The ``typescript`` version installed beside typescript-language-server. */
  typescriptVersion?: string;
  /** How jdtls is run: its own JDK, its heap ceiling, where it is downloaded from. */
  jdtls?: JdtlsOptions;
  mise: string;
  python: string;
  paranoid: boolean;
  installTimeoutMs: number;
  store: FileEnvironmentStore;
  ownership: WorkspaceOwnership;
  run?: ProcessRunner;
  log?: (msg: string) => void;
}

export type JobStatus = "running" | "done" | "failed" | "cancelled";

export interface JobView {
  id: string;
  workspace: string;
  tool: ToolId;
  version: string;
  status: JobStatus;
  /** The last lines of output, for the progress panel. */
  log: string[];
  error?: string;
  notes: string[];
  startedAt: string;
  finishedAt?: string;
}

interface Job extends JobView {
  sub: string;
  abort: AbortController;
  done: Promise<void>;
}

/** What the page may bind: a toolchain, its allowed versions and whether its language server is pinned. */
export interface AllowedTool {
  tool: ToolId;
  label: string;
  versions: string[];
  server: { id: ServerId; version: string } | null;
}

/** One proposal the page shows as a chip. */
export interface Proposal {
  tool: ToolId;
  label: string;
  /** The allowed version the repository's pin maps to, else the first allowed one. */
  version: string;
  /** The repository's pin, verbatim; ``null`` when its files name none. */
  pin: string | null;
  /** ``false`` when the repository pins a version the operator does not allow; ``version`` is then only a suggestion. */
  pinAllowed: boolean;
  source: string;
}

export interface EnvironmentStatus {
  workspace: string;
  toolchains: Array<Pick<BoundToolchain, "tool" | "version" | "bin" | "boundAt"> & { server: { id: ServerId; version: string } | null }>;
  allowed: AllowedTool[];
  proposals: Proposal[];
  declined: ToolId[];
  job: JobView | null;
  guidance: string[];
}

const LOG_TAIL = 200;
const JOB_RETENTION_MS = 60 * 60 * 1000;

export class EnvironmentService {
  private readonly o: EnvironmentOptions;
  private readonly _root: string;
  private readonly _run: ProcessRunner;
  private readonly _log: (msg: string) => void;
  private readonly _jobs = new Map<string, Job>();

  constructor(opts: EnvironmentOptions) {
    this.o = opts;
    this._root = realpathSafe(opts.workspaceRoot) ?? opts.workspaceRoot;
    this._run = opts.run ?? spawnRunner;
    this._log = opts.log ?? (() => undefined);
  }

  /** The toolchains the operator allows, in catalog order. */
  allowedTools(): AllowedTool[] {
    const out: AllowedTool[] = [];
    for (const tool of TOOL_IDS) {
      const server = pinnedServer(tool, this.o.lsp);
      if (tool === "python") {
        // Nothing to install but the server, so python is offered iff basedpyright is pinned.
        if (server) out.push({ tool, label: TOOLCHAINS[tool].label, versions: ["system"], server });
        continue;
      }
      const versions = this.o.tools[tool] ?? [];
      if (versions.length) out.push({ tool, label: TOOLCHAINS[tool].label, versions: [...versions], server });
    }
    return out;
  }

  /** Contain ``workspace`` under the root (symlinks resolved), then ask the daemon whether ``user`` owns it. */
  private async _authorize(user: string, workspace: string): Promise<string> {
    if (typeof workspace !== "string" || !workspace || !isAbsolute(workspace)) throw new EnvironmentError("workspace must be an absolute path");
    const real = realpathSafe(workspace);
    if (real === null || !real.startsWith(this._root + sep)) {
      throw new EnvironmentError("this server manages toolchains only for workspaces under its workspace_root", 404);
    }
    const owns = await this.o.ownership.owns(user, workspace);
    if (owns === null) throw new EnvironmentError("the daemon could not confirm you own this workspace; try again", 503);
    if (!owns) throw new EnvironmentError("no such workspace for this user", 404);
    return real;
  }

  async status(sub: string, user: string, workspace: string): Promise<EnvironmentStatus> {
    const real = await this._authorize(user, workspace);
    return this._status(sub, real, workspace);
  }

  private _status(sub: string, real: string, workspace: string): EnvironmentStatus {
    const manifest = readManifest(real);
    const allowed = this.allowedTools();
    // The offer follows the allow-list as well as the bindings, so every read refreshes it (a no-op when unchanged).
    this._writeOffer(real, manifest);
    const declined = this.o.store.declined(sub, workspace);
    const bound = new Set(manifest.toolchains.map((t) => t.tool));
    const proposals: Proposal[] = [];
    for (const d of detectWorkspace(real)) {
      const a = allowed.find((x) => x.tool === d.tool);
      if (!a || bound.has(d.tool) || declined.includes(d.tool)) continue;
      const matched = d.tool === "python" ? "system" : d.pin ? matchToolVersion(d.tool, d.pin, a.versions) : null;
      proposals.push({
        tool: d.tool, label: a.label, version: matched ?? a.versions[0]!, pin: d.pin,
        pinAllowed: d.pin === null || matched !== null, source: d.source,
      });
    }
    return {
      workspace,
      toolchains: manifest.toolchains.map((t) => ({ tool: t.tool, version: t.version, bin: t.bin, boundAt: t.boundAt, server: t.server ? { id: t.server.id, version: t.server.version } : null })),
      allowed, proposals, declined,
      job: this._jobFor(real),
      guidance: findRepoGuidance(real),
    };
  }

  private _jobFor(real: string): JobView | null {
    let latest: Job | null = null;
    for (const j of this._jobs.values()) {
      if (j.workspace !== real) continue;
      if (!latest || j.startedAt > latest.startedAt) latest = j;
    }
    return latest ? view(latest) : null;
  }

  async decline(sub: string, user: string, workspace: string, tool: ToolId): Promise<void> {
    await this._authorize(user, workspace);
    this.o.store.decline(sub, workspace, tool);
  }

  /** Rewrite (or remove) the repository-guidance pointer from what the workspace holds now. */
  async refreshGuidance(sub: string, user: string, workspace: string): Promise<{ status: EnvironmentStatus; outcome: string }> {
    const real = await this._authorize(user, workspace);
    const paths = findRepoGuidance(real);
    const file = repoGuidanceFile(paths);
    const outcome = file ? writeManagedFile(real, file, atomicWrite, this._log).action : removeManagedFile(real, repoGuidanceIdentity(), this._log).action;
    this._log(`environment: repo guidance for ${workspace}: ${outcome} (${paths.length} file(s))`);
    return { status: this._status(sub, real, workspace), outcome };
  }

  /** Start installing ``tool@version``; refuses a version off the allow-list and a workspace with a job running. */
  async bind(sub: string, user: string, workspace: string, tool: ToolId, version: string): Promise<JobView> {
    const real = await this._authorize(user, workspace);
    const allowed = this.allowedTools().find((a) => a.tool === tool);
    if (!allowed) throw new EnvironmentError(`${tool} is not offered on this server`, 403);
    if (!allowed.versions.includes(version)) throw new EnvironmentError(`${tool} ${version} is not an allowed version (allowed: ${allowed.versions.join(", ")})`, 403);
    const running = this._jobFor(real);
    if (running && running.status === "running") throw new EnvironmentError("an install is already running in this workspace", 409);
    this._sweep();
    this.o.store.undecline(sub, workspace, tool);

    const abort = new AbortController();
    const job: Job = {
      id: randomUUID(), sub, workspace: real, tool, version, status: "running", log: [], notes: [],
      startedAt: new Date().toISOString(), abort, done: Promise.resolve(),
    };
    this._jobs.set(job.id, job);
    job.done = this._runJob(job, allowed.server).catch(() => undefined);
    return view(job);
  }

  private async _runJob(job: Job, server: { id: ServerId; version: string } | null): Promise<void> {
    const push = (line: string) => { job.log.push(line); if (job.log.length > LOG_TAIL) job.log.splice(0, job.log.length - LOG_TAIL); };
    const inst = new Installer({
      workspace: job.workspace, mise: this.o.mise, python: this.o.python, paranoid: this.o.paranoid,
      timeoutMs: this.o.installTimeoutMs, run: this._run, signal: job.abort.signal, log: push,
    });
    try {
      const tc = await inst.installToolchain(job.tool, job.version);
      let installed = null;
      let serverBin: string[] = [];
      if (server) {
        installed = await inst.installServer(server.id, server.version, { typescriptVersion: this.o.typescriptVersion, jdtls: this.o.jdtls });
        if (server.id === "gopls") serverBin = ["gopls"];
      }
      const manifest = readManifest(job.workspace);
      manifest.toolchains = manifest.toolchains.filter((t) => t.tool !== job.tool);
      manifest.toolchains.push({ tool: job.tool, version: job.version, installDir: tc.installDir, bin: tc.bin, server: installed, serverBin, boundAt: new Date().toISOString() });
      job.notes.push(...this._writeFiles(job.workspace, manifest));
      job.status = "done";
      this._log(`environment: bound ${job.tool} ${job.version} in ${job.workspace}${installed ? ` with ${installed.id}` : ""}`);
    } catch (e) {
      if (e instanceof InstallCancelled || job.abort.signal.aborted) { job.status = "cancelled"; }
      else { job.status = "failed"; job.error = (e as Error).message; this._log(`environment: ${job.tool} ${job.version} in ${job.workspace} failed: ${job.error}`); }
    } finally {
      job.finishedAt = new Date().toISOString();
    }
  }

  /** Write the five managed files from ``manifest``; returns a note for each one not written as asked. */
  private _writeFiles(real: string, manifest: Manifest): string[] {
    const notes: string[] = [];
    const put = (file: ManagedFile | null, identity: ManagedFile) => {
      const outcome = file ? writeManagedFile(real, file, atomicWrite, this._log) : removeManagedFile(real, identity, this._log);
      if (outcome.action === "skipped-user-file") notes.push(`kept your own ${identity.relativePath} (its jaato-managed marker was removed), so it does not reflect this change`);
      if (outcome.action === "error") notes.push(`could not write ${identity.relativePath}: ${(outcome as Extract<ManagedWriteOutcome, { action: "error" }>).message}`);
    };
    // The manifest last: it is the record, so it never names a toolchain whose other files did not get the chance.
    put(miseConfigFile(manifest), miseConfigFile(manifest));
    put(lspConfigFile(manifest), lspConfigIdentity());
    put(environmentInstructionsFile(manifest), environmentInstructionsIdentity());
    put(apparmorFragmentFile(manifest, real, notes), apparmorFragmentIdentity());
    if (manifest.toolchains.length) put(manifestFile(manifest), manifestFile(manifest));
    else put(null, manifestFile(manifest));
    this._writeOffer(real, manifest);
    return notes;
  }

  /** Rewrite (or remove) ``.jaato/toolchain-offer.json``; a failure is logged, never raised: the offer is a convenience. */
  private _writeOffer(real: string, manifest: Manifest): void {
    const file = toolchainOfferFile(this.allowedTools(), manifest);
    const outcome = file ? writeManagedFile(real, file, atomicWrite, this._log) : removeManagedFile(real, toolchainOfferIdentity(), this._log);
    if (outcome.action === "error") this._log(`environment: could not write the toolchain offer in ${real}: ${(outcome as Extract<ManagedWriteOutcome, { action: "error" }>).message}`);
  }

  /** Remove a binding: its links, its server's binaries, and its entry in the managed files.  The installed data is kept. */
  async unbind(sub: string, user: string, workspace: string, tool: ToolId): Promise<{ status: EnvironmentStatus; notes: string[]; removed: string[] }> {
    const real = await this._authorize(user, workspace);
    const running = this._jobFor(real);
    if (running && running.status === "running") throw new EnvironmentError("an install is running in this workspace; cancel it first", 409);
    const manifest = readManifest(real);
    const entry = manifest.toolchains.find((t) => t.tool === tool);
    if (!entry) throw new EnvironmentError(`${tool} is not bound to this workspace`, 404);
    const bin = join(real, LOCAL_BIN);
    const removed = unlinkBinaries(bin, entry.bin, join(real, MISE_DATA_DIR));
    for (const name of entry.serverBin) {
      try { rmSync(join(bin, name), { force: true }); removed.push(name); } catch { /* already gone */ }
    }
    manifest.toolchains = manifest.toolchains.filter((t) => t.tool !== tool);
    const notes = this._writeFiles(real, manifest);
    this._log(`environment: unbound ${tool} from ${workspace} (removed ${removed.length} link(s))`);
    return { status: this._status(sub, real, workspace), notes, removed };
  }

  /** A job the signed-in user started; ``null`` for anyone else's (the same answer an unknown id gets). */
  job(sub: string, id: string): JobView | null {
    const j = this._jobs.get(id);
    return j && j.sub === sub ? view(j) : null;
  }

  cancel(sub: string, id: string): JobView | null {
    const j = this._jobs.get(id);
    if (!j || j.sub !== sub) return null;
    if (j.status === "running") j.abort.abort();
    return view(j);
  }

  /** Test seam: wait for a job to finish. */
  async waitFor(id: string): Promise<JobView | null> {
    const j = this._jobs.get(id);
    if (!j) return null;
    await j.done;
    return view(j);
  }

  private _sweep(): void {
    const now = Date.now();
    for (const [id, j] of this._jobs) {
      if (j.finishedAt && now - Date.parse(j.finishedAt) > JOB_RETENTION_MS) this._jobs.delete(id);
    }
  }
}

function view(j: Job): JobView {
  return {
    id: j.id, workspace: j.workspace, tool: j.tool, version: j.version, status: j.status,
    log: [...j.log], error: j.error, notes: [...j.notes], startedAt: j.startedAt, finishedAt: j.finishedAt,
  };
}

function realpathSafe(p: string): string | null {
  try { return realpathSync(p); } catch { return null; }
}

function atomicWrite(path: string, body: string, mode: number): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = join(dirname(path), `.${randomBytes(6).toString("hex")}.tmp`);
  writeFileSync(tmp, body, { mode });
  renameSync(tmp, path);
}
