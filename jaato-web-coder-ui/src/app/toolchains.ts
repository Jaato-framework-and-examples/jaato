/**
 * A workspace's toolchains, as the ``web_coder_toolchains`` plugin keeps
 * them (#1344).
 *
 * The plugin runs in the session's runner and is the only thing that writes
 * a workspace's toolchains: it installs into ``<ws>/.home`` and records
 * what is bound, the proposals from the repositories' markers, the
 * repositories' guidance files and the current or last job in
 * ``.jaato/environment.json``.  The page:
 *
 * - READS that file through the daemon (``workspace.file.fetch``, protocol
 *   1.20), quiet and correlated, polling while a job runs;
 * - sends ``toolchain bind|unbind|scan|cancel`` (a user command; its one-line
 *   answer appears in the transcript as the daemon's system message).  A
 *   command needs a session: the plugin runs in its runner.
 *
 * **Pending binds.**  A toolchain chosen where there is no session yet (the
 * New workspace plate, right after the clone) is remembered per workspace
 * in this browser and bound when a session in that workspace starts, one
 * job after another (the plugin runs one at a time).  ``app/toolchainOffer.ts``
 * calls :func:`flushPending` once it has staged the offer, which the plugin
 * needs before it accepts a bind.
 */
import { useJaato } from "@/store/store";
import { getClient, isConnected } from "@/sdk/connection";
import { MIN_FILE_FETCH_PROTOCOL, isProtocolCompatible } from "@jaato/sdk";
import type { ToolId } from "./environment";

export const MANIFEST_PATH = ".jaato/environment.json";
export const POLL_MS = 1500;
/** How long a pending bind waits for its job to appear before it is given up as refused. */
export const START_GRACE_MS = 15_000;
/** The longest one pending bind is followed (an install's own timeout is the operator's, in the offer). */
export const MAX_JOB_MS = 2 * 60 * 60 * 1000;

export interface BoundToolchain {
  tool: ToolId;
  version: string;
  bin: string[];
  boundAt: string | null;
  server: { id: string; version: string } | null;
}

export interface Proposal {
  tool: ToolId;
  label: string;
  version: string;
  pin: string | null;
  pinAllowed: boolean;
  source: string;
}

export interface ToolchainJob {
  id: string;
  action: "bind" | "unbind";
  tool: ToolId;
  version: string | null;
  status: "running" | "done" | "failed" | "cancelled";
  log: string[];
  error: string | null;
  notes: string[];
  startedAt: string;
  finishedAt: string | null;
}

export interface Manifest {
  toolchains: BoundToolchain[];
  proposals: Proposal[];
  guidance: string[];
  job: ToolchainJob | null;
}

export const EMPTY_MANIFEST: Manifest = { toolchains: [], proposals: [], guidance: [], job: null };

/** Whether this connection can read the manifest: a daemon that serves ``workspace.file.fetch``. */
export function canReadManifest(): boolean {
  if (!isConnected()) return false;
  const v = getClient().serverProtocolVersion;
  return !!v && isProtocolCompatible(v, MIN_FILE_FETCH_PROTOCOL);
}

/** Parse the manifest the plugin wrote; anything malformed is empty (the file is model-writable). */
export function parseManifest(text: string): Manifest {
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { return EMPTY_MANIFEST; }
  if (!raw || typeof raw !== "object") return EMPTY_MANIFEST;
  const r = raw as Record<string, unknown>;
  const list = <T>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);
  return {
    toolchains: list<BoundToolchain>(r.toolchains).filter((t) => t && typeof t.tool === "string" && typeof t.version === "string"),
    proposals: list<Proposal>(r.proposals).filter((p) => p && typeof p.tool === "string" && typeof p.version === "string"),
    guidance: list<string>(r.guidance).filter((g) => typeof g === "string"),
    job: r.job && typeof r.job === "object" ? (r.job as ToolchainJob) : null,
  };
}

/** The manifest of this connection's workspace; ``null`` when it cannot be read, ``EMPTY_MANIFEST`` when absent. */
export async function readManifest(): Promise<Manifest | null> {
  if (!canReadManifest()) return null;
  const { event, data } = await getClient().fetchWorkspaceFile(MANIFEST_PATH);
  if (!event.ok) return event.category === "not_found" ? EMPTY_MANIFEST : null;
  return parseManifest(new TextDecoder().decode(data ?? new Uint8Array()));
}

/** Send ``toolchain <args…>`` to the session's plugin. */
export async function toolchainCommand(...args: string[]): Promise<void> {
  await getClient().executeCommand("toolchain", args);
}

// ── pending binds ───────────────────────────────────────────────────────

const pendingKey = (ws: string) => `jaato.toolchains.pending:${ws}`;

export interface PendingBind { tool: ToolId; version: string }

export function pendingBinds(ws: string): PendingBind[] {
  try {
    const raw = JSON.parse(localStorage.getItem(pendingKey(ws)) ?? "[]") as unknown;
    return Array.isArray(raw) ? raw.filter((p): p is PendingBind => !!p && typeof p.tool === "string" && typeof p.version === "string") : [];
  } catch { return []; }
}

function savePending(ws: string, list: PendingBind[]): void {
  try {
    if (list.length) localStorage.setItem(pendingKey(ws), JSON.stringify(list));
    else localStorage.removeItem(pendingKey(ws));
  } catch { /* storage unavailable: the choice lives for this page only */ }
}

export function addPending(ws: string, tool: ToolId, version: string): void {
  savePending(ws, [...pendingBinds(ws).filter((p) => p.tool !== tool), { tool, version }]);
}

export function removePending(ws: string, tool: ToolId): void {
  savePending(ws, pendingBinds(ws).filter((p) => p.tool !== tool));
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let flushing = "";

/**
 * Bind this workspace's pending choices in the session now open there, one
 * job at a time.  A no-op without a session, or while another flush of the
 * same workspace runs.  A choice leaves the list once its job has ENDED
 * (done or failed): the job's own record then says how it went.
 */
export async function flushPending(ws: string, pollMs = POLL_MS): Promise<void> {
  if (!ws || flushing === ws) return;
  flushing = ws;
  try {
    for (;;) {
      const st = useJaato.getState();
      const next = pendingBinds(ws)[0];
      if (!next || !st.sessionId || !isConnected()) return;
      const before = await readManifest();
      if (before === null) return;
      if (before.job?.status === "running") { await sleep(pollMs); continue; }
      await toolchainCommand("bind", next.tool, next.version);
      // Wait for THIS job (its record replaces the previous one).  A bind the
      // plugin refused (a version no longer allowed, no mise) starts no job:
      // give up on it after START_GRACE_MS rather than wait forever.
      let started = false;
      for (let waited = 0; waited < MAX_JOB_MS; waited += pollMs) {
        await sleep(pollMs);
        const m = await readManifest();
        if (m === null) return;
        if (m.toolchains.some((t) => t.tool === next.tool && t.version === next.version)) break;
        const j = m.job;
        if (j && j.id !== before.job?.id && j.tool === next.tool) {
          started = true;
          if (j.status !== "running") break;
        } else if (!started && waited >= START_GRACE_MS) break;
      }
      removePending(ws, next.tool);
    }
  } finally {
    if (flushing === ws) flushing = "";
  }
}
