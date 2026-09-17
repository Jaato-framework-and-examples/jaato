/**
 * Attaching a file from the browser stages it into the session's
 * workspace on the daemon — ``StageFilesRequest`` (one TEXT frame naming
 * the files, one BINARY frame per file, one ``StageFilesEvent`` back),
 * the verb the premium ``<jaato-task>`` component uses and the canonical
 * primitive ``docs/sdk-file-staging.md`` describes.  The SDK's
 * ``stageFiles`` does the framing; this module decides WHEN and reports
 * WHAT happened.
 *
 * Two moments, one verb:
 *
 *  • mid-session (the composer): staged the moment they are dropped,
 *    into the workspace the session runs in;
 *  • at bootstrap (the session picker): a workspace that is already
 *    selected takes them BEFORE ``session.new``, so the session starts
 *    with them on disk; a daemon that provisions the workspace AS PART of
 *    ``session.new`` has nowhere to put them until its ``session.info``
 *    arrives, so they wait for it, still ahead of the first turn.
 *
 * The SDK correlates the daemon's answer to a ``stageFiles`` call by
 * order, so calls are serialised through one promise chain — two drops in
 * quick succession are two requests, never interleaved.  ``File`` objects
 * are kept here rather than in the store: the store holds what the strip
 * renders, this holds what the wire needs.
 */
import { EventTypeValue } from "@jaato/sdk";
import { MAIN_AGENT, useJaato } from "@/store/store";
import type { StagedUpload } from "@/store/types";
import { checkSizes, stagedName } from "@/protocol/attachments";
import { getClient, isConnected } from "@/sdk/connection";

const bytesOf = new Map<string, File>();
let chain: Promise<void> = Promise.resolve();
let seq = 0;

/** Is there a workspace on the daemon side to stage into right now? */
export function canStageNow(): boolean {
  return isConnected() && hasWorkspace(useJaato.getState());
}

function hasWorkspace(st: { sessionId?: string | null; workspace: { selected?: string } }): boolean {
  return Boolean(st.sessionId || st.workspace.selected);
}

// A file attached before the daemon has said which workspace this client
// is in -- the picker is on screen the moment ``workspace.select`` is SENT,
// and ``selected`` is written when its ``config.status`` reply is reduced,
// a frame or a round-trip later -- is queued, and stages the moment that
// answer (or a session's ``session.info``) lands.  Sampling once at attach
// time left such a file queued until a profile was picked.
useJaato.subscribe((st, prev) => {
  if (hasWorkspace(st) && !hasWorkspace(prev) && isConnected() && st.uploads.some((u) => u.status === "queued")) {
    stageQueued().catch(() => undefined);
  }
});

/**
 * Attach ``files`` (a pick, a drop, a paste) under ``folder``.  Names the
 * daemon would refuse and sizes over its caps are failed here, without a
 * request.  Everything else is staged now when a workspace exists, or
 * queued for ``stageQueued`` — the picker's case.
 */
export function attachFiles(files: File[], folder: string): void {
  if (!files.length) return;
  const st = useJaato.getState();
  const verdicts = checkSizes(files.map((f) => f.size));
  const items: StagedUpload[] = files.map((f, i) => {
    const id = `up-${++seq}`;
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath ?? "";
    const path = stagedName(f.name, rel, folder);
    const reason = path === null ? "name must be a non-empty workspace-relative path with no '..' components" : verdicts[i]!.reason;
    if (reason) return { id, path: path ?? f.name, size: f.size, status: "failed", error: reason };
    bytesOf.set(id, f);
    return { id, path: path!, size: f.size, status: "queued" };
  });
  st.addUploads(items);
  if (canStageNow()) stageQueued().catch(() => undefined);
}

/** Drop a chip; a file not yet sent is forgotten, one already staged stays in the workspace. */
export function discardUpload(id: string): void {
  bytesOf.delete(id);
  useJaato.getState().removeUpload(id);
}

/** Stage every ``queued`` entry, one request for the batch.  Resolves once the daemon has answered. */
export function stageQueued(): Promise<void> {
  chain = chain.then(() => stageBatch()).catch(() => undefined);
  return chain;
}

async function stageBatch(): Promise<void> {
  const st = useJaato.getState();
  const queued = st.uploads.filter((u) => u.status === "queued" && bytesOf.has(u.id));
  if (!queued.length) return;
  for (const u of queued) st.updateUpload(u.id, { status: "staging" });
  const agentId = st.selectedAgentId || MAIN_AGENT;
  // Read each file on its own: a directory dropped alongside real files
  // yields a File that cannot be read, and it must fail alone.
  const payloads: Array<{ name: string; data: ArrayBuffer; contentType?: string }> = [];
  const sending: StagedUpload[] = [];
  for (const u of queued) {
    const f = bytesOf.get(u.id)!;
    try {
      payloads.push({ name: u.path, data: await f.arrayBuffer(), contentType: f.type || undefined });
      sending.push(u);
    } catch (err) {
      bytesOf.delete(u.id);
      st.updateUpload(u.id, { status: "failed", error: `could not read the file: ${err instanceof Error ? err.message : String(err)}` });
    }
  }
  if (!sending.length) return;
  try {
    const result = await getClient().stageFiles("", payloads);
    const staged = new Set(result.staged ?? []);
    const failed = new Map((result.failed ?? []).map((f) => [f.name, f.error || f.category || "failed"]));
    const ok: string[] = [];
    const bad: string[] = [];
    for (const u of sending) {
      bytesOf.delete(u.id);
      if (staged.has(u.path)) { st.updateUpload(u.id, { status: "staged" }); ok.push(u.path); }
      else {
        const error = failed.get(u.path) ?? "the daemon did not report this file";
        st.updateUpload(u.id, { status: "failed", error });
        bad.push(`${u.path}: ${error}`);
      }
    }
    if (ok.length) st.addSystemBlock(agentId, `Staged into the workspace: ${ok.join(", ")}`, "info");
    if (bad.length) st.addSystemBlock(agentId, `Not staged:\n  ${bad.join("\n  ")}`, "error");
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    for (const u of sending) { bytesOf.delete(u.id); st.updateUpload(u.id, { status: "failed", error }); }
    st.addSystemBlock(agentId, `Staging failed: ${error}`, "error");
  }
}

/** Resolve on the next ``session.info``, or after ``timeoutMs`` — a session that did not open stages nothing. */
export function awaitSessionInfo(timeoutMs = 60_000): Promise<boolean> {
  return new Promise((resolve) => {
    let unsub = () => undefined as void;
    const timer = setTimeout(() => { unsub(); resolve(false); }, timeoutMs);
    unsub = getClient().subscribe(EventTypeValue.SESSION_INFO, () => { clearTimeout(timer); unsub(); resolve(true); });
  });
}

/**
 * Open a session with the picker's queued files in its workspace:
 * ``open`` is ``createSession`` or ``attachSession``.  See the module
 * docstring for why the order depends on who provisions the workspace.
 */
export async function openSessionWithQueued(open: () => Promise<void>): Promise<void> {
  const st = useJaato.getState();
  const hasQueued = st.uploads.some((u) => u.status === "queued");
  if (!hasQueued) return open();
  if (st.workspace.selected) {
    await stageQueued();
    return open();
  }
  const opened = awaitSessionInfo();
  await open();
  if (await opened) await stageQueued();
}
