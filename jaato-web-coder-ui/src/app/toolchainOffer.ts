/**
 * ``.jaato/toolchain-offer.json`` reaches the workspace from the PAGE.
 *
 * The ``web_coder_toolchains`` plugin (runner-side) reads that file as the
 * operator's policy: what it may bind, and which toolchain provides a
 * missing command.  The backend computes it but does not write it: it may run as an account that
 * cannot write the workspace (a web coder backend beside a root daemon).
 * The daemon can, so the page stages the backend's content through
 * ``StageFilesRequest``, the verb attachments use (``app/staging.ts``).
 *
 * When:
 *   • every environment status the backend returns (the Toolchains panel's
 *     reads, through ``environmentApi``'s ``onStatus``);
 *   • the session's workspace becoming known (attach, create, a workspace
 *     selected), so the file is there before anyone opens the rail.  In a
 *     session, the page then asks the plugin to scan when the workspace has
 *     no manifest yet (its first session: the plugin scans at start only
 *     when the offer was already there), and binds what was chosen before
 *     any session existed (``flushPending``).
 *
 * Only into the workspace this connection is in: staging writes there, so a
 * status for another workspace (the new-workspace plate's) is not staged.
 * The same content is staged once per workspace per page load.
 */
import { useJaato } from "@/store/store";
import { isConnected } from "@/sdk/connection";
import { environmentApi, type EnvironmentStatus } from "@/app/environment";
import { stageGenerated } from "@/app/staging";
import { EMPTY_MANIFEST, flushPending, pendingBinds, readManifest, toolchainCommand, waitingProposals } from "@/app/toolchains";

/** The session's workspace path, as the Toolchains rail resolves it; ``""`` when unknown. */
export function currentWorkspacePath(s: ReturnType<typeof useJaato.getState>): string {
  return s.workspace.list.find((w) => w.name === s.workspace.selected)?.path
    ?? s.sessions.find((x) => x.isCurrent)?.workspacePath ?? "";
}

const lastStaged = new Map<string, string>();

/** Stage ``st.offer`` into this connection's workspace when it is that workspace's and has changed. */
export async function stageToolchainOffer(st: Pick<EnvironmentStatus, "workspace" | "offer">): Promise<boolean> {
  if (!st.offer || !isConnected()) return false;
  const here = currentWorkspacePath(useJaato.getState());
  if (!here || here !== st.workspace) return false;
  if (lastStaged.get(st.workspace) === st.offer.content) return false;
  const ok = await stageGenerated(st.offer.path, st.offer.content);
  if (ok) lastStaged.set(st.workspace, st.offer.content);
  else console.warn(`toolchain offer: the daemon did not stage ${st.offer.path} into ${st.workspace}`);
  return ok;
}

/** Test seam: forget what was staged. */
export function resetStagedOffers(): void {
  lastStaged.clear();
}

let asked = "";
useJaato.subscribe((s) => {
  const url = s.environmentUrl;
  const path = currentWorkspacePath(s);
  // The session is part of the key: a new session in the workspace is when a scan and the pending binds run.
  const key = url && path ? `${url}\n${path}\n${s.sessionId ?? ""}` : "";
  if (!key || key === asked || !isConnected()) return;
  asked = key;
  void (async () => {
    try {
      const st = await environmentApi(url!, fetch).status(path);
      await stageToolchainOffer(st);
      if (!useJaato.getState().sessionId || currentWorkspacePath(useJaato.getState()) !== path) return;
      // The plugin scans at session start only when the offer was already there; the first session in a workspace needs asking.
      const m = await readManifest();
      if (m !== null && m === EMPTY_MANIFEST) await toolchainCommand("scan");
      await flushPending(path);
      // The rail badge counts the proposals with the section closed.
      const after = await readManifest();
      if (after && currentWorkspacePath(useJaato.getState()) === path) {
        useJaato.getState().setEnvironmentWaiting(waitingProposals(after, st, pendingBinds(path)).length);
      }
    } catch {
      if (asked === key) asked = "";
    }
  })();
});
