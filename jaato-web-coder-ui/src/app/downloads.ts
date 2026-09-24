/**
 * Taking files OUT of the workspace (protocol 1.20).
 *
 * Two ways in, one way out:
 *
 * - **the Files panel** -- a file's name is a button; clicking it downloads
 *   the file.  A failure is the panel's notice.
 * - **``offer_download``**, a host tool this client registers with the
 *   session, so the model can put a download button in the chat -- when the
 *   user asks for a file, or when it produced one they will want.  The tool
 *   only CHECKS the file (a metadata-only fetch) and answers the model with
 *   what it offered; the bytes move when the person clicks, never on the
 *   model's say-so, and a file that is not there is reported to the model
 *   rather than drawn as a button that fails.
 *
 * Both end in {@link downloadWorkspaceFile}, which asks the daemon for the
 * bytes (``JaatoClient.fetchWorkspaceFile``) and hands them to the browser
 * as a download.  What may leave the workspace is the DAEMON's decision
 * (``server/workspace_download.py``): nothing outside it, never ``.env`` or
 * a stored ``*_auth.json``, nothing over its size cap.  This module only
 * renders the answer.
 */
import { EventTypeValue, isProtocolCompatible, MIN_FILE_FETCH_PROTOCOL, type JaatoClient, type JaatoEvent } from "@jaato/sdk";
import { getClient } from "@/sdk/connection";
import { useJaato } from "@/store/store";

/** The host tool's model-facing name. */
export const OFFER_DOWNLOAD_TOOL = "offer_download";

/** Why a download was refused, in the words the panel and the chip show. */
const CATEGORY_TEXT: Record<string, string> = {
  not_found: "no such file",
  not_a_file: "not a file",
  unsafe_path: "outside the workspace",
  credential: "holds credentials",
  too_large: "too large to download",
  workspace_not_found: "no workspace selected",
  io_error: "could not be read",
};

/** A refusal's category as a reader-facing phrase (the daemon's message if the category is new). */
export function refusalText(category: string | undefined, error: string | undefined): string {
  return (category && CATEGORY_TEXT[category]) || error || "download refused";
}

/** True when a daemon speaking ``protocolVersion`` serves ``workspace.file.fetch``. */
export function servesDownloads(protocolVersion: string | null | undefined): boolean {
  return !!protocolVersion && isProtocolCompatible(protocolVersion, MIN_FILE_FETCH_PROTOCOL);
}

/** Hand bytes to the browser as a file download. */
export function saveBytes(name: string, mimeType: string, data: Uint8Array): void {
  const blob = new Blob([data as BlobPart], { type: mimeType || "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoked on the next tick, after the click has started the download.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Download ``path`` from the workspace.  Resolves with ``null`` on success
 * and with a reader-facing reason on refusal or failure -- never throws, so
 * both callers render the outcome the same way.
 */
export async function downloadWorkspaceFile(path: string): Promise<string | null> {
  try {
    const { event, data } = await getClient().fetchWorkspaceFile(path);
    if (!event.ok || !data) return refusalText(event.category ?? undefined, event.error ?? undefined);
    saveBytes(event.name || path.split("/").pop() || "download", event.mime_type ?? "", data);
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
}

/** The Files panel's click: download, and say so in the panel on failure. */
export async function downloadFromPanel(path: string): Promise<void> {
  const st = useJaato.getState();
  st.setWorkspaceNotice({ text: `Downloading ${path}…` });
  const failure = await downloadWorkspaceFile(path);
  useJaato.getState().setWorkspaceNotice(failure ? { text: `${path}: ${failure}`, error: true } : null);
}

/** The host tool as the daemon registers it (``ToolsRegisterClientRequest``). */
export const OFFER_DOWNLOAD_SPEC = {
  name: OFFER_DOWNLOAD_TOOL,
  description:
    "Offer the user a file from the workspace as a download button in the chat. " +
    "Use it when the user asks to get, download or export a file, or when you produced a file " +
    "they will want to keep (a report, an export, an image, an archive). " +
    "The user clicks the button to download; you do not send the file's content. " +
    "The path is relative to the workspace root. The answer says what was offered, or why " +
    "it could not be (no such file, outside the workspace, holds credentials, too large).",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "The file's path, relative to the workspace root." },
      label: { type: "string", description: "Optional short caption for the button." },
    },
    required: ["path"],
  },
  timeout: 30000,
  auto_approve: true,
};

type ExecuteRequest = { call_id?: string; tool_name?: string; tool_args?: Record<string, unknown> };

/**
 * Answer one ``offer_download`` call: check the file exists and may leave,
 * then report what was offered.  Throws the refusal, which the caller
 * returns to the model as the tool's error.
 */
export async function answerOfferDownload(args: Record<string, unknown>, client: Pick<JaatoClient, "fetchWorkspaceFile">): Promise<Record<string, unknown>> {
  const path = typeof args.path === "string" ? args.path.trim() : "";
  if (!path) throw new Error("offer_download needs a path");
  const { event } = await client.fetchWorkspaceFile(path, { metadataOnly: true });
  if (!event.ok) throw new Error(`${path}: ${refusalText(event.category ?? undefined, event.error ?? undefined)}`);
  return {
    offered: true,
    path: event.path,
    name: event.name,
    size: event.size,
    mime_type: event.mime_type,
    note: "A download button is shown to the user in the chat.",
  };
}

/**
 * Wire the host tool onto a connected client: answer execution requests,
 * and (re-)register the tool whenever this connection is attached to a
 * session -- registration is per SESSION on the daemon, and a reconnect is
 * a new client there, so both a new session and a re-attach need it.
 * Nothing is registered against a daemon that cannot serve the fetch.
 */
export function wireDownloadTool(client: JaatoClient): void {
  let registeredFor: string | null = null;
  client.onStatus((status) => {
    if (String((status as unknown as { state: unknown }).state).toUpperCase() !== "CONNECTED") registeredFor = null;
  });
  client.subscribeAll((raw: JaatoEvent) => {
    const ev = raw as unknown as { type?: string; session_id?: string } & ExecuteRequest;
    if (ev.type === EventTypeValue.SESSION_INFO && ev.session_id && ev.session_id !== registeredFor && servesDownloads(client.serverProtocolVersion)) {
      registeredFor = ev.session_id;
      void client.registerClientTools([OFFER_DOWNLOAD_SPEC]).catch(() => { registeredFor = null; });
      return;
    }
    if (ev.type !== EventTypeValue.TOOL_EXECUTE_REQUEST || ev.tool_name !== OFFER_DOWNLOAD_TOOL || !ev.call_id) return;
    const callId = ev.call_id;
    answerOfferDownload(ev.tool_args ?? {}, client).then(
      (result) => client.respondToToolExecution(callId, JSON.stringify(result), ""),
      (err: unknown) => client.respondToToolExecution(callId, "", err instanceof Error ? err.message : String(err)),
    ).catch(() => undefined);
  });
}
