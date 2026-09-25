/**
 * Adapter between ``@jaato/sdk``'s ``JaatoClient`` and the store.
 *
 * Owns exactly four things:
 *   1. the client's lifecycle (one live client per page; ``connect`` /
 *      ``disconnect``), mirrored into ``state.connection``;
 *   2. event delivery — every wire event is queued and flushed into the
 *      store once per animation frame, so a token stream costs one
 *      commit per frame (the SDK already preserves order; we preserve
 *      it too by flushing the queue in FIFO);
 *   3. the request helpers the UI needs that the SDK exposes only as raw
 *      verbs (workspace list/select/create, config update);
 *   4. wiring the ``offer_download`` host tool onto each client
 *      (``app/downloads.ts`` owns what it does), and the Memories rail's
 *      refresh triggers (``app/memories.ts``).
 *
 * The client identifies itself as ``client_type: "web"`` with
 * ``supports_expandable_content: true`` — the model is told its output
 * will be shown in a browser that can collapse overflow, so it need not
 * shorten for a narrow terminal.
 */
import { EventTypeValue, JaatoClient, type ConnectionStatus, type JaatoEvent, type TokenProvider } from "@jaato/sdk";
import { useJaato } from "@/store/store";
import type { WorkspaceInfo } from "@/store/types";
import { wireDownloadTool } from "@/app/downloads";
import { wireMemoryRail } from "@/app/memories";

export interface ConnectOptions {
  url: string;
  /**
   * The daemon's shared token, or a provider the SDK calls before every
   * connection attempt.  The provider form is how a single-use per-user
   * ticket (#1074) survives reconnects — see ``app/tickets.ts``.
   */
  token?: string | TokenProvider;
}

let client: JaatoClient | null = null;
let queue: JaatoEvent[] = [];
let flushScheduled = false;

function flush(): void {
  flushScheduled = false;
  if (!queue.length) return;
  const batch = queue;
  queue = [];
  useJaato.getState().dispatch(batch);
}

function enqueue(ev: JaatoEvent): void {
  queue.push(ev);
  if (flushScheduled) return;
  flushScheduled = true;
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(flush);
  else setTimeout(flush, 16);
}

/** Force pending events into the store now (tests, and before reading state after a request). */
export function flushNow(): void {
  flush();
}

// Set while the SDK is retrying, so the transition INTO connected can be
// told from a first connect (which has nothing to re-assert).
let sawReconnecting = false;

function onStatus(status: ConnectionStatus): void {
  const phaseMap: Record<string, "disconnected" | "connecting" | "connected" | "reconnecting" | "closed"> = {
    DISCONNECTED: "disconnected",
    CONNECTING: "connecting",
    CONNECTED: "connected",
    RECONNECTING: "reconnecting",
    CLOSED: "closed",
  };
  const raw = String((status as unknown as { state: unknown }).state);
  const phase = phaseMap[raw] ?? phaseMap[raw.toUpperCase()] ?? "disconnected";
  useJaato.getState().setConnection({
    phase,
    detail: (status as unknown as { reason?: string }).reason,
    attempt: (status as unknown as { attempt?: number }).attempt,
    serverVersion: (status as unknown as { serverVersion?: string | null }).serverVersion ?? undefined,
  });
  if (phase === "reconnecting") sawReconnecting = true;
  else if (phase === "connected" && sawReconnecting) {
    sawReconnecting = false;
    void reassertAfterReconnect();
  }
}

export function presentationContext(): Record<string, unknown> {
  const width = typeof window !== "undefined" ? Math.max(40, Math.floor((window.innerWidth * 0.7) / 8)) : 120;
  return {
    client_type: "web",
    content_width: width,
    content_height: null,
    supports_markdown: true,
    supports_tables: true,
    supports_code_blocks: true,
    supports_images: true,
    supports_rich_text: true,
    supports_unicode: true,
    supports_mermaid: false,
    supports_expandable_content: true,
    renderable_media: ["image/*", "audio/*"],
  };
}

export async function connect(opts: ConnectOptions): Promise<JaatoClient> {
  await disconnect();
  const st = useJaato.getState();
  st.setUrl(opts.url);
  st.setConnection({ phase: "connecting", detail: undefined });
  const c = new JaatoClient({
    url: opts.url,
    token: opts.token,
    clientConfig: { presentation: presentationContext() },
  });
  sawReconnecting = false;
  c.onStatus(onStatus);
  c.subscribeAll((ev) => enqueue(ev));
  // The ``offer_download`` host tool (protocol 1.20): registered per session.
  wireDownloadTool(c);
  // The Memories rail's refresh triggers (#1232): session.info, a memory
  // tool's success, the `memory` command's push.
  wireMemoryRail(c);
  client = c;
  try {
    await c.connect();
  } catch (err) {
    client = null;
    useJaato.getState().setConnection({ phase: "disconnected", detail: err instanceof Error ? err.message : String(err) });
    throw err;
  }
  return c;
}

/**
 * Leave the daemon.
 *
 * The session state goes with the connection, because that is what it
 * describes: ``sessionId`` means "this client is attached to that
 * session", and a client that has closed its socket is attached to
 * nothing — the daemon detaches it on the way out.  Keeping it left the
 * screen rendering a dead session's transcript with a live composer, and
 * ``SessionScreen`` shows its picker only when no session is held, so
 * after a Detach and a reconnect there was no way back to the picker at
 * all: the next connection opened straight into the transcript of the
 * session it had just left.
 *
 * Queued attachments survive (``uploads`` lives outside
 * ``emptySessionState``), so files picked before leaving still stage into
 * whatever session is opened next.
 */
export async function disconnect(): Promise<void> {
  const c = client;
  client = null;
  useJaato.getState().resetSessionState();
  if (c) await c.close().catch(() => undefined);
}

export function getClient(): JaatoClient {
  if (!client) throw new Error("Not connected");
  return client;
}

export function isConnected(): boolean {
  return client !== null;
}

/**
 * Re-assert what this client believes, after a reconnect.
 *
 * A reconnect is a NEW client on the daemon, and none of the daemon's
 * per-connection state survives it: the disconnect path calls
 * ``remove_client`` on the workspace manager and on the event-sink
 * adapter (dropping this connection's ``workspace.select``) and detaches
 * the client from its session.  The store keeps both, so the screen goes
 * on naming a workspace and a session the live connection does not have,
 * and the next verb keyed on either is answered against nothing:
 * ``stageFiles`` refuses with ``workspace_not_found`` naming a client id
 * nobody recognises, and a ``session.new`` that resolves no workspace
 * resolves no ``.env`` either, so its envelope carries no model and the
 * runner bootstrap fails on an empty model name.  Both were measured on a
 * tablet, where a backgrounded tab drops the socket routinely.
 *
 * Order matters: the workspace FIRST, because ``session.attach`` compares
 * the session's workspace against the client's and opens a mismatch
 * prompt when they differ — and a client that has just reconnected has
 * none.
 *
 * The session is re-attached with the SDK verb alone, NOT the app's
 * ``attachSession``: that one resets the pane and re-requests history,
 * which is right when switching sessions and wrong here, where the
 * transcript on screen is already this session's and the daemon replays
 * nothing on a bare attach.  The SDK's own opt-in re-attach
 * (``autoReattachSessionId``) is left off for the other half of the same
 * reason — it sends the attach without the workspace that has to precede
 * it.
 */
export async function reassertAfterReconnect(): Promise<void> {
  const st = useJaato.getState();
  const workspace = st.workspace.selected;
  const sessionId = st.sessionId;
  if (!workspace && !sessionId) return;
  try {
    if (workspace) await selectWorkspace(workspace);
    if (sessionId) await getClient().attachSession(sessionId);
  } catch (err) {
    const state = useJaato.getState();
    state.addSystemBlock(
      state.selectedAgentId,
      `Reconnected, but this connection could not be restored: ${err instanceof Error ? err.message : String(err)}`,
      "error",
    );
  }
}

// ── Workspace-first flow (WS daemons started with --workspace-root) ──

export async function requestWorkspaceList(): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_LIST_REQUEST });
}

export async function selectWorkspace(name: string): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_SELECT_REQUEST, name });
}

/**
 * A ``workspace.created`` reply's row, mirroring the store's own
 * ``WORKSPACE_CREATED`` extraction (``store.ts``'s dispatcher) exactly so
 * the two cannot disagree about what the daemon sent: the row lives under
 * ``workspace``, with ``name`` / ``path`` repeated beside it as a fallback
 * for an older daemon.  Exported so the shape can be pinned in a test
 * without a live client.
 */
export function parseWorkspaceCreatedEvent(ev: unknown): WorkspaceInfo | null {
  const body = ev as { workspace?: Partial<WorkspaceInfo>; name?: string; path?: string };
  const raw = body.workspace ?? {};
  const name = String(raw.name ?? body.name ?? "");
  if (!name) return null; // a reply naming nothing is not a row (see store.ts)
  const path = (raw.path as string | null | undefined) ?? body.path ?? null;
  return { name, configured: raw.configured === true, path };
}

/**
 * ``workspace.create``, awaiting the daemon's ``workspace.created`` reply so
 * the caller has the new workspace's row -- name AND absolute ``path`` --
 * before doing anything that needs to identify it precisely.  Binding a
 * GitHub account is exactly that: the daemon resolves ``app://`` references
 * against the ABSOLUTE path (``SecretResolveContext.workspace_path``),
 * never the bare name this form takes, so a caller that only had the name
 * to hand would record a binding ``secret.resolve`` can never find (see
 * ``autoBindDefaultGitHubAccount`` in ``app/github.ts``).
 *
 * Resolves with the created ``WorkspaceInfo``, or ``null`` on a timeout or
 * an ``ErrorEvent`` -- never rejects, so a caller that only wants
 * fire-and-forget can await it and ignore the result exactly as before.
 * Filtered by name because a reply for a DIFFERENT create (another tab on
 * this connection, in principle) must not resolve this one.
 */
export async function createWorkspace(name: string, timeoutMs = 8000): Promise<WorkspaceInfo | null> {
  const c = getClient();
  return new Promise((resolve) => {
    let done = false;
    const finish = (info: WorkspaceInfo | null) => {
      if (done) return;
      done = true;
      unsubCreated();
      unsubErr();
      clearTimeout(timer);
      resolve(info);
    };
    const unsubCreated = c.subscribe(EventTypeValue.WORKSPACE_CREATED, (ev) => {
      const parsed = parseWorkspaceCreatedEvent(ev);
      if (parsed && parsed.name === name) finish(parsed); // ignore a different create's reply
    });
    const unsubErr = c.subscribe(EventTypeValue.ERROR, () => finish(null));
    const timer = setTimeout(() => finish(null), timeoutMs);
    c.sendRawEvent({ type: EventTypeValue.WORKSPACE_CREATE_REQUEST, name }).catch(() => finish(null));
  });
}

/** ``workspace.delete`` (protocol 1.13): the daemon answers with ``workspace.deleted``. */
export async function deleteWorkspace(name: string): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_DELETE_REQUEST, name });
}

export async function updateConfig(cfg: { provider?: string; model?: string; api_key?: string }): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.CONFIG_UPDATE_REQUEST, ...cfg });
}

/**
 * Probe workspace mode: ask for the list and settle on the first of
 * ``WorkspaceListEvent`` (enabled) or an ``ErrorEvent`` naming
 * workspace mode (disabled — a plain single-workspace daemon).
 */
export function probeWorkspaceMode(timeoutMs = 4000): Promise<"enabled" | "disabled" | "unknown"> {
  const c = getClient();
  return new Promise((resolve) => {
    let done = false;
    const finish = (mode: "enabled" | "disabled" | "unknown") => {
      if (done) return;
      done = true;
      unsubList();
      unsubErr();
      clearTimeout(timer);
      useJaato.getState().setWorkspaceMode(mode);
      resolve(mode);
    };
    const unsubList = c.subscribe(EventTypeValue.WORKSPACE_LIST, () => finish("enabled"));
    const unsubErr = c.subscribe(EventTypeValue.ERROR, (ev) => {
      const msg = String((ev as unknown as { error?: string }).error ?? "").toLowerCase();
      if (msg.includes("workspace mode")) finish("disabled");
    });
    const timer = setTimeout(() => finish("unknown"), timeoutMs);
    requestWorkspaceList().catch(() => finish("unknown"));
  });
}
