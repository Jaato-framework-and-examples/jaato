/**
 * Adapter between ``@jaato/sdk``'s ``JaatoClient`` and the store.
 *
 * Owns exactly three things:
 *   1. the client's lifecycle (one live client per page; ``connect`` /
 *      ``disconnect``), mirrored into ``state.connection``;
 *   2. event delivery — every wire event is queued and flushed into the
 *      store once per animation frame, so a token stream costs one
 *      commit per frame (the SDK already preserves order; we preserve
 *      it too by flushing the queue in FIFO);
 *   3. the request helpers the UI needs that the SDK exposes only as raw
 *      verbs (workspace list/select/create, config update).
 *
 * The client identifies itself as ``client_type: "web"`` with
 * ``supports_expandable_content: true`` — the model is told its output
 * will be shown in a browser that can collapse overflow, so it need not
 * shorten for a narrow terminal.
 */
import { EventTypeValue, JaatoClient, type ConnectionStatus, type JaatoEvent } from "@jaato/sdk";
import { useJaato } from "@/store/store";

export interface ConnectOptions {
  url: string;
  token?: string;
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
  c.onStatus(onStatus);
  c.subscribeAll((ev) => enqueue(ev));
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

export async function disconnect(): Promise<void> {
  const c = client;
  client = null;
  if (c) await c.close().catch(() => undefined);
}

export function getClient(): JaatoClient {
  if (!client) throw new Error("Not connected");
  return client;
}

export function isConnected(): boolean {
  return client !== null;
}

// ── Workspace-first flow (WS daemons started with --workspace-root) ──

export async function requestWorkspaceList(): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_LIST_REQUEST });
}

export async function selectWorkspace(name: string): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_SELECT_REQUEST, name });
}

export async function createWorkspace(name: string): Promise<void> {
  await getClient().sendRawEvent({ type: EventTypeValue.WORKSPACE_CREATE_REQUEST, name });
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
