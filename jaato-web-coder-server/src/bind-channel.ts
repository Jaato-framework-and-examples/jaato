/**
 * The bind channel: one WebSocket to the daemon, authenticated with the
 * application credential, over which tickets are minted and revoked.
 *
 * It is an ordinary ``JaatoClient`` (protocol 1.10 events, #1074) with two
 * deliberate settings: the credential goes in the ``Authorization`` header
 * (Node can set it; a query string would end up in logs), and **no
 * ``clientConfig``** — an app-credential connection answers any frame other
 * than ``ticket.bind`` / ``ticket.revoke`` with an error, and the client
 * only sends ``client.config`` when that option is set.  The client's own
 * reconnect loop keeps the channel up across daemon restarts.
 *
 * Both verbs are request/response correlated by ``request_id``, so many
 * browsers may be minting through one channel at once.  A result that does
 * not arrive within ``timeoutMs`` is a :class:`BindUnavailableError`; so is
 * a channel that is not connected.  The route turns both into a 503 the
 * page retries, never into a ticket.
 */
import { randomUUID } from "node:crypto";
import {
  ConnectionState,
  EventTypeValue,
  JaatoClient,
  MIN_WORKSPACE_APP_WRITE_PROTOCOL,
  SecretResolveResponder,
  isProtocolCompatible,
  type JaatoEvent,
  type SecretResolveHandler,
} from "@jaato/sdk";

export class BindUnavailableError extends Error {
  override name = "BindUnavailableError";
}

/** The daemon answered, and the answer is not a ticket. */
export class BindRefusedError extends Error {
  override name = "BindRefusedError";
  constructor(public readonly status: string, detail?: string | null) {
    super(`ticket.bind ${status}${detail ? `: ${detail}` : ""}`);
  }
}

/** A ``workspace.app_write`` body: ``env`` values are ``app://`` references or ``null`` (remove). */
export interface WorkspaceWriteRequest {
  env: Record<string, string | null>;
  files: Array<{ path: string; content: string | null; managed_by?: string | null }>;
}

/** The daemon's answer: a status, then one action per env name and per file. */
export interface WorkspaceWriteAnswer {
  status: string;
  env: Record<string, string>;
  files: Array<{ path: string; action: string; detail?: string }>;
  detail?: string;
}

export interface BoundTicket {
  ticket: string;
  qualified: string;
  appId: string;
  expiresAt: string;
}

export interface BindChannelOptions {
  bindUrl: string;
  appCredential: string;
  timeoutMs?: number;
  /** Test seam: construct the client differently. */
  clientFactory?: (opts: { url: string; headers: Record<string, string> }) => JaatoClient;
}

interface Pending {
  resolve: (ev: Record<string, unknown>) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class BindChannel {
  private readonly _client: JaatoClient;
  private readonly _pending = new Map<string, Pending>();
  private readonly _timeoutMs: number;
  private _resolveResponder: SecretResolveResponder | null = null;

  constructor(opts: BindChannelOptions) {
    this._timeoutMs = opts.timeoutMs ?? 10_000;
    const make = opts.clientFactory ?? ((o) => new JaatoClient({ ...o, recovery: { autoReconnect: true, maxReconnectAttempts: null } }));
    this._client = make({ url: opts.bindUrl, headers: { Authorization: `Bearer ${opts.appCredential}` } });
    this._client.subscribeAll((ev: JaatoEvent) => this._onEvent(ev as unknown as Record<string, unknown>));
  }

  /** Open the channel; a failure here is fatal for the server start. */
  async connect(): Promise<void> {
    await this._client.connect();
  }

  /**
   * Answer the daemon's ``secret.resolve`` requests (#1226) with ``handler``.
   * This is the daemon -> application direction on the same channel; the
   * responder subscribes to ``secret.resolve`` and replies correlated by
   * ``request_id``, carrying no credential policy of its own — that is the
   * handler's.  Idempotent: a second call replaces the responder.
   */
  attachSecretResolver(handler: SecretResolveHandler): void {
    this._resolveResponder?.stop();
    this._resolveResponder = new SecretResolveResponder(this._client, handler);
    this._resolveResponder.start();
  }

  async close(): Promise<void> {
    this._resolveResponder?.stop();
    this._resolveResponder = null;
    for (const [id, p] of this._pending) { clearTimeout(p.timer); p.reject(new BindUnavailableError("bind channel closed")); this._pending.delete(id); }
    await this._client.close();
  }

  get connected(): boolean {
    return this._client.state === ConnectionState.CONNECTED;
  }

  onStatus(handler: (state: string) => void): void {
    this._client.onStatus((s) => handler(String(s.state)));
  }

  /** Call ``handler`` each time the channel (re)connects — after the daemon's protocol is known. */
  onConnected(handler: () => void): void {
    this._client.onStatus((s) => { if (s.state === ConnectionState.CONNECTED) handler(); });
  }

  /**
   * Whether the daemon answers ``workspace.app_write`` (protocol 1.30).  An
   * older daemon replies to it with an error frame and never the result, so
   * the request would only time out; the caller checks this instead.
   */
  canWriteWorkspaces(): boolean {
    const v = this._client.serverProtocolVersion;
    return this.connected && typeof v === "string" && isProtocolCompatible(v, MIN_WORKSPACE_APP_WRITE_PROTOCOL);
  }

  /**
   * Ask the daemon to write a binding's files into ``workspace`` (1.30): the
   * ``app://`` reference in ``.env`` and allow-listed home / instruction
   * files.  For a deployment where this application cannot reach the
   * daemon's workspaces (it runs as another account).  The daemon checks
   * that ``user`` of this application owns the workspace and refuses a
   * literal secret or a path off its allow-list; per-item outcomes come back.
   */
  async writeWorkspace(user: string, workspace: string, request: WorkspaceWriteRequest): Promise<WorkspaceWriteAnswer> {
    const ev = await this._request({ type: EventTypeValue.WORKSPACE_APP_WRITE_REQUEST, user, workspace, env: request.env, files: request.files });
    const files = Array.isArray(ev.files) ? (ev.files as WorkspaceWriteAnswer["files"]) : [];
    const env = ev.env && typeof ev.env === "object" ? (ev.env as Record<string, string>) : {};
    return { status: String(ev.status ?? "unknown"), env, files, detail: typeof ev.detail === "string" ? ev.detail : undefined };
  }

  /**
   * Whether ``user`` of this application owns ``workspace`` on the daemon.
   * An EMPTY ``workspace.app_write`` is the probe: the daemon checks the
   * ownership before anything else and writes nothing for an empty request,
   * so ``ok`` means owned and ``not_found`` means not (an unknown path and
   * another user's workspace answer alike).  ``null`` when the daemon could
   * not be asked at all (channel down, a daemon below protocol 1.30), which
   * a caller about to install software must treat as a refusal.
   */
  async owns(user: string, workspace: string): Promise<boolean | null> {
    if (!this.canWriteWorkspaces()) return null;
    let ans: WorkspaceWriteAnswer;
    try { ans = await this.writeWorkspace(user, workspace, { env: {}, files: [] }); }
    catch { return null; }
    if (ans.status === "ok") return true;
    if (ans.status === "not_found") return false;
    return null;
  }

  private _onEvent(ev: Record<string, unknown>): void {
    const type = ev.type;
    if (
      type !== EventTypeValue.TICKET_BIND_RESULT &&
      type !== EventTypeValue.TICKET_REVOKE_RESULT &&
      type !== EventTypeValue.SECRET_RELOAD_RESULT &&
      type !== EventTypeValue.WORKSPACE_APP_WRITE_RESULT
    ) return;
    const id = typeof ev.request_id === "string" ? ev.request_id : "";
    const p = this._pending.get(id);
    if (!p) return;
    this._pending.delete(id);
    clearTimeout(p.timer);
    p.resolve(ev);
  }

  private async _request(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
    if (!this.connected) throw new BindUnavailableError("bind channel is not connected to the daemon");
    const request_id = randomUUID();
    const result = new Promise<Record<string, unknown>>((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pending.delete(request_id);
        reject(new BindUnavailableError(`no ${String(payload.type)} result within ${this._timeoutMs} ms`));
      }, this._timeoutMs);
      this._pending.set(request_id, { resolve, reject, timer });
    });
    try {
      await this._client.sendRawEvent({ ...payload, request_id });
    } catch (e) {
      const p = this._pending.get(request_id);
      if (p) { clearTimeout(p.timer); this._pending.delete(request_id); }
      throw new BindUnavailableError(`could not send ${String(payload.type)}: ${(e as Error).message}`);
    }
    return result;
  }

  /** Mint a single-use ticket for ``user``. */
  async bind(user: string, ttlSeconds: number): Promise<BoundTicket> {
    const ev = await this._request({ type: EventTypeValue.TICKET_BIND_REQUEST, user, ttl_seconds: ttlSeconds, single_use: true });
    if (ev.status !== "bound" || typeof ev.ticket !== "string" || !ev.ticket) {
      throw new BindRefusedError(String(ev.status ?? "unknown"), typeof ev.detail === "string" ? ev.detail : null);
    }
    return { ticket: ev.ticket, qualified: String(ev.qualified ?? ""), appId: String(ev.app_id ?? ""), expiresAt: String(ev.expires_at ?? "") };
  }

  /** Revoke every outstanding ticket of ``user``; ``not_found`` is the normal answer after a completed login. */
  async revokeUser(user: string): Promise<{ status: string; revoked: number }> {
    const ev = await this._request({ type: EventTypeValue.TICKET_REVOKE_REQUEST, user });
    return { status: String(ev.status ?? "unknown"), revoked: Number(ev.revoked ?? 0) };
  }

  /**
   * Ask the daemon to re-resolve ``user``'s LOADED sessions (#1226 §6.4), so a
   * just-changed ``app://`` reference (a GitHub disconnect, a workspace -> none)
   * drops out of a live session's environment rather than lingering.  The
   * daemon qualifies ``user`` with this connection's app id, so one
   * application can never reload another's ``alice``.  ``reloaded: 0`` — the
   * user has nothing loaded — is the ordinary, non-error answer.
   */
  async reloadUser(user: string): Promise<{ status: string; reloaded: number }> {
    const ev = await this._request({ type: EventTypeValue.SECRET_RELOAD_REQUEST, user });
    return { status: String(ev.status ?? "unknown"), reloaded: Number(ev.reloaded ?? 0) };
  }
}
