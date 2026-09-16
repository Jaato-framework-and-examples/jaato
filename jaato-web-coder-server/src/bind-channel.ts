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
import { ConnectionState, EventTypeValue, JaatoClient, type JaatoEvent } from "@jaato/sdk";

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

  async close(): Promise<void> {
    for (const [id, p] of this._pending) { clearTimeout(p.timer); p.reject(new BindUnavailableError("bind channel closed")); this._pending.delete(id); }
    await this._client.close();
  }

  get connected(): boolean {
    return this._client.state === ConnectionState.CONNECTED;
  }

  onStatus(handler: (state: string) => void): void {
    this._client.onStatus((s) => handler(String(s.state)));
  }

  private _onEvent(ev: Record<string, unknown>): void {
    const type = ev.type;
    if (type !== EventTypeValue.TICKET_BIND_RESULT && type !== EventTypeValue.TICKET_REVOKE_RESULT) return;
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
}
