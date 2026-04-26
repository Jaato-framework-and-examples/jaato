// Low-level WebSocket transport for JaatoClient.
//
// Owns: connection open/close, frame parse/serialise, auth.  Does
// NOT own: handshake, reconnect, request/response correlation,
// connection state machine — those live in client.ts on top of
// this transport.
//
// The implementation deliberately uses only the standard
// WebSocket API (Node 21+ ships it; browsers always have it).  No
// dependency on the npm `ws` package, so the SDK works in
// browsers without a polyfill.  Header-based auth is browser-
// hostile (the standard WebSocket constructor rejects custom
// headers); we use the ?token=<token> query-param form documented
// in jaato-server's WS handshake, which is browser-compatible and
// also accepted by the server when issued from Node.

import { ConnectionError } from "./errors.js";
import type { JaatoEvent } from "./events.js";

/**
 * Connection options accepted by {@link openTransport}.
 */
export interface TransportOptions {
  /** Full ``ws://`` or ``wss://`` URL of the jaato daemon. */
  url: string;
  /**
   * Bearer token presented as ``?token=<token>`` query parameter.
   * Omit when the daemon is started with ``--ws-unsafe-no-auth``.
   * Mutually exclusive with {@link headers} — when both are
   * supplied, ``headers`` wins (Node only).
   */
  token?: string;
  /**
   * Custom request headers (Node only).  Browsers cannot set
   * headers on the ``WebSocket`` upgrade request; supplying this
   * in a browser environment is a no-op.  Use the ``token`` field
   * instead for browser auth.
   */
  headers?: Record<string, string>;
  /** Connection open timeout in milliseconds.  Default 5000. */
  openTimeoutMs?: number;
}

/**
 * The result of opening a transport.
 *
 * Wraps a live ``WebSocket`` with helpers for typed-event send /
 * receive.  Caller owns the lifecycle — call ``close()`` when done.
 */
export interface Transport {
  /** Send a fully-typed jaato event over the wire. */
  sendEvent(event: JaatoEvent): void;
  /**
   * Send a raw binary frame over the wire.
   *
   * Used by multi-frame protocols like ``StageFilesRequest`` —
   * caller sends the TEXT request frame via {@link sendEvent}, then
   * follows with N binary frames in declared order.  WebSocket
   * preserves frame order per-connection so the server-side
   * inline-binary-read pattern works deterministically.
   */
  sendBinary(data: ArrayBuffer | Uint8Array): void;
  /**
   * Async iterator of incoming events.  Yields until the
   * underlying WebSocket closes.  Rejects with
   * {@link ConnectionError} on parse failure (the wire format is
   * strict — every frame must be a valid serialised event).
   */
  events(): AsyncIterableIterator<JaatoEvent>;
  /** Close the WebSocket cleanly. */
  close(code?: number, reason?: string): void;
  /** Lifecycle observer — called once the WebSocket closes. */
  onClose(handler: (event: { code: number; reason: string }) => void): void;
  /** True until the WebSocket transitions to CLOSED. */
  readonly isOpen: boolean;
}

function _resolveAuthUrl(options: TransportOptions): string {
  // Token goes in the query string.  Headers (when supplied in
  // Node) take precedence and are passed via constructor options
  // below — in that case we leave the URL untouched.
  if (options.headers) {
    return options.url;
  }
  if (options.token == null) {
    return options.url;
  }
  const url = new URL(options.url);
  url.searchParams.set("token", options.token);
  return url.toString();
}

/**
 * Open a WebSocket connection to a jaato daemon.
 *
 * Resolves once the underlying socket reports OPEN.  Rejects with
 * {@link ConnectionError} on auth failure (server closes 1008),
 * timeout, or any other open-time error.
 */
export function openTransport(options: TransportOptions): Promise<Transport> {
  const openTimeoutMs = options.openTimeoutMs ?? 5000;
  const url = _resolveAuthUrl(options);

  // The standard WebSocket constructor accepts only (url, protocols).
  // Node-specific WebSocket implementations may accept a third
  // options arg for custom headers; cast through any to support
  // that without taking a hard dep on @types/ws.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ctor: any = (globalThis as any).WebSocket;
  if (ctor == null) {
    return Promise.reject(
      new ConnectionError(
        "No WebSocket implementation available.  Ensure Node 21+ or a browser environment.",
      ),
    );
  }

  let ws: WebSocket;
  try {
    if (options.headers) {
      // Node's WebSocket signature: new WebSocket(url, protocols, options)
      // Browsers don't accept a third arg — calling like this is harmless
      // because the headers are sent only when the runtime understands them.
      ws = new ctor(url, undefined, { headers: options.headers });
    } else {
      ws = new ctor(url);
    }
  } catch (e) {
    return Promise.reject(
      new ConnectionError(`WebSocket constructor failed: ${(e as Error).message}`),
    );
  }
  ws.binaryType = "arraybuffer";

  // The frames arriving while no consumer is reading buffer up
  // here so the consumer (events()) sees them in order on first
  // call.  Same shape as jaato-sdk/jaato_sdk/client/ipc.py's
  // _buffered_events, but at the transport altitude.
  const incoming: JaatoEvent[] = [];
  const waiters: Array<(event: JaatoEvent | null) => void> = [];
  let closed = false;
  let closeInfo: { code: number; reason: string } | null = null;
  const closeHandlers: Array<(info: { code: number; reason: string }) => void> = [];

  const _drainOnClose = (): void => {
    closed = true;
    while (waiters.length) {
      const w = waiters.shift()!;
      w(null);
    }
  };

  return new Promise<Transport>((resolve, reject) => {
    let openTimer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
      try {
        ws.close();
      } catch {
        // ignore
      }
      reject(new ConnectionError(`WebSocket open timed out after ${openTimeoutMs}ms`));
    }, openTimeoutMs);

    ws.onopen = (): void => {
      if (openTimer) {
        clearTimeout(openTimer);
        openTimer = null;
      }
      const transport: Transport = {
        sendEvent(event: JaatoEvent): void {
          if (closed) {
            throw new ConnectionError("Transport is closed");
          }
          ws.send(JSON.stringify(event));
        },
        sendBinary(data: ArrayBuffer | Uint8Array): void {
          if (closed) {
            throw new ConnectionError("Transport is closed");
          }
          // The standard WebSocket API accepts both ArrayBuffer and
          // ArrayBufferView (Uint8Array is a view).  Pass through
          // verbatim — no copy.
          ws.send(data);
        },
        async *events(): AsyncIterableIterator<JaatoEvent> {
          while (true) {
            if (incoming.length > 0) {
              yield incoming.shift()!;
              continue;
            }
            if (closed) {
              return;
            }
            const next = await new Promise<JaatoEvent | null>((res) => {
              waiters.push(res);
            });
            if (next == null) {
              return;
            }
            yield next;
          }
        },
        close(code?: number, reason?: string): void {
          try {
            ws.close(code ?? 1000, reason ?? "");
          } catch {
            // ignore
          }
        },
        onClose(handler): void {
          if (closeInfo) {
            handler(closeInfo);
          } else {
            closeHandlers.push(handler);
          }
        },
        get isOpen(): boolean {
          return !closed;
        },
      };
      resolve(transport);
    };

    ws.onerror = (): void => {
      if (openTimer) {
        clearTimeout(openTimer);
        openTimer = null;
        reject(new ConnectionError("WebSocket error during open"));
      }
      // After open, errors are surfaced to the consumer via close.
    };

    ws.onmessage = (msg: MessageEvent): void => {
      let payload: string;
      if (typeof msg.data === "string") {
        payload = msg.data;
      } else if (msg.data instanceof ArrayBuffer) {
        payload = new TextDecoder().decode(msg.data);
      } else if ((msg.data as Blob)?.text) {
        // Browser Blob — convert async then dispatch
        (msg.data as Blob).text().then((text) => {
          _dispatch(text);
        });
        return;
      } else {
        // Unknown frame — surface as close-with-error rather than swallow
        try {
          ws.close(1003, "unsupported frame type");
        } catch {
          // ignore
        }
        return;
      }
      _dispatch(payload);
    };

    ws.onclose = (event: CloseEvent): void => {
      if (openTimer) {
        clearTimeout(openTimer);
        openTimer = null;
        reject(
          new ConnectionError(
            `WebSocket closed before open (code=${event.code}, reason=${event.reason || "<empty>"})`,
          ),
        );
        return;
      }
      closeInfo = { code: event.code, reason: event.reason || "" };
      _drainOnClose();
      for (const h of closeHandlers) {
        try {
          h(closeInfo);
        } catch {
          // never let a handler crash the close path
        }
      }
    };

    function _dispatch(text: string): void {
      let parsed: JaatoEvent;
      try {
        parsed = JSON.parse(text) as JaatoEvent;
      } catch (e) {
        // Bad frame: close with 1003 and let consumers see it via the
        // events() generator returning early.
        try {
          ws.close(1003, "invalid JSON frame");
        } catch {
          // ignore
        }
        return;
      }
      if (waiters.length > 0) {
        const w = waiters.shift()!;
        w(parsed);
      } else {
        incoming.push(parsed);
      }
    }
  });
}
