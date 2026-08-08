// Connection state machine for JaatoClient.
//
// Mirror of jaato-sdk/jaato_sdk/client/recovery.py:ConnectionState
// + ConnectionStatus.  The TS client uses the same three states
// and the same status payload so cross-language status reporters
// can interop (e.g. a multi-client dashboard rendering connection
// health for both Python and TS clients side-by-side).

/**
 * Connection lifecycle states.
 *
 * @see ConnectionStatus for the rich status payload emitted on transitions.
 */
export enum ConnectionState {
  /** Not yet connected; initial state. */
  DISCONNECTED = "disconnected",
  /** Connecting or fully connected and ready to send/receive. */
  CONNECTED = "connected",
  /** Connection lost; auto-reconnect attempts in progress. */
  RECONNECTING = "reconnecting",
  /** Connection closed by user or unrecoverable error; no further attempts. */
  CLOSED = "closed",
}

/**
 * Status snapshot emitted on every state transition.
 *
 * Subscribe via {@link JaatoClient.onStatus} to receive these.
 */
export interface ConnectionStatus {
  state: ConnectionState;
  /** Human-readable reason or context (e.g. "auth failed", "network unreachable"). */
  reason?: string;
  /** Reconnection attempt number, when state is RECONNECTING. */
  reconnectAttempt?: number;
  /** Seconds until the next reconnect attempt, when state is RECONNECTING. */
  reconnectDelaySeconds?: number;
  /** Server version reported in ConnectedEvent (when state is CONNECTED). */
  serverVersion?: string;
  /** Client ID assigned by the server (when state is CONNECTED). */
  clientId?: string;
}

/**
 * Recovery policy — controls reconnect behaviour.
 *
 * Mirror of jaato-sdk/jaato_sdk/client/recovery.py:RecoveryConfig.
 * Defaults match the Python side so deployments can swap clients
 * without surprising changes in retry cadence.
 */
export interface RecoveryConfig {
  /** Auto-reconnect on connection loss.  Default: true. */
  autoReconnect: boolean;
  /** Maximum reconnect attempts before giving up; null = infinite. */
  maxReconnectAttempts: number | null;
  /** Initial backoff delay in seconds.  Doubles each attempt up to max. */
  initialBackoffSeconds: number;
  /** Cap on backoff delay between attempts. */
  maxBackoffSeconds: number;
  /** Random jitter factor (0-1) applied to backoff. */
  jitterFactor: number;
  /**
   * After a successful reconnect, automatically call
   * ``attachSession(sessionId)`` if the client knows a sessionId
   * (from a prior ``SessionInfoEvent`` or explicit
   * ``attachSession``).  The server then replays buffered events
   * from the session journal so the consumer picks up where it
   * left off.  Default: ``false`` — the consumer wires re-attach
   * via an ``onStatus`` handler.
   */
  autoReattachSessionId: boolean;
}

export const DEFAULT_RECOVERY_CONFIG: RecoveryConfig = {
  autoReconnect: true,
  maxReconnectAttempts: null,
  initialBackoffSeconds: 1.0,
  maxBackoffSeconds: 30.0,
  jitterFactor: 0.1,
  autoReattachSessionId: false,
};
