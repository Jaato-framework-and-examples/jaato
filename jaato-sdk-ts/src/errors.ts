// Error types thrown by JaatoClient.
//
// Mirrors of jaato-sdk/jaato_sdk/client/{ipc,recovery}.py exception
// classes — naming + semantics line up so cross-language consumers
// can recognise the same failure modes by name.

/**
 * The connection closed before, during, or after the handshake.
 * Caller-facing distinct from network-level WebSocket close events.
 */
export class ConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConnectionError";
  }
}

/**
 * Operation rejected because the client is mid-reconnect.
 *
 * Sends are buffered or rejected (depending on the call) while the
 * client is in the RECONNECTING state.  Catch this to back off
 * client-initiated work and wait for the next CONNECTED status
 * notification.
 */
export class ReconnectingError extends Error {
  constructor(message = "Client is reconnecting") {
    super(message);
    this.name = "ReconnectingError";
  }
}

/**
 * Operation rejected because the connection is permanently closed.
 *
 * Distinguishable from {@link ReconnectingError} because there is
 * no recovery path — caller must construct a new {@link JaatoClient}
 * and reconnect.
 */
export class ConnectionClosedError extends Error {
  constructor(message = "Connection is closed") {
    super(message);
    this.name = "ConnectionClosedError";
  }
}

/**
 * The server reported a version older than this SDK supports.
 *
 * Non-retryable: an old server will not become newer on retry.
 * Catch this and surface a clear "upgrade the server" message.
 *
 * Mirrors jaato-sdk/jaato_sdk/client/ipc.py:IncompatibleServerError
 * one-for-one — the property names match.
 */
export class IncompatibleServerError extends Error {
  readonly serverVersion: string;
  readonly minVersion: string;

  constructor(serverVersion: string, minVersion: string) {
    super(
      `Server version ${serverVersion} is not supported by this client ` +
      `(requires >= ${minVersion}). Please upgrade the server.`,
    );
    this.name = "IncompatibleServerError";
    this.serverVersion = serverVersion;
    this.minVersion = minVersion;
  }
}
