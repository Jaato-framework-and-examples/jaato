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
 * The server's wire-protocol version is incompatible with this SDK.
 *
 * Non-retryable — an old (or wrong-major) daemon will not become
 * compatible on retry.  Catch this and surface a clear protocol-
 * version mismatch message; the package version is reported for
 * diagnostics but isn't the actionable signal.
 *
 * Mirrors jaato-sdk/jaato_sdk/client/ipc.py:IncompatibleServerError
 * one-for-one — the property names match.
 */
export class IncompatibleServerError extends Error {
  /** Wire-protocol version reported by the daemon. */
  readonly serverProtocol: string;
  /** Minimum protocol version this client requires. */
  readonly minProtocol: string;
  /**
   * Daemon package version (from ``server_info.server_version``).
   * Diagnostics only — not used by the compat check.
   */
  readonly serverVersion: string;

  constructor(
    serverProtocol: string,
    minProtocol: string,
    serverVersion?: string,
  ) {
    const sv = serverVersion ?? "unknown";
    const sParts = serverProtocol.split(".").map((p) => parseInt(p, 10));
    const cParts = minProtocol.split(".").map((p) => parseInt(p, 10));
    let hint = "version mismatch";
    if (
      !Number.isNaN(sParts[0]) &&
      !Number.isNaN(cParts[0]) &&
      sParts[0] !== cParts[0]
    ) {
      hint =
        `major-version mismatch (server speaks ${sParts[0]}.x, ` +
        `client needs ${cParts[0]}.x) — wire shapes are incompatible`;
    } else if (
      !Number.isNaN(sParts[1]) &&
      !Number.isNaN(cParts[1]) &&
      sParts[1] < cParts[1]
    ) {
      hint =
        `server minor ${sParts[1]} is below client's required minor ` +
        `${cParts[1]} — daemon is missing fields the client depends on`;
    }
    super(
      `Server protocol ${serverProtocol} is not supported by this client ` +
        `(requires >= ${minProtocol}): ${hint}. Daemon package: ${sv}.`,
    );
    this.name = "IncompatibleServerError";
    this.serverProtocol = serverProtocol;
    this.minProtocol = minProtocol;
    this.serverVersion = sv;
  }

  /** Alias for {@link minProtocol} (pre-1.0 compatibility). */
  get minVersion(): string {
    return this.minProtocol;
  }
}
