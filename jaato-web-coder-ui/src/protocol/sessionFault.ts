/**
 * Whether an ``ErrorEvent`` ends a session, per the daemon's own
 * ``recoverable`` contract (``jaato_sdk/events.py``: ``recoverable: bool =
 * True``; every site that sends ``False`` is a config or provider-connect
 * failure that ends session initialization -- ``RunnerBootstrapFailed``
 * among them, and the CLAUDE.md-documented case: a bootstrap failure that
 * used to reach the client only as one more line in the transcript,
 * indistinguishable from a recoverable tool error, while the status bar
 * went on reading the WebSocket transport's own state and stayed green
 * "connected" through a session that had already failed to come up.
 *
 * ``recoverable`` and ``is_processing`` are different axes -- the
 * transport can be perfectly healthy while the SESSION it was trying to
 * open never existed -- so this is read off the event, not derived from
 * the connection phase.
 */
export interface SessionFault {
  errorType: string;
  message: string;
}

/**
 * ``recoverable === false`` is the one signal that means "this session is
 * not coming up"; every other ``ErrorEvent`` (a refused tool call, a
 * missing session by id, a turn that failed) is recoverable and changes
 * nothing here.  Absent ``recoverable`` defaults to ``true`` on the wire,
 * so an event that omits the field is read the same way.
 */
export function faultFromError(errorType: unknown, message: unknown, recoverable: unknown): SessionFault | null {
  if (recoverable !== false) return null;
  return { errorType: typeof errorType === "string" && errorType ? errorType : "SessionError", message: typeof message === "string" ? message : "" };
}
