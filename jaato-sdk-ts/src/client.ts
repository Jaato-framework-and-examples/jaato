// JaatoClient — WebSocket client for jaato-server.
//
// Mirror of jaato-sdk/jaato_sdk/client/{ipc,recovery}.py method-for-method.
// The Python side does both IPC (Unix socket / Windows pipe) and WS;
// the TS side is WS-only because browsers can't speak Unix sockets.
//
// Layout:
//   - JaatoClient class — the full public surface
//   - Handshake: send ClientConfigRequest, await ConnectedEvent,
//     enforce min_server_version
//   - Reconnect: exponential backoff with state machine
//     (CONNECTED → RECONNECTING → CONNECTED, or → CLOSED)
//   - Typed methods: every IPCClient method mirrored 1:1 with the
//     same noun (camelCase per JS convention)
//
// See ../README.md and project_backlog_sdk_feature_parity.md for
// the contract this implements.

import {
  ConnectionClosedError,
  ConnectionError,
  IncompatibleServerError,
  ReconnectingError,
} from "./errors.js";
import {
  ConnectionState,
  type ConnectionStatus,
  DEFAULT_RECOVERY_CONFIG,
  type RecoveryConfig,
} from "./state.js";
import { openTransport, type Transport } from "./transport.js";
import {
  EventTypeValue,
  type EventType,
  type ClientConfigRequest,
  type CommandRequest,
  type ConnectedEvent,
  type JaatoEvent,
  type SendMessageRequest,
  type StopRequest,
  type PermissionResponseRequest,
  type ClarificationResponseRequest,
  type ReferenceSelectionResponseRequest,
  type CommandListRequest,
  type HistoryRequest,
  type ToolDisableRequest,
  type ToolsRegisterClientRequest,
  type ToolExecuteResultEvent,
  type StageFilesRequest,
  type StageFilesEvent,
  type StagedFileSpec,
  type InjectPromptRequest,
  type ReplayMessagesRequest,
  type ResolveForkPointRequest,
  type PermissionAddWhitelistRequest,
  type PermissionAddBlacklistRequest,
  type PermissionRemoveRequest,
  type PermissionClearRequest,
  type PermissionSetDefaultRequest,
  type PermissionPolicySnapshotRequest,
} from "./events.js";
import type {
  CatchallEventHandler,
  EventByType,
  EventHandler,
  SubscribeManyMap,
  Unsubscribe,
} from "./event-typing.js";

/**
 * Internal record of a registered handler.
 *
 * Stored in {@link JaatoClient}'s typed-handler map (keyed by EventType
 * string) or catchall list. `id` is unique across all entries and used
 * by the unsubscribe closure to find and remove this record.
 */
interface HandlerEntry {
  handler: (event: JaatoEvent) => void | Promise<void>;
  once: boolean;
  id: number;
}

/**
 * The earliest wire-protocol version this SDK is compatible with.
 *
 * Bumped when the SDK depends on a new wire shape — typically when
 * gap-numbered changes land in `events.py`.  Compat is checked on
 * connect against ``ConnectedEvent.protocol_version`` from the daemon.
 *
 * Distinct from the daemon's *package* version (``server_version``),
 * which is no longer used for compat checks — it remains in
 * ``server_info`` for diagnostics only.
 *
 * See ``docs/sdk-protocol-versioning.md`` for the bump policy.
 */
export const MIN_PROTOCOL_VERSION = "1.0";

/**
 * Parse ``"MAJOR.MINOR"`` into ``[major, minor]``.  Extra components
 * are tolerated and dropped (e.g. ``"1.0.5"`` → ``[1, 0]``).  Returns
 * ``null`` on malformed input rather than throwing — the compat check
 * treats unparseable as incompatible.
 */
function _parseProtocolVersion(v: string): [number, number] | null {
  const parts = v.split(".");
  if (parts.length < 2) return null;
  const major = parseInt(parts[0]!, 10);
  const minor = parseInt(parts[1]!, 10);
  if (Number.isNaN(major) || Number.isNaN(minor)) return null;
  return [major, minor];
}

/**
 * Whether ``serverProtocol`` satisfies ``clientMin``.
 *
 * Rule (semver-flavoured):
 * - Server's MAJOR must equal client's MAJOR.  Different majors mean
 *   incompatible wire shapes.
 * - Server's MINOR must be >= client's required minor.  Server minor
 *   higher is fine — additive optional fields the client may not yet
 *   read.
 *
 * Either side malformed → ``false`` (refuse rather than guess).
 */
export function isProtocolCompatible(
  serverProtocol: string,
  clientMin: string,
): boolean {
  const s = _parseProtocolVersion(serverProtocol);
  const c = _parseProtocolVersion(clientMin);
  if (s == null || c == null) return false;
  if (s[0] !== c[0]) return false;
  return s[1] >= c[1];
}

/**
 * Constructor options for {@link JaatoClient}.
 */
export interface JaatoClientOptions {
  /** Full ``ws://`` or ``wss://`` URL of the jaato daemon. */
  url: string;
  /**
   * Bearer token presented as ``?token=<token>`` query parameter.
   * Omit when the daemon is started with ``--ws-unsafe-no-auth``.
   */
  token?: string;
  /**
   * Custom request headers (Node only).  Mutually exclusive with
   * {@link token}.  See {@link openTransport} for caveats.
   */
  headers?: Record<string, string>;
  /**
   * Override the SDK's compile-time minimum protocol version.  Use
   * only for development against unreleased daemons; production
   * deployments should leave this unset so the SDK refuses to connect
   * to incompatible servers.
   */
  minProtocolVersion?: string;
  /**
   * Recovery policy for automatic reconnection.  Override fields
   * piecewise; unspecified fields fall back to
   * {@link DEFAULT_RECOVERY_CONFIG}.  Pass ``{ autoReconnect: false }``
   * to disable reconnect entirely (the client transitions
   * straight to CLOSED on connection loss).
   */
  recovery?: Partial<RecoveryConfig>;
  /**
   * Client config sent in the post-connect handshake.  All fields
   * map to ClientConfigRequest on the wire — see jaato-sdk's
   * events.py for semantics.
   */
  clientConfig?: Omit<ClientConfigRequest, "type" | "timestamp">;
  /** Connection open timeout in milliseconds.  Default 5000. */
  openTimeoutMs?: number;
}

/**
 * Compare two semver-ish version strings.
 *
 * Returns -1 if a < b, 0 if equal, 1 if a > b.  Treats missing
 * components as 0 ("0.5" == "0.5.0").  Pre-release tags ignored —
 * sufficient for the integer-only bumps jaato-server uses.
 */
function _compareVersions(a: string, b: string): number {
  const partsA = a.split(".").map((p) => parseInt(p, 10) || 0);
  const partsB = b.split(".").map((p) => parseInt(p, 10) || 0);
  const length = Math.max(partsA.length, partsB.length);
  for (let i = 0; i < length; i++) {
    const valA = partsA[i] ?? 0;
    const valB = partsB[i] ?? 0;
    if (valA < valB) return -1;
    if (valA > valB) return 1;
  }
  return 0;
}

/**
 * WebSocket client for jaato-server.
 *
 * Mirrors jaato-sdk/jaato_sdk/client/IPCRecoveryClient method-for-method
 * with TS-idiomatic naming (camelCase) and Promise-based async.
 */
export class JaatoClient {
  private _options: JaatoClientOptions;
  private _recovery: RecoveryConfig;
  private _transport: Transport | null = null;
  private _state: ConnectionState = ConnectionState.DISCONNECTED;
  private _serverVersion: string | null = null;
  private _serverProtocolVersion: string | null = null;
  private _clientId: string | null = null;
  private _sessionId: string | null = null;
  private _statusHandlers: Array<(s: ConnectionStatus) => void> = [];
  // Typed handler buckets keyed by EventType string.  Catchall handlers
  // live in `_catchallHandlers`.  Mutated and dispatched on the JS event
  // loop only — no thread safety guarantees because there is no other
  // thread.  `_dispatchEvent` snapshots both buckets before iterating
  // so subscribe/unsubscribe calls inside a handler only take effect
  // for the next event.
  private _typedHandlers: Map<EventType, HandlerEntry[]> = new Map();
  private _catchallHandlers: HandlerEntry[] = [];
  private _handlerIdCounter = 0;
  private _bufferedEvents: JaatoEvent[] = [];
  private _eventLoopActive = false;
  private _reconnectAttempts = 0;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _explicitClose = false;

  constructor(options: JaatoClientOptions) {
    this._options = options;
    this._recovery = { ...DEFAULT_RECOVERY_CONFIG, ...(options.recovery ?? {}) };

    // Opt-in auto re-attach.  Wires an internal status handler that
    // fires attachSession(sessionId) on every RECONNECTING →
    // CONNECTED transition (i.e. after a successful reconnect, not
    // on the initial connect — sessionId is null at that point).
    // The server then replays buffered events from the session
    // journal so the consumer doesn't have to wire this manually.
    if (this._recovery.autoReattachSessionId) {
      let sawReconnecting = false;
      this.onStatus((status) => {
        if (status.state === ConnectionState.RECONNECTING) {
          sawReconnecting = true;
          return;
        }
        if (
          status.state === ConnectionState.CONNECTED
          && sawReconnecting
          && this._sessionId
        ) {
          sawReconnecting = false;
          // Fire-and-forget — re-attach failures will surface as
          // ErrorEvents on the event stream and trigger the next
          // status transition naturally.
          void this.attachSession(this._sessionId);
        }
      });
    }
  }

  // ──── Status / state ─────────────────────────────────────────────

  /** Current connection state. */
  get state(): ConnectionState {
    return this._state;
  }

  /** True iff the WebSocket is open and the handshake completed. */
  get isConnected(): boolean {
    return this._state === ConnectionState.CONNECTED;
  }

  /**
   * Server's package version reported in {@link ConnectedEvent}, after
   * handshake.  **Diagnostics only** — compat is checked against
   * {@link serverProtocolVersion}.
   */
  get serverVersion(): string | null {
    return this._serverVersion;
  }

  /**
   * Server's wire-protocol version from {@link ConnectedEvent}, after
   * handshake.  This is what the compat check ran against — distinct
   * from {@link serverVersion} (the daemon's package version).
   */
  get serverProtocolVersion(): string | null {
    return this._serverProtocolVersion;
  }

  /** Client ID assigned by the server in {@link ConnectedEvent}. */
  get clientId(): string | null {
    return this._clientId;
  }

  /** Last observed session ID (set by SessionInfoEvent). */
  get sessionId(): string | null {
    return this._sessionId;
  }

  /** Subscribe to connection-state transitions. */
  onStatus(handler: (status: ConnectionStatus) => void): () => void {
    this._statusHandlers.push(handler);
    return (): void => {
      const i = this._statusHandlers.indexOf(handler);
      if (i >= 0) this._statusHandlers.splice(i, 1);
    };
  }

  // ──── Connect / disconnect ───────────────────────────────────────

  /**
   * Open the WebSocket and complete the handshake.
   *
   * Resolves once {@link ConnectedEvent} arrives and the server
   * version passes the {@link MIN_SERVER_VERSION} check.  Throws
   * {@link IncompatibleServerError} on version mismatch (no retry —
   * an old server won't become newer); {@link ConnectionError} on
   * other failures.
   */
  async connect(): Promise<void> {
    if (this._state === ConnectionState.CONNECTED) {
      return;
    }
    if (this._state === ConnectionState.CLOSED) {
      throw new ConnectionClosedError("Client was closed; construct a new instance");
    }
    this._explicitClose = false;
    await this._openOnce();
    this._startEventLoop();
  }

  /**
   * Close the WebSocket and cancel any in-flight reconnect.
   *
   * After close, the client is permanently in the CLOSED state —
   * call sites that need to reconnect must construct a new
   * {@link JaatoClient}.
   */
  async close(): Promise<void> {
    this._explicitClose = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._transport) {
      this._transport.close(1000, "client close");
      this._transport = null;
    }
    this._transition(ConnectionState.CLOSED);
  }

  // ──── Event subscription API ─────────────────────────────────────

  /**
   * Subscribe to events of a specific type.
   *
   * The handler receives only events whose `type` field equals
   * `eventType`. Sync handlers run inline; async handlers are
   * dispatched fire-and-forget — order of *delivery* is FIFO, but
   * order of *completion* of async handlers is not guaranteed.
   *
   * Throwing inside a handler (or rejecting an async handler) is
   * logged and swallowed — it never breaks the event loop or affects
   * other subscribers.
   *
   * @returns Idempotent unsubscribe function.
   */
  subscribe<T extends EventType>(
    eventType: T,
    handler: EventHandler<T>,
  ): Unsubscribe {
    return this._addTypedHandler(eventType, handler as HandlerEntry["handler"], false);
  }

  /**
   * Subscribe to a single event of `eventType`, then auto-unsubscribe.
   *
   * The handler fires exactly once when the next matching event
   * arrives. The returned unsubscribe can be called early to cancel.
   */
  subscribeOnce<T extends EventType>(
    eventType: T,
    handler: EventHandler<T>,
  ): Unsubscribe {
    return this._addTypedHandler(eventType, handler as HandlerEntry["handler"], true);
  }

  /**
   * Subscribe to every event regardless of type (catchall firehose).
   *
   * Use sparingly — typed `subscribe` is preferred when you only care
   * about a specific event family.
   */
  subscribeAll(handler: CatchallEventHandler): Unsubscribe {
    return this._addCatchallHandler(handler, false);
  }

  /**
   * Register multiple typed handlers in one call.
   *
   * Returns a single unsubscribe that removes all of them atomically —
   * useful for "set up my client" call sites that want a single cleanup
   * point.
   */
  subscribeMany(map: SubscribeManyMap): Unsubscribe {
    const unsubs: Unsubscribe[] = [];
    for (const key of Object.keys(map) as EventType[]) {
      const handler = map[key];
      if (handler) {
        unsubs.push(
          this._addTypedHandler(key, handler as HandlerEntry["handler"], false),
        );
      }
    }
    return (): void => {
      for (const u of unsubs) u();
    };
  }

  private _addTypedHandler(
    type: EventType,
    handler: HandlerEntry["handler"],
    once: boolean,
  ): Unsubscribe {
    const id = ++this._handlerIdCounter;
    const entry: HandlerEntry = { handler, once, id };
    let bucket = this._typedHandlers.get(type);
    if (!bucket) {
      bucket = [];
      this._typedHandlers.set(type, bucket);
    }
    bucket.push(entry);
    return (): void => this._removeTypedHandlerId(type, id);
  }

  private _addCatchallHandler(
    handler: HandlerEntry["handler"],
    once: boolean,
  ): Unsubscribe {
    const id = ++this._handlerIdCounter;
    const entry: HandlerEntry = { handler, once, id };
    this._catchallHandlers.push(entry);
    return (): void => this._removeCatchallHandlerId(id);
  }

  private _removeTypedHandlerId(type: EventType, id: number): void {
    const bucket = this._typedHandlers.get(type);
    if (!bucket) return;
    const i = bucket.findIndex((e) => e.id === id);
    if (i >= 0) bucket.splice(i, 1);
  }

  private _removeCatchallHandlerId(id: number): void {
    const i = this._catchallHandlers.findIndex((e) => e.id === id);
    if (i >= 0) this._catchallHandlers.splice(i, 1);
  }

  // ──── Typed methods (parity with Python IPCClient) ───────────────

  async sendMessage(
    text: string,
    attachments?: Array<Record<string, unknown>>,
    parallelTools?: boolean | null,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.SEND_MESSAGE,
      text,
      attachments: attachments ?? [],
      parallel_tools: parallelTools ?? null,
    } as SendMessageRequest);
  }

  async stop(agentId?: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.STOP,
      agent_id: agentId ?? null,
    } as StopRequest);
  }

  async respondToPermission(
    requestId: string,
    response: string,
    editedArguments?: Record<string, unknown>,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_RESPONSE,
      request_id: requestId,
      response,
      edited_arguments: editedArguments ?? null,
    } as PermissionResponseRequest);
  }

  async respondToClarification(
    requestId: string,
    response: string,
    questionIndex = 0,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.CLARIFICATION_RESPONSE,
      request_id: requestId,
      question_index: questionIndex,
      response,
    } as ClarificationResponseRequest);
  }

  async respondToReferenceSelection(requestId: string, response: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.REFERENCE_SELECTION_RESPONSE,
      request_id: requestId,
      response,
    } as ReferenceSelectionResponseRequest);
  }

  /**
   * Return the result of a client-side tool execution.
   *
   * Sends ``ToolExecuteResultEvent`` so the server can resume the
   * model loop with the tool's result.  Caller-side counterpart of
   * the ``ToolExecuteRequestEvent`` the server emits when the model
   * invokes a client-registered tool (see {@link registerClientTools}).
   *
   * Mirror of Python ``IPCClient.respond_to_tool_execution``.
   *
   * @param callId The ``call_id`` from the originating
   *   ``ToolExecuteRequestEvent``.  Server uses this to correlate
   *   the response with the in-flight tool call.
   * @param result JSON-encoded tool result.  Empty string when
   *   ``error`` is set.
   * @param error Error message when execution failed.  Empty when
   *   ``result`` is set.  Setting both is undefined.
   */
  async respondToToolExecution(callId: string, result = "", error = ""): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.TOOL_EXECUTE_RESULT,
      call_id: callId,
      result,
      error,
    } as ToolExecuteResultEvent);
  }

  // ──── Session management (mirror of Python IPCClient) ────────────

  /**
   * Create a new session on the server.
   *
   * Fire-and-forget: the resulting ``SessionInfoEvent`` arrives
   * via the event stream and updates {@link sessionId}.  Subscribe
   * via {@link subscribe} to react to session creation.
   *
   * Mirror of Python ``IPCClient.create_session``.
   *
   * @param options Session-creation parameters.  When omitted the
   *   server uses its defaults.
   */
  async createSession(options: {
    name?: string;
    /**
     * Either a profile **name** (string) referencing a JSON file under
     * ``.jaato/profiles/`` on the server, **or** an inline **spec**
     * record with the same shape — recognised keys include ``model``
     * (required), ``provider``, ``plugins``, ``plugin_configs``,
     * ``system_instructions``, ``gc``, ``env``, ``max_turns``,
     * ``runtime_limits``, ``model_tiers``, ``completion_payload_schema``.
     * The two forms are mutually exclusive — pass one or the other.
     * The server validates the dict and rejects it with a clear
     * ``ErrorEvent`` if ``model`` is missing.
     */
    profile?: string | Record<string, unknown>;
    agent?: string;
    agentParams?: Record<string, string>;
  } = {}): Promise<void> {
    const args: string[] = options.name ? [options.name] : [];
    let payload: Record<string, unknown> | undefined;

    if (typeof options.profile === "string") {
      args.push("--profile", options.profile);
    } else if (
      options.profile !== undefined &&
      options.profile !== null &&
      typeof options.profile === "object"
    ) {
      payload = { spec: options.profile };
    } else if (options.profile !== undefined && options.profile !== null) {
      throw new TypeError(
        `createSession: 'profile' must be string (name) or object ` +
          `(inline spec), got ${typeof options.profile}`,
      );
    }

    if (options.agent) {
      args.push("--agent", options.agent);
    }
    if (options.agentParams) {
      for (const [k, v] of Object.entries(options.agentParams)) {
        args.push(`${k}=${v}`);
      }
    }
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.new",
      args,
      payload,
    } as CommandRequest);
  }

  /**
   * Attach to an existing session.
   *
   * After successful attach, the server replays buffered events
   * from the session journal (per the WS reconnect contract) so
   * the client picks up where it left off.  Combined with the
   * reconnect state-machine, this is the building block for
   * "survive a network blip" workflows.
   *
   * Mirror of Python ``IPCClient.attach_session``.
   *
   * @param sessionId The session to attach to.
   */
  async attachSession(sessionId: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.attach",
      args: [sessionId],
    } as CommandRequest);
    this._sessionId = sessionId;
  }

  /**
   * Get or create the default session.
   *
   * Fire-and-forget: response arrives via the event stream as a
   * ``SessionInfoEvent``.  Mirror of Python
   * ``IPCClient.get_default_session``.
   */
  async getDefaultSession(): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.default",
      args: [],
    } as CommandRequest);
  }

  /**
   * Request the list of sessions on the server.
   *
   * Response arrives via the event stream.  Mirror of Python
   * ``IPCClient.list_sessions``.
   */
  async listSessions(): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.list",
      args: [],
    } as CommandRequest);
  }

  /**
   * Request the list of available agent profiles.
   *
   * Response arrives via the event stream as a
   * ``SessionProfilesEvent``.  Mirror of Python
   * ``IPCClient.list_profiles``.
   */
  async listProfiles(): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.profiles",
      args: [],
    } as CommandRequest);
  }

  /**
   * Terminate the currently-attached session.
   *
   * Sends ``session.end`` — the server stops the session's
   * in-flight activity and emits a ``[SESSION_TERMINATED]``
   * marker so attached clients know the session is no longer
   * active.  The session record itself stays on disk; use
   * {@link deleteSession} to purge it.  Mirror of Python
   * ``IPCClient.end_session``.
   */
  async endSession(): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.end",
      args: [],
    } as CommandRequest);
  }

  /**
   * Permanently delete a session by ID.
   *
   * Sends ``session.delete`` — the server removes both
   * in-memory state and the on-disk journal for the named
   * session.  Response arrives via the event stream as a
   * ``SystemMessageEvent`` ("Session 'X' deleted." on success;
   * "Session 'X' not found." otherwise).  Mirror of Python
   * ``IPCClient.delete_session``.
   *
   * @param sessionId The session to delete.  Must be a known
   *   session ID (visible in {@link listSessions}).
   */
  async deleteSession(sessionId: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.delete",
      args: [sessionId],
    } as CommandRequest);
  }

  async executeCommand(command: string, args?: string[]): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command,
      args: args ?? [],
    } as CommandRequest);
  }

  /**
   * Send an arbitrary event-shaped object over the wire — escape
   * hatch for daemon-extension verbs that aren't in the public
   * {@link JaatoEvent} union.
   *
   * Use cases:
   * - premium's ``reconnect.list`` / ``reconnect.delete`` /
   *   ``auth.token`` verbs from ``session_reconnect.extension``
   * - premium's ``assets.list`` from ``asset_picker``
   * - any third-party daemon extension that registers its own WS
   *   message handlers (typed envelopes, not wrapped in
   *   ``command.execute``)
   *
   * The envelope must include a ``type`` string that the server's
   * dispatcher recognises.  No validation is performed on the
   * client side — the caller owns shape correctness.
   *
   * Responses (if any) arrive via the regular event stream and
   * surface in {@link subscribe} / {@link events} as
   * ``JaatoEvent``-typed values that won't narrow against the
   * public union; the caller filters by ``event.type``.
   *
   * Prefer {@link executeCommand} when the verb is dispatched via
   * ``command.execute`` (the stringly-typed escape hatch for
   * command-router verbs).  This method is for verbs that
   * register their OWN top-level message type.
   */
  async sendRawEvent(event: object): Promise<void> {
    if (this._state === ConnectionState.RECONNECTING) {
      throw new ReconnectingError();
    }
    if (this._state === ConnectionState.CLOSED) {
      throw new ConnectionClosedError();
    }
    if (this._transport == null) {
      throw new ConnectionError("No active transport — call connect() first");
    }
    this._transport.sendRawEvent(event);
  }

  async disableTool(toolName: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.TOOL_DISABLE_REQUEST,
      tool_name: toolName,
    } as ToolDisableRequest);
  }

  async requestCommandList(): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND_LIST_REQUEST,
    } as CommandListRequest);
  }

  async requestHistory(agentId = "main"): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.HISTORY_REQUEST,
      agent_id: agentId,
    } as HistoryRequest);
  }

  async registerClientTools(
    tools: Array<Record<string, unknown>>,
    categories?: Record<string, string>,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.TOOLS_REGISTER_CLIENT,
      tools,
      categories: categories ?? {},
    } as ToolsRegisterClientRequest);
  }

  // ──── SDK feature parity — session-primitive verbs ───────────────

  async injectPrompt(text: string, sourceType = "user", sourceId?: string): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.INJECT_PROMPT_REQUEST,
      text,
      source_type: sourceType,
      source_id: sourceId ?? null,
    } as InjectPromptRequest);
  }

  async replayMessages(
    requestId: string,
    messages?: Array<Record<string, unknown>> | null,
    timeoutSeconds = 120.0,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.REPLAY_MESSAGES_REQUEST,
      request_id: requestId,
      messages: messages ?? null,
      timeout_seconds: timeoutSeconds,
    } as ReplayMessagesRequest);
  }

  async resolveForkPoint(
    requestId: string,
    options: {
      afterMessage?: number;
      afterToolCall?: string;
      afterTimestamp?: string;
    } = {},
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.RESOLVE_FORK_POINT_REQUEST,
      request_id: requestId,
      after_message: options.afterMessage ?? null,
      after_tool_call: options.afterToolCall ?? null,
      after_timestamp: options.afterTimestamp ?? null,
    } as ResolveForkPointRequest);
  }

  // ──── SDK feature parity — permission policy verbs ───────────────

  async addWhitelistTools(tools?: string[], patterns?: string[]): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_ADD_WHITELIST_REQUEST,
      tools: tools ?? [],
      patterns: patterns ?? [],
    } as PermissionAddWhitelistRequest);
  }

  async addBlacklistTools(tools?: string[], patterns?: string[]): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_ADD_BLACKLIST_REQUEST,
      tools: tools ?? [],
      patterns: patterns ?? [],
    } as PermissionAddBlacklistRequest);
  }

  async removePermissionRules(
    target: "whitelist" | "blacklist",
    tools?: string[],
    patterns?: string[],
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_REMOVE_REQUEST,
      target,
      tools: tools ?? [],
      patterns: patterns ?? [],
    } as PermissionRemoveRequest);
  }

  async clearPermissionRules(target: "whitelist" | "blacklist" | "all" = "all"): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_CLEAR_REQUEST,
      target,
    } as PermissionClearRequest);
  }

  async setDefaultPolicy(policy: "allow" | "deny" | "ask"): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_SET_DEFAULT_REQUEST,
      policy,
    } as PermissionSetDefaultRequest);
  }

  async requestPolicySnapshot(requestId = ""): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.PERMISSION_POLICY_SNAPSHOT_REQUEST,
      request_id: requestId,
    } as PermissionPolicySnapshotRequest);
  }

  // ──── File staging (multi-frame WS protocol) ─────────────────────

  /**
   * Stage files into a workspace via the multi-frame WS protocol.
   *
   * Wire shape (per the server-side handler in
   * ``websocket.py:_handle_stage_files_request``):
   *
   * 1. Client sends ``StageFilesRequest`` as a TEXT WS frame
   *    declaring the file names + sizes.
   * 2. Client immediately sends N raw BINARY frames in the same
   *    order as ``files``.  Each frame's byte length must equal
   *    the corresponding ``size`` value.
   * 3. Server responds with a TEXT ``StageFilesEvent`` listing
   *    what was written / what failed.
   *
   * This method handles all three steps and returns the resulting
   * ``StageFilesEvent`` so the caller can inspect successes /
   * failures per-file.  The response is correlated to this call
   * by ordering: WebSocket preserves frame order per-connection
   * and the server reads the binaries inline before producing the
   * response, so the next ``StageFilesEvent`` arriving after this
   * call is the response to it.  Concurrent stageFiles calls on
   * the same client will interleave incorrectly — serialise them
   * caller-side.
   *
   * @param workspaceId Target workspace.  Empty targets the
   *   connection's currently-selected workspace.
   * @param files Each entry needs ``name`` (workspace-relative
   *   path) and ``data`` (the bytes).  ``contentType`` and ``mode``
   *   are optional informational hints.
   * @returns The server's ``StageFilesEvent`` reporting per-file
   *   success / failure.
   */
  async stageFiles(
    workspaceId: string,
    files: Array<{
      name: string;
      data: ArrayBuffer | Uint8Array;
      contentType?: string;
      mode?: number;
    }>,
  ): Promise<StageFilesEvent> {
    if (this._state !== ConnectionState.CONNECTED) {
      throw this._state === ConnectionState.RECONNECTING
        ? new ReconnectingError()
        : new ConnectionClosedError();
    }
    if (this._transport == null) {
      throw new ConnectionError("No active transport — call connect() first");
    }
    const transport = this._transport;

    // Build the spec list (TEXT request body).  Coerce Uint8Array
    // to its byte length for the size field; the server validates
    // that the binary frame's byte length matches.
    const specs: StagedFileSpec[] = files.map((f) => ({
      name: f.name,
      size:
        f.data instanceof ArrayBuffer
          ? f.data.byteLength
          : (f.data as Uint8Array).byteLength,
      content_type: f.contentType ?? null,
      mode: f.mode ?? null,
    }) as StagedFileSpec);

    // Set up the response waiter BEFORE sending — otherwise the
    // server's response could race with handler installation.
    const responsePromise = this._waitForNextEvent<StageFilesEvent>(
      (e) => e.type === EventTypeValue.WORKSPACE_FILES_STAGED,
    );

    transport.sendEvent({
      type: EventTypeValue.WORKSPACE_FILES_STAGE_REQUEST,
      workspace_id: workspaceId,
      files: specs,
    } as StageFilesRequest);

    // Send the binary frames in declared order — WebSocket
    // preserves order so the server's inline-binary-read pattern
    // works as designed.
    for (const f of files) {
      transport.sendBinary(f.data);
    }

    return responsePromise;
  }

  /**
   * Internal: resolve with the next event matching ``predicate``.
   *
   * One-shot subscription used by request/response methods like
   * {@link stageFiles}.  Auto-unsubscribes after the first match.
   */
  private _waitForNextEvent<T extends JaatoEvent = JaatoEvent>(
    predicate: (event: JaatoEvent) => boolean,
  ): Promise<T> {
    return new Promise<T>((resolve) => {
      const unsub = this.subscribeAll((event) => {
        if (predicate(event)) {
          unsub();
          resolve(event as T);
        }
      });
    });
  }

  // ──── Internals ──────────────────────────────────────────────────

  private async _sendEvent(event: JaatoEvent): Promise<void> {
    if (this._state === ConnectionState.RECONNECTING) {
      throw new ReconnectingError();
    }
    if (this._state === ConnectionState.CLOSED) {
      throw new ConnectionClosedError();
    }
    if (this._transport == null) {
      throw new ConnectionError("No active transport — call connect() first");
    }
    this._transport.sendEvent(event);
  }

  private async _openOnce(): Promise<void> {
    const transport = await openTransport({
      url: this._options.url,
      token: this._options.token,
      headers: this._options.headers,
      openTimeoutMs: this._options.openTimeoutMs,
    });
    this._transport = transport;
    transport.onClose((info) => this._handleClose(info));

    // Handshake: server sends ConnectedEvent first, then we send
    // ClientConfigRequest.  Pull the first frame off the events
    // generator and verify it's a ConnectedEvent.  Anything else is
    // a protocol violation and we surface it as ConnectionError.
    const iter = transport.events();
    const firstFrame = await iter.next();
    if (firstFrame.done) {
      this._transport = null;
      throw new ConnectionError("Server closed connection before sending ConnectedEvent");
    }
    const first = firstFrame.value;
    if (first.type !== EventTypeValue.CONNECTED) {
      this._transport = null;
      transport.close();
      throw new ConnectionError(
        `Expected ConnectedEvent from server, got ${first.type}`,
      );
    }
    const connected = first as ConnectedEvent;
    const serverInfo = connected.server_info ?? {};
    this._serverVersion = (serverInfo.server_version as string) ?? null;
    this._clientId = (serverInfo.client_id as string) ?? null;
    this._serverProtocolVersion = connected.protocol_version ?? null;

    // Wire-protocol compat gate.  Compares against
    // ``protocol_version`` (not the daemon package version) — the
    // package can bump without changing the wire and vice versa.
    const minRequired =
      this._options.minProtocolVersion ?? MIN_PROTOCOL_VERSION;
    if (
      this._serverProtocolVersion == null ||
      !isProtocolCompatible(this._serverProtocolVersion, minRequired)
    ) {
      this._transport = null;
      transport.close(1002, "incompatible protocol version");
      throw new IncompatibleServerError(
        this._serverProtocolVersion ?? "unknown",
        minRequired,
        this._serverVersion ?? undefined,
      );
    }

    // Send ClientConfigRequest (handshake completion from our side).
    if (this._options.clientConfig) {
      const cfg: ClientConfigRequest = {
        type: EventTypeValue.CLIENT_CONFIG,
        timestamp: new Date().toISOString(),
        ...this._options.clientConfig,
      } as unknown as ClientConfigRequest;
      transport.sendEvent(cfg as unknown as JaatoEvent);
    }

    this._reconnectAttempts = 0;
    this._transition(ConnectionState.CONNECTED, {
      serverVersion: this._serverVersion ?? undefined,
      clientId: this._clientId ?? undefined,
    });

    // Surface the inaugural ConnectedEvent to subscribers — handlers
    // registered before connect() rely on this to react to the very
    // first connection (parity with Python's IPCClient, which yields
    // ConnectedEvent through its events() loop).
    this._dispatchEvent(connected);

    // Buffer the post-handshake events that arrived between now and
    // when the event loop picks up.  The loop drains _bufferedEvents
    // first so handshake-phase events aren't dropped.
    void this._pumpFromIterator(iter);
  }

  private async _pumpFromIterator(iter: AsyncIterableIterator<JaatoEvent>): Promise<void> {
    try {
      while (true) {
        const next = await iter.next();
        if (next.done) {
          return;
        }
        this._dispatchEvent(next.value);
      }
    } catch (e) {
      // Iterator failed — close path will be triggered via onClose
      // observer; nothing to do here.
    }
  }

  private _startEventLoop(): void {
    if (this._eventLoopActive) return;
    this._eventLoopActive = true;
    // Drain anything buffered during reconnect intervals.
    while (this._bufferedEvents.length > 0) {
      const e = this._bufferedEvents.shift()!;
      this._dispatchEvent(e);
    }
  }

  private _dispatchEvent(event: JaatoEvent): void {
    // Track session_id from SessionInfoEvent so callers can read it.
    // Use a runtime check rather than narrow because the discriminated
    // union doesn't include SessionInfoEvent's session_id field on
    // every member.
    const maybeSession = (event as unknown as { session_id?: string }).session_id;
    if (maybeSession && event.type === EventTypeValue.SESSION_INFO) {
      this._sessionId = maybeSession;
    }

    // Snapshot before iterating so subscribe/unsubscribe calls made
    // inside a handler only take effect for the *next* event.
    const typedSnapshot = this._typedHandlers.get(event.type as EventType);
    const typedEntries = typedSnapshot ? typedSnapshot.slice() : [];
    const catchallEntries = this._catchallHandlers.slice();

    for (const entry of typedEntries) {
      if (entry.once) {
        this._removeTypedHandlerId(event.type as EventType, entry.id);
      }
      this._invokeHandler(entry.handler, event);
    }

    for (const entry of catchallEntries) {
      if (entry.once) {
        this._removeCatchallHandlerId(entry.id);
      }
      this._invokeHandler(entry.handler, event);
    }
  }

  private _invokeHandler(
    handler: HandlerEntry["handler"],
    event: JaatoEvent,
  ): void {
    let result: void | Promise<void>;
    try {
      result = handler(event);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[JaatoClient] subscriber threw:", err);
      return;
    }
    if (result && typeof (result as Promise<void>).then === "function") {
      (result as Promise<void>).catch((err: unknown) => {
        // eslint-disable-next-line no-console
        console.error("[JaatoClient] async subscriber rejected:", err);
      });
    }
  }

  private _handleClose(info: { code: number; reason: string }): void {
    this._transport = null;
    if (this._explicitClose || this._state === ConnectionState.CLOSED) {
      this._transition(ConnectionState.CLOSED, {
        reason: info.reason || `code ${info.code}`,
      });
      return;
    }
    if (!this._recovery.autoReconnect) {
      this._transition(ConnectionState.CLOSED, {
        reason: `connection lost (code ${info.code})`,
      });
      return;
    }
    this._scheduleReconnect();
  }

  private _scheduleReconnect(): void {
    this._reconnectAttempts += 1;
    if (
      this._recovery.maxReconnectAttempts != null &&
      this._reconnectAttempts > this._recovery.maxReconnectAttempts
    ) {
      this._transition(ConnectionState.CLOSED, {
        reason: `max reconnect attempts (${this._recovery.maxReconnectAttempts}) exceeded`,
      });
      return;
    }

    const baseDelay = Math.min(
      this._recovery.initialBackoffSeconds * Math.pow(2, this._reconnectAttempts - 1),
      this._recovery.maxBackoffSeconds,
    );
    const jitter = baseDelay * this._recovery.jitterFactor * (Math.random() * 2 - 1);
    const delaySeconds = Math.max(0, baseDelay + jitter);

    this._transition(ConnectionState.RECONNECTING, {
      reconnectAttempt: this._reconnectAttempts,
      reconnectDelaySeconds: delaySeconds,
    });

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._attemptReconnect().catch(() => {
        // Either rescheduled inside (next attempt) or transitioned
        // to CLOSED (max attempts hit) — nothing else to do here.
      });
    }, delaySeconds * 1000);
  }

  private async _attemptReconnect(): Promise<void> {
    try {
      await this._openOnce();
    } catch (e) {
      if (e instanceof IncompatibleServerError) {
        this._transition(ConnectionState.CLOSED, { reason: e.message });
        return;
      }
      // Schedule the next attempt with exponential backoff.
      this._scheduleReconnect();
    }
  }

  private _transition(next: ConnectionState, extra: Partial<ConnectionStatus> = {}): void {
    if (next === this._state && next !== ConnectionState.RECONNECTING) {
      return;
    }
    this._state = next;
    const status: ConnectionStatus = { state: next, ...extra };
    for (const h of this._statusHandlers) {
      try {
        h(status);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[JaatoClient] status handler threw:", err);
      }
    }
  }
}
