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

/**
 * The earliest jaato-server version this SDK is wire-compatible with.
 *
 * Bumped whenever the SDK starts depending on a new server feature
 * (e.g. a new WS verb).  Mirrors the TUI's MIN_SERVER_VERSION
 * convention — clients refuse to connect to older daemons rather
 * than send requests the server can't handle.
 */
export const MIN_SERVER_VERSION = "0.5.27";

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
   * Override the SDK's compile-time minimum.  Use only for
   * development; production deployments should leave this unset
   * so the SDK refuses to connect to incompatible servers.
   */
  minServerVersion?: string;
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
  private _clientId: string | null = null;
  private _sessionId: string | null = null;
  private _statusHandlers: Array<(s: ConnectionStatus) => void> = [];
  private _eventHandlers: Array<(e: JaatoEvent) => void> = [];
  private _bufferedEvents: JaatoEvent[] = [];
  private _eventLoopActive = false;
  private _reconnectAttempts = 0;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _explicitClose = false;

  constructor(options: JaatoClientOptions) {
    this._options = options;
    this._recovery = { ...DEFAULT_RECOVERY_CONFIG, ...(options.recovery ?? {}) };
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

  /** Server version reported in {@link ConnectedEvent}, after handshake. */
  get serverVersion(): string | null {
    return this._serverVersion;
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

  // ──── Event stream ───────────────────────────────────────────────

  /**
   * Subscribe a handler for every incoming event.
   *
   * Handlers are invoked synchronously on event arrival.  Throwing
   * inside a handler is logged and swallowed — it never breaks the
   * event loop or other subscribers.
   *
   * @returns Unsubscribe function.
   */
  subscribe(handler: (event: JaatoEvent) => void): () => void {
    this._eventHandlers.push(handler);
    return (): void => {
      const i = this._eventHandlers.indexOf(handler);
      if (i >= 0) this._eventHandlers.splice(i, 1);
    };
  }

  /**
   * Async iterator alternative to {@link subscribe}.
   *
   * Yields events as they arrive.  When the connection drops and
   * {@link RecoveryConfig.autoReconnect} is true, the iterator
   * suspends until the next CONNECTED state and then resumes.
   * When the connection enters CLOSED, the iterator returns.
   */
  async *events(): AsyncIterableIterator<JaatoEvent> {
    const queue: JaatoEvent[] = [];
    const waiters: Array<(e: JaatoEvent | null) => void> = [];
    let stop = false;

    const handler = (e: JaatoEvent): void => {
      if (waiters.length > 0) {
        const w = waiters.shift()!;
        w(e);
      } else {
        queue.push(e);
      }
    };
    const statusUnsub = this.onStatus((status) => {
      if (status.state === ConnectionState.CLOSED) {
        stop = true;
        while (waiters.length) {
          const w = waiters.shift()!;
          w(null);
        }
      }
    });
    const unsub = this.subscribe(handler);

    try {
      while (!stop) {
        if (queue.length > 0) {
          yield queue.shift()!;
          continue;
        }
        const next = await new Promise<JaatoEvent | null>((res) => {
          waiters.push(res);
        });
        if (next == null) {
          return;
        }
        yield next;
      }
    } finally {
      unsub();
      statusUnsub();
    }
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
    profile?: string;
    agent?: string;
    agentParams?: Record<string, string>;
  } = {}): Promise<void> {
    const args: string[] = options.name ? [options.name] : [];
    if (options.profile) {
      args.push("--profile", options.profile);
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

  async executeCommand(command: string, args?: string[]): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command,
      args: args ?? [],
    } as CommandRequest);
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

    // Version gate.
    const minRequired = this._options.minServerVersion ?? MIN_SERVER_VERSION;
    if (this._serverVersion != null && _compareVersions(this._serverVersion, minRequired) < 0) {
      this._transport = null;
      transport.close(1002, "incompatible server version");
      throw new IncompatibleServerError(this._serverVersion, minRequired);
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
    for (const h of this._eventHandlers) {
      try {
        h(event);
      } catch (err) {
        // Swallow — never let a subscriber crash the dispatcher.
        // eslint-disable-next-line no-console
        console.error("[JaatoClient] subscriber threw:", err);
      }
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
