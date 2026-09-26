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
  RequestInterruptedError,
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
// Runtime import is used only inside the static session() method body, so the
// client.ts <-> convenience.ts cycle resolves fine under ESM (no top-level use).
import { openSession, type Session, type SessionOpenOptions } from "./convenience.js";
import {
  EventTypeValue,
  type EventType,
  type ClientConfigRequest,
  type CommandRequest,
  type ConnectedEvent,
  type ExternalEventRequest,
  type JaatoEvent,
  type SendMessageRequest,
  type StopRequest,
  type PermissionResponseRequest,
  type PostAuthSetupResponse,
  type ClarificationResponseRequest,
  type ClarificationBatchResponseEvent,
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
  type WorkspaceFileContentEvent,
  type WorkspaceFileFetchRequest,
  type MemoryListEvent,
  type MemoryGetResultEvent,
  type MemoryUpdateResultEvent,
  type MemoryDeleteResultEvent,
  type DiagnosticsResultEvent,
  type WorkspaceInspectEvent,
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
 * Wire-protocol minor from which the two RESUME verbs (`injectPrompt` and
 * `session.wake`) carry `attachments`.
 *
 * Deliberately NOT the client's connect-time floor: a TS client that never
 * sends binary content works against any 1.x daemon, and raising
 * {@link MIN_PROTOCOL_VERSION} would refuse those connections for a field
 * they do not use.  It is checked per call instead, by the two methods that
 * would otherwise let a payload vanish.
 */
export const MIN_ATTACHMENT_RESUME_PROTOCOL = "1.5";

/**
 * Wire-protocol minor from which a clarification ANSWER carries
 * `answer_attachments` (#989).
 *
 * Per-call for the same reason as {@link MIN_ATTACHMENT_RESUME_PROTOCOL},
 * and refused for the same one: an older daemon ignores the field and
 * answers the clarification with the media gone, which for a voice-only
 * answer is a BLANK answer reported as a successful one — the server reads
 * an empty response as `free_text: ""` and the agent proceeds on nothing.
 */
export const MIN_CLARIFICATION_ATTACHMENT_PROTOCOL = "1.6";

/**
 * Protocol floor for {@link JaatoClient.reloadSessionEnv}.  A daemon that
 * does not know the verb ignores it silently, and "reloaded" would then be
 * reported about a session still running on its old credential -- so the
 * call is refused below this version rather than sent blind.
 */
export const MIN_SESSION_RELOAD_ENV_PROTOCOL = "1.11";

/**
 * Protocol floor for {@link JaatoClient.toggleWorkspaceIgnore}.  Same rule
 * as {@link MIN_SESSION_RELOAD_ENV_PROTOCOL}: an older daemon ignores the
 * verb, and a client that then reported the entry as ignored would be
 * describing a ``.gitignore`` nobody changed.
 */
export const MIN_WORKSPACE_IGNORE_PROTOCOL = "1.12";

/**
 * Protocol floor for {@link JaatoClient.fetchWorkspaceFile}.  A missing
 * VERB: an older daemon answers ``ErrorEvent("Unknown message type")`` and
 * never the ``workspace.file.content`` the call waits on, so the call is
 * refused below this version rather than left to time out.
 */
export const MIN_FILE_FETCH_PROTOCOL = "1.20";

/**
 * Protocol floor for {@link JaatoClient.runScaffoldIntegration}.  Same rule
 * as {@link MIN_WORKSPACE_IGNORE_PROTOCOL}: an older daemon ignores
 * ``scaffold.integration`` silently, and a client that then reported the
 * skill as installed would be describing a copy nobody wrote.  So the call
 * is refused below this version rather than sent blind.
 */
export const MIN_SCAFFOLD_INTEGRATION_PROTOCOL = "1.21";

/**
 * Protocol floor for the memory verbs ({@link JaatoClient.listMemories} and
 * friends, #1232).  A missing VERB: an older daemon answers ``ErrorEvent(
 * "Unknown request type")`` with no ``request_id`` and never the result the
 * call waits on, so every memory call is refused below this version rather
 * than left to time out.
 */
export const MIN_MEMORY_VERBS_PROTOCOL = "1.22";

/**
 * Protocol floor for {@link JaatoClient.sendSessionMessage}.  Same rule as
 * {@link MIN_WORKSPACE_IGNORE_PROTOCOL}: an older daemon ignores
 * ``session.message`` silently, and a client that then reported the message
 * as delivered would be describing one nobody carried.  So the call is
 * refused below this version rather than sent blind.
 */
export const MIN_SESSION_MESSAGE_PROTOCOL = "1.23";

/**
 * Protocol floor for a {@link JaatoClient.sendSessionMessage} that carries
 * `fileRefs` or `textAttachments` (1.24).  Two NEW keys on an existing
 * verb: a 1.23 daemon reads neither, delivers the text alone and answers
 * `accepted` — a degraded call that reads as success — so a call carrying
 * either is refused below this while a text-only call keeps the 1.23 floor.
 */
export const MIN_SESSION_MESSAGE_FILES_PROTOCOL = "1.24";

/**
 * Protocol floor for {@link JaatoClient.getDiagnostics} (#1294). A missing
 * VERB, the same shape as {@link MIN_MEMORY_VERBS_PROTOCOL}: an older
 * daemon answers ``ErrorEvent("Unknown request type")`` with no
 * ``request_id`` and never the result the call waits on, so the call is
 * refused below this version rather than left to time out.
 */
export const MIN_DIAGNOSTICS_PROTOCOL = "1.25";

/**
 * Protocol floor for the workspace/session pickers (1.26):
 * {@link JaatoClient.inspectWorkspace}, {@link JaatoClient.cloneIntoWorkspace},
 * ``WorkspaceDeleteRequest.stop_sessions`` and the ``model`` / ``provider``
 * override on {@link JaatoClient.createSession}.  Below it a new verb is
 * answered "Unknown message type" and never the result, and an older
 * ``session.new`` parser would read ``--model`` as the session NAME -- so
 * each is refused client-side rather than silently degraded.
 */
export const MIN_WORKSPACE_PICKER_PROTOCOL = "1.26";

/**
 * Size limits a daemon enforces, advertised in ``ConnectedEvent.server_info``.
 *
 * ``maxMessageSize`` is the largest single WebSocket message the daemon
 * accepts; a larger one makes it close the connection with 1009.  A staged
 * file travels as ONE binary message, so ``stagePerFileLimit`` is never
 * above it.
 */
export interface ServerLimits {
  maxMessageSize: number;
  stagePerFileLimit: number;
  stageTotalLimit: number;
  /** ``false`` when the daemon advertised nothing and these are the legacy values. */
  advertised: boolean;
}

/**
 * What a daemon that advertises no limits enforces: the ``websockets``
 * default of 1 MiB per message (the daemon never set one), which also caps
 * a staged file whatever the 10 MB staging cap says.
 */
export const LEGACY_SERVER_LIMITS: ServerLimits = {
  maxMessageSize: 1024 * 1024,
  stagePerFileLimit: 1024 * 1024,
  stageTotalLimit: 50 * 1024 * 1024,
  advertised: false,
};

/** Read {@link ServerLimits} out of ``server_info``; legacy values when absent. */
export function serverLimitsFrom(info: Record<string, unknown> | null | undefined): ServerLimits {
  const num = (v: unknown): number | null =>
    typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
  const max = num(info?.max_message_size);
  if (max === null) return { ...LEGACY_SERVER_LIMITS };
  return {
    maxMessageSize: max,
    stagePerFileLimit: Math.min(num(info?.stage_per_file_limit) ?? max, max),
    stageTotalLimit: num(info?.stage_total_limit) ?? LEGACY_SERVER_LIMITS.stageTotalLimit,
    advertised: true,
  };
}

/**
 * How long {@link JaatoClient.stageFiles} waits for the daemon's
 * ``workspace.files.staged`` response before giving up (default 120 s, the
 * sibling {@link JaatoClient.fetchWorkspaceFile} value).  The wait had no
 * deadline: if the response never arrived (lost, the daemon errored before
 * emitting it, the connection stalled without closing) the promise never
 * settled and the caller's ``staging`` indicator pulsed forever with no
 * error to react to.  The timeout rejects with a clear message so a caller's
 * ``catch`` marks the upload ``failed`` instead of hanging (#1248).
 */
export const STAGE_FILES_TIMEOUT_MS = 120_000;

/**
 * What {@link JaatoClient.fetchWorkspaceFile} resolves with.  ``event`` is
 * the daemon's header; ``data`` is the file's bytes when it was fetched
 * (``null`` for a metadata-only fetch or a refusal).
 */
export interface WorkspaceFileFetchResult {
  event: WorkspaceFileContentEvent;
  data: Uint8Array | null;
}

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
 * Supplies the credential for one connection attempt.
 *
 * Called by {@link JaatoClient} immediately before each WebSocket open,
 * never cached: the value it returns is presented on that attempt and
 * on no other.  May return the credential directly or as a promise.
 * Returning ``undefined`` connects with no token, the
 * ``--ws-unsafe-no-auth`` posture.
 */
export type TokenProvider = () => string | undefined | Promise<string | undefined>;

/**
 * Constructor options for {@link JaatoClient}.
 */
export interface JaatoClientOptions {
  /** Full ``ws://`` or ``wss://`` URL of the jaato daemon. */
  url: string;
  /**
   * Bearer token presented as ``?token=<token>`` query parameter.
   * Omit when the daemon is started with ``--ws-unsafe-no-auth``.
   *
   * A **string** is presented as-is on every connection attempt — right
   * for the daemon's shared token (``--ws-token-file``), which is valid
   * until the operator rotates it.
   *
   * A **function** (a {@link TokenProvider}) is called before *each*
   * attempt, the initial ``connect()`` and every automatic reconnect,
   * and its result is presented once.  This is the shape a per-user
   * ticket needs (protocol 1.10, #1074): a ticket is single-use and
   * consumed at accept, so replaying the value that opened the last
   * connection can never open the next one.  A browser client behind a
   * backend-for-frontend passes a provider that asks that backend for a
   * fresh ticket; see ``docs/design/web-server-bff.md``.
   *
   * A provider that throws fails that attempt only: on ``connect()`` the
   * error propagates to the caller; during reconnect the attempt is
   * counted and the next one is scheduled with the usual backoff, so a
   * backend that is briefly down degrades into the reconnect loop rather
   * than into a dead connection.
   */
  token?: string | TokenProvider;
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
  /** Monotonic part of each ``fetchWorkspaceFile`` request id. */
  private _fileFetchSeq = 0;
  private _memorySeq = 0;
  /**
   * Requests waiting for an answer on the CURRENT connection, told when it
   * closes so they reject at once instead of waiting out their deadline.
   * Cleared by each waiter as it settles.
   */
  private _closeWaiters: Set<(info: { code: number; reason: string }) => void> = new Set();
  private _limits: ServerLimits | null = null;
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

  /**
   * The size limits the connected daemon enforces, from
   * ``ConnectedEvent.server_info``.  Against a daemon that advertises none
   * (every release before it did) this is {@link LEGACY_SERVER_LIMITS}:
   * those daemons closed the connection on any message over 1 MiB whatever
   * their staging caps said.  ``null`` before the first handshake.
   */
  get serverLimits(): ServerLimits | null {
    return this._limits;
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

  /**
   * Answer the daemon's post-auth setup offer (``auth.setup``).
   *
   * After a daemon-level auth command succeeds with no session open
   * (``anthropic-auth login`` and friends), the daemon offers to create
   * one: pick a model, optionally persist ``JAATO_PROVIDER`` /
   * ``MODEL_NAME`` to the workspace ``.env``.  ``connect: false``
   * declines.  Mirrors the Python SDK's ``respond_to_post_auth_setup``.
   */
  async respondToPostAuthSetup(
    requestId: string,
    options: { connect: boolean; modelName?: string; persistEnv?: boolean },
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.POST_AUTH_SETUP_RESPONSE,
      request_id: requestId,
      connect: options.connect,
      model_name: options.modelName ?? "",
      persist_env: options.persistEnv ?? false,
    } as PostAuthSetupResponse);
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

  /**
   * Answer a batched clarification — every answer at once.
   *
   * Required for a ``ClarificationBatchEvent`` carrying
   * ``batch_only``: the server relays the whole request in that one
   * event and nothing else follows it, so this is the only reply that
   * unblocks the tool call and the turn behind it.  A client that
   * ignores such an event hangs the session (#704).
   *
   * Mirror of Python ``IPCClient.respond_to_clarification_batch``.
   *
   * @param requestId The ``request_id`` from the ``ClarificationBatchEvent``.
   * @param answers Ordered answers, one per question by index.
   * @param cancelled Abandon the clarification instead of answering it:
   *   the tool returns ``{cancelled: true}`` to the model and the turn
   *   continues.  ``answers`` is ignored.  This is the way out of a
   *   question the user cannot or will not answer.
   * @param answerAttachments Media attached to individual answers
   *   (protocol 1.6+, #989), keyed by 1-based question index as a string:
   *   `{"1": [{mime_type, data, display_name}]}`.  `data` must already be
   *   base64 — this SDK does no file reading, exactly as `sendMessage`
   *   and `wakeSession` do none.  A voice note answering a `free_text`
   *   question is the motivating case, and the matching entry in
   *   `answers` is then legitimately `""`: the utterance IS the answer,
   *   and the daemon does not read it as a skip.  Equally valid on a
   *   CHOICE answer — the ordinal says which branch, the attachment says
   *   what content.  Refused (not degraded) below protocol 1.6.
   */
  async respondToClarificationBatch(
    requestId: string,
    answers: string[],
    cancelled = false,
    answerAttachments?: Record<string, Array<Record<string, unknown>>>,
  ): Promise<void> {
    const hasMedia =
      !cancelled &&
      answerAttachments !== undefined &&
      Object.keys(answerAttachments).length > 0;
    if (hasMedia) {
      this._requireClarificationAttachmentProtocol();
    }
    await this._sendEvent({
      type: EventTypeValue.CLARIFICATION_BATCH_RESPONSE,
      request_id: requestId,
      answers,
      cancelled,
      answer_attachments: hasMedia ? answerAttachments : {},
    } as ClarificationBatchResponseEvent);
  }

  /**
   * Refuse answer attachments against a daemon too old to carry them.
   *
   * Sibling of {@link _requireAttachmentResumeProtocol}, and the same
   * argument: the degraded call does not mean what the caller asked for.
   * Here it answers the agent's question with the recording thrown away —
   * and the agent acts on it, because a clarification answer reaches the
   * model as a tool result it reads as fact.
   */
  private _requireClarificationAttachmentProtocol(): void {
    if (
      this._serverProtocolVersion !== null &&
      isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_CLARIFICATION_ATTACHMENT_PROTOCOL,
      )
    ) {
      return;
    }
    throw new Error(
      `respondToClarificationBatch: this daemon speaks protocol ` +
        `${this._serverProtocolVersion ?? "unknown"} and would DROP the ` +
        `answer attachments (needs >= ` +
        `${MIN_CLARIFICATION_ATTACHMENT_PROTOCOL}).  Answer the ` +
        `clarification in text, or upgrade the daemon.`,
    );
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
     * ``system_instructions``, ``gc``, ``env``,
     * ``runtime_limits``, ``model_tiers``, ``completion_payload_schema``.
     * The two forms are mutually exclusive — pass one or the other.
     * The server validates the dict and rejects it with a clear
     * ``ErrorEvent`` if ``model`` is missing.
     */
    profile?: string | Record<string, unknown>;
    agent?: string;
    agentParams?: Record<string, string>;
    /**
     * Phase 2 cascade-sharing tenant id (server 0.6.144+).  Sessions
     * sharing the same id reuse the same pre-warm runner slot across
     * cascade stages.  Sent as ``--cascade-driver-id`` on ``session.new``,
     * mirroring Python ``IPCClient.create_session(cascade_driver_id=...)``.
     */
    cascadeDriverId?: string;
    /**
     * Override the model the resolved profile (or, with no profile, the
     * workspace ``.env``) binds.  Sent as ``--model`` on ``session.new``
     * (protocol 1.26).  The daemon applies it to a COPY of the profile, and
     * a revived session keeps it.
     */
    model?: string;
    /**
     * Override the provider too (``--provider``).  Only with ``model``:
     * passing it alone throws here, and the daemon refuses it as well.
     */
    provider?: string;
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
    if (options.cascadeDriverId) {
      args.push("--cascade-driver-id", options.cascadeDriverId);
    }
    if (options.provider && !options.model) {
      throw new TypeError("createSession: 'provider' requires 'model'");
    }
    if (
      options.model &&
      (this._serverProtocolVersion === null ||
        !isProtocolCompatible(this._serverProtocolVersion, MIN_WORKSPACE_PICKER_PROTOCOL))
    ) {
      throw new Error(
        `createSession: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and would read --model ` +
          `as the session name (needs >= ${MIN_WORKSPACE_PICKER_PROTOCOL}).`,
      );
    }
    if (options.model) {
      args.push("--model", options.model);
    }
    if (options.provider) {
      args.push("--provider", options.provider);
    }
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.new",
      args,
      payload,
    } as CommandRequest);
  }

  /**
   * Open a session with the high-level convenience facade.
   *
   * Connects, registers any host tools, and creates the session, then
   * returns a {@link Session} that bundles the send-and-wait recipe so the
   * common path never reproduces the ``SESSION_TERMINATED``-only hang.
   * ``Session`` is an ``AsyncDisposable`` — use ``await using`` for
   * automatic teardown, or call ``await s.close()`` explicitly:
   *
   * ```ts
   * await using s = await JaatoClient.session({
   *   url: "wss://host:8089", profile: "researcher",
   * });
   * console.log(await s.ask("Who are you?"));
   * ```
   *
   * Additive — every low-level method stays; reach the underlying client
   * via {@link Session.client}.  See ``convenience.ts``.
   */
  static session(options: SessionOpenOptions): Promise<Session> {
    return openSession(options);
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
   * Re-read a live session's ``.env`` and credentials and rebuild its
   * provider.
   *
   * A session resolves its environment and its provider credential once,
   * when its runner boots; a key stored with ``<provider>-auth key`` or a
   * ``.env`` line written afterwards never reaches the open session.  This
   * sends ``session.reload_env``: the daemon re-resolves and the runner
   * re-applies the whole environment and re-creates the provider, so the
   * next turn runs on the credential now on disk.  Refused by the daemon,
   * with nothing changed, while a turn is running.  The daemon confirms
   * with a ``system.message`` naming the outcome and the credential source
   * the rebuilt provider resolved.  Mirror of Python
   * ``IPCClient.reload_session_env``.
   *
   * @param sessionId The session to reload; omit for the one this client
   *   is attached to.
   * @throws Error against a daemon below {@link MIN_SESSION_RELOAD_ENV_PROTOCOL}.
   */
  async reloadSessionEnv(sessionId?: string): Promise<void> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_SESSION_RELOAD_ENV_PROTOCOL,
      )
    ) {
      throw new Error(
        `reloadSessionEnv: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `session.reload_env (needs >= ${MIN_SESSION_RELOAD_ENV_PROTOCOL}).  ` +
          `It would ignore the command silently.  Upgrade the daemon, or ` +
          `start a new session to pick up the credential.`,
      );
    }
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "session.reload_env",
      args: sessionId ? [sessionId] : [],
    } as CommandRequest);
  }

  /**
   * Add an entry to the session workspace's ``.gitignore``, or remove it
   * again — the TUI workspace panel's ``i`` key, served daemon-side
   * (protocol 1.12) so a browser client can make the same edit.  Exact-match
   * toggle of ONE line: a directory entry keeps its trailing ``/``.  The
   * daemon answers with one ``workspace.ignore.result`` whatever happened —
   * ``ok`` / ``ignored`` on success, ``ok: false`` with the reason otherwise.
   * Mirror of Python ``IPCClient.toggle_workspace_ignore``.
   *
   * @param path The workspace-relative entry, as the workspace panel shows it.
   * @throws Error against a daemon below {@link MIN_WORKSPACE_IGNORE_PROTOCOL}.
   */
  async toggleWorkspaceIgnore(path: string): Promise<void> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_WORKSPACE_IGNORE_PROTOCOL,
      )
    ) {
      throw new Error(
        `toggleWorkspaceIgnore: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `workspace.ignore (needs >= ${MIN_WORKSPACE_IGNORE_PROTOCOL}).  ` +
          `It would ignore the command silently.  Upgrade the daemon, or ` +
          `edit the workspace's .gitignore directly.`,
      );
    }
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "workspace.ignore",
      args: [path],
    } as CommandRequest);
  }

  /**
   * Message another session in this session's GROUP, waking it if it is
   * cold (protocol 1.22).  The client-tier form of the ``courier`` plugin's
   * ``send_to_session``: the sender is this connection's OWN session,
   * resolved daemon-side, and the target must share a group with it — a
   * cascade, or the same authenticated creator.  A cold target is revived
   * from disk and driven; a busy one queues the message on the idle-only
   * peer tier; a terminated one is never woken.  Mirror of Python
   * ``IPCClient.send_session_message``.
   *
   * Fire-and-forget: the daemon answers with one ``session.message.result``
   * event carrying the receipt — ``status`` is ``accepted`` / ``queued`` /
   * ``spooled`` (delivered, or held in the target's durable inbox),
   * ``no_such_session``, ``ambiguous`` (with ``candidates``),
   * ``session_cold``, ``duplicate``, ``terminated`` or ``refused`` (with
   * ``error``).  No delivered status claims the peer read or acted on
   * anything.
   *
   * @param target A session id, or a cascade-scoped sibling name.
   * @param text The message; may be empty when attachments, file
   *   references or text attachments carry the content.
   * @param options.attachments Binary content in the canonical wire shape
   *   (``{mime_type, data, display_name}``), delivered on the drive branch
   *   only — a busy target's message is spooled rather than stripped.
   * @param options.fileRefs Files in the SENDER session's workspace to hand
   *   the target (1.24): workspace-relative paths, or ``{path, workspace?}``
   *   rows.  References the DAEMON resolves on its own host — never bytes
   *   read here.  A target sharing the workspace reads the file in place;
   *   one in another workspace gets a copy under its own inbox (10 MB per
   *   file, 50 MB per message).  The result event's ``files`` says per file
   *   what became of it; one refused file refuses the whole message.
   * @param options.textAttachments ``{text, display_name?, mime_type?}``
   *   rows (1.24) — a patch, a snippet — inlined for the target up to 32 KiB
   *   in total, stored as files beyond it.
   * @param options.eventId Idempotency key; a redelivered id answers
   *   ``duplicate``, a benign no-op.
   * @param options.requestId Correlation id echoed on the result event.
   * @throws Error against a daemon below {@link MIN_SESSION_MESSAGE_PROTOCOL};
   *   against one below {@link MIN_SESSION_MESSAGE_FILES_PROTOCOL} when
   *   ``fileRefs`` or ``textAttachments`` are given (it would deliver the
   *   text without them and call that accepted); or when no content at all
   *   is given.
   */
  async sendSessionMessage(
    target: string,
    text = "",
    options?: {
      attachments?: Array<Record<string, unknown>>;
      fileRefs?: Array<string | Record<string, unknown>>;
      textAttachments?: Array<Record<string, unknown>>;
      eventId?: string;
      requestId?: string;
    },
  ): Promise<void> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_SESSION_MESSAGE_PROTOCOL,
      )
    ) {
      throw new Error(
        `sendSessionMessage: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `session.message (needs >= ${MIN_SESSION_MESSAGE_PROTOCOL}).  ` +
          `It would ignore the command silently.  Upgrade the daemon.`,
      );
    }
    const fileRefs = options?.fileRefs ?? [];
    const textAttachments = options?.textAttachments ?? [];
    if (
      (fileRefs.length > 0 || textAttachments.length > 0) &&
      !isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_SESSION_MESSAGE_FILES_PROTOCOL,
      )
    ) {
      throw new Error(
        `sendSessionMessage: this daemon speaks protocol ` +
          `${this._serverProtocolVersion} and does not carry fileRefs / ` +
          `textAttachments on session.message (needs >= ` +
          `${MIN_SESSION_MESSAGE_FILES_PROTOCOL}).  It would deliver the ` +
          `text without them and report accepted.  Upgrade the daemon, or ` +
          `send text only.`,
      );
    }
    const attachments = options?.attachments ?? [];
    if (
      !text &&
      attachments.length === 0 &&
      fileRefs.length === 0 &&
      textAttachments.length === 0
    ) {
      throw new Error(
        "sendSessionMessage requires text, attachments, fileRefs or " +
          "textAttachments — a message with no content drives a turn the " +
          "peer has nothing to answer",
      );
    }
    const payload: Record<string, unknown> = { target, text };
    if (options?.eventId !== undefined) payload.event_id = options.eventId;
    if (attachments.length > 0) payload.attachments = attachments;
    if (fileRefs.length > 0) payload.file_refs = fileRefs;
    if (textAttachments.length > 0) payload.text_attachments = textAttachments;
    // CommandRequest carries no request_id of its own, so the correlation
    // id rides the payload and the daemon echoes it from there.
    if (options?.requestId !== undefined) payload.request_id = options.requestId;
    await this.executeCommand("session.message", [], payload);
  }

  /**
   * Run ``jaato-scaffold integration <name>`` on the daemon (protocol 1.21)
   * — install or refresh an integration payload, the ``jaato-sdk`` skill
   * among them, into the workspace this connection is in.  The sibling of
   * ``scaffold.explain``: the command must run on the install that serves
   * the session, because the copy it writes is stamped with THAT
   * ``jaato-server``'s version and the workspace directory is on THAT host —
   * so the daemon runs it rather than the caller shelling out to its own
   * venv, and the application never carries (and so never drifts) a copy of
   * the skill's text.
   *
   * The daemon keeps the copy current with the ``--refresh`` contract: it
   * re-applies an ``absent`` / ``stale`` / ``outdated`` copy and LEAVES an
   * ``edited`` / ``diverged`` / ``unstamped`` one alone.  It answers with one
   * ``scaffold.integration.result`` event whatever happened — ``ok`` with
   * ``changed`` / ``state_before`` / ``state_after`` / ``skipped_reason``, or
   * ``ok: false`` with the reason and the ``available`` integrations it
   * ships.  A refresh it declined to apply is ``ok: true`` with a
   * ``skipped_reason``, so a client reports it in a notice, not as an error.
   * Mirror of Python ``IPCClient.run_integration``.
   *
   * The integration installs into the caller's OWN workspace, resolved
   * daemon-side (the same entitlement path ``workspace.file.fetch`` uses), so
   * there is no directory parameter.
   *
   * @param name The integration to run, e.g. ``"claude-code"``.
   * @throws Error against a daemon below
   *   {@link MIN_SCAFFOLD_INTEGRATION_PROTOCOL}, which would ignore the
   *   command silently — indistinguishable from the skill having been
   *   installed, which a caller must not report.
   */
  async runScaffoldIntegration(name: string): Promise<void> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_SCAFFOLD_INTEGRATION_PROTOCOL,
      )
    ) {
      throw new Error(
        `runScaffoldIntegration: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `scaffold.integration (needs >= ${MIN_SCAFFOLD_INTEGRATION_PROTOCOL}).  ` +
          `It would ignore the command silently, which is indistinguishable ` +
          `from the skill having been installed.  Upgrade the daemon, or run ` +
          `jaato-scaffold integration in the daemon's own virtualenv.`,
      );
    }
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command: "scaffold.integration",
      args: [name],
    } as CommandRequest);
  }

  /**
   * Download one file from the workspace this connection is in (protocol
   * 1.20, WS only) -- the reverse of {@link stageFiles}.
   *
   * ``path`` is workspace-relative (or absolute inside the workspace).  The
   * daemon answers with a ``workspace.file.content`` header and, on success,
   * one binary frame the transport attaches to it, so the result carries
   * the bytes.  A refusal is NOT thrown: it resolves with ``event.ok ===
   * false`` and ``event.category`` naming why (``not_found``,
   * ``unsafe_path``, ``credential``, ``too_large``, ...), because a caller
   * renders those differently.
   *
   * Calls are correlated by ``request_id``, so several may be in flight.
   *
   * @param options.metadataOnly Ask whether the file exists, its size and
   *   type, without transferring it.
   * @param options.timeoutMs Give up after this long (default 120 s).
   * @throws Error against a daemon below {@link MIN_FILE_FETCH_PROTOCOL},
   *   or on timeout.
   */
  async fetchWorkspaceFile(
    path: string,
    options: { metadataOnly?: boolean; timeoutMs?: number } = {},
  ): Promise<WorkspaceFileFetchResult> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(this._serverProtocolVersion, MIN_FILE_FETCH_PROTOCOL)
    ) {
      throw new Error(
        `fetchWorkspaceFile: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `workspace.file.fetch (needs >= ${MIN_FILE_FETCH_PROTOCOL}).  ` +
          `Upgrade the daemon to download workspace files.`,
      );
    }
    const requestId = `dl-${++this._fileFetchSeq}-${Date.now().toString(36)}`;
    const timeoutMs = options.timeoutMs ?? 120_000;
    const answer = new Promise<WorkspaceFileFetchResult>((resolve, reject) => {
      const done = () => {
        clearTimeout(timer);
        unsub();
        this._closeWaiters.delete(onClose);
      };
      const timer = setTimeout(() => {
        done();
        reject(new Error(`fetchWorkspaceFile: no answer for ${path} after ${timeoutMs} ms`));
      }, timeoutMs);
      const onClose = (info: { code: number; reason: string }) => {
        done();
        reject(new RequestInterruptedError(`fetchWorkspaceFile ${path}`, info.code, info.reason));
      };
      this._closeWaiters.add(onClose);
      const unsub = this.subscribeAll((raw) => {
        const event = raw as WorkspaceFileContentEvent & { data?: Uint8Array };
        if (event.type !== EventTypeValue.WORKSPACE_FILE_CONTENT) return;
        if (event.request_id !== requestId) return;
        done();
        resolve({ event, data: event.data ?? null });
      });
    });
    await this._sendEvent({
      type: EventTypeValue.WORKSPACE_FILE_FETCH_REQUEST,
      request_id: requestId,
      path,
      metadata_only: options.metadataOnly ?? false,
    } as WorkspaceFileFetchRequest);
    return answer;
  }

  /**
   * Send one quiet request/result-pair request and resolve with its
   * correlated answer (#1232's ``_memoryRequest``, generalised for #1294's
   * diagnostics verb so a second copy of this plumbing does not drift from
   * the first).
   *
   * Refuses a daemon below ``minProtocol``; rejects on timeout and on a
   * closed connection -- never resolves with a fabricated empty answer,
   * which for a list would read as "nothing remembered". The answer is
   * matched on ``request_id`` AND on its result type, so a daemon echo of
   * the request itself is not mistaken for the answer.
   */
  private async _quietRequest<T>(
    method: string,
    request: Record<string, unknown>,
    resultType: string,
    timeoutMs: number,
    minProtocol: string,
    featureLabel: string,
    idPrefix: string,
  ): Promise<T> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(this._serverProtocolVersion, minProtocol)
    ) {
      throw new Error(
        `${method}: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `${featureLabel} (needs >= ${minProtocol}).`,
      );
    }
    const requestId = `${idPrefix}-${++this._memorySeq}-${Date.now().toString(36)}`;
    const answer = new Promise<T>((resolve, reject) => {
      const done = () => {
        clearTimeout(timer);
        unsub();
        this._closeWaiters.delete(onClose);
      };
      const timer = setTimeout(() => {
        done();
        reject(new Error(`${method}: no answer after ${timeoutMs} ms`));
      }, timeoutMs);
      const onClose = (info: { code: number; reason: string }) => {
        done();
        reject(new RequestInterruptedError(method, info.code, info.reason));
      };
      this._closeWaiters.add(onClose);
      const unsub = this.subscribeAll((raw) => {
        const event = raw as { type?: string; request_id?: string };
        if (event.type !== resultType) return;
        if (event.request_id !== requestId) return;
        done();
        resolve(raw as unknown as T);
      });
    });
    await this._sendEvent({ ...request, request_id: requestId } as unknown as JaatoEvent);
    return answer;
  }

  /**
   * Inspect a workspace (protocol 1.26, WS only): path, size, session counts
   * by state (``total`` / ``waiting`` / ``awake`` / ``sleeping``) and, per
   * git checkout, uncommitted / unpushed counts.  ``ok === false`` (with
   * ``error``) when the daemon refused -- another user's workspace, a name
   * that left the root, or no such workspace.
   */
  async inspectWorkspace(
    name: string,
    options: { timeoutMs?: number } = {},
  ): Promise<WorkspaceInspectEvent> {
    return this._quietRequest<WorkspaceInspectEvent>(
      "inspectWorkspace",
      { type: EventTypeValue.WORKSPACE_INSPECT_REQUEST, name },
      EventTypeValue.WORKSPACE_INSPECTED,
      options.timeoutMs ?? 30_000,
      MIN_WORKSPACE_PICKER_PROTOCOL,
      "workspace.inspect",
      "wsi",
    );
  }

  /**
   * Clone repositories into a workspace (protocol 1.26, WS only).
   *
   * Sends one ``workspace.clone`` and returns its ``request_id``; progress
   * arrives as ``workspace.clone_progress`` events carrying that id (every
   * repo ``queued`` first, then one at a time to ``done`` / ``failed``,
   * with ``done === total`` on the last).  Subscribe BEFORE calling.  A
   * retry is a new call naming the one repo.
   */
  async cloneIntoWorkspace(
    name: string,
    repos: Array<{ repo: string; branch: string; forge?: string }>,
  ): Promise<string> {
    if (
      this._serverProtocolVersion === null ||
      !isProtocolCompatible(this._serverProtocolVersion, MIN_WORKSPACE_PICKER_PROTOCOL)
    ) {
      throw new Error(
        `cloneIntoWorkspace: this daemon speaks protocol ` +
          `${this._serverProtocolVersion ?? "unknown"} and does not serve ` +
          `workspace.clone (needs >= ${MIN_WORKSPACE_PICKER_PROTOCOL}).`,
      );
    }
    const requestId = `wsc-${++this._memorySeq}-${Date.now().toString(36)}`;
    await this._sendEvent({
      type: EventTypeValue.WORKSPACE_CLONE_REQUEST,
      name,
      request_id: requestId,
      repos: repos.map((r) => ({
        repo: r.repo,
        branch: r.branch,
        forge: r.forge ?? "github",
      })),
    } as unknown as JaatoEvent);
    return requestId;
  }

  /**
   * List the attached session's memory store, quietly (protocol 1.22).
   *
   * Unlike the ``memory list`` command this writes nothing to the
   * transcript.  ``ok === false`` (with ``error`` and ``category``) means
   * the store could not be read, never "nothing remembered".  Rows carry
   * ``tier`` (``workspace`` / ``global``), timestamps, usage, both
   * provenance stamps and the two this-session flags -- not the content;
   * see {@link getMemory}.  ``may_curate`` says whether this connection may
   * change them.
   */
  async listMemories(options: { timeoutMs?: number } = {}): Promise<MemoryListEvent> {
    return this._quietRequest<MemoryListEvent>(
      "listMemories",
      { type: EventTypeValue.MEMORY_LIST_REQUEST },
      EventTypeValue.MEMORY_LIST,
      options.timeoutMs ?? 10_000,
      MIN_MEMORY_VERBS_PROTOCOL,
      'the memory verbs (upgrade the daemon, or use the "memory" command)',
      "mem",
    );
  }

  /** Fetch one memory WITH its content and evidence (protocol 1.22). */
  async getMemory(
    memoryId: string,
    options: { timeoutMs?: number } = {},
  ): Promise<MemoryGetResultEvent> {
    return this._quietRequest<MemoryGetResultEvent>(
      "getMemory",
      { type: EventTypeValue.MEMORY_GET_REQUEST, memory_id: memoryId },
      EventTypeValue.MEMORY_GET_RESULT,
      options.timeoutMs ?? 10_000,
      MIN_MEMORY_VERBS_PROTOCOL,
      'the memory verbs (upgrade the daemon, or use the "memory" command)',
      "mem",
    );
  }

  /**
   * Edit a memory, or move its maturity (protocol 1.22).
   *
   * The structured replacement for ``memory edit`` (which opens
   * ``$EDITOR`` on the runner's host).  An omitted field is left alone.  A
   * maturity change is recorded in ``curated_by`` with this connection's
   * authenticated identity.  Limited to the workspace owner: a refusal
   * resolves with ``ok === false, category === "not_owner"``.
   */
  async updateMemory(
    memoryId: string,
    fields: { description?: string; content?: string; tags?: string[]; maturity?: string },
    options: { timeoutMs?: number } = {},
  ): Promise<MemoryUpdateResultEvent> {
    const request: Record<string, unknown> = {
      type: EventTypeValue.MEMORY_UPDATE_REQUEST,
      memory_id: memoryId,
    };
    for (const key of ["description", "content", "tags", "maturity"] as const) {
      if (fields[key] !== undefined) request[key] = fields[key];
    }
    return this._quietRequest<MemoryUpdateResultEvent>(
      "updateMemory",
      request,
      EventTypeValue.MEMORY_UPDATE_RESULT,
      options.timeoutMs ?? 10_000,
      MIN_MEMORY_VERBS_PROTOCOL,
      'the memory verbs (upgrade the daemon, or use the "memory" command)',
      "mem",
    );
  }

  /** Approve a memory: ``updateMemory(id, { maturity: "validated" })``. */
  async approveMemory(memoryId: string, options: { timeoutMs?: number } = {}): Promise<MemoryUpdateResultEvent> {
    return this.updateMemory(memoryId, { maturity: "validated" }, options);
  }

  /**
   * Dismiss a memory: ``updateMemory(id, { maturity: "dismissed" })``.  The
   * store keeps no dismissed trace, so it is gone from the next list.
   */
  async dismissMemory(memoryId: string, options: { timeoutMs?: number } = {}): Promise<MemoryUpdateResultEvent> {
    return this.updateMemory(memoryId, { maturity: "dismissed" }, options);
  }

  /** Remove a memory through the plugin's own delete path (protocol 1.22). */
  async deleteMemory(
    memoryId: string,
    options: { timeoutMs?: number } = {},
  ): Promise<MemoryDeleteResultEvent> {
    return this._quietRequest<MemoryDeleteResultEvent>(
      "deleteMemory",
      { type: EventTypeValue.MEMORY_DELETE_REQUEST, memory_id: memoryId },
      EventTypeValue.MEMORY_DELETE_RESULT,
      options.timeoutMs ?? 10_000,
      MIN_MEMORY_VERBS_PROTOCOL,
      'the memory verbs (upgrade the daemon, or use the "memory" command)',
      "mem",
    );
  }

  /**
   * Self-diagnose the attached session's confinement and runtime facts
   * (#1294): cached record fields (``runner_identity``, ``confinement_id``,
   * ``sandbox_mode``, ``consumption``, ``notebook_boundary_kind``,
   * ``protocol_version``, ``server_version``) plus ``probe`` -- a LIVE
   * re-check of whether the runner's threads are genuinely confined right
   * now, measured fresh on every call and never merged into the cached
   * fields. Always about the caller's OWN attached session -- there is no
   * way to name a different one. ``ok === false`` (with ``error`` and
   * ``category``) means nothing could be reported at all (no session
   * attached, or the workspace owner gate refused the caller); even then
   * ``probe`` may independently be ``null`` while the cached fields are
   * populated, which is what a runner that did not answer looks like.
   */
  async getDiagnostics(options: { timeoutMs?: number } = {}): Promise<DiagnosticsResultEvent> {
    return this._quietRequest<DiagnosticsResultEvent>(
      "getDiagnostics",
      { type: EventTypeValue.DIAGNOSTICS_REQUEST },
      EventTypeValue.DIAGNOSTICS_RESULT,
      options.timeoutMs ?? 10_000,
      MIN_DIAGNOSTICS_PROTOCOL,
      "the diagnostics verb",
      "diag",
    );
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

  /**
   * Execute a daemon command verb.
   *
   * `payload` is the structured body for verbs that take one
   * (`CommandRequest.payload`) — used where a dict is the natural shape and
   * squeezing it into positional `args` would be lossy, e.g.
   * `cascade.budget.set`, or `session.wake` carrying attachments.  Omitted
   * it is `null`, exactly as before.
   */
  async executeCommand(
    command: string,
    args?: string[],
    payload?: Record<string, unknown>,
  ): Promise<void> {
    await this._sendEvent({
      type: EventTypeValue.COMMAND,
      command,
      args: args ?? [],
      payload: payload ?? null,
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

  /**
   * Inject a prompt into the session's message queue.
   *
   * `attachments` (protocol 1.5+) carries binary user content in the same
   * canonical wire shape `sendMessage` accepts —
   * `{mime_type, data: base64-string, display_name}`.  Note that an
   * attachment-bearing inject is IDLE-ONLY: the queued path folds a message
   * into the running turn as text and has nowhere to put binary content, so
   * the daemon offers it with `require_idle` and a busy target answers
   * `"busy"` with nothing enqueued.  Retry when the target goes idle rather
   * than assuming delivery.
   */
  async injectPrompt(
    text: string,
    sourceType = "user",
    sourceId?: string,
    attachments?: Array<Record<string, unknown>>,
  ): Promise<void> {
    if (attachments && attachments.length > 0) {
      this._requireAttachmentResumeProtocol("injectPrompt");
    }
    await this._sendEvent({
      type: EventTypeValue.INJECT_PROMPT_REQUEST,
      text,
      source_type: sourceType,
      source_id: sourceId ?? null,
      attachments: attachments ?? [],
    } as InjectPromptRequest);
  }

  /**
   * Wake a session by id — revive it if cold and start a USER turn on it.
   *
   * The typed form of `executeCommand("session.wake", [], payload)`.  Like
   * the command it wraps this is fire-and-forget: a refusal arrives on the
   * event stream as an `ErrorEvent` with `error_type: "WakeError"`.
   *
   * `attachments` (protocol 1.5+) carries binary content in the same shape
   * `sendMessage` accepts, already base64-encoded — this SDK does no file
   * reading.  `text` may be empty when the attachments ARE the message (a
   * spoken utterance), which is why the two are checked together.  The
   * daemon wraps the payload in its untrusted-content boundary and names
   * each attachment inside it, so media arriving this way is data the model
   * interprets, never instructions it follows.
   */
  async wakeSession(
    sessionId: string,
    text = "",
    options?: {
      attachments?: Array<Record<string, unknown>>;
      source?: string;
      eventId?: string;
    },
  ): Promise<void> {
    const attachments = options?.attachments ?? [];
    if (!text && attachments.length === 0) {
      throw new Error(
        "wakeSession requires text or attachments — a wake with no content " +
          "drives a turn the model has nothing to answer",
      );
    }
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      text,
      source: options?.source ?? "user",
    };
    if (options?.eventId !== undefined) payload.event_id = options.eventId;
    if (attachments.length > 0) {
      this._requireAttachmentResumeProtocol("wakeSession");
      payload.attachments = attachments;
    }
    await this.executeCommand("session.wake", [], payload);
  }

  /**
   * Publish an external event onto the session's `EventBus`.
   *
   * The host's way of telling a running session that something happened
   * outside it — `order.placed`, `build.finished`, `ticket.assigned`.  It
   * reaches every agent that called
   * `subscribeToEvents(event_types: ['external_event'])`, and sinks onward to
   * the daemon-wide reactor bus, so it is the one verb that can trigger a
   * reactor from a client.  Mirror of Python
   * ``IPCClient.send_external_event``.
   *
   * `ExternalEventRequest` has existed as a TYPE in both SDKs and as a METHOD
   * in neither (#1167): the only producer was an out-of-tree web component
   * hand-rolling the JSON frame.
   *
   * **Not** {@link wakeSession}.  A wake drives a USER turn on one session;
   * this publishes a bus event and drives no turn of its own.  A session with
   * no `external_event` subscriber receives it and does nothing, which is a
   * success and the normal state before any agent has subscribed.
   *
   * Fire-and-forget.  A refusal arrives on the event stream as an
   * `ErrorEvent` — `error_type: "ExternalEventError"` when the session has no
   * bus, `"SessionError"` when the session is gone, and `"RequestError"`
   * (`Unknown request type: ExternalEventRequest`) from a daemon predating
   * #1167 on IPC.  That last one is why this method takes no protocol floor:
   * the refusal is a named error on the stream rather than the silence an
   * unknown command verb produces, and a floor would ALSO refuse against the
   * WebSocket daemons where the request has always worked.
   *
   * @param name The event name the host chose, e.g. `order.placed`.  This is
   *   what an agent's `subscribeToEvents(event_names)` filter matches, so it
   *   must agree with what the persona asked to hear.
   * @param data Arbitrary JSON-serialisable payload; omitted sends `{}`.
   * @param options.timestamp ISO 8601, when the thing being reported
   *   happened.  Omitted lets the daemon stamp arrival time.
   * @param options.sessionId The target session; omitted means the one this
   *   client is attached to.
   * @throws Error if `name` is empty — an unnamed event matches no
   *   subscriber filter and reaches the model with nothing to say what
   *   happened.
   */
  async sendExternalEvent(
    name: string,
    data?: Record<string, unknown>,
    options?: { timestamp?: string; sessionId?: string },
  ): Promise<void> {
    if (!name) {
      throw new Error(
        "sendExternalEvent requires a name — an unnamed event matches no " +
          "subscribeToEvents filter and reaches the model with nothing to " +
          "say what happened",
      );
    }
    await this._sendEvent({
      type: EventTypeValue.EVENT_EXTERNAL,
      name,
      data: data ?? {},
      timestamp: options?.timestamp ?? "",
      session_id: options?.sessionId ?? "",
    } as ExternalEventRequest);
  }

  /**
   * Refuse an attachment-bearing resume against a daemon too old to carry it.
   *
   * An additive optional field is normally safe to send blind: an older peer
   * ignores it and the call degrades to what it always did.  That reasoning
   * holds for a `request_id` and NOT for an attachment — the degraded call is
   * a turn driven with the text and WITHOUT the audio that was the whole
   * message, which for a blank-text utterance is an empty turn reported as a
   * success.  So this is checked, not hoped.
   */
  private _requireAttachmentResumeProtocol(verb: string): void {
    if (
      this._serverProtocolVersion !== null &&
      isProtocolCompatible(
        this._serverProtocolVersion,
        MIN_ATTACHMENT_RESUME_PROTOCOL,
      )
    ) {
      return;
    }
    throw new Error(
      `${verb}: this daemon speaks protocol ` +
        `${this._serverProtocolVersion ?? "unknown"} and would DROP the ` +
        `attachments (needs >= ${MIN_ATTACHMENT_RESUME_PROTOCOL}).  Send the ` +
        `content through sendMessage() on a live session, or upgrade the daemon.`,
    );
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
   * @param options.timeoutMs How long to wait for the daemon's response
   *   before rejecting (default {@link STAGE_FILES_TIMEOUT_MS}).  Pass 0 to
   *   wait indefinitely (the pre-#1248 behaviour).
   * @returns The server's ``StageFilesEvent`` reporting per-file
   *   success / failure.
   * @throws Error if no ``workspace.files.staged`` response arrives within
   *   the timeout — so a lost response is a settled rejection, not a hang.
   */
  async stageFiles(
    workspaceId: string,
    files: Array<{
      name: string;
      data: ArrayBuffer | Uint8Array;
      contentType?: string;
      mode?: number;
    }>,
    options: { timeoutMs?: number } = {},
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
    // server's response could race with handler installation.  The wait
    // is bounded (#1248): a response that never arrives would otherwise
    // leave this promise unsettled forever, and the caller's status stuck
    // on "staging".
    const responsePromise = this._waitForNextEvent<StageFilesEvent>(
      (e) => e.type === EventTypeValue.WORKSPACE_FILES_STAGED,
      {
        timeoutMs: options.timeoutMs ?? STAGE_FILES_TIMEOUT_MS,
        label: "stageFiles: no workspace.files.staged response",
        // The usual reason no response arrives: a file over the daemon's
        // message limit makes it close the connection (1009) mid-upload.
        closeLabel: "stageFiles",
      },
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
   *
   * With ``options.timeoutMs`` set (> 0), the wait is bounded: if no
   * matching event arrives in time it **rejects** with an ``Error`` naming
   * ``options.label`` and the deadline, and unsubscribes — so a request
   * whose response is lost surfaces as a settled promise the caller can
   * catch, not a hang (#1248).  Omit ``timeoutMs`` (or pass 0) to keep the
   * resolve-only behaviour.
   *
   * With ``options.closeLabel`` set, the wait also rejects -- at once, with
   * a {@link RequestInterruptedError} carrying the close code -- when the
   * connection closes first.  For a request the daemon answers on the same
   * connection, a close means no answer is coming.
   */
  private _waitForNextEvent<T extends JaatoEvent = JaatoEvent>(
    predicate: (event: JaatoEvent) => boolean,
    options: { timeoutMs?: number; label?: string; closeLabel?: string } = {},
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | null = null;
      let onClose: ((info: { code: number; reason: string }) => void) | null = null;
      const done = () => {
        if (timer) clearTimeout(timer);
        unsub();
        if (onClose) this._closeWaiters.delete(onClose);
      };
      const unsub = this.subscribeAll((event) => {
        if (predicate(event)) {
          done();
          resolve(event as T);
        }
      });
      if (options.closeLabel) {
        const label = options.closeLabel;
        onClose = (info) => {
          done();
          reject(new RequestInterruptedError(label, info.code, info.reason));
        };
        this._closeWaiters.add(onClose);
      }
      const timeoutMs = options.timeoutMs ?? 0;
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          done();
          reject(new Error(`${options.label ?? "waitForNextEvent: no matching event"} after ${timeoutMs} ms`));
        }, timeoutMs);
      }
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

  /**
   * Resolve the credential for the attempt about to be made.
   *
   * A string option is returned as-is; a {@link TokenProvider} is
   * called now, so a single-use ticket is minted per attempt rather
   * than once per client.  Errors propagate to the caller
   * (``connect()`` or ``_attemptReconnect()``), each of which already
   * decides what a failed attempt means.
   */
  private async _resolveToken(): Promise<string | undefined> {
    const t = this._options.token;
    if (typeof t === "function") {
      return await t();
    }
    return t;
  }

  private async _openOnce(): Promise<void> {
    const token = await this._resolveToken();
    const transport = await openTransport({
      url: this._options.url,
      token,
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
    this._limits = serverLimitsFrom(serverInfo);

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
    // Requests in flight on this connection cannot be answered on the
    // next one; reject them now, naming the close.
    const waiters = [...this._closeWaiters];
    this._closeWaiters.clear();
    for (const w of waiters) {
      try {
        w(info);
      } catch {
        // A waiter only rejects its own promise.
      }
    }
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
