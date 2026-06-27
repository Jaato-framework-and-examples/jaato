// High-level convenience facade over {@link JaatoClient}.
//
// The low-level client is event-loop primitives (subscribe / sendMessage /
// the connect+createSession+done dance). The common path — open a session,
// ask, get the answer — is ~10 lines of that boilerplate, and the recipe is
// subtle enough that a naive version hangs (waiting on SESSION_TERMINATED only,
// which a plain turn never emits). This module owns the recipe so consumer code
// can't reproduce it (or its hangs):
//
//   await using s = await JaatoClient.session({ url, profile: "researcher" });
//   console.log(await s.ask("Research tide pools."));
//
// Purely additive sugar — every JaatoClient method stays; reach the underlying
// client via `s.client` to mix facade calls with the raw event API on the same
// connection. Mirror of jaato-sdk/jaato_sdk/client/convenience.py
// (#400/#402/#403/#404 — the org-wide client-simplification effort).

import { JaatoClient, type JaatoClientOptions } from "./client.js";
import { AgentError, PermissionUnhandled } from "./errors.js";
import { EventTypeValue } from "./events.js";
import type { ConnectionStatus, RecoveryConfig } from "./state.js";

// Ensure `[Symbol.asyncDispose]` is definable even on runtimes/polyfill gaps
// older than the well-known-symbol (Node < 20.4). `await using` still needs a
// runtime that recognises it; the explicit `close()` works everywhere.
(Symbol as { asyncDispose?: symbol }).asyncDispose ??= Symbol.for(
  "Symbol.asyncDispose",
);

type PermissionHandler = (event: unknown) => string | Promise<string>;

/** Runs client-side when the agent invokes a host tool; its return is sent
 *  back as the result (JSON-encoded if not a string). */
export type ClientToolHandler = (
  args: Record<string, unknown>,
) => unknown | Promise<unknown>;

/** A host/client tool spec. ``handler`` runs in YOUR process on invocation;
 *  the rest is the schema sent to the daemon. */
export interface ClientToolSpec {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  handler?: ClientToolHandler;
  /** Optional per-tool timeout (ms) / auto-approve — forwarded on the wire. */
  timeout?: number;
  auto_approve?: boolean;
  [key: string]: unknown;
}

/** Options for {@link JaatoClient.session} / {@link openSession}. */
export interface SessionOpenOptions {
  // ── connection (WebSocket) ──
  /** Full ``ws://`` / ``wss://`` URL of the jaato daemon. */
  url: string;
  /** Bearer token (``?token=`` query param). */
  token?: string;
  /** Custom request headers (Node only; mutually exclusive with token). */
  headers?: Record<string, string>;
  /** Connection open timeout in ms (default 5000). */
  openTimeoutMs?: number;
  /** Override the SDK's compile-time minimum protocol version. */
  minProtocolVersion?: string;
  /** Reconnection policy; pass to enable the recovery (auto-reconnect) path. */
  recovery?: Partial<RecoveryConfig>;
  /** Reconnection-status callback (wired before connect). */
  onStatusChange?: (status: ConnectionStatus) => void;

  // ── client config (post-connect handshake) ──
  /**
   * Presentation client type. Defaults to ``"api"`` — load-bearing for
   * headless completion (the daemon strips ``signal_completion`` for
   * ``terminal``/``web``/``chat`` roots). Set into ``presentation.client_type``.
   */
  clientType?: string;
  /** Workspace dir for file ops / sandbox scoping (→ ``working_dir``). */
  workspacePath?: string;
  /** ``.env`` path for the daemon to load (→ ``env_file``). */
  envFile?: string;
  /** Read-only framework-config root override (→ ``config_root``). */
  configRoot?: string;
  /** Opt-in per-session AppArmor confinement (→ ``apparmor``). */
  apparmor?: boolean;
  /** Extra presentation fields (merged; ``clientType`` sets ``client_type``). */
  presentation?: Record<string, unknown>;

  // ── session ──
  /** Profile name (string) or inline spec (object). */
  profile?: string | Record<string, unknown>;
  /** Agent/persona name. */
  agent?: string;
  /** Values for the agent's ``{{param}}`` placeholders. */
  agentParams?: Record<string, string>;
  /** Cascade-sharing tenant id (warm-slot reuse across stages). */
  cascadeDriverId?: string;

  // ── facade extras ──
  /**
   * Per gated-tool callback; returns the response (``"y"``/``"n"``/``"a"``/…,
   * sync or async). Without it, a gated tool makes the in-flight call throw
   * {@link PermissionUnhandled} (auto-denied so the daemon never hangs).
   */
  onPermission?: PermissionHandler;
  /**
   * Host/client tool specs (same shape as ``registerClientTools``). Registered
   * AFTER connect but BEFORE createSession (the runner-tier model only sees
   * tools registered before the session exists).
   */
  clientTools?: ClientToolSpec[];
}

/** Per-call options for {@link Session.ask} / {@link Session.stream}. */
export interface AskOptions {
  /**
   * Which ``AGENT_OUTPUT`` sources to collect, by ``.source``. Default
   * ``["model"]`` (a clean answer); ``null`` collects everything.
   */
  sources?: string[] | null;
  /** ``false`` forces sequential tool execution for the turn. */
  parallelTools?: boolean | null;
  /** Multimodal attachments forwarded to ``sendMessage``. */
  attachments?: Array<Record<string, unknown>>;
}

/** Per-call options for {@link Session.complete}. */
export interface CompleteOptions {
  parallelTools?: boolean | null;
  attachments?: Array<Record<string, unknown>>;
}

const DEFAULT_SOURCES = ["model"];

interface TerminalBox {
  reason?: string;
  errorType?: string;
  errorSummary?: string;
}

/**
 * High-level handle over an open session. Owns the send-and-wait recipe
 * (first-of ``{TURN_COMPLETED, SESSION_TERMINATED}``) so a turn never hangs.
 * Construct via {@link JaatoClient.session}, not directly. Implements
 * ``AsyncDisposable`` for ``await using``.
 */
export class Session implements AsyncDisposable {
  #client: JaatoClient;
  #onPermission?: PermissionHandler;
  #unhandledPerm: string | null = null;
  #toolHandlers?: Map<string, ClientToolHandler>;

  constructor(
    client: JaatoClient,
    onPermission?: PermissionHandler,
    toolHandlers?: Map<string, ClientToolHandler>,
  ) {
    this.#client = client;
    this.#onPermission = onPermission;
    // Wire permissions once for the session's lifetime — fail loud, never hang.
    client.subscribe(EventTypeValue.PERMISSION_REQUESTED, (event) => {
      void this.#onPerm(event);
    });
    // Host-tool dispatch: the low-level client only ships the schemas + exposes
    // respondToToolExecution; the facade owns invoking the local handler on
    // TOOL_EXECUTE_REQUEST (mirror of Python's _on_tool_execute_request).
    if (toolHandlers && toolHandlers.size > 0) {
      this.#toolHandlers = toolHandlers;
      client.subscribe(EventTypeValue.TOOL_EXECUTE_REQUEST, (event) => {
        void this.#onToolExecute(event);
      });
    }
  }

  /** The underlying low-level client — mix facade + raw event API freely. */
  get client(): JaatoClient {
    return this.#client;
  }

  /** The daemon-side session id (null until created). */
  get sessionId(): string | null {
    return this.#client.sessionId;
  }

  async #onPerm(event: unknown): Promise<void> {
    const ev = event as { request_id?: string; tool_name?: string };
    const requestId = ev?.request_id ?? "";
    if (this.#onPermission) {
      const resp = await this.#onPermission(event);
      await this.#client.respondToPermission(requestId, resp || "n");
    } else {
      // No policy to apply — deny to unstick the daemon, flag so the in-flight
      // ask/complete/stream throws PermissionUnhandled.
      this.#unhandledPerm = ev?.tool_name ?? "?";
      await this.#client.respondToPermission(requestId, "n");
    }
  }

  async #onToolExecute(event: unknown): Promise<void> {
    const ev = event as {
      call_id?: string;
      tool_name?: string;
      tool_args?: Record<string, unknown>;
    };
    const callId = ev?.call_id ?? "";
    const handler = ev?.tool_name ? this.#toolHandlers?.get(ev.tool_name) : undefined;
    if (!handler) {
      await this.#client.respondToToolExecution(
        callId,
        "",
        `no handler for host tool ${JSON.stringify(ev?.tool_name)}`,
      );
      return;
    }
    try {
      const out = await handler(ev?.tool_args ?? {});
      const result = typeof out === "string" ? out : JSON.stringify(out);
      await this.#client.respondToToolExecution(callId, result);
    } catch (err) {
      await this.#client.respondToToolExecution(
        callId,
        "",
        err instanceof Error ? err.message : String(err),
      );
    }
  }

  #raiseIfNeeded(box: TerminalBox): void {
    if (this.#unhandledPerm !== null) {
      const tool = this.#unhandledPerm;
      this.#unhandledPerm = null;
      throw new PermissionUnhandled(tool);
    }
    if (box.reason === "error") {
      throw new AgentError(box.errorType, box.errorSummary);
    }
  }

  #noteTerminal(box: TerminalBox, event: unknown): void {
    const ev = event as { reason?: string; error_type?: string; error_summary?: string };
    // first-of {TURN_COMPLETED (no reason -> "natural"), SESSION_TERMINATED
    // (rich reason/error)}; the ??= lets a real error reason win on a race.
    box.reason ??= ev?.reason ?? "natural";
    if (ev?.error_type) box.errorType = ev.error_type;
    if (ev?.error_summary) box.errorSummary = ev.error_summary;
  }

  /**
   * Send ``prompt``, wait for the turn to finish, return collected text.
   * Waits on first-of ``{TURN_COMPLETED, SESSION_TERMINATED}`` (never hangs).
   * Throws {@link AgentError} on an error terminal / {@link PermissionUnhandled}
   * on an unanswered gated tool.
   */
  async ask(prompt: string, opts: AskOptions = {}): Promise<string> {
    const sources = opts.sources === undefined ? DEFAULT_SOURCES : opts.sources;
    const chunks: string[] = [];
    const box: TerminalBox = {};
    let resolveDone!: () => void;
    const done = new Promise<void>((resolve) => {
      resolveDone = resolve;
    });

    const onOutput = (event: unknown): void => {
      const ev = event as { source?: string; text?: string };
      if (sources === null || (ev?.source != null && sources.includes(ev.source))) {
        if (ev?.text) chunks.push(ev.text);
      }
    };
    const onTerminal = (event: unknown): void => {
      this.#noteTerminal(box, event);
      resolveDone();
    };

    const unsubOutput = this.#client.subscribe(EventTypeValue.AGENT_OUTPUT, onOutput);
    const unsubTerm = this.#client.subscribeOnce(EventTypeValue.SESSION_TERMINATED, onTerminal);
    const unsubTurn = this.#client.subscribeOnce(EventTypeValue.TURN_COMPLETED, onTerminal);
    try {
      await this.#client.sendMessage(prompt, opts.attachments, opts.parallelTools);
      await done;
    } finally {
      unsubOutput();
      unsubTerm();
      unsubTurn();
    }
    this.#raiseIfNeeded(box);
    return chunks.join("");
  }

  /**
   * Send ``prompt`` and return the typed completion payload (or null when the
   * profile declared no schema / the model didn't complete). For
   * completion-gated profiles. Throws {@link AgentError} on an error terminal.
   */
  async complete(
    prompt: string,
    opts: CompleteOptions = {},
  ): Promise<Record<string, unknown> | null> {
    const box: TerminalBox = {};
    let payload: Record<string, unknown> | null = null;
    let resolveDone!: () => void;
    const done = new Promise<void>((resolve) => {
      resolveDone = resolve;
    });

    const onCompleted = (event: unknown): void => {
      payload = (event as { payload?: Record<string, unknown> })?.payload ?? null;
    };
    const onTerminal = (event: unknown): void => {
      this.#noteTerminal(box, event);
      resolveDone();
    };

    const unsubComp = this.#client.subscribeOnce(EventTypeValue.AGENT_COMPLETED, onCompleted);
    const unsubTerm = this.#client.subscribeOnce(EventTypeValue.SESSION_TERMINATED, onTerminal);
    const unsubTurn = this.#client.subscribeOnce(EventTypeValue.TURN_COMPLETED, onTerminal);
    try {
      await this.#client.sendMessage(prompt, opts.attachments, opts.parallelTools);
      await done;
    } finally {
      unsubComp();
      unsubTerm();
      unsubTurn();
    }
    this.#raiseIfNeeded(box);
    return payload;
  }

  /**
   * Send ``prompt`` and yield text chunks live as they arrive, stopping at the
   * terminal. ``TURN_COMPLETED`` fires after all output, so no chunk is
   * dropped. Throws {@link AgentError} / {@link PermissionUnhandled} after the
   * stream drains.
   */
  async *stream(prompt: string, opts: AskOptions = {}): AsyncIterable<string> {
    const sources = opts.sources === undefined ? DEFAULT_SOURCES : opts.sources;
    const queue: string[] = [];
    const box: TerminalBox = {};
    let terminated = false;
    let wake: (() => void) | null = null;
    const bump = (): void => {
      if (wake) {
        const w = wake;
        wake = null;
        w();
      }
    };

    const onOutput = (event: unknown): void => {
      const ev = event as { source?: string; text?: string };
      if (sources === null || (ev?.source != null && sources.includes(ev.source))) {
        if (ev?.text) {
          queue.push(ev.text);
          bump();
        }
      }
    };
    const onTerminal = (event: unknown): void => {
      this.#noteTerminal(box, event);
      terminated = true;
      bump();
    };

    const unsubOutput = this.#client.subscribe(EventTypeValue.AGENT_OUTPUT, onOutput);
    const unsubTerm = this.#client.subscribeOnce(EventTypeValue.SESSION_TERMINATED, onTerminal);
    const unsubTurn = this.#client.subscribeOnce(EventTypeValue.TURN_COMPLETED, onTerminal);
    try {
      await this.#client.sendMessage(prompt, opts.attachments, opts.parallelTools);
      for (;;) {
        if (queue.length > 0) {
          yield queue.shift() as string;
          continue;
        }
        if (terminated) break;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
    } finally {
      unsubOutput();
      unsubTerm();
      unsubTurn();
    }
    this.#raiseIfNeeded(box);
  }

  /** Disconnect the underlying client (idempotent). */
  async close(): Promise<void> {
    await this.#client.close();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}

/**
 * Build a {@link JaatoClient}, connect, register host tools, create the
 * session, and return a ready {@link Session}. Backs {@link JaatoClient.session}.
 */
export async function openSession(opts: SessionOpenOptions): Promise<Session> {
  const presentation: Record<string, unknown> = {
    ...(opts.presentation ?? {}),
    client_type: opts.clientType ?? "api",
  };
  const clientConfig: Record<string, unknown> = { presentation };
  if (opts.workspacePath !== undefined) clientConfig.working_dir = opts.workspacePath;
  if (opts.envFile !== undefined) clientConfig.env_file = opts.envFile;
  if (opts.configRoot !== undefined) clientConfig.config_root = opts.configRoot;
  if (opts.apparmor !== undefined) clientConfig.apparmor = opts.apparmor;

  const clientOptions: JaatoClientOptions = {
    url: opts.url,
    token: opts.token,
    headers: opts.headers,
    minProtocolVersion: opts.minProtocolVersion,
    recovery: opts.recovery,
    openTimeoutMs: opts.openTimeoutMs,
    clientConfig: clientConfig as JaatoClientOptions["clientConfig"],
  };
  const client = new JaatoClient(clientOptions);
  if (opts.onStatusChange) client.onStatus(opts.onStatusChange);

  await client.connect(); // throws ConnectionError on failure
  // Host tools MUST be registered after connect, before createSession. The
  // handler is local — strip it from the wire spec and keep it for the
  // facade's TOOL_EXECUTE_REQUEST dispatcher (the low-level client only ships
  // the schema + exposes respondToToolExecution).
  const toolHandlers = new Map<string, ClientToolHandler>();
  if (opts.clientTools && opts.clientTools.length > 0) {
    const wireSpecs = opts.clientTools.map((spec) => {
      const { handler, ...rest } = spec;
      if (handler && typeof spec.name === "string") {
        toolHandlers.set(spec.name, handler);
      }
      return rest as Record<string, unknown>;
    });
    await client.registerClientTools(wireSpecs);
  }
  await client.createSession({
    profile: opts.profile,
    agent: opts.agent,
    agentParams: opts.agentParams,
    cascadeDriverId: opts.cascadeDriverId,
  });
  if (!client.sessionId) {
    await client.close();
    throw new Error(
      "session.new did not produce a session id — check provider auth / the daemon log",
    );
  }
  return new Session(client, opts.onPermission, toolHandlers);
}

/**
 * One-shot: open a session, ask, return the answer, tear down. Sugar over
 * {@link openSession} — one connect per call; for repeated calls hold a
 * {@link Session} via {@link JaatoClient.session} and reuse it.
 */
export async function ask(
  prompt: string,
  opts: SessionOpenOptions & AskOptions,
): Promise<string> {
  const session = await openSession(opts);
  try {
    return await session.ask(prompt, {
      sources: opts.sources,
      parallelTools: opts.parallelTools,
      attachments: opts.attachments,
    });
  } finally {
    await session.close();
  }
}
