// Tests for JaatoClient.
//
// Mirror of the Python-side test_sdk_parity_methods.py — every
// typed method gets a "constructs the right event" check, plus
// handshake / version-gate / reconnect coverage.
//
// Uses node:test (built into Node 18+) so the SDK has zero test-
// time dependencies.  A small MockWebSocket replaces the global
// WebSocket constructor, capturing every event the client sends
// and replaying server-shaped events back as desired.

import { strict as assert } from "node:assert";
import { afterEach, beforeEach, describe, mock, test } from "node:test";

import {
  JaatoClient,
  MIN_ATTACHMENT_RESUME_PROTOCOL,
  MIN_SESSION_RELOAD_ENV_PROTOCOL,
  MIN_WORKSPACE_IGNORE_PROTOCOL,
  MIN_SCAFFOLD_INTEGRATION_PROTOCOL,
  MIN_FILE_FETCH_PROTOCOL,
  MIN_WORKSPACE_PICKER_PROTOCOL,
  MIN_MEMORY_VERBS_PROTOCOL,
  MIN_SESSION_MESSAGE_PROTOCOL,
  MIN_SESSION_MESSAGE_FILES_PROTOCOL,
  MIN_PROTOCOL_VERSION,
  STAGE_FILES_TIMEOUT_MS,
  LEGACY_SERVER_LIMITS,
  serverLimitsFrom,
} from "./client.js";
import {
  ConnectionClosedError,
  RequestInterruptedError,
  IncompatibleServerError,
  ReconnectingError,
} from "./errors.js";
import { ConnectionState } from "./state.js";
import { EventTypeValue, type JaatoEvent } from "./events.js";

// ──── MockWebSocket ──────────────────────────────────────────────

interface MockInstance {
  url: string;
  sent: string[];
  sentBinary: ArrayBuffer[];
  readyState: number;
  binaryType: string;
  onopen: (() => void) | null;
  onmessage: ((msg: { data: string }) => void) | null;
  onerror: (() => void) | null;
  onclose: ((info: { code: number; reason: string }) => void) | null;
  emit(event: object): void;
  emitClose(code?: number, reason?: string): void;
  send(data: string | ArrayBuffer | Uint8Array): void;
  close(code?: number, reason?: string): void;
}

let lastInstance: MockInstance | null = null;
let lastCtorArgs: unknown[] = [];
const realWebSocket = (globalThis as Record<string, unknown>).WebSocket;

function installMockWebSocket(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket = function (url: string, ...rest: unknown[]): MockInstance {
    lastCtorArgs = [url, ...rest];
    const instance: MockInstance = {
      url,
      sent: [],
      sentBinary: [],
      readyState: 0,
      binaryType: "arraybuffer",
      onopen: null,
      onmessage: null,
      onerror: null,
      onclose: null,
      emit(event: object): void {
        if (this.onmessage) {
          this.onmessage({ data: JSON.stringify(event) });
        }
      },
      emitClose(code = 1000, reason = ""): void {
        this.readyState = 3;
        if (this.onclose) {
          this.onclose({ code, reason });
        }
      },
      send(data: string | ArrayBuffer | Uint8Array): void {
        if (typeof data === "string") {
          this.sent.push(data);
        } else {
          // Normalise Uint8Array view → underlying ArrayBuffer slice.
          const buf =
            data instanceof ArrayBuffer
              ? data
              : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
          this.sentBinary.push(buf);
        }
      },
      close(code?: number, reason?: string): void {
        this.emitClose(code ?? 1000, reason ?? "");
      },
    };
    lastInstance = instance;
    // Open synchronously on next microtask so connect() resolves cleanly.
    queueMicrotask(() => {
      instance.readyState = 1;
      if (instance.onopen) instance.onopen();
    });
    return instance;
  };
}

function restoreWebSocket(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket = realWebSocket;
  lastInstance = null;
}

function makeConnectedEvent(
  protocolVersion = MIN_PROTOCOL_VERSION,
  serverVersion = "0.7.1",
): JaatoEvent {
  return {
    type: EventTypeValue.CONNECTED,
    timestamp: new Date().toISOString(),
    protocol_version: protocolVersion,
    server_info: {
      client_id: "client_1",
      server_version: serverVersion,
    },
  } as unknown as JaatoEvent;
}

async function connectAndAck(
  client: JaatoClient,
  protocolVersion = MIN_PROTOCOL_VERSION,
  serverVersion = "0.7.1",
): Promise<void> {
  const connectPromise = client.connect();
  // Let the WS open microtask fire, then emit ConnectedEvent.
  await new Promise<void>((resolve) => queueMicrotask(resolve));
  await new Promise<void>((resolve) => queueMicrotask(resolve));
  if (lastInstance == null) {
    throw new Error("MockWebSocket was not constructed");
  }
  lastInstance.emit(makeConnectedEvent(protocolVersion, serverVersion));
  await connectPromise;
}

function getSent(): JaatoEvent[] {
  if (lastInstance == null) return [];
  return lastInstance.sent.map((s) => JSON.parse(s) as JaatoEvent);
}

// ──── Tests ──────────────────────────────────────────────────────

describe("JaatoClient handshake", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("connect resolves on ConnectedEvent and transitions to CONNECTED", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080", token: "t" });
    assert.equal(client.state, ConnectionState.DISCONNECTED);
    await connectAndAck(client);
    assert.equal(client.state, ConnectionState.CONNECTED);
    assert.equal(client.serverProtocolVersion, MIN_PROTOCOL_VERSION);
    assert.equal(client.serverVersion, "0.7.1");
    assert.equal(client.clientId, "client_1");
    await client.close();
  });

  test("custom headers travel as the SECOND constructor argument (Node's built-in WebSocket ignores a third)", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      headers: { Authorization: "Bearer app-credential" },
    });
    await connectAndAck(client);
    assert.equal(lastCtorArgs.length, 2, `expected (url, options), got ${lastCtorArgs.length} args`);
    assert.deepEqual(lastCtorArgs[1], { headers: { Authorization: "Bearer app-credential" } });
    assert.ok(!lastInstance!.url.includes("token="), "headers must not also leak into the query string");
    await client.close();
  });

  test("token appended as ?token= query param", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080/", token: "secret" });
    await connectAndAck(client);
    assert.ok(lastInstance!.url.includes("token=secret"));
    await client.close();
  });

  test("incompatible major-version protocol throws IncompatibleServerError", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const connectPromise = client.connect();
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    // Server speaks 2.x; client requires 1.x — major mismatch.
    lastInstance!.emit(makeConnectedEvent("2.0"));
    await assert.rejects(connectPromise, IncompatibleServerError);
  });

  test("server with newer minor still connects (additive forward-compat)", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    // Server is at 1.5; client requires 1.0 — fine, server has more.
    await connectAndAck(client, "1.5");
    assert.equal(client.state, ConnectionState.CONNECTED);
    await client.close();
  });

  test("explicit minProtocolVersion override is honoured", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      minProtocolVersion: "1.0",
    });
    await connectAndAck(client, "1.2");
    assert.equal(client.state, ConnectionState.CONNECTED);
    await client.close();
  });

  test("ClientConfigRequest is sent when clientConfig is provided", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      clientConfig: { working_dir: "/home/app", permission_timeout: 0 },
    });
    await connectAndAck(client);
    const sent = getSent();
    assert.equal(sent.length, 1);
    assert.equal(sent[0].type, EventTypeValue.CLIENT_CONFIG);
    assert.equal((sent[0] as { working_dir?: string }).working_dir, "/home/app");
    await client.close();
  });
});

describe("JaatoClient typed methods", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    // Drop the handshake-phase events from the capture so each
    // test only sees the events its method sends.
    if (lastInstance) lastInstance.sent = [];
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  test("sendMessage with parallel_tools propagates the field", async () => {
    await client.sendMessage("hi", undefined, true);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.SEND_MESSAGE);
    assert.equal((ev as { parallel_tools?: boolean }).parallel_tools, true);
  });

  test("injectPrompt defaults source_type to 'user'", async () => {
    await client.injectPrompt("steer me");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.INJECT_PROMPT_REQUEST);
    assert.equal((ev as { source_type?: string }).source_type, "user");
  });

  test("injectPrompt with source_type='child' for follow-up", async () => {
    await client.injectPrompt("follow up", "child", "ui");
    const [ev] = getSent();
    assert.equal((ev as { source_type?: string }).source_type, "child");
    assert.equal((ev as { source_id?: string }).source_id, "ui");
  });

  test("replayMessages omits messages → null (continue from current)", async () => {
    await client.replayMessages("r1");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.REPLAY_MESSAGES_REQUEST);
    assert.equal((ev as { request_id?: string }).request_id, "r1");
    assert.equal((ev as { messages?: unknown }).messages, null);
    assert.equal((ev as { timeout_seconds?: number }).timeout_seconds, 120);
  });

  test("resolveForkPoint with after_message specifier", async () => {
    await client.resolveForkPoint("r2", { afterMessage: 5 });
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.RESOLVE_FORK_POINT_REQUEST);
    assert.equal((ev as { after_message?: number }).after_message, 5);
    assert.equal((ev as { after_tool_call?: unknown }).after_tool_call, null);
  });

  test("resolveForkPoint with after_tool_call specifier", async () => {
    await client.resolveForkPoint("r3", { afterToolCall: "call_42" });
    const [ev] = getSent();
    assert.equal((ev as { after_tool_call?: string }).after_tool_call, "call_42");
  });

  test("addWhitelistTools sends both tools and patterns", async () => {
    await client.addWhitelistTools(["read_file"], ["safe_*"]);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_ADD_WHITELIST_REQUEST);
    assert.deepEqual((ev as { tools?: string[] }).tools, ["read_file"]);
    assert.deepEqual((ev as { patterns?: string[] }).patterns, ["safe_*"]);
  });

  test("addBlacklistTools defaults patterns to []", async () => {
    await client.addBlacklistTools(["dangerous"]);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_ADD_BLACKLIST_REQUEST);
    assert.deepEqual((ev as { tools?: string[] }).tools, ["dangerous"]);
    assert.deepEqual((ev as { patterns?: string[] }).patterns, []);
  });

  test("removePermissionRules requires target", async () => {
    await client.removePermissionRules("blacklist", ["t1"], ["p1"]);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_REMOVE_REQUEST);
    assert.equal((ev as { target?: string }).target, "blacklist");
  });

  test("clearPermissionRules defaults to 'all'", async () => {
    await client.clearPermissionRules();
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_CLEAR_REQUEST);
    assert.equal((ev as { target?: string }).target, "all");
  });

  test("setDefaultPolicy sends the policy verbatim", async () => {
    await client.setDefaultPolicy("allow");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_SET_DEFAULT_REQUEST);
    assert.equal((ev as { policy?: string }).policy, "allow");
  });

  test("requestPolicySnapshot accepts request_id", async () => {
    await client.requestPolicySnapshot("snap1");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_POLICY_SNAPSHOT_REQUEST);
    assert.equal((ev as { request_id?: string }).request_id, "snap1");
  });

  test("stop sends StopRequest", async () => {
    await client.stop();
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.STOP);
  });

  test("respondToPermission carries request_id and edited_arguments", async () => {
    await client.respondToPermission("req_42", "y", { foo: "bar" });
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.PERMISSION_RESPONSE);
    assert.equal((ev as { request_id?: string }).request_id, "req_42");
    assert.deepEqual((ev as { edited_arguments?: unknown }).edited_arguments, { foo: "bar" });
  });

  test("respondToPostAuthSetup mirrors the Python SDK's PostAuthSetupResponse", async () => {
    await client.respondToPostAuthSetup("req_7", { connect: true, modelName: "claude-sonnet-4", persistEnv: true });
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.POST_AUTH_SETUP_RESPONSE);
    assert.equal((ev as { request_id?: string }).request_id, "req_7");
    assert.equal((ev as { connect?: boolean }).connect, true);
    assert.equal((ev as { model_name?: string }).model_name, "claude-sonnet-4");
    assert.equal((ev as { persist_env?: boolean }).persist_env, true);
  });

  test("respondToPostAuthSetup declining sends connect=false with the Python defaults", async () => {
    await client.respondToPostAuthSetup("req_8", { connect: false });
    const [ev] = getSent();
    assert.equal((ev as { connect?: boolean }).connect, false);
    assert.equal((ev as { model_name?: string }).model_name, "");
    assert.equal((ev as { persist_env?: boolean }).persist_env, false);
  });

  test("executeCommand sends CommandRequest", async () => {
    await client.executeCommand("permissions", ["whitelist", "tool1"]);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "permissions");
    assert.deepEqual((ev as { args?: string[] }).args, ["whitelist", "tool1"]);
  });

  test("registerClientTools sends ToolsRegisterClientRequest", async () => {
    const tools = [{ name: "browser_pick", description: "..." }];
    await client.registerClientTools(tools);
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.TOOLS_REGISTER_CLIENT);
    assert.deepEqual((ev as { tools?: unknown }).tools, tools);
  });

  test("sendRawEvent emits arbitrary type-checking-bypassed envelopes", async () => {
    // Premium-style daemon-extension verb — not in JaatoEvent union.
    await client.sendRawEvent({
      type: "reconnect.list",
      filter: { only_attached: true },
    });
    const [ev] = getSent();
    assert.equal(ev.type, "reconnect.list");
    assert.deepEqual(
      (ev as { filter?: unknown }).filter,
      { only_attached: true },
    );
  });

  test("sendRawEvent rejects after close", async () => {
    await client.close();
    await assert.rejects(
      client.sendRawEvent({ type: "reconnect.list" }),
      ConnectionClosedError,
    );
  });

  test("respondToToolExecution success path", async () => {
    await client.respondToToolExecution("call_42", '{"ok":true}');
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.TOOL_EXECUTE_RESULT);
    assert.equal((ev as { call_id?: string }).call_id, "call_42");
    assert.equal((ev as { result?: string }).result, '{"ok":true}');
    assert.equal((ev as { error?: string }).error, "");
  });

  test("respondToToolExecution error path", async () => {
    await client.respondToToolExecution("call_99", "", "tool crashed");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.TOOL_EXECUTE_RESULT);
    assert.equal((ev as { call_id?: string }).call_id, "call_99");
    assert.equal((ev as { error?: string }).error, "tool crashed");
  });
});

describe("JaatoClient resume verbs carry attachments (#845)", () => {
  // Both ways of driving an EXISTING session — session.wake and
  // injectPrompt — were text-only, while `attachments` sat on sendMessage,
  // the LIVE-session path.  A session whose input is audio could be started
  // with an utterance and never driven again with one.
  let client: JaatoClient;

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  async function connected(protocol = MIN_ATTACHMENT_RESUME_PROTOCOL) {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, protocol);
    if (lastInstance) lastInstance.sent = [];
  }

  test("injectPrompt forwards attachments", async () => {
    await connected();
    await client.injectPrompt("look", "user", undefined, [
      { mime_type: "image/png", data: "QUJD" },
    ]);
    const [ev] = getSent();
    const atts = (ev as { attachments?: Array<Record<string, unknown>> })
      .attachments;
    assert.equal(atts?.[0].mime_type, "image/png");
  });

  test("wakeSession puts the utterance in the command payload", async () => {
    await connected();
    await client.wakeSession("sess_1", "", {
      attachments: [{ mime_type: "audio/wav", data: "QUJD" }],
      source: "phone",
      eventId: "e1",
    });
    const [ev] = getSent();
    const payload = (ev as { payload?: Record<string, unknown> }).payload!;
    assert.equal((ev as { command?: string }).command, "session.wake");
    assert.equal(payload.session_id, "sess_1");
    // Blank text is the NORMAL shape: for a spoken message the attachment
    // IS the message.
    assert.equal(payload.text, "");
    assert.equal(payload.source, "phone");
    assert.equal(payload.event_id, "e1");
  });

  test("wakeSession with neither text nor attachments is refused", async () => {
    await connected();
    await assert.rejects(
      () => client.wakeSession("sess_1"),
      /text or attachments/,
    );
    assert.equal(getSent().length, 0);
  });

  test("a daemon that would DROP the bytes is refused, not degraded", async () => {
    // The degraded call is a turn driven without the audio that was the whole
    // message — for a blank-text utterance, an empty turn reported as success.
    await connected("1.4");
    await assert.rejects(
      () =>
        client.wakeSession("sess_1", "", {
          attachments: [{ mime_type: "audio/wav", data: "QUJD" }],
        }),
      /DROP the attachments/,
    );
    await assert.rejects(
      () => client.injectPrompt("hi", "user", undefined, [{ data: "QUJD" }]),
      /DROP the attachments/,
    );
    assert.equal(getSent().length, 0);
  });

  test("a text-only resume still works against an older daemon", async () => {
    await connected("1.4");
    await client.wakeSession("sess_1", "have another look");
    await client.injectPrompt("steer");
    assert.equal(getSent().length, 2);
  });
});

describe("JaatoClient session management", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    if (lastInstance) lastInstance.sent = [];
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  test("createSession with no options sends bare session.new", async () => {
    await client.createSession();
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "session.new");
    assert.deepEqual((ev as { args?: string[] }).args, []);
  });

  test("createSession threads name + profile + agent + agentParams", async () => {
    await client.createSession({
      name: "test-session",
      profile: "researcher",
      agent: "code-reviewer",
      agentParams: { topic: "auth", depth: "deep" },
    });
    const [ev] = getSent();
    const args = (ev as { args?: string[] }).args ?? [];
    assert.equal(args[0], "test-session");
    assert.ok(args.includes("--profile"));
    assert.ok(args.includes("researcher"));
    assert.ok(args.includes("--agent"));
    assert.ok(args.includes("code-reviewer"));
    assert.ok(args.includes("topic=auth"));
    assert.ok(args.includes("depth=deep"));
    // Profile-by-name path → no payload, all-via-argv.
    assert.equal((ev as { payload?: unknown }).payload, undefined);
  });

  test("createSession with inline profile dict routes to payload", async () => {
    const spec = {
      model: "claude-sonnet-4-5",
      provider: "anthropic",
      plugins: ["cli", "web_search"],
      system_instructions: "You are a researcher.",
    };
    await client.createSession({ profile: spec });
    const [ev] = getSent();
    const args = (ev as { args?: string[] }).args ?? [];
    // No --profile flag in argv when the spec is inline.
    assert.ok(!args.includes("--profile"));
    assert.deepEqual((ev as { payload?: unknown }).payload, { spec });
  });

  test("createSession inline profile composes with name + agent", async () => {
    await client.createSession({
      name: "ops-task",
      profile: { model: "claude-sonnet-4-5" },
      agent: "reviewer",
      agentParams: { focus: "security" },
    });
    const [ev] = getSent();
    const args = (ev as { args?: string[] }).args ?? [];
    assert.equal(args[0], "ops-task");
    assert.ok(args.includes("--agent"));
    assert.ok(args.includes("reviewer"));
    assert.ok(args.includes("focus=security"));
    assert.deepEqual((ev as { payload?: unknown }).payload, {
      spec: { model: "claude-sonnet-4-5" },
    });
  });

  test("createSession rejects invalid profile types", async () => {
    await assert.rejects(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      client.createSession({ profile: 42 as any }),
      TypeError,
    );
  });

  test("createSession refuses a model override against a pre-1.27 daemon", async () => {
    await assert.rejects(client.createSession({ model: "gpt-5.1" }), /1\.27/);
    assert.equal(getSent().length, 0);
  });

  test("attachSession sends session.attach and updates sessionId", async () => {
    await client.attachSession("sess_abc");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "session.attach");
    assert.deepEqual((ev as { args?: string[] }).args, ["sess_abc"]);
    assert.equal(client.sessionId, "sess_abc");
  });

  test("getDefaultSession sends session.default", async () => {
    await client.getDefaultSession();
    const [ev] = getSent();
    assert.equal((ev as { command?: string }).command, "session.default");
  });

  test("listSessions sends session.list", async () => {
    await client.listSessions();
    const [ev] = getSent();
    assert.equal((ev as { command?: string }).command, "session.list");
  });

  test("listProfiles sends session.profiles", async () => {
    await client.listProfiles();
    const [ev] = getSent();
    assert.equal((ev as { command?: string }).command, "session.profiles");
  });

  test("endSession sends session.end with no args", async () => {
    await client.endSession();
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "session.end");
    assert.deepEqual((ev as { args?: string[] }).args, []);
  });

  test("reloadSessionEnv sends session.reload_env for the attached session", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_SESSION_RELOAD_ENV_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
    await client.reloadSessionEnv();
    const [ev] = getSent();
    assert.equal((ev as { command?: string }).command, "session.reload_env");
    assert.deepEqual((ev as { args?: string[] }).args, []);
    await client.reloadSessionEnv("sess_9");
    assert.deepEqual((getSent()[1] as { args?: string[] }).args, ["sess_9"]);
  });

  test("reloadSessionEnv is refused below protocol 1.11", async () => {
    await assert.rejects(() => client.reloadSessionEnv(), /session\.reload_env/);
    assert.equal(getSent().length, 0);
  });

  test("toggleWorkspaceIgnore sends workspace.ignore with the entry as its one arg", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_WORKSPACE_IGNORE_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
    await client.toggleWorkspaceIgnore(".jaato/logs/");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "workspace.ignore");
    assert.deepEqual((ev as { args?: string[] }).args, [".jaato/logs/"]);
  });

  test("toggleWorkspaceIgnore is refused below protocol 1.12", async () => {
    await assert.rejects(() => client.toggleWorkspaceIgnore("x"), /workspace\.ignore/);
    assert.equal(getSent().length, 0);
  });

  test("runScaffoldIntegration sends scaffold.integration with the name as its one arg", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_SCAFFOLD_INTEGRATION_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
    await client.runScaffoldIntegration("claude-code");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "scaffold.integration");
    assert.deepEqual((ev as { args?: string[] }).args, ["claude-code"]);
  });

  test("runScaffoldIntegration is refused below protocol 1.21 with nothing sent", async () => {
    await assert.rejects(
      () => client.runScaffoldIntegration("claude-code"),
      /scaffold\.integration/,
    );
    assert.equal(getSent().length, 0);
  });

  test("sendSessionMessage carries fileRefs and textAttachments at 1.24", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_SESSION_MESSAGE_FILES_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
    await client.sendSessionMessage("s-b", "", {
      fileRefs: ["reports/q3.md", { path: "a", workspace: "/w" }],
      textAttachments: [{ name: "fix.patch", text: "--- a" }],
      requestId: "r1",
    });
    const [ev] = getSent();
    assert.equal((ev as { command?: string }).command, "session.message");
    const payload = (ev as { payload?: Record<string, unknown> }).payload ?? {};
    assert.deepEqual(payload.file_refs, ["reports/q3.md", { path: "a", workspace: "/w" }]);
    assert.deepEqual(payload.text_attachments, [{ name: "fix.patch", text: "--- a" }]);
    assert.equal(payload.text, "");
    assert.equal(payload.request_id, "r1");
    assert.equal("attachments" in payload, false);
  });

  test("sendSessionMessage refuses files below 1.24 and still sends text alone at 1.23", async () => {
    // A 1.23 daemon reads neither key: it would deliver the text WITHOUT
    // the files and answer accepted -- a degraded call that reads as success.
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_SESSION_MESSAGE_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
    await assert.rejects(
      () => client.sendSessionMessage("s-b", "see", { fileRefs: ["a.md"] }),
      /fileRefs \/ textAttachments/,
    );
    await assert.rejects(
      () => client.sendSessionMessage("s-b", "see", { textAttachments: [{ text: "x" }] }),
      /fileRefs \/ textAttachments/,
    );
    assert.equal(getSent().length, 0);
    await client.sendSessionMessage("s-b", "see");
    assert.equal(getSent().length, 1);
    await assert.rejects(() => client.sendSessionMessage("s-b"), /requires text/);
  });

  test("deleteSession carries the session id as the first arg", async () => {
    await client.deleteSession("sess_xyz");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "session.delete");
    assert.deepEqual((ev as { args?: string[] }).args, ["sess_xyz"]);
  });

  // #1167 -- the type existed in both SDKs and the method in neither, so the
  // only producer was a web component hand-rolling the frame.  These pin the
  // three things a hand-rolled frame kept getting wrong.
  test("sendExternalEvent sends a typed event.external frame", async () => {
    await client.sendExternalEvent("order.placed", { id: 7 });
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.EVENT_EXTERNAL);
    assert.equal((ev as { name?: string }).name, "order.placed");
    assert.deepEqual((ev as { data?: unknown }).data, { id: 7 });
  });

  test("sendExternalEvent sends {} rather than undefined for an absent payload", async () => {
    await client.sendExternalEvent("build.finished");
    const [ev] = getSent();
    assert.deepEqual((ev as { data?: unknown }).data, {});
  });

  test("sendExternalEvent carries timestamp and sessionId when given", async () => {
    await client.sendExternalEvent(
      "ticket.assigned",
      {},
      { timestamp: "2026-09-20T12:00:00Z", sessionId: "sess_abc" },
    );
    const [ev] = getSent();
    assert.equal((ev as { timestamp?: string }).timestamp, "2026-09-20T12:00:00Z");
    assert.equal((ev as { session_id?: string }).session_id, "sess_abc");
  });

  test("sendExternalEvent refuses an empty name and sends nothing", async () => {
    await assert.rejects(() => client.sendExternalEvent(""), /requires a name/);
    assert.equal(getSent().length, 0);
  });
});

describe("JaatoClient.stageFiles", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    if (lastInstance) {
      lastInstance.sent = [];
      lastInstance.sentBinary = [];
    }
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  test("sends TEXT request frame followed by binary frames in order", async () => {
    const file1 = new TextEncoder().encode("hello world");
    const file2 = new TextEncoder().encode("second file content");

    const stageFilesPromise = client.stageFiles("workspace_abc", [
      { name: "first.txt", data: file1, contentType: "text/plain" },
      { name: "second.txt", data: file2 },
    ]);

    // Server normally responds inline; for the test, queue up a
    // StageFilesEvent on the next tick.
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    lastInstance!.emit({
      type: EventTypeValue.WORKSPACE_FILES_STAGED,
      timestamp: new Date().toISOString(),
      workspace_id: "workspace_abc",
      staged: [{ name: "first.txt" }, { name: "second.txt" }],
      failed: [],
    });

    const result = await stageFilesPromise;

    // The TEXT frame is the StageFilesRequest with declared specs.
    const requestFrame = JSON.parse(lastInstance!.sent[0]!) as Record<string, unknown>;
    assert.equal(requestFrame.type, EventTypeValue.WORKSPACE_FILES_STAGE_REQUEST);
    assert.equal(requestFrame.workspace_id, "workspace_abc");
    const specs = requestFrame.files as Array<{ name: string; size: number }>;
    assert.equal(specs[0].name, "first.txt");
    assert.equal(specs[0].size, file1.byteLength);
    assert.equal(specs[1].name, "second.txt");
    assert.equal(specs[1].size, file2.byteLength);

    // The binary frames are sent in declared order.
    assert.equal(lastInstance!.sentBinary.length, 2);
    assert.equal(lastInstance!.sentBinary[0].byteLength, file1.byteLength);
    assert.equal(lastInstance!.sentBinary[1].byteLength, file2.byteLength);
    assert.deepEqual(
      new Uint8Array(lastInstance!.sentBinary[0]),
      file1,
    );

    // The returned event is the response.
    assert.equal(result.type, EventTypeValue.WORKSPACE_FILES_STAGED);
    assert.equal((result as unknown as { workspace_id: string }).workspace_id, "workspace_abc");
  });

  test("rejects with ConnectionClosedError after close", async () => {
    await client.close();
    await assert.rejects(
      client.stageFiles("workspace_abc", [
        { name: "x.txt", data: new Uint8Array([1, 2, 3]) },
      ]),
      ConnectionClosedError,
    );
  });

  test("accepts ArrayBuffer input as well as Uint8Array", async () => {
    const buf = new ArrayBuffer(16);
    new Uint8Array(buf).set([1, 2, 3, 4, 5]);

    const promise = client.stageFiles("ws", [
      { name: "blob.bin", data: buf },
    ]);
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    lastInstance!.emit({
      type: EventTypeValue.WORKSPACE_FILES_STAGED,
      timestamp: new Date().toISOString(),
      workspace_id: "ws",
      staged: [{ name: "blob.bin" }],
      failed: [],
    });
    await promise;

    const requestFrame = JSON.parse(lastInstance!.sent[0]!) as Record<string, unknown>;
    const specs = requestFrame.files as Array<{ size: number }>;
    assert.equal(specs[0].size, 16);
    assert.equal(lastInstance!.sentBinary[0].byteLength, 16);
  });

  // #1248: the wait had no deadline, so a lost workspace.files.staged
  // response left the promise unsettled forever and the caller's status
  // stuck on "staging".  Drive the deadline with fake timers — no real
  // clock sleep.
  test("rejects and cleans up the subscription when no staged response arrives", async () => {
    mock.timers.enable({ apis: ["setTimeout"] });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handlersBefore = (client as any)._catchallHandlers.length as number;
      const promise = client.stageFiles(
        "workspace_abc",
        [{ name: "x.txt", data: new Uint8Array([1, 2, 3]) }],
        { timeoutMs: 5_000 },
      );
      const settled = assert.rejects(promise, /no workspace\.files\.staged response after 5000 ms/);
      // Before the deadline, the one-shot subscription is still installed.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      assert.equal((client as any)._catchallHandlers.length, handlersBefore + 1);
      mock.timers.tick(5_000);
      await settled;
      // On timeout the subscription is removed — no leaked listener.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      assert.equal((client as any)._catchallHandlers.length, handlersBefore);
    } finally {
      mock.timers.reset();
    }
  });

  // A file over the daemon's message limit makes it close the connection
  // (1009) mid-upload.  The next connection is a new client that cannot
  // answer, so waiting out the 120 s deadline only hid the refusal.
  test("rejects at once, naming the close, when the connection drops first", async () => {
    mock.timers.enable({ apis: ["setTimeout"] });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handlersBefore = (client as any)._catchallHandlers.length as number;
      const promise = client.stageFiles(
        "workspace_abc",
        [{ name: "big.pdf", data: new Uint8Array([1, 2, 3]) }],
      );
      lastInstance!.emitClose(1009, "message too big");
      await assert.rejects(promise, (err: unknown) => {
        assert.ok(err instanceof RequestInterruptedError);
        assert.equal(err.code, 1009);
        assert.match(err.message, /larger than its limit/);
        return true;
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      assert.equal((client as any)._catchallHandlers.length, handlersBefore);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      assert.equal((client as any)._closeWaiters.size, 0);
      // No timer left to fire a second rejection later.
      mock.timers.tick(STAGE_FILES_TIMEOUT_MS + 1);
    } finally {
      mock.timers.reset();
    }
  });

  test("a settled request is not rejected by a later close", async () => {
    const promise = client.stageFiles("ws", [{ name: "a.txt", data: new Uint8Array([1]) }]);
    lastInstance!.emit({
      type: EventTypeValue.WORKSPACE_FILES_STAGED,
      timestamp: new Date().toISOString(),
      workspace_id: "ws",
      staged: [{ name: "a.txt" }],
      failed: [],
    });
    await promise;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    assert.equal((client as any)._closeWaiters.size, 0);
  });

  test("uses STAGE_FILES_TIMEOUT_MS as the default deadline", () => {
    assert.equal(typeof STAGE_FILES_TIMEOUT_MS, "number");
    assert.ok(STAGE_FILES_TIMEOUT_MS > 0);
  });

  test("a normal staged response resolves and clears the timer (no late rejection)", async () => {
    mock.timers.enable({ apis: ["setTimeout"] });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handlersBefore = (client as any)._catchallHandlers.length as number;
      const promise = client.stageFiles(
        "workspace_abc",
        [{ name: "ok.txt", data: new Uint8Array([9]) }],
        { timeoutMs: 5_000 },
      );
      lastInstance!.emit({
        type: EventTypeValue.WORKSPACE_FILES_STAGED,
        timestamp: new Date().toISOString(),
        workspace_id: "workspace_abc",
        staged: [{ name: "ok.txt" }],
        failed: [],
      });
      const result = await promise;
      assert.equal(result.type, EventTypeValue.WORKSPACE_FILES_STAGED);
      // The subscription is gone and the timer, ticked past its deadline,
      // fires no rejection at nothing.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      assert.equal((client as any)._catchallHandlers.length, handlersBefore);
      mock.timers.tick(10_000);
    } finally {
      mock.timers.reset();
    }
  });
});

describe("JaatoClient.fetchWorkspaceFile (protocol 1.20)", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_FILE_FETCH_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  const tick = (): Promise<void> => new Promise<void>((resolve) => setTimeout(resolve, 0));
  const lastRequest = (): Record<string, unknown> =>
    JSON.parse(lastInstance!.sent[lastInstance!.sent.length - 1]!) as Record<string, unknown>;
  const header = (requestId: unknown, fields: Record<string, unknown>): object => ({
    type: EventTypeValue.WORKSPACE_FILE_CONTENT,
    timestamp: new Date().toISOString(),
    request_id: requestId,
    ...fields,
  });
  const binary = (bytes: Uint8Array): void => {
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    lastInstance!.onmessage!({ data: buf } as any);
  };

  test("the binary frame after a header is that file's bytes", async () => {
    const promise = client.fetchWorkspaceFile("out/report.pdf");
    await tick();
    const req = lastRequest();
    assert.equal(req.type, EventTypeValue.WORKSPACE_FILE_FETCH_REQUEST);
    assert.equal(req.path, "out/report.pdf");
    assert.equal(req.metadata_only, false);
    const bytes = new TextEncoder().encode("%PDF-1.7");
    lastInstance!.emit(header(req.request_id, { ok: true, path: "out/report.pdf", name: "report.pdf", size: bytes.byteLength }));
    binary(bytes);
    const result = await promise;
    assert.equal(result.event.ok, true);
    assert.deepEqual(result.data, bytes);
  });

  test("a text frame after the binary is parsed as an event again", async () => {
    const promise = client.fetchWorkspaceFile("a.txt");
    await tick();
    const req = lastRequest();
    lastInstance!.emit(header(req.request_id, { ok: true, path: "a.txt", size: 1 }));
    binary(new Uint8Array([65]));
    await promise;
    const seen: string[] = [];
    client.subscribeAll((e) => { seen.push(String(e.type)); });
    lastInstance!.emit({ type: EventTypeValue.SYSTEM_MESSAGE, timestamp: new Date().toISOString(), message: "hi" });
    await tick();
    assert.ok(seen.includes(EventTypeValue.SYSTEM_MESSAGE));
  });

  test("a metadata-only answer and a refusal carry no bytes and wait for none", async () => {
    const meta = client.fetchWorkspaceFile("a.txt", { metadataOnly: true });
    await tick();
    const r1 = lastRequest();
    assert.equal(r1.metadata_only, true);
    lastInstance!.emit(header(r1.request_id, { ok: true, metadata_only: true, size: 3 }));
    const m = await meta;
    assert.equal(m.data, null);
    assert.equal(m.event.size, 3);

    const refused = client.fetchWorkspaceFile(".env");
    await tick();
    lastInstance!.emit(header(lastRequest().request_id, { ok: false, category: "credential" }));
    const r = await refused;
    assert.equal(r.data, null);
    assert.equal(r.event.category, "credential");
  });

  test("concurrent fetches are matched by request_id, not by arrival order", async () => {
    const first = client.fetchWorkspaceFile("one.txt");
    await tick();
    const id1 = lastRequest().request_id;
    const second = client.fetchWorkspaceFile("two.txt");
    await tick();
    const id2 = lastRequest().request_id;
    assert.notEqual(id1, id2);
    lastInstance!.emit(header(id2, { ok: true, path: "two.txt", size: 1 }));
    binary(new Uint8Array([2]));
    lastInstance!.emit(header(id1, { ok: true, path: "one.txt", size: 1 }));
    binary(new Uint8Array([1]));
    assert.deepEqual((await first).data, new Uint8Array([1]));
    assert.deepEqual((await second).data, new Uint8Array([2]));
  });

  test("is refused below protocol 1.20 and sends nothing", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, "1.19");
    if (lastInstance) lastInstance.sent = [];
    await assert.rejects(() => client.fetchWorkspaceFile("a.txt"), /workspace\.file\.fetch/);
    assert.equal(getSent().length, 0);
  });
});

describe("JaatoClient auto re-attach on reconnect", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("opt-in flag triggers attachSession after RECONNECTING → CONNECTED", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      recovery: {
        autoReconnect: true,
        autoReattachSessionId: true,
        initialBackoffSeconds: 0.01,
        maxBackoffSeconds: 0.05,
        jitterFactor: 0.0,
        maxReconnectAttempts: 5,
      },
    });
    await connectAndAck(client);

    // Simulate the server emitting a SessionInfoEvent so the
    // client knows its session_id (the auto re-attach handler
    // depends on this being non-null).
    lastInstance!.emit({
      type: EventTypeValue.SESSION_INFO,
      timestamp: new Date().toISOString(),
      session_id: "sess_42",
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
    assert.equal(client.sessionId, "sess_42");

    // Drop the connection.  Reconnect machinery kicks in.
    lastInstance!.emitClose(1006, "abnormal");
    await new Promise<void>((resolve) => setTimeout(resolve, 50));

    // After reconnect, the new mock WS receives the handshake,
    // we ack it, and the auto re-attach should fire.
    if (lastInstance!.readyState !== 1) {
      // Wait for the new connection's open + ack.
      await new Promise<void>((resolve) => setTimeout(resolve, 30));
    }
    // Drain any handshake frames the client sent on the new
    // connection.  We only care about post-reconnect activity.
    lastInstance!.sent = [];
    lastInstance!.emit(makeConnectedEvent());
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // The auto-reattach handler should have called attachSession,
    // which sends a CommandRequest with session.attach + sess_42.
    const sent = lastInstance!.sent.map((s) => JSON.parse(s) as Record<string, unknown>);
    const reattach = sent.find(
      (ev) => ev.type === EventTypeValue.COMMAND && ev.command === "session.attach",
    );
    assert.ok(reattach, `expected session.attach in ${JSON.stringify(sent)}`);
    assert.deepEqual((reattach as { args?: string[] }).args, ["sess_42"]);
    await client.close();
  });

  test("opt-out (default) does NOT auto-re-attach", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      recovery: {
        autoReconnect: true,
        initialBackoffSeconds: 0.01,
        maxBackoffSeconds: 0.05,
        jitterFactor: 0.0,
        maxReconnectAttempts: 5,
      },
    });
    await connectAndAck(client);
    lastInstance!.emit({
      type: EventTypeValue.SESSION_INFO,
      timestamp: new Date().toISOString(),
      session_id: "sess_99",
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 5));

    lastInstance!.emitClose(1006, "lost");
    await new Promise<void>((resolve) => setTimeout(resolve, 50));
    if (lastInstance!.readyState !== 1) {
      await new Promise<void>((resolve) => setTimeout(resolve, 30));
    }
    lastInstance!.sent = [];
    lastInstance!.emit(makeConnectedEvent());
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    const sent = lastInstance!.sent.map((s) => JSON.parse(s) as Record<string, unknown>);
    const reattach = sent.find(
      (ev) => ev.type === EventTypeValue.COMMAND && ev.command === "session.attach",
    );
    assert.equal(reattach, undefined);
    await client.close();
  });
});

describe("JaatoClient event stream", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("subscribe receives events emitted from the server", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const received: JaatoEvent[] = [];
    client.subscribeAll((e) => received.push(e));
    await connectAndAck(client);
    // Drop the inaugural ConnectedEvent that fires during handshake.
    received.length = 0;

    const evt = {
      type: EventTypeValue.AGENT_OUTPUT,
      timestamp: new Date().toISOString(),
      agent_id: "main",
      source: "model",
      text: "hello",
    } as unknown as JaatoEvent;
    lastInstance!.emit(evt);
    await new Promise<void>((resolve) => setTimeout(resolve, 5));

    assert.equal(received.length, 1);
    assert.equal(received[0].type, EventTypeValue.AGENT_OUTPUT);
    await client.close();
  });

  test("unsubscribe stops handler invocation", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const received: JaatoEvent[] = [];
    const unsub = client.subscribeAll((e) => received.push(e));
    await connectAndAck(client);
    // After connect the inaugural ConnectedEvent has been dispatched.
    const baseline = received.length;
    unsub();
    lastInstance!.emit({
      type: EventTypeValue.SYSTEM_MESSAGE,
      timestamp: new Date().toISOString(),
    } as unknown as JaatoEvent);
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
    assert.equal(received.length, baseline);
    await client.close();
  });
});

describe("JaatoClient reconnect", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("connection loss with auto-reconnect enabled transitions through RECONNECTING", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      recovery: {
        autoReconnect: true,
        initialBackoffSeconds: 0.01,
        maxBackoffSeconds: 0.05,
        jitterFactor: 0.0,
        maxReconnectAttempts: 1,
      },
    });
    const states: ConnectionState[] = [];
    client.onStatus((s) => states.push(s.state));
    await connectAndAck(client);

    // Drop the connection
    lastInstance!.emitClose(1006, "abnormal");
    // Wait for the reconnect-loop microtask + the backoff timer
    await new Promise<void>((resolve) => setTimeout(resolve, 50));

    assert.ok(states.includes(ConnectionState.RECONNECTING),
      `expected RECONNECTING in transitions, got ${states.join(",")}`);
    await client.close();
  });

  test("auto-reconnect disabled goes straight to CLOSED on close", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      recovery: { autoReconnect: false },
    });
    await connectAndAck(client);
    lastInstance!.emitClose(1006, "lost");
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
    assert.equal(client.state, ConnectionState.CLOSED);
  });

  test("send while RECONNECTING throws ReconnectingError", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      recovery: {
        autoReconnect: true,
        initialBackoffSeconds: 1.0, // long enough to stay in RECONNECTING
        maxBackoffSeconds: 1.0,
        jitterFactor: 0.0,
        maxReconnectAttempts: null,
      },
    });
    await connectAndAck(client);
    lastInstance!.emitClose(1006, "lost");
    await new Promise<void>((resolve) => setTimeout(resolve, 10));
    assert.equal(client.state, ConnectionState.RECONNECTING);
    await assert.rejects(client.sendMessage("hi"), ReconnectingError);
    await client.close();
  });

  test("a TokenProvider is consulted afresh on every attempt (single-use tickets, #1074)", async () => {
    // A ticket is consumed at accept, so the value that opened the last
    // connection can never open the next one.  The provider must be
    // called per ATTEMPT, and each attempt must present its own value.
    let minted = 0;
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      token: async () => `ticket-${++minted}`,
      recovery: {
        autoReconnect: true,
        initialBackoffSeconds: 0.01,
        maxBackoffSeconds: 0.05,
        jitterFactor: 0.0,
        maxReconnectAttempts: 3,
      },
    });
    await connectAndAck(client);
    assert.equal(minted, 1);
    assert.ok(lastInstance!.url.endsWith("?token=ticket-1"), lastInstance!.url);

    const first = lastInstance!;
    first.emitClose(1006, "lost");
    await new Promise<void>((resolve) => setTimeout(resolve, 40));

    assert.equal(minted, 2, "reconnect must mint a fresh ticket, not replay the consumed one");
    assert.notEqual(lastInstance, first);
    assert.ok(lastInstance!.url.endsWith("?token=ticket-2"), lastInstance!.url);
    await client.close();
  });

  test("a TokenProvider that throws on connect() propagates to the caller", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      token: async () => { throw new Error("backend says 401"); },
      recovery: { autoReconnect: false },
    });
    await assert.rejects(client.connect(), /backend says 401/);
    assert.equal(lastInstance, null, "no WebSocket may be opened without a credential");
  });

  test("a TokenProvider that throws during reconnect fails that attempt and schedules the next", async () => {
    let calls = 0;
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      token: async () => {
        calls += 1;
        if (calls === 2) throw new Error("backend briefly down");
        return `ticket-${calls}`;
      },
      recovery: {
        autoReconnect: true,
        initialBackoffSeconds: 0.01,
        maxBackoffSeconds: 0.02,
        jitterFactor: 0.0,
        maxReconnectAttempts: 5,
      },
    });
    await connectAndAck(client);
    lastInstance!.emitClose(1006, "lost");
    // attempt 2 throws inside the provider, attempt 3 opens a socket
    await new Promise<void>((resolve) => setTimeout(resolve, 80));
    assert.ok(calls >= 3, `expected the loop to continue past the throwing attempt, calls=${calls}`);
    assert.ok(lastInstance!.url.endsWith("?token=ticket-3"), lastInstance!.url);
    assert.equal(client.state, ConnectionState.RECONNECTING);
    await client.close();
  });

  test("a TokenProvider returning undefined connects with no token", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080", token: () => undefined });
    await connectAndAck(client);
    assert.ok(!lastInstance!.url.includes("token="), lastInstance!.url);
    await client.close();
  });

  test("close() transitions to CLOSED and rejects further sends", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    await client.close();
    assert.equal(client.state, ConnectionState.CLOSED);
    await assert.rejects(client.connect(), ConnectionClosedError);
  });
});

// ──── Subscribe API parity tests ─────────────────────────────────
//
// Mirror jaato-sdk/jaato_sdk/tests/test_subscribe_api.py one-to-one
// to keep cross-language semantics in lockstep.

function _output(text = "x"): JaatoEvent {
  return {
    type: EventTypeValue.AGENT_OUTPUT,
    timestamp: new Date().toISOString(),
    agent_id: "main",
    source: "model",
    text,
    mode: "append",
  } as unknown as JaatoEvent;
}

function _completed(): JaatoEvent {
  return {
    type: EventTypeValue.AGENT_COMPLETED,
    timestamp: new Date().toISOString(),
  } as unknown as JaatoEvent;
}

function _perm(): JaatoEvent {
  return {
    type: EventTypeValue.PERMISSION_REQUESTED,
    timestamp: new Date().toISOString(),
    request_id: "r1",
    tool_name: "cli",
    arguments: {},
  } as unknown as JaatoEvent;
}

describe("JaatoClient subscribe API", () => {
  beforeEach(() => installMockWebSocket());
  afterEach(() => restoreWebSocket());

  test("subscribe filters by type", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: string[] = [];
    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) => seen.push(e.text));

    lastInstance!.emit(_output("a"));
    lastInstance!.emit(_completed());
    lastInstance!.emit(_output("b"));
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, ["a", "b"]);
    await client.close();
  });

  test("subscribeAll receives everything", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: string[] = [];
    client.subscribeAll((e) => seen.push(e.type));

    lastInstance!.emit(_output());
    lastInstance!.emit(_completed());
    lastInstance!.emit(_perm());
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, [
      EventTypeValue.AGENT_OUTPUT,
      EventTypeValue.AGENT_COMPLETED,
      EventTypeValue.PERMISSION_REQUESTED,
    ]);
    await client.close();
  });

  test("subscribeOnce fires exactly once", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: string[] = [];
    client.subscribeOnce(EventTypeValue.AGENT_OUTPUT, (e) => seen.push(e.text));

    lastInstance!.emit(_output("first"));
    lastInstance!.emit(_output("second"));
    lastInstance!.emit(_output("third"));
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, ["first"]);
    await client.close();
  });

  test("subscribeMany atomic unsub", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: Array<[string, ...unknown[]]> = [];
    const unsubAll = client.subscribeMany({
      [EventTypeValue.AGENT_OUTPUT]: (e) => seen.push(["out", e.text]),
      [EventTypeValue.AGENT_COMPLETED]: () => seen.push(["done"]),
      [EventTypeValue.PERMISSION_REQUESTED]: (e) => seen.push(["perm", e.tool_name]),
    });

    lastInstance!.emit(_output("hi"));
    lastInstance!.emit(_completed());
    lastInstance!.emit(_perm());
    await new Promise<void>((r) => setTimeout(r, 5));

    unsubAll();
    lastInstance!.emit(_output("after"));
    lastInstance!.emit(_completed());
    lastInstance!.emit(_perm());
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, [
      ["out", "hi"],
      ["done"],
      ["perm", "cli"],
    ]);
    await client.close();
  });

  test("async handler does not block stream", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const order: Array<[string, string]> = [];

    client.subscribe(EventTypeValue.AGENT_OUTPUT, async (e) => {
      await new Promise<void>((r) => setTimeout(r, 50));
      order.push(["async-done", e.text]);
    });
    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) => {
      order.push(["sync", e.text]);
    });

    lastInstance!.emit(_output("first"));
    lastInstance!.emit(_output("second"));
    await new Promise<void>((r) => setTimeout(r, 100));

    const syncs = order.filter((o) => o[0] === "sync");
    const asyncs = order.filter((o) => o[0] === "async-done").sort();
    assert.deepEqual(syncs, [
      ["sync", "first"],
      ["sync", "second"],
    ]);
    assert.deepEqual(asyncs, [
      ["async-done", "first"],
      ["async-done", "second"],
    ]);
    await client.close();
  });

  test("async handlers run concurrently", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const timings: Array<[number, number]> = [];

    const slow = async (): Promise<void> => {
      const start = Date.now();
      await new Promise<void>((r) => setTimeout(r, 50));
      timings.push([start, Date.now()]);
    };

    client.subscribe(EventTypeValue.AGENT_OUTPUT, slow);
    client.subscribe(EventTypeValue.AGENT_OUTPUT, slow);
    client.subscribe(EventTypeValue.AGENT_OUTPUT, slow);

    lastInstance!.emit(_output());
    await new Promise<void>((r) => setTimeout(r, 120));

    assert.equal(timings.length, 3);
    const starts = timings.map((t) => t[0]).sort();
    const ends = timings.map((t) => t[1]).sort();
    assert.ok(
      starts[2]! < ends[0]!,
      "Handlers serialized instead of running concurrently",
    );
    await client.close();
  });

  test("sync exception is isolated", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: Array<[string, string]> = [];

    client.subscribe(EventTypeValue.AGENT_OUTPUT, () => {
      throw new Error("intentional");
    });
    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) =>
      seen.push(["good", e.text]),
    );

    lastInstance!.emit(_output("a"));
    lastInstance!.emit(_output("b"));
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, [
      ["good", "a"],
      ["good", "b"],
    ]);
    await client.close();
  });

  test("async rejection is isolated", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: string[] = [];

    client.subscribe(EventTypeValue.AGENT_OUTPUT, async () => {
      await new Promise<void>((r) => setTimeout(r, 5));
      throw new Error("intentional async failure");
    });
    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) => seen.push(e.text));

    lastInstance!.emit(_output("a"));
    lastInstance!.emit(_output("b"));
    await new Promise<void>((r) => setTimeout(r, 30));

    assert.deepEqual(seen, ["a", "b"]);
    await client.close();
  });

  test("unsub during dispatch takes effect next event", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: Array<[string, string]> = [];
    let unsubB: (() => void) | null = null;

    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) => {
      seen.push(["a", e.text]);
      if (unsubB) {
        unsubB();
        unsubB = null;
      }
    });
    unsubB = client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) =>
      seen.push(["b", e.text]),
    );

    lastInstance!.emit(_output("first"));
    await new Promise<void>((r) => setTimeout(r, 5));
    assert.ok(seen.some(([h, t]) => h === "b" && t === "first"));

    lastInstance!.emit(_output("second"));
    await new Promise<void>((r) => setTimeout(r, 5));
    assert.ok(!seen.some(([h, t]) => h === "b" && t === "second"));
    assert.ok(seen.some(([h, t]) => h === "a" && t === "second"));
    await client.close();
  });

  test("handler registered before connect captures inaugural events", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const seen: string[] = [];
    client.subscribe(EventTypeValue.CONNECTED, () => seen.push("connected"));

    await connectAndAck(client);
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, ["connected"]);
    await client.close();
  });

  test("no replay on late subscribe", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);

    lastInstance!.emit(_output("missed"));
    await new Promise<void>((r) => setTimeout(r, 5));

    const seen: string[] = [];
    client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) => seen.push(e.text));

    lastInstance!.emit(_output("seen"));
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, ["seen"]);
    await client.close();
  });

  test("unsubscribe is idempotent", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client);
    const seen: string[] = [];
    const unsub = client.subscribe(EventTypeValue.AGENT_OUTPUT, (e) =>
      seen.push(e.text),
    );

    lastInstance!.emit(_output("a"));
    await new Promise<void>((r) => setTimeout(r, 5));
    unsub();
    unsub(); // second call must not throw
    lastInstance!.emit(_output("b"));
    await new Promise<void>((r) => setTimeout(r, 5));

    assert.deepEqual(seen, ["a"]);
    await client.close();
  });
});

describe("serverLimits", () => {
  test("a daemon that advertises nothing gets the legacy 1 MiB limits", () => {
    const limits = serverLimitsFrom({ client_id: "c", server_version: "1.1.0rc2" });
    assert.deepEqual(limits, LEGACY_SERVER_LIMITS);
    assert.equal(limits.stagePerFileLimit, 1024 * 1024);
    assert.equal(limits.advertised, false);
  });

  test("advertised limits are read, and the per-file cap never exceeds a message", () => {
    const limits = serverLimitsFrom({
      max_message_size: 2 * 1024 * 1024,
      stage_per_file_limit: 10 * 1024 * 1024,
      stage_total_limit: 50 * 1024 * 1024,
    });
    assert.equal(limits.maxMessageSize, 2 * 1024 * 1024);
    assert.equal(limits.stagePerFileLimit, 2 * 1024 * 1024);
    assert.equal(limits.stageTotalLimit, 50 * 1024 * 1024);
    assert.equal(limits.advertised, true);
  });

  test("the client exposes the handshake's limits", async () => {
    installMockWebSocket();
    try {
      const client = new JaatoClient({ url: "ws://localhost:8080" });
      assert.equal(client.serverLimits, null);
      await connectAndAck(client);
      assert.equal(client.serverLimits?.advertised, false);
      await client.close();
    } finally {
      restoreWebSocket();
    }
  });
});

describe("JaatoClient memory verbs (protocol 1.22, #1232)", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_MEMORY_VERBS_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  const tick = (): Promise<void> => new Promise<void>((resolve) => setTimeout(resolve, 0));
  const lastRequest = (): Record<string, unknown> =>
    JSON.parse(lastInstance!.sent[lastInstance!.sent.length - 1]!) as Record<string, unknown>;

  test("the floor is 1.22", () => {
    assert.equal(MIN_MEMORY_VERBS_PROTOCOL, "1.22");
  });

  test("listMemories resolves with the answer carrying ITS request_id", async () => {
    const promise = client.listMemories();
    await tick();
    const req = lastRequest();
    assert.equal(req.type, EventTypeValue.MEMORY_LIST_REQUEST);
    assert.ok(req.request_id);
    // A decoy answer for somebody else, then an echo of the request, then ours.
    lastInstance!.emit({ type: EventTypeValue.MEMORY_LIST, request_id: "other", memories: [{ id: "x" }] });
    lastInstance!.emit({ ...req });
    lastInstance!.emit({
      type: EventTypeValue.MEMORY_LIST, request_id: req.request_id,
      memories: [{ id: "mine" }], ok: true, may_curate: false,
    });
    const got = await promise;
    assert.deepEqual(got.memories, [{ id: "mine" }]);
    assert.equal(got.may_curate, false);
  });

  test("updateMemory sends only the fields given", async () => {
    const promise = client.updateMemory("m1", { description: "d", tags: ["aa", "bb"] });
    await tick();
    const req = lastRequest();
    assert.equal(req.type, EventTypeValue.MEMORY_UPDATE_REQUEST);
    assert.equal(req.memory_id, "m1");
    assert.equal(req.description, "d");
    assert.deepEqual(req.tags, ["aa", "bb"]);
    assert.ok(!("content" in req));
    assert.ok(!("maturity" in req));
    lastInstance!.emit({ type: EventTypeValue.MEMORY_UPDATE_RESULT, request_id: req.request_id, memory_id: "m1", ok: true });
    assert.equal((await promise).ok, true);
  });

  test("approve and dismiss are maturity updates", async () => {
    for (const [call, maturity] of [
      [() => client.approveMemory("m1"), "validated"],
      [() => client.dismissMemory("m1"), "dismissed"],
    ] as const) {
      const promise = call();
      await tick();
      const req = lastRequest();
      assert.equal(req.maturity, maturity);
      lastInstance!.emit({ type: EventTypeValue.MEMORY_UPDATE_RESULT, request_id: req.request_id, memory_id: "m1", ok: true });
      await promise;
    }
  });

  test("getMemory and deleteMemory send their typed requests", async () => {
    const got = client.getMemory("m1");
    await tick();
    const r1 = lastRequest();
    assert.equal(r1.type, EventTypeValue.MEMORY_GET_REQUEST);
    lastInstance!.emit({ type: EventTypeValue.MEMORY_GET_RESULT, request_id: r1.request_id, memory_id: "m1", ok: true, memory: { id: "m1", content: "c" } });
    assert.equal(((await got).memory as { content?: string }).content, "c");

    const del = client.deleteMemory("m1");
    await tick();
    const r2 = lastRequest();
    assert.equal(r2.type, EventTypeValue.MEMORY_DELETE_REQUEST);
    lastInstance!.emit({ type: EventTypeValue.MEMORY_DELETE_RESULT, request_id: r2.request_id, memory_id: "m1", ok: false, category: "not_owner" });
    assert.equal((await del).category, "not_owner");
  });

  test("a closed connection rejects rather than resolving empty", async () => {
    const promise = client.listMemories();
    await tick();
    lastInstance!.close(1006, "gone");
    await assert.rejects(promise, RequestInterruptedError);
  });

  test("every verb is refused below 1.22 with nothing sent", async () => {
    await client.close();
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, "1.21");
    if (lastInstance) lastInstance.sent = [];
    for (const call of [
      () => client.listMemories(),
      () => client.getMemory("m1"),
      () => client.updateMemory("m1", { description: "d" }),
      () => client.approveMemory("m1"),
      () => client.dismissMemory("m1"),
      () => client.deleteMemory("m1"),
    ]) {
      await assert.rejects(call, /memory verbs/);
    }
    assert.equal(lastInstance!.sent.length, 0);
  });
});

describe("JaatoClient workspace/session pickers (1.27)", () => {
  let client: JaatoClient;

  beforeEach(async () => {
    installMockWebSocket();
    client = new JaatoClient({ url: "ws://localhost:8080" });
    await connectAndAck(client, MIN_WORKSPACE_PICKER_PROTOCOL);
    if (lastInstance) lastInstance.sent = [];
  });

  afterEach(async () => {
    await client.close();
    restoreWebSocket();
  });

  test("createSession sends a model override as --model/--provider (1.27)", async () => {
    await client.createSession({
      profile: "researcher",
      model: "gpt-5.1",
      provider: "openai",
    });
    const [ev] = getSent();
    const args = (ev as { args?: string[] }).args ?? [];
    assert.deepEqual(args.slice(args.indexOf("--model"), args.indexOf("--model") + 2),
      ["--model", "gpt-5.1"]);
    assert.deepEqual(args.slice(args.indexOf("--provider"), args.indexOf("--provider") + 2),
      ["--provider", "openai"]);
  });

  test("createSession model override without provider sends only --model", async () => {
    await client.createSession({ model: "gpt-5.1" });
    const [ev] = getSent();
    assert.deepEqual((ev as { args?: string[] }).args, ["--model", "gpt-5.1"]);
  });

  test("createSession refuses a provider without a model", async () => {
    await assert.rejects(
      client.createSession({ provider: "openai" }),
      TypeError,
    );
    assert.equal(getSent().length, 0);
  });

  test("inspectWorkspace sends workspace.inspect and resolves on its answer", async () => {
    const pending = client.inspectWorkspace("proj");
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    const [ev] = getSent() as unknown as Array<Record<string, unknown>>;
    assert.equal(ev.type, "workspace.inspect");
    assert.equal(ev.name, "proj");
    const rid = ev.request_id as string;
    lastInstance!.emit({ type: "workspace.inspected", request_id: "other", name: "x" });
    lastInstance!.emit({
      type: "workspace.inspected", request_id: rid, name: "proj", ok: true,
      sessions: { total: 2, waiting: 0, awake: 1, sleeping: 1 }, repos: [],
    });
    const answer = await pending;
    assert.equal(answer.name, "proj");
    assert.equal(answer.sessions.total, 2);
  });

  test("cloneIntoWorkspace sends the repos with a request id", async () => {
    const rid = await client.cloneIntoWorkspace("proj", [
      { repo: "octo/one", branch: "main" },
    ]);
    const [ev] = getSent() as unknown as Array<Record<string, unknown>>;
    assert.equal(ev.type, "workspace.clone");
    assert.equal(ev.request_id, rid);
    assert.deepEqual(ev.repos, [{ repo: "octo/one", branch: "main", forge: "github" }]);
  });
});
