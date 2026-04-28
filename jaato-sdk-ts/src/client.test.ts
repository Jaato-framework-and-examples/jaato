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
import { afterEach, beforeEach, describe, test } from "node:test";

import { JaatoClient, MIN_SERVER_VERSION } from "./client.js";
import {
  ConnectionClosedError,
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
const realWebSocket = (globalThis as Record<string, unknown>).WebSocket;

function installMockWebSocket(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).WebSocket = function (url: string): MockInstance {
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

function makeConnectedEvent(serverVersion = MIN_SERVER_VERSION): JaatoEvent {
  return {
    type: EventTypeValue.CONNECTED,
    timestamp: new Date().toISOString(),
    protocol_version: "1.0",
    server_info: {
      client_id: "client_1",
      server_version: serverVersion,
    },
  } as unknown as JaatoEvent;
}

async function connectAndAck(
  client: JaatoClient,
  serverVersion = MIN_SERVER_VERSION,
): Promise<void> {
  const connectPromise = client.connect();
  // Let the WS open microtask fire, then emit ConnectedEvent.
  await new Promise<void>((resolve) => queueMicrotask(resolve));
  await new Promise<void>((resolve) => queueMicrotask(resolve));
  if (lastInstance == null) {
    throw new Error("MockWebSocket was not constructed");
  }
  lastInstance.emit(makeConnectedEvent(serverVersion));
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
    assert.equal(client.serverVersion, MIN_SERVER_VERSION);
    assert.equal(client.clientId, "client_1");
    await client.close();
  });

  test("token appended as ?token= query param", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080/", token: "secret" });
    await connectAndAck(client);
    assert.ok(lastInstance!.url.includes("token=secret"));
    await client.close();
  });

  test("incompatible server version throws IncompatibleServerError", async () => {
    const client = new JaatoClient({ url: "ws://localhost:8080" });
    const connectPromise = client.connect();
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    lastInstance!.emit(makeConnectedEvent("0.0.1"));
    await assert.rejects(connectPromise, IncompatibleServerError);
  });

  test("explicit minServerVersion override is honoured", async () => {
    const client = new JaatoClient({
      url: "ws://localhost:8080",
      minServerVersion: "0.0.1",
    });
    await connectAndAck(client, "0.0.5");
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

  test("deleteSession carries the session id as the first arg", async () => {
    await client.deleteSession("sess_xyz");
    const [ev] = getSent();
    assert.equal(ev.type, EventTypeValue.COMMAND);
    assert.equal((ev as { command?: string }).command, "session.delete");
    assert.deepEqual((ev as { args?: string[] }).args, ["sess_xyz"]);
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
